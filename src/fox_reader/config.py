from __future__ import annotations

from pathlib import Path
from typing import Any

import logging
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


def _coerce_to_bool(value: Any, field_name: str) -> bool:
    """Convert common string values to bool."""

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.strip().lower()

        if value in {"true", "1", "y", "yes", "on"}:
            return True

        if value in {"false", "0", "n", "no", "off"}:
            return False

    raise ValueError(f"{field_name} must be a boolean")


class FoxConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=7954, ge=1, le=65535)
    mtl_dir: Path = Path("models")
    theme: str = "dark"
    layout: str = "basic"

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("host cannot be empty")

        return value

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, value: str) -> str:
        value = value.strip().lower()

        if value not in {"dark", "light"}:
            raise ValueError("theme must be one of: dark, light")

        return value

    @field_validator("layout")
    @classmethod
    def validate_layout(cls, value: str) -> str:
        value = value.strip().lower()

        if value not in {"basic", "default", "reader", "translation"}:
            raise ValueError(
                "layout must be one of: basic, default, reader, translation"
            )

        return value

    def resolve_mtl_dir(self, root: Path) -> Path:
        if self.mtl_dir.is_absolute():
            return self.mtl_dir
        return root / self.mtl_dir


class ConfigManager:
    IMMUTABLE_FIELDS = {"host", "port"}

    def __init__(self, config_root: Path):
        DEFAULT_CONFIG_FILENAME = "fox_config.yaml"
        USER_CONFIG_FILENAME = "user_config.yaml"

        self.config_root = Path(config_root)

        user_config_path = self.config_root / USER_CONFIG_FILENAME
        default_config_path = self.config_root / DEFAULT_CONFIG_FILENAME

        if user_config_path.exists():
            self.config_path = user_config_path
            self.config_file_class = "user"
        else:
            self.config_path = default_config_path
            self.config_file_class = "default"

        self.config: FoxConfig | None = None

    def load(self) -> FoxConfig:
        self.config_root.mkdir(parents=True, exist_ok=True)

        if not self.config_path.exists():
            logger.info("Creating default config: %s", self.config_path)

            self.config = FoxConfig()
            self.save()

            return self.config

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            self.config = FoxConfig.model_validate(data)

        except (ValidationError, yaml.YAMLError, OSError) as e:
            logger.warning("Failed to load config: %s", e)
            logger.warning("Restoring default configuration.")

            self.config = FoxConfig()
            self.save()

        return self.config

    def save(self) -> None:
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded.")

        with self.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.config.model_dump(mode="json"),
                f,
                sort_keys=False,
            )

    def update(self, **changes) -> FoxConfig:
        if self.config is None:
            raise RuntimeError("Call load() first.")

        forbidden = self.IMMUTABLE_FIELDS & changes.keys()

        if forbidden:
            raise PermissionError(
                f"Cannot modify runtime-protected fields: "
                f"{', '.join(sorted(forbidden))}"
            )

        # Validated into a local first, and only then adopted. Assigning the
        # unvalidated copy to self.config -- which is what this used to do --
        # leaves a rejected update in place when the validation below raises,
        # and the next save() writes it out. On the following launch load()
        # rejects that file and restores *every* field to its default, so one
        # bad value costs the user the rest of their settings.
        candidate = self.config.model_copy(update=changes)
        candidate = FoxConfig.model_validate(candidate.model_dump())

        self.config = candidate

        self.save()

        return self.config

    def reset(self) -> FoxConfig:
        self.config = FoxConfig()
        self.save()

        return self.config