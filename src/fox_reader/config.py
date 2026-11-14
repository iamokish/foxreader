from __future__ import annotations

import logging
import os
import re
import threading
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from uuid import uuid4

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
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("host must be an IP address or hostname")
        try:
            ip_address(value)
        except ValueError:
            # Validate syntax locally; DNS and the availability of an interface
            # are runtime concerns, not reasons to erase a saved preference.
            try:
                hostname = value.rstrip(".").encode("idna").decode("ascii")
            except UnicodeError as exc:
                raise ValueError("host must be an IP address or hostname") from exc
            if (
                len(hostname) > 253
                or re.fullmatch(r"[\d.]+", hostname)
                or not all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", part)
                           for part in hostname.split("."))
            ):
                raise ValueError("host must be an IP address or hostname")
            value = hostname + ("." if value.endswith(".") else "")
        if len(value) >= 256 or not re.fullmatch(r"[A-Za-z0-9.:%_-]+", value):
            raise ValueError("host must be an IP address or hostname")
        return value

    @field_validator("port", mode="before")
    @classmethod
    def validate_port(cls, value: Any) -> int:
        if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
            value = int(value.strip())
        if type(value) is not int:
            raise ValueError("port must be an integer")
        return value

    @field_validator("mtl_dir", mode="before")
    @classmethod
    def validate_mtl_dir(cls, value: Any) -> Any:
        if isinstance(value, (str, Path)):
            text = str(value)
            if not text.strip() or any(ord(char) < 32 or ord(char) == 127 for char in text):
                raise ValueError("mtl_dir must be a non-empty path without control characters")
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

        # Only "basic" and "vulpes" are real layouts. The other three are retired
        # and the frontend maps them (see normalizeLayout): "default" onto Vulpes,
        # its successor, and "reader"/"translation" onto Basic. They stay accepted
        # so an existing fox_config.yaml does not stop the app from starting.
        if value not in {"basic", "vulpes", "default", "reader", "translation"}:
            raise ValueError(
                "layout must be one of: basic, vulpes "
                "(retired, still accepted: default, reader, translation)"
            )

        return value

    def resolve_mtl_dir(self, root: Path) -> Path:
        if self.mtl_dir.is_absolute():
            return self.mtl_dir
        return root / self.mtl_dir


class _ConfigLoader(yaml.SafeLoader):
    """Keep PyYAML's last-value precedence, but make duplicate keys visible."""

    def __init__(self, stream: bytes):
        super().__init__(stream)
        self.duplicate_lines: list[int] = []

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            # YAML merge overrides are intentional, not duplicate user keys.
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in seen:
                    self.duplicate_lines.append(key_node.start_mark.line + 1)
                seen.add(key)
            except TypeError:
                pass  # SafeLoader reports unhashable keys with source context.
        return super().construct_mapping(node, deep=deep)


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _write_config(path: Path, data: dict, expected: bytes | None) -> bytes:
    """Replace a complete document atomically, retaining the old file on failure."""
    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{path.name}.{uuid4().hex}.tmp")
    # A single exclusive open fails promptly on Windows ACL errors. tempfile's
    # retry loop can mistake denied access for name collisions on that platform.
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() and os.name != "nt":
            temporary.chmod(destination.stat().st_mode & 0o777)
        result = temporary.read_bytes()
        if _read_bytes(destination) != expected:
            raise OSError(f"Config changed on disk; reload before saving: {path}")
        os.replace(temporary, destination)
        return result
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove temporary config %s", temporary)


def _launcher_readable(source: bytes, config: FoxConfig) -> bool:
    """The native launcher's address reader requires ordinary top-level lines.

    Normalize flow mappings, aliases, quoted keys, BOMs and alternative numeric
    spellings once, even when their YAML values already pass validation. Normal
    complete files (including comments) are left alone on subsequent launches.
    """
    try:
        lines = source.decode("utf-8").splitlines()
    except UnicodeError:
        return False
    for name in ("host", "port"):
        canonical = yaml.safe_dump({name: getattr(config, name)}, allow_unicode=True).strip()
        if not any(line.rstrip() == canonical or line.startswith(canonical + " #") for line in lines):
            return False
    return True


class ConfigManager:
    """Reconcile both files against FoxConfig; the optional user file wins.

    Known fields are validated independently on load. Unknown YAML data is kept
    outside the runtime model so upgrading/downgrading never removes it. Runtime
    edits remain strict, and are published only after a successful save.
    """

    IMMUTABLE_FIELDS = {"host", "port"}

    def __init__(self, config_root: Path):
        self.config_root = Path(config_root)
        self._lock = threading.RLock()
        self.config: FoxConfig | None = None
        self._extra: dict = {}
        self._source: bytes | None = None
        self._write_blocked = False
        self._select_path()

    def _select_path(self) -> None:
        user = self.config_root / "user_config.yaml"
        try:
            exists = user.exists()
        except OSError:
            # An unreadable user file still takes precedence. _load_file logs
            # the actual read failure and prevents subsequent overwrites.
            exists = True
        self.config_file_class = "user" if exists else "default"
        self.config_path = user if self.config_file_class == "user" else self.config_root / "fox_config.yaml"

    def _load_file(self, path: Path) -> tuple[FoxConfig, dict, bytes | None, bool]:
        defaults = FoxConfig()
        try:
            source = _read_bytes(path)
        except OSError as exc:
            logger.warning("Cannot read config %s: %s; using defaults without overwriting it", path, exc)
            return defaults, {}, None, True

        invalid = False
        data: dict = {}
        if source is None:
            logger.info("Creating default config: %s", path)
        else:
            loader = None
            try:
                # Passing bytes lets PyYAML recognize UTF-8/UTF-16 BOMs from
                # Windows editors, while rejecting undecodable content safely.
                loader = _ConfigLoader(source)
                document = loader.get_single_data()
                if document is not None and not isinstance(document, dict):
                    raise ValueError("expected a YAML mapping")
                data = document if isinstance(document, dict) else {}
                if loader.duplicate_lines:
                    invalid = True
                    logger.warning("Duplicate config keys in %s at lines %s; keeping the last values",
                                   path, ", ".join(map(str, loader.duplicate_lines)))
            except (yaml.YAMLError, UnicodeError, ValueError, RecursionError) as exc:
                invalid = True
                # Parser/validation exception strings can include entire lines
                # of private user data. Report location and category instead.
                mark = getattr(exc, "problem_mark", None)
                location = f" at line {mark.line + 1}" if mark is not None else ""
                logger.warning("Invalid config document %s%s (%s); using defaults",
                               path, location, type(exc).__name__)
            finally:
                if loader is not None:
                    loader.dispose()

        fields = type(defaults).model_fields
        known = {name: data[name] for name in fields if name in data}
        extra = {name: value for name, value in data.items() if name not in fields}
        try:
            config = FoxConfig.model_validate(known)
        except ValidationError as exc:
            invalid = True
            # Defaults live only in the schema. Adding a field there is enough
            # to populate every older file on its next launch.
            for name in dict.fromkeys(error["loc"][0] for error in exc.errors(include_input=False)):
                known.pop(name, None)
                logger.warning("Invalid config field %s in %s; using its default", name, path)
            config = FoxConfig.model_validate(known)

        normalized = config.model_dump(mode="json")
        missing = [name for name in fields if name not in data]
        changed = [name for name in fields if name in data and
                   (type(data[name]) is not type(normalized[name]) or data[name] != normalized[name])]
        if missing and source is not None:
            logger.info("Adding missing config fields in %s: %s", path, ", ".join(missing))
        if changed:
            logger.info("Normalizing config fields in %s: %s", path, ", ".join(changed))
        blocked = False
        if source is None or invalid or missing or changed or not _launcher_readable(source, config):
            try:
                source = _write_config(path, {**normalized, **extra}, source)
                logger.info("Saved reconciled config: %s", path)
            except (OSError, yaml.YAMLError, RecursionError) as exc:
                blocked = True
                logger.warning("Could not persist config %s (%s); using reconciled values in memory; "
                               "the original file is unchanged", path, type(exc).__name__)
        return config, extra, source, blocked

    def load(self) -> FoxConfig:
        with self._lock:
            # Re-select at load time so a manually added/removed user file is
            # honored even when this manager was constructed earlier.
            self._select_path()
            default_path = self.config_root / "fox_config.yaml"
            result = self._load_file(default_path)
            if self.config_file_class == "user":
                result = self._load_file(self.config_path)
            self.config, self._extra, self._source, self._write_blocked = result
            return self.config

    def save(self) -> None:
        with self._lock:
            if self.config is None:
                raise RuntimeError("Configuration has not been loaded.")
            self._save(self.config)

    def _save(self, config: FoxConfig) -> None:
        if self._write_blocked:
            raise OSError(f"Config could not be safely persisted; reload before saving: {self.config_path}")
        self._source = _write_config(
            self.config_path, {**config.model_dump(mode="json"), **self._extra}, self._source,
        )

    def _replace(self, candidate: FoxConfig) -> FoxConfig:
        self._save(candidate)
        self.config = candidate
        return candidate

    def update(self, **changes) -> FoxConfig:
        with self._lock:
            if self.config is None:
                raise RuntimeError("Call load() first.")
            forbidden = self.IMMUTABLE_FIELDS & changes.keys()
            if forbidden:
                raise PermissionError(
                    f"Cannot modify runtime-protected fields: {', '.join(sorted(forbidden))}"
                )
            candidate = FoxConfig.model_validate({**self.config.model_dump(), **changes})
            return self._replace(candidate)

    def reset(self) -> FoxConfig:
        with self._lock:
            if self.config is None:
                self.load()
            return self._replace(FoxConfig())
