from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .constants import mtl_model_downloaded, mtl_models_for_language
from .device import AUTO as AUTO_DEVICE
from .device import AUTO_THREADS
from .device import SLOTS as DEVICE_SLOTS
from .device import normalize_id as normalize_device_id
from .device import normalize_threads as normalize_thread_count

logger = logging.getLogger(__name__)


SUPPORTED_MTL_LANGUAGES = ("japanese", "korean", "chinese")
DEFAULT_SETTINGS_FILENAME = "settings.yaml"


def _normalize_ocr_engine(value: object) -> str:
    """A saved OCR engine id, falling back to classic PaddleOCR."""
    from fox_reader.constants import DEFAULT_OCR_ENGINE, SUPPORTED_OCR_ENGINES

    text = str(value or "").strip().lower() or DEFAULT_OCR_ENGINE
    if text in SUPPORTED_OCR_ENGINES:
        return text
    raise ValueError(
        f"OCR engine must be one of: {', '.join(SUPPORTED_OCR_ENGINES)}"
    )


def _normalize_language(value: Any, field_name: str = "language") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    value = value.strip().lower()

    if value not in SUPPORTED_MTL_LANGUAGES:
        raise ValueError(
            f"{field_name} must be one of: "
            + ", ".join(SUPPORTED_MTL_LANGUAGES)
        )

    return value


class MTLDefaults(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    japanese: str | None = None
    korean: str | None = None
    chinese: str | None = None

    @field_validator("japanese", "korean", "chinese", mode="before")
    @classmethod
    def normalize_model_id(cls, value: Any) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError("MTL model ID must be a string or null")

        value = value.strip()
        return value or None


class DeviceSettings(BaseModel):
    """Which device each model should run on, and how much CPU to give it.

    Stored as ids rather than as resolved device strings: `auto` has to mean
    "whatever is best on this machine today", and a pinned `cuda:1` has to
    survive a week with the card removed. Both are questions for
    `fox_reader.device` at startup, not for this file. `mtl_threads` is the same
    idea for the CPU — `auto` or a count, resolved against the cores that are
    actually there.

    Validation here is grammar only, for the same reason — a settings.yaml
    carried over from a two-GPU desktop must still load on a laptop, with the
    fallback to the CPU happening at resolve time and the preference left as
    written.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    paddleocr: str = AUTO_DEVICE
    #: PaddleOCR-VL transcriber (llama.cpp GGUF). Separate from `paddleocr` so
    #: the two OCR engines can sit on different devices; older settings files
    #: predate it (see `SettingsManager.load`, which inherits the paddleocr
    #: choice once).
    paddleocr_vl: str = AUTO_DEVICE
    bubble: str = AUTO_DEVICE
    translator: str = AUTO_DEVICE
    #: Text Seg cleaning network. Separate from `paddleocr` so OCR and cleaning
    #: can sit on different devices; older settings files predate it (see
    #: `SettingsManager.load`, which inherits the paddleocr choice once).
    textseg: str = AUTO_DEVICE

    #: Threads for the GGUF translators, which pick their own: `auto` for the
    #: heuristic in `fox_reader.device.thread_count`, or a number to pin it.
    mtl_threads: int | str = AUTO_THREADS

    @field_validator("paddleocr", "paddleocr_vl", "bubble", "translator", "textseg", mode="before")
    @classmethod
    def normalize_device(cls, value: Any) -> str:
        if value is None:
            return AUTO_DEVICE

        try:
            return normalize_device_id(value)
        except ValueError as exc:
            # A malformed id is not worth refusing to start over; the whole
            # settings file would be discarded with it.
            logger.warning("%s -- using 'auto'", exc)
            return AUTO_DEVICE

    @field_validator("mtl_threads", mode="before")
    @classmethod
    def normalize_threads(cls, value: Any) -> int | str:
        try:
            return normalize_thread_count(value)
        except ValueError as exc:
            logger.warning("%s -- using 'auto'", exc)
            return AUTO_THREADS


class FoxSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    mtl_defaults: MTLDefaults = Field(default_factory=MTLDefaults)
    devices: DeviceSettings = Field(default_factory=DeviceSettings)
    deepl_api_token: str = ""
    jpdb_api_token: str = ""
    #: Which text-recognition engine OCR uses. ``paddleocr`` is the classic
    #: PP-OCRv6 pipeline; ``paddleocr-vl`` is the GGUF vision-language model.
    #: Default is classic; VL is only honoured when its package and weights
    #: are present (see OCRService fallback), so an old or hand-edited file
    #: can never break startup.
    ocr_engine: str = Field(default="paddleocr")

    @field_validator("ocr_engine", mode="before")
    @classmethod
    def normalize_ocr_engine(cls, value: object) -> str:
        from fox_reader.constants import DEFAULT_OCR_ENGINE

        if value is None or (isinstance(value, str) and not value.strip()):
            return DEFAULT_OCR_ENGINE
        try:
            return _normalize_ocr_engine(value)
        except ValueError as exc:
            logger.warning("%s -- using %r", exc, DEFAULT_OCR_ENGINE)
            return DEFAULT_OCR_ENGINE

    @field_validator(
        "deepl_api_token",
        "jpdb_api_token",
        mode="before",
    )
    @classmethod
    def normalize_api_token(cls, value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("API token must be a string")
        return value.strip()


class SettingsManager:
    """Persistent settings manager for settings.yaml.

    Settings are loaded and saved independently from FoxConfig. Runtime
    consumers read this manager directly, so changes are effective without
    restarting the application.
    """

    def __init__(self, config_root: Path):
        self.config_root = Path(config_root)
        self.settings_path = self.config_root / DEFAULT_SETTINGS_FILENAME
        self.settings: FoxSettings | None = None

    def load(self) -> FoxSettings:
        self.config_root.mkdir(parents=True, exist_ok=True)

        if not self.settings_path.exists():
            self.settings = FoxSettings()
            self.save()
            return self.settings

        try:
            with self.settings_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # One-time migration for files written before the Text Seg slot
            # existed: inherit the paddleocr choice, which is what cleaning
            # used to run on. Without this, upgrading would silently move an
            # install that had pinned OCR to the CPU back onto the GPU for
            # cleaning. Files that already carry `textseg` are left alone, and
            # the migrated value is persisted so the two slots can diverge
            # from here on.
            #
            # The same applies to the PaddleOCR-VL slot, which used to follow
            # the paddleocr choice at runtime: inherit it once so an upgrade
            # neither strands the VL engine on another device nor resets a
            # deliberate pin.
            migrated = False
            if isinstance(data, dict):
                devices_raw = data.get("devices")
                if isinstance(devices_raw, dict):
                    if "bubble" not in devices_raw and "yolo" in devices_raw:
                        # Pre-rename device key: carry a saved `yolo` choice
                        # over once so a pinned bubble device survives the
                        # upgrade instead of resetting to `auto`.
                        try:
                            devices_raw["bubble"] = normalize_device_id(devices_raw.get("yolo"))
                        except ValueError:
                            devices_raw["bubble"] = AUTO_DEVICE
                        migrated = True
                    if "textseg" not in devices_raw:
                        paddle_raw = devices_raw.get("paddleocr", AUTO_DEVICE)
                        try:
                            devices_raw["textseg"] = normalize_device_id(paddle_raw)
                        except ValueError:
                            devices_raw["textseg"] = AUTO_DEVICE
                        migrated = True
                    if "paddleocr_vl" not in devices_raw:
                        paddle_raw = devices_raw.get("paddleocr", AUTO_DEVICE)
                        try:
                            devices_raw["paddleocr_vl"] = normalize_device_id(paddle_raw)
                        except ValueError:
                            devices_raw["paddleocr_vl"] = AUTO_DEVICE
                        migrated = True

            self.settings = FoxSettings.model_validate(data)
            if migrated:
                self.save()

        except (ValidationError, yaml.YAMLError, OSError) as exc:
            logger.warning("Failed to load settings: %s", exc)
            logger.warning("Restoring default settings.")
            self.settings = FoxSettings()
            self.save()

        return self.settings

    def save(self) -> None:
        if self.settings is None:
            raise RuntimeError("Settings have not been loaded.")

        with self.settings_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.settings.model_dump(mode="json"),
                f,
                sort_keys=False,
            )

    def _ensure_loaded(self) -> FoxSettings:
        if self.settings is None:
            return self.load()
        return self.settings

    def _replace(self, **changes: Any) -> FoxSettings:
        """Rebuild the settings with some blocks swapped out, and persist.

        Going through a full dump matters: rebuilding `FoxSettings` from one
        block alone would quietly reset every other block to its default, which
        is how the API tokens used to disappear whenever an MTL default was
        picked.
        """
        settings = self._ensure_loaded()

        self.settings = FoxSettings.model_validate({**settings.model_dump(), **changes})
        self.save()

        return self.settings

    def get_default_model(self, language: str) -> str | None:
        language = _normalize_language(language)
        settings = self._ensure_loaded()
        return getattr(settings.mtl_defaults, language)

    def get_defaults(self) -> dict[str, str | None]:
        settings = self._ensure_loaded()
        return {
            language: getattr(settings.mtl_defaults, language)
            for language in SUPPORTED_MTL_LANGUAGES
        }

    def get_api_tokens(self) -> dict[str, str]:
        settings = self._ensure_loaded()
        return {
            "deepl_api_token": settings.deepl_api_token,
            "jpdb_api_token": settings.jpdb_api_token,
        }

    def get_translate_status(self) -> dict[str, bool]:
        """Which token-gated translators may be offered, without leaking keys.

        Tokens are validated on save (JPDB ping, DeepL usage), so a non-empty
        value means "valid when it was saved". The reader page gates its
        DeepL/JPDB buttons on this rather than re-pinging on every load.
        """
        tokens = self.get_api_tokens()
        return {
            "deepl": bool((tokens.get("deepl_api_token") or "").strip()),
            "jpdb": bool((tokens.get("jpdb_api_token") or "").strip()),
        }

    def update_api_tokens(self, **changes: str | None) -> FoxSettings:
        allowed = {
            "deepl_api_token",
            "jpdb_api_token",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown API token setting(s): {', '.join(sorted(unknown))}")

        values: dict[str, Any] = {}
        for key, value in changes.items():
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            values[key] = value.strip()

        return self._replace(**values)

    def get_devices(self) -> dict[str, str]:
        """The saved device id per slot — `auto` unless someone pinned one."""
        settings = self._ensure_loaded()

        # `getattr` with a default on purpose: a settings object built from an
        # older file (or an older FoxSettings in memory) may not have the
        # textseg attribute yet — answer `auto` rather than raising.
        return {slot: getattr(settings.devices, slot, AUTO_DEVICE) for slot in DEVICE_SLOTS}

    def get_mtl_threads(self) -> int | str:
        """The saved GGUF thread count: `auto`, or a pinned number."""
        settings = self._ensure_loaded()

        return settings.devices.mtl_threads

    def get_ocr_engine(self) -> str:
        """The saved OCR engine id, defaulting to classic PaddleOCR."""
        from fox_reader.constants import DEFAULT_OCR_ENGINE

        settings = self._ensure_loaded()
        # getattr with a default: objects built from older files (or older
        # FoxSettings in memory) may not carry the field yet.
        engine = getattr(settings, "ocr_engine", DEFAULT_OCR_ENGINE)
        try:
            return _normalize_ocr_engine(engine)
        except ValueError:
            return DEFAULT_OCR_ENGINE

    def update_ocr_engine(self, value: object) -> FoxSettings:
        """Select the OCR engine. Takes effect on next launch.

        Strict like update_devices: this is a user picking from a list, so
        an unknown id is an error rather than a silent fallback. Whether the
        selected engine can actually run (llama_cpp + weights for VL) is a
        runtime question answered by describe_ocr_engines, not here.
        """
        engine = _normalize_ocr_engine(value)
        return self._replace(ocr_engine=engine)

    def update_devices(self, changes: dict[str, Any]) -> FoxSettings:
        """Pin (or un-pin, with `auto`) the device for one or more slots.

        Unlike the rest of the settings this one only takes effect on the next
        launch: the OCR engines bake their device in when they are constructed,
        and the translator does the same when it loads a model.
        """
        settings = self._ensure_loaded()
        devices = settings.devices.model_dump()

        unknown = set(changes) - set(DEVICE_SLOTS)
        if unknown:
            raise ValueError(f"Unknown device slot(s): {', '.join(sorted(unknown))}")

        for slot, device_id in changes.items():
            # Raise here rather than logging: this is a user pressing a button,
            # so a bad value is worth an error message instead of a silent
            # 'auto'. Loading a file is the lenient path, not this one.
            devices[slot] = normalize_device_id(device_id)

        return self._replace(devices=devices)

    def update_mtl_threads(self, value: Any) -> FoxSettings:
        """Pin the GGUF translator's thread count, or hand it back to `auto`.

        Strict for the same reason `update_devices` is: this is a user picking
        from a list, not a file being read, so a value that is not a count is
        worth an error rather than a silent 'auto'.

        Not clamped to this machine's cores — a number larger than the box can
        run is narrowed when the model loads, the same way a pinned `cuda:5`
        survives in the file and resolves to the CPU.
        """
        settings = self._ensure_loaded()
        devices = settings.devices.model_dump()
        devices["mtl_threads"] = normalize_thread_count(value)

        return self._replace(devices=devices)

    def update_default_model(self, language: str, model_id: str | None) -> FoxSettings:
        language = _normalize_language(language)

        if model_id is not None:
            model_id = model_id.strip() or None

        settings = self._ensure_loaded()
        defaults = settings.mtl_defaults.model_dump()
        defaults[language] = model_id

        return self._replace(mtl_defaults=defaults)

    def update_defaults(self, changes: dict[str, str | None]) -> FoxSettings:
        settings = self._ensure_loaded()
        defaults = settings.mtl_defaults.model_dump()

        for language, model_id in changes.items():
            language = _normalize_language(language)

            if model_id is not None:
                model_id = str(model_id).strip() or None

            defaults[language] = model_id

        return self._replace(mtl_defaults=defaults)

    def resolve_mtl_defaults(self, mtl_dir: Path | str) -> dict[str, str | None]:
        """Point every language at a model whose weights are actually present.

        A default only means anything once the model is downloaded, so it is
        settled against `mtl_dir` rather than trusted from the file. A fresh
        install writes nulls before anything has been fetched, and a model can
        be deleted from under a value that was valid when it was written;
        neither should leave the translator raising "no local MTL model is
        configured" for a language that has one sitting on disk.

        Per language: keep the saved choice when it is still downloaded and
        still supports that language, otherwise take the first downloaded model
        that does, otherwise fall back to null.

        Called at startup so the file is correct before anything reads it, and
        again whenever the settings page is built so a model downloaded (or
        deleted) since launch is picked up. Writes only when something actually
        changed, so a plain GET does not rewrite settings.yaml every time.
        """
        settings = self._ensure_loaded()
        saved = settings.mtl_defaults.model_dump()
        resolved = dict(saved)

        for language in SUPPORTED_MTL_LANGUAGES:
            downloaded = [
                model["id"]
                for model in mtl_models_for_language(language)
                if mtl_model_downloaded(mtl_dir, model)
            ]

            if saved.get(language) in downloaded:
                continue

            resolved[language] = downloaded[0] if downloaded else None

        changed = {
            language: resolved[language]
            for language in SUPPORTED_MTL_LANGUAGES
            if resolved[language] != saved.get(language)
        }

        if not changed:
            return resolved

        logger.info(
            "Resolved MTL default model(s) against what is downloaded: %s",
            ", ".join(
                f"{language}: {saved.get(language)!r} -> {model_id!r}"
                for language, model_id in changed.items()
            ),
        )

        self._replace(mtl_defaults=resolved)

        return resolved
