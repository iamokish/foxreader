"""Loading, unloading and gating the local MTL model.

Every one of these models is big enough that a machine which cannot hold it
fails badly rather than slowly: on the CPU the process starts swapping or is
killed outright, and on CUDA the allocator raises from somewhere deep inside the
framework with a message no reader should have to interpret. So the free memory
is measured against the model's recorded requirement before anything is
constructed, and a shortfall comes back as a plain sentence the frontend can
show (see `check_memory` and `InsufficientMemoryError`).

A device that is too small is a refusal, not a downgrade. Silently moving a
model from the GPU the user pinned onto the CPU makes the reader look hung
rather than fast, and hides the one fact worth telling them.

Nothing here may raise on import: torch is an optional extra (see the ``cpu`` /
``cu126`` / ``macos`` groups in pyproject.toml), llama-cpp-python is another (the
``gguf`` group), and psutil can be absent from a trimmed environment. A probe
that cannot answer is treated as "unknown", which allows the load — refusing on
the strength of a failed measurement would lock a working machine out of its own
models. The translator modules below are imported eagerly to register them, so
each of them has to keep its own optional dependency out of module scope.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fox_reader.constants import mtl_model
from fox_reader.device import CPU, CUDA, MPS, FOX_DEVICE
from fox_reader.device import configure_threads
from fox_reader.device import probe as probe_devices
from fox_reader.device import torch_module
from fox_reader.settings import SUPPORTED_MTL_LANGUAGES, SettingsManager
from fox_reader.translate.characters import normalize_character_info, normalize_character_links, normalise_meta_id
from fox_reader.translate.context import (
    model_supports_characters,
    model_supports_context,
    normalize_context,
)
from fox_reader.translate.machine_translation.gemma_e4b_q6_llamacpp import GemmaE4BQ6Translator
from fox_reader.translate.machine_translation.gemma_e4b_q8_llamacpp import GemmaE4BQ8Translator
from fox_reader.translate.machine_translation.vntl_llama3_8b_llamacpp import VntlLlamaTranslator


logger = logging.getLogger(__name__)

_BYTES_PER_MIB = 1024 * 1024

#: Set this to skip the memory gate. The measurement can be pessimistic —
#: `psutil.virtual_memory().available` on Windows ignores the page file, for
#: instance — and no check should be able to permanently lock someone out of a
#: model that does in fact run on their machine.
IGNORE_MEMORY_CHECK_ENV = "FOX_READER_IGNORE_MTL_MEMORY"

# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------

_psutil_probed = False
_psutil_module: Any | None = None


def _psutil() -> Any | None:
    """The psutil module, or None when it cannot be used. Probed once."""
    global _psutil_probed, _psutil_module

    if _psutil_probed:
        return _psutil_module

    _psutil_probed = True

    try:
        import psutil
    except ImportError as exc:
        logger.info("psutil is not installed, cannot measure free RAM: %s", exc)
        return None
    except Exception as exc:
        logger.warning("psutil could not be imported: %s", exc)
        return None

    _psutil_module = psutil
    return _psutil_module


def _cuda_index(device: str) -> int | None:
    """The ordinal in a ``cuda:N`` device string, or None if it is not one."""
    text = (device or "").strip().lower()

    if text == CUDA:
        return 0

    prefix = f"{CUDA}:"

    if text.startswith(prefix) and text[len(prefix):].isdigit():
        return int(text[len(prefix):])

    return None


def free_ram_bytes() -> int | None:
    """Memory that can be handed out without swapping, or None if unknown."""
    psutil = _psutil()

    if psutil is None:
        return None

    try:
        return int(psutil.virtual_memory().available)
    except Exception as exc:
        logger.warning("Could not read the free system memory: %s", exc)
        return None


def free_vram_bytes(index: int) -> int | None:
    """Free VRAM on one CUDA device, or None if it could not be asked.

    `mem_get_info` reports what the driver has left, so another process holding
    the card is counted — which is the number that decides whether this load
    will succeed.
    """
    torch = torch_module()

    if torch is None:
        return None

    try:
        free, _total = torch.cuda.mem_get_info(index)
        return int(free)
    except Exception as exc:
        logger.warning("Could not read the free VRAM on cuda:%d: %s", index, exc)
        return None


def _gpu_name(index: int) -> str | None:
    try:
        gpu = probe_devices().gpu(index)
    except Exception:
        return None

    return gpu.name if gpu is not None else None


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryCheck:
    """One model measured against one device.

    `measured` separates "there is enough" from "nobody could tell": both leave
    `fits` true, but only the first is a clean answer, and the page should say
    so differently.
    """

    model: str
    model_name: str
    device: str
    #: Which pool `available_mib` came from: "vram" or "ram".
    pool: str
    required_mib: int
    available_mib: int | None
    fits: bool
    measured: bool
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


class InsufficientMemoryError(RuntimeError):
    """Raised instead of loading a model that does not fit.

    The message is written for the reader UI: both ``/ml/control/load`` and
    ``/translate/ml`` pass ``str(exc)`` straight through to the page, so it has
    to name the model, both figures, and what the user can do about it.
    """

    def __init__(self, check: MemoryCheck) -> None:
        super().__init__(check.message)
        self.check = check


def effective_device(model_id: str, device: str | None = None) -> str:
    """The device `model_id` will really load on, given the configured one.

    All remaining translators are GGUF via llama.cpp, which has a Metal
    backend — so every model runs on the configured device as-is, including
    MPS. ``model_id`` is kept so callers do not need to change if a future
    model ever needs remapping again.
    """
    _ = model_id
    target = (device if device is not None else FOX_DEVICE.translator) or CPU
    target = target.strip().lower() or CPU

    return target


def check_memory(model_id: str, device: str | None = None) -> MemoryCheck | None:
    """Whether `model_id` fits on `device` (the configured one by default).

    None means there is nothing to check — an unknown model id, or one with no
    recorded requirement — and the caller should go ahead and load.
    """
    model = mtl_model(model_id)

    if model is None:
        logger.debug("No metadata for MTL model %r; loading unchecked", model_id)
        return None

    target = effective_device(model_id, device)
    index = _cuda_index(target)
    name = str(model.get("name") or model_id)

    if index is not None:
        pool = "vram"
        required = model.get("gpu_vram_mib")
        free = free_vram_bytes(index)
        card = _gpu_name(index)
        where = f"GPU {index}" + (f" ({card})" if card else "")
    elif target == MPS:
        # Metal shares one pool with everything else on the machine, so the
        # figure to measure is the system's — against the GPU requirement,
        # because that is the footprint the weights will have.
        pool = "ram"
        required = model.get("gpu_vram_mib")
        free = free_ram_bytes()
        where = "the Apple GPU"
    else:
        pool = "ram"
        required = model.get("cpu_ram_mib")
        free = free_ram_bytes()
        where = "the CPU"

    try:
        required_mib = int(required or 0)
    except (TypeError, ValueError):
        required_mib = 0

    if required_mib <= 0:
        logger.debug("No memory requirement recorded for %r; loading unchecked", model_id)
        return None

    unit = "VRAM" if pool == "vram" else "RAM"

    if free is None:
        return MemoryCheck(
            model=model_id,
            model_name=name,
            device=target,
            pool=pool,
            required_mib=required_mib,
            available_mib=None,
            fits=True,
            measured=False,
            message=(
                f"{name} needs about {required_mib} MiB of {unit} on {where}. "
                f"The free {unit} could not be measured here, so it will be "
                "loaded without checking."
            ),
        )

    available_mib = int(free) // _BYTES_PER_MIB
    fits = available_mib >= required_mib

    if fits:
        message = (
            f"{name} needs {required_mib} MiB of {unit} on {where}; "
            f"{available_mib} MiB is free."
        )
    elif pool == "vram":
        message = (
            f"{name} needs {required_mib} MiB of VRAM on {where}, but only "
            f"{available_mib} MiB is free. Close whatever else is using the "
            "GPU, choose another device under Settings → Compute Devices, "
            "or pick a model that fits."
        )
    else:
        message = (
            f"{name} needs {required_mib} MiB of RAM on {where}, but only "
            f"{available_mib} MiB is available. Close some applications, or "
            "pick a smaller model under Settings."
        )

    return MemoryCheck(
        model=model_id,
        model_name=name,
        device=target,
        pool=pool,
        required_mib=required_mib,
        available_mib=available_mib,
        fits=fits,
        measured=True,
        message=message,
    )


def memory_check_disabled() -> bool:
    return os.environ.get(IGNORE_MEMORY_CHECK_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class LocalMTLManager:
    def __init__(self, mtl_dir: Path, settings: SettingsManager | None = None):
        self.mtl_dir = mtl_dir
        self.settings = settings
        self.model = None
        self.current_lang = None
        self.available_translators = [
            GemmaE4BQ8Translator,
            GemmaE4BQ6Translator,
            VntlLlamaTranslator,
        ]
        # load / unload / translate all run in the threadpool now, so they can
        # overlap. Unloading while a generation is in flight would free the
        # weights under it, and two concurrent loads would each build a model and
        # leak one. Reentrant because load() calls unload() and translate() calls
        # load().
        self._lock = threading.RLock()

    @property
    def device(self) -> str:
        """The configured translator device, read late.

        Read on every call rather than captured in __init__: this manager is
        built while the app starts up, and `fox_reader.device.configure` may
        not have applied the saved selection yet.
        """
        return FOX_DEVICE.translator or CPU

    def unload(self):
        """Free the loaded translator, serialised against load and translate."""
        with self._lock:
            self._unload_locked()

    def _unload_locked(self):
        if self.model is not None:
            # Every remaining translator is a GGUF model owning memory Python
            # cannot see -- a llama.cpp context in a native allocation or in
            # VRAM -- so it releases itself. Deleting the handle without
            # closing it would leak the context and everything under it.
            release = getattr(self.model, "unload", None)

            if callable(release):
                try:
                    release()
                except Exception as e:
                    logger.debug("Translator-owned unload failed: %s", e)

        self.model = None
        self.current_lang = None
        gc.collect()

        # Returning the freed blocks to the driver is what makes the next
        # `check_memory` tell the truth, so it is worth doing even though torch
        # would reuse them itself.
        torch = torch_module()

        if torch is None:
            return

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("Failed to empty the CUDA cache: %s", e)

        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            logger.debug("Failed to empty the MPS cache: %s", e)

    def status(self) -> dict:
        """What is loaded right now, for the reader page after a refresh.

        The frontend's MTL switch resets on reload while this manager keeps
        the weights, so the page asks on startup and re-checks the box when
        something is still here. Never raises: every accessor is guarded, so
        a half-built model reads as whatever could be answered, and "nothing
        loaded" is the honest default.
        """
        with self._lock:
            model = self.model

            if model is None:
                return {"loaded": False, "lang": None, "model_id": None, "languages": []}

            try:
                model_id = model.model_id()
            except Exception:
                model_id = None

            try:
                raw_langs = model.model_langs() or []
            except Exception:
                raw_langs = []

            languages: list[str] = []

            try:
                for item in raw_langs:
                    lang = str(item).strip().lower()
                    if lang and lang not in languages:
                        languages.append(lang)
            except Exception:
                languages = []

            return {
                "loaded": True,
                "lang": self.current_lang,
                "model_id": model_id,
                "languages": languages,
            }

    def _compatible_translators(self, lang: str):
        lang = lang.strip().lower()
        return [
            translator_cls
            for translator_cls in self.available_translators
            if lang in translator_cls.model_langs()
        ]

    def _configured_model_id(self, lang: str) -> str | None:
        if self.settings is None:
            return None
        return self.settings.get_default_model(lang)

    def _apply_thread_preference(self) -> None:
        """Hand the saved CPU thread count to `fox_reader.device`.

        Read here rather than once at startup because llama.cpp binds its
        threads when it builds a context: doing it on every load is what lets
        the number be changed on the settings page without a restart. All
        remaining translators are GGUF and read it.
        """
        if self.settings is None:
            return

        try:
            configure_threads(self.settings.get_mtl_threads())
        except Exception as exc:
            logger.debug("Could not read the saved translator thread count: %s", exc)

    def _ensure_fits(self, model_id: str) -> None:
        """Refuse the load when the model will not fit on its device."""
        check = check_memory(model_id, self.device)

        if check is None:
            return

        if check.fits:
            # An unmeasurable machine still loads, but says so louder: the next
            # thing in the log may well be an allocator failure.
            if check.measured:
                logger.info("Memory check — %s", check.message)
            else:
                logger.warning("Memory check — %s", check.message)
            return

        if memory_check_disabled():
            logger.warning(
                "Loading %s anyway, %s is set — %s",
                model_id,
                IGNORE_MEMORY_CHECK_ENV,
                check.message,
            )
            return

        logger.error("Refusing to load %s — %s", model_id, check.message)
        raise InsufficientMemoryError(check)

    def memory_report(self) -> dict:
        """Pre-flight for the reader page: does every configured model fit?

        Reported per model rather than per language, since the same model
        usually serves several. `ok` is false only for a measured shortfall — a
        machine that could not be measured is not something to nag about.
        """
        device = self.device
        defaults: dict[str, str | None] = {}

        if self.settings is not None:
            try:
                defaults = self.settings.get_defaults()
            except Exception as exc:
                logger.debug("Could not read the configured MTL models: %s", exc)

        languages: dict[str, list[str]] = {}

        for language in SUPPORTED_MTL_LANGUAGES:
            model_id = defaults.get(language)

            if model_id:
                languages.setdefault(model_id, []).append(language)

        checks: list[dict] = []

        for model_id, langs in languages.items():
            check = check_memory(model_id, device)

            if check is None:
                continue

            row = check.as_dict()
            row["languages"] = langs
            checks.append(row)

        issues = [row for row in checks if not row["fits"]]

        return {
            "enabled": True,
            "device": device,
            "ok": not issues,
            "checks": checks,
            "issues": issues,
        }

    def load(self, lang):
        """Make `lang`'s configured model current, serialised against unload."""
        with self._lock:
            self._load_locked(lang)

    def _load_locked(self, lang):
        lang = lang.strip().lower()

        compatible = self._compatible_translators(lang)
        if not compatible:
            raise ValueError(f"Unsupported language: {lang}")

        configured_id = self._configured_model_id(lang)

        # A changed settings.yaml value must take effect without restarting.
        # If the current language is loaded with the same model, keep it;
        # otherwise unload the old model before loading the new one.
        if (
            lang == self.current_lang
            and self.model is not None
            and configured_id is not None
            and self.model.model_id() == configured_id
        ):
            return

        if configured_id is None:
            if self.model is not None and lang == self.current_lang:
                self.unload()
            raise ValueError(
                f"No local MTL model is configured for language: {lang}"
            )

        target_class = next(
            (
                translator_cls
                for translator_cls in compatible
                if translator_cls.model_id() == configured_id
            ),
            None,
        )

        if target_class is None:
            raise ValueError(
                f"Configured model '{configured_id}' does not support language: {lang}"
            )

        if self.model is not None:
            if self.model.model_id() == target_class.model_id():
                self.current_lang = lang
                return
            self.unload()

        # Deliberately after the unload above: whatever was loaded was holding
        # the memory this measurement is about.
        self._ensure_fits(target_class.model_id())
        self._apply_thread_preference()

        self.model = target_class(self.mtl_dir)
        self.current_lang = lang

    def translate(
        self,
        lang: str,
        text: str,
        context: object = None,
        character_info: object = None,
        context_character_links: object = None,
        meta_id: object = None,
    ):
        # Held across the generation, not just the load: releasing it early would
        # let a concurrent /ml/control/unload free the weights mid-inference.
        with self._lock:
            self._load_locked(lang)

            if self.model is None:
                return ""

            # Context is only ever rendered by models that ask for it. Anything
            # else translates exactly as before, context or no context.
            try:
                pairs = normalize_context(context)
            except Exception:
                pairs = []

            try:
                model_id = self.model.model_id()
            except Exception:
                model_id = ""

            try:
                ctx_supported = model_supports_context(model_id)
            except Exception:
                ctx_supported = False

            try:
                char_supported = model_supports_characters(model_id)
            except Exception:
                char_supported = False

            # Characters resolve here so the translator always sees validated
            # data. The request carries a roster snapshot; the in-memory store
            # is only the UI's roster (CRUD), never an implicit input -- so a
            # caller that sends nothing gets no characters, even when the
            # store holds some (e.g. the Characters switch is off).
            characters: dict[str, dict] = {}
            links: list[str | None] = []
            speaker: str | None = None
            if char_supported:
                try:
                    characters = normalize_character_info(character_info)
                except Exception:
                    characters = {}
                try:
                    links = normalize_character_links(context_character_links, len(pairs))
                except Exception:
                    links = [None] * len(pairs)
                try:
                    speaker = normalise_meta_id(meta_id)
                except Exception:
                    speaker = None

                if pairs or characters or speaker:
                    try:
                        return self.model.translate(
                            text,
                            context=pairs,
                            lang=lang,
                            character_info=list(characters.values()),
                            context_character_links=links,
                            meta_id=speaker,
                        )
                    except TypeError:
                        # A supporting model with an older signature: translate
                        # without rather than fail the whole request.
                        pass

            if ctx_supported and pairs:
                try:
                    return self.model.translate(text, context=pairs, lang=lang)
                except TypeError:
                    # A supporting model with an older signature: translate
                    # without rather than fail the whole request.
                    pass

            return self.model.translate(text)
