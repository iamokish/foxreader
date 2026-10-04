"""PaddleOCR-VL-1.6 GGUF text recognition via llama.cpp.

The classic pipeline in :mod:`fox_reader.ocr` detects lines and then reads
them. This module is the alternative selected under Settings -> Text
Recognition: a single vision-language model
(``PaddlePaddle/PaddleOCR-VL-1.6-GGUF``) that transcribes a bubble crop
directly, with no detector involved.

Robustness rules, in order:

* importing this module must never require ``llama_cpp``. The ``gguf`` extra
  is optional, so every ``llama_cpp`` import happens lazily inside a function
  and a missing package reads as "unavailable", not an ImportError;
* constructing :class:`PaddleOCRVLEngine` must fail with a plain sentence
  when the package is missing, the weights are missing, or the GPU build
  cannot satisfy the configured device -- never with a bare traceback the
  settings page cannot show;
* the ``llama-cpp-python`` vision API has moved between releases
  (``clip_model_path`` + chat handler vs the newer ``mmproj_path`` generic
  MTMD handler). All known spellings are tried so one pinned version cannot
  lock the engine out.
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any, Iterator

from fox_reader.constants import (
    PADDLEOCR_VL_MMPROJ_FILE,
    PADDLEOCR_VL_MODEL_DIR,
    PADDLEOCR_VL_MODEL_FILE,
)
from fox_reader.device import CPU, CUDA, FOX_DEVICE, MPS
from fox_reader.utils import MODELS_DIR

logger = logging.getLogger(__name__)

#: Prompt the upstream model card uses for transcription. Kept verbatim:
#: the GGUF's chat template was tuned around it, and a longer instruction
#: ("transcribe exactly, no explanations") drifts the short bubble crops.
OCR_PROMPT = "OCR:"

#: Generation budget for a bubble crop. Crops are small; 512 tokens is ample
#: and keeps a runaway completion from stalling the OCR queue.
MAX_TOKENS = 512

#: Context window. Matches the upstream ``llama-server -c 8192`` guidance: a
#: 1024px crop encodes to ~160 image tokens, and anything smaller risks
#: truncating the embeddings on tall bubbles.
N_CTX = 8192

VERBOSE_ENV = "FOX_READER_VL_VERBOSE"

MISSING_PACKAGE = (
    "PaddleOCR-VL needs llama-cpp-python, which is not installed. "
    "Install it with the 'gguf' extra (uv pip install .[gguf]), or use "
    "the classic PaddleOCR engine under Settings."
)

# ---------------------------------------------------------------------------
# availability probing (never raises)
# ---------------------------------------------------------------------------

_llama_available: bool | None = None
_llama_version: str | None = None


def _probe_llama_cpp() -> tuple[bool, str | None]:
    """Import ``llama_cpp`` once and remember the answer."""
    global _llama_available, _llama_version
    if _llama_available is not None:
        return _llama_available, _llama_version
    try:
        import llama_cpp  # noqa: F401

        _llama_available = True
        try:
            _llama_version = str(getattr(llama_cpp, "__version__", "") or "") or None
        except Exception:
            _llama_version = None
    except ImportError:
        _llama_available = False
        _llama_version = None
    except Exception as exc:  # installed but unusable: same answer, louder log
        logger.warning("llama-cpp-python could not be loaded: %s", exc)
        _llama_available = False
        _llama_version = None
    return _llama_available, _llama_version


def is_llama_cpp_available() -> bool:
    """Whether the ``gguf`` extra is installed and importable."""
    available, _ = _probe_llama_cpp()
    return bool(available)


def llama_cpp_version() -> str | None:
    """Installed llama-cpp-python version, or None when unavailable."""
    _, version = _probe_llama_cpp()
    return version


def supports_gpu_offload() -> bool | None:
    """Ask the installed llama.cpp whether GPU offload exists.

    True / False / None (cannot answer). None must not block a load; False
    blocks only a GPU request, never a CPU one.
    """
    available, _ = _probe_llama_cpp()
    if not available:
        return None
    try:
        import llama_cpp

        probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        if not callable(probe):
            return None
        return bool(probe())
    except Exception as exc:
        logger.debug("Could not query llama.cpp GPU offload: %s", exc)
        return None


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def vl_dir() -> Path:
    """Directory the two VL GGUF files live under."""
    return Path(MODELS_DIR) / PADDLEOCR_VL_MODEL_DIR


def _legacy_vl_dir() -> Path:
    """Where the weights lived before they moved in beside classic PaddleOCR."""
    from fox_reader.constants import PADDLEOCR_VL_LEGACY_MODEL_DIR

    return Path(MODELS_DIR) / PADDLEOCR_VL_LEGACY_MODEL_DIR


_migration_lock = threading.Lock()
_migration_done = False


def migrate_legacy_vl_dir() -> bool:
    """Move weights from ``models/paddleocr-vl/...`` to ``models/paddleocr/...``.

    One-time and idempotent: a process-wide flag keeps it to a single attempt
    per process, and every filesystem step is guarded. Returns True when it
    moved at least one file. Never raises -- a failed move just means the
    weights stay where they are and resolution below reports them missing, in
    which case the setup page re-downloads only what is actually absent (the
    hub resumes per file, so nothing already on disk is fetched twice).

    Never deletes user data: files are moved, never removed, and directories
    are removed only when left empty by the move itself.
    """
    global _migration_done
    with _migration_lock:
        if _migration_done:
            return False
        _migration_done = True
        try:
            new_root = vl_dir()
            old_root = _legacy_vl_dir()
            if new_root == old_root or not old_root.is_dir():
                return False
            from fox_reader.constants import MTL_COMPLETION_MARKER

            if (new_root / MTL_COMPLETION_MARKER).is_file():
                # New home already complete (e.g. a fresh download landed
                # there while the old folder lingered): nothing to move.
                logger.debug("PaddleOCR-VL already complete under %s; leaving %s alone", new_root, old_root)
                return False
            import shutil

            moved = False
            new_root.mkdir(parents=True, exist_ok=True)
            for child in sorted(old_root.iterdir()):
                if not child.is_file():
                    continue
                target = new_root / child.name
                if target.exists():
                    continue
                shutil.move(str(child), str(target))
                moved = True
            if moved:
                logger.info("Moved PaddleOCR-VL weights from %s to %s", old_root, new_root)
            # Remove only what the move itself emptied; anything else stays.
            try:
                old_root.rmdir()
                try:
                    old_root.parent.rmdir()
                except OSError:
                    pass
            except OSError:
                pass
            return moved
        except Exception as exc:  # noqa: BLE001 -- migration is best-effort by design
            logger.warning("Could not migrate PaddleOCR-VL weights to %s: %s", vl_dir(), exc)
            return False


def chat_template_path(model_dir: Path | str | None = None) -> Path:
    """Where the upstream ``chat_template.jinja`` lives next to the weights."""
    from fox_reader.constants import PADDLEOCR_VL_CHAT_TEMPLATE_FILE

    root = Path(model_dir) if model_dir is not None else vl_dir()
    return root / PADDLEOCR_VL_CHAT_TEMPLATE_FILE


_template_lock = threading.Lock()
_template_attempted: set[str] = set()


def ensure_chat_template(model_dir: Path | str | None = None) -> Path | None:
    """Fetch ``chat_template.jinja`` next to resolved weights, best-effort.

    Installs predating the file carry a ``_completed`` marker, so the setup
    worker would skip them forever -- this repair is what delivers the ~2 KiB
    template to them instead of forcing a ~1.8 GiB re-download. Only ever
    fetches, never raises: ``None`` means "still missing, carry on without
    it" (inference reads the template embedded in the GGUF metadata anyway).
    One attempt per directory per process, so an offline machine pays at most
    one failed fetch instead of one per OCR request.
    """
    from fox_reader.constants import PADDLEOCR_VL_CHAT_TEMPLATE_FILE, PADDLEOCR_VL_REPO

    root = Path(model_dir) if model_dir is not None else vl_dir()
    target = root / PADDLEOCR_VL_CHAT_TEMPLATE_FILE
    if target.is_file():
        return target
    key = str(root)
    with _template_lock:
        if key in _template_attempted:
            return None
        _template_attempted.add(key)
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=PADDLEOCR_VL_REPO, filename=PADDLEOCR_VL_CHAT_TEMPLATE_FILE, local_dir=str(root))
    except Exception as exc:  # noqa: BLE001 -- best-effort by design; see above
        logger.debug("Could not fetch PaddleOCR-VL chat template: %s", exc)
        return None
    if target.is_file():
        logger.info("Fetched PaddleOCR-VL chat template to %s", target)
        return target
    return None


def resolve_vl_weights(model_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Return ``(model.gguf, mmproj.gguf)`` or raise FileNotFoundError.

    Prefers the declared filenames, then falls back to the first ``*.gguf``
    pair found under the directory so a renamed quant still loads. Raises
    only FileNotFoundError with a sentence the UI can show.
    """
    root = Path(model_dir) if model_dir is not None else vl_dir()
    if root == vl_dir():
        # Default location only: an explicit directory is used verbatim, so a
        # caller pointing elsewhere never triggers a move behind their back.
        migrate_legacy_vl_dir()
    model = root / PADDLEOCR_VL_MODEL_FILE
    mmproj = root / PADDLEOCR_VL_MMPROJ_FILE
    if model.is_file() and mmproj.is_file():
        pair = (model, mmproj)
    else:
        found = sorted(p for p in root.rglob("*.gguf") if p.is_file()) if root.is_dir() else []
        if not found:
            raise FileNotFoundError(
                f"PaddleOCR-VL weights are not downloaded under {root}. "
                "Download them from the setup page or Settings, or use the "
                "classic PaddleOCR engine."
            )
        # Prefer a non-mmproj file as the model and an mmproj one as projector.
        non_mmproj = [p for p in found if "mmproj" not in p.name.lower()]
        mmprojs = [p for p in found if "mmproj" in p.name.lower()]
        if not (non_mmproj and mmprojs):
            raise FileNotFoundError(
                f"PaddleOCR-VL under {root} is incomplete: need both the model "
                f"({PADDLEOCR_VL_MODEL_FILE}) and the projector ({PADDLEOCR_VL_MMPROJ_FILE}). "
                "Re-download it from the setup page."
            )
        if len(found) > 2:
            logger.info("Multiple GGUF files under %s; loading %s + %s", root, non_mmproj[0].name, mmprojs[0].name)
        pair = (non_mmproj[0], mmprojs[0])
    # Top up the sidecar template for installs that predate it. Best-effort:
    # a missing template never fails resolution.
    ensure_chat_template(root)
    return pair


def vl_model_downloaded(model_dir: Path | str | None = None) -> bool:
    """Whether both VL GGUF files are on disk."""
    try:
        resolve_vl_weights(model_dir)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# device helpers (mirrors the GGUF translators)
# ---------------------------------------------------------------------------


def _normalize_device(device: str | None) -> str:
    return (device or CPU).strip().lower() or CPU


def _cuda_index(device: str) -> int | None:
    text = _normalize_device(device)
    if text == CUDA:
        return 0
    prefix = f"{CUDA}:"
    if text.startswith(prefix) and text[len(prefix):].isdigit():
        return int(text[len(prefix):])
    return None


def _is_mps(device: str) -> bool:
    return _normalize_device(device) == MPS


def _is_cpu(device: str) -> bool:
    return _normalize_device(device) == CPU


def _verbose_enabled() -> bool:
    return os.environ.get(VERBOSE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _flush_logs() -> None:
    """Flush log handlers before entering native llama.cpp code.

    llama.cpp can terminate the process without unwinding Python, so the last
    Python log line must already have reached the output stream -- and, below,
    nothing still-buffered may leak into the suppression window.
    """
    for target in (logger, logging.getLogger("fox_reader"), logging.getLogger()):
        for handler in list(target.handlers):
            try:
                handler.flush()
            except Exception:
                pass


def _native_suppressor() -> Any | None:
    """llama-cpp-python's own fd-level output suppressor, or None.

    The per-request chatter (prompt dumps, image-encode timings, perf tables)
    is printed by native code straight to fds 1/2, so Python-level redirect
    cannot catch it. The library ships exactly this tool for its own load
    path; importing it lazily keeps the ``gguf`` extra optional. Any failure
    degrades to Python-level redirect in the caller, never to an exception.
    """
    try:
        from llama_cpp._utils import suppress_stdout_stderr
    except Exception:
        return None
    try:
        # NOTE the inverted flag: disable=True is the no-op. We want the
        # suppression active, hence False.
        return suppress_stdout_stderr(disable=False)
    except Exception:
        return None


@contextlib.contextmanager
def _suppressed_native_output() -> Iterator[None]:
    """Run native inference quietly unless verbose mode is on.

    Verbose (``FOX_READER_VL_VERBOSE=1``) passes everything through for
    debugging. Otherwise native stdout/stderr are pointed at devnull for the
    call: no prompt dumps, no per-slice timings. Our own ``logging`` lines go
    through handlers flushed beforehand, so the concise log stays intact.

    The context always restores the streams, including on exception, and never
    raises itself: worst case the chatter shows, the transcription still runs.
    """
    if _verbose_enabled():
        yield
        return
    _flush_logs()
    suppressor = _native_suppressor()
    if suppressor is None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
        return
    with suppressor:
        yield


# ---------------------------------------------------------------------------
# image + response helpers
# ---------------------------------------------------------------------------


def _image_to_data_uri(image: Any) -> str:
    """A PIL image as a ``data:image/...;base64,`` URI for llama.cpp."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected a PIL image, got {type(image).__name__}")
    # VL wants the natural crop: grayscale flag is honoured by the classic
    # pipeline's binarisation, but thresholding destroys the anti-aliased
    # fringe this model reads, so here it only means "flatten to RGB".
    rgb = image.convert("RGB")
    # Downscale huge crops: vision embeddings grow with pixels, and a bubble
    # crop above ~1024px on a side buys nothing for transcription speed.
    try:
        w, h = rgb.size
        longest = max(w, h)
        if longest > 1024:
            scale = 1024.0 / float(longest)
            rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    except Exception as exc:
        logger.debug("Could not downscale VL crop: %s", exc)
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _message_content(response: Any) -> str:
    """Extract assistant text from a chat-completion dict, defensively."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Unexpected chat completion shape from llama.cpp: %s", exc)
        return ""
    content = message.get("content") if hasattr(message, "get") else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in (None, "text"):
                continue
            text = part.get("text", "")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    if content is not None:
        logger.warning("Ignoring llama.cpp content of type %s", type(content).__name__)
    return ""


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class PaddleOCRVLEngine:
    """PaddleOCR-VL-1.6 GGUF transcriber, loaded lazily via llama.cpp."""

    def __init__(self, model_dir: Path | str | None = None, device: str | None = None):
        self.root = Path(model_dir) if model_dir is not None else vl_dir()
        if device is None:
            # Own slot on the settings page, like every other consumer. The
            # getattr fallback is for a process still holding a FOX_DEVICE
            # built before the slot existed (hot-reload, old pickle): answer
            # with the classic OCR device rather than raising, which is also
            # what the engine used before the slot was introduced.
            device = getattr(FOX_DEVICE, "paddleocr_vl", None) or getattr(FOX_DEVICE, "paddleocr", CPU)
        self.device = _normalize_device(device)
        self.model_path, self.mmproj_path = resolve_vl_weights(self.root)
        self.model: Any = None
        self._lock = threading.RLock()
        self.languages = ["japanese", "chinese", "english", "korean"]

    # -- construction ----------------------------------------------------

    def _import_llama(self) -> Any:
        try:
            import llama_cpp
        except ImportError as exc:
            raise RuntimeError(MISSING_PACKAGE) from exc
        except Exception as exc:
            raise RuntimeError(f"llama-cpp-python is installed but could not be loaded: {exc}") from exc
        if not hasattr(llama_cpp, "Llama"):
            raise RuntimeError("The installed llama-cpp-python package does not expose llama_cpp.Llama.")
        return llama_cpp

    def _validate_device(self, llama_cpp: Any) -> None:
        if _is_cpu(self.device):
            return
        offload = supports_gpu_offload()
        logger.info("PaddleOCR-VL requested device: %s (llama.cpp GPU offload: %s)", self.device, offload)
        if offload is False:
            raise RuntimeError(
                f"PaddleOCR-VL is configured to use {self.device}, but the installed "
                "llama-cpp-python build is CPU-only. Install a build with GPU support, "
                "or pin Text recognition to the CPU."
            )

    def _placement(self) -> dict[str, Any]:
        if _is_cpu(self.device):
            return {"n_gpu_layers": 0}
        index = _cuda_index(self.device)
        if index is not None:
            kwargs: dict[str, Any] = {"n_gpu_layers": -1}
            if index > 0:
                kwargs["main_gpu"] = index
            return kwargs
        if _is_mps(self.device):
            return {"n_gpu_layers": -1}
        logger.warning("Unknown PaddleOCR-VL device %r; using CPU execution", self.device)
        return {"n_gpu_layers": 0}

    def _n_threads(self, on_gpu: bool) -> int:
        try:
            from fox_reader.device import thread_count

            return int(thread_count(on_gpu=on_gpu))
        except Exception:
            try:
                import os as _os

                return max(1, int(_os.cpu_count() or 4))
            except Exception:
                return 4

    def _build_llama(self, llama_cpp: Any) -> Any:
        placement = self._placement()
        on_gpu = placement.get("n_gpu_layers", 0) != 0
        n_threads = self._n_threads(on_gpu)
        common = dict(n_ctx=N_CTX, n_threads=n_threads, verbose=_verbose_enabled(), **placement)
        logger.info(
            "Loading PaddleOCR-VL %s + %s on %s: %s, n_threads=%d, n_ctx=%d",
            self.model_path.name,
            self.mmproj_path.name,
            self.device,
            ", ".join(f"{k}={v}" for k, v in placement.items()),
            n_threads,
            N_CTX,
        )
        _flush_logs()

        # Vision handler first. On llama-cpp-python 0.3.35 Llama.__init__
        # has neither mmproj_path nor clip_model_path -- both vanish into
        # **kwargs and load a *text-only* model that answers "OCR:" with
        # CJK repetition (𠮷𠮷...). The only spelling that actually wires
        # the projector on this version is chat_handler=MTMDChatHandler(...),
        # so it must come before any Llama-kwarg attempt, and there is no
        # text-only fallback: silent garbage is worse than a loud error.
        chat_handler = self._vision_handler(llama_cpp, use_gpu=on_gpu)
        if chat_handler is not None:
            try:
                return llama_cpp.Llama(
                    model_path=str(self.model_path), chat_handler=chat_handler, logits_all=True, **common
                )
            except TypeError:
                return llama_cpp.Llama(model_path=str(self.model_path), chat_handler=chat_handler, **common)

        # Newer/future API: mmproj passed straight to the constructor. Only
        # tried when the parameter really exists in the signature -- otherwise
        # **kwargs would swallow it and repeat the text-only failure above.
        try:
            import inspect as _inspect

            params = _inspect.signature(llama_cpp.Llama.__init__).parameters
        except Exception:
            params = {}
        for key in ("mmproj_path", "clip_model_path"):
            if key not in params:
                continue
            try:
                kwargs = dict(common)
                kwargs[key] = str(self.mmproj_path)
                if key == "clip_model_path":
                    kwargs.setdefault("logits_all", True)
                return llama_cpp.Llama(model_path=str(self.model_path), **kwargs)
            except TypeError as exc:
                logger.debug("Llama(%s=...) unsupported (%s)", key, exc)
                continue
        raise RuntimeError(
            "The installed llama-cpp-python has no vision-capable handler "
            "(need MTMDChatHandler or an mmproj_path/clip_model_path Llama "
            "parameter). Upgrade llama-cpp-python to >= 0.3.35."
        )

    def _vision_handler(self, llama_cpp: Any, use_gpu: bool = False) -> Any | None:
        """Best chat handler the installed package offers, or None."""
        try:
            from llama_cpp import llama_chat_format as fmt  # type: ignore
        except Exception:
            return None
        verbose = _verbose_enabled()
        # Dedicated PaddleOCR handler on newer forks first, then the generic
        # MTMD handler, then the legacy Llava handler family.
        for attr in ("PaddleOCRChatHandler", "MTMDChatHandler", "GenericMTMDChatHandler", "Llava15ChatHandler"):
            cls = getattr(fmt, attr, None)
            if cls is None:
                continue
            kw_variants: list[dict[str, Any]] = [
                {"clip_model_path": str(self.mmproj_path), "verbose": verbose, "use_gpu": use_gpu},
                {"clip_model_path": str(self.mmproj_path), "verbose": verbose},
                {"clip_model_path": str(self.mmproj_path)},
                {"mmproj_path": str(self.mmproj_path), "verbose": verbose},
                {"mmproj_path": str(self.mmproj_path)},
            ]
            for kw in kw_variants:
                try:
                    return cls(**kw)  # type: ignore[call-arg]
                except TypeError:
                    continue
                except Exception as exc:
                    logger.debug("Vision handler %s failed: %s", attr, exc)
                    break
        # llama_multimodal module on some forks.
        try:
            import importlib as _il

            mm = _il.import_module("llama_cpp.llama_multimodal")
            for attr in ("PaddleOCRChatHandler",):
                cls = getattr(mm, attr, None)
                if cls is None:
                    continue
                for kw in ({"mmproj_path": str(self.mmproj_path)}, {"clip_model_path": str(self.mmproj_path)}):
                    try:
                        return cls(**kw)  # type: ignore[call-arg]
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    def ensure_loaded(self) -> None:
        """Build the llama.cpp handle once, thread-safely."""
        with self._lock:
            if self.model is not None:
                return
            llama_cpp = self._import_llama()
            try:
                ver = getattr(llama_cpp, "__version__", "?")
                logger.info("llama-cpp-python %s for PaddleOCR-VL", ver)
            except Exception:
                pass
            self._validate_device(llama_cpp)
            try:
                self.model = self._build_llama(llama_cpp)
            except RuntimeError:
                raise
            except Exception as exc:
                logger.error("llama.cpp failed loading PaddleOCR-VL on %s: %s", self.device, exc)
                raise RuntimeError(f"llama.cpp could not load PaddleOCR-VL on {self.device}: {exc}") from exc

    # -- inference -------------------------------------------------------

    def predict(self, input: Any, isGrayScaled: bool = False, lang: str = "japanese", **kwargs: Any) -> str:
        """Transcribe a bubble crop. ``lang`` is accepted and ignored: the VL
        model is multilingual, and the parameter exists so the engine can be
        swapped with the classic one without touching callers."""
        _ = (isGrayScaled, lang, kwargs)  # documented above; VL reads raw pixels
        self.ensure_loaded()
        with self._lock:
            model = self.model
            if model is None:
                raise RuntimeError("PaddleOCR-VL has been unloaded.")
            try:
                data_uri = _image_to_data_uri(input)
            except Exception as exc:
                logger.warning("PaddleOCR-VL could not encode the crop: %s", exc)
                return ""
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ]
            try:
                # Quiet by default: native code dumps the prompt, slice
                # timings and perf tables per request otherwise. Logging
                # lines above stay; set FOX_READER_VL_VERBOSE=1 to see it all.
                with _suppressed_native_output():
                    response = model.create_chat_completion(
                        messages=messages, temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS
                    )
            except ValueError as exc:
                raise RuntimeError(f"PaddleOCR-VL rejected the crop ({exc})") from exc
            text = _message_content(response).strip()
            # The model sometimes echoes the prompt or wraps in quotes; strip
            # only those artefacts, never the transcription itself.
            if text == OCR_PROMPT.strip():
                return ""
            return text

    # -- lifecycle -------------------------------------------------------

    def unload(self) -> None:
        with self._lock:
            model = getattr(self, "model", None)
            self.model = None
            if model is None:
                return
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.debug("llama.cpp did not close cleanly: %s", exc)
