"""Gemma E4B GGUF translator using llama.cpp.

Device policy
-------------
The translator device comes from fox_reader.device.FOX_DEVICE:

    cpu       -> CPU execution, n_gpu_layers=0
    cuda      -> CUDA GPU 0, n_gpu_layers=-1
    cuda:N    -> CUDA GPU N, n_gpu_layers=-1
    mps       -> Apple Metal, n_gpu_layers=-1

llama.cpp has its own hardware backend and build configuration. Torch is used
to resolve Fox Reader's configured device, but this module separately verifies
that the installed llama-cpp-python build can actually execute that request.

A GPU request on a CPU-only llama-cpp-python installation is refused before the
large GGUF allocation begins. This avoids silently attempting to load a ~7.6 GiB
model into system RAM and potentially killing the process.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from fox_reader.constants import (
    GEMMA_E4B_Q8_MODEL_DIR,
    GEMMA_E4B_Q8_MODEL_FILE,
)
from fox_reader.device import (
    CPU,
    CUDA,
    MPS,
    FOX_DEVICE,
    thread_count,
)
from fox_reader.translate.context import render_context_block

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# model configuration
# ---------------------------------------------------------------------------

N_CTX = 8192
MAX_NEW_TOKENS = 1024

VERBOSE_ENV = "FOX_READER_MTL_VERBOSE"

_AFFIRMATIVE = frozenset({"1", "true", "yes", "on"})

MISSING_PACKAGE = (
    "This model is a GGUF file and needs llama-cpp-python, which is not "
    "installed. Install it with the 'gguf' extra (uv pip install llama-cpp-python)"
    ", or choose another model under Settings → Local Translation Models."
)


# Only strip a markdown fence when it wraps the entire response.
_WRAPPING_FENCE = re.compile(
    r"\A```[\w+-]*[ \t]*\r?\n(.*?)\r?\n?```\Z",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def verbose_enabled() -> bool:
    """Whether llama.cpp should emit its native loader output."""
    return (
        os.environ.get(VERBOSE_ENV, "")
        .strip()
        .lower()
        in _AFFIRMATIVE
    )


def _flush_logs() -> None:
    """Flush log handlers before entering native llama.cpp code.

    llama.cpp can terminate the process without unwinding Python, so the last
    Python log line must already have reached the output stream.
    """
    for target in (
        logger,
        logging.getLogger("fox_reader"),
        logging.getLogger(),
    ):
        for handler in list(target.handlers):
            try:
                handler.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# llama.cpp import / capabilities
# ---------------------------------------------------------------------------


def _import_llama_cpp() -> Any:
    """Import llama_cpp only when Gemma is actually being loaded."""
    logger.info("Importing llama-cpp-python")
    _flush_logs()

    try:
        import llama_cpp
    except ImportError as exc:
        raise RuntimeError(MISSING_PACKAGE) from exc
    except Exception as exc:
        raise RuntimeError(
            f"llama-cpp-python is installed but could not be loaded: {exc}"
        ) from exc

    return llama_cpp


def _llama_class(llama_cpp: Any) -> Any:
    """Return the Llama class from llama_cpp."""
    try:
        return llama_cpp.Llama
    except AttributeError as exc:
        raise RuntimeError(
            "The installed llama-cpp-python package does not expose "
            "llama_cpp.Llama."
        ) from exc


def supports_gpu_offload(llama_cpp: Any) -> bool | None:
    """Ask llama.cpp whether this build supports accelerator offload.

    Returns:
        True  - accelerator offload is available
        False - this is a CPU-only build
        None  - this version/build cannot answer
    """
    probe = getattr(
        llama_cpp,
        "llama_supports_gpu_offload",
        None,
    )

    if not callable(probe):
        return None

    try:
        return bool(probe())
    except Exception as exc:
        logger.debug(
            "Could not query llama.cpp GPU offload support: %s",
            exc,
        )
        return None


def _log_build(llama_cpp: Any) -> None:
    """Log the llama.cpp version and backend capability."""
    version = getattr(
        llama_cpp,
        "__version__",
        None,
    )

    offload = supports_gpu_offload(llama_cpp)

    if offload is True:
        backend = "GPU offload available"
    elif offload is False:
        backend = "CPU-only build"
    else:
        backend = "GPU offload support unknown"

    logger.info(
        "llama-cpp-python %s, %s",
        version or "version unknown",
        backend,
    )

    _flush_logs()


# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------


def _normalize_device(device: str | None) -> str:
    """Normalize a configured device string."""
    return (device or CPU).strip().lower() or CPU


def _cuda_index(device: str) -> int | None:
    """Return the ordinal from cuda:N, or None for non-CUDA devices."""
    text = _normalize_device(device)

    if text == CUDA:
        return 0

    prefix = f"{CUDA}:"

    if text.startswith(prefix):
        tail = text[len(prefix):]

        if tail.isdigit():
            return int(tail)

    return None


def _is_cuda(device: str) -> bool:
    return _cuda_index(device) is not None


def _is_mps(device: str) -> bool:
    return _normalize_device(device) == MPS


def _is_cpu(device: str) -> bool:
    return _normalize_device(device) == CPU


def _split_mode_none(llama_cpp: Any) -> int:
    """Get llama.cpp's split-none enum.

    Different llama.cpp versions have used different names.
    """
    for name in (
        "LLAMA_SPLIT_MODE_NONE",
        "LLAMA_SPLIT_NONE",
    ):
        value = getattr(
            llama_cpp,
            name,
            None,
        )

        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return 0


def llama_device_kwargs(
    device: str,
    llama_cpp: Any | None = None,
) -> dict[str, Any]:
    """Convert Fox Reader's device into llama.cpp constructor arguments.

    CPU:
        n_gpu_layers=0

    CUDA:
        n_gpu_layers=-1

    CUDA:N:
        n_gpu_layers=-1
        main_gpu=N
        split_mode=NONE

    MPS:
        n_gpu_layers=-1
    """
    text = _normalize_device(device)

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------
    if _is_cpu(text):
        return {
            "n_gpu_layers": 0,
        }

    # ------------------------------------------------------------------
    # CUDA
    # ------------------------------------------------------------------
    index = _cuda_index(text)

    if index is not None:
        kwargs: dict[str, Any] = {
            "n_gpu_layers": -1,
        }

        # CUDA / cuda:0 does not need extra placement arguments.
        #
        # For cuda:N we explicitly select that device. Otherwise llama.cpp
        # may choose its default GPU.
        if index > 0:
            kwargs["main_gpu"] = index

            if llama_cpp is not None:
                kwargs["split_mode"] = _split_mode_none(
                    llama_cpp
                )

        return kwargs

    # ------------------------------------------------------------------
    # Apple Metal / MPS
    # ------------------------------------------------------------------
    if _is_mps(text):
        return {
            "n_gpu_layers": -1,
        }

    # ------------------------------------------------------------------
    # Unknown device
    # ------------------------------------------------------------------
    logger.warning(
        "Unknown Gemma device %r; using CPU execution",
        text,
    )

    return {
        "n_gpu_layers": 0,
    }


# ---------------------------------------------------------------------------
# device/build validation
# ---------------------------------------------------------------------------


def _validate_requested_device(
    device: str,
    llama_cpp: Any,
) -> None:
    """Verify that llama.cpp can satisfy the requested device.

    We deliberately do not silently fall back from GPU to CPU.

    A CPU fallback here would turn the GGUF into a large system-RAM
    allocation and can terminate the whole Fox Reader process before Python
    gets an exception.
    """
    text = _normalize_device(device)

    if _is_cpu(text):
        return

    offload = supports_gpu_offload(llama_cpp)

    logger.info(
        "Gemma requested device: %s",
        text,
    )

    logger.info(
        "llama.cpp GPU offload capability: %s",
        (
            "yes"
            if offload is True
            else "no"
            if offload is False
            else "unknown"
        ),
    )

    if offload is False:
        raise RuntimeError(
            f"Gemma is configured to use {text}, but the installed "
            "llama-cpp-python build is CPU-only. "
            "Install a llama-cpp-python build with the appropriate GPU "
            "backend, then restart Fox Reader."
        )

    # We currently cannot reliably distinguish CUDA and Metal capability
    # using llama_supports_gpu_offload() alone across every llama.cpp version.
    #
    # The actual constructor remains the authority. We therefore only perform
    # the generic capability check here and let llama.cpp validate the exact
    # device during construction.
    #
    # This is preferable to pretending an old llama.cpp API exposes capability
    # information that it does not actually provide.


# ---------------------------------------------------------------------------
# model path
# ---------------------------------------------------------------------------


def resolve_weights(mtl_dir: Path | str) -> Path:
    """Find the Gemma GGUF file."""
    root = Path(mtl_dir) / GEMMA_E4B_Q8_MODEL_DIR

    declared = root / GEMMA_E4B_Q8_MODEL_FILE

    if declared.is_file():
        return declared

    found = sorted(
        path
        for path in root.rglob("*.gguf")
        if path.is_file()
    )

    if not found:
        raise FileNotFoundError(
            f"No .gguf file under {root}. The model directory exists but "
            "its weights are missing — re-download it from the setup page."
        )

    if len(found) > 1:
        logger.info(
            "%d .gguf files under %s; loading %s",
            len(found),
            root,
            found[0].name,
        )
    else:
        logger.info(
            "Loading %s instead of the expected %s",
            found[0].name,
            GEMMA_E4B_Q8_MODEL_FILE,
        )

    return found[0]


def _size_on_disk(path: Path) -> str:
    """Return the GGUF size for diagnostics."""
    try:
        return f"{path.stat().st_size / (1024 ** 3):.1f} GiB"
    except OSError:
        return "unknown size"


# ---------------------------------------------------------------------------
# response handling
# ---------------------------------------------------------------------------


def _message_content(response: Any) -> str:
    """Extract assistant text defensively."""
    try:
        message = response["choices"][0]["message"]
    except (
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        logger.warning(
            "Unexpected chat completion shape from llama.cpp: %s",
            exc,
        )
        return ""

    content = (
        message.get("content")
        if hasattr(message, "get")
        else None
    )

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
        logger.warning(
            "Ignoring llama.cpp content of type %s",
            type(content).__name__,
        )

    return ""


def clean_translation(text: str) -> str:
    """Remove only a whole-response markdown fence."""
    stripped = (text or "").strip()

    match = _WRAPPING_FENCE.match(stripped)

    if match:
        return match.group(1).strip()

    return stripped


# ---------------------------------------------------------------------------
# token-budget context trimming
# ---------------------------------------------------------------------------


def _answer_reserve() -> int:
    """Tokens kept free for the reply: capped at half the window."""
    return min(MAX_NEW_TOKENS, N_CTX // 2) + 4


def _estimate_tokens(model: Any, text: str) -> int:
    """Measured token cost of `text`, falling back to a chars estimate."""
    if not text:
        return 0
    try:
        tokenize = getattr(model, "tokenize", None)
        if callable(tokenize):
            try:
                return len(list(tokenize(text.encode("utf-8"), add_bos=False, special=False)))
            except TypeError:
                return len(list(tokenize(text.encode("utf-8"), add_bos=False)))
    except Exception:
        pass
    return max(1, len(text) // 4 + 1)


def _trim_pairs_for_window(
    model: Any,
    pairs: list,
    base_tokens: int,
    per_pair_overhead: int = 16,
) -> tuple[list, int]:
    """Newest pairs that fit ``N_CTX - reserve``; oldest dropped first.

    Costs are measured with the loaded tokenizer, not estimated from chars.
    Returns ``(kept, dropped)``. Mirrors the VNTL policy (newest wins, one
    oversized pair stops the walk) so both GGUF models degrade the same way.
    """
    try:
        clean = [pair for pair in (pairs or []) if isinstance(pair, (list, tuple)) and len(pair) == 2]
    except Exception:
        return [], 0
    budget = N_CTX - base_tokens - _answer_reserve()
    if budget <= 0:
        return [], len(clean)
    # Pre-encode newest-first so a trim loses the oldest turns.
    costs: list[tuple[Any, int]] = []
    for pair in clean:
        try:
            cost = _estimate_tokens(model, str(pair[0])) + _estimate_tokens(model, str(pair[1])) + per_pair_overhead
        except Exception:
            cost = 0
        costs.append((pair, cost))
    chosen: list[Any] = []
    for pair, cost in reversed(costs):
        if cost > budget:
            break
        budget -= cost
        chosen.append(pair)
    chosen.reverse()
    dropped = len(clean) - len(chosen)
    if dropped:
        logger.warning(
            "Gemma context trimmed: dropped the %d oldest of %d turns to fit %d tokens",
            dropped,
            len(clean),
            N_CTX,
        )
    return chosen, dropped


# ---------------------------------------------------------------------------
# translator
# ---------------------------------------------------------------------------


class GemmaE4BQ8Translator:
    """Gemma E4B uncensored GGUF translator."""

    def __init__(
        self,
        mtl_dir: Path,
        device: str | None = None,
    ):
        self.model_path = resolve_weights(mtl_dir)

        # Device comes from FOX_DEVICE unless a caller explicitly supplied one.
        self.device = _normalize_device(
            device if device is not None else FOX_DEVICE.translator
        )

        self.model: Any = None

        # llama.cpp's KV cache/context is mutable. Serialize generation with a
        # reentrant lock: translate() holds it across the call and unload()
        # takes the same one, so weights cannot be freed mid-inference.
        self._lock = threading.RLock()

        llama_cpp = _import_llama_cpp()
        llama_cls = _llama_class(llama_cpp)

        _log_build(llama_cpp)

        # ---------------------------------------------------------------
        # Verify build capability BEFORE allocating the GGUF.
        # ---------------------------------------------------------------
        _validate_requested_device(
            self.device,
            llama_cpp,
        )

        placement = llama_device_kwargs(
            self.device,
            llama_cpp,
        )

        self.on_gpu = (
            placement.get("n_gpu_layers", 0) != 0
        )

        # Same policy as the working standalone implementation.
        self.n_threads = thread_count(
            on_gpu=self.on_gpu,
        )

        verbose = verbose_enabled()

        logger.info(
            "Loading %s (%s) on %s: %s, "
            "n_threads=%d, n_ctx=%d",
            self.model_path.name,
            _size_on_disk(self.model_path),
            self.device,
            ", ".join(
                f"{key}={value}"
                for key, value in placement.items()
            ),
            self.n_threads,
            N_CTX,
        )

        if self.on_gpu:
            logger.info(
                "Gemma is using full accelerator offload "
                "(n_gpu_layers=-1)"
            )
        else:
            logger.info(
                "Gemma is using CPU execution "
                "(n_gpu_layers=0)"
            )

        if not verbose:
            logger.info(
                "llama.cpp will load quietly; set %s=1 for "
                "native llama.cpp loader output",
                VERBOSE_ENV,
            )

        _flush_logs()

        # ---------------------------------------------------------------
        # IMPORTANT
        #
        # Keep the actual Llama constructor close to the known-good code.
        # ---------------------------------------------------------------
        try:
            self.model = llama_cls(
                model_path=str(self.model_path),
                n_ctx=N_CTX,
                n_threads=self.n_threads,
                verbose=verbose,
                **placement,
            )

        except Exception as exc:
            logger.error(
                "llama.cpp failed loading %s on %s: %s",
                self.model_path.name,
                self.device,
                exc,
            )

            if self.on_gpu:
                raise RuntimeError(
                    f"llama.cpp could not load Gemma on "
                    f"{self.device}: {exc}"
                ) from exc

            raise

    # ------------------------------------------------------------------
    # prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        context: object = None,
        lang: str | None = None,
    ) -> str:
        system = [
            "You are a professional translator.",
            "Keep the original tone of text",
            "Use slangs if given",
            "Don't censor anything",
            "Translate all user-provided text into English.",
        ]

        system.extend(
            [
                "Return ONLY valid translated text.",
                "Provide the final translation immediately without any other text.",
                "Do not wrap any text in markdown.",
                "Do not explain anything.",
            ]
        )

        block = render_context_block(context, lang)

        if block:
            system.append(block)

        return "\n".join(system)

    # ------------------------------------------------------------------
    # translate
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        context: object = None,
        lang: str | None = None,
        character_info: object = None,
        context_character_links: object = None,
        meta_id: object = None,
        **kwargs,
    ) -> str:
        source = (text or "").strip()

        if not source:
            return ""

        if self.model is None:
            raise RuntimeError(
                "This translator has been unloaded."
            )

        if character_info is not None or context_character_links is not None or meta_id is not None:
            logger.debug("Gemma does not render character metadata; ignoring it.")

        # Token-budget trim before building the system prompt: measured with
        # the loaded tokenizer, newest pairs win, oldest dropped first.
        pairs = context if isinstance(context, (list, tuple)) else []
        try:
            base = _estimate_tokens(self.model, source)
            system_base = _estimate_tokens(
                self.model,
                "\n".join(
                    [
                        "You are a professional translator.",
                        "Translate all user-provided text into English.",
                    ]
                ),
            )
            pairs, _dropped = _trim_pairs_for_window(self.model, list(pairs or []), base + system_base)
        except Exception as exc:
            logger.debug("Gemma context pre-trim failed, using context as-is: %s", exc)
            pairs = list(pairs or [])

        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(
                    pairs,
                    lang,
                ),
            },
            {
                "role": "user",
                "content": source,
            },
        ]

        try:
            with self._lock:
                response = self.model.create_chat_completion(
                    messages=messages,
                    top_k=64,
                    temperature=1.0,
                    top_p=0.95,
                    max_tokens=MAX_NEW_TOKENS,
                )

        except ValueError as exc:
            raise RuntimeError(
                f"This text is too long for the model's "
                f"{N_CTX}-token context window: {exc}"
            ) from exc

        return clean_translation(
            _message_content(response)
        )

    # ------------------------------------------------------------------
    # unload
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Explicitly release the native llama.cpp context.

        Idempotent and serialised against translate(): takes the same RLock
        so an in-flight generation cannot lose its weights underneath it.
        """
        with self._lock:
            model = getattr(
                self,
                "model",
                None,
            )

            self.model = None

            if model is None:
                return

            close = getattr(
                model,
                "close",
                None,
            )

            if not callable(close):
                return

            try:
                close()
            except Exception as exc:
                logger.debug(
                    "llama.cpp did not close cleanly: %s",
                    exc,
                )

    # ------------------------------------------------------------------
    # model metadata
    # ------------------------------------------------------------------

    @classmethod
    def model_id(cls) -> str:
        return "gemma-4-e4b-q8-uncensored"

    @classmethod
    def model_langs(cls) -> list[str]:
        return ["japanese", "chinese", "korean"]