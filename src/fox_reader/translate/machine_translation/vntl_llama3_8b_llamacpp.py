"""VNTL Llama3 8B v2 GGUF translator using llama.cpp.

Prompt dialect
--------------
This fine-tune speaks the stock Llama-3 header format with three non-standard
roles (``Metadata`` / ``Japanese`` / ``English``). A prompt is a ``Metadata``
block with one ``[character]`` line per known character, then alternating
``Japanese`` / ``English`` history turns, then the Japanese line to translate,
ending on an **open** ``English`` header for the model to complete. Prompts are
built as token-ID lists (special tokens spliced as integers): handed to
``tokenize()`` as text they shred into ordinary pieces and the model loses its
turn boundaries. See ``translator.py`` (``D:\\Workspace\\vntl-llm``) this port
follows.

Device policy
-------------
Same as the Gemma GGUF translators -- the device comes from
``fox_reader.device.FOX_DEVICE``::

    cpu     -> CPU execution, n_gpu_layers=0
    cuda    -> CUDA GPU 0, n_gpu_layers=-1
    cuda:N  -> CUDA GPU N, n_gpu_layers=-1 (+ main_gpu/split_mode for N > 0)
    mps     -> Apple Metal, n_gpu_layers=-1

A GPU request on a CPU-only llama-cpp-python build is refused before the ~8.5
GiB allocation begins. Threads come from ``fox_reader.device.thread_count``
(the saved preference / env override / heuristic), not from a local default.

Robustness contract (ported from the reference implementation)
--------------------------------------------------------------
* Repair-and-report validation: malformed context rows, links and character
  fields are dropped with a WARNING, never raised. Only an empty ``text`` and
  an over-budget fixed cost raise.
* Context trimming is measured, not estimated: every pair is pre-encoded,
  then newest-first is kept while it fits ``n_ctx - fixed_cost``. The loop
  ``break``\\ s (not ``continue``\\ s) on the first oversized pair, so one huge
  turn discards it and everything older; ``dropped`` counts them all.
* Speaker tags are symmetric: a JA/EN pair is tagged on both sides or neither.
* ``meta_id`` is compared byte-exact (strip only, never sanitised): it is a
  dict key, never rendered, so sanitising one side would lose every tag.
* Calls are serialised with an RLock (llama.cpp owns one KV cache); ``close``
  / ``unload`` take the same lock so weights cannot be freed mid-generation.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fox_reader.constants import VNTL_LLAMA3_MODEL_DIR, VNTL_LLAMA3_MODEL_FILE
from fox_reader.device import CPU, CUDA, FOX_DEVICE, MPS, thread_count

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# model configuration
# ---------------------------------------------------------------------------

N_CTX = 8192
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 0.95
TOP_K = 40
REPEAT_PENALTY = 1.0
SEED = 0
SPEAKER_TAGGING = "auto"

VERBOSE_ENV = "FOX_READER_MTL_VERBOSE"

_AFFIRMATIVE = frozenset({"1", "true", "yes", "on"})

MISSING_PACKAGE = (
    "This model is a GGUF file and needs llama-cpp-python, which is not "
    "installed. Install it with the 'gguf' extra (uv pip install llama-cpp-python)"
    ", or choose another model under Settings → Local Translation Models."
)

# ---------------------------------------------------------------------------
# Llama-3 wire format (verified against the GGUF token table)
# ---------------------------------------------------------------------------

TOK_BEGIN_OF_TEXT = 128000
TOK_END_OF_TEXT = 128001
TOK_START_HEADER = 128006
TOK_END_HEADER = 128007
TOK_EOT = 128009

ROLE_METADATA = "Metadata"
ROLE_JAPANESE = "Japanese"
ROLE_ENGLISH = "English"
_ROLES = (ROLE_METADATA, ROLE_JAPANESE, ROLE_ENGLISH)

NEWLINE = 10

_CONTROL_MARKERS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<<START>>",
    "<<JAPANESE>>",
    "<<ENGLISH>>",
)

_SPEAKER_RE = re.compile(r"^\s*\[([^\[\]\n]{1,64})\]\s*[:：]\s*")

_VALID_GENDERS = frozenset({"male", "female"})
_CHARACTER_KEYS = frozenset({"meta_id", "name_en", "name_ja", "gender", "alias_en", "alias_ja"})


class ModelLoadError(RuntimeError):
    """The GGUF could not be loaded (missing file / package / backend)."""


class PromptBudgetError(RuntimeError):
    """The non-context parts alone do not fit the window."""


# ---------------------------------------------------------------------------
# logging helpers (same policy as the Gemma translators)
# ---------------------------------------------------------------------------


def verbose_enabled() -> bool:
    return os.environ.get(VERBOSE_ENV, "").strip().lower() in _AFFIRMATIVE


def _flush_logs() -> None:
    for target in (logger, logging.getLogger("fox_reader"), logging.getLogger()):
        for handler in list(target.handlers):
            try:
                handler.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# llama.cpp import / capabilities
# ---------------------------------------------------------------------------


def _import_llama_cpp() -> Any:
    logger.info("Importing llama-cpp-python")
    _flush_logs()
    try:
        import llama_cpp
    except ImportError as exc:
        raise ModelLoadError(MISSING_PACKAGE) from exc
    except Exception as exc:
        raise ModelLoadError(f"llama-cpp-python is installed but could not be loaded: {exc}") from exc
    return llama_cpp


def _llama_class(llama_cpp: Any) -> Any:
    try:
        return llama_cpp.Llama
    except AttributeError as exc:
        raise ModelLoadError("The installed llama-cpp-python package does not expose llama_cpp.Llama.") from exc


def supports_gpu_offload(llama_cpp: Any) -> bool | None:
    probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if not callable(probe):
        return None
    try:
        return bool(probe())
    except Exception as exc:
        logger.debug("Could not query llama.cpp GPU offload support: %s", exc)
        return None


def _log_build(llama_cpp: Any) -> None:
    version = getattr(llama_cpp, "__version__", None)
    offload = supports_gpu_offload(llama_cpp)
    backend = "GPU offload available" if offload is True else "CPU-only build" if offload is False else "GPU offload support unknown"
    logger.info("llama-cpp-python %s, %s", version or "version unknown", backend)
    _flush_logs()


# ---------------------------------------------------------------------------
# device helpers (same spelling as the Gemma translators)
# ---------------------------------------------------------------------------


def _normalize_device(device: str | None) -> str:
    return (device or CPU).strip().lower() or CPU


def _cuda_index(device: str) -> int | None:
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
    for name in ("LLAMA_SPLIT_MODE_NONE", "LLAMA_SPLIT_NONE"):
        value = getattr(llama_cpp, name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def llama_device_kwargs(device: str, llama_cpp: Any | None = None) -> dict[str, Any]:
    text = _normalize_device(device)
    if _is_cpu(text):
        return {"n_gpu_layers": 0}
    index = _cuda_index(text)
    if index is not None:
        kwargs: dict[str, Any] = {"n_gpu_layers": -1}
        if index > 0:
            kwargs["main_gpu"] = index
            if llama_cpp is not None:
                kwargs["split_mode"] = _split_mode_none(llama_cpp)
        return kwargs
    if _is_mps(text):
        return {"n_gpu_layers": -1}
    logger.warning("Unknown VNTL device %r; using CPU execution", text)
    return {"n_gpu_layers": 0}


def _validate_requested_device(device: str, llama_cpp: Any) -> None:
    """Refuse a GPU request on a CPU-only build before allocating ~8.5 GiB."""
    text = _normalize_device(device)
    if _is_cpu(text):
        return
    offload = supports_gpu_offload(llama_cpp)
    logger.info("VNTL requested device: %s", text)
    logger.info("llama.cpp GPU offload capability: %s", "yes" if offload is True else "no" if offload is False else "unknown")
    if offload is False:
        raise ModelLoadError(
            f"VNTL is configured to use {text}, but the installed llama-cpp-python build is CPU-only. "
            "Install a llama-cpp-python build with the appropriate GPU backend, then restart Fox Reader."
        )


# ---------------------------------------------------------------------------
# model path
# ---------------------------------------------------------------------------


def resolve_weights(mtl_dir: Path | str) -> Path:
    root = Path(mtl_dir) / VNTL_LLAMA3_MODEL_DIR
    declared = root / VNTL_LLAMA3_MODEL_FILE
    if declared.is_file():
        return declared
    found = sorted(path for path in root.rglob("*.gguf") if path.is_file()) if root.is_dir() else []
    if not found:
        raise FileNotFoundError(
            f"No .gguf file under {root}. The model directory exists but its weights are missing — "
            "re-download it from the setup page."
        )
    if len(found) > 1:
        logger.info("%d .gguf files under %s; loading %s", len(found), root, found[0].name)
    else:
        logger.info("Loading %s instead of the expected %s", found[0].name, VNTL_LLAMA3_MODEL_FILE)
    return found[0]


def _size_on_disk(path: Path) -> str:
    try:
        return f"{path.stat().st_size / (1024 ** 3):.1f} GiB"
    except OSError:
        return "unknown size"


# ---------------------------------------------------------------------------
# text helpers (ported from the reference translator.py)
# ---------------------------------------------------------------------------


def _sanitise(value: str) -> str:
    for marker in _CONTROL_MARKERS:
        if marker in value:
            value = value.replace(marker, "")
    return " ".join(value.split())


def _normalise_meta_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return None
    elif isinstance(value, (list, tuple, set, dict)):
        return None
    text = str(value).strip()
    return text or None


def _coerce_side(value: Any, index: int, side: str) -> str | None:
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", "replace").strip()
        except Exception:
            return None
    if isinstance(value, (list, tuple, set, dict)):
        logger.warning("context[%d]'s %s side is a %s, not text; dropping the row", index, side, type(value).__name__)
        return None
    return str(value).strip()


def _tokenize_text(llm: Any, text: str) -> list[int]:
    if not text:
        return []
    try:
        return list(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
    except TypeError:
        return list(llm.tokenize(text.encode("utf-8"), add_bos=False))


def _encode_header(role: str, role_tokens: dict[str, list[int]]) -> list[int]:
    return [TOK_START_HEADER] + role_tokens[role] + [TOK_END_HEADER, NEWLINE, NEWLINE]


def _encode_body(role: str, body: str, llm: Any, role_tokens: dict[str, list[int]]) -> list[int]:
    return _encode_header(role, role_tokens) + _tokenize_text(llm, body) + [TOK_EOT]


@dataclass(frozen=True)
class CharacterInfo:
    meta_id: str | None = None
    name_en: str | None = None
    name_ja: str | None = None
    gender: str | None = None
    alias_en: str | None = None
    alias_ja: str | None = None

    @classmethod
    def from_mapping(cls, raw: Any, index: int = 0) -> CharacterInfo:
        if not isinstance(raw, dict):
            logger.warning("character_info[%d] is %s, not a dict; ignoring it", index, type(raw).__name__)
            return cls()

        def pick(key: str) -> str | None:
            value = raw.get(key)
            if value is None:
                return None
            if isinstance(value, (bytes, bytearray)):
                try:
                    value = value.decode("utf-8", "replace")
                except Exception:
                    return None
            elif isinstance(value, (list, tuple, set, dict)):
                logger.warning("character_info[%d][%r] is a %s; ignoring it", index, key, type(value).__name__)
                return None
            text = _sanitise(str(value))
            return text or None

        gender = pick("gender")
        if gender is not None:
            gender = gender.lower()
            if gender not in _VALID_GENDERS:
                logger.debug(
                    "character_info[%d] gender %r is non-standard; it will be omitted from the prompt",
                    index,
                    gender,
                )
        return cls(
            meta_id=_normalise_meta_id(raw.get("meta_id")),
            name_en=pick("name_en"),
            name_ja=pick("name_ja"),
            gender=gender,
            alias_en=pick("alias_en"),
            alias_ja=pick("alias_ja"),
        )

    def resolve_ja(self) -> str | None:
        return self.name_ja or self.alias_ja or self.name_en or self.alias_en

    def resolve_en(self) -> str | None:
        return self.name_en or self.alias_en or self.name_ja or self.alias_ja

    def is_empty(self) -> bool:
        return not any((self.name_en, self.name_ja, self.gender, self.alias_en, self.alias_ja))


def format_character_line(char: CharacterInfo) -> str | None:
    fields: list[str] = []

    def pair(en: str | None, ja: str | None, label: str) -> str | None:
        if en and ja:
            return f"{label}: {en}" if en == ja else f"{label}: {en} ({ja})"
        if en:
            return f"{label}: {en}"
        if ja:
            return f"{label}: {ja}"
        return None

    name = pair(char.name_en, char.name_ja, "Name")
    if name:
        fields.append(name)
    gender = (char.gender or "").strip().lower()
    if gender in _VALID_GENDERS:
        fields.append(f"Gender: {gender.capitalize()}")
    alias = pair(char.alias_en, char.alias_ja, "Aliases")
    if alias:
        fields.append(alias)
    if not fields:
        return None
    return "[character] " + " | ".join(fields)


@dataclass
class _PreparedInput:
    text_ja: str
    context: list[tuple[str, str]]
    context_links: list[str | None]
    characters: dict[str, CharacterInfo]
    meta_id: str | None = None


class _InputValidator:
    @staticmethod
    def prepare(text_ja: Any, context: Any, character_info: Any, context_character_links: Any, meta_id: Any) -> _PreparedInput:
        text = _InputValidator._text(text_ja)
        characters = _InputValidator._characters(character_info)
        ctx, links = _InputValidator._context_and_links(context, context_character_links)
        return _PreparedInput(text, ctx, links, characters, _InputValidator._meta_id(meta_id))

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            raise ValueError("text_ja must be a non-empty string, got None")
        if not isinstance(value, str):
            raise TypeError(f"text_ja must be a string, got {type(value).__name__}")
        stripped = "".join(ch for ch in value if unicodedata.category(ch) != "Cf").strip()
        if not stripped:
            raise ValueError("text_ja is empty or whitespace-only")
        return stripped

    @staticmethod
    def _meta_id(value: Any) -> str | None:
        if value is not None and isinstance(value, (list, tuple, set, dict)):
            logger.warning("meta_id is a %s; ignoring it", type(value).__name__)
            return None
        return _normalise_meta_id(value)

    @staticmethod
    def _characters(raw: Any) -> dict[str, CharacterInfo]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            if any(key in raw for key in _CHARACTER_KEYS):
                raw = [raw]
            elif raw and all(isinstance(v, dict) for v in raw.values()):
                logger.debug("character_info looks like a meta_id -> info map")
                raw = [{**value, "meta_id": value.get("meta_id", key)} for key, value in raw.items()]
            else:
                logger.warning("character_info is a dict with no recognised character fields; ignoring it")
                return {}
        if not isinstance(raw, (list, tuple)):
            logger.warning("character_info is a %s, expected a list; ignoring it", type(raw).__name__)
            return {}
        characters: dict[str, CharacterInfo] = {}
        for index, item in enumerate(raw):
            char = CharacterInfo.from_mapping(item, index)
            if char.meta_id is None:
                if not char.is_empty():
                    logger.warning(
                        "character_info[%d] has no meta_id and cannot be linked to any context; "
                        "its name and gender will be unused",
                        index,
                    )
                continue
            if char.meta_id in characters:
                logger.warning("duplicate meta_id %r in character_info; keeping the first", char.meta_id)
                continue
            characters[char.meta_id] = char
        return characters

    @staticmethod
    def _context_rows(raw: Any) -> list[tuple[int, str, str]]:
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)):
            logger.warning("context is a %s, expected a list; ignoring it", type(raw).__name__)
            return []
        rows: list[tuple[int, str, str]] = []
        for index, item in enumerate(raw):
            if item is None:
                continue
            if isinstance(item, (list, tuple)):
                if len(item) < 2:
                    logger.warning("context[%d] has %d element(s), expected 2; dropping it", index, len(item))
                    continue
                src, dst = item[0], item[1]
            elif isinstance(item, dict):
                src = item.get("ja", item.get("source"))
                dst = item.get("en", item.get("target"))
            else:
                logger.warning("context[%d] is a %s, expected a pair; dropping it", index, type(item).__name__)
                continue
            if src is None or dst is None:
                logger.warning("context[%d] is missing a side; dropping it", index)
                continue
            src_text = _coerce_side(src, index, "Japanese")
            dst_text = _coerce_side(dst, index, "English")
            if src_text is None or dst_text is None:
                continue
            if not src_text:
                logger.warning("context[%d] has no Japanese text; dropping it", index)
                continue
            rows.append((index, src_text, dst_text))
        return rows

    @staticmethod
    def _coerce_links(raw: Any) -> list[str | None] | None:
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)):
            logger.warning("context_character_links is a %s, expected a list; ignoring it", type(raw).__name__)
            return None
        links: list[str | None] = []
        for item in raw:
            if isinstance(item, (list, tuple, set, dict)):
                logger.warning("context_character_links entry is a %s; treating as None", type(item).__name__)
                links.append(None)
            else:
                links.append(_normalise_meta_id(item))
        return links

    @staticmethod
    def _context_and_links(raw_context: Any, raw_links: Any) -> tuple[list[tuple[str, str]], list[str | None]]:
        rows = _InputValidator._context_rows(raw_context)
        links = _InputValidator._coerce_links(raw_links)
        if links is None:
            return [(ja, en) for _, ja, en in rows], [None] * len(rows)
        raw_len = len(raw_context) if isinstance(raw_context, (list, tuple)) else 0
        if len(links) != raw_len:
            logger.warning(
                "context_character_links has %d entries but context has %d; unlisted turns will have no speaker attribution",
                len(links),
                raw_len,
            )
        pairs: list[tuple[str, str]] = []
        out_links: list[str | None] = []
        for original_index, ja, en in rows:
            pairs.append((ja, en))
            out_links.append(links[original_index] if original_index < len(links) else None)
        return pairs, out_links


# ---------------------------------------------------------------------------
# translator
# ---------------------------------------------------------------------------


class VntlLlamaTranslator:
    """VNTL Llama3 8B v2 GGUF translator (Japanese -> English)."""

    def __init__(self, mtl_dir: Path | str, device: str | None = None):
        self.model_path = resolve_weights(mtl_dir)
        self.device = _normalize_device(device if device is not None else FOX_DEVICE.translator)
        # Set before anything that can raise, so unload() on a half-built
        # object is safe.
        self.model: Any = None
        self._role_tokens: dict[str, list[int]] = {}
        self._load_lock = threading.RLock()

        llama_cpp = _import_llama_cpp()
        llama_cls = _llama_class(llama_cpp)
        _log_build(llama_cpp)
        _validate_requested_device(self.device, llama_cpp)
        placement = llama_device_kwargs(self.device, llama_cpp)
        self.on_gpu = placement.get("n_gpu_layers", 0) != 0
        self.n_threads = thread_count(on_gpu=self.on_gpu)
        verbose = verbose_enabled()

        logger.info(
            "Loading %s (%s) on %s: %s, n_threads=%d, n_ctx=%d",
            self.model_path.name,
            _size_on_disk(self.model_path),
            self.device,
            ", ".join(f"{key}={value}" for key, value in placement.items()),
            self.n_threads,
            N_CTX,
        )
        if self.on_gpu:
            logger.info("VNTL is using full accelerator offload (n_gpu_layers=-1)")
        else:
            logger.info("VNTL is using CPU execution (n_gpu_layers=0)")
        if not verbose:
            logger.info("llama.cpp will load quietly; set %s=1 for native llama.cpp loader output", VERBOSE_ENV)
        _flush_logs()

        try:
            llm = llama_cls(
                model_path=str(self.model_path),
                n_ctx=N_CTX,
                n_threads=self.n_threads,
                verbose=verbose,
                **placement,
            )
        except Exception as exc:
            logger.error("llama.cpp failed loading %s on %s: %s", self.model_path.name, self.device, exc)
            if self.on_gpu:
                raise ModelLoadError(f"llama.cpp could not load VNTL on {self.device}: {exc}") from exc
            raise ModelLoadError(f"llama.cpp could not load VNTL on {self.device}: {exc}") from exc

        with self._load_lock:
            self.model = llm
            try:
                self._role_tokens = {role: _tokenize_text(llm, role) for role in _ROLES}
            except Exception as exc:
                try:
                    llm.close()
                except Exception:
                    pass
                self.model = None
                raise ModelLoadError(f"VNTL role tokenization failed: {exc}") from exc
        logger.info("VNTL model loaded (%s)", self.model_path.name)

    # -- labels ---------------------------------------------------------
    def _labels(self, prepared: _PreparedInput) -> dict[str, tuple[str | None, str | None]]:
        labels: dict[str, tuple[str | None, str | None]] = {}
        referenced = {link for link in prepared.context_links if link is not None}
        if prepared.meta_id:
            referenced.add(prepared.meta_id)
        for link in referenced:
            char = prepared.characters.get(link)
            if char is None:
                logger.warning("meta_id %r has no character_info entry; those lines will have no speaker tag", link)
                continue
            labels[link] = (char.resolve_ja(), char.resolve_en())
        return labels

    def _label_for(self, labels: dict[str, tuple[str | None, str | None]], meta_id: str | None, *, english: bool) -> str | None:
        if meta_id is None:
            return None
        pair = labels.get(meta_id)
        if pair is None:
            return None
        return pair[1] if english else pair[0]

    @staticmethod
    def _format_line(label: str | None, text: str) -> str:
        return f"[{label}]: {text}" if label else text

    # -- prompt ---------------------------------------------------------
    def _metadata_body(self, prepared: _PreparedInput) -> str:
        lines: list[str] = []
        for meta_id, char in prepared.characters.items():
            line = format_character_line(char)
            if line is None:
                logger.debug("meta_id %r has nothing renderable; skipping", meta_id)
                continue
            lines.append(line)
        if not lines:
            return ""
        return "\n".join(lines) + "\n"

    def _input_body(self, prepared: _PreparedInput, labels: dict[str, tuple[str | None, str | None]]) -> str:
        if SPEAKER_TAGGING == "never":
            return prepared.text_ja
        label = self._label_for(labels, prepared.meta_id, english=False)
        if label is None:
            return prepared.text_ja
        return self._format_line(label, prepared.text_ja)

    @staticmethod
    def _answer_reserve() -> int:
        return min(MAX_NEW_TOKENS, N_CTX // 2) + 4

    def _build_prompt(self, prepared: _PreparedInput) -> tuple[list[int], int]:
        llm = self.model
        if llm is None:
            raise ModelLoadError("VNTL translator has been unloaded.")
        roles = self._role_tokens
        labels = self._labels(prepared)
        head = [TOK_BEGIN_OF_TEXT] + _encode_body(ROLE_METADATA, self._metadata_body(prepared), llm, roles)
        tail = _encode_body(ROLE_JAPANESE, self._input_body(prepared, labels), llm, roles) + _encode_header(ROLE_ENGLISH, roles)
        fixed_cost = len(head) + len(tail) + self._answer_reserve()
        if fixed_cost >= N_CTX:
            raise PromptBudgetError(
                f"the input alone needs {fixed_cost} tokens but the window is {N_CTX}; shorten the line or clear characters"
            )
        budget = N_CTX - fixed_cost
        encoded: list[list[int]] = []
        for (ja_text, en_text), link in zip(prepared.context, prepared.context_links):
            ja_label = self._label_for(labels, link, english=False)
            en_label = self._label_for(labels, link, english=True)
            ja_line = self._format_line(ja_label, ja_text) + "\n"
            en_line = self._format_line(en_label, en_text) + "\n"
            encoded.append(
                _encode_body(ROLE_JAPANESE, ja_line, llm, roles) + _encode_body(ROLE_ENGLISH, en_line, llm, roles)
            )
        chosen: list[list[int]] = []
        for pair_tokens in reversed(encoded):
            if len(pair_tokens) > budget:
                break
            budget -= len(pair_tokens)
            chosen.append(pair_tokens)
        chosen.reverse()
        dropped = len(encoded) - len(chosen)
        if dropped:
            logger.warning(
                "VNTL context trimmed: dropped the %d oldest of %d turns to fit %d tokens",
                dropped,
                len(encoded),
                N_CTX,
            )
        prompt = head + [tok for pair in chosen for tok in pair] + tail
        return prompt, dropped

    # -- generation ------------------------------------------------------
    def _generate(self, prompt_ids: list[int]) -> tuple[str, float]:
        started = time.time()
        try:
            response = self.model(
                prompt_ids,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                repeat_penalty=REPEAT_PENALTY,
                seed=SEED,
                stop=["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
                echo=False,
            )
        except Exception as exc:
            raise RuntimeError(f"VNTL generation failed: {exc}") from exc
        elapsed = time.time() - started
        try:
            text = response["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("Unexpected llama.cpp response shape: %s", exc)
            text = ""
        return text or "", elapsed

    # -- output parsing ---------------------------------------------------
    def _resolve_tag(self, tag: str, labels: dict[str, tuple[str | None, str | None]], expected: str | None) -> str:
        folded = tag.casefold()
        for ja_label, en_label in labels.values():
            if folded in {(lbl or "").casefold() for lbl in (ja_label, en_label)} - {""}:
                return en_label or ja_label or tag
        if expected and folded != expected.casefold():
            logger.warning("VNTL tagged the line %r but %r was expected; reporting the model's tag", tag, expected)
        return tag

    def _parse(self, output: str, prepared: _PreparedInput, dropped: int, prompt_tokens: int, elapsed: float = 0.0) -> str:
        text = (output or "").strip()
        labels = self._labels(prepared)
        expected = self._label_for(labels, prepared.meta_id, english=True)
        speaker: str | None = None
        plain = text
        # Strip every consecutive leading speaker tag. The model sometimes
        # echoes the input line's tag and then generates its own, yielding
        # "[A]: [A]: ..."; stripping only the first would re-emit the second
        # as content and the caller would show a doubled prefix. The LAST
        # stripped tag wins: it sits closest to the content, so it is the
        # model's final attribution. Bounded so a pathological all-brackets
        # output terminates.
        stripped = 0
        while stripped < 4:
            match = _SPEAKER_RE.match(plain)
            if not match:
                break
            tag = match.group(1).strip()
            plain = plain[match.end() :].lstrip()
            stripped += 1
            if tag:
                speaker = self._resolve_tag(tag, labels, expected)
            elif expected is not None and speaker is None:
                speaker = expected
        if stripped == 0 and expected is not None:
            speaker = expected
        if not text:
            logger.warning("VNTL returned an empty translation for %r", prepared.text_ja[:80])
            speaker = None
        if not plain:
            return ""
        if not prepared.characters and prepared.meta_id is None and not any(prepared.context_links):
            # No character data rode along (the Characters switch is off), so
            # no speaker prefix is ever emitted. The model can still generate
            # "[Name]:" from its own priors; the tags were stripped above, so
            # returning plain keeps the switch a real off-switch.
            return plain
        if SPEAKER_TAGGING != "never" and speaker:
            return self._format_line(speaker, plain)
        return plain

    def count_prompt_tokens(
        self,
        text: str,
        context: object = None,
        character_info: object = None,
        context_character_links: object = None,
        meta_id: object = None,
    ) -> int:
        prepared = _InputValidator.prepare(text, context, character_info, context_character_links, meta_id)
        with self._load_lock:
            tokens, _ = self._build_prompt(prepared)
        return len(tokens)

    # -- translate ---------------------------------------------------------
    def translate(
        self,
        text: str,
        context: object = None,
        lang: str | None = None,
        character_info: object = None,
        context_character_links: object = None,
        meta_id: object = None,
        **kwargs: Any,
    ) -> str:
        source = (text or "").strip() if isinstance(text, str) else ""
        if not source:
            return ""
        if self.model is None:
            raise RuntimeError("This translator has been unloaded.")
        try:
            prepared = _InputValidator.prepare(source, context, character_info, context_character_links, meta_id)
        except (ValueError, TypeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"VNTL could not prepare the request: {exc}") from exc
        with self._load_lock:
            try:
                prompt_ids, dropped = self._build_prompt(prepared)
            except PromptBudgetError as exc:
                raise RuntimeError(
                    f"This text is too long for the model's {N_CTX}-token context window: {exc}"
                ) from exc
            output, _elapsed = self._generate(prompt_ids)
            return self._parse(output, prepared, dropped, len(prompt_ids), _elapsed)

    # -- unload ------------------------------------------------------------
    def unload(self) -> None:
        """Release the native llama.cpp context. Idempotent, thread-safe."""
        with self._load_lock:
            model = getattr(self, "model", None)
            self.model = None
            if model is None:
                return
            close = getattr(model, "close", None)
            if not callable(close):
                return
            try:
                close()
            except Exception as exc:
                logger.debug("llama.cpp did not close cleanly: %s", exc)

    # -- metadata -----------------------------------------------------------
    @classmethod
    def model_id(cls) -> str:
        return "vntl-llama3-8b-v2"

    @classmethod
    def model_langs(cls) -> list[str]:
        return ["japanese"]
