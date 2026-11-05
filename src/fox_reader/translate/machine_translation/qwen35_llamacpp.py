"""Text translation with the Qwen3.5 GGUF family.

Qwen3.5 uses ChatML with an empty, closed thinking block for non-thinking
generation. Build that wire format as tokens: only our role boundaries are
special tokens, while source text and character/context data remain literal.
The exact same tokens are measured for budgeting and sent to llama.cpp.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from fox_reader.constants import mtl_model
from fox_reader.device import CPU, FOX_DEVICE, MPS, thread_count
from fox_reader.translate.context_window import resolve_context_size
from fox_reader.translate.machine_translation.gemma_e4b_llamacpp import (
    _cuda_index,
    _flush_logs,
    _import_llama_cpp,
    _llama_class,
    _log_build,
    _normalize_device,
    clean_translation,
    llama_device_kwargs,
    supports_gpu_offload,
    verbose_enabled,
)
from fox_reader.translate.prompt import build_translation_messages

logger = logging.getLogger(__name__)

N_CTX = 8192
MAX_NEW_TOKENS = 1024
_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
_STOP = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def resolve_weights(mtl_dir: Path | str, model_id: str) -> Path:
    """Require the selected quantization, never another GGUF in its place."""
    entry = mtl_model(model_id)
    if entry is None:
        raise ValueError(f"Unknown Qwen3.5 model: {model_id}")
    path = Path(mtl_dir) / entry["dest"] / entry["files"][0]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Download this model from the setup page."
        )
    return path


def clean_qwen_translation(text: str) -> str:
    """Discard explicitly marked reasoning if a model ignores its prefix."""
    result = text.strip()
    while result.startswith("<think>"):
        end = result.find("</think>")
        if end == -1:
            raise RuntimeError("The model returned incomplete reasoning instead of a translation. Try again.")
        result = result[end + len("</think>"):].lstrip()
    # A stray leading close marker can follow the prefilled empty block.
    # Markers inside translated prose are literal content and must survive.
    if result.startswith("</think>"):
        result = result[len("</think>"):].lstrip()
    return clean_translation(result)


class Qwen35Translator:
    """Shared loader and prompt handling for the supported Qwen3.5 quants."""

    MODEL_ID = ""

    def __init__(self, mtl_dir: Path, device: str | None = None, *, n_ctx: int = N_CTX):
        self._lock = threading.RLock()
        self.model: Any = None
        self.model_path = resolve_weights(mtl_dir, self.model_id())
        self.n_ctx = resolve_context_size(self.model_id(), n_ctx, self.model_path)
        self.device = _normalize_device(device if device is not None else FOX_DEVICE.translator)
        if self.device not in (CPU, MPS) and _cuda_index(self.device) is None:
            raise RuntimeError(f"Unsupported Qwen3.5 translation device: {self.device}")

        llama_cpp = _import_llama_cpp()
        llama_cls = _llama_class(llama_cpp)
        _log_build(llama_cpp)
        if self.device != CPU and supports_gpu_offload(llama_cpp) is False:
            raise RuntimeError(
                f"Qwen3.5 is configured to use {self.device}, but the installed "
                "llama-cpp-python build is CPU-only. Install a build with the "
                "appropriate GPU backend, then restart Fox Reader."
            )
        placement = llama_device_kwargs(self.device, llama_cpp)
        self.on_gpu = placement["n_gpu_layers"] != 0
        self.n_threads = thread_count(on_gpu=self.on_gpu)
        logger.info(
            "Loading %s on %s, n_ctx=%d, n_threads=%d",
            self.model_path.name, self.device, self.n_ctx, self.n_threads,
        )
        _flush_logs()
        try:
            self.model = llama_cls(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_threads_batch=thread_count(on_gpu=self.on_gpu, reserve=False),
                verbose=verbose_enabled(),
                **placement,
            )
            if self.model.metadata.get("general.architecture") != "qwen35":
                raise RuntimeError("The selected GGUF does not use the Qwen3.5 (qwen35) architecture.")
            self._special = {}
            for token in _SPECIAL_TOKENS:
                ids = self.model.tokenize(token.encode("utf-8"), add_bos=False, special=True)
                if len(ids) != 1:
                    raise RuntimeError(f"The selected GGUF lacks the Qwen3.5 control token {token}.")
                self._special[token] = ids[0]
        except Exception as exc:
            self.unload()
            raise RuntimeError(
                f"Could not load Qwen3.5 on {self.device}: {exc} "
                "This model needs a llama-cpp-python build supporting Qwen3.5 (qwen35)."
            ) from exc

    def _encode(self, text: str) -> list[int]:
        return list(self.model.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def _prompt_tokens(self, messages: list[dict[str, str]]) -> list[int]:
        """The text-only, non-thinking branch of Qwen3.5's chat template."""
        tokens: list[int] = []
        for message in messages:
            tokens.append(self._special["<|im_start|>"])
            tokens.extend(self._encode(message["role"] + "\n" + message["content"].strip()))
            tokens.append(self._special["<|im_end|>"])
            tokens.extend(self._encode("\n"))
        tokens.append(self._special["<|im_start|>"])
        tokens.extend(self._encode("assistant\n"))
        tokens.append(self._special["<think>"])
        tokens.extend(self._encode("\n\n"))
        tokens.append(self._special["</think>"])
        tokens.extend(self._encode("\n\n"))
        return tokens

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
        # Tokenization also touches the native model, so it belongs under the
        # same lock as generation and unload.
        with self._lock:
            if self.model is None:
                raise RuntimeError("This translator has been unloaded.")
            messages = build_translation_messages(
                self.model,
                source,
                context,
                lang,
                character_info,
                context_character_links,
                meta_id,
                n_ctx=self.n_ctx,
                max_new_tokens=MAX_NEW_TOKENS,
                prompt_token_count=lambda items: len(self._prompt_tokens(items)),
            )
            prompt = self._prompt_tokens(messages)
            response = self.model.create_completion(
                prompt=prompt,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                repeat_penalty=1.0,
                max_tokens=MAX_NEW_TOKENS,
                stop=list(_STOP),
            )
            try:
                choice = response["choices"][0]
                content = choice["text"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("The model returned an invalid translation response.") from exc
            if not isinstance(content, str):
                raise RuntimeError("The model returned an invalid translation response.")
            if choice.get("finish_reason") == "length":
                raise RuntimeError("The translation exceeded the model's output limit. Translate a shorter passage.")
            result = clean_qwen_translation(content)
            if not result:
                raise RuntimeError("The model returned an empty translation. Try again.")
            return result

    def unload(self) -> None:
        with self._lock:
            model, self.model = self.model, None
            if model is not None:
                try:
                    model.close()
                except Exception as exc:
                    logger.debug("llama.cpp did not close cleanly: %s", exc)

    @classmethod
    def model_id(cls) -> str:
        return cls.MODEL_ID

    @classmethod
    def model_langs(cls) -> list[str]:
        return ["japanese", "chinese", "korean"]


class Qwen35_9BQ6Translator(Qwen35Translator):
    MODEL_ID = "qwen3.5-9b-q6-k-uncensored"


class Qwen35_9BQ8Translator(Qwen35Translator):
    MODEL_ID = "qwen3.5-9b-q8-0-uncensored"


class Qwen35_4BQ8Translator(Qwen35Translator):
    MODEL_ID = "qwen3.5-4b-q8-0-uncensored"
