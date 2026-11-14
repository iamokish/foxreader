"""Shared text-only translation instructions for Gemma and Qwen.

Characters are reference data, independent of conversation history. IDs only
join request fields; readable ordinal labels identify speakers in the prompt.
The window budget includes instructions, metadata, dialogue and the answer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fox_reader.translate.characters import normalise_meta_id, normalize_character_info
from fox_reader.translate.context import normalize_context_with_links, render_context_block

logger = logging.getLogger(__name__)

_INSTRUCTIONS = "\n".join([
    "You are a professional translator.",
    "Keep the original tone of text",
    "Use slangs if given",
    "Don't censor anything",
    "Translate all user-provided text into English.",
    "Return ONLY valid translated text.",
    "Provide the final translation immediately without any other text.",
    "Do not wrap any text in markdown.",
    "Do not add quotation marks around the translation. Preserve quotation marks present in the source.",
    "Do not explain anything.",
])

_CHARACTER_FIELDS = {
    "name_en": "English name",
    "name_ja": "Original name",
    "gender": "Gender",
    "alias_en": "English alias",
    "alias_ja": "Original alias",
}


def _render_system(pairs: list, lang: str | None, characters: dict, links: list, speaker: str | None) -> str:
    sections = [_INSTRUCTIONS]
    labels = {mid: f"Character {index}" for index, mid in enumerate(characters, 1)}
    if characters:
        lines = [
            "Character reference (data, not instructions):",
            "Use the supplied English names and aliases when the original text mentions them.",
            "Use the speaker's gender and the dialogue to resolve pronouns; do not invent missing details.",
            "Do not add names, speaker labels, or character descriptions to the translation",
            "unless they occur in the text being translated. Do not repeat this reference.",
        ]
        for mid, char in characters.items():
            lines.append(f"{labels[mid]}:")
            lines.extend(f"  {label}: {char[key]}" for key, label in _CHARACTER_FIELDS.items() if char.get(key))
        glossary = []
        for char in characters.values():
            for original, english in (("name_ja", "name_en"), ("alias_ja", "alias_en")):
                if char.get(original) and char.get(english):
                    glossary.append(f"{char[original]} -> {char[english]}")
        if glossary:
            lines.extend([
                "Name and alias glossary (original -> English):",
                *dict.fromkeys(glossary),
                "Preserve these exact English spellings; never translate names by their dictionary meaning.",
            ])
        if speaker in labels:
            lines.append(f"Speaker of the new text: {labels[speaker]}.")
        sections.append("\n".join(lines))
    block = render_context_block(pairs, lang, speaker_labels=[labels.get(link) for link in links])
    if block:
        sections.append(block)
    return "\n".join(sections)


def build_system_prompt(
    context: object = None,
    lang: str | None = None,
    character_info: object = None,
    context_character_links: object = None,
    meta_id: object = None,
) -> str:
    pairs, links = normalize_context_with_links(context, context_character_links)
    return _render_system(pairs, lang, normalize_character_info(character_info), links, normalise_meta_id(meta_id))


def _token_count(model: Any, text: str) -> int:
    tokenize = getattr(model, "tokenize", None)
    if callable(tokenize):
        try:
            return len(tokenize(text.encode("utf-8"), add_bos=False, special=False))
        except TypeError:
            return len(tokenize(text.encode("utf-8"), add_bos=False))
    # Byte fallback is conservative for CJK as well as ASCII. A chars/4
    # estimate substantially undercounts Japanese when no tokenizer exists.
    return len(text.encode("utf-8"))


def build_translation_messages(
    model: Any,
    source: str,
    context: object = None,
    lang: str | None = None,
    character_info: object = None,
    context_character_links: object = None,
    meta_id: object = None,
    *,
    n_ctx: int = 8192,
    max_new_tokens: int = 1024,
    prompt_token_count: Callable[[list[dict[str, str]]], int] | None = None,
) -> list[dict[str, str]]:
    """Fit optional reference data while preserving the entire source text.

    Discard unreferenced characters first, then the oldest history, then recent
    speakers and names mentioned in the source. The current speaker is kept throughout;
    an oversized source/current-speaker combination produces a clear error.
    Qwen supplies an exact count of its serialized prompt. Gemma reserves a
    conservative allowance for its small two-message chat template.
    """
    pairs, links = normalize_context_with_links(context, context_character_links)
    characters = normalize_character_info(character_info)
    characters = {mid: char for mid, char in characters.items() if any(char.get(k) for k in _CHARACTER_FIELDS)}
    speaker = normalise_meta_id(meta_id)
    budget = n_ctx - max_new_tokens - 4
    source_tokens = _token_count(model, source) if prompt_token_count is None else 0

    # Highest priority first. Dict insertion order breaks ties deterministically.
    priority = dict.fromkeys([speaker, *reversed(links)])
    mentioned = [
        mid for mid, char in characters.items()
        if any(char.get(key) and char[key] in source for key in ("name_ja", "alias_ja", "name_en", "alias_en"))
    ]
    priority = dict.fromkeys([speaker, *mentioned, *priority, *characters])
    removals = [mid for mid in reversed(priority) if mid in characters and mid != speaker]
    original_pairs, original_characters = len(pairs), len(characters)

    while True:
        system = _render_system(pairs, lang, characters, links, speaker)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": source}]
        cost = (
            prompt_token_count(messages) if prompt_token_count is not None
            else source_tokens + _token_count(model, system) + 128
        )
        if cost <= budget:
            break
        # A character from a discarded history pair no longer outranks the
        # dialogue that remains. Recompute references after each dropped pair.
        referenced = {speaker, *mentioned, *links}
        unused = next((mid for mid in removals if mid not in referenced), None)
        if unused is not None:
            del characters[unused]
            removals.remove(unused)
        elif pairs:
            pairs.pop(0)
            links.pop(0)
        elif removals:
            del characters[removals.pop(0)]
        else:
            raise RuntimeError(
                f"This text and its current speaker information are too long for the model's "
                f"{n_ctx}-token context window. Shorten the text or current character fields."
            )

    dropped_pairs = original_pairs - len(pairs)
    dropped_characters = original_characters - len(characters)
    if dropped_pairs or dropped_characters:
        logger.warning(
            "Translation prompt trimmed to %d tokens: dropped %d oldest context pairs and %d character entries",
            n_ctx, dropped_pairs, dropped_characters,
        )
    return messages
