"""Translation context: earlier bubbles helping the MTL model stay consistent.

The frontend sends the already-translated entries before the one being
translated as ``[[source, english], ...]`` in user (reading) order. Only the
Gemma GGUF translators render it into the prompt, plus custom endpoints whose
request schema carries a Context node; every other model ignores it. This
module is deliberately dependency-free so the contract (validation,
capability, prompt block) can be unit-tested without torch or llama.cpp.
"""

from __future__ import annotations

#: How many previous pairs travel with a request at most. The single obvious
#: knob: raise it and prompts grow towards the 8192-token window (2048 of
#: which is reserved for the answer); lower it and long pages lose early story.
MAX_CONTEXT_PAIRS = 10

#: Per-side safety cut. Bubble text is short; anything longer is pasted logs,
#: not dialogue, and must not be allowed to eat the window on its own.
MAX_PAIR_CHARS = 1000

#: Model ids that render context into the prompt. Everything else translates
#: exactly as before, context or no context.
CONTEXT_MODEL_IDS = frozenset(
    {
        "gemma-4-e4b-q8-uncensored",
        "gemma-4-e4b-q6-uncensored",
        "vntl-llama3-8b-v2",
    }
)

#: Model ids that additionally render per-character metadata (name/gender /
#: aliases) and per-turn speaker attribution. A subset of CONTEXT_MODEL_IDS:
#: characters only mean anything alongside the history they annotate.
CHARACTER_MODEL_IDS = frozenset(
    {
        "vntl-llama3-8b-v2",
    }
)

#: Source-language names for the prompt header, keyed by request language.
SOURCE_NAMES = {
    "japanese": "Japanese",
    "chinese": "Chinese",
    "korean": "Korean",
}


def model_supports_context(model_id: str | None) -> bool:
    """Whether this model id renders context into its prompt."""
    return str(model_id or "").strip() in CONTEXT_MODEL_IDS


def model_supports_characters(model_id: str | None) -> bool:
    """Whether this model id renders character metadata + speaker tags."""
    return str(model_id or "").strip() in CHARACTER_MODEL_IDS


def _clean_side(value: object) -> str:
    """One side of a pair: stripped, single-line, length-capped."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    if len(text) > MAX_PAIR_CHARS:
        text = text[:MAX_PAIR_CHARS].rstrip()
    return text


def normalize_context(value: object) -> list[list[str]]:
    """Keep only usable ``[source, english]`` pairs, oldest first.

    A pair counts when both sides are non-empty strings after cleaning; the
    tail (most recent) wins when there are more than ``MAX_CONTEXT_PAIRS``.
    Anything else -- ``None``, a flat list, pairs with a blank side -- reads
    as "no context", which callers treat exactly like the first entry having
    nothing before it.
    """
    if not isinstance(value, (list, tuple)):
        return []

    pairs: list[list[str]] = []

    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        source = _clean_side(item[0])
        english = _clean_side(item[1])
        if source and english:
            pairs.append([source, english])

    if len(pairs) > MAX_CONTEXT_PAIRS:
        pairs = pairs[-MAX_CONTEXT_PAIRS:]

    return pairs


def source_name(lang: str | None) -> str:
    """A human name for the request language, for the prompt header."""
    return SOURCE_NAMES.get(str(lang or "").strip().lower(), "Original")


def render_context_block(pairs: list[list[str]], lang: str | None = None) -> str:
    """The verbose context section of the system prompt.

    States what the pairs are (earlier bubbles, oldest first), what they are
    for (consistent tone, pronouns, names, terminology), and what they are not
    (not to be translated or repeated). Empty in, empty out.
    """
    clean = normalize_context(pairs)

    if not clean:
        return ""

    lines = [
        "Conversation context: the pairs below are earlier bubbles from the same page,",
        "oldest first. They establish the story, the speakers, character names,",
        "pronouns and terminology.",
        "Use them to keep the new translation consistent in tone, pronouns, names",
        "and word choices with what came before.",
        "Do NOT translate, repeat or explain the context. Translate ONLY the new",
        "user-provided text that follows the context.",
    ]

    label = source_name(lang)

    for index, (source, english) in enumerate(clean, start=1):
        lines.append(f'{index}. {label}: "{source}"')
        lines.append(f'   English: "{english}"')

    lines.append("Now translate the new text that follows into English.")
    return "\n".join(lines)
