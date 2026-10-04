"""Character metadata for the VNTL-style MTL models.

The roster lives only in memory on the backend: nothing is written to disk by
this module. The frontend offers CSV import/export, and the user re-imports the
file themselves after a restart. Switching the source folder clears the roster
(the frontend calls :meth:`CharacterStore.clear` when a new folder is loaded).

A character is a plain dict with exactly these keys::

    {"meta_id": str, "name_en": str|None, "name_ja": str|None,
     "gender": "male"|"female"|None, "alias_en": str|None, "alias_ja": str|None}

``meta_id`` is opaque: it is never rendered into the prompt, only used as the
join key between ``character_info``, ``context_character_links`` and ``meta_id``
(the speaker of the current line). It is generated here (short hex) when a
character is added or imported -- the CSV format therefore carries no
``meta_id`` column::

    name_en,name_ja,gender,alias_en,alias_ja

This module is deliberately dependency-free (stdlib + logging only) so the
contract can be unit-tested without torch or llama.cpp. The VNTL translator
validates the same way again at translate time (defence in depth); these
helpers are what the HTTP layer and ``LocalMTLManager`` use.
"""

from __future__ import annotations

import csv
import io
import logging
import threading
import uuid

logger = logging.getLogger(__name__)

#: CSV columns, in order. No ``meta_id``: ids are generated on add/import.
CSV_FIELDS = ("name_en", "name_ja", "gender", "alias_en", "alias_ja")

#: Header row written by :func:`to_csv` and expected by :func:`parse_csv`.
CSV_HEADER = ",".join(CSV_FIELDS)

#: Field keys a character dict may carry (plus ``meta_id``).
CHARACTER_KEYS = frozenset({"meta_id", *CSV_FIELDS})

#: Genders the VNTL prompt renders. Anything else is dropped (kept as None).
VALID_GENDERS = frozenset({"male", "female"})

#: Markers stripped from name/alias fields so a pasted prompt fragment cannot
#: poison the ``[character]`` metadata block. Mirrors translator.py.
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

#: Per-field safety cut. Names are short; anything longer is pasted prose.
MAX_FIELD_CHARS = 100

#: How many characters travel with a request at most. The metadata block is
#: re-tokenized on every call, so an unbounded roster would eat the window.
MAX_CHARACTERS = 50


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------


def _sanitise(value: str) -> str:
    """Strip control markers and flatten whitespace (one line per character)."""
    for marker in _CONTROL_MARKERS:
        if marker in value:
            value = value.replace(marker, "")
    return " ".join(value.split())


def _clean_field(value: object) -> str | None:
    """One name/alias field as text, or None when unusable."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return None
    elif isinstance(value, (list, tuple, set, dict)):
        return None
    text = _sanitise(str(value))
    if not text:
        return None
    if len(text) > MAX_FIELD_CHARS:
        text = text[:MAX_FIELD_CHARS].rstrip() or None
        if not text:
            return None
    return text


def _clean_gender(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return None
    elif isinstance(value, (list, tuple, set, dict)):
        return None
    text = str(value).strip().lower()
    return text if text in VALID_GENDERS else None


def normalise_meta_id(value: object) -> str | None:
    """Canonical form of an opaque character key, or None.

    Deliberately NOT sanitised beyond strip: a meta_id is only a dict key,
    never rendered, so sanitising one side of a lookup and not the other
    would silently lose every speaker tag.
    """
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


def generate_meta_id(existing: set[str] | None = None) -> str:
    """A fresh opaque id, unique against ``existing``."""
    seen = existing or set()
    for _ in range(16):
        candidate = uuid.uuid4().hex[:8]
        if candidate not in seen:
            return candidate
    return uuid.uuid4().hex


def clean_character_dict(raw: object, *, index: int = 0) -> dict | None:
    """One raw mapping as a clean character dict (with meta_id), or None.

    Returns None when the entry is unusable (not a dict, or no meta_id and
    nothing renderable). Logsand drops bad fields rather than raising: a
    translation costs seconds, so one malformed row must not discard the work.
    """
    if not isinstance(raw, dict):
        logger.warning("character_info[%d] is %s, not a dict; ignoring it", index, type(raw).__name__)
        return None
    meta_id = normalise_meta_id(raw.get("meta_id"))
    name_en = _clean_field(raw.get("name_en"))
    name_ja = _clean_field(raw.get("name_ja"))
    alias_en = _clean_field(raw.get("alias_en"))
    alias_ja = _clean_field(raw.get("alias_ja"))
    gender = _clean_gender(raw.get("gender"))
    if raw.get("gender") is not None and gender is None:
        logger.debug("character_info[%d] gender %r is non-standard; omitting it", index, raw.get("gender"))
    if meta_id is None:
        # No key to link this character to any line. An entry that carried a
        # real name is worth a warning; an empty placeholder is not.
        if any((name_en, name_ja, gender, alias_en, alias_ja)):
            logger.warning(
                "character_info[%d] has no meta_id and cannot be linked; ignoring it", index
            )
        return None
    return {
        "meta_id": meta_id,
        "name_en": name_en,
        "name_ja": name_ja,
        "gender": gender,
        "alias_en": alias_en,
        "alias_ja": alias_ja,
    }


def normalize_character_info(raw: object) -> dict[str, dict]:
    """Caller-supplied ``character_info`` as ``{meta_id: character}``.

    Accepts a list of dicts, a single character dict, or a
    ``{meta_id: dict}`` map (mirroring translator.py). Anything malformed is
    dropped with a warning; duplicates keep the first. Capped at
    ``MAX_CHARACTERS`` (oldest... first-seen wins, extras dropped).
    """
    if raw is None:
        return {}
    items: list[object]
    if isinstance(raw, dict):
        if any(key in raw for key in CHARACTER_KEYS):
            items = [raw]
        elif raw and all(isinstance(v, dict) for v in raw.values()):
            items = [{**v, "meta_id": v.get("meta_id", k)} for k, v in raw.items()]
        else:
            logger.warning("character_info is a dict with no recognised fields; ignoring it")
            return {}
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        logger.warning("character_info is a %s, expected a list; ignoring it", type(raw).__name__)
        return {}
    out: dict[str, dict] = {}
    for index, item in enumerate(items):
        if len(out) >= MAX_CHARACTERS:
            logger.warning("character_info has more than %d entries; ignoring the rest", MAX_CHARACTERS)
            break
        cleaned = clean_character_dict(item, index=index)
        if cleaned is None:
            continue
        mid = cleaned["meta_id"]
        if mid in out:
            logger.warning("duplicate meta_id %r in character_info; keeping the first", mid)
            continue
        out[mid] = cleaned
    return out


def normalize_character_links(raw: object, count: int) -> list[str | None]:
    """``context_character_links`` padded/truncated to ``count`` entries."""
    if raw is None:
        return [None] * count
    if not isinstance(raw, (list, tuple)):
        logger.warning(
            "context_character_links is a %s, expected a list; ignoring it", type(raw).__name__
        )
        return [None] * count
    links: list[str | None] = []
    for item in raw:
        if isinstance(item, (list, tuple, set, dict)):
            links.append(None)
        else:
            links.append(normalise_meta_id(item))
    if len(links) < count:
        links.extend([None] * (count - len(links)))
    elif len(links) > count:
        links = links[:count]
    return links


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def parse_csv(text: str) -> list[dict]:
    """Parse roster CSV (no meta_id column) into clean field dicts.

    Header ``name_en,name_ja,gender,alias_en,alias_ja`` (case-insensitive,
    surrounding whitespace ignored). Extra columns are ignored; missing
    columns read as blank. Blank rows are skipped. Never raises on content:
    malformed rows are skipped with a warning.
    """
    if text is None:
        return []
    if isinstance(text, (bytes, bytearray)):
        try:
            text = text.decode("utf-8-sig")
        except Exception:
            return []
    content = str(text)
    # Strip a BOM left by Excel/Notepad; csv would otherwise keep it in the
    # first header name and the first column would never match.
    if content.startswith("\ufeff"):
        content = content.lstrip("\ufeff")
    if not content.strip():
        return []
    try:
        reader = csv.DictReader(io.StringIO(content))
    except Exception as exc:
        logger.warning("Could not parse character CSV: %s", exc)
        return []
    if reader.fieldnames is None:
        return []
    # Normalise header names once so "Name_EN " still maps.
    lowered = {str(name or "").strip().lower(): name for name in reader.fieldnames}
    out: list[dict] = []
    for row_index, row in enumerate(reader):
        try:
            mapped = {field: row.get(lowered.get(field, field)) for field in CSV_FIELDS}
        except Exception:
            logger.warning("Skipping malformed character CSV row %d", row_index + 1)
            continue
        name_en = _clean_field(mapped.get("name_en"))
        name_ja = _clean_field(mapped.get("name_ja"))
        alias_en = _clean_field(mapped.get("alias_en"))
        alias_ja = _clean_field(mapped.get("alias_ja"))
        gender = _clean_gender(mapped.get("gender"))
        if not any((name_en, name_ja, gender, alias_en, alias_ja)):
            continue
        if len(out) >= MAX_CHARACTERS:
            logger.warning("Character CSV has more than %d rows; ignoring the rest", MAX_CHARACTERS)
            break
        out.append(
            {"name_en": name_en, "name_ja": name_ja, "gender": gender, "alias_en": alias_en, "alias_ja": alias_ja}
        )
    return out


def to_csv(characters: list[dict]) -> str:
    """Roster as CSV text (header + one row per character, no meta_id)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_FIELDS), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for char in characters or []:
        if not isinstance(char, dict):
            continue
        writer.writerow(
            {
                "name_en": char.get("name_en") or "",
                "name_ja": char.get("name_ja") or "",
                "gender": char.get("gender") or "",
                "alias_en": char.get("alias_en") or "",
                "alias_ja": char.get("alias_ja") or "",
            }
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


class CharacterStore:
    """The in-memory character roster.

    Thread-safe (RLock). Nothing here touches disk: import/export move CSV
    text through the caller, and a restart simply starts empty again.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._characters: dict[str, dict] = {}

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(char) for char in self._characters.values()]

    def get(self, meta_id: object) -> dict | None:
        mid = normalise_meta_id(meta_id)
        if mid is None:
            return None
        with self._lock:
            found = self._characters.get(mid)
            return dict(found) if found is not None else None

    def _insert_locked(self, fields: dict) -> dict:
        existing = set(self._characters)
        mid = generate_meta_id(existing)
        char = {
            "meta_id": mid,
            "name_en": fields.get("name_en"),
            "name_ja": fields.get("name_ja"),
            "gender": fields.get("gender"),
            "alias_en": fields.get("alias_en"),
            "alias_ja": fields.get("alias_ja"),
        }
        self._characters[mid] = char
        return dict(char)

    def add(self, raw: object) -> dict:
        """Add one character (no meta_id expected); returns it with its id."""
        if not isinstance(raw, dict):
            raise ValueError("character must be an object with name/gender/alias fields")
        cleaned = {
            "name_en": _clean_field(raw.get("name_en")),
            "name_ja": _clean_field(raw.get("name_ja")),
            "gender": _clean_gender(raw.get("gender")),
            "alias_en": _clean_field(raw.get("alias_en")),
            "alias_ja": _clean_field(raw.get("alias_ja")),
        }
        if not any(cleaned.values()):
            raise ValueError("character needs at least one of: name_en, name_ja, gender, alias_en, alias_ja")
        with self._lock:
            if len(self._characters) >= MAX_CHARACTERS:
                raise ValueError(f"too many characters (max {MAX_CHARACTERS})")
            return self._insert_locked(cleaned)

    def update(self, meta_id: object, raw: object) -> dict:
        mid = normalise_meta_id(meta_id)
        if mid is None:
            raise KeyError("unknown character")
        if not isinstance(raw, dict):
            raise ValueError("character must be an object with name/gender/alias fields")
        with self._lock:
            if mid not in self._characters:
                raise KeyError(f"unknown character {mid!r}")
            current = self._characters[mid]
            # Partial update: only keys present in the payload change. An
            # explicit null/blank clears that field.
            updated = dict(current)
            if "name_en" in raw:
                updated["name_en"] = _clean_field(raw.get("name_en"))
            if "name_ja" in raw:
                updated["name_ja"] = _clean_field(raw.get("name_ja"))
            if "gender" in raw:
                updated["gender"] = _clean_gender(raw.get("gender"))
            if "alias_en" in raw:
                updated["alias_en"] = _clean_field(raw.get("alias_en"))
            if "alias_ja" in raw:
                updated["alias_ja"] = _clean_field(raw.get("alias_ja"))
            if not any((updated.get("name_en"), updated.get("name_ja"), updated.get("gender"), updated.get("alias_en"), updated.get("alias_ja"))):
                raise ValueError("character needs at least one of: name_en, name_ja, gender, alias_en, alias_ja")
            self._characters[mid] = updated
            return dict(updated)

    def delete(self, meta_id: object) -> bool:
        mid = normalise_meta_id(meta_id)
        if mid is None:
            return False
        with self._lock:
            return self._characters.pop(mid, None) is not None

    def clear(self) -> int:
        """Drop the whole roster (folder change). Returns rows removed."""
        with self._lock:
            removed = len(self._characters)
            self._characters.clear()
            return removed

    def replace_all(self, raws: object) -> list[dict]:
        """Replace the roster (CSV import path). Generates fresh meta_ids."""
        if raws is None:
            raws = []
        if isinstance(raws, dict):
            raws = [raws]
        if not isinstance(raws, (list, tuple)):
            raise ValueError("characters must be a list")
        cleaned: list[dict] = []
        for index, item in enumerate(raws):
            if not isinstance(item, dict):
                logger.warning("import character[%d] is %s; skipping it", index, type(item).__name__)
                continue
            fields = {
                "name_en": _clean_field(item.get("name_en")),
                "name_ja": _clean_field(item.get("name_ja")),
                "gender": _clean_gender(item.get("gender")),
                "alias_en": _clean_field(item.get("alias_en")),
                "alias_ja": _clean_field(item.get("alias_ja")),
            }
            if not any(fields.values()):
                continue
            cleaned.append(fields)
            if len(cleaned) >= MAX_CHARACTERS:
                logger.warning("Import has more than %d characters; ignoring the rest", MAX_CHARACTERS)
                break
        with self._lock:
            self._characters.clear()
            out = [self._insert_locked(fields) for fields in cleaned]
            return out

    def import_csv(self, text: str) -> list[dict]:
        """Replace the roster from CSV text (no meta_id column)."""
        return self.replace_all(parse_csv(text))

    def export_csv(self) -> str:
        with self._lock:
            rows = list(self._characters.values())
        return to_csv(rows)


#: The process-wide roster. In-memory only: a restart starts empty and the
#: user re-imports their CSV.
store = CharacterStore()
