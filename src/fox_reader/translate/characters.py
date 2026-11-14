"""Character metadata for local translation models and custom endpoints.

The active destination holds ``fox_characters_data.json``. Every roster change
is saved atomically; loading a folder restores its validated roster and speaker
ids. Before a folder is loaded, the roster stays in memory. CSV import/export
remains available for sharing characters between projects.

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
import json
import logging
import os
import stat
import tempfile
import threading
import uuid
from pathlib import Path

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
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<|turn>",
    "<turn|>",
    "<think>",
    "</think>",
    "<<START>>",
    "<<JAPANESE>>",
    "<<ENGLISH>>",
)

#: Per-field safety cut. Names are short; anything longer is pasted prose.
MAX_FIELD_CHARS = 100

#: How many characters travel with a request at most. The metadata block is
#: re-tokenized on every call, so an unbounded roster would eat the window.
MAX_CHARACTERS = 50

#: Versioned destination-side roster, including stable speaker ids.
CHARACTERS_FILENAME = "fox_characters_data.json"
# Includes JSON escaping overhead at the row, field and id limits below.
MAX_ROSTER_BYTES = 256 * 1024
MAX_META_ID_CHARS = 128


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
    """Thread-safe character roster with optional destination persistence.

    Folder loads and mutations share one lock. Changes reach memory only after
    their atomic save succeeds, so failed writes leave the previous roster intact.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._characters: dict[str, dict] = {}
        self._path: Path | None = None

    @staticmethod
    def _unique_object(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    @staticmethod
    def _validate_saved(data: object) -> dict[str, dict]:
        if (not isinstance(data, dict) or set(data) != {"version", "characters"}
                or type(data["version"]) is not int or data["version"] != 1):
            raise ValueError("unsupported character document")
        rows = data["characters"]
        if not isinstance(rows, list) or len(rows) > MAX_CHARACTERS:
            raise ValueError("invalid character list")
        characters: dict[str, dict] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != CHARACTER_KEYS:
                raise ValueError(f"invalid character row {index}")
            mid = row["meta_id"]
            if (not isinstance(mid, str) or not mid or mid != mid.strip()
                    or len(mid) > MAX_META_ID_CHARS or any(ord(c) < 32 for c in mid)
                    or mid in characters):
                raise ValueError(f"invalid or duplicate character id at row {index}")
            if any(row[field] is not None and not isinstance(row[field], str) for field in CSV_FIELDS):
                raise ValueError(f"invalid character field at row {index}")
            for value in row.values():
                if isinstance(value, str):
                    value.encode("utf-8")  # JSON escapes can hide unpaired surrogates.
            cleaned = clean_character_dict(row, index=index)
            if cleaned != row or not any(row[field] for field in CSV_FIELDS):
                raise ValueError(f"invalid character values at row {index}")
            characters[mid] = cleaned
        return characters

    def load_destination(self, directory: str | Path | None) -> list[dict]:
        """Bind to a destination and restore its roster without writing a file.

        Missing, malformed, oversized and non-regular files are ignored as a
        whole. The next edit can replace an invalid file. ``None`` detaches the
        store and clears memory without altering the previous destination.
        """
        path = Path(directory).absolute() / CHARACTERS_FILENAME if directory else None
        with self._lock:
            characters: dict[str, dict] = {}
            if path is not None:
                try:
                    metadata = path.lstat()
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_ROSTER_BYTES:
                        raise ValueError("character file is not a bounded regular file")
                    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
                             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(os.open(path, flags), "rb") as stream:
                        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                            raise ValueError("character file is not regular")
                        raw = stream.read(MAX_ROSTER_BYTES + 1)
                    if len(raw) > MAX_ROSTER_BYTES:
                        raise ValueError("character file is too large")
                    data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=self._unique_object)
                    characters = self._validate_saved(data)
                except FileNotFoundError:
                    pass
                except (OSError, ValueError, UnicodeError, RecursionError) as exc:
                    logger.warning("Ignoring character file %s: %s", path, exc)
            self._path = path
            self._characters = characters
            return [dict(char) for char in characters.values()]

    def _commit_locked(self, characters: dict[str, dict]) -> None:
        if self._path is not None:
            path = self._path
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent,
                    prefix=".fox_characters_data-", suffix=".tmp", delete=False,
                ) as stream:
                    temporary = stream.name
                    json.dump({"version": 1, "characters": list(characters.values())},
                              stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        logger.warning("Could not remove temporary character file %s", temporary)
        self._characters = characters

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

    def _insert_locked(self, fields: dict, characters: dict[str, dict]) -> dict:
        existing = set(characters)
        mid = generate_meta_id(existing)
        char = {
            "meta_id": mid,
            "name_en": fields.get("name_en"),
            "name_ja": fields.get("name_ja"),
            "gender": fields.get("gender"),
            "alias_en": fields.get("alias_en"),
            "alias_ja": fields.get("alias_ja"),
        }
        characters[mid] = char
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
            characters = dict(self._characters)
            created = self._insert_locked(cleaned, characters)
            self._commit_locked(characters)
            return created

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
            if not any(updated[field] for field in CSV_FIELDS):
                raise ValueError("character needs at least one of: name_en, name_ja, gender, alias_en, alias_ja")
            characters = dict(self._characters)
            characters[mid] = updated
            self._commit_locked(characters)
            return dict(updated)

    def delete(self, meta_id: object) -> bool:
        mid = normalise_meta_id(meta_id)
        if mid is None:
            return False
        with self._lock:
            if mid not in self._characters:
                return False
            characters = dict(self._characters)
            del characters[mid]
            self._commit_locked(characters)
            return True

    def clear(self) -> int:
        """Delete and save the whole roster. Returns rows removed."""
        with self._lock:
            removed = len(self._characters)
            self._commit_locked({})
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
            characters: dict[str, dict] = {}
            out = [self._insert_locked(fields, characters) for fields in cleaned]
            self._commit_locked(characters)
            return out

    def import_csv(self, text: str) -> list[dict]:
        """Replace the roster from CSV text (no meta_id column)."""
        return self.replace_all(parse_csv(text))

    def export_csv(self) -> str:
        with self._lock:
            rows = list(self._characters.values())
        return to_csv(rows)


#: Unbound fallback for standalone character API users. The running app keeps
#: its destination-bound roster on SessionService.
store = CharacterStore()
