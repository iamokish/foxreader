"""Character roster routes: in-memory CRUD + CSV import/export.

Nothing here touches disk. The roster lives in
``fox_reader.translate.characters.store`` for the life of the process; the
frontend downloads it as CSV (a real file on the user's disk) and re-imports
it after a restart. Switching the source folder clears it -- the frontend
calls ``DELETE /api/characters`` when a new folder is loaded.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from fox_reader.translate.characters import parse_csv, store, to_csv

logger = logging.getLogger(__name__)

router = APIRouter()


class CharacterUpsert(BaseModel):
    name_en: Any = None
    name_ja: Any = None
    gender: Any = None
    alias_en: Any = None
    alias_ja: Any = None


class CharacterImport(BaseModel):
    characters: Any = None
    csv: Any = None


def _ok(data: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(content=data, status_code=status)


@router.get("/api/characters")
async def list_characters():
    try:
        return _ok({"characters": store.list()})
    except Exception as exc:
        logger.warning("Could not list characters: %s", exc)
        return _ok({"characters": [], "error": str(exc)})


@router.post("/api/characters")
async def add_character(payload: CharacterUpsert):
    try:
        # Only keys the caller sent: absent fields stay null rather than
        # clearing anything (matters for update; harmless for add).
        created = store.add(payload.model_dump(exclude_unset=True))
    except ValueError as exc:
        return _ok({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.warning("Could not add character: %s", exc)
        return _ok({"error": str(exc)}, status=500)
    return _ok({"character": created}, status=201)


@router.put("/api/characters/{meta_id}")
async def update_character(meta_id: str, payload: CharacterUpsert):
    try:
        updated = store.update(meta_id, payload.model_dump(exclude_unset=True))
    except KeyError:
        return _ok({"error": "unknown character"}, status=404)
    except ValueError as exc:
        return _ok({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.warning("Could not update character %r: %s", meta_id, exc)
        return _ok({"error": str(exc)}, status=500)
    return _ok({"character": updated})


@router.delete("/api/characters/{meta_id}")
async def delete_character(meta_id: str):
    try:
        removed = store.delete(meta_id)
    except Exception as exc:
        logger.warning("Could not delete character %r: %s", meta_id, exc)
        return _ok({"error": str(exc)}, status=500)
    if not removed:
        return _ok({"error": "unknown character"}, status=404)
    return _ok({"status": True})


@router.delete("/api/characters")
async def clear_characters():
    """Drop the whole roster (called when the source folder changes)."""
    try:
        removed = store.clear()
    except Exception as exc:
        logger.warning("Could not clear characters: %s", exc)
        return _ok({"error": str(exc)}, status=500)
    return _ok({"status": True, "removed": removed})


@router.post("/api/characters/import")
async def import_characters(payload: CharacterImport):
    """Replace the roster from JSON rows and/or CSV text.

    Accepts ``{"characters": [{name_en, ...}]}`` (what the editor sends after
    parsing a file locally) and/or ``{"csv": "name_en,...\\n..."}``. Fresh
    ``meta_id`` values are generated for every row; the CSV carries none.
    """
    try:
        rows: list[dict] = []
        data = payload.model_dump()
        if isinstance(data.get("characters"), (list, tuple)):
            rows.extend(item for item in data["characters"] if isinstance(item, dict))
        csv_text = data.get("csv")
        if isinstance(csv_text, (bytes, bytearray)):
            try:
                csv_text = csv_text.decode("utf-8-sig")
            except Exception:
                csv_text = None
        if isinstance(csv_text, str) and csv_text.strip():
            rows.extend(parse_csv(csv_text))
        created = store.replace_all(rows)
    except ValueError as exc:
        return _ok({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.warning("Could not import characters: %s", exc)
        return _ok({"error": str(exc)}, status=500)
    return _ok({"characters": created, "added": len(created)})


@router.get("/api/characters/export")
async def export_characters():
    """The roster as CSV (``name_en,name_ja,gender,alias_en,alias_ja``)."""
    try:
        return PlainTextResponse(to_csv(store.list()), media_type="text/csv")
    except Exception as exc:
        logger.warning("Could not export characters: %s", exc)
        return PlainTextResponse("", media_type="text/csv")
