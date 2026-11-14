"""Character roster CRUD with destination autosave and optional CSV transfer."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from fox_reader.services.session_service import complete_workspace_operation
from fox_reader.translate.characters import CharacterStore, parse_csv, store, to_csv

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


@asynccontextmanager
async def _roster(request: Request, *, write: bool = False) -> AsyncIterator[CharacterStore]:
    session = getattr(request.app.state, "session", None)
    if session is None:
        # Standalone API users retain the original unbound, in-memory store.
        yield store
    else:
        async with session.workspace_lock:
            expected = request.headers.get("X-Fox-Character-Workspace")
            if write and expected is not None and expected != session.character_workspace:
                raise HTTPException(status_code=409, detail="The folder changed. Reload it before editing characters.")
            yield session.characters


@router.get("/api/characters")
async def list_characters(request: Request):
    try:
        async with _roster(request) as roster:
            session = getattr(request.app.state, "session", None)
            data: dict[str, Any] = {"characters": roster.list()}
            if session is not None:
                data["character_workspace"] = session.character_workspace
            return _ok(data)
    except Exception as exc:
        logger.warning("Could not list characters: %s", exc)
        return _ok({"characters": [], "error": str(exc)})


@router.post("/api/characters")
async def add_character(request: Request, payload: CharacterUpsert):
    try:
        # Only keys the caller sent: absent fields stay null rather than
        # clearing anything (matters for update; harmless for add).
        async with _roster(request, write=True) as roster:
            created = await complete_workspace_operation(
                asyncio.to_thread(roster.add, payload.model_dump(exclude_unset=True)))
    except HTTPException as exc:
        return _ok({"error": exc.detail}, status=exc.status_code)
    except ValueError as exc:
        return _ok({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.warning("Could not add character: %s", exc)
        return _ok({"error": str(exc)}, status=500)
    return _ok({"character": created}, status=201)


@router.put("/api/characters/{meta_id}")
async def update_character(request: Request, meta_id: str, payload: CharacterUpsert):
    try:
        async with _roster(request, write=True) as roster:
            updated = await complete_workspace_operation(
                asyncio.to_thread(roster.update, meta_id, payload.model_dump(exclude_unset=True)))
    except HTTPException as exc:
        return _ok({"error": exc.detail}, status=exc.status_code)
    except KeyError:
        return _ok({"error": "unknown character"}, status=404)
    except ValueError as exc:
        return _ok({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.warning("Could not update character %r: %s", meta_id, exc)
        return _ok({"error": str(exc)}, status=500)
    return _ok({"character": updated})


@router.delete("/api/characters/{meta_id}")
async def delete_character(request: Request, meta_id: str):
    try:
        async with _roster(request, write=True) as roster:
            removed = await complete_workspace_operation(asyncio.to_thread(roster.delete, meta_id))
    except HTTPException as exc:
        return _ok({"error": exc.detail}, status=exc.status_code)
    except Exception as exc:
        logger.warning("Could not delete character %r: %s", meta_id, exc)
        return _ok({"error": str(exc)}, status=500)
    if not removed:
        return _ok({"error": "unknown character"}, status=404)
    return _ok({"status": True})


@router.delete("/api/characters")
async def clear_characters(request: Request):
    """Explicitly clear the current roster and save that empty list."""
    try:
        async with _roster(request, write=True) as roster:
            removed = await complete_workspace_operation(asyncio.to_thread(roster.clear))
    except HTTPException as exc:
        return _ok({"error": exc.detail}, status=exc.status_code)
    except Exception as exc:
        logger.warning("Could not clear characters: %s", exc)
        return _ok({"error": str(exc)}, status=500)
    return _ok({"status": True, "removed": removed})


@router.post("/api/characters/import")
async def import_characters(request: Request, payload: CharacterImport):
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
        async with _roster(request, write=True) as roster:
            created = await complete_workspace_operation(asyncio.to_thread(roster.replace_all, rows))
    except HTTPException as exc:
        return _ok({"error": exc.detail}, status=exc.status_code)
    except ValueError as exc:
        return _ok({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.warning("Could not import characters: %s", exc)
        return _ok({"error": str(exc)}, status=500)
    return _ok({"characters": created, "added": len(created)})


@router.get("/api/characters/export")
async def export_characters(request: Request):
    """The roster as CSV (``name_en,name_ja,gender,alias_en,alias_ja``)."""
    try:
        async with _roster(request) as roster:
            return PlainTextResponse(to_csv(roster.list()), media_type="text/csv")
    except Exception as exc:
        logger.warning("Could not export characters: %s", exc)
        return PlainTextResponse("", media_type="text/csv")
