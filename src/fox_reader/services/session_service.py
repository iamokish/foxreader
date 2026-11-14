from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar
from uuid import uuid4

from fastapi import WebSocket

from fox_reader.fsaccess import normalize_dir
from fox_reader.translate.characters import CharacterStore

_Result = TypeVar("_Result")


async def complete_workspace_operation(operation: Coroutine[Any, Any, _Result]) -> _Result:
    """Finish an in-flight workspace write before releasing its lock on cancel.

    Cancelling an await does not stop filesystem work in a worker thread.
    Shield the complete operation so paths and roster state stay in agreement.
    """
    task = asyncio.create_task(operation)
    cancelled = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancelled = exc
    if cancelled is not None:
        raise cancelled
    return result


class SessionService:
    """Manages runtime session state: active tabs, current directory, caches."""

    def __init__(self) -> None:
        self._active_tabs: list[WebSocket] = []
        self._current_dir: str | None = None
        self._dest_dir: str | None = None
        self._bubble_cache: dict[str, dict] = {}
        self.characters = CharacterStore()
        self.character_workspace = uuid4().hex
        # Folder activation and roster writes share a destination. Serialize
        # them while disk I/O runs off the event loop.
        self.workspace_lock = asyncio.Lock()

    @property
    def current_dir(self) -> str | None:
        return self._current_dir

    @current_dir.setter
    def current_dir(self, value: str | None) -> None:
        if value is None:
            self._current_dir = None
            return
        self._current_dir = normalize_dir(value)

    @property
    def dest_dir(self) -> str | None:
        """Where typeset pages are written. ``None`` means "the source folder".

        Kept separate from `current_dir` rather than defaulting to it in the
        setter: `save_dir` is what callers should ask for, and a `None` here is
        how "no folder has been loaded yet" stays distinguishable.
        """
        return self._dest_dir

    @dest_dir.setter
    def dest_dir(self, value: str | None) -> None:
        if value is None:
            self._dest_dir = None
            return
        self._dest_dir = normalize_dir(value)

    @property
    def save_dir(self) -> str | None:
        """The directory a save lands in: the destination, or the source."""
        return self._dest_dir or self._current_dir

    @property
    def writes_in_place(self) -> bool:
        """Whether saving overwrites the originals, i.e. dest *is* source.

        Compared as plain strings because both went through `normalize_dir` on
        the way in; `fsaccess.same_dir` is the answer for two paths that came
        from different places.
        """
        return bool(self._current_dir) and (
            not self._dest_dir or self._dest_dir == self._current_dir
        )

    def set_workspace(self, source: str | None, dest: str | None = None) -> None:
        """Point the session at a source folder and its destination."""
        self.current_dir = source
        self.dest_dir = dest
        self.character_workspace = uuid4().hex

    # Bubble cache ------------------------------------------------------------------
    #
    # Deliberately not keyed by view variant: OCR and bubble detection always read
    # the original page (typesetting does too), so there is only ever one set of
    # regions per page and no variant for them to collide on.

    def get_cached_bubble(self, filename: str, cache_key: str) -> list | None:
        entry = self._bubble_cache.get(filename)
        if entry and cache_key in entry:
            return entry[cache_key]
        return None

    def set_cached_bubble(self, filename: str, cache_key: str, data: list) -> None:
        self._bubble_cache.setdefault(filename, {})[cache_key] = data

    def clear_bubble_cache(self) -> None:
        self._bubble_cache = {}

    # Active tabs -------------------------------------------------------------------

    def add_tab(self, ws: WebSocket) -> None:
        self._active_tabs.append(ws)

    def remove_tab(self, ws: WebSocket) -> None:
        if ws in self._active_tabs:
            self._active_tabs.remove(ws)

    def has_active_tabs(self) -> bool:
        return len(self._active_tabs) > 0

    def get_active_tabs(self) -> list[WebSocket]:
        return self._active_tabs
