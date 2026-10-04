"""Where each source folder's typeset pages go, remembered across restarts.

A user works through one scan over several sittings, and re-typing the
destination every time is both tedious and a chance to typo a path that then
quietly becomes a second output folder. So the pairing is stored: load the same
source again and the destination it had last time is offered back.

Modelled on :class:`fox_reader.user_endpoints.UserEndpointStore` -- same yaml
file under ``config/``, same lock, and the same rule that one unreadable record
is skipped rather than allowed to take the file down with it. This is a
convenience cache; nothing in here is worth failing a startup over.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml

from fox_reader.fsaccess import resolve_dir, same_dir

logger = logging.getLogger(__name__)

#: How many pairings to keep. Well past what anyone will revisit, and small
#: enough that the file stays a few kilobytes.
MAX_ENTRIES = 200


class WorkspaceStore:
    """Source folder -> destination folder, most recently used first."""

    def __init__(self, config_root: Path):
        self.config_root = Path(config_root)
        self.path = self.config_root / "workspaces.yaml"

        self._lock = threading.RLock()
        #: [{"source": ..., "dest": ...}], most recent first.
        self._entries: list[dict[str, str]] = []

        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(record: Any) -> dict[str, str] | None:
        """One stored record, or None if it is not usable."""
        if not isinstance(record, dict):
            return None
        source = resolve_dir(str(record.get("source") or ""))
        dest = resolve_dir(str(record.get("dest") or ""))
        if not source or not dest:
            return None
        return {"source": source, "dest": dest}

    def load(self) -> None:
        with self._lock:
            self.config_root.mkdir(parents=True, exist_ok=True)

            if not self.path.exists():
                self._entries = []
                self.save()
                return

            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                raw = data.get("workspaces", [])
                if not isinstance(raw, list):
                    raise ValueError("workspaces must be a list")

                # One at a time, like the endpoint store: a record written by an
                # older build must not cost the user every other pairing.
                entries: list[dict[str, str]] = []
                for record in raw:
                    cleaned = self._clean(record)
                    if cleaned is None:
                        logger.warning("Skipping unusable workspace record: %r", record)
                        continue
                    if any(same_dir(cleaned["source"], kept["source"]) for kept in entries):
                        continue
                    entries.append(cleaned)

                self._entries = entries[:MAX_ENTRIES]

            except (yaml.YAMLError, OSError, ValueError) as exc:
                logger.warning("Failed to load remembered workspaces: %s", exc)
                self._entries = []

            self.save()

    def save(self) -> None:
        with self._lock:
            payload = {"workspaces": [dict(entry) for entry in self._entries]}
            try:
                with self.path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
            except OSError as exc:
                # A read-only config directory must not stop a folder from
                # loading; the pairing is simply not remembered.
                logger.warning("Could not write %s: %s", self.path, exc)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def dest_for(self, source: str) -> str | None:
        """The destination this source folder was last loaded with."""
        source = resolve_dir(source)
        if not source:
            return None
        with self._lock:
            for entry in self._entries:
                if same_dir(entry["source"], source):
                    return entry["dest"]
        return None

    def last_source(self) -> str | None:
        """The most recently loaded source folder, for prefilling the picker."""
        with self._lock:
            return self._entries[0]["source"] if self._entries else None

    def all(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(entry) for entry in self._entries]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def remember(self, source: str, dest: str) -> None:
        """Record a pairing, moving it to the front of the list."""
        source = resolve_dir(source)
        dest = resolve_dir(dest)
        if not source or not dest:
            return

        with self._lock:
            self._entries = [
                entry for entry in self._entries
                if not same_dir(entry["source"], source)
            ]
            self._entries.insert(0, {"source": source, "dest": dest})
            del self._entries[MAX_ENTRIES:]
            self.save()

    def forget(self, source: str) -> None:
        source = resolve_dir(source)
        if not source:
            return
        with self._lock:
            before = len(self._entries)
            self._entries = [
                entry for entry in self._entries
                if not same_dir(entry["source"], source)
            ]
            if len(self._entries) != before:
                self.save()
