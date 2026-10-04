"""Progress reporting for long, synchronous jobs.

Typesetting a page can mean a 40 s segmentation pass followed by a second of
reconstruction per text cluster, all of it CPU work running in a worker thread.
The browser needs to say what is happening while that goes on, and it cannot be
told over the request that is doing the work -- the response is a single image.

So the worker writes here and the UI reads here, over a tiny polling endpoint.
Every method is safe to call from any thread, and reading is a dict copy under a
lock, so a poll can never be what blocks the health check.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

#: How long a finished job stays readable. Long enough that a poll in flight
#: still sees the final message, short enough to be self-cleaning.
_KEEP_DONE = 30.0


class TaskProgress:
    """The state of one named job, plus a log of what it has said."""

    #: Messages kept per job. A page with 200 text clusters would otherwise grow
    #: an unbounded list for a UI that only shows the last few lines.
    MAX_LINES = 60

    def __init__(self, name: str) -> None:
        self.name = name
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.finished: float | None = None
        self.message = ""
        self.lines: list[str] = []
        self.done = 0
        self.total = 0
        self.error: str | None = None
        self.result: dict[str, Any] = {}

    # -- writing (worker thread) --------------------------------------------

    def say(self, message: str, *, step: bool = False) -> None:
        """Record a line of progress. ``step`` also advances the counter."""
        text = str(message).strip()
        with self.lock:
            if step:
                self.done += 1
            if not text:
                return
            self.message = text
            self.lines.append(text)
            if len(self.lines) > self.MAX_LINES:
                del self.lines[:-self.MAX_LINES]

    def plan(self, total: int) -> None:
        with self.lock:
            self.total = max(0, int(total))

    def advance(self, done: int | None = None, total: int | None = None) -> None:
        with self.lock:
            if total is not None:
                self.total = max(0, int(total))
            self.done = self.done + 1 if done is None else max(0, int(done))

    def fail(self, error: str) -> None:
        with self.lock:
            self.error = str(error)
            self.finished = time.monotonic()

    def finish(self, message: str = "", **result: Any) -> None:
        with self.lock:
            if message:
                self.message = message
                self.lines.append(message)
            self.result.update(result)
            self.finished = time.monotonic()

    # -- reading (event loop) ----------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "name": self.name,
                "running": self.finished is None,
                "message": self.message,
                "lines": list(self.lines),
                "done": self.done,
                "total": self.total,
                "elapsed": round((self.finished or time.monotonic())
                                 - self.started, 2),
                "error": self.error,
                **self.result,
            }


class ProgressRegistry:
    """The jobs the UI may ask about, keyed by name (``"preview"``, ...)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskProgress] = {}

    def start(self, name: str) -> TaskProgress:
        """Begin (or restart) a job, dropping whatever the last one left."""
        task = TaskProgress(name)
        with self._lock:
            self._sweep_locked()
            self._tasks[name] = task
        return task

    def get(self, name: str) -> TaskProgress | None:
        with self._lock:
            return self._tasks.get(name)

    def snapshot(self, name: str) -> dict[str, Any]:
        task = self.get(name)
        if task is None:
            return {"name": name, "running": False, "message": "",
                    "lines": [], "done": 0, "total": 0, "idle": True}
        return task.snapshot()

    def _sweep_locked(self) -> None:
        now = time.monotonic()
        for key, task in list(self._tasks.items()):
            if task.finished is not None and now - task.finished > _KEEP_DONE:
                del self._tasks[key]
