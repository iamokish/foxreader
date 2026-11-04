"""Quiet per-file progress without changing Hugging Face's transfer machinery."""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterator
from functools import wraps
from typing import Any

import httpx

download_context = threading.local()
_hook_lock = threading.Lock()
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)
_PUBLISH_INTERVAL = 0.25


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


class FileProgress:
    """One file owns its callbacks, including callbacks from Xet's native thread.

    HTTP observations replace, rather than add to, tqdm's coarser byte counts.
    Each HTTP attempt starts at its response's Range offset, so discarded bytes
    from a failed request and servers ignoring Range are handled correctly.
    """

    def __init__(self, publish: Callable[[int, int, str], None]) -> None:
        self._lock = threading.RLock()
        self._publish = publish
        self._active = True
        self._bar_ready = False
        self._source: object | None = None
        self._count = 0
        self._total = 0
        self._phase = "preparing"
        self._transfer: dict[str, int] = {}
        self._last_published = 0.0

    def transfer_snapshot(self) -> dict[str, int]:
        """Physical transfer counters, separate from assembled file bytes."""
        with self._lock:
            return dict(self._transfer)

    def _transfer_received(self) -> bool:
        total = self._transfer.get("transfer_total", 0)
        return total > 0 and self._transfer.get("transfer_bytes", 0) >= total

    def _emit(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if self._active and (force or now - self._last_published >= _PUBLISH_INTERVAL):
            # Display failure must never interrupt a model transfer.
            try:
                self._publish(self._count, self._total, self._phase)
            except Exception:
                pass
            self._last_published = now

    def bar_started(self, total: int, initial: int) -> None:
        with self._lock:
            if not self._active:
                return
            self._bar_ready = True
            self._count, self._total = initial, total
            self._phase = "downloading"
            self._emit(force=True)

    def bar_advanced(self, count: int, total: int, *, force: bool = False, finished: bool = False) -> None:
        with self._lock:
            if not self._active:
                return
            if self._source is None:
                self._count, self._total = count, total
            self._phase = "finalizing" if finished or self._transfer_received() else "downloading"
            self._emit(force=force)

    def native_transfer(self, report: Any) -> None:
        """Read Xet's wire counters without pretending they are file sizes.

        Xet may discover more transfer work while running. Its reported total
        is therefore used as-is; reaching it does not mark the file complete.
        Older native APIs without these fields retain ordinary tqdm progress.
        """
        try:
            count = int(report.total_transfer_bytes_completed)
            total = int(report.total_transfer_bytes)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return
        if count < 0 or total < 0:
            return
        with self._lock:
            if not self._active:
                return
            self._transfer = {"transfer_bytes": count, "transfer_total": total}
            self._phase = "finalizing" if self._transfer_received() else "downloading"
            self._emit()

    def use_bar(self) -> None:
        with self._lock:
            self._source = None

    def http_started(self, start: int, total: int) -> object | None:
        with self._lock:
            # Metadata/API bodies are consumed before Hub creates its file bar.
            if not self._active or not self._bar_ready:
                return None
            self._source = token = object()
            self._count = start
            self._total = total or self._total
            self._phase = "downloading"
            self._emit(force=True)
            return token

    def http_advanced(self, token: object, amount: int) -> None:
        with self._lock:
            if self._active and self._source is token:
                self._count += amount
                self._emit()

    def http_finished(self, token: object, *, exhausted: bool) -> None:
        with self._lock:
            if self._active and self._source is token:
                if exhausted:
                    self._phase = "finalizing"
                else:
                    # Hub may replace its HTTP client after a connection
                    # failure. If the retry has no stream observer, its normal
                    # tqdm updates must take ownership instead of being ignored.
                    self._source = None
                self._emit(force=True)

    def finish(self, size: int | None = None) -> None:
        with self._lock:
            if size is not None:
                self._count = self._total = size
                self._phase = "finalizing"
            self._emit(force=True)
            self._active = False
            # Old native callbacks cannot retain a completed job's state.
            self._publish = lambda *_args: None


class HubByteProgress:
    """The small tqdm interface used by Hub's HTTP and Xet download paths."""

    def __init__(self, total: Any = None, initial: Any = 0, **_kwargs: Any) -> None:
        self._total = _integer(total)
        self._count = _integer(initial)
        self._closed = False
        progress = getattr(download_context, "current", None)
        self._progress = progress if isinstance(progress, FileProgress) else None
        if self._progress is not None:
            self._progress.bar_started(self._total, self._count)

    def update(self, n: Any) -> None:
        if self._closed:
            return
        try:
            step = int(n or 0)
        except (TypeError, ValueError, OverflowError):
            return
        self._count = max(0, self._count + step)
        if self._progress is not None:
            self._progress.bar_advanced(self._count, self._total)

    def close(self, *, finished: bool = True) -> None:
        if not self._closed and self._progress is not None:
            self._progress.bar_advanced(self._count, self._total, force=True, finished=finished)
        self._closed = True

    def __enter__(self) -> HubByteProgress:
        return self

    def __exit__(self, exc_type: Any, *_args: Any) -> None:
        self.close(finished=exc_type is None)


class _ObservedStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, progress: FileProgress, start: int, total: int) -> None:
        self._stream, self._progress = stream, progress
        self._start, self._total = start, total

    def __iter__(self) -> Iterator[bytes]:
        token = self._progress.http_started(self._start, self._total)
        exhausted = False
        try:
            for chunk in self._stream:
                if token is not None:
                    self._progress.http_advanced(token, len(chunk))
                yield chunk
            exhausted = True
        finally:
            if token is not None:
                self._progress.http_finished(token, exhausted=exhausted)

    def close(self) -> None:
        self._stream.close()


def _observe_response(response: httpx.Response) -> None:
    progress = getattr(download_context, "current", None)
    if not isinstance(progress, FileProgress) or response.request.method != "GET":
        return
    if response.status_code not in (200, 206) or response.is_stream_consumed:
        return
    # Wire bytes and saved bytes differ for compression: keep Hub's decoded
    # tqdm counts in that case. Never guess progress from compressed lengths.
    if response.headers.get("Content-Encoding", "identity").lower().strip() not in ("", "identity"):
        progress.use_bar()
        return
    if not isinstance(response.stream, httpx.SyncByteStream):
        return
    start, total = 0, _integer(response.headers.get("Content-Length"))
    if response.status_code == 206:
        match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", "").strip())
        if match is None:
            progress.use_bar()
            return
        try:
            start, end = int(match[1]), int(match[2])
        except ValueError:
            # Python rejects absurdly long numeric headers. Instrumentation
            # must not turn such a header into an otherwise unrelated failure.
            progress.use_bar()
            return
        total = _integer(match[3])
        if end < start or (total and end >= total):
            progress.use_bar()
            return
    response.stream = _ObservedStream(response.stream, progress, start, total)


class _XetProgressSession:
    """Delegate to the native session and observe its existing group callback."""

    def __init__(self, session: Any, progress: FileProgress) -> None:
        self._session, self._progress = session, progress

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def new_file_download_group(self, *args: Any, **kwargs: Any) -> Any:
        callback = kwargs.get("progress_callback")
        if callable(callback):
            progress = self._progress

            def observe(report: Any, *callback_args: Any, **callback_kwargs: Any) -> Any:
                try:
                    return callback(report, *callback_args, **callback_kwargs)
                finally:
                    # Preserve both the original callback's result/errors and
                    # native transfer behavior if instrumentation cannot run.
                    try:
                        progress.native_transfer(report)
                    except Exception:
                        pass

            kwargs["progress_callback"] = observe
        return self._session.new_file_download_group(*args, **kwargs)


def _observe_xet_progress() -> None:
    # This private integration point belongs to the pinned Hub version. Guard
    # its presence so an API change degrades to its existing tqdm callbacks.
    try:
        from huggingface_hub.utils import _xet
    except ImportError:
        return
    with _hook_lock:
        original = getattr(_xet, "get_xet_session", None)
        if not callable(original) or getattr(original, "_fox_reader_progress_bridge", False):
            return

        @wraps(original)
        def get_session(*args: Any, **kwargs: Any) -> Any:
            session = original(*args, **kwargs)
            progress = getattr(download_context, "current", None)
            if isinstance(progress, FileProgress):
                try:
                    if callable(getattr(session, "new_file_download_group", None)):
                        return _XetProgressSession(session, progress)
                except Exception:
                    pass
            return session

        get_session._fox_reader_progress_bridge = True
        _xet.get_xet_session = get_session


def observe_hub_client() -> None:
    """Install scoped progress observers, retaining Hub's configured clients.

    No client factory, proxy, authentication, retry, or transfer-size setting is
    replaced. The thread-local scope prevents unrelated/concurrent requests
    from being attributed to the active setup file.
    """
    from huggingface_hub.utils import get_session

    try:
        client = get_session()
        with _hook_lock:
            hooks = client.event_hooks["response"]
            if _observe_response not in hooks:
                hooks.append(_observe_response)
    except Exception:
        # A custom/future client may not support mutable response hooks. Let
        # Hub use it normally, retaining its own errors and tqdm reporting.
        pass
    try:
        _observe_xet_progress()
    except Exception:
        pass
