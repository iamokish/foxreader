"""Bounded local diagnostics shared by the console and desktop entry points."""
from __future__ import annotations

import io
import logging
import os
import re
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_LEVEL_ENV = "FOX_READER_LOG_LEVEL"
_configured: Path | None = None
_lock = threading.RLock()
_SECRET = re.compile(
    r"(?i)\b(authorization|api[ _-]?key|access[ _-]?token|token|password|secret)"
    r"([\s\"']*[:=][\s\"']*)(?:bearer\s+)?[^\s,;\"'<>]+"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_URL_QUERY = re.compile(r"(https?://[^\s?#]+)[?#][^\s]*", re.I)
_URL_USERINFO = re.compile(r"(https?://)[^\s/@]+:[^\s/@]+@", re.I)


def redact(text: str) -> str:
    """Remove common credential forms and URL query strings before persistence."""
    text = _URL_QUERY.sub(r"\1?[redacted]", text)
    text = _URL_USERINFO.sub(r"\1[redacted]@", text)
    text = _SECRET.sub(r"\1\2[redacted]", text)
    text = _BEARER.sub("Bearer [redacted]", text)
    token = os.environ.get("FOX_READER_SHUTDOWN_TOKEN", "")
    if token:
        text = text.replace(token, "[redacted]")
    return text


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Format a copy so another handler never receives a mutated record.
        import copy

        return redact(super().format(copy.copy(record)))


class _SafeRotatingHandler(RotatingFileHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        # Logging must not recurse through a redirected stderr when a disk is
        # full, nor take down a healthy reader because diagnostics cannot write.
        pass


class _LogStream(io.TextIOBase):
    """Python libraries still print in a no-console build; keep those lines."""

    def __init__(self, name: str, level: int):
        self._logger = logging.getLogger(name)
        self._level = level
        self._pending = ""
        self._lock = threading.RLock()

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("write() argument must be str")
        with self._lock:
            self._pending += value
            while "\n" in self._pending or len(self._pending) >= 8192:
                end = self._pending.find("\n")
                end = min(end, 8192) if end >= 0 else 8192
                line = self._pending[:end].rstrip("\r")
                self._pending = self._pending[end + (self._pending[end:end + 1] == "\n"):]
                if line:
                    self._logger.log(self._level, line)
        return len(value)

    def flush(self) -> None:
        with self._lock:
            if self._pending:
                self._logger.log(self._level, self._pending.rstrip("\r"))
                self._pending = ""


def configure_logging(directory: Path | None = None) -> bool:
    """Install once, without changing third-party log levels or existing handlers.

    Returns false for an unwritable data directory. Console/source launches keep
    their terminal output; a frozen launch writes diagnostics only to files.
    """
    global _configured
    if directory is None:
        from fox_reader.utils import LOG_DIR

        directory = LOG_DIR
    directory = directory.resolve()
    with _lock:
        if _configured == directory:
            return True
        formatter = _RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            backend = _SafeRotatingHandler(
                directory / "backend.log", maxBytes=5 * 1024 * 1024, backupCount=3,
                encoding="utf-8", errors="replace",
            )
            try:
                frontend = _SafeRotatingHandler(
                    directory / "frontend.log", maxBytes=1024 * 1024, backupCount=2,
                    encoding="utf-8", errors="replace",
                )
            except OSError:
                backend.close()
                raise
        except OSError:
            # The reader can still start from a read-only folder. A detached
            # process needs usable streams even when its diagnostics are lost.
            for name in ("stdout", "stderr"):
                if getattr(sys, name) is None:
                    setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
            return False
        for handler in (backend, frontend):
            handler.setFormatter(formatter)
            handler._fox_reader_handler = True  # type: ignore[attr-defined]
        root = logging.getLogger()
        client = logging.getLogger("fox_reader.frontend")
        for logger in (root, client):
            for handler in logger.handlers[:]:
                if getattr(handler, "_fox_reader_handler", False):
                    logger.removeHandler(handler)
                    handler.close()
        root.addHandler(backend)
        if root.level == logging.NOTSET:
            root.setLevel(logging.WARNING)
        client.addHandler(frontend)
        client.setLevel(logging.INFO)
        client.propagate = False
        wanted = os.environ.get(LOG_LEVEL_ENV, "").strip().upper() or "INFO"
        logging.getLogger("fox_reader").setLevel(logging.getLevelNamesMapping().get(wanted, logging.INFO))
        # uvicorn's default formatter assumes stderr exists. All server logs
        # use the same file handler; access logs are disabled by the entry point.
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            server_logger = logging.getLogger(name)
            server_logger.handlers.clear()
            server_logger.propagate = True
            server_logger.setLevel(logging.INFO)
        if (not getattr(sys, "frozen", False) and sys.stderr is not None
                and not isinstance(sys.stderr, _LogStream)):
            console = logging.StreamHandler(sys.stderr)
            console.setFormatter(formatter)
            console._fox_reader_handler = True  # type: ignore[attr-defined]
            root.addHandler(console)
        for name, level in (("stdout", logging.INFO), ("stderr", logging.ERROR)):
            if getattr(sys, name) is None:
                setattr(sys, name, _LogStream(f"fox_reader.{name}", level))
        _configured = directory
        return True
