"""Bounded, on-demand gallery previews; original page bytes stay untouched."""
from __future__ import annotations

import asyncio
import logging
import os
import struct
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from threading import Lock

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

logger = logging.getLogger(__name__)

MAX_DIMENSION = 1024
MAX_SOURCE_PIXELS = 16_000_000
MAX_SOURCE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _Key:
    path: str
    signature: tuple[int, int, int, int, int]
    width: int
    height: int
    fit: str
    revision: str


def _signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _png_colour_metadata(source: Image.Image) -> PngInfo:
    metadata = PngInfo()
    # PNGs can describe their colour space without an ICC profile. Dropping
    # these chunks can change brightness even when the resized pixels are right.
    if not source.info.get("icc_profile"):
        if "srgb" in source.info:
            metadata.add(b"sRGB", bytes([source.info["srgb"]]))
        if "gamma" in source.info:
            metadata.add(b"gAMA", struct.pack(">I", round(source.info["gamma"] * 100_000)))
        if "chromaticity" in source.info:
            metadata.add(b"cHRM", struct.pack(">8I", *(round(x * 100_000) for x in source.info["chromaticity"])))
    return metadata


class ThumbnailService:
    """One resize worker, a small admission queue, and an encoded-byte LRU.

    Distinct requests beyond the admission limit fall back to the original.
    Requests for the same preview share a future, including across event loops.
    Cancelling one HTTP request cannot cancel another request's preview.
    """

    def __init__(self, *, max_bytes: int = 32 * 1024 * 1024,
                 max_entries: int = 256, max_pending: int = 8):
        self._max_bytes = max(0, max_bytes)
        self._max_entries = max(0, max_entries)
        self._max_pending = max(1, max_pending)
        self._cache: OrderedDict[_Key, bytes | None] = OrderedDict()
        self._cache_bytes = 0
        self._pending: dict[_Key, Future[bytes | None]] = {}
        self._lock = Lock()
        # ThreadPoolExecutor creates its worker on the first submitted job.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fox-thumbnail")
        self._closed = False

    async def get(self, path: str, width: int, height: int,
                  fit: str = "contain", revision: str = "") -> bytes | None:
        if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
            return None
        if fit not in {"contain", "cover"} or len(revision) > 128:
            return None
        try:
            resolved = os.path.normcase(os.path.realpath(path))
            stat = os.stat(resolved)
            if stat.st_size > MAX_SOURCE_BYTES:
                return None
            key = _Key(resolved, _signature(stat), width, height, fit, revision)
        except (OSError, ValueError):
            return None

        created = False
        with self._lock:
            if self._closed:
                return None
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            future = self._pending.get(key)
            if future is None:
                if len(self._pending) >= self._max_pending:
                    return None
                future = self._executor.submit(self._render, key)
                self._pending[key] = future
                created = True
        # Register outside the lock: a completed future invokes this immediately.
        if created:
            future.add_done_callback(lambda done: self._complete(key, done))
        try:
            # The worker is independent of the HTTP request: a slow filesystem
            # must not leave a thumbnail request waiting indefinitely.
            return await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), 5.0)
        except TimeoutError:
            return None
        except asyncio.CancelledError:
            if future.cancelled():  # App shutdown cancelled a queued resize.
                return None
            raise

    @staticmethod
    def _render(key: _Key) -> bytes | None:
        try:
            with open(key.path, "rb") as stream, Image.open(stream) as source:
                if (source.width * source.height > MAX_SOURCE_PIXELS
                        or getattr(source, "is_animated", False)
                        or source.mode not in {"1", "P", "L", "LA", "RGB", "RGBA"}):
                    return None
                # A saved page can be replaced between the initial stat and
                # decoding. Do not publish bytes under the old page's cache key.
                # Windows Python can expose different ctime values through
                # stat(path) and fstat(handle); compare identity/size/mtime here,
                # and the complete path signature again after rendering.
                if _signature(os.fstat(stream.fileno()))[:4] != key.signature[:4]:
                    return None
                orientation = source.getexif().get(274, 1)
                oriented = ImageOps.exif_transpose(source) if orientation in range(2, 9) else source
                try:
                    if key.fit == "contain":
                        scale = min(key.width / oriented.width, key.height / oriented.height)
                    else:
                        scale = max(key.width / oriented.width, key.height / oriented.height)
                    if scale >= 1:
                        return None  # Already small enough; let the browser use the original.
                    # Pillow otherwise resamples palette and 1-bit images with
                    # nearest-neighbour even when LANCZOS was requested.
                    if oriented.mode == "P":
                        working = oriented.convert("RGBA" if oriented.has_transparency_data else "RGB")
                    elif oriented.mode == "RGB" and oriented.has_transparency_data:
                        working = oriented.convert("RGBA")
                    elif oriented.mode in {"1", "L"} and oriented.has_transparency_data:
                        working = oriented.convert("LA")
                    elif oriented.mode == "1":
                        working = oriented.convert("L")
                    else:
                        working = oriented
                    try:
                        if key.fit == "cover":
                            preview = ImageOps.fit(working, (key.width, key.height), Image.Resampling.LANCZOS)
                        else:
                            size = (max(1, round(working.width * scale)), max(1, round(working.height * scale)))
                            preview = working.resize(size, Image.Resampling.LANCZOS)
                        with preview, BytesIO() as output:
                            # Carry the profile matching the retained pixel mode;
                            # EXIF orientation has already been applied to pixels.
                            preview.save(output, format="PNG", icc_profile=source.info.get("icc_profile"),
                                         pnginfo=_png_colour_metadata(source), compress_level=3)
                            result = output.getvalue()
                    finally:
                        if working is not oriented:
                            working.close()
                finally:
                    if oriented is not source:
                        oriented.close()
            if _signature(os.stat(key.path)) != key.signature:
                return None
            return result
        except Exception:  # A bad/unsupported page must still have its original-source fallback.
            logger.debug("Thumbnail generation failed; serving the original", exc_info=True)
            return None

    def _complete(self, key: _Key, future: Future[bytes | None]) -> None:
        try:
            result = None if future.cancelled() else future.result()
        except Exception:
            logger.debug("Thumbnail worker failed; serving the original", exc_info=True)
            result = None
        with self._lock:
            self._pending.pop(key, None)
            if self._closed or not self._max_entries:
                return
            size = len(result) if result is not None else 0
            if size > self._max_bytes:
                return
            self._cache[key] = result
            self._cache_bytes += size
            while len(self._cache) > self._max_entries or self._cache_bytes > self._max_bytes:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= len(old) if old is not None else 0

    def close(self) -> None:
        """Discard previews and queued work; let the one active resize finish."""
        with self._lock:
            self._closed = True
            self._cache.clear()
            self._cache_bytes = 0
        self._executor.shutdown(wait=False, cancel_futures=True)
