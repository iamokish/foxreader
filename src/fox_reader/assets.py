"""The frontend, served from inside the binary.

A compiled build ships no ``frontend/`` folder. The build generates a module of
compressed byte constants -- one per template, static file and bundled font --
which Nuitka compiles into the executable alongside the rest of the code; this
module is the only thing that reads it. In a checkout the module does not exist
and everything falls through to ``frontend/`` on disk, so ``uv run`` behaves
exactly as it always has and edits to a template still show up on reload.

The generated module's contract, which ``packaging/embed_assets.py`` writes and
:func:`_bundle` checks::

    FORMAT = 1
    TEMPLATES: dict[str, Entry]
    STATIC:    dict[str, Entry]
    FONTS:     dict[str, Entry]

    Entry = (chunks: tuple[bytes, ...], raw_size: int, encoding: str, content_type: str)

``chunks`` is the payload split into pieces small enough that no single C string
literal can run into a compiler limit, whatever Nuitka decides to do with it;
join them to get the payload. ``encoding`` is ``"br"`` or ``"raw"`` -- brotli is
kept only where it actually wins, which excludes the PNGs and the woff2, and a
``br`` payload is handed to the browser still compressed when it says it can
take one.

Content types are recorded at build time rather than guessed with
:mod:`mimetypes`, which on Windows answers from the registry and has been known
to call ``.js`` ``text/plain``.
"""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

#: Written by the build, importable only in a compiled bundle. Top-level rather
#: than inside ``fox_reader`` so a stale copy can never shadow a real source
#: file, and so the generator never has to write into ``src/``.
BUNDLE_MODULE = "fox_reader_assets"

#: The only format this code knows how to read. A bundle announcing anything
#: else is ignored rather than mis-parsed.
BUNDLE_FORMAT = 1

#: Set to ignore an embedded bundle and read ``frontend/`` from disk instead --
#: the escape hatch for debugging a compiled build without rebuilding it.
DISK_ASSETS_ENV = "FOX_READER_DISK_ASSETS"

#: The two faces embedded as the typesetter's last resort.
BUNDLED_FONTS = ("ComicMono.ttf", "ComicMono-Bold.ttf")

#: Inpainting/typesetting-only symbol fallback. Embedded alongside the faces
#: above so a compiled build still has it when ``fonts/`` ships empty, but
#: never offered to the frontend (see :func:`bundled_font_names` and
#: :mod:`fox_reader.fonts`). Must agree with
#: ``packaging/embed_assets.py::SYMBOL_FONT``; ``tests/test_assets.py`` checks.
SYMBOL_FONT = "NotoSansSymbols2-Regular.otf"

_lock = threading.Lock()
_decoded: dict[tuple[str, str], bytes] = {}


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _bundle() -> Any | None:
    """The embedded asset module, or None when there is not a usable one."""
    if _truthy(os.environ.get(DISK_ASSETS_ENV, "")):
        logger.info("%s set: reading frontend assets from disk.", DISK_ASSETS_ENV)
        return None

    try:
        module = __import__(BUNDLE_MODULE)
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 - a broken bundle must not be fatal
        logger.warning("Embedded asset bundle failed to import: %s", exc)
        return None

    fmt = getattr(module, "FORMAT", None)
    if fmt != BUNDLE_FORMAT:
        logger.warning(
            "Embedded asset bundle has format %r, expected %r; ignoring it.",
            fmt,
            BUNDLE_FORMAT,
        )
        return None

    if not all(isinstance(getattr(module, name, None), dict) for name in ("TEMPLATES", "STATIC", "FONTS")):
        logger.warning("Embedded asset bundle is missing a section; ignoring it.")
        return None

    return module


def has_bundle() -> bool:
    return _bundle() is not None


# ── Payload decoding ──────────────────────────────────────────────────────────

def _decode(entry: tuple, what: str) -> bytes:
    chunks, raw_size, encoding, _content_type = entry
    data = chunks[0] if len(chunks) == 1 else b"".join(chunks)

    if encoding == "br":
        import brotli

        data = brotli.decompress(data)
    elif encoding != "raw":
        raise ValueError(f"{what}: unknown encoding {encoding!r}")

    if len(data) != raw_size:
        raise ValueError(f"{what}: expected {raw_size} bytes, got {len(data)}")

    return data


def _cached(section: str, name: str, entry: tuple) -> bytes:
    key = (section, name)
    cached = _decoded.get(key)
    if cached is not None:
        return cached

    data = _decode(entry, f"{section}/{name}")
    # The whole bundle is well under a megabyte, so this never needs evicting.
    with _lock:
        _decoded[key] = data
    return data


def _entry(section: str, name: str) -> tuple | None:
    bundle = _bundle()
    if bundle is None:
        return None
    return getattr(bundle, section).get(name)


# ── Templates ─────────────────────────────────────────────────────────────────

def template_names() -> list[str]:
    bundle = _bundle()
    if bundle is None:
        from fox_reader.utils import TEMPLATES_DIR

        if not TEMPLATES_DIR.is_dir():
            return []
        return sorted(p.name for p in TEMPLATES_DIR.iterdir() if p.is_file())
    return sorted(bundle.TEMPLATES)


def template_source(name: str) -> str | None:
    entry = _entry("TEMPLATES", name)
    if entry is None:
        return None
    return _cached("TEMPLATES", name, entry).decode("utf-8")


@lru_cache(maxsize=1)
def templates():
    """The shared :class:`Jinja2Templates`, embedded or on disk.

    One instance for every router: Jinja caches compiled templates per
    environment, and four environments would each compile ``index.html``
    separately for no gain.
    """
    import jinja2
    from fastapi.templating import Jinja2Templates

    if _bundle() is None:
        from fox_reader.utils import TEMPLATES_DIR

        return Jinja2Templates(directory=str(TEMPLATES_DIR))

    class _EmbeddedLoader(jinja2.BaseLoader):
        def get_source(self, environment, template):  # noqa: ANN001, ANN202
            source = template_source(template)
            if source is None:
                raise jinja2.TemplateNotFound(template)
            # No filename, and never stale: the bundle cannot change under a
            # running process.
            return source, None, lambda: True

        def list_templates(self):  # noqa: ANN202
            return template_names()

    # Same environment settings Jinja2Templates builds for a directory, so
    # switching between the two changes nothing about how a template renders.
    # `Jinja2Templates(env=...)` still installs `url_for`, which every template
    # uses for its icon links.
    env = jinja2.Environment(loader=_EmbeddedLoader(), autoescape=jinja2.select_autoescape())
    return Jinja2Templates(env=env)


# ── Static files ──────────────────────────────────────────────────────────────

def static_names() -> list[str]:
    bundle = _bundle()
    if bundle is None:
        from fox_reader.utils import STATIC_DIR

        if not STATIC_DIR.is_dir():
            return []
        return sorted(p.name for p in STATIC_DIR.iterdir() if p.is_file())
    return sorted(bundle.STATIC)


def static_bytes(name: str) -> bytes | None:
    """A static file's real contents, decompressed, or None if there is no such file."""
    entry = _entry("STATIC", name)
    if entry is not None:
        return _cached("STATIC", name, entry)

    if _bundle() is not None:
        return None

    from fox_reader.utils import STATIC_DIR

    path = STATIC_DIR / name
    try:
        # `resolve` before the containment check, so a `..` in the name cannot
        # walk out of the directory even though this only ever sees names the
        # frontend asks for.
        resolved = path.resolve()
        if not resolved.is_relative_to(STATIC_DIR.resolve()):
            return None
        return resolved.read_bytes()
    except OSError:
        return None


def _sub_path(scope) -> str:  # noqa: ANN001
    """The requested name, relative to the mount.

    Starlette's ``Mount`` does not rewrite ``scope["path"]``: it leaves the full
    request path in place and records the matched prefix in ``root_path``, so the
    prefix comes off here. Getting this wrong does not fail loudly -- every
    lookup simply misses and the whole frontend 404s.
    """
    path = scope.get("path", "")
    root = scope.get("root_path", "")
    if root and path.startswith(root):
        path = path[len(root) :]
    return path.lstrip("/")


def static_app():
    """An ASGI app serving the embedded static files, for ``app.mount``.

    Mounted under the name ``static`` it keeps ``url_for('static', path=...)``
    resolving exactly as :class:`StaticFiles` did, which is what every template
    builds its icon and manifest links with.
    """
    from starlette.responses import Response

    from fox_reader.utils import STATIC_DIR

    if _bundle() is None:
        from starlette.staticfiles import StaticFiles

        return StaticFiles(directory=str(STATIC_DIR), check_dir=False)

    async def app(scope, receive, send):  # noqa: ANN001, ANN202
        if scope["type"] != "http":
            # Mount forwards websockets too; nothing here speaks one.
            await send({"type": "websocket.close", "code": 1000})
            return

        method = scope.get("method", "GET").upper()
        if method not in ("GET", "HEAD"):
            await Response(status_code=405, headers={"Allow": "GET, HEAD"})(scope, receive, send)
            return

        name = _sub_path(scope)
        entry = _entry("STATIC", name)
        if entry is None:
            await Response(status_code=404)(scope, receive, send)
            return

        _chunks, _raw_size, encoding, content_type = entry
        headers = {}
        if encoding == "br" and _accepts_brotli(scope):
            # Handed over still compressed. Worth doing rather than skipping:
            # the app's own middleware marks /static no-store, so the browser
            # refetches app.js and style.css on every load -- 45 KB instead of
            # 184 KB.
            body = b"".join(entry[0])
            headers["Content-Encoding"] = "br"
        else:
            body = _cached("STATIC", name, entry)

        if encoding == "br":
            # The body depends on the request headers, so say so -- otherwise a
            # proxy may hand a brotli body to a client that cannot read one.
            headers["Vary"] = "Accept-Encoding"

        response = Response(
            content=b"" if method == "HEAD" else body,
            media_type=content_type or "application/octet-stream",
            headers=headers,
        )
        if method == "HEAD":
            # Response computed Content-Length from the empty body it was given.
            response.headers["Content-Length"] = str(len(body))
        await response(scope, receive, send)

    return app


def _accepts_brotli(scope) -> bool:  # noqa: ANN001
    for key, value in scope.get("headers", ()):
        if key == b"accept-encoding":
            return b"br" in value.lower()
    return False


def mount_static(app) -> None:  # noqa: ANN001
    app.mount("/static", static_app(), name="static")


def static_response(name: str):
    """A ready response for one static file, embedded or on disk.

    Used by the routes that serve a static file outside the ``/static`` mount --
    ``/favicon.ico``, which browsers request from the root whatever the HTML says.
    """
    from starlette.responses import Response

    data = static_bytes(name)
    if data is None:
        return Response(status_code=404)

    entry = _entry("STATIC", name)
    content_type = entry[3] if entry else _guess_type(name)
    return Response(content=data, media_type=content_type)


def _guess_type(name: str) -> str:
    suffix = name.rpartition(".")[2].lower()
    return {
        "css": "text/css; charset=utf-8",
        "html": "text/html; charset=utf-8",
        "ico": "image/vnd.microsoft.icon",
        "js": "application/javascript",
        "json": "application/json",
        "png": "image/png",
        "svg": "image/svg+xml",
        "webmanifest": "application/manifest+json",
        "woff2": "font/woff2",
        "ttf": "font/ttf",
        "otf": "font/otf",
    }.get(suffix, "application/octet-stream")


# ── Bundled fonts ─────────────────────────────────────────────────────────────
# ComicMono is the typesetter's last resort. It is embedded so a build with an
# empty fonts/ folder still renders text with a real face instead of Pillow's
# 11 px bitmap default, and it is *only* consulted when the folder turns up
# nothing usable -- a user who drops their own fonts in never sees it.
#
# NotoSansSymbols2 (SYMBOL_FONT) is the typesetter's per-glyph fallback for
# characters the selected face does not cover. It lives in the same embedded
# FONTS section so a compiled build carries it, but bundled_font_names() hides
# it and fox_reader.fonts filters it from the frontend list -- it is only ever
# reached through symbol_font_bytes() from fox_reader.typeset.

def _is_symbol_name(name: str) -> bool:
    return (name or "").strip().lower() == SYMBOL_FONT.lower()


def bundled_font_names() -> list[str]:
    bundle = _bundle()
    if bundle is None:
        from fox_reader.utils import FONT_DIR

        # In a checkout the repository's own fonts/ is the bundle's source, so
        # the fallback reads from there. If it is empty there is nothing to fall
        # back to -- but then the folder scan found nothing either, and the
        # embedded copies only exist because the build put them there.
        return [name for name in BUNDLED_FONTS if (FONT_DIR / name).is_file()]
    return sorted(name for name in bundle.FONTS if not _is_symbol_name(name))


def bundled_font_bytes(name: str) -> bytes | None:
    entry = _entry("FONTS", name)
    if entry is not None:
        return _cached("FONTS", name, entry)

    if _bundle() is not None:
        # Exact key missed in a bundle: retry case-insensitively before giving
        # up, so a checkout asking for "comicmono.ttf" still resolves. Symbol
        # faces stay reachable here too when addressed by name.
        try:
            bundle = _bundle()
            wanted = (name or "").strip().lower()
            for key in getattr(bundle, "FONTS", {}):
                if isinstance(key, str) and key.lower() == wanted:
                    alt = _entry("FONTS", key)
                    if alt is not None:
                        return _cached("FONTS", key, alt)
        except Exception:  # noqa: BLE001 - lookup must never be fatal
            pass
        return None

    from fox_reader.utils import FONT_DIR

    path = FONT_DIR / name
    try:
        if path.is_file():
            return path.read_bytes()
    except OSError:
        pass
    return None


def _symbol_entry_key() -> str | None:
    """The bundle key holding the symbol fallback, exact or case-insensitive."""
    bundle = _bundle()
    if bundle is None:
        return None
    try:
        section = getattr(bundle, "FONTS", {})
        if SYMBOL_FONT in section:
            return SYMBOL_FONT
        wanted = SYMBOL_FONT.lower()
        for key in section:
            if isinstance(key, str) and key.lower() == wanted:
                return key
    except Exception:  # noqa: BLE001 - lookup must never be fatal
        return None
    return None


def symbol_font_bytes(name: str | None = None) -> bytes | None:
    """Raw bytes of the inpainting-only symbol fallback, or None.

    ``name`` defaults to :data:`SYMBOL_FONT` and anything else is refused, so
    this cannot be repurposed into a generic font loader that would expose the
    face to the frontend. Reads from the embedded bundle when present (a
    compiled build, where ``fonts/`` ships empty) and from ``fonts/`` on disk
    otherwise -- mirroring :func:`bundled_font_bytes`. Never raises: a missing
    or unreadable face disables the fallback and the typesetter keeps using the
    primary face alone.
    """
    target = (name or SYMBOL_FONT).strip() or SYMBOL_FONT
    if target.lower() != SYMBOL_FONT.lower():
        return None

    key = _symbol_entry_key()
    if key is not None:
        entry = _entry("FONTS", key)
        if entry is not None:
            try:
                return _cached("FONTS", key, entry)
            except Exception:  # noqa: BLE001 - a bad entry disables fallback
                logger.warning("Embedded symbol font %s is unusable.", key)
                return None
        # Bundle present but the key vanished between the two lookups: fall
        # through to the explicit None below rather than to disk, so a stale
        # checkout next to a binary cannot shadow the bundle (same rule as
        # bundled_font_bytes).
        if _bundle() is not None:
            return None

    if _bundle() is not None:
        return None

    from fox_reader.utils import FONT_DIR

    try:
        exact = FONT_DIR / SYMBOL_FONT
        if exact.is_file():
            return exact.read_bytes()
        # Case-insensitive disk retry (Linux is case-sensitive; the file the
        # user dropped in may differ in case).
        try:
            for child in FONT_DIR.iterdir():
                if child.is_file() and child.name.lower() == SYMBOL_FONT.lower():
                    return child.read_bytes()
        except OSError:
            pass
    except OSError:
        pass
    except Exception:  # noqa: BLE001 - lookup must never be fatal
        pass
    return None
