"""Turns ``frontend/`` and the fallback fonts into a Python module of byte constants.

The build runs this, Nuitka compiles the result into the executable, and
:mod:`fox_reader.assets` reads it. The point is that a shipped build has no
``frontend/`` folder for anyone to read or edit -- the templates, the CSS and the
JavaScript live inside the binary as constants, not as files.

Written module -- the format :mod:`fox_reader.assets` checks on import::

    FORMAT = 1
    TEMPLATES: dict[str, Entry]
    STATIC:    dict[str, Entry]
    FONTS:     dict[str, Entry]

    Entry = (chunks: tuple[bytes, ...], raw_size: int, encoding: str, content_type: str)

Two decisions worth stating, because neither is arbitrary:

*Brotli only where it wins.* Compression is tried on everything and kept only if
the result is actually smaller. That is not a formality -- the PNGs and the woff2
are already compressed and come out four or five bytes *larger*, so a
blanket-compress build would pay a decompression pass at every request to save
nothing. What does win wins big: ``app.js`` 137,987 -> 35,715 and ``style.css``
46,503 -> 9,409, and those two are refetched on every page load because the app
marks ``/static`` ``no-store``.

*Chunked payloads.* Each payload is split into :data:`CHUNK_SIZE` pieces. Nuitka
serialises constants into a blob rather than emitting C string literals, so this
is belt-and-braces: even if a payload did end up as a literal, 8 KiB of bytes is
at most ~32 KB of escaped source, comfortably under MSVC's 65,535-byte limit for
one string literal (C2026). The reader joins them back.

Run standalone to inspect what a build would embed::

    python packaging/embed_assets.py --out build/fox_reader_assets.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import brotli

# Same pipe problem as packaging/build.py: under CI stdout is block-buffered,
# so these totals would flush after whatever the caller prints next.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
del _stream

#: The format number written into the module. Bump only alongside
#: ``fox_reader.assets.BUNDLE_FORMAT``; a mismatch makes the reader ignore the
#: bundle and fall back to disk rather than mis-parse it.
FORMAT = 1

#: Bytes of payload per emitted constant. See the module docstring.
CHUNK_SIZE = 8192

#: Maximum compression. This runs once per build, so the time is free.
BROTLI_QUALITY = 11
BROTLI_LGWIN = 24

#: The faces embedded as the typesetter's last resort. Must agree with
#: ``fox_reader.assets.BUNDLED_FONTS``; ``tests/test_assets.py`` checks that it
#: does. Only these two are offered to the frontend: the rest of ``fonts/`` is
#: the user's, and ships empty.
BUNDLED_FONTS = ("ComicMono.ttf", "ComicMono-Bold.ttf")

#: Inpainting/typesetting-only symbol fallback, embedded with the faces above
#: but never listed as a frontend font (see ``fox_reader.assets.SYMBOL_FONT``
#: and ``fox_reader.fonts``). Used per-glyph for characters the selected face
#: does not cover, at the same size. Must agree with
#: ``fox_reader.assets.SYMBOL_FONT``.
SYMBOL_FONT = "NotoSansSymbols2-Regular.otf"

#: Recorded at build time rather than guessed at request time. :mod:`mimetypes`
#: answers from the registry on Windows, where ``.js`` has been known to come
#: back as ``text/plain`` -- which a browser will refuse to execute.
CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".gif": "image/gif",
    ".htm": "text/html; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".ico": "image/vnd.microsoft.icon",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".js": "application/javascript",
    ".json": "application/json",
    ".map": "application/json",
    ".mjs": "application/javascript",
    ".otf": "font/otf",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ttf": "font/ttf",
    ".txt": "text/plain; charset=utf-8",
    ".webmanifest": "application/manifest+json",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".xml": "application/xml",
}

#: ``frontend/static/LICENSE-material-icons`` and ``fonts/LICENSE-ComicMono``
#: have no extension and are plain text.
NO_EXTENSION_TYPE = "text/plain; charset=utf-8"

#: Never embedded: editor litter, OS litter, and the font cache -- which is
#: keyed by the mtime and size of files in the user's own fonts folder and so is
#: meaningless anywhere but the machine that wrote it.
SKIP_NAMES = {".DS_Store", "Thumbs.db", ".fonts_cache.json", "desktop.ini"}
SKIP_SUFFIXES = (".pyc", ".swp", "~", ".orig", ".rej", ".bak")


class AssetError(RuntimeError):
    """Raised when there is nothing usable to embed -- always a build bug."""


# ── Encoding ──────────────────────────────────────────────────────────────────

def content_type(name: str) -> str:
    """The type to serve ``name`` as. Never guesses silently."""
    suffix = Path(name).suffix.lower()
    if not suffix:
        return NO_EXTENSION_TYPE
    guess = CONTENT_TYPES.get(suffix)
    if guess is None:
        print(
            f"  ! {name}: no content type for {suffix!r}, "
            f"serving as application/octet-stream "
            f"(add it to CONTENT_TYPES in packaging/embed_assets.py)",
            file=sys.stderr,
        )
        return "application/octet-stream"
    return guess


def encode(data: bytes) -> tuple[tuple[bytes, ...], int, str]:
    """``(chunks, raw_size, encoding)`` for one payload.

    Compressed only when that makes it smaller, which for the already-compressed
    assets it does not.
    """
    payload = data
    encoding = "raw"

    if data:
        compressed = brotli.compress(data, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN)
        if len(compressed) < len(data):
            payload, encoding = compressed, "br"

    # `or (b"",)`: an empty file would otherwise produce no chunks at all, and
    # the reader's join would be reading past the end of a zero-length tuple.
    chunks = tuple(payload[i : i + CHUNK_SIZE] for i in range(0, len(payload), CHUNK_SIZE)) or (b"",)
    return chunks, len(data), encoding


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render_entry(name: str, data: bytes, ctype: str) -> tuple[str, int, int]:
    """Source for one dict item, plus ``(raw_size, stored_size)`` for the log."""
    chunks, raw_size, encoding = encode(data)
    stored = sum(len(c) for c in chunks)

    lines = [f"    {name!r}: ((", *(f"        {chunk!r}," for chunk in chunks)]
    lines.append(f"    ), {raw_size}, {encoding!r}, {ctype!r}),")
    return "\n".join(lines), raw_size, stored


def _render_section(title: str, items: dict[str, tuple[str, bytes]]) -> tuple[str, int, int]:
    if not items:
        return f"{title} = {{}}\n", 0, 0

    body: list[str] = [f"{title} = {{"]
    raw_total = stored_total = 0
    for name in sorted(items):
        ctype, data = items[name]
        rendered, raw, stored = _render_entry(name, data, ctype)
        body.append(rendered)
        raw_total += raw
        stored_total += stored
    body.append("}\n")
    return "\n".join(body), raw_total, stored_total


def render_module(
    templates: dict[str, bytes],
    static: dict[str, tuple[str, bytes]],
    fonts: dict[str, bytes],
) -> str:
    """The generated module's source.

    ``templates`` and ``fonts`` map name to bytes; ``static`` maps name to
    ``(content_type, bytes)``, because a static file's type is what it is served
    with and a template's never reaches the wire.
    """
    sections = [
        ("TEMPLATES", {name: (content_type(name), data) for name, data in templates.items()}),
        ("STATIC", static),
        ("FONTS", {name: (content_type(name), data) for name, data in fonts.items()}),
    ]

    parts = [
        '"""The frontend, as constants.\n\n'
        "Generated by packaging/embed_assets.py. Do not edit, and do not commit:\n"
        "the build writes this next to the Nuitka entry point and Nuitka compiles\n"
        "it in. fox_reader.assets is the only thing that reads it.\n"
        '"""\n',
        f"FORMAT = {FORMAT}\n",
    ]

    totals: list[tuple[str, int, int, int]] = []
    for title, items in sections:
        rendered, raw, stored = _render_section(title, items)
        parts.append(rendered)
        totals.append((title, len(items), raw, stored))

    source = "\n".join(parts)
    _log_totals(totals, len(source))
    return source


def _log_totals(totals: list[tuple[str, int, int, int]], source_size: int) -> None:
    for title, count, raw, stored in totals:
        if not count:
            continue
        saved = f"-{100 - stored * 100 // max(raw, 1)}%" if stored < raw else "as-is"
        print(f"  {title.lower():10} {count:3} files  {raw:>8,} -> {stored:>8,} bytes  {saved}")
    print(f"  {'module':10}     {'':3}        {'':>8}    {source_size:>8,} bytes of source")


# ── Collection ────────────────────────────────────────────────────────────────

def _skip(path: Path) -> bool:
    """Whether this asset is litter rather than something to ship.

    ``path`` is relative to the asset root it was found under. That matters for
    the directory check below: given an absolute path, a checkout living
    somewhere like ``~/.cache/src/foxreader`` would skip every asset in the
    build.

    Tool metadata directories are skipped wholesale. Vite writes
    ``frontend/static/.vite/`` (a manifest and a bundled licence file) beside
    the bundle it builds, and .gitignore already treats that as build output.
    Embedding it would serve files nothing requests and grow the binary for no
    one, so the embedder agrees with .gitignore instead of learning a content
    type per build tool.
    """
    return (
        path.name in SKIP_NAMES
        or path.name.endswith(SKIP_SUFFIXES)
        or any(part.startswith(".") for part in path.parts[:-1])
    )


def _walk(directory: Path) -> dict[str, bytes]:
    """Every file under ``directory``, keyed by its relative posix path.

    Nested paths are kept as-is rather than flattened: the reader looks names up
    as dict keys and the mount hands it the request path verbatim, so
    ``static/vendor/x.js`` would just work if the frontend ever grows one.
    """
    if not directory.is_dir():
        raise AssetError(f"{directory} is missing")

    found: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(directory)

        if _skip(relative):
            continue

        found[relative.as_posix()] = path.read_bytes()

    if not found:
        raise AssetError(f"{directory} holds no files to embed")
    return found


def collect(root: Path) -> tuple[dict[str, bytes], dict[str, tuple[str, bytes]], dict[str, bytes]]:
    """The three sections, read from a checkout."""
    templates = _walk(root / "frontend" / "templates")
    static_raw = _walk(root / "frontend" / "static")
    static = {name: (content_type(name), data) for name, data in static_raw.items()}

    font_dir = root / "fonts"
    fonts: dict[str, bytes] = {}
    for name in BUNDLED_FONTS:
        path = font_dir / name
        if not path.is_file():
            raise AssetError(f"bundled font {name} is missing from {font_dir}")
        fonts[name] = path.read_bytes()

    # Symbol fallback: same FONTS section so the compiled binary carries it,
    # but fox_reader.assets hides it from the frontend list and fox_reader.fonts
    # filters it from getFonts(). Required -- a build without it would silently
    # ship without per-glyph fallback.
    symbol_path = font_dir / SYMBOL_FONT
    if not symbol_path.is_file():
        # Case-insensitive retry before failing, so a checkout on a
        # case-sensitive filesystem with a differently-cased copy still builds.
        found: Path | None = None
        try:
            if font_dir.is_dir():
                wanted = SYMBOL_FONT.lower()
                for child in font_dir.iterdir():
                    if child.is_file() and child.name.lower() == wanted:
                        found = child
                        break
        except OSError:
            found = None
        if found is None:
            raise AssetError(f"bundled symbol font {SYMBOL_FONT} is missing from {font_dir}")
        symbol_path = found
    # Canonical key: the reader looks this name up case-insensitively anyway,
    # and a canonical key keeps the bundle stable across filesystems.
    fonts[SYMBOL_FONT] = symbol_path.read_bytes()

    return templates, static, fonts


def build(root: Path, out: Path) -> Path:
    """Generate the module for the checkout at ``root``, writing it to ``out``."""
    templates, static, fonts = collect(root)
    source = render_module(templates, static, fonts)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(source, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=repo_root, help="checkout to read (default: this one)")
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "packaging" / "build" / "fox_reader_assets.py",
        help="module to write",
    )
    args = parser.parse_args(argv)

    try:
        written = build(args.root.resolve(), args.out.resolve())
    except AssetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"  wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
