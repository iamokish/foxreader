import base64
import io
import json
import logging
from pathlib import Path

from fontTools.ttLib import TTFont

logger = logging.getLogger(__name__)

#: What counts as a font in ``fonts/``. Nothing else in that folder does --
#: not ``.fonts_cache.json``, not the ``LICENSE-ComicMono`` that ships beside the
#: fallbacks, not an editor's stray backup. This matters for more than tidiness:
#: whether the bundled fallbacks load is decided by how many fonts are in the
#: folder, so a folder holding only those files has to count as empty.
FONT_EXTENSIONS = (".ttf", ".otf")

#: Inpainting/typesetting-only symbol fallback. Lives in ``fonts/`` on disk and
#: in the embedded bundle, but is never offered to the frontend: it is only
#: ever reached per-glyph from :mod:`fox_reader.typeset`. Must agree with
#: ``fox_reader.assets.SYMBOL_FONT`` and ``packaging/embed_assets.py::SYMBOL_FONT``.
SYMBOL_FONT_FILENAME = "NotoSansSymbols2-Regular.otf"


def _is_symbol_font(filename: str) -> bool:
    """Whether ``filename`` is the hidden symbol fallback (case-insensitive)."""
    try:
        return (filename or "").strip().lower() == SYMBOL_FONT_FILENAME.lower()
    except Exception:  # noqa: BLE001 - filtering must never be fatal
        return False


def _extract_font_metadata(source, stem: str):
    """Family/style names for a face.

    ``source`` is whatever ``TTFont`` accepts -- a path for a font in the folder,
    a ``BytesIO`` for one of the embedded fallbacks -- and ``stem`` is the name to
    fall back on when the face has no usable ``name`` table.
    """
    try:
        font = TTFont(source, lazy=True)
        name_table = font["name"]

        family_name = None
        for name_id in (16, 1):
            for platform_id, enc_id in ((3, 1), (1, 0)):
                record = name_table.getName(name_id, platform_id, enc_id)
                if record:
                    family_name = record.toUnicode()
                    break
            if family_name:
                break

        subfamily = "Regular"
        for platform_id, enc_id in ((3, 1), (1, 0)):
            record = name_table.getName(2, platform_id, enc_id)
            if record:
                subfamily = record.toUnicode()
                break

        bold = False
        italic = False

        if "OS/2" in font:
            fs = font["OS/2"].fsSelection
            bold = bool(fs & 0x20)
            italic = bool(fs & 0x01)
        elif "head" in font:
            mac = font["head"].macStyle
            bold |= bool(mac & 0x01)
            italic |= bool(mac & 0x02)

        font.close()
        return {
            "family_name": family_name or stem,
            "subfamily": subfamily,
            "bold": bold,
            "italic": italic,
        }
    except Exception as e:
        logger.debug("Failed to extract font metadata from %s: %s", stem, e)
        return {
            "family_name": stem,
            "subfamily": "",
            "bold": False,
            "italic": False,
        }


def _sanitize_and_get_bytes(source, filename: str) -> bytes:
    font_obj = TTFont(source, lazy=True)

    if "kern" in font_obj:
        try:
            kern_table = font_obj["kern"]
            if kern_table.version == 1:
                del font_obj["kern"]
            else:
                has_vertical = False
                for subtable in getattr(kern_table, "kernTables", []):
                    if hasattr(subtable, "coverage") and not (subtable.coverage & 0x01):
                        has_vertical = True
                        break
                if has_vertical:
                    del font_obj["kern"]
        except Exception as e:
            logger.debug("Failed to parse kern table in %s: %s — deleting", filename, e)
            del font_obj["kern"]

    # Align 'head' flags if a legacy hdmx table is present to avoid browser drops
    if "hdmx" in font_obj and "head" in font_obj:
        font_obj["head"].flags |= 0x0004
        font_obj["head"].flags |= 0x0010

    # Fix "maxp: Bad maxZones: 0"
    if "maxp" in font_obj:
        try:
            if hasattr(font_obj["maxp"], "maxZones") and font_obj["maxp"].maxZones == 0:
                font_obj["maxp"].maxZones = 1
        except AttributeError:
            pass

    # Fix "glyf: Glyph bbox was incorrect" (ONLY run if table is already loaded to save time)
    if "glyf" in font_obj:
        try:
            # High-cost operation: only recalculated during initial clean optimization run
            font_obj["glyf"].recalcBBoxes(font_obj)
        except Exception as e:
            logger.debug("recalcBBoxes failed for %s: %s", filename, e)

    # Fix "cmap: language id should be zero"
    if "cmap" in font_obj:
        try:
            for table in font_obj["cmap"].tables:
                if hasattr(table, "language") and table.language != 0:
                    table.language = 0
            font_obj["cmap"].compile(font_obj)
        except Exception:
            raise ValueError("Font has a critically unrecoverable cmap table.")

    # Fix "OS/2: Bad sxHeight"
    if "OS/2" in font_obj:
        try:
            if hasattr(font_obj["OS/2"], "sxHeight") and font_obj["OS/2"].sxHeight < 0:
                font_obj["OS/2"].sxHeight = 0
        except Exception as e:
            logger.debug("Failed to fix OS/2 sxHeight in %s: %s", filename, e)
    if "hdmx" in font_obj:
        del font_obj["hdmx"]
    if "LTSH" in font_obj:
        del font_obj["LTSH"]

    buffered_stream = io.BytesIO()
    font_obj.save(buffered_stream)
    font_bytes = buffered_stream.getvalue()
    font_obj.close()

    return font_bytes


def _payload(font_bytes: bytes, filename: str, metadata: dict) -> dict:
    """What the frontend is handed for one face: metadata plus the bytes inline."""
    font_title = (
        f"{metadata['family_name']} {metadata['subfamily']}"
        if metadata["subfamily"]
        else metadata["family_name"]
    )
    font_format = "opentype" if filename.lower().endswith(".otf") else "truetype"
    mime_type = "font/otf" if font_format == "opentype" else "font/ttf"

    encoded_bytes = base64.b64encode(font_bytes).decode("utf-8")

    return {
        "data_uri": f"data:{mime_type};base64,{encoded_bytes}",
        "format": font_format,
        "font_name": font_title,
        "font_filename": filename,
        "family_name": metadata["family_name"],
        "subfamily": metadata["subfamily"],
        "bold": metadata["bold"],
        "italic": metadata["italic"],
    }


def _font_files(font_dir: Path) -> list[Path]:
    """The font files in ``font_dir``, sorted; empty if it holds none or cannot be read.

    Deliberately separate from loading them. The fallback decision is "how many
    fonts are in the folder", and that has to be answerable before a single face
    has been parsed.
    """
    try:
        entries = sorted(font_dir.iterdir())
    except OSError as e:
        logger.warning("Cannot read font directory %s: %s", font_dir, e)
        return []

    return [
        path for path in entries
        if path.suffix.lower() in FONT_EXTENSIONS and path.is_file()
    ]


#: The payload keys anything downstream actually reads. ``data_uri`` is the face
#: itself, so the frontend renders nothing without it, and the two names are
#: lowercased while sorting in :func:`_filterFonts` -- where a non-string is an
#: AttributeError rather than merely a bad sort. ``subfamily`` is not checked
#: because nothing reads it.
_PAYLOAD_STR_KEYS = ("data_uri", "format", "font_name", "font_filename", "family_name")


def _valid_payload(payload: object) -> bool:
    """Whether a cached payload is shaped like one :func:`_payload` produced.

    ``.fonts_cache.json`` sits in the folder users are invited to drop fonts
    into, so it is untrusted input that merely happens to usually be ours.
    Rejecting an odd entry costs one re-parse of a font that is still sitting
    right there next to it; trusting one costs the request.
    """
    if not isinstance(payload, dict):
        return False
    if not all(isinstance(payload.get(key), str) for key in _PAYLOAD_STR_KEYS):
        return False
    return isinstance(payload.get("bold"), bool) and isinstance(payload.get("italic"), bool)


def _load_fonts_to_ram(font_files: list[Path], cache_file_path: Path):
    FONTS = []

    font_cache = {}
    if cache_file_path.exists():
        try:
            with open(cache_file_path, "r", encoding="utf-8") as f:
                font_cache = json.load(f)
        except Exception as e:
            logger.debug("Corrupt font cache, resetting: %s", e)
            font_cache = {}

        # Parsing is not validating. Every read below assumes a mapping of
        # mappings, and a file of valid JSON that is a list, a number or a
        # string satisfies `json.load` while satisfying none of that.
        if not isinstance(font_cache, dict):
            logger.debug(
                "Font cache is a %s rather than an object; resetting.",
                type(font_cache).__name__,
            )
            font_cache = {}

    cache_updated = False
    current_cache_signature = {}

    for font_path in font_files:
        filename = font_path.name

        # Calculate distinct file state properties (Name + Modification Time + Size)
        try:
            stat = font_path.stat()
            file_signature = f"{filename}_{stat.st_mtime}_{stat.st_size}"
        except Exception as e:
            # Deliberately not a string. A sentinel *is* a signature, so a cache
            # file containing that same sentinel would "match" a font this
            # process cannot even stat. None never equals anything JSON can hold.
            logger.debug("Cannot stat font file %s: %s", filename, e)
            file_signature = None

        entry = font_cache.get(filename)
        if (
            file_signature is not None
            and isinstance(entry, dict)
            and entry.get("signature") == file_signature
            and _valid_payload(entry.get("payload"))
        ):
            FONTS.append(entry["payload"])
            current_cache_signature[filename] = entry
            continue

        # CACHE MISS: Perform heavy validation processing once
        try:
            font_bytes = _sanitize_and_get_bytes(font_path, filename)
        except Exception as e:
            logger.warning("Skipping dangerous or corrupt font file: %s — %s", filename, e)
            continue

        metadata = _extract_font_metadata(font_path, font_path.stem)
        font_payload = _payload(font_bytes, filename, metadata)

        FONTS.append(font_payload)

        current_cache_signature[filename] = {
            "signature": file_signature,
            "payload": font_payload,
        }
        cache_updated = True

    if cache_updated:
        try:
            with open(cache_file_path, "w", encoding="utf-8") as f:
                json.dump(current_cache_signature, f, indent=2, ensure_ascii=False)
        except Exception as cache_ex:
            logger.warning("Failed writing font cache index: %s", cache_ex)
    else:
        # A cache written before the symbol fallback was hidden still holds its
        # entry. It is ignored above (the file is filtered before this is
        # called), but without a rewrite it lingers forever. Dropping it is a
        # pure cleanup: failure stays a debug, never a warning.
        try:
            stale = [k for k in font_cache if _is_symbol_font(k)]
            if stale and isinstance(font_cache, dict):
                pruned = {k: v for k, v in font_cache.items() if k not in stale}
                # Only rewrite when pruning actually changes the file; the
                # common case (no symbol entry) skips I/O entirely.
                if len(pruned) != len(font_cache):
                    with open(cache_file_path, "w", encoding="utf-8") as f:
                        json.dump(pruned, f, indent=2, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001 - cache cleanup never fails a request
            logger.debug("Could not prune symbol entries from font cache: %s", e)

    return FONTS


def _bundled_font_payloads() -> list[dict]:
    """The two ComicMono faces, from inside the binary.

    Reached when ``fonts/`` holds no fonts at all -- which is how a shipped build
    starts, since the folder ships empty. Without this the user is offered no font
    whatsoever and the typesetter drops to Pillow's bitmap default.

    The symbol fallback is deliberately excluded even if the bundle lists it:
    it is a per-glyph typesetting aid, never a frontend face.

    Not written to ``.fonts_cache.json``: that index is keyed by the size and
    mtime of a file in the folder, and these have no file.
    """
    from fox_reader import assets

    payloads: list[dict] = []
    for filename in assets.bundled_font_names():
        if _is_symbol_font(filename):
            continue
        raw = assets.bundled_font_bytes(filename)
        if not raw:
            continue
        try:
            # A fresh stream per read: fontTools consumes the one it is given.
            font_bytes = _sanitize_and_get_bytes(io.BytesIO(raw), filename)
            metadata = _extract_font_metadata(io.BytesIO(raw), Path(filename).stem)
        except Exception as e:  # noqa: BLE001 - a bad fallback is not fatal
            logger.warning("Bundled font %s is unusable: %s", filename, e)
            continue
        payloads.append(_payload(font_bytes, filename, metadata))

    return payloads


def _filterFonts(fonts):
    seen_names = set()
    unique_fonts = []
    for font in fonts:
        name = font.get("font_name")
        if name not in seen_names:
            seen_names.add(name)
            unique_fonts.append(font)

    def get_family_priority(font):
        font_name = font.get("font_name", "").lower()
        family_name = font.get("family_name", "").lower()
        if "cc wild words" in font_name or "cc wild words" in family_name:
            return 0
        elif "anime ace" in font_name or "anime ace" in family_name:
            return 1
        return 2

    def get_style_priority(font):
        is_bold = font.get("bold", False)
        is_italic = font.get("italic", False)
        if not is_italic and not is_bold:
            return 0
        elif is_italic and not is_bold:
            return 1
        elif not is_italic and is_bold:
            return 2
        return 3

    return sorted(
        unique_fonts,
        key=lambda x: (
            get_family_priority(x),
            get_style_priority(x),
            x.get("family_name", "").lower(),
        ),
    )


def getFonts(font_dir: Path | None = None) -> list[dict]:
    if font_dir is None:
        from fox_reader.utils import FONT_DIR

        font_dir = FONT_DIR
    try:
        font_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # A read-only install, a path that exists as a file, a revoked ACL. None
        # of what follows needs the folder to exist -- the scan finds nothing and
        # the bundled faces take over -- so this is a warning rather than the end
        # of the request. Unguarded it propagated out of getFonts, through
        # FontService's constructor, and took the server's startup with it.
        logger.warning("Cannot create font directory %s: %s", font_dir, e)
    CACHE_FILE = font_dir / ".fonts_cache.json"

    # Read the folder first, and count. The folder is what decides whether the
    # bundled fallbacks are used -- not the cache index, which is only ever an
    # optimisation for files that are already there. The symbol fallback is
    # filtered here, not in _font_files: it lives in the same folder on disk
    # (and in the bundle) but is never a frontend font and must not count as
    # "a font in the folder" -- a folder holding only it is an empty folder.
    font_files = [p for p in _font_files(font_dir) if not _is_symbol_font(p.name)]

    fonts = _load_fonts_to_ram(font_files, CACHE_FILE) if font_files else []

    if not fonts:
        fallback = _bundled_font_payloads()
        if not font_files:
            # Zero fonts in the folder. This is a fresh install: fonts/ ships
            # empty, holding at most a licence file.
            logger.info(
                "No fonts in %s; using the %d bundled fallback face(s).",
                font_dir,
                len(fallback),
            )
        else:
            # Fonts were there but not one of them survived sanitising. Falling
            # back anyway, because the alternative is offering no font at all.
            logger.warning(
                "All %d font file(s) in %s are corrupt or unreadable; "
                "using the %d bundled fallback face(s).",
                len(font_files),
                font_dir,
                len(fallback),
            )
        fonts = fallback

    return _filterFonts(fonts)
