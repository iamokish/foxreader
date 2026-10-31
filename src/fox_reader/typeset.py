"""Typesetting: repaint each region, then draw its text into it.

Ordering, colour and layout are all per entry now (see
:class:`fox_reader.models.requests.DataItem`):

* **layer** decides who is drawn over whom -- 1 at the bottom, 10 at the top --
  so a caption can be laid over a cleaned plate without the plate erasing it;
* **background** is one of ``auto`` (the region's own dominant colour), ``color``
  (a chosen one), ``transparent`` (draw nothing) or ``clean``, which repaints the
  artwork under the lettering instead of covering it
  (:mod:`fox_reader.services.clean_service`);
* **font size, colour, outline colour and outline width** are each either
  detected or given;
* **alignment** is ``left``, ``center`` or ``right``, and ``\\n`` in the text is
  honoured everywhere;
* **geometry** -- word spacing, line spacing, an X/Y glyph scale, an X/Y shift
  and an X/Y/Z rotation -- is per entry too (:class:`TextGeometry`). The
  spacings and the scales take part in the auto font fit; the shift and the
  rotation are applied afterwards, to the *lettering only*, so the plate behind
  it never moves.

An entry with no text at all is still processed: a clean-only entry is how a page
gets its lettering removed without anything put back.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache, partial
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageColor, ImageDraw, ImageFont

from fox_reader.font_calibration import FontProfile, profile_for, scale_for
from fox_reader.models.requests import DataItem
from fox_reader.utils import FONT_DIR, is_frozen

logger = logging.getLogger(__name__)

#: What the UI's font-size slider allows, and therefore what auto may pick.
MIN_FONT_SIZE = 6
MAX_FONT_SIZE = 72

#: What the outline-width control allows.
MIN_STROKE = 0
MAX_STROKE = 32

MAX_LAYER = 10

#: Per-entry geometry ranges. These are the single source of truth: the model
#: (`fox_reader.models.requests.DataItem`) accepts anything numeric and this
#: module clamps, so a project saved by a build with a wider range still
#: renders. The frontend mirrors them in `core/state.ts`.
MIN_WORD_SPACING = 0.0
MAX_WORD_SPACING = 3.0
MIN_LINE_SPACING = 0.5
MAX_LINE_SPACING = 3.0
MIN_FONT_SCALE = 0.5
MAX_FONT_SCALE = 2.0
MAX_TEXT_ANGLE = 180
#: Raw guard only -- the real limit is the page, applied in `_clamp_shift`.
MAX_TEXT_SHIFT = 100_000

TEXT_ALIGNMENTS = ("left", "center", "right")

#: Two pixels off every wrap budget, shared with the frontend's `textFit`
#: (`WRAP_SAFETY_PX`). Both engines now lay out with the faces' unhinted hmtx
#: advances (see :func:`_raw_length`), so their measurements of one string are
#: equal to well under a pixel; the margin's remaining jobs are to absorb the
#: rendered line's paste rounding (each glyph lands on an integer pixel on the
#: render, at a fractional position in the preview) and the kern pair the
#: browsers apply on the few strapped faces that carry `kern`/`GPOS`. The traced
#: "just once," measured 69.0 hinted / 71.2 unhinted against a 70px limit: a
#: shared margin makes every boundary call land the same way on both sides.
_WRAP_EPS_PX = 2.0

WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)

_FALLBACK_FONT = "ComicMono.ttf"

#: Loading a TrueType face is milliseconds, and fitting one entry asks for a
#: dozen sizes; a page of thirty entries would otherwise spend most of its time
#: in FreeType. Bounded because a page can legitimately use many faces.
_FONT_CACHE: dict[tuple[str, int], Any] = {}
_FONT_CACHE_MAX = 512

# Settings snapshot. User fallbacks use the sanitised bytes served to SVG;
# primary disk faces keep their original FreeType rendering.
_REGISTERED_FONT_BYTES: dict[str, bytes] = {}
_USER_SYMBOL_FONTS: tuple[str, ...] = ()
_FORCED_SYMBOL_OWNERS: dict[str, str] = {}
_FORCED_SYMBOL_ADJUSTMENTS: dict = {}
_DEFAULT_TEXT_FONT = _FALLBACK_FONT
_USER_SYMBOL_CACHE: dict[tuple[str, int], Any] = {}
_PRIMARY_PROFILE_CACHE: dict[str, tuple[Any, FontProfile]] = {}


def configure_fonts(raw_fonts: dict[str, bytes], symbols: tuple[str, ...], default: str,
                    profiles: dict[str, FontProfile] | None = None,
                    characters: dict[str, str] | None = None,
                    glyph_adjustments: dict | None = None) -> None:
    global _REGISTERED_FONT_BYTES, _USER_SYMBOL_FONTS, _DEFAULT_TEXT_FONT, _FORCED_SYMBOL_OWNERS
    global _FORCED_SYMBOL_ADJUSTMENTS
    same_bytes = (raw_fonts.keys() == _REGISTERED_FONT_BYTES.keys()
                  and all(raw is _REGISTERED_FONT_BYTES[name] for name, raw in raw_fonts.items()))
    _REGISTERED_FONT_BYTES, _USER_SYMBOL_FONTS, _DEFAULT_TEXT_FONT = dict(raw_fonts), symbols, default
    _FORCED_SYMBOL_OWNERS = {ch: name for name, chars in (characters or {}).items() for ch in chars}
    _FORCED_SYMBOL_ADJUSTMENTS = glyph_adjustments or {}
    if same_bytes:
        return  # Changing priority does not invalidate font faces or profiles.
    _FONT_CACHE.clear()
    _USER_SYMBOL_CACHE.clear()
    _PRIMARY_PROFILE_CACHE.clear()
    _PRIMARY_CMAP_CACHE.clear()
    _SYMBOL_SCALE_CACHE.clear()
    _LINE_METRICS_CACHE.clear()
    if profiles:
        _PRIMARY_PROFILE_CACHE.update({name: (id(raw_fonts[name]), profile) for name, profile in profiles.items()})


def _text_font_name(name: str) -> str:
    hidden = (name.lower() == _SYMBOL_FONT.lower()
              or any(name.casefold() == font.casefold() for font in _USER_SYMBOL_FONTS))
    return _DEFAULT_TEXT_FONT if hidden else name


# --------------------------------------------------------------------- colours

def parse_color(value: Any, default: tuple[int, int, int, int] | None = None
                ) -> tuple[int, int, int, int] | None:
    """A colour as RGBA, or ``default``.

    Accepts what the palette and the eyedropper produce (``#rgb``, ``#rrggbb``,
    ``#rrggbbaa``) and anything else Pillow knows, so a hand-typed ``red`` also
    works. Never raises: an unparseable colour falls back rather than failing a
    whole page of typesetting.
    """
    if value is None:
        return default
    if isinstance(value, (tuple, list)):
        parts = [int(max(0, min(255, int(round(float(v)))))) for v in value[:4]]
        if len(parts) == 3:
            parts.append(255)
        return tuple(parts) if len(parts) == 4 else default  # type: ignore[return-value]

    text = str(value).strip()
    if not text or text.lower() in ("auto", "none", "transparent"):
        return default
    if text.startswith("#") and len(text) == 9:  # #rrggbbaa
        try:
            r, g, b, a = (int(text[i:i + 2], 16) for i in (1, 3, 5, 7))
            return (r, g, b, a)
        except ValueError:
            return default
    try:
        rgba = ImageColor.getcolor(text, "RGBA")
    except (ValueError, AttributeError):
        return default
    if len(rgba) == 3:
        return (*rgba, 255)
    return rgba  # type: ignore[return-value]


def _brightness(color: tuple[int, int, int, int]) -> float:
    return (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000.0


def _contrast_color(background: tuple[int, int, int, int]
                    ) -> tuple[int, int, int, int]:
    return BLACK if _brightness(background) > 127 else WHITE


#: How far apart a hinted font colour and the plate have to be, on the 0-255
#: brightness scale, for the hint to be worth using. The hint was measured off
#: the original artwork (see `fox_reader.ocr_colors`), so when the user then
#: paints an explicit plate under it the two can collide -- white lettering
#: hinted off a dark panel, set on a white plate, would be invisible. Below this
#: the hint is dropped and `_contrast_color` decides, which can never be.
_MIN_PLATE_CONTRAST = 40.0


def _legible_on(hint: tuple[int, int, int, int] | None,
                plate: tuple[int, int, int, int]
                ) -> tuple[int, int, int, int] | None:
    """``hint``, unless it would disappear into ``plate``.

    A hint that is transparent is dropped too. The user's own ``font_color`` is
    allowed to be, because invisible text is a thing someone can mean to ask
    for; a measurement that came out transparent is only ever a bad hint.
    """
    if hint is None or hint[3] == 0:
        return None
    if abs(_brightness(hint) - _brightness(plate)) < _MIN_PLATE_CONTRAST:
        return None
    return hint


def _dominant_color(img: Image.Image, box: tuple[int, int, int, int]
                    ) -> tuple[int, int, int, int]:
    """The most common colour in ``box`` -- the region's background.

    The crop is capped at 128x128 before counting. ``getcolors`` over a
    full-resolution speech bubble builds a dict of every distinct RGBA value in
    it, which on a scanned page is most of the pixels; sampling loses nothing,
    because what is wanted is the *most frequent* colour and the flat plate is
    still the most frequent after downsampling.
    """
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return WHITE
    crop = img.crop((x0, y0, x1, y1)).convert("RGBA")
    w, h = crop.size
    if w * h > 128 * 128:
        scale = (128 * 128 / float(w * h)) ** 0.5
        crop = crop.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                           Image.Resampling.NEAREST)
    colors = crop.getcolors(maxcolors=crop.size[0] * crop.size[1])
    if not colors:
        return WHITE
    count, color = max(colors, key=lambda item: item[0])
    if isinstance(color, int):  # paletted / single-band crop
        return (color, color, color, 255)
    if len(color) == 3:
        return (*color, 255)
    return color


# ----------------------------------------------------------------------- fonts

def _load_font(fontfile: str, size: int) -> Any:
    """A face at ``size``, falling back rather than failing.

    Looks in ``fonts/`` first and then inside the binary. A shipped build has an
    empty ``fonts/`` folder, so without the embedded step every entry would be
    typeset with ``ImageFont.load_default()`` -- an 11 px bitmap face that ignores
    ``size`` entirely, which does not read as a fallback so much as as a bug.
    """
    size = int(max(1, size))
    name = _text_font_name((fontfile or "").strip())
    for candidate in (name, _text_font_name(_FALLBACK_FONT)):
        if not candidate:
            continue
        path = str(FONT_DIR / candidate)
        key = (path, size)
        cached = _FONT_CACHE.get(key)
        if cached is not None:
            return cached
        font = None
        raw = _REGISTERED_FONT_BYTES.get(candidate)
        if os.path.isfile(path):
            try:
                font = ImageFont.truetype(path, size)
            except OSError as exc:
                logger.warning("Could not load font %s: %s", candidate, exc)
        if font is None and raw is not None:
            try:
                font = ImageFont.truetype(BytesIO(raw), size)
                _attach_raw_advances(font, raw)
            except OSError:
                pass
        if font is None:
            font = _embedded_font(candidate, size)
        if font is None:
            continue
        if font.path == path:
            _attach_raw_advances(font, path)
        if len(_FONT_CACHE) >= _FONT_CACHE_MAX:
            _FONT_CACHE.clear()
        _FONT_CACHE[key] = font
        return font
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without a sizable default
        return ImageFont.load_default()


def _embedded_font(candidate: str, size: int) -> Any | None:
    """One of the faces compiled into the binary, at ``size``, or None."""
    from fox_reader import assets

    raw = assets.bundled_font_bytes(candidate)
    if not raw:
        return None
    try:
        font = ImageFont.truetype(BytesIO(raw), size)
    except OSError as exc:
        logger.warning("Could not load bundled font %s: %s", candidate, exc)
        return None
    _attach_raw_advances(font, raw)
    return font


# ---------------------------------------------------------------- line metrics
# The preview must advance lines by the *same integer* Pillow's
# ``font.getmetrics()`` yields, but FreeType's per-pixel-size rounding cannot be
# reproduced from the font-table units: for ComicMono, 49 of the 67 sizes the
# UI searches (6..72) disagree by a pixel between ``round(size*a/upem) +
# round(size*d/upem)`` and the real ``getmetrics()``, and the ascent alone is
# off for ~70 sizes of 1..144. So the backend measures its own engine once per
# face and ships the table with ``/api/fonts``; the preview looks sizes up in
# it and keeps its rounded-font-unit formula only for a payload that predates
# the field (an empty table).

_LINE_METRICS_CACHE: dict[str, list[int]] = {}


def line_metrics_for(fontfile: str) -> list[int]:
    """``[ascent, descent, ascent, descent, ...]`` for every size in the fit
    range (:data:`MIN_FONT_SIZE` .. :data:`MAX_FONT_SIZE`, ascending), via the
    same :func:`_load_font` call the real drawing uses.

    A missing face resolves through the fallback (``ComicMono``), exactly as
    drawing would, so the table always describes the face the backend would
    actually paint with. All-or-nothing: an empty list when *any* size fails to
    give a usable ascent+descent, because a preview that mixes table entries
    with its own estimate could wrap or centre unlike either engine. Never
    raises; a whole unusable face is simply "no table" and the preview keeps
    its estimate.
    """
    key = (fontfile or "").strip() or _FALLBACK_FONT
    cached = _LINE_METRICS_CACHE.get(key)
    if cached is not None:
        return cached
    table: list[int] = []
    for size in range(MIN_FONT_SIZE, MAX_FONT_SIZE + 1):
        try:
            ascent, descent = _load_font(key, size).getmetrics()
        except Exception:  # noqa: BLE001 - metrics are best-effort
            table = []
            break
        if int(ascent) + int(descent) <= 0:
            table = []
            break
        table.extend((int(ascent), int(descent)))
    if len(_LINE_METRICS_CACHE) >= _FONT_CACHE_MAX:
        # FreeType face objects are already RAM-cached by size in `_FONT_CACHE`
        # (these tables share that loading), so the cap here is about not
        # holding an unbounded number of distinct face names, not memory
        # pressure -- each table is a few hundred bytes.
        _LINE_METRICS_CACHE.clear()
    _LINE_METRICS_CACHE[key] = table
    return table


# ---------------------------------------------------------------- raw advances
# Pillow's BASIC layout (this build has no raqm) advances each glyph by a
# hinted, per-size-rounded step, and every line it measures comes out as a
# whole number of pixels. The preview lays the same string out in the browser,
# which walks the face's unhinted hmtx advances scaled only by size/upem -- the
# same string can differ by 2px per line, which is what made the same font size
# render and preview differently. The fix is to measure (and place) with the
# face's own advance table: the two engines then lay out identically, and the
# residual differences are the per-glyph paste rounding plus the kern pairs of
# the few faces that carry `kern`/`GPOS`.

#: The parsed ``{upem, adv, notdef}`` tables, keyed by file signature (disk) or
#: by size-of-content hash (bundle). Parsed once per face, never re-read per
#: load, which would re-parse the file for every cached Pillow size.
_RAW_ADV_CACHE: dict[tuple[str, ...], dict[str, Any] | None] = {}
_RAW_ADV_CACHE_MAX = 64


def _parse_raw_advances(source: Any) -> dict[str, Any] | None:
    """``{upem, adv, notdef}`` from a font file, or None when unreadable.

    ``adv`` maps codepoint -> advance units along the best unicode cmap, and
    ``notdef`` is glyph 0's advance -- what a browser falls back to for any
    character the face does not cover. An unreadable face returning None only
    keeps the hinted measurement: never worth failing a render over.
    """
    try:
        tt = TTFont(source, lazy=True)
        try:
            upem = int(tt["head"].unitsPerEm or 1000)
            if upem <= 0:
                return None
            cmap = tt.getBestCmap()
            hmtx = tt["hmtx"]
            glyph_order = tt.getGlyphOrder()
            notdef = int(hmtx[glyph_order[0]][0])
            adv: dict[int, int] = {}
            if cmap:
                for cp, glyph in cmap.items():
                    try:
                        adv[int(cp)] = int(hmtx[glyph][0])
                    except (KeyError, IndexError, TypeError, ValueError):
                        continue
            return {"upem": upem, "adv": adv, "notdef": notdef}
        finally:
            try:
                tt.close()
            except Exception:  # noqa: BLE001 - close must never be fatal
                pass
    except Exception:  # noqa: BLE001 - coverage is best-effort
        return None


def _attach_raw_advances(font: Any, source: Any) -> None:
    """Give ``font`` its unhinted advance table, or a False sentinel.

    ``source`` is the face's path (a table is parsed straight off the file,
    never out of the Pillow font, whose buffer must stay untouched) or the
    bundle's bytes for an embedded face. The sentinel distinguishes
    "tried and failed" from "not tried yet" so a bad face is not re-read on
    every measure.
    """
    try:
        if getattr(font, "_fox_raw_adv", None) is not None:
            return
        if isinstance(source, bytes):
            key = ("embed", hashlib.md5(source).hexdigest())
        else:
            try:
                st = os.stat(source)
            except OSError:
                font._fox_raw_adv = False
                return
            key = ("file", str(source), st.st_mtime_ns, st.st_size)
        if key not in _RAW_ADV_CACHE:
            if len(_RAW_ADV_CACHE) >= _RAW_ADV_CACHE_MAX:
                _RAW_ADV_CACHE.clear()
            # A bytes source must reach fontTools as a stream: TTFont wants a
            # path or a file object, and a bare bytes object is neither, which
            # silently downgraded every bytes-loaded face (the bundled faces
            # and the symbol fallback) to the hinted measurement.
            if isinstance(source, bytes):
                parsed = _parse_raw_advances(BytesIO(source))
            else:
                parsed = _parse_raw_advances(str(source))
            _RAW_ADV_CACHE[key] = parsed
        table = _RAW_ADV_CACHE[key]
        font._fox_raw_adv = table if table else False
    except Exception:  # noqa: BLE001 - a table is an improvement, never a duty
        try:
            font._fox_raw_adv = False
        except Exception:  # noqa: BLE001
            pass


def _raw_length(font: Any, text: str) -> float | None:
    """Sum of ``text``'s unhinted advance widths, or None without a table.

    This is what the browser lays: each glyph's hmtx advance times
    ``size/upem``, glyphs the face lacks falling to .notdef. The hinted walk
    (:func:`_length`'s fallback) differs from it by up to a couple of pixels
    per line, which is why the tables exist.
    """
    table = getattr(font, "_fox_raw_adv", None)
    if table is None or table is False:
        return None
    try:
        adv = table["adv"]
        notdef = table["notdef"]
        total = 0
        for ch in text:
            total += adv.get(ord(ch), notdef)
        scale = float(font.size) / float(table["upem"])
        return total * scale
    except Exception:  # noqa: BLE001 - keep hinted measuring on any surprise
        return None


# ------------------------------------------------- symbol fallback (inpainting)
# NotoSansSymbols2 covers the symbols/characters user faces usually lack. It is
# typesetting-only: never offered to the frontend (see fox_reader.fonts), only
# used per-glyph here when the selected face does not contain a character.
#
# Loading is lazy and RAM-resident: the first typeset that needs it reads the
# bytes once -- from the embedded bundle in a compiled build (where fonts/
# ships empty) or from fonts/ on disk in a checkout -- and every later call
# reuses the cached bytes, cmap and Pillow faces. A missing or corrupt symbol
# face never fails typesetting; it only disables the fallback.

_SYMBOL_FONT = "NotoSansSymbols2-Regular.otf"

_SYMBOL_FONT_CACHE: dict[int, Any] = {}
_SYMBOL_BYTES: bytes | None = None
_SYMBOL_BYTES_SIG: str | None = None
_SYMBOL_BYTES_MISSING = False
_SYMBOL_CMAP: set[int] | None = None
_SYMBOL_CMAP_SIG: str | None = None
_SYMBOL_WARNED = False

#: Primary-face coverages, keyed by effective font filename. Values are
#: ``(signature, cmap)`` where ``cmap`` is None when the face cannot be read
#: (unknown coverage => no fallback, preserving the old single-face behaviour).
_PRIMARY_CMAP_CACHE: dict[str, tuple[str | None, set[int] | None]] = {}
_PRIMARY_CMAP_MAX = 64


def _warn_symbol_once(message: str, *args: Any) -> None:
    global _SYMBOL_WARNED
    if _SYMBOL_WARNED:
        return
    _SYMBOL_WARNED = True
    logger.warning(message, *args)


def _disk_sig(path: Any) -> str | None:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}_{st.st_size}"
    except OSError:
        return None
    except Exception:  # noqa: BLE001 - stat must never be fatal
        return None


def _symbol_font_bytes() -> bytes | None:
    """Raw bytes of the symbol fallback, cached in RAM, or None.

    Compiled (frozen) builds read the embedded asset only and never touch
    ``fonts/`` on disk: the staged ``fonts/`` folder holds licences, never the
    face. Checkouts read the embedded bundle first when one is present (tests,
    ``FOX_READER_DISK_ASSETS`` aside) and ``fonts/`` on disk otherwise.
    Disk reads re-check mtime/size, so replacing the .otf is picked up without
    a restart; a missing/unreadable face is remembered to avoid disk I/O on
    every line, but a later call still retries if the file appears.
    Never raises.
    """
    global _SYMBOL_BYTES, _SYMBOL_BYTES_SIG, _SYMBOL_BYTES_MISSING
    if is_frozen:
        # Strict bundle-only path: no stat, no read, no scan of FONT_DIR.
        # ``assets.symbol_font_bytes`` itself refuses disk while a bundle is
        # present, and without a bundle there is no disk source a shipped
        # tree may use -- so a failed lookup ends here rather than falling
        # through to the checkout-only code below.
        try:
            from fox_reader import assets

            if assets.has_bundle():
                try:
                    raw = assets.symbol_font_bytes()
                except Exception:  # noqa: BLE001 - lookup must never be fatal
                    raw = None
                if raw:
                    if _SYMBOL_BYTES is None or _SYMBOL_BYTES_SIG != "bundled":
                        if _SYMBOL_BYTES_SIG is not None and _SYMBOL_BYTES_SIG != "bundled":
                            _SYMBOL_FONT_CACHE.clear()
                        _SYMBOL_BYTES = raw
                        _SYMBOL_BYTES_SIG = "bundled"
                        _SYMBOL_BYTES_MISSING = False
                    return _SYMBOL_BYTES
        except Exception:  # noqa: BLE001 - assets import must never be fatal
            pass
        if not _SYMBOL_BYTES_MISSING:
            _warn_symbol_once(
                "Symbol fallback %s is not in the embedded bundle; "
                "typesetting without per-glyph fallback.",
                _SYMBOL_FONT,
            )
        _SYMBOL_BYTES_MISSING = True
        return None
    try:
        from fox_reader import assets

        # Bundled build: the bytes are static, cache forever.
        try:
            raw = assets.symbol_font_bytes()
        except Exception:  # noqa: BLE001 - lookup must never be fatal
            raw = None
        if raw:
            if _SYMBOL_BYTES is None or _SYMBOL_BYTES_SIG != "bundled":
                if _SYMBOL_BYTES_SIG is not None and _SYMBOL_BYTES_SIG != "bundled":
                    _SYMBOL_FONT_CACHE.clear()
                _SYMBOL_BYTES = raw
                _SYMBOL_BYTES_SIG = "bundled"
                _SYMBOL_BYTES_MISSING = False
            return _SYMBOL_BYTES
        # No bundle (checkout, or DISK_ASSETS override): fall through to disk.
        # When a bundle *is* present but holds no symbol face, disk must not
        # shadow it -- same rule as bundled_font_bytes.
        try:
            if assets.has_bundle():
                if not _SYMBOL_BYTES_MISSING:
                    _warn_symbol_once(
                        "Symbol fallback %s is not in the embedded bundle; "
                        "typesetting without per-glyph fallback.",
                        _SYMBOL_FONT,
                    )
                _SYMBOL_BYTES_MISSING = True
                return None
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001 - assets import must never be fatal
        pass

    # Disk: fonts/NotoSansSymbols2-Regular.otf (case-insensitive retry).
    try:
        exact = FONT_DIR / _SYMBOL_FONT
        sig = _disk_sig(exact)
        if sig is not None and exact.is_file():
            if _SYMBOL_BYTES is not None and _SYMBOL_BYTES_SIG == sig:
                return _SYMBOL_BYTES
            try:
                data = exact.read_bytes()
            except OSError as exc:
                _warn_symbol_once("Could not read symbol font %s: %s", _SYMBOL_FONT, exc)
                return None
            if data:
                if _SYMBOL_BYTES_SIG is not None and _SYMBOL_BYTES_SIG != sig:
                    _SYMBOL_FONT_CACHE.clear()
                _SYMBOL_BYTES = data
                _SYMBOL_BYTES_SIG = sig
                _SYMBOL_BYTES_MISSING = False
                return data
            return None
        # Case-insensitive scan for a differently-cased copy.
        try:
            for child in FONT_DIR.iterdir():
                try:
                    if not child.is_file() or child.name.lower() != _SYMBOL_FONT.lower():
                        continue
                except OSError:
                    continue
                csig = _disk_sig(child)
                if _SYMBOL_BYTES is not None and _SYMBOL_BYTES_SIG == csig:
                    return _SYMBOL_BYTES
                try:
                    data = child.read_bytes()
                except OSError:
                    continue
                if data:
                    if _SYMBOL_BYTES_SIG is not None and _SYMBOL_BYTES_SIG != csig:
                        _SYMBOL_FONT_CACHE.clear()
                    _SYMBOL_BYTES = data
                    _SYMBOL_BYTES_SIG = csig
                    _SYMBOL_BYTES_MISSING = False
                    return data
        except OSError:
            pass
    except Exception:  # noqa: BLE001 - disk lookup must never be fatal
        pass
    if not _SYMBOL_BYTES_MISSING:
        logger.debug("Symbol font %s not found in %s.", _SYMBOL_FONT, FONT_DIR)
    _SYMBOL_BYTES_MISSING = True
    # A replaced file must still be picked up: a "missing" verdict is only
    # sticky until the file appears, because the exact-path stat above runs on
    # every call. Keep any previously-loaded bytes (from an older file) rather
    # than dropping them while the file is absent.
    return _SYMBOL_BYTES


def _load_symbol_font(size: int) -> Any | None:
    """The symbol face at ``size``, or None.

    Callers pass the calibrated size from :func:`_symbol_px_size` -- the
    fallback is drawn per-primary-face so its symbols read the primary's cap
    height rather than the raw em size, which is what made them look right
    next to ComicMono and undersized next to the other faces.
    """
    size = int(max(1, size))
    raw = _symbol_font_bytes()
    if not raw:
        return None
    # Bytes call above cleared the cache when the .otf changed, so a hit here
    # is guaranteed to be from the current file/bundle.
    cached = _SYMBOL_FONT_CACHE.get(size)
    if cached is not None:
        return cached
    try:
        font = ImageFont.truetype(BytesIO(raw), size)
    except OSError as exc:
        _warn_symbol_once("Could not load symbol font %s: %s", _SYMBOL_FONT, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - a bad fallback is not fatal
        _warn_symbol_once("Could not load symbol font %s: %s", _SYMBOL_FONT, exc)
        return None
    _attach_raw_advances(font, raw)
    if len(_SYMBOL_FONT_CACHE) >= _FONT_CACHE_MAX:
        _SYMBOL_FONT_CACHE.clear()
    _SYMBOL_FONT_CACHE[size] = font
    return font


#: Used only to verify the face is loadable. Optical calibration uses outlines
#: in font units (font_calibration), so there is no size-specific raster probe.
_SYMBOL_SCALE_PROBE = 64

#: Effective primary face, refreshed Noto signature and immutable profile.
_SYMBOL_SCALE_CACHE: dict[tuple[str, str | None, FontProfile], float] = {}
_SYMBOL_SCALE_CACHE_MAX = 64


def symbol_scale_for(fontfile: str) -> float:
    """Outline-based Noto scale; refresh byte signatures before cache lookup."""
    try:
        raw = _symbol_font_bytes()
        if not raw:
            return 1.0
        effective = _effective_fontfile(fontfile)
        primary = _profile_for_font(effective)
        if _load_symbol_font(_SYMBOL_SCALE_PROBE) is None:
            return 1.0
        _load_font(effective, _SYMBOL_SCALE_PROBE)
        key = (effective, _SYMBOL_BYTES_SIG, primary)
        cached = _SYMBOL_SCALE_CACHE.get(key)
        if cached is not None:
            return cached
        scale = scale_for(primary, profile_for(raw))
        if len(_SYMBOL_SCALE_CACHE) >= _SYMBOL_SCALE_CACHE_MAX:
            _SYMBOL_SCALE_CACHE.clear()
        _SYMBOL_SCALE_CACHE[key] = scale
        return scale
    except Exception:
        logger.debug("Symbol scale lookup failed for %r", fontfile, exc_info=True)
        return 1.0


def _profile_for_font(effective: str) -> FontProfile:
    registered = _REGISTERED_FONT_BYTES.get(effective)
    signature = id(registered) if registered is not None else _disk_sig(FONT_DIR / effective)
    cached = _PRIMARY_PROFILE_CACHE.get(effective)
    if cached is not None and cached[0] == signature:
        return cached[1]
    raw = _primary_font_bytes(effective)
    profile = profile_for(raw) if raw else FontProfile(frozenset())
    if len(_PRIMARY_PROFILE_CACHE) >= _FONT_CACHE_MAX:
        _PRIMARY_PROFILE_CACHE.clear()
    _PRIMARY_PROFILE_CACHE[effective] = (signature, profile)
    return profile


def _symbol_px_size(fontfile: str, size: int) -> int:
    """The pixel size the symbol face renders at next to ``size`` primary px.

    Every path that materialises the fallback face -- measuring
    (:func:`_mixed_length` via the layout), laying out (``_line_pieces``),
    drawing (:func:`_draw_block` through ``_symbol_faces``) and the stretch
    tile (``_text_layer``'s ``ksize``) -- goes through the two
    :func:`_symbol_faces`/:func:`_max_size` callers of this, so both engines
    agree integer-for-integer on what `round(size * scale)` produced. It is
    never below one: the rounding can only make small sizes *equal*, never
    sub-pixel.
    """
    size = max(1, int(size))
    try:
        return max(1, _round_half_up(size * symbol_scale_for(fontfile)))
    except Exception:  # noqa: BLE001 - an odd factor never breaks the size
        return size


def _cmap_from_bytes(raw: bytes) -> set[int] | None:
    """Codepoints in ``raw`` font bytes, or None when unreadable.

    Uses the top-level :class:`fontTools.ttLib.TTFont` import -- the same one
    :mod:`fox_reader.fonts` uses -- rather than a function-level import, so the
    dependency is visible to static analysis (and Nuitka's import recursion) up
    front instead of surfacing mid-optimisation.
    """
    if not raw:
        return None
    try:
        tt = TTFont(BytesIO(raw), lazy=True)
        try:
            return set(profile_for(raw).coverage)
        finally:
            try:
                tt.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001 - coverage is best-effort
        logger.debug("Could not read font cmap: %s", exc)
        return None


def _effective_fontfile(fontfile: str) -> str:
    """Which face :func:`_load_font` will actually use for ``fontfile``."""
    name = _text_font_name((fontfile or "").strip())
    for candidate in (name, _text_font_name(_FALLBACK_FONT)):
        if not candidate:
            continue
        if candidate in _REGISTERED_FONT_BYTES:
            return candidate
        try:
            if (FONT_DIR / candidate).is_file():
                return candidate
        except OSError:
            pass
        except Exception:  # noqa: BLE001
            pass
        try:
            from fox_reader import assets

            if assets.bundled_font_bytes(candidate):
                return candidate
        except Exception:  # noqa: BLE001
            pass
    return _FALLBACK_FONT


def _primary_font_bytes(effective: str) -> bytes | None:
    if effective in _REGISTERED_FONT_BYTES:
        return _REGISTERED_FONT_BYTES[effective]
    try:
        path = FONT_DIR / effective
        if path.is_file():
            try:
                return path.read_bytes()
            except OSError:
                pass
    except OSError:
        pass
    except Exception:  # noqa: BLE001
        pass
    try:
        from fox_reader import assets

        return assets.bundled_font_bytes(effective)
    except Exception:  # noqa: BLE001
        return None


def _primary_cmap(fontfile: str) -> set[int] | None:
    """Codepoints the effective primary face covers, or None when unknown."""
    try:
        effective = _effective_fontfile(fontfile)
    except Exception:  # noqa: BLE001
        return None
    # Signature so a replaced .ttf invalidates the cached cmap.
    sig: str | None = None
    try:
        p = FONT_DIR / effective
        if effective in _REGISTERED_FONT_BYTES:
            sig = str(id(_REGISTERED_FONT_BYTES[effective]))
        elif p.is_file():
            sig = _disk_sig(p)
        else:
            sig = "bundled"
    except Exception:  # noqa: BLE001
        sig = None
    try:
        hit = _PRIMARY_CMAP_CACHE.get(effective)
        if hit is not None and hit[0] == sig:
            return hit[1]
    except Exception:  # noqa: BLE001
        pass
    raw = _primary_font_bytes(effective)
    cmap = set(_profile_for_font(effective).coverage) if raw else None
    try:
        if len(_PRIMARY_CMAP_CACHE) >= _PRIMARY_CMAP_MAX:
            _PRIMARY_CMAP_CACHE.clear()
        _PRIMARY_CMAP_CACHE[effective] = (sig, cmap)
    except Exception:  # noqa: BLE001
        pass
    return cmap


def _symbol_cmap() -> set[int] | None:
    """Codepoints the symbol fallback covers, or None when unavailable."""
    global _SYMBOL_CMAP, _SYMBOL_CMAP_SIG
    # Invalidate when the underlying bytes changed (replaced .otf).
    try:
        current = _symbol_font_bytes()
        sig = _SYMBOL_BYTES_SIG
    except Exception:  # noqa: BLE001
        current = None
        sig = None
    if current is None:
        _SYMBOL_CMAP = None
        _SYMBOL_CMAP_SIG = sig
        return None
    if _SYMBOL_CMAP is not None and _SYMBOL_CMAP_SIG == sig:
        return _SYMBOL_CMAP
    cmap = _cmap_from_bytes(current)
    _SYMBOL_CMAP = cmap
    _SYMBOL_CMAP_SIG = sig
    return cmap


def _needs_symbol(char: str, primary_cmap: set[int] | None,
                  symbol_cmap: set[int] | None) -> bool:
    """Whether ``char`` must be set in the symbol face."""
    try:
        if not char or len(char) != 1:
            return False
        if primary_cmap is None or symbol_cmap is None:
            return False
        cp = ord(char)
        if cp in primary_cmap:
            return False
        return cp in symbol_cmap
        # Covered by neither: keep the primary's .notdef rather than swapping
        # one tofu for another.
    except Exception:  # noqa: BLE001 - coverage check never fails a line
        return False


def _text_needs_symbol(text: str, primary_cmap: set[int] | None,
                       symbol_cmap: set[int] | None) -> bool:
    if not text or primary_cmap is None or symbol_cmap is None:
        return False
    try:
        for ch in text:
            if ch in ("\n", "\r"):
                continue
            if _needs_symbol(ch, primary_cmap, symbol_cmap):
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _split_runs(text: str, primary_cmap: set[int] | None,
                symbol_cmap: set[int] | None) -> list[tuple[str, bool]]:
    """``text`` as ``(segment, use_symbol)`` runs of consecutive faces."""
    if not text:
        return []
    if primary_cmap is None or symbol_cmap is None:
        return [(text, False)]
    try:
        runs: list[tuple[str, bool]] = []
        buf: list[str] = []
        cur: bool | None = None
        for ch in text:
            need = _needs_symbol(ch, primary_cmap, symbol_cmap)
            if cur is None:
                cur = need
                buf.append(ch)
            elif need == cur:
                buf.append(ch)
            else:
                runs.append(("".join(buf), cur))
                buf = [ch]
                cur = need
        if buf:
            runs.append(("".join(buf), cur if cur is not None else False))
        return runs or [(text, False)]
    except Exception:  # noqa: BLE001
        return [(text, False)]


def _mixed_length(font: Any, symbol_font: Any | None, text: str,
                  primary_cmap: set[int] | None = None,
                  symbol_cmap: set[int] | None = None) -> float:
    """Width of ``text`` with per-glyph fallback applied."""
    if not text:
        return 0.0
    if symbol_font is None or primary_cmap is None or symbol_cmap is None:
        return _length(font, text)
    if isinstance(symbol_font, _SymbolCascade):
        return sum(_length(symbol_font.faces.get(ch, font), ch) for ch in text)
    try:
        total = 0.0
        for seg, use_sym in _split_runs(text, primary_cmap, symbol_cmap):
            total += _length(symbol_font if use_sym else font, seg)
        return total
    except Exception:  # noqa: BLE001 - measurement must never be fatal
        return _length(font, text)


def _length(font: Any, text: str) -> float:
    """Advance width of ``text``, unhinted when the face's table is attached.

    The unhinted sum (:func:`_raw_length`) is what the preview lays out, so
    with a table this is the browser-scale width; faces that never got a
    table (unreadable file, tests' fakes) keep the hinted ``getlength``
    fall path they have always had.
    """
    if not text:
        return 0.0
    raw = _raw_length(font, text)
    if raw is not None:
        return raw
    try:
        return float(font.getlength(text))
    except (AttributeError, OSError):
        try:
            box = font.getbbox(text)
            return float(box[2] - box[0])
        except Exception:  # noqa: BLE001 - measurement must never be fatal
            return float(len(text) * 8)


def _line_height(font: Any, size: int) -> int:
    """Ascent + descent, so every line advances by the same amount.

    Measuring each line's own bounding box instead would make the gap depend on
    whether the line happens to contain a descender, which reads as ragged
    leading. A blank line -- what ``\\n\\n`` produces -- then also gets exactly the
    same advance as a full one, which is what "normal vertical spacing" means.
    """
    try:
        ascent, descent = font.getmetrics()
        if ascent + descent > 0:
            return int(ascent + descent)
    except (AttributeError, OSError):
        pass
    return int(max(1, round(size * 1.2)))


def _ascent(font: Any) -> int | None:
    """Baseline offset of ``font`` (pixels above the baseline), or None.

    Needed to convert a line's ascender-top ``y`` (the ``la``/``ma``/``ra``
    convention used throughout :func:`_draw_block`) into the baseline ``y``
    that ``ls``-anchored drawing expects. None when the face cannot say --
    the caller then falls back to unshifted ``la`` drawing rather than
    guessing.
    """
    try:
        ascent, _descent = font.getmetrics()
        ascent = int(ascent)
        if ascent > 0:
            return ascent
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    except Exception:  # noqa: BLE001 - metrics must never be fatal
        pass
    return None


# ---------------------------------------------------------------------- layout

def _hyphen_pieces(word: str) -> list[str]:
    """``word`` cut after each of its own single hyphens, hyphen kept.

    ``"tot-al"`` -> ``["tot-", "al"]``. Only breaks the author already wrote are
    offered, which is the whole rule: a run of two or more hyphens is punctuation,
    never a break, and a hyphen at either end of the word is not one either, since
    the piece it would leave behind is nothing but punctuation.
    """
    pieces: list[str] = []
    start = i = 0
    n = len(word)
    while i < n:
        if word[i] != "-":
            i += 1
            continue
        j = i
        while j < n and word[j] == "-":
            j += 1
        if j - i == 1 and i > start and j < n:
            pieces.append(word[start:j])
            start = j
        i = j
    pieces.append(word[start:])
    return [p for p in pieces if p] or [word]


def _break_word(font: Any, word: str, max_w: float,
                 symbol_font: Any | None = None,
                 primary_cmap: set[int] | None = None,
                 symbol_cmap: set[int] | None = None) -> list[str]:
    """A word too wide for the line, split at its own hyphens or left whole.

    Never mid-glyph-run: cutting ``Total`` into ``tot`` / ``al`` reads as a typo,
    so a word with no usable hyphen comes back as one piece. The caller then puts
    it on a line of its own and :func:`_fit_font_size` shrinks the type until the
    block fits, which is the right lever -- the alternative invents a hyphen the
    author did not write.

    Widths are measured with per-glyph symbol fallback when ``symbol_font`` and
    both cmaps are given; otherwise exactly as before (single face). No
    word-spacing term here: the pieces are fragments of one word, so there is no
    inter-word gap to stretch.
    """
    pieces = _hyphen_pieces(word)
    if len(pieces) < 2:
        return [word]
    def _w(text: str) -> float:
        return _mixed_length(font, symbol_font, text, primary_cmap, symbol_cmap)
    lines: list[str] = []
    current = ""
    for piece in pieces:
        trial = current + piece
        if current and _w(trial) > max_w:
            lines.append(current)
            current = piece
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [word]


def _wrap_paragraph(font: Any, text: str, max_w: float,
                    symbol_font: Any | None = None,
                    primary_cmap: set[int] | None = None,
                    symbol_cmap: set[int] | None = None,
                    word_extra: float = 0.0) -> list[str]:
    def _w(text: str) -> float:
        return _line_width(font, symbol_font, text, primary_cmap, symbol_cmap,
                           word_extra)
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if _w(word) > max_w and not current:
            pieces = _break_word(font, word, max_w, symbol_font, primary_cmap, symbol_cmap)
            lines.extend(pieces[:-1])
            current = [pieces[-1]]
            continue
        trial = " ".join(current + [word]) if current else word
        if _w(trial) <= max_w or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            if _w(word) > max_w:
                pieces = _break_word(font, word, max_w, symbol_font, primary_cmap, symbol_cmap)
                lines.extend(pieces[:-1])
                current = [pieces[-1]]
            else:
                current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _layout(font: Any, text: str, max_w: float, align: str,
            symbol_font: Any | None = None,
            primary_cmap: set[int] | None = None,
            symbol_cmap: set[int] | None = None,
            word_extra: float = 0.0) -> list[str]:
    """The wrapped lines for ``text`` at this size.

    ``align`` is accepted and unused: left, center and right all wrap the same
    way and differ only in where each line is placed (:func:`_draw_block`). It
    stays in the signature because the fit search passes it through and a future
    alignment may well wrap differently.

    ``symbol_font``/cmaps enable per-glyph fallback measuring; omitted means
    single-face measuring, exactly as before. ``word_extra`` is the extra
    advance per inter-word gap (:func:`_word_extra`), so wrapping sees the same
    widths the drawing will produce.
    """
    del align
    # The wrap budget backs off by `_WRAP_EPS_PX`: the frontend's preview wraps
    # with the same reduction (see `textFit.WRAP_SAFETY_PX`), so a line-break
    # decision cannot be flipped by the engines' independent advance rounding.
    # The un-reduced width is still honoured afterwards (`_block_size` and the
    # fit checks compare against the full budget), which keeps an exactly-fitting
    # single word from shrinking further.
    max_w = max(1.0, max_w - _WRAP_EPS_PX)
    lines: list[str] = []
    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for para in paragraphs:
        wrapped = _wrap_paragraph(font, para, max_w, symbol_font, primary_cmap,
                                  symbol_cmap, word_extra) or [""]
        lines.extend(wrapped)
    return lines


def _block_size(font: Any, lines: list[str], size: int, spacing: int,
                symbol_font: Any | None = None,
                primary_cmap: set[int] | None = None,
                symbol_cmap: set[int] | None = None,
                word_extra: float = 0.0,
                line_factor: float | None = None
                ) -> tuple[float, float]:
    if not lines:
        return 0.0, 0.0
    width = max((_line_width(font, symbol_font, line, primary_cmap, symbol_cmap,
                             word_extra) for line in lines), default=0.0)
    line_h = _line_height(font, size)
    height = (len(lines) - 1) * _advance_for(line_h, spacing, line_factor) + line_h
    return width, float(height)


def _round_half_up(value: Any) -> int:
    """Round to the nearest integer with halves going *up*, like JS `Math.round`.

    Python's ``round`` is banker's rounding (``round(22.5) == 22``) while the
    preview's JavaScript rounds halves up (`Math.round(22.5) == 23`). Every
    quantity that both engines compute from the same float -- a symbol em
    (``round(size * scale)`), a scaled line advance, a clamped user size --
    would otherwise land on different sides of a `.5` and shift everything
    after it by a pixel. This mirrors the browser half exactly for the
    positive *and* negative quantities in use, and never raises: an
    unreadable value is zero, the same answer ``float(value) or 0`` would
    give.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if math.isnan(number):
        return 0
    if math.isinf(number):
        # JS Math.round(inf) is inf; callers clamp right afterwards, so a
        # huge finite sign-safe value keeps that behaviour without raising.
        return MAX_TEXT_SHIFT if number > 0 else -MAX_TEXT_SHIFT
    return int(math.floor(number + 0.5))


def _spacing_for(size: int) -> int:
    """Leading between lines, proportional to the type size."""
    return max(1, _round_half_up(size * 0.14))


def _advance_for(line_h: int, spacing: int, factor: float | None) -> int:
    """How far the text drops between two lines.

    ``None`` is the untouched entry and returns exactly what the leading logic
    has always produced. A factor scales the whole advance rather than only the
    leading, which is what "line spacing 1.5" means anywhere else -- scaling the
    0.14-of-a-size leading alone would give a range too narrow to be worth a
    control.

    Halves round up: the preview computes the same product in JavaScript, where
    `Math.round` goes the same way (see :func:`_round_half_up`).
    """
    if factor is None:
        return line_h + spacing
    return max(1, _round_half_up((line_h + spacing) * factor))


def _word_extra(font: Any, factor: float | None) -> float:
    """Extra advance per inter-word gap, in pixels; 0 when untouched.

    Expressed against the face's own space advance so the setting means the
    same thing at every type size, and so it survives the fit search changing
    the size underneath it.
    """
    if factor is None or factor == 1.0:
        return 0.0
    try:
        return (float(factor) - 1.0) * _length(font, " ")
    except (TypeError, ValueError):
        return 0.0


def _line_width(font: Any, symbol_font: Any | None, text: str,
                primary_cmap: set[int] | None = None,
                symbol_cmap: set[int] | None = None,
                word_extra: float = 0.0) -> float:
    """Width of one laid-out line, word spacing included.

    Lines are built by joining tokens with a single space, so the gap count is
    just the space count -- and with ``word_extra`` at 0 this is the plain
    measurement it has always been.
    """
    base = _mixed_length(font, symbol_font, text, primary_cmap, symbol_cmap)
    if word_extra and text:
        base += text.count(" ") * word_extra
    return base


def _fit_font_size(fontfile: str, text: str, align: str, max_w: float,
                   max_h: float, lo: int = MIN_FONT_SIZE,
                   hi: int = MAX_FONT_SIZE,
                   geom: TextGeometry | None = None) -> int:
    """The largest size in ``[lo, hi]`` whose block fits the region.

    Binary search, not the descending scan this used to do: the old loop started
    at the region's height in px and tried every integer down from there, which
    on a full-page panel is several hundred complete wrap-and-measure passes for
    one entry. Fitting is monotone in the size -- if a size fits, every smaller
    one does -- so ~7 passes answer the same question.

    Monotonicity can be violated by a hair in a proportional face (a smaller size
    occasionally wraps one word differently and needs an extra line), so the
    result is verified and walked down if the winning size does not actually fit.

    Measuring uses the per-glyph symbol fallback when the text needs it, so a
    line containing a wide symbol is fitted at its real width rather than at
    the primary face's .notdef width.

    ``geom`` folds the per-entry spacing and scale into the search: the block is
    drawn ``scale_x``/``scale_y`` times its measured size, so the space it has
    to fit into shrinks by the same factors, and stretched lettering auto-sizes
    down instead of spilling out of the bubble. The shift and the rotation are
    deliberately *not* here -- they happen after fitting, by definition.
    """
    lo = max(MIN_FONT_SIZE, int(lo))
    hi = max(lo, min(MAX_FONT_SIZE, int(hi)))

    word_factor = geom.word_spacing if geom is not None else None
    line_factor = geom.line_spacing if geom is not None else None
    # Wrapping and fitting both happen in unscaled units, so the budget is
    # divided once here rather than the measurements multiplied everywhere.
    fit_w = max_w / geom.scale_x if geom is not None else max_w
    fit_h = max_h / geom.scale_y if geom is not None else max_h

    def fits(size: int) -> bool:
        font = _load_font(fontfile, size)
        sym, pc, sc = _symbol_faces(fontfile, text, size)
        extra = _word_extra(font, word_factor)
        lines = _layout(font, text, fit_w, align, sym, pc, sc, extra)
        w, h = _block_size(font, lines, size, _spacing_for(size), sym, pc, sc,
                           extra, line_factor)
        return w <= fit_w and h <= fit_h

    best = lo
    a, b = lo, hi
    while a <= b:
        mid = (a + b) // 2
        if fits(mid):
            best = mid
            a = mid + 1
        else:
            b = mid - 1
    while best > lo and not fits(best):
        best -= 1
    return best


# --------------------------------------------------------------------- drawing

def _line_pieces(line: str, font: Any, symbol_font: Any | None,
                 primary_cmap: set[int] | None, symbol_cmap: set[int] | None,
                 word_extra: float, space_w: float
                 ) -> list[tuple[str, Any, float]]:
    """``line`` as ``(char, face, advance)`` pieces, left to right.

    One piece per character, with each inter-word gap an empty piece advancing
    by the space (plus its ``word_extra``). Pillow's own string run would
    advance each glyph by a hinted, per-size-rounded step, while the preview
    walks the unhinted hmtx table -- the granularity here is what lets
    :func:`_draw_block` place every glyph at the cumulative unhinted position
    instead, which is what makes one font size read the same on both sides
    (each glyph's *paste* still snaps to a pixel, but our walk tracks the
    fractional pen exactly like the browser's does). Face selection per
    character is the same run split the measurer uses, so wide symbols/emoji
    still switch to the symbol face mid-line.
    """
    mixed = (symbol_font is not None and primary_cmap is not None
             and symbol_cmap is not None)
    gap = space_w + word_extra
    pieces: list[tuple[str, Any, float]] = []
    for ch in line:
        if ch == " ":
            pieces.append(("", font, gap))
            continue
        face = font
        if mixed:
            try:
                if isinstance(symbol_font, _SymbolCascade):
                    face = symbol_font.faces.get(ch, font)
                elif _needs_symbol(ch, primary_cmap, symbol_cmap):
                    face = symbol_font
            except Exception:  # noqa: BLE001
                face = font
        pieces.append((ch, face, _length(face, ch)))
    return pieces


def _expand_outline_mask(size: tuple[int, int], pixels: bytes, radius: int) -> Image.Image:
    """Circular dilation of clean glyph coverage, including hollow contours."""
    width, height = size
    padded_width, padded_height = width + 2 * radius, height + 2 * radius
    if padded_width * padded_height > _MAX_LAYER_PIXELS:
        raise ValueError("Glyph outline mask is too large")
    source = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)
    source = np.pad(source, radius)
    ink = source > 0
    if not ink.any():
        return Image.new("L", (padded_width, padded_height))
    outside = (~ink).astype(np.uint8)
    # Euclidean distance gives round joins without radius-sized convolutions.
    # Keep the nearest ink's coverage to antialias the expanded boundary, while
    # allowing a faint, subpixel glyph to acquire an opaque thick outline.
    _, labels = cv2.distanceTransformWithLabels(outside, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    coverage_by_label = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    coverage_by_label[labels[ink]] = source[ink].astype(np.float32) / 255
    coverage = np.clip(radius + coverage_by_label[labels] - distance, 0, 1)
    return Image.fromarray(np.rint(coverage * 255).astype(np.uint8))


@lru_cache(maxsize=64)
def _cached_outline_mask(size: tuple[int, int], pixels: bytes, radius: int) -> Image.Image:
    """Read-only masks; keys contain pixels, never retained font objects.

    Only masks up to 64 KiB enter this cache, bounding keys and images to 8 MiB.
    """
    return _expand_outline_mask(size, pixels, radius)


def _draw_outline_text(draw: Any, xy: tuple[float, float], text: str, *, anchor: str, **kwargs: Any) -> None:
    """Draw the outline from unstroked ink; thick FreeType strokes can invert holes."""
    font = kwargs["font"]
    if not isinstance(draw, ImageDraw.ImageDraw) or not isinstance(font, ImageFont.FreeTypeFont):
        # Preserve the draw protocol for custom surfaces and bitmap fonts.
        draw.text(xy, text, anchor=anchor, **kwargs)
        return
    x, y = xy
    # Match ImageDraw.text's fractional raster origin and integer paste exactly.
    core, offset = font.getmask2(text, draw.fontmode, anchor=anchor,
                                 start=(math.modf(x)[0], math.modf(y)[0]))
    if not core.size[0] or not core.size[1]:
        return
    radius = int(kwargs["stroke_width"])
    size = core.size
    pixels = bytes(core)
    expanded_pixels = (size[0] + 2 * radius) * (size[1] + 2 * radius)
    expand = _cached_outline_mask if expanded_pixels <= 65_536 else _expand_outline_mask
    mask = expand(size, pixels, radius)
    draw.bitmap((int(x) + offset[0] - radius, int(y) + offset[1] - radius), mask, fill=kwargs["stroke_fill"])


def _draw_block(draw: ImageDraw.ImageDraw, font: Any, size: int,
                lines: list[str], align: str,
                box: tuple[float, float, float, float],
                fill: tuple[int, int, int, int],
                stroke_width: int,
                stroke_fill: tuple[int, int, int, int] | None,
                symbol_font: Any | None = None,
                primary_cmap: set[int] | None = None,
                symbol_cmap: set[int] | None = None,
                word_extra: float = 0.0,
                line_factor: float | None = None) -> None:
    """Draw ``lines`` centred vertically in ``box``, aligned as asked.

    Every line is drawn on its own, rather than through ``multiline_text``: a
    uniform line advance (see :func:`_line_height`) needs the baseline computed
    here rather than inferred from each line's ink, which would make the leading
    jump wherever a line happens to have no descender.

    Every line is painted through the piece machinery, one character at a time
    at the pen positions the unhinted advance walk produces (see
    :func:`_line_pieces`). Pillow's in-string walk advances by hinted,
    per-size-rounded steps, while the preview's browser walks the raw hmtx
    table -- drawing the string as one call would put the *same font size*
    visibly further along the line than the preview shows. Per-character
    placement tracks the fractional pen exactly as the browser does; the only
    residual is that each glyph's paste snaps to an integer pixel
    (<= 0.5px, never cumulative, invisible next to the ink's own
    antialiasing).

    When ``symbol_font`` and both cmaps are given, a character the primary face
    lacks is drawn from the symbol face at its calibrated size. The reference
    baseline for all glyphs comes from each line's ``y``, which is its
    ascender top (the ``la`` convention), converted once to the baseline via
    the primary face's ascent, and every glyph is drawn ``ls``-anchored there.
    Drawing at the same ascender-top ``y`` instead would align ascender tops,
    and faces with different ascents (ComicMono 32 vs NotoSansSymbols2 43 at
    size 40) would sit their baselines ~11 px apart -- hearts floating or
    sinking next to the letters. When the ascent is unreadable the glyphs
    degrade to the old same-y ``la`` placement rather than guessing. Forced
    replacements match the original character's visible bottom by shifting
    from this baseline using both outline bottoms and the rounded symbol size.

    ``word_extra`` widens (or tightens) every inter-word gap and ``line_factor``
    scales the line advance. Both are inert at their defaults.

    Outlined blocks paint all outlines before any letter fills. Keeping the
    passes separate prevents a thick stroke from covering another character
    or line. Positions are computed once and reused by the fill pass.
    Outlines grow clean raster coverage by Euclidean distance; FreeType's
    native stroke expansion can invert small counters at large stroke widths.
    """
    x0, y0, x1, y1 = box
    max_w = x1 - x0
    spacing = _spacing_for(size)
    line_h = _line_height(font, size)
    advance = _advance_for(line_h, spacing, line_factor)
    block_h = ((len(lines) - 1) * advance + line_h) if lines else 0

    y = y0 + (y1 - y0 - block_h) / 2.0
    use_fallback = symbol_font is not None and primary_cmap is not None and symbol_cmap is not None
    baseline_offsets = symbol_font.baseline_offsets if use_fallback and isinstance(symbol_font, _SymbolCascade) else {}
    outlined = stroke_width > 0 and stroke_fill is not None
    paint_text = partial(_draw_outline_text, draw) if outlined else draw.text
    fill_pieces: list[tuple[float, float, float | None, str, Any]] = []
    space_w = _length(font, " ")
    # Ascender-top -> baseline conversion for mixed lines (see docstring).
    # Read once: it depends on the face/size, not on the line.
    try:
        primary_ascent = _ascent(font)
    except Exception:  # noqa: BLE001
        primary_ascent = None

    for line in lines:
        try:
            pieces = _line_pieces(
                line, font,
                symbol_font if use_fallback else None,
                primary_cmap if use_fallback else None,
                symbol_cmap if use_fallback else None,
                word_extra, space_w)
        except Exception:  # noqa: BLE001
            pieces = [(line, font, _length(font, line))] if line else []
        total = sum(adv for _, _, adv in pieces)
        if align == "center":
            cur_x = x0 + (max_w - total) / 2.0
        elif align == "right":
            cur_x = x1 - total
        else:
            cur_x = x0
        base_y = y + primary_ascent if primary_ascent is not None else None
        for seg, face, adv in pieces:
            if seg:
                glyph_baseline = base_y + baseline_offsets.get(seg, 0.0) if base_y is not None else None
                seg_kw: dict[str, Any] = {"font": face, "fill": stroke_fill if outlined else fill}
                if outlined:
                    seg_kw["stroke_width"] = stroke_width
                    seg_kw["stroke_fill"] = stroke_fill
                    fill_pieces.append((cur_x, y, glyph_baseline, seg, face))
                try:
                    # Baseline-shared drawing (see docstring). When the
                    # primary ascent is unreadable, degrade to the old
                    # same-y ``la`` placement rather than guessing.
                    if base_y is None:
                        paint_text((cur_x, y), seg, anchor="la", **seg_kw)
                    else:
                        try:
                            paint_text((cur_x, glyph_baseline), seg, anchor="ls", **seg_kw)
                        except Exception:
                            # Ancient Pillow without baseline anchors:
                            # visible-but-unshifted beats a gap.
                            paint_text((cur_x, y), seg, anchor="la", **seg_kw)
                except Exception:  # noqa: BLE001 - one bad run, not the page
                    logger.exception("Could not draw a text run")
                    # Keep advancing so the rest of the line stays placed.
            cur_x += adv
        y += advance

    for glyph_x, glyph_top, glyph_baseline, seg, face in fill_pieces:
        try:
            if glyph_baseline is None:
                draw.text((glyph_x, glyph_top), seg, font=face, fill=fill, anchor="la")
            else:
                try:
                    draw.text((glyph_x, glyph_baseline), seg, font=face, fill=fill, anchor="ls")
                except Exception:
                    draw.text((glyph_x, glyph_top), seg, font=face, fill=fill, anchor="la")
        except Exception:  # noqa: BLE001 - one bad glyph must not lose the block
            logger.exception("Could not draw a text fill")


# ------------------------------------------------------------------- per entry

def _clamp(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return int(max(lo, min(hi, _round_half_up(float(value)))))
    except (TypeError, ValueError):
        return default


def _clamp_factor(value: Any, lo: float, hi: float) -> float | None:
    """A multiplier, or None for "leave that logic alone".

    Unset and unreadable collapse to the same answer on purpose: the neutral
    value of these two controls is *the existing code path*, not 1.0, so there
    is nothing to distinguish them into.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(lo, min(hi, number))


def _clamp_scale(value: Any, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(number):
        return 1.0
    return max(lo, min(hi, number))


@dataclass(frozen=True, slots=True)
class TextGeometry:
    """One entry's typesetting geometry, already clamped to this module's ranges.

    The two spacings are multipliers where ``None`` means "use the logic that
    was here before"; everything else is neutral at its default. That is what
    :attr:`transformed` is for: an entry nobody has touched takes the original
    drawing path untouched, so these controls cannot change a page that does
    not use them.

    The spacings and the scales feed the fit search (:func:`_fit_font_size`).
    The shift and the angles are applied afterwards, to the lettering alone.
    """

    word_spacing: float | None = None
    line_spacing: float | None = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    shift_x: int = 0
    shift_y: int = 0
    angle_x: int = 0
    angle_y: int = 0
    angle_z: int = 0

    @property
    def scaled(self) -> bool:
        return self.scale_x != 1.0 or self.scale_y != 1.0

    @property
    def rotated(self) -> bool:
        return bool(self.angle_x or self.angle_y or self.angle_z)

    @property
    def shifted(self) -> bool:
        return bool(self.shift_x or self.shift_y)

    @property
    def transformed(self) -> bool:
        """Whether anything happens *after* the fit -- scale, spin or shift."""
        return self.scaled or self.rotated or self.shifted


#: The "nobody touched this" geometry. A shared instance so the common case
#: allocates nothing per entry.
PLAIN_GEOMETRY = TextGeometry()


def _geometry_of(item: DataItem) -> TextGeometry:
    """``item``'s geometry, clamped. Anything unreadable reads as the default.

    ``getattr`` throughout for the same reason :func:`_clean_jobs` uses it: an
    entry saved before these options existed has none of these attributes, and
    a project saved by a build with wider ranges has to keep rendering rather
    than 422 the page.
    """
    word = _clamp_factor(getattr(item, "word_spacing", None),
                         MIN_WORD_SPACING, MAX_WORD_SPACING)
    line = _clamp_factor(getattr(item, "line_spacing", None),
                         MIN_LINE_SPACING, MAX_LINE_SPACING)
    scale_x = _clamp_scale(getattr(item, "font_scale_x", 1.0),
                           MIN_FONT_SCALE, MAX_FONT_SCALE)
    scale_y = _clamp_scale(getattr(item, "font_scale_y", 1.0),
                           MIN_FONT_SCALE, MAX_FONT_SCALE)
    shift_x = _clamp(getattr(item, "shift_x", 0), -MAX_TEXT_SHIFT, MAX_TEXT_SHIFT, 0)
    shift_y = _clamp(getattr(item, "shift_y", 0), -MAX_TEXT_SHIFT, MAX_TEXT_SHIFT, 0)
    angle_x = _clamp(getattr(item, "angle_x", 0), -MAX_TEXT_ANGLE, MAX_TEXT_ANGLE, 0)
    angle_y = _clamp(getattr(item, "angle_y", 0), -MAX_TEXT_ANGLE, MAX_TEXT_ANGLE, 0)
    angle_z = _clamp(getattr(item, "angle_z", 0), -MAX_TEXT_ANGLE, MAX_TEXT_ANGLE, 0)
    if (word is None and line is None and scale_x == 1.0 and scale_y == 1.0
            and not (shift_x or shift_y or angle_x or angle_y or angle_z)):
        return PLAIN_GEOMETRY
    return TextGeometry(word, line, scale_x, scale_y, shift_x, shift_y,
                        angle_x, angle_y, angle_z)


# ------------------------------------------------------- text-only transforms
# Shift and rotation move the lettering and nothing else, which rules out
# transforming the page: the plate, the cleaned artwork and every other entry
# have to stay exactly where they are. So a transformed entry is drawn onto its
# own transparent tile -- sized to its own block, not to the page -- and that
# tile is transformed and composited back. Everything here is best-effort: a
# transform that cannot be computed returns the layer it was given, and
# `_draw_entry` falls back to drawing in place rather than dropping the entry.

#: Ceiling on a text layer, in pixels. A tile is bounded by the block it holds,
#: so this is only reachable through an absurd size/scale combination -- and
#: there, degrading to untransformed text beats allocating a gigabyte.
_MAX_LAYER_PIXELS = 16_000_000

#: Camera distance for the X/Y tilt, in multiples of the layer's longest side.
#: Far enough that a tilt reads as depth rather than as a funhouse keystone,
#: and -- since it exceeds the half-diagonal at every angle -- far enough that
#: no corner can reach the camera plane.
_TILT_DEPTH = 2.0


class _SymbolCascade:
    """Resolved real Pillow faces; a proxy is never passed to ImageDraw."""

    def __init__(self, faces: dict[str, Any], baseline_offsets: dict[str, float] | None = None):
        self.faces = faces
        self.baseline_offsets = baseline_offsets or {}

    def face_for(self, char: str) -> Any:
        return self.faces[char]


def _valid_glyph_adjustment(value: Any) -> bool:
    """Mirror the preview's validation before using assigned-glyph metrics."""
    if not isinstance(value, dict):
        return False
    scale, primary_bottom, fallback_bottom = (
        value.get(key) for key in ("scale", "primary_bottom", "fallback_bottom")
    )
    return (
        all(type(v) in (int, float) and math.isfinite(v) for v in (scale, primary_bottom, fallback_bottom))
        and 0.5 <= scale <= 3
        and abs(primary_bottom) <= 8
        and abs(fallback_bottom) <= 8
    )


def _load_user_symbol_font(filename: str, size: int) -> Any | None:
    key = (filename, size)
    if key in _USER_SYMBOL_CACHE:
        return _USER_SYMBOL_CACHE[key]
    raw = _REGISTERED_FONT_BYTES.get(filename)
    if not raw:
        return None
    try:
        face = ImageFont.truetype(BytesIO(raw), size)
        _attach_raw_advances(face, raw)
    except Exception:
        logger.debug("Could not load user symbol font %s", filename, exc_info=True)
        if len(_USER_SYMBOL_CACHE) >= _FONT_CACHE_MAX:
            _USER_SYMBOL_CACHE.clear()
        _USER_SYMBOL_CACHE[key] = None
        return None
    if len(_USER_SYMBOL_CACHE) >= _FONT_CACHE_MAX:
        _USER_SYMBOL_CACHE.clear()
    _USER_SYMBOL_CACHE[key] = face
    return face


def _symbol_faces(fontfile: str, text: str, size: int
                  ) -> tuple[Any | None, set[int] | None, set[int] | None]:
    """The symbol fallback face and both cmaps for ``text`` at ``size``.

    ``(None, None, None)`` when the text needs no fallback or a face cannot be
    read; callers then take the single-face path, exactly as before.
    """
    try:
        primary_cmap = _primary_cmap(fontfile)
        if (_USER_SYMBOL_FONTS or _FORCED_SYMBOL_OWNERS) and primary_cmap is not None:
            characters = {ch for ch in text if not ch.isspace()}
            missing = {ch for ch in characters if ord(ch) not in primary_cmap}
            forced = {ch for ch in characters if ch in _FORCED_SYMBOL_OWNERS}
            if not missing and not forced:
                return None, None, None
            primary = _profile_for_font(_effective_fontfile(fontfile))
            resolved: dict[str, Any] = {}
            baseline_offsets: dict[str, float] = {}
            adjustments = _FORCED_SYMBOL_ADJUSTMENTS.get(_effective_fontfile(fontfile), {})
            loaded: dict[tuple[str, int], Any] = {}

            def load_face(filename: str, glyph_scale: float | None = None) -> Any | None:
                if glyph_scale is not None:
                    px = max(1, _round_half_up(size * glyph_scale))
                elif filename == _SYMBOL_FONT:
                    px = _symbol_px_size(fontfile, size)
                else:
                    px = max(1, _round_half_up(size * scale_for(primary, _profile_for_font(filename))))
                key = (filename, px)
                if key not in loaded:
                    if filename == _SYMBOL_FONT:
                        loaded[key] = _load_symbol_font(px)
                    else:
                        loaded[key] = _load_user_symbol_font(filename, px)
                return loaded[key]

            # Explicit ownership precedes the primary face and the hierarchy.
            # Only a successfully loaded, covering face can replace a character.
            for filename in (*_USER_SYMBOL_FONTS, _SYMBOL_FONT):
                assigned = {ch for ch in forced if _FORCED_SYMBOL_OWNERS[ch] == filename}
                if not assigned:
                    continue
                coverage = ((_symbol_cmap() or set()) if filename == _SYMBOL_FONT
                            else _profile_for_font(filename).coverage)
                covered = {ch for ch in assigned if ord(ch) in coverage}
                for char in covered:
                    adjustment = adjustments.get(filename, {}).get(str(ord(char)))
                    if ord(char) not in primary_cmap or not _valid_glyph_adjustment(adjustment):
                        adjustment = None
                    face = load_face(filename, adjustment["scale"] if adjustment else None)
                    if face is not None:
                        resolved[char] = face
                        if adjustment:
                            # Font coordinates grow up; screen coordinates grow
                            # down. Use the actual rounded em, as the SVG does.
                            baseline_offsets[char] = (
                                adjustment["fallback_bottom"] * face.size - adjustment["primary_bottom"] * size
                            )
                        missing.discard(char)
            for filename in _USER_SYMBOL_FONTS:
                profile = _profile_for_font(filename)
                covered = {ch for ch in missing if ord(ch) in profile.coverage}
                if not covered:
                    continue
                face = load_face(filename)
                if face is not None:
                    resolved.update(dict.fromkeys(covered, face))
                    missing -= covered
            # The embedded Noto face is always last, regardless of user order.
            if missing:
                noto_cmap = _symbol_cmap() or set()
                covered = {ch for ch in missing if ord(ch) in noto_cmap}
                if covered:
                    face = load_face(_SYMBOL_FONT)
                    if face is not None:
                        resolved.update(dict.fromkeys(covered, face))
            if resolved:
                return _SymbolCascade(resolved, baseline_offsets), primary_cmap, {ord(ch) for ch in resolved}
            return None, None, None
        symbol_cmap = _symbol_cmap()
        symbol_font = None
        if _text_needs_symbol(text, primary_cmap, symbol_cmap):
            symbol_font = _load_symbol_font(_symbol_px_size(fontfile, size))
        if symbol_font is None:
            return None, None, None
        return symbol_font, primary_cmap, symbol_cmap
    except Exception:  # noqa: BLE001 - fallback lookup is never fatal
        logger.debug("Symbol fallback lookup failed; using primary face alone.",
                     exc_info=True)
        return None, None, None


def _block_rect(box: tuple[float, float, float, float], align: str,
                block_w: float, block_h: float
                ) -> tuple[float, float, float, float]:
    """Where a ``block_w`` x ``block_h`` block lands inside ``box``.

    Mirrors :func:`_draw_block`'s placement -- centred vertically, and against
    whichever edge the alignment anchors to. Callers pass the *scaled* size, so
    stretched lettering grows away from its own anchor instead of drifting out
    of the bubble on the anchored side.
    """
    x0, y0, x1, y1 = box
    if align == "center":
        rx0 = x0 + (x1 - x0 - block_w) / 2.0
    elif align == "right":
        rx0 = x1 - block_w
    else:  # left
        rx0 = float(x0)
    ry0 = y0 + (y1 - y0 - block_h) / 2.0
    return rx0, ry0, rx0 + block_w, ry0 + block_h


def _clamp_shift(delta: int, lo: float, hi: float, limit: int) -> int:
    """``delta`` trimmed so ``[lo, hi]`` ends up inside ``[0, limit]``.

    The bounds are rounded inwards and the result is a whole number of pixels,
    so the answer is always strictly inside rather than half a pixel over.

    Zero when the span is wider than the page: no offset keeps it inside, and
    refusing to move is more predictable than silently pinning it to an edge.
    """
    low = math.ceil(-lo)
    high = math.floor(limit - hi)
    if low > high:
        return 0
    return int(max(low, min(high, int(delta))))


def _text_layer(fontfile: str, text: str, size: int, lines: list[str],
                align: str, fill: tuple[int, int, int, int],
                stroke_width: int,
                stroke_fill: tuple[int, int, int, int] | None,
                geom: TextGeometry) -> Image.Image | None:
    """The block alone on a transparent tile, at the asked-for X/Y scale.

    Set at ``max(scale_x, scale_y)`` times the type size and then resampled to
    the target aspect, so both axes are *down*-sampled: enlarging a rasterised
    block is what would make stretched lettering soft, and at an equal X and Y
    scale there is no resampling step at all.

    The block is centred in the tile, which is what lets every later transform
    work about the tile's own centre and lets the caller line the result up by
    matching centres.

    None when the tile cannot be built -- a degenerate block, or one too large
    to be worth the memory. The caller then draws in place instead.
    """
    k = max(geom.scale_x, geom.scale_y)
    ksize = max(1, int(round(size * k)))
    kfont = _load_font(fontfile, ksize)
    ksym, kpc, ksc = _symbol_faces(fontfile, text, ksize)
    kextra = _word_extra(kfont, geom.word_spacing)
    block_w, block_h = _block_size(kfont, lines, ksize, _spacing_for(ksize),
                                   ksym, kpc, ksc, kextra, geom.line_spacing)
    if block_w <= 0 or block_h <= 0:
        return None

    kstroke = max(0, int(round(stroke_width * k)))
    # Advance widths are not ink widths: swashes, accents, italics and the
    # outline all reach past the measured box, so the tile carries a margin
    # rather than shaving the edges off the lettering.
    margin = kstroke + max(4, int(round(ksize * 0.5)))
    lw = int(math.ceil(block_w)) + 2 * margin
    lh = int(math.ceil(block_h)) + 2 * margin
    if lw < 1 or lh < 1 or lw * lh > _MAX_LAYER_PIXELS:
        logger.debug("Text layer %dx%d is too large; drawing it in place.", lw, lh)
        return None

    layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    _draw_block(ImageDraw.Draw(layer), kfont, ksize, lines, align,
                (margin, margin, margin + block_w, margin + block_h),
                fill, kstroke, stroke_fill, ksym, kpc, ksc,
                kextra, geom.line_spacing)

    tw = max(1, int(round(lw * geom.scale_x / k)))
    th = max(1, int(round(lh * geom.scale_y / k)))
    if (tw, th) != (lw, lh):
        layer = layer.resize((tw, th), Image.Resampling.LANCZOS)
    return layer


def _perspective_coeffs(dst: list[tuple[float, float]],
                        src: list[tuple[float, float]]
                        ) -> tuple[float, ...] | None:
    """The eight coefficients mapping ``dst`` back to ``src``.

    Pillow's ``PERSPECTIVE`` transform samples the *source* for each output
    pixel, so the matrix it wants runs the opposite way to the projection --
    hence the argument order. None when the system is singular (a quad that
    has collapsed to a line or a point).
    """
    rows: list[list[float]] = []
    for (dx, dy), (sx, sy) in zip(dst, src):
        rows.append([dx, dy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dy * sx])
        rows.append([0.0, 0.0, 0.0, dx, dy, 1.0, -dx * sy, -dy * sy])
    try:
        solved = np.linalg.solve(
            np.array(rows, dtype=np.float64),
            np.array([v for point in src for v in point], dtype=np.float64),
        )
    except np.linalg.LinAlgError:
        return None
    except Exception:  # noqa: BLE001 - a bad solve is not a failed page
        logger.debug("Could not solve a perspective transform.", exc_info=True)
        return None
    if not np.all(np.isfinite(solved)):
        return None
    return tuple(float(v) for v in solved)


def _tilt(layer: Image.Image, angle_x: int, angle_y: int) -> Image.Image:
    """``layer`` as a plane turned about its own horizontal/vertical axis.

    A pinhole projection at :data:`_TILT_DEPTH` layer-widths. The output is
    padded symmetrically about the plane's centre rather than cropped to the
    projected quad, so the centre of the lettering stays the centre of the
    tile and the caller's "match the centres" placement keeps holding.

    Returns ``layer`` unchanged when the tilt degenerates -- edge-on at +/-90,
    or a quad too thin to sample. A sliver of a pixel is not a useful result,
    and leaving the text flat is the legible failure.
    """
    w, h = layer.size
    rx = math.radians(angle_x)
    ry = math.radians(angle_y)
    cos_x, sin_x = math.cos(rx), math.sin(rx)
    cos_y, sin_y = math.cos(ry), math.sin(ry)
    depth = max(w, h) * _TILT_DEPTH

    projected: list[tuple[float, float]] = []
    for px, py in ((-w / 2.0, -h / 2.0), (w / 2.0, -h / 2.0),
                   (w / 2.0, h / 2.0), (-w / 2.0, h / 2.0)):
        # Flat to begin with (z = 0): turn about X, then about Y, then project.
        ty = py * cos_x
        tz = py * sin_x
        tx = px * cos_y + tz * sin_y
        tz = tz * cos_y - px * sin_y
        denom = depth + tz
        if denom <= 0.0:
            return layer
        scale = depth / denom
        projected.append((tx * scale, ty * scale))

    half_w = max(abs(x) for x, _ in projected)
    half_h = max(abs(y) for _, y in projected)
    out_w = int(math.ceil(half_w * 2.0))
    out_h = int(math.ceil(half_h * 2.0))
    if out_w < 1 or out_h < 1 or out_w * out_h > _MAX_LAYER_PIXELS:
        return layer

    dst = [(x + out_w / 2.0, y + out_h / 2.0) for x, y in projected]
    coeffs = _perspective_coeffs(dst, [(0.0, 0.0), (float(w), 0.0),
                                       (float(w), float(h)), (0.0, float(h))])
    if coeffs is None:
        return layer
    try:
        return layer.transform((out_w, out_h), Image.Transform.PERSPECTIVE,
                               coeffs, resample=Image.Resampling.BICUBIC)
    except (ValueError, OSError, MemoryError):
        logger.debug("Could not tilt a text layer.", exc_info=True)
        return layer


def _spin(layer: Image.Image, angle_z: int) -> Image.Image:
    """``layer`` turned in its own plane, expanding to keep the corners.

    Positive is clockwise, matching the frontend's SVG preview; Pillow's
    ``rotate`` is counter-clockwise, hence the negation. ``expand`` keeps the
    rotation centred on the layer's centre, which is the point the caller
    lines up against.
    """
    w, h = layer.size
    rad = math.radians(angle_z)
    cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
    if (w * cos_a + h * sin_a) * (w * sin_a + h * cos_a) > _MAX_LAYER_PIXELS:
        return layer
    try:
        return layer.rotate(-angle_z, resample=Image.Resampling.BICUBIC,
                            expand=True)
    except (ValueError, OSError, MemoryError):
        logger.debug("Could not rotate a text layer.", exc_info=True)
        return layer


def _composite(img: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    """Blend ``layer`` onto ``img`` at ``(x, y)``, cropped to the page.

    Cropping first because ``alpha_composite`` rejects an overhanging box, and
    lettering rotated near an edge legitimately overhangs even after the shift
    clamp -- the clamp keeps the *ink* inside, not the tile's transparent
    corners.
    """
    lw, lh = layer.size
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.width, x + lw), min(img.height, y + lh)
    if x1 <= x0 or y1 <= y0:
        return
    if (x0, y0, x1, y1) != (x, y, x + lw, y + lh):
        layer = layer.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    if img.mode == "RGBA":
        img.alpha_composite(layer, (x0, y0))
    else:
        img.paste(layer, (x0, y0), layer)


def _points_of(item: DataItem) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for point in item.points or ():
        try:
            x, y = point[0], point[1]
        except (TypeError, IndexError, KeyError):
            continue
        try:
            out.append((int(round(float(x))), int(round(float(y)))))
        except (TypeError, ValueError):
            continue
    return out


def _ordered(dataitems: list[DataItem]) -> list[DataItem]:
    """Bottom layer first, original order within a layer.

    A stable sort is the point: two entries on the same layer keep the order the
    user sees in the list, so the result does not depend on dictionary ordering
    somewhere upstream.
    """
    return sorted(
        dataitems,
        key=lambda item: _clamp(getattr(item, "layer", 1), 1, MAX_LAYER, 1),
    )


def _clean_jobs(dataitems: list[DataItem]) -> list[Any]:
    """The clean-mode entries, as `CleanJob`s. Imports late; see `_apply_clean`.

    `getattr` throughout because `item.clean` is optional: an entry saved before
    text clean existed has no options object at all, and every field falls back to
    the package default rather than to a literal spelled out here. `CleanJob`
    normalises whatever comes through, so a stale preset name is harmless.
    """
    from fox_reader.clean import (
        DEFAULT_CLEAN_METHOD,
        DEFAULT_INPAINT_METHOD,
        DEFAULT_SPEED,
        DEFAULT_TILE,
        DEFAULT_TTA,
    )
    from fox_reader.services.clean_service import CleanJob

    jobs = []
    for item in dataitems:
        if getattr(item, "bg_mode", "auto") != "clean":
            continue
        points = _points_of(item)
        if len(points) < 3:
            continue
        opts = getattr(item, "clean", None)
        jobs.append(CleanJob(
            polygon=points,
            method=getattr(opts, "method", DEFAULT_CLEAN_METHOD) or DEFAULT_CLEAN_METHOD,
            fill=getattr(opts, "fill", DEFAULT_INPAINT_METHOD) or DEFAULT_INPAINT_METHOD,
            glow=bool(getattr(opts, "glow", True)),
            transport=bool(getattr(opts, "transport", True)),
            speed=getattr(opts, "speed", DEFAULT_SPEED) or DEFAULT_SPEED,
            tta=bool(getattr(opts, "tta", DEFAULT_TTA)),
            # `or` would turn a deliberate 0 into the default -- which is also 0
            # today, but the fallback has to be the one for a *missing* field.
            tile=getattr(opts, "tile", DEFAULT_TILE),
        ))
    return jobs


def _apply_clean(img: Image.Image, jobs: list[Any], clean_service: Any,
                 task: Any | None, plan: Any | None) -> tuple[Image.Image, Any]:
    """Repaint the artwork under every clean-mode entry on this frame."""
    rgba = np.array(img.convert("RGBA"))
    bgr = np.ascontiguousarray(rgba[:, :, 2::-1])
    cleaned, plan = clean_service.clean(bgr, jobs, task=task, plan=plan)
    if cleaned is not bgr:
        rgba[:, :, :3] = cleaned[:, :, ::-1]
        img = Image.fromarray(rgba, "RGBA")
    return img, plan


def _draw_entry(img: Image.Image, draw: ImageDraw.ImageDraw,
                item: DataItem) -> None:
    points = _points_of(item)
    if len(points) < 3:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = max(0, min(xs)), min(img.width, max(xs))
    min_y, max_y = max(0, min(ys)), min(img.height, max(ys))
    if max_x <= min_x or max_y <= min_y:
        return

    # 1. What the region looks like now -- needed for every "auto" below, and
    #    read before the background is painted over it. The OCR step may already
    #    have measured it (`fox_reader.ocr_colors`), off the same pixels and with
    #    the lettering excluded, which is both a better estimate than the raw
    #    mode and one crop-and-count per entry saved.
    region_bg = parse_color(getattr(item, "detected_bg", None), None)
    if region_bg is None or region_bg[3] == 0:
        region_bg = _dominant_color(img, (min_x, min_y, max_x, max_y))

    bg_mode = (getattr(item, "bg_mode", "auto") or "auto").strip().lower()
    bg_color: tuple[int, int, int, int] | None
    if bg_mode == "color":
        bg_color = parse_color(getattr(item, "bg_color", None), region_bg)
    elif bg_mode in ("transparent", "clean"):
        bg_color = None
    else:  # auto
        bg_color = region_bg

    if bg_color is not None and bg_color[3] > 0:
        draw.polygon(points, fill=bg_color)

    text = item.text or ""
    if not text.strip():
        # A clean-only or plate-only entry: the region work is done above and
        # there is deliberately nothing to set.
        return

    # 2. Type area. 12% horizontal padding keeps the text inside the curve of an
    #    oval bubble rather than against its bounding box.
    box_w = max_x - min_x
    box_h = max_y - min_y
    pad_x = max(5, int(box_w * 0.12))
    pad_y = max(4, int(box_h * 0.10))
    max_text_w = max(1, box_w - 2 * pad_x)
    max_text_h = max(1, box_h - 2 * pad_y)

    align = (item.text_align or "left").strip().lower()
    if align not in TEXT_ALIGNMENTS:
        align = "left"

    geom = _geometry_of(item)

    # 3. Size: given, or the largest that fits. The spacings and the scales are
    #    part of that question -- stretched or widely-spaced lettering has to
    #    auto-size down rather than spill out of the bubble -- so the fit is
    #    told about them. Wrapping then happens in unscaled units against the
    #    same divided budget the fit used.
    fontfile = item.fontfile or _FALLBACK_FONT
    if getattr(item, "font_size", None):
        size = _clamp(item.font_size, MIN_FONT_SIZE, MAX_FONT_SIZE, 16)
    else:
        size = _fit_font_size(fontfile, text, align, max_text_w, max_text_h,
                              geom=geom)
    font = _load_font(fontfile, size)
    # Per-glyph symbol fallback, same size: only for characters the primary
    # face lacks that the symbol face has. Everything is best-effort -- any
    # failure degrades to the old single-face path rather than failing the
    # entry.
    symbol_font, primary_cmap, symbol_cmap = _symbol_faces(fontfile, text, size)
    word_extra = _word_extra(font, geom.word_spacing)
    wrap_w = max_text_w / geom.scale_x if geom.scaled else max_text_w
    lines = _layout(font, text, wrap_w, align, symbol_font, primary_cmap,
                    symbol_cmap, word_extra)

    # 4. Colours, in three tiers: the user's own pick, then what the OCR step
    #    measured off the artwork, then the fallback that works from the plate
    #    alone -- the text takes whichever of black or white it contrasts with,
    #    and the outline takes the plate's own colour, which is what keeps
    #    lettering legible where it crosses the bubble border.
    #
    #    The font hint has to survive the plate changing under it, so it is
    #    dropped when it would not read there (`_legible_on`). The outline hint
    #    needs no such guard: it is only ever sent when the lettering really has
    #    an outline, and an outline's whole job is to separate the glyphs from
    #    whatever is behind them.
    plate = bg_color if bg_color is not None and bg_color[3] > 0 else region_bg
    auto_fill = _legible_on(
        parse_color(getattr(item, "detected_font_color", None), None), plate
    ) or _contrast_color(plate)
    fill = parse_color(getattr(item, "font_color", None), auto_fill)
    auto_stroke = parse_color(getattr(item, "detected_stroke_color", None), plate)
    stroke_fill = parse_color(getattr(item, "stroke_color", None), auto_stroke)
    raw_width = getattr(item, "stroke_width", None)
    if raw_width is None:
        # Auto: proportional to the type size, so a 60 px title is not outlined
        # like 8 px furigana. Clamped to the range the control offers.
        stroke_width = _clamp(round(size * 0.09), MIN_STROKE, MAX_STROKE, 2)
        stroke_width = max(1, stroke_width)
    else:
        stroke_width = _clamp(raw_width, MIN_STROKE, MAX_STROKE, 2)

    text_box = (min_x + pad_x, min_y + pad_y,
                min_x + pad_x + max_text_w, min_y + pad_y + max_text_h)
    fill = fill or BLACK

    def _in_place(box: tuple[float, float, float, float]) -> None:
        _draw_block(draw, font, size, lines, align, box,
                    fill, stroke_width, stroke_fill,
                    symbol_font, primary_cmap, symbol_cmap,
                    word_extra, geom.line_spacing)

    # 5. Placement. Nothing after the fit -- the overwhelmingly common case, and
    #    every entry written before these options existed -- draws exactly as it
    #    always has, straight onto the page.
    if not geom.transformed:
        _in_place(text_box)
        return

    block_w, block_h = _block_size(font, lines, size, _spacing_for(size),
                                   symbol_font, primary_cmap, symbol_cmap,
                                   word_extra, geom.line_spacing)

    if not (geom.scaled or geom.rotated):
        # Shift alone: moving the type area moves the block by the same amount,
        # so this stays a direct draw -- no tile, no resampling, and the text
        # is as sharp as it is unshifted.
        rx0, ry0, rx1, ry1 = _block_rect(text_box, align, block_w, block_h)
        pad = float(stroke_width)
        dx = _clamp_shift(geom.shift_x, rx0 - pad, rx1 + pad, img.width)
        dy = _clamp_shift(geom.shift_y, ry0 - pad, ry1 + pad, img.height)
        _in_place((text_box[0] + dx, text_box[1] + dy,
                   text_box[2] + dx, text_box[3] + dy))
        return

    # Scaled and/or rotated: the lettering goes onto its own tile so the plate,
    # the cleaned artwork and every other entry stay where they are.
    layer = _text_layer(fontfile, text, size, lines, align, fill,
                        stroke_width, stroke_fill, geom)
    if layer is None:
        _in_place(text_box)
        return

    if geom.angle_x or geom.angle_y:
        layer = _tilt(layer, geom.angle_x, geom.angle_y)
    if geom.angle_z:
        layer = _spin(layer, geom.angle_z)

    ink = layer.getbbox()
    if ink is None:  # every transform landed the lettering outside its own tile
        return

    # The block is centred in its tile and every transform above preserves that
    # centre, so matching centres is what puts the lettering back where it was
    # laid out. The shift is then clamped against the ink itself rather than the
    # tile, whose corners are transparent once the text has been turned.
    rx0, ry0, rx1, ry1 = _block_rect(text_box, align,
                                     block_w * geom.scale_x,
                                     block_h * geom.scale_y)
    origin_x = (rx0 + rx1) / 2.0 - layer.width / 2.0
    origin_y = (ry0 + ry1) / 2.0 - layer.height / 2.0
    dx = _clamp_shift(geom.shift_x, origin_x + ink[0], origin_x + ink[2], img.width)
    dy = _clamp_shift(geom.shift_y, origin_y + ink[1], origin_y + ink[3], img.height)
    _composite(img, layer, int(round(origin_x)) + dx, int(round(origin_y)) + dy)


# ---------------------------------------------------------------------- public

def process_typesetting(img_path: str, save_path: str,
                        dataitems: list[DataItem], *,
                        clean_service: Any | None = None,
                        task: Any | None = None) -> None:
    """Typeset ``dataitems`` onto ``img_path`` and write ``save_path``.

    ``task`` is an optional
    :class:`~fox_reader.services.progress.TaskProgress` -- the caller runs this in
    a worker thread, and a page with a Text Seg clean on it takes long enough
    that the UI has to be told what is happening.
    """
    items = _ordered([item for item in dataitems or [] if item is not None])
    jobs = _clean_jobs(items) if clean_service is not None else []

    with Image.open(img_path) as src_img:
        is_animated = bool(getattr(src_img, "is_animated", False))
        n_frames = int(getattr(src_img, "n_frames", 1) or 1)
        original_idx = src_img.tell()

        if task is not None:
            task.plan(max(1, n_frames))

        processed_frames: list[Image.Image] = []
        plan = None

        for frame_idx in range(n_frames):
            try:
                src_img.seek(frame_idx)
            except (EOFError, OSError):
                break

            img = src_img.convert("RGBA")

            if jobs:
                # Detection is reused across frames: the lettering is drawn over
                # every frame in the same place, so segmenting each one would
                # multiply a 40 s pass by the frame count for an identical mask.
                if task is not None and n_frames > 1:
                    task.say(f"Text clean: frame {frame_idx + 1}/{n_frames}")
                try:
                    img, plan = _apply_clean(img, jobs, clean_service, task, plan)
                except Exception as exc:  # noqa: BLE001 - keep the typesetting
                    logger.exception("Text clean failed")
                    if task is not None:
                        task.say(f"Text clean failed: {exc}")

            draw = ImageDraw.Draw(img)
            for item in items:
                try:
                    _draw_entry(img, draw, item)
                except Exception:  # noqa: BLE001 - one bad entry, not the page
                    logger.exception("Could not typeset an entry")

            if is_animated:
                img.info["duration"] = src_img.info.get("duration", 100)
                img.info["loop"] = src_img.info.get("loop", 0)

            processed_frames.append(img)
            if task is not None:
                task.advance(done=frame_idx + 1, total=n_frames)

        try:
            src_img.seek(original_idx)
        except (EOFError, OSError):
            pass

        if not processed_frames:
            raise ValueError(f"no frames could be read from {img_path}")

        if task is not None:
            task.say("Saving")
        save_img(processed_frames, save_path, src_img)

    if plan is not None and getattr(plan, "notes", None) and task is not None:
        for note in plan.notes:
            task.say(note)


def save_img(image_data: list | Image.Image, file_path: str,
             src_img_context: Image.Image | None = None) -> None:
    img_map = {
        "png": "PNG",
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "gif": "GIF",
        "webp": "WEBP",
        "bmp": "BMP",
        "avif": "AVIF",
    }
    filename: str = os.path.basename(file_path)
    ext: str = os.path.splitext(filename)[-1].strip(".").lower()
    file_format = img_map.get(ext, "PNG")

    # 1. Standardize input (If it's a single image, turn it into a 1-item list)
    if not isinstance(image_data, list):
        frames = [image_data]
    else:
        frames = list(image_data)

    # 2. Enforce color space profiles cleanly on every frame in the list
    if ext in ["jpg", "jpeg", "bmp"]:
        for i in range(len(frames)):
            if frames[i].mode != "RGB":
                frames[i] = frames[i].convert("RGB")

    # 3. Base Engine Arguments
    save_args: dict[str, Any] = {"format": file_format}

    # 4. Centralized Quality Settings
    if ext in ["jpg", "jpeg"]:
        save_args.update(
            {"quality": 100, "subsampling": 0, "qtables": "web_high"}
        )
    elif ext == "png":
        save_args.update({"compress_level": 1})
    elif ext in ["webp", "gif"]:
        save_args.update({"lossless": True, "quality": 100, "exact": True})
    elif ext == "avif":
        save_args.update({"speed": 0, "quality": 100})

    # 5. Native Save Routing Decision
    if len(frames) > 1:
        # Extract animation loop timing directly from original context if provided
        loop_val = src_img_context.info.get("loop", 0) if src_img_context else 0
        duration_val = (
            src_img_context.info.get("duration", 100) if src_img_context else 100
        )

        save_args.update(
            {
                "save_all": True,
                "append_images": frames[1:],
                "loop": loop_val,
                "duration": duration_val,
                "disposal": 2,
                "optimize": False,
            }
        )
        frames[0].save(file_path, **save_args)
    else:
        frames[0].save(file_path, **save_args)
