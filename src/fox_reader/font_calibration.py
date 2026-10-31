"""Coverage-aware, bounded optical calibration shared by both renderers.

Measure real outlines in font units, never a missing-glyph box or a raster at
one arbitrary size. A profile is computed once per content hash; pairing faces
then costs only a division. The browser receives those same ratios.
"""

from __future__ import annotations

import hashlib
import logging
import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from statistics import median

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

logger = logging.getLogger(__name__)
SCALE_LIMITS = (0.5, 3.0)
PROFILE_SCHEMA = 2
MAX_SAMPLES = 48
_CACHE_MAX = 128
_PROFILES: dict[str, FontProfile] = {}
_CAPS = "HEMNITXABDFKLOPRSUVYZ"
_SYMBOLS = "\u2661\u2605\u2606\u273f\u2764\u25b2\u25a0\u2731\u2763\u2713"


@dataclass(frozen=True)
class FontProfile:
    coverage: frozenset[int]
    cap_height: float = 0.0
    symbol_height: float = 0.0
    symbol_samples: int = 0


def _robust_height(values: list[float]) -> float:
    """Median with MAD rejection; tiny marks and huge logos cannot dominate."""
    values = [v for v in values if math.isfinite(v) and v > 0]
    if not values:
        return 0.0
    middle = median(values)
    deviation = median(abs(v - middle) for v in values)
    tolerance = max(3 * 1.4826 * deviation, middle * 0.2)
    inliers = [v for v in values if abs(v - middle) <= tolerance]
    return float(median(inliers)) if inliers else float(middle)


def _sample(items: list[int]) -> list[int]:
    """Deterministic samples spread over the charset, including both ends."""
    if len(items) <= MAX_SAMPLES:
        return items
    return [items[i * (len(items) - 1) // (MAX_SAMPLES - 1)] for i in range(MAX_SAMPLES)]


def profile_for(raw: bytes) -> FontProfile:
    signature = hashlib.sha256(raw).hexdigest()
    cached = _PROFILES.get(signature)
    if cached is not None:
        return cached
    profile = FontProfile(frozenset())
    try:
        with TTFont(BytesIO(raw), lazy=True) as font:
            units = float(font["head"].unitsPerEm)
            if not math.isfinite(units) or units <= 0:
                raise ValueError("Invalid unitsPerEm")
            glyphs = font.getGlyphSet()
            cmap = font.getBestCmap() or {}
            heights: dict[str, float] = {}
            usable: dict[int, str] = {}
            for cp, name in sorted(cmap.items()):
                if not 0 <= cp <= 0x10FFFF or name in (".notdef", ".null"):
                    continue
                if name not in heights:
                    try:
                        pen = BoundsPen(glyphs)
                        glyphs[name].draw(pen)
                        bounds = pen.bounds
                        height = (bounds[3] - bounds[1]) / units if bounds else 0.0
                        # Bound pathological metrics without assuming every font
                        # fits in its em. Zero-ink glyphs cannot fill a symbol gap.
                        heights[name] = height if math.isfinite(height) and 0 < height <= 8 else 0.0
                    except Exception:  # one corrupt outline does not lose a face
                        heights[name] = 0.0
                if heights[name] > 0 or chr(cp).isspace():
                    usable[cp] = name

            def measure(cps: list[int]) -> list[float]:
                names = dict.fromkeys(usable[cp] for cp in cps if cp in usable)
                return [heights[name] for name in names if heights[name] > 0]

            cap = _robust_height(measure([ord(ch) for ch in _CAPS]))
            if not cap:
                # Non-Latin text faces: sample actual letters before consulting
                # optional OS/2 hints; whitespace/control/marks are never probes.
                letters = [cp for cp in usable if unicodedata.category(chr(cp)).startswith("L")]
                cap = _robust_height(measure(_sample(letters)))
            if not cap and "OS/2" in font:
                hinted = float(getattr(font["OS/2"], "sCapHeight", 0)) / units
                cap = hinted if math.isfinite(hinted) and 0 < hinted <= 4 else 0.0

            samples = measure([ord(ch) for ch in _SYMBOLS])
            if len(samples) < 5:
                candidates = [
                    cp
                    for cp in usable
                    if unicodedata.category(chr(cp)).startswith("S") or unicodedata.category(chr(cp)) == "Co"
                ]
                samples = measure(_sample(candidates))
            if len(samples) < 5:
                # Icon fonts can map artwork to ASCII letters. Do not measure
                # their absent Unicode hearts as missing-character boxes.
                candidates = [
                    cp
                    for cp in usable
                    if unicodedata.category(chr(cp))[0] in "LSN" or unicodedata.category(chr(cp)) == "Co"
                ]
                extra = measure(_sample(candidates))
                if extra:
                    samples = extra
            if not samples:
                samples = measure(_sample([cp for cp in usable if not chr(cp).isspace()]))
            profile = FontProfile(frozenset(usable), cap, _robust_height(samples), len(samples))
    except Exception:
        logger.debug("Could not calibrate font outlines", exc_info=True)
    if len(_PROFILES) >= _CACHE_MAX:
        _PROFILES.clear()
    _PROFILES[signature] = profile
    return profile


def scale_for(primary: FontProfile, fallback: FontProfile) -> float:
    if primary.cap_height <= 0 or fallback.symbol_height <= 0:
        return 1.0
    ratio = primary.cap_height / fallback.symbol_height
    return min(SCALE_LIMITS[1], max(SCALE_LIMITS[0], ratio)) if math.isfinite(ratio) else 1.0


@lru_cache(maxsize=256)
def glyph_metrics_for(raw: bytes, characters: str) -> dict[int, tuple[float, float]]:
    """Normalised outline height and bottom for a bounded assignment set."""
    result = {}
    try:
        with TTFont(BytesIO(raw), lazy=True) as font:
            units = float(font["head"].unitsPerEm)
            if not math.isfinite(units) or units <= 0:
                return result
            glyphs = font.getGlyphSet()
            cmap = font.getBestCmap() or {}
            measured = {}
            for character in characters:
                codepoint = ord(character)
                name = cmap.get(codepoint)
                if not name or name in (".notdef", ".null"):
                    continue
                if name not in measured:
                    measured[name] = None
                    try:
                        pen = BoundsPen(glyphs)
                        glyphs[name].draw(pen)
                        if pen.bounds:
                            bottom, top = pen.bounds[1] / units, pen.bounds[3] / units
                            height = top - bottom
                            valid = all(math.isfinite(value) and abs(value) <= 8 for value in (bottom, top, height))
                            if height > 0 and valid:
                                measured[name] = (height, bottom)
                    except Exception:
                        logger.debug("Could not measure assigned glyph %s", name, exc_info=True)
                if measured[name] is not None:
                    result[codepoint] = measured[name]
    except Exception:
        logger.debug("Could not measure assigned characters", exc_info=True)
    return result


def replacement_metrics_for(primary_raw: bytes, fallback_raw: bytes, characters: str) -> dict[str, dict[str, float]]:
    """Calibrate assigned glyph heights and visible-bottom alignment.

    Both renderers offset the reference baseline using these normalised bounds
    and the same rounded symbol size; no per-size outline parsing is needed.
    """
    characters = "".join(sorted(set(characters)))
    primary = glyph_metrics_for(primary_raw, characters)
    fallback = glyph_metrics_for(fallback_raw, characters)
    return {
        str(codepoint): {
            "scale": min(SCALE_LIMITS[1], max(SCALE_LIMITS[0], primary[codepoint][0] / fallback[codepoint][0])),
            "primary_bottom": primary[codepoint][1],
            "fallback_bottom": fallback[codepoint][1],
        }
        for codepoint in primary.keys() & fallback.keys()
    }
