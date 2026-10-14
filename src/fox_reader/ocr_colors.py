"""What colour the lettering in a region actually is.

The typesetter has always had an ``auto`` for every colour, and
:func:`fox_reader.typeset._draw_entry` resolves it the only way it can from a
finished page: the plate is the region's dominant colour, the text is whichever
of black or white contrasts with it, and the outline is the plate again. That is
right for black-on-white body text and wrong for everything else -- white SFX
lettering with a black outline comes back black-on-artwork. The browser's live
preview was worse still: it had no pixels at all, so it drew a white plate and
black text for every ``auto`` entry (see the note in
``frontend/src/entries/shapes.ts``).

This module reads the three colours off the crop the OCR step already has in
hand, so both sides can stop guessing. The result travels as a *hint*: the user's
own picks still win, and a hint that is missing or unreadable falls straight
through to the detection that was there before.

How it reads them
-----------------

1. **The plate.** The modal colour -- quantised to 5 bits a channel, then the
   median of the winning bin, so the answer is a colour that is really in the
   crop rather than a bin centre. Sampled *outside* the detector's text quads
   when they are known, which is the one sample that cannot be polluted by the
   lettering itself.

2. **The glyphs.** Lab distance from the plate, split by Otsu over a floor: Otsu
   adapts to how far the ink really sits from the paper, and the floor stops it
   inventing a split in a region that holds no text at all. Implausible outcomes
   (almost nothing, or most of the region) are reported as "no idea" rather than
   as a colour, because a wrong hint is worse than none.

3. **Ink and outline.** A glyph with an outline is a core inside a ring, so the
   distance transform of the glyph mask separates them: the interior is the ink,
   the outermost pixels are the ring. The ring is only an *outline* if its colour
   is not simply ink and paper mixed by anti-aliasing -- which is to say, if it
   does not lie on the segment joining them. (The same reading of a text plate's
   palette as :func:`fox_reader.clean.mask._palette_dist`, which measures to the
   segments between palette centres for exactly this reason.) When the ring is
   just anti-aliasing, no outline colour is reported at all and the renderers
   keep their old rule of outlining in the plate's own colour. A reported outline
   therefore always means "there is a real outline, and it is this colour".

Every entry point is best-effort: nothing here raises, because an OCR result must
not be lost to a colour that could not be measured.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Colour statistics want pixels, not resolution. A speech bubble is well under
#: this and is measured untouched; a full-page freeform selection is box-filtered
#: down to it first, which costs a resize and saves counting ten million pixels
#: to learn what the lettering in them is coloured.
_MAX_PIXELS = 512 * 512

#: Too few pixels to say anything about. Separate numbers because they answer
#: different questions: ``_MIN_SAMPLE`` is "is this sub-region worth preferring
#: over the whole crop", ``_MIN_GLYPH`` is "is this enough ink to take a median
#: of".
_MIN_SAMPLE = 64
_MIN_GLYPH = 40

#: Plausible share of the sampled zone that is lettering. Below this the Otsu
#: split found noise; above it, it split the artwork rather than the text, and
#: either way the honest answer is that the colours were not measured.
_MIN_GLYPH_SHARE = 0.004
_MAX_GLYPH_SHARE = 0.72

#: Floor under the Otsu split, in OpenCV's 8-bit Lab units (L scaled to 0-255,
#: a/b offset by 128). Paper that is JPEG-noisy or lightly shaded stays within
#: this of its own modal colour; ink never does.
_PLATE_FLOOR = 18.0

#: How well the Otsu split has to separate the two populations before either is
#: called a colour -- Otsu's own separability measure, between-class variance
#: over total. Crisp lettering measures 1.00 and a resampled page 0.98; a smooth
#: gradient measures about 0.75 wherever it is cut, and flat noisy paper about
#: 0.71, because neither has a valley to find. The gap between those two groups
#: is wide, so the line sits well clear of both.
_MIN_SEPARATION = 0.88

#: How far the glyph rim has to sit off the ink-to-paper segment before it counts
#: as an outline rather than anti-aliasing. Generous, because the rim of a thin
#: stroke is a mixture under any lighting and a false outline is visible.
_MIX_TOLERANCE = 18.0

#: Strokes thinner than this (half-width, px) cannot hold a separable outline:
#: the rim *is* the stroke, and reading its colour would report the ink twice.
_MIN_OUTLINE_RADIUS = 2.0

#: How far outward a detector quad is grown before sampling. DB predicts a shrunk
#: text region and expands it by a fixed ratio, so the quad lands close to the
#: ink -- close enough that a white outline drawn around the lettering falls
#: outside it. The outline is the thing being measured, so it has to be inside.
#: A fraction of the quad's short side keeps the margin proportional to the type
#: size, the same way :func:`fox_reader.clean.ppocr._pad_px` does.
_QUAD_PAD_FRAC = 0.18
_QUAD_PAD_MIN = 1
_QUAD_PAD_MAX = 12

_WHITE = np.array([255, 255, 255], np.uint8)


class RegionColors(NamedTuple):
    """The three colours, named as :class:`fox_reader.models.requests.DataItem`
    names them, with ``None`` for "could not be measured".

    ``stroke_color`` is ``None`` whenever the lettering has no outline of its
    own, which is the common case; see the module docstring.
    """

    font_color: str | None = None
    stroke_color: str | None = None
    bg_color: str | None = None

    @property
    def measured(self) -> bool:
        return any(value is not None for value in self)

    def payload(self) -> dict[str, str | None]:
        """The shape the OCR routes put on the wire."""
        return {
            "font_color": self.font_color,
            "stroke_color": self.stroke_color,
            "bg_color": self.bg_color,
        }


#: Nothing could be measured. Shared rather than rebuilt: it is the answer on
#: every guard and every failure path.
NOTHING = RegionColors()


# ------------------------------------------------------------------- conversion

def _hex(rgb: np.ndarray) -> str:
    r, g, b = (int(v) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgb_array(image: Any) -> np.ndarray | None:
    """``image`` as a contiguous ``(H, W, 3)`` uint8 RGB array, or None."""
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim != 3:
            return None
        if arr.shape[2] >= 3:
            return np.ascontiguousarray(arr[:, :, :3])
        return None
    # A PIL image. `convert` is a copy, so it is skipped when it would be a
    # no-op -- the routes hand us an RGB crop they have already converted.
    if getattr(image, "mode", None) != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    return arr if arr.ndim == 3 and arr.shape[2] == 3 else None


def _bool_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    """``mask`` as a bool array of ``shape``, or None when it is not usable."""
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        arr = mask
    else:
        if getattr(mask, "mode", None) != "L":
            mask = mask.convert("L")
        arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2 or arr.shape != shape:
        return None
    return arr if arr.dtype == np.bool_ else arr > 0


def _lab_of(rgb: np.ndarray) -> np.ndarray:
    """One RGB colour in OpenCV's 8-bit Lab space, as float32."""
    one = np.asarray(rgb, np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(one, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)


# -------------------------------------------------------------------- statistics

def _modal_color(pixels: np.ndarray) -> np.ndarray:
    """The most common colour in ``pixels`` (N x 3 uint8), to full precision.

    Quantised to 5 bits a channel so that anti-aliased neighbours of one colour
    count as that colour rather than as a thousand singletons -- the same reason
    :func:`fox_reader.typeset._dominant_color` downsamples before counting -- and
    then resolved to the median of the pixels that landed in the winning bin, so
    the answer is a colour the artwork really contains.
    """
    if pixels.size == 0:
        return _WHITE.copy()
    quantised = (pixels >> 3).astype(np.uint32)
    keys = (quantised[:, 0] << 10) | (quantised[:, 1] << 5) | quantised[:, 2]
    winner = int(np.bincount(keys).argmax())
    members = pixels[keys == winner]
    return np.asarray(np.rint(np.median(members, axis=0)), np.uint8)


def _median_color(pixels: np.ndarray) -> np.ndarray:
    if pixels.size == 0:
        return _WHITE.copy()
    return np.asarray(np.rint(np.median(pixels, axis=0)), np.uint8)


def _split(values: np.ndarray) -> tuple[float, float]:
    """Otsu's split of ``values``, and how well it separates them.

    ``values`` are Lab distances from the plate, clipped to 255 first: the
    maximum distance in this space is about 441, but everything past 255 is
    unambiguously ink and compressing it cannot move the split.

    The second number is Otsu's own separability measure -- between-class
    variance over total variance -- which is what tells "ink and paper" from
    "a gradient cut down the middle". Two tight populations give almost 1;
    anything spread smoothly across the range gives about 0.75 however it is
    cut, because there is no valley to find. It costs one pass over values that
    are already in hand, and it is the only signal here that answers "is this
    lettering at all" rather than "what colour is it".
    """
    if values.size < 2:
        return 0.0, 0.0
    bounded = np.clip(values, 0.0, 255.0).astype(np.uint8)
    threshold, _ = cv2.threshold(bounded.reshape(-1, 1), 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    scalar = bounded.astype(np.float64)
    total = float(scalar.var())
    if total <= 0.0:
        return float(threshold), 0.0
    high = scalar > threshold
    share = float(high.mean())
    if share <= 0.0 or share >= 1.0:
        return float(threshold), 0.0
    gap = float(scalar[high].mean() - scalar[~high].mean())
    return float(threshold), share * (1.0 - share) * gap * gap / total


def _segment_distance(point: np.ndarray, start: np.ndarray,
                      end: np.ndarray) -> float:
    """Distance from ``point`` to the segment ``start``-``end``.

    An anti-aliased pixel is an alpha composite of the two colours it sits
    between, so it lies on the segment joining them. Measuring to the segment
    rather than to its ends is what tells "a mixture of ink and paper" from "a
    third colour", without a tolerance wide enough to swallow real outlines.
    """
    span = end - start
    length2 = float(span @ span)
    if length2 < 1e-6:
        return float(np.linalg.norm(point - start))
    t = float(np.clip((point - start) @ span / length2, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * span)))


# -------------------------------------------------------------------- the zones

def _text_zone(shape: tuple[int, int], polys: Any,
               scale: float) -> tuple[np.ndarray, int] | None:
    """Where the detector found lettering, grown outward by :data:`_QUAD_PAD_FRAC`.

    Answers the zone and the widest pad it applied, which is the scale the plate
    ring outside it is measured at -- both are properties of the type size, and a
    page carries a cover title and furigana at once.
    """
    height, width = shape
    zone = np.zeros((height, width), np.uint8)
    widest = 0
    for poly in polys:
        quad = np.asarray(poly, np.float32).reshape(-1, 2)
        if quad.shape[0] < 3 or not np.isfinite(quad).all():
            continue
        if scale != 1.0:
            quad = quad * scale
        (_, (quad_w, quad_h), _) = cv2.minAreaRect(quad)
        pad = int(round(float(np.clip(_QUAD_PAD_FRAC * min(quad_w, quad_h),
                                      _QUAD_PAD_MIN, _QUAD_PAD_MAX))))
        whole = [np.round(quad).astype(np.int32)]
        cv2.fillPoly(zone, whole, 255)
        cv2.polylines(zone, whole, True, 255, thickness=2 * pad + 1)
        widest = max(widest, pad)
    return ((zone > 0), widest) if widest else None


def _halo(zone: np.ndarray, radius: int) -> np.ndarray:
    """The band of ``radius`` px just outside ``zone``.

    This, and not "everything that is not the lettering", is the paper the
    lettering sits on. A capture is often much larger than the text in it -- a
    rectangle dragged over a panel, a bubble with artwork showing at the corners
    -- and the modal colour of all of that is whatever the panel is painted,
    which is not what the glyphs are set against.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius + 1, 2 * radius + 1))
    grown = np.asarray(cv2.dilate(zone.astype(np.uint8), kernel)) > 0
    return grown & ~zone


# ---------------------------------------------------------------------- the read

def estimate(image: Any, *, mask: Any = None, polys: Any = None) -> RegionColors:
    """Read the font, outline and background colours off one region.

    ``image`` is the crop the OCR engine was given (a PIL image or an RGB array).
    ``mask`` restricts the read to the part of it that belongs to the region --
    a freeform selection is a polygon inside a white-filled rectangle, and
    without the mask that filler would be measured as the plate. ``polys`` are
    the detector's text quads in the crop's own coordinates, when the engine
    produced any.

    Never raises: an unmeasurable region answers :data:`NOTHING`, and both
    renderers then do exactly what they did before this existed.
    """
    try:
        return _estimate(image, mask, polys)
    except Exception:  # noqa: BLE001 -- a hint is never worth failing an OCR for
        logger.debug("Region colour estimation failed", exc_info=True)
        return NOTHING


def _estimate(image: Any, mask: Any, polys: Any) -> RegionColors:
    rgb = _rgb_array(image)
    if rgb is None:
        return NOTHING
    height, width = rgb.shape[:2]
    if height < 3 or width < 3:
        return NOTHING

    region = _bool_mask(mask, (height, width))

    # One resize for everything below, so the Lab conversion, the distance
    # transform and the three medians all run on a bounded number of pixels.
    scale = 1.0
    if height * width > _MAX_PIXELS:
        scale = (_MAX_PIXELS / float(height * width)) ** 0.5
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        if region is not None:
            region = cv2.resize(region.astype(np.uint8), (width, height),
                                interpolation=cv2.INTER_NEAREST) > 0

    if region is None or not region.any():
        region = np.ones((height, width), bool)

    zone: np.ndarray | None = None
    ring = _QUAD_PAD_MIN
    found = _text_zone((height, width), polys, scale) if polys is not None else None
    if found is not None:
        zone, ring = found
        zone &= region
        if int(zone.sum()) < _MIN_SAMPLE:
            zone = None

    # The plate, from the paper immediately around the lettering. Two fallbacks,
    # each one step less local: the rest of the region, then the region entire.
    # A vertical column nearly spans the bubble that holds it, so the ring can
    # come out too thin to measure.
    plate_source = region
    if zone is not None:
        for candidate in (_halo(zone, max(2, ring)) & region, region & ~zone):
            if int(candidate.sum()) >= _MIN_SAMPLE:
                plate_source = candidate
                break
    plate_rgb = _modal_color(rgb[plate_source])
    background = _hex(plate_rgb)

    sample = zone if zone is not None else region
    if int(sample.sum()) < _MIN_GLYPH:
        return RegionColors(bg_color=background)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    distance = np.linalg.norm(lab[sample].astype(np.float32) - _lab_of(plate_rgb),
                              axis=1)
    threshold, separation = _split(distance)
    if separation < _MIN_SEPARATION:
        # No valley between ink and paper: this is artwork, not lettering, and
        # any colour taken from it would be whatever side of the midpoint the
        # split happened to land on.
        return RegionColors(bg_color=background)
    inked = distance > max(_PLATE_FLOOR, threshold)
    share = float(inked.mean())
    if not (_MIN_GLYPH_SHARE <= share <= _MAX_GLYPH_SHARE):
        return RegionColors(bg_color=background)
    if int(inked.sum()) < _MIN_GLYPH:
        return RegionColors(bg_color=background)

    glyphs = np.zeros((height, width), np.uint8)
    glyphs[sample] = inked.astype(np.uint8) * 255
    body = glyphs > 0

    # Inside the strokes is the ink, unmixed; the outermost pixels are either an
    # outline or the anti-aliased edge of the ink itself.
    depth = cv2.distanceTransform(glyphs, cv2.DIST_L2, 3)
    radius = float(np.percentile(depth[body], 90.0))
    core = body & (depth >= max(1.0, 0.6 * radius))
    font_rgb = _median_color(rgb[core if int(core.sum()) >= 8 else body])

    stroke: str | None = None
    rim = body & (depth <= 1.0)
    if radius >= _MIN_OUTLINE_RADIUS and int(rim.sum()) >= _MIN_GLYPH:
        # Modal rather than median: a rim that is part outline and part
        # anti-aliasing holds two populations, and the median of the two is a
        # colour that is in neither.
        rim_rgb = _modal_color(rgb[rim])
        if _segment_distance(_lab_of(rim_rgb), _lab_of(plate_rgb),
                             _lab_of(font_rgb)) > _MIX_TOLERANCE:
            stroke = _hex(rim_rgb)

    return RegionColors(font_color=_hex(font_rgb), stroke_color=stroke,
                        bg_color=background)
