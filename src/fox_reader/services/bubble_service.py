from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from fox_reader.device import FOX_DEVICE
from fox_reader.utils import MODELS_DIR

# Slack around the region so a cut that grazes the border still has room to be
# drawn, and so the returned contours are never clipped by the canvas edge.
_SPLIT_MARGIN = 6
# A 3 px erasure is the narrowest gap that still separates two pieces under the
# 8-connectivity that `connectedComponents` uses below.
_MIN_CUT_THICKNESS = 3
_MAX_CUT_THICKNESS = 7
# Below this share of the region a piece is a sliver produced by the cut itself.
_MIN_PIECE_FRACTION = 0.004
_MIN_PIECE_AREA = 24
# A piece this close to the area that survived the cut means the cut achieved
# nothing. Only a floor -- the real threshold is derived from the sliver filter
# below, so the two tests cannot both reject the same genuine split.
_SAME_REGION_FRACTION = 0.99
# How far back from a tip to look for the direction the stroke is heading in.
# One adjacent sample is 1 px away and its direction is rounding noise; scaled
# by a whole region diagonal that noise becomes a ray pointing anywhere.
_TANGENT_SPAN = 8.0
# A stroke shorter than this share of the region diagonal is mouse jitter or a
# stray click, not a cut. Without the test, prolonging both of its ends would
# turn a 4 px twitch into a line straight through the bubble.
_MIN_STROKE_SPAN_FRACTION = 0.05


def _as_array(points: list) -> np.ndarray:
    """`[{"x":..,"y":..}, ...]` (or `[[x,y], ...]`) -> float64 (N,2)."""
    out: list[tuple[float, float]] = []
    for p in points:
        if isinstance(p, dict):
            x, y = p.get("x"), p.get("y")
        else:
            try:
                x, y = p[0], p[1]
            except (TypeError, IndexError, KeyError):
                continue
        if x is None or y is None:
            continue
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(fx) and np.isfinite(fy)):
            continue
        out.append((fx, fy))
    return np.asarray(out, dtype=np.float64).reshape((-1, 2))


def _drop_repeats(pts: np.ndarray) -> np.ndarray:
    """Remove consecutive duplicates; they carry no direction information."""
    if len(pts) < 2:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)
    return pts[keep]


def _inner_ref(pts: np.ndarray, tip_idx: int, step: int) -> np.ndarray | None:
    """The point to aim away from when prolonging the stroke past `pts[tip_idx]`.

    Walks inward until the chord back to the tip is at least `_TANGENT_SPAN`
    long. Taking the immediate neighbour instead -- which is what this used to do
    -- reads a direction off two samples that a freehand stroke leaves barely a
    pixel apart, so the angle is quantisation noise. Multiplied by the region
    diagonal, a one-pixel wobble swings the extension by hundreds of pixels and
    the cut lands somewhere the user never drew. That is precisely why a stroke
    with many bends used to split inaccurately: bends are where consecutive
    samples crowd together.

    Returns `None` only when every point coincides with the tip.
    """
    tip = pts[tip_idx]
    fallback: np.ndarray | None = None
    i = tip_idx + step
    while 0 <= i < len(pts):
        d = pts[i] - tip
        dist = float(np.hypot(d[0], d[1]))
        if dist > 1e-9:
            # Keep the furthest point seen so far, so a stroke shorter than
            # `_TANGENT_SPAN` still uses its own full length as the direction.
            fallback = pts[i]
            if dist >= _TANGENT_SPAN:
                return pts[i]
        i += step
    return fallback


def _extend_ends(pts: np.ndarray, reach: float, tip_inside) -> np.ndarray:
    """Prolong the first and last segment so the cut always exits the region.

    A freehand cut is drawn by hand, so it habitually starts or ends a few pixels
    inside the bubble. Such a cut carves a notch instead of separating anything,
    which is the difference between "the splitter did nothing" and a working
    split -- and it is invisible to the user, who did draw a line right across.
    Continuing both ends along their own direction, far enough to leave the
    region's bounding box, makes the outcome depend on the shape of the stroke
    rather than on where exactly the mouse went down.

    Only an end that stops *inside* the region is prolonged. An end that already
    lies outside has nothing left to reach, and extending it is actively harmful:
    the ray carries on and can re-enter the region on the far side, carving a
    second cut the user never drew. A stroke whose two ends meet is already closed
    and is left alone entirely -- prolonging it would fire a ray off the loop's
    tangent and slice the region a second time.
    """
    if len(pts) < 2 or reach <= 0:
        return pts

    gap = pts[0] - pts[-1]
    if float(np.hypot(gap[0], gap[1])) <= _TANGENT_SPAN:
        return pts

    def _outward(tip: np.ndarray, inner: np.ndarray | None) -> np.ndarray | None:
        if inner is None:
            return None
        d = tip - inner
        n = float(np.hypot(d[0], d[1]))
        if n < 1e-9:
            return None
        return tip + d / n * reach

    head = _outward(pts[0], _inner_ref(pts, 0, 1)) if tip_inside(pts[0]) else None
    tail = _outward(pts[-1], _inner_ref(pts, len(pts) - 1, -1)) if tip_inside(pts[-1]) else None
    parts = [pts]
    if head is not None:
        parts.insert(0, head.reshape(1, 2))
    if tail is not None:
        parts.append(tail.reshape(1, 2))
    return np.vstack(parts)


def _contour_of(component: np.ndarray, ox: int, oy: int) -> list[dict] | None:
    """Outer contour of a single-piece mask, simplified and de-offset."""
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    # Freehand cuts produce contours with thousands of points; the frontend has
    # to re-render every one of them on each overlay update, so they are reduced
    # to the ~1 px-faithful polyline that draws identically.
    epsilon = float(np.clip(0.0015 * perimeter, 0.5, 2.0))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        approx = contour
    if len(approx) < 3:
        return None
    return [
        {"x": int(p[0][0]) + ox, "y": int(p[0][1]) + oy} for p in approx
    ]


class BubbleService:
    """Torch-based speech bubble detection (safetensors weights)."""

    def __init__(self) -> None:
        from fox_reader.bubble import getBubbleModel

        self.device = FOX_DEVICE.bubble
        self._model = getBubbleModel(MODELS_DIR / "bubble" / "model.safetensors", device=self.device)

    def detect(self, image: Image.Image, grayscale: bool) -> list[dict]:
        # Grayscale lives in the model (`fox_reader.bubble._to_grayscale`), so
        # every caller binarises identically; the service only forwards.
        return self._model.detect(image, grayscale=bool(grayscale))

    @staticmethod
    def split(region_coords: list, split_object: list) -> list[dict]:
        """Cut a region into pieces along a freehand stroke.

        The stroke is rasterised as an *open* polyline. Treating a many-point
        stroke as a polygon to be erased -- which is what this used to do -- is
        what made a curvy cut inaccurate: the chord from the last point back to
        the first closes the stroke into a lasso, and everything that lasso
        happens to enclose is erased along with the intended line. The more bends
        a stroke has, the further that phantom chord wanders from it.

        Returns one polygon per piece, or `[]` when the stroke failed to divide
        the region -- a notch, a miss, a twitch, or a cut that only shaved a
        sliver off. Callers must not treat "one piece" as a successful split,
        since that piece is the region they started with.
        """
        region = _as_array(region_coords)
        cut = _drop_repeats(_as_array(split_object))
        if len(region) < 3 or len(cut) < 2:
            return []

        lo = region.min(axis=0)
        hi = region.max(axis=0)
        span = hi - lo
        if span[0] < 1 or span[1] < 1:
            return []

        # The canvas covers the region only. The prolonged cut deliberately runs
        # past it and OpenCV clips the drawing, so the extension costs nothing.
        ox = int(np.floor(lo[0])) - _SPLIT_MARGIN
        oy = int(np.floor(lo[1])) - _SPLIT_MARGIN
        width = int(np.ceil(hi[0])) - ox + _SPLIT_MARGIN + 1
        height = int(np.ceil(hi[1])) - oy + _SPLIT_MARGIN + 1
        if width < 3 or height < 3 or width > 20000 or height > 20000:
            return []

        origin = np.array([ox, oy], dtype=np.float64)

        mask = np.zeros((height, width), dtype=np.uint8)
        region_px = np.rint(region - origin).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [region_px], 255, lineType=cv2.LINE_8)
        region_area = int(cv2.countNonZero(mask))
        if region_area <= 0:
            return []

        thickness = int(
            np.clip(
                round(min(width, height) * 0.01),
                _MIN_CUT_THICKNESS,
                _MAX_CUT_THICKNESS,
            )
        )

        diagonal = float(np.hypot(span[0], span[1]))
        cut_span = cut.max(axis=0) - cut.min(axis=0)
        if float(np.hypot(cut_span[0], cut_span[1])) < diagonal * _MIN_STROKE_SPAN_FRACTION:
            return []

        # Rasterise the stroke as drawn, before any extension, and require that it
        # actually touches the region. Without this the ends were prolonged
        # unconditionally, so a stroke drawn *beside* the bubble grew two rays a
        # full diagonal long, one of which crossed it -- and the caller hides the
        # source entry as soon as pieces come back, so a mis-aimed stroke silently
        # destroyed the entry it was supposed to divide.
        probe = np.zeros_like(mask)
        raw_px = np.rint(cut - origin).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            probe, [raw_px], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_8
        )
        if not np.any(cv2.bitwise_and(probe, mask)):
            return []

        def _tip_inside(point: np.ndarray) -> bool:
            px, py = np.rint(point - origin).astype(np.int32)
            if not (0 <= px < width and 0 <= py < height):
                return False
            return bool(mask[py, px])

        reach = diagonal + _SPLIT_MARGIN * 4.0
        cut = _extend_ends(cut, reach, _tip_inside)

        cut_px = np.rint(cut - origin).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            mask,
            [cut_px],
            isClosed=False,
            color=0,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )
        remaining_area = int(cv2.countNonZero(mask))

        # Components, not contours: a stroke that loops back on itself leaves an
        # island of region inside the loop, and RETR_EXTERNAL would silently drop
        # it. Every component is exactly one piece.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        min_area = max(_MIN_PIECE_AREA, int(region_area * _MIN_PIECE_FRACTION))
        # The "nothing happened" test is measured against what survived the cut,
        # not the region it started as -- the erased band alone is enough to put a
        # genuine split's larger piece under 99% of the original. It also has to
        # stay looser than `min_area`, or a real split whose smaller piece sits
        # just above the sliver threshold would have its larger piece thrown out
        # as "the whole region" and the split rejected. Deriving the bound from
        # `min_area` makes that overlap impossible by construction.
        same_region_area = max(
            remaining_area - min_area,
            int(remaining_area * _SAME_REGION_FRACTION),
        )

        pieces: list[tuple[int, list[dict]]] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            if area >= same_region_area:
                # Effectively the input region: the stroke never divided it.
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            # Cropping keeps the contour trace proportional to the piece rather
            # than to the whole region, which is what keeps this fast on the
            # large bubbles where strokes have the most points.
            piece = (labels[y : y + h, x : x + w] == label).astype(np.uint8) * 255
            coords = _contour_of(piece, ox + x, oy + y)
            if coords is None:
                continue
            pieces.append((area, coords))

        if len(pieces) < 2:
            return []

        pieces.sort(key=lambda item: item[0], reverse=True)
        return [{"type": "polygon", "coords": coords} for _, coords in pieces]
