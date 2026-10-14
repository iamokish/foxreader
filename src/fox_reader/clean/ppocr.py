"""Text detection with PP-OCRv6, as an alternative to :mod:`.seg`.

Ported from the standalone ``ocr.py`` research script, with the model swapped for
the detector FoxReader has already loaded --
``MultiLangOCR._engines["default"].detector`` -- so nothing is loaded twice and
:mod:`fox_reader.ocr` stays untouched.

This is a *detector*, not a segmenter: DB (differentiable binarization) answers
with one quadrilateral per text line, so the mask is a set of blocks rather than
the glyph shapes themselves. It costs about two seconds on the CPU for a page
where the Unet++ model costs thirty, and it never mistakes artwork for lettering
-- but it also removes everything else inside the block, so a quad that clips a
speech-bubble border takes the border with it.

The interface mirrors :func:`fox_reader.clean.seg.detect` so the two can be
swapped: ``detect`` returns ``(coverage, agreement)``, where coverage carries each
quad's own confidence and agreement is simply everywhere a quad was found --
there is no multi-scale vote to take here, and every detection is one the network
is already sure of (DB post-processing has dropped anything below ``box_thresh``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

#: Post-processing knobs the detector's ``predict`` may accept. Absent ones mean
#: "use the value the detector resolved at construction" -- for
#: PP-OCRv6_medium_det that is ``thresh`` 0.2, ``box_thresh`` 0.45 and
#: ``unclip_ratio`` 1.4 from the model's ``inference.yml``, with
#: ``limit_side_len`` 960 and ``limit_type: max``.
#:
#: Those last two are not what the model's ``preprocessor_config.json`` says
#: (736 / ``min``); see :data:`fox_reader.ocr_predict._TEXT_DET_MAX_LIMIT_MODELS`
#: for why the larger value is the one that has always run.
_PREDICT_KEYS = ("limit_side_len", "limit_type", "thresh", "box_thresh",
                 "unclip_ratio")


class DetectorUnavailable(RuntimeError):
    """No PP-OCR detector to borrow -- the OCR models are not loaded."""


def _as_rgb(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    if bgr.shape[2] == 4:
        return cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def boxes(detector: Any, bgr: np.ndarray,
          **kw: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(polys, scores)`` -- an ``(N, 4, 2)`` float array and ``(N,)``.

    Takes BGR, because that is what the clean chain carries a page around as.
    Callers that already hold RGB should use :func:`boxes_rgb` instead rather
    than converting twice for nothing.
    """
    return boxes_rgb(detector, _as_rgb(bgr), **kw)


def boxes_rgb(detector: Any, rgb: Any,
              **kw: Any) -> tuple[np.ndarray, np.ndarray]:
    """:func:`boxes`, for an image already in the order the detector wants.

    The detector has always been fed RGB -- see :func:`_as_rgb` above, and
    `PaddleOCRWrapper._preprocess`, which hands it `np.array(img.convert("RGB"))`.
    This is the same call with the conversion left out, so a caller that already
    holds RGB (the colour read in :mod:`fox_reader.services.ocr_service`) does
    not pay an RGB->BGR->RGB round trip -- two full-frame copies on a page-sized
    freeform selection.

    ``rgb`` is passed to the model untouched, so it may be either of the things
    `fox_reader.ocr_predict._as_pil` accepts: an RGB ndarray, or a PIL image.
    A PIL crop is the cheapest of all, since that is what the detector's own
    preprocessing wants and it would otherwise rebuild one from the array.
    """
    if detector is None:
        raise DetectorUnavailable(
            "the OCR detector is not loaded, so PaddleOCR text clean is "
            "unavailable")
    opts = {k: v for k, v in kw.items() if k in _PREDICT_KEYS and v is not None}
    t0 = time.time()
    try:
        result = detector.predict(rgb, **opts) if opts else detector.predict(rgb)
    except TypeError:
        # Older builds take the post-processing values at construction time only.
        # Detecting with the shipped defaults beats failing the clean.
        result = detector.predict(rgb)

    res = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(res, dict):
        raw_polys = res.get("dt_polys")
        raw_scores = res.get("dt_scores")
    else:
        raw_polys = getattr(res, "dt_polys", None)
        raw_scores = getattr(res, "dt_scores", None)

    polys = np.asarray(raw_polys if raw_polys is not None else [], np.float32)
    polys = polys.reshape(-1, 4, 2) if polys.size else np.zeros((0, 4, 2), np.float32)
    if raw_scores is None:
        scores = np.ones(len(polys), np.float32)
    else:
        scores = np.asarray(raw_scores, np.float32).reshape(-1)
        if len(scores) != len(polys):
            scores = np.ones(len(polys), np.float32)
    log.debug("PP-OCRv6 det: %d regions in %.1fs", len(polys), time.time() - t0)
    return polys, scores


def _pad_px(poly: np.ndarray, frac: float, lo: int, hi: int) -> int:
    """How far to grow one quad outward, from the quad's own short side.

    DB is trained to predict a *shrunk* text region and expands it again by a
    fixed ratio, so the quad lands close to the ink -- close enough that the
    anti-aliased fringe, and any white outline drawn around the lettering, can
    fall outside it and survive as a ghost. A fraction of the short side keeps
    that margin proportional to the type size, the same way the segmentation
    chain works in units of stroke radius.

    The fraction is deliberately small, and ``lo`` carries most of the weight for
    body text. A vertical text column nearly spans the bubble that holds it, so a
    generous pad pushes the block across the bubble border; the ring of known
    pixels around the hole then contains ink, ``_flat_fill`` refuses it, and the
    reconstructor answers with 251-253 instead of paper white -- a grey ghost of
    the block, which is the one artefact a flat white area cannot hide.
    """
    (_, (w, h), _) = cv2.minAreaRect(poly.astype(np.float32))
    return int(round(float(np.clip(frac * min(w, h), lo, hi))))


def coverage(shape: tuple[int, ...], polys: np.ndarray, scores: np.ndarray,
             pad_frac: float = 0.03, pad_min: int = 2, pad_max: int = 12
             ) -> np.ndarray:
    """Paint every quad, grown by :func:`_pad_px`, with its own confidence."""
    h, w = shape[:2]
    cov = np.zeros((h, w), np.float32)
    for poly, s in zip(polys, scores):
        q = np.round(poly).astype(np.int32)
        pad = _pad_px(poly, pad_frac, pad_min, pad_max)
        x0 = max(0, int(q[:, 0].min()) - pad - 1)
        y0 = max(0, int(q[:, 1].min()) - pad - 1)
        x1 = min(w, int(q[:, 0].max()) + pad + 2)
        y1 = min(h, int(q[:, 1].max()) + pad + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        local = [q - np.array([x0, y0], np.int32)]
        sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(sub, local, 255)
        if pad > 0:
            cv2.polylines(sub, local, True, 255, thickness=2 * pad + 1)
        roi = cov[y0:y1, x0:x1]
        np.maximum(roi, float(s) * (sub > 0), out=roi)
    return cov


def detect(detector: Any, bgr: np.ndarray, pad_frac: float = 0.03,
           **kw: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coverage, agreement)``, matching :func:`.seg.detect`."""
    polys, scores = boxes(detector, bgr, **kw)
    cov = coverage(bgr.shape, polys, scores, pad_frac=pad_frac)
    return cov, cov > 0


def refine_boxes(prob: np.ndarray, thr: float = 0.35, grow: int = 0) -> np.ndarray:
    """Turn the score map into a mask.

    Almost none of :func:`fox_reader.clean.mask.refine_mask` applies to blocks.
    Hole filling and speck dropping have nothing to act on in a solid rectangle,
    and the cross-scale vote filter needs scales that can disagree. Halo growth
    looks for the glyph outline just outside the detection, but the outline is
    already inside the quad. The aura test, measured from a rectangle rather than
    a glyph rim, fires on none of the reference pages, because the block's ends
    run into the panel gutter where there is no aura at all. Padding therefore
    happens per quad in :func:`coverage`, where each block's own size is still
    known -- a single dilation afterwards would treat a furigana column and a
    cover title alike -- and nothing else is done here.
    """
    mask = ((prob > thr).astype(np.uint8)) * 255
    if grow > 0:
        mask = cv2.dilate(mask, K3, iterations=grow)
    return mask
