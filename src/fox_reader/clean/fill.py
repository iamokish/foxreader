"""Reconstruction of the pixels under the text.

Ported from the standalone ``clean_text.py`` research script. :func:`inpaint_regions`
is the entry point: it takes the page and a mask of what to remove
(:mod:`.mask`), and rebuilds each text cluster inside its own neighbourhood using
one of the fill methods in :data:`INPAINT_METHODS`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import cv2
import numpy as np

from fox_reader.clean import caps
from fox_reader.clean.hybridfill import inpaint_hybrid, inpaint_transport
from fox_reader.clean.patchsynth import inpaint_patchsynth
from fox_reader.clean.smoothfill import inpaint_pyramid

log = logging.getLogger(__name__)

#: Fill methods, ordered best-first. The first entry that is not "transport" is
#: the default the UI offers, because transport costs seconds per cluster.
INPAINT_METHODS = ("transport", "hybrid-level", "level", "hybrid-fsr",
                   "hybrid-patch", "hybrid", "patch", "pyramid", "fsr",
                   "fsr-fast", "shiftmap", "telea", "ns")

DEFAULT_INPAINT_METHOD = "hybrid-level"

# Kept as an alias so code ported from the script keeps working.
METHODS = INPAINT_METHODS


def _flat_fill(bgr: np.ndarray, hole: np.ndarray, ring: int = 4,
               tol: float = 8.0, min_ring: int = 12
               ) -> tuple[np.ndarray, np.ndarray]:
    """Paint holes whose surroundings are one flat colour with exactly that colour.

    Most manga lettering sits on blank paper, and the paper is a single value --
    255. Every reconstructor here is built for *texture*, so it answers with
    something very slightly different: the pyramid fill interpolates to 251-253
    and the grain term scatters a few levels around it. On paint that is invisible;
    on a white speech bubble it leaves a faint grey ghost in the shape of the
    removed glyph, which is the one artefact a flat area cannot hide.

    So each hole component is examined on its own: the ring of known pixels just
    outside it is measured, and if that ring is uniform (robust spread within
    ``tol``) the component is filled with the ring's median and removed from the
    hole passed on to the real reconstructor. A component that touches art has a
    spread-out ring and is left alone, so this can only fire where it is exact.

    Returns the partly filled image and the hole that is left.
    """
    holeb = hole > 0
    if not holeb.any():
        return bgr.copy(), hole.copy()
    out = bgr.copy()
    rest = hole.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1,) * 2)
    h, w = hole.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        holeb.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        pad = ring + 1
        X0, Y0 = max(0, x - pad), max(0, y - pad)
        X1, Y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        sl = (slice(Y0, Y1), slice(X0, X1))
        sub = labels[sl] == i
        band = (cv2.dilate((sub.astype(np.uint8)) * 255, k) > 0) & ~holeb[sl]
        if int(band.sum()) < min_ring:
            continue
        vals = bgr[sl][band].astype(np.float32)
        lo, hi = np.percentile(vals, (2.5, 97.5), axis=0)
        if float(np.max(hi - lo)) > tol:
            continue
        out[sl][sub] = np.median(vals, axis=0).round().astype(np.uint8)
        rest[sl][sub] = 0
    return out, rest


def _inpaint_patch(patch: np.ndarray, hole: np.ndarray, method: str) -> np.ndarray:
    """Fill ``hole`` (255 = missing) in ``patch``; returns the filled patch."""
    patch, hole = _flat_fill(patch, hole)
    if not hole.any():
        return patch
    if method == "hybrid":
        return inpaint_hybrid(patch, hole)
    if method == "hybrid-patch":
        return inpaint_hybrid(patch, hole, source="patch")
    if method == "hybrid-fsr":
        return inpaint_hybrid(patch, hole, source="fsr")
    if method == "level":
        return inpaint_pyramid(patch, hole, solver="level")
    if method == "hybrid-level":
        return inpaint_hybrid(patch, hole, source="fsr", solver="level")
    if method == "transport":
        return inpaint_transport(patch, hole)
    if method == "patch":
        return inpaint_patchsynth(patch, hole)
    if method == "pyramid":
        return inpaint_pyramid(patch, hole)
    if method in ("telea", "ns"):
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        return cv2.inpaint(patch, hole, 7, flag)

    valid = 255 - hole  # xphoto convention: non-zero == valid pixels
    src = patch.copy()
    if valid.any():
        # Neutralise the text colour so nothing of it can bleed through.
        fill = cv2.mean(patch, valid)[:3]
        src[hole > 0] = np.array(fill, np.float32).astype(np.uint8)

    # `getattr(cv2, "xphoto", None) is None` is not a usable test -- OpenCV 5
    # ships an empty `xphoto` module that passes it and then raises on the call.
    # `caps` probes for the function and the flag themselves; see that module.
    flag = caps.xphoto_flag("shiftmap" if method == "shiftmap"
                            else "fsr" if method == "fsr" else "fsr-fast")
    if flag is None:
        caps.note_fallback(method, "Telea diffusion")
        return cv2.inpaint(patch, hole, 7, cv2.INPAINT_TELEA)

    if method == "shiftmap":
        lab = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
        dst = np.zeros_like(lab)
        cv2.xphoto.inpaint(lab, valid, dst, flag)
        return cv2.cvtColor(dst, cv2.COLOR_Lab2BGR)

    dst = np.zeros_like(src)
    cv2.xphoto.inpaint(src, valid, dst, flag)
    return dst


_SMEARS = ("hybrid-fsr", "hybrid-level", "hybrid", "level", "pyramid",
           "telea", "ns", "fsr", "fsr-fast")


def _beyond_interpolation(bgr: np.ndarray, cluster: np.ndarray,
                          mask: np.ndarray, half_min: float = 60.0,
                          tex_min: float = 2.0, near: tuple[int, int] = (2, 10),
                          sigma: float = 2.0) -> tuple[bool, float, float]:
    """Is this hole too wide to interpolate, in paint too busy to smear?

    Every reconstructor here except offset transport works from the hole's
    boundary, so what it can say about the middle of the hole falls off with the
    distance to the nearest known pixel. Past a hundred px there is nothing left
    to say: a cover title laid across hair, ornament and background comes back as
    one smooth stain, and no amount of added grain or edge detail changes that.
    Offset transport is the one method that does not interpolate -- it covers the
    hole with whole displaced pieces of the same picture -- so it keeps structure
    at every scale, at the price of inventing content and of twenty seconds.

    That price is only worth paying where a smear would be obvious, which is not
    the same as "where the hole is wide". A wide hole in flat airbrushed paint is
    a place where a smooth continuation is not merely acceptable but *correct*,
    and transporting hair into it would be a clear regression. Both conditions
    are therefore required: the hole has to be wider than boundary information
    reaches, and the paint right at its edge has to be textured artwork rather
    than flat.

    Texture is measured in a shell only a few px outside the hole -- the paint the
    reconstruction actually draws from -- and as a *median*, so a single strong
    contour cannot carry it. That distinction matters on a bitonal page: measured
    over a wider ring, or by range instead of density, the flattest holes on the
    page score highest, because a speech-bubble border spans black to white.
    """
    cu = (cluster.astype(np.uint8)) * 255
    half = float(cv2.distanceTransform(cu, cv2.DIST_L2, 5).max())
    if half < half_min:
        return False, half, 0.0
    d = cv2.distanceTransform(255 - cu, cv2.DIST_L2, 5)
    ring = (d >= near[0]) & (d <= near[1]) & (mask == 0)
    if ring.sum() < 200:
        return False, half, 0.0
    img = bgr.astype(np.float32)
    hi = img - cv2.GaussianBlur(img, (0, 0), sigma)
    tex = float(np.median(np.sqrt(
        np.einsum("ijk,ijk->ij", hi, hi)[ring] / 3.0)))
    return tex >= tex_min, half, tex


def inpaint_regions(bgr: np.ndarray, mask: np.ndarray,
                    method: str = DEFAULT_INPAINT_METHOD, gap: int = 12,
                    transport: bool = True,
                    progress: Callable[[int, int], None] | None = None
                    ) -> np.ndarray:
    """Inpaint each text cluster inside its own neighbourhood.

    Working on a local crop keeps the reconstruction sourced from pixels *near*
    the hole, and keeps the cost of the expensive reconstructors bounded.
    Validity always comes from the global mask, so text pixels of a neighbouring
    cluster are never used as source material.

    ``progress`` is called with ``(done, total)`` after each cluster. It runs on
    whichever worker thread called this function, so it must not touch the event
    loop -- see :mod:`fox_reader.services.clean_service` for the counter it feeds.
    """
    if method not in INPAINT_METHODS:
        method = DEFAULT_INPAINT_METHOD
    h, w = mask.shape
    out = bgr.copy()

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    grouped = cv2.dilate(mask, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)

    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    total = len(order)
    failures: list[Exception] = []
    for idx, i in enumerate(order, 1):
        cluster = ((labels == i) & (mask > 0))
        if not cluster.any():
            if progress is not None:
                progress(idx, total)
            continue
        ys, xs = np.nonzero(cluster)
        x0b, x1b, y0b, y1b = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        bw, bh = x1b - x0b, y1b - y0b

        margin = int(np.clip(0.6 * max(bw, bh), 28, 140))
        X0 = Y0 = 0
        X1, Y1 = w, h
        sub_hole = mask
        valid_ratio = 0.0
        for _ in range(4):  # grow the crop until enough known pixels remain
            X0, Y0 = max(0, x0b - margin), max(0, y0b - margin)
            X1, Y1 = min(w, x1b + margin), min(h, y1b + margin)
            sub_hole = mask[Y0:Y1, X0:X1]
            valid_ratio = 1.0 - float((sub_hole > 0).mean())
            if valid_ratio >= 0.45 or margin >= 320:
                break
            margin = int(margin * 1.6)

        patch = bgr[Y0:Y1, X0:X1]
        local = cluster[Y0:Y1, X0:X1]
        use, note = method, ""
        # Escalating needs offset transport to actually exist: without it
        # `inpaint_transport` is diffusion plus a level fix, i.e. the very smear
        # the escalation is trying to avoid, at no gain and some cost.
        if transport and method in _SMEARS and caps.fill_available("transport"):
            wide, half, tex = _beyond_interpolation(patch, local, sub_hole)
            if wide:
                use, note = "transport", (f" -> transport "
                                          f"(half-w {half:.0f}, tex {tex:.2f})")
        t0 = time.time()
        try:
            filled = _inpaint_patch(patch, mask[Y0:Y1, X0:X1], use)
        except Exception as exc:  # noqa: BLE001 - one cluster must not lose the page
            log.exception("inpaint failed for cluster %d/%d", idx, total)
            failures.append(exc)
            if progress is not None:
                progress(idx, total)
            continue
        out[Y0:Y1, X0:X1][local] = filled[local]
        log.debug("  [%d/%d] %dx%d at (%d,%d) crop %dx%d valid=%.2f %.1fs%s",
                  idx, total, bw, bh, x0b, y0b, X1 - X0, Y1 - Y0, valid_ratio,
                  time.time() - t0, note)
        if progress is not None:
            progress(idx, total)

    # Per-cluster tolerance is right for one awkward hole and wrong for a broken
    # method: if *nothing* was reconstructed, the caller has to hear about it,
    # otherwise the page comes back untouched and the run looks like a success.
    # `clean_service` turns this into a note the UI shows.
    if failures and len(failures) == total:
        raise RuntimeError(
            f"every region failed to reconstruct with {method!r}: "
            f"{failures[0]}") from failures[0]
    return out
