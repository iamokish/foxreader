"""Multi-scale patch-synthesis inpainting.

Every missing pixel is rebuilt from *real pixels that sit near it*: for each hole
pixel the algorithm searches a bounded neighbourhood for the best matching known
patch, then each pixel is averaged over all patches covering it (patch voting).
Searching coarse-to-fine lets a large hole agree on structure first and refine
detail afterwards, which avoids both the blur of diffusion inpainting and the
wrong-patch blobs of a single-scale search.

It is an EM-style variant of PatchMatch / "Image Melding" with three practical
refinements that matter for clean results:

* the randomised nearest-neighbour search is replaced by an exhaustive search
  over a dense offset grid, vectorised with box filters -- deterministic, and
  affordable because the grid shrinks once structure is settled;
* the offset field is median filtered each iteration, so neighbouring pixels
  copy from a *coherent* region instead of each picking its own best match;
* votes are weighted by patch similarity, so a poor match cannot drag a pixel.
"""

from __future__ import annotations

import cv2
import numpy as np

INF = np.float32(np.inf)


def _box(a: np.ndarray, p: int) -> np.ndarray:
    return cv2.boxFilter(a, -1, (p, p), normalize=False,
                         borderType=cv2.BORDER_REFLECT101)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``out[p] = a[p + (dy, dx)]``, edges replicated."""
    H, W = a.shape[:2]
    out = np.empty_like(a)
    ys = slice(max(0, dy), H + min(0, dy))
    xs = slice(max(0, dx), W + min(0, dx))
    yd = slice(max(0, -dy), H + min(0, -dy))
    xd = slice(max(0, -dx), W + min(0, -dx))
    out[yd, xd] = a[ys, xs]
    # replicate into the strip that fell outside
    if dy > 0:
        out[H - dy:] = out[H - dy - 1:H - dy]
    elif dy < 0:
        out[:-dy] = out[-dy:-dy + 1]
    if dx > 0:
        out[:, W - dx:] = out[:, W - dx - 1:W - dx]
    elif dx < 0:
        out[:, :-dx] = out[:, -dx:-dx + 1]
    return out


def _pyramid(lab: np.ndarray, hole: np.ndarray, levels: int
             ) -> list[tuple[np.ndarray, np.ndarray]]:
    pyr = [(lab, hole)]
    for _ in range(levels):
        prev_img, prev_hole = pyr[-1]
        h, w = prev_hole.shape
        if min(h, w) < 56:
            break
        size = (max(28, w // 2), max(28, h // 2))
        small = cv2.resize(prev_img, size, interpolation=cv2.INTER_AREA)
        # a coarse pixel counts as hole if it contains any hole pixel
        small_hole = cv2.resize(prev_hole.astype(np.float32), size,
                                interpolation=cv2.INTER_AREA) > 1e-3
        pyr.append((small, small_hole))
    return pyr[::-1]  # coarsest first


def _em_level(lab: np.ndarray, hole: np.ndarray, patch: int, radius: int,
              iters: int, min_valid: float = 0.5,
              coherence: int = 5) -> np.ndarray:
    """Alternate best-offset search and weighted patch voting on one level."""
    h, w = hole.shape
    pad = radius + patch // 2
    lab = cv2.copyMakeBorder(lab, pad, pad, pad, pad, cv2.BORDER_REFLECT101)
    hole = cv2.copyMakeBorder(hole.astype(np.uint8), pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=0).astype(bool)
    known = (~hole).astype(np.float32)
    H, W = hole.shape
    core = (slice(pad, pad + h), slice(pad, pad + w))

    span = 2 * radius + 1
    offs = [(dy, dx) for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)]
    area = float(patch * patch)
    holef = hole.astype(np.float32)

    for _ in range(iters):
        best = np.full((H, W), INF, np.float32)
        bdy = np.zeros((H, W), np.float32)
        bdx = np.zeros((H, W), np.float32)
        for dy, dx in offs:
            if dy == 0 and dx == 0:
                continue
            diff = lab - _shift(lab, dy, dx)
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            vs = _shift(known, dy, dx)
            cnt = _box(vs, patch)
            cost = _box(d2 * vs, patch) / np.maximum(cnt, 1e-6)
            cost[cnt < min_valid * area] = INF
            upd = cost < best
            best[upd] = cost[upd]
            bdy[upd] = dy
            bdx[upd] = dx

        found = np.isfinite(best)
        if not (found & hole).any():
            break

        if coherence >= 3:
            # make neighbouring pixels copy from the same place
            bdy_s = cv2.medianBlur(bdy, coherence)
            bdx_s = cv2.medianBlur(bdx, coherence)
            keep = found & ~hole
            bdy = np.where(keep, bdy, bdy_s)
            bdx = np.where(keep, bdx, bdx_s)
            bdy = np.clip(np.rint(bdy), -radius, radius)
            bdx = np.clip(np.rint(bdx), -radius, radius)

        # similarity weighting: a bad match should not dominate a vote
        scale = np.median(best[found & hole]) if (found & hole).any() else 1.0
        weight = np.exp(-best / max(float(scale), 1e-3)).astype(np.float32)
        weight[~found] = 0.0
        wsel = weight * holef

        num = np.zeros((H, W, 3), np.float32)
        den = np.zeros((H, W), np.float32)
        idx = ((bdy + radius) * span + (bdx + radius)).astype(np.int32)
        for i, (dy, dx) in enumerate(offs):
            if dy == 0 and dx == 0:
                continue
            sel = wsel * (idx == i)
            if not sel.any():
                continue
            ci = _box(sel, patch)
            num += ci[:, :, None] * _shift(lab, dy, dx)
            den += ci
        upd = (den > 1e-6) & hole
        lab[upd] = num[upd] / den[upd][:, None]

    return lab[core]


def inpaint_patchsynth(bgr: np.ndarray, hole: np.ndarray, patch: int = 7,
                       radius: int = 12, coarse_radius: int = 20,
                       iters: int = 3, coarse_iters: int = 6,
                       max_levels: int = 4) -> np.ndarray:
    """Fill ``hole`` (255 = missing) in ``bgr`` from nearby known patches."""
    holeb = hole > 0
    if not holeb.any():
        return bgr.copy()

    # Structural initialisation: diffusion is a sane low-frequency guess which
    # the coarse level then re-synthesises out of real patches.
    init = cv2.inpaint(bgr, (holeb.astype(np.uint8)) * 255, 5, cv2.INPAINT_TELEA)
    lab0 = cv2.cvtColor(init, cv2.COLOR_BGR2Lab).astype(np.float32)

    thick = float(cv2.distanceTransform((holeb.astype(np.uint8)) * 255,
                                        cv2.DIST_L2, 5).max())
    levels = int(np.clip(np.ceil(np.log2(max(thick, 1.0) / 6.0)), 0, max_levels))
    pyr = _pyramid(lab0, holeb, levels)

    cur = None
    for lvl, (l_img, l_hole) in enumerate(pyr):
        h, w = l_hole.shape
        if cur is not None:
            up = cv2.resize(cur, (w, h), interpolation=cv2.INTER_LINEAR)
            l_img = l_img.copy()
            l_img[l_hole] = up[l_hole]
        first = lvl == 0
        cur = _em_level(l_img, l_hole, patch,
                        coarse_radius if first else radius,
                        coarse_iters if first else iters)

    out_lab = lab0.copy()
    out_lab[holeb] = cur[holeb]
    filled = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8),
                          cv2.COLOR_Lab2BGR)
    res = bgr.copy()
    res[holeb] = filled[holeb]
    return res
