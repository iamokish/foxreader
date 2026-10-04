"""Pyramid ("push-pull") hole filling with matched grain.

For soft, airbrushed artwork the eye forgives a smooth fill but instantly spots a
duplicated ridge, so the safest reconstruction of a large hole is one that
continues the surrounding low frequencies exactly and then restores the local
amount of high-frequency detail.

``inpaint_pyramid`` does that in two steps:

1. normalised-convolution pyramid: the known pixels are pushed up a Gaussian
   pyramid together with their coverage weights and pulled back down, so each
   hole pixel is interpolated from progressively larger neighbourhoods around it
   -- a seamless, streak-free continuation of the surrounding shading;
2. grain matching: the local standard deviation of the surrounding
   high-frequency residual is measured, extrapolated into the hole the same way,
   and re-applied as correlated noise, so the filled area does not read as a
   suspiciously clean patch next to grainy paint.
"""

from __future__ import annotations

import cv2
import numpy as np


def _push_pull(values: np.ndarray, weight: np.ndarray, levels: int = 9,
               soft: float = 0.35) -> np.ndarray:
    """Interpolate ``values`` where ``weight`` is 0, from where it is 1."""
    single = values.ndim == 2
    if single:
        values = values[:, :, None]
    w = weight.astype(np.float32)
    iw = [values.astype(np.float32) * w[:, :, None]]
    ww = [w]
    for _ in range(levels):
        if min(iw[-1].shape[:2]) < 4:
            break
        iw.append(cv2.pyrDown(iw[-1]))
        ww.append(cv2.pyrDown(ww[-1]))
        if iw[-1].ndim == 2:  # pyrDown drops a trailing single channel
            iw[-1] = iw[-1][:, :, None]

    est = iw[-1] / np.maximum(ww[-1], 1e-6)[:, :, None]
    for lvl in range(len(iw) - 2, -1, -1):
        h, w_ = ww[lvl].shape
        up = cv2.pyrUp(est, dstsize=(w_, h))
        if up.ndim == 2:
            up = up[:, :, None]
        here = iw[lvl] / np.maximum(ww[lvl], 1e-6)[:, :, None]
        alpha = np.clip(ww[lvl] / soft, 0.0, 1.0)[:, :, None]
        est = alpha * here + (1.0 - alpha) * up
    return est[:, :, 0] if single else est


def _level_fix(img: np.ndarray, hole: np.ndarray, smooth: np.ndarray,
               frac: float = 0.5, lo: float = 8.0, hi: float = 80.0
               ) -> np.ndarray:
    """Re-level ``smooth`` inside ``hole`` to what the hole's boundary implies.

    A hole pixel in the normalised-convolution pyramid is interpolated from a
    neighbourhood as wide as the level it had to climb to find any coverage, so
    the middle of a 200 px hole is roughly a 200 px average centred on it --
    including paint the hole does not touch at all. Over a title laid across a
    dark background that is how a pale wash appears where the artwork is dark:
    highlights and skin a hundred px away are averaged in, and no amount of added
    high-frequency detail can rescue a level that is simply wrong.

    Diffusion has the opposite balance. It marches in from the hole boundary, so
    its level is right by construction -- dark boundary, dark interior -- but its
    mid frequencies drift into the faceted wedges and pale streaks that made the
    pyramid preferable in the first place. Only the disagreement between the two
    at a scale of the hole itself is taken here, which is exactly the level and
    none of the drift; on any hole whose surroundings the pyramid did read
    correctly the two agree and the correction is nothing at all.
    """
    holeb = hole > 0
    inside = cv2.distanceTransform(holeb.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    sigma = float(np.clip(frac * float(inside.max()), lo, hi))
    diff = (cv2.inpaint(img.astype(np.uint8), hole, 7, cv2.INPAINT_TELEA
                        ).astype(np.float32) - smooth) * holeb[:, :, None]
    w = holeb.astype(np.float32)
    num = cv2.GaussianBlur(diff, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return smooth + num / np.maximum(den, 1e-4)[:, :, None]


def inpaint_pyramid(bgr: np.ndarray, hole: np.ndarray, grain: float = 0.8,
                    grain_sigma: float = 0.8, guard: int = 6, win: int = 15,
                    seed: int = 12345, solver: str = "pyramid") -> np.ndarray:
    """Fill ``hole`` (255 = missing) with a smooth continuation plus grain.

    ``guard`` px next to the hole are excluded from the grain measurement: the
    mask ends inside the anti-aliased fringe of the removed text, so those
    pixels still carry glyph-edge contrast and would otherwise inflate the
    estimated grain into visible mottling. The estimate is also clipped to the
    median level of the surrounding paint for the same reason.
    """
    holeb = hole > 0
    if not holeb.any():
        return bgr.copy()

    img = bgr.astype(np.float32)
    known = (~holeb).astype(np.float32)
    smooth = _push_pull(img, known)
    if solver == "level":
        smooth = _level_fix(img, hole, smooth)

    out = img.copy()
    out[holeb] = smooth[holeb]

    if grain > 0:
        # High-frequency energy of paint that is safely away from the glyph.
        dist = cv2.distanceTransform((~holeb).astype(np.uint8), cv2.DIST_L2, 5)
        clean = ((dist > guard).astype(np.float32))
        if clean.sum() < 64:
            clean = known
        resid = img - cv2.GaussianBlur(img, (0, 0), grain_sigma)
        energy = np.einsum("ijk,ijk->ij", resid, resid) / 3.0
        energy = cv2.blur(energy * clean, (win, win)) / np.maximum(
            cv2.blur(clean, (win, win)), 1e-6)
        cap = float(np.median(energy[clean > 0])) * 2.0
        sigma = np.sqrt(np.clip(_push_pull(energy, clean), 0.0, max(cap, 1e-6)))

        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(img.shape[:2]).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), grain_sigma)
        std = float(noise.std())
        if std > 1e-6:
            noise /= std
        out[holeb] += (grain * sigma * noise)[holeb, None]

    res = bgr.copy()
    res[holeb] = np.clip(out[holeb], 0, 255).astype(np.uint8)
    return res
