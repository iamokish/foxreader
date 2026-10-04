"""Hybrid fill: pyramid shading + transported edge detail.

Neither reconstructor alone is good enough on soft, painterly artwork:

* the normalised-convolution pyramid (``smoothfill``) continues the surrounding
  shading exactly and can never invent a colour, but it leaves a texture-free
  haze where the text used to be, and structures crossing the hole (a fold, a
  hair strand) dissolve;
* diffusion inpainting (``cv2.inpaint``) transports those structures across the
  hole, but its low frequencies drift, which shows as faceted wedges and pale
  streaks;
* patch synthesis (``patchsynth``) returns real paint texture, but over a hole
  as wide as a glyph it mismatches the shading badly -- bright blocks and
  stair-stepped ridges.

So each contributes only what it is good at: the shading comes from the pyramid
fill and only the *high-frequency* part of a second reconstructor -- the
continued edges and grain -- is added on top. The cut-off follows the hole size,
because that is the scale above which the second reconstructor stops being
reliable, and the added amplitude is capped by the amount of detail actually
present in the paint around the hole, so a spurious ridge cannot survive.
"""

from __future__ import annotations

import cv2
import numpy as np

from fox_reader.clean import caps
from fox_reader.clean.smoothfill import _push_pull, inpaint_pyramid

DETAIL_SOURCES = ("telea", "ns", "patch", "fsr", "shiftmap")


def _detail_image(bgr: np.ndarray, hole: np.ndarray, source: str,
                  **patch_kw) -> np.ndarray:
    """The reconstruction whose *high frequencies* the hybrid fill will keep.

    Only the detail term comes from here, so an ``xphoto`` source that this
    OpenCV build cannot run degrades to Telea rather than failing the fill: the
    pyramid shading -- which is what the hybrid is mostly made of -- is
    unaffected, and Telea is the closest built-in that still transports edges
    across the hole. See :mod:`.caps` for why a plain ``hasattr`` guard is not
    enough to detect that.
    """
    if source == "patch":
        from fox_reader.clean.patchsynth import inpaint_patchsynth  # optional, slow
        return inpaint_patchsynth(bgr, hole, **patch_kw)
    if source == "shiftmap":
        # Offset optimisation: the hole is covered by whole displaced pieces of
        # the same picture, so hair strands, folds and ornament come back as
        # themselves at every scale instead of dissolving. What it gets wrong is
        # the level -- a transported piece carries the tone of wherever it came
        # from, which shows as a block-shaped step across an otherwise smooth
        # gradient. Taking only its high frequencies here discards exactly that.
        flag = caps.xphoto_flag("shiftmap")
        if flag is None:
            caps.note_fallback("shiftmap", "Telea diffusion")
        else:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
            dst = np.zeros_like(lab)
            cv2.xphoto.inpaint(lab, 255 - hole, dst, flag)
            return cv2.cvtColor(dst, cv2.COLOR_Lab2BGR)
    elif source == "fsr":
        # Frequency-selective reconstruction extrapolates the band-limited
        # signal, so a straight edge or a run of hatching continues as itself
        # instead of dissolving (diffusion) or being re-invented from elsewhere
        # (patch synthesis). That is what line art is mostly made of.
        flag = caps.xphoto_flag("fsr-fast")
        if flag is None:
            caps.note_fallback("fsr-fast", "Telea diffusion")
        else:
            src = bgr.copy()
            valid = 255 - hole
            if valid.any():
                src[hole > 0] = np.array(cv2.mean(bgr, valid)[:3],
                                         np.float32).astype(np.uint8)
            dst = np.zeros_like(src)
            cv2.xphoto.inpaint(src, valid, dst, flag)
            return dst
    flag = cv2.INPAINT_NS if source == "ns" else cv2.INPAINT_TELEA
    return cv2.inpaint(bgr, hole, 7, flag)


def inpaint_transport(bgr: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Offset-transported fill with its block-level tone drift taken out.

    Over a hole two hundred px thick there is nothing to interpolate: every
    boundary-driven reconstructor can only produce a smear, and on a cover title
    laid across hair and ornament a smear is the one thing that cannot pass for
    artwork. Offset optimisation is the only method here that returns structure at
    that scale, because it does not interpolate at all -- it covers the hole with
    whole displaced pieces of the same picture.

    What it gets wrong is the level. A transported piece carries the tone of
    wherever it came from, so it lands as a block-shaped step across an otherwise
    smooth gradient. Taking only its high frequencies is not the answer -- a step
    edge is broadband, and the sharp part of it survives any high-pass -- but the
    *drift* is exactly what :func:`smoothfill._level_fix` removes, since diffusion
    from the hole boundary has the level right and the structure wrong.

    Offset optimisation lives in ``cv2.xphoto``, which not every OpenCV build
    ships (see :mod:`.caps`). Where it is missing this degrades to diffusion plus
    the same level fix -- a smear, which is the one thing transport exists to
    avoid -- so the method is not offered by ``/api/clean/methods`` on such a
    build and :func:`.fill.inpaint_regions` will not escalate to it. The path
    keeps working only so a project saved elsewhere still renders.
    """
    from fox_reader.clean.smoothfill import _level_fix  # local: only this path needs it

    holeb = hole > 0
    if not holeb.any():
        return bgr.copy()
    tex = _detail_image(bgr, hole, "shiftmap").astype(np.float32)
    out = _level_fix(bgr.astype(np.float32), hole, tex)
    res = bgr.copy()
    res[holeb] = np.clip(out[holeb], 0, 255).astype(np.uint8)
    return res


def _local_amplitude(ref: np.ndarray, holeb: np.ndarray, sigma: float,
                     guard: int = 4, win: int = 25, cap_k: float = 4.0
                     ) -> np.ndarray:
    """Std of ``ref``'s detail at scale ``sigma``, extrapolated into the hole.

    ``guard`` px next to the hole are ignored: the mask ends inside the
    anti-aliased fringe of the removed text, so those pixels still carry
    glyph-edge contrast and would overstate how busy the paint really is.
    """
    dist = cv2.distanceTransform((~holeb).astype(np.uint8), cv2.DIST_L2, 5)
    clean = (dist > guard).astype(np.float32)
    if clean.sum() < 64:
        clean = (~holeb).astype(np.float32)
    hi = ref - cv2.GaussianBlur(ref, (0, 0), sigma)
    energy = np.einsum("ijk,ijk->ij", hi, hi) / 3.0
    energy = cv2.blur(energy * clean, (win, win)) / np.maximum(
        cv2.blur(clean, (win, win)), 1e-6)
    cap = float(np.median(energy[clean > 0])) * cap_k
    return np.sqrt(np.clip(_push_pull(energy, clean), 0.0, max(cap, 1e-6)))


def inpaint_hybrid(bgr: np.ndarray, hole: np.ndarray, source: str = "telea",
                   detail: float = 1.0, cutoff: float = 0.6,
                   cutoff_min: float = 3.0, cutoff_max: float = 8.0,
                   clip_k: float = 3.0, fade: float = 1.5, grain: float = 0.8,
                   solver: str = "pyramid", **patch_kw) -> np.ndarray:
    """Fill ``hole`` (255 = missing) with pyramid shading plus edge detail."""
    holeb = hole > 0
    if not holeb.any():
        return bgr.copy()

    base = inpaint_pyramid(bgr, hole, grain=grain,
                           solver=solver).astype(np.float32)
    if detail <= 0:
        res = bgr.copy()
        res[holeb] = np.clip(base[holeb], 0, 255).astype(np.uint8)
        return res

    # Hole half-width sets the scale that the detail source can be trusted at.
    inside = cv2.distanceTransform((holeb.astype(np.uint8)) * 255,
                                   cv2.DIST_L2, 5)
    sigma = float(np.clip(cutoff * float(inside.max()), cutoff_min, cutoff_max))

    tex = _detail_image(bgr, hole, source, **patch_kw).astype(np.float32)
    hi = tex - cv2.GaussianBlur(tex, (0, 0), sigma)
    lim = (clip_k * _local_amplitude(bgr.astype(np.float32), holeb, sigma)
           )[:, :, None]
    hi = np.clip(hi, -lim, lim)

    # Detail is credible next to known paint and invented deep inside a wide
    # hole, where diffusion only ever produces a ghost of the hole's own shape,
    # so its weight decays with the distance from the hole boundary.
    if fade > 0:
        hi *= np.exp(-inside / (fade * sigma))[:, :, None]

    # The base must contribute *only* the frequencies below the cut-off. Right
    # at the hole boundary the pyramid fill reproduces the image exactly, detail
    # included, so adding `hi` on top of it there would count the same detail
    # twice and leave a glyph-shaped rim around every filled area.
    out = cv2.GaussianBlur(base, (0, 0), sigma) + detail * hi

    res = bgr.copy()
    res[holeb] = np.clip(out[holeb], 0, 255).astype(np.uint8)
    return res
