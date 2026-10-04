"""Mask refinement for text cleaning.

Ported from the standalone ``clean_text.py`` research script. The probability map
comes from either backend (Unet++ glyph segmentation in :mod:`.seg`, PP-OCRv6
block detection in :mod:`.ppocr`); this module turns it into the mask that is
actually repainted.

Every distance here is measured in units of the *stroke radius* of the lettering
it belongs to (:func:`_stroke_radius`), because a band of 18 px is three quarters
of a stroke on a cover title and four times the whole stroke of body text -- at a
fixed size the bands of neighbouring glyphs merge and the mask fuses into a
ribbon over the whole block.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _stroke_radius(mask: np.ndarray, q: float = 90.0) -> float:
    """Half-width of the strokes in ``mask``, in px.

    Every distance in the mask stage -- how far the outline can reach, how wide
    a letter counter can be, how deep a pocket is -- is a property of the
    lettering, not of the page, and a page can carry a 50 px cover title and 3 px
    furigana at once. The 90th percentile of the distance-to-background inside
    the strokes is a robust stand-in for the stroke half-width: insensitive to
    the thin tails of the glyphs, and not thrown off by the single fattest blob.
    """
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    v = d[mask > 0]
    if v.size == 0:
        return 1.0
    return max(1.0, float(np.percentile(v, q)))


def _fill_holes(mask: np.ndarray, max_frac: float = 0.004,
                fill_k: float | None = None) -> np.ndarray:
    """Fill background regions fully enclosed by ``mask``.

    Letter counters (the inside of 'p', of the enclosed part of a CJK glyph)
    carry the outline colour, not the page background, so they belong to the
    region we have to repaint. The gap between two glyphs does not, and in dense
    vertical lettering it is just as enclosed. A counter is as narrow as the
    strokes that form it, so a region is only filled when it is both small and
    *narrow*: its largest inscribed circle must be no wider than ``fill_k``
    stroke radii.

    Whose strokes is the whole question. Measured over the page, the stroke
    radius is whatever the largest lettering on the page happens to be, and a
    manga page carries a 70 px cover title and 4 px script at once -- the case
    :func:`_stroke_radius` warns about. Each hole is therefore judged against the
    lettering that actually encloses it: the strokes of the mask component it is
    a hole in.
    """
    h, w = mask.shape
    limit = max_frac * h * w
    n, labels, stats, _ = cv2.connectedComponentsWithStats(255 - mask, 4)
    if fill_k is not None:
        dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        inner = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        mn, mlab, mstats, _ = cv2.connectedComponentsWithStats(mask, 8)
        sr = np.ones(max(mn, 1), np.float32)
        for c in range(1, mn):
            cx, cy, cw, ch, _ = mstats[c]
            sub = mlab[cy:cy + ch, cx:cx + cw] == c
            sr[c] = max(1.0, float(np.percentile(
                inner[cy:cy + ch, cx:cx + cw][sub], 90.0)))
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches_border or area > limit:
            continue
        if fill_k is not None:
            sel = labels[y:y + bh, x:x + bw] == i
            # Which component encloses the hole: the one holding most of the ring
            # just outside it. A hole can abut two touching components, and then
            # either answer is defensible; the majority is the stable one.
            X0, Y0 = max(0, x - 1), max(0, y - 1)
            X1, Y1 = min(w, x + bw + 1), min(h, y + bh + 1)
            hl = ((labels[Y0:Y1, X0:X1] == i).astype(np.uint8)) * 255
            ring = (cv2.dilate(hl, K3) > 0) & (hl == 0)
            ids = mlab[Y0:Y1, X0:X1][ring]
            ids = ids[ids > 0]
            if ids.size == 0:
                continue
            enclosing = int(np.bincount(ids).argmax())
            if (float(dist[y:y + bh, x:x + bw][sel].max()) >
                    fill_k * sr[enclosing]):
                continue
        out[labels == i] = 255
    return out


def _keep_connected(cand: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Keep only components of ``cand`` that overlap ``core``."""
    n, labels = cv2.connectedComponents(cand, 8)
    keep = np.unique(labels[core > 0])
    keep = keep[keep != 0]
    if keep.size == 0:
        return core.copy()
    lut = np.zeros(n, np.uint8)
    lut[keep] = 255
    return lut[labels]


def _lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _palette(colors: np.ndarray, k: int = 3, min_share: float = 0.08,
             bg: np.ndarray | None = None, bg_tol: float = 30.0
             ) -> np.ndarray | None:
    """Dominant colour centres of ``colors`` (N x 3 Lab), sorted by support.

    ``k`` is deliberately larger than the number of colours a text plate really
    has. The seed pixels are taken next to the glyph, so a large share of them
    are anti-aliased mixtures; with too few centres those mixtures pull each
    centre away from the pure stroke colour -- a white outline came out as
    L=224 instead of 255, far enough to fail the match. Extra centres let the
    mixtures form their own clusters and leave the pure colours pure.

    ``bg`` is the per-pixel background estimate of the same pixels. A centre that
    sits within ``bg_tol`` of the background under it is not a stroke colour at
    all, only the background reproduced with some shading error, and keeping it
    would make the palette match the background it is meant to exclude.
    """
    if len(colors) < 40:
        return None
    z = np.ascontiguousarray(colors, np.float32)
    k = min(k, max(1, len(z) // 20))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lbl, centres = cv2.kmeans(z, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    keep = []
    for i in range(k):
        m = lbl == i
        if m.mean() < min_share:
            continue
        if bg is not None and np.linalg.norm(
                centres[i] - np.median(bg[m], axis=0)) <= bg_tol:
            continue
        keep.append((int(m.sum()), centres[i]))
    if not keep:
        return None
    keep.sort(key=lambda t: -t[0])
    return np.array([c for _, c in keep], np.float32)


def _palette_dist(colors: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Distance from each colour (N x 3 Lab) to the palette.

    The palette is read as the set of stroke colours *and* the mixtures between
    them: a pixel on the border between ink and outline is an alpha composite of
    the two, so it lies on the segment joining them. Measuring to the segments
    rather than only to the centres therefore covers the anti-aliased fringe
    without having to widen the tolerance, which would start matching unrelated
    paint in every direction.
    """
    d = np.min(np.linalg.norm(colors[:, None, :] - pal[None, :, :], axis=2),
               axis=1)
    for i in range(len(pal)):
        for j in range(i + 1, len(pal)):
            seg = pal[j] - pal[i]
            n2 = float(seg @ seg)
            if n2 < 1e-6:
                continue
            t = np.clip((colors - pal[i]) @ seg / n2, 0.0, 1.0)
            proj = pal[i] + t[:, None] * seg[None, :]
            d = np.minimum(d, np.linalg.norm(colors - proj, axis=1))
    return d


def _geodesic(seed: np.ndarray, allowed: np.ndarray, iters: int) -> np.ndarray:
    """Dilate ``seed`` inside ``allowed``, at most ``iters`` px."""
    cur = seed.copy()
    reach = (allowed | seed).astype(np.uint8)
    for _ in range(max(1, iters)):
        nxt = (cv2.dilate(cur.astype(np.uint8), K3) & reach).astype(bool)
        if nxt.sum() == cur.sum():
            break
        cur = nxt
    return cur


def _grow_over_outline(bgr: np.ndarray, core: np.ndarray, halo: int,
                       dev_thr: float, pal_tol: float, gap: int = 12,
                       pocket: int = 9, halo_k: float = 0.8,
                       pocket_k: float = 0.4, ring_k: float = 0.6,
                       glow_k: float = 2.0, glow_thr: float = 12.0,
                       verbose: bool = False) -> np.ndarray:
    """Extend ``core`` across the glyph outline / anti-aliased fringe.

    A band pixel is absorbed when it is close to the text, does *not* look like
    the background there (compared against a smooth estimate extrapolated from
    pixels outside the band), and matches the stroke palette of its own text
    cluster. Growth is geodesic, so only pixels reachable from the text through
    other absorbed pixels are taken -- art that merely happens to sit next to the
    text stays untouched.

    Everything is done per text cluster and at that cluster's own scale, because
    "close to the text" only means anything relative to the stroke width. A band
    of a fixed 18 px is three quarters of a stroke on a cover title but four
    times the whole stroke of body lettering: the bands of neighbouring glyphs
    then merge, the region handed to the background estimator covers the entire
    text block, and the estimate inside it degenerates into a blur extrapolated
    from the speech-bubble border. Estimating the background on a crop, from a
    band only as wide as the lettering, keeps it honest.

    The palette is sampled from a shell *straddling* the detection boundary, not
    only from outside it. How much of the outline the network already returned
    varies with the lettering: for body text it returns the ink and the outline
    is entirely outside, while for an oversized title it returns the whole plate
    and only a few px of the outer edge are left over.
    """
    h, w = core.shape
    lab = _lab(bgr)
    grown = core.copy()

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(core, k), 8)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        pad = halo + 6
        X0, Y0 = max(0, x - pad), max(0, y - pad)
        X1, Y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        sl = (slice(Y0, Y1), slice(X0, X1))
        sub_core = (core[sl] > 0) & (labels[sl] == i)
        if not sub_core.any():
            continue
        cu = (sub_core.astype(np.uint8)) * 255

        sr = _stroke_radius(cu)
        halo_i = int(np.clip(round(halo_k * sr), 2, halo))
        dist = cv2.distanceTransform(255 - cu, cv2.DIST_L2, 5)
        band = (dist > 0) & (dist <= halo_i)

        unknown = cv2.bitwise_or(cu, (band.astype(np.uint8)) * 255)
        bg = cv2.inpaint(bgr[sl], unknown, halo_i + 4, cv2.INPAINT_TELEA)
        bg_lab = _lab(bg)
        sub_lab = lab[sl]
        strong = np.linalg.norm(sub_lab - bg_lab, axis=2) > dev_thr

        ring = max(2, int(round(ring_k * halo_i)))
        inner = cv2.distanceTransform(cu, cv2.DIST_L2, 5)
        sel = (((dist >= 1) & (dist <= ring)) |
               ((inner >= 1) & (inner <= ring)))
        pal = _palette(sub_lab[sel], bg=bg_lab[sel], bg_tol=pal_tol)
        pal_ok = np.zeros(sub_core.shape, bool)
        if pal is not None:
            pal_ok[band] = _palette_dist(sub_lab[band], pal) <= pal_tol

        # Where the band has merged, ``pal_tol`` is the only gate left, and an
        # absolute tolerance is not enough to be one. A 16 px band around script
        # set 25 px apart covers the whole gap; the background estimate inside it
        # is then extrapolated from beyond the text block, so *everything* in the
        # band deviates from it. Those pixels are recognisable without reference
        # to colour: no known paint lies within the band's own width of them,
        # which in an ordinary shell is never true. Only there is the candidate
        # also required to resemble the stroke *more than it resembles its own
        # surroundings*, read from a ring past the band.
        if pal is not None:
            shadow = band & (cv2.distanceTransform(unknown, cv2.DIST_L2, 5) >
                             halo_i)
            out_ring = ((dist > halo_i) & (dist <= halo_i + ring) &
                        (core[sl] == 0))
            art = (_palette(sub_lab[out_ring])
                   if shadow.any() and out_ring.any() else None)
            if art is not None:
                nearer = (_palette_dist(sub_lab[band], pal) <
                          _palette_dist(sub_lab[band], art))
                pal_ok[band] &= nearer | ~shadow[band]

        cur = _geodesic(sub_core, band & strong & pal_ok, halo_i)
        pocket_i = int(np.clip(round(pocket_k * sr), 0, pocket))
        if pocket_i >= 2:
            cur = _close_pockets((cur.astype(np.uint8)) * 255,
                                 strong, pocket_i) > 0
        if glow_k > 0:
            # The aura reaches further than the outline band, so it gets a wider
            # crop of its own; widening the shared one would move the background
            # estimate the palette stage depends on.
            gp = 46
            A0, B0 = max(0, x - gp), max(0, y - gp)
            A1, B1 = min(w, x + bw + gp), min(h, y + bh + gp)
            gsl = (slice(B0, B1), slice(A0, A1))
            seed = np.zeros((B1 - B0, A1 - A0), bool)
            seed[Y0 - B0:Y1 - B0, X0 - A0:X1 - A0] = cur
            lit = _absorb_glow(bgr[gsl], seed, sr, glow_k, glow_thr,
                               verbose=verbose)
            grown[gsl][lit] = 255
        grown[sl][cur] = 255
        if verbose:
            log.debug("  cluster %d px at (%d,%d): stroke r=%.1f halo=%d "
                      "pocket=%d -> %d px", int(sub_core.sum()), x, y, sr,
                      halo_i, pocket_i, int(cur.sum()))

    return grown


def _excess(bgr: np.ndarray, m: np.ndarray, reach: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(dist, band, dL, dC)``: how the page departs from the background.

    The background is estimated by inpainting over the mask *and* everything
    within ``reach`` of it, so the estimate is drawn from past the region under
    test and an aura cannot vote for its own background. Two departures are
    measured, because a glow can be either kind: ``dL`` is the signed luminance
    above the estimate, ``dC`` the distance in Lab's a/b plane, i.e. how much the
    colour has been shifted regardless of whether it got lighter or darker. On a
    greyscale page ``dC`` is exactly zero.
    """
    dist = cv2.distanceTransform(255 - m, cv2.DIST_L2, 5)
    band = (dist > 0) & (dist <= reach)
    unknown = cv2.bitwise_or(m, (band.astype(np.uint8)) * 255)
    bg = cv2.inpaint(bgr, unknown, reach + 4, cv2.INPAINT_TELEA)
    dL = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) -
          cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32))
    dab = _lab(bgr)[..., 1:] - _lab(bg)[..., 1:]
    return dist, band, dL, np.sqrt((dab * dab).sum(2))


def _wall_thickness(bgr: np.ndarray, seed: np.ndarray, grown: np.ndarray,
                    ring: int = 2) -> float:
    """Half-width of the dark structure the growth came to rest against.

    ``-1`` when it came to rest against nothing dark at all. A *drawn* boundary
    -- a speech-bubble border, a panel edge -- is a thin curve two or three px
    wide, so a small value says the growth ran into an enclosure, and that the
    bright region it just filled is an interior rather than an aura that faded
    out over artwork. Ink that belongs to the artwork is bulkier than that: hair,
    shadow, a filled gutter.
    """
    add = grown & ~seed
    if not add.any():
        return -1.0
    front = add & (cv2.erode((grown.astype(np.uint8)) * 255,
                             np.ones((3, 3), np.uint8)) == 0)
    if not front.any():
        return -1.0
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lit = float(np.percentile(lum[add], 75))
    dark = ((lum < np.clip(0.5 * lit, 90.0, 140.0)).astype(np.uint8)) * 255
    kr = np.ones((2 * ring + 1,) * 2, np.uint8)
    touch = (dark > 0) & (cv2.dilate((front.astype(np.uint8)) * 255, kr) > 0)
    if touch.sum() <= 10:
        return -1.0
    return float(np.percentile(
        cv2.distanceTransform(dark, cv2.DIST_L2, 5)[touch], 75))


def _absorb_glow(bgr: np.ndarray, mask: np.ndarray, sr: float,
                 reach_k: float = 2.0, thr: float = 12.0, reach_max: int = 22,
                 rim: float = 0.6, decay: float = 0.25, tail_k: float = 2.0,
                 tail_max: int = 40, wall_max: float = 4.0,
                 verbose: bool = False) -> np.ndarray:
    """Absorb a soft aura drawn around the lettering.

    Text laid over artwork is often given a wide, feathered glow so that it
    stays legible. The glow carries no ink, so the segmentation stops at the
    outline and :func:`_grow_over_outline` has nothing to hold on to either: the
    aura matches neither the stroke palette nor -- past the first couple of px --
    the deviation threshold. What is left behind is a bright ribbon in the exact
    shape of the text block, which is the most conspicuous failure of all,
    because the reconstruction then has to blend *into* it.

    An aura is recognised by its shape, not by its colour. It is a departure from
    the local background that (a) *decays* with the distance from the text, (b)
    covers nearly the whole rim of the lettering, because it was drawn from the
    glyph outline, and (c) is not walled in. All three tests are needed:

    * without the decay test, lettering inside a speech bubble is destroyed --
      the band is wide enough there that the background estimate gets
      extrapolated from the artwork outside the bubble, so the entire interior
      reads as one enormous excess, only a *flat* one;
    * without the rim test, any pale artwork that happens to lie against the text
      is eaten, since it too is brighter close in than the estimate drawn from
      further out;
    * without the enclosure test, a *translucent* bubble laid over artwork is
      destroyed. What gives it away is where the growth stops: against the thin
      drawn curve of the bubble border, whereas an aura stops against nothing.

    The three tests are run twice, once on each departure :func:`_excess`
    measures, and either channel may confirm the cluster. A white glow shows up
    as luminance; a *coloured* one need not show up there at all.

    Recognising the aura and removing it want different geometry, so they get
    their own passes. Once a cluster *is* confirmed, absorption continues over a
    band ``tail_k`` times wider, down to half the threshold, in the channel that
    confirmed it: within a confirmed aura, faint and connected is enough.
    """
    reach = int(np.clip(round(reach_k * sr), 3, reach_max))
    m = (mask.astype(np.uint8)) * 255
    dist, band, dL, dC = _excess(bgr, m, reach)
    near, far = band & (dist <= 2), band & (dist > 0.7 * reach)
    if not near.any() or not far.any():
        return mask
    tried = []
    for what, d in (("L", dL), ("C", dC)):
        p_near = float(np.percentile(d[near], 75))
        p_far = float(np.percentile(d[far], 75))
        cov = float((d[near] >= thr).mean())
        tried.append((what, d, p_near, p_far, cov))
    hit = next((t for t in tried if t[2] >= 2.0 * thr
                and t[3] <= decay * t[2] and t[4] >= rim), None)
    if hit is None:
        return mask
    what, d, p_near, p_far, cov = hit

    tail = min(int(round(tail_k * reach)), tail_max)
    if tail > reach:
        _, band, dL, dC = _excess(bgr, m, tail)
        d = dL if what == "L" else dC
    out = _geodesic(mask, band & (d >= 0.5 * thr), tail)

    wt = _wall_thickness(bgr, mask, out)
    if 0.0 <= wt <= wall_max:
        if verbose:
            log.debug("    no glow (reach=%d d%s %.0f -> %.0f, rim %.2f) -- "
                      "enclosed by a %.0f px line", reach, what, p_near, p_far,
                      cov, 2.0 * wt)
        return mask
    if verbose:
        log.debug("    glow reach=%d/%d d%s %.0f -> %.0f, rim %.2f wall %.0f: "
                  "+%d px", reach, tail, what, p_near, p_far, cov, wt,
                  int(out.sum() - mask.sum()))
    return out


def _close_pockets(mask: np.ndarray, allow: np.ndarray, pocket: int,
                   iters: int = 6) -> np.ndarray:
    """Absorb non-background pockets in concave corners of the mask.

    Inside a narrow bay of the text plate -- the notch between the arms of '<',
    say -- a pixel is a mixture of ink, outline and background, so the geodesic
    growth of :func:`_grow_over_outline` stalls at the mouth of the bay. Those
    pixels are not enclosed either, so hole filling misses them, and what
    survives is a sliver of outline that bleeds back into the reconstruction. A
    morphological closing finds every bay narrower than ``pocket`` px; only the
    pixels there that ``allow`` marks as not-background are taken, and the step
    is repeated so that a wide bay is consumed gradually, always through
    plausible material, instead of by a single kernel large enough to jump over
    artwork.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pocket + 1,) * 2)
    out = mask
    for _ in range(iters):
        added = (cv2.morphologyEx(out, cv2.MORPH_CLOSE, k) > 0) & (out == 0)
        added &= allow
        if not added.any():
            break
        out = out.copy()
        out[added] = 255
    return out


def _drop_specks(mask: np.ndarray, prob: np.ndarray, min_area: int,
                 near_k: float = 3.0, conf_mean: float = 0.6,
                 conf_max: float = 0.9) -> np.ndarray:
    """Remove isolated specks, keeping small marks that really are lettering.

    A flat area threshold is wrong for lettering: the dot of an 'i', a dakuten, a
    comma, a whole furigana glyph are all smaller than the noise the threshold is
    there to remove. Two things separate them from noise. They are usually in
    company -- within a couple of stroke widths of larger text, which JPEG ringing
    and screentone dots are not -- and the network is *certain* about them: on a
    reference page every small mark that was real scored a mean probability of
    0.68-0.86 and peaked at 1.00, while every speck that was not stayed at
    0.38-0.56 and never peaked above 0.64. Either signal is enough to keep it.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    areas = stats[:, cv2.CC_STAT_AREA]
    big = np.zeros(n, bool)
    big[1:] = areas[1:] >= min_area
    if not big.any():
        return np.zeros_like(mask)
    lut = np.zeros(n, np.uint8)
    lut[big] = 255
    out = lut[labels]

    small = [i for i in range(1, n)
             if not big[i] and areas[i] >= max(6, min_area // 8)]
    if not small:
        return out
    reach = near_k * _stroke_radius(out)
    dist = cv2.distanceTransform(255 - out, cv2.DIST_L2, 5)
    for i in small:
        sel = labels == i
        if (float(dist[sel].min()) <= reach or prob[sel].mean() >= conf_mean
                or prob[sel].max() >= conf_max):
            out[sel] = 255
    return out


def _drop_unsupported(mask: np.ndarray, agree: np.ndarray,
                      min_frac: float = 0.25, min_px: int = 200,
                      vote_area: int = 600, vote_stroke: float = 3.5,
                      keep_area: int = 400, keep_dist: float = 16.0,
                      keep_agree: int = 30,
                      verbose: bool = False) -> np.ndarray:
    """Remove detections that only one inference scale believes in.

    The network occasionally reads a piece of artwork as a glyph -- a hair curl
    with a bright highlight behind it, a lip. Such a detection is not stable
    under rescaling, while real text is: a glyph is still a glyph at 60% size.
    A component is therefore kept when a good share of it is confirmed by
    several scales, and otherwise only when it sits right next to a confirmed
    component -- the dot of an 'i', a letter counter, the tail of a stroke -- and
    is either small or has some cross-scale support of its own.

    The vote can only be *evidence* for something the coarse passes were able to
    see at all. A mark with a stroke half-width of 2 px is a single pixel wide at
    scale 0.4 and gone at 0.6, so it can never collect two votes no matter how
    plainly it is text. Components that are both small and thin-stroked are
    therefore exempt; the artwork this filter exists to remove is neither.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    supported = np.zeros(mask.shape, np.uint8)
    weak = []
    for i in range(1, n):
        sel = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if int(agree[sel].sum()) >= min_px or agree[sel].mean() >= min_frac:
            supported[sel] = 255
        elif area < vote_area and _stroke_radius(
                (sel.astype(np.uint8)) * 255) <= vote_stroke:
            supported[sel] = 255  # too thin to be visible at the coarse scales
        else:
            weak.append((i, area, sel))

    if not weak:
        return mask
    if not supported.any():
        return np.zeros_like(mask)

    dist = cv2.distanceTransform(255 - supported, cv2.DIST_L2, 5)
    out = supported
    for i, area, sel in weak:
        near = float(dist[sel].min())
        votes = int(agree[sel].sum())
        if near <= keep_dist and (area <= keep_area or votes >= keep_agree):
            out[sel] = 255
        elif verbose:
            log.debug("  dropped unstable detection %d px at (%d,%d), %.0f px "
                      "from confirmed text, %d agreeing px", area,
                      stats[i, 0], stats[i, 1], near, votes)
    return out


def refine_mask(bgr: np.ndarray, prob: np.ndarray, thr: float = 0.35,
                close_k: int = 3, min_area: int = 40, halo: int = 18,
                halo_dev: float = 12.0, pal_tol: float = 30.0, grow: int = 2,
                gap: int = 12, pocket: int = 9, fill_k: float = 3.0,
                glow_k: float = 2.0,
                agree: np.ndarray | None = None,
                verbose: bool = False) -> np.ndarray:
    """Turn a probability map into a mask that also covers glyph outlines."""
    mask = ((prob > thr).astype(np.uint8)) * 255

    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if min_area > 0:
        mask = _drop_specks(mask, prob, min_area)
        if not mask.any():
            return mask

    if agree is not None:
        mask = _drop_unsupported(mask, agree, verbose=verbose)
        if not mask.any():
            return mask

    # Distances below are all relative to the lettering: a letter counter is as
    # narrow as the strokes enclosing it, whatever the type size.
    core = _fill_holes(mask, fill_k=fill_k)

    if halo > 0:
        mask = _grow_over_outline(bgr, core, halo, halo_dev, pal_tol, gap,
                                  pocket, glow_k=glow_k, verbose=verbose)
    else:
        mask = core

    mask = _fill_holes(mask, fill_k=fill_k)
    if grow > 0:
        mask = cv2.dilate(mask, K3, iterations=grow)
        mask = _fill_holes(mask, fill_k=fill_k)
    return mask


def overlay(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """The mask drawn over the page -- diagnostics only."""
    vis = bgr.copy()
    red = np.zeros_like(bgr)
    red[:, :, 2] = 255
    sel = mask > 0
    vis[sel] = (bgr[sel] * (1 - alpha) + red[sel] * alpha).astype(np.uint8)
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, K3)
    vis[edge > 0] = (0, 255, 0)
    return vis
