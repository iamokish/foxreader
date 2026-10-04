"""Detector presets: how many forward passes to spend, and on what tiles.

Split out of :mod:`fox_reader.clean.seg` deliberately. ``seg`` imports torch and
:mod:`fox_reader.device`, and importing ``device`` probes the hardware -- so
anything that only wants to *name* a preset (a request model, the capability
endpoint, a job normaliser, a health check) must not have to pay for that.
Nothing here imports anything outside the standard library, so it is safe to
import from a request handler and from module scope anywhere.

The numbers are the ones the reference CLI uses (``tt/seg.py``); they are not
tuning knobs invented here. The default is ``fastest``: it is within 6.2 % of
``best`` on the reference masks for an eighth of the passes, which is the right
trade for an interactive panel where the user can re-run.
"""

from __future__ import annotations

#: ``name -> (flip TTA, scales, tone-inverted pass)``.
#:
#: Passes = scales x polarities x flips. The three multipliers are not equally
#: worth their time, so they are given up in a fixed order:
#:
#: * flips first -- flip TTA only averages four views of the same evidence;
#: * then polarity -- the inverted pass is the only way light lettering on dark
#:   ground is seen, so it goes second;
#: * then scales -- scales are what make an oversized cover title detectable at
#:   all *and* what the cross-scale vote filter votes with.
#:
#: ``single`` is last for a reason: with one scale there is no vote left to
#: filter artwork with, and :func:`.seg.detect` then reports no agreement at all
#: (see its docstring). It is here because it is the cheapest thing the model can
#: be asked, not because the answer is good.
SPEEDS: dict[str, tuple[bool, tuple[float, ...], bool]] = {
    "best": (True, (1.0, 0.6, 0.4), True),       # 24 passes
    "fast": (False, (1.0, 0.6, 0.4), True),      #  6
    "fastest": (False, (1.0, 0.6, 0.4), False),  #  3
    "single": (False, (1.0,), False),            #  1
}

#: Slowest first, so a picker reads best -> cheapest.
SPEED_ORDER: tuple[str, ...] = ("best", "fast", "fastest", "single")

DEFAULT_SPEED = "fastest"

#: Flip TTA is on unless asked otherwise; it costs 4x and is the first thing a
#: speed preset gives up, so only ``best`` has anything for this to switch off.
DEFAULT_TTA = True

#: Tile size in px -> overlap in px. ``0`` means "the whole image at once", and
#: carries an overlap anyway so that switching to a tiled size and back does not
#: lose the number.
#:
#: The overlap has to be a large fraction of the tile: the tiled path blends with
#: a cosine window, and a glyph split across a seam is only reconstructed from
#: the tile that holds enough of it. 48 px on a 256 px tile is already thin --
#: that size exists for machines that cannot hold a page of activations at once,
#: not because it detects better.
TILE_OVERLAPS: dict[int, int] = {
    0: 192,
    256: 48,
    512: 96,
    1024: 192,
    2048: 192,
}

#: Ascending, with "off" first -- the default, and the only one that sees the
#: page whole.
TILE_ORDER: tuple[int, ...] = (0, 256, 512, 1024, 2048)

DEFAULT_TILE = 0


def normalise_speed(speed: object) -> str:
    """``speed`` if it names a preset, else :data:`DEFAULT_SPEED`.

    Anything unrecognised falls back rather than raising: these values arrive
    from saved projects and from the network, and a stale preset name is not a
    reason to fail a clean run.
    """
    return speed if isinstance(speed, str) and speed in SPEEDS else DEFAULT_SPEED


def normalise_tile(tile: object) -> int:
    """``tile`` if it is one of the offered sizes, else :data:`DEFAULT_TILE`.

    Only the tabulated sizes are accepted, because each one is paired with an
    overlap; an arbitrary size would have no overlap to go with it. ``bool`` is
    rejected explicitly -- it is an ``int`` subclass, and ``True`` is not tile 1.
    """
    if isinstance(tile, bool) or not isinstance(tile, int):
        return DEFAULT_TILE
    return tile if tile in TILE_OVERLAPS else DEFAULT_TILE


def overlap_for(tile: object) -> int:
    """The overlap that goes with ``tile``, normalising ``tile`` first."""
    return TILE_OVERLAPS[normalise_tile(tile)]


def passes_for(speed: object, tta: bool = DEFAULT_TTA) -> int:
    """How many forward passes ``speed`` costs, at most.

    At most, because :func:`.seg.scale_maps` skips the inverted pass entirely on
    a page with no dark ground -- which halves the count when it fires. Used for
    the "24 passes" hint beside the picker, so an over-estimate is the safe
    direction.
    """
    preset_tta, scales, polarity = SPEEDS[normalise_speed(speed)]
    return len(scales) * (2 if polarity else 1) * (4 if preset_tta and tta else 1)
