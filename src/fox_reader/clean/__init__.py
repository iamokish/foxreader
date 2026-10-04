"""Text cleaning: find what to remove, then rebuild the pixels underneath.

Two stages, either of which can be used on its own:

* **detection** -- where the lettering is. Three sources, exposed to the user as
  the clean *method*:

  ``region``
      The user's own selection, used verbatim. No model runs, so it is instant
      and it cannot miss text; it removes everything inside the shape, which is
      exactly what "clean this area" means.
  ``ppocr``
      :mod:`.ppocr` -- the PP-OCRv6 detector FoxReader already has loaded, one
      quad per text line. About two seconds a page.
  ``textseg``
      :mod:`.seg` -- glyph-level Unet++ segmentation, refined by :mod:`.mask`.
      The most precise, and the slowest (20-40 s a page on the CPU).

* **reconstruction** -- :mod:`.fill`, with the method chosen from
  :data:`INPAINT_METHODS`. Everything from a plain diffusion inpaint to
  offset-transported patch synthesis; the default trades a second a cluster for
  results that hold up on painted artwork.

:mod:`fox_reader.services.clean_service` is the orchestrator that turns a page
plus a list of per-entry jobs into one cleaned image.
"""

from __future__ import annotations

from fox_reader.clean.caps import fill_available, usable_fills
from fox_reader.clean.fill import (
    DEFAULT_INPAINT_METHOD,
    INPAINT_METHODS,
    inpaint_regions,
)
from fox_reader.clean.mask import overlay, refine_mask
from fox_reader.clean.tuning import (
    DEFAULT_SPEED,
    DEFAULT_TILE,
    DEFAULT_TTA,
    SPEED_ORDER,
    SPEEDS,
    TILE_ORDER,
    TILE_OVERLAPS,
    normalise_speed,
    normalise_tile,
    overlap_for,
    passes_for,
)

#: Detection sources, in the order the UI lists them. ``region`` is the default:
#: it is instant, needs no model, and is what a user drawing a box means.
CLEAN_METHODS: tuple[str, ...] = ("region", "ppocr", "textseg")

DEFAULT_CLEAN_METHOD = "region"

#: Which detector each tuning knob actually reaches. The panel uses this to offer
#: only controls that do something for the selected method -- a Glow switch under
#: PP-OCR would be a lie, since box coverage has no halo to absorb.
#:
#: ``fill`` and ``transport`` are absent because they belong to the second stage
#: and apply to every method, ``region`` included.
METHOD_KNOBS: dict[str, tuple[str, ...]] = {
    "region": (),
    "ppocr": (),
    "textseg": ("glow", "speed", "tta", "tile"),
}

__all__ = [
    "CLEAN_METHODS",
    "DEFAULT_CLEAN_METHOD",
    "DEFAULT_INPAINT_METHOD",
    "DEFAULT_SPEED",
    "DEFAULT_TILE",
    "DEFAULT_TTA",
    "INPAINT_METHODS",
    "METHOD_KNOBS",
    "SPEEDS",
    "SPEED_ORDER",
    "TILE_ORDER",
    "TILE_OVERLAPS",
    "fill_available",
    "inpaint_regions",
    "normalise_speed",
    "normalise_tile",
    "overlap_for",
    "overlay",
    "passes_for",
    "refine_mask",
    "usable_fills",
]
