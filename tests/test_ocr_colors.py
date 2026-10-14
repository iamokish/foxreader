"""What :mod:`fox_reader.ocr_colors` reads off a region.

Synthetic plates and bars rather than rendered type: the question here is
"which of these three populations of pixels is the ink, which is the outline and
which is the paper", and a bar of a known width answers it without a font, a
hinting setting or a platform's rasteriser in the way. Every case is one the real
thing has to get right:

* black body text on paper -- the only case the old plate-and-contrast rule
  already handled, which has to keep coming out the same
* white lettering with a black outline over artwork -- the case that came out
  black-on-grey before, and the reason this module exists
* a region holding no lettering at all, and one whose colours are not separable,
  which must answer "not measured" rather than a colour
* the two inputs that narrow the read: the polygon mask of a freeform capture,
  and the detector's quads
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fox_reader import ocr_colors


# ── helpers ───────────────────────────────────────────────────────────────────

def _plate(size: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    """A flat RGB field."""
    field = np.zeros((size[1], size[0], 3), np.uint8)
    field[:, :] = color
    return field


def _bars(field: np.ndarray, color: tuple[int, int, int], *, width: int,
          gap: int, top: int, bottom: int, left: int, right: int) -> None:
    """Paint vertical bars of ``width`` px, in place. Stands in for strokes."""
    x = left
    while x + width <= right:
        field[top:bottom, x:x + width] = color
        x += width + gap


def _quad(left: int, top: int, right: int, bottom: int) -> np.ndarray:
    """One detector quad, in the order PP-OCR emits them."""
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]],
                    np.float32)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    body = hex_color.lstrip("#")
    return tuple(int(body[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _near(hex_color: str | None, expected: tuple[int, int, int],
          tolerance: int = 12) -> bool:
    """Whether a reported colour is ``expected``, give or take a little.

    Exact equality would be testing the resize filter and the 5-bit quantiser
    rather than the separation, which is what matters.
    """
    if hex_color is None:
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(_rgb(hex_color), expected))


# ── the plain case ────────────────────────────────────────────────────────────

def test_black_body_text_on_paper():
    """The case the old rule already got right, and must keep getting right.

    Paper white, ink black, and -- the part that is easy to get wrong -- *no*
    outline: the rim of a black bar on white paper is a mixture of the two, and
    reporting that mixture as an outline colour would draw a grey halo around
    every line of body text.
    """
    field = _plate((160, 90), (255, 255, 255))
    _bars(field, (0, 0, 0), width=5, gap=7, top=20, bottom=70, left=15, right=145)

    read = ocr_colors.estimate(Image.fromarray(field))

    assert _near(read.bg_color, (255, 255, 255))
    assert _near(read.font_color, (0, 0, 0))
    assert read.stroke_color is None, (
        f"a plain black-on-white line reported an outline ({read.stroke_color})"
    )


def test_coloured_ink_is_reported_as_itself():
    """Not as whichever of black or white contrasts with the paper, which is all
    the old rule could say."""
    field = _plate((160, 90), (250, 248, 240))
    _bars(field, (178, 24, 36), width=6, gap=8, top=20, bottom=70, left=15, right=145)

    read = ocr_colors.estimate(Image.fromarray(field))

    assert _near(read.font_color, (178, 24, 36))


# ── the case this module was written for ──────────────────────────────────────

def test_white_lettering_with_a_black_outline_over_artwork():
    """SFX lettering. The plate is artwork, the ink is white and there really is
    an outline -- three populations, and the old rule could name none of them.

    Bars 15 px wide in black with 5 px of white down the middle, so the ink is
    the interior of the blob and the outline is its rim: exactly the geometry of
    an outlined glyph, at a width where the distance transform can tell the two
    apart.
    """
    field = _plate((200, 120), (120, 118, 125))
    _bars(field, (0, 0, 0), width=15, gap=9, top=25, bottom=95, left=20, right=180)
    _bars(field, (255, 255, 255), width=5, gap=19, top=25, bottom=95, left=25,
          right=180)

    read = ocr_colors.estimate(Image.fromarray(field))

    assert _near(read.bg_color, (120, 118, 125))
    assert _near(read.font_color, (255, 255, 255))
    assert _near(read.stroke_color or "", (0, 0, 0)), (
        f"the outline came back as {read.stroke_color}, not black"
    )


# ── when it must decline to answer ────────────────────────────────────────────

def test_a_region_with_no_lettering_reports_only_its_background():
    """A clean-only entry, or a bubble whose text the user has already removed.

    The background is still worth having -- it is what the plate and the contrast
    fall back to -- but inventing a font colour out of JPEG noise would hand the
    renderer a hint that is pure chance.
    """
    field = _plate((120, 90), (240, 236, 228))
    rng = np.random.default_rng(7)
    noise = rng.integers(-4, 5, field.shape, dtype=np.int16)
    noisy = np.clip(field.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    read = ocr_colors.estimate(Image.fromarray(noisy))

    assert _near(read.bg_color, (240, 236, 228))
    assert read.font_color is None
    assert read.stroke_color is None


def test_artwork_with_no_dominant_colour_is_not_measured():
    """A capture that landed on painted artwork rather than on lettering.

    Nothing here is modal, so almost every pixel sits far from whichever colour
    wins -- and a split that calls 90% of a region "ink" has not found text, it
    has found a gradient. Reported as unmeasured, so both renderers fall back to
    the rule they had.
    """
    ramp = np.linspace(0, 255, 120, dtype=np.uint8)
    field = np.repeat(ramp[None, :, None], 90, axis=0).repeat(3, axis=2)
    field[:, :, 1] = field[:, :, 1] // 2  # a second axis, so it is not grey

    read = ocr_colors.estimate(Image.fromarray(field))

    assert read.font_color is None
    assert read.stroke_color is None


def test_a_flat_crop_reports_its_colour_and_nothing_else():
    """The smallest thing that is still a region. Not an error -- there is a
    background and it is knowable; there is just no lettering in it."""
    read = ocr_colors.estimate(np.zeros((4, 4), np.uint8))

    assert read.bg_color == "#000000"
    assert read.font_color is None
    assert read.stroke_color is None


@pytest.mark.parametrize("bad", [
    None,
    "not an image",
    np.zeros((2, 2, 3), np.uint8),
    np.zeros((6, 6, 2), np.uint8),
    object(),
])
def test_nothing_unmeasurable_raises(bad):
    """An OCR result must never be lost to a colour that could not be measured."""
    assert ocr_colors.estimate(bad) == ocr_colors.NOTHING


# ── what narrows the read ─────────────────────────────────────────────────────

def test_the_mask_keeps_the_freeform_filler_out_of_the_plate():
    """`/api/ocr_freeform` composites the selection onto white and crops to its
    bounding box, so most of what arrives can be filler. Measured without the
    mask the plate comes out white; with it, the panel's own colour."""
    field = _plate((120, 90), (255, 255, 255))
    panel = (40, 44, 70)
    field[10:80, 10:60] = panel
    _bars(field, (235, 230, 225), width=4, gap=6, top=20, bottom=70, left=15,
          right=55)

    mask = np.zeros((90, 120), np.uint8)
    mask[10:80, 10:60] = 255

    masked = ocr_colors.estimate(Image.fromarray(field),
                                 mask=Image.fromarray(mask, "L"))
    unmasked = ocr_colors.estimate(Image.fromarray(field))

    assert _near(masked.bg_color, panel)
    assert _near(masked.font_color, (235, 230, 225))
    assert _near(unmasked.bg_color, (255, 255, 255)), (
        "the unmasked read was supposed to be fooled by the filler; if it is "
        "not, this test no longer shows that the mask is what fixes it"
    )


def test_the_detector_quads_keep_distant_artwork_out_of_the_plate():
    """A rectangle dragged over a panel holds far more artwork than lettering.

    The plate is read from the paper *around* the glyphs, not from whatever else
    is in the crop, so a block of unrelated colour taking up half the region
    cannot become the background the text is said to sit on.
    """
    field = _plate((240, 160), (255, 255, 255))
    field[0:160, 150:240] = (200, 30, 40)  # artwork, well away from the text
    _bars(field, (0, 0, 0), width=5, gap=7, top=40, bottom=110, left=20, right=130)

    quads = [_quad(18, 38, 132, 112)]
    read = ocr_colors.estimate(Image.fromarray(field), polys=quads)

    assert _near(read.bg_color, (255, 255, 255))
    assert _near(read.font_color, (0, 0, 0))


def test_quads_that_cannot_be_read_are_simply_ignored():
    """They come from a model's output, so they can be any shape at all. A bad
    one must degrade to the whole-crop read, not fail it."""
    field = _plate((160, 90), (255, 255, 255))
    _bars(field, (0, 0, 0), width=5, gap=7, top=20, bottom=70, left=15, right=145)

    read = ocr_colors.estimate(
        Image.fromarray(field),
        polys=[np.array([[0.0, 0.0], [1.0, 1.0]]),            # too few points
               np.array([[np.nan, 0.0], [1.0, 1.0], [2.0, 2.0]])],
    )

    assert _near(read.bg_color, (255, 255, 255))
    assert _near(read.font_color, (0, 0, 0))


def test_a_huge_region_is_measured_at_a_bounded_cost():
    """A page-sized freeform selection is box-filtered down first. The answer has
    to survive that, because it is the answer for every large capture.

    Handed in as a PIL image, which is what the routes pass: the array behind one
    is read-only, so this is also the case where the resize runs on a buffer it
    may not write to.
    """
    field = _plate((2400, 1600), (252, 250, 245))
    _bars(field, (20, 18, 25), width=40, gap=60, top=300, bottom=1300, left=200,
          right=2200)

    read = ocr_colors.estimate(Image.fromarray(field))

    assert _near(read.bg_color, (252, 250, 245))
    assert _near(read.font_color, (20, 18, 25))


# ── the payload the routes put on the wire ────────────────────────────────────

def test_the_payload_is_named_after_the_fields_it_feeds():
    """One assignment per key on the page side -- see `applyDetectedColors`."""
    read = ocr_colors.RegionColors("#010203", "#040506", "#070809")
    assert read.payload() == {
        "font_color": "#010203",
        "stroke_color": "#040506",
        "bg_color": "#070809",
    }
    assert read.measured is True
    assert ocr_colors.NOTHING.measured is False
    assert ocr_colors.NOTHING.payload() == {
        "font_color": None, "stroke_color": None, "bg_color": None,
    }
