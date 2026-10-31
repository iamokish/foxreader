"""Compare frontend layout (textFit) against the backend typesetter (typeset).

Focuses on vertical placement: per-line baselines and block height.
Renders identical text through both paths and asserts baselines match within 1px.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
from PIL import ImageFont

from fox_reader.typeset import (
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
    _ascent,
    _fit_font_size,
    _layout,
    _length,
    _line_height,
    _load_font,
    _raw_length,
    _spacing_for,
    line_metrics_for,
)
from tests.font_fixtures import TEXT_FIXTURE_NAMES

FONT_PATH = "fonts/ComicMono.ttf"


@dataclass
class RecordedLine:
    baseline_y: float
    text: str
    anchor: str
    x: float


class RecorderDraw:
    def __init__(self) -> None:
        self.lines: list[RecordedLine] = []

    def text(self, xy, text, anchor="la", **kw) -> None:
        self.lines.append(
            RecordedLine(baseline_y=float(xy[1]), text=text, anchor=anchor, x=float(xy[0]))
        )


def backend_baselines(text, size, align, box):
    """Run backend _draw_block; return (size, baselines).

    Pillow ``anchor="la"`` records ``y`` as the ascender top, so add the face's
    ascent to get the baseline (matching SVG's default baseline origin).
    """
    font = _load_font(FONT_PATH, size if size else 16)
    draw = RecorderDraw()
    x0, y0, x1, y1 = box
    pad_x = max(5, int((x1 - x0) * 0.12))
    pad_y = max(4, int((y1 - y0) * 0.10))
    max_w = max(1, x1 - x0 - 2 * pad_x)
    max_h = max(1, y1 - y0 - 2 * pad_y)
    text_box = (x0 + pad_x, y0 + pad_y, x0 + pad_x + max_w, y0 + pad_y + max_h)

    if size:
        fitted = size
    else:
        fitted = _fit_font_size(FONT_PATH, text, align, max_w, max_h)

    font = _load_font(FONT_PATH, fitted)
    ascent, _descent = font.getmetrics()
    lines = _layout(font, text, max_w, align)
    _draw_block_backend(draw, font, fitted, lines, align, text_box)
    # Convert recorded ascender-top y to baseline y.
    return fitted, [
        RecordedLine(baseline_y=line.baseline_y + ascent, text=line.text, anchor=line.anchor, x=line.x)
        for line in draw.lines
    ]


def _draw_block_backend(draw, font, size, lines, align, box):
    """Mirror of typeset._draw_block single-face path."""
    x0, y0, x1, y1 = box
    max_w = x1 - x0
    spacing = _spacing_for(size)
    line_h = _line_height(font, size)
    advance = line_h + spacing
    block_h = ((len(lines) - 1) * advance + line_h) if lines else 0
    y = y0 + (y1 - y0 - block_h) / 2.0
    for line in lines:
        if align == "center":
            draw.text((x0 + max_w / 2.0, y), line, anchor="ma")
        elif align == "right":
            draw.text((x1, y), line, anchor="ra")
        else:
            draw.text((x0, y), line, anchor="la")
        y += advance


# ---------------------------------------------------------------------------
# Frontend vertical placement (port of textFit.drawFitLayout)


def font_metrics(font_path, size):
    """Backend font.getmetrics() -> (ascent, descent)."""
    f = ImageFont.truetype(font_path, size)
    a, d = f.getmetrics()
    return a, d


def frontend_baselines(text, size, align, box):
    """Port of textFit.drawFitLayout vertical placement.

    Mirrors: top = y0 + (h - blockH)/2; baseline = top + ascentFor(size);
             advance = advanceFor(size).
    """
    body = text
    x0, y0, x1, y1 = box
    pad_x = max(5, int((x1 - x0) * 0.12))
    pad_y = max(4, int((y1 - y0) * 0.10))
    max_w = max(1, x1 - x0 - 2 * pad_x)
    max_h = max(1, y1 - y0 - 2 * pad_y)
    text_box = (x0 + pad_x, y0 + pad_y, x0 + pad_x + max_w, y0 + pad_y + max_h)

    if size:
        fitted = size
    else:
        fitted = _frontend_fit(text, None, align, text_box)

    # `_load_font`, not a bare ``ImageFont.truetype``: the real preview lays
    # text out with the face's unhinted advances, and a `_load_font` font is
    # what carries the table that reproduces them (a bare truetype would keep
    # Pillow's hinted walk, which the backend no longer measures with).
    f = _load_font(FONT_PATH, fitted)
    ascent, descent = font_metrics(FONT_PATH, fitted)
    line_h = ascent + descent
    spacing = _spacing_for(fitted)
    advance = line_h + spacing

    # wrap
    lines = _layout(f, body, max_w, align)
    block_h = (len(lines) - 1) * advance + line_h if lines else 0

    top = text_box[1] + (text_box[3] - text_box[1] - block_h) / 2
    out = []
    anchor = "middle" if align == "center" else "end" if align == "right" else "start"
    for line in lines:
        # Pillow anchor="la" treats y as ascender-top; SVG default baseline means
        # SVG y = backend ascender-top + ascent.
        baseline = top + ascent
        line_x = text_box[3] if anchor == "end" else text_box[0] + max_w / 2 if anchor == "middle" else text_box[0]
        out.append(RecordedLine(baseline_y=baseline, text=line, anchor=anchor, x=line_x))
        top += advance
    return fitted, out


def _frontend_fit(text, size, align, box):
    x0, y0, x1, y1 = box
    max_w = max(1, x1 - x0)
    max_h = max(1, y1 - y0)
    lo, hi = MIN_FONT_SIZE, MAX_FONT_SIZE
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        # Unhinted measure + exact fits, mirroring `textFit.attempt` and the
        # backend's own `fits()` — both engines now see the same widths.
        f = _load_font(FONT_PATH, mid)
        lines = _layout(f, text, max_w, align)
        line_h = _line_height(f, mid)
        spacing = _spacing_for(mid)
        advance = line_h + spacing
        block_h = (len(lines) - 1) * advance + line_h if lines else 0
        widest = max((_length(f, line) for line in lines), default=0.0)
        if widest <= max_w and block_h <= max_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


class TestFrontendVsBackend:
    TEXTS = [
        "She really went and did it not just once, but twice!",
        "Seriously, this girl really can't read the room at all, can she.",
        "Come on, come on. If you don't spit it out quickly.",
        "Um...",
        "I was just going with the flow and passively looking on, but...",
    ]
    BOX = (100, 100, 400, 300)

    @pytest.mark.parametrize("text", TEXTS)
    def test_auto_baselines_match(self, text):
        b_size, b_lines = backend_baselines(text, None, "left", self.BOX)
        f_size, f_lines = frontend_baselines(text, None, "left", self.BOX)
        assert f_size == b_size, f"size: frontend={f_size} backend={b_size} text={text!r}"
        assert len(f_lines) == len(b_lines), (
            f"line count: frontend={len(f_lines)} backend={len(b_lines)} text={text!r}"
        )
        for i, (fl, bl) in enumerate(zip(f_lines, b_lines)):
            assert fl.text == bl.text, f"line {i} text: {fl.text!r} vs {bl.text!r}"
            assert abs(fl.baseline_y - bl.baseline_y) <= 1.0, (
                f"line {i} baseline: frontend={fl.baseline_y} backend={bl.baseline_y} "
                f"diff={fl.baseline_y - bl.baseline_y} text={text!r}"
            )

    @pytest.mark.parametrize("size", [12, 20, 36, 48])
    def test_explicit_size_baselines_match(self, size):
        text = self.TEXTS[0]
        b_size, b_lines = backend_baselines(text, size, "left", self.BOX)
        f_size, f_lines = frontend_baselines(text, size, "left", self.BOX)
        assert b_size == f_size == size
        for i, (fl, bl) in enumerate(zip(f_lines, b_lines)):
            assert abs(fl.baseline_y - bl.baseline_y) <= 1.0, (
                f"size={size} line {i} baseline: frontend={fl.baseline_y} "
                f"backend={bl.baseline_y} diff={fl.baseline_y - bl.baseline_y}"
            )


class TestExactBaselineParity:
    """With the metrics table, vertical parity is exact, not "within a pixel".

    Both engines now compute from the same integers
    (`typeset.line_metrics_for` ships `font.getmetrics()` verbatim and the
    preview reads sizes out of it), so the earlier 1px tolerance -- which the
    rounded font-unit estimate needed -- would only hide a regression
    elsewhere. These assert bit-equality.
    """

    TEXTS = TestFrontendVsBackend.TEXTS
    BOX = TestFrontendVsBackend.BOX

    def _require_face(self):
        if not os.path.isfile(FONT_PATH):
            pytest.skip("ComicMono not present in this checkout")

    @pytest.mark.parametrize("text", TEXTS)
    def test_wrapped_baselines_match_exactly(self, text):
        self._require_face()
        b_size, b_lines = backend_baselines(text, None, "left", self.BOX)
        f_size, f_lines = frontend_baselines(text, None, "left", self.BOX)
        assert f_size == b_size, f"size: frontend={f_size} backend={b_size} text={text!r}"
        assert len(f_lines) == len(b_lines), text
        for i, (fl, bl) in enumerate(zip(f_lines, b_lines)):
            assert fl.text == bl.text, f"line {i} text: {fl.text!r} vs {bl.text!r}"
            assert fl.baseline_y == bl.baseline_y, (
                f"line {i} baseline: frontend={fl.baseline_y} "
                f"backend={bl.baseline_y} text={text!r}"
            )

    @pytest.mark.parametrize(
        "size", range(MIN_FONT_SIZE, MAX_FONT_SIZE + 1)
    )
    def test_every_fit_size_matches_exactly(self, size):
        self._require_face()
        b_size, b_lines = backend_baselines("Sand!", size, "center", self.BOX)
        f_size, f_lines = frontend_baselines("Sand!", size, "center", self.BOX)
        assert b_size == f_size == size
        assert len(f_lines) == len(b_lines) == 1
        fl, bl = f_lines[0], b_lines[0]
        assert fl.baseline_y == bl.baseline_y, (
            f"size {size}: frontend={fl.baseline_y} backend={bl.baseline_y}"
        )


class TestUnhintedAdvanceParity:
    """Generated hmtx advances expose drift without optional font dependencies."""

    def test_loaded_face_measures_unhinted(self, typeset_font_dir):
        font = _load_font(TEXT_FIXTURE_NAMES[1], 12)
        table = getattr(font, "_fox_raw_adv", None)
        assert isinstance(table, dict) and table["adv"]
        for text, unhinted in (
            ("She", 22.5),
            ("once, but", 67.5),
            ("just once,", 75.0),
            (" ", 7.5),
        ):
            assert _length(font, text) == pytest.approx(unhinted, abs=0.01), text

    def test_plain_truetype_still_measures_hinted(self, typeset_font_dir):
        font = ImageFont.truetype(
            str(typeset_font_dir / TEXT_FIXTURE_NAMES[1]), 12, layout_engine=ImageFont.Layout.BASIC,
        )
        assert getattr(font, "_fox_raw_adv", None) is None
        assert _length(font, "once, but") == pytest.approx(72.0, abs=0.01)


class TestMixedRunParity:
    """Composite widths match the primary and calibrated symbol runs."""

    @pytest.mark.parametrize("face", ["ComicMono.ttf", "ComicMono-Bold.ttf", *TEXT_FIXTURE_NAMES])
    def test_composite_is_the_sum_of_its_runs(self, typeset_font_dir, face):
        from fox_reader import typeset

        scale = typeset.symbol_scale_for(face)
        text = "Hi♡!"
        runs = [("Hi", False), ("♡", True), ("!", False)]
        for size in (12, 24):
            primary = _load_font(face, size)
            px = typeset._symbol_px_size(face, size)
            assert px == max(1, typeset._round_half_up(size * scale))
            sf, pc, sc = typeset._symbol_faces(face, text, size)
            assert sf is not None and sf.size == px
            assert typeset._split_runs(text, pc, sc) == runs
            composite = typeset._mixed_length(primary, sf, text, pc, sc)
            run_sum = sum(
                _raw_length(sf, run_text) if is_symbol else _raw_length(primary, run_text)
                for run_text, is_symbol in runs
            )
            assert composite == pytest.approx(run_sum, abs=0.05), (
                f"{face} size={size}: composite {composite} vs raw run sum {run_sum}"
            )


class TestWrapSafetyMargin:
    """The two-pixel wrap margin must survive with only shipped/test fonts."""

    @pytest.fixture(autouse=True)
    def _font(self, typeset_font_dir):
        self.font = ImageFont.truetype(
            str(typeset_font_dir / TEXT_FIXTURE_NAMES[1]), 12, layout_engine=ImageFont.Layout.BASIC,
        )

    def test_the_marginal_line_sits_within_one_pixel_of_the_budget(self):
        # Nine 8px BASIC advances fit 73px, but cannot fit after the 2px margin.
        assert self.font.getlength("HHHH HHHH") == 72.0

    def test_the_margin_wraps_the_marginal_line(self):
        assert _layout(self.font, "HHHH HHHH HHHH", 73.0, "center") == ["HHHH", "HHHH", "HHHH"]

    def test_a_line_with_a_wide_enough_margin_stays_whole(self):
        assert _layout(self.font, "HHHH HHHH", 74.0, "center") == ["HHHH HHHH"]

    @pytest.mark.parametrize("face", ["ComicMono.ttf", *TEXT_FIXTURE_NAMES])
    def test_the_margin_shrinks_the_budget_on_each_face(self, face):
        font = _load_font(face, 12)
        width = _length(font, "abcd efgh")
        assert _layout(font, "abcd efgh", width + 2.5, "left") == ["abcd efgh"]
        assert _layout(font, "abcd efgh", width + 1.9, "left") == ["abcd", "efgh"]


class TestLineMetricsParity:
    """Vertical parity rides on the backend-shipped per-size metrics table.

    FreeType's per-size rounding cannot be reproduced from the font-table
    units: for ComicMono, 49 of the 67 sizes the UI searches disagree by a
    pixel between the preview's rounded formula and the real
    ``font.getmetrics()``. The fix ships the exact ints with ``/api/fonts``
    (``typeset.line_metrics_for``), and the preview reads sizes out of that
    table instead of estimating. These tests pin the two halves of that
    contract.
    """

    FONT = "fonts/ComicMono.ttf"

    def _require_face(self):
        if not os.path.isfile(self.FONT):
            pytest.skip("ComicMono not present in this checkout")

    def test_the_table_is_exactly_what_the_render_advances_by(self):
        self._require_face()
        table = line_metrics_for("ComicMono.ttf")
        assert len(table) == (MAX_FONT_SIZE - MIN_FONT_SIZE + 1) * 2
        for i, size in enumerate(range(MIN_FONT_SIZE, MAX_FONT_SIZE + 1)):
            font = _load_font("ComicMono.ttf", size)
            # Not raw getmetrics(): the helpers the renderer computes with.
            # Row-for-row, this is what `_draw_block` walks and what the
            # preview's `lineHeightFor`/`ascentFor` must reproduce.
            assert table[2 * i] == _ascent(font), f"size {size} ascent"
            assert table[2 * i] + table[2 * i + 1] == _line_height(font, size), (
                f"size {size} line height"
            )

    def test_the_font_unit_estimate_cannot_replace_it(self):
        """The rounded formula disagrees with the real metrics for many sizes.

        The estimate stays as the fallback for payloads without the table; a
        1px line-height difference is exactly the height of a descender-capable
        row, enough to wrap a block or spill it out of a bubble, so it can
        never be the *primary* contract. The backend declares fit sizes
        ``ascent/upem`` and ``descent/upem`` (hhea, descent stored positive --
        ``fonts._extract_font_metadata``), and this is the formula the
        preview's no-table path applies.
        """
        self._require_face()
        import io as _io

        from fontTools.ttLib import TTFont

        tt = TTFont(self.FONT, lazy=True)
        try:
            upem = int(tt["head"].unitsPerEm)
            ascent = int(tt["hhea"].ascent)
            descent = abs(int(tt["hhea"].descent))
        finally:
            tt.close()
        assert upem > 0 and ascent > 0
        with open(self.FONT, "rb") as fh:
            raw = fh.read()

        mismatches: list[tuple[int, int, int]] = []
        for size in range(MIN_FONT_SIZE, MAX_FONT_SIZE + 1):
            f = ImageFont.truetype(_io.BytesIO(raw), size)
            real = int(sum(f.getmetrics()))
            formula = round(size * ascent / upem) + round(size * descent / upem)
            if formula != real:
                mismatches.append((size, real, formula))
        # 49/67 on today's Pillow/freetype; a constant is deliberately not
        # pinned -- the point is that *some* sizes drift, so the estimate must
        # not be trusted where the table can be read instead.
        assert len(mismatches) >= 1, (
            "the font-unit estimate matches every size; the table could be "
            f"retired, this test with it (last run: {len(mismatches)} drifts)"
        )

