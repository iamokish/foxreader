"""Tests for the typeset line breaker and the alignments it still supports.

Two rules are pinned here, both from the spec:

* a word is never cut mid-run -- ``Total`` may not become ``tot`` / ``al`` -- but
  a hyphen the author wrote *is* a break, so ``tot-al`` may become ``tot-`` /
  ``al``. A run of two or more hyphens is punctuation and never a break.
* justify and vertical alignment are gone. Only ``left``, ``center`` and
  ``right`` exist, and anything else (a value saved by an older build) has to
  degrade to left rather than raise.

Line-breaker cases use a fixed-width stub to make wrap boundaries exact.
Raster and calibration cases use shipped faces and generated font fixtures,
so optional user fonts cannot change which regressions run.
"""
import io
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from fox_reader import typeset
from fox_reader.typeset import (
    TEXT_ALIGNMENTS,
    _ascent,
    _break_word,
    _draw_block,
    _hyphen_pieces,
    _layout,
    _load_font,
    _wrap_paragraph,
)
from tests.font_fixtures import TEXT_FIXTURE_NAMES, make_font


class FakeFont:
    """A monospace face: every character is exactly ``advance`` wide."""

    def __init__(self, advance: float = 10.0, ascent: int = 8, descent: int = 2):
        self.advance = advance
        self._ascent = ascent
        self._descent = descent

    def getlength(self, text):
        return len(text) * self.advance

    def getmetrics(self):
        return (self._ascent, self._descent)


class FakeDraw:
    """Records what :func:`_draw_block` asked to be drawn, and where."""

    def __init__(self):
        self.calls = []

    def text(self, xy, line, **kw):
        self.calls.append((xy, line, kw.get("anchor")))


class RecordingDraw:
    """Records font + anchor + position per run (for fallback tests)."""

    def __init__(self):
        self.calls = []

    def text(self, xy, line, **kw):
        self.calls.append({"xy": xy, "line": line,
                           "anchor": kw.get("anchor"), "font": kw.get("font")})


class TestHyphenPieces:
    def test_single_hyphen_is_a_break_and_is_kept(self):
        assert _hyphen_pieces("tot-al") == ["tot-", "al"]

    def test_double_hyphen_is_punctuation_not_a_break(self):
        assert _hyphen_pieces("a--b") == ["a--b"]
        assert _hyphen_pieces("re--do-it") == ["re--do-", "it"]

    def test_word_without_a_hyphen_is_one_piece(self):
        assert _hyphen_pieces("Total") == ["Total"]

    def test_every_interior_hyphen_offers_a_break(self):
        assert _hyphen_pieces("well-known-thing") == ["well-", "known-", "thing"]

    def test_edge_hyphens_are_not_breaks(self):
        # Breaking here would leave a line that is nothing but punctuation.
        assert _hyphen_pieces("-lead") == ["-lead"]
        assert _hyphen_pieces("trail-") == ["trail-"]


class TestBreakWord:
    def test_unhyphenated_word_survives_whole(self):
        font = FakeFont()
        # 50px of glyphs into a 20px line: still one piece, because the fix is to
        # shrink the type, not to invent a hyphen.
        assert _break_word(font, "Total", 20.0) == ["Total"]

    def test_hyphenated_word_breaks_at_its_hyphen(self):
        font = FakeFont()
        assert _break_word(font, "tot-al", 45.0) == ["tot-", "al"]

    def test_pieces_that_fit_together_stay_together(self):
        font = FakeFont()
        assert _break_word(font, "tot-al", 1000.0) == ["tot-al"]


class TestWrapParagraph:
    def test_wraps_on_spaces(self):
        font = FakeFont()
        assert _wrap_paragraph(font, "aaa bbb ccc", 70.0) == ["aaa bbb", "ccc"]

    def test_an_overlong_word_is_not_chopped(self):
        font = FakeFont()
        assert _wrap_paragraph(font, "Totalitarian", 40.0) == ["Totalitarian"]

    def test_empty_text_still_yields_a_line(self):
        assert _wrap_paragraph(FakeFont(), "   ", 100.0) == [""]


class TestLayout:
    def test_newlines_start_new_lines(self):
        lines = _layout(FakeFont(), "one\ntwo", 1000.0, "left")
        assert lines == ["one", "two"]

    def test_blank_line_is_preserved(self):
        assert _layout(FakeFont(), "a\n\nb", 1000.0, "left") == ["a", "", "b"]

    def test_alignment_does_not_change_the_wrap(self):
        font = FakeFont()
        text = "aaa bbb ccc"
        left = _layout(font, text, 70.0, "left")
        assert all(_layout(font, text, 70.0, align) == left for align in TEXT_ALIGNMENTS)


class TestAlignments:
    def test_only_left_center_and_right_remain(self):
        assert TEXT_ALIGNMENTS == ("left", "center", "right")

    @pytest.mark.parametrize("gone", ["justify", "top", "middle", "bottom", "vertical"])
    def test_removed_alignments_are_not_offered(self, gone):
        assert gone not in TEXT_ALIGNMENTS

    @pytest.mark.parametrize(
        "align",
        ["left", "center", "right", "justify", ""],
    )
    def test_draw_block_anchors_and_falls_back_to_left(self, align):
        draw = FakeDraw()
        # "hi" is 2 x 10px glyphs; unknown alignments degrade to left.
        _draw_block(draw, FakeFont(), 20, ["hi"], align,
                    (0, 0, 100, 40), (0, 0, 0, 255), 0, None)
        assert [call[2] for call in draw.calls] == ["ls", "ls"]
        x = [call[0][0] for call in draw.calls]
        if align == "center":
            assert x == [pytest.approx(40.0), pytest.approx(50.0)]
        elif align == "right":
            assert x == [pytest.approx(80.0), pytest.approx(90.0)]
        else:  # left, and anything the block does not know
            assert x == [pytest.approx(0.0), pytest.approx(10.0)]

    def test_block_is_centred_vertically(self):
        draw = FakeDraw()
        # One 10px-tall line box (ascent 8 + descent 2) centred in a 40px box:
        # its top is 15.0, and glyphs sit baseline-anchored on its ascender.
        _draw_block(draw, FakeFont(), 20, ["hi"], "left",
                    (0, 0, 100, 40), (0, 0, 0, 255), 0, None)
        assert draw.calls[0][0][1] == pytest.approx(15.0 + 8)

    def test_blank_lines_advance_but_are_not_drawn(self):
        draw = FakeDraw()
        _draw_block(draw, FakeFont(), 20, ["a", "", "b"], "left",
                    (0, 0, 100, 100), (0, 0, 0, 255), 0, None)
        drawn = [call[1] for call in draw.calls]
        assert drawn == ["a", "b"]
        # The gap between "a" and "b" is two advances, not one -- the empty line
        # takes up exactly as much room as a full one. One advance is the 10px
        # line height plus `_spacing_for(20)` == 3.
        gap = draw.calls[1][0][1] - draw.calls[0][0][1]
        assert gap == pytest.approx(2 * (10 + 3))


class TestSymbolBaseline:
    """Mixed primary/symbol runs must share one baseline, not one ascender top.

    Regression test for "Ngiiii♡♡♡": the symbol face (NotoSansSymbols2, ascent
    43 at size 40) is much taller than a typical primary (ComicMono, ascent 32),
    so drawing every run at the same ascender-top ``y`` puts the hearts' baseline
    ~11 px below the letters'. Every run of a mixed line is therefore drawn
    ``ls``-anchored on ``y + primary_ascent``.
    """

    HEART = "♡"

    def test_ascent_reads_the_face(self):
        assert _ascent(FakeFont(ascent=8)) == 8

    def test_ascent_is_none_when_unreadable(self):
        class Broken:
            def getmetrics(self):
                raise OSError("no metrics")

        assert _ascent(Broken()) is None
        assert _ascent(object()) is None

    def test_mixed_runs_share_one_baseline(self):
        primary = FakeFont(advance=10.0, ascent=8, descent=2)
        symbol = FakeFont(advance=10.0, ascent=20, descent=5)
        pc, sc = {ord("A")}, {ord(self.HEART)}
        draw = RecordingDraw()
        # One line, 10px tall (8+2), in a 40px box -> line y is 15.0.
        _draw_block(draw, primary, 20, ["A" + self.HEART], "left",
                    (0, 0, 100, 40), (0, 0, 0, 255), 0, None,
                    symbol, pc, sc)
        assert [c["line"] for c in draw.calls] == ["A", self.HEART]
        assert [c["font"] for c in draw.calls] == [primary, symbol]
        # Both baseline-anchored on the primary baseline, not the ascender top.
        assert [c["anchor"] for c in draw.calls] == ["ls", "ls"]
        assert draw.calls[0]["xy"][1] == pytest.approx(15.0 + 8)
        assert draw.calls[1]["xy"][1] == pytest.approx(15.0 + 8)
        # Horizontal advance still follows the primary run's width.
        assert draw.calls[1]["xy"][0] == pytest.approx(10.0)

    def test_mixed_center_keeps_total_width_but_baseline_anchors(self):
        primary = FakeFont(advance=10.0, ascent=8, descent=2)
        symbol = FakeFont(advance=10.0, ascent=20, descent=5)
        pc, sc = {ord("A")}, {ord(self.HEART)}
        draw = RecordingDraw()
        _draw_block(draw, primary, 20, ["A" + self.HEART], "center",
                    (0, 0, 100, 40), (0, 0, 0, 255), 0, None,
                    symbol, pc, sc)
        # 20px of ink in a 100px box -> starts at 40.
        assert draw.calls[0]["xy"][0] == pytest.approx(40.0)
        assert all(c["anchor"] == "ls" for c in draw.calls)

    def test_primary_only_line_with_fallback_armed_stays_single_face(self):
        """Idle fallback draws every glyph from the primary, on one baseline."""
        primary = FakeFont(ascent=8)
        symbol = FakeFont(ascent=20)
        draw = RecordingDraw()
        _draw_block(draw, primary, 20, ["hi"], "center",
                    (0, 0, 100, 40), (0, 0, 0, 255), 0, None,
                    symbol, {ord("h"), ord("i")}, {ord(self.HEART)})
        assert [call["font"] for call in draw.calls] == [primary, primary]
        assert [call["line"] for call in draw.calls] == ["h", "i"]
        assert all(call["anchor"] == "ls" for call in draw.calls)
        # Same converted baseline as a mixed line would get.
        assert all(call["xy"][1] == pytest.approx(15.0 + 8) for call in draw.calls)
        # Glyphs placed at the per-character advance, not one string.
        assert draw.calls[1]["xy"][0] == pytest.approx(draw.calls[0]["xy"][0] + 10.0)

    def test_unreadable_ascent_degrades_to_unshifted(self):
        class Broken:
            def getlength(self, text):
                return len(text) * 10.0

            def getmetrics(self):
                raise OSError("no metrics")

        primary, symbol = Broken(), FakeFont(ascent=20)
        pc, sc = {ord("A")}, {ord(self.HEART)}
        draw = RecordingDraw()
        _draw_block(draw, primary, 20, ["A" + self.HEART], "left",
                    (0, 0, 100, 40), (0, 0, 0, 255), 0, None,
                    symbol, pc, sc)
        # No baseline to convert to: old same-y placement, still two runs.
        assert [c["anchor"] for c in draw.calls] == ["la", "la"]


class TestFrozenSymbolSource:
    """A compiled build reads the symbol face from the bundle, never from disk.

    The staged ``fonts/`` folder holds licences only, so any ``FONT_DIR`` access
    in frozen mode is a bug -- not just a wasted stat, but a lookup that can
    only answer wrongly. These pin the branch with a ``FONT_DIR`` that explodes
    on any touch.
    """

    @pytest.fixture
    def typeset(self):
        import fox_reader.typeset as module

        saved = {
            name: getattr(module, name)
            for name in ("_SYMBOL_BYTES", "_SYMBOL_BYTES_SIG",
                         "_SYMBOL_BYTES_MISSING", "_SYMBOL_WARNED")
        }
        module._SYMBOL_BYTES = None
        module._SYMBOL_BYTES_SIG = None
        module._SYMBOL_BYTES_MISSING = False
        module._SYMBOL_WARNED = False
        try:
            yield module
        finally:
            for name, value in saved.items():
                setattr(module, name, value)

    class _ExplodingDir:
        """Stands in for ``FONT_DIR``; any attribute use is a failure."""

        def __getattribute__(self, _name):
            raise AssertionError("frozen symbol lookup touched FONT_DIR")

    def test_frozen_reads_bundle_without_touching_disk(self, typeset, monkeypatch):
        import fox_reader.assets as assets

        monkeypatch.setattr(typeset, "is_frozen", True)
        monkeypatch.setattr(typeset, "FONT_DIR", self._ExplodingDir())
        monkeypatch.setattr(assets, "has_bundle", lambda: True)
        monkeypatch.setattr(assets, "symbol_font_bytes", lambda *a, **k: b"fake-symbol")

        assert typeset._symbol_font_bytes() == b"fake-symbol"

    def test_frozen_without_bundle_face_is_none_without_touching_disk(
            self, typeset, monkeypatch):
        import fox_reader.assets as assets

        monkeypatch.setattr(typeset, "is_frozen", True)
        monkeypatch.setattr(typeset, "FONT_DIR", self._ExplodingDir())
        monkeypatch.setattr(assets, "has_bundle", lambda: True)
        monkeypatch.setattr(assets, "symbol_font_bytes", lambda *a, **k: None)

        assert typeset._symbol_font_bytes() is None

    def test_checkout_without_any_source_is_none_not_fatal(self, typeset, monkeypatch, tmp_path):
        import fox_reader.assets as assets

        monkeypatch.setattr(typeset, "is_frozen", False)
        monkeypatch.setattr(typeset, "FONT_DIR", tmp_path)
        monkeypatch.setattr(assets, "has_bundle", lambda: False)
        monkeypatch.setattr(assets, "symbol_font_bytes", lambda *a, **k: None)

        assert typeset._symbol_font_bytes() is None


class TestEntryColours:
    """The three tiers an ``auto`` colour resolves through.

    The user's own pick wins; failing that, what the OCR step measured off the
    artwork (:mod:`fox_reader.ocr_colors`); failing that, the rule that works
    from the plate alone, which is all there was before the measurements
    existed. The last tier has to keep behaving exactly as it did, because it is
    what every entry nobody has OCR'd still renders through.
    """

    class Recorder:
        """A draw surface that remembers the plate and the text colours."""

        def __init__(self):
            self.plate = None
            self.text_calls = []

        def polygon(self, _points, fill=None):
            self.plate = fill

        def text(self, _xy, _line, **kw):
            self.text_calls.append(kw)

        @property
        def fill(self):
            return self.text_calls[-1]["fill"]

        @property
        def stroke(self):
            return self.text_calls[0]["stroke_fill"]

    @staticmethod
    def _item(**over):
        from fox_reader.models.requests import DataItem

        fields = dict(
            og_text="x", text="Hi", fontfile="", text_align="center",
            points=[(0, 0), (80, 0), (80, 60), (0, 60)],
        )
        fields.update(over)
        return DataItem(**fields)

    def _render(self, item, plate_rgb=(255, 255, 255)):
        from PIL import Image

        from fox_reader.typeset import _draw_entry

        img = Image.new("RGBA", (80, 60), (*plate_rgb, 255))
        recorder = self.Recorder()
        _draw_entry(img, recorder, item)
        assert recorder.text_calls, "nothing was drawn, so no colours were chosen"
        return recorder

    def test_without_measurements_the_old_rule_still_decides(self):
        """White paper, so black text outlined in the paper's own colour."""
        drawn = self._render(self._item())
        assert drawn.fill == (0, 0, 0, 255)
        assert drawn.stroke == (255, 255, 255, 255)

    def test_a_measurement_beats_the_old_rule(self):
        """The whole point: white lettering measured on a dark panel is set in
        white, where the plate-and-contrast rule would have made it black."""
        drawn = self._render(
            self._item(detected_font_color="#ffffff",
                       detected_stroke_color="#101010",
                       detected_bg="#1b1b1e"),
        )
        assert drawn.fill == (255, 255, 255, 255)
        assert drawn.stroke == (16, 16, 16, 255)
        assert drawn.plate == (27, 27, 30, 255), "the measured plate was not painted"

    def test_the_users_own_pick_beats_a_measurement(self):
        drawn = self._render(
            self._item(font_color="#ff0000", stroke_color="#00ff00",
                       detected_font_color="#ffffff",
                       detected_stroke_color="#101010"),
        )
        assert drawn.fill == (255, 0, 0, 255)
        assert drawn.stroke == (0, 255, 0, 255)

    def test_a_measurement_that_would_vanish_is_dropped(self):
        """White lettering measured on a dark panel, then set on a white plate
        the user chose. Keeping the measurement would draw white on white."""
        drawn = self._render(
            self._item(bg_mode="color", bg_color="#ffffff",
                       detected_font_color="#fdfdfd",
                       detected_bg="#1b1b1e"),
        )
        assert drawn.fill == (0, 0, 0, 255), "white-on-white survived the guard"
        assert drawn.plate == (255, 255, 255, 255)

    def test_a_measurement_that_still_reads_survives_a_new_plate(self):
        """The guard is about legibility, not about the plate having changed: red
        ink measured off the artwork still reads on a pale plate, so it stays."""
        drawn = self._render(
            self._item(bg_mode="color", bg_color="#fffaf0",
                       detected_font_color="#b21824"),
        )
        assert drawn.fill == (178, 24, 36, 255)

    @pytest.mark.parametrize("bad", ["", "   ", "auto", "not-a-colour", "#12345"])
    def test_an_unreadable_measurement_falls_through(self, bad):
        """They arrive over the wire, so they can be anything. An unreadable one
        has to degrade to the old rule rather than fail the entry."""
        drawn = self._render(
            self._item(detected_font_color=bad, detected_stroke_color=bad,
                       detected_bg=bad),
        )
        assert drawn.fill == (0, 0, 0, 255)
        assert drawn.stroke == (255, 255, 255, 255)

    def test_a_measured_plate_is_not_painted_over_a_clean(self):
        """`clean` repaints the artwork and deliberately paints no plate. A
        measured background must not reintroduce one."""
        drawn = self._render(
            self._item(bg_mode="clean", detected_bg="#1b1b1e",
                       detected_font_color="#ffffff"),
        )
        assert drawn.plate is None
        # The plate for contrast purposes is still the measured background, so
        # the lettering is set as it was measured.
        assert drawn.fill == (255, 255, 255, 255)

    def test_a_transparent_measurement_cannot_hide_the_text(self):
        """A pick may legitimately be transparent; a measurement never is, so one
        that arrives that way is a bad hint rather than an instruction."""
        drawn = self._render(self._item(detected_font_color="#00000000"))
        assert drawn.fill == (0, 0, 0, 255)


class TestStrokeRing:
    """Pillow's ``stroke_width`` sits wholly outside the glyph.

    The frontend's SVG outline (`textFit.STROKE_ATTR_COEFF`) is built on this
    contract: SVG centres a stroke on the outline, so `stroke-width: N` shows
    only N/2 of it outside the fill, while the backend's `stroke_width: N` is a
    solid N-pixel band of ring on every side. Measured at sizes 24/48 and
    widths 1..3 on the shipped ComicMono face; a Pillow upgrade that moves these
    numbers would silently double or halve every outline in the reader, so they
    are pinned rather than assumed.
    """

    RED = (255, 0, 0, 255)
    BLUE = (0, 0, 255, 255)

    @pytest.fixture(autouse=True)
    def _font(self):
        from pathlib import Path

        from PIL import ImageFont

        from fox_reader.utils import FONT_DIR

        source = FONT_DIR / "ComicMono.ttf"
        if not source.is_file():
            pytest.skip("no fonts/ComicMono.ttf in the checkout")
        self._font_bytes = Path(source).read_bytes()
        self._ImageFont = ImageFont

    def _render(self, text: str, size: int, stroke: int):
        import io

        from PIL import Image, ImageDraw

        font = self._ImageFont.truetype(io.BytesIO(self._font_bytes), size)
        img = Image.new("RGBA", (size * 6, size * 6), (0, 0, 0, 0))
        ImageDraw.Draw(img).text(
            (size, size), text, font=font, fill=self.RED,
            stroke_width=stroke, stroke_fill=self.BLUE, anchor="la",
        )
        return img

    def _ring_and_widening(self, size: int, stroke: int) -> tuple[float, float]:
        img = self._render("HI", size, stroke)
        plain = self._render("HI", size, 0)
        px, ppx = img.load(), plain.load()

        def ink(canvas_px, imgs, x, y):
            return canvas_px[x, y][3] > 96

        def is_stroke(x, y):
            p = px[x, y]
            return p[3] > 96 and p[2] > 128 and p[0] < 128

        def bbox(canvas_px, image):
            xs = [
                x for y in range(image.height) for x in range(image.width)
                if ink(canvas_px, image, x, y)
            ]
            return min(xs), max(xs)

        w0 = bbox(ppx, plain)[1] - bbox(ppx, plain)[0] + 1
        w1 = bbox(px, img)[1] - bbox(px, img)[0] + 1
        widening = (w1 - w0) / 2.0

        # Ring thickness: contiguous stroke pixels walking in from the left edge
        # of the H's left bar, on rows through its clean middle third (where the
        # crossbar cannot interfere).
        ys = [
            y for y in range(img.height)
            if any(ink(px, img, x, y) for x in range(img.width))
        ]
        lo, hi = min(ys), max(ys)
        lo += (hi - lo) // 3
        hi -= (hi - lo) // 3
        runs = []
        for y in range(lo, hi + 1):
            row = [x for x in range(img.width) if ink(px, img, x, y)]
            if not row:
                continue
            t = 0
            for x in range(row[0], img.width):
                if is_stroke(x, y):
                    t += 1
                else:
                    break
            if t:
                runs.append(t)
        ring = sum(runs) / len(runs)
        return widening, ring

    @pytest.mark.parametrize("size,stroke", [(24, 1), (24, 2), (24, 3), (48, 2), (48, 3)])
    def test_the_ring_matches_the_stroke_width(self, size, stroke):
        widening, ring = self._ring_and_widening(size, stroke)
        assert ring == stroke, f"ring {ring} != stroke {stroke}"
        assert widening == stroke, f"per-side widening {widening} != stroke {stroke}"


class TestUnhintedInkPaint:
    """Per-character painting must retain fractional hmtx advances.

    The generated face advances 7.5px at size 12, while Pillow's BASIC string
    walk rounds each advance to 8px. Eight letters expose cumulative drift.
    """

    @pytest.fixture(autouse=True)
    def _font(self, typeset_font_dir):
        self.face = TEXT_FIXTURE_NAMES[1]
        self.source = str(typeset_font_dir / self.face)

    def _ink_from_origin(self, img, x_origin: int = 30) -> tuple[float, float]:
        xs = np.nonzero(np.array(img))[1]
        assert xs.size, "nothing was drawn"
        return float(xs.min() - x_origin), float(xs.max() - x_origin)

    def _paint_new(self):
        img = Image.new("L", (240, 36), 0)
        font = _load_font(self.face, 12)
        _draw_block(ImageDraw.Draw(img), font, 12, ["HHHHHHHH"], "left", (30, 6, 230, 24), 255, 0, None)
        return self._ink_from_origin(img)

    def _paint_old_style(self):
        img = Image.new("L", (240, 36), 0)
        font = ImageFont.truetype(self.source, 12, layout_engine=ImageFont.Layout.BASIC)
        ImageDraw.Draw(img).text((30, 12), "HHHHHHHH", font=font, fill=255, anchor="ls")
        return self._ink_from_origin(img)

    def test_fractional_paint_does_not_accumulate_hinted_advance_rounding(self):
        new_left, new_right = self._paint_new()
        old_left, old_right = self._paint_old_style()
        assert (old_right - old_left) - (new_right - new_left) >= 2, (
            f"painted widths: unhinted {new_right - new_left} vs hinted {old_right - old_left}"
        )

    def test_the_last_glyph_lands_at_the_fractional_pen_position(self):
        _left, new_right = self._paint_new()
        # Seven 7.5px advances plus the final glyph's 6px ink, with paste rounding.
        assert new_right == pytest.approx(58.0, abs=1.0)


class TestSymbolSizeCalibration:
    """Known cap heights produce independent scales and calibrated run widths."""

    @pytest.fixture(autouse=True)
    def _fonts(self, typeset_font_dir, monkeypatch):
        self.primary = TEXT_FIXTURE_NAMES[1]
        raw = make_font(typeset_font_dir / "TestSymbols.ttf", dict.fromkeys("♡★✓■●", 400))
        monkeypatch.setattr(typeset, "_symbol_font_bytes", lambda: raw)

    def test_scale_matches_each_generated_cap_height(self):
        for face, expected in zip(TEXT_FIXTURE_NAMES, (1.5, 2.0, 2.5), strict=True):
            assert typeset.symbol_scale_for(face) == pytest.approx(expected), face

    def test_symbol_px_size_follows_the_scale(self):
        for face in (*TEXT_FIXTURE_NAMES, "ComicMono.ttf"):
            scale = typeset.symbol_scale_for(face)
            for size in (1, 12, 24):
                assert typeset._symbol_px_size(face, size) == max(1, typeset._round_half_up(size * scale))

    def test_the_missing_or_broken_face_stays_sane(self, monkeypatch):
        assert typeset.symbol_scale_for("absent-face.ttf") == typeset.symbol_scale_for("ComicMono.ttf")

        def broken(name, size):
            raise OSError("no face")

        monkeypatch.setattr(typeset, "_load_font", broken)
        monkeypatch.setattr(typeset, "_load_symbol_font", lambda size: None)
        assert typeset.symbol_scale_for("whatever.ttf") == 1.0
        assert typeset._symbol_px_size("whatever.ttf", 12) == 12

    def test_mixed_measure_runs_the_calibrated_symbol_face(self):
        size = 24
        primary = typeset._load_font(self.primary, size)
        sf, pc, sc = typeset._symbol_faces(self.primary, "Hi♡!", size)
        assert sf is not None and sf.size == typeset._symbol_px_size(self.primary, size)
        expected = (
            typeset._raw_length(primary, "Hi")
            + typeset._raw_length(sf, "♡")
            + typeset._raw_length(primary, "!")
        )
        assert typeset._mixed_length(primary, sf, "Hi♡!", pc, sc) == pytest.approx(expected, abs=0.02)


class TestBytesSourceAdvances:
    """Bytes-loaded faces must expose the same raw hmtx widths as disk faces."""

    def test_attach_from_raw_bytes_walks_the_table(self, typeset_font_dir):
        raw = (typeset_font_dir / TEXT_FIXTURE_NAMES[1]).read_bytes()
        font = ImageFont.truetype(io.BytesIO(raw), 12)
        typeset._attach_raw_advances(font, raw)
        assert isinstance(getattr(font, "_fox_raw_adv", None), dict)
        # Nine 625-unit advances at 12px in a 1000-unit em.
        assert typeset._raw_length(font, "once, but") == pytest.approx(67.5, abs=0.01)

    def test_attach_from_bad_bytes_sets_the_sentinel(self, typeset_font_dir):
        raw = (typeset_font_dir / TEXT_FIXTURE_NAMES[1]).read_bytes()
        font = ImageFont.truetype(io.BytesIO(raw), 12)
        typeset._attach_raw_advances(font, b"still not a font")
        assert getattr(font, "_fox_raw_adv", None) is False


class TestHalfUpRounding:
    """Both engines must round shared floats the same way.

    The preview computes `round(size * scale)`, `round((lh + spacing) *
    factor)` and clamps in JavaScript, where ``Math.round`` sends ``x.5`` up.
    Python's ``round`` sends ``x.5`` to the *even* neighbour, so the two
    engines would disagree exactly there -- a class of positionoff-by-one that
    no amount of shared font data fixes. ``_round_half_up`` is floor(x + 0.5),
    which is JS's rule for every sign.
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.5, 1),
            (1.5, 2),
            (2.5, 3),  # banker's round() gives 2 here -- the whole point
            (3.5, 4),
            (-0.5, 0),
            (-1.5, -1),  # banker's gives -2
            (22.5, 23),
            (0.0, 0),
            (1.4999999, 1),
        ],
    )
    def test_halves_go_up_like_javascript(self, value, expected):
        assert typeset._round_half_up(value) == expected

    @pytest.mark.parametrize(
        "junk", [None, "abc", float("nan"), object()]
    )
    def test_junk_is_zero_never_an_exception(self, junk):
        assert typeset._round_half_up(junk) == 0

    def test_scaled_line_advance_agrees_with_javascript(self, monkeypatch):
        # 15 * 1.5 = 22.5 -> Math.round says 23, banker's round() says 22.
        assert typeset._advance_for(12, 3, 1.5) == 23

    def test_clamp_rounds_halves_up(self):
        # int(round(2.5)) would clamp to 2; the browser rounds to 3.
        assert typeset._clamp(2.5, 0, 32, 2) == 3
        assert typeset._clamp(0.5, 1, 72, 16) == 1, "clamping still wins"

    def test_symbol_px_size_rounds_halves_up(self, monkeypatch):
        monkeypatch.setattr(
            typeset, "symbol_scale_for", lambda *_a, **_k: 0.5, raising=False
        )
        # 13 * 0.5 = 6.5 -> 7; banker's would produce 6.
        assert typeset._symbol_px_size("half.ttf", 13) == 7

    def test_real_faces_never_land_on_a_disagreement(self):
        """No shipped-calibration size sits on a half, and rounding is half-up.

        If some ``size * scale`` were exactly ``x.5``, banker's ``round`` would
        have produced one fewer pixel than the browser does before this change
        -- worth knowing about, not silently forgetting.
        """
        if not os.path.isfile("fonts/ComicMono.ttf"):
            pytest.skip("ComicMono not present")
        scale = typeset.symbol_scale_for("ComicMono.ttf")
        for size in range(typeset.MIN_FONT_SIZE, typeset.MAX_FONT_SIZE + 1):
            raw = size * scale
            assert typeset._symbol_px_size("ComicMono.ttf", size) == max(
                1, typeset._round_half_up(raw)
            )
            assert raw != int(raw) + 0.5, (
                f"size {size} lands exactly on a half ({raw}): engines would split"
            )


class TestLineMetricsTable:
    """`line_metrics_for` ships the exact ints `font.getmetrics()` returns.

    The preview's rounded font-unit estimate disagrees with FreeType's
    per-size rounding for most sizes in the fit range (measured: 49 of the 67
    sizes 6..72 for ComicMono are off by a pixel in line height, and the
    ascent alone is off for ~70 of sizes 1..144), so the backend measures its
    own engine once per face and ships the table. These tests pin the table's
    contract: full range, real ints, fallback-consistent, all-or-nothing.
    """

    FACE = "ComicMono.ttf"

    @pytest.mark.skipif(
        not os.path.isfile(f"fonts/{FACE}"), reason="ComicMono not present"
    )
    def test_table_covers_every_fit_size_with_real_metrics(self):
        table = typeset.line_metrics_for(self.FACE)
        assert len(table) == (typeset.MAX_FONT_SIZE - typeset.MIN_FONT_SIZE + 1) * 2
        for i, size in enumerate(range(typeset.MIN_FONT_SIZE, typeset.MAX_FONT_SIZE + 1)):
            ascent, descent = (int(v) for v in _load_font(self.FACE, size).getmetrics())
            got_a, got_d = table[2 * i], table[2 * i + 1]
            assert (got_a, got_d) == (ascent, descent), (
                f"size {size}: table {got_a}/{got_d} vs getmetrics {ascent}/{descent}"
            )
            assert got_a + got_d > 0

    @pytest.mark.skipif(
        not os.path.isfile(f"fonts/{FACE}"), reason="ComicMono not present"
    )
    def test_a_missing_face_resolves_the_same_face_drawing_would(self):
        # `_load_font` falls through to the fallback, and so must the table:
        # an entry referencing a font the folder no longer has must still
        # describe what the render actually paints with.
        assert typeset.line_metrics_for("definitely-absent.ttf") == typeset.line_metrics_for(
            self.FACE
        )

    def test_a_face_that_cannot_be_measured_ships_no_table(self, monkeypatch):
        monkeypatch.setattr(typeset, "_LINE_METRICS_CACHE", {})

        def broken(name, size):
            raise OSError("no face")

        monkeypatch.setattr(typeset, "_load_font", broken)
        assert typeset.line_metrics_for("whatever.ttf") == []

    def test_the_table_is_cached_per_face(self, monkeypatch):
        monkeypatch.setattr(typeset, "_LINE_METRICS_CACHE", {})
        calls = []

        real_load = typeset._load_font

        def counting(name, size):
            calls.append((name, size))
            return real_load(name, size)

        monkeypatch.setattr(typeset, "_load_font", counting)
        typeset.line_metrics_for(self.FACE)
        first = len(calls)
        typeset.line_metrics_for(self.FACE)
        assert len(calls) == first, "second call must be a pure cache hit"


class TestBlockOutlines:
    def test_outline_expansion_is_circular_and_preserves_coverage(self):
        mask = typeset._expand_outline_mask((1, 1), b"\xff", 14)
        assert mask.size == (29, 29)
        assert mask.getpixel((14, 14)) == 255
        assert mask.getpixel((0, 14)) == mask.getpixel((14, 0)) == 255
        assert mask.getpixel((0, 0)) == 0, "a round stroke must not grow square corners"
        faint = typeset._expand_outline_mask((1, 1), b"\x40", 2)
        assert faint.getpixel((2, 2)) == 255, "a subpixel glyph must still acquire an opaque thick stroke"
        assert faint.getpixel((0, 2)) == 64, "the outline boundary must retain antialiasing"

    def test_outline_cache_is_bounded_and_large_masks_are_rejected(self):
        typeset._cached_outline_mask.cache_clear()
        first = typeset._cached_outline_mask((1, 1), b"\xff", 14)
        assert typeset._cached_outline_mask((1, 1), b"\xff", 14) is first
        assert typeset._cached_outline_mask.cache_info().maxsize == 64
        with pytest.raises(ValueError, match="too large"):
            typeset._expand_outline_mask((1, 1), b"\xff", 5000)

    @pytest.mark.parametrize("align", ["left", "center", "right"])
    def test_all_outlines_precede_all_fills_with_identical_glyph_positions(self, align):
        class Trace:
            def __init__(self):
                self.calls = []

            def text(self, xy, text, **kwargs):
                self.calls.append({"xy": xy, "text": text, **kwargs})

        primary, symbol = FakeFont(), FakeFont(advance=15, ascent=20)
        cascade = typeset._SymbolCascade({"?": symbol}, {"?": 3.5})
        draw = Trace()
        _draw_block(draw, primary, 20, ["H ?", "? H"], align, (0, 0, 200, 80),
                    (0, 0, 0, 255), 23, (255, 255, 255, 255), cascade, {ord("H")}, {ord("?")}, 4, 0.5)
        outlines, fills = draw.calls[:4], draw.calls[4:]
        assert len(fills) == 4
        assert [call["text"] for call in outlines] == ["H", "?", "?", "H"]
        for outline, fill in zip(outlines, fills, strict=True):
            assert outline["fill"] == outline["stroke_fill"] == (255, 255, 255, 255)
            assert outline["stroke_width"] == 23
            assert fill["fill"] == (0, 0, 0, 255)
            assert "stroke_width" not in fill
            assert "stroke_fill" not in fill
            for key in ("text", "xy", "font", "anchor"):
                assert outline[key] == fill[key]

    @pytest.mark.parametrize("filename", ["ComicMono.ttf", *TEXT_FIXTURE_NAMES])
    @pytest.mark.parametrize("stroke_width", [6, 23, 32])
    @pytest.mark.parametrize("lines,line_factor", [(["HELLO"], None), (["HELLO", "HELLO"], 0.5)])
    def test_thick_strokes_preserve_every_solid_letter_pixel(
        self, typeset_font_dir, filename, stroke_width, lines, line_factor,
    ):
        font = _load_font(filename, 40)
        mask = Image.new("L", (400, 220), 0)
        canvas = Image.new("RGB", mask.size, (180, 0, 180))
        box = (60, 40, 340, 180)
        _draw_block(ImageDraw.Draw(mask), font, 40, lines, "center", box,
                    255, 0, None, line_factor=line_factor)
        _draw_block(ImageDraw.Draw(canvas), font, 40, lines, "center", box,
                    (0, 0, 0), stroke_width, (255, 255, 255), line_factor=line_factor)
        solid_ink = np.asarray(mask) == 255
        assert solid_ink.any()
        assert np.all(np.asarray(canvas)[solid_ink] == 0), "a neighboring outline covered solid letter ink"
        assert np.any(np.all(np.asarray(canvas) == 255, axis=2)), "the outline must still be visible"
