"""Tests for the typeset line breaker and the alignments it still supports.

Two rules are pinned here, both from the spec:

* a word is never cut mid-run -- ``Total`` may not become ``tot`` / ``al`` -- but
  a hyphen the author wrote *is* a break, so ``tot-al`` may become ``tot-`` /
  ``al``. A run of two or more hyphens is punctuation and never a break.
* justify and vertical alignment are gone. Only ``left``, ``center`` and
  ``right`` exist, and anything else (a value saved by an older build) has to
  degrade to left rather than raise.

The font here is a stub rather than a real face: ``_length`` only ever asks for
``getlength``, and a fixed 10px per character makes "too wide for the line"
exact instead of dependent on whichever TTF happens to be installed.
"""
import pytest

from fox_reader.typeset import (
    TEXT_ALIGNMENTS,
    _ascent,
    _break_word,
    _draw_block,
    _hyphen_pieces,
    _layout,
    _wrap_paragraph,
)


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
        "align, anchor",
        [("left", "la"), ("center", "ma"), ("right", "ra"), ("justify", "la"), ("", "la")],
    )
    def test_draw_block_anchors_and_falls_back_to_left(self, align, anchor):
        draw = FakeDraw()
        _draw_block(draw, FakeFont(), 20, ["hi"], align,
                    (0, 0, 100, 40), (0, 0, 0, 255), 0, None)
        assert [call[2] for call in draw.calls] == [anchor]

    def test_block_is_centred_vertically(self):
        draw = FakeDraw()
        # One line, 10px tall (ascent 8 + descent 2), in a 40px-tall box.
        _draw_block(draw, FakeFont(), 20, ["hi"], "left",
                    (0, 0, 100, 40), (0, 0, 0, 255), 0, None)
        assert draw.calls[0][0][1] == pytest.approx(15.0)

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
        """Idle fallback is pixel-identical: no splitting, old anchors."""
        primary = FakeFont(ascent=8)
        symbol = FakeFont(ascent=20)
        draw = RecordingDraw()
        _draw_block(draw, primary, 20, ["hi"], "center",
                    (0, 0, 100, 40), (0, 0, 0, 255), 0, None,
                    symbol, {ord("h"), ord("i")}, {ord(self.HEART)})
        assert len(draw.calls) == 1
        assert draw.calls[0]["line"] == "hi"
        assert draw.calls[0]["anchor"] == "ma"

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
