"""Ordered fallback selection, optical calibration and preview/export contract."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from io import BytesIO
from itertools import product
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from fox_reader import assets, fonts, typeset
from fox_reader.font_calibration import glyph_metrics_for, profile_for, replacement_metrics_for, scale_for
from fox_reader.services import font_service
from fox_reader.settings import FoxSettings, SettingsManager
from tests.font_fixtures import ART_FIXTURE_NAME, TEXT_FIXTURE_NAMES, make_font, make_layout_fonts

REPO_FONTS = Path(__file__).resolve().parents[1] / "fonts"
TEXT_TEST_FONTS = (*assets.BUNDLED_FONTS, *TEXT_FIXTURE_NAMES)


class RecordingDraw:
    def __init__(self, canvas):
        self.draw = ImageDraw.Draw(canvas) if canvas is not None else None
        self.calls = []

    def text(self, xy, text, **kwargs):
        self.calls.append({"xy": xy, "text": text, **kwargs})
        if self.draw is not None:
            self.draw.text(xy, text, **kwargs)


@pytest.mark.parametrize("filename", ["ComicMono.ttf", "ComicMono-Bold.ttf"])
def test_font_registration_preserves_original_disk_rendering(monkeypatch, filename):
    path = REPO_FONTS / filename
    sanitized = fonts._sanitize_and_get_bytes(path, filename)
    monkeypatch.setattr(typeset, "FONT_DIR", REPO_FONTS)
    monkeypatch.setattr(typeset, "_REGISTERED_FONT_BYTES", {filename: sanitized})
    monkeypatch.setattr(typeset, "_USER_SYMBOL_FONTS", ())
    monkeypatch.setattr(typeset, "_FONT_CACHE", {})
    for size in (12, 20, 24):
        expected = ImageFont.truetype(str(path), size)
        actual = typeset._load_font(filename, size)
        assert actual.getmetrics() == expected.getmetrics()
        assert bytes(actual.getmask("She really went and did it")) == bytes(
            expected.getmask("She really went and did it")
        )
        assert typeset._raw_length(actual, "Hello") is not None


@pytest.fixture
def service(tmp_path, monkeypatch):
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    make_font(font_dir / "Text.ttf", {"H": 800, "i": 600, "!": 800, " ": 0})
    make_font(
        font_dir / "First.ttf", {**dict.fromkeys("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 400), "♡": 400, "★": 400}, advance=900
    )
    make_font(font_dir / "Second.ttf", {"H": 800, "♡": 800, "✓": 800}, advance=1000)
    make_font(font_dir / "Third.ttf", {"\ue001": 1600, "😀": 1600}, advance=1100)
    shutil.copy(REPO_FONTS / fonts.SYMBOL_FONT_FILENAME, font_dir)
    old = (typeset._REGISTERED_FONT_BYTES, typeset._USER_SYMBOL_FONTS, typeset._DEFAULT_TEXT_FONT)
    monkeypatch.setattr(font_service, "FONT_DIR", font_dir)
    monkeypatch.setattr(typeset, "FONT_DIR", font_dir)
    # A disk signature from an earlier test must not masquerade as this folder.
    monkeypatch.setattr(typeset, "_SYMBOL_BYTES", None)
    monkeypatch.setattr(typeset, "_SYMBOL_BYTES_SIG", None)
    monkeypatch.setattr(typeset, "_SYMBOL_BYTES_MISSING", False)
    manager = SettingsManager(tmp_path / "config")
    manager.load()
    result = font_service.FontService(manager)
    yield result, manager
    typeset.configure_fonts(*old)


def test_order_is_persisted_hidden_and_reversible(service):
    svc, manager = service
    order = ["First.ttf", "Second.ttf", "Third.ttf"]
    svc.update_symbol_fallbacks(order)
    assert [font["font_filename"] for font in svc.get_all()] == ["Text.ttf"]
    assert [font["font_filename"] for font in svc.get_symbol_fallbacks()] == [*order, fonts.SYMBOL_FONT_FILENAME]
    reloaded = SettingsManager(manager.config_root)
    assert reloaded.load().fonts.symbol_fallbacks == tuple(order)
    svc.update_symbol_fallbacks(["Second.ttf", "First.ttf"])
    assert [font["font_filename"] for font in svc.get_symbol_fallbacks()][:2] == ["Second.ttf", "First.ttf"]
    svc.update_symbol_fallbacks([])
    assert len(svc.get_all()) == 4
    assert [font["font_filename"] for font in svc.get_symbol_fallbacks()] == [fonts.SYMBOL_FONT_FILENAME]


def test_backend_resolves_each_character_and_uses_transmitted_scales(service):
    svc, _ = service
    svc.update_symbol_fallbacks(["First.ttf", "Second.ttf", "Third.ttf"])
    primary = typeset._load_font("Text.ttf", 20)
    text = "Hi♡✓\ue001■!"
    cascade, pc, sc = typeset._symbol_faces("Text.ttf", text, 20)
    scales = svc.get_all()[0]["symbol_scales"]
    assert scales["First.ttf"] == 2
    assert scales["Second.ttf"] == 1
    assert scales["Third.ttf"] == 0.5
    assert cascade.face_for("♡").size == 40
    assert cascade.face_for("✓").size == 20
    assert cascade.face_for("\ue001").size == 10
    expected = sum(typeset._length(cascade.face_for(ch) if ord(ch) in sc else primary, ch) for ch in text)
    assert typeset._mixed_length(primary, cascade, text, pc, sc) == pytest.approx(expected)
    pieces = typeset._line_pieces(text, primary, cascade, pc, sc, 0, typeset._length(primary, " "))
    assert pieces[0][1] is primary
    assert pieces[2][1] is cascade.face_for("♡")
    image = Image.new("RGBA", (500, 200), "white")
    typeset._draw_block(
        ImageDraw.Draw(image), primary, 20, [text], "center", (0, 0, 500, 200), (0, 0, 0, 255), 0, None, cascade, pc, sc
    )
    assert image.getextrema()[0][0] == 0
    fitted = typeset._fit_font_size("Text.ttf", text, "center", 150, 100)
    face = typeset._load_font("Text.ttf", fitted)
    sym, pc, sc = typeset._symbol_faces("Text.ttf", text, fitted)
    lines = typeset._layout(face, text, 150, "center", sym, pc, sc)
    assert max(typeset._mixed_length(face, sym, line, pc, sc) for line in lines) <= 150
    svc.update_symbol_fallbacks(["Second.ttf", "First.ttf", "Third.ttf"])
    reordered, _, _ = typeset._symbol_faces("Text.ttf", "♡", 20)
    assert reordered.face_for("♡").size == 20


def test_existing_primary_characters_and_unknown_glyphs_stay_primary(service):
    svc, _ = service
    svc.update_symbol_fallbacks(["First.ttf"])
    assert typeset._symbol_faces("Text.ttf", "Hi!\U0010ffff", 20) == (None, None, None)
    default = svc.get_all()[0]["font_filename"]
    assert typeset._effective_fontfile("First.ttf") == default
    assert typeset._load_font("First.ttf", 20) is typeset._load_font(default, 20)


@pytest.mark.parametrize(
    "invalid",
    [
        ["First.ttf", "First.ttf"],
        ["First.ttf", "FIRST.TTF"],
        ["Missing.ttf"],
        [fonts.SYMBOL_FONT_FILENAME],
        ["../First.ttf"],
        ["C:\\First.ttf"],
        [5],
        "First.ttf",
        None,
        ["First.ttf", "Second.ttf", "Third.ttf", "Text.ttf"],
    ],
)
def test_invalid_selections_leave_settings_and_runtime_unchanged(service, invalid):
    svc, manager = service
    before = manager.settings_path.read_bytes()
    with pytest.raises(ValueError):
        svc.update_symbol_fallbacks(invalid)
    assert manager.settings_path.read_bytes() == before
    assert typeset._USER_SYMBOL_FONTS == ()


def test_keep_a_text_font_and_handle_deleted_preferences(service):
    svc, manager = service
    original = svc._fonts
    svc._fonts = original[:3]
    with pytest.raises(ValueError, match="at least one"):
        svc.validate_symbol_fallbacks([font["font_filename"] for font in svc._fonts])
    reserved = tuple(font["font_filename"] for font in svc._fonts)
    svc._apply(reserved)
    assert len(svc.get_all()) == 1
    assert len(svc.settings_payload()["active"]) == 2
    svc._fonts = original
    manager.update_symbol_fallbacks(["Gone.ttf", "First.ttf"])
    svc._apply(manager.get_symbol_fallbacks())
    assert svc.settings_payload()["symbol_fallbacks"] == ["Gone.ttf", "First.ttf"]
    assert svc.settings_payload()["active"] == ["First.ttf"]


def test_route_contract_and_invalid_json(service):
    from fox_reader.routes.folder import router as font_router
    from fox_reader.routes.settings import router as settings_router

    svc, _ = service
    app = FastAPI()
    app.state.fonts = svc
    app.include_router(font_router)
    app.include_router(settings_router)
    client = TestClient(app)
    assert client.put("/api/settings/fonts", json={"symbol_fallbacks": ["First.ttf", "Second.ttf"]}).status_code == 200
    text = client.get("/api/fonts").json()
    assert all(font["font_filename"] not in ("First.ttf", "Second.ttf") for font in text)
    assert text[0]["symbol_scales"]["First.ttf"] == 2
    fallback = client.get("/api/fonts/symbol-fallbacks").json()
    assert [f["font_filename"] for f in fallback] == ["First.ttf", "Second.ttf", fonts.SYMBOL_FONT_FILENAME]
    assert all(f["hidden"] for f in fallback)
    invalid = client.put("/api/settings/fonts", content="{", headers={"Content-Type": "application/json"})
    assert invalid.status_code == 400
    assert client.put("/api/settings/fonts", json=[]).status_code == 400


@pytest.mark.parametrize("symbol_height,expected_scale", [(200, 3.0), (320, 2.5)])
def test_scales_above_two_are_transmitted_and_used_for_rendering(service, symbol_height, expected_scale):
    _, manager = service
    make_font(font_service.FONT_DIR / "Small.ttf", {"♡": symbol_height}, advance=900)
    svc = font_service.FontService(manager)
    svc.update_symbol_fallbacks(["Small.ttf"])
    text_font = next(font for font in svc.get_all() if font["font_filename"] == "Text.ttf")
    assert text_font["symbol_scales"]["Small.ttf"] == expected_scale
    primary = typeset._load_font("Text.ttf", 20)
    cascade, pc, sc = typeset._symbol_faces("Text.ttf", "H♡", 20)
    symbol = cascade.face_for("♡")
    assert symbol.size == int(20 * expected_scale)
    assert typeset._mixed_length(primary, cascade, "H♡", pc, sc) == pytest.approx(
        typeset._length(primary, "H") + 18 * expected_scale
    )


def test_calibration_ignores_tofu_empty_glyphs_aliases_and_outliers(tmp_path):
    primary = profile_for(make_font(tmp_path / "Primary.ttf", {"E": 800, "M": 800, "N": 800}))
    raw = make_font(
        tmp_path / "Sparse.ttf",
        {
            "\ue001": 400,
            "\ue002": 400,
            "\ue003": 400,
            "\ue004": 400,
            "\ue005": 400,
            "\ue006": 3500,
            "\ue007": 0,
        },
    )
    with TTFont(BytesIO(raw)) as face:
        original = face.getBestCmap()[0xE001]
        for table in face["cmap"].tables:
            if table.isUnicode():
                table.cmap[0xE011] = original
        face.save(tmp_path / "Alias.ttf")
    fallback = profile_for((tmp_path / "Alias.ttf").read_bytes())
    assert ord("H") not in primary.coverage
    assert ord("♡") not in fallback.coverage
    assert ord("\ue007") not in fallback.coverage
    assert fallback.symbol_height == pytest.approx(0.4)
    assert fallback.symbol_samples == 6  # one alias and one empty glyph are excluded
    assert scale_for(primary, fallback) == 2
    assert profile_for(b"bad font").symbol_samples == 0
    assert scale_for(primary, profile_for(b"bad font")) == 1


def test_em_units_do_not_change_calibration_and_non_latin_caps_work(tmp_path):
    primary = profile_for(make_font(tmp_path / "CJK.ttf", {"漢": 800, "字": 800}))
    a = profile_for(make_font(tmp_path / "A.ttf", {"\ue001": 400}))
    b = profile_for(make_font(tmp_path / "B.ttf", {"\ue001": 800}, units=2000))
    assert scale_for(primary, a) == scale_for(primary, b) == 2


def test_forced_glyph_metrics_use_individual_geometry_and_bounded_scales(tmp_path, monkeypatch):
    primary = make_font(
        tmp_path / "Primary.ttf", {"H": 800, "?": 600, ".": 100, " ": 0}, bottoms={"?": -50}
    )
    fallback = make_font(
        tmp_path / "Fallback.ttf", {"H": 200, "?": 800, ".": 1000, "x": 800, " ": 0},
        units=2000, bottoms={"?": 400},
    )
    metrics = replacement_metrics_for(primary, fallback, "?H.x ")
    assert set(metrics) == {str(ord(ch)) for ch in "?H."}
    assert metrics[str(ord("?"))] == pytest.approx(
        {"scale": 1.5, "primary_bottom": -0.05, "fallback_bottom": 0.2}
    )
    assert metrics[str(ord("H"))]["scale"] == 3
    assert metrics[str(ord("."))]["scale"] == 0.5
    assert glyph_metrics_for(b"bad font", "?") == {}
    pathological = make_font(tmp_path / "Pathological.ttf", {"?": 9000})
    assert glyph_metrics_for(pathological, "?") == {}

    def unexpected(*args, **kwargs):
        raise AssertionError("unchanged assignment geometry was parsed again")

    monkeypatch.setattr("fox_reader.font_calibration.TTFont", unexpected)
    assert replacement_metrics_for(primary, fallback, " x.H??") == metrics


def test_forced_glyph_without_a_primary_reference_keeps_font_calibration(service):
    _, manager = service
    make_font(font_service.FONT_DIR / "Raised.ttf", {"?": 400}, bottoms={"?": 200})
    svc = font_service.FontService(manager)
    svc.update_symbol_fallbacks(["Raised.ttf"], {"Raised.ttf": "?"})
    text_font = next(font for font in svc.get_all() if font["font_filename"] == "Text.ttf")
    assert "Raised.ttf" not in text_font["symbol_glyph_adjustments"]
    cascade, pc, sc = typeset._symbol_faces("Text.ttf", "H?", 20)
    assert cascade.face_for("?").size == 40
    primary = typeset._load_font("Text.ttf", 20)
    draw = RecordingDraw(Image.new("RGBA", (240, 240)))
    typeset._draw_block(
        draw, primary, 20, ["H?"], "left", (0, 0, 240, 240), (0, 0, 0, 255), 0, None, cascade, pc, sc
    )
    assert [call["anchor"] for call in draw.calls] == ["ls", "ls"]
    assert draw.calls[0]["xy"][1] == draw.calls[1]["xy"][1]


@pytest.mark.parametrize("align", ["left", "center", "right"])
@pytest.mark.parametrize("word_extra", [0, 4])
def test_assigned_glyphs_match_original_bottoms_and_missing_glyphs_keep_baselines(service, align, word_extra):
    _, manager = service
    make_font(
        font_service.FONT_DIR / "Raised.ttf", {"H": 400, "!": 400}, bottoms={"H": 300, "!": -250}
    )
    svc = font_service.FontService(manager)
    svc.update_symbol_fallbacks(["Raised.ttf", "First.ttf", "Second.ttf"], {"Raised.ttf": "H!"})
    size = 20
    primary = typeset._load_font("Text.ttf", size)
    line = "H i!♡✓■"
    cascade, pc, sc = typeset._symbol_faces("Text.ttf", line, size)
    assert cascade.face_for("H").size == cascade.face_for("!").size == 40
    assert cascade.face_for("♡") is not cascade.face_for("✓")
    assert cascade.face_for("■") is not cascade.face_for("♡")
    draw = RecordingDraw(Image.new("RGBA", (500, 400)))
    typeset._draw_block(
        draw, primary, size, [line, line], align, (0, 0, 500, 400), (0, 0, 0, 255), 0, None,
        cascade, pc, sc, word_extra=word_extra,
    )
    ascent, descent = primary.getmetrics()
    advance = typeset._advance_for(ascent + descent, typeset._spacing_for(size), None)
    baseline = (400 - (advance + ascent + descent)) / 2 + ascent
    glyphs_per_line = len(line.replace(" ", ""))
    assert len(draw.calls) == 2 * glyphs_per_line
    for index, call in enumerate(draw.calls):
        assert call["anchor"] == "ls"
        offset = {"H": 12, "!": -10}.get(call["text"], 0)
        assert call["xy"][1] == pytest.approx(baseline + (index // glyphs_per_line) * advance + offset)
    assert draw.calls[0]["font"] is cascade.face_for("H")
    assert draw.calls[1]["font"] is primary
    assert draw.calls[2]["font"] is cascade.face_for("!")


@pytest.mark.parametrize("primary_filename", TEXT_TEST_FONTS)
@pytest.mark.parametrize("character", ["?", "!"])
def test_generated_punctuation_matches_original_height_and_bottom(service, primary_filename, character):
    _, manager = service
    make_layout_fonts(font_service.FONT_DIR)
    symbol_path = font_service.FONT_DIR / ART_FIXTURE_NAME
    if primary_filename in assets.BUNDLED_FONTS:
        shutil.copy(REPO_FONTS / primary_filename, font_service.FONT_DIR)
    svc = font_service.FontService(manager)
    svc.update_symbol_fallbacks([symbol_path.name], {symbol_path.name: character})
    text_font = next(font for font in svc.get_all() if font["font_filename"] == primary_filename)
    metric = text_font["symbol_glyph_adjustments"][symbol_path.name][str(ord(character))]
    primary_outline = glyph_metrics_for(svc._raw[primary_filename], character)[ord(character)]
    fallback_outline = glyph_metrics_for(svc._raw[symbol_path.name], character)[ord(character)]
    for size in range(typeset.MIN_FONT_SIZE, typeset.MAX_FONT_SIZE + 1):
        primary = typeset._load_font(primary_filename, size)
        cascade, pc, sc = typeset._symbol_faces(primary_filename, character, size)
        replacement = cascade.face_for(character)
        assert replacement.size == typeset._round_half_up(size * metric["scale"])
        assert typeset._mixed_length(primary, cascade, character * 2, pc, sc) == pytest.approx(
            2 * typeset._length(replacement, character)
        )
        reference_image = Image.new("RGBA", (240, 240))
        replacement_image = Image.new("RGBA", (240, 240))
        expected_image = Image.new("RGBA", (240, 240))
        args = (primary, size, [character], "left", (30, 0, 240, 240), (0, 0, 0, 255), 0, None)
        typeset._draw_block(ImageDraw.Draw(reference_image), *args)
        typeset._draw_block(ImageDraw.Draw(replacement_image), *args, cascade, pc, sc)
        ascent, descent = primary.getmetrics()
        baseline = (240 - ascent - descent) / 2 + ascent
        offset = metric["fallback_bottom"] * replacement.size - metric["primary_bottom"] * size
        assert cascade.baseline_offsets[character] - fallback_outline[1] * replacement.size == pytest.approx(
            -primary_outline[1] * size
        )
        assert abs(fallback_outline[0] * replacement.size - primary_outline[0] * size) <= fallback_outline[0] / 2
        ImageDraw.Draw(expected_image).text(
            (30, baseline + offset), character, font=replacement, anchor="ls", fill=(0, 0, 0, 255)
        )
        assert replacement_image.tobytes() == expected_image.tobytes()
        reference_bounds, replacement_bounds = reference_image.getbbox(), replacement_image.getbbox()
        assert reference_bounds is not None and replacement_bounds is not None
        # Hinting, antialiasing and paste rounding differ between the two
        # artworks. Geometry above is exact; the raster envelope is bounded.
        assert abs(replacement_bounds[3] - reference_bounds[3]) <= 3
        assert abs((replacement_bounds[3] - replacement_bounds[1])
                   - (reference_bounds[3] - reference_bounds[1])) <= 4
    svc.update_symbol_fallbacks([symbol_path.name], {})
    assert typeset._symbol_faces(primary_filename, character, 20) == (None, None, None)
    assert next(font for font in svc.get_all() if font["font_filename"] == primary_filename)[
        "symbol_glyph_adjustments"
    ] == {}


@pytest.mark.parametrize("invalid", [
    {"scale": float("nan")}, {"scale": 0.49}, {"scale": 3.01}, {"scale": True},
    {"primary_bottom": float("inf")}, {"primary_bottom": -9},
    {"fallback_bottom": float("nan")}, {"fallback_bottom": 9}, {"fallback_bottom": None},
])
def test_invalid_replacement_metrics_keep_normal_size_and_baseline(service, invalid):
    _, manager = service
    make_font(font_service.FONT_DIR / "Raised.ttf", {"H": 400}, bottoms={"H": 300})
    svc = font_service.FontService(manager)
    svc.update_symbol_fallbacks(["Raised.ttf"], {"Raised.ttf": "H"})
    typeset._FORCED_SYMBOL_ADJUSTMENTS["Text.ttf"]["Raised.ttf"][str(ord("H"))] = {
        "scale": 1.5, "primary_bottom": 0.05, "fallback_bottom": 0.1, **invalid,
    }
    cascade, _, _ = typeset._symbol_faces("Text.ttf", "H", 20)
    assert cascade.face_for("H").size == 40
    assert cascade.baseline_offsets == {}


def test_actual_typescript_matches_backend_symbol_layout(service):
    """Exercise the real TS fit/draw code, not a Python port of the formulas."""
    root = REPO_FONTS.parent
    node = shutil.which("node")
    if not node or not (root / "frontend/node_modules/rolldown").is_dir():
        pytest.skip("Node and installed frontend dependencies are required")
    _, manager = service
    filenames = TEXT_TEST_FONTS
    symbol_name = ART_FIXTURE_NAME
    make_layout_fonts(font_service.FONT_DIR)
    for name in assets.BUNDLED_FONTS:
        shutil.copy(REPO_FONTS / name, font_service.FONT_DIR)
    svc = font_service.FontService(manager)
    svc.update_symbol_fallbacks([symbol_name], {symbol_name: "?!"})
    from fox_reader.routes.folder import router

    app = FastAPI()
    app.state.fonts = svc
    app.include_router(router)
    client = TestClient(app)
    fonts_payload = client.get("/api/fonts").json()
    symbols_payload = client.get("/api/fonts/symbol-fallbacks").json()
    faces = {font["font_filename"]: font for font in (*fonts_payload, *symbols_payload)}
    texts = ("HELLO", "HELLO?!$%@", "HELLO? Hello!\n\n!? Hello?", "HELLO? HELLO! HELLO?!\u2661")
    characters = set("M " + "".join(texts)) - {"\n"}
    advances = {}
    for name, payload in faces.items():
        if name == fonts.SYMBOL_FONT_FILENAME:
            face = typeset._load_symbol_font(100)
        elif name == symbol_name:
            face = typeset._load_user_symbol_font(name, 100)
        else:
            face = typeset._load_font(name, 100)
        family = payload["font_name"].replace(" ", "_")
        advances[family] = {character: typeset._length(face, character) for character in characters}

    cases = []
    for name, explicit_size, align, word_spacing, line_spacing, text, stroke_width in product(
        filenames, (None, 6, 12, 20, 21, 40, 72), ("left", "center", "right"),
        (None, 2), (None, 1.3), texts, (0, 6, 14, 23),
    ):
        box = {"x": 20, "y": 30, "w": 240 if "\n" in text else 120, "h": 180}
        geom = typeset.TextGeometry(word_spacing=word_spacing, line_spacing=line_spacing)
        size = explicit_size or typeset._fit_font_size(name, text, align, box["w"], box["h"], geom=geom)
        primary = typeset._load_font(name, size)
        cascade, pc, sc = typeset._symbol_faces(name, text, size)
        extra = typeset._word_extra(primary, word_spacing)
        lines = typeset._layout(primary, text, box["w"], align, cascade, pc, sc, extra)
        draw = RecordingDraw(None)
        typeset._draw_block(
            draw, primary, size, lines, align,
            (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]),
            (0, 0, 0, 255), stroke_width, (255, 255, 255, 255), cascade, pc, sc, extra, line_spacing,
        )
        calls = []
        for call in draw.calls:
            filename = name if call["font"] is primary else (
                symbol_name if call["text"] in "?!" else fonts.SYMBOL_FONT_FILENAME
            )
            calls.append({"text": call["text"], "x": call["xy"][0], "y": call["xy"][1],
                          "family": faces[filename]["font_name"].replace(" ", "_"), "size": call["font"].size,
                          "fill": "black" if call["fill"] == (0, 0, 0, 255) else "white",
                          "strokeWidth": call.get("stroke_width", 0)})
        cases.append({"filename": name, "explicitSize": explicit_size, "size": size, "align": align,
                      "wordSpacing": word_spacing, "lineSpacing": line_spacing, "text": text,
                      "box": box, "lines": lines, "calls": calls, "strokeWidth": stroke_width})
    result = subprocess.run(
        [node, str(root / "tests/manual/symbol_typeset_contract.mjs")], cwd=root,
        input=json.dumps({"fonts": fonts_payload, "symbols": symbols_payload, "advances": advances, "cases": cases}),
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"{len(cases)} frontend/backend symbol layouts matched"


@pytest.mark.parametrize("stroke_width", [14, 23, 32])
def test_thick_symbol_outlines_preserve_hollows_and_antialiased_ink(service, stroke_width):
    """The complete clean glyph, including its hollow counter, must survive."""
    _, manager = service
    symbol_name = ART_FIXTURE_NAME
    names = TEXT_TEST_FONTS
    make_layout_fonts(font_service.FONT_DIR)
    for name in assets.BUNDLED_FONTS:
        shutil.copy(REPO_FONTS / name, font_service.FONT_DIR)
    svc = font_service.FontService(manager)
    svc.update_symbol_fallbacks([symbol_name], {symbol_name: "?!"})
    for name, character, (mode, stroke_color) in product(
        names, "?!", (("RGB", (255, 255, 255)), ("RGBA", (255, 255, 255, 128))),
    ):
        font = typeset._load_font(name, 40)
        cascade, pc, sc = typeset._symbol_faces(name, character, 40)
        box = (50, 0, 240, 240)
        ink = Image.new("L", (240, 240))
        args = (font, 40, [character], "left", box)
        typeset._draw_block(ImageDraw.Draw(ink), *args, 255, 0, None, cascade, pc, sc)
        bounds = ink.getbbox()
        assert bounds is not None
        assert ink.crop(bounds).getextrema() == (0, 255), "the test glyph must have a hollow counter"
        reference = Image.new(mode, ink.size, stroke_color)
        actual = Image.new(mode, ink.size, 0)
        fill = (0, 0, 0) if mode == "RGB" else (0, 0, 0, 255)
        typeset._draw_block(ImageDraw.Draw(reference), *args, fill, 0, None, cascade, pc, sc)
        typeset._draw_block(ImageDraw.Draw(actual), *args, fill, stroke_width, stroke_color, cascade, pc, sc)
        assert actual.crop(bounds).tobytes() == reference.crop(bounds).tobytes(), (
            name, character, stroke_width, mode
        )


def test_letter_mapped_symbol_font_uses_artwork_not_absent_unicode_probes(tmp_path):
    make_layout_fonts(tmp_path)
    path = tmp_path / ART_FIXTURE_NAME
    profile = profile_for(path.read_bytes())
    assert profile.symbol_samples >= 5
    assert math.isfinite(profile.symbol_height) and profile.symbol_height > 0
    assert ord("♡") not in profile.coverage
    assert ord("A") in profile.coverage
    assert ord("Á") not in profile.coverage  # listed in cmap but has an empty outline


def test_forced_letter_uses_artwork_for_measurement_and_drawing(service):
    _, manager = service
    make_layout_fonts(font_service.FONT_DIR)
    path = font_service.FONT_DIR / ART_FIXTURE_NAME
    svc = font_service.FontService(manager)
    svc.update_symbol_fallbacks([path.name], {path.name: "H"})
    primary = typeset._load_font("Text.ttf", 20)
    cascade, pc, sc = typeset._symbol_faces("Text.ttf", "Hi", 20)
    artwork = cascade.face_for("H")
    assert artwork is not primary
    text_font = next(font for font in svc.get_all() if font["font_filename"] == "Text.ttf")
    scale = text_font["symbol_glyph_adjustments"][path.name][str(ord("H"))]["scale"]
    assert artwork.size == typeset._round_half_up(20 * scale)
    assert typeset._mixed_length(primary, cascade, "Hi", pc, sc) == pytest.approx(
        typeset._length(artwork, "H") + typeset._length(primary, "i")
    )
    pieces = typeset._line_pieces("Hi", primary, cascade, pc, sc, 0, typeset._length(primary, " "))
    assert [(text, face) for text, face, _ in pieces] == [("H", artwork), ("i", primary)]
    image = Image.new("L", (200, 200))
    ImageDraw.Draw(image).text((50, 100), "H", font=artwork, fill=255, anchor="ls")
    assert image.getbbox() is not None
    svc.update_symbol_fallbacks([path.name], {})
    assert typeset._symbol_faces("Text.ttf", "H", 20)[0] is None


def test_optional_invalid_font_settings_do_not_erase_other_preferences():
    settings = FoxSettings.model_validate(
        {
            "fonts": {"symbol_fallbacks": ["First.ttf", "../bad.ttf", 7]},
            "deepl_api_token": "keep",
            "devices": {"paddleocr": "cpu"},
        }
    )
    assert settings.fonts.symbol_fallbacks == ("First.ttf",)
    assert settings.deepl_api_token == "keep"
    assert settings.devices.paddleocr == "cpu"
    assert FoxSettings.model_validate({"fonts": "broken", "jpdb_api_token": "keep"}).jpdb_api_token == "keep"


def test_failed_save_does_not_apply_selection(service, monkeypatch):
    svc, manager = service
    before = manager.settings_path.read_bytes()

    def fail(*args):
        raise OSError("read-only")

    monkeypatch.setattr("fox_reader.settings.os.replace", fail)
    with pytest.raises(OSError):
        svc.update_symbol_fallbacks(["First.ttf"])
    assert manager.get_symbol_fallbacks() == ()
    assert typeset._USER_SYMBOL_FONTS == ()
    assert manager.settings_path.read_bytes() == before


def test_version_change_rereads_all_fonts_and_matching_version_skips_parsing(tmp_path, monkeypatch):
    import fox_reader

    make_font(tmp_path / "A.ttf", {"H": 800})
    make_font(tmp_path / "B.ttf", {"H": 900})
    fonts.getFonts(tmp_path)
    cache = tmp_path / ".fonts_cache.json"
    assert json.loads(cache.read_text(encoding="utf-8"))["__version__"] == fox_reader.__version__
    original = fonts._sanitize_and_get_bytes
    parsed = []

    def record(source, filename):
        parsed.append(filename)
        return original(source, filename)

    monkeypatch.setattr(fonts, "_sanitize_and_get_bytes", record)
    fonts.getFonts(tmp_path)
    assert not parsed
    monkeypatch.setattr(fox_reader, "__version__", "test-new-version")
    fonts.getFonts(tmp_path)
    assert parsed == ["A.ttf", "B.ttf"]
    assert json.loads(cache.read_text(encoding="utf-8"))["__version__"] == "test-new-version"
    parsed.clear()
    fonts.getFonts(tmp_path)
    assert not parsed
    (tmp_path / "B.ttf").unlink()
    fonts.getFonts(tmp_path)
    assert "B.ttf" not in json.loads(cache.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("data_uri", "not a data URI"),
        ("data_uri", "data:font/ttf;base64,AA=="),
        ("symbol_scale", float("nan")),
        ("coverage", [100, 90]),
    ],
)
def test_malformed_cached_font_is_reparsed(tmp_path, monkeypatch, field, value):
    make_font(tmp_path / "A.ttf", {"H": 800})
    fonts.getFonts(tmp_path)
    cache = tmp_path / ".fonts_cache.json"
    index = json.loads(cache.read_text(encoding="utf-8"))
    index["A.ttf"]["payload"][field] = value
    cache.write_text(json.dumps(index), encoding="utf-8")
    original = fonts._sanitize_and_get_bytes
    parsed = []

    def record(source, filename):
        parsed.append(filename)
        return original(source, filename)

    monkeypatch.setattr(fonts, "_sanitize_and_get_bytes", record)
    assert fonts.getFonts(tmp_path)[0]["font_filename"] == "A.ttf"
    assert parsed == ["A.ttf"]


def test_reordering_reuses_loaded_font_faces_and_calibration_profiles(service, monkeypatch):
    svc, _ = service
    svc.update_symbol_fallbacks(["First.ttf", "Second.ttf"])
    cascade, _, _ = typeset._symbol_faces("Text.ttf", "♡", 20)
    loaded = cascade.face_for("♡")

    def unexpected(*args, **kwargs):
        raise AssertionError("reordering parsed an unchanged font")

    # The initial scan's profiles are restored from cache metadata, so no
    # outline walk or sanitisation is needed to change the hierarchy.
    monkeypatch.setattr(fonts, "_sanitize_and_get_bytes", unexpected)
    monkeypatch.setattr(typeset, "profile_for", unexpected)
    svc.update_symbol_fallbacks(["Second.ttf", "First.ttf"])
    svc.update_symbol_fallbacks(["First.ttf", "Second.ttf"])
    updated, _, _ = typeset._symbol_faces("Text.ttf", "♡", 20)
    assert updated.face_for("♡") is loaded


def test_assignments_override_primary_and_order_and_clear_reversibly(service):
    svc, manager = service
    order = ["First.ttf", "Second.ttf", "Third.ttf"]
    rules = {"Second.ttf": "H♡", "Third.ttf": "😀"}
    svc.update_symbol_fallbacks(order, rules)
    primary = typeset._load_font("Text.ttf", 20)
    text = "Hi♡★😀!"
    cascade, pc, sc = typeset._symbol_faces("Text.ttf", text, 20)
    assert cascade.face_for("H") is not primary  # primary covers H
    assert cascade.face_for("H").size == 20
    assert cascade.face_for("♡").size == 20  # Second wins despite First also covering it
    assert cascade.face_for("★").size == 40  # unassigned still follows the hierarchy
    pieces = typeset._line_pieces(text, primary, cascade, pc, sc, 0, typeset._length(primary, " "))
    assert pieces[0][1] is cascade.face_for("H")
    assert pieces[1][1] is primary
    expected = sum(typeset._length(cascade.faces.get(ch, primary), ch) for ch in text)
    assert typeset._mixed_length(primary, cascade, text, pc, sc) == pytest.approx(expected)
    assert SettingsManager(manager.config_root).load().fonts.symbol_characters == rules
    api_fonts = svc.get_symbol_fallbacks()
    assert api_fonts[1]["forced_codepoints"] == [ord("H"), ord("♡")]
    svc.update_symbol_fallbacks(list(reversed(order)))  # old callers preserve surviving assignments
    assert manager.get_symbol_characters() == rules
    svc.update_symbol_fallbacks(order, {})
    plain, _, _ = typeset._symbol_faces("Text.ttf", "H♡", 20)
    assert "H" not in plain.faces
    assert plain.face_for("♡").size == 40
    assert typeset._FORCED_SYMBOL_OWNERS == {}


def test_noto_can_own_a_character_a_user_fallback_already_covers(service):
    svc, _ = service
    svc.update_symbol_fallbacks(["First.ttf"], {fonts.SYMBOL_FONT_FILENAME: "★"})
    cascade, _, _ = typeset._symbol_faces("Text.ttf", "★", 20)
    assert cascade.face_for("★") is typeset._load_symbol_font(typeset._symbol_px_size("Text.ttf", 20))
    assert svc.get_symbol_fallbacks()[-1]["forced_codepoints"] == [ord("★")]
    svc.update_symbol_fallbacks([], {fonts.SYMBOL_FONT_FILENAME: "★"})
    assert typeset._symbol_faces("Text.ttf", "★", 20)[0].face_for("★") is not None


@pytest.mark.parametrize(
    "rules",
    [
        {"First.ttf": "H", "Second.ttf": "H"},
        {"First.ttf": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
        {"First.ttf": "✓"},
        {"Text.ttf": "H"},
        {"Ghost.ttf": "H"},
        {"First.ttf": " "},
        {"First.ttf": "\ud800"},
        {"First.ttf": "\u200d"},
        {"First.ttf": 7},
        [],
        {fonts.SYMBOL_FONT_FILENAME: "漢"},
    ],
)
def test_bad_character_rules_are_rejected_atomically(service, rules):
    svc, manager = service
    before = manager.settings_path.read_bytes()
    with pytest.raises(ValueError):
        svc.update_symbol_fallbacks(["First.ttf", "Second.ttf"], rules)
    assert manager.settings_path.read_bytes() == before
    assert manager.get_symbol_characters() == {}
    assert typeset._FORCED_SYMBOL_OWNERS == {}


def test_25_unique_characters_duplicate_input_and_unicode_scalar_counts(service):
    svc, manager = service
    text = "ABCDEFGHIJKLMNOPQRSTUVWXY"
    svc.update_symbol_fallbacks(["First.ttf", "Third.ttf"], {"First.ttf": text + "AA", "Third.ttf": "😀😀"})
    assert manager.get_symbol_characters() == {"First.ttf": text, "Third.ttf": "😀"}
    assert len(manager.get_symbol_characters()["First.ttf"]) == 25
    svc.update_symbol_fallbacks(["Third.ttf"])
    assert manager.get_symbol_characters() == {"Third.ttf": "😀"}


def test_glyph_catalog_search_and_paging_exclude_empty_and_control_glyphs(service):
    svc, _ = service
    stars = svc.get_glyphs("First.ttf", "BLACK STAR")
    assert [glyph["character"] for glyph in stars["glyphs"]] == ["★"]
    assert svc.get_glyphs("First.ttf", "U+0048")["glyphs"][0]["character"] == "H"
    page = svc.get_glyphs("First.ttf", offset=5, limit=3)
    assert len(page["glyphs"]) == 3
    assert page["offset"] == 5
    assert page["total"] == 28
    assert all(not glyph["character"].isspace() for glyph in svc.get_glyphs("Text.ttf")["glyphs"])
    with pytest.raises(ValueError):
        svc.get_glyphs("../First.ttf")


def test_character_rules_and_glyph_inventory_api(service):
    from fox_reader.routes.folder import router as font_router
    from fox_reader.routes.settings import router as settings_router

    svc, _ = service
    app = FastAPI()
    app.state.fonts = svc
    app.include_router(font_router)
    app.include_router(settings_router)
    client = TestClient(app)
    saved = client.put(
        "/api/settings/fonts", json={"symbol_fallbacks": ["First.ttf"], "symbol_characters": {"First.ttf": "H"}}
    )
    assert saved.status_code == 200
    assert saved.json()["fonts"]["symbol_characters"] == {"First.ttf": "H"}
    assert client.get("/api/fonts/symbol-fallbacks").json()[0]["forced_codepoints"] == [ord("H")]
    assert "First.ttf" in {f["font_filename"] for f in client.get("/api/fonts/available").json()}
    glyphs = client.get("/api/fonts/glyphs", params={"filename": "First.ttf", "query": "star"})
    assert glyphs.json()["glyphs"][0]["character"] == "★"
    assert client.get("/api/fonts/glyphs", params={"filename": "First.ttf", "limit": 500}).status_code == 400
    assert client.get("/api/fonts/glyphs", params={"filename": "Missing.ttf"}).status_code == 400
    assert (
        client.put("/api/settings/fonts", json={"symbol_fallbacks": [], "symbol_characters": None}).status_code == 400
    )


def test_invalid_saved_rules_do_not_reset_other_settings():
    settings = FoxSettings.model_validate(
        {
            "deepl_api_token": "keep",
            "fonts": {
                "symbol_fallbacks": ["First.ttf", "Second.ttf"],
                "symbol_characters": {"First.ttf": "HH\u200d", "Second.ttf": "H✓", "Other.ttf": "X"},
            },
        }
    )
    assert settings.deepl_api_token == "keep"
    assert settings.fonts.symbol_characters == {"First.ttf": "H", "Second.ttf": "✓"}
