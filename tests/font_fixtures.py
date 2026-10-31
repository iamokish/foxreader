"""Generated font fixtures; no optional user fonts are needed by the tests."""

from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

TEXT_FIXTURE_NAMES = ("TestShort.ttf", "TestFractional.ttf", "TestTall.ttf")
ART_FIXTURE_NAME = "TestArtSymbols.ttf"


def make_font(
    path: Path, characters: dict[str, int], *, units: int = 1000, advance: int = 600,
    bottoms: dict[str, int] | None = None, hollow: str = "",
):
    """Build simple outlines with known heights, bottoms and fractional advances.

    Hollow glyphs have a reversed inner contour, protecting counter rendering
    at large outline widths without requiring a particular third-party face.
    """
    builder = FontBuilder(units, isTTF=True)
    names = {ord(ch): f"u{ord(ch):X}" for ch in characters}
    builder.setupGlyphOrder([".notdef", *names.values()])
    builder.setupCharacterMap(names)
    glyphs = {}
    glyph_bottoms = {names[ord(ch)]: bottom for ch, bottom in (bottoms or {}).items()}
    hollow_names = {names[ord(ch)] for ch in hollow}
    for name, height in [(".notdef", 950), *((names[ord(ch)], h) for ch, h in characters.items())]:
        pen = TTGlyphPen(None)
        if height:
            bottom = glyph_bottoms.get(name, 0)
            pen.moveTo((0, bottom))
            pen.lineTo((500, bottom))
            pen.lineTo((500, bottom + height))
            pen.lineTo((0, bottom + height))
            pen.closePath()
            if name in hollow_names:
                inset = max(1, height // 5)
                pen.moveTo((100, bottom + inset))
                pen.lineTo((100, bottom + height - inset))
                pen.lineTo((400, bottom + height - inset))
                pen.lineTo((400, bottom + inset))
                pen.closePath()
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({name: (advance, 0) for name in glyphs})
    builder.setupHorizontalHeader(ascent=units, descent=-units // 4)
    builder.setupNameTable({
        "familyName": path.stem,
        "styleName": "Regular",
        "uniqueFontIdentifier": path.stem,
        "fullName": path.stem,
        "psName": path.stem,
    })
    builder.setupOS2(sTypoAscender=units, sTypoDescender=-units // 4,
                     usWinAscent=units * 4, usWinDescent=units // 4)
    builder.setupPost()
    builder.setupMaxp()
    builder.save(path)
    return path.read_bytes()


def make_layout_fonts(directory: Path) -> None:
    """Cover cap heights, em units, descenders, raised art and empty glyphs."""
    for name, units, cap_height, advance in (
        (TEXT_FIXTURE_NAMES[0], 1000, 600, 550),
        (TEXT_FIXTURE_NAMES[1], 1000, 800, 625),
        (TEXT_FIXTURE_NAMES[2], 2000, 2000, 1550),
    ):
        characters = {chr(cp): cap_height for cp in range(33, 127)}
        characters[" "] = 0
        characters["?"] = cap_height * 3 // 4
        characters[","] = cap_height // 4
        make_font(directory / name, characters, units=units, advance=advance,
                  bottoms={"?": -units // 20, ",": -units // 5})
    make_font(directory / ART_FIXTURE_NAME,
              {**dict.fromkeys("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 400), "?": 400, "!": 500, "Á": 0, " ": 0},
              advance=675, bottoms={"?": 200, "!": 100}, hollow="?!")
