from __future__ import annotations

import base64
import logging
import unicodedata

from fox_reader.font_calibration import FontProfile, profile_for, replacement_metrics_for, scale_for
from fox_reader.fonts import SYMBOL_FONT_FILENAME, getFonts, symbol_fallback_payload
from fox_reader.settings import SettingsManager, normalize_symbol_characters, normalize_symbol_fallbacks
from fox_reader.utils import FONT_DIR

logger = logging.getLogger(__name__)


class FontService:
    """Loads and caches font data at startup."""

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._fonts = getFonts(FONT_DIR)
        self._settings = settings
        self._raw = {
            f["font_filename"]: base64.b64decode(f["data_uri"].split(",", 1)[1], validate=True) for f in self._fonts
        }
        self._profiles = {}
        for font in self._fonts:
            calibration = font["calibration"]
            ranges = font["coverage"]
            coverage = frozenset(cp for i in range(0, len(ranges), 2) for cp in range(ranges[i], ranges[i + 1] + 1))
            self._profiles[font["font_filename"]] = FontProfile(
                coverage, calibration["cap_height"], calibration["symbol_height"], calibration["symbol_samples"]
            )
        self._view: list[dict] | None = None
        self._view_noto: dict | None = None
        self._active: tuple[str, ...] = ()
        self._characters: dict[str, str] = {}
        self._glyph_catalogs: dict[str, tuple[dict, ...]] = {}
        self._apply(settings.get_symbol_fallbacks() if settings else ())

    def validate_symbol_fallbacks(self, value: object) -> tuple[str, ...]:
        names = normalize_symbol_fallbacks(value, strict=True)
        unknown = set(names) - self._raw.keys()
        if unknown:
            raise ValueError("Font is not loaded: " + ", ".join(sorted(unknown)))
        if names and len(names) >= len(self._fonts):
            raise ValueError("Keep at least one font available for text")
        for name in names:
            profile = self._profiles[name]
            if not profile.symbol_samples:
                raise ValueError(f"Font has no usable symbol outlines: {name}")
        return names

    def _apply(self, names: tuple[str, ...]) -> None:
        from fox_reader.typeset import configure_fonts

        active = tuple(name for name in names if name in self._raw)
        if active and len(active) >= len(self._fonts):
            # A user may delete their remaining text font between launches.
            # Preserve their saved preference and reserve one usable text face.
            active = active[:-1]
            logger.warning("Reserving a text font because all remaining fonts were selected as fallbacks")
        self._active = active
        saved_characters = self._settings.get_symbol_characters() if self._settings else {}
        self._characters = {}
        for name in (*active, SYMBOL_FONT_FILENAME):
            profile = self._fallback_profile(name)
            valid = "".join(ch for ch in saved_characters.get(name, "") if ord(ch) in profile.coverage)
            if valid:
                self._characters[name] = valid
        self._view = None
        default = next((f["font_filename"] for f in self._fonts if f["font_filename"] not in active), "ComicMono.ttf")
        self._glyph_adjustments = self._build_glyph_adjustments()
        configure_fonts(self._raw, active, default, self._profiles, self._characters, self._glyph_adjustments)

    def _build_glyph_adjustments(self) -> dict:
        if not self._characters:
            return {}
        fallback_bytes = {name: self._raw[name] for name in self._characters if name in self._raw}
        if SYMBOL_FONT_FILENAME in self._characters:
            noto = symbol_fallback_payload()
            if noto:
                fallback_bytes[SYMBOL_FONT_FILENAME] = base64.b64decode(noto["data_uri"].split(",", 1)[1])
        result = {}
        for name, raw in self._raw.items():
            if name in self._active:
                continue
            adjustments = {}
            for filename, fallback_raw in fallback_bytes.items():
                metrics = replacement_metrics_for(raw, fallback_raw, self._characters[filename])
                if metrics:
                    adjustments[filename] = metrics
            if adjustments:
                result[name] = adjustments
        return result

    def _fallback_profile(self, name: str) -> FontProfile:
        if name in self._profiles:
            return self._profiles[name]
        noto = symbol_fallback_payload() if name == SYMBOL_FONT_FILENAME else None
        return profile_for(base64.b64decode(noto["data_uri"].split(",", 1)[1])) if noto else FontProfile(frozenset())

    def update_symbol_fallbacks(self, value: object, characters: object = None) -> dict:
        names = self.validate_symbol_fallbacks(value)
        if self._settings is None:
            raise RuntimeError("Font settings manager is unavailable")
        if characters is None:
            characters = {
                name: chars
                for name, chars in self._settings.get_symbol_characters().items()
                if name in (*names, SYMBOL_FONT_FILENAME)
            }
        rules = normalize_symbol_characters(characters, names, strict=True)
        for name, chars in rules.items():
            coverage = self._fallback_profile(name).coverage
            missing = [f"U+{ord(ch):04X}" for ch in chars if ord(ch) not in coverage]
            if missing:
                raise ValueError(f"{name} does not contain a usable glyph for: {', '.join(missing)}")
        self._settings.update_symbol_fallbacks(names, rules)
        self._apply(names)
        return self.settings_payload()

    def settings_payload(self) -> dict:
        saved = self._settings.get_symbol_fallbacks() if self._settings else self._active
        available = []
        for font in self._fonts:
            profile = self._profiles[font["font_filename"]]
            unicode_symbols = any(
                cp >= 0x2190
                and (unicodedata.category(chr(cp)).startswith("S") or unicodedata.category(chr(cp)) == "Co")
                for cp in profile.coverage
            )
            available.append(
                {
                    "font_filename": font["font_filename"],
                    "font_name": font["font_name"],
                    "limited_symbol_mapping": not unicode_symbols,
                    "calibrated": bool(profile.symbol_samples),
                }
            )
        return {
            "symbol_fallbacks": list(saved),
            "symbol_characters": self._settings.get_symbol_characters() if self._settings else dict(self._characters),
            "active": list(self._active),
            "available": available,
            "max_symbol_fallbacks": 3,
            "max_characters_per_font": 25,
            "final_fallback": SYMBOL_FONT_FILENAME,
        }

    def get_all(self) -> list[dict]:
        noto = symbol_fallback_payload()
        if self._view is not None and noto is self._view_noto:
            return self._view
        symbol_profiles = {name: self._profiles[name] for name in self._active}
        if noto:
            symbol_profiles[SYMBOL_FONT_FILENAME] = profile_for(base64.b64decode(noto["data_uri"].split(",", 1)[1]))
        result = []
        for font in self._fonts:
            name = font["font_filename"]
            if name in self._active:
                continue
            primary = self._profiles[name]
            scales = {filename: scale_for(primary, profile) for filename, profile in symbol_profiles.items()}
            result.append(
                {
                    **font,
                    "symbol_scales": scales,
                    "symbol_scale": scales.get(SYMBOL_FONT_FILENAME, 1.0),
                    "symbol_glyph_adjustments": self._glyph_adjustments.get(name, {}),
                }
            )
        self._view, self._view_noto = result, noto
        return result

    def get_symbol_fallbacks(self) -> list[dict]:
        fonts = {font["font_filename"]: font for font in self._fonts}
        result = [fonts[name] for name in self._active]
        noto = symbol_fallback_payload()
        if noto:
            result.append(noto)
        return [
            {**font, "forced_codepoints": [ord(ch) for ch in self._characters.get(font["font_filename"], "")]}
            for font in result
        ]

    def get_available(self) -> list[dict]:
        noto = symbol_fallback_payload()
        return [
            {**font, "hidden": font["font_filename"] in (*self._active, SYMBOL_FONT_FILENAME)}
            for font in [*self._fonts, *([noto] if noto else [])]
        ]

    def get_glyphs(self, filename: str, query: str = "", offset: int = 0, limit: int = 96) -> dict:
        if filename not in self._raw and filename != SYMBOL_FONT_FILENAME:
            raise ValueError("Font is not loaded")
        catalog = self._glyph_catalogs.get(filename)
        if catalog is None:
            coverage = self._fallback_profile(filename).coverage
            catalog = tuple(
                {
                    "character": chr(cp),
                    "codepoint": cp,
                    "unicode": f"U+{cp:04X}",
                    "name": unicodedata.name(chr(cp), "PRIVATE USE CHARACTER"),
                }
                for cp in sorted(coverage)
                if not chr(cp).isspace()
                and (not unicodedata.category(chr(cp)).startswith("C") or unicodedata.category(chr(cp)) == "Co")
            )
            self._glyph_catalogs[filename] = catalog
        key = query.strip().casefold()
        matches = [
            glyph
            for glyph in catalog
            if not key
            or key in glyph["character"].casefold()
            or key in glyph["name"].casefold()
            or key in glyph["unicode"].casefold()
        ]
        return {
            "font_filename": filename,
            "glyphs": matches[offset : offset + limit],
            "total": len(matches),
            "offset": offset,
            "limit": limit,
        }
