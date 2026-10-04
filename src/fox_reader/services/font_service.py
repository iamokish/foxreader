from __future__ import annotations

from pathlib import Path

from fox_reader.fonts import getFonts
from fox_reader.utils import FONT_DIR


class FontService:
    """Loads and caches font data at startup."""

    def __init__(self) -> None:
        self._fonts = getFonts(FONT_DIR)

    def get_all(self) -> list[dict]:
        return self._fonts
