"""Opt-in scope for the two image display improvements, shared by every viewer."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)
IMAGE_RENDER_QUALITY_ENV = "FOX_IMAGE_RENDER_QUALITY"


@lru_cache(maxsize=16)
def _parse_mode(value: str | None) -> int:
    if value is None:
        return 1
    normalized = value.strip()
    if normalized in {"1", "2", "3"}:
        return int(normalized)
    # A typo must not unexpectedly enable processing in a browser or stop startup.
    logger.warning("FOX_IMAGE_RENDER_QUALITY must be 1, 2 or 3; image improvements disabled.")
    return 3


def image_render_quality_mode() -> int:
    """1: native windows only (default); 2: all viewers; 3: original rendering."""
    return _parse_mode(os.environ.get(IMAGE_RENDER_QUALITY_ENV))
