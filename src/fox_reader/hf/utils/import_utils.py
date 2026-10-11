"""Import-time helpers the vendored model files call.

Upstream's ``transformers.utils.import_utils`` is a large lazy-import and
optional-backend framework. Fox Reader needs exactly one decorator from it, and
needs it to do nothing: ``@requires(backends=(...))`` exists so transformers can
raise a tidy error when an extra is missing. Here torch, torchvision and cv2 are
hard dependencies, so the check can only ever pass -- and a decorator that is
guaranteed to pass is better written as a no-op than as a probe that costs an
import on every model load.

:func:`is_cv2_available` and friends stay real, because the vendored detector
image processor branches on one of them at import time.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Any, Callable


def requires(*args: Any, **kwargs: Any) -> Callable[[type], type]:
    """No-op stand-in for transformers' backend gate.

    Accepts and discards whatever upstream passes (today ``backends=(...)``), so
    a model folder copied from a newer transformers keeps importing even if the
    signature grows.
    """

    def decorator(obj):
        return obj

    return decorator


@lru_cache(maxsize=None)
def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # ValueError: a parent package exists but has no __spec__.
        return False


def is_cv2_available() -> bool:
    return _module_available("cv2")


def is_torch_available() -> bool:
    return _module_available("torch")


def is_torchvision_available() -> bool:
    return _module_available("torchvision")


def is_vision_available() -> bool:
    return _module_available("PIL")


__all__ = [
    "is_cv2_available",
    "is_torch_available",
    "is_torchvision_available",
    "is_vision_available",
    "requires",
]
