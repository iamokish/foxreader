"""What this OpenCV build can actually do.

``cv2.xphoto`` is not a capability signal. OpenCV 5.0 ships an ``xphoto``
*module object* that imports cleanly, is truthy, and exposes nothing at all --
``dir(cv2.xphoto)`` is dunders only. So every guard of the shape::

    xphoto = getattr(cv2, "xphoto", None)
    if xphoto is None:
        ...fall back...

passes, and the call on the next line raises ``AttributeError: module
'cv2.xphoto' has no attribute 'inpaint'``. Inside :func:`.fill.inpaint_regions`
that exception is caught per cluster, so the failure never reaches the user as an
error: text clean appears to run and silently leaves the lettering in place.

The fix is to probe instead of assume, and to probe for the exact thing that is
about to be called -- ``xphoto.inpaint`` *and* the specific flag constant, since
a partial build can have one without the other. Verdicts are cached: they are
asked once per cluster on a hot path and cannot change while the process lives.

Nothing here imports from the rest of :mod:`fox_reader.clean` (``fill`` and
``hybridfill`` both need it, and ``clean/__init__`` imports ``fill``), and
nothing here does real work -- attribute lookups only -- so it is safe to call
from a request handler.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import cv2

log = logging.getLogger(__name__)

#: The ``cv2.xphoto`` inpaint flag each capability needs, by short name.
_FLAGS = {
    "shiftmap": "INPAINT_SHIFTMAP",
    "fsr": "INPAINT_FSR_BEST",
    "fsr-fast": "INPAINT_FSR_FAST",
}

#: Fill methods that are *nothing* without ``cv2.xphoto``, and what each needs.
#:
#: The hybrids are deliberately absent: they use an xphoto reconstructor for the
#: high-frequency term only, and degrade to the closest built-in with the rest of
#: the method intact (see :func:`.hybridfill._detail_image`) -- ``hybrid-fsr``
#: degraded is still pyramid shading with transported detail, which is what
#: ``hybrid-level``, the default, does on the same build. These four have no
#: remainder -- degrading ``fsr`` leaves plain Telea wearing the name ``fsr`` --
#: so they are not offered on a build that cannot run them.
#:
#: The flag each one names is the flag its own code path passes to
#: ``xphoto.inpaint``, which is why ``fsr`` needs ``INPAINT_FSR_BEST`` while the
#: hybrids' detail term asks for ``INPAINT_FSR_FAST``.
_FILL_NEEDS = {
    "transport": "shiftmap",
    "shiftmap": "shiftmap",
    "fsr": "fsr",
    "fsr-fast": "fsr-fast",
}

_warned: set[str] = set()


@lru_cache(maxsize=None)
def xphoto_flag(kind: str) -> int | None:
    """``cv2.xphoto``'s inpaint flag for ``kind``, or ``None`` if unusable.

    ``None`` means "do not call it": either the function is missing, or the flag
    is, or ``xphoto`` is one of the empty stand-in modules described above.
    """
    name = _FLAGS.get(kind)
    if name is None:
        return None
    xphoto = getattr(cv2, "xphoto", None)
    if xphoto is None or not callable(getattr(xphoto, "inpaint", None)):
        return None
    flag = getattr(xphoto, name, None)
    # bool is an int subclass and a stray `True` would index as 1.
    if isinstance(flag, bool) or not isinstance(flag, int):
        return None
    return int(flag)


def has_xphoto(kind: str) -> bool:
    """Whether ``xphoto.inpaint`` can be called with ``kind``'s flag."""
    return xphoto_flag(kind) is not None


def note_fallback(kind: str, instead: str) -> None:
    """Say once, per capability, that a reconstructor is being substituted.

    Once and not per cluster: a page can hold dozens, and a log line per cluster
    would bury the rest of the clean narration.
    """
    if kind in _warned:
        return
    _warned.add(kind)
    log.warning(
        "cv2.xphoto.inpaint (%s) is unavailable in this OpenCV build (%s); "
        "using %s instead", _FLAGS.get(kind, kind), cv2.__version__, instead)


def fill_available(method: str) -> bool:
    """Whether ``method`` can run here as the method it claims to be."""
    need = _FILL_NEEDS.get(method)
    return True if need is None else has_xphoto(need)


def usable_fills(methods: tuple[str, ...] | list[str]) -> list[str]:
    """``methods`` minus the ones this build cannot honestly perform.

    Used to decide what the entry panel offers. The full list stays valid input
    -- a project saved on a contrib build keeps rendering, degraded, rather than
    having its fill silently swapped for the default.
    """
    return [m for m in methods if fill_available(m)]
