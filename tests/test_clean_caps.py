"""Tests for the OpenCV capability probes behind text clean.

The bug these pin: OpenCV 5.0 ships a ``cv2.xphoto`` module that imports fine,
is truthy, and exposes nothing -- so ``getattr(cv2, "xphoto", None) is None``
guards pass and the very next call raises ``AttributeError: module 'cv2.xphoto'
has no attribute 'inpaint'``. Every probe here therefore asks about the exact
attribute that is about to be called, not about the module.
"""
import types
from functools import lru_cache

import cv2

from fox_reader.clean import caps
from fox_reader.clean.fill import DEFAULT_INPAINT_METHOD, INPAINT_METHODS


def fake_xphoto(*, inpaint=True,
                flags=("INPAINT_SHIFTMAP", "INPAINT_FSR_BEST", "INPAINT_FSR_FAST"),
                flag_value=1):
    """A stand-in ``cv2.xphoto``: `inpaint` and `flags` say what it carries."""
    mod = types.SimpleNamespace()
    if inpaint:
        mod.inpaint = lambda *a, **kw: None
    for name in flags:
        setattr(mod, name, flag_value)
    return mod


def install(monkeypatch, xphoto):
    """Point ``caps`` at a build whose ``cv2.xphoto`` is ``xphoto``.

    ``None`` stands for a build with no ``xphoto`` at all: :func:`caps.xphoto_flag`
    reaches it through ``getattr(cv2, "xphoto", None)``, so an attribute holding
    ``None`` and a missing attribute are the same code path.

    The probe is ``lru_cache``d for the life of the process -- it is asked once
    per cluster on a hot path -- and ``monkeypatch``'s undo cannot reach inside a
    cache, so each test gets a fresh one wrapped around the same function.
    """
    monkeypatch.setattr(cv2, "xphoto", xphoto, raising=False)
    monkeypatch.setattr(
        caps, "xphoto_flag", lru_cache(maxsize=None)(caps.xphoto_flag.__wrapped__),
    )


class TestXphotoFlag:
    def test_unknown_capability_is_never_usable(self, monkeypatch):
        install(monkeypatch, fake_xphoto())
        assert caps.xphoto_flag("bogus") is None

    def test_a_complete_build_reports_the_flag(self, monkeypatch):
        install(monkeypatch, fake_xphoto(flag_value=7))
        assert caps.xphoto_flag("shiftmap") == 7
        assert caps.has_xphoto("fsr") is True

    def test_the_empty_opencv_5_module_is_not_a_capability(self, monkeypatch):
        # The reported failure: present, truthy, and holds nothing.
        install(monkeypatch, types.SimpleNamespace())
        assert caps.xphoto_flag("shiftmap") is None
        assert caps.has_xphoto("shiftmap") is False

    def test_a_missing_module_is_handled(self, monkeypatch):
        install(monkeypatch, None)
        assert caps.xphoto_flag("fsr") is None

    def test_inpaint_without_its_flag_is_not_usable(self, monkeypatch):
        # A partial build can have the function and not the constant.
        install(monkeypatch, fake_xphoto(flags=("INPAINT_SHIFTMAP",)))
        assert caps.xphoto_flag("shiftmap") is not None
        assert caps.xphoto_flag("fsr") is None

    def test_a_flag_that_is_not_an_int_is_rejected(self, monkeypatch):
        install(monkeypatch, fake_xphoto(flag_value=None))
        assert caps.xphoto_flag("shiftmap") is None

    def test_a_boolean_flag_is_rejected(self, monkeypatch):
        # `bool` is an `int` subclass, so a stray True would pass as flag 1.
        install(monkeypatch, fake_xphoto(flag_value=True))
        assert caps.xphoto_flag("shiftmap") is None

    def test_a_non_callable_inpaint_is_rejected(self, monkeypatch):
        xphoto = fake_xphoto()
        xphoto.inpaint = "not a function"
        install(monkeypatch, xphoto)
        assert caps.xphoto_flag("shiftmap") is None


class TestFillAvailability:
    def test_builtin_fills_never_depend_on_xphoto(self, monkeypatch):
        install(monkeypatch, types.SimpleNamespace())
        for method in ("telea", "ns", "pyramid", "level", "hybrid",
                       "hybrid-level", "hybrid-fsr", "hybrid-patch", "patch"):
            assert caps.fill_available(method) is True

    def test_xphoto_only_fills_are_dropped_on_an_empty_build(self, monkeypatch):
        install(monkeypatch, types.SimpleNamespace())
        # Order is preserved: the panel offers them best-first.
        assert caps.usable_fills(INPAINT_METHODS) == [
            "hybrid-level", "level", "hybrid-fsr", "hybrid-patch", "hybrid",
            "patch", "pyramid", "telea", "ns",
        ]

    def test_the_default_fill_survives_an_empty_build(self, monkeypatch):
        # Otherwise the panel opens on a method it refuses to list.
        install(monkeypatch, types.SimpleNamespace())
        assert caps.fill_available(DEFAULT_INPAINT_METHOD) is True

    def test_everything_is_offered_on_a_complete_build(self, monkeypatch):
        install(monkeypatch, fake_xphoto())
        assert caps.usable_fills(INPAINT_METHODS) == list(INPAINT_METHODS)

    def test_an_unknown_method_is_left_alone(self, monkeypatch):
        # `usable_fills` filters; it is not a whitelist of known names.
        install(monkeypatch, types.SimpleNamespace())
        assert caps.usable_fills(["telea", "made-up"]) == ["telea", "made-up"]


class TestNoteFallback:
    def test_warns_once_per_capability(self, monkeypatch):
        lines = []
        monkeypatch.setattr(caps, "_warned", set(), raising=False)
        monkeypatch.setattr(
            caps, "log",
            types.SimpleNamespace(warning=lambda msg, *a: lines.append(msg % a)),
            raising=False,
        )

        caps.note_fallback("shiftmap", "telea")
        caps.note_fallback("shiftmap", "telea")
        caps.note_fallback("fsr", "ns")

        # A page can hold dozens of clusters; one line each, not one per cluster.
        assert len(lines) == 2
        assert "INPAINT_SHIFTMAP" in lines[0]
        assert "telea" in lines[0]
        assert "INPAINT_FSR_BEST" in lines[1]
