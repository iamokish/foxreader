"""Environment policy must preserve the browser/native opt-in boundary."""
import pytest

from fox_reader.image_rendering import image_render_quality_mode


def test_native_only_is_the_default(monkeypatch):
    monkeypatch.delenv("FOX_IMAGE_RENDER_QUALITY", raising=False)
    assert image_render_quality_mode() == 1


@pytest.mark.parametrize("raw, expected", [("1", 1), ("2", 2), ("3", 3), (" 2 ", 2)])
def test_explicit_modes(monkeypatch, raw, expected):
    monkeypatch.setenv("FOX_IMAGE_RENDER_QUALITY", raw)
    assert image_render_quality_mode() == expected


@pytest.mark.parametrize("raw", ["", "0", "4", "true", "2.0", "native"])
def test_invalid_values_disable_improvements_without_breaking_startup(monkeypatch, raw):
    monkeypatch.setenv("FOX_IMAGE_RENDER_QUALITY", raw)
    assert image_render_quality_mode() == 3
