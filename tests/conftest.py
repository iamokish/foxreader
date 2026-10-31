import pytest
from PIL import Image


@pytest.fixture
def sample_image(tmp_path):
    """Create a small test image."""
    img = Image.new("RGB", (100, 100), color="white")
    path = tmp_path / "test.png"
    img.save(path)
    return path


@pytest.fixture
def fox_config():
    """Return default FoxConfig."""
    from fox_reader.config import FoxConfig
    return FoxConfig()


@pytest.fixture
def typeset_font_dir(tmp_path, monkeypatch):
    """An isolated folder with shipped faces and generated layout fixtures."""
    from fox_reader import assets, typeset
    from tests.font_fixtures import make_layout_fonts

    directory = tmp_path / "layout-fonts"
    directory.mkdir()
    for name in (*assets.BUNDLED_FONTS, assets.SYMBOL_FONT):
        raw = assets.bundled_font_bytes(name)
        assert raw, f"missing officially shipped font: {name}"
        (directory / name).write_bytes(raw)
    make_layout_fonts(directory)
    monkeypatch.setattr(typeset, "FONT_DIR", directory)
    for name in ("_FONT_CACHE", "_RAW_ADV_CACHE", "_LINE_METRICS_CACHE", "_PRIMARY_CMAP_CACHE",
                 "_SYMBOL_FONT_CACHE", "_SYMBOL_SCALE_CACHE", "_PRIMARY_PROFILE_CACHE", "_USER_SYMBOL_CACHE",
                 "_REGISTERED_FONT_BYTES", "_FORCED_SYMBOL_OWNERS", "_FORCED_SYMBOL_ADJUSTMENTS"):
        monkeypatch.setattr(typeset, name, {})
    for name in ("_SYMBOL_BYTES", "_SYMBOL_BYTES_SIG", "_SYMBOL_CMAP", "_SYMBOL_CMAP_SIG"):
        monkeypatch.setattr(typeset, name, None)
    monkeypatch.setattr(typeset, "_SYMBOL_BYTES_MISSING", False)
    monkeypatch.setattr(typeset, "_USER_SYMBOL_FONTS", ())
    monkeypatch.setattr(typeset, "_DEFAULT_TEXT_FONT", "ComicMono.ttf")
    return directory
