import pytest
from pathlib import Path
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
