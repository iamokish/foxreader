"""The torch bubble model loader.

`fox_reader.bubble` is pure torch + safetensors with no detection framework.
These cover the contract the loader keeps: a missing weight file reads as a
plain sentence (never a traceback type), the reader's grayscale binarisation
is the model's own, device spellings normalise to torch form with a CPU
fallback, and cached construction returns the same instance.
"""
import importlib
import sys

import pytest
from PIL import Image


def _fresh_bubble():
    """A clean import of the module, so cache state never leaks between tests."""
    for name in ("fox_reader.bubble",):
        sys.modules.pop(name, None)
    return importlib.import_module("fox_reader.bubble")


class TestWeights:
    def test_missing_file_is_a_sentence(self, tmp_path):
        B = _fresh_bubble()
        with pytest.raises(B.BubbleUnavailable, match="missing"):
            B.BubbleModel(weights=tmp_path / "model.safetensors", device="cpu")

    def test_directory_resolves_the_filename(self, tmp_path):
        B = _fresh_bubble()
        with pytest.raises(B.BubbleUnavailable, match="missing"):
            B.BubbleModel(weights=tmp_path, device="cpu")

    def test_corrupt_file_is_a_sentence(self, tmp_path):
        bad = tmp_path / "model.safetensors"
        bad.write_bytes(b"not safetensors")
        B = _fresh_bubble()
        with pytest.raises(B.BubbleUnavailable):
            B.BubbleModel(weights=bad, device="cpu")


class TestGrayscale:
    def test_threshold_matches_the_reader(self):
        B = _fresh_bubble()
        assert B._GRAY_THRESHOLD == 160
        dark = B._to_grayscale(Image.new("L", (1, 1), 159).convert("RGB"))
        light = B._to_grayscale(Image.new("L", (1, 1), 161).convert("RGB"))
        assert dark.getpixel((0, 0)) == (0, 0, 0)
        assert light.getpixel((0, 0)) == (255, 255, 255)
        assert dark.mode == "RGB"

    def test_detect_accepts_the_flag_without_weights(self, tmp_path, monkeypatch):
        """`grayscale=True` must not crash before weights are even read."""
        B = _fresh_bubble()
        with pytest.raises(B.BubbleUnavailable):
            B.BubbleModel(weights=tmp_path / "model.safetensors", device="cpu").detect(
                Image.new("RGB", (8, 8)), grayscale=True
            )


class TestDevice:
    def test_normalises_to_torch_spelling(self):
        B = _fresh_bubble()
        assert B._normalize_device("cuda:1") == "cuda:1"
        assert B._normalize_device("cuda") == "cuda:0"
        assert B._normalize_device("gpu:2") == "cuda:2"
        assert B._normalize_device("mps") == "mps"
        assert B._normalize_device("cpu") == "cpu"
        assert B._normalize_device("nonsense") == "cpu"

    def test_unusable_gpu_falls_back_to_cpu(self):
        B = _fresh_bubble()
        # No card in this environment; a cuda request must not refuse.
        assert B._effective_device("cuda:0") == "cpu"

    def test_cached_construction(self, tmp_path):
        B = _fresh_bubble()
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"not safetensors")
        with pytest.raises(B.BubbleUnavailable):
            B.getBubbleModel(weights, device="cpu")
        with pytest.raises(B.BubbleUnavailable):
            B.getBubbleModel(weights, device="cpu")
