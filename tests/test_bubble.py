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


class TestPadding:
    def test_normalize_keeps_valid_values(self):
        B = _fresh_bubble()
        assert B.normalize_padding(0) == 0
        assert B.normalize_padding(5) == 5
        assert B.normalize_padding(10) == 10

    def test_normalize_clamps_instead_of_refusing(self):
        B = _fresh_bubble()
        assert B.normalize_padding(-3) == 0
        assert B.normalize_padding(15) == 10

    def test_normalize_falls_back_on_garbage(self):
        B = _fresh_bubble()
        assert B.normalize_padding(None) == B._DEFAULT_PADDING
        assert B.normalize_padding("abc") == B._DEFAULT_PADDING
        assert B.normalize_padding("") == B._DEFAULT_PADDING

    def test_normalize_accepts_numeric_strings(self):
        B = _fresh_bubble()
        assert B.normalize_padding("7") == 7

    def test_detect_defaults_to_the_historical_constant(self):
        """Omitting padding must behave exactly like the old hardcoded 5."""
        import inspect

        B = _fresh_bubble()
        sig = inspect.signature(B.BubbleModel.detect)
        assert sig.parameters["padding"].default == 5
        assert sig.parameters["padding"].default == B._DEFAULT_PADDING


class TestBubbleRequest:
    def test_padding_defaults_to_five(self):
        from fox_reader.models.requests import BubbleRequest

        req = BubbleRequest(filename="p.png", isGrayScale=False)
        assert req.padding == 5

    def test_padding_round_trips(self):
        from fox_reader.models.requests import BubbleRequest

        req = BubbleRequest(filename="p.png", isGrayScale=True, padding=8)
        assert req.padding == 8


class TestBubbleServiceRoute:
    """The app serves a BubbleService, not the raw model.

    The route passes the slider padding straight through it, so the service
    must accept and forward it -- a signature mismatch here is a 500 on every
    bubble capture, exactly as shipped once. No weights needed: the model
    behind the service is stubbed, the route and the forwarding are real.
    """

    def _service_with_stub(self, monkeypatch):
        B = _fresh_bubble()
        seen = {}

        class StubModel:
            def detect(self, image, **kwargs):
                seen.update(kwargs)
                return []

        monkeypatch.setattr(B, "getBubbleModel", lambda *args, **kwargs: StubModel())
        from fox_reader.services.bubble_service import BubbleService

        return BubbleService(), seen

    def _request(self, tmp_path, service, padding):
        from types import SimpleNamespace

        from PIL import Image

        from fox_reader.models.requests import BubbleRequest
        from fox_reader.services.session_service import SessionService

        Image.new("RGB", (32, 32)).save(tmp_path / "p.png")
        session = SessionService()
        session.set_workspace(str(tmp_path), str(tmp_path))
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(session=session, bubble=service))
        )
        return request, BubbleRequest(filename="p.png", isGrayScale=False, padding=padding)

    def test_service_forwards_padding_to_the_model(self, tmp_path, monkeypatch):
        service, seen = self._service_with_stub(monkeypatch)
        service.detect(Image.new("RGB", (8, 8)), False, padding=8)
        assert seen == {"grayscale": False, "padding": 8}

    def test_service_keeps_the_historical_default(self, tmp_path, monkeypatch):
        service, seen = self._service_with_stub(monkeypatch)
        service.detect(Image.new("RGB", (8, 8)), False)
        assert seen == {"grayscale": False}

    def test_route_serves_padded_detection(self, tmp_path, monkeypatch):
        import asyncio
        import json

        from fox_reader.routes import bubble as bubble_route

        service, seen = self._service_with_stub(monkeypatch)
        request, req = self._request(tmp_path, service, padding=8)
        response = asyncio.run(bubble_route.detect_bubbles(request, req))
        assert response.status_code == 200
        assert json.loads(response.body.decode()) == []
        assert seen == {"grayscale": False, "padding": 8}


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
