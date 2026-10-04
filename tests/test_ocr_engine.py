"""OCR engine selection: classic vs PaddleOCR-VL GGUF.

No real weights and no llama_cpp needed: VL paths are exercised through
stubs and the missing-package fallback.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader import ocr as ocr_mod
from fox_reader.constants import DEFAULT_OCR_ENGINE


def test_normalize_ocr_engine_defaults_to_classic():
    assert ocr_mod.normalize_ocr_engine(None) == DEFAULT_OCR_ENGINE
    assert ocr_mod.normalize_ocr_engine("") == DEFAULT_OCR_ENGINE
    assert ocr_mod.normalize_ocr_engine("paddleocr-vl") == "paddleocr-vl"
    # Unknown ids never raise here: settings validation rejects, runtime falls back.
    assert ocr_mod.normalize_ocr_engine("bogus") == DEFAULT_OCR_ENGINE


def test_vl_unavailable_without_package_or_weights(monkeypatch):
    import fox_reader.ocr_vl as vl_mod

    # Forced-absent reads as unavailable with a reason, whatever the machine
    # actually has installed (CI may or may not carry the gguf extra).
    monkeypatch.setattr(vl_mod, "is_llama_cpp_available", lambda: False)
    ok, reason = ocr_mod.vl_engine_available()
    assert ok is False
    assert isinstance(reason, str) and reason

    monkeypatch.setattr(vl_mod, "is_llama_cpp_available", lambda: True)
    monkeypatch.setattr(vl_mod, "vl_model_downloaded", lambda *a, **k: False)
    ok, reason = ocr_mod.vl_engine_available()
    assert ok is False
    assert isinstance(reason, str) and reason


def _isolated_models_dir(monkeypatch, tmp_path):
    """Point ocr_vl at a scratch MODELS_DIR with a fresh migration flag."""
    import fox_reader.ocr_vl as vl_mod

    monkeypatch.setattr(vl_mod, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(vl_mod, "_migration_done", False)
    return vl_mod


def _write_legacy_weights(models_dir):
    """Populate models/paddleocr-vl/... with fake weights + marker; return old root."""
    from fox_reader.constants import (
        MTL_COMPLETION_MARKER,
        PADDLEOCR_VL_LEGACY_MODEL_DIR,
        PADDLEOCR_VL_MMPROJ_FILE,
        PADDLEOCR_VL_MODEL_FILE,
    )

    old_root = models_dir / PADDLEOCR_VL_LEGACY_MODEL_DIR
    old_root.mkdir(parents=True)
    (old_root / PADDLEOCR_VL_MODEL_FILE).write_bytes(b"fake-model")
    (old_root / PADDLEOCR_VL_MMPROJ_FILE).write_bytes(b"fake-mmproj")
    (old_root / MTL_COMPLETION_MARKER).touch()
    return old_root


def test_legacy_vl_weights_migrate_to_paddleocr_dir(tmp_path, monkeypatch):
    from fox_reader.constants import PADDLEOCR_VL_MODEL_DIR

    vl_mod = _isolated_models_dir(monkeypatch, tmp_path)
    old_root = _write_legacy_weights(tmp_path)

    model, mmproj = vl_mod.resolve_vl_weights()
    new_root = tmp_path / PADDLEOCR_VL_MODEL_DIR
    assert model == new_root / model.name
    assert mmproj == new_root / mmproj.name
    assert model.is_file() and mmproj.is_file()
    # Moved, not copied: the old folder is gone, no 1.8 GiB duplicated.
    assert not old_root.exists()
    # Second call is a no-op.
    assert vl_mod.migrate_legacy_vl_dir() is False


def test_legacy_migration_skipped_when_new_home_complete(tmp_path, monkeypatch):
    from fox_reader.constants import MTL_COMPLETION_MARKER, PADDLEOCR_VL_MODEL_DIR

    vl_mod = _isolated_models_dir(monkeypatch, tmp_path)
    old_root = _write_legacy_weights(tmp_path)
    new_root = tmp_path / PADDLEOCR_VL_MODEL_DIR
    new_root.mkdir(parents=True)
    (new_root / "other.gguf").write_bytes(b"x")
    (new_root / MTL_COMPLETION_MARKER).touch()

    assert vl_mod.migrate_legacy_vl_dir() is False
    assert old_root.is_dir()  # never delete user data unasked


def test_legacy_migration_never_raises(tmp_path, monkeypatch):
    import shutil

    import pytest

    vl_mod = _isolated_models_dir(monkeypatch, tmp_path)
    old_root = _write_legacy_weights(tmp_path)

    def _boom(src, dst):
        raise OSError("simulated locked file")

    monkeypatch.setattr(shutil, "move", _boom)
    assert vl_mod.migrate_legacy_vl_dir() is False
    assert old_root.is_dir()
    with pytest.raises(FileNotFoundError):
        vl_mod.resolve_vl_weights()


def test_explicit_model_dir_skips_migration(tmp_path, monkeypatch):
    import pytest

    vl_mod = _isolated_models_dir(monkeypatch, tmp_path)
    old_root = _write_legacy_weights(tmp_path)
    custom = tmp_path / "elsewhere"
    custom.mkdir()

    # A caller pointing elsewhere uses it verbatim: no move behind their back.
    with pytest.raises(FileNotFoundError):
        vl_mod.resolve_vl_weights(custom)
    assert old_root.is_dir()


def _write_new_weights(models_dir):
    """Populate models/paddleocr/... with fake GGUFs (no template); return root."""
    from fox_reader.constants import (
        PADDLEOCR_VL_MMPROJ_FILE,
        PADDLEOCR_VL_MODEL_DIR,
        PADDLEOCR_VL_MODEL_FILE,
    )

    new_root = models_dir / PADDLEOCR_VL_MODEL_DIR
    new_root.mkdir(parents=True)
    (new_root / PADDLEOCR_VL_MODEL_FILE).write_bytes(b"fake-model")
    (new_root / PADDLEOCR_VL_MMPROJ_FILE).write_bytes(b"fake-mmproj")
    return new_root


def _fresh_template_attempts(monkeypatch):
    import fox_reader.ocr_vl as vl_mod

    monkeypatch.setattr(vl_mod, "_template_attempted", set())


def test_chat_template_listed_with_the_bundle():
    from fox_reader.constants import (
        HF_BASE_MODELS,
        PADDLEOCR_VL_CHAT_TEMPLATE_FILE,
        PADDLEOCR_VL_MMPROJ_FILE,
        PADDLEOCR_VL_MODEL_FILE,
    )

    entry = next(m for m in HF_BASE_MODELS["paddleocr_vl"])
    assert entry["files"] == [PADDLEOCR_VL_MODEL_FILE, PADDLEOCR_VL_MMPROJ_FILE, PADDLEOCR_VL_CHAT_TEMPLATE_FILE]


def test_ensure_chat_template_present_needs_no_network(tmp_path, monkeypatch):
    import huggingface_hub

    from fox_reader.constants import PADDLEOCR_VL_CHAT_TEMPLATE_FILE

    vl_mod = _isolated_models_dir(monkeypatch, tmp_path)
    _fresh_template_attempts(monkeypatch)
    new_root = _write_new_weights(tmp_path)
    target = new_root / PADDLEOCR_VL_CHAT_TEMPLATE_FILE
    target.write_text("already here", encoding="utf-8")

    def _must_not_run(*args, **kwargs):
        raise AssertionError("no download when the file is present")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _must_not_run)
    assert vl_mod.ensure_chat_template() == target


def test_ensure_chat_template_fetched_when_missing(tmp_path, monkeypatch):
    import huggingface_hub

    from fox_reader.constants import PADDLEOCR_VL_CHAT_TEMPLATE_FILE

    vl_mod = _isolated_models_dir(monkeypatch, tmp_path)
    _fresh_template_attempts(monkeypatch)
    new_root = _write_new_weights(tmp_path)
    calls = []

    def _fake_download(*, repo_id, filename, local_dir, **kw):
        from pathlib import Path

        calls.append((repo_id, filename))
        assert filename == PADDLEOCR_VL_CHAT_TEMPLATE_FILE
        (Path(local_dir) / filename).write_text("{% chat %}", encoding="utf-8")
        return str(Path(local_dir) / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    target = vl_mod.ensure_chat_template()
    assert target == new_root / PADDLEOCR_VL_CHAT_TEMPLATE_FILE
    assert target.is_file()
    # Second call is a pure is_file check, not another download.
    assert vl_mod.ensure_chat_template() == target
    assert len(calls) == 1


def test_ensure_chat_template_failure_is_silent(tmp_path, monkeypatch):
    import huggingface_hub

    vl_mod = _isolated_models_dir(monkeypatch, tmp_path)
    _fresh_template_attempts(monkeypatch)
    _write_new_weights(tmp_path)

    def _boom(*args, **kwargs):
        raise ConnectionError("simulated offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    assert vl_mod.ensure_chat_template() is None
    # Failed once: no retry storm on the next OCR request either.
    assert vl_mod.ensure_chat_template() is None


def test_resolve_tops_up_template(tmp_path, monkeypatch):
    import huggingface_hub


    vl_mod = _isolated_models_dir(monkeypatch, tmp_path)
    _fresh_template_attempts(monkeypatch)
    _write_new_weights(tmp_path)

    def _fake_download(*, repo_id, filename, local_dir, **kw):
        from pathlib import Path

        (Path(local_dir) / filename).write_text("{% chat %}", encoding="utf-8")
        return str(Path(local_dir) / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    vl_mod.resolve_vl_weights()
    assert vl_mod.chat_template_path().is_file()


def test_suppressed_output_restores_streams(monkeypatch):
    import sys

    import fox_reader.ocr_vl as vl_mod

    monkeypatch.delenv(vl_mod.VERBOSE_ENV, raising=False)
    before_out, before_err = sys.stdout, sys.stderr
    with vl_mod._suppressed_native_output():
        # Streams are pointed away for the duration (devnull fds, or a
        # StringIO fallback when the native helper is unavailable).
        assert sys.stdout is not before_out
    assert sys.stdout is before_out
    assert sys.stderr is before_err


def test_suppressed_output_restores_on_exception(monkeypatch):
    import sys

    import pytest

    import fox_reader.ocr_vl as vl_mod

    monkeypatch.delenv(vl_mod.VERBOSE_ENV, raising=False)
    before_out, before_err = sys.stdout, sys.stderr
    with pytest.raises(RuntimeError):
        with vl_mod._suppressed_native_output():
            raise RuntimeError("boom inside inference")
    assert sys.stdout is before_out
    assert sys.stderr is before_err


def test_verbose_mode_passes_output_through(monkeypatch):
    import sys

    import fox_reader.ocr_vl as vl_mod

    monkeypatch.setenv(vl_mod.VERBOSE_ENV, "1")
    before_out = sys.stdout
    with vl_mod._suppressed_native_output():
        assert sys.stdout is before_out


def test_suppression_degrades_without_native_helper(monkeypatch):
    import sys

    import fox_reader.ocr_vl as vl_mod

    monkeypatch.delenv(vl_mod.VERBOSE_ENV, raising=False)
    monkeypatch.setattr(vl_mod, "_native_suppressor", lambda: None)
    before_out, before_err = sys.stdout, sys.stderr
    with vl_mod._suppressed_native_output():
        print("captured, never shown")
    assert sys.stdout is before_out
    assert sys.stderr is before_err


def test_predict_runs_quietly(monkeypatch, capsys):
    from pathlib import Path

    import fox_reader.ocr_vl as vl_mod

    monkeypatch.delenv(vl_mod.VERBOSE_ENV, raising=False)
    monkeypatch.setattr(
        vl_mod, "resolve_vl_weights",
        lambda root=None: (Path("/tmp/m.gguf"), Path("/tmp/p.gguf")),
    )
    eng = vl_mod.PaddleOCRVLEngine.__new__(vl_mod.PaddleOCRVLEngine)
    eng.root = Path("/tmp")
    eng.device = "cpu"
    eng.model_path = Path("/tmp/m.gguf")
    eng.mmproj_path = Path("/tmp/p.gguf")
    import threading

    eng._lock = threading.RLock()

    class NoisyModel:
        def create_chat_completion(self, **kw):
            print("add_text: User: OCR:")
            import sys as _sys

            _sys.stderr.write("clip_encode: copying image\n")
            return {"choices": [{"message": {"content": "quiet あ"}}]}

    eng.model = NoisyModel()
    monkeypatch.setattr(eng, "ensure_loaded", lambda: None)
    from PIL import Image

    assert eng.predict(input=Image.new("RGB", (16, 16))) == "quiet あ"
    out, err = capsys.readouterr()
    assert "add_text" not in out and "clip_encode" not in err


def test_describe_ocr_engines_never_raises():
    payload = ocr_mod.describe_ocr_engines("paddleocr-vl")
    assert payload["selected"] == "paddleocr-vl"
    assert payload["default"] == DEFAULT_OCR_ENGINE
    assert len(payload["engines"]) == 2
    classic = next(e for e in payload["engines"] if e["id"] == "paddleocr")
    assert classic["available"] is True


def test_settings_manager_ocr_engine_roundtrip(tmp_path):
    from fox_reader.settings import SettingsManager

    mgr = SettingsManager(tmp_path)
    mgr.load()
    assert mgr.get_ocr_engine() == DEFAULT_OCR_ENGINE
    mgr.update_ocr_engine("paddleocr-vl")
    assert mgr.get_ocr_engine() == "paddleocr-vl"
    try:
        mgr.update_ocr_engine("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("bogus engine must raise")
    # Failed update leaves the saved value alone.
    assert mgr.get_ocr_engine() == "paddleocr-vl"


def _stub_classic(monkeypatch):
    import fox_reader.services.ocr_service as svc_mod

    class FakeClassic:
        last_kwargs: dict = {}

        def __init__(self, *args, **kwargs):
            FakeClassic.last_kwargs = dict(kwargs)
            self._engines = {"default": type("D", (), {"detector": object()})()}
            self.ensure_calls = 0

        def ensure_recognizers(self):
            self.ensure_calls += 1

        def predict(self, lang, input, isGrayScaled, **kw):
            return "classic"

    monkeypatch.setattr(svc_mod, "MultiLangOCR", FakeClassic)
    return FakeClassic


def _stub_vl(monkeypatch, fail=False):
    """Stub the VL engine class: no gigabytes, no llama.cpp, fully observed."""
    import fox_reader.services.ocr_service as svc_mod

    class FakeVL:
        instances: list = []

        def __init__(self, *args, **kwargs):
            FakeVL.instances.append(self)
            self.load_calls = 0
            self.unloaded = False

        def ensure_loaded(self):
            self.load_calls += 1
            if fail:
                raise RuntimeError("simulated broken VL")

        def predict(self, input, isGrayScaled=False, lang="japanese", **kw):
            return "vl-text"

        def unload(self):
            self.unloaded = True

    FakeVL.instances.clear()
    monkeypatch.setattr(svc_mod, "PaddleOCRVLEngine", FakeVL)
    return FakeVL


def test_vl_loads_eagerly_at_startup(monkeypatch):
    fake_classic = _stub_classic(monkeypatch)
    fake_vl = _stub_vl(monkeypatch)
    from PIL import Image

    from fox_reader.services.ocr_service import OCRService

    svc = OCRService(engine="paddleocr-vl")
    # Detector-only classic in VL mode: no recognizers at construction.
    assert fake_classic.last_kwargs == {"recognizers": False}
    # VL loaded during construction, not on first predict.
    assert len(fake_vl.instances) == 1
    assert svc._vl is fake_vl.instances[0]
    assert svc._vl.load_calls == 1
    assert svc.active_engine == "paddleocr-vl"
    # Detector always comes from classic, even in VL mode (for ppocr clean).
    assert svc.detector is not None
    out = svc.predict(lang="japanese", image=Image.new("RGB", (16, 16)), grayscale=False)
    assert out == "vl-text"
    assert svc.active_engine == "paddleocr-vl"
    # No lazy load on the request path, and classic recognizers never touched.
    assert svc._vl.load_calls == 1
    assert svc._engine.ensure_calls == 0


def test_ocr_service_falls_back_to_classic(monkeypatch):
    fake_classic = _stub_classic(monkeypatch)
    _stub_vl(monkeypatch, fail=True)
    from PIL import Image

    from fox_reader.services.ocr_service import OCRService

    # A broken VL must not fail startup: construction falls back to classic
    # (loading its recognizers eagerly, since classic is now the live path).
    svc = OCRService(engine="paddleocr-vl")
    assert fake_classic.last_kwargs == {"recognizers": False}
    assert svc._vl is None
    assert svc._vl_error == "simulated broken VL"
    assert svc.active_engine == "paddleocr"
    assert svc._engine.ensure_calls == 1
    # Detector always comes from classic, even in VL mode (for ppocr clean).
    assert svc.detector is not None
    out = svc.predict(lang="japanese", image=Image.new("RGB", (16, 16)), grayscale=False)
    assert out == "classic"
    assert svc.active_engine == "paddleocr"
    # Recognizers were already loaded at fallback time: exactly one more
    # ensure from the predict path, no repeated VL load attempts.
    assert svc._engine.ensure_calls == 2
    assert svc.engine_info()["last_error"] == "simulated broken VL"


def test_ocr_service_loads_recognizers_for_classic_mode(monkeypatch):
    fake_classic = _stub_classic(monkeypatch)
    from PIL import Image

    from fox_reader.services.ocr_service import OCRService

    svc = OCRService(engine="paddleocr")
    assert fake_classic.last_kwargs == {"recognizers": True}
    out = svc.predict(lang="japanese", image=Image.new("RGB", (16, 16)), grayscale=False)
    assert out == "classic"
    assert svc._engine.ensure_calls == 0


def _stub_manager(monkeypatch):
    """Stub the model manager: count builds, serve tiny fakes, load nothing."""
    import fox_reader.ocr as ocr_mod

    calls = {"detector": 0, "recognizer": 0}

    class FakeDetector:
        def predict(self, img_np):
            return [{"dt_polys": [[[0, 0], [10, 0], [10, 10], [0, 10]]]}]

    class FakeRecognizer:
        def predict(self, crop_np):
            return [{"rec_text": "あ", "rec_score": 0.9}]

    @classmethod
    def fake_detector(cls, model_name, device):
        calls["detector"] += 1
        return FakeDetector()

    @classmethod
    def fake_recognizer(cls, model_name, device, config_model_name=None):
        calls["recognizer"] += 1
        return FakeRecognizer()

    monkeypatch.setattr(ocr_mod.PaddleOCRModelManager, "get_detector", fake_detector)
    monkeypatch.setattr(ocr_mod.PaddleOCRModelManager, "get_recognizer", fake_recognizer)
    return calls


def test_detector_only_mode_skips_recognizers(monkeypatch):
    from fox_reader.ocr import MultiLangOCR

    calls = _stub_manager(monkeypatch)
    classic = MultiLangOCR(recognizers=False)
    assert calls["detector"] == 2  # main + korean share one det model, built per wrapper
    assert calls["recognizer"] == 0
    # The detector clean borrows is alive from the start.
    assert classic._engines["default"].detector is not None
    assert classic._engines["default"].recognizer is None
    assert classic._engines["korean"].recognizer is None


def test_recognizers_load_once_on_first_use(monkeypatch):
    from PIL import Image

    from fox_reader.ocr import MultiLangOCR

    calls = _stub_manager(monkeypatch)
    classic = MultiLangOCR(recognizers=False)
    classic.ensure_recognizers()
    # Two unique wrappers (main is shared across four language keys).
    assert calls["recognizer"] == 2
    classic.ensure_recognizers()
    assert calls["recognizer"] == 2
    # End to end through the lazy path, on a 20x20 crop with one box.
    text = classic.predict(lang="japanese", input=Image.new("RGB", (20, 20)), isGrayScaled=False)
    assert text == "あ"
    assert calls["recognizer"] == 2


def test_full_mode_unchanged(monkeypatch):
    from fox_reader.ocr import MultiLangOCR

    calls = _stub_manager(monkeypatch)
    MultiLangOCR()
    assert calls["detector"] == 2
    assert calls["recognizer"] == 2


def test_vl_build_prefers_chat_handler_over_kwargs():
    """Regression: 𠮷𠮷 repetition came from Llama(mmproj_path=...) being
    swallowed by **kwargs on 0.3.35, loading text-only. The handler must win
    and text-only must never be built silently."""
    import sys
    import types

    from fox_reader import ocr_vl as vl_mod

    calls: dict = {}

    class FakeHandler:
        def __init__(self, **kw):
            calls["handler_kw"] = kw

    fake_fmt = types.ModuleType("llama_chat_format")
    fake_fmt.MTMDChatHandler = FakeHandler
    class FakeLlama:
        def __init__(self, model_path, chat_handler=None, **kw):
            calls["chat_handler"] = chat_handler
            calls["kwargs"] = kw
            if chat_handler is None:
                raise AssertionError("text-only Llama must never be built")

    import unittest.mock as mock
    from pathlib import Path

    eng = vl_mod.PaddleOCRVLEngine.__new__(vl_mod.PaddleOCRVLEngine)
    eng.model_path = Path("/tmp/model.gguf")
    eng.mmproj_path = Path("/tmp/mmproj.gguf")
    eng.device = "cpu"
    # _vision_handler does `from llama_cpp import llama_chat_format`, so both
    # sys.modules entries must point at the fakes.
    fake_parent = types.ModuleType("llama_cpp")
    fake_parent.Llama = FakeLlama  # type: ignore[attr-defined]
    fake_parent.llama_chat_format = fake_fmt  # type: ignore[attr-defined]
    with mock.patch.dict(
        sys.modules, {"llama_cpp": fake_parent, "llama_cpp.llama_chat_format": fake_fmt}
    ):
        eng._build_llama(fake_parent)
    assert isinstance(calls.get("chat_handler"), FakeHandler)
    assert "mmproj" in str(calls.get("handler_kw", {}))


def test_ocr_settings_routes(tmp_path):
    _stub_settings_routes = None
    from fox_reader.routes import settings as settings_routes
    from fox_reader.settings import SettingsManager

    mgr = SettingsManager(tmp_path)
    mgr.load()
    app = FastAPI()
    app.include_router(settings_routes.router)
    app.state.settings = mgr
    app.state.mtl_dir = tmp_path
    app.state.ocr = None
    client = TestClient(app)

    assert client.get("/api/settings").status_code == 200
    assert client.get("/api/settings").json()["ocr"]["selected"] == "paddleocr"

    resp = client.put("/api/settings/ocr-engine", json={"engine": "paddleocr-vl"})
    assert resp.status_code == 200
    assert resp.json()["ocr"]["selected"] == "paddleocr-vl"

    assert client.put("/api/settings/ocr-engine", json={"engine": "bogus"}).status_code == 400
    assert client.put("/api/settings/ocr-engine", json={}).status_code == 400
