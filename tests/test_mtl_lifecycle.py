"""All registered local models own and release their native handles safely."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from fox_reader.constants import MTL_MODELS, mtl_model
from fox_reader.translate import local_mtl
from fox_reader.translate.machine_translation import gemma_e4b_llamacpp as gemma
from fox_reader.translate.machine_translation import qwen35_llamacpp as qwen
from fox_reader.translate.machine_translation import vntl_llama3_8b_llamacpp as vntl

_CLASSES = tuple(local_mtl.LocalMTLManager(Path(".")).available_translators)
_MODELS = [pytest.param(cls, id=cls.model_id()) for cls in _CLASSES]
_MODULES = (gemma, qwen, vntl)
_SPECIAL = {token.encode(): 200000 + index for index, token in enumerate(qwen._SPECIAL_TOKENS)}


class FakeNativeModel:
    """Expose each family's native entry points and observable ownership."""

    def __init__(self, state, kwargs):
        self.state = state
        self.kwargs = kwargs
        self.metadata = {"general.architecture": "qwen35"}
        self.filename = Path(kwargs["model_path"]).name
        self.calls = []
        self.close_count = 0
        self.closed = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_phase = None
        self.active = False
        self.closed_while_active = False

    def _enter(self, phase):
        assert not self.closed.is_set(), f"{phase} used a released native handle"
        if phase == self.block_phase:
            self.active = True
            self.entered.set()
            try:
                assert self.release.wait(5), f"test did not release {phase}"
                assert not self.closed.is_set(), f"handle released during {phase}"
            finally:
                self.active = False

    def tokenize(self, text, *, add_bos=False, special=False):
        self._enter("tokenize")
        if self.state.tokenize_error:
            raise RuntimeError(self.state.tokenize_error)
        return [_SPECIAL[text]] if special and text in _SPECIAL else list(text)

    def create_completion(self, **kwargs):
        self._enter("generate")
        self.calls.append(kwargs)
        return {"choices": [{"text": "Hello.", "finish_reason": "stop"}]}

    def create_chat_completion(self, **kwargs):
        self._enter("generate")
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "Hello."}, "finish_reason": "stop"}]}

    def __call__(self, prompt, **kwargs):
        return self.create_completion(prompt=prompt, **kwargs)

    def close(self):
        self.close_count += 1
        self.closed_while_active |= self.active
        self.state.events.append(("close", self.filename))
        self.closed.set()


@pytest.fixture
def native(tmp_path, monkeypatch):
    state = SimpleNamespace(
        root=tmp_path, instances=[], attempts=[], events=[], constructor_error=None, tokenize_error=None,
    )

    def construct(**kwargs):
        state.attempts.append(kwargs)
        if state.constructor_error:
            raise RuntimeError(state.constructor_error)
        model = FakeNativeModel(state, kwargs)
        state.instances.append(model)
        state.events.append(("load", model.filename))
        return model

    state.package = SimpleNamespace(
        Llama=construct, llama_supports_gpu_offload=lambda: True, LLAMA_SPLIT_MODE_NONE=0,
    )
    for module in _MODULES:
        monkeypatch.setattr(module, "_import_llama_cpp", lambda: state.package)
        monkeypatch.setattr(module, "thread_count", lambda **kwargs: 2)
    monkeypatch.setattr(local_mtl.FOX_DEVICE, "translator", "cpu")
    monkeypatch.setattr(local_mtl, "check_memory", lambda *args: None)
    monkeypatch.setattr(local_mtl, "torch_module", lambda: None)
    monkeypatch.setattr(local_mtl, "configure_threads", lambda preference: None)
    for entry in MTL_MODELS:
        path = tmp_path / entry["dest"] / entry["files"][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return state


@pytest.mark.parametrize("cls", _MODELS)
@pytest.mark.parametrize("device,placement", [
    ("cpu", {"n_gpu_layers": 0}),
    ("cuda:2", {"n_gpu_layers": -1, "main_gpu": 2, "split_mode": 0}),
    ("mps", {"n_gpu_layers": -1}),
])
def test_catalog_models_load_translate_unload_and_reload(native, cls, device, placement):
    entry = mtl_model(cls.model_id())
    expected_path = native.root / entry["dest"] / entry["files"][0]
    for _ in range(2):
        translator = cls(native.root, device=device)
        handle = translator.model
        try:
            assert translator.model_path == expected_path
            assert handle.kwargs["model_path"] == str(expected_path)
            assert {key: handle.kwargs[key] for key in placement} == placement
            assert handle.kwargs["n_ctx"] == 8192
            assert handle.kwargs["n_threads"] == 2
            if device != "cuda:2":
                assert "main_gpu" not in handle.kwargs
                assert "split_mode" not in handle.kwargs
            assert translator.translate("こんにちは。") == "Hello."
            assert len(handle.calls) == 1
        finally:
            translator.unload()
        translator.unload()
        assert translator.model is None
        assert handle.close_count == 1
        with pytest.raises(RuntimeError, match="unloaded"):
            translator.translate("こんにちは。")
        assert len(handle.calls) == 1
    assert len(native.instances) == 2
    assert native.instances[0] is not native.instances[1]


@pytest.mark.parametrize("cls", _MODELS)
@pytest.mark.parametrize("device", ["cuda:2", "mps"])
def test_every_model_refuses_gpu_on_cpu_only_build_without_allocating(native, cls, device):
    native.package.llama_supports_gpu_offload = lambda: False
    translator = cls.__new__(cls)
    with pytest.raises(RuntimeError, match="CPU-only"):
        translator.__init__(native.root, device=device)
    translator.unload()
    assert native.attempts == []
    assert translator.model is None


@pytest.mark.parametrize("cls", _MODELS)
@pytest.mark.parametrize("failure", ["missing_weights", "missing_package", "native_constructor"])
def test_failed_initialization_is_safe_to_unload(native, monkeypatch, cls, failure):
    root = native.root
    expected = RuntimeError
    message = "simulated"
    if failure == "missing_weights":
        root /= "missing"
        expected = FileNotFoundError
        message = "gguf"
    elif failure == "missing_package":
        def missing_package():
            raise RuntimeError("simulated missing llama-cpp-python")

        for module in _MODULES:
            monkeypatch.setattr(module, "_import_llama_cpp", missing_package)
    else:
        native.constructor_error = "simulated native allocation failure"

    translator = cls.__new__(cls)
    with pytest.raises(expected, match=message):
        translator.__init__(root, device="cpu")
    translator.unload()
    translator.unload()
    assert translator.model is None
    assert native.instances == []
    assert len(native.attempts) == (1 if failure == "native_constructor" else 0)


@pytest.mark.parametrize("cls", [
    pytest.param(vntl.VntlLlamaTranslator, id="vntl-q8"),
    pytest.param(vntl.VntlLlamaQ5Translator, id="vntl-q5"),
    pytest.param(qwen.Qwen35_9BQ6Translator, id="qwen-9b-q6"),
    pytest.param(qwen.Qwen35_9BQ8Translator, id="qwen-9b-q8"),
    pytest.param(qwen.Qwen35_4BQ8Translator, id="qwen-4b-q8"),
])
def test_tokenizer_setup_failure_releases_constructed_handle(native, cls):
    native.tokenize_error = "simulated role/control tokenization failure"
    translator = cls.__new__(cls)
    with pytest.raises(RuntimeError, match="simulated role/control"):
        translator.__init__(native.root, device="cpu")
    assert len(native.instances) == 1
    handle = native.instances[0]
    assert handle.closed.is_set()
    assert handle.close_count == 1
    assert translator.model is None
    translator.unload()
    translator.unload()
    assert handle.close_count == 1


@pytest.mark.parametrize("cls", _MODELS)
@pytest.mark.parametrize("phase", ["tokenize", "generate"])
def test_each_model_keeps_native_handle_alive_through_inference(native, cls, phase):
    translator = cls(native.root, device="cpu")
    handle = translator.model
    handle.block_phase = phase
    unloading = threading.Event()

    def unload():
        unloading.set()
        translator.unload()

    with ThreadPoolExecutor(max_workers=2) as pool:
        translation = pool.submit(translator.translate, "こんにちは。")
        closing = None
        try:
            assert handle.entered.wait(3), f"translation did not enter {phase}"
            closing = pool.submit(unload)
            assert unloading.wait(3)
            assert not handle.closed.wait(0.1), f"unload freed the handle during {phase}"
        finally:
            handle.release.set()
        assert translation.result(timeout=3) == "Hello."
        assert closing is not None
        closing.result(timeout=3)
    assert not handle.closed_while_active
    assert handle.close_count == 1
    assert translator.model is None


def test_manager_switches_all_catalog_models_without_overlapping_allocations(native):
    configured = {"japanese": _CLASSES[0].model_id()}
    settings = SimpleNamespace(get_default_model=configured.get, get_mtl_threads=lambda: "auto")
    manager = local_mtl.LocalMTLManager(native.root, settings=settings)
    assert {cls.model_id() for cls in _CLASSES} == {entry["id"] for entry in MTL_MODELS}
    expected_events = []
    try:
        for cls in _CLASSES:
            configured["japanese"] = cls.model_id()
            if native.instances:
                expected_events.append(("close", native.instances[-1].filename))
            entry = mtl_model(cls.model_id())
            expected_events.append(("load", entry["files"][0]))
            manager.load("japanese")
            current = manager.model
            assert isinstance(current, cls)
            assert native.events == expected_events
            assert sum(not handle.closed.is_set() for handle in native.instances) == 1
            manager.load("japanese")
            assert manager.model is current
            assert manager.translate("japanese", "こんにちは。") == "Hello."
            assert native.events == expected_events
        assert len(native.instances) == len(MTL_MODELS)
    finally:
        manager.unload()
    manager.unload()
    assert all(handle.close_count == 1 for handle in native.instances)
    assert manager.status()["loaded"] is False
