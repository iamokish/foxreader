"""Exercise both Gemma quantizations through their shared native lifecycle."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from fox_reader.constants import (
    GEMMA_E4B_Q6_MODEL_DIR,
    GEMMA_E4B_Q6_MODEL_FILE,
    GEMMA_E4B_Q8_MODEL_DIR,
    GEMMA_E4B_Q8_MODEL_FILE,
)
from fox_reader.translate.machine_translation import gemma_e4b_llamacpp as gemma

_VARIANTS = (
    pytest.param(gemma.GemmaE4BQ8Translator, GEMMA_E4B_Q8_MODEL_DIR, GEMMA_E4B_Q8_MODEL_FILE, id="q8"),
    pytest.param(gemma.GemmaE4BQ6Translator, GEMMA_E4B_Q6_MODEL_DIR, GEMMA_E4B_Q6_MODEL_FILE, id="q6"),
)


class FakeLlama:
    def __init__(self, events, **kwargs):
        self.kwargs = kwargs
        self.events = events
        self.calls = []
        self.close_count = 0
        self.block_phase = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.active = False
        self.closed_while_active = False
        self.response = {"choices": [{"message": {"content": "```text\nHello.\n```"}}]}

    def _native_call(self, phase):
        assert not self.closed.is_set(), f"{phase} used a closed native handle"
        if phase != self.block_phase:
            return
        self.active = True
        self.entered.set()
        try:
            assert self.release.wait(5), f"test did not release {phase}"
            assert not self.closed.is_set(), f"native handle closed during {phase}"
        finally:
            self.active = False

    def tokenize(self, text, *, add_bos=False, special=False):
        self._native_call("tokenize")
        return list(text)

    def create_chat_completion(self, **kwargs):
        self._native_call("generate")
        self.calls.append(kwargs)
        return self.response

    def close(self):
        self.close_count += 1
        self.closed_while_active |= self.active
        self.events.append(("close", Path(self.kwargs["model_path"]).name))
        self.closed.set()


@pytest.fixture
def native_loader(tmp_path, monkeypatch):
    events = []
    instances = []

    def construct(**kwargs):
        events.append(("load", Path(kwargs["model_path"]).name))
        model = FakeLlama(events, **kwargs)
        instances.append(model)
        return model

    package = SimpleNamespace(
        Llama=construct,
        llama_supports_gpu_offload=lambda: True,
        LLAMA_SPLIT_MODE_NONE=0,
    )
    monkeypatch.setattr(gemma, "_import_llama_cpp", lambda: package)
    monkeypatch.setattr(gemma, "thread_count", lambda **kwargs: 2)
    monkeypatch.setattr(gemma.FOX_DEVICE, "translator", "cpu")
    for directory, filename in (
        (GEMMA_E4B_Q8_MODEL_DIR, GEMMA_E4B_Q8_MODEL_FILE),
        (GEMMA_E4B_Q6_MODEL_DIR, GEMMA_E4B_Q6_MODEL_FILE),
    ):
        path = tmp_path / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        # A stray GGUF must not displace the declared quantization.
        (path.parent / "aaa-other.gguf").touch()
    return SimpleNamespace(root=tmp_path, package=package, instances=instances, events=events)


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
@pytest.mark.parametrize("device,placement", [
    ("cpu", {"n_gpu_layers": 0}),
    ("cuda", {"n_gpu_layers": -1}),
    ("cuda:0", {"n_gpu_layers": -1}),
    ("cuda:2", {"n_gpu_layers": -1, "main_gpu": 2, "split_mode": 0}),
    ("mps", {"n_gpu_layers": -1}),
])
def test_variants_load_their_own_quant_on_requested_device(native_loader, cls, directory, filename, device, placement):
    translator = cls(native_loader.root, device=device)
    native = native_loader.instances[-1]
    expected = native_loader.root / directory / filename
    assert translator.model_path == expected
    assert native.kwargs["model_path"] == str(expected)
    assert native.kwargs["n_ctx"] == gemma.N_CTX
    assert native.kwargs["n_threads"] == 2
    assert {key: native.kwargs[key] for key in placement} == placement
    if "main_gpu" not in placement:
        assert "main_gpu" not in native.kwargs
        assert "split_mode" not in native.kwargs
    translator.unload()
    assert native.close_count == 1


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
@pytest.mark.parametrize("renamed", [Path("renamed.gguf"), Path("nested") / "renamed.gguf"])
def test_renamed_weight_fallback_stays_inside_selected_quant(tmp_path, cls, directory, filename, renamed):
    selected = tmp_path / directory / renamed
    selected.parent.mkdir(parents=True)
    selected.touch()
    other_directory, other_filename = (
        (GEMMA_E4B_Q6_MODEL_DIR, GEMMA_E4B_Q6_MODEL_FILE)
        if filename == GEMMA_E4B_Q8_MODEL_FILE
        else (GEMMA_E4B_Q8_MODEL_DIR, GEMMA_E4B_Q8_MODEL_FILE)
    )
    other = tmp_path / other_directory / other_filename
    other.parent.mkdir(parents=True)
    other.touch()
    assert gemma.resolve_weights(tmp_path, model_dir=directory, model_file=filename) == selected


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
def test_other_quant_weights_do_not_satisfy_missing_selected_quant(native_loader, cls, directory, filename):
    root = native_loader.root / "isolated-install"
    other_directory, other_filename = (
        (GEMMA_E4B_Q6_MODEL_DIR, GEMMA_E4B_Q6_MODEL_FILE)
        if filename == GEMMA_E4B_Q8_MODEL_FILE
        else (GEMMA_E4B_Q8_MODEL_DIR, GEMMA_E4B_Q8_MODEL_FILE)
    )
    other = root / other_directory / other_filename
    other.parent.mkdir(parents=True)
    other.touch()
    with pytest.raises(FileNotFoundError):
        gemma.resolve_weights(root, model_dir=directory, model_file=filename)
    with pytest.raises(FileNotFoundError):
        cls(root, device="cpu")
    assert native_loader.events == []


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
@pytest.mark.parametrize("device", ["cuda", "cuda:2", "mps"])
def test_cpu_only_build_rejects_gpu_before_native_allocation(native_loader, cls, directory, filename, device):
    native_loader.package.llama_supports_gpu_offload = lambda: False
    with pytest.raises(RuntimeError, match="CPU-only"):
        cls(native_loader.root, device=device)
    assert native_loader.instances == []
    assert native_loader.events == []


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
def test_default_device_is_read_at_construction(native_loader, monkeypatch, cls, directory, filename):
    monkeypatch.setattr(gemma.FOX_DEVICE, "translator", "mps")
    translator = cls(native_loader.root)
    assert translator.device == "mps"
    assert native_loader.instances[-1].kwargs["n_gpu_layers"] == -1
    translator.unload()


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_failed_native_constructor_has_no_retry_and_can_be_unloaded(native_loader, cls, directory, filename, device):
    attempts = []

    def fail(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("native allocation failed")

    native_loader.package.Llama = fail
    translator = cls.__new__(cls)
    with pytest.raises(RuntimeError, match="native allocation failed"):
        translator.__init__(native_loader.root, device=device)
    translator.unload()
    translator.unload()
    assert translator.model is None
    assert len(attempts) == 1
    assert attempts[0]["n_gpu_layers"] == (0 if device == "cpu" else -1)
    assert native_loader.instances == []


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
def test_missing_weights_leave_partial_instance_safe_to_unload(native_loader, cls, directory, filename):
    translator = cls.__new__(cls)
    with pytest.raises(FileNotFoundError):
        translator.__init__(native_loader.root / "missing-install", device="cpu")
    translator.unload()
    translator.unload()
    assert translator.model is None
    assert native_loader.events == []


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
def test_missing_runtime_package_leaves_no_handle(native_loader, monkeypatch, cls, directory, filename):
    def fail():
        raise RuntimeError("llama-cpp-python is not installed")

    monkeypatch.setattr(gemma, "_import_llama_cpp", fail)
    translator = cls.__new__(cls)
    with pytest.raises(RuntimeError, match="not installed"):
        translator.__init__(native_loader.root, device="cpu")
    translator.unload()
    assert translator.model is None
    assert native_loader.events == []


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
def test_translate_and_repeated_unload(native_loader, cls, directory, filename):
    translator = cls(native_loader.root, device="cpu")
    native = translator.model
    assert translator.translate("こんにちは。") == "Hello."
    assert native.calls[-1]["max_tokens"] == gemma.MAX_NEW_TOKENS
    assert native.calls[-1]["messages"][-1] == {"role": "user", "content": "こんにちは。"}
    translator.unload()
    translator.unload()
    assert native.close_count == 1
    assert translator.model is None
    with pytest.raises(RuntimeError, match="unloaded"):
        translator.translate("こんにちは。")
    assert len(native.calls) == 1
    assert translator.translate("  ") == ""


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
@pytest.mark.parametrize("use_context,use_characters", [(False, False), (True, False), (False, True), (True, True)])
def test_translate_removes_added_quotes_but_preserves_quoted_source(
    native_loader, cls, directory, filename, use_context, use_characters,
):
    translator = cls(native_loader.root, device="cpu")
    metadata = {
        "context": [["おはよう。", "Good morning."]] if use_context else None,
        "character_info": [{"meta_id": "a", "name_en": "Aiko"}] if use_characters else None,
        "meta_id": "a" if use_characters else None,
    }
    try:
        translator.model.response = {"choices": [{"message": {"content": '"Hello."'}}]}
        assert translator.translate("こんにちは。", **metadata) == "Hello."
        assert translator.translate("「こんにちは。」", **metadata) == '"Hello."'
    finally:
        translator.unload()


@pytest.mark.parametrize("text,source,expected", [
    ('"Hello."', "こんにちは。", "Hello."),
    ('  “Hello.”  ', "こんにちは。", "Hello."),
    ('„Hello.“', "こんにちは。", "Hello."),
    ('＂Hello.＂', "こんにちは。", "Hello."),
    ('```text\n"Hello."\n```', "こんにちは。", "Hello."),
    ('"Line one.\nLine two."', "一行目。\n二行目。", "Line one.\nLine two."),
    ('"Hello."', None, '"Hello."'),
    ('"Hello."', "「こんにちは。」", '"Hello."'),
    ('"Hello."', "『こんにちは。』", '"Hello."'),
    ('“Hello.”', "“你好。”", '“Hello.”'),
    ('"Hello."', '"안녕."', '"Hello."'),
    ('He said "Go."', "行けと言った。", 'He said "Go."'),
    ('"Hello." "Goodbye."', "こんにちは。さようなら。", '"Hello." "Goodbye."'),
    ('“Hello.” “Goodbye.”', "こんにちは。さようなら。", '“Hello.” “Goodbye.”'),
    ('"Hello.', "こんにちは。", '"Hello.'),
    ('"He said \'Go.\'"', "「行け」と言った。", '"He said \'Go.\'"'),
])
def test_cleanup_only_removes_an_unambiguous_added_quote_pair(text, source, expected):
    assert gemma.clean_translation(text, source=source) == expected


def test_quantizations_own_independent_handles_and_locks(native_loader):
    q8 = gemma.GemmaE4BQ8Translator(native_loader.root, device="cpu")
    q6 = gemma.GemmaE4BQ6Translator(native_loader.root, device="cpu")
    assert q8.model is not q6.model
    assert q8._lock is not q6._lock
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            # Holding one instance's lock must not stall another quantization.
            with q8._lock:
                assert pool.submit(q6.translate, "こんにちは。").result(timeout=3) == "Hello."
        q8.unload()
        assert not q6.model.closed.is_set()
        assert q6.translate("ありがとう。") == "Hello."
    finally:
        q8.unload()
        q6.unload()
    assert [model.close_count for model in native_loader.instances] == [1, 1]


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
@pytest.mark.parametrize("phase", ["tokenize", "generate"])
def test_unload_waits_for_native_tokenization_and_generation(native_loader, cls, directory, filename, phase):
    translator = cls(native_loader.root, device="cpu")
    native = translator.model
    native.block_phase = phase
    unloading = threading.Event()

    def unload():
        unloading.set()
        translator.unload()

    with ThreadPoolExecutor(max_workers=2) as pool:
        translation = pool.submit(translator.translate, "こんにちは。")
        closing = None
        try:
            assert native.entered.wait(3), f"translation did not enter {phase}"
            closing = pool.submit(unload)
            assert unloading.wait(3)
            assert not native.closed.wait(0.1), f"unload released weights during {phase}"
        finally:
            native.release.set()
        assert translation.result(timeout=3) == "Hello."
        assert closing is not None
        closing.result(timeout=3)
    assert native.close_count == 1
    assert not native.closed_while_active
    assert translator.model is None


@pytest.mark.parametrize("cls,directory,filename", _VARIANTS)
def test_native_close_failure_still_detaches_handle(native_loader, cls, directory, filename):
    translator = cls(native_loader.root, device="cpu")
    attempts = []

    def fail():
        attempts.append(True)
        raise RuntimeError("native close failed")

    translator.model.close = fail
    translator.unload()
    translator.unload()
    assert translator.model is None
    assert attempts == [True]


def test_manager_closes_q8_before_loading_q6_and_reuses_current_model(native_loader, monkeypatch):
    from fox_reader.translate import local_mtl

    monkeypatch.setattr(local_mtl, "check_memory", lambda *args: None)
    monkeypatch.setattr(local_mtl, "torch_module", lambda: None)
    monkeypatch.setattr(local_mtl, "configure_threads", lambda preference: None)
    configured = {"japanese": "gemma-4-e4b-q8-uncensored", "chinese": "gemma-4-e4b-q6-uncensored"}
    settings = SimpleNamespace(get_default_model=configured.get, get_mtl_threads=lambda: "auto")
    manager = local_mtl.LocalMTLManager(native_loader.root, settings=settings)
    try:
        manager.load("japanese")
        old = manager.model
        assert isinstance(old, gemma.GemmaE4BQ8Translator)
        manager.load("japanese")
        assert manager.model is old
        assert len(native_loader.instances) == 1

        configured["japanese"] = "gemma-4-e4b-q6-uncensored"
        manager.load("japanese")
        current = manager.model
        assert isinstance(current, gemma.GemmaE4BQ6Translator)
        assert current is not old
        assert old.model is None
        assert native_loader.events == [
            ("load", GEMMA_E4B_Q8_MODEL_FILE),
            ("close", GEMMA_E4B_Q8_MODEL_FILE),
            ("load", GEMMA_E4B_Q6_MODEL_FILE),
        ]
        manager.load("japanese")
        manager.load("chinese")
        assert manager.model is current
        assert manager.current_lang == "chinese"
        assert len(native_loader.instances) == 2
    finally:
        manager.unload()
    assert [model.close_count for model in native_loader.instances] == [1, 1]
