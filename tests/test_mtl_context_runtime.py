"""Context preferences bind to one load and to that load's prompt budget."""

import struct
from concurrent.futures import ThreadPoolExecutor

import pytest

from fox_reader.constants import MTL_MODELS, mtl_model
from fox_reader.settings import SettingsManager
from fox_reader.translate import local_mtl
from fox_reader.translate.machine_translation import gemma_e4b_llamacpp as gemma
from fox_reader.translate.machine_translation import qwen35_llamacpp as qwen
from fox_reader.translate.prompt import build_translation_messages
from tests.test_mtl_lifecycle import _CLASSES
from tests.test_mtl_lifecycle import native as native


def _write_context_header(path, limit):
    def string(value):
        data = value.encode("utf-8")
        return struct.pack("<Q", len(data)) + data

    path.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
        + string("general.architecture") + struct.pack("<I", 8) + string("qwen35")
        + string("qwen35.context_length") + struct.pack("<II", 4, limit)
    )


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda cls: cls.model_id())
def test_native_window_never_exceeds_model_limit(native, cls):
    expected = min(16384, mtl_model(cls.model_id())["max_context_tokens"])
    translator = cls(native.root, device="cpu", n_ctx=16384)
    assert translator.n_ctx == expected
    assert translator.model.kwargs["n_ctx"] == expected
    assert translator.translate("こんにちは。") == "Hello."
    translator.unload()
    assert native.instances[0].close_count == 1


@pytest.mark.parametrize("cls", [gemma.GemmaE4BQ6Translator, qwen.Qwen35_4BQ8Translator])
def test_replacement_gguf_can_lower_the_configured_window_before_allocation(native, cls):
    entry = mtl_model(cls.model_id())
    _write_context_header(native.root / entry["dest"] / entry["files"][0], 8192)
    translator = cls(native.root, device="cpu", n_ctx=16384)
    assert translator.n_ctx == 8192
    assert native.attempts[0]["n_ctx"] == 8192
    translator.unload()


@pytest.mark.parametrize("cls", [gemma.GemmaE4BQ6Translator, qwen.Qwen35_4BQ8Translator])
def test_too_small_actual_context_is_rejected_before_native_allocation(native, cls):
    entry = mtl_model(cls.model_id())
    _write_context_header(native.root / entry["dest"] / entry["files"][0], 4096)
    translator = cls.__new__(cls)
    with pytest.raises(ValueError, match="minimum 8192"):
        translator.__init__(native.root, device="cpu", n_ctx=16384)
    translator.unload()
    assert native.attempts == []


@pytest.mark.parametrize("cls", [gemma.GemmaE4BQ6Translator, qwen.Qwen35_4BQ8Translator])
def test_prompt_budget_uses_each_loaded_instances_window(native, cls):
    small = cls(native.root, device="cpu", n_ctx=8192)
    large = cls(native.root, device="cpu", n_ctx=16384)
    source = "Long dialogue. " * 600
    try:
        with pytest.raises(RuntimeError, match="8192-token context window"):
            small.translate(source)
        assert large.translate(source) == "Hello."
        assert small.n_ctx == 8192
        assert large.n_ctx == 16384
        assert not small.model.calls
        assert len(large.model.calls) == 1
    finally:
        small.unload()
        large.unload()


def test_larger_window_retains_more_context_without_mutating_source():
    class Tokenizer:
        def tokenize(self, value, **kwargs):
            return list(value)

    pairs = [[f"old{i}" + "a" * 990, "b" * 990] for i in range(8)]
    small = build_translation_messages(Tokenizer(), "New source", pairs, n_ctx=8192)
    large = build_translation_messages(Tokenizer(), "New source", pairs, n_ctx=16384)
    assert len(large[0]["content"]) > len(small[0]["content"])
    assert large[1] == small[1] == {"role": "user", "content": "New source"}
    assert "old7" in small[0]["content"] and "old7" in large[0]["content"]


def test_saving_context_only_applies_after_explicit_unload_and_next_load(native, monkeypatch, tmp_path):
    monkeypatch.setattr(local_mtl, "check_memory", lambda *args, **kwargs: None)
    settings = SettingsManager(tmp_path / "settings")
    settings.load()
    model_id = "gemma-4-e4b-q6-uncensored"
    settings.update_defaults({"japanese": model_id, "chinese": model_id})
    manager = local_mtl.LocalMTLManager(native.root, settings)
    manager.load("japanese")
    original = manager.model
    try:
        settings.update_mtl_context_size(model_id, 16384)
        manager.load("japanese")
        assert manager.translate("chinese", "你好。") == "Hello."
        assert manager.model is original
        assert manager.status()["context_size"] == 8192
        assert len(native.instances) == 1
        # The settings page remains responsive while the generation lock is held.
        with ThreadPoolExecutor(max_workers=1) as pool:
            with manager._lock:
                snapshot = pool.submit(manager.context_status).result(timeout=2)
        assert snapshot == {"model_id": model_id, "context_size": 8192}
        manager.unload()
        manager.load("japanese")
        assert manager.model is not original
        assert manager.status()["context_size"] == 16384
        assert native.instances[1].kwargs["n_ctx"] == 16384
        assert native.instances[0].close_count == 1
    finally:
        manager.unload()


@pytest.mark.parametrize("model", [model for model in MTL_MODELS if model["max_context_tokens"] >= 16384])
def test_larger_window_memory_gate_adds_cache_cost_without_doubling_weights(monkeypatch, model):
    monkeypatch.setattr(local_mtl, "free_ram_bytes", lambda: (model["cpu_ram_mib"] + 200) * 1024**2)
    small = local_mtl.check_memory(model["id"], "cpu", context_size=8192)
    large = local_mtl.check_memory(model["id"], "cpu", context_size=16384)
    assert small.fits is True and large.fits is False
    assert large.required_mib == small.required_mib + 384
    assert "16,384-token context" in large.message


def test_unusable_model_context_does_not_hide_other_models_memory_warnings(tmp_path, monkeypatch):
    small_id = gemma.GemmaE4BQ6Translator.model_id()
    large_id = qwen.Qwen35_9BQ8Translator.model_id()
    entry = mtl_model(small_id)
    model_path = tmp_path / entry["dest"] / entry["files"][0]
    model_path.parent.mkdir(parents=True)
    _write_context_header(model_path, 4096)
    settings = SettingsManager(tmp_path / "settings.yaml")
    settings.update_defaults({"japanese": small_id, "korean": large_id})
    manager = local_mtl.LocalMTLManager(tmp_path, settings)
    monkeypatch.setattr(local_mtl.LocalMTLManager, "device", property(lambda _self: "cpu"))
    monkeypatch.setattr(local_mtl, "free_ram_bytes", lambda: 1024**3)

    report = manager.memory_report()

    assert report["ok"] is False
    assert [row["model"] for row in report["checks"]] == [large_id]
    assert [row["model"] for row in report["issues"]] == [large_id]
    assert report["issues"][0]["languages"] == ["korean"]
