"""The paddleocr_vl device slot: resolution, spelling, migration.

torch is stubbed at the probe boundary, so these run the same on a laptop
with no GPU and on CI without the torch extra at all.
"""

import pytest

from fox_reader import device as device_mod
from fox_reader.device import _Gpu, _Hardware

GIB = 1024**3


def _two_cards(monkeypatch):
    hardware = _Hardware(
        gpus=[_Gpu(index=0, name="RTX 4060", total_memory=8 * GIB),
              _Gpu(index=1, name="RTX 4090", total_memory=24 * GIB)],
        cuda_version="12.6",
    )
    monkeypatch.setattr(device_mod, "probe", lambda refresh=False: hardware)
    return hardware


def test_slot_order_and_presence():
    assert device_mod.SLOTS == ("paddleocr", "paddleocr_vl", "bubble", "translator", "textseg")


def test_auto_resolves_to_best_card_in_torch_spelling(monkeypatch):
    _two_cards(monkeypatch)
    resolution = device_mod.resolve({})
    assert resolution.resolved["paddleocr_vl"] == "cuda:1"
    assert resolution.device.paddleocr_vl == "cuda:1"
    assert resolution.fallbacks == []


def test_missing_slot_inherits_paddleocr(monkeypatch):
    _two_cards(monkeypatch)
    resolution = device_mod.resolve({"paddleocr": "cuda:0", "bubble": "cpu", "translator": "auto"})
    assert resolution.selection["paddleocr_vl"] == "cuda:0"
    assert resolution.device.paddleocr_vl == "cuda:0"


def test_explicit_pin_wins_over_inheritance(monkeypatch):
    _two_cards(monkeypatch)
    resolution = device_mod.resolve({"paddleocr": "cpu", "paddleocr_vl": "cuda:1"})
    assert resolution.selection["paddleocr_vl"] == "cuda:1"
    assert resolution.device.paddleocr == "cpu"
    assert resolution.device.paddleocr_vl == "cuda:1"


def test_inherited_missing_card_falls_back_with_its_own_line(monkeypatch):
    _two_cards(monkeypatch)
    resolution = device_mod.resolve({"paddleocr": "cuda:7"})
    assert resolution.device.paddleocr == "cpu"
    assert resolution.device.paddleocr_vl == "cpu"
    assert any("paddleocr_vl" in line for line in resolution.fallbacks)
    # The saved preference survives, exactly like every other slot.
    assert resolution.selection["paddleocr_vl"] == "cuda:7"


def test_metal_is_honoured_by_all_torch_and_llama_slots(monkeypatch):
    monkeypatch.setattr(device_mod, "probe", lambda refresh=False: _Hardware(mps=True))
    resolution = device_mod.resolve({})
    for slot in device_mod.SLOTS:
        assert resolution.resolved[slot] == "mps"
        assert getattr(resolution.device, slot) == "mps"
    assert resolution.fallbacks == []


def test_metal_keeps_explicit_cpu_choices(monkeypatch):
    monkeypatch.setattr(device_mod, "probe", lambda refresh=False: _Hardware(mps=True))
    resolution = device_mod.resolve({"paddleocr": "cpu", "paddleocr_vl": "mps"})
    assert resolution.device.paddleocr == "cpu"
    assert resolution.device.textseg == "cpu"  # inherited legacy preference
    assert resolution.device.paddleocr_vl == "mps"
    assert resolution.device.bubble == "mps"
    assert resolution.fallbacks == []


def test_snapshot_carries_the_slot(monkeypatch):
    _two_cards(monkeypatch)
    snapshot = device_mod.snapshot({"paddleocr": "cpu"})
    assert "paddleocr_vl" in snapshot["slots"]
    assert snapshot["selection"]["paddleocr_vl"] == "cpu"
    assert snapshot["resolved"]["paddleocr_vl"] == "cpu"
    assert snapshot["active"]["paddleocr_vl"] == device_mod.CPU


def test_settings_migration_inherits_paddleocr(tmp_path):
    import yaml

    from fox_reader.settings import SettingsManager

    root = tmp_path / "config"
    root.mkdir(parents=True)
    (root / "settings.yaml").write_text(
        yaml.safe_dump({"devices": {"paddleocr": "cpu"}}), encoding="utf-8"
    )
    mgr = SettingsManager(root)
    settings = mgr.load()
    assert settings.devices.paddleocr_vl == "cpu"
    # Persisted, so the two can diverge from here on.
    assert "paddleocr_vl" in (root / "settings.yaml").read_text(encoding="utf-8")


def test_settings_migration_leaves_explicit_value_alone(tmp_path):
    import yaml

    from fox_reader.settings import SettingsManager

    root = tmp_path / "config"
    root.mkdir(parents=True)
    (root / "settings.yaml").write_text(
        yaml.safe_dump({"devices": {"paddleocr": "cpu", "paddleocr_vl": "auto"}}),
        encoding="utf-8",
    )
    mgr = SettingsManager(root)
    assert mgr.load().devices.paddleocr_vl == "auto"
    assert mgr.get_devices()["paddleocr_vl"] == "auto"


def test_update_devices_accepts_and_rejects(tmp_path):
    from fox_reader.settings import SettingsManager

    mgr = SettingsManager(tmp_path)
    mgr.load()
    mgr.update_devices({"paddleocr_vl": "cuda:0"})
    assert mgr.get_devices()["paddleocr_vl"] == "cuda:0"
    with pytest.raises(ValueError):
        mgr.update_devices({"paddleocr_vl": "rubbish"})


def test_vl_engine_reads_its_own_slot(monkeypatch):
    import types
    from pathlib import Path

    import fox_reader.ocr_vl as vl_mod

    monkeypatch.setattr(
        vl_mod, "FOX_DEVICE",
        types.SimpleNamespace(paddleocr_vl="cuda:1", paddleocr="cpu"),
        raising=False,
    )
    monkeypatch.setattr(
        vl_mod, "resolve_vl_weights",
        lambda root=None: (Path("/tmp/m.gguf"), Path("/tmp/p.gguf")),
    )
    eng = vl_mod.PaddleOCRVLEngine()
    assert eng.device == "cuda:1"


def test_vl_engine_falls_back_to_classic_slot_without_one(monkeypatch):
    import types
    from pathlib import Path

    import fox_reader.ocr_vl as vl_mod

    # A FOX_DEVICE built before the slot existed has no attribute at all.
    monkeypatch.setattr(
        vl_mod, "FOX_DEVICE", types.SimpleNamespace(paddleocr="cpu"), raising=False
    )
    monkeypatch.setattr(
        vl_mod, "resolve_vl_weights",
        lambda root=None: (Path("/tmp/m.gguf"), Path("/tmp/p.gguf")),
    )
    eng = vl_mod.PaddleOCRVLEngine()
    assert eng.device == "cpu"


@pytest.mark.parametrize("module_name", [
    "gemma_e4b_q8_llamacpp", "gemma_e4b_q6_llamacpp", "vntl_llama3_8b_llamacpp",
])
def test_gguf_metal_offload_and_cpu_only_build_rejection(module_name):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module(f"fox_reader.translate.machine_translation.{module_name}")
    metal = SimpleNamespace(llama_supports_gpu_offload=lambda: True)
    cpu = SimpleNamespace(llama_supports_gpu_offload=lambda: False)
    # Metal handles its own GPU placement: never pass CUDA ordinals/split modes.
    assert module.llama_device_kwargs("mps", metal) == {"n_gpu_layers": -1}
    module._validate_requested_device("mps", metal)
    with pytest.raises(RuntimeError, match="CPU-only"):
        module._validate_requested_device("mps", cpu)
    module._validate_requested_device("cpu", cpu)


def test_vl_metal_offloads_both_model_and_projector(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import fox_reader.ocr_vl as vl

    monkeypatch.setattr(vl, "resolve_vl_weights", lambda root=None: (
        tmp_path / "model.gguf", tmp_path / "projector.gguf",
    ))
    engine = vl.PaddleOCRVLEngine(device="mps")
    seen = {}
    handler = object()

    def vision_handler(llama_cpp, use_gpu=False):
        seen["projector_gpu"] = use_gpu
        return handler

    def llama(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(engine, "_vision_handler", vision_handler)
    monkeypatch.setattr(vl, "supports_gpu_offload", lambda: True)
    library = SimpleNamespace(Llama=llama)
    engine._validate_device(library)
    engine._build_llama(library)
    assert seen["n_gpu_layers"] == -1
    assert seen["projector_gpu"] is True
    assert seen["chat_handler"] is handler
    assert "main_gpu" not in seen
    monkeypatch.setattr(vl, "supports_gpu_offload", lambda: False)
    with pytest.raises(RuntimeError, match="CPU-only"):
        engine._validate_device(library)


@pytest.mark.parametrize(("platform_name", "configured", "expected"), [
    ("darwin", None, "1"),
    ("darwin", "0", "0"),
    ("win32", None, None),
    ("linux", None, None),
])
def test_mps_fallback_is_configured_before_submodule_imports(monkeypatch, platform_name, configured, expected):
    import os
    import runpy
    import sys
    from pathlib import Path

    package_init = Path(device_mod.__file__).with_name("__init__.py")
    monkeypatch.setattr(sys, "platform", platform_name)
    if configured is None:
        monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    else:
        monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", configured)
    runpy.run_path(str(package_init))
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == expected


def test_textseg_unload_releases_metal_cache_even_when_cuda_probe_fails(monkeypatch):
    from types import SimpleNamespace

    from fox_reader.clean import seg

    released = []

    def unavailable_cuda():
        raise RuntimeError("no CUDA runtime")

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=unavailable_cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(empty_cache=lambda: released.append(True)),
    )
    monkeypatch.setattr(device_mod, "torch_module", lambda: torch)
    monkeypatch.setattr(seg, "_model", object())
    monkeypatch.setattr(seg, "_model_device", "mps")
    monkeypatch.setattr(seg, "_model_want", "mps")
    seg.unload()
    assert seg._model is seg._model_device is seg._model_want is None
    assert released == [True]
