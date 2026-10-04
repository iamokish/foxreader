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


def test_metal_is_honoured_unlike_classic_paddleocr(monkeypatch):
    monkeypatch.setattr(device_mod, "probe", lambda refresh=False: _Hardware(mps=True))
    resolution = device_mod.resolve({})
    assert resolution.device.paddleocr == "cpu"
    assert resolution.device.paddleocr_vl == "mps"


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
