"""Persisted context choices and the settings API's next-load contract."""

from __future__ import annotations

import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader.constants import MTL_MODELS, mtl_model
from fox_reader.routes import settings as routes
from fox_reader.settings import FoxSettings, SettingsManager

GEMMA = "gemma-4-e4b-q6-uncensored"
QWEN = "qwen3.5-9b-q6-k-uncensored"
QWEN_OTHER = "qwen3.5-4b-q8-0-uncensored"
VNTL = "vntl-llama3-8b-v2"


@pytest.fixture
def manager(tmp_path):
    result = SettingsManager(tmp_path / "config")
    result.load()
    return result


def _write_gguf_limit(path, limit):
    def string(text):
        encoded = text.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
        + string("general.architecture") + struct.pack("<I", 8) + string("qwen35")
        + string("qwen35.context_length") + struct.pack("<II", 4, limit)
    )


@pytest.fixture
def api(manager, tmp_path, monkeypatch):
    # These independent sections do not require OCR/device probing to verify
    # the translation settings contract.
    monkeypatch.setattr(routes, "_build_devices_payload", lambda request: {})
    monkeypatch.setattr(routes, "_build_ocr_payload", lambda request: {})
    status = {"loaded": False, "model_id": None}
    service = SimpleNamespace(
        ml_context_status=Mock(side_effect=lambda: dict(status)),
        ml_status=Mock(side_effect=AssertionError("must not wait on the generation lock")),
        ml_load=Mock(), ml_unload=Mock(),
    )
    app = FastAPI()
    app.state.settings = manager
    app.state.mtl_dir = tmp_path / "models"
    app.state.translate = service
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, app=app, manager=manager, status=status, service=service)


def _model(payload, model_id, language="japanese"):
    return next(model for model in payload["languages"][language]["models"] if model["id"] == model_id)


def test_missing_preferences_default_to_8192_for_every_model(manager):
    assert manager.settings.mtl_context_sizes == {}
    assert all(manager.get_mtl_context_size(model["id"]) == 8192 for model in MTL_MODELS)


def test_independent_model_choices_persist_and_preserve_other_settings(manager):
    manager.update_api_tokens(deepl_api_token="keep-deepl", jpdb_api_token="keep-jpdb")
    manager.update_default_model("japanese", GEMMA)
    manager.update_devices({"translator": "cpu"})
    manager.update_mtl_context_size(QWEN, 16384)
    manager.update_mtl_context_size(GEMMA, 16384)
    manager.update_mtl_context_size(VNTL, 8192)
    manager.update_mtl_context_size(GEMMA, 8192)
    reloaded = SettingsManager(manager.config_root)
    reloaded.load()
    assert reloaded.settings.mtl_context_sizes == {QWEN: 16384, GEMMA: 8192, VNTL: 8192}
    assert reloaded.get_default_model("japanese") == GEMMA
    assert reloaded.get_devices()["translator"] == "cpu"
    assert reloaded.get_api_tokens() == {"deepl_api_token": "keep-deepl", "jpdb_api_token": "keep-jpdb"}


@pytest.mark.parametrize("value", [True, False, 8192.0, 16384.0, "8192", None, 4096, 32768, [], {}])
def test_invalid_updates_never_change_memory_or_disk(manager, value):
    manager.update_mtl_context_size(QWEN, 16384)
    before = manager.settings_path.read_bytes()
    settings = manager.settings
    with pytest.raises(ValueError):
        manager.update_mtl_context_size(QWEN, value)
    assert manager.settings is settings
    assert manager.settings_path.read_bytes() == before


@pytest.mark.parametrize("model_id", ["missing-model", "", None, True, 123, []])
def test_unknown_or_invalid_model_is_rejected_before_initial_settings_write(tmp_path, model_id):
    manager = SettingsManager(tmp_path / "not-yet-created")
    with pytest.raises(ValueError):
        manager.update_mtl_context_size(model_id, 8192)
    assert not manager.settings_path.exists()


@pytest.mark.parametrize("model_id", [VNTL, "vntl-llama3-8b-v2-q5-k-m"])
def test_vntl_does_not_accept_16384(manager, model_id):
    before = manager.settings_path.read_bytes()
    with pytest.raises(ValueError, match="8192"):
        manager.update_mtl_context_size(model_id, 16384)
    assert manager.settings_path.read_bytes() == before


@pytest.mark.parametrize("optional", [None, "16384", [], True, {QWEN: 16384, GEMMA: "16384", VNTL: 16384,
                                                           QWEN_OTHER: True, "unknown": 8192, 42: 8192}])
def test_bad_saved_optional_entries_do_not_reset_other_settings(manager, optional):
    saved = FoxSettings().model_dump()
    saved["deepl_api_token"] = "preserved-token"
    saved["mtl_defaults"]["japanese"] = GEMMA
    saved["devices"]["translator"] = "cpu"
    saved["mtl_context_sizes"] = optional
    manager.settings_path.write_text(yaml.safe_dump(saved), encoding="utf-8")
    loaded = manager.load()
    assert loaded.deepl_api_token == "preserved-token"
    assert loaded.mtl_defaults.japanese == GEMMA
    assert loaded.devices.translator == "cpu"
    assert loaded.mtl_context_sizes == ({QWEN: 16384} if isinstance(optional, dict) else {})


def test_failed_persistence_restores_previous_context_preference(manager, monkeypatch):
    manager.update_mtl_context_size(QWEN, 16384)
    before = manager.settings_path.read_bytes()
    settings = manager.settings
    monkeypatch.setattr(manager, "save", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        manager.update_mtl_context_size(QWEN, 8192)
    assert manager.settings is settings
    assert manager.settings_path.read_bytes() == before


@pytest.mark.parametrize("fails", [False, True], ids=["commit", "rollback"])
def test_runtime_reader_waits_for_save_commit_or_rollback(manager, monkeypatch, fails):
    saving = threading.Event()
    release_save = threading.Event()
    reading = threading.Event()
    read_finished = threading.Event()
    original_save = manager.save

    def save():
        saving.set()
        assert release_save.wait(3), "test did not release settings save"
        if fails:
            raise OSError("disk full")
        original_save()

    def read():
        reading.set()
        try:
            return manager.get_mtl_context_size(QWEN)
        finally:
            read_finished.set()

    monkeypatch.setattr(manager, "save", save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        updating = pool.submit(manager.update_mtl_context_size, QWEN, 16384)
        try:
            assert saving.wait(3)
            reader = pool.submit(read)
            assert reading.wait(3)
            assert not read_finished.wait(0.1), "reader observed an uncommitted preference"
        finally:
            release_save.set()
        if fails:
            with pytest.raises(OSError, match="disk full"):
                updating.result(timeout=3)
        else:
            updating.result(timeout=3)
        expected = 8192 if fails else 16384
        assert reader.result(timeout=3) == expected
    reloaded = SettingsManager(manager.config_root)
    assert reloaded.get_mtl_context_size(QWEN) == expected


def test_concurrent_context_updates_preserve_both_preferences(manager):
    start = threading.Barrier(2)

    def update(model_id):
        start.wait(timeout=3)
        manager.update_mtl_context_size(model_id, 16384)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(update, QWEN)
        second = pool.submit(update, GEMMA)
        first.result(timeout=3)
        second.result(timeout=3)
    reloaded = SettingsManager(manager.config_root)
    assert reloaded.get_mtl_context_size(QWEN) == 16384
    assert reloaded.get_mtl_context_size(GEMMA) == 16384


def test_context_api_returns_all_models_and_verified_choices(api):
    response = api.client.get("/api/settings")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {"languages", "defaults", "api_tokens", "devices", "ocr", "fonts"}
    assert len(payload["languages"]["japanese"]["models"]) == len(MTL_MODELS)
    for model in MTL_MODELS:
        context = _model(payload, model["id"])["context"]
        assert context["selected"] == context["effective"] == 8192
        assert context["running"] is None
        assert context["pending"] is False
        assert context["options"] == ([8192] if model["id"].startswith("vntl-") else [8192, 16384])
        assert context["max"] == model["max_context_tokens"]
    api.service.ml_context_status.assert_called_once()
    api.service.ml_status.assert_not_called()


def test_save_affects_next_load_without_changing_running_model(api):
    api.status.update(loaded=True, model_id=QWEN, context_size=8192)
    response = api.client.put("/api/settings/mtl-context", json={"model_id": QWEN, "context_size": 16384})
    assert response.status_code == 200
    payload = response.json()
    context = _model(payload, QWEN)["context"]
    assert context["selected"] == context["effective"] == 16384
    assert context["running"] == 8192 and context["pending"] is True
    for language in ("japanese", "chinese", "korean"):
        assert _model(payload, QWEN, language)["context"] == context
    assert _model(payload, GEMMA)["context"]["running"] is None
    assert api.status["context_size"] == 8192
    assert api.manager.get_mtl_context_size(QWEN) == 16384
    api.service.ml_load.assert_not_called()
    api.service.ml_unload.assert_not_called()
    api.service.ml_status.assert_not_called()
    response = api.client.put("/api/settings/mtl-context", json={"model_id": QWEN, "context_size": 8192})
    assert _model(response.json(), QWEN)["context"]["pending"] is False


def test_preference_can_be_saved_before_model_download(api):
    assert not _model(api.client.get("/api/settings").json(), QWEN)["downloaded"]
    response = api.client.put("/api/settings/mtl-context", json={"model_id": QWEN, "context_size": 16384})
    assert response.status_code == 200
    assert _model(response.json(), QWEN)["context"]["effective"] == 16384


def test_minimal_advisory_snapshot_reports_current_context_and_pending_change(api):
    api.manager.update_mtl_context_size(QWEN, 16384)
    api.status.clear()
    api.status.update(model_id=QWEN, context_size=8192)
    context = _model(api.client.get("/api/settings").json(), QWEN)["context"]
    assert context["running"] == 8192 and context["pending"] is True
    api.service.ml_status.assert_not_called()


@pytest.mark.parametrize("body", [
    None, [], "invalid", {}, {"model_id": QWEN}, {"context_size": 8192},
    {"model_id": QWEN, "context_size": 8192, "typo": True},
    {"model_id": "unknown", "context_size": 8192},
    {"model_id": 123, "context_size": 8192},
    {"model_id": QWEN, "context_size": True},
    {"model_id": QWEN, "context_size": 16384.0},
    {"model_id": QWEN, "context_size": "16384"},
    {"model_id": QWEN, "context_size": 32768},
    {"model_id": VNTL, "context_size": 16384},
])
def test_context_route_rejects_bad_body_without_partial_write(api, body):
    before = api.manager.settings_path.read_bytes()
    response = api.client.put("/api/settings/mtl-context", json=body)
    assert response.status_code == 400
    assert response.json()["error"]
    assert api.manager.settings_path.read_bytes() == before


def test_context_route_rejects_malformed_json(api):
    before = api.manager.settings_path.read_bytes()
    response = api.client.put(
        "/api/settings/mtl-context", content="{broken", headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert api.manager.settings_path.read_bytes() == before


def test_actual_gguf_limit_narrows_options_and_rejects_oversized_save(api):
    api.manager.update_mtl_context_size(QWEN, 16384)
    entry = mtl_model(QWEN)
    _write_gguf_limit(api.app.state.mtl_dir / entry["dest"] / entry["files"][0], 8192)
    context = _model(api.client.get("/api/settings").json(), QWEN)["context"]
    assert context["selected"] == 16384
    assert context["effective"] == context["max"] == 8192
    assert context["options"] == [8192]
    before = api.manager.settings_path.read_bytes()
    response = api.client.put("/api/settings/mtl-context", json={"model_id": QWEN, "context_size": 16384})
    assert response.status_code == 400
    assert api.manager.settings_path.read_bytes() == before


def test_unsupported_actual_gguf_keeps_settings_page_usable(api):
    entry = mtl_model(QWEN)
    _write_gguf_limit(api.app.state.mtl_dir / entry["dest"] / entry["files"][0], 4096)
    api.status.update(loaded=True, model_id=QWEN, context_size=8192)
    response = api.client.get("/api/settings")
    assert response.status_code == 200
    context = _model(response.json(), QWEN)["context"]
    assert context["options"] == [] and context["effective"] is None
    assert context["max"] == 4096 and context["running"] == 8192
    assert context["pending"] is False and "8192" in context["error"]
    assert api.client.put("/api/settings/mtl-context", json={"model_id": QWEN, "context_size": 8192}).status_code == 400


@pytest.mark.parametrize("status", [None, [], {"loaded": True, "model_id": QWEN},
                                    {"loaded": True, "model_id": QWEN, "context_size": True}])
def test_missing_or_incomplete_runtime_snapshot_is_tolerated(api, status):
    api.service.ml_context_status = Mock(return_value=status)
    response = api.client.get("/api/settings")
    assert response.status_code == 200
    assert _model(response.json(), QWEN)["context"]["running"] is None


def test_legacy_service_and_optional_getter_do_not_break_settings(api, monkeypatch):
    api.app.state.translate = SimpleNamespace(ml_status=Mock(side_effect=AssertionError("never call locking status")))
    monkeypatch.setattr(api.manager, "get_mtl_context_size", None)
    response = api.client.get("/api/settings")
    assert response.status_code == 200
    assert _model(response.json(), QWEN)["context"]["selected"] == 8192
    api.app.state.translate.ml_status.assert_not_called()


def test_route_disk_failure_preserves_previous_settings(api, monkeypatch):
    before = api.manager.settings_path.read_bytes()
    monkeypatch.setattr(api.manager, "save", Mock(side_effect=OSError("disk full")))
    response = api.client.put("/api/settings/mtl-context", json={"model_id": QWEN, "context_size": 16384})
    assert response.status_code == 500
    assert api.manager.get_mtl_context_size(QWEN) == 8192
    assert api.manager.settings_path.read_bytes() == before
