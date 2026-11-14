"""The setup location is validated, applied and restored as one user preference."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from fox_reader import config as config_module
from fox_reader import fsaccess, model_terms
from fox_reader.config import ConfigManager
from fox_reader.routes import setup


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "reader"
    root.mkdir()
    (root / "models").mkdir()
    manager = ConfigManager(root / "config")
    config = manager.load()
    monkeypatch.setattr(setup, "PROJECT_ROOT", root)
    monkeypatch.setattr(setup, "MODELS_DIR", root / "models")
    monkeypatch.setattr(setup, "_MTL_DIR", root / "models")
    monkeypatch.setattr(setup, "_storage_apply_generation", 0)
    monkeypatch.setattr(setup, "_download_state", {
        "active": False, "stage": "mtl_selection", "current_model": None,
        "progress": {}, "error": None, "complete": False,
    })
    monkeypatch.setattr(setup, "_required_models_ready", lambda: True)
    monkeypatch.setattr(setup, "_serialize_required_models", lambda *args: [])
    monkeypatch.setattr(setup, "_optional_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(setup, "_get_mtl_models", lambda path, *args: [
        {"id": "example", "downloaded": (path / "ready").is_file()},
    ])
    monkeypatch.setattr(setup, "_all_mtl_models_downloaded", lambda path: (path / "ready").is_file())
    monkeypatch.setattr(model_terms, "state", lambda: {})
    app = FastAPI()
    app.include_router(setup.router)
    app.state.config_manager = manager
    app.state.config = config
    app.state.mtl_dir = root / "models"
    with TestClient(app) as client:
        yield SimpleNamespace(root=root, manager=manager, app=app, client=client)


@pytest.mark.parametrize("spelling", ['  "models"  ', "'models'", "models", "./models"])
def test_inspect_normalizes_quoted_and_relative_paths_without_saving(env, spelling):
    before = env.manager.config_path.read_bytes()
    report = env.client.post("/api/setup/storage/inspect", json={"path": spelling}).json()
    assert report["ok"] is True
    assert report["path"] == str(env.root / "models")
    assert report["exists"] is True and report["can_create"] is False
    assert report["free_bytes"] >= 0
    assert env.manager.config_path.read_bytes() == before
    assert setup._MTL_DIR == env.root / "models"
    assert not list((env.root / "models").glob(".fox_write_probe.*"))


def test_inspect_expands_home_and_accepts_new_child_without_creating_it(env, monkeypatch):
    monkeypatch.setenv("HOME", str(env.root))
    monkeypatch.setenv("USERPROFILE", str(env.root))
    target = env.root / "MTL weights"
    report = env.client.post("/api/setup/storage/inspect", json={"path": "~/MTL weights"}).json()
    assert report["ok"] and report["can_create"]
    assert report["path"] == str(target)
    assert not target.exists()


@pytest.mark.parametrize("path", ["", "   ", '""', "bad\x00folder", "missing/child"])
def test_inspect_rejects_empty_invalid_and_missing_parent_paths(env, path):
    before = env.manager.config_path.read_bytes()
    response = env.client.post("/api/setup/storage/inspect", json={"path": path})
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["error"]
    assert env.manager.config_path.read_bytes() == before
    assert not (env.root / "missing").exists()


def test_file_and_unwritable_existing_target_are_rejected(env, monkeypatch):
    (env.root / "a-file").write_text("keep me", encoding="utf-8")
    report = env.client.post("/api/setup/storage/inspect", json={"path": "a-file"}).json()
    assert not report["ok"] and "file" in report["error"]
    calls = []

    def deny(path):
        calls.append(path)
        return False

    monkeypatch.setattr(fsaccess, "_probe_writable", deny)
    report = env.client.post("/api/setup/storage/inspect", json={"path": "models"}).json()
    assert not report["ok"] and "write" in report["error"]
    assert calls == [str(env.root / "models")], "check the target ACL, not just its parent"


@pytest.mark.skipif(os.name != "nt", reason="Windows filesystem rules")
@pytest.mark.parametrize("path", ["CON", "NUL.txt", "bad:name", "bad?name", "bad.", "C:relative"])
def test_windows_filename_rules(env, path):
    assert env.client.post("/api/setup/storage/inspect", json={"path": path}).json()["ok"] is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX filesystem names")
@pytest.mark.parametrize("name", ['colon:name', 'question?mark', 'quoted"', "back\\slash", "CON"])
def test_posix_names_are_not_rewritten_or_rejected_as_windows_names(env, name):
    folder = env.root / name
    folder.mkdir()
    report = env.client.post("/api/setup/storage/inspect", json={"path": str(folder)}).json()
    assert report["ok"] and report["path"] == str(folder)
    assert not list(folder.iterdir())


@pytest.mark.parametrize("endpoint", ["/api/setup/storage", "/api/setup/storage/inspect"])
@pytest.mark.parametrize("body", [None, [], {}, {"path": 5}])
def test_storage_endpoints_reject_wrong_payloads(env, endpoint, body):
    assert env.client.post(endpoint, json=body).status_code == 400


def test_apply_persists_a_relative_preference_and_refreshes_status(env):
    target = env.root / "translation models"
    target.mkdir()
    (target / "ready").touch()
    setup._download_state["progress"] = {"old-model": {"done": True, "downloaded": True}}
    setup._download_state["error"] = "old download failure"
    response = env.client.post("/api/setup/storage", json={"path": '"translation models"'})
    assert response.status_code == 200
    assert response.json() == {
        "path": str(target), "mtl_dir": "translation models",
        "mtl_models": [{"id": "example", "downloaded": True}], "all_mtl_downloaded": True,
    }
    assert env.app.state.config.mtl_dir == Path("translation models")
    assert env.app.state.mtl_dir == setup._MTL_DIR == target
    assert ConfigManager(env.root / "config").load().mtl_dir == Path("translation models")
    status = env.client.get("/api/setup/status").json()
    assert status["mtl_dir"] == str(target)
    assert status["progress"] == {} and status["error"] is None


def test_apply_absolute_custom_and_default_locations_preserves_portability(env):
    custom = env.root.parent / "shared models"
    assert env.client.post("/api/setup/storage", json={"path": str(custom)}).status_code == 200
    assert env.manager.config.mtl_dir == custom
    assert not custom.exists(), "applying a location must not create it"
    response = env.client.post("/api/setup/storage", json={"path": str(env.root / "models")})
    assert response.json()["mtl_dir"] == "models"
    assert ConfigManager(env.root / "config").load().mtl_dir == Path("models")


@pytest.mark.parametrize("busy", ["download", "closing"])
def test_busy_apply_does_not_inspect_or_mutate(env, monkeypatch, busy):
    before = env.manager.config_path.read_bytes()
    setup._download_state["active"] = busy == "download"
    env.app.state.setup_closing = busy == "closing"
    monkeypatch.setattr(setup, "_inspect_storage", lambda value: pytest.fail("busy requests must not inspect"))
    assert env.client.post("/api/setup/storage", json={"path": "new"}).status_code == 409
    assert env.manager.config_path.read_bytes() == before
    assert setup._MTL_DIR == env.root / "models"


def test_apply_rechecks_busy_state_after_inspection(env, monkeypatch):
    inspect = setup._inspect_storage

    def became_busy(value):
        report = inspect(value)
        setup._download_state["active"] = True
        return report

    monkeypatch.setattr(setup, "_inspect_storage", became_busy)
    assert env.client.post("/api/setup/storage", json={"path": "new"}).status_code == 409
    assert env.manager.config.mtl_dir == Path("models")
    assert setup._MTL_DIR == env.root / "models"


def test_later_apply_wins_when_earlier_inspection_finishes_last(env, monkeypatch):
    inspect = setup._inspect_storage

    def delayed_inspection(value):
        report = inspect(value)
        if value == "earlier":
            # The first request is awaiting this worker thread, leaving the
            # event loop free to finish a later Apply before it resumes.
            later = env.client.post("/api/setup/storage", json={"path": "later"})
            assert later.status_code == 200
        return report

    monkeypatch.setattr(setup, "_inspect_storage", delayed_inspection)
    earlier = env.client.post("/api/setup/storage", json={"path": "earlier"})
    assert earlier.status_code == 409
    assert "newer" in earlier.json()["error"]
    assert env.manager.config.mtl_dir == Path("later")
    assert ConfigManager(env.root / "config").load().mtl_dir == Path("later")
    assert setup._MTL_DIR == env.app.state.mtl_dir == env.root / "later"


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_save_preserves_memory_disk_and_active_path(env, monkeypatch, failure):
    original = env.manager.config
    before = env.manager.config_path.read_bytes()

    def fail_write(data, stream, **kwargs):
        stream.write("partial document")
        raise OSError("disk full")

    def fail_replace(*args):
        raise PermissionError("config locked")

    if failure == "write":
        monkeypatch.setattr(config_module.yaml, "safe_dump", fail_write)
    else:
        monkeypatch.setattr(config_module.os, "replace", fail_replace)
    response = env.client.post("/api/setup/storage", json={"path": "new"})
    assert response.status_code == 503
    assert env.manager.config is original
    assert env.app.state.config is original
    assert env.manager.config_path.read_bytes() == before
    assert setup._MTL_DIR == env.app.state.mtl_dir == env.root / "models"
    assert not list((env.root / "config").glob(".*.tmp"))


def test_setup_page_keeps_an_active_jobs_location(env, monkeypatch):
    active = env.root / "active job"
    setup._MTL_DIR = active
    setup._download_state["active"] = True
    monkeypatch.setattr(setup.templates, "TemplateResponse", lambda **kwargs: JSONResponse(kwargs["context"]))
    context = env.client.get("/setup").json()
    assert setup._MTL_DIR == active
    assert context["mtl_resolved_dir"] == str(active)
    assert context["mtl_default_dir"] == str(env.root / "models")


def test_setup_page_derives_stage_without_reentering_download_lock(env, monkeypatch):
    def derive():
        assert setup._download_lock.acquire(blocking=False), "stage lookup must run outside the download lock"
        setup._download_lock.release()
        return "mtl_selection"

    monkeypatch.setattr(setup, "_derive_idle_stage", derive)
    monkeypatch.setattr(setup.templates, "TemplateResponse", lambda **kwargs: JSONResponse(kwargs["context"]))
    response = env.client.get("/setup")
    assert response.status_code == 200
    assert response.json()["mtl_resolved_dir"] == str(env.root / "models")


def test_setup_page_does_not_restore_a_config_snapshot_older_than_a_concurrent_apply(env, monkeypatch):
    target = env.root / "new location"

    def derive():
        # An apply publishes its configuration/path together while GET /setup
        # is between its stage lookup and the protected location snapshot.
        with setup._download_lock:
            env.app.state.config = env.manager.update(mtl_dir=Path("new location"))
            env.app.state.mtl_dir = setup._MTL_DIR = target
        return "mtl_selection"

    monkeypatch.setattr(setup, "_derive_idle_stage", derive)
    monkeypatch.setattr(setup.templates, "TemplateResponse", lambda **kwargs: JSONResponse(kwargs["context"]))
    context = env.client.get("/setup").json()
    assert context["mtl_dir"] == "new location"
    assert context["mtl_resolved_dir"] == str(target)
    assert setup._MTL_DIR == target


@pytest.mark.parametrize("error", [None, "Download interrupted"])
def test_status_returns_selection_when_mtl_worker_finishes_during_readiness_check(env, monkeypatch, error):
    setup._download_state.update(active=True, stage="mtl", current_model="test-mtl")

    def finish_worker():
        with setup._download_lock:
            setup._download_state.update(
                active=False, stage="mtl_selection", current_model=None, error=error,
            )
        return True

    monkeypatch.setattr(setup, "_required_models_ready", finish_worker)
    status = env.client.get("/api/setup/status").json()
    assert status["active"] is False
    assert status["stage"] == "mtl_selection"
    assert status["current_model"] is None
    assert status["error"] == error


def test_status_retains_new_job_stage_if_a_download_starts_during_readiness_check(env, monkeypatch):
    def start_worker():
        with setup._download_lock:
            setup._download_state.update(active=True, stage="mtl", current_model="test-mtl")
        return True

    monkeypatch.setattr(setup, "_required_models_ready", start_worker)
    status = env.client.get("/api/setup/status").json()
    assert status["active"] is True
    assert status["stage"] == "mtl"
    assert status["current_model"] == "test-mtl"


@pytest.mark.parametrize("stage", ["non_mtl", "mtl"])
def test_status_does_not_replace_an_active_stage_with_filesystem_readiness(env, monkeypatch, stage):
    setup._download_state.update(active=True, stage=stage)
    monkeypatch.setattr(setup, "_required_models_ready", lambda: stage != "non_mtl")
    status = env.client.get("/api/setup/status").json()
    assert status["active"] is True
    assert status["stage"] == stage


def test_direct_download_persists_only_after_terms_acceptance(env, monkeypatch):
    model = {"id": "test-mtl", "dest": "model"}
    accepted = False
    started = []
    monkeypatch.setattr(setup, "_resolve_mtl_models", lambda ids: [model])
    monkeypatch.setattr(setup, "_model_completed", lambda path, item: False)
    monkeypatch.setattr(
        setup, "_terms_refusal",
        lambda models: None if accepted else JSONResponse({"error": "terms"}, status_code=403),
    )

    class Worker:
        def __init__(self, *, target, args, daemon):
            self.args = args

        def start(self):
            started.append(self.args)

    monkeypatch.setattr(setup, "threading", SimpleNamespace(Thread=Worker))
    payload = {"mtl_dir": "new models", "selected_mtl": [model["id"]]}
    assert env.client.post("/api/setup/download", json=payload).status_code == 403
    assert env.manager.config.mtl_dir == Path("models")
    assert setup._MTL_DIR == env.root / "models"
    accepted = True
    response = env.client.post("/api/setup/download", json=payload)
    assert response.status_code == 200 and response.json()["status"] == "started"
    assert ConfigManager(env.root / "config").load().mtl_dir == Path("new models")
    assert started[0][0] == env.root / "new models"
    assert not (env.root / "new models").exists()
    setup._download_state["active"] = False
    payload["mtl_dir"] = str(env.root / "new models")
    assert env.client.post("/api/setup/download", json=payload).status_code == 200
    assert ConfigManager(env.root / "config").load().mtl_dir == Path("new models")


def test_direct_download_save_failure_does_not_start_worker(env, monkeypatch):
    monkeypatch.setattr(setup, "_resolve_mtl_models", lambda ids: [{"id": "test-mtl"}])
    monkeypatch.setattr(setup, "_model_completed", lambda path, item: False)
    monkeypatch.setattr(setup, "_terms_refusal", lambda models: None)

    def fail_save(*args):
        raise OSError("disk full")

    monkeypatch.setattr(config_module.os, "replace", fail_save)
    response = env.client.post("/api/setup/download", json={"mtl_dir": "new", "selected_mtl": ["test-mtl"]})
    assert response.status_code == 503
    assert setup._download_state["active"] is False
    assert setup._MTL_DIR == env.root / "models"
    assert env.manager.config.mtl_dir == Path("models")
