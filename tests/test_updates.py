"""Updates download verified bytes without touching the working installation."""
import hashlib
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader.routes.updates import router
from fox_reader.services import updates


def digest(data):
    return hashlib.sha256(data).hexdigest()


def manifest(parts=(b"one", b"two")):
    name = "fox-reader-v1.2.3-cpu-gguf-windows-x64.zip"
    whole = b"".join(parts)
    files = []
    for number, part in enumerate(parts, 1):
        part_name = f"{name}.{number:03d}" if len(parts) > 1 else name
        files.append({"name": part_name, "size": len(part), "sha256": digest(part),
                      "url": f"https://github.com/{updates.REPOSITORY}/releases/download/v1.2.3/{part_name}"})
    return {"schema_version": 1, "version": "1.2.3", "tag": "v1.2.3", "notes": "Fixed reader startup.",
            "pub_date": "2026-09-22T00:44:18.046Z",
            "release_url": f"https://github.com/{updates.REPOSITORY}/releases/tag/v1.2.3",
            "platform": {"windows": {"x86_64": {"variants": {"cpu-gguf": {
                "device": "cpu", "gguf": "cpu", "parts": files,
                "archive": {"name": name, "format": "zip", "size": len(whole), "sha256": digest(whole)},
            }}}}}}


def selected(value):
    return value["platform"]["windows"]["x86_64"]["variants"]["cpu-gguf"]


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(updates.utils, "is_frozen", True)
    build = {"version": "1.2.2", "platform": "windows", "architecture": "x86_64",
             "device": "cpu", "gguf": "cpu", "repository": updates.REPOSITORY}
    (tmp_path / "build-info.json").write_text(json.dumps(build), encoding="utf-8")
    return updates.UpdateService(tmp_path)


@pytest.fixture
def network(monkeypatch):
    bodies = {}
    calls = []

    @contextmanager
    def open_fake(url):
        calls.append(url)
        if url not in bodies:
            raise OSError("network unavailable")
        body = bodies[url]

        def blocks(_):
            if callable(body):
                yield from body()
            else:
                yield body
        yield SimpleNamespace(iter_bytes=blocks)
    monkeypatch.setattr(updates, "open_download", open_fake)
    return bodies, calls


def prepare(service, network, value=None, parts=(b"one", b"two")):
    value = value or manifest(parts)
    bodies, _ = network
    bodies[updates.MANIFEST_URL] = json.dumps(value).encode()
    for item, part in zip(selected(value)["parts"], parts):
        bodies[item["url"]] = part
    return value


def finish(service):
    service._thread.join(timeout=5)
    assert not service._thread.is_alive()
    return service.status()


@pytest.mark.parametrize("metadata_present", [False, True])
def test_source_cannot_update_even_with_copied_build_metadata(service, network, monkeypatch, metadata_present):
    if not metadata_present:
        (service.root / "build-info.json").unlink()
    monkeypatch.setattr(updates.utils, "is_frozen", False)
    source = updates.UpdateService(service.root)
    status = source.status()
    assert source.build is None
    assert status["current_version"] == updates.__version__
    assert status["updates_supported"] is False
    assert status["download_supported"] is False
    assert status["install_supported"] is False
    for action in (source.check, source.start):
        with pytest.raises(updates.UpdateError, match="running from source"):
            action()
    assert source._thread is None
    assert not (service.root / "updates").exists()
    assert network[1] == []
    result, chosen = updates.validate_manifest(manifest(), None, "1.2.2")
    assert result["available"] and not result["compatible"] and chosen is None


@pytest.mark.parametrize("platform", ["windows", "linux", "macos"])
def test_no_window_build_checks_release_but_cannot_download_native_windows(service, network, platform):
    build = {**service.build, "no_window": True, "platform": platform,
             "architecture": "arm64" if platform == "macos" else "x86_64"}
    (service.root / "build-info.json").write_text(json.dumps(build), encoding="utf-8")
    service = updates.UpdateService(service.root)
    value = prepare(service, network)
    client = client_for(service)
    headers = {"X-Fox-Reader-Update": "1", "Origin": "http://127.0.0.1"}

    status = client.get("/api/updates/status").json()
    assert status["current_version"] == "1.2.2"
    assert status["download_supported"] is False
    assert "--no-window" in status["reason"]
    assert network[1] == []

    response = client.get("/api/updates/check", headers=headers)
    assert response.status_code == 200
    release = response.json()
    assert release["available"] is True and release["compatible"] is False
    assert release["version"] == value["version"] and release["notes"] == value["notes"]
    assert release["variant"] is None and release["download_size"] == 0
    assert "--no-window" in release["reason"]
    assert client.post("/api/updates/download", headers=headers).status_code == 409
    assert network[1] == [updates.MANIFEST_URL]
    assert not (service.root / "updates").exists()


def test_explicit_native_window_build_keeps_update_support(service, network):
    build = {**service.build, "no_window": False}
    (service.root / "build-info.json").write_text(json.dumps(build), encoding="utf-8")
    service = updates.UpdateService(service.root)
    prepare(service, network)
    assert service.status()["updates_supported"] is True
    assert service.status()["download_supported"] is True
    assert service.check()["compatible"] is True


@pytest.mark.parametrize("no_window", [None, 0, 1, "false", "true", {}, []])
def test_invalid_no_window_metadata_cannot_select_update_archives(service, no_window):
    build = {**service.build, "no_window": no_window}
    (service.root / "build-info.json").write_text(json.dumps(build), encoding="utf-8")
    service = updates.UpdateService(service.root)
    assert service.build is None
    assert service.status()["download_supported"] is False


def test_checks_and_downloads_matching_variant_with_join_verification(service, network):
    value = prepare(service, network)
    result = service.check()
    assert result["available"] and result["compatible"]
    assert result["variant"] == "cpu-gguf" and result["download_size"] == 6
    assert service.start()["state"] == "downloading"
    status = finish(service)
    assert status["state"] == "complete"
    destination = service.root / "updates" / "1.2.3" / selected(value)["archive"]["name"]
    assert destination.read_bytes() == b"onetwo"
    assert status["archive"] == str(destination)
    assert "config, fonts, and models" in status["message"]
    assert len(list(destination.parent.iterdir())) == 1, "joined parts should not double disk usage forever"
    assert (service.root / "build-info.json").is_file()


def test_unsplit_download_has_one_verified_file(service, network):
    prepare(service, network, parts=(b"archive",))
    service.check()
    service.start()
    assert finish(service)["state"] == "complete"
    assert len(list((service.root / "updates" / "1.2.3").iterdir())) == 1


def test_resume_reuses_only_hash_verified_parts(service, network):
    value = prepare(service, network)
    first, second = selected(value)["parts"]
    network[0][second["url"]] = b"bad"
    service.check()
    service.start()
    assert finish(service)["state"] == "error"
    folder = service.root / "updates" / "1.2.3"
    assert (folder / first["name"]).read_bytes() == b"one"
    assert not list(folder.glob("*.part"))
    network[0][second["url"]] = b"two"
    service.start()
    assert finish(service)["state"] == "complete"
    assert network[1].count(first["url"]) == 1
    assert network[1].count(second["url"]) == 2


def test_corrupted_previously_downloaded_part_is_fetched_again(service, network):
    value = prepare(service, network)
    part = selected(value)["parts"][0]
    folder = service.root / "updates" / "1.2.3"
    folder.mkdir(parents=True)
    (folder / part["name"]).write_bytes(b"bad")
    service.check()
    service.start()
    assert finish(service)["state"] == "complete"
    assert part["url"] in network[1]


def test_cancel_removes_unverified_partial_and_allows_retry(service, network):
    value = prepare(service, network)
    first = selected(value)["parts"][0]

    def interrupted():
        yield b"o"
        service.cancel()
        yield b"ne"
    network[0][first["url"]] = interrupted
    service.check()
    service.start()
    assert finish(service)["state"] == "cancelled"
    assert not list((service.root / "updates" / "1.2.3").iterdir())
    network[0][first["url"]] = b"one"
    service.start()
    assert finish(service)["state"] == "complete"


def test_joined_hash_failure_never_creates_completed_archive(service, network):
    value = manifest()
    selected(value)["archive"]["sha256"] = digest(b"foobar")
    prepare(service, network, value)
    service.check()
    service.start()
    assert finish(service)["state"] == "error"
    folder = service.root / "updates" / "1.2.3"
    assert not (folder / selected(value)["archive"]["name"]).exists()
    assert len(list(folder.iterdir())) == 2, "verified parts should survive a failed join"


def test_disk_space_failure_precedes_asset_requests(service, network, monkeypatch):
    prepare(service, network)
    monkeypatch.setattr(updates.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    service.check()
    service.start()
    status = finish(service)
    assert status["state"] == "error" and "disk space" in status["message"]
    assert network[1] == [updates.MANIFEST_URL]


@pytest.mark.parametrize("mutation", ["url", "path", "part_order", "size", "hash", "variant", "tag", "date"])
def test_rejects_untrusted_or_inconsistent_manifest(service, mutation):
    value = manifest()
    variant = selected(value)
    if mutation == "url":
        variant["parts"][0]["url"] = "https://github.com/attacker/repo/releases/download/v1.2.3/asset"
    elif mutation == "path":
        variant["archive"]["name"] = "../launcher.exe"
    elif mutation == "part_order":
        variant["parts"].reverse()
    elif mutation == "size":
        variant["parts"][0]["size"] = 100
    elif mutation == "hash":
        variant["parts"][0]["sha256"] = "xyz"
    elif mutation == "variant":
        value["platform"]["windows"]["x86_64"]["variants"]["cpu"] = variant
    elif mutation == "tag":
        value["tag"] = "v2.0.0"
    else:
        value["pub_date"] = "2026-09-22"
    with pytest.raises(updates.UpdateError):
        updates.validate_manifest(value, service.build, service.current_version)


@pytest.mark.parametrize("url", ["http://github.com/x", "https://github.com.evil.test/x", "https://127.0.0.1/x",
                                  "https://github.com:444/x", "https://user:pass@github.com/x"])
def test_redirects_cannot_leave_https_github(url):
    with pytest.raises(updates.UpdateError):
        updates._network_url(url)


def test_missing_variant_does_not_select_a_different_cuda_backend(service):
    value = manifest()
    selected(value)["device"] = "cu130"
    result, chosen = updates.validate_manifest(value, service.build, service.current_version)
    assert not result["compatible"] and chosen is None


def test_start_requires_explicit_check_and_newer_release(service, network):
    with pytest.raises(updates.UpdateError, match="Check for"):
        service.start()
    prepare(service, network)
    service.current_version = "1.3.0"
    assert not service.check()["available"]
    with pytest.raises(updates.UpdateError, match="Check for"):
        service.start()


def test_oversized_manifest_does_not_become_downloadable(service, network):
    network[0][updates.MANIFEST_URL] = b" " * (updates.MAX_MANIFEST_BYTES + 1)
    with pytest.raises(updates.UpdateError, match="size limit"):
        service.check()
    with pytest.raises(updates.UpdateError):
        service.start()


def test_symlink_download_directory_cannot_escape_installation(service, network, tmp_path):
    outside = tmp_path / "other"
    outside.mkdir()
    try:
        (service.root / "updates").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks")
    prepare(service, network)
    service.check()
    service.start()
    assert finish(service)["state"] == "error"
    assert list(outside.iterdir()) == []


def client_for(service, host="127.0.0.1", base_url="http://127.0.0.1"):
    app = FastAPI()
    app.include_router(router)
    app.state.updates = service
    return TestClient(app, base_url=base_url, client=(host, 1234))


@pytest.mark.parametrize("method,path", [("GET", "status"), ("GET", "check"),
                                         ("POST", "download"), ("POST", "cancel")])
def test_source_update_routes_reject_requests_before_creating_a_service(network, monkeypatch, method, path):
    monkeypatch.setattr(updates.utils, "is_frozen", False)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1234))
    response = client.request(method, f"/api/updates/{path}",
                              headers={"X-Fox-Reader-Update": "1", "Origin": "http://127.0.0.1"})
    assert response.status_code == 403
    assert "running from source" in response.json()["error"]
    assert not hasattr(app.state, "updates")
    assert network[1] == []


@pytest.mark.parametrize("packaged", [False, True])
def test_settings_only_renders_updates_for_packaged_builds(monkeypatch, packaged):
    from fox_reader.routes import settings

    monkeypatch.setattr(updates.utils, "is_frozen", packaged)
    app = FastAPI()
    app.include_router(settings.router)
    response = TestClient(app).get("/settings")
    assert response.status_code == 200
    assert ('id="updatesPanel"' in response.text) is packaged
    assert ("/static/updates-entry.js" in response.text) is packaged
    assert "/static/desktop-entry.js" in response.text
    assert 'id="settings"' in response.text


def test_routes_reject_remote_and_cross_origin_mutations(service, network):
    prepare(service, network)
    client = client_for(service)
    assert client.get("/api/updates/status").status_code == 200
    assert network[1] == []
    assert client.post("/api/updates/download").status_code == 403
    headers = {"X-Fox-Reader-Update": "1", "Origin": "https://evil.test"}
    assert client.post("/api/updates/download", headers=headers).status_code == 403
    remote = client_for(service, host="192.0.2.10")
    assert remote.get("/api/updates/check").status_code == 403
    rebound = client_for(service, base_url="http://attacker.test")
    assert rebound.get("/api/updates/status").status_code == 403
    assert network[1] == []


def test_routes_check_and_explicit_download(service, network):
    prepare(service, network)
    client = client_for(service)
    headers = {"X-Fox-Reader-Update": "1", "Origin": "http://127.0.0.1"}
    assert client.post("/api/updates/download", headers=headers).status_code == 409
    assert client.get("/api/updates/check").status_code == 403
    assert client.get("/api/updates/check", headers=headers).json()["available"] is True
    assert client.post("/api/updates/download", headers=headers).status_code == 200
    assert finish(service)["state"] == "complete"


def test_repeat_checks_are_cached_to_avoid_hammering_github(service, network):
    prepare(service, network)
    first = service.check()
    assert service.check() == first
    assert network[1] == [updates.MANIFEST_URL]


@pytest.mark.parametrize("failure", [None, "checksum", "prepare", "handoff", "shutdown", "shutdown_exception"])
def test_installation_handoff_preserves_data_and_shuts_down_only_after_ready(service, network, monkeypatch, failure):
    value = prepare(service, network)
    if failure == "checksum":
        network[0][selected(value)["parts"][0]["url"]] = b"bad"
    calls = []
    staged = []

    def preparation(root, destination, archive, build, version, cancel):
        calls.append("prepare")
        assert destination.read_bytes() == b"onetwo", "split parts must be joined and verified before preparation"
        if failure == "prepare":
            raise updates.InstallError("preparation failed")
        transaction = root / (".foxreader-update-" + "a" * 32)
        transaction.mkdir()
        path = transaction / "archive.zip"
        destination.replace(path)
        prepared = SimpleNamespace(transaction=transaction, archive=path)
        staged.append(prepared)
        return prepared

    process = SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append("terminate"),
                              wait=lambda **_: calls.append("wait"))

    def handoff(prepared, platform):
        calls.append("handoff")
        assert service.status()["state"] == "handoff"
        service.cancel()
        assert not service._cancel.is_set(), "cancellation is locked after handing off"
        if failure == "handoff":
            raise updates.InstallError("handoff failed")
        calls.append("READY")
        return process

    def shutdown():
        calls.append("shutdown")
        assert calls[-2] == "READY"
        if failure == "shutdown_exception":
            raise RuntimeError("shutdown failed")
        return failure != "shutdown"

    monkeypatch.setattr(updates, "install_supported", lambda *_: True)
    monkeypatch.setattr(updates, "prepare_update", preparation)
    monkeypatch.setattr(updates, "launch_updater", handoff)
    service._shutdown = shutdown
    assert service.status()["install_supported"] is True
    service.check()
    service.start()
    status = finish(service)
    if failure is None:
        assert status["state"] == "installing" and status["archive"] is None
        assert calls == ["prepare", "handoff", "READY", "shutdown"]
        assert staged[0].archive.read_bytes() == b"onetwo"
        with pytest.raises(updates.UpdateError):
            service.start()
        with pytest.raises(updates.UpdateError):
            service.check()
    else:
        assert status["state"] == "error"
        if failure in {"checksum", "prepare", "handoff"}:
            assert "shutdown" not in calls
        else:
            assert calls[-2:] == ["terminate", "wait"]
        if failure != "checksum":
            assert (service.root / "updates" / "1.2.3" / selected(value)["archive"]["name"]).read_bytes() == b"onetwo"
        assert not list(service.root.glob(".foxreader-update-*"))
    assert json.loads((service.root / "build-info.json").read_text())["version"] == "1.2.2"


def test_cancellation_at_handoff_never_reports_acceptance_then_installs(service, network, monkeypatch):
    prepare(service, network)
    prepared = threading.Event()
    final_check = threading.Event()
    continue_handoff = threading.Event()
    cancelling = threading.Event()
    cancelled = threading.Event()
    responses = []
    check_cancelled = service._cancelled

    def preparation(*_):
        prepared.set()
        return SimpleNamespace()

    def cancellation_check():
        check_cancelled()
        if prepared.is_set():
            final_check.set()
            assert continue_handoff.wait(5)

    def cancel():
        cancelling.set()
        responses.append(service.cancel())
        cancelled.set()

    monkeypatch.setattr(updates, "install_supported", lambda *_: True)
    monkeypatch.setattr(updates, "prepare_update", preparation)
    monkeypatch.setattr(updates, "launch_updater", lambda *_: SimpleNamespace())
    monkeypatch.setattr(service, "_cancelled", cancellation_check)
    service._shutdown = lambda: True
    service.check()
    service.start()
    worker = threading.Thread(target=cancel)
    try:
        assert final_check.wait(5)
        worker.start()
        assert cancelling.wait(5)
        # Give the request a chance to reach cancel() while the update thread
        # is paused immediately after its final cancellation check.
        cancelled.wait(0.25)
    finally:
        continue_handoff.set()
        if worker.ident is not None:
            worker.join(timeout=5)
    assert finish(service)["state"] == "installing"
    assert len(responses) == 1
    assert responses[0]["state"] in {"handoff", "installing"}
    assert not service._cancel.is_set()
