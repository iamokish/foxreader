"""Updates download verified bytes without touching the working installation."""
import hashlib
import json
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
def service(tmp_path):
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


def test_status_does_not_check_network_and_source_build_does_not_guess(tmp_path, network):
    source = updates.UpdateService(tmp_path)
    assert source.status()["download_supported"] is False
    assert network[1] == []
    result, chosen = updates.validate_manifest(manifest(), None, "1.2.2")
    assert result["available"] and not result["compatible"] and chosen is None


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
