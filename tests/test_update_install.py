"""Release preparation is isolated, bounded and authenticated before shutdown."""
from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

from fox_reader.services import update_install as install


@pytest.fixture(params=["windows", "linux", "macos"])
def release(tmp_path, request):
    return make_release(tmp_path, request.param)


def make_release(tmp_path, platform):
    root = tmp_path / "Fox Reader"
    root.mkdir()
    build = {"version": "1.0.0", "platform": platform, "architecture": "arm64" if platform == "macos" else "x86_64",
             "device": "cpu", "gguf": None, "repository": "iamokish/foxreader"}
    new_build = {**build, "version": "1.1.0"}
    files = {"build-info.json": json.dumps(new_build).encode()}
    for name in ("LICENSE", "NOTICE", "TERMS.md", "FONT.md", "OFL.txt", "LICENSE-ComicMono", "LICENSE-material-icons"):
        files[f"legal/{name}"] = f"Legal text: {name}".encode()
    executable = ".exe" if platform == "windows" else ""
    files[f"bin/fox-reader{executable}"] = b"new backend"
    files[f"launcher{executable}"] = b"new launcher"
    files[f"foxreader-updater{executable}"] = b"new updater"
    if platform == "windows":
        files["legal/WebView2-LICENSE.txt"] = b"WebView license"
    elif platform == "linux":
        for name in ("libfox-window.so", "libfox-browser.so", "install-desktop.sh"):
            files[name] = b"native helper"
    else:
        files["Fox Reader.app/Contents/MacOS/launcher"] = b"Mac launcher"
        files["Fox Reader.app/Contents/Info.plist"] = b"Mac metadata"
    (root / f"foxreader-updater{executable}").write_bytes(b"trusted installed updater")
    for folder in ("config", "models", "fonts", "bin", "legal"):
        (root / folder).mkdir()
        (root / folder / "old").write_bytes(f"old {folder}".encode())
    return SimpleNamespace(root=root, build=build, files=files, platform=platform)


def archive_for(release, files=None, extra_zip=None):
    files = release.files if files is None else files
    extension = "zip" if release.platform == "windows" else "tar.gz"
    destination = release.root / "updates" / "1.1.0" / f"fox-reader-update.{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if extension == "zip":
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in files.items():
                archive.writestr(f"fox-reader/{name}", content)
            for name in ("models", "cache", "logs", "fonts"):
                archive.writestr(f"fox-reader/{name}/", b"")
            if extra_zip:
                extra_zip(archive)
    else:
        with tarfile.open(destination, "w:gz") as archive:
            for name, content in files.items():
                item = tarfile.TarInfo(f"fox-reader/{name}")
                item.size, item.mode = len(content), 0o755
                archive.addfile(item, io.BytesIO(content))
            for name in ("models", "cache", "logs", "fonts"):
                item = tarfile.TarInfo(f"fox-reader/{name}")
                item.type, item.mode = tarfile.DIRTYPE, 0o755
                archive.addfile(item)
    raw = destination.read_bytes()
    metadata = {"name": destination.name, "format": extension, "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}
    return destination, metadata


def prepare(release, path, metadata):
    return install.prepare_update(release.root, path, metadata, release.build, "1.1.0")


def assert_originals(release):
    for folder in ("config", "models", "fonts", "bin", "legal"):
        assert (release.root / folder / "old").read_bytes() == f"old {folder}".encode()


def test_prepare_checks_every_payload_and_preserves_installation(release):
    path, metadata = archive_for(release)
    original = path.read_bytes()
    prepared = prepare(release, path, metadata)
    assert_originals(release)
    assert prepared.archive.read_bytes() == original
    assert not path.exists()
    assert not (prepared.transaction / "payload" / "config").exists()
    assert not (prepared.transaction / "payload" / "models").exists()
    inventory = (prepared.transaction / "inventory.txt").read_bytes()
    assert hashlib.sha256(inventory).hexdigest() == prepared.inventory_sha256
    assert (prepared.transaction / "inventory.sha256").read_text().strip() == prepared.inventory_sha256
    runner = prepared.transaction / "runner" / install.updater_name(release.platform)
    assert runner.read_bytes() == b"trusted installed updater"
    for name, content in release.files.items():
        assert (prepared.transaction / "payload" / name).read_bytes() == content


def test_archive_changed_after_download_is_rejected(release):
    path, metadata = archive_for(release)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(install.InstallError, match="changed after verification"):
        prepare(release, path, metadata)
    assert_originals(release)
    assert not list(release.root.glob(".foxreader-update-*"))


@pytest.mark.parametrize("name", ["../escaped", "/absolute", "config/settings.yaml", "models/replace", "extra.txt"])
def test_unsafe_or_protected_archive_entries_never_reach_installation(release, name):
    path, metadata = archive_for(release, {**release.files, name: b"must not write"})
    with pytest.raises(install.InstallError):
        prepare(release, path, metadata)
    assert_originals(release)
    assert path.exists()
    assert not list(release.root.glob(".foxreader-update-*"))


@pytest.mark.parametrize("change", [{"version": "9.9.9"}, {"device": "cu130"}, {"no_window": True}])
def test_extracted_build_metadata_must_match_verified_selection(release, change):
    target = {**json.loads(release.files["build-info.json"]), **change}
    path, metadata = archive_for(release, {**release.files, "build-info.json": json.dumps(target).encode()})
    with pytest.raises(install.InstallError):
        prepare(release, path, metadata)
    assert_originals(release)


def test_missing_required_release_file_is_rejected(release):
    files = dict(release.files)
    files.pop("legal/NOTICE")
    path, metadata = archive_for(release, files)
    with pytest.raises((install.InstallError, OSError)):
        prepare(release, path, metadata)
    assert_originals(release)


def test_low_space_fails_before_extraction(release, monkeypatch):
    path, metadata = archive_for(release)
    monkeypatch.setattr(install.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(install.InstallError, match="GiB"):
        prepare(release, path, metadata)
    assert_originals(release)
    assert path.exists()
    assert not list(release.root.glob(".foxreader-update-*"))


def test_space_budget_includes_both_copies_of_packaged_fonts(release, monkeypatch):
    new_font = b"f" * 4096
    release.files["fonts/added.ttf"] = new_font
    path, metadata = archive_for(release)
    required = sum(map(len, release.files.values())) + len(b"old fonts") + len(new_font) + 64 * 1024**2
    monkeypatch.setattr(install.shutil, "disk_usage", lambda _: SimpleNamespace(free=required - 1))
    with pytest.raises(install.InstallError, match="GiB"):
        prepare(release, path, metadata)
    assert_originals(release)
    assert path.exists()


def test_space_budget_keeps_room_for_launcher_recovery(release, monkeypatch):
    name = "launcher.exe" if release.platform == "windows" else "launcher"
    (release.root / name).write_bytes(b"old launcher")
    extra = len(b"old launcher")
    if release.platform == "macos":
        app = release.root / "Fox Reader.app/Contents/MacOS"
        app.mkdir(parents=True)
        (app / "launcher").write_bytes(b"old app launcher")
        extra += 2 * len(b"old app launcher")
    path, metadata = archive_for(release)
    required = sum(map(len, release.files.values())) + len(b"old fonts") + extra + 64 * 1024**2
    monkeypatch.setattr(install.shutil, "disk_usage", lambda _: SimpleNamespace(free=required - 1))
    with pytest.raises(install.InstallError, match="GiB"):
        prepare(release, path, metadata)
    assert (release.root / name).read_bytes() == b"old launcher"
    assert path.exists()


def test_cancelled_preparation_keeps_verified_archive(release):
    path, metadata = archive_for(release)

    def cancel():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        install.prepare_update(release.root, path, metadata, release.build, "1.1.0", cancel)
    assert path.exists()
    assert_originals(release)


def test_interrupted_transaction_requires_launcher_recovery(release):
    path, metadata = archive_for(release)
    pending = release.root / (".foxreader-update-" + "a" * 32)
    pending.mkdir()
    (pending / "journal.bin").write_bytes(b"unfinished")
    with pytest.raises(install.InstallError, match="interrupted"):
        prepare(release, path, metadata)
    assert_originals(release)


def test_escaping_and_duplicate_zip_links_are_rejected_before_following(tmp_path):
    for spelling, target in (("bin/redirect", "../../outside"), ("bin/redirect", "/outside")):
        info = zipfile.ZipInfo("fox-reader/" + spelling)
        info.create_system, info.external_attr = 3, (stat.S_IFLNK | 0o777) << 16
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr(info, target)
        with zipfile.ZipFile(data) as archive, pytest.raises(install.InstallError, match="link"):
            install._members(archive, False)


def test_launch_requires_ready_acknowledgement_before_return(release, monkeypatch):
    path, metadata = archive_for(release)
    prepared = prepare(release, path, metadata)
    captured = {}

    class Input(io.BytesIO):
        def close(self):
            captured["ticket"] = self.getvalue()
            super().close()

    class Process:
        stdin = Input()
        stdout = io.BytesIO(b"READY\n")

    def spawn(command, **kwargs):
        captured.update(command=command, options=kwargs)
        return Process()

    monkeypatch.setattr(install.subprocess, "Popen", spawn)
    install.launch_updater(prepared, release.platform)
    fields = captured["ticket"].decode().splitlines()
    assert fields[0:2] == ["FOXREADER-UPDATER-1", "install"]
    assert fields[2] == str(release.root.resolve())
    assert fields[3] == prepared.transaction.name
    assert len(fields[4]) == 64
    assert fields[5] == str(os.getpid())
    assert fields[7] == prepared.inventory_sha256
    assert not (prepared.transaction / "ticket.token").exists()
    if os.name == "nt":
        assert captured["options"]["creationflags"] & install.subprocess.CREATE_BREAKAWAY_FROM_JOB
    else:
        assert captured["options"]["start_new_session"] is True


def test_failed_handoff_terminates_only_the_unready_helper(release, monkeypatch):
    path, metadata = archive_for(release)
    prepared = prepare(release, path, metadata)
    calls = []
    process = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(b"REJECTED\n"),
                              poll=lambda: None, terminate=lambda: calls.append("terminate"),
                              wait=lambda **_: calls.append("wait"))
    monkeypatch.setattr(install.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(install.InstallError, match="still running"):
        install.launch_updater(prepared, release.platform)
    assert calls == ["terminate", "wait"]
    assert_originals(release)


def link_archive(files, links):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        for name, value in files.items():
            archive.writestr("fox-reader/" + name, value)
        for name, target in links.items():
            item = zipfile.ZipInfo("fox-reader/" + name)
            item.create_system, item.external_attr = 3, (stat.S_IFLNK | 0o777) << 16
            archive.writestr(item, target)
    return zipfile.ZipFile(data)


def test_macos_framework_directory_symlink_chain_is_valid():
    prefix = "bin/Python.framework/"
    files = {prefix + "Versions/3.12/Python": b"Python", prefix + "Versions/3.12/Resources/Info.plist": b"plist"}
    links = {prefix + "Versions/Current": "3.12", prefix + "Python": "Versions/Current/Python",
             prefix + "Resources": "Versions/Current/Resources"}
    with link_archive(files, links) as archive:
        members = install._members(archive, False)
    assert sum(item.kind == "L" for item in members) == 3


@pytest.mark.parametrize("links", [
    {"bin/first": "second", "bin/second": "first"},
    {"bin/alias": "../legal", "bin/escape": "alias/../../outside"},
    {"bin/alias": "missing"},
])
def test_invalid_directory_link_chains_are_rejected_before_extraction(links):
    with link_archive({"legal/LICENSE": b"license"}, links) as archive:
        with pytest.raises(install.InstallError, match="link"):
            install._members(archive, False)


def test_materialized_hardlinks_count_towards_unpacked_limit(monkeypatch):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        item = tarfile.TarInfo("fox-reader/bin/data")
        item.size = 8
        archive.addfile(item, io.BytesIO(b"12345678"))
        item = tarfile.TarInfo("fox-reader/bin/copy")
        item.type, item.linkname = tarfile.LNKTYPE, "fox-reader/bin/data"
        archive.addfile(item)
    monkeypatch.setattr(install, "MAX_UNPACKED", 15)
    data.seek(0)
    with tarfile.open(fileobj=data, mode="r") as archive:
        with pytest.raises(install.InstallError, match="size limit"):
            install._members(archive, False)


def test_implicit_parents_count_towards_entry_limit(monkeypatch):
    monkeypatch.setattr(install, "MAX_ENTRIES", 2)
    with link_archive({"bin/deep/directory/data": b"data"}, {}) as archive:
        with pytest.raises(install.InstallError, match="too many entries"):
            install._members(archive, False)


def test_source_and_missing_helper_builds_never_offer_installation(release, monkeypatch):
    monkeypatch.setattr(install.sys, "frozen", False, raising=False)
    assert not install.install_supported(release.root, release.build)
    monkeypatch.setattr(install.sys, "frozen", True, raising=False)
    executable = "fox-reader.exe" if release.platform == "windows" else "fox-reader"
    monkeypatch.setattr(install.sys, "executable", str(release.root / "bin" / executable))
    assert install.install_supported(release.root, release.build)
    (release.root / install.updater_name(release.platform)).unlink()
    assert not install.install_supported(release.root, release.build)
