"""Exercise the real C++ transaction core against disposable installations."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = "launcher.exe" if os.name == "nt" else "launcher"
UPDATER = "foxreader-updater.exe" if os.name == "nt" else "foxreader-updater"


@pytest.fixture(scope="module")
def core(tmp_path_factory):
    directory = tmp_path_factory.mktemp("updater-native")
    output = directory / ("core.exe" if os.name == "nt" else "core")
    spec = importlib.util.spec_from_file_location("fox_updater_toolchain", REPO / "packaging/toolchain.py")
    toolchain = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(REPO / "packaging"))
        imports.setitem(sys.modules, spec.name, toolchain)
        spec.loader.exec_module(toolchain)
        try:
            compiler = toolchain.find_compiler(allow_download=False)
        except toolchain.ToolchainError as exc:
            pytest.skip(f"transaction tests require a C++17 compiler: {exc}")
        sources = [str(REPO / "tests/updater_core.cpp"), str(REPO / "packaging/updater/core.cpp")]
        if compiler.kind == "msvc":
            command = [str(compiler.path), "/nologo", "/std:c++17", "/EHsc", "/utf-8", "/MT",
                       "/DFOX_UPDATER_TESTING", *sources, f"/Fe:{output}", f"/Fo:{directory}\\"]
        else:
            candidate = compiler.path.with_name("g++.exe" if os.name == "nt" else "g++")
            if "clang" in compiler.path.name:
                candidate = compiler.path.with_name("clang++.exe" if os.name == "nt" else "clang++")
            cxx = str(candidate) if candidate.is_file() else shutil.which("c++")
            if not cxx:
                pytest.skip("transaction tests require a C++17 compiler")
            command = [cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-DFOX_UPDATER_TESTING",
                       *sources, "-o", str(output)]
            if os.name == "nt":
                command += ["-static"]
        built = subprocess.run(command, cwd=directory, env=compiler.env, capture_output=True, text=True, timeout=120)
        assert built.returncode == 0, built.stdout + built.stderr
    return output


def write(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def manifest(transaction):
    payload = transaction / "payload"
    lines = ["FOXREADER-UPDATER-INVENTORY 1"]
    for path in sorted(payload.rglob("*")):
        name = path.relative_to(payload).as_posix().encode("utf-8").hex()
        if path.is_symlink():
            lines.append(f"L\t{os.readlink(path).encode('utf-8').hex()}\t{name}")
        elif path.is_dir():
            lines.append(f"D\t{name}")
        else:
            raw = path.read_bytes()
            lines.append(f"F\t{len(raw)}\t{hashlib.sha256(raw).hexdigest()}\t{name}")
    raw = ("\n".join(lines) + "\n").encode()
    (transaction / "inventory.txt").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def installation(tmp_path):
    root = tmp_path / "Fox Reader installation"
    root.mkdir()
    for name, data in {
        "bin/old-only.dll": b"old", "bin/common.dll": b"original library",
        "legal/old-notice": b"old notice", "icons/old.ico": b"old icon",
        LAUNCHER: b"old launcher", UPDATER: b"old updater", "build-info.json": b'{"version":"old"}',
        "fonts/custom.ttf": b"custom font", "fonts/shared.ttf": b"old shared font",
        "fonts/sub/old.ttf": b"old nested font", "config/user_config.yaml": b"theme: light\n",
        "models/weights.bin": b"weights", "cache/preview": b"preview", "logs/log": b"log",
        "updates/other-download": b"other", "personal-note.txt": b"personal",
    }.items():
        write(root, name, data)
    transaction = root / (".foxreader-update-" + "a" * 32)
    payload = transaction / "payload"
    for name, data in {
        "bin/common.dll": b"new library", "bin/new-only.dll": b"new",
        "legal/LICENSE": b"new license", "icons/new.ico": b"new icon",
        LAUNCHER: b"new launcher", UPDATER: b"new updater", "build-info.json": b'{"version":"new"}',
        "fonts/shared.ttf": b"new shared font", "fonts/sub/new.ttf": b"new nested font",
    }.items():
        write(payload, name, data)
    if sys.platform == "darwin":
        for directory, version in ((root, "old"), (payload, "new")):
            write(directory, "Fox Reader.app/Contents/MacOS/launcher", f"{version} app launcher".encode())
            write(directory, "Fox Reader.app/Contents/Info.plist", f"{version} app metadata".encode())
    write(transaction, "runner/helper", b"runner")
    write(transaction, "archive.zip", b"verified archive")
    return root, transaction, manifest(transaction)


def run(core, installation, mode="apply", event=""):
    root, transaction, digest = installation
    result = subprocess.run([str(core), mode, str(root).encode().hex(), str(transaction).encode().hex(), digest, event],
                            capture_output=True, text=True, timeout=20)
    if "crash" in mode:
        assert result.returncode == 73, result.stdout + result.stderr
        return None
    assert result.returncode == 0, result.stdout + result.stderr
    flags, _, message = result.stdout.partition("\n")
    return tuple(map(int, flags.split())), message


def snapshot(root, *, exclude_transactions=True):
    result = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if exclude_transactions and relative.parts[0].startswith(".foxreader-update-"):
            continue
        if path.is_symlink():
            result[relative.as_posix()] = ("symlink", os.readlink(path))
        elif path.is_file():
            result[relative.as_posix()] = path.read_bytes()
        else:
            result[relative.as_posix()] = "directory"
    return result


def test_install_replaces_managed_trees_and_merges_fonts(core, installation):
    root, transaction, _ = installation
    before = snapshot(root)
    flags, message = run(core, installation)
    assert flags[:4] == (1, 0, 0, 0), message
    assert not (root / "bin/old-only.dll").exists()
    assert not (root / "legal/old-notice").exists()
    assert not (root / "icons/old.ico").exists()
    assert (root / "bin/common.dll").read_bytes() == b"new library"
    assert (root / "build-info.json").read_bytes() == b'{"version":"new"}'
    assert (transaction / "old/build-info.json").read_bytes() == before["build-info.json"]
    assert (root / "fonts/shared.ttf").read_bytes() == b"new shared font"
    assert (root / "fonts/custom.ttf").read_bytes() == b"custom font"
    assert (root / "fonts/sub/old.ttf").read_bytes() == b"old nested font"
    assert (root / "fonts/sub/new.ttf").read_bytes() == b"new nested font"
    for name in ("config/user_config.yaml", "models/weights.bin", "cache/preview", "logs/log",
                 "updates/other-download", "personal-note.txt"):
        assert (root / name).read_bytes() == before[name]
    assert (transaction / "state").read_bytes() == b"COMMITTED\n"
    installed = snapshot(root)
    assert run(core, installation, "recover")[0][0] == 1
    assert snapshot(root) == installed
    assert run(core, installation, "cleanup")[0][0] == 1
    assert not transaction.exists()
    assert snapshot(root) == installed


@pytest.mark.parametrize("change", ["data", "extra", "missing", "inventory", "protected", "absolute_link"])
def test_bad_payload_never_changes_installation(core, installation, change):
    root, transaction, digest = installation
    before = snapshot(root)
    payload = transaction / "payload"
    if change == "data":
        (payload / "bin/common.dll").write_bytes(b"different payload")
    elif change == "extra":
        write(payload, "bin/unlisted", b"unlisted")
    elif change == "missing":
        (payload / "legal/LICENSE").unlink()
    elif change == "inventory":
        with (transaction / "inventory.txt").open("ab") as stream:
            stream.write(b"tampered\n")
    elif change == "protected":
        write(payload, "config/user_config.yaml", b"erased")
        digest = manifest(transaction)
    else:
        try:
            (payload / "bin/escape").symlink_to(root / "models")
        except OSError:
            pytest.skip("symlink creation unavailable")
        digest = manifest(transaction)
    flags, message = run(core, (root, transaction, digest))
    assert not any(flags[:4]), message
    assert snapshot(root) == before
    assert not (transaction / "journal.bin").exists()


LAUNCH_EVENTS = [f"{stage}:{LAUNCHER}" for stage in (
    "before:copy", "during:backup", "after:copy", "before:backup", "after:backup",
    "before:install", "after:install",
)]
APP_EVENTS = [f"{stage}:Fox Reader.app" for stage in (
    "before:copy", "during:backup", "after:copy", "before:backup", "after:backup",
    "before:install", "after:install",
)] if sys.platform == "darwin" else []
EVENTS = ["journal", "before:backup:bin", "after:backup:bin", "after:install:bin",
          "after:backup:legal", "after:install:legal", *LAUNCH_EVENTS, *APP_EVENTS,
          "after:install:build-info.json", "after:backup:fonts", "after:install:fonts"]


@pytest.mark.parametrize("event", EVENTS)
def test_failure_rolls_back_every_changed_component(core, installation, event):
    root, transaction, _ = installation
    before = snapshot(root)
    flags, message = run(core, installation, "fail", event)
    assert flags[:4] == (0, 0, 1, 0), message
    assert (root / "build-info.json").read_bytes() == b'{"version":"old"}'
    assert snapshot(root) == before
    assert (transaction / "state").read_bytes() == b"ROLLED_BACK\n"
    assert run(core, installation, "recover")[0][2] == 1
    assert snapshot(root) == before


@pytest.mark.parametrize("event", EVENTS)
def test_crash_can_be_recovered_idempotently(core, installation, event):
    root, transaction, _ = installation
    before = snapshot(root)
    run(core, installation, "crash", event)
    if event == "after:install:build-info.json":
        assert (root / "build-info.json").read_bytes() == b'{"version":"new"}'
        assert (transaction / "old/build-info.json").read_bytes() == before["build-info.json"]
    assert (root / LAUNCHER).read_bytes() in (before[LAUNCHER], b"new launcher")
    if sys.platform == "darwin":
        assert (root / "Fox Reader.app/Contents/MacOS/launcher").read_bytes() in (
            b"old app launcher", b"new app launcher",
        )
    flags, _ = run(core, installation, "cleanup")
    assert flags[3] == 1
    assert transaction.exists()
    flags, message = run(core, installation, "recover")
    assert flags[:4] == (0, 0, 1, 0), message
    assert (root / "build-info.json").read_bytes() == b'{"version":"old"}'
    assert snapshot(root) == before
    assert run(core, installation, "recover")[0][2] == 1
    assert snapshot(root) == before


def test_cancel_has_no_installation_side_effects(core, installation):
    root, transaction, _ = installation
    before = snapshot(root)
    flags, message = run(core, installation, "cancel")
    assert flags[:4] == (0, 1, 0, 0), message
    assert snapshot(root) == before
    assert not (transaction / "journal.bin").exists()


def test_partial_launcher_backup_never_replaces_live_launcher(core, installation):
    root, transaction, _ = installation
    original = b"complete original launcher" * 16000
    (root / LAUNCHER).write_bytes(original)
    before = snapshot(root)
    run(core, installation, "crash", f"during:backup:{LAUNCHER}")
    assert (root / LAUNCHER).read_bytes() == original
    assert not (transaction / "old" / LAUNCHER).exists()
    partial = transaction / "old" / f"{LAUNCHER}.tmp"
    assert partial.is_file() and partial.stat().st_size < len(original)
    assert run(core, installation, "recover")[0][2] == 1
    assert snapshot(root) == before


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS atomic bundle exchange")
@pytest.mark.parametrize("stage", ["before", "after"])
def test_bundle_exchange_capability_is_checked_before_mutation(core, installation, stage):
    root, transaction, _ = installation
    before = snapshot(root)
    flags, message = run(core, installation, "fail", f"{stage}:check:bundle-exchange")
    assert not any(flags[:4]), message
    assert snapshot(root) == before
    assert not (transaction / "journal.bin").exists()
    assert run(core, installation, "cleanup")[0][0] == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable permissions")
def test_launcher_backup_preserves_executable_permissions(core, installation):
    root, transaction, _ = installation
    (root / LAUNCHER).chmod(0o751)
    (transaction / "payload" / LAUNCHER).chmod(0o755)
    run(core, installation, "crash", f"after:install:{LAUNCHER}")
    assert (root / LAUNCHER).stat().st_mode & 0o777 == 0o755
    assert (transaction / "old" / LAUNCHER).stat().st_mode & 0o777 == 0o751
    assert run(core, installation, "recover")[0][2] == 1
    assert (root / LAUNCHER).read_bytes() == b"old launcher"
    assert (root / LAUNCHER).stat().st_mode & 0o777 == 0o751


def test_wrong_transaction_location_is_refused(core, installation):
    root, transaction, digest = installation
    wrong = root / "my-personal-folder"
    transaction.rename(wrong)
    before = snapshot(root)
    flags, message = run(core, (root, wrong, digest))
    assert not any(flags[:4]), message
    assert snapshot(root) == before


def test_new_components_are_removed_on_rollback(core, installation):
    root, transaction, _ = installation
    (root / UPDATER).unlink()
    before = snapshot(root)
    flags, message = run(core, installation, "fail", "after:install:fonts")
    assert flags[2] == 1, message
    assert snapshot(root) == before


def test_commit_crash_keeps_successful_update(core, installation):
    root, transaction, _ = installation
    run(core, installation, "crash", "committed")
    installed = snapshot(root)
    assert (root / "bin/common.dll").read_bytes() == b"new library"
    assert run(core, installation, "recover")[0][0] == 1
    assert snapshot(root) == installed


def test_fonts_overlay_does_not_follow_old_external_symlink(core, installation, tmp_path):
    root, transaction, _ = installation
    outside = tmp_path / "private fonts"
    outside.mkdir()
    (outside / "keep.ttf").write_bytes(b"external unchanged")
    shutil.rmtree(root / "fonts/sub")
    try:
        (root / "fonts/sub").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    flags, message = run(core, installation)
    assert flags[0] == 1, message
    assert not (root / "fonts/sub").is_symlink()
    assert (outside / "keep.ttf").read_bytes() == b"external unchanged"
    assert list(outside.iterdir()) == [outside / "keep.ttf"]


RESTORE_EVENTS = ["after:discard:fonts", "after:restore:fonts", f"before:restore:{LAUNCHER}",
                  f"after:restore:{LAUNCHER}", "after:discard:bin", "after:restore:bin", "rolled-back"]
if sys.platform == "darwin":
    RESTORE_EVENTS += [f"{stage}:Fox Reader.app" for stage in (
        "before:restore-copy", "during:restore-copy", "after:restore-copy", "before:restore", "after:restore",
    )]


@pytest.mark.parametrize("event", RESTORE_EVENTS)
def test_recovery_survives_another_crash(core, installation, event):
    root, _, _ = installation
    before = snapshot(root)
    run(core, installation, "crash", "after:install:fonts")
    run(core, installation, "recover-crash", event)
    assert (root / LAUNCHER).read_bytes() in (b"old launcher", b"new launcher")
    if sys.platform == "darwin":
        assert (root / "Fox Reader.app/Contents/MacOS/launcher").read_bytes() in (
            b"old app launcher", b"new app launcher",
        )
    flags, message = run(core, installation, "recover")
    assert flags[2] == 1 and flags[3] == 0, message
    assert snapshot(root) == before


def test_failed_rollback_preserves_journal_for_retry(core, installation):
    root, transaction, _ = installation
    before = snapshot(root)
    run(core, installation, "crash", "after:install:fonts")
    flags, message = run(core, installation, "recover-fail", "before:restore:fonts")
    assert flags[3] == 1, message
    assert (transaction / "journal.bin").is_file()
    assert run(core, installation, "cleanup")[0][3] == 1
    assert run(core, installation, "recover")[0][2] == 1
    assert snapshot(root) == before


def test_unicode_installation_and_payload_paths(core, installation):
    root, transaction, _ = installation
    renamed = root.with_name("Fox 読者 café")
    root.rename(renamed)
    transaction = renamed / transaction.name
    write(transaction / "payload", "bin/日本語.bin", b"unicode file")
    write(transaction / "payload", "fonts/新しい.ttf", b"unicode font")
    flags, message = run(core, (renamed, transaction, manifest(transaction)))
    assert flags[0] == 1, message
    assert (renamed / "bin/日本語.bin").read_bytes() == b"unicode file"
    assert (renamed / "fonts/新しい.ttf").read_bytes() == b"unicode font"


def test_corrupt_journal_refuses_cleanup_and_recovery(core, installation):
    root, transaction, _ = installation
    before = snapshot(root)
    (transaction / "journal.bin").write_bytes(b"corrupt")
    assert run(core, installation, "recover")[0][3] == 1
    assert run(core, installation, "cleanup")[0][3] == 1
    assert snapshot(root) == before
    assert (transaction / "journal.bin").read_bytes() == b"corrupt"


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing locks")
@pytest.mark.parametrize("installed", [False, True])
def test_cleanup_retains_terminal_marker_while_runner_is_locked(core, installation, installed):
    import ctypes
    root, transaction, _ = installation
    if installed:
        assert run(core, installation)[0][0] == 1
    before = snapshot(root)
    create = ctypes.windll.kernel32.CreateFileW
    create.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p,
                       ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
    create.restype = ctypes.c_void_p
    close = ctypes.windll.kernel32.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    handle = create(str(transaction / "runner/helper"), 0x80000000, 1, None, 3, 0x80, None)
    assert handle not in (None, ctypes.c_void_p(-1).value)
    try:
        flags, message = run(core, installation, "cleanup")
        assert flags[4] == 1 and flags[0] == 0, message
        assert (transaction / "state").read_bytes() == (b"COMMITTED\n" if installed else b"ROLLED_BACK\n")
        assert (transaction / "journal.bin").is_file() == installed
        assert not (transaction / "payload").exists()
        assert not (transaction / "old").exists()
        assert not (transaction / "archive.zip").exists()
        assert snapshot(root) == before
        recovered, message = run(core, installation, "recover")
        assert recovered[:4] == ((1, 0, 0, 0) if installed else (0, 0, 1, 0)), message
    finally:
        close(handle)
    assert run(core, installation, "cleanup")[0][0] == 1
    assert not transaction.exists()


@pytest.mark.parametrize("length", [0, 55, 56, 63, 64, 65, 127, 128, 131071, 131072, 131073])
def test_sha256_padding_and_stream_boundaries(core, installation, length):
    root, transaction, _ = installation
    write(transaction / "payload", "bin/hash-boundary", (bytes(range(256)) * ((length + 255) // 256))[:length])
    flags, message = run(core, (root, transaction, manifest(transaction)))
    assert flags[0] == 1, message
    assert (root / "bin/hash-boundary").stat().st_size == length


def test_chained_relative_symlink_escape_is_rejected(core, installation):
    root, transaction, _ = installation
    payload = transaction / "payload"
    write(transaction, "outside", b"must not become a packaged link")
    try:
        (payload / "bin/alias").symlink_to("../legal", target_is_directory=True)
        (payload / "bin/escape").symlink_to("alias/../../outside")
    except OSError:
        pytest.skip("symlink creation unavailable")
    before = snapshot(root)
    flags, message = run(core, (root, transaction, manifest(transaction)))
    assert not any(flags[:4]), message
    assert snapshot(root) == before
    assert not (transaction / "journal.bin").exists()


@pytest.mark.parametrize("fail", [False, True])
def test_python_prepared_release_installs_and_rolls_back_in_native_core(core, tmp_path, fail):
    from tests.test_update_install import archive_for, make_release, prepare

    platform = "windows" if os.name == "nt" else "macos" if sys.platform == "darwin" else "linux"
    release = make_release(tmp_path, platform)
    (release.root / "build-info.json").write_text(json.dumps(release.build), encoding="utf-8")
    release.files["fonts/added.ttf"] = b"new packaged font"
    release.files["fonts/old"] = b"replacement packaged font"
    (release.root / "fonts/custom.ttf").write_bytes(b"user font")
    path, metadata = archive_for(release)
    prepared = prepare(release, path, metadata)
    before = snapshot(release.root)
    installation = (release.root, prepared.transaction, prepared.inventory_sha256)
    if fail:
        flags, message = run(core, installation, "fail", "after:install:fonts")
        assert flags[2] == 1 and flags[3] == 0, message
        assert json.loads((release.root / "build-info.json").read_bytes()) == release.build
        assert snapshot(release.root) == before
    else:
        flags, message = run(core, installation)
        assert flags[0] == 1, message
        assert json.loads((release.root / "build-info.json").read_bytes()) == {**release.build, "version": "1.1.0"}
        for name, data in release.files.items():
            assert (release.root / name).read_bytes() == data
        assert not (release.root / "bin/old").exists()
        assert not (release.root / "legal/old").exists()
        assert (release.root / "fonts/custom.ttf").read_bytes() == b"user font"
        for name in ("config", "models"):
            assert (release.root / name / "old").read_bytes() == f"old {name}".encode()
    assert run(core, installation, "cleanup")[0][0] == 1
    assert not prepared.transaction.exists()
