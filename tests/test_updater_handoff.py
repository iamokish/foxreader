"""Authenticated native handoff, real process waits, cleanup, and Windows GUI."""
from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.test_updater_core import manifest, write

ROOT = Path(__file__).resolve().parents[1]
UPDATER = "foxreader-updater.exe" if os.name == "nt" else "foxreader-updater"
BACKEND = "fox-reader.exe" if os.name == "nt" else "fox-reader"
LAUNCHER = "launcher.exe" if os.name == "nt" else "launcher"


def wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for native updater evidence")


@pytest.fixture(scope="module")
def native_binaries(tmp_path_factory):
    directory = tmp_path_factory.mktemp("updater-handoff-build")
    spec = importlib.util.spec_from_file_location("fox_handoff_toolchain", ROOT / "packaging/toolchain.py")
    toolchain = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(ROOT / "packaging"))
        imports.setitem(sys.modules, spec.name, toolchain)
        spec.loader.exec_module(toolchain)
        compiler = toolchain.find_compiler(allow_download=False)
        paths = {}
        for name, sources in {
            "parent": [ROOT / "tests/updater_parent.cpp"],
            "handoff": [ROOT / "tests/updater_handoff.cpp", ROOT / "packaging/updater/handoff.cpp",
                        ROOT / "packaging/updater/core.cpp"],
        }.items():
            output = directory / (name + (".exe" if os.name == "nt" else ""))
            if compiler.kind == "msvc":
                command = [str(compiler.path), "/nologo", "/std:c++17", "/EHsc", "/utf-8", "/MT",
                           *map(str, sources), f"/Fe{output}", f"/Fo{directory}\\", "/link", "bcrypt.lib"]
            else:
                driver = compiler.path.with_name("clang++" if "clang" in compiler.path.name else "g++")
                command = [str(driver), "-std=c++17", "-Wall", "-Wextra", "-pthread", *map(str, sources),
                           "-o", str(output)]
                if sys.platform == "darwin":
                    command += ["-lproc"]
            compiled = subprocess.run(command, cwd=directory, env=compiler.env, capture_output=True,
                                      text=True, timeout=120)
            assert compiled.returncode == 0, compiled.stdout + compiled.stderr
            paths[name] = output
        if os.name == "nt":
            paths["gui"] = toolchain.build_updater(directory / UPDATER, compiler)
    return paths


@pytest.fixture
def prepared(tmp_path, native_binaries):
    def create(kind="handoff"):
        root = tmp_path / "Fox Reader installation"
        transaction = root / (".foxreader-update-" + "a" * 32)
        payload = transaction / "payload"
        write(root, "config/user_config.yaml", b"theme: light\n")
        write(root, "models/weights", b"weights")
        write(root, LAUNCHER, b"old launcher")
        write(root, UPDATER, native_binaries[kind].read_bytes()).chmod(0o755)
        write(root, "build-info.json", b'{"version":"old"}')
        write(root, f"bin/{BACKEND}", native_binaries["parent"].read_bytes()).chmod(0o755)
        write(transaction, f"runner/{UPDATER}", native_binaries[kind].read_bytes()).chmod(0o755)
        write(payload, f"bin/{BACKEND}", b"new backend")
        write(payload, "legal/LICENSE", b"license fixture")
        write(payload, "icons/icon.png", b"icon fixture")
        write(payload, LAUNCHER, b"new launcher")
        write(payload, UPDATER, native_binaries[kind].read_bytes()).chmod(0o755)
        write(payload, "build-info.json", b'{"version":"new"}')
        write(transaction, "archive.zip", b"archive")
        digest = manifest(transaction)
        return root, transaction, digest
    return create


def start_parent(prepared, *, mode="good", kind="handoff"):
    root, transaction, digest = prepared(kind)
    process = subprocess.Popen([str(root / "bin" / BACKEND), str(root), transaction.name, digest, mode],
                               cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wait_for(lambda: (root / "handoff-response").exists())
    return root, transaction, process


def test_handoff_waits_for_parent_then_installs_and_cleans_up(prepared):
    root, transaction, parent = start_parent(prepared)
    try:
        assert (root / "handoff-response").read_text() == "READY\n"
        assert not (transaction / "journal.bin").exists()
        assert (root / "build-info.json").read_bytes() == b'{"version":"old"}'
        assert parent.poll() is None
        (root / "exit-parent").touch()
        assert parent.wait(timeout=10) == 0
        wait_for(lambda: (root / "updater-result").exists())
        assert (root / "updater-result").read_text().startswith("1 0 0 0\n")
        wait_for(lambda: not transaction.exists())
        assert (root / "bin" / BACKEND).read_bytes() == b"new backend"
        assert (root / "config/user_config.yaml").read_bytes() == b"theme: light\n"
        assert (root / "models/weights").read_bytes() == b"weights"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)


@pytest.mark.parametrize("mode", ["bad-token", "bad-parent"])
def test_handoff_refuses_forged_ticket_without_mutation(prepared, mode):
    root, transaction, parent = start_parent(prepared, mode=mode)
    assert parent.wait(timeout=10) == 6
    assert (root / "handoff-response").read_text() == "ERROR\n"
    assert not (transaction / "journal.bin").exists()
    assert (root / "build-info.json").read_bytes() == b'{"version":"old"}'


def test_cancel_during_process_wait_cleans_without_mutating_installation(prepared):
    root, transaction, parent = start_parent(prepared)
    try:
        (root / "cancel-request").touch()
        wait_for(lambda: (root / "updater-result").exists())
        assert (root / "updater-result").read_text().startswith("0 1 0 0\n")
        assert not (transaction / "journal.bin").exists()
        assert (root / "build-info.json").read_bytes() == b'{"version":"old"}'
        wait_for(lambda: not transaction.exists())
        assert parent.poll() is None
        assert (root / LAUNCHER).read_bytes() == b"old launcher"
        assert (root / "config/user_config.yaml").read_bytes() == b"theme: light\n"
        assert (root / "models/weights").read_bytes() == b"weights"
    finally:
        (root / "exit-parent").touch()
        parent.wait(timeout=10)


def test_verification_failure_cleans_without_mutating_installation(prepared):
    root, transaction, parent = start_parent(prepared)
    try:
        write(transaction, f"payload/bin/{BACKEND}", b"corrupted download")
        (root / "exit-parent").touch()
        assert parent.wait(timeout=10) == 0
        wait_for(lambda: (root / "updater-result").exists())
        assert (root / "updater-result").read_text().startswith("0 0 0 0\n")
        wait_for(lambda: not transaction.exists())
        assert (root / "build-info.json").read_bytes() == b'{"version":"old"}'
        assert (root / LAUNCHER).read_bytes() == b"old launcher"
        assert (root / "config/user_config.yaml").read_bytes() == b"theme: light\n"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)


def test_cancel_with_interrupted_transaction_refuses_cleanup(prepared):
    root, transaction, parent = start_parent(prepared)
    try:
        write(transaction, "journal.bin", b"interrupted transaction fixture")
        (root / "probe-cleanup").touch()
        (root / "cancel-request").touch()
        wait_for(lambda: (root / "cleanup-started").exists())
        assert (root / "updater-result").read_text().startswith("0 0 0 1\n")
        assert (root / "cleanup-started").read_text() == "0"
        assert (transaction / "journal.bin").read_bytes() == b"interrupted transaction fixture"
        assert (transaction / "archive.zip").read_bytes() == b"archive"
        assert (transaction / "payload" / "bin" / BACKEND).read_bytes() == b"new backend"
        assert not (transaction / "ticket.token").exists()
        assert (root / "build-info.json").read_bytes() == b'{"version":"old"}'
    finally:
        (root / "exit-parent").touch()
        parent.wait(timeout=10)


@pytest.mark.parametrize("relative", [f"bin/{'worker.exe' if os.name == 'nt' else 'worker'}", LAUNCHER])
def test_waits_for_remaining_installation_workers_and_controllers(prepared, native_binaries, relative):
    root, transaction, digest = prepared()
    extra = root / relative
    shutil.copy2(native_binaries["parent"], extra)
    extra.chmod(0o755)
    release = root / "release-worker"
    worker = subprocess.Popen([str(extra), "--hold", str(release)], cwd=root)
    parent = subprocess.Popen([str(root / "bin" / BACKEND), str(root), transaction.name, digest, "good"], cwd=root)
    try:
        wait_for(lambda: (root / "handoff-response").exists())
        assert (root / "handoff-response").read_text() == "READY\n"
        (root / "exit-parent").touch()
        assert parent.wait(timeout=10) == 0
        time.sleep(0.3)
        assert worker.poll() is None
        assert not (transaction / "journal.bin").exists()
        assert (root / "build-info.json").read_bytes() == b'{"version":"old"}'
        release.touch()
        assert worker.wait(timeout=10) == 0
        wait_for(lambda: not transaction.exists())
        assert (root / "bin" / BACKEND).read_bytes() == b"new backend"
    finally:
        release.touch()
        for process in (parent, worker):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_launcher_can_clean_terminal_transaction_without_exiting(prepared, native_binaries):
    root, transaction, _ = prepared()
    shutil.copy2(native_binaries["parent"], root / LAUNCHER)
    shutil.copy2(native_binaries["handoff"], root / UPDATER)
    (root / LAUNCHER).chmod(0o755)
    (root / UPDATER).chmod(0o755)
    write(transaction, "state", b"COMMITTED\n")
    launcher = subprocess.Popen([str(root / LAUNCHER), str(root), transaction.name, "0" * 64, "cleanup"], cwd=root)
    try:
        wait_for(lambda: (root / "handoff-response").exists())
        assert (root / "handoff-response").read_text() == "READY\n"
        wait_for(lambda: not transaction.exists())
        assert launcher.poll() is None
        assert (root / "build-info.json").read_bytes() == b'{"version":"old"}'
    finally:
        (root / "exit-parent").touch()
        assert launcher.wait(timeout=10) == 0


def updater_window(pid):
    user = ctypes.windll.user32
    found = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def visit(hwnd, _):
        owner = ctypes.c_ulong()
        user.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(owner))
        name = ctypes.create_unicode_buffer(128)
        user.GetClassNameW(ctypes.c_void_p(hwnd), name, 128)
        if owner.value == pid and name.value == "FoxReaderUpdater":
            found.append(hwnd)
        return True

    user.EnumWindows(callback_type(visit), 0)
    return found[0] if found else None


def close_button_text(window):
    user = ctypes.windll.user32
    user.GetDlgItem.restype = ctypes.c_void_p
    button = user.GetDlgItem(ctypes.c_void_p(window), 1)
    text = ctypes.create_unicode_buffer(128)
    user.GetWindowTextW(ctypes.c_void_p(button), text, len(text))
    return text.value


@pytest.mark.skipif(os.name != "nt", reason="Win32 native control automation")
def test_manual_updater_launch_exits_without_opening_window(native_binaries):
    process = subprocess.Popen([str(native_binaries["gui"])], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = process.communicate(timeout=10)
    assert process.returncode != 0
    assert b"Start updates from Settings" in errors
    assert not updater_window(process.pid)


@pytest.mark.skipif(os.name != "nt", reason="Win32 native control automation")
def test_windows_gui_waits_then_success_close_cleans_runner(prepared):
    root, transaction, parent = start_parent(prepared, kind="gui")
    pid = int((root / "updater-pid").read_text())
    window = wait_for(lambda: updater_window(pid))
    try:
        assert (root / "handoff-response").read_text() == "READY\n"
        assert not (transaction / "journal.bin").exists()
        (root / "exit-parent").touch()
        assert parent.wait(timeout=10) == 0
        # The marker is atomically published with a brief Windows write lock.
        # Wait for the GUI's finished state before reading durable evidence.
        wait_for(lambda: close_button_text(window) == "Close")
        assert (transaction / "state").read_bytes() == b"COMMITTED\n"
        # Success remains visible until the user chooses Close.
        assert updater_window(pid) == window
        assert (root / "bin" / BACKEND).read_bytes() == b"new backend"
        ctypes.windll.user32.PostMessageW(ctypes.c_void_p(window), 0x0010, 0, 0)
        wait_for(lambda: not transaction.exists())
        wait_for(lambda: not updater_window(pid))
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        remaining = updater_window(pid)
        if remaining:
            ctypes.windll.user32.PostMessageW(ctypes.c_void_p(remaining), 0x0010, 0, 0)


@pytest.mark.skipif(os.name != "nt", reason="Win32 native control automation")
def test_windows_gui_cancel_then_close_cleans_prepared_transaction(prepared):
    root, transaction, parent = start_parent(prepared, kind="gui")
    pid = int((root / "updater-pid").read_text())
    window = wait_for(lambda: updater_window(pid))
    try:
        assert (root / "handoff-response").read_text() == "READY\n"
        wait_for(lambda: close_button_text(window) == "Cancel")
        ctypes.windll.user32.PostMessageW(ctypes.c_void_p(window), 0x0010, 0, 0)
        wait_for(lambda: close_button_text(window) == "Close")
        # Cancellation removes large files immediately; the locked runner waits
        # for Close and its authenticated installed-helper handoff.
        assert updater_window(pid) == window
        assert not (transaction / "archive.zip").exists()
        assert not (transaction / "payload").exists()
        assert not (transaction / "journal.bin").exists()
        assert (transaction / "state").read_bytes() == b"ROLLED_BACK\n"
        assert (transaction / "runner" / UPDATER).exists()
        assert parent.poll() is None
        assert (root / "build-info.json").read_bytes() == b'{"version":"old"}'
        assert (root / LAUNCHER).read_bytes() == b"old launcher"
        ctypes.windll.user32.PostMessageW(ctypes.c_void_p(window), 0x0010, 0, 0)
        wait_for(lambda: not transaction.exists())
        wait_for(lambda: not updater_window(pid))
        assert parent.poll() is None
        assert (root / "config/user_config.yaml").read_bytes() == b"theme: light\n"
        assert (root / "models/weights").read_bytes() == b"weights"
    finally:
        (root / "exit-parent").touch()
        parent.wait(timeout=10)
        remaining = updater_window(pid)
        if remaining:
            ctypes.windll.user32.PostMessageW(ctypes.c_void_p(remaining), 0x0010, 0, 0)
