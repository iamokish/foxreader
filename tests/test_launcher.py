"""Exercise compiled launchers in isolated folders; no real backend is started.

Set FOX_TEST_LAUNCHER and FOX_TEST_BACKEND to precompiled fixture binaries.
The native CI workflow does this on all supported platforms.
"""
import ctypes
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import time

import pytest


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for launcher")


def window_for(pid):
    if os.name != "nt":
        return None
    user = ctypes.windll.user32
    found = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def collect(hwnd, _):
        process = ctypes.c_ulong()
        user.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(process))
        name = ctypes.create_unicode_buffer(128)
        user.GetClassNameW(ctypes.c_void_p(hwnd), name, 128)
        if process.value == pid and name.value == "FoxReaderWindow":
            found.append(hwnd)
        return True

    user.EnumWindows(callback_type(collect), 0)
    return found[0] if found else None


@pytest.fixture
def installation(tmp_path):
    source = os.environ.get("FOX_TEST_LAUNCHER")
    backend = os.environ.get("FOX_TEST_BACKEND")
    if not source or not backend:
        pytest.skip("native launcher binaries are not supplied")
    root = tmp_path / "Fox Reader Unicode \u65e5\u672c"
    root.mkdir()
    name = "launcher.exe" if os.name == "nt" else "launcher"
    launcher = root / name
    shutil.copy2(source, launcher)
    (root / "bin").mkdir()
    shutil.copy2(backend, root / "bin" / ("fox-reader.exe" if os.name == "nt" else "fox-reader"))
    (root / "config").mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    (root / "config" / "fox_config.yaml").write_text(f"host: 127.0.0.1\nport: {port}\n")
    env = {**os.environ, "FOX_TEST_PORT": str(port), "FOX_TEST_DELAY": "250"}
    return root, launcher, port, env


def stop_launcher(process):
    if os.name == "nt":
        hwnd = wait_for(lambda: window_for(process.pid))
        ctypes.windll.user32.PostMessageW(ctypes.c_void_p(hwnd), 0x0010, 0, 0)
    else:
        process.send_signal(signal.SIGTERM)


def health(port):
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.3) as response:
            return b"fox-reader" in response.read()
    except OSError:
        return False


@pytest.mark.parametrize(("argument", "output"), [
    ("--help", r"(?s)^Fox Reader launcher .*--browser"),
    ("-h", r"(?s)^Fox Reader launcher .*--browser"),
    ("--version", r"^\d+\.\d+\.\d+\s*$"),
], ids=["help", "short-help", "version"])
def test_command_line_info_exits_without_starting_backend(installation, argument, output):
    root, launcher, _, env = installation
    # A broken Windows entry point must fail the test, not wait indefinitely
    # behind an inherited OS crash dialog. Restore the parent's mode afterward.
    previous_mode = None
    if os.name == "nt":
        previous_mode = ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    try:
        result = subprocess.run([str(launcher), argument], env=env, cwd=root,
                                capture_output=True, text=True, timeout=10)
    finally:
        if previous_mode is not None:
            ctypes.windll.kernel32.SetErrorMode(previous_mode)

    assert result.returncode == 0, (
        f"{argument} exited 0x{result.returncode & 0xFFFFFFFF:08X}: "
        f"{result.stdout}{result.stderr}"
    )
    assert re.search(output, result.stdout), result.stdout
    assert not result.stderr
    assert not (root / "started").exists()


def test_owned_close_and_native_output(installation):
    root, launcher, port, env = installation
    args = [] if os.name == "nt" else ["--no-browser"]
    process = subprocess.Popen([str(launcher), *args], env=env, cwd=root,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(lambda: health(port))
        if os.name == "nt":
            wait_for(lambda: (root / "ui-ready").exists())
            hwnd = wait_for(lambda: window_for(process.pid))
            user = ctypes.windll.user32
            user.PostMessageW(ctypes.c_void_p(hwnd), 0x0100, 0x7A, 0)  # F11
            wait_for(lambda: not user.GetWindowLongW(ctypes.c_void_p(hwnd), -16) & 0x00C00000)
            user.PostMessageW(ctypes.c_void_p(hwnd), 0x0100, 0x7A, 0)
            wait_for(lambda: user.GetWindowLongW(ctypes.c_void_p(hwnd), -16) & 0x00C00000)
        stop_launcher(process)
        assert process.wait(timeout=25) == 0
        assert (root / "shutdown-ok").read_text() == "token accepted"
        assert "fixture ready" in (root / "logs" / "backend-output.log").read_text()
        assert not health(port)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows PE subsystem")
def test_windows_binary_uses_gui_subsystem(installation):
    _, launcher, _, _ = installation
    data = launcher.read_bytes()
    header = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[header:header + 4] == b"PE\x00\x00"
    assert struct.unpack_from("<H", data, header + 24 + 68)[0] == 2  # IMAGE_SUBSYSTEM_WINDOWS_GUI


def test_closing_attached_launcher_preserves_owner(installation):
    root, launcher, port, env = installation
    args = [] if os.name == "nt" else ["--no-browser"]
    owner = subprocess.Popen([str(launcher), *args], env=env, cwd=root,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peer = None
    try:
        wait_for(lambda: health(port))
        peer = subprocess.Popen([str(launcher), *args], env=env, cwd=root,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for(lambda: "already running" in (root / "logs" / "launcher.log").read_text())
        stop_launcher(peer)
        assert peer.wait(timeout=10) == 0
        assert health(port)
        assert not (root / "shutdown-ok").exists()
        stop_launcher(owner)
        assert owner.wait(timeout=25) == 0
        assert (root / "shutdown-ok").exists()
    finally:
        for process in (peer, owner):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_close_during_startup_releases_installation(installation):
    root, launcher, port, env = installation
    env["FOX_TEST_DELAY"] = "1200"
    args = [] if os.name == "nt" else ["--no-browser"]
    process = subprocess.Popen([str(launcher), *args], env=env, cwd=root,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(lambda: (root / "started").exists())
        stop_launcher(process)
        assert process.wait(timeout=25) == 0
        assert not health(port)
        # A retry in the same directory must be able to acquire its lock.
        env["FOX_TEST_DELAY"] = "10"
        process = subprocess.Popen([str(launcher), *args], env=env, cwd=root,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for(lambda: health(port))
        stop_launcher(process)
        assert process.wait(timeout=25) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
