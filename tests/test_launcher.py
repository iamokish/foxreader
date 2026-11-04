"""Exercise compiled launchers in isolated folders; no real backend is started.

Set FOX_TEST_LAUNCHER and FOX_TEST_BACKEND to precompiled fixture binaries.
The native CI workflow does this on all supported platforms.
"""
import ctypes
import importlib.util
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


def _installation(tmp_path, launcher_variable):
    source = os.environ.get(launcher_variable)
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
    env = {key: value for key, value in os.environ.items() if key != "FOX_READER_CLI"}
    env.update(FOX_TEST_PORT=str(port), FOX_TEST_DELAY="250")
    return root, launcher, port, env


@pytest.fixture
def installation(tmp_path):
    return _installation(tmp_path, "FOX_TEST_LAUNCHER")


@pytest.fixture
def no_window_installation(tmp_path):
    return _installation(tmp_path, "FOX_TEST_NO_WINDOW_LAUNCHER")


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
    ("--help", r"(?s)^Fox Reader launcher .*--browser.*--cli"),
    ("-h", r"(?s)^Fox Reader launcher .*--browser.*--cli"),
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


def test_removed_headless_option_is_rejected_before_starting_backend(installation):
    root, launcher, _, env = installation
    result = subprocess.run([str(launcher), "--no-browser"], cwd=root, env=env,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "unknown option" in result.stderr
    assert not (root / "started").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX CLI shutdown uses SIGTERM")
@pytest.mark.parametrize("use_environment", [False, True], ids=["cli-option", "cli-environment"])
def test_cli_mode_runs_without_opening_a_browser(installation, use_environment):
    root, launcher, port, env = installation
    commands = root / "browser commands"
    commands.mkdir()
    for name in ("open", "xdg-open"):
        opener = commands / name
        opener.write_text('#!/bin/sh\nprintf opened > "$FOX_TEST_BROWSER_MARKER"\n', encoding="utf-8")
        opener.chmod(0o755)
    marker = root / "browser-opened"
    env.update(PATH=str(commands), FOX_TEST_BROWSER_MARKER=str(marker))
    if use_environment:
        env["FOX_READER_CLI"] = "1"
    process = subprocess.Popen([str(launcher), *([] if use_environment else ["--cli"])], cwd=root, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(lambda: health(port))
        wait_for(lambda: "Leave this window open" in (root / "logs/launcher.log").read_text())
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=25) == 0
        assert not marker.exists()
        assert not (root / "ui-requested").exists()
        assert (root / "shutdown-ok").read_text() == "token accepted"
        assert not health(port)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux terminal fallback")
@pytest.mark.parametrize("args", [[], ["--terminal-manager", "/nonexistent/fox-reader-test/control"]],
                         ids=["no-controller-or-terminal", "unverified-terminal-successor"])
def test_graphical_launch_without_a_controller_cannot_start_an_invisible_backend(installation, args):
    root, launcher, _, env = installation
    # This installation deliberately has neither GTK plugin. A private empty
    # PATH prevents the check from opening a terminal on the developer's desk.
    commands = root / "empty commands"
    commands.mkdir()
    env["PATH"] = str(commands)
    result = subprocess.run([str(launcher), *args], cwd=root, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, start_new_session=True, timeout=15)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (root / "started").exists()


def test_owned_close_and_native_output(installation):
    root, launcher, port, env = installation
    args = [] if os.name == "nt" else ["--cli"]
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
    args = [] if os.name == "nt" else ["--cli"]
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
    args = [] if os.name == "nt" else ["--cli"]
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


@pytest.fixture
def launcher_toolchain(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "packaging"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("fox_no_window_toolchain_tests", directory / "toolchain.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("system,kind", [
    ("Windows", "msvc"), ("Windows", "gcc"), ("Linux", "gcc"), ("Darwin", "clang"),
])
def test_no_window_build_never_builds_or_links_graphical_hosts(
    launcher_toolchain, monkeypatch, tmp_path, system, kind,
):
    toolchain = launcher_toolchain
    monkeypatch.setattr(toolchain, "IS_WINDOWS", system == "Windows")
    monkeypatch.setattr(toolchain.platform, "system", lambda: system)
    monkeypatch.setattr(toolchain, "_native_command", lambda *_: pytest.fail("native host build requested"))
    compiler = toolchain.Compiler(kind, Path("fixture-compiler"), "fixture compiler")
    output = tmp_path / ("launcher.exe" if system == "Windows" else "launcher")
    commands = []

    def compile_fixture(command, **kwargs):
        commands.append(command)
        output.write_bytes(b"console launcher fixture")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(toolchain.subprocess, "run", compile_fixture)
    assert toolchain.build_launcher(output, compiler=compiler, no_window=True) == output

    assert len(commands) == 1
    command = " ".join(commands[0])
    assert "FOX_NO_WINDOW" in command
    for graphical_part in ("FOX_GUI", "window_windows.cpp", "window_macos.m", "window_linux",
                           "WebView2", "WebKit", "Cocoa", "gtk", "/SUBSYSTEM:WINDOWS", "-mwindows"):
        assert graphical_part not in command
    assert ("terminal_linux.c" in command) == (system == "Linux")


def test_standard_launcher_still_builds_native_host(launcher_toolchain, monkeypatch, tmp_path):
    toolchain = launcher_toolchain
    output = tmp_path / "launcher"
    compiler = toolchain.Compiler("fixture", Path("fixture-compiler"), "fixture compiler")
    native_calls = []

    def native_command(*arguments):
        native_calls.append(arguments)
        return ["fixture-native-build"]

    def compile_fixture(command, **kwargs):
        assert command == ["fixture-native-build"]
        output.write_bytes(b"native launcher fixture")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(toolchain, "_native_command", native_command)
    monkeypatch.setattr(toolchain.subprocess, "run", compile_fixture)
    assert toolchain.build_launcher(output, compiler=compiler) == output
    assert len(native_calls) == 1


def test_toolchain_command_line_propagates_no_window(launcher_toolchain, monkeypatch, tmp_path):
    toolchain = launcher_toolchain
    compiler = toolchain.Compiler("fixture", Path("fixture-compiler"), "fixture compiler")
    monkeypatch.setattr(toolchain, "find_compiler", lambda **_: compiler)
    calls = []
    monkeypatch.setattr(toolchain, "build_launcher", lambda *args, **kwargs: calls.append((args, kwargs)))

    assert toolchain.main(["--no-window", "--out", str(tmp_path / "launcher")]) == 0
    assert len(calls) == 1
    assert calls[0][1]["no_window"] is True


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_no_window_command_line_info_does_not_start_backend(no_window_installation, argument):
    root, launcher, _, env = no_window_installation
    result = subprocess.run([str(launcher), argument], cwd=root, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip()
    assert not (root / "started").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows PE subsystem")
def test_no_window_windows_binary_uses_console_subsystem(no_window_installation):
    _, launcher, _, _ = no_window_installation
    data = launcher.read_bytes()
    header = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[header:header + 4] == b"PE\x00\x00"
    assert struct.unpack_from("<H", data, header + 24 + 68)[0] == 3  # IMAGE_SUBSYSTEM_WINDOWS_CUI


@pytest.mark.parametrize("use_environment", [False, True], ids=["cli-option", "cli-environment"])
def test_no_window_cli_runs_and_cleans_up_owned_backend(no_window_installation, use_environment):
    root, launcher, port, env = no_window_installation
    if use_environment:
        env["FOX_READER_CLI"] = "1"
    process = subprocess.Popen([str(launcher), *([] if use_environment else ["--cli"])], cwd=root, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(lambda: health(port))
        wait_for(lambda: "Leave this window open" in (root / "logs/launcher.log").read_text())
        assert not window_for(process.pid)
        assert not (root / "ui-requested").exists()
        assert "Opening it in your browser" not in (root / "logs/launcher.log").read_text()
        if os.name == "nt":
            # TerminateProcess exercises the job-object backstop without sending
            # console events to pytest's own terminal or other running apps.
            process.terminate()
            process.wait(timeout=10)
            wait_for(lambda: not health(port))
        else:
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=25) == 0
            assert (root / "shutdown-ok").read_text() == "token accepted"
            assert not health(port)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows console control events")
def test_no_window_ctrl_break_gracefully_stops_owned_backend(no_window_installation):
    root, launcher, port, env = no_window_installation
    # Only this short-lived helper and its launcher share the hidden console.
    # The event targets the launcher's new process group, never pytest's console
    # or other applications. Output pipes still preserve assertion diagnostics.
    helper = r'''
import ctypes
import signal
import subprocess
import sys
import time
import urllib.request

assert ctypes.windll.kernel32.GetConsoleWindow(), "isolated console was not created"
process = subprocess.Popen(
    [sys.argv[1], "--cli"], cwd=sys.argv[2], stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)
try:
    deadline = time.monotonic() + 10
    ready = False
    while time.monotonic() < deadline:
        assert process.poll() is None, "launcher exited before readiness"
        try:
            with urllib.request.urlopen(sys.argv[3], timeout=0.3) as response:
                ready = b"fox-reader" in response.read()
        except OSError:
            pass
        if ready:
            break
        time.sleep(0.05)
    assert ready, "fixture backend did not become ready"
    process.send_signal(signal.CTRL_BREAK_EVENT)
    assert process.wait(timeout=25) == 0, "launcher did not shut down cleanly"
finally:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)
'''
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    result = subprocess.run(
        [sys.executable, "-c", helper, str(launcher), str(root), f"http://127.0.0.1:{port}/api/health"],
        cwd=root, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        startupinfo=startup, creationflags=subprocess.CREATE_NEW_CONSOLE, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "shutdown-ok").read_text() == "token accepted"
    assert not health(port)


@pytest.mark.parametrize("arguments", [[], ["--browser"]], ids=["default", "explicit-browser"])
def test_no_window_browser_without_terminal_fails_before_backend(no_window_installation, arguments):
    root, launcher, _, env = no_window_installation
    commands = root / "commands"
    commands.mkdir()
    marker = root / "unexpected-graphical-output"
    for name in ("notify-send", "open", "xdg-open"):
        helper = commands / name
        helper.write_text('#!/bin/sh\nprintf invoked > "$FOX_TEST_GRAPHICAL_MARKER"\n', encoding="utf-8")
        helper.chmod(0o755)
    # If the terminal check regresses, prevent the fixture becoming ready and
    # opening the real Windows browser before subprocess.run kills the launcher.
    env.update(PATH=str(commands), FOX_TEST_GRAPHICAL_MARKER=str(marker), FOX_TEST_DELAY="30000")
    process_options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {
        "start_new_session": True,
    }

    result = subprocess.run([str(launcher), *arguments], cwd=root, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=15, **process_options)

    assert result.returncode != 0, result.stdout + result.stderr
    assert not (root / "started").exists()
    assert not marker.exists()


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="POSIX controlling PTY")
@pytest.mark.parametrize("arguments", [[], ["--browser"]], ids=["default", "explicit-browser"])
def test_no_window_browser_uses_terminal_and_preserves_owned_shutdown(no_window_installation, arguments):
    import fcntl
    import pty
    import termios

    root, launcher, port, env = no_window_installation
    commands = root / "browser commands"
    commands.mkdir()
    marker = root / "browser-opened"
    for name in ("open", "xdg-open"):
        helper = commands / name
        helper.write_text('#!/bin/sh\nprintf "%s" "$1" > "$FOX_TEST_BROWSER_MARKER"\n', encoding="utf-8")
        helper.chmod(0o755)
    env.update(PATH=str(commands), FOX_TEST_BROWSER_MARKER=str(marker))
    master, slave = pty.openpty()
    process = None

    def claim_terminal():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        os.tcsetpgrp(0, os.getpgrp())

    try:
        process = subprocess.Popen([str(launcher), *arguments], cwd=root, env=env,
                                   stdin=slave, stdout=slave, stderr=slave, preexec_fn=claim_terminal)
        os.close(slave)
        slave = None
        wait_for(lambda: marker.exists())
        assert marker.read_text() == f"http://127.0.0.1:{port}/"
        assert health(port)
        assert not (root / "ui-requested").exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=25) == 0
        assert (root / "shutdown-ok").read_text() == "token accepted"
        assert not health(port)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if slave is not None:
            os.close(slave)
        os.close(master)
