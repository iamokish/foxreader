"""Exercise compiled launchers in isolated folders; no real backend is started.

Set FOX_TEST_LAUNCHER and FOX_TEST_BACKEND to precompiled fixture binaries.
The native CI workflow does this on all supported platforms.
"""
import ctypes
import importlib.util
import os
import re
import shlex
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

# The native host allows 30 seconds each for engine initialization and first
# navigation. Include both phases and a little runner scheduling time; ordinary
# backend/ownership waits retain their shorter deadline.
NATIVE_PAGE_TIMEOUT = 65


def wait_for(predicate, timeout=15, *, description="launcher"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"Timed out after {timeout}s waiting for {description}")


def log_tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 8192))
            return stream.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"Log unavailable: {exc}"


def launcher_diagnostics(root):
    # Pytest displays captured teardown output on failure. Keep these bounded
    # and confined to fixture logs, without printing inherited environment data.
    print(f"\nLauncher fixture: {root}")
    print("Readiness markers:", sorted(path.name for path in root.glob("ui-ready-*")))
    for name in ("launcher.log", "backend-output.log"):
        print(f"\n{name}:\n{log_tail(root / 'logs' / name)}")


def wait_for_native_page(root, process, *, page=1):
    def ready():
        code = process.poll()
        assert code is None, f"Launcher exited before native page {page} was ready (code {code})"
        log = log_tail(root / "logs" / "launcher.log")
        assert "WebView2 fallback" not in log, f"Native page {page} fell back to the browser:\n{log}"
        return (root / f"ui-ready-{page}").is_file()

    wait_for(ready, timeout=NATIVE_PAGE_TIMEOUT, description=f"native page {page} readiness")


def post_window_message(hwnd, message, wparam=0, lparam=0):
    user = ctypes.WinDLL("user32", use_last_error=True)
    post = user.PostMessageW
    post.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
    post.restype = ctypes.c_int
    if not post(hwnd, message, wparam, lparam):
        raise ctypes.WinError(ctypes.get_last_error())


def windows_for(pid, class_name, title=None):
    if os.name != "nt":
        return []
    user = ctypes.windll.user32
    found = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def collect(hwnd, _):
        process = ctypes.c_ulong()
        user.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(process))
        name = ctypes.create_unicode_buffer(128)
        user.GetClassNameW(ctypes.c_void_p(hwnd), name, 128)
        caption = ctypes.create_unicode_buffer(128)
        user.GetWindowTextW(ctypes.c_void_p(hwnd), caption, 128)
        if process.value == pid and name.value == class_name and (title is None or caption.value == title):
            found.append(hwnd)
        return True

    user.EnumWindows(callback_type(collect), 0)
    return found


def window_for(pid):
    found = windows_for(pid, "FoxReaderWindow")
    return found[0] if found else None


def close_confirmation_for(pid):
    dialogs = windows_for(pid, "#32770", "Close Fox Reader?")
    assert len(dialogs) <= 1, "Repeated close requests opened multiple confirmation dialogs"
    return dialogs[0] if dialogs else None


def request_window_close(process):
    hwnd = wait_for(lambda: window_for(process.pid))
    post_window_message(hwnd, 0x0010)  # WM_CLOSE
    return wait_for(lambda: close_confirmation_for(process.pid), description="native close confirmation")


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
    result = _installation(tmp_path, "FOX_TEST_LAUNCHER")
    yield result
    launcher_diagnostics(result[0])


@pytest.fixture
def no_window_installation(tmp_path):
    result = _installation(tmp_path, "FOX_TEST_NO_WINDOW_LAUNCHER")
    yield result
    launcher_diagnostics(result[0])


def stop_launcher(process):
    if os.name == "nt":
        dialog = request_window_close(process)
        post_window_message(dialog, 0x0111, 6)  # WM_COMMAND, IDYES
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
            wait_for_native_page(root, process)
            hwnd = wait_for(lambda: window_for(process.pid))
            user = ctypes.windll.user32
            post_window_message(hwnd, 0x0100, 0x7A)  # F11
            wait_for(lambda: not user.GetWindowLongW(ctypes.c_void_p(hwnd), -16) & 0x00C00000)
            post_window_message(hwnd, 0x0100, 0x7A)
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


@pytest.mark.skipif(os.name != "nt", reason="Windows native close dialog")
@pytest.mark.parametrize("response", ["no", "default", "escape", "dismiss"])
def test_declining_native_close_preserves_backend_and_allows_later_shutdown(installation, response):
    root, launcher, port, env = installation
    process = subprocess.Popen([str(launcher)], env=env, cwd=root,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(lambda: health(port))
        wait_for_native_page(root, process)
        hwnd = wait_for(lambda: window_for(process.pid))
        dialog = request_window_close(process)
        user = ctypes.windll.user32
        user.GetWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        user.GetWindow.restype = ctypes.c_void_p
        assert user.GetWindow(dialog, 4) == hwnd  # GW_OWNER
        assert not user.IsWindowEnabled(ctypes.c_void_p(hwnd))
        for _ in range(3):
            assert request_window_close(process) == dialog
        assert process.poll() is None and health(port)
        assert not (root / "shutdown-ok").exists()
        message, value = {
            "no": (0x0111, 7),           # WM_COMMAND, IDNO
            "default": (0x0100, 0x0D),   # WM_KEYDOWN, Return
            "escape": (0x0100, 0x1B),    # WM_KEYDOWN, Escape
            "dismiss": (0x0010, 0),      # dialog's own title-bar close
        }[response]
        post_window_message(dialog, message, value)
        wait_for(lambda: close_confirmation_for(process.pid) is None)
        assert user.IsWindowEnabled(ctypes.c_void_p(hwnd))
        assert process.poll() is None and health(port)
        assert not (root / "shutdown-ok").exists()
        stop_launcher(process)
        assert process.wait(timeout=25) == 0
        assert (root / "shutdown-ok").read_text() == "token accepted"
        assert not health(port)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("close_during_startup", [
    pytest.param(False, id="ready"),
    pytest.param(True, id="starting", marks=pytest.mark.skipif(
        os.name != "nt", reason="Windows WebView startup",
    )),
])
def test_closing_attached_launcher_preserves_owner(installation, close_during_startup):
    root, launcher, port, env = installation
    args = [] if os.name == "nt" else ["--cli"]
    owner = subprocess.Popen([str(launcher), *args], env=env, cwd=root,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peer = None
    try:
        wait_for(lambda: health(port))
        if os.name == "nt" and not close_during_startup:
            wait_for_native_page(root, owner)
        peer = subprocess.Popen([str(launcher), *args], env=env, cwd=root,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for(lambda: "already running" in (root / "logs" / "launcher.log").read_text())
        if os.name == "nt" and not close_during_startup:
            # Separate normal attached close from closing during WebView startup.
            wait_for_native_page(root, peer, page=2)
        stop_launcher(peer)
        assert peer.wait(timeout=25) == 0
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


def test_workflow_temp_directories_start_clean_and_preserve_logs(tmp_path):
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/launcher.yml").read_text(
        encoding="utf-8",
    )
    commands = [shlex.split(line.strip()[5:]) for line in workflow.splitlines()
                if line.strip().startswith("run: ") and " -m pytest " in line]
    paths = {command[command.index("--basetemp") + 1] for command in commands}
    ownership_paths = {command[command.index("--basetemp") + 1] for command in commands
                       if "tests/test_launcher.py" in command}
    recovery_paths = {command[command.index("--basetemp") + 1] for command in commands
                      if "tests/test_window_recovery.py" in command}
    patterns = [line.strip() for line in workflow.splitlines()
                if line.strip().startswith("${{ runner.temp }}/")]
    assert paths and patterns, "Workflow test paths and diagnostics must be configured"
    assert ownership_paths and recovery_paths
    assert ownership_paths.isdisjoint(recovery_paths), "Recovery must not clear ownership test logs"

    runner_temp = tmp_path / "fresh runner temp"
    runner_temp.mkdir()
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "def test_temp_directory(tmp_path):\n"
        "    logs = tmp_path / 'logs'\n"
        "    logs.mkdir()\n"
        "    (logs / 'launcher.log').write_text('fixture log')\n"
        "    (tmp_path / 'window.log').write_text('recovery fixture log')\n",
        encoding="utf-8",
    )
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    evidence = []
    for template in sorted(paths):
        assert template.startswith("${{ runner.temp }}/")
        base = Path(template.replace("${{ runner.temp }}", runner_temp.as_posix())).resolve()
        # Pytest clears --basetemp. Confine this regression check to its fixture.
        assert runner_temp.resolve() in base.parents
        assert not base.exists()
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-c", str(config), "--confcutdir", str(tmp_path),
             str(probe), "--basetemp", str(base), "-p", "no:cacheprovider", "-q"],
            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        assert result.returncode == 0, f"{template}:\n{result.stdout}\n{result.stderr}"
        logs = list(base.rglob("launcher.log"))
        assert logs, "The real pytest tmp_path fixture must create its log"
        evidence.extend(logs)
        if template in recovery_paths:
            evidence.extend(base.rglob("window.log"))
        assert all(path.is_file() for path in evidence), "Later test runs must preserve earlier logs"

    uploaded = {path.resolve() for pattern in patterns
                for path in runner_temp.glob(pattern.removeprefix("${{ runner.temp }}/"))}
    assert set(evidence) <= uploaded, "The failure artifact must include logs from every pytest directory"


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
