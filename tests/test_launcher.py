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


@pytest.fixture(scope="module")
def config_reader(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(root / "packaging"))
        import toolchain

        try:
            compiler = toolchain.find_compiler(allow_download=False)
        except toolchain.ToolchainError as exc:
            pytest.skip(f"native configuration checks need a C compiler: {exc}")
        output = tmp_path_factory.mktemp("launcher-config") / ("config.exe" if os.name == "nt" else "config")
        toolchain.build_launcher(output, root / "tests/launcher_config.c", compiler, gui=False)
    return output


@pytest.fixture(scope="module")
def timing_probe(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(root / "packaging"))
        import toolchain

        try:
            compiler = toolchain.find_compiler(allow_download=False)
        except toolchain.ToolchainError as exc:
            pytest.skip(f"native timing checks need a C compiler: {exc}")
        output = tmp_path_factory.mktemp("launcher-timing") / ("timing.exe" if os.name == "nt" else "timing")
        toolchain.build_launcher(output, root / "tests/launcher_timing.c", compiler, gui=False)
    return output


@pytest.mark.parametrize("mode", [
    "oversleep", "interrupted-sleep", "short-timeout", "zero-timeout", "already-exited",
    "exit-during-wait", "clock-failure", "clock-failure-during-wait",
])
def test_shutdown_wait_uses_elapsed_time(timing_probe, mode):
    result = subprocess.run([str(timing_probe), mode], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def cli_config_binaries(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(root / "packaging"))
        import toolchain

        try:
            compiler = toolchain.find_compiler(allow_download=False)
        except toolchain.ToolchainError as exc:
            pytest.skip(f"native startup checks need a C compiler: {exc}")
        directory = tmp_path_factory.mktemp("launcher-config-cli")
        launcher = directory / ("launcher.exe" if os.name == "nt" else "launcher")
        backend = directory / ("backend.exe" if os.name == "nt" else "backend")
        toolchain.build_launcher(launcher, compiler=compiler, no_window=True)
        toolchain.build_launcher(backend, root / "tests/launcher_backend.c", compiler, gui=False)
    return launcher, backend


@pytest.fixture
def cli_config_installation(tmp_path, monkeypatch, cli_config_binaries):
    launcher, backend = cli_config_binaries
    monkeypatch.setenv("FOX_TEST_NO_WINDOW_LAUNCHER", str(launcher))
    monkeypatch.setenv("FOX_TEST_BACKEND", str(backend))
    result = _installation(tmp_path, "FOX_TEST_NO_WINDOW_LAUNCHER")
    yield result
    launcher_diagnostics(result[0])


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS bundle exec forwarding")
@pytest.mark.parametrize("bundle_name", ["Fox Reader.app", "Renamed Reader.app"])
def test_macos_bundle_forwards_future_options_to_updated_root_launcher(cli_config_installation, bundle_name):
    root, launcher, _, env = cli_config_installation
    bundled = root / bundle_name / "Contents" / "MacOS" / "launcher"
    bundled.parent.mkdir(parents=True)
    shutil.copy2(launcher, bundled)
    original_bundle = bundled.read_bytes()
    marker = root / "forwarded-arguments"
    env["FOX_TEST_FORWARD_MARKER"] = str(marker)
    arguments = ["--future-option", "value with spaces", "--version"]
    for exit_code in (37, 38):
        launcher.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$$" "$@" > "$FOX_TEST_FORWARD_MARKER"\n'
            f"exit {exit_code}\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        process = subprocess.Popen([str(bundled), *arguments], cwd=root, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, errors = process.communicate(timeout=10)
        assert process.returncode == exit_code, (output + errors).decode(errors="replace")
        assert marker.read_text(encoding="utf-8").splitlines() == [str(process.pid), *arguments]
    assert bundled.read_bytes() == original_bundle
    assert not (root / "started").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS bundle exec forwarding")
@pytest.mark.parametrize("problem", ["missing", "symlink", "directory", "not-executable"])
def test_macos_bundle_refuses_missing_or_unsafe_root_launcher(cli_config_installation, problem):
    root, launcher, _, env = cli_config_installation
    bundled = root / "Renamed Reader.app" / "Contents" / "MacOS" / "launcher"
    bundled.parent.mkdir(parents=True)
    shutil.copy2(launcher, bundled)
    if problem == "not-executable":
        launcher.chmod(0o600)
    else:
        launcher.unlink()
        if problem == "symlink":
            launcher.symlink_to(bundled)
        elif problem == "directory":
            launcher.mkdir()

    result = subprocess.run([str(bundled), "--cli"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=10)

    assert result.returncode == 1
    assert "current Fox Reader launcher" in result.stderr
    assert "root launcher" in log_tail(root / "logs" / "launcher.log")
    assert not (root / "started").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS root launcher entry point")
def test_macos_root_launcher_runs_directly(cli_config_installation):
    root, launcher, _, env = cli_config_installation
    result = subprocess.run([str(launcher), "--version"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert re.fullmatch(r"\d+\.\d+\.\d+\s*", result.stdout)
    assert not (root / "started").exists()


@pytest.fixture(scope="module")
def recovery_helper(tmp_path_factory):
    """A native child records the pipe contract and waits for launcher exit."""
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("launcher-recovery-helper")
    source = directory / "recovery.c"
    source.write_text(
        '#define main fox_launcher_main\n'
        f'#include "{(root / "packaging/launcher/launcher.c").as_posix()}"\n'
        '#undef main\n' + r'''
#ifdef _WIN32
#include <fcntl.h>
#endif

static void mark(const char *name, const char *value)
{
    FILE *file = fopen(name, "wb");
    if (file) { fputs(value, file); fclose(file); }
}

static int wait_for_launcher(unsigned long pid)
{
#ifdef _WIN32
    HANDLE parent = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)pid);
    DWORD result;
    if (!parent) return 0;
    result = WaitForSingleObject(parent, 15000);
    CloseHandle(parent);
    return result == WAIT_OBJECT_0;
#else
    int waited;
    for (waited = 0; waited < 15000; waited += 25) {
        if (kill((pid_t)pid, 0) < 0 && errno == ESRCH) return 1;
        sleep_ms(25);
    }
    return 0;
#endif
}

int main(int argc, char **argv)
{
    char lines[8][PATH_CAP * 4], temporary[PATH_CAP], token[65], pid_text[32];
    pchar current[PATH_CAP];
    const char *mode = getenv("FOX_TEST_RECOVERY_MODE");
    char *cwd;
    FILE *file;
    int i;
    unsigned long parent;
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc == 2 && strcmp(argv[1], "--breakaway") == 0) {
        parent = strtoul(getenv("FOX_READER_LAUNCHER_PID"), NULL, 10);
        if (!wait_for_launcher(parent)) return 11;
        mark("detached-finished", "survived launcher exit");
        return 0;
    }
    if (mode && strcmp(mode, "backend") == 0) {
        const char *owner = getenv("FOX_READER_LAUNCHER_PID");
        if (!owner || !*owner) return 12;
        mark("launcher-pid", owner);
#ifdef _WIN32
        STARTUPINFOW startup;
        PROCESS_INFORMATION process;
        wchar_t executable[PATH_CAP], command[PATH_CAP + 32];
        memset(&startup, 0, sizeof startup); startup.cb = sizeof startup;
        memset(&process, 0, sizeof process);
        if (!GetModuleFileNameW(NULL, executable, PATH_CAP)) return 13;
        swprintf(command, PATH_CAP + 32, L"\"%ls\" --breakaway", executable);
        if (!CreateProcessW(executable, command, NULL, NULL, FALSE,
                            DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB,
                            NULL, NULL, &startup, &process)) return 14;
        CloseHandle(process.hThread); CloseHandle(process.hProcess);
#else
        pid_t child = fork();
        if (child < 0) return 13;
        if (child == 0) {
            if (setsid() < 0) _exit(14);
            execl(argv[0], argv[0], "--breakaway", (char *)NULL);
            _exit(15);
        }
#endif
        return 0;
    }
    if (argc != 1) return 2;
    for (i = 0; i < 8; ++i) {
        size_t size;
        if (!fgets(lines[i], sizeof lines[i], stdin)) return 3;
        size = strlen(lines[i]);
        if (!size || lines[i][size - 1] != '\n') return 4;
        lines[i][size - 1] = 0;
    }
    if (getchar() != EOF || strcmp(lines[0], "FOXREADER-UPDATER-1") ||
        (strcmp(lines[1], "recover") && strcmp(lines[1], "cleanup")) || strcmp(lines[6], "0") ||
        !update_hex(lines[4], 64) || !update_hex(lines[7], 64)) return 5;
#ifdef _WIN32
    if (!GetCurrentDirectoryW(PATH_CAP, current)) return 6;
    snprintf(pid_text, sizeof pid_text, "%lu", (unsigned long)GetCurrentProcessId());
#else
    if (!getcwd(current, sizeof current)) return 6;
    snprintf(pid_text, sizeof pid_text, "%lu", (unsigned long)getpid());
#endif
    cwd = PRINTABLE(current);
    if (!cwd || strcmp(cwd, lines[2])) return 7;
    PRINTABLE_FREE(cwd);
    snprintf(temporary, sizeof temporary, "%s/ticket.token", lines[3]);
    file = fopen(temporary, "rb");
    if (!file || fread(token, 1, 64, file) != 64) return 8;
    fclose(file); token[64] = 0;
    if (strcmp(token, lines[4])) return 9;
    file = fopen("recovery-handoff", "wb");
    if (!file) return 10;
    for (i = 0; i < 8; ++i) fprintf(file, "%s\n", lines[i]);
    fclose(file);
    mark("recovery-pid", pid_text);
    if (strcmp(lines[1], "cleanup") == 0) {
        const char *names[] = {"journal.bin", "state", "ticket.token"};
        if (mode && strcmp(mode, "cleanup-error") == 0) {
            fputs("ERROR\n", stdout); fflush(stdout);
            return 18;
        }
        fputs("READY\n", stdout); fflush(stdout);
        if (mode && strcmp(mode, "cleanup-fail") == 0) return 19;
        if (mode && strcmp(mode, "cleanup-hang") == 0) sleep_ms(40000);
        for (i = 0; i < 3; ++i) {
            snprintf(temporary, sizeof temporary, "%s/%s", lines[3], names[i]);
            remove(temporary);
        }
#ifdef _WIN32
        if (!RemoveDirectoryA(lines[3])) return 20;
#else
        if (rmdir(lines[3])) return 20;
#endif
        mark("cleanup-finished", lines[3]);
        return 0;
    }
    if (mode && strcmp(mode, "exit") == 0) return 1;
    if (mode && strcmp(mode, "late-ready") == 0) sleep_ms(20000);
    parent = strtoul(lines[5], NULL, 10);
#ifdef _WIN32
    /* Capture a process handle before READY, as the real updater does. */
    {
        HANDLE owner = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)parent);
        DWORD result;
        if (!owner) return 16;
        fputs(mode && strcmp(mode, "error") == 0 ? "ERROR\n" : "READY\n", stdout);
        fflush(stdout);
        result = WaitForSingleObject(owner, 15000);
        CloseHandle(owner);
        if (result != WAIT_OBJECT_0) return 17;
    }
#else
    if ((unsigned long)getppid() != parent) return 16;
    fputs(mode && strcmp(mode, "error") == 0 ? "ERROR\n" : "READY\n", stdout);
    fflush(stdout);
    if (!wait_for_launcher(parent)) return 17;
#endif
    mark("recovered", "launcher exited before mutation");
    return 0;
}
''',
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(root / "packaging"))
        import toolchain

        try:
            compiler = toolchain.find_compiler(allow_download=False)
        except toolchain.ToolchainError as exc:
            pytest.skip(f"native recovery checks need a C compiler: {exc}")
        output = directory / ("helper.exe" if os.name == "nt" else "helper")
        toolchain.build_launcher(output, source, compiler, gui=False)
    return output


def prepare_recovery(root, *, state=None, helper=None, identifier="a"):
    transaction = root / (".foxreader-update-" + identifier * 32)
    transaction.mkdir()
    (transaction / "journal.bin").write_bytes(b"fixture journal")
    if state is not None:
        (transaction / "state").write_bytes(state)
    if helper is not None:
        runner = transaction / "runner"
        runner.mkdir()
        shutil.copy2(helper, runner / ("foxreader-updater.exe" if os.name == "nt" else "foxreader-updater"))
        (transaction / "inventory.sha256").write_bytes(b"b" * 64 + b"\n")
    return transaction


def fixture_process_running(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_launcher_does_not_start_backend_during_update(cli_config_installation):
    root, launcher, _, env = cli_config_installation
    path = root / ".fox-reader.update.lock"
    with path.open("w+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([str(launcher), "--cli"], cwd=root, env=env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "being updated" in result.stdout
        assert not (root / "started").exists()
    # Releasing the lock leaves a harmless file; ordinary startup is restored.
    (root / "bin" / ("fox-reader.exe" if os.name == "nt" else "fox-reader")).unlink()
    result = subprocess.run([str(launcher), "--cli"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=10)
    assert "no Fox Reader binary" in result.stderr
    assert path.is_file()


@pytest.mark.parametrize("state,terminal", [
    (None, False), (b"COMMITTED\n", True), (b"ROLLED_BACK\n", True),
    (b"COMMITTED", False), (b"COMMITTED\nextra", False), (b"", False),
])
def test_launcher_checks_interrupted_update_before_backend(cli_config_installation, state, terminal):
    root, launcher, _, env = cli_config_installation
    prepare_recovery(root, state=state)
    (root / "bin" / ("fox-reader.exe" if os.name == "nt" else "fox-reader")).unlink()

    result = subprocess.run([str(launcher), "--cli"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=10)

    assert result.returncode != 0
    assert not (root / "started").exists()
    if terminal:
        assert "no Fox Reader binary" in result.stderr
    else:
        assert "interrupted Fox Reader update needs recovery" in result.stderr
        assert "no Fox Reader binary" not in result.stderr


def test_launcher_blocks_multiple_interrupted_updates(cli_config_installation):
    root, launcher, _, env = cli_config_installation
    prepare_recovery(root, identifier="a")
    prepare_recovery(root, identifier="b")

    result = subprocess.run([str(launcher), "--cli"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=10)

    assert result.returncode != 0
    assert "Cannot safely check" in result.stderr
    assert not (root / "started").exists()


def test_launcher_recovers_with_fresh_pipe_ticket_and_stays_closed(cli_config_installation, recovery_helper):
    root, launcher, _, env = cli_config_installation
    transaction = prepare_recovery(root, helper=recovery_helper)
    (transaction / "ticket.token").write_text("old authorization", encoding="utf-8")
    (transaction / "ticket.new").write_text("interrupted token write", encoding="utf-8")
    (root / "bin" / ("fox-reader.exe" if os.name == "nt" else "fox-reader")).unlink()
    process = subprocess.Popen([str(launcher), "--cli"], cwd=root, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = process.communicate(timeout=20)

    assert process.returncode == 0, (output + errors).decode(errors="replace")
    wait_for(lambda: (root / "recovered").exists(), description="recovery after launcher exit")
    lines = (root / "recovery-handoff").read_text(encoding="utf-8").splitlines()
    assert lines[:4] == ["FOXREADER-UPDATER-1", "recover", str(root), transaction.name]
    assert re.fullmatch("[0-9a-f]{64}", lines[4])
    assert lines[4] == (transaction / "ticket.token").read_text()
    assert lines[5:] == [str(process.pid), "0", "b" * 64]
    assert not (root / "started").exists()
    assert "will remain closed" in output.decode()


@pytest.mark.parametrize("mode", ["error", "exit", "late-ready"])
def test_failed_recovery_handoff_kills_helper_before_launcher_exit(
    cli_config_installation, recovery_helper, mode,
):
    root, launcher, _, env = cli_config_installation
    prepare_recovery(root, helper=recovery_helper)
    env["FOX_TEST_RECOVERY_MODE"] = mode

    result = subprocess.run([str(launcher), "--cli"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=25)

    assert result.returncode != 0, result.stdout + result.stderr
    assert "Could not start update recovery" in result.stderr
    assert not fixture_process_running(int((root / "recovery-pid").read_text()))
    assert not (root / "recovered").exists()
    assert not (root / "started").exists()


@pytest.mark.parametrize("digest", [b"b" * 64 + b"\0", b"b" * 63, b"B" * 64, b"b" * 64 + b"\nextra"])
def test_launcher_rejects_invalid_recovery_digest(cli_config_installation, recovery_helper, digest):
    root, launcher, _, env = cli_config_installation
    transaction = prepare_recovery(root, helper=recovery_helper)
    (transaction / "inventory.sha256").write_bytes(digest)

    result = subprocess.run([str(launcher), "--cli"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=10)

    assert result.returncode != 0
    assert "inventory digest is invalid" in result.stderr
    assert not (root / "recovery-pid").exists()
    assert not (root / "started").exists()


def test_backend_receives_launcher_pid_and_can_detach_updater(cli_config_installation, recovery_helper):
    root, launcher, _, env = cli_config_installation
    backend = root / "bin" / ("fox-reader.exe" if os.name == "nt" else "fox-reader")
    shutil.copy2(recovery_helper, backend)
    env.update(FOX_TEST_RECOVERY_MODE="backend", FOX_READER_LAUNCHER_PID="1")
    process = subprocess.Popen([str(launcher), "--cli"], cwd=root, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = process.communicate(timeout=20)

    # This fixture intentionally exits before serving HTTP. The launcher reports
    # that startup failure while still releasing its job/process-group ownership.
    assert process.returncode == 1, (output + errors).decode(errors="replace")
    assert b"backend exited before it was ready (code 0)" in errors
    assert (root / "launcher-pid").read_text() == str(process.pid)
    wait_for(lambda: (root / "detached-finished").exists(), description="updater survives owner shutdown")


def test_launcher_cleans_terminal_transactions_before_normal_startup(cli_config_installation, recovery_helper):
    root, launcher, _, env = cli_config_installation
    transactions = [
        prepare_recovery(root, state=b"COMMITTED\n", identifier="a"),
        prepare_recovery(root, state=b"ROLLED_BACK\n", identifier="b"),
        prepare_recovery(root, state=b"COMMITTED\n", identifier="c"),
    ]
    # Cleanup can have removed its journal before crashing; the terminal marker
    # still proves that the remaining transaction is safe to remove.
    (transactions[-1] / "journal.bin").unlink()
    shutil.copy2(recovery_helper, root / ("foxreader-updater.exe" if os.name == "nt" else "foxreader-updater"))
    (root / "bin" / ("fox-reader.exe" if os.name == "nt" else "fox-reader")).unlink()
    process = subprocess.Popen([str(launcher), "--cli"], cwd=root, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = process.communicate(timeout=20)

    assert process.returncode == 1, (output + errors).decode(errors="replace")
    assert b"no Fox Reader binary" in errors  # Normal startup proceeded after cleanup.
    assert all(not transaction.exists() for transaction in transactions)
    lines = (root / "recovery-handoff").read_text(encoding="utf-8").splitlines()
    assert lines[1] == "cleanup"
    assert lines[5:] == [str(process.pid), "0", "0" * 64]
    assert not (root / "recovered").exists()


@pytest.mark.parametrize("mode", ["cleanup-error", "cleanup-fail", "cleanup-hang"])
def test_launcher_cleanup_failure_does_not_block_backend(cli_config_installation, recovery_helper, mode):
    root, launcher, port, env = cli_config_installation
    transaction = prepare_recovery(root, state=b"COMMITTED\n")
    shutil.copy2(recovery_helper, root / ("foxreader-updater.exe" if os.name == "nt" else "foxreader-updater"))
    env["FOX_TEST_RECOVERY_MODE"] = mode
    process = subprocess.Popen([str(launcher), "--cli"], cwd=root, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(lambda: health(port), timeout=45 if mode == "cleanup-hang" else 15,
                 description="backend startup after cleanup failure")
        assert process.poll() is None
        assert transaction.is_dir()
        assert not fixture_process_running(int((root / "recovery-pid").read_text()))
        assert "Cleanup will be retried next launch" in log_tail(root / "logs" / "launcher.log")
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=25)
        wait_for(lambda: not health(port), description="fixture backend shutdown")


@pytest.mark.parametrize("default,user,host,port", [
    (None, None, "127.0.0.1", 7954),
    ("host: 0.0.0.0\nport: 8123\n", None, "0.0.0.0", 8123),
    ("host: 0.0.0.0\nport: 8123\n", "theme: light\n", "127.0.0.1", 7954),
    ("port: 8123\n", "port: 9000\n", "127.0.0.1", 9000),
    ("port: 8123\n", "port: broken\n", "127.0.0.1", 7954),
    ("port: 8123oops\n", None, "127.0.0.1", 7954),
    ("host: ::1\nport: 8123\nextra:\n  host: elsewhere\n  port: 9000\n", None, "::1", 8123),
])
def test_native_configuration_selection(config_reader, tmp_path, default, user, host, port):
    config = tmp_path / "config"
    for name, contents in (("fox_config.yaml", default), ("user_config.yaml", user)):
        if contents is not None:
            config.mkdir(exist_ok=True)
            (config / name).write_text(contents, encoding="utf-8")
    result = subprocess.run([str(config_reader)], cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [host, str(port)]


@pytest.mark.parametrize("filename", ["fox_config.yaml", "user_config.yaml"])
@pytest.mark.parametrize("source", [
    b"{host: 127.0.0.1, port: 8123, theme: light, layout: basic, mtl_dir: models}\n",
    b"'host': 127.0.0.1\n'port': 8123\n",
    "host: 127.0.0.1\nport: 8123\n".encode("utf-8-sig"),
    "host: 127.0.0.1\nport: 8123\n".encode("utf-16"),
    b"host: 127.0.0.1\nport: 8123\ntheme: invalid\nextra:\n  host: other\n  port: 9000\n",
], ids=["flow", "quoted-keys", "utf8-bom", "utf16-bom", "invalid-theme"])
def test_repaired_yaml_agrees_with_native_reader(config_reader, tmp_path, filename, source):
    from fox_reader.config import ConfigManager

    config = tmp_path / "config"
    config.mkdir()
    (config / "fox_config.yaml").write_text("host: 0.0.0.0\nport: 9000\n", encoding="utf-8")
    (config / filename).write_bytes(source)
    expected = ConfigManager(config).load()

    result = subprocess.run([str(config_reader)], cwd=tmp_path, capture_output=True, text=True, timeout=5)

    assert result.returncode == 0, result.stderr
    assert expected.port == 8123
    assert result.stdout.splitlines() == [expected.host, str(expected.port)]


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


@pytest.mark.parametrize("filename", ["fox_config.yaml", "user_config.yaml"])
def test_startup_refreshes_configuration_after_backend_repair(installation, filename):
    root, launcher, port, env = installation
    config = root / "config" / filename
    # Flow-style YAML is valid but intentionally beyond the small native reader.
    # The real backend canonicalizes it before binding. Wait for child startup
    # here so the rewrite occurs after the launcher has read the original file.
    config.write_text(f"{{host: 127.0.0.1, port: {port}}}\n", encoding="utf-8")
    args = [] if os.name == "nt" else ["--cli"]
    process = subprocess.Popen([str(launcher), *args], env=env, cwd=root,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(lambda: (root / "started").exists())
        config.write_text(f"host: 127.0.0.1\nport: {port}\n", encoding="utf-8")
        wait_for(lambda: f"Fox Reader is ready at http://127.0.0.1:{port}/" in
                 log_tail(root / "logs" / "launcher.log"))
        if os.name == "nt":
            wait_for_native_page(root, process)
        stop_launcher(process)
        assert process.wait(timeout=25) == 0
        assert (root / "shutdown-ok").read_text() == "token accepted"
        assert not health(port)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("filename", ["fox_config.yaml", "user_config.yaml"])
def test_cli_startup_observes_config_manager_repair(cli_config_installation, filename):
    root, launcher, port, env = cli_config_installation
    config = root / "config" / filename
    config.write_text(f"{{host: 127.0.0.1, port: {port}, theme: invalid}}\n", encoding="utf-8")
    # Use a private hidden console for CTRL_BREAK on Windows. This exercises
    # the real graceful shutdown handler without signaling pytest's console.
    helper = r'''
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from fox_reader.config import ConfigManager

root, port = Path(sys.argv[2]), int(sys.argv[3])
options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
process = subprocess.Popen([sys.argv[1], "--cli"], cwd=root, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
try:
    deadline = time.monotonic() + 15
    while not (root / "started").exists():
        assert process.poll() is None, "launcher exited before backend startup"
        assert time.monotonic() < deadline, "backend startup marker missing"
        time.sleep(0.05)
    config = ConfigManager(root / "config").load()
    assert config.port == port and config.theme == "dark"
    expected = f"Fox Reader is ready at http://127.0.0.1:{port}/"
    while expected not in (root / "logs/launcher.log").read_text(encoding="utf-8"):
        assert process.poll() is None, "launcher exited before repaired address was ready"
        assert time.monotonic() < deadline, "launcher kept probing the original address"
        time.sleep(0.05)
    process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
    assert process.wait(timeout=25) == 0, "launcher did not shut down cleanly"
finally:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)
'''
    options = {}
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        options.update(startupinfo=startup, creationflags=subprocess.CREATE_NEW_CONSOLE)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-c", helper, str(launcher), str(root), str(port)],
                            cwd=root, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=45, **options)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "shutdown-ok").read_text() == "token accepted"
    assert not (root / "ui-requested").exists()
    assert not health(port)


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


@pytest.mark.parametrize("system,kind", [("Windows", "msvc"), ("Linux", "gcc"), ("Darwin", "clang")])
def test_updater_build_uses_native_controls_and_cpp17(launcher_toolchain, monkeypatch, tmp_path, system, kind):
    toolchain = launcher_toolchain
    monkeypatch.setattr(toolchain.platform, "system", lambda: system)
    compiler_path = tmp_path / ("cl.exe" if kind == "msvc" else kind)
    compiler = toolchain.Compiler(kind, compiler_path, "fixture compiler")
    if kind != "msvc":
        compiler_path.with_name("g++" if kind == "gcc" else "clang++").write_bytes(b"fixture driver")
    monkeypatch.setattr(toolchain.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="-lgtk-3"))

    command = toolchain._updater_command(compiler, tmp_path / "updater", tmp_path)

    assert "/std:c++17" in command if kind == "msvc" else "-std=c++17" in command
    assert not any("WebView2" in argument or "WebKit" in argument or "python" in argument for argument in command)
    if system == "Windows":
        assert "/MT" in command and "/SUBSYSTEM:WINDOWS" in command
        assert "comctl32.lib" in command
    elif system == "Linux":
        assert "-static-libstdc++" in command and "-lgtk-3" in command
    else:
        assert "Cocoa" in command and "-lproc" in command
        assert command[command.index("-arch") + 1] == "arm64"


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
    import errno
    import fcntl
    import pty
    import select
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
    terminal_output = bytearray()

    def claim_terminal():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        os.tcsetpgrp(0, os.getpgrp())

    def wait_in_terminal(predicate, timeout=15):
        def ready():
            # A real terminal consumes its output. Darwin can wait for unread
            # PTY output during process exit, even after the backend has stopped.
            readable, _, _ = select.select([master], [], [], 0)
            if readable:
                try:
                    terminal_output.extend(os.read(master, 4096))
                    del terminal_output[:-8192]
                except OSError as exc:
                    if exc.errno != errno.EIO:  # Linux reports PTY EOF as EIO.
                        raise
            return predicate()

        return wait_for(ready, timeout=timeout)

    try:
        process = subprocess.Popen([str(launcher), *arguments], cwd=root, env=env,
                                   stdin=slave, stdout=slave, stderr=slave, preexec_fn=claim_terminal)
        os.close(slave)
        slave = None
        wait_in_terminal(lambda: marker.exists())
        assert marker.read_text() == f"http://127.0.0.1:{port}/"
        assert health(port)
        assert not (root / "ui-requested").exists()
        process.send_signal(signal.SIGTERM)
        wait_in_terminal(lambda: process.poll() is not None, timeout=25)
        assert process.returncode == 0
        assert (root / "shutdown-ok").read_text() == "token accepted"
        assert not health(port)
    finally:
        print("\nTerminal output:\n" + terminal_output.decode("utf-8", errors="replace"))
        try:
            if slave is not None:
                os.close(slave)
        finally:
            # Release the terminal before reaping, including on assertion failure.
            os.close(master)
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
