"""Exercise the real native viewer without models or opening a browser.

Windows uses WebView2, macOS uses WKWebView, and Linux needs GTK/WebKitGTK
and a display (Xvfb works).
The fixture records fallback callbacks and can terminate its own Linux renderer.
Timeout coverage uses the production 30-second deadline, without test switches.
"""
from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    sys.platform not in {"win32", "linux", "darwin"}
    or (sys.platform == "linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))),
    reason="real viewer checks require Windows, macOS, or a Linux graphical session",
)


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for native viewer evidence")


@pytest.fixture(scope="module")
def native_host(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-window-recovery-build")
    scratch = directory / "objects"
    scratch.mkdir()
    output = directory / ("window-host.exe" if os.name == "nt" else "window-host")
    name = "fox_window_recovery_toolchain"
    spec = importlib.util.spec_from_file_location(name, ROOT / "packaging" / "toolchain.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous_path = sys.path[:]
    sys.path.insert(0, str(ROOT / "packaging"))
    try:
        spec.loader.exec_module(module)
        try:
            compiler = module.find_compiler(allow_download=False)
        except module.ToolchainError as exc:
            pytest.skip(f"native compiler unavailable: {exc}")
        source = "window_host_macos.m" if sys.platform == "darwin" else "window_host.c"
        command = module._native_command(compiler, ROOT / "tests" / source, output, scratch)
        if sys.platform == "linux":
            # The test host inspects its own GTK tree to request a real renderer
            # termination. These dependencies remain confined to this fixture.
            flags = subprocess.run(
                ["pkg-config", "--cflags", "--libs", "webkit2gtk-4.1"],
                check=True, capture_output=True, text=True, timeout=30,
            )
            command += shlex.split(flags.stdout)
        result = subprocess.run(command, cwd=scratch, env=compiler.env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.is_file()
    finally:
        sys.path[:] = previous_path
        sys.modules.pop(name, None)
    return output


@pytest.fixture
def page_server():
    requested, release, loaded = threading.Event(), threading.Event(), threading.Event()
    html = b"<!doctype html><title>Fox Reader fixture</title><script>fetch('/ui-ready')</script><p>Ready</p>"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body, kind = html, "text/html"
            elif self.path == "/api/health":
                body, kind = b'{"app":"fox-reader"}', "application/json"
            else:
                body, kind = b"ok", "text/plain"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.path == "/":
                requested.set()
                release.wait(45)
            elif self.path == "/ui-ready":
                loaded.set()
            try:
                self.wfile.write(body)
            except OSError:
                pass  # The timeout deliberately cancels the initial request.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", requested, release, loaded
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def lines(root, name):
    path = root / name
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


@contextmanager
def running_host(binary, tmp_path, url, *, owns_backend=True, startup=False, defer_finish=False):
    root = tmp_path / "native viewer with spaces"
    root.mkdir()
    executable = root / binary.name
    shutil.copy2(binary, executable)
    for name in ("libfox-window.so", "libfox-browser.so"):
        plugin = binary.parent / name
        if plugin.is_file():
            shutil.copy2(plugin, root / plugin.name)
    env = {**os.environ, "FOX_TEST_URL": url, "FOX_TEST_OWNS_BACKEND": str(int(owns_backend))}
    if startup:
        env["FOX_TEST_DEFER_READY"] = "1"
    if defer_finish:
        env["FOX_TEST_DEFER_FINISH"] = "1"
    process = subprocess.Popen([str(executable)], cwd=root, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        def created():
            if (root / "unavailable").exists():
                pytest.skip("native viewer cannot open a graphical window")
            assert process.poll() is None, f"native viewer exited: {lines(root, 'window.log')}"
            return (root / "created").exists()
        wait_for(created)
        yield root, process
    finally:
        if process.poll() is None:
            (root / "close-view").touch()
            (root / "confirm-yes").touch()
            if defer_finish:
                (root / "finish").touch()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def host_command(root, process, name):
    """Wait for a nonterminating command while requiring the host to stay alive."""
    path = root / name
    path.touch()

    def consumed():
        assert process.poll() is None, f"host exited while handling {name}: {lines(root, 'window.log')}"
        return not path.exists()
    wait_for(consumed)


def confirmation_snapshot(root, process, *, count):
    def inspect():
        path = root / "confirmation.txt"
        path.unlink(missing_ok=True)
        host_command(root, process, "inspect-confirmation")
        wait_for(path.exists)
        values = dict(line.split("=", 1) for line in path.read_text().splitlines())
        return values if values["count"] == str(count) else None
    return wait_for(inspect)


def assert_confirmation(root, process):
    values = confirmation_snapshot(root, process, count=1)
    for name in ("parent", "modal", "default-no", "yes", "no"):
        assert values[name] == "1", values
    assert not lines(root, "stop-requested")
    return values


def accept_close(root, process, *, command="close-view"):
    host_command(root, process, command)
    assert_confirmation(root, process)
    (root / "confirm-yes").touch()


def wait_for_navigation(root, process, requested):
    def began():
        if (root / "unavailable").exists():
            pytest.skip("native viewer cannot open a graphical window")
        if "Microsoft WebView2 is not installed" in "\n".join(lines(root, "window.log")):
            pytest.skip("Microsoft WebView2 runtime is unavailable")
        assert process.poll() is None, f"native viewer exited: {lines(root, 'window.log')}"
        assert not lines(root, "browser-open"), f"viewer failed before navigation: {lines(root, 'window.log')}"
        return requested.is_set()
    wait_for(began, timeout=40)


def test_stalled_initial_navigation_falls_back_once_and_ignores_late_response(native_host, tmp_path, page_server):
    url, requested, release, loaded = page_server
    with running_host(native_host, tmp_path, url) as (root, process):
        wait_for_navigation(root, process, requested)
        wait_for(lambda: lines(root, "browser-open"), timeout=40)
        assert lines(root, "browser-open") == [url]
        assert lines(root, "log-calls") == ["log"]
        assert "30 seconds" in "\n".join(lines(root, "window.log"))
        assert process.poll() is None
        assert not lines(root, "stop-requested")

        # The same request can complete after recovery; it must not reveal the
        # discarded view, reopen a browser, or shut down the manager window.
        release.set()
        assert not loaded.wait(1)
        assert lines(root, "browser-open") == [url]
        assert lines(root, "log-calls") == ["log"]
        assert process.poll() is None
        accept_close(root, process)
        assert process.wait(timeout=10) == 0
        assert lines(root, "stop-requested") == ["stop"]


def test_close_during_initial_navigation_does_not_fall_back(native_host, tmp_path, page_server):
    url, requested, release, _ = page_server
    with running_host(native_host, tmp_path, url) as (root, process):
        wait_for_navigation(root, process, requested)
        accept_close(root, process)
        assert process.wait(timeout=10) == 0
        release.set()
        assert lines(root, "stop-requested") == ["stop"]
        assert not lines(root, "browser-open")
        assert not lines(root, "log-calls")


@pytest.mark.skipif(sys.platform != "linux", reason="WebKitGTK renderer lifecycle")
@pytest.mark.parametrize("owns_backend", [True, False], ids=["owned", "attached"])
def test_linux_renderer_termination_falls_back_without_closing_manager(
    native_host, tmp_path, page_server, owns_backend,
):
    url, requested, release, loaded = page_server
    release.set()
    with running_host(native_host, tmp_path, url, owns_backend=owns_backend) as (root, process):
        wait_for_navigation(root, process, requested)
        assert loaded.wait(10), "native page did not execute its ready script"
        (root / "crash-view").touch()
        wait_for(lambda: lines(root, "browser-open"))
        assert lines(root, "crash-requested") == ["crash"]
        assert lines(root, "browser-open") == [url]
        assert lines(root, "log-calls") == ["log"]
        assert not lines(root, "stop-requested")
        assert process.poll() is None
        (root / "inspect-controls").touch()
        expected = "Quit Fox Reader" if owns_backend else "Close This Window"
        wait_for(lambda: lines(root, "close-label") == [expected])
        (root / "open-reader").touch()
        wait_for(lambda: lines(root, "browser-open") == [url, url])
        accept_close(root, process, command="quit-view")
        assert process.wait(timeout=10) == 0
        assert lines(root, "stop-requested") == ["stop"]


@pytest.mark.parametrize("owns_backend", [True, False], ids=["owned", "attached"])
@pytest.mark.parametrize("response", ["no", "default", "escape"])
def test_declining_close_keeps_native_reader_usable(native_host, tmp_path, page_server, owns_backend, response):
    url, requested, release, loaded = page_server
    release.set()
    with running_host(native_host, tmp_path, url, owns_backend=owns_backend) as (root, process):
        wait_for_navigation(root, process, requested)
        assert loaded.wait(10)
        host_command(root, process, "close-view")
        original = assert_confirmation(root, process)
        for _ in range(3):
            host_command(root, process, "close-view")
            assert assert_confirmation(root, process)["id"] == original["id"]
        host_command(root, process, f"confirm-{response}")
        dismissed = confirmation_snapshot(root, process, count=0)
        if os.name == "nt":
            assert dismissed["main-enabled"] == "1"
        assert not lines(root, "stop-requested")
        assert not lines(root, "browser-open")
        accept_close(root, process)
        assert process.wait(timeout=10) == 0
        assert lines(root, "stop-requested") == ["stop"]


@pytest.mark.parametrize("ready_while_prompt", [False, True], ids=["cancel-then-ready", "ready-during-prompt"])
def test_startup_close_can_be_declined_and_loading_continues(
    native_host, tmp_path, page_server, ready_while_prompt,
):
    url, requested, release, loaded = page_server
    release.set()
    with running_host(native_host, tmp_path, url, startup=True) as (root, process):
        host_command(root, process, "close-view")
        original = assert_confirmation(root, process)
        assert not requested.is_set()
        if ready_while_prompt:
            host_command(root, process, "ready-owned")
            wait_for_navigation(root, process, requested)
            assert loaded.wait(10)
            assert assert_confirmation(root, process)["id"] == original["id"]
        host_command(root, process, "confirm-default")
        confirmation_snapshot(root, process, count=0)
        assert not lines(root, "stop-requested")
        if not ready_while_prompt:
            host_command(root, process, "ready-owned")
            wait_for_navigation(root, process, requested)
            assert loaded.wait(10)
        accept_close(root, process)
        assert process.wait(timeout=10) == 0
        assert lines(root, "stop-requested") == ["stop"]


def test_yes_during_startup_stops_once_and_ignores_late_ready(native_host, tmp_path, page_server):
    url, requested, _, _ = page_server
    with running_host(native_host, tmp_path, url, startup=True, defer_finish=True) as (root, process):
        accept_close(root, process)
        wait_for(lambda: lines(root, "stop-requested") == ["stop"])
        host_command(root, process, "ready-owned")
        for _ in range(3):
            host_command(root, process, "close-view")
        confirmation_snapshot(root, process, count=0)
        assert lines(root, "stop-requested") == ["stop"]
        assert not requested.is_set()
        assert not lines(root, "browser-open")
        (root / "finish").touch()
        assert process.wait(timeout=10) == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows dialog keyboard and owner-drawn buttons")
def test_windows_focused_yes_accepts_return_and_preserves_button_theme(native_host, tmp_path, page_server):
    url, _, _, _ = page_server
    with running_host(native_host, tmp_path, url, startup=True) as (root, process):
        host_command(root, process, "close-view")
        initial = assert_confirmation(root, process)
        assert initial["yes-ownerdraw"] == initial["no-ownerdraw"] == "1"
        host_command(root, process, "confirm-focus-yes")
        focused = confirmation_snapshot(root, process, count=1)
        assert focused["yes-ownerdraw"] == focused["no-ownerdraw"] == "1"
        # Accepting Yes can exit before host_command's next liveness check.
        (root / "confirm-yes-default").touch()
        assert process.wait(timeout=10) == 0
        assert lines(root, "stop-requested") == ["stop"]
        assert lines(root, "window-exited") == ["exited"]


@pytest.mark.parametrize("finish", ["finish", "failure"])
def test_finish_while_confirmation_is_open_dismisses_it_without_requesting_stop(
    native_host, tmp_path, page_server, finish,
):
    url, _, _, _ = page_server
    with running_host(native_host, tmp_path, url, startup=True) as (root, process):
        host_command(root, process, "close-view")
        assert_confirmation(root, process)
        (root / finish).touch()
        if finish == "failure":
            dismissed = confirmation_snapshot(root, process, count=0)
            if os.name == "nt":
                assert dismissed["main-enabled"] == "1"
            # Finished errors dismiss directly, without asking again or stopping.
            (root / "close-view").touch()
        assert process.wait(timeout=10) == (1 if finish == "failure" else 0)
        assert not lines(root, "stop-requested")
        assert lines(root, "window-exited") == ["exited"]


@pytest.mark.skipif(sys.platform != "darwin", reason="AppKit application Quit action")
def test_macos_quit_uses_the_same_cancellable_sheet(native_host, tmp_path, page_server):
    url, _, _, _ = page_server
    with running_host(native_host, tmp_path, url, startup=True) as (root, process):
        host_command(root, process, "quit-application")
        original = assert_confirmation(root, process)
        host_command(root, process, "close-view")
        host_command(root, process, "quit-application")
        assert assert_confirmation(root, process)["id"] == original["id"]
        host_command(root, process, "confirm-escape")
        confirmation_snapshot(root, process, count=0)
        assert not lines(root, "stop-requested")
        accept_close(root, process, command="quit-application")
        assert process.wait(timeout=10) == 0
        assert lines(root, "stop-requested") == ["stop"]
