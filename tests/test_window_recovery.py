"""Exercise the real native viewer without models or opening a browser.

Windows uses WebView2; Linux needs GTK/WebKitGTK and a display (Xvfb works).
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
    sys.platform not in {"win32", "linux"}
    or (sys.platform == "linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))),
    reason="real viewer checks require Windows or a Linux graphical session",
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
        command = module._native_command(compiler, ROOT / "tests" / "window_host.c", output, scratch)
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
def running_host(binary, tmp_path, url, *, owns_backend=True):
    root = tmp_path / "native viewer with spaces"
    root.mkdir()
    executable = root / binary.name
    shutil.copy2(binary, executable)
    for name in ("libfox-window.so", "libfox-browser.so"):
        plugin = binary.parent / name
        if plugin.is_file():
            shutil.copy2(plugin, root / plugin.name)
    env = {**os.environ, "FOX_TEST_URL": url, "FOX_TEST_OWNS_BACKEND": str(int(owns_backend))}
    process = subprocess.Popen([str(executable)], cwd=root, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield root, process
    finally:
        if process.poll() is None:
            (root / "close-view").touch()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


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
        (root / "close-view").touch()
        assert process.wait(timeout=10) == 0
        assert lines(root, "stop-requested") == ["stop"]


def test_close_during_initial_navigation_does_not_fall_back(native_host, tmp_path, page_server):
    url, requested, release, _ = page_server
    with running_host(native_host, tmp_path, url) as (root, process):
        wait_for_navigation(root, process, requested)
        (root / "close-view").touch()
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
        (root / "quit-view").touch()
        assert process.wait(timeout=10) == 0
        assert lines(root, "stop-requested") == ["stop"]
