"""Exercise the real GTK-only browser controller without WebKit or a backend."""
from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
    reason="GTK controller checks require a Linux graphical session (Xvfb works)",
)
URL = "http://127.0.0.1:7999/"


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for browser controller evidence")


@pytest.fixture(scope="module")
def controller_binaries(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gtk-only-controller-build")
    spec = importlib.util.spec_from_file_location("fox_browser_test_toolchain", ROOT / "packaging/toolchain.py")
    toolchain = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(ROOT / "packaging"))
        imports.setitem(sys.modules, spec.name, toolchain)
        spec.loader.exec_module(toolchain)
        try:
            compiler = toolchain.find_compiler(allow_download=False)
        except toolchain.ToolchainError as exc:
            pytest.skip(f"native compiler unavailable: {exc}")
        flags = subprocess.run(["pkg-config", "--cflags", "--libs", "gtk+-3.0"],
                               check=True, capture_output=True, text=True, timeout=30)
        gtk = shlex.split(flags.stdout)
        launcher = ROOT / "packaging/launcher"

        def compile_file(*args):
            command = [str(compiler.path), "-Wall", "-Wextra", *map(str, args)]
            result = subprocess.run(command, cwd=directory, env=compiler.env,
                                    capture_output=True, text=True, timeout=120)
            assert result.returncode == 0, result.stdout + result.stderr

        plugin = directory / "libfox-browser.so"
        host = directory / "browser-host"
        compile_file("-fPIC", "-shared", launcher / "window_linux_browser.c", "-o", plugin, *gtk)
        compile_file(ROOT / "tests/browser_host.c", launcher / "window_linux_loader.c", "-o", host,
                     *gtk, "-ldl", "-pthread")

        # A present native plugin whose dependency is missing reproduces the
        # actual WebKit load failure without depending on WebKit on this host.
        dependency = directory / "missing.c"
        dependency.write_text("void fixture_dependency(void) {}\n", encoding="utf-8")
        missing = directory / "libfox-fixture-missing.so"
        compile_file("-fPIC", "-shared", dependency, "-o", missing)
        native_source = directory / "native.c"
        native_source.write_text(
            '#include <stdio.h>\n'
            'void fixture_dependency(void);\n'
            '__attribute__((constructor)) static void loaded(void) {\n'
            '  FILE *f = fopen("native-loaded", "w");\n'
            '  if (f) { fputs("loaded", f); fclose(f); }\n'
            '  fixture_dependency();\n'
            '}\n', encoding="utf-8",
        )
        native = directory / "libfox-window.so"
        compile_file("-fPIC", "-shared", native_source, "-o", native,
                     "-L", directory, "-lfox-fixture-missing", "-Wl,-rpath,$ORIGIN")

        for binary in (host, plugin):
            dependencies = subprocess.run(["ldd", str(binary)], check=True, capture_output=True,
                                          text=True, timeout=10).stdout.lower()
            assert "webkit" not in dependencies and "javascriptcore" not in dependencies, dependencies
    return directory


def lines(root, name):
    path = root / name
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


@contextmanager
def running_controller(binaries, tmp_path, *, browser_mode=False, native="missing"):
    root = tmp_path / "controller with spaces 日本"
    root.mkdir()
    for name in ("browser-host", "libfox-browser.so"):
        shutil.copy2(binaries / name, root / name)
    if native != "absent":
        shutil.copy2(binaries / "libfox-window.so", root / "libfox-window.so")
    if native == "loadable":
        shutil.copy2(binaries / "libfox-fixture-missing.so", root / "libfox-fixture-missing.so")
    env = {**os.environ, "FOX_TEST_URL": URL, "FOX_TEST_BROWSER_MODE": str(int(browser_mode))}
    with (root / "stderr.log").open("wb") as errors:
        process = subprocess.Popen([str(root / "browser-host")], cwd=root, env=env,
                                   stdout=subprocess.DEVNULL, stderr=errors)
        try:
            def created():
                assert process.poll() is None, (root / "stderr.log").read_text()
                return (root / "created").exists()
            wait_for(created)
            yield root, process
        finally:
            if process.poll() is None:
                (root / "close-view").touch()
                (root / "finish").touch()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def snapshot(root, process, expected):
    def inspect():
        assert process.poll() is None, (root / "stderr.log").read_text()
        path = root / "snapshot.txt"
        path.unlink(missing_ok=True)
        (root / "inspect").touch()
        wait_for(path.exists)
        text = path.read_text(encoding="utf-8")
        return text if expected in text else None
    return wait_for(inspect)


@pytest.mark.parametrize("browser_mode,native", [(False, "missing"), (False, "absent"), (True, "loadable")])
def test_browser_fallback_is_visible_and_can_reopen_and_quit(controller_binaries, tmp_path, browser_mode, native):
    with running_controller(controller_binaries, tmp_path, browser_mode=browser_mode, native=native) as (root, process):
        initial = snapshot(root, process, "fox-browser-state=Starting")
        assert "title=Fox Reader Control" in initial and "visible=1" in initial
        assert "fox-browser-open-sensitive=0" in initial
        assert "fox-browser-close=Cancel Startup" in initial
        maps = Path(f"/proc/{process.pid}/maps").read_text().lower()
        assert "libwebkit" not in maps and "libjavascriptcore" not in maps
        if native == "missing":
            assert "libfox-fixture-missing.so" in "\n".join(lines(root, "window.log"))
        assert not (root / "native-loaded").exists()  # explicit browser mode skips even a loadable native plugin
        (root / "ready-owned").touch()
        running = snapshot(root, process, "fox-browser-state=Running")
        assert f"fox-browser-url={URL}" in running
        assert "fox-browser-close=Quit Fox Reader" in running
        assert "fox-browser-open-sensitive=1" in running
        wait_for(lambda: lines(root, "browser-open") == [URL])
        (root / "ready-owned").touch()
        snapshot(root, process, "fox-browser-state=Running")
        assert lines(root, "browser-open") == [URL]
        (root / "open-reader").touch()
        wait_for(lambda: lines(root, "browser-open") == [URL, URL])
        (root / "quit-view").touch()
        stopping = snapshot(root, process, "fox-browser-state=Stopping")
        assert "fox-browser-open-sensitive=0" in stopping
        assert lines(root, "stop-requested") == ["stop"]
        (root / "ready-attached").touch()  # a late lifecycle update must not undo shutdown
        (root / "quit-view").touch()
        snapshot(root, process, "fox-browser-state=Stopping")
        assert lines(root, "stop-requested") == ["stop"]
        assert lines(root, "browser-open") == [URL, URL]
        (root / "finish").touch()
        assert process.wait(timeout=5) == 0
        assert lines(root, "window-exited") == ["exited"]


def test_attached_controller_explains_that_closing_leaves_original_running(controller_binaries, tmp_path):
    with running_controller(controller_binaries, tmp_path) as (root, process):
        (root / "ready-attached").touch()
        view = snapshot(root, process, "fox-browser-state=Running")
        assert "fox-browser-close=Close This Window" in view
        assert "running" in view.lower() and "original" in view.lower()
        (root / "close-view").touch()
        snapshot(root, process, "fox-browser-state=Closing")
        assert lines(root, "stop-requested") == ["stop"]
        (root / "finish").touch()
        assert process.wait(timeout=5) == 0


def test_close_during_startup_never_opens_a_late_browser(controller_binaries, tmp_path):
    with running_controller(controller_binaries, tmp_path) as (root, process):
        (root / "close-view").touch()
        snapshot(root, process, "fox-browser-state=Closing")
        assert lines(root, "stop-requested") == ["stop"]
        (root / "ready-owned").touch()
        snapshot(root, process, "fox-browser-state=Closing")
        assert not lines(root, "browser-open")
        (root / "finish").touch()
        assert process.wait(timeout=5) == 0


def test_backend_error_stays_visible_until_dismissed(controller_binaries, tmp_path):
    with running_controller(controller_binaries, tmp_path) as (root, process):
        (root / "failure").touch()
        view = snapshot(root, process, "fox-browser-state=Error")
        assert "Fixture backend failed." in view
        assert "fox-browser-open-sensitive=0" in view
        assert "fox-browser-close=Close" in view
        assert not lines(root, "stop-requested")
        (root / "ready-owned").touch()
        snapshot(root, process, "fox-browser-state=Error")
        assert not lines(root, "browser-open")
        (root / "quit-view").touch()
        assert process.wait(timeout=5) == 1
        assert lines(root, "window-exited") == ["exited"]
