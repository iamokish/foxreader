"""Portable native entry points retain the external user-data layout."""
import importlib.util
import os
import plistlib
import struct
import subprocess
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "packaging" / "desktop.py"
    spec = importlib.util.spec_from_file_location("fox_desktop_packaging", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_macos_bundle_icon_and_local_network_policy(tmp_path):
    root = tmp_path / "portable"
    root.mkdir()
    launcher = tmp_path / "launcher"
    launcher.write_bytes(b"executable fixture")
    module().stage_desktop(root, launcher, "Darwin", "0.2.1")
    contents = root / "Fox Reader.app" / "Contents"
    info = plistlib.loads((contents / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "launcher"
    assert info["LSMinimumSystemVersion"] == "14.0"
    assert info["NSAppTransportSecurity"] == {"NSAllowsLocalNetworking": True}
    assert (contents / "MacOS" / "launcher").read_bytes() == launcher.read_bytes()
    icon = (contents / "Resources" / "FoxReader.icns").read_bytes()
    assert icon[:4] == b"icns"
    assert struct.unpack(">I", icon[4:8])[0] == len(icon)
    assert icon[16:] == (root / "icons" / "fox-reader.png").read_bytes()


def test_linux_viewer_stays_separate_from_backend(tmp_path):
    root = tmp_path / "portable"
    root.mkdir()
    launcher = tmp_path / "launcher"
    launcher.write_bytes(b"executable fixture")
    (tmp_path / "libfox-window.so").write_bytes(b"viewer fixture")
    (tmp_path / "libfox-browser.so").write_bytes(b"browser controller fixture")
    module().stage_desktop(root, launcher, "Linux", "0.2.1")
    assert (root / "libfox-window.so").read_bytes() == b"viewer fixture"
    assert (root / "libfox-browser.so").read_bytes() == b"browser controller fixture"
    assert "Terminal=false" in (root / "install-desktop.sh").read_text()
    assert not (root / "bin").exists()


@pytest.mark.parametrize("missing", ["libfox-window.so", "libfox-browser.so"])
def test_linux_distribution_requires_both_controller_and_viewer(tmp_path, missing):
    root = tmp_path / "portable"
    root.mkdir()
    launcher = tmp_path / "launcher"
    launcher.write_bytes(b"executable fixture")
    for name in ("libfox-window.so", "libfox-browser.so"):
        if name != missing:
            (tmp_path / name).write_bytes(b"plugin fixture")
    with pytest.raises(RuntimeError, match="missing"):
        module().stage_desktop(root, launcher, "Linux", "0.2.1")


@pytest.mark.parametrize("existing_legal", [False, True])
def test_windows_distribution_requires_and_preserves_webview_license(tmp_path, existing_legal):
    root = tmp_path / "portable"
    root.mkdir()
    if existing_legal:
        (root / "legal").mkdir()
        (root / "legal" / "LICENSE").write_text("Fox Reader license fixture", encoding="utf-8")
    launcher = tmp_path / "launcher.exe"
    launcher.write_bytes(b"executable fixture")
    desktop = module()
    with pytest.raises(RuntimeError, match="license is missing"):
        desktop.stage_desktop(root, launcher, "Windows", "0.2.1")
    license = tmp_path / "WebView2-LICENSE.txt"
    license.write_text("Microsoft WebView2 SDK license fixture", encoding="utf-8")
    desktop.stage_desktop(root, launcher, "Windows", "0.2.1")
    assert (root / "legal" / license.name).read_bytes() == license.read_bytes()
    assert not (root / license.name).exists()
    if existing_legal:
        assert (root / "legal" / "LICENSE").read_text(encoding="utf-8") == "Fox Reader license fixture"


@pytest.mark.parametrize("system", ["Windows", "Linux", "Darwin"])
@pytest.mark.parametrize("stale_native_assets", [False, True])
def test_no_window_distribution_omits_all_native_hosts(tmp_path, system, stale_native_assets):
    root = tmp_path / "portable"
    root.mkdir()
    launcher = tmp_path / ("launcher.exe" if system == "Windows" else "launcher")
    launcher.write_bytes(b"console launcher fixture")
    native_files = ("WebView2-LICENSE.txt", "libfox-window.so", "libfox-browser.so")
    if stale_native_assets:
        for name in native_files:
            (tmp_path / name).write_bytes(b"stale native build fixture")

    module().stage_desktop(root, launcher, system, "0.2.1", no_window=True)

    assert all(not (root / name).exists() for name in native_files)
    assert not (root / "legal" / "WebView2-LICENSE.txt").exists()
    assert not (root / "Fox Reader.app").exists()
    if system == "Linux":
        script = (root / "install-desktop.sh").read_bytes()
        assert b"Terminal=true\n" in script
        assert b"Terminal=false" not in script
        assert b"\r" not in script
    elif system == "Darwin":
        script = (root / "Fox Reader.command").read_bytes()
        assert script.startswith(b"#!/bin/sh\n")
        assert b'exec "$root/launcher" "$@"\n' in script
        assert b"\r" not in script


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX executable scripts")
def test_macos_terminal_entry_preserves_paths_and_arguments(tmp_path):
    root = tmp_path / "Fox Reader 日本語 $ ' folder"
    root.mkdir()
    launcher = root / "launcher"
    launcher.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    module().stage_desktop(root, launcher, "Darwin", "0.2.1", no_window=True)
    arguments = ["--cli", "argument with spaces", "$(printf unsafe)", "'quoted'"]

    result = subprocess.run(
        [str(root / "Fox Reader.command"), *arguments],
        cwd=tmp_path, check=True, text=True, capture_output=True,
    )

    assert result.stdout.splitlines() == arguments


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX executable scripts")
@pytest.mark.parametrize("no_window", [False, True])
def test_linux_installed_menu_entry_uses_required_terminal(tmp_path, no_window):
    root = tmp_path / "Fox Reader 日本語"
    root.mkdir()
    launcher = tmp_path / "launcher"
    launcher.write_bytes(b"launcher fixture")
    for name in ("libfox-window.so", "libfox-browser.so"):
        (tmp_path / name).write_bytes(b"plugin fixture")
    module().stage_desktop(root, launcher, "Linux", "0.2.1", no_window=no_window)
    data = tmp_path / "xdg-data"

    subprocess.run(
        [str(root / "install-desktop.sh")], cwd=tmp_path,
        env={**os.environ, "XDG_DATA_HOME": str(data)}, check=True, capture_output=True,
    )

    entry = (data / "applications" / "fox-reader.desktop").read_text(encoding="utf-8")
    assert f"Terminal={'true' if no_window else 'false'}\n" in entry
