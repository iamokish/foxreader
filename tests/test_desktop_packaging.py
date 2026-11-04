"""Portable native entry points retain the external user-data layout."""
import importlib.util
import plistlib
import struct
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
    module().stage_desktop(root, launcher, "Linux", "0.2.1")
    assert (root / "libfox-window.so").read_bytes() == b"viewer fixture"
    assert "Terminal=false" in (root / "install-desktop.sh").read_text()
    assert not (root / "bin").exists()


def test_windows_distribution_requires_and_preserves_webview_license(tmp_path):
    root = tmp_path / "portable"
    root.mkdir()
    launcher = tmp_path / "launcher.exe"
    launcher.write_bytes(b"executable fixture")
    desktop = module()
    with pytest.raises(RuntimeError, match="license is missing"):
        desktop.stage_desktop(root, launcher, "Windows", "0.2.1")
    license = tmp_path / "WebView2-LICENSE.txt"
    license.write_text("Microsoft WebView2 SDK license fixture", encoding="utf-8")
    desktop.stage_desktop(root, launcher, "Windows", "0.2.1")
    assert (root / license.name).read_bytes() == license.read_bytes()
