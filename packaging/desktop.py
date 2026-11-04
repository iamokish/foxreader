"""Stage desktop entry points without moving the portable user-data folders."""
from __future__ import annotations

import plistlib
import shutil
import struct
from pathlib import Path


def stage_desktop(root: Path, launcher: Path, system: str, version: str) -> None:
    assets = Path(__file__).resolve().parent.parent / "frontend" / "static"
    icons = root / "icons"
    icons.mkdir(exist_ok=True)
    shutil.copy2(assets / "android-chrome-512x512.png", icons / "fox-reader.png")
    if system == "Windows":
        license = launcher.parent / "WebView2-LICENSE.txt"
        if not license.is_file() or not license.stat().st_size:
            raise RuntimeError("WebView2 SDK license is missing beside the compiled launcher")
        shutil.copy2(license, root / license.name)
    elif system == "Darwin":
        contents = root / "Fox Reader.app" / "Contents"
        executable = contents / "MacOS"
        resources = contents / "Resources"
        executable.mkdir(parents=True)
        resources.mkdir()
        shutil.copy2(launcher, executable / "launcher")
        (executable / "launcher").chmod(0o755)
        png = (icons / "fox-reader.png").read_bytes()
        # ic09 stores a 512-pixel PNG. macOS derives the smaller display sizes.
        (resources / "FoxReader.icns").write_bytes(
            b"icns" + struct.pack(">I", len(png) + 16) + b"ic09" + struct.pack(">I", len(png) + 8) + png
        )
        (contents / "Info.plist").write_bytes(plistlib.dumps({
            "CFBundleName": "Fox Reader", "CFBundleDisplayName": "Fox Reader",
            "CFBundleIdentifier": "io.github.iamokish.foxreader", "CFBundleExecutable": "launcher",
            "CFBundlePackageType": "APPL", "CFBundleShortVersionString": version,
            "CFBundleVersion": version, "CFBundleIconFile": "FoxReader.icns",
            "LSMinimumSystemVersion": "14.0", "NSHighResolutionCapable": True,
            "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
        }))
    elif system == "Linux":
        plugin = launcher.parent / "libfox-window.so"
        if not plugin.is_file():
            raise RuntimeError("native Linux viewer library is missing")
        shutil.copy2(plugin, root / plugin.name)
        installer = root / "install-desktop.sh"
        installer.write_text((Path(__file__).parent / "launcher" / "install-desktop.sh").read_text(), encoding="utf-8")
        installer.chmod(0o755)
