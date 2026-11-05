"""Stage desktop entry points without moving the portable user-data folders."""
from __future__ import annotations

import plistlib
import shutil
import struct
from pathlib import Path


def stage_desktop(root: Path, launcher: Path, system: str, version: str,
                  *, no_window: bool = False) -> None:
    assets = Path(__file__).resolve().parent.parent / "frontend" / "static"
    icons = root / "icons"
    icons.mkdir(exist_ok=True)
    shutil.copy2(assets / "android-chrome-512x512.png", icons / "fox-reader.png")
    if system == "Windows" and not no_window:
        license = launcher.parent / "WebView2-LICENSE.txt"
        if not license.is_file() or not license.stat().st_size:
            raise RuntimeError("WebView2 SDK license is missing beside the compiled launcher")
        shutil.copy2(license, root / license.name)
    elif system == "Darwin":
        if no_window:
            # Finder opens .command files in Terminal. Keep arguments literal and
            # resolve the portable folder independently of Terminal's directory.
            command = root / "Fox Reader.command"
            command.write_text(
                '#!/bin/sh\nset -eu\n'
                'root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)\n'
                'exec "$root/launcher" "$@"\n',
                encoding="utf-8", newline="\n",
            )
            command.chmod(0o755)
            return
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
        plugins = [] if no_window else [
            launcher.parent / name for name in ("libfox-window.so", "libfox-browser.so")
        ]
        for plugin in plugins:
            if not plugin.is_file() or not plugin.stat().st_size:
                raise RuntimeError(f"Linux window library is missing or empty: {plugin.name}")
        for plugin in plugins:
            shutil.copy2(plugin, root / plugin.name)
        installer = root / "install-desktop.sh"
        script = (Path(__file__).parent / "launcher" / "install-desktop.sh").read_text(encoding="utf-8")
        if no_window:
            script = script.replace("Terminal=false\n", "Terminal=true\n")
        installer.write_text(script, encoding="utf-8", newline="\n")
        installer.chmod(0o755)
