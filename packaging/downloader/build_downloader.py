"""Build script to compile the Fox Reader Downloader into a standalone single-file binary using Nuitka."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
ENTRY_POINT = HERE / "main.py"
ICON_PATH = HERE / "favicon.ico"


def _read_version() -> str:
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        raise FileNotFoundError(f"Could not find pyproject.toml at {pyproject}")
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find a version in {pyproject}")
    return match.group(1)


def _file_version(version: str) -> str:
    """A version Windows will accept in a resource: exactly four numbers."""
    parts = re.findall(r"\d+", version)[:4]
    parts += ["0"] * (4 - len(parts))
    return ".".join(parts)


def build_downloader(output_dir: Path, binary_name: str | None = None) -> Path:
    """Compile main.py using Nuitka into a single standalone executable."""
    if not ENTRY_POINT.is_file():
        raise FileNotFoundError(f"Entry point not found: {ENTRY_POINT}")

    # Check Nuitka
    try:
        import nuitka  # noqa: F401
    except ImportError:
        raise RuntimeError("Nuitka is not installed. Run 'uv pip install nuitka' or 'pip install nuitka'.")

    # Check customtkinter
    try:
        import customtkinter  # noqa: F401
    except ImportError:
        raise RuntimeError("customtkinter is not installed. Run 'uv pip install customtkinter'.")

    output_dir.mkdir(parents=True, exist_ok=True)

    is_windows = sys.platform.startswith("win")
    if not binary_name:
        binary_name = "fox-downloader.exe" if is_windows else "fox-downloader"

    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--onefile",
        "--assume-yes-for-downloads",
        "--enable-plugin=tk-inter",
        "--include-package-data=customtkinter",
        f"--output-dir={output_dir}",
        f"--output-filename={binary_name}",
        # Avoid caching bytecode issues
        "--disable-cache=bytecode",
    ]

    if ICON_PATH.is_file():
        cmd.append(f"--include-data-file={ICON_PATH}=favicon.ico")

    if is_windows:
        version = _read_version()
        file_ver = _file_version(version)
        cmd.extend([
            "--windows-console-mode=disable",
            "--company-name=Fox Reader",
            "--product-name=Fox Reader Downloader",
            "--file-description=Fox Reader Setup & Downloader",
            f"--file-version={file_ver}",
            f"--product-version={file_ver}",
        ])
        if ICON_PATH.is_file():
            cmd.append(f"--windows-icon-from-ico={ICON_PATH}")

    cmd.append(str(ENTRY_POINT))

    print(f"Building standalone downloader with Nuitka...")
    print(f"Command: {' '.join(cmd)}")

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Nuitka compilation failed with exit code {proc.returncode}")

    target_binary = output_dir / binary_name
    if not target_binary.is_file():
        raise FileNotFoundError(f"Expected compiled binary not found at {target_binary}")

    size_mb = target_binary.stat().st_size / (1024 * 1024)
    print(f"\n========================================================")
    print(f"Downloader successfully built:")
    print(f"  Path: {target_binary}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"========================================================\n")

    return target_binary


def main():
    parser = argparse.ArgumentParser(description="Compile Fox Reader Downloader into a single-file binary.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "dist",
        help="Directory to write the compiled binary to (default: dist/)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output binary name (default: fox-downloader.exe or fox-downloader)",
    )
    args = parser.parse_args()

    try:
        build_downloader(args.out_dir, args.name)
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
