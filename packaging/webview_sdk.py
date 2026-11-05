"""Fetch only the pinned WebView2 SDK headers and static loader, never a browser.

FOX_WEBVIEW2_SDK can point at an already-unpacked Microsoft.Web.WebView2 NuGet
package for offline builds. Otherwise the official NuGet package is verified
against its pinned SHA-512 before the build inputs and license are extracted.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

VERSION = "1.0.2903.40"
PACKAGE = "microsoft.web.webview2"
BASE_URL = f"https://api.nuget.org/v3-flatcontainer/{PACKAGE}/{VERSION}/{PACKAGE}.{VERSION}.nupkg"
# NuGet catalog packageHash (also the download's x-ms-meta-SHA512):
# https://api.nuget.org/v3/catalog0/data/2024.11.18.06.53.26/microsoft.web.webview2.1.0.2903.40.json
PACKAGE_SHA512 = base64.b64decode(
    "pDLSrpyma3xyOGXlw3Aowy7XceBdeJbukZzKLVdPCFvyfnn8PrCLylMn6c456ja0TcOOal8IdQCVONDRU2OGkw=="
)
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024


class WebViewSDKError(RuntimeError):
    """The requested SDK could not be downloaded or verified."""


@dataclass(frozen=True)
class WebViewSDK:
    include: Path
    static_library: Path
    license: Path


def _paths(root: Path, arch: str) -> tuple[Path, Path, Path]:
    return (root / "build/native/include/WebView2.h", root / f"build/native/{arch}/WebView2LoaderStatic.lib",
            root / "LICENSE.txt")


def _result(root: Path, arch: str) -> WebViewSDK:
    header, library, license = _paths(root, arch)
    if not header.is_file() or not library.is_file() or not license.is_file():
        raise WebViewSDKError(f"WebView2 SDK at {root} needs WebView2.h, the {arch} static loader, and LICENSE.txt")
    if not 1024 < header.stat().st_size <= MAX_MEMBER_BYTES or not 8 < library.stat().st_size <= MAX_MEMBER_BYTES:
        raise WebViewSDKError(f"WebView2 SDK at {root} contains invalid build inputs")
    with library.open("rb") as stream:
        if stream.read(8) != b"!<arch>\n":
            raise WebViewSDKError(f"WebView2 static loader at {library} is not a COFF library")
    if not 0 < license.stat().st_size <= MAX_MEMBER_BYTES:
        raise WebViewSDKError(f"WebView2 SDK at {root} has no valid license text")
    return WebViewSDK(header.parent, library, license)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cached(root: Path, arch: str) -> WebViewSDK | None:
    try:
        metadata = json.loads((root / "verified.json").read_text(encoding="utf-8"))
        if (metadata.get("version") != VERSION or metadata.get("arch") != arch
                or metadata.get("package_sha512") != PACKAGE_SHA512.hex()):
            return None
        paths = _paths(root, arch)
        if metadata.get("sha256") != {path.relative_to(root).as_posix(): _sha256(path) for path in paths}:
            return None
        return _result(root, arch)
    except (OSError, ValueError, TypeError, AttributeError, WebViewSDKError):
        return None


def _find_cached(cache_dir: Path, arch: str) -> WebViewSDK | None:
    # Older builds used a single mutable directory. Keep valid copies usable,
    # including read-only caches, but never repair one in place: a compiler or
    # antivirus may have its files open without Windows delete sharing.
    legacy = cache_dir / f"webview2-{VERSION}-{arch}"
    if cached := _cached(legacy, arch):
        return cached
    for candidate in sorted(cache_dir.glob(f"{legacy.name}-*")):
        if cached := _cached(candidate, arch):
            return cached
    return None


def _publish(staging: Path, destination: Path) -> None:
    # All streams are closed before publishing. A brief scanner lock on the
    # newly extracted files can still prevent a directory rename on Windows.
    for delay in (0.0, 0.1, 0.3, 0.6):
        if delay:
            time.sleep(delay)
        try:
            staging.rename(destination)
            return
        except PermissionError:
            if delay == 0.6:
                raise


def _download(url: str, destination: Path, limit: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "FoxReader-build/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > limit:
                    raise WebViewSDKError(f"WebView2 download exceeded the {limit}-byte limit")
                output.write(chunk)
    except (OSError, urllib.error.URLError) as exc:
        raise WebViewSDKError(
            f"Could not download the WebView2 SDK from NuGet: {exc}. "
            "For an offline build set FOX_WEBVIEW2_SDK to an unpacked SDK package."
        ) from exc


def _extract(package: Path, root: Path, arch: str) -> None:
    # An allowlist avoids path traversal and avoids copying any runtime DLLs.
    members = {path.relative_to(root).as_posix(): path for path in _paths(root, arch)}
    try:
        with zipfile.ZipFile(package) as archive:
            for name, destination in members.items():
                matching = [entry for entry in archive.infolist() if entry.filename == name]
                if len(matching) != 1:
                    raise WebViewSDKError(f"The WebView2 SDK must contain exactly one {name}")
                member = matching[0]
                if (member.is_dir() or stat.S_ISLNK(member.external_attr >> 16)
                        or not 0 < member.file_size <= MAX_MEMBER_BYTES):
                    raise WebViewSDKError(f"Unsafe WebView2 SDK member: {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as output:
                    written = 0
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        if written > MAX_MEMBER_BYTES:
                            raise WebViewSDKError(f"WebView2 SDK member is too large: {name}")
                        output.write(chunk)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise WebViewSDKError(f"Invalid WebView2 SDK package: {exc}") from exc


def prepare_sdk(cache_dir: Path, arch: str = "x64") -> WebViewSDK:
    """Return verified build inputs; ``cache_dir`` belongs to the build workspace."""
    if arch not in {"x64", "arm64", "x86"}:
        raise WebViewSDKError(f"Unsupported WebView2 architecture: {arch}")
    try:
        if override := os.environ.get("FOX_WEBVIEW2_SDK"):
            return _result(Path(override).expanduser().resolve(), arch)
        return _prepare_cached_sdk(cache_dir.resolve(), arch)
    except OSError as exc:
        location = os.environ.get("FOX_WEBVIEW2_SDK") or str(cache_dir)
        raise WebViewSDKError(
            f"Could not prepare the WebView2 SDK at {location}: {exc}. "
            "Check this folder's access permissions, or set FOX_WEBVIEW2_SDK "
            "to a readable, unpacked SDK package."
        ) from exc


def _prepare_cached_sdk(cache_dir: Path, arch: str) -> WebViewSDK:
    if cached := _find_cached(cache_dir, arch):
        return cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory only removes this newly created directory, inside the
    # explicitly resolved cache. Existing SDKs are never recursively deleted.
    with tempfile.TemporaryDirectory(prefix="webview2-download-", dir=cache_dir) as temporary:
        scratch = Path(temporary)
        package = scratch / "sdk.nupkg"
        _download(BASE_URL, package, MAX_PACKAGE_BYTES)
        with package.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha512").digest()
        if actual != PACKAGE_SHA512:
            raise WebViewSDKError("WebView2 SDK SHA-512 verification failed; no SDK files were installed")
        staging = scratch / "extracted"
        _extract(package, staging, arch)
        _result(staging, arch)
        # Include the marker in the staged tree, then publish the entire SDK to
        # a fresh directory. Existing files are never replaced, deleted, or
        # chmod'ed, so cached files can remain open in another build. Concurrent
        # publishers get separate complete generations without a mutable pointer.
        hashes = {path.relative_to(staging).as_posix(): _sha256(path) for path in _paths(staging, arch)}
        marker = staging / "verified.json"
        marker.write_text(json.dumps({"version": VERSION, "arch": arch,
                                      "package_sha512": actual.hex(), "sha256": hashes}), encoding="utf-8")
        root = cache_dir / f"webview2-{VERSION}-{arch}-{uuid.uuid4().hex}"
        _publish(staging, root)
    return _result(root, arch)
