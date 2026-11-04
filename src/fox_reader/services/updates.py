"""Explicit update checks and verified downloads; never installs or runs code."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import httpx

from fox_reader import __version__

logger = logging.getLogger(__name__)
REPOSITORY = "iamokish/foxreader"
RELEASE_URL = f"https://github.com/{REPOSITORY}/releases/latest"
MANIFEST_URL = f"{RELEASE_URL}/download/latest.json"
CHUNK_SIZE = 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024**3
CHECK_COOLDOWN_SECONDS = 15
_CDN_HOSTS = {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com",
              "github-releases.githubusercontent.com"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
INSTALL_MESSAGE = (
    "Download verified. Close Fox Reader, extract this archive into a new folder, and copy your "
    "config, fonts, and models folders from the old installation before starting the new launcher. "
    "Keep the old installation until the new version works."
)


class UpdateError(ValueError):
    """An update cannot be selected or verified safely."""


class _Cancelled(Exception):
    pass


def _network_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == "https" and parsed.hostname in _CDN_HOSTS and parsed.port in (None, 443)
                 and not parsed.username and not parsed.password and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise UpdateError("Update server redirected outside GitHub's release download hosts.")


@contextmanager
def open_download(url: str):
    """Check every redirect before following it; never accept a downgrade to HTTP."""
    with httpx.Client(timeout=httpx.Timeout(20.0, connect=10.0), follow_redirects=False,
                      headers={"User-Agent": "FoxReader-Updater/1", "Accept-Encoding": "identity"}) as client:
        for _ in range(6):
            _network_url(url)
            with client.stream("GET", url) as response:
                if response.is_redirect:
                    target = response.headers.get("location")
                    if not target:
                        raise UpdateError("Update server sent an invalid redirect.")
                    url = urljoin(url, target)
                    continue
                response.raise_for_status()
                yield response
                return
        raise UpdateError("Update server sent too many redirects.")


def _stable_version(value: object) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value):
        raise UpdateError("The stable update manifest has an invalid version.")
    return tuple(map(int, value.split(".")))


def _newer(version: str, current: str) -> bool:
    target = _stable_version(version)
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+)(.*)", current)
    if not match:
        raise UpdateError("The installed version cannot be compared safely.")
    installed = tuple(int(match[index]) for index in (1, 2, 3))
    return target > installed or (target == installed and bool(match[4]) and not match[4].startswith("+"))


def _asset(asset: object) -> dict:
    if not isinstance(asset, dict):
        raise UpdateError("The update manifest contains an invalid archive.")
    name, size, digest = asset.get("name"), asset.get("size"), asset.get("sha256")
    if (not isinstance(name, str) or len(name) > 240
            or not re.fullmatch(r"fox-reader-[A-Za-z0-9._-]+", name)
            or not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_ARCHIVE_BYTES
            or not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
        raise UpdateError("The update manifest has an invalid filename, size, or checksum.")
    return asset


def validate_manifest(manifest: object, build: dict | None, current_version: str) -> tuple[dict, dict | None]:
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1):
        raise UpdateError("This release uses an unsupported update manifest.")
    version = manifest.get("version")
    _stable_version(version)
    tag = manifest.get("tag")
    if tag not in (version, f"v{version}"):
        raise UpdateError("The release tag and version do not agree.")
    expected_url = f"https://github.com/{REPOSITORY}/releases/tag/{quote(tag, safe='')}"
    if manifest.get("release_url") != expected_url:
        raise UpdateError("The manifest does not belong to Fox Reader's release repository.")
    notes, date = manifest.get("notes"), manifest.get("pub_date")
    if not isinstance(notes, str) or len(notes) > 128 * 1024 or not isinstance(date, str):
        raise UpdateError("The release notes or publication date are invalid.")
    try:
        if datetime.fromisoformat(date.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError
    except ValueError as error:
        raise UpdateError("The release publication date needs a valid timezone.") from error
    result = {"available": _newer(version, current_version), "current_version": current_version,
              "version": version, "notes": notes, "pub_date": date, "release_url": expected_url,
              "compatible": False, "variant": None, "download_size": 0}
    if build is None:
        result["reason"] = "This is a source checkout or an older build without update metadata. Use the release page."
        return result, None
    try:
        variants = manifest["platform"][build["platform"]][build["architecture"]]["variants"]
        if not isinstance(variants, dict) or len(variants) > 100:
            raise KeyError
        matches = [(key, item) for key, item in variants.items() if isinstance(item, dict)
                   and item.get("device") == build["device"] and item.get("gguf") == build["gguf"]]
    except (KeyError, TypeError):
        matches = []
    if not matches:
        result["reason"] = "This release has no download matching this installation's OS, CPU/GPU and GGUF variant."
        return result, None
    if len(matches) != 1:
        raise UpdateError("The release contains ambiguous builds for this installation.")
    key, selected = matches[0]
    if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9._-]{0,100}", key):
        raise UpdateError("Invalid release variant.")
    expected_variant = build["device"]
    if build["gguf"] is not None:
        expected_variant += "-gguf"
        if build["gguf"] != build["device"]:
            expected_variant += f"-{build['gguf']}"
    if key != expected_variant:
        raise UpdateError("The archive variant and build metadata do not agree.")
    archive = _asset(selected.get("archive"))
    parts = selected.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= 999:
        raise UpdateError("The update manifest has no valid ordered download parts.")
    extension = "zip" if build["platform"] == "windows" else "tar.gz"
    platform_suffix = {"windows": "windows-x64", "linux": "linux-x64", "macos": "macos-arm64"}[build["platform"]]
    if (archive.get("format") != extension
            or archive["name"] != f"fox-reader-v{version}-{key}-{platform_suffix}.{extension}"):
        raise UpdateError("The archive name does not match the selected version and platform.")
    split = parts[0].get("name") != archive["name"] if isinstance(parts[0], dict) else False
    for index, part in enumerate(parts, 1):
        _asset(part)
        name = f"{archive['name']}.{index:03d}" if split else archive["name"]
        url = f"https://github.com/{REPOSITORY}/releases/download/{quote(tag, safe='')}/{quote(name, safe='')}"
        if part["name"] != name or part.get("url") != url:
            raise UpdateError("Update parts are not ordered assets from the expected GitHub release.")
    if not split and len(parts) != 1:
        raise UpdateError("The manifest repeats an unsplit archive.")
    if sum(part["size"] for part in parts) != archive["size"]:
        raise UpdateError("Update part sizes do not add up to the archive size.")
    if not split and parts[0]["sha256"] != archive["sha256"]:
        raise UpdateError("The archive and download checksums do not agree.")
    result.update(compatible=True, variant=key, download_size=archive["size"])
    return result, deepcopy(selected)


class UpdateService:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._lock = threading.RLock()
        self._checking = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._selected: dict | None = None
        self._release: dict | None = None
        self._last_check = float("-inf")
        self.build = self._read_build()
        self.current_version = self.build["version"] if self.build else __version__
        self._status = {"state": "idle", "downloaded": 0, "total": 0, "message": "", "archive": None}

    def _read_build(self) -> dict | None:
        try:
            path = self.root / "build-info.json"
            if path.stat().st_size > 16 * 1024:
                return None
            build = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(build, dict) or build.get("repository") != REPOSITORY
                    or build.get("platform") not in {"windows", "linux", "macos"}
                    or build.get("architecture") != ("arm64" if build["platform"] == "macos" else "x86_64")
                    or not isinstance(build.get("version"), str)
                    or not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]*", build["version"])
                    or not isinstance(build.get("device"), str)
                    or "gguf" not in build or not (build["gguf"] is None or isinstance(build["gguf"], str))):
                return None
            return build
        except (OSError, ValueError, TypeError):
            return None

    def status(self) -> dict:
        with self._lock:
            result = {**self._status, "current_version": self.current_version,
                      "download_supported": self.build is not None}
            if not self.build:
                result["reason"] = "Source checkouts and older builds use the release page for updates."
            return result

    def _set(self, **fields) -> None:
        with self._lock:
            self._status.update(fields)

    def check(self) -> dict:
        if not self._checking.acquire(blocking=False):
            raise UpdateError("An update check is already running.")
        try:
            with self._lock:
                if self._thread and self._thread.is_alive():
                    raise UpdateError("Wait for the current download before checking another release.")
                if time.monotonic() - self._last_check < CHECK_COOLDOWN_SECONDS:
                    if self._release is not None:
                        return deepcopy(self._release)
                    raise UpdateError("Please wait 15 seconds before checking GitHub again.")
                self._last_check = time.monotonic()
                self._selected = None
                self._release = None
            with open_download(MANIFEST_URL) as response:
                content = bytearray()
                for block in response.iter_bytes(CHUNK_SIZE):
                    content.extend(block)
                    if len(content) > MAX_MANIFEST_BYTES:
                        raise UpdateError("The release manifest exceeds the size limit.")
            try:
                manifest = json.loads(content)
            except (ValueError, UnicodeError) as error:
                raise UpdateError("The release manifest is not valid JSON.") from error
            result, selected = validate_manifest(manifest, self.build, self.current_version)
            with self._lock:
                self._release = result
                self._selected = selected
            return result
        finally:
            self._checking.release()

    def start(self) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise UpdateError("An update download is already running.")
            if not self._release or not self._selected or not self._release["available"]:
                raise UpdateError("Check for a compatible newer release before downloading.")
            self._cancel.clear()
            self._status.update(state="downloading", downloaded=0, total=self._selected["archive"]["size"],
                                message="Preparing verified download…", archive=None)
            self._thread = threading.Thread(target=self._download, args=(deepcopy(self._selected),
                                            self._release["version"]), name="fox-reader-update", daemon=True)
            self._thread.start()
            return self.status()

    def cancel(self) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._cancel.set()
                self._status["message"] = "Cancelling download… Verified parts will be kept."
            return self.status()

    def close(self) -> None:
        self.cancel()

    def _cancelled(self) -> None:
        if self._cancel.is_set():
            raise _Cancelled

    def _path(self, parent: Path, name: str) -> Path:
        path = parent / name
        # Both symlinks and Windows junctions resolve here. Never follow one out
        # of the dedicated download directory or overwrite its target.
        if path.is_symlink() or path.resolve() != path or not path.resolve().is_relative_to(self.root):
            raise UpdateError("The update folder contains a redirected path; choose a regular installation folder.")
        return path

    def _verified(self, path: Path, metadata: dict) -> bool:
        if not path.is_file() or path.stat().st_size != metadata["size"]:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(CHUNK_SIZE):
                self._cancelled()
                digest.update(block)
        return digest.hexdigest() == metadata["sha256"]

    def _download(self, selected: dict, version: str) -> None:
        temporary: Path | None = None
        try:
            updates = self._path(self.root, "updates")
            updates.mkdir(exist_ok=True)
            folder = self._path(updates, version)
            folder.mkdir(exist_ok=True)
            archive = selected["archive"]
            destination = self._path(folder, archive["name"])
            self._set(message="Checking previously downloaded files…")
            if self._verified(destination, archive):
                self._set(state="complete", downloaded=archive["size"], message=INSTALL_MESSAGE,
                          archive=str(destination))
                return
            completed = []
            downloaded = 0
            for part in selected["parts"]:
                path = self._path(folder, part["name"])
                valid = self._verified(path, part)
                completed.append((part, path, valid))
                if valid:
                    downloaded += part["size"]
            split = selected["parts"][0]["name"] != archive["name"]
            need = archive["size"] - downloaded + (archive["size"] if split else 0) + 64 * 1024**2
            if shutil.disk_usage(folder).free < need:
                raise UpdateError(f"The update needs at least {need / 1024**3:.1f} GiB of free disk space.")
            self._set(downloaded=downloaded)
            for index, (part, path, valid) in enumerate(completed, 1):
                self._cancelled()
                if valid:
                    continue
                temporary = self._path(folder, part["name"] + ".part")
                digest = hashlib.sha256()
                size = 0
                last_progress = 0.0
                self._set(message=f"Downloading part {index} of {len(completed)}…")
                with open_download(part["url"]) as response, temporary.open("wb") as stream:
                    for block in response.iter_bytes(CHUNK_SIZE):
                        self._cancelled()
                        size += len(block)
                        if size > part["size"]:
                            raise UpdateError("The downloaded part is larger than its declared size.")
                        stream.write(block)
                        digest.update(block)
                        now = time.monotonic()
                        if now - last_progress >= 0.15:
                            self._set(downloaded=downloaded + size)
                            last_progress = now
                if size != part["size"] or digest.hexdigest() != part["sha256"]:
                    raise UpdateError("Download verification failed. Retry to download the affected part again.")
                self._cancelled()
                temporary.replace(path)
                temporary = None
                downloaded += size
                self._set(downloaded=downloaded)
            if split:
                self._set(message="Joining and verifying the complete archive…")
                temporary = self._path(folder, archive["name"] + ".part")
                digest = hashlib.sha256()
                size = 0
                with temporary.open("wb") as target:
                    for _, path, _ in completed:
                        with path.open("rb") as stream:
                            while block := stream.read(CHUNK_SIZE):
                                self._cancelled()
                                target.write(block)
                                digest.update(block)
                                size += len(block)
                if size != archive["size"] or digest.hexdigest() != archive["sha256"]:
                    raise UpdateError("The joined archive did not match its release checksum. Please retry.")
                self._cancelled()
                temporary.replace(destination)
                temporary = None
                for _, path, _ in completed:
                    try:
                        path.unlink()
                    except OSError:
                        logger.debug("Could not remove completed update part %s", path, exc_info=True)
            self._set(state="complete", downloaded=archive["size"], message=INSTALL_MESSAGE,
                      archive=str(destination))
        except _Cancelled:
            self._set(state="cancelled", message="Download cancelled. Verified parts are kept for the next attempt.")
        except (OSError, httpx.HTTPError, UpdateError) as error:
            logger.warning("Update download failed: %s", error)
            self._set(state="error", message=str(error))
        except Exception:
            logger.exception("Unexpected update download failure")
            self._set(state="error", message="The update download failed. See logs/backend.log for details.")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Could not remove incomplete update file %s", temporary, exc_info=True)
