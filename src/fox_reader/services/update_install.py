"""Prepare verified releases and hand them to the independent native updater.

Extraction never writes into the installed application. The native helper owns
process waits, replacement, the durable rollback journal, and final cleanup.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable
from uuid import uuid4

logger = logging.getLogger(__name__)
CHUNK = 1024 * 1024
MAX_ENTRIES = 250_000
MAX_UNPACKED = 256 * 1024**3
MAX_INVENTORY = 64 * 1024**2
TRANSACTION = re.compile(r"\.foxreader-update-[0-9a-f]{32}")
MANAGED = frozenset({
    "bin", "legal", "fonts", "icons", "launcher", "launcher.exe", "foxreader-updater", "foxreader-updater.exe",
    "build-info.json", "libfox-window.so", "libfox-browser.so", "install-desktop.sh",
    "Fox Reader.app", "Fox Reader.command",
})
PROTECTED = frozenset({"config", "models", "cache", "logs", "updates"})


class InstallError(ValueError):
    """Preparation or the authenticated helper handoff could not complete."""


@dataclass(frozen=True)
class PreparedUpdate:
    root: Path
    transaction: Path
    inventory_sha256: str
    archive: Path


@dataclass(frozen=True)
class Member:
    name: str
    kind: str
    size: int
    mode: int
    source: object
    link: str = ""


def updater_name(platform: str) -> str:
    return "foxreader-updater.exe" if platform == "windows" else "foxreader-updater"


def install_supported(root: Path, build: dict | None) -> bool:
    if not build or build.get("no_window", False) or not getattr(sys, "frozen", False):
        return False
    helper = root / updater_name(build["platform"])
    backend = root / "bin" / ("fox-reader.exe" if build["platform"] == "windows" else "fox-reader")
    return helper.is_file() and not helper.is_symlink() and Path(sys.executable).resolve() == backend.resolve()


def _plain(path: Path, *, directory: bool = False) -> None:
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise InstallError(f"Update path is not a regular {'directory' if directory else 'file'}: {path}")


def _sha256(path: Path, cancel: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            cancel()
            digest.update(block)
    return digest.hexdigest()


def _name(value: str, windows: bool) -> str:
    if not value or "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise InstallError("The update archive contains an invalid path.")
    parts = value.rstrip("/").split("/")
    if any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise InstallError("The update archive contains an unsafe path.")
    if windows and any(
        part.endswith((".", " ")) or any(c in part for c in '<>"|?*')
        or re.fullmatch(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part, re.I)
        for part in parts
    ):
        raise InstallError("The update archive contains a filename Windows cannot safely install.")
    if parts[0] != "fox-reader":
        raise InstallError("The update archive must contain one fox-reader directory.")
    return "/".join(parts[1:])


def _link_target(name: str, target: str) -> str:
    if (not target or len(target) > 4096 or target.startswith("/") or "\\" in target or ":" in target
            or any(ord(c) < 32 or ord(c) == 127 for c in target)):
        raise InstallError("The update archive contains an unsafe symbolic link.")
    parts = name.split("/")[:-1]
    for part in target.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise InstallError("An update symbolic link escapes its payload.")
            parts.pop()
        else:
            parts.append(part)
    if not parts or parts[0] not in MANAGED:
        raise InstallError("An update symbolic link targets protected or external data.")
    return "/".join(parts)


def _resolved_link(item: Member, members: dict[str, Member]) -> str:
    """Resolve directory-link chains as the filesystem does, before extraction."""
    pending = deque(item.name.split("/")[:-1] + item.link.split("/"))
    parts: list[str] = []
    followed = 0
    while pending:
        part = pending.popleft()
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise InstallError("An update symbolic link escapes its payload.")
            parts.pop()
            continue
        parts.append(part)
        member = members.get("/".join(parts))
        if member is None:
            raise InstallError("The update archive contains a dangling symbolic link.")
        if member.kind == "L":
            followed += 1
            if followed > 128:
                raise InstallError("The update archive contains a symbolic-link cycle or excessive chain.")
            parts.pop()
            pending.extendleft(reversed(member.link.split("/")))
        elif pending and member.kind != "D":
            raise InstallError("An update symbolic link traverses a regular file.")
    resolved = "/".join(parts)
    if not parts or parts[0] not in MANAGED or resolved not in members:
        raise InstallError("An update symbolic link targets protected or external data.")
    return resolved


def _members(archive, windows: bool) -> list[Member]:
    entries = archive.infolist() if isinstance(archive, zipfile.ZipFile) else archive
    members: dict[str, Member] = {}
    spellings: dict[str, str] = {}
    total = 0
    for index, entry in enumerate(entries, 1):
        if index > MAX_ENTRIES:
            raise InstallError("The update archive contains too many entries.")
        zipped = isinstance(entry, zipfile.ZipInfo)
        path = entry.filename if zipped else entry.name
        name = _name(path, windows)
        mode = (entry.external_attr >> 16) if zipped else entry.mode
        if zipped:
            kind = "D" if entry.is_dir() else "L" if stat.S_ISLNK(mode) else "F"
            if stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK} or entry.flag_bits & 1:
                raise InstallError("The update archive contains an unsupported file type.")
            size = entry.file_size
            if kind == "L":
                if size > 4096:
                    raise InstallError("The update archive contains an oversized symbolic link.")
                target = archive.read(entry).decode("utf-8")
            else:
                target = ""
        else:
            kind = ("D" if entry.isdir() else "F" if entry.isfile()
                    else "L" if entry.issym() else "H" if entry.islnk() else "")
            if not kind or entry.sparse:
                raise InstallError("The update archive contains an unsupported file type.")
            size, target = entry.size, entry.linkname
        if not name:
            if kind != "D":
                raise InstallError("The update archive root is not a directory.")
            continue
        top = name.split("/")[0]
        if top in PROTECTED:
            # Releases ship empty cache/models/logs. Never stage user data,
            # even if a malformed release attempts to provide replacements.
            if name != top or kind != "D":
                raise InstallError("The update archive unexpectedly contains user data.")
            continue
        if top not in MANAGED:
            raise InstallError(f"The update archive contains an unexpected component: {top}")
        if name in members or (name.casefold() in spellings and spellings[name.casefold()] != name):
            raise InstallError("The update archive contains duplicate or colliding paths.")
        if kind == "L":
            _link_target(name, target)
        if kind == "H":
            target = _name(target, windows)
        if size < 0:
            raise InstallError("The update archive contains an invalid file size.")
        total += size
        if total > MAX_UNPACKED:
            raise InstallError("The unpacked update exceeds the size limit.")
        members[name] = Member(name, kind, size, mode, entry, target)
        spellings[name.casefold()] = name
    # Parents omitted from ZIP directory entries are still part of inventory.
    for item in list(members.values()):
        for parent in PurePosixPath(item.name).parents:
            name = str(parent)
            if name == ".":
                continue
            if name in members and members[name].kind != "D":
                raise InstallError("An update file is nested beneath a link or non-directory.")
            if name.casefold() in spellings and spellings[name.casefold()] != name:
                raise InstallError("The update archive contains colliding parent paths.")
            members.setdefault(name, Member(name, "D", 0, 0o755, None))
            spellings[name.casefold()] = name
            if len(members) > MAX_ENTRIES:
                raise InstallError("The update archive contains too many entries.")
    for item in list(members.values()):
        if item.kind == "H":
            if item.link not in members or members[item.link].kind != "F":
                raise InstallError("An update hard link does not reference a regular payload file.")
            # Hard links become independent files. Account for their actual
            # expanded bytes in both the limit and the free-space estimate.
            size = members[item.link].size
            total += size - item.size
            members[item.name] = Member(item.name, item.kind, size, item.mode, item.source, item.link)
            if total > MAX_UNPACKED:
                raise InstallError("The unpacked update exceeds the size limit.")
        if item.kind == "L":
            _resolved_link(item, members)
    return sorted(members.values(), key=lambda item: (item.name.count("/"), item.name))


def _write_file(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _extract(archive, members: list[Member], payload: Path, cancel: Callable[[], None]) -> None:
    for item in members:
        cancel()
        target = payload / item.name
        if item.kind == "D":
            target.mkdir(exist_ok=True)
        elif item.kind == "F":
            source = (archive.open(item.source) if isinstance(archive, zipfile.ZipFile)
                      else archive.extractfile(item.source))
            count = 0
            with source, target.open("xb") as output:
                while block := source.read(CHUNK):
                    cancel()
                    count += len(block)
                    if count > item.size:
                        raise InstallError("An extracted update file exceeds its declared size.")
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if count != item.size:
                raise InstallError("The update archive contains a truncated file.")
            if os.name != "nt":
                target.chmod((item.mode & 0o777) or 0o644)
    for item in members:
        cancel()
        target = payload / item.name
        if item.kind == "H":
            with (payload / item.link).open("rb") as source, target.open("xb") as output:
                while block := source.read(CHUNK):
                    cancel()
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            shutil.copystat(payload / item.link, target)
        elif item.kind == "L":
            target.symlink_to(item.link, target_is_directory=(payload / _link_target(item.name, item.link)).is_dir())
    for item in members:
        target = payload / item.name
        if item.kind == "L" and (not target.resolve().is_relative_to(payload) or not target.exists()):
            raise InstallError("An extracted symbolic link is invalid.")


def _validate_payload(payload: Path, build: dict, version: str) -> None:
    metadata = payload / "build-info.json"
    _plain(metadata)
    if metadata.stat().st_size > 16 * 1024:
        raise InstallError("The release build metadata is too large.")
    target = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(target, dict) or target.get("version") != version:
        raise InstallError("The extracted release has the wrong version.")
    for key in ("platform", "architecture", "device", "gguf", "repository"):
        if target.get(key) != build.get(key):
            raise InstallError("The extracted release does not match this installation.")
    if target.get("no_window", False) is not False:
        raise InstallError("The extracted release uses an incompatible window variant.")
    windows = build["platform"] == "windows"
    required = [
        "bin/fox-reader.exe" if windows else "bin/fox-reader",
        "launcher.exe" if windows else "launcher", updater_name(build["platform"]),
        "legal/LICENSE", "legal/NOTICE", "legal/TERMS.md", "legal/FONT.md",
        "legal/OFL.txt", "legal/LICENSE-ComicMono", "legal/LICENSE-material-icons",
    ]
    if windows:
        required.append("legal/WebView2-LICENSE.txt")
    elif build["platform"] == "linux":
        required.extend(("libfox-window.so", "libfox-browser.so", "install-desktop.sh"))
    else:
        required.extend(("Fox Reader.app/Contents/MacOS/launcher", "Fox Reader.app/Contents/Info.plist"))
    for name in required:
        _plain(payload / name)
        if not (payload / name).stat().st_size:
            raise InstallError(f"The release contains an empty required file: {name}")
    if not windows:
        executables = ["bin/fox-reader", "launcher", "foxreader-updater"]
        if build["platform"] == "macos":
            executables.append("Fox Reader.app/Contents/MacOS/launcher")
        if os.name != "nt" and any(not (payload / name).stat().st_mode & 0o111 for name in executables):
            raise InstallError("The release is missing executable permissions on a required program.")
    for name in ("bin", "legal"):
        _plain(payload / name, directory=True)


def _inventory(payload: Path, cancel: Callable[[], None]) -> bytes:
    lines = ["FOXREADER-UPDATER-INVENTORY 1"]
    size = len(lines[0]) + 1
    for directory, folders, files in os.walk(payload, followlinks=False):
        for name in sorted(folders + files):
            cancel()
            path = Path(directory) / name
            relative = path.relative_to(payload).as_posix().encode("utf-8").hex()
            if path.is_symlink():
                lines.append(f"L\t{os.readlink(path).encode('utf-8').hex()}\t{relative}")
            elif path.is_dir():
                lines.append(f"D\t{relative}")
            else:
                lines.append(f"F\t{path.stat().st_size}\t{_sha256(path, cancel)}\t{relative}")
            size += len(lines[-1]) + 1
            if size > MAX_INVENTORY:
                raise InstallError("The prepared update inventory exceeds the size limit.")
    return ("\n".join(lines) + "\n").encode("ascii")


def _copy_bytes(path: Path, cancel: Callable[[], None]) -> int:
    """Count copied bytes without following user font links or reparse points."""
    if not path.exists():
        return 0
    if path.is_file():
        _plain(path)
        return path.stat().st_size
    _plain(path, directory=True)
    size = 0
    for folder, directories, files in os.walk(path, followlinks=False):
        cancel()
        directories[:] = [name for name in directories
                          if not (Path(folder) / name).is_symlink()
                          and not getattr((Path(folder) / name).lstat(), "st_file_attributes", 0) & 0x400]
        for name in files:
            info = (Path(folder) / name).lstat()
            if stat.S_ISREG(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400:
                size += info.st_size
    return size


def prepare_update(root: Path, archive: Path, metadata: dict, build: dict, version: str,
                   cancel: Callable[[], None] = lambda: None) -> PreparedUpdate:
    root = root.resolve()
    _plain(root, directory=True)
    _plain(archive)
    if not archive.resolve().is_relative_to(root / "updates"):
        raise InstallError("The verified archive is outside this installation's update directory.")
    if archive.stat().st_size != metadata["size"] or _sha256(archive, cancel) != metadata["sha256"]:
        raise InstallError("The downloaded archive changed after verification. Download it again.")
    for path in root.iterdir():
        if not TRANSACTION.fullmatch(path.name):
            continue
        _plain(path, directory=True)
        if (path / "journal.bin").exists():
            state = path / "state"
            terminal = False
            if state.exists():
                _plain(state)
                terminal = state.stat().st_size <= 12 and state.read_bytes() in {b"COMMITTED\n", b"ROLLED_BACK\n"}
            if not terminal:
                raise InstallError("An interrupted update needs recovery. Close Fox Reader and open its launcher.")
    transaction = root / f".foxreader-update-{uuid4().hex}"
    transaction.mkdir(mode=0o700)
    payload = transaction / "payload"
    payload.mkdir()
    try:
        opener = zipfile.ZipFile if metadata["format"] == "zip" else tarfile.open
        with opener(archive, "r") as source:
            members = _members(source, build["platform"] == "windows")
            # bin/legal move without copying. Font merging temporarily holds
            # both payload fonts and the merged tree. Launch entries also need
            # durable copied backups so they remain usable after interruption.
            copied = _copy_bytes(root / "fonts", cancel)
            copied += sum(item.size for item in members if item.name.startswith("fonts/"))
            for name in ("launcher", "launcher.exe", "Fox Reader.command"):
                if any(item.name == name for item in members):
                    copied += _copy_bytes(root / name, cancel)
            if build["platform"] == "macos":
                copied += 2 * _copy_bytes(root / "Fox Reader.app", cancel)
            required = sum(item.size for item in members) + copied + 64 * 1024**2
            if shutil.disk_usage(root).free < required:
                raise InstallError(f"Preparing this update needs at least {required / 1024**3:.1f} GiB free.")
            _extract(source, members, payload, cancel)
        _validate_payload(payload, build, version)
        inventory = _inventory(payload, cancel)
        digest = hashlib.sha256(inventory).hexdigest()
        _write_file(transaction / "inventory.txt", inventory)
        _write_file(transaction / "inventory.sha256", (digest + "\n").encode("ascii"))
        runner = transaction / "runner"
        runner.mkdir()
        helper = root / updater_name(build["platform"])
        _plain(helper)
        shutil.copyfile(helper, runner / helper.name)
        with (runner / helper.name).open("r+b") as stream:
            os.fsync(stream.fileno())
        shutil.copystat(helper, runner / helper.name)
        if os.name != "nt":
            # File fsync does not persist directory entries after a power loss.
            for directory, _, _ in os.walk(transaction, topdown=False, followlinks=False):
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        cancel()
        staged_archive = transaction / ("archive.zip" if metadata["format"] == "zip" else "archive.tar.gz")
        if os.name != "nt":
            for directory in (transaction, root):
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        os.replace(archive, staged_archive)
        try:
            archive.parent.rmdir()  # Only remove it when no other downloads remain.
        except OSError:
            pass
        return PreparedUpdate(root, transaction, digest, staged_archive)
    except BaseException:
        shutil.rmtree(transaction, ignore_errors=True)
        raise


def launch_updater(prepared: PreparedUpdate, platform: str, *, timeout: float = 20.0) -> subprocess.Popen:
    """Return only once the authenticated GUI owns the handoff; do not stop the app here."""
    nonce = secrets.token_hex(32)
    token = prepared.transaction / "ticket.token"
    _write_file(token, (nonce + "\n").encode("ascii"))
    try:
        owner = int(os.environ.get("FOX_READER_LAUNCHER_PID", "0"))
    except ValueError:
        owner = 0
    ticket = "\n".join((
        "FOXREADER-UPDATER-1", "install", str(prepared.root), prepared.transaction.name,
        nonce, str(os.getpid()), str(max(0, owner)), prepared.inventory_sha256, "",
    )).encode("utf-8")
    process = None
    try:
        options = ({"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_BREAKAWAY_FROM_JOB}
                   if os.name == "nt" else {"start_new_session": True})
        logs = prepared.root / "logs"
        logs.mkdir(exist_ok=True)
        _plain(logs, directory=True)
        with (logs / "updater.log").open("ab") as errors:
            process = subprocess.Popen(
                [str(prepared.transaction / "runner" / updater_name(platform))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                cwd=prepared.root, close_fds=True, **options,
            )
        process.stdin.write(ticket)
        process.stdin.close()
        ready = threading.Event()
        answer = []

        def read_ready():
            try:
                answer.append(process.stdout.readline(1024))
            except (OSError, ValueError):
                pass
            finally:
                ready.set()

        threading.Thread(target=read_ready, name="fox-updater-handoff", daemon=True).start()
        if not ready.wait(timeout) or not answer or answer[0] != b"READY\n":
            raise InstallError("The updater could not open safely. Fox Reader is still running; see logs/updater.log.")
        process.stdout.close()
        return process
    except BaseException:
        if process is not None:
            # Before READY, this helper has not been allowed to replace files.
            stop_waiting_updater(process)
            if process.stdout:
                process.stdout.close()
        raise
    finally:
        token.unlink(missing_ok=True)


def stop_waiting_updater(process: subprocess.Popen) -> None:
    """Reap our own waiting child before discarding its untouched transaction."""
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
