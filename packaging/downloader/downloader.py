"""Download manager, split-file rejoinder, SHA256 verification, and extractor."""

from __future__ import annotations

import hashlib
import os
import queue
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable

CHUNK_SIZE = 256 * 1024  # 256 KB read chunk for fast I/O
USER_AGENT = "FoxReader-Downloader/1.0"
PARALLEL_WORKERS = 8  # Parallel streams per file for 5x-10x faster downloads


def format_bytes(num_bytes: int) -> str:
    """Format byte count into human-readable string (e.g. '225 MB' or '3.20 GB')."""
    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.2f} GB"
    elif num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.1f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes} B"


def check_disk_space(path: str, required_bytes: int = 0) -> tuple[bool, int, int]:
    """Check if the disk containing path has at least required_bytes free.

    Returns (is_sufficient, free_bytes, required_bytes).
    """
    target = os.path.abspath(path)
    while not os.path.exists(target) and os.path.dirname(target) != target:
        target = os.path.dirname(target)

    usage = shutil.disk_usage(target)
    free = usage.free
    return (free >= required_bytes, free, required_bytes)


def check_write_permission(path: str) -> bool:
    """Test if a path can be created/written to."""
    target = os.path.abspath(path)
    try:
        os.makedirs(target, exist_ok=True)
        test_file = os.path.join(target, f".write_test_{os.getpid()}")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception:
        return False


def verify_sha256(file_path: str, expected_digest: str) -> bool:
    """Compute sha256 of file in 1MB chunks and compare against expected_digest."""
    if not os.path.isfile(file_path):
        return False

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)

    actual = hasher.hexdigest().lower()
    return actual == expected_digest.strip().lower()


def _download_single_stream(
    url: str,
    temp_path: str,
    target_path: str,
    expected_size: int,
    progress_callback: Callable[[int, int, float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Fallback single-stream download with resume capability."""
    existing_bytes = 0
    if os.path.isfile(temp_path):
        existing_bytes = os.path.getsize(temp_path)

    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=20)
    except urllib.error.HTTPError as err:
        if err.code == 416:  # Range Not Satisfiable, reset
            existing_bytes = 0
            mode = "wb"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            resp = urllib.request.urlopen(req, timeout=20)
        else:
            raise

    content_range = resp.headers.get("Content-Range")
    if content_range:
        try:
            total_size = int(content_range.split("/")[-1])
        except Exception:
            total_size = expected_size or (existing_bytes + int(resp.headers.get("Content-Length", 0)))
    else:
        total_size = expected_size or int(resp.headers.get("Content-Length", 0))

    bytes_downloaded = existing_bytes
    last_time = time.time()
    last_bytes = bytes_downloaded
    speed = 0.0

    with open(temp_path, mode) as f:
        while True:
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Download was cancelled by user.")

            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break

            f.write(chunk)
            bytes_downloaded += len(chunk)

            now = time.time()
            elapsed = now - last_time
            if elapsed >= 0.4:
                speed = (bytes_downloaded - last_bytes) / elapsed
                last_time = now
                last_bytes = bytes_downloaded
                if progress_callback:
                    progress_callback(bytes_downloaded, total_size, speed)

    if os.path.isfile(target_path):
        os.remove(target_path)
    shutil.move(temp_path, target_path)

    if progress_callback:
        progress_callback(bytes_downloaded, total_size, 0.0)


def _download_parallel_stream(
    direct_url: str,
    temp_path: str,
    target_path: str,
    total_size: int,
    progress_callback: Callable[[int, int, float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """High-speed parallel chunked download using multiple concurrent HTTP Range requests."""
    # Pre-allocate sparse/zeroed output file
    with open(temp_path, "wb") as f:
        f.seek(total_size - 1)
        f.write(b"\0")

    num_workers = min(PARALLEL_WORKERS, max(2, total_size // (2 * 1024 * 1024)))
    seg_size = total_size // num_workers

    write_queue: queue.Queue[tuple[int, bytes] | None] = queue.Queue(maxsize=48)
    bytes_downloaded = 0
    bytes_lock = threading.Lock()
    worker_error: list[Exception] = []

    def writer_thread_func():
        nonlocal bytes_downloaded
        with open(temp_path, "r+b") as out_f:
            while True:
                item = write_queue.get()
                if item is None:
                    write_queue.task_done()
                    break
                offset, data = item
                out_f.seek(offset)
                out_f.write(data)
                with bytes_lock:
                    bytes_downloaded += len(data)
                write_queue.task_done()

    writer = threading.Thread(target=writer_thread_func, daemon=True)
    writer.start()

    def worker_func(idx: int):
        start = idx * seg_size
        end = start + seg_size - 1 if idx < num_workers - 1 else total_size - 1
        curr = start

        for attempt in range(3):
            if cancel_event and cancel_event.is_set():
                return
            try:
                headers = {
                    "User-Agent": USER_AGENT,
                    "Range": f"bytes={curr}-{end}",
                }
                req = urllib.request.Request(direct_url, headers=headers)
                with urllib.request.urlopen(req, timeout=25) as resp:
                    while curr <= end:
                        if cancel_event and cancel_event.is_set():
                            return
                        to_read = min(CHUNK_SIZE, end - curr + 1)
                        chunk = resp.read(to_read)
                        if not chunk:
                            break
                        write_queue.put((curr, chunk))
                        curr += len(chunk)
                if curr > end:
                    return  # Segment completed successfully
            except Exception as exc:
                if cancel_event and cancel_event.is_set():
                    return
                if attempt == 2:
                    worker_error.append(exc)
                time.sleep(1)

    workers = [threading.Thread(target=worker_func, args=(i,), daemon=True) for i in range(num_workers)]
    for w in workers:
        w.start()

    # Progress tracking loop
    last_time = time.time()
    last_bytes = 0
    speed = 0.0

    while any(w.is_alive() for w in workers):
        if cancel_event and cancel_event.is_set():
            write_queue.put(None)
            if os.path.isfile(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise InterruptedError("Download was cancelled by user.")

        if worker_error:
            write_queue.put(None)
            raise RuntimeError(f"Download stream error: {worker_error[0]}")

        now = time.time()
        elapsed = now - last_time
        if elapsed >= 0.3:
            with bytes_lock:
                current_b = bytes_downloaded
            speed = (current_b - last_bytes) / elapsed
            last_time = now
            last_bytes = current_b
            if progress_callback:
                progress_callback(current_b, total_size, speed)

        time.sleep(0.1)

    for w in workers:
        w.join()

    # Finish writing
    write_queue.put(None)
    writer.join()

    if cancel_event and cancel_event.is_set():
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise InterruptedError("Download was cancelled by user.")

    if worker_error:
        raise RuntimeError(f"Download stream error: {worker_error[0]}")

    if os.path.isfile(target_path):
        os.remove(target_path)
    shutil.move(temp_path, target_path)

    if progress_callback:
        progress_callback(total_size, total_size, 0.0)


def download_file(
    url: str,
    target_path: str,
    expected_size: int = 0,
    progress_callback: Callable[[int, int, float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Download a file with multi-threaded parallel streaming for maximum speed.

    Automatically uses parallel Range connections if supported by the server,
    or falls back to single-stream downloading.
    """
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    temp_path = target_path + ".part"

    # If target already exists and size matches
    if expected_size and os.path.isfile(target_path) and os.path.getsize(target_path) == expected_size:
        if progress_callback:
            progress_callback(expected_size, expected_size, 0.0)
        return

    # Probe server for direct CDN URL, size, and Range support
    probe_req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(probe_req, timeout=20) as probe_resp:
            final_url = probe_resp.geturl()
            cl_header = probe_resp.headers.get("Content-Length")
            accept_ranges = probe_resp.headers.get("Accept-Ranges", "")
            remote_size = int(cl_header) if cl_header else expected_size
    except Exception:
        # Fall back directly to single-stream if probe fails
        _download_single_stream(url, temp_path, target_path, expected_size, progress_callback, cancel_event)
        return

    # Use parallel stream if >= 4 MB and ranges supported
    if remote_size >= 4 * 1024 * 1024 and ("bytes" in accept_ranges or "github" in final_url or "s3" in final_url):
        try:
            _download_parallel_stream(final_url, temp_path, target_path, remote_size, progress_callback, cancel_event)
            return
        except InterruptedError:
            raise
        except Exception:
            # Fall back to single-stream on any parallel stream issue
            pass

    _download_single_stream(url, temp_path, target_path, remote_size or expected_size, progress_callback, cancel_event)


def rejoin_split_parts(
    part_paths: list[str],
    output_path: str,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Concatenate split parts (.001, .002, ...) into output_path."""
    total_size = sum(os.path.getsize(p) for p in part_paths if os.path.isfile(p))
    bytes_written = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_output = output_path + ".joining"

    with open(temp_output, "wb") as outfile:
        for part in part_paths:
            with open(part, "rb") as infile:
                while True:
                    if cancel_event and cancel_event.is_set():
                        outfile.close()
                        if os.path.isfile(temp_output):
                            os.remove(temp_output)
                        raise InterruptedError("Rejoining was cancelled by user.")

                    chunk = infile.read(1024 * 1024)
                    if not chunk:
                        break
                    outfile.write(chunk)
                    bytes_written += len(chunk)

                    if progress_callback:
                        progress_callback(bytes_written, total_size)

    if os.path.isfile(output_path):
        os.remove(output_path)
    shutil.move(temp_output, output_path)

    if progress_callback:
        progress_callback(total_size, total_size)


def _safe_extract_zip(
    zip_path: str,
    target_dir: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Extract .zip file safely, stripping any single top-level root directory (e.g. 'fox-reader/')."""
    target_abs = os.path.abspath(target_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total = len(members)

        # Detect if all archive members share a single top-level folder
        root_dir = ""
        first_segments = set()
        for m in members:
            clean = m.filename.replace("\\", "/").strip("/")
            parts = clean.split("/")
            if parts and parts[0]:
                first_segments.add(parts[0])
        if len(first_segments) == 1:
            root_dir = list(first_segments)[0] + "/"

        for idx, member in enumerate(members, 1):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Extraction was cancelled by user.")

            rel_path = member.filename.replace("\\", "/")
            if root_dir and rel_path.startswith(root_dir):
                rel_path = rel_path[len(root_dir):]

            if not rel_path:
                continue

            member_dest = os.path.abspath(os.path.join(target_abs, rel_path))
            # Security check: prevent zip slip
            if not member_dest.startswith(target_abs + os.sep) and member_dest != target_abs:
                continue

            if member.is_dir():
                os.makedirs(member_dest, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(member_dest), exist_ok=True)
                with zf.open(member) as source, open(member_dest, "wb") as target:
                    shutil.copyfileobj(source, target)

            if progress_callback and (idx % 25 == 0 or idx == total):
                progress_callback(idx, total, rel_path)


def _safe_extract_tar(
    tar_path: str,
    target_dir: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Extract .tar.gz safely, stripping any single top-level root directory."""
    target_abs = os.path.abspath(target_dir)

    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        total = len(members)

        root_dir = ""
        first_segments = set()
        for m in members:
            clean = m.name.replace("\\", "/").strip("/")
            parts = clean.split("/")
            if parts and parts[0]:
                first_segments.add(parts[0])
        if len(first_segments) == 1:
            root_dir = list(first_segments)[0] + "/"

        for idx, member in enumerate(members, 1):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Extraction was cancelled by user.")

            rel_path = member.name.replace("\\", "/")
            if root_dir and rel_path.startswith(root_dir):
                rel_path = rel_path[len(root_dir):]

            if not rel_path:
                continue

            member_dest = os.path.abspath(os.path.join(target_abs, rel_path))
            if not member_dest.startswith(target_abs + os.sep) and member_dest != target_abs:
                continue

            if member.isdir():
                os.makedirs(member_dest, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(member_dest), exist_ok=True)
                with tf.extractfile(member) as source:
                    if source:
                        with open(member_dest, "wb") as target:
                            shutil.copyfileobj(source, target)

            if progress_callback and (idx % 25 == 0 or idx == total):
                progress_callback(idx, total, rel_path)


def extract_archive(
    archive_path: str,
    target_dir: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Extract archive (.zip or .tar.gz) into target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    if archive_path.endswith(".zip"):
        _safe_extract_zip(archive_path, target_dir, progress_callback, cancel_event)
    elif archive_path.endswith((".tar.gz", ".tgz")):
        _safe_extract_tar(archive_path, target_dir, progress_callback, cancel_event)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def cleanup_temp_files(files_or_dirs: list[str]) -> None:
    """Safely delete temporary files or directories."""
    for item in files_or_dirs:
        try:
            if os.path.isfile(item):
                os.remove(item)
            elif os.path.isdir(item):
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass
