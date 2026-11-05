"""SDK download integrity and extraction boundaries; no network or compiler."""
import hashlib
import importlib.util
import io
import os
import stat
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "fox_webview_sdk_tests", Path(__file__).resolve().parents[1] / "packaging/webview_sdk.py"
)
sdk = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sdk
SPEC.loader.exec_module(sdk)


def package(*, malicious=False, duplicate=False, symlink=False, license=True):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        name = "build/native/include/WebView2.h"
        if symlink:
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b"outside")
        else:
            archive.writestr(name, b"// verified header\n" * 100)
        if duplicate:
            archive.writestr(name, b"another header")
        archive.writestr("build/native/x64/WebView2LoaderStatic.lib", b"!<arch>\nloader")
        if license:
            archive.writestr("LICENSE.txt", b"Microsoft WebView2 SDK license fixture")
        archive.writestr("build/native/x64/WebView2Loader.dll", b"must not ship")
        if malicious:
            archive.writestr("../../outside", b"must not escape")
    return buffer.getvalue()


def install_fake_download(monkeypatch, payload, *, bad_digest=False):
    calls = []
    digest = hashlib.sha512(payload if not bad_digest else b"tampered").digest()

    def download(url, path, limit):
        calls.append(url)
        path.write_bytes(payload)

    monkeypatch.delenv("FOX_WEBVIEW2_SDK", raising=False)
    monkeypatch.setattr(sdk, "_download", download)
    monkeypatch.setattr(sdk, "PACKAGE_SHA512", digest)
    return calls


def legacy_cache(tmp_path):
    """An installed cache from before immutable generations were introduced."""
    result = sdk.prepare_sdk(tmp_path)
    legacy = tmp_path / f"webview2-{sdk.VERSION}-x64"
    result.license.parent.rename(legacy)
    return sdk._result(legacy, "x64")


@contextmanager
def deny_windows_delete_sharing(path):
    """Model a compiler/scanner that permits reads but not replacement."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        kernel.CloseHandle(handle)


def test_extracts_only_verified_build_inputs_and_reuses_cache(tmp_path, monkeypatch):
    calls = install_fake_download(monkeypatch, package(malicious=True))
    result = sdk.prepare_sdk(tmp_path)
    assert result.static_library.read_bytes() == b"!<arch>\nloader"
    assert {path.name for path in tmp_path.rglob("*") if path.is_file()} == {
        "WebView2.h", "WebView2LoaderStatic.lib", "LICENSE.txt", "verified.json"
    }
    assert sdk.prepare_sdk(tmp_path) == result
    assert len(calls) == 1
    assert result.license.read_text() == "Microsoft WebView2 SDK license fixture"


def test_rejects_digest_mismatch_without_publishing(tmp_path, monkeypatch):
    install_fake_download(monkeypatch, package(), bad_digest=True)
    with pytest.raises(sdk.WebViewSDKError, match="SHA-512 verification failed"):
        sdk.prepare_sdk(tmp_path)
    assert not list(tmp_path.rglob("WebView2.h"))


@pytest.mark.parametrize("options", [{"duplicate": True}, {"symlink": True}])
def test_rejects_ambiguous_or_link_members(tmp_path, monkeypatch, options):
    with pytest.warns(UserWarning) if options.get("duplicate") else nullcontext():
        payload = package(**options)
    install_fake_download(monkeypatch, payload)
    with pytest.raises(sdk.WebViewSDKError):
        sdk.prepare_sdk(tmp_path)
    assert not list(tmp_path.rglob("verified.json"))


def test_modified_cache_is_reverified_in_a_new_generation(tmp_path, monkeypatch):
    calls = install_fake_download(monkeypatch, package())
    result = sdk.prepare_sdk(tmp_path)
    (result.include / "WebView2.h").write_text("corrupt", encoding="utf-8")
    repaired = sdk.prepare_sdk(tmp_path)
    assert (repaired.include / "WebView2.h").read_bytes().startswith(b"// verified header")
    assert repaired.include != result.include
    assert (result.include / "WebView2.h").read_text() == "corrupt"
    assert len(calls) == 2


def test_invalid_offline_override_fails_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("FOX_WEBVIEW2_SDK", str(tmp_path))
    monkeypatch.setattr(sdk, "_download", lambda *args: pytest.fail("offline override must not download"))
    with pytest.raises(sdk.WebViewSDKError, match="needs WebView2.h"):
        sdk.prepare_sdk(tmp_path / "cache")


def test_architecture_cannot_escape_cache(tmp_path):
    with pytest.raises(sdk.WebViewSDKError, match="Unsupported"):
        sdk.prepare_sdk(tmp_path, "../../outside")


def test_missing_license_rejects_package_before_publication(tmp_path, monkeypatch):
    install_fake_download(monkeypatch, package(license=False))
    with pytest.raises(sdk.WebViewSDKError, match="LICENSE.txt"):
        sdk.prepare_sdk(tmp_path)
    assert not list(tmp_path.rglob("verified.json"))


def test_modified_license_invalidates_cached_sdk(tmp_path, monkeypatch):
    calls = install_fake_download(monkeypatch, package())
    result = sdk.prepare_sdk(tmp_path)
    result.license.write_text("altered")
    repaired = sdk.prepare_sdk(tmp_path)
    assert repaired.license.read_text() == "Microsoft WebView2 SDK license fixture"
    assert repaired.license != result.license
    assert result.license.read_text() == "altered"
    assert len(calls) == 2


def test_valid_readonly_legacy_cache_needs_no_writes_or_downloads(tmp_path, monkeypatch):
    calls = install_fake_download(monkeypatch, package())
    existing = legacy_cache(tmp_path)
    paths = [*sdk._paths(existing.license.parent, "x64"), existing.license.parent / "verified.json"]
    modes = {path: path.stat().st_mode for path in paths}
    original_open = Path.open

    def forbid_writes(path, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+"), f"cached SDK should not be opened for writing: {path}"
        return original_open(path, mode, *args, **kwargs)

    try:
        for path, mode in modes.items():
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        with monkeypatch.context() as guard:
            guard.setattr(Path, "open", forbid_writes)
            guard.setattr(Path, "mkdir", lambda *a, **kw: pytest.fail("cache reuse must not mkdir"))
            guard.setattr(sdk, "_download", lambda *a: pytest.fail("cache reuse must not download"))
            guard.setattr(sdk, "_publish", lambda *a: pytest.fail("cache reuse must not publish"))
            assert sdk.prepare_sdk(tmp_path) == existing
        assert len(calls) == 1
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


@pytest.mark.skipif(os.name != "nt", reason="Windows file handles enforce delete-sharing flags")
@pytest.mark.parametrize("damage", ["missing_marker", "corrupt_header", "corrupt_marker"])
def test_locked_legacy_cache_recovers_without_touching_existing_files(tmp_path, monkeypatch, damage):
    calls = install_fake_download(monkeypatch, package())
    existing = legacy_cache(tmp_path)
    header = existing.include / "WebView2.h"
    marker = existing.license.parent / "verified.json"
    if damage == "missing_marker":
        marker.unlink()
    elif damage == "corrupt_header":
        header.write_text("corrupt header", encoding="utf-8")
    else:
        marker.write_text("invalid metadata", encoding="utf-8")
    locked = marker if damage == "corrupt_marker" else header
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in existing.license.parent.rglob("*") if path.is_file()}
    probe = tmp_path / "replacement-probe"
    probe.write_bytes(b"replacement must fail")
    with deny_windows_delete_sharing(locked):
        # Verify this test really reproduces the Windows replacement failure.
        with pytest.raises(PermissionError):
            os.replace(probe, locked)
        repaired = sdk.prepare_sdk(tmp_path)
        assert repaired.license.parent != existing.license.parent
        assert sdk._cached(repaired.license.parent, "x64") == repaired
        for path, (content, modified) in before.items():
            assert path.read_bytes() == content
            assert path.stat().st_mtime_ns == modified
        if damage == "missing_marker":
            assert not marker.exists()
        assert sdk.prepare_sdk(tmp_path) == repaired
    probe.unlink()
    assert len(calls) == 2


def test_concurrent_publishers_expose_only_complete_verified_generations(tmp_path, monkeypatch):
    calls = install_fake_download(monkeypatch, package())
    barrier = threading.Barrier(2)
    publish = sdk._publish

    def publish_together(staging, destination):
        assert sdk._cached(staging, "x64") is not None
        assert not destination.exists()
        barrier.wait(timeout=10)
        publish(staging, destination)

    monkeypatch.setattr(sdk, "_publish", publish_together)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(sdk.prepare_sdk, tmp_path) for _ in range(2)]
        results = [future.result(timeout=15) for future in futures]
    assert results[0].license.parent != results[1].license.parent
    for result in results:
        assert sdk._cached(result.license.parent, "x64") == result
        assert result.static_library.read_bytes() == b"!<arch>\nloader"
    assert sdk.prepare_sdk(tmp_path) in results
    assert len(calls) == 2
    assert {path for path in tmp_path.iterdir()} == {result.license.parent for result in results}


def test_permanent_publish_denial_is_bounded_and_leaves_no_partial_cache(tmp_path, monkeypatch):
    install_fake_download(monkeypatch, package())
    attempts, delays = [], []

    def reject_publish(staging, destination):
        assert staging.name == "extracted"
        assert sdk._cached(staging, "x64") is not None
        assert not destination.exists()
        attempts.append(destination)
        raise PermissionError(13, "directory rename denied", str(destination))

    monkeypatch.setattr(Path, "rename", reject_publish)
    monkeypatch.setattr(sdk.time, "sleep", delays.append)
    with pytest.raises(sdk.WebViewSDKError, match="Check this folder's access permissions") as failure:
        sdk.prepare_sdk(tmp_path)
    assert isinstance(failure.value.__cause__, PermissionError)
    assert str(tmp_path) in str(failure.value)
    assert "FOX_WEBVIEW2_SDK" in str(failure.value)
    assert 1 < len(attempts) <= 5
    assert len(delays) == len(attempts) - 1
    assert 0 < sum(delays) <= 2
    assert not list(tmp_path.iterdir())
    assert sdk._find_cached(tmp_path, "x64") is None


def test_transient_publish_denial_retries_complete_staging(tmp_path, monkeypatch):
    calls = install_fake_download(monkeypatch, package())
    rename = Path.rename
    attempts, delays = [], []

    def initially_locked(staging, destination):
        assert sdk._cached(staging, "x64") is not None
        attempts.append(destination)
        if len(attempts) <= 2:
            raise PermissionError(13, "temporary scanner lock", str(destination))
        return rename(staging, destination)

    monkeypatch.setattr(Path, "rename", initially_locked)
    monkeypatch.setattr(sdk.time, "sleep", delays.append)
    result = sdk.prepare_sdk(tmp_path)
    assert sdk._cached(result.license.parent, "x64") == result
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert len(delays) == 2 and 0 < sum(delays) <= 2
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == [result.license.parent]


def test_unwritable_cache_reports_actionable_error_before_download(tmp_path, monkeypatch):
    install_fake_download(monkeypatch, package())
    directory = tmp_path / "denied"
    mkdir = Path.mkdir

    def deny_cache(path, *args, **kwargs):
        if path == directory:
            raise PermissionError(13, "cache directory access denied", str(path))
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_cache)
    monkeypatch.setattr(sdk, "_download", lambda *a: pytest.fail("unwritable cache must not download"))
    with pytest.raises(sdk.WebViewSDKError, match="Check this folder's access permissions") as failure:
        sdk.prepare_sdk(directory)
    assert isinstance(failure.value.__cause__, PermissionError)
    assert str(directory) in str(failure.value)
