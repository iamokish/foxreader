"""Unit tests for the Fox Reader downloader components."""

import hashlib
import os
import tarfile
import tempfile
import zipfile
import pytest

import sys

_DOWNLOADER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packaging", "downloader"))
if _DOWNLOADER_DIR not in sys.path:
    sys.path.insert(0, _DOWNLOADER_DIR)

from hardware import (
    detect_os,
    detect_arch,
    HW_RTX_MODERN,
    HW_GTX_LEGACY,
    HW_CPU,
    HARDWARE_CHOICES,
)
from github_client import (
    parse_release_assets,
    parse_sha256sums_text,
    match_recommended_variant,
)
from downloader import (
    check_disk_space,
    check_write_permission,
    format_bytes,
    verify_sha256,
    rejoin_split_parts,
    extract_archive,
)


def test_hardware_detection():
    os_name = detect_os()
    assert os_name in ("windows", "linux")
    arch = detect_arch()
    assert arch in ("x64", "arm64")
    assert len(HARDWARE_CHOICES) == 3
    assert any(code == HW_RTX_MODERN for code, _ in HARDWARE_CHOICES)
    assert any(code == HW_GTX_LEGACY for code, _ in HARDWARE_CHOICES)
    assert any(code == HW_CPU for code, _ in HARDWARE_CHOICES)


def test_parse_release_assets():
    mock_release = {
        "tag_name": "v0.2.1",
        "name": "Fox Reader v0.2.1",
        "assets": [
            {
                "name": "SHA256SUMS.txt",
                "browser_download_url": "https://example.com/SHA256SUMS.txt",
                "size": 500,
            },
            {
                "name": "fox-reader-v0.2.1-cpu-gguf-windows-x64.zip",
                "browser_download_url": "https://example.com/cpu-gguf.zip",
                "size": 300000000,
            },
            {
                "name": "fox-reader-v0.2.1-cu126-windows-x64.zip.001",
                "browser_download_url": "https://example.com/cu126.zip.001",
                "size": 1000000000,
            },
            {
                "name": "fox-reader-v0.2.1-cu126-windows-x64.zip.002",
                "browser_download_url": "https://example.com/cu126.zip.002",
                "size": 500000000,
            },
        ],
    }

    parsed = parse_release_assets(mock_release)
    assert parsed["tag"] == "v0.2.1"
    assert parsed["sha256sums_url"] == "https://example.com/SHA256SUMS.txt"
    assert len(parsed["variants"]) == 2

    # Check split grouping
    cu126_var = next(v for v in parsed["variants"] if v["device"] == "cu126")
    assert cu126_var["is_split"] is True
    assert len(cu126_var["parts"]) == 2
    assert cu126_var["total_size"] == 1500000000
    assert cu126_var["base_name"] == "fox-reader-v0.2.1-cu126-windows-x64.zip"

    # Check single file
    cpu_var = next(v for v in parsed["variants"] if v["device"] == "cpu")
    assert cpu_var["is_split"] is False
    assert cpu_var["has_gguf"] is True
    assert len(cpu_var["parts"]) == 1


def test_parse_sha256sums_text():
    sample_text = """
# fox-reader-v0.2.1-cu126-windows-x64.zip (rejoined)  9f1312ea5c6ad81ec351d65f6de47e1c793dc6bf9b814400a3d60893c1b1fb85
3175f7475235441819bddaf1fa8155384d97555366b4362dbb0a5ae545aefcea  fox-reader-v0.2.1-cpu-gguf-windows-x64.zip
029693d6c6119bab2915d5f4abaf6237c21e026368e851baee85135227523a42  fox-reader-v0.2.1-cu126-windows-x64.zip.001
"""
    sums = parse_sha256sums_text(sample_text)
    assert sums["fox-reader-v0.2.1-cu126-windows-x64.zip"] == "9f1312ea5c6ad81ec351d65f6de47e1c793dc6bf9b814400a3d60893c1b1fb85"
    assert sums["fox-reader-v0.2.1-cpu-gguf-windows-x64.zip"] == "3175f7475235441819bddaf1fa8155384d97555366b4362dbb0a5ae545aefcea"
    assert sums["fox-reader-v0.2.1-cu126-windows-x64.zip.001"] == "029693d6c6119bab2915d5f4abaf6237c21e026368e851baee85135227523a42"


def test_match_recommended_variant():
    mock_variants = [
        {"os": "windows", "device": "cpu", "has_gguf": True, "base_name": "cpu-gguf"},
        {"os": "windows", "device": "cpu", "has_gguf": False, "base_name": "cpu-no-gguf"},
        {"os": "windows", "device": "cu126", "has_gguf": False, "base_name": "cu126-no-gguf"},
        {"os": "windows", "device": "cu118", "has_gguf": False, "base_name": "cu118-no-gguf"},
        {"os": "linux", "device": "cpu", "has_gguf": True, "base_name": "linux-cpu"},
        {"os": "linux", "device": "cu126", "has_gguf": True, "base_name": "linux-cu126-gguf"},
    ]

    # Modern RTX on Windows
    rec_rtx = match_recommended_variant(
        mock_variants, target_os="windows", hardware_choice=HW_RTX_MODERN, prefer_gguf=True
    )
    assert rec_rtx["base_name"] == "cu126-no-gguf"

    # Legacy GTX on Windows
    rec_gtx = match_recommended_variant(
        mock_variants, target_os="windows", hardware_choice=HW_GTX_LEGACY, prefer_gguf=True
    )
    assert rec_gtx["base_name"] == "cu118-no-gguf"

    # CPU on Windows with GGUF
    rec_cpu_gguf = match_recommended_variant(
        mock_variants, target_os="windows", hardware_choice=HW_CPU, prefer_gguf=True
    )
    assert rec_cpu_gguf["base_name"] == "cpu-gguf"

    # CPU on Windows without GGUF
    rec_cpu_no_gguf = match_recommended_variant(
        mock_variants, target_os="windows", hardware_choice=HW_CPU, prefer_gguf=False
    )
    assert rec_cpu_no_gguf["base_name"] == "cpu-no-gguf"

    # Linux OS filtering
    rec_linux = match_recommended_variant(
        mock_variants, target_os="linux", hardware_choice=HW_RTX_MODERN, prefer_gguf=True
    )
    assert rec_linux["base_name"] == "linux-cu126-gguf"


def test_rejoin_split_parts_and_verify():
    with tempfile.TemporaryDirectory() as tmpdir:
        part1_data = b"Hello " * 1000
        part2_data = b"World!" * 1000
        whole_data = part1_data + part2_data

        p1 = os.path.join(tmpdir, "archive.zip.001")
        p2 = os.path.join(tmpdir, "archive.zip.002")
        out = os.path.join(tmpdir, "archive.zip")

        with open(p1, "wb") as f:
            f.write(part1_data)
        with open(p2, "wb") as f:
            f.write(part2_data)

        rejoin_split_parts([p1, p2], out)

        assert os.path.isfile(out)
        with open(out, "rb") as f:
            assert f.read() == whole_data

        expected_hash = hashlib.sha256(whole_data).hexdigest()
        assert verify_sha256(out, expected_hash) is True
        assert verify_sha256(out, "0000000000000000000000000000000000000000000000000000000000000000") is False


def test_extract_archive():
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "test.zip")
        extract_dir = os.path.join(tmpdir, "extracted")

        # Test archive with single root folder (like fox-reader/launcher.exe)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("fox-reader/launcher.exe", b"fake binary")
            zf.writestr("fox-reader/config/settings.json", b'{"key": "value"}')

        extract_archive(zip_path, extract_dir)

        # Root folder should be stripped so files are placed directly in target
        assert os.path.isfile(os.path.join(extract_dir, "launcher.exe"))
        assert os.path.isfile(os.path.join(extract_dir, "config", "settings.json"))


def test_disk_space_and_permissions():
    cwd = os.getcwd()
    assert check_write_permission(cwd) is True
    # Test checking 1 KB
    is_enough, free, req = check_disk_space(cwd, required_bytes=1024)
    assert is_enough is True
    assert free > 1024


def test_format_bytes():
    assert format_bytes(500) == "500 B"
    assert format_bytes(1024) == "1 KB"
    assert format_bytes(236 * 1024 * 1024) == "236.0 MB"
    assert format_bytes(3 * 1024 * 1024 * 1024) == "3.00 GB"
