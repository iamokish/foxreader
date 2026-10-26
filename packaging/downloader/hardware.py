"""Platform detection and hardware categories for Fox Reader Downloader.

Detects OS and architecture, and defines clear, user-friendly hardware
categories so users select their build directly without fragile CUDA probing.
"""

from __future__ import annotations

import platform
import sys

# User-selectable hardware categories
HW_RTX_MODERN = "rtx_modern"
HW_GTX_LEGACY = "gtx_legacy"
HW_CPU = "cpu"

HARDWARE_CHOICES = [
    (HW_RTX_MODERN, "NVIDIA GeForce RTX (30 / 40 / 50 Series)"),
    (HW_GTX_LEGACY, "NVIDIA GeForce GTX / RTX 20 (10xx / 16xx / 20xx)"),
    (HW_CPU, "CPU (Standard / No NVIDIA GPU)"),
]


def detect_os() -> str:
    """Return normalized operating system: 'windows' or 'linux'."""
    p = sys.platform
    if p.startswith("win"):
        return "windows"
    if p.startswith("linux"):
        return "linux"
    return p


def detect_arch() -> str:
    """Return normalized architecture ('x64' or 'arm64')."""
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine
