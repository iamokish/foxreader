"""Hardware and system detection for Fox Reader Downloader.

Detects OS, architecture, NVIDIA GPUs, CUDA driver versions, and maps
them to recommended build variants.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys


def detect_os() -> str:
    """Return 'windows', 'linux', or 'macos'."""
    p = sys.platform
    if p.startswith("win"):
        return "windows"
    if p.startswith("linux"):
        return "linux"
    if p == "darwin":
        return "macos"
    return p


def detect_arch() -> str:
    """Return normalized architecture ('x64' or 'arm64')."""
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine


def _query_nvcuda_driver_version() -> tuple[int, int] | None:
    """Try querying driver CUDA version via ctypes nvcuda/libcuda."""
    try:
        if sys.platform.startswith("win"):
            lib = ctypes.windll.LoadLibrary("nvcuda.dll")
        else:
            lib = ctypes.CDLL("libcuda.so.1")

        if lib.cuInit(0) != 0:
            return None

        version = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(version)) != 0:
            return None

        v = version.value
        major = v // 1000
        minor = (v % 1000) // 10
        return (major, minor)
    except Exception:
        return None


def _query_nvidia_smi() -> dict | None:
    """Query nvidia-smi if available."""
    smi = shutil.which("nvidia-smi")
    if not smi and sys.platform.startswith("win"):
        candidates = [
            os.path.expandvars(r"%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
            os.path.expandvars(r"%SystemRoot%\System32\nvidia-smi.exe"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                smi = c
                break

    if not smi:
        return None

    try:
        proc = subprocess.run(
            [smi, "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            name = parts[0] if parts else "NVIDIA GPU"
            driver = parts[1] if len(parts) > 1 else None

            # Also try to extract CUDA version from default nvidia-smi output banner
            cuda_version = None
            banner_proc = subprocess.run(
                [smi],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if banner_proc.returncode == 0:
                match = re.search(r"CUDA Version:\s*(\d+\.\d+)", banner_proc.stdout)
                if match:
                    cuda_version = match.group(1)

            return {
                "name": name,
                "driver": driver,
                "cuda_version": cuda_version,
            }
    except Exception:
        pass
    return None


def _query_windows_wmi_gpu() -> list[str]:
    """Query Windows video controllers using PowerShell as fallback."""
    if not sys.platform.startswith("win"):
        return []
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0:
            return [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    except Exception:
        pass
    return []


def _query_linux_proc_gpu() -> dict | None:
    """Query Linux /proc/driver/nvidia/version as fallback."""
    path = "/proc/driver/nvidia/version"
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # Example: NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.183.01
                match = re.search(r"Kernel Module\s+([\d.]+)", content)
                driver = match.group(1) if match else None
                return {"name": "NVIDIA GPU", "driver": driver, "cuda_version": None}
        except Exception:
            pass
    return None


def detect_system_hardware() -> dict:
    """Inspect the machine's OS, CPU arch, and GPU capabilities.

    Returns a dictionary with:
      - os: 'windows' | 'linux' | 'macos'
      - arch: 'x64' | 'arm64'
      - has_nvidia: bool
      - gpu_name: str
      - driver_version: str | None
      - cuda_version: str | None
      - recommended_device: 'cu130' | 'cu129' | 'cu126' | 'cu118' | 'cpu'
      - details: str
    """
    os_name = detect_os()
    arch = detect_arch()

    has_nvidia = False
    gpu_name = "Integrated / Software"
    driver_version = None
    cuda_version = None

    # Step 1: Try nvidia-smi
    smi_info = _query_nvidia_smi()
    if smi_info:
        has_nvidia = True
        gpu_name = smi_info["name"]
        driver_version = smi_info["driver"]
        cuda_version = smi_info["cuda_version"]

    # Step 2: Try ctypes libcuda / nvcuda for exact CUDA driver version
    if not cuda_version:
        cuda_tuple = _query_nvcuda_driver_version()
        if cuda_tuple:
            has_nvidia = True
            cuda_version = f"{cuda_tuple[0]}.{cuda_tuple[1]}"

    # Step 3: Windows fallback via WMI
    if not has_nvidia and os_name == "windows":
        controllers = _query_windows_wmi_gpu()
        nvidia_cards = [c for c in controllers if "nvidia" in c.lower() or "geforce" in c.lower()]
        if nvidia_cards:
            has_nvidia = True
            gpu_name = nvidia_cards[0]
        elif controllers:
            gpu_name = controllers[0]

    # Step 4: Linux fallback
    if not has_nvidia and os_name == "linux":
        proc_info = _query_linux_proc_gpu()
        if proc_info:
            has_nvidia = True
            gpu_name = proc_info["name"]
            driver_version = proc_info["driver"]

    # Step 5: Map detected hardware to recommended CUDA device
    rec_device = "cpu"
    if has_nvidia:
        # Check CUDA version first if found
        if cuda_version:
            try:
                major, minor = [int(x) for x in cuda_version.split(".")[:2]]
                if major > 13 or (major == 13 and minor >= 0):
                    rec_device = "cu130"
                elif major == 12 and minor >= 9:
                    rec_device = "cu129"
                elif major == 12 and minor >= 6:
                    rec_device = "cu126"
                elif major == 11 and minor >= 8 or (major == 12 and minor < 6):
                    rec_device = "cu118"
                elif major > 11:
                    rec_device = "cu118"
            except Exception:
                pass

        # Fallback to model name heuristics if rec_device is still cpu
        if rec_device == "cpu":
            name_upper = gpu_name.upper()
            if "5090" in name_upper or "5080" in name_upper or "5070" in name_upper or "RTX 50" in name_upper:
                rec_device = "cu130"
            elif "4090" in name_upper or "4080" in name_upper or "4070" in name_upper or "4060" in name_upper or "RTX 40" in name_upper:
                rec_device = "cu126"
            elif "3090" in name_upper or "3080" in name_upper or "3070" in name_upper or "3060" in name_upper or "RTX 30" in name_upper:
                rec_device = "cu126"
            elif any(k in name_upper for k in ["2080", "2070", "2060", "RTX 20", "1660", "1650", "GTX 10", "1080", "1070", "1060"]):
                rec_device = "cu118"
            else:
                # Default modern NVIDIA GPU without known version
                rec_device = "cu126"

    # Friendly details summary
    if has_nvidia:
        cuda_str = f" (CUDA {cuda_version})" if cuda_version else ""
        driver_str = f" [Driver {driver_version}]" if driver_version else ""
        details = f"{gpu_name}{cuda_str}{driver_str} -> Recommended: {rec_device}"
    else:
        details = f"{gpu_name} (No NVIDIA CUDA GPU detected) -> Recommended: CPU"

    return {
        "os": os_name,
        "arch": arch,
        "has_nvidia": has_nvidia,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "recommended_device": rec_device,
        "details": details,
    }
