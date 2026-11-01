"""Run a relocated Mac distribution on real Apple Silicon, without model downloads.

Called by the macOS build workflow. Use --gguf for the Metal GGUF variant.
Only the temporary copy is modified; the staged release remains untouched.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _metal_smoke(root: Path) -> None:
    libdir = root / "bin" / "llama_cpp" / "lib"
    # Keep both handles alive. Checking libllama alone does not prove the
    # separately built Metal library or its embedded shaders work.
    llama = ctypes.CDLL(str(libdir / "libllama.dylib"))
    metal = ctypes.CDLL(str(libdir / "libggml-metal.dylib"))
    base = ctypes.CDLL(str(libdir / "libggml-base.dylib"))
    registry = metal.ggml_backend_metal_reg
    registry.argtypes = []
    registry.restype = ctypes.c_void_p
    device_count = base.ggml_backend_reg_dev_count
    device_count.argtypes = [ctypes.c_void_p]
    device_count.restype = ctypes.c_size_t
    metal_registry = registry()
    if not metal_registry or not device_count(metal_registry):
        raise RuntimeError("the relocated Metal backend found no usable Apple GPU")
    initialize = metal.ggml_backend_metal_init
    initialize.argtypes = []
    initialize.restype = ctypes.c_void_p
    release = base.ggml_backend_free
    release.argtypes = [ctypes.c_void_p]
    release.restype = None
    backend = initialize()
    if not backend:
        raise RuntimeError("the relocated Metal backend could not initialize its embedded shaders")
    release(backend)
    # This also catches a mismatched libllama beside otherwise valid ggml libs.
    initialize_llama = llama.llama_backend_init
    initialize_llama.argtypes = []
    initialize_llama.restype = None
    initialize_llama()
    free_llama = llama.llama_backend_free
    free_llama.argtypes = []
    free_llama.restype = None
    free_llama()


def _get(base: str, path: str, *, timeout: float = 5) -> dict:
    # Do not let a machine's HTTP proxy redirect local probes elsewhere.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{base}{path}", timeout=timeout) as response:
        return json.load(response)


def _run(root: Path, *, gguf: bool, scratch: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # A private config and fresh root avoid attaching to an existing app or
    # downloading models already configured in a developer's stage directory.
    (root / "config" / "fox_config.yaml").write_text(
        f"host: 127.0.0.1\nport: {port}\nmtl_dir: models\ntheme: dark\nlayout: basic\n",
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("DYLD_", "PYTHON", "FOX_READER_", "GGML_"))
           and key != "LLAMA_CPP_LIB_PATH"}
    env["FOX_READER_NO_BROWSER"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    if gguf:
        # dyld reads some environment variables at process startup. A fresh
        # interpreter prevents host library/resource overrides hiding a broken
        # relocated library or missing embedded shader source.
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--native-only"],
            cwd=scratch, env=env, check=True, timeout=120,
        )
    base = f"http://127.0.0.1:{port}"
    log_path = scratch / "launcher.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [str(root / "launcher"), "--no-browser"], cwd=scratch, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        children: set[int] = set()
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if not children and process.poll() is None:
                    child_probe = subprocess.run(
                        ["/usr/bin/pgrep", "-P", str(process.pid)], capture_output=True,
                        text=True, timeout=5,
                    )
                    children.update(int(pid) for pid in child_probe.stdout.split() if pid.isdigit())
                if process.poll() is not None:
                    raise RuntimeError("launcher exited before the relocated backend became ready")
                try:
                    health = _get(base, "/api/health", timeout=2)
                    if health.get("app") == "fox-reader":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError("the relocated backend did not become ready in 120 seconds")

            devices = _get(base, "/api/settings", timeout=15)["devices"]
            if not any(option.get("id") == "mps" and option.get("available")
                       for option in devices["options"]):
                raise RuntimeError(f"the compiled backend cannot use MPS: {devices}")
            if any(devices["resolved"].get(slot) != "mps" for slot in ("paddleocr", "bubble", "textseg")):
                raise RuntimeError(f"the compiled Torch consumers did not select MPS by default: {devices}")
            print("macOS smoke passed: relocated launcher, compiled backend, MPS device settings"
                  + (", Metal shader initialization" if gguf else ""))
        except Exception:
            print(log_path.read_text(encoding="utf-8", errors="replace"))
            raise
        finally:
            if process.poll() is None:
                process.terminate()  # launcher requests orderly backend shutdown
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            # The launcher creates a separate group for its backend. Clean up
            # only children recorded from this test, including a startup crash
            # that would otherwise orphan a backend on macOS.
            for child in children:
                try:
                    os.killpg(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", action="store_true", help="require and initialize bundled Metal llama.cpp")
    parser.add_argument("--native-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, default=(
        Path(__file__).resolve().parents[2] / "packaging" / "build" / "stage" / "fox-reader"
    ))
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        parser.error("this smoke test requires native Apple Silicon macOS with a working GPU")
    source = args.root.resolve()
    if not (source / "launcher").is_file():
        parser.error(f"no staged launcher at {source}")
    if args.native_only:
        _metal_smoke(source)
        return
    with tempfile.TemporaryDirectory(prefix="fox macOS smoke ") as temporary:
        scratch = Path(temporary)
        root = scratch / "relocated Fox Reader"
        # Copy only shipped, immutable content. Existing local settings/models
        # must not alter this first-run smoke or cause a multi-GB weight copy.
        root.mkdir()
        shutil.copy2(source / "launcher", root / "launcher")
        shutil.copytree(source / "bin", root / "bin", symlinks=True)
        for directory in ("config", "models", "fonts", "cache"):
            (root / directory).mkdir()
        _run(root, gguf=args.gguf, scratch=scratch)


if __name__ == "__main__":
    main()
