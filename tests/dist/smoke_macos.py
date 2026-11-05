"""Run a relocated ARM64 Mac distribution without model downloads.

Called by the macOS build workflow. Use --gguf for the Metal GGUF variant.
Only the temporary copy is modified; the staged release remains untouched.
FOX_BUILD_ALLOW_NO_GPU=1 permits a host that exposes no GPU; available GPUs
must still pass their live checks.
"""
from __future__ import annotations

import argparse
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

PACKAGING = Path(__file__).resolve().parents[2] / "packaging"
sys.path.insert(0, str(PACKAGING))

import llamacpp  # noqa: E402
import macos  # noqa: E402


def _metal_smoke(root: Path, env: dict[str, str]) -> dict:
    # The installed-package and relocated-library checks share one probe. It
    # always checks exported APIs, embedded source and CPU initialization, and
    # runs Metal when the framework reports a device. The subprocess runs in
    # a fresh directory with build-host library/resource overrides removed.
    return llamacpp.run_metal_probe(
        Path(sys.executable), env=env, libdir=root / "bin" / "llama_cpp" / "lib",
    )


def _torch_smoke(root: Path, env: dict[str, str], scratch: Path) -> dict:
    # Probe the bundled torch through the compiled entry point. A missing
    # torch library must not pass merely because the app's device discovery
    # catches ImportError and offers CPU-only settings.
    result = subprocess.run(
        [str(root / "bin" / "fox-reader"), "--verify-macos-runtime"],
        cwd=scratch, env=env, capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(
            f"the compiled Torch smoke failed (exit {result.returncode}):\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        state = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError(f"the compiled Torch smoke returned invalid output: {result.stdout}") from exc
    if (not isinstance(state, dict) or state.get("mps_built") is not True or state.get("cpu_ok") is not True
            or not isinstance(state.get("mps_available"), bool)
            or not isinstance(state.get("gpu_available"), bool)):
        raise RuntimeError(f"the compiled Torch smoke did not verify its MPS build and CPU operation: {state}")
    if state["gpu_available"] != state["mps_available"]:
        raise RuntimeError(f"the compiled Torch smoke's MPS availability disagrees with Apple's GPU probe: {state}")
    if state["mps_available"]:
        if state.get("mps_ok") is not True:
            raise RuntimeError(f"the compiled Torch smoke did not exercise its available MPS device: {state}")
    elif not macos.allow_no_gpu(env):
        raise RuntimeError("the compiled Torch smoke found no usable Apple GPU")
    return state


def _get(base: str, path: str, *, timeout: float = 5) -> dict:
    # Do not let a machine's HTTP proxy redirect local probes elsewhere.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{base}{path}", timeout=timeout) as response:
        return json.load(response)


def _clean_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("DYLD_", "PYTHON", "FOX_READER_", "GGML_"))
           and key != "LLAMA_CPP_LIB_PATH"}
    env["FOX_READER_CLI"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    return env


def _check_devices(devices: dict, *, mps_available: bool) -> None:
    """Compare UI placement with the independent compiled Torch smoke result."""
    actual_mps = any(option.get("id") == "mps" and option.get("available")
                     for option in devices["options"])
    if actual_mps != mps_available:
        raise RuntimeError(f"the compiled settings API disagrees with Torch's MPS probe: {devices}")
    expected = "mps" if mps_available else "cpu"
    if not any(option.get("id") == expected and option.get("available") for option in devices["options"]):
        raise RuntimeError(f"the compiled backend has no available {expected} option: {devices}")
    if any(devices["resolved"].get(slot) != expected for slot in ("paddleocr", "bubble", "textseg")):
        raise RuntimeError(f"the compiled Torch consumers did not select {expected} by default: {devices}")


def _write_summary(env: dict[str, str], *, torch_state: dict, metal_state: dict | None) -> None:
    summary = env.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    mps = "Passed" if torch_state["mps_available"] else "Not run: no GPU exposed by this runner"
    if metal_state is None:
        metal_libraries = metal_execution = "Not included (--no-gguf)"
    else:
        metal_libraries = "Passed: library loading, embedded source, CPU initialization"
        metal_execution = "Passed" if metal_state["initialized"] else "Not run: no GPU exposed by this runner"
    with Path(summary).open("a", encoding="utf-8") as output:
        output.write(
            "### macOS runtime validation\n\n"
            "| Check | Result |\n|---|---|\n"
            "| Relocated launcher and settings API | Passed |\n"
            "| Compiled Torch MPS support and CPU tensor operation | Passed |\n"
            f"| MPS tensor execution | {mps} |\n"
            f"| llama.cpp package checks | {metal_libraries} |\n"
            f"| Metal initialization and shader compilation | {metal_execution} |\n\n"
        )


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
    env = _clean_env()
    torch_state = _torch_smoke(root, env, scratch)
    metal_state = _metal_smoke(root, env) if gguf else None
    base = f"http://127.0.0.1:{port}"
    log_path = scratch / "launcher.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [str(root / "launcher"), "--cli"], cwd=scratch, env=env,
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
            _check_devices(devices, mps_available=torch_state["mps_available"])
            checked = "MPS execution and settings" if torch_state["mps_available"] else (
                "CPU execution and settings; GPU execution unavailable on this host"
            )
            print(f"macOS smoke passed: relocated launcher, compiled backend, {checked}")
            _write_summary(env, torch_state=torch_state, metal_state=metal_state)
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
    parser.add_argument("--gguf", action="store_true", help="verify bundled Metal llama.cpp libraries")
    parser.add_argument("--root", type=Path, default=(
        Path(__file__).resolve().parents[2] / "packaging" / "build" / "stage" / "fox-reader"
    ))
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        parser.error("this smoke test requires native ARM64 macOS")
    source = args.root.resolve()
    if not (source / "launcher").is_file():
        parser.error(f"no staged launcher at {source}")
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
