"""Compile and exercise the shared native recovery policy without a GUI or sleeps."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def recovery_policy(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("fox_recovery_toolchain", root / "packaging/toolchain.py")
    toolchain = importlib.util.module_from_spec(spec)
    # Loading a script by path does not expose its sibling modules. Keep the
    # packaging helpers importable through discovery and compilation (macOS
    # imports macos.py in both), then restore our temporary path/module binding.
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(root / "packaging"))
        imports.setitem(sys.modules, spec.name, toolchain)
        spec.loader.exec_module(toolchain)
        try:
            compiler = toolchain.find_compiler(allow_download=False)
        except toolchain.ToolchainError as exc:
            pytest.skip(f"native recovery checks need a C compiler: {exc}")
        output = tmp_path_factory.mktemp("recovery-policy") / ("policy.exe" if os.name == "nt" else "policy")
        toolchain.build_launcher(output, root / "tests/window_recovery_policy.c", compiler, gui=False)
    return output


@pytest.mark.parametrize("scenario", [
    "backend_wait", "initialization_deadline", "separate_navigation_budget",
    "ready_has_no_deadline", "repeated_events_keep_deadline", "fallback_before_backend_ready",
    "renderer_failure_after_ready", "shutdown_suppresses_recovery",
])
def test_recovery_policy(recovery_policy, scenario):
    result = subprocess.run([str(recovery_policy), scenario], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
