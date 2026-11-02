"""Apple Silicon build policy and standalone Mach-O validation.

This module is called only by the macOS build. Nothing is loaded from Homebrew
or the build environment to repair an incomplete archive: unresolved libraries
are build errors. Apple system frameworks remain supplied by macOS.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# The locked torch/torchvision wheels require macosx_14_0_arm64.
DEPLOYMENT_TARGET = "14.0"
ALLOW_NO_GPU_ENV = "FOX_BUILD_ALLOW_NO_GPU"
_SYSTEM_ROOTS = ("/usr/lib/", "/System/Library/")
_MACHO_MAGIC = {
    b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca",
}


class MacOSBuildError(RuntimeError):
    """The build cannot produce a portable Apple Silicon distribution."""


def allow_no_gpu(env: dict[str, str] | None = None) -> bool:
    """Opt in to checking compiled GPU support without executing it on a GPU.

    Used by hosted ARM64 CI. This changes verification only, never what gets
    compiled or packaged. A detected GPU must still pass its execution checks.
    """
    return (env if env is not None else os.environ).get(ALLOW_NO_GPU_ENV) == "1"


def _run(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise MacOSBuildError(f"could not run {cmd[0]}: {exc}") from exc
    if result.returncode:
        raise MacOSBuildError(
            f"{' '.join(cmd)} failed (exit {result.returncode}):\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _version(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    if not 1 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        raise MacOSBuildError(f"invalid macOS version: {value!r}")
    return tuple(int(p) for p in (parts + ["0"] * (3 - len(parts))))


def preflight() -> dict[str, str]:
    """Find Apple tools and return a fixed, non-host-tuned build environment."""
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        raise MacOSBuildError("build on an Apple Silicon Mac using native arm64 Python (not Rosetta)")
    if _version(platform.mac_ver()[0]) < _version(DEPLOYMENT_TARGET):
        raise MacOSBuildError(f"the locked dependencies require macOS {DEPLOYMENT_TARGET} or newer")
    try:
        found = {name: _run(["xcrun", "--find", name]) for name in
                 ("clang", "clang++", "otool", "install_name_tool", "lipo", "codesign")}
        sdk = _run(["xcrun", "--sdk", "macosx", "--show-sdk-path"])
        sdk_version = _run(["xcrun", "--sdk", "macosx", "--show-sdk-version"])
    except MacOSBuildError as exc:
        raise MacOSBuildError(f"{exc}\nInstall/select Apple's Command Line Tools: xcode-select --install") from exc
    if _version(sdk_version) < _version(DEPLOYMENT_TARGET):
        raise MacOSBuildError(f"the selected SDK must support macOS {DEPLOYMENT_TARGET} or newer")
    # Override inherited host-specific compiler/linker flags, including -mcpu=native.
    # llama.cpp sets its own explicit M1 CPU baseline in addition to these flags.
    flags = f"-arch arm64 -mmacosx-version-min={DEPLOYMENT_TARGET}"
    return {
        "CC": found["clang"], "CXX": found["clang++"], "SDKROOT": sdk,
        "MACOSX_DEPLOYMENT_TARGET": DEPLOYMENT_TARGET, "ARCHFLAGS": "-arch arm64",
        "CFLAGS": flags, "CXXFLAGS": flags, "CCFLAGS": "", "CPPFLAGS": "", "LDFLAGS": flags,
        "OBJCFLAGS": flags, "OBJCXXFLAGS": flags, "SKBUILD_CMAKE_ARGS": "",
    }


_VENV_PROBE = """\
import ctypes, json, platform, struct, sys
import torch
if (torch.ones(2) + 1).tolist() != [2.0, 2.0]:
    raise RuntimeError('Torch CPU operation failed')
metal = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
default_device = metal.MTLCreateSystemDefaultDevice
default_device.argtypes = []
default_device.restype = ctypes.c_void_p
gpu_available = bool(default_device())
available = torch.backends.mps.is_available()
if available:
    value = (torch.ones(2, device='mps') + 1).cpu().tolist()
    if value != [2.0, 2.0]:
        raise RuntimeError('Torch MPS operation failed')
print(json.dumps({
    'machine': platform.machine(), 'bits': struct.calcsize('P') * 8,
    'python': list(sys.version_info[:2]), 'mps_built': torch.backends.mps.is_built(),
    'mps_available': available,
    'gpu_available': gpu_available,
}))
"""


def verify_environment(python: Path, env: dict[str, str]) -> None:
    """Reject a stale Intel venv or an incorrect torch wheel before compiling."""
    import json

    try:
        result = subprocess.run([str(python), "-c", _VENV_PROBE], env=env,
                                capture_output=True, text=True, timeout=120, check=True)
        state = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        detail = (getattr(exc, "stderr", "") or "").strip()
        raise MacOSBuildError(f"could not validate the Mac build environment: {exc}\n{detail}") from exc
    if state.get("machine") != "arm64" or state.get("bits") != 64 or state.get("python") != [3, 12]:
        raise MacOSBuildError("the build venv must use native arm64 Python 3.12; rebuild with --clean")
    if not state.get("mps_built"):
        raise MacOSBuildError("the macos build requires an MPS-enabled torch wheel")
    if not state.get("mps_available"):
        if state.get("gpu_available") is not False:
            raise MacOSBuildError("MPS is unavailable despite a present or unverified Metal GPU")
        if not allow_no_gpu(env):
            raise MacOSBuildError(
                "MPS is unavailable: use a Mac with GPU access for execution checks, "
                f"or set {ALLOW_NO_GPU_ENV}=1 on a builder without a GPU"
            )
        print("      MPS is built into Torch; this host exposes no MPS device. "
              "CPU execution checked; MPS GPU execution not tested.")
    # Nuitka inherits its deployment target from Python itself. An interpreter
    # built for a newer macOS cannot produce our 14.0 release, even with env set.
    _inspect(python.resolve())


@dataclass
class MachO:
    dependencies: list[str] = field(default_factory=list)
    rpaths: list[str] = field(default_factory=list)
    install_id: str | None = None
    minimum_os: str | None = None


def _load_commands(text: str) -> MachO:
    """Parse arm64 load commands, identifying dylibs by LC_ID, not suffix."""
    info = MachO()
    for block in re.split(r"(?m)^Load command \d+\s*$", text)[1:]:
        command = re.search(r"(?m)^\s*cmd (LC_\w+)\s*$", block)
        if not command:
            continue
        kind = command[1]
        if kind == "LC_RPATH":
            match = re.search(r"(?m)^\s*path (.+) \(offset \d+\)\s*$", block)
            if match:
                info.rpaths.append(match[1])
        elif kind == "LC_ID_DYLIB" or kind.endswith("_DYLIB"):
            match = re.search(r"(?m)^\s*name (.+) \(offset \d+\)\s*$", block)
            if match:
                if kind == "LC_ID_DYLIB":
                    info.install_id = match[1]
                else:
                    info.dependencies.append(match[1])
        elif kind in ("LC_BUILD_VERSION", "LC_VERSION_MIN_MACOSX"):
            if kind == "LC_BUILD_VERSION":
                target = re.search(r"(?m)^\s*platform (\S+)\s*$", block)
                if target is None or target[1].lower() not in ("1", "macos"):
                    raise MacOSBuildError("a bundled arm64 binary targets a platform other than macOS")
            match = re.search(r"(?m)^\s*(?:minos|version) ([\d.]+)\s*$", block)
            if match:
                info.minimum_os = match[1]
    return info


def _is_macho(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(4) in _MACHO_MAGIC


def _inspect(path: Path) -> MachO:
    _run(["xcrun", "lipo", "-verify_arch", "arm64", str(path)])
    info = _load_commands(_run(["xcrun", "otool", "-arch", "arm64", "-l", str(path)]))
    if info.minimum_os is None:
        raise MacOSBuildError(f"{path}: no macOS deployment target recorded")
    if _version(info.minimum_os) > _version(DEPLOYMENT_TARGET):
        raise MacOSBuildError(
            f"{path}: requires macOS {info.minimum_os}, above the {DEPLOYMENT_TARGET} release baseline; "
            "use compatible dependency wheels/Python and rebuild"
        )
    return info


def _expand(name: str, loader: Path, executable: Path) -> Path | None:
    for token, directory in (("@loader_path", loader.parent), ("@executable_path", executable.parent)):
        if name == token or name.startswith(token + "/"):
            return directory / name[len(token):].lstrip("/")
    return Path(name) if name.startswith("/") else None


def _dependency_target(path: Path, name: str, root: Path, infos: dict[Path, MachO],
                       by_name: dict[str, set[Path]] | None = None) -> Path:
    executable = root / "bin" / "fox-reader" if path.parent != root else root / "launcher"
    direct = [_expand(name, path, executable)]
    if name.startswith("@rpath/"):
        direct = []
        # dyld also inherits the executable's rpaths. All rewritten references
        # become direct @loader_path links, so later loads need no search stack.
        for owner in dict.fromkeys((path, executable)):
            for rpath in infos.get(owner, MachO()).rpaths:
                base = _expand(rpath, owner, executable)
                if base is not None:
                    direct.append(base / name.removeprefix("@rpath/"))
    for candidate in direct:
        if candidate is not None:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and resolved in infos:
                return resolved

    # Nuitka moves dependent libraries, while ctypes libraries arrive later.
    # Prefer the package's own directory; otherwise accept only a unique match.
    leaf = name.rsplit("/", 1)[-1]
    local = (path.parent / leaf).resolve()
    if local.is_relative_to(root) and local in infos:
        return local
    if by_name is None:
        by_name = _library_names(root, infos)
    candidates = by_name.get(leaf, set())
    if len(candidates) == 1:
        return next(iter(candidates))
    raise MacOSBuildError(
        f"{path.relative_to(root)} needs {name}: "
        f"{'ambiguous' if candidates else 'missing'} bundled library; refusing a dependency on the build host"
    )


def _library_names(root: Path, infos: dict[Path, MachO]) -> dict[str, set[Path]]:
    names: dict[str, set[Path]] = {}
    for path in root.rglob("*"):
        resolved = path.resolve()
        if resolved.is_relative_to(root) and resolved in infos:
            names.setdefault(path.name, set()).add(resolved)
    return names


def finalize(root: Path) -> None:
    """Audit the finished tree, relocate its dependencies, then ad-hoc sign it."""
    root = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink() and (not path.resolve().is_relative_to(root) or not path.exists()):
            raise MacOSBuildError(f"{path}: dangling symlink or link outside the distribution")
    files = sorted(path for path in root.rglob("*") if _is_macho(path))
    for binary in (root / "launcher", root / "bin" / "fox-reader"):
        if binary not in files:
            raise MacOSBuildError(f"{binary}: missing native executable")
    infos = {path: _inspect(path) for path in files}
    by_name = _library_names(root, infos)
    for path, info in infos.items():
        changes: list[str] = []
        for name in info.dependencies:
            if name.startswith(_SYSTEM_ROOTS):
                continue
            target = _dependency_target(path, name, root, infos, by_name)
            relative = os.path.relpath(target, path.parent).replace(os.sep, "/")
            relocated = f"@loader_path/{relative}"
            if relocated != name:
                changes += ["-change", name, relocated]
        if info.install_id:
            changes += ["-id", f"@loader_path/{path.name}"]
        # Keep internal search paths for libraries opened by name at runtime;
        # discard host paths so dlopen cannot accidentally use Homebrew/the venv.
        for rpath in dict.fromkeys(info.rpaths):
            expanded = _expand(rpath, path, root / "bin" / "fox-reader")
            if expanded is None or not expanded.resolve().is_relative_to(root):
                changes += ["-delete_rpath", rpath]
            elif rpath.startswith("/"):
                relative = os.path.relpath(expanded, path.parent).replace(os.sep, "/")
                changes += ["-rpath", rpath, f"@loader_path/{relative}"]
        if changes:
            _run(["xcrun", "install_name_tool", *changes, str(path)])

    # Check the rewritten commands before signing; all non-system loads must
    # resolve directly inside the tree, with no fallback lookup helping them.
    for path in files:
        actual = _load_commands(_run(["xcrun", "otool", "-arch", "arm64", "-l", str(path)]))
        for name in actual.dependencies:
            if name.startswith(_SYSTEM_ROOTS):
                continue
            target = _expand(name, path, root / "bin" / "fox-reader")
            if not name.startswith("@loader_path/") or target is None or target.resolve() not in infos:
                raise MacOSBuildError(f"{path}: unresolved dependency after relocation: {name}")
    # Ad-hoc signatures are required by Apple Silicon, but are not notarization.
    for path in files:
        _run(["xcrun", "codesign", "--force", "--sign", "-", str(path)])
        _run(["xcrun", "codesign", "--verify", "--strict", str(path)])
    print(f"      macOS: {len(files)} arm64 binaries checked, relocated and ad-hoc signed (macOS {DEPLOYMENT_TARGET}+)")
