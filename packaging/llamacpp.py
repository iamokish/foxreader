#!/usr/bin/env python3
"""Builds llama-cpp-python for a distribution, for one device, portably.

The GGUF translators (`fox_reader.translate.machine_translation.gemma_e4b_*`) are
ctypes wrappers around `llama_cpp`, which is a wrapper around a shared library
that does not exist until someone compiles it. PyPI has no wheels for it -- only
an sdist -- so *every* install compiles llama.cpp, and what it compiles for is
decided entirely by ``CMAKE_ARGS``. Nothing else in this build has that property:
torch arrives prebuilt from an index chosen by an extra, and the extra is the
whole decision. Here the extra is only half of it.

Two rules follow from that, and this module exists to enforce them.

**Not native.** llama.cpp's default is ``GGML_NATIVE=ON``, which compiles for the
machine doing the compiling -- ``-march=native``, and for CUDA the architectures
of the cards it can see. That binary then runs on the build machine and crashes
with an illegal instruction, or refuses to load a model, on anything older. A
distribution has to be built for a *baseline*, so every argument list here starts
from ``GGML_NATIVE=OFF`` and names the instruction sets and GPU architectures
explicitly.

**The device is one decision, made twice.** ``--device cu129`` picks a CUDA torch
*and* has to pick a CUDA llama.cpp. Getting the second one wrong is quiet: a
CUDA build with a CPU-only `libllama` loads, answers, and translates at a tenth
of the speed with nothing in the log to say why. So the device drives the CMake
arguments, an argument list that cannot be satisfied is a build failure rather
than a downgrade, and the finished install is asked what it actually built (see
`verify`) before the build carries on.

What gets generated, by device:

  cpu        an explicit x86-64 ISA baseline (AVX2 by default), plus OpenBLAS
             when one can be found or fetched
  cu<NNN>    ``GGML_CUDA=ON`` and an explicit ``CMAKE_CUDA_ARCHITECTURES``,
             version-gated against the toolkit actually installed. Any
             ``cu<NNN>`` extra works, including ones added to pyproject later --
             the version is read out of the name, not from a table.
  rocm*      ``GGML_HIP=ON`` and a broad ``gfx`` list, filtered to what the
             installed ROCm can compile. Builds on Linux, and on Windows with
             the HIP SDK -- but torch ships ROCm wheels for Linux only, so on
             Windows this is a `--gguf-device` and not a `--device`.
  macos      Metal, embedded library, no BLAS (Accelerate is already there).

OpenBLAS goes into every one of those but macos, GPU devices included: their CPU
backend is still there and still runs whatever the device has no kernel for.

Standalone, for looking at what a device would produce without building it::

    python packaging/llamacpp.py --device cu129 --print-args
    python packaging/llamacpp.py --device cpu --openblas auto --print-args
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# Same pipe problem as packaging/build.py: when stdout is not a TTY (CI logs),
# print() block-buffers while the compiles this module drives stream past it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
del _stream

PACKAGING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGING_DIR.parent

#: Scratch space for anything downloaded rather than built. Gitignored (`/nc/`),
#: outside `packaging/build/` on purpose: a 40 MB OpenBLAS is worth keeping
#: across a `--clean`, which would otherwise re-download it every time.
NC_DIR = PROJECT_ROOT / "nc"

#: The distribution installed, and the package it provides.
DISTRIBUTION = "llama-cpp-python"
PACKAGE = "llama_cpp"

#: Where llama-cpp-python puts the shared libraries it built, relative to the
#: package. `llama_cpp/llama_cpp.py` derives this from `__file__` at import, so
#: the compiled build needs the same layout -- which is what `bundle` produces.
LIB_DIR = "lib"

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

#: llama.cpp needs 11.7 for the CUDA backend it has now.
MIN_CUDA = (11, 7)
#: ROCm below this cannot build ggml's HIP backend.
MIN_ROCM = (6, 1)

#: The ISA baselines offered, most to least demanding, with the ggml options each
#: turns on. Everything not named is switched off explicitly rather than left to
#: ggml's defaults: with `GGML_NATIVE=OFF` those defaults are *on* for AVX2 and
#: friends (llama.cpp's `INS_ENB`), which is a fine default for someone building
#: for themselves and not a decision this build should inherit silently.
#:
#: avx2 is the default because it is the floor for practically every x86-64 CPU
#: since 2013 and worth roughly a third of prompt-processing speed. sse42 is the
#: universal one -- anything x86-64 at all -- and is there for a build that has to
#: run on a Nehalem or an Atom.
CPU_BASELINES: dict[str, tuple[str, ...]] = {
    "avx2": ("GGML_SSE42", "GGML_AVX", "GGML_AVX2", "GGML_FMA", "GGML_F16C", "GGML_BMI2"),
    "avx": ("GGML_SSE42", "GGML_AVX"),
    "sse42": ("GGML_SSE42",),
}

#: Every ISA switch ggml has that a portable build must decide about. The ones a
#: baseline does not name are set to OFF.
CPU_ISA_OPTIONS = (
    "GGML_SSE42", "GGML_AVX", "GGML_AVX2", "GGML_AVX_VNNI", "GGML_FMA", "GGML_F16C", "GGML_BMI2",
    "GGML_AVX512", "GGML_AVX512_VBMI", "GGML_AVX512_VNNI", "GGML_AVX512_BF16",
    "GGML_AMX_TILE", "GGML_AMX_INT8", "GGML_AMX_BF16",
)

#: AMD GPU architectures a ROCm build covers. Roughly what PyTorch's own ROCm
#: wheels target, which is the useful definition of "the cards people have":
#: Vega 20 through CDNA 3 on the compute side, RDNA 2 through RDNA 4 on the
#: desktop side. Filtered against the installed ROCm before use -- a gfx the
#: local LLVM has never heard of is a compile error, not a smaller binary.
DEFAULT_HIP_ARCH = (
    "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950",
    "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1151", "gfx1200", "gfx1201",
)

#: OpenBLAS, when the machine has none. The release page is the only place that
#: publishes prebuilt Windows binaries; the version is discovered rather than
#: pinned, and this is the fallback for when GitHub's API cannot be reached.
OPENBLAS_API = "https://api.github.com/repos/OpenMathLib/OpenBLAS/releases/latest"
OPENBLAS_FALLBACK_VERSION = "0.3.34"
OPENBLAS_ASSET = re.compile(r"^OpenBLAS-(?P<version>\d+\.\d+\.\d+)-x64\.zip$")

#: On Linux there are no official binaries, and building OpenBLAS from source
#: needs a Fortran compiler and half an hour. This wheel is what numpy and scipy
#: ship their own builds from: prebuilt, manylinux, headers included.
OPENBLAS_WHEEL = "scipy-openblas32"

#: Header files that prove an include directory is OpenBLAS's and not some other
#: BLAS. ggml includes `cblas.h`; `openblas_config.h` is what makes it OpenBLAS.
OPENBLAS_HEADERS = ("cblas.h", "openblas_config.h")


class GgufError(RuntimeError):
    """Anything that should stop the build with a readable message."""


# ── What device we are building for ───────────────────────────────────────────

@dataclass(frozen=True)
class Target:
    """The device an install is for, parsed out of a `--device` name."""

    #: "cpu", "cuda", "rocm" or "metal".
    kind: str
    #: The name it came from, e.g. "cu129_win".
    device: str
    #: For CUDA, the version the extra asks for. The toolkit on the machine is
    #: found separately and gets the last word -- see `cuda_toolkit`.
    cuda: tuple[int, int] | None = None
    #: For ROCm, the version named by the extra, when it names one.
    rocm: tuple[int, int] | None = None

    @property
    def label(self) -> str:
        if self.kind == "cuda" and self.cuda:
            return f"CUDA {self.cuda[0]}.{self.cuda[1]}"
        if self.kind == "rocm":
            return f"ROCm {self.rocm[0]}.{self.rocm[1]}" if self.rocm else "ROCm"
        return {"cpu": "CPU", "metal": "Metal"}.get(self.kind, self.kind)

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "rocm", "metal")


def _split_version(digits: str) -> tuple[int, int]:
    """`118` -> (11, 8), `130` -> (13, 0), `72` -> (7, 2), `1210` -> (12, 10).

    PyTorch's index names run the major and a one-digit minor together, so the
    split is by position. Four digits is read as a two-digit minor, which is what
    a CUDA 12.10 would have to look like; two digits is a bare major.minor pair,
    which is how the ROCm extras are spelled.
    """
    if not digits.isdigit():
        raise GgufError(f"could not read a version out of {digits!r}")
    if len(digits) <= 2:
        return int(digits[0]), int(digits[1]) if len(digits) > 1 else 0
    if len(digits) == 3:
        return int(digits[:2]), int(digits[2])
    return int(digits[:2]), int(digits[2:])


def parse_device(device: str) -> Target:
    """The device name from `--device`, as something to build against.

    Deliberately pattern-based rather than a lookup table: pyproject gains CUDA
    extras over time, and a new `cu132` should need no change here. Anything
    unrecognised is an error -- guessing CPU for a name meant to be a GPU is the
    silent downgrade this module exists to prevent.
    """
    name = (device or "").strip().lower()
    if not name:
        raise GgufError("no device given")

    if name in ("cpu", "cpu_win"):
        return Target("cpu", name)
    if name in ("macos", "mac", "metal", "darwin"):
        return Target("metal", name)

    cuda = re.fullmatch(r"cu(\d{2,4})(?:[_-].*)?", name)
    if cuda:
        return Target("cuda", name, cuda=_split_version(cuda.group(1)))

    rocm = re.fullmatch(r"(?:rocm|hip)[_-]?(\d{1,4}|\d+\.\d+)?(?:[_-].*)?", name)
    if rocm:
        raw = rocm.group(1)
        version: tuple[int, int] | None = None
        if raw:
            version = _split_version(raw.replace(".", "")) if "." not in raw else (
                int(raw.split(".")[0]), int(raw.split(".")[1])
            )
        return Target("rocm", name, rocm=version)

    raise GgufError(
        f"cannot tell what {device!r} is a device for.\n"
        "  Recognised: cpu, macos, cu<version> (cu118, cu129, cu130, ...), rocm<version>.\n"
        "  A new torch extra in pyproject.toml only needs a name in one of those shapes."
    )


# ── The toolkits, as installed ────────────────────────────────────────────────

def _capture(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 180) -> tuple[int, str]:
    """Run a command for its output. Never raises."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return proc.returncode, proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")


def find_nvcc(prefer: tuple[int, int] | None = None) -> Path | None:
    """The CUDA compiler, from PATH, the usual environment variables, or on disk.

    ``prefer`` is the version wanted, and is a preference and not a filter: an exact
    match wins, then the newest toolkit with the same major version, then the newest
    of any. `prepare` is what refuses a major-version mismatch, with a reason.

    The disk search is not a nicety. Windows machines accumulate toolkits, and both
    of the cheaper answers point at whichever one was installed last: PATH keeps its
    ``bin`` entry after an uninstall, so it can name a directory that is no longer
    there, and ``CUDA_PATH`` is left pointing at an older toolkit -- possibly one
    stripped down to ``include`` and ``lib``. Either way the two toolkits sitting
    beside it go unnoticed, and a machine with CUDA installed reports that it has
    none.
    """
    def version_of(root: Path) -> tuple[int, int] | None:
        match = re.fullmatch(r"v?(\d+)\.(\d+)", root.name)
        return (int(match.group(1)), int(match.group(2))) if match else None

    exe = "nvcc.exe" if IS_WINDOWS else "nvcc"
    found = shutil.which("nvcc")
    if found and Path(found).is_file():
        return Path(found)
    for key in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDAToolkit_ROOT"):
        root = os.environ.get(key)
        candidate = Path(root) / "bin" / exe if root else None
        if candidate is not None and candidate.is_file():
            return candidate

    roots = [Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
             / "NVIDIA GPU Computing Toolkit" / "CUDA"] if IS_WINDOWS else [Path("/usr/local")]
    installed: list[tuple[tuple[int, int], Path]] = []
    for parent in roots:
        try:
            children = list(parent.iterdir())
        except OSError:
            continue
        for child in children:
            candidate = child / "bin" / exe
            version = version_of(child) or version_of(Path(child.name.replace("cuda-", "v")))
            if version is not None and candidate.is_file():
                installed.append((version, candidate))
    if not installed:
        return None
    installed.sort(reverse=True)
    if prefer is not None:
        for wanted in (lambda v: v == prefer, lambda v: v[0] == prefer[0]):
            for version, candidate in installed:
                if wanted(version):
                    return candidate
    return installed[0][1]


def cuda_toolkit(nvcc: Path) -> tuple[int, int] | None:
    """What `nvcc --version` says, as a pair."""
    code, text = _capture([str(nvcc), "--version"])
    if code != 0:
        return None
    match = re.search(r"release\s+(\d+)\.(\d+)", text) or re.search(r"V(\d+)\.(\d+)\.", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


#: The host-compiler gate in CUDA's `crt/host_config.h`, which reads
#: `#if _MSC_VER < 1910 || _MSC_VER >= 1940`. The second number is the ceiling.
_HOST_CONFIG_MSVC = re.compile(r"_MSC_VER\s*<\s*(\d+)\s*\|\|\s*_MSC_VER\s*>=\s*(\d+)")

#: The same gate for GNU, which the same file writes as `#if __GNUC__ > 15`. One
#: number, and it is the last version accepted rather than the first refused.
_HOST_CONFIG_GNU = re.compile(r"__GNUC__\s*>\s*(\d+)")


def nvcc_gnu_limit(nvcc: Path) -> int | None:
    """The newest GCC major this toolkit's nvcc will accept, or None if unreadable.

    The Linux half of `nvcc_msvc_limit`, and it matters for the same reason and
    more often: distributions move GCC faster than CUDA moves its ceiling, so a
    current Fedora or Arch has a compiler nvcc refuses -- CUDA 13.0 stops after
    GCC 15, CUDA 11.8 after GCC 11, and Arch has shipped 15 since 2025. The
    failure is ``#error -- unsupported GNU version!`` on the first ``.cu`` file,
    inside ``enable_language(CUDA)``, before any of llama.cpp is read.
    """
    header = nvcc.parent.parent / "include" / "crt" / "host_config.h"
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _HOST_CONFIG_GNU.search(text)
    return int(match.group(1)) if match else None


def nvcc_msvc_limit(nvcc: Path) -> int | None:
    """The first ``_MSC_VER`` this CUDA toolkit's nvcc refuses, or None if unreadable.

    nvcc will not compile against an MSVC newer than the toolkit was released
    against: CUDA 11.8 stops at ``_MSC_VER`` 1940, so the 14.44 toolset that a
    current Visual Studio installs (1944) fails on the first CUDA file with
    *unsupported Microsoft Visual Studio version* -- during CMake's
    ``enable_language(CUDA)``, before any of llama.cpp is read.

    Read out of the toolkit rather than kept in a table here. The bound moves with
    every CUDA release, the file it is read from is the same file that raises the
    error, and a table would be wrong about the next toolkit either way.
    """
    header = nvcc.parent.parent / "include" / "crt" / "host_config.h"
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _HOST_CONFIG_MSVC.search(text)
    return int(match.group(2)) if match else None


def msvc_toolset_name(msc_ver: int) -> str:
    """The toolset that carries a given ``_MSC_VER``: 1939 -> "14.39".

    For naming the Visual Studio component a user has to install, which is
    versioned by toolset and not by ``_MSC_VER``.
    """
    return f"14.{msc_ver - 1900}"


def find_rocm() -> Path | None:
    """The ROCm root, from the environment or where each platform puts it."""
    for key in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        root = os.environ.get(key)
        if root and Path(root).is_dir():
            return Path(root)
    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        code, text = _capture([hipconfig, "--rocmpath"])
        candidate = text.strip().splitlines()[-1].strip() if code == 0 and text.strip() else ""
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    if IS_WINDOWS:
        # The HIP SDK for Windows, which installs versioned directories and sets
        # HIP_PATH -- but only for shells started after it, so the newest one is
        # worth finding by hand. Sorted by version, not by name: `6.10` must beat
        # `6.4`, and a plain reverse sort puts it second.
        roots = [Path(base) / "AMD" / "ROCm"
                 for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                              os.environ.get("ProgramW6432", r"C:\Program Files"))]
        found: list[tuple[tuple[int, ...], Path]] = []
        for root in roots:
            if not root.is_dir():
                continue
            for entry in root.iterdir():
                if not entry.is_dir():
                    continue
                digits = re.findall(r"\d+", entry.name)
                if digits:
                    found.append((tuple(int(part) for part in digits), entry))
        if found:
            return max(found)[1]
        return None
    for candidate in [Path("/opt/rocm"), *sorted(Path("/opt").glob("rocm-*"), reverse=True)]:
        if candidate.is_dir():
            return candidate
    return None


def _rocm_llvm(root: Path) -> Path | None:
    """ROCm's own clang directory, whichever layout this install uses.

    Linux puts it at ``<root>/llvm/bin``; the Windows HIP SDK puts it at
    ``<root>/lib/llvm/bin``.
    """
    for relative in (("llvm", "bin"), ("lib", "llvm", "bin")):
        candidate = root.joinpath(*relative)
        if candidate.is_dir():
            return candidate
    return None


def rocm_version(root: Path) -> tuple[int, int] | None:
    """ROCm's own version, read from the file it writes for exactly this."""
    for name in (".info/version", ".info/version-dev", ".info/version-hip-libraries"):
        path = root / name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"(\d+)\.(\d+)", text)
        if match:
            return int(match.group(1)), int(match.group(2))
    # Fall back to the directory name (`/opt/rocm-6.4.1`), which is the other
    # place the version is written down.
    match = re.search(r"rocm[-_]?(\d+)\.(\d+)", root.name.lower())
    return (int(match.group(1)), int(match.group(2))) if match else None


# ── GPU architectures ─────────────────────────────────────────────────────────

def cuda_architectures(version: tuple[int, int]) -> list[str]:
    """An explicit `CMAKE_CUDA_ARCHITECTURES` for a portable build.

    Mostly ``-virtual``, deliberately. A ``-real`` entry embeds machine code for
    that architecture and costs several minutes of nvcc per entry; a ``-virtual``
    one embeds PTX, which the driver JITs on first load (once -- it is cached in
    ``~/.nv/ComputeCache`` afterwards) and which is forward compatible, so one
    PTX target covers every card newer than it. So the real entries are the
    architectures people actually have -- Turing, Ampere consumer, Ada, Blackwell
    consumer -- and the rest ride on PTX.

    Version gating matters in both directions: 8.9 and 9.0 did not exist before
    11.8, the ``12Xa`` family arrives with 12.8 and 12.9, and CUDA 13 dropped
    everything below 7.5. Compiling for an architecture the toolkit has never
    heard of is a hard error, and *not* compiling for one is a card that cannot
    run the build at all.
    """
    if version < MIN_CUDA:
        raise GgufError(
            f"CUDA {version[0]}.{version[1]} is too old for llama.cpp's CUDA backend, "
            f"which needs {MIN_CUDA[0]}.{MIN_CUDA[1]} or newer."
        )

    entries: list[tuple[str, str]] = []
    if version >= (13, 0):
        # 5.0 through 7.0 were removed in CUDA 13; 7.5 is the floor.
        entries += [("75", "real"), ("80", "virtual"), ("86", "real")]
    else:
        if version[0] == 11:
            # Maxwell. Deprecated in 12.x loudly enough that it is not worth the
            # warnings there, but a 11.x build is being made for old cards.
            entries.append(("50", "virtual"))
        entries += [("61", "virtual"), ("70", "virtual"), ("75", "real"), ("80", "virtual")]
        if version >= (11, 1):
            entries.append(("86", "real"))
    if version >= (11, 8):
        entries += [("89", "real"), ("90", "virtual")]
    if version >= (12, 8):
        # Blackwell. Family-specific ("a") targets are not forward compatible, so
        # these have to be real -- PTX from 9.0 is what covers anything newer.
        entries.append(("120a", "real"))
    if version >= (12, 9):
        entries.append(("121a", "real"))

    return [f"{arch}-{kind}" for arch, kind in entries]


def _nvcc_known_arch(nvcc: Path) -> set[str]:
    """The `compute_*` targets this nvcc advertises, or nothing if it cannot say."""
    code, text = _capture([str(nvcc), "--list-gpu-arch"])
    if code != 0:
        return set()
    return {match.group(1) for match in re.finditer(r"compute_(\w+)", text)}


def _arch_family(entry: str) -> str:
    """The number in an architecture entry, as `--list-gpu-arch` would spell it.

    ``120a-real`` is a family-specific target: same compute capability, kernels
    that will not run on anything else in the family. nvcc takes the suffix in
    ``-arch``/``-gencode`` and CMake takes it in ``CMAKE_CUDA_ARCHITECTURES``, but
    ``--list-gpu-arch`` prints ``compute_120`` with no suffix at all -- there is
    one architecture, and ``a`` is a compilation mode for it. Comparing the
    unstripped entry against that list therefore drops every ``a``/``f`` target on
    every toolkit that has one, which is to say all of them from 12.8 on. That is
    silent: the build succeeds, `verify` sees a CUDA backend, and the result has
    no Blackwell-consumer kernels in it -- an RTX 50-series card falls back to
    whatever PTX is present, or to the CPU.
    """
    return re.sub(r"[af]$", "", entry.split("-", 1)[0])


def filter_cuda_architectures(entries: list[str], nvcc: Path | None) -> tuple[list[str], list[str]]:
    """Keep the architectures nvcc knows. Returns (kept, dropped).

    The version gating above is a table, and a table is a thing that goes stale.
    This asks the compiler instead, so a toolkit that adds or removes an
    architecture is handled without an edit here -- and so a mistake in the table
    costs a slightly smaller binary rather than a failed build an hour in.
    """
    if nvcc is None:
        return entries, []
    known = _nvcc_known_arch(nvcc)
    if not known:
        return entries, []
    kept, dropped = [], []
    for entry in entries:
        (kept if _arch_family(entry) in known else dropped).append(entry)
    # If the filter would leave nothing, it is the filter that is wrong.
    return (kept, dropped) if kept else (entries, [])


def _llc(root: Path) -> Path | None:
    llvm = _rocm_llvm(root)
    if llvm is None:
        return None
    candidate = llvm / ("llc.exe" if IS_WINDOWS else "llc")
    return candidate if candidate.is_file() else None


def filter_hip_architectures(entries: list[str], root: Path | None) -> tuple[list[str], list[str]]:
    """Keep the `gfx` targets the installed ROCm's LLVM can compile."""
    if root is None:
        return entries, []
    llc = _llc(root)
    if llc is None:
        return entries, []
    code, text = _capture([str(llc), "-march=amdgcn", "-mcpu=help"])
    if code != 0:
        return entries, []
    known = {match.group(0) for match in re.finditer(r"gfx\d+[a-z]*", text)}
    if not known:
        return entries, []
    kept = [entry for entry in entries if entry in known]
    dropped = [entry for entry in entries if entry not in known]
    return (kept, dropped) if kept else (entries, [])


# ── OpenBLAS ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OpenBlas:
    """A usable OpenBLAS: headers, something to link, something to ship."""

    root: Path
    include: Path
    #: The import library (Windows) or shared object (Linux) CMake links against.
    library: Path
    #: What has to travel with the build. The DLL on Windows; on Linux the
    #: shared object and whatever it drags in (see `runtime_files`).
    runtime: tuple[Path, ...]
    label: str

    @property
    def summary(self) -> str:
        return f"{self.label} ({self.library.name})"


def _posix(path: Path) -> str:
    """A path CMake will accept on either platform.

    Forward slashes, always. ``CMAKE_ARGS`` is split with `shlex` before it
    reaches CMake, and in POSIX mode a backslash is an escape character -- so a
    Windows path put in verbatim arrives with its separators eaten.
    """
    return path.as_posix()


def _short_path(path: Path) -> Path:
    """The 8.3 form of a Windows path, when there is one.

    Only used to get rid of spaces. A quoted argument inside ``CMAKE_ARGS``
    survives `shlex` but is at the mercy of how the build backend splits it, and
    a path with no space in it needs no quoting at all.
    """
    if not IS_WINDOWS or " " not in str(path):
        return path
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(1024)
        length = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))  # type: ignore[attr-defined]
        if 0 < length < len(buffer) and " " not in buffer.value:
            return Path(buffer.value)
    except Exception:  # noqa: BLE001 - a best effort; the caller quotes instead
        pass
    return path


def _include_dir(root: Path) -> Path | None:
    """Where OpenBLAS's headers are under `root`, if they are there at all."""
    candidates = [root / "include", root, root / "include" / "openblas"]
    candidates += sorted(root.glob("include/openblas*"))
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / header).is_file() for header in OPENBLAS_HEADERS):
            return candidate
    return None


#: Answered once, and only on Windows. Deciding it means finding the compiler,
#: which for MSVC is a `vcvarsall.bat` subshell.
_GNU_LINKER: list[bool] = []


def _prefers_gnu_libraries() -> bool:
    """Whether the linker this host will use reads GNU import libraries.

    Never raises: a machine with no compiler at all still gets to be told where its
    OpenBLAS is (``--find-openblas``), and the missing compiler is `prepare`'s error
    to report, with a better message than this one could give.
    """
    if not IS_WINDOWS:
        return False
    if not _GNU_LINKER:
        try:
            compiler = _find_host_compiler(required=False)
        except GgufError:
            compiler = None
        _GNU_LINKER.append(compiler is not None and compiler.kind != "msvc")
    return _GNU_LINKER[0]


def _link_library(root: Path, *, gnu: bool | None = None) -> Path | None:
    """The file CMake should link against, in the form this host's linker reads.

    The Windows OpenBLAS release ships both import libraries side by side --
    ``lib/libopenblas.lib`` for MSVC's link.exe and ``lib/libopenblas.dll.a`` for
    MinGW's ld -- so which one is right is a property of the compiler, not of the
    OpenBLAS. Handing ld the MSVC one is a mistake that survives configuration,
    because CMake checks only that the file exists: it becomes an undefined
    reference to `cblas_sgemm` at the point llama.cpp's BLAS backend is linked,
    thousands of lines into a build that already looked like it was working.

    So the preference follows the compiler `_host_toolchain` will actually pass to
    CMake. Both forms stay in the list either way -- an OpenBLAS carrying only one
    of them is still better than no BLAS -- only the order changes.
    """
    if gnu is None:
        gnu = _prefers_gnu_libraries()
    if IS_WINDOWS:
        msvc = ["lib/*openblas*.lib", "*openblas*.lib"]
        mingw = ["lib/*openblas*.dll.a", "*openblas*.dll.a"]
        patterns = [*mingw, *msvc] if gnu else [*msvc, *mingw]
    else:
        patterns = ["lib/lib*openblas*.so", "lib/lib*openblas*.so.*", "lib*openblas*.so",
                    "lib/lib*openblas*.a"]
    for pattern in patterns:
        found = sorted(root.glob(pattern))
        if found:
            return found[0]
    return None


def _runtime_files(root: Path, library: Path) -> tuple[Path, ...]:
    """What has to be copied next to `libllama` for it to load on a clean machine."""
    if IS_WINDOWS:
        return tuple(sorted({*root.glob("bin/*openblas*.dll"), *root.glob("*openblas*.dll")}))

    # The shared object itself, plus the parts of gcc's runtime a distribution's
    # OpenBLAS is linked against and a user's machine may not have. Read out of
    # `ldd` rather than guessed: which of libgfortran, libquadmath and libgomp
    # are needed depends on how the distribution built it, and a missing one
    # means `libllama.so` does not load at all.
    wanted = re.compile(r"lib(openblas|gfortran|quadmath|gomp|scipy_openblas)[.\-]")
    files = {library.resolve()} if library.suffix != ".a" else set()
    code, text = _capture(["ldd", str(library)])
    if code == 0:
        for line in text.splitlines():
            match = re.search(r"=>\s*(\S+)", line)
            if match and wanted.search(Path(match.group(1)).name):
                path = Path(match.group(1))
                if path.is_file():
                    files.add(path.resolve())
    # The SONAME symlink chain matters: `libggml-blas.so` asks for
    # `libopenblas.so.0`, not for whatever `libopenblas.so` points at.
    for path in list(files):
        for sibling in path.parent.glob(path.name.split(".so")[0] + ".so*"):
            if sibling.is_file():
                files.add(sibling.resolve())
    return tuple(sorted(files))


def _as_openblas(root: Path, label: str) -> OpenBlas | None:
    """`root` as an OpenBlas, if it has everything needed."""
    if not root.is_dir():
        return None
    include = _include_dir(root)
    library = _link_library(root)
    if include is None or library is None:
        return None
    runtime = _runtime_files(root, library)
    if IS_WINDOWS and not runtime:
        # An import library with no DLL beside it links and then fails to load.
        return None
    return OpenBlas(root=root, include=include, library=library, runtime=runtime, label=label)


def _pkg_config_openblas() -> OpenBlas | None:
    """What pkg-config knows, which on Linux is usually everything."""
    pkg_config = shutil.which("pkg-config")
    if not pkg_config:
        return None
    for name in ("openblas", "openblas64"):
        code, text = _capture([pkg_config, "--variable=prefix", name])
        prefix = text.strip().splitlines()[0].strip() if code == 0 and text.strip() else ""
        if not prefix:
            continue
        found = _as_openblas(Path(prefix), f"OpenBLAS (pkg-config: {name})")
        if found is not None:
            return found
        # A prefix without headers under it is normal on Debian, where the
        # headers live in a multiarch subdirectory. Ask for the include path.
        code, text = _capture([pkg_config, "--cflags-only-I", name])
        includes = [Path(part[2:]) for part in text.split() if part.startswith("-I")] if code == 0 else []
        code, text = _capture([pkg_config, "--variable=libdir", name])
        libdir = Path(text.strip().splitlines()[0].strip()) if code == 0 and text.strip() else None
        for include in includes:
            if not all((include / header).is_file() for header in OPENBLAS_HEADERS):
                continue
            library = _link_library(libdir) if libdir else None
            if library is None and libdir is not None:
                found_libs = sorted(libdir.glob("libopenblas.so*"))
                library = found_libs[0] if found_libs else None
            if library is not None:
                return OpenBlas(
                    root=library.parent.parent,
                    include=include,
                    library=library,
                    runtime=_runtime_files(library.parent, library),
                    label=f"OpenBLAS (pkg-config: {name})",
                )
    return None


def _openblas_candidates() -> list[tuple[Path, str]]:
    """Where to look, best first."""
    found: list[tuple[Path, str]] = []

    def add(path: Path | None, label: str) -> None:
        if path is not None and path.is_dir():
            found.append((path, label))

    # Something downloaded by an earlier build wins: it is the one this build
    # knows the shape of, and it is already the one that would be shipped.
    for cached in sorted(NC_DIR.glob("openblas/*"), reverse=True):
        add(cached, f"OpenBLAS (nc/{cached.name})")

    for key in ("OPENBLAS_HOME", "OPENBLAS_ROOT", "OpenBLAS_HOME", "OPENBLAS_PATH"):
        value = os.environ.get(key)
        if value:
            add(Path(value), f"OpenBLAS (${key})")

    if IS_WINDOWS:
        add(Path("C:/OpenBLAS"), "OpenBLAS (C:/OpenBLAS)")
        for env in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            root = os.environ.get(env)
            if root:
                add(Path(root) / "OpenBLAS", f"OpenBLAS ({env})")
        vcpkg = os.environ.get("VCPKG_ROOT")
        if vcpkg:
            add(Path(vcpkg) / "installed" / "x64-windows", "OpenBLAS (vcpkg)")
        add(Path("C:/msys64/mingw64"), "OpenBLAS (MSYS2 mingw64)")
        conda = os.environ.get("CONDA_PREFIX")
        if conda:
            add(Path(conda) / "Library", "OpenBLAS (conda)")
    else:
        add(Path("/usr"), "OpenBLAS (/usr)")
        add(Path("/usr/local"), "OpenBLAS (/usr/local)")
        add(Path("/opt/OpenBLAS"), "OpenBLAS (/opt/OpenBLAS)")
        # Debian and Ubuntu: headers in a multiarch openblas-* directory, the
        # library one level up from it.
        for include in sorted(Path("/usr/include").glob("*/openblas*")):
            add(include.parent.parent.parent, "OpenBLAS (multiarch)")
    return found


def find_openblas() -> tuple[OpenBlas | None, list[str]]:
    """The best OpenBLAS on this machine. Returns it and what was passed over."""
    passed_over: list[str] = []
    seen: set[Path] = set()

    if not IS_WINDOWS:
        found = _pkg_config_openblas()
        if found is not None:
            return found, passed_over

    for root, label in _openblas_candidates():
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        found = _as_openblas(root, label)
        if found is not None:
            return found, passed_over
        missing = []
        if _include_dir(root) is None:
            missing.append("no cblas.h/openblas_config.h")
        if _link_library(root) is None:
            missing.append("nothing to link")
        passed_over.append(f"{label}: {', '.join(missing) or 'not usable'}")
    return None, passed_over


def _install_hint() -> str:
    """The package to install, for whichever Linux this is."""
    for manager, command in (
        ("apt-get", "sudo apt install libopenblas-dev"),
        ("dnf", "sudo dnf install openblas-devel"),
        ("pacman", "sudo pacman -S openblas"),
        ("zypper", "sudo zypper install openblas-devel"),
        ("apk", "sudo apk add openblas-dev"),
    ):
        if shutil.which(manager):
            return command
    return "install your distribution's OpenBLAS development package"


def _download(url: str, target: Path) -> None:
    """Fetch `url` to `target`, saying how it is going."""
    print(f"      downloading {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - https, fixed host
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            step = 8 * 1024 * 1024
            next_report = step
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        next_report += step
                        if total:
                            print(f"        {done / 1e6:.0f} of {total / 1e6:.0f} MB")
                        else:
                            print(f"        {done / 1e6:.0f} MB")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise GgufError(f"could not download {url}: {exc}") from exc
    partial.replace(target)


def _safe_extract(archive: Path, into: Path) -> None:
    """Unpack, refusing anything that would write outside `into`."""
    into.mkdir(parents=True, exist_ok=True)
    root = into.resolve()

    def allowed(name: str) -> bool:
        candidate = (root / name).resolve()
        return candidate == root or root in candidate.parents

    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zipped:
            members = [name for name in zipped.namelist() if allowed(name)]
            if len(members) != len(zipped.namelist()):
                raise GgufError(f"{archive.name} contains paths outside the archive root")
            zipped.extractall(into, members=members)
        return

    with tarfile.open(archive) as tarred:
        members = [member for member in tarred.getmembers() if allowed(member.name)]
        if len(members) != len(tarred.getmembers()):
            raise GgufError(f"{archive.name} contains paths outside the archive root")
        tarred.extractall(into, members=members)


def _openblas_release() -> tuple[str, str]:
    """(url, version) for the current Windows x64 build."""
    try:
        with urllib.request.urlopen(OPENBLAS_API, timeout=60) as response:  # noqa: S310 - https, fixed host
            release = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        print(f"      note: could not ask GitHub for the latest OpenBLAS ({exc})")
        version = OPENBLAS_FALLBACK_VERSION
        return (
            f"https://github.com/OpenMathLib/OpenBLAS/releases/download/v{version}/"
            f"OpenBLAS-{version}-x64.zip",
            version,
        )

    for asset in release.get("assets", []):
        # `-x64.zip`, not `-x64-64.zip`: the latter is the ILP64 build, whose
        # 64-bit integer interface is not what ggml's cblas calls expect.
        match = OPENBLAS_ASSET.match(asset.get("name", ""))
        if match and asset.get("browser_download_url"):
            return asset["browser_download_url"], match.group("version")

    raise GgufError(
        "the latest OpenBLAS release has no OpenBLAS-<version>-x64.zip asset.\n"
        f"  Look at {OPENBLAS_API} and download one by hand into {NC_DIR / 'openblas'}."
    )


def download_openblas() -> OpenBlas | None:
    """Fetch OpenBLAS into `nc/`. Returns None if it could not be made usable.

    Not fatal, on purpose. BLAS makes prompt processing faster and changes
    nothing else, so a build that could not get hold of it is a slightly slower
    build rather than no build -- and the reason is printed either way.
    """
    NC_DIR.mkdir(parents=True, exist_ok=True)
    if IS_WINDOWS:
        url, version = _openblas_release()
        root = NC_DIR / "openblas" / f"OpenBLAS-{version}-x64"
        marker = root / ".complete"
        if not marker.is_file():
            archive = NC_DIR / "openblas" / Path(url).name
            if not archive.is_file():
                _download(url, archive)
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            print(f"      unpacking {archive.name}")
            _safe_extract(archive, root)
            marker.write_text(f"{url}\n", encoding="utf-8")
        found = _as_openblas(root, f"OpenBLAS {version} (downloaded)")
        if found is None:
            print(f"      note: {root} is not the layout expected of an OpenBLAS release")
        return found

    # Linux: no official binaries exist, so this borrows the one numpy and scipy
    # build their wheels against. Installed into `nc/` with --target so nothing
    # is added to the build venv, which must stay exactly what uv.lock says.
    uv = shutil.which("uv")
    if not uv:
        print("      note: uv is not on PATH, so OpenBLAS cannot be fetched")
        return None
    root = NC_DIR / "openblas" / OPENBLAS_WHEEL
    if not (root / ".complete").is_file():
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        cmd = [uv, "pip", "install", "--target", str(root), OPENBLAS_WHEEL]
        print(f"      $ {' '.join(cmd)}", flush=True)
        sys.stdout.flush()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"      note: could not install {OPENBLAS_WHEEL} (exit {result.returncode})")
            return None
        (root / ".complete").write_text(f"{OPENBLAS_WHEEL}\n", encoding="utf-8")
    # The wheel unpacks as `scipy_openblas32/{include,lib}`, which is the shape
    # `_as_openblas` already knows how to read.
    for candidate in (root / OPENBLAS_WHEEL.replace("-", "_"), root):
        found = _as_openblas(candidate, f"{OPENBLAS_WHEEL} (downloaded)")
        if found is not None:
            return found
    print(f"      note: {root} does not carry the headers and library expected of {OPENBLAS_WHEEL}")
    return None


def resolve_openblas(mode: str) -> OpenBlas | None:
    """Find OpenBLAS, or fetch it, or do without. Never raises for want of it.

    ``mode`` is ``auto`` (use it if it is here, fetch it into ``nc/`` if not),
    ``download`` (same as ``auto``) or ``off``.

    Deliberately non-interactive: this runs inside ``packaging/build.py``, which
    must also work unattended (CI, scheduled builds), so there is no prompt and
    ``auto`` fetches without asking. ``off`` (or ``--no-blas``) is the only way
    to build without it.
    """
    if mode == "off":
        print("      OpenBLAS: off (--openblas off / --no-blas)")
        print("        building without BLAS. Prompt processing will be slower;")
        print("        nothing else changes.")
        return None

    if mode not in ("auto", "download"):
        print(f"      note: unknown OpenBLAS mode {mode!r}; treating it as 'auto'")
        mode = "auto"

    found, passed_over = find_openblas()
    for line in passed_over:
        print(f"      passed over {line}")
    if found is not None:
        print(f"      OpenBLAS: {found.summary}")
        print(f"        headers {found.include}")
        return found

    print("      OpenBLAS: not found on this machine; fetching it into"
          f" {NC_DIR} (pass --no-blas to build without it).")
    if not IS_WINDOWS:
        print(f"        the packaged one is better than a download: {_install_hint()}")

    try:
        fetched = download_openblas()
    except GgufError as exc:
        print(f"      note: {exc}")
        return None
    if fetched is not None:
        print(f"      OpenBLAS: {fetched.summary}")
    return fetched


# ── The arguments ─────────────────────────────────────────────────────────────

def cmake_args(
    target: Target,
    *,
    openblas: OpenBlas | None = None,
    cpu_baseline: str = "avx2",
    cuda_arch: str | None = None,
    hip_arch: str | None = None,
    toolkit: tuple[int, int] | None = None,
    rocm_root: Path | None = None,
    machine: str | None = None,
    nvcc: Path | None = None,
) -> list[str]:
    """The `-D` arguments for one device. Pure: everything it looks at is a
    parameter, which is what makes it testable without a CUDA install.

    ``toolkit`` is the CUDA version to gate architectures against -- the one
    actually installed, which may be older than the one the extra names.
    """
    if cpu_baseline not in CPU_BASELINES:
        raise GgufError(
            f"unknown CPU baseline {cpu_baseline!r}. One of: {', '.join(CPU_BASELINES)}"
        )
    machine = (machine if machine is not None else platform.machine()).lower()
    is_x86 = machine in ("x86_64", "amd64", "x64", "i386", "i686")

    args = [
        # The whole point. Everything below is here because this is off.
        "-DGGML_NATIVE=OFF",
        # No -DCMAKE_BUILD_TYPE here. llama-cpp-python builds through
        # scikit-build-core, which owns the build type (`cmake.build-type`,
        # Release by default) and logs `Unsupported CMAKE_ARGS ignored` for any
        # attempt to set it from the environment. Passing it anyway would be three
        # warnings in every build log and no effect either way. `install` asks for
        # Release the way scikit-build-core wants to be asked, via
        # `SKBUILD_CMAKE_BUILD_TYPE`.
        # Nothing here runs llama.cpp's own binaries; the python package needs
        # the libraries and nothing else. Skipping them is several minutes.
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_CURL=OFF",
        # Two OpenMP runtimes in one process is a known way to crash or hang, and
        # torch brings its own -- this library is loaded into the same process as
        # torch, always. ggml has its own thread pool, so what is given up is a
        # little prompt-processing speed rather than threading.
        #
        # It also removes a runtime dependency the target machine may not have:
        # libgomp.so.1 on Linux, vcomp140.dll on Windows.
        "-DGGML_OPENMP=OFF",
    ]

    if is_x86:
        wanted = set(CPU_BASELINES[cpu_baseline])
        args += [f"-D{option}={'ON' if option in wanted else 'OFF'}" for option in CPU_ISA_OPTIONS]
    else:
        # arm64 and friends: ggml's own defaults with NATIVE off are already a
        # baseline (armv8-a), and naming x86 options here would only produce
        # warnings.
        print(f"      note: {machine} is not x86-64; leaving the ISA baseline to ggml")

    if target.kind == "cuda":
        version = toolkit or target.cuda
        if version is None:
            raise GgufError(f"{target.device} names no CUDA version")
        entries = cuda_architectures(version)
        if cuda_arch:
            entries = [part.strip() for part in re.split(r"[;,\s]+", cuda_arch) if part.strip()]
        else:
            entries, dropped = filter_cuda_architectures(entries, nvcc)
            if dropped:
                print(f"      note: this toolkit does not know {', '.join(dropped)}; dropped")
        args += [
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(entries)}",
        ]

    elif target.kind == "rocm":
        entries = list(DEFAULT_HIP_ARCH)
        if hip_arch:
            entries = [part.strip() for part in re.split(r"[;,\s]+", hip_arch) if part.strip()]
        else:
            entries, dropped = filter_hip_architectures(entries, rocm_root)
            if dropped:
                print(f"      note: this ROCm cannot compile {', '.join(dropped)}; dropped")
        joined = ";".join(entries)
        args += [
            "-DGGML_HIP=ON",
            # All three spellings: `GPU_TARGETS` is the current one,
            # `AMDGPU_TARGETS` the deprecated one older ROCm still reads, and
            # `CMAKE_HIP_ARCHITECTURES` is what CMake itself uses. An unused one
            # is a warning at configure time and nothing else.
            f"-DGPU_TARGETS={joined}",
            f"-DAMDGPU_TARGETS={joined}",
            f"-DCMAKE_HIP_ARCHITECTURES={joined}",
        ]

    elif target.kind == "metal":
        args += [
            "-DGGML_METAL=ON",
            # Without this the shaders are a separate .metallib file that has to
            # sit next to the library, and nothing in the packaging copies it.
            "-DGGML_METAL_EMBED_LIBRARY=ON",
        ]

    # Not gated on `target.kind`: a GPU backend never replaces ggml's CPU one, it
    # sits beside it, and that is what runs for every op the GPU has no kernel for
    # and for every layer that did not fit in VRAM. So a CUDA or HIP build wants
    # BLAS for exactly the reason a CPU-only build does. `prepare` is where metal is
    # kept out of it -- Apple's Accelerate is already the better BLAS there.
    if openblas is not None:
        args += [
            "-DGGML_BLAS=ON",
            "-DGGML_BLAS_VENDOR=OpenBLAS",
            # BLAS_INCLUDE_DIRS is the important one: ggml only reaches for
            # pkg-config (which it then *requires*) when this is empty, and there
            # is no pkg-config on Windows.
            f"-DBLAS_INCLUDE_DIRS={_posix(_short_path(openblas.include))}",
            f"-DBLAS_LIBRARIES={_posix(_short_path(openblas.library))}",
            f"-DBLAS_ROOT={_posix(_short_path(openblas.root))}",
            f"-DCMAKE_LIBRARY_PATH={_posix(_short_path(openblas.library.parent))}",
        ]
        if not IS_WINDOWS:
            # So the OpenBLAS copied in beside the libraries is the one found at
            # load time, on a machine that has none of its own.
            args += [
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            ]

    return args


def cmake_args_value(args: list[str]) -> str:
    """The arguments as one `CMAKE_ARGS` string.

    Quoted where it has to be. Paths are turned into their space-free form
    first, so quoting is normally not reached at all -- the shape that survives
    every splitter is the one with no spaces in it.
    """
    parts = []
    for arg in args:
        parts.append(f'"{arg}"' if " " in arg else arg)
    return " ".join(parts)


# ── Installing ────────────────────────────────────────────────────────────────

def locked_version() -> str | None:
    """The llama-cpp-python version uv.lock resolved, if it names one.

    Read with a regex rather than a TOML parser for the same reason build.py
    reads the project version that way: this runs on whatever interpreter the
    user has, and the answer is one line.
    """
    lock = PROJECT_ROOT / "uv.lock"
    try:
        text = lock.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(
        r'^\[\[package\]\]\s*^name\s*=\s*"llama-cpp-python"\s*^version\s*=\s*"([^"]+)"',
        text,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def _installed_version(python: Path, env: dict[str, str] | None = None) -> str | None:
    """The version of llama-cpp-python already in `python`'s environment."""
    code, text = _capture(
        [str(python), "-c",
         "import importlib.metadata as m;"
         f"print(m.version({DISTRIBUTION!r}))"],
        env=env,
    )
    return text.strip().splitlines()[-1].strip() if code == 0 and text.strip() else None


@dataclass
class Build:
    """What an install was asked to be, and what it turned out to be."""

    target: Target
    args: list[str]
    value: str
    spec: str
    openblas: OpenBlas | None = None
    #: Environment the compile needs beyond ``CMAKE_ARGS`` -- how CMake finds
    #: nvcc or hipcc. Worked out once, in `prepare`.
    env: dict[str, str] = field(default_factory=dict)
    #: Compiler runtime DLLs the result cannot load without, copied in after the
    #: install. Empty everywhere but a MinGW host -- see `_host_toolchain`.
    runtime_libs: list[Path] = field(default_factory=list)
    system_info: str = ""
    notes: list[str] = field(default_factory=list)


def _jobs(target: Target) -> int:
    """How many compiles to run at once.

    Halved for GPU builds: nvcc and hipcc want a gigabyte or two each, and a
    16-way parallel CUDA build on a 16 GB machine ends in the OOM killer an hour
    in -- which looks like a compiler crash, not like a resource problem.
    """
    count = max(1, os.cpu_count() or 1)
    return max(1, count // 2) if target.is_gpu else count


def _describe_environment(target: Target, nvcc: Path | None, rocm: Path | None) -> dict[str, str]:
    """Environment variables the compile needs, beyond CMAKE_ARGS."""
    env: dict[str, str] = {}
    if target.kind == "rocm" and rocm is not None:
        # What llama.cpp's own HIP instructions set. CMake finds hipcc through
        # these; without them a HIP build configures against the system clang and
        # fails on the first `__global__`.
        env["ROCM_PATH"] = str(rocm)
        env["HIP_PATH"] = str(rocm)
        llvm = _rocm_llvm(rocm)
        if llvm is not None:
            suffix = ".exe" if IS_WINDOWS else ""
            hipcc = llvm / f"clang++{suffix}"
            if hipcc.is_file():
                env["HIPCXX"] = str(hipcc)
            if IS_WINDOWS:
                # llama.cpp's documented Windows HIP build, which differs from the
                # Linux one in both respects: the Visual Studio generator cannot
                # drive a HIP compile, and MSVC cannot compile the device code, so
                # ROCm's own clang is named for both languages. scikit-build-core
                # reads CMAKE_GENERATOR and adds ninja to the build requirements
                # when it is Ninja and there is none on PATH.
                env["CMAKE_GENERATOR"] = "Ninja"
                for variable, tool in (("CC", "clang"), ("CXX", "clang++")):
                    exe = llvm / f"{tool}{suffix}"
                    if exe.is_file():
                        env[variable] = str(exe)
    if target.kind == "cuda" and nvcc is not None:
        # Assigned, not setdefault. Machines accumulate half-removed toolkits --
        # an uninstall that keeps `include` and `lib` while deleting `bin`, with
        # CUDA_PATH and the PATH entry still pointing at it -- and whatever stale
        # value the environment carries must lose to the nvcc that was actually
        # found. Otherwise CMake configures against one toolkit while compiling
        # with another's compiler.
        env["CUDACXX"] = str(nvcc)
        root = nvcc.parent.parent
        if (root / "include").is_dir():
            env["CUDA_PATH"] = str(root)
    return env


#: What MSVC needs to be usable, and nothing else. `Compiler.env` is a whole
#: environment dump taken from a subshell, so merging it wholesale would drag that
#: shell's PYTHONPATH and VIRTUAL_ENV into the compile as well.
_MSVC_VARIABLES = (
    "PATH", "INCLUDE", "LIB", "LIBPATH",
    "VCINSTALLDIR", "VCToolsInstallDir", "VCToolsVersion", "VSINSTALLDIR",
    "WindowsSdkDir", "WindowsSdkBinPath", "WindowsSdkVerBinPath",
    "WindowsSDKLibVersion", "WindowsSDKVersion",
    "UCRTVersion", "UniversalCRTSdkDir",
    "VSCMD_ARG_HOST_ARCH", "VSCMD_ARG_TGT_ARCH",
)


def _msvc_variables(compiler) -> dict[str, str]:
    """Only what cl.exe needs. ``Compiler.env`` is a whole environment dump."""
    source = compiler.env or dict(os.environ)
    return {name: source[name] for name in _MSVC_VARIABLES if name in source}


#: Directories whose contents must not be found by an MSVC build. A machine with
#: MSYS2 or a MinGW toolchain installed has its `cmake`, `ninja`, `gcc`, `cc`, `ar`
#: and `link` on PATH, and CMake takes the first of each it finds -- so an MSVC
#: build configures itself with MSYS2's CMake, which then reports a policy warning
#: about MSVC not being an assembler and looks for a GNU driver it will not find.
#: Matched on the path rather than by name: the offenders are whole toolchains.
_FOREIGN_TOOLCHAIN = ("\\msys64\\", "\\msys32\\", "\\mingw64\\", "\\mingw32\\",
                      "\\ucrt64\\", "\\clang64\\", "\\clang32\\", "\\cygwin64\\",
                      "\\cygwin\\", "\\w64devkit\\", "\\git\\usr\\bin")


def _without_foreign_toolchains(path: str) -> str:
    """`path` with any MinGW, MSYS2 or Cygwin directory removed.

    Only ever applied to the environment handed to an MSVC or CUDA compile. The
    MinGW build wants those directories, and the shell the build was started from
    keeps its own PATH regardless -- this changes what the compiler sees, nothing else.
    """
    kept = [entry for entry in path.split(os.pathsep)
            if entry and not any(tag in entry.lower().replace("/", "\\")
                                 for tag in _FOREIGN_TOOLCHAIN)]
    return os.pathsep.join(kept)


def _make_program(bin_dir: Path) -> Path | None:
    """The make that CMake's ``MinGW Makefiles`` generator will drive.

    It looks for ``mingw32-make`` and nothing else, so a toolchain shipping only
    ``make.exe`` -- w64devkit, MSYS2 -- has to be pointed at it by name.
    """
    for name in ("mingw32-make.exe", "make.exe"):
        candidate = bin_dir / name
        if candidate.is_file():
            return candidate
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


#: Found once per ``max_msc_ver``. Discovering MSVC means running vcvarsall.bat in a
#: subshell, which is three seconds, and `prepare` is not the only thing that asks.
_HOST_COMPILER: dict[int | None, object] = {}


def _find_host_compiler(*, required: bool, max_msc_ver: int | None = None):
    """The compiler `packaging/toolchain.py` would build the launcher with.

    Deliberately the same one: two toolchains in one distribution is two C runtimes.
    ``required`` is False for a HIP build, where the compiler is ROCm's clang and
    this is asked only for the Windows SDK paths that come with MSVC.
    ``max_msc_ver`` asks for an MSVC old enough for a CUDA toolkit to accept.
    """
    if max_msc_ver not in _HOST_COMPILER:
        if str(PACKAGING_DIR) not in sys.path:
            sys.path.insert(0, str(PACKAGING_DIR))
        try:
            from toolchain import ToolchainError, find_compiler
        except ImportError as exc:  # pragma: no cover - packaging/ ships both files
            raise GgufError(
                f"cannot import packaging/toolchain.py to find a compiler ({exc}).\n"
                "  It is what knows where this machine's MSVC or MinGW is.\n"
                "  Or build without the GGUF backend (--no-gguf)."
            ) from exc
        try:
            # No download: Nuitka's private MinGW is a fine last resort for a
            # 300-line launcher, and fetching 80 MB of compiler is not something to
            # do behind a --gguf-device flag.
            _HOST_COMPILER[max_msc_ver] = find_compiler(
                allow_download=False, max_msc_ver=max_msc_ver
            )
        except ToolchainError:
            _HOST_COMPILER[max_msc_ver] = None

    compiler = _HOST_COMPILER[max_msc_ver]
    if compiler is None and required:
        raise GgufError(
            "no C or C++ compiler on this machine to build llama.cpp with.\n"
            '  Install Visual Studio Build Tools with the "Desktop development with C++"\n'
            "  workload, or a MinGW-w64 with gcc on PATH -- w64devkit and MSYS2's\n"
            "  mingw-w64-x86_64-gcc both work. `python packaging/toolchain.py --which`\n"
            "  reports what is found here.\n"
            "  Or build without the GGUF backend (--no-gguf)."
        )
    return compiler


#: The runtime a MinGW-built library loads at startup. Globs rather than exact
#: names because the unwinder's is configuration-dependent (`seh` on x86-64,
#: `dw2` and `sjlj` elsewhere), and a name that does not match would be a
#: missing DLL at import time rather than an error here.
MINGW_RUNTIME = ("libwinpthread-1.dll", "libgcc_s_*.dll", "libstdc++-6.dll")


def _runtime_libraries(bin_dir: Path) -> list[Path]:
    """The MinGW runtime DLLs in `bin_dir`, in the order they depend on each other."""
    found: dict[str, Path] = {}
    for pattern in MINGW_RUNTIME:
        for dll in sorted(bin_dir.glob(pattern)):
            if dll.is_file():
                found.setdefault(dll.name.lower(), dll)
    return list(found.values())


#: A translation unit that needs the C++ *standard library*, not just a C++ compiler.
#: `<string>` and `<vector>` pull in `bits/c++config.h` and the allocator headers,
#: which is where a libstdc++ the compiler cannot reach actually breaks. A C probe
#: says nothing about it: llama.cpp is C++, and a toolchain can compile every C file
#: in it and not one C++ file.
_CXX_PROBE = """\
#include <cstdlib>
#include <string>
#include <vector>
int main() {
    std::vector<std::string> words{"llama"};
    return words[0].empty() ? EXIT_FAILURE : EXIT_SUCCESS;
}
"""

#: `<path>:<line>:<col>: fatal error: <header>: No such file or directory`, which is
#: how both gcc and clang report a header they could not open.
_MISSING_HEADER = re.compile(r"fatal error: (?P<header>[^\s:]+): No such file", re.MULTILINE)

#: Windows' legacy path limit. A compiler built without long-path support opens its
#: headers through the ANSI CRT and cannot see past this, whatever the filesystem
#: and the registry allow.
MAX_PATH = 260

#: Set it to skip the probe. Here for the case where the probe is wrong and the
#: build is not -- an escape hatch, not a supported configuration.
SKIP_PROBE_ENV = "FOX_READER_SKIP_CXX_PROBE"


def _skip_probe() -> bool:
    return os.environ.get(SKIP_PROBE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _cxx_search_dirs(cxx: Path, env: dict[str, str]) -> list[str]:
    """The `#include <...>` directories `cxx` will search, verbatim.

    Verbatim matters: gcc joins a search directory to the header name without
    normalising either, so the path it opens keeps every ``bin/../lib/gcc/../../..``
    segment it was built from and is far longer than the file's real path. Measuring
    the tidied-up version would measure the wrong thing.

    Preprocesses an empty file rather than `os.devnull`, which gcc on Windows reads
    as a relative path called ``nul`` and refuses before printing anything.
    """
    with tempfile.TemporaryDirectory(prefix="fox-cxx-dirs-") as work:
        empty = Path(work) / "empty.cpp"
        empty.write_text("", encoding="utf-8")
        code, text = _capture([str(cxx), "-E", "-x", "c++", "-v", str(empty)], env=env, timeout=60)
    if code != 0 and "search starts here" not in text:
        return []
    dirs: list[str] = []
    collecting = False
    for line in text.splitlines():
        if line.startswith("#include <...>"):
            collecting = True
        elif line.startswith("End of search list"):
            break
        elif collecting and line.startswith(" "):
            dirs.append(line.strip())
    return dirs


def _diagnose_missing_header(cxx: Path, env: dict[str, str], header: str) -> str | None:
    """Why `cxx` cannot open `header`, when the reason is that its path is too long.

    The tell is an asymmetry: this process can stat the file (Python's manifest opts
    it into long paths) at a path the compiler reports as not existing. When that is
    what happened, the length is the whole story and the compiler is fine everywhere
    else -- so it is worth saying, because the alternative reading, a corrupt
    toolchain, sends you off reinstalling one that was never broken.
    """
    for directory in _cxx_search_dirs(cxx, env):
        joined = directory.rstrip("/\\") + "/" + header
        if len(joined) <= MAX_PATH - 1:
            continue
        try:
            exists = Path(joined).is_file()
        except OSError:
            exists = False
        if not exists:
            continue
        return (
            f"its own <{header}> is at a path {len(joined)} characters long, past the "
            f"{MAX_PATH}-character limit\n"
            f"  the compiler reads headers through:\n"
            f"    {joined}\n"
            "  That file is there -- this script just opened it -- so the toolchain is\n"
            "  intact and only its location is the problem. gcc joins its search\n"
            "  directory to the header name without tidying the '..' segments out, so\n"
            "  the path it opens is much longer than the file's real one, and gcc.exe\n"
            "  has no long-path manifest to get past the limit.\n"
            "  Fixes, cheapest first:\n"
            "    * build with MSVC instead -- it has no such limit and is what a\n"
            "      release should be compiled with anyway (Visual Studio Build Tools,\n"
            '      "Desktop development with C++")\n'
            "    * put a MinGW-w64 somewhere short, C:\\mingw64, and have it on PATH\n"
            "    * if this is running inside a sandboxed or virtualised app, run the\n"
            "      build from an ordinary shell: the redirected path prefix is what\n"
            "      pushed this over the limit"
        )
    return None


def _verify_cxx(compiler, cxx: Path, env: dict[str, str]) -> None:
    """Compile one C++ file before committing to compiling twelve hundred.

    `packaging/toolchain.py` already proves the compiler can build a program, but it
    proves it in C, for a launcher written in C. llama.cpp is C++, and the two are
    not the same claim about a toolchain -- a gcc whose libstdc++ it cannot reach
    passes that check and then fails on the first `.cpp`, twenty minutes in, under a
    hundred lines of make output that name a header rather than a cause.

    Raises `GgufError` with what actually went wrong. Everything it can diagnose is a
    property of the machine, not of the build, so it is worth the second it costs.
    """
    if _skip_probe():
        return
    merged = {**os.environ, **env}
    with tempfile.TemporaryDirectory(prefix="fox-cxx-probe-") as work:
        source = Path(work) / "probe.cpp"
        source.write_text(_CXX_PROBE, encoding="utf-8")
        if compiler.kind == "msvc":
            command = [str(cxx), "/nologo", "/EHsc", "/c", str(source), f"/Fo{work}\\probe.obj"]
        else:
            command = [str(cxx), "-c", str(source), "-o", str(Path(work) / "probe.o")]
        code, text = _capture(command, env=merged, timeout=180)
        if code == 0:
            return

    detail = ""
    missing = _MISSING_HEADER.search(text)
    if missing is not None and compiler.kind != "msvc":
        detail = _diagnose_missing_header(cxx, merged, missing.group("header")) or ""
    if not detail:
        excerpt = "\n".join(f"    {line}" for line in text.strip().splitlines()[:12])
        detail = "it cannot compile a three-line C++ program\n" + (excerpt or "    (no output)")
    raise GgufError(
        f"{compiler.label} cannot build llama.cpp: {detail}\n"
        f"  `python packaging/toolchain.py --which` lists what else is on this machine.\n"
        f"  Or build without the GGUF backend (--no-gguf), and set {SKIP_PROBE_ENV}=1 to\n"
        f"  attempt the build anyway."
    )


#: A CUDA translation unit that trips every host-compiler gate there is. ``<vector>``
#: is the entire point of it. nvcc's version check lives in `crt/host_config.h`, but
#: MSVC's C++ library carries a second one in ``yvals_core.h`` pointing the opposite
#: way -- 14.44's refuses any CUDA before 12.4 -- and only a translation unit that
#: includes a real STL header ever reaches it. Every CUDA source in llama.cpp
#: includes considerably more than this.
_CUDA_PROBE = """\
#include <vector>
#include <type_traits>
__global__ void probe_kernel(float *out) { out[threadIdx.x] = 1.0f; }
void probe_call(float *out) {
    probe_kernel<<<1, 32>>>(out);
    std::vector<int> v{1, 2, 3};
    (void)v;
}
"""

#: nvcc refusing the host compiler, as opposed to any other compile failure. Worth
#: telling apart because it is the only one a flag can get past.
_UNSUPPORTED_HOST = re.compile(r"host_config\.h|unsupported .{0,20}(?:Visual Studio|GNU) version",
                               re.IGNORECASE)


def _probe_nvcc(nvcc: Path, compiler, env: dict[str, str], flags: list[str]) -> tuple[int, str]:
    """Compile one .cu the way CMake will, and report how it went."""
    with tempfile.TemporaryDirectory(prefix="fox-cuda-probe-") as work:
        source = Path(work) / "probe.cu"
        source.write_text(_CUDA_PROBE, encoding="utf-8")
        return _capture(
            [str(nvcc), *flags, f"-ccbin={compiler.path}",
             "-c", str(source), "-o", str(Path(work) / "probe.obj")],
            env={**os.environ, **env}, timeout=600,
        )


def _cuda_host_compiler(nvcc: Path, compiler, limit: int | None, env: dict[str, str],
                        notes: list[str]) -> dict[str, str]:
    """Pin nvcc to this exact cl.exe, having proved the two can work together.

    nvcc compiles no host code itself; it hands it to cl.exe. Three things go wrong
    there, and every one of them stops a Windows CUDA build during CMake's
    ``enable_language(CUDA)``, before a line of llama.cpp is read.

    **nvcc chooses its own cl.** Given a ``-ccbin`` under a Visual Studio install it
    runs that install's ``vcvars64.bat`` and then takes whatever cl.exe lands on PATH
    -- the newest toolset present, not the one it was pointed at. So naming one is not
    enough. ``--use-local-env`` is what stops it: it tells nvcc the environment it was
    handed is already correct and to leave it alone. Visual Studio's own CUDA
    integration passes it, and it is what makes ``-ccbin`` mean what it says.

    **nvcc's version gate.** `crt/host_config.h` rejects an MSVC at or above the
    toolkit's ceiling -- *unsupported Microsoft Visual Studio version*.
    ``-allow-unsupported-compiler`` suppresses that check.

    **MSVC's version gate, pointing the other way.** ``yvals_core.h`` rejects a CUDA
    older than the C++ library expects -- 14.44's wants 12.4 or newer, *error STL1002*
    -- and no nvcc flag touches it, because it is MSVC refusing, not nvcc. So the two
    checks can exclude each other outright: CUDA 11.8 wants a toolset no newer than
    14.39, MSVC 14.44 wants a toolkit no older than 12.4, and a machine with only
    those two installed cannot build CUDA at all until one of them changes.

    Which means the override cannot be applied on the strength of a version
    comparison -- it has to be tried. So one .cu is compiled here: unadorned first,
    with the override second and only if nvcc's own gate was the thing that
    complained. If neither works this raises, naming both gates and what to install,
    rather than handing CMake a configuration that dies twenty minutes later under a
    screen of errors from inside ``<type_traits>``.

    The flags travel in ``CUDAFLAGS`` rather than ``-DCMAKE_CUDA_FLAGS``. Both are
    read by the compiler-identification step, which is where the failure happens --
    but a cache entry *replaces* ``CMAKE_CUDA_FLAGS_INIT``, silently dropping the
    ``-D_WINDOWS`` and ``-Xcompiler="/EHsc"`` that CMake's own Windows-NVIDIA-CUDA
    platform file put there, whereas the environment variable is prepended to it.
    """
    # No `-DCMAKE_CUDA_HOST_COMPILER`, deliberately. CMake would turn it into `-ccbin`,
    # and `--use-local-env` makes nvcc compare that against the cl.exe it finds on PATH
    # and refuse the build if the two strings differ -- which they always do here, since
    # CMake shortens the path it passes ("C:/PROGRA~2/MIB055~1/...") and PATH carries the
    # long one. There is nothing to reconcile: `env` below is vcvarsall's own, so PATH
    # already leads to the toolset chosen above, and that is the cl nvcc picks up.
    plain = ["--use-local-env"]
    override = [*plain, "-allow-unsupported-compiler"]

    if _skip_probe():
        # Unprobed, the override is the better guess: without it a toolkit older than
        # the host MSVC cannot get past its own header, and with it a build that was
        # going to work still works.
        return _cudaflags(override)

    code, text = _probe_nvcc(nvcc, compiler, env, plain)
    if code == 0:
        if compiler.toolset:
            notes.append(f"nvcc host compiler: MSVC toolset {compiler.toolset}, picked to "
                         f"stay under this toolkit's _MSC_VER limit of {limit}")
        return _cudaflags(plain)

    if _UNSUPPORTED_HOST.search(text):
        code, text = _probe_nvcc(nvcc, compiler, env, override)
        if code == 0:
            print(f"      note: this CUDA toolkit does not list MSVC {compiler.msc_ver} as "
                  "supported,")
            print("            but the two do compile together, so the build goes ahead with")
            print("            -allow-unsupported-compiler.")
            notes.append(f"MSVC {compiler.msc_ver} is past this toolkit's limit of {limit}; "
                         "-allow-unsupported-compiler, probed working")
            return _cudaflags(override)

    toolkit = cuda_toolkit(nvcc)
    named = f"CUDA {toolkit[0]}.{toolkit[1]}" if toolkit else "this CUDA toolkit"
    excerpt = "\n".join(f"    {line}" for line in text.strip().splitlines()[:8])
    older = ""
    if limit is not None:
        wanted = msvc_toolset_name(limit - 1)
        older = (
            f"  Install an MSVC {named} accepts -- toolset v{wanted} or older:\n"
            "    Visual Studio Installer -> Modify -> Individual components ->\n"
            f'    "MSVC v143 - VS 2022 C++ x64/x86 build tools (v{wanted}-...)"\n'
            "    Several toolsets can be installed side by side, and this picks whichever\n"
            "    one fits -- so adding it costs nothing that is already working.\n"
        )
    raise GgufError(
        f"{named} and {compiler.label} cannot compile a CUDA file together, so a CUDA\n"
        f"  llama.cpp cannot be built on this machine as it stands. nvcc said:\n"
        f"{excerpt}\n"
        f"{older}"
        "  Or install a newer CUDA toolkit -- MSVC's C++ library gates on the toolkit\n"
        "  version too, in the opposite direction, so a current MSVC wants a current CUDA\n"
        "  and raising either one far enough closes the gap.\n"
        "  Or build the CPU variant of the GGUF backend (--gguf-device cpu), or leave it\n"
        f"  out (--no-gguf). {SKIP_PROBE_ENV}=1 attempts the build regardless."
    )


def _cudaflags(flags: list[str]) -> dict[str, str]:
    """``CUDAFLAGS``, with anything the caller's environment already had folded in.

    Folded rather than overwritten: `install` merges this over the environment it was
    given, so a CUDAFLAGS set by whoever started the build would otherwise vanish.
    """
    inherited = os.environ.get("CUDAFLAGS", "").strip()
    return {"CUDAFLAGS": " ".join([*flags, inherited] if inherited else flags)}


def _gcc_major(compiler: str) -> int | None:
    """The major version of a GCC on PATH, or None if it is not a GCC.

    None for clang deliberately, and not because clang cannot build this. nvcc
    keeps a separate ceiling for clang in the same header, and a clang major
    compared against the GCC one is a number that means nothing -- clang 21 and
    GCC 21 are years apart. Returning None leaves a clang host compiler alone,
    which is the right answer until there is a clang ceiling to read as well.
    """
    code, version = _capture([compiler, "--version"])
    if code != 0 or "clang" in version.lower():
        return None
    # `-dumpversion` prints the major alone on GCC 7 and up and `4.8.5` before
    # that; the first number is the major either way.
    code, text = _capture([compiler, "-dumpversion"])
    if code != 0:
        return None
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def _posix_cuda_host_compiler(nvcc: Path, notes: list[str]) -> tuple[dict[str, str], list[str]]:
    """Give nvcc a GCC it will accept, on a machine whose default is too new.

    Same problem as MSVC on Windows and a different shape. nvcc hands host code to
    the system compiler and refuses one newer than its toolkit knew about, and
    Linux distributions move GCC every six months while CUDA moves its ceiling
    once a year -- so the default compiler being unusable is the ordinary case on
    a current distribution, not an edge one.

    Three outcomes, in order of preference. The default compiler is under the
    ceiling and nothing is done. It is not, but a versioned ``g++-N`` under the
    ceiling is installed -- Debian, Ubuntu and Fedora all package several -- and
    that one is named for the whole build, host code included: naming it for nvcc
    alone would put two libstdc++ ABIs in one library, which is a crash while a
    model loads rather than an error here. Or there is nothing suitable, and nvcc
    is told to proceed anyway, which is what upstream llama.cpp tells people to do
    and usually works, with a note saying it was done.
    """
    ceiling = nvcc_gnu_limit(nvcc)
    if ceiling is None:
        return {}, []
    default = _gcc_major("c++") or _gcc_major("g++")
    if default is None:
        notes.append("no g++ found to check against nvcc's ceiling; leaving the host "
                     "compiler to CMake")
        return {}, []
    if default <= ceiling:
        notes.append(f"nvcc accepts GCC up to {ceiling}; this one is {default}")
        return {}, []

    for major in range(ceiling, 7, -1):
        cxx, cc = shutil.which(f"g++-{major}"), shutil.which(f"gcc-{major}")
        if cxx is None or cc is None:
            continue
        notes.append(f"GCC {default} is past nvcc's ceiling of {ceiling}; building with "
                     f"gcc-{major} throughout")
        return {}, [
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
            f"-DCMAKE_CUDA_HOST_COMPILER={cxx}",
        ]

    notes.append(f"GCC {default} is past nvcc's ceiling of {ceiling} and no g++-{ceiling} "
                 "or older is installed; compiling with -allow-unsupported-compiler")
    notes.append(f"install g++-{ceiling} for a build nvidia supports")
    # Through CUDAFLAGS rather than `-DCMAKE_CUDA_FLAGS`, which CMake overwrites
    # with its own per-configuration flags at `enable_language(CUDA)` -- the point
    # where this is needed and the point where that would already be gone.
    return _cudaflags(["-allow-unsupported-compiler"]), []


def _posix_host_toolchain(
    target: Target, nvcc: Path | None
) -> tuple[dict[str, str], list[str], list[str], list[Path]]:
    """What a POSIX build needs told to it, which is nearly nothing.

    ``cc`` and ``c++`` are on PATH, CMake's default generator drives them, and a
    bare ``uv pip install`` compiles -- so unlike Windows there is no generator to
    name, no environment to capture and no runtime to ship. The exception is
    nvcc's host compiler, which is the same problem on both platforms because it
    is nvcc's problem rather than the platform's.
    """
    if target.kind != "cuda" or nvcc is None:
        return {}, [], [], []
    notes: list[str] = []
    env, args = _posix_cuda_host_compiler(nvcc, notes)
    return env, args, notes, []


def _host_toolchain(
    target: Target, nvcc: Path | None = None
) -> tuple[dict[str, str], list[str], list[str], list[Path]]:
    """The generator, the compiler, and the environment those two need here.

    Returns environment variables, extra ``-D`` arguments, notes to print, and any
    runtime DLLs that have to be shipped beside the result.

    Linux and macOS need almost none of it: ``cc`` and ``c++`` are on PATH, CMake's
    default generator drives them, and a bare ``uv pip install`` compiles. The one
    thing they do need is nvcc's host compiler, which is nvcc's problem rather than
    the platform's -- see `_posix_host_toolchain`. Windows is where every part of
    the rest is untrue, and every part fails in the same place -- CMake's
    ``project()`` call, before a source file is read.

    **The generator.** scikit-build-core prefers Ninja and falls back to
    ``NMake Makefiles`` when there is no ninja to be found. nmake exists only inside
    a Visual Studio Developer Command Prompt, so a build started from cmd.exe or
    PowerShell gets a generator whose ``nmake -?`` fails with *no such file or
    directory*, and never reaches a compiler at all. So the generator is named
    rather than guessed: ``MinGW Makefiles`` for gcc, which is what
    llama-cpp-python's own documentation uses, and Ninja for MSVC and for HIP's
    clang.

    **The compiler.** Passed as ``-DCMAKE_C_COMPILER``/``-DCMAKE_CXX_COMPILER``
    rather than left to ``CC``/``CXX``: a Makefiles generator otherwise takes the
    first ``cc`` on PATH, which on a machine with two toolchains need not be the one
    that was verified to work.

    **Their environment.** MSVC cannot simply be put on PATH -- cl.exe finds no
    stdio.h without the ``INCLUDE`` and ``LIB`` that ``vcvarsall.bat`` sets, and
    `packaging/toolchain.py` already runs that batch file for the launcher, so the
    same environment is reused rather than discovered a second way. gcc needs its own
    ``bin`` ahead on PATH instead, for ``mingw32-make`` and for the DLLs it loads
    while it runs.

    **nvcc's host compiler**, which is a third thing again. nvcc does not compile
    host code; it hands it to cl.exe, and refuses outright to hand it to one newer
    than the toolkit knew about. So a CUDA build asks for an MSVC under its toolkit's
    ceiling, names it explicitly so nvcc cannot pick a different one off PATH, and
    then compiles one ``.cu`` to find out whether the pair actually works --
    see `_cuda_host_compiler`.

    None of it depends on the shell the build was started from: the compile is run
    with an environment handed to it, so cmd.exe, PowerShell and a POSIX shell all
    arrive at the same configuration.
    """
    if not IS_WINDOWS:
        return _posix_host_toolchain(target, nvcc)

    env: dict[str, str] = {}
    args: list[str] = []
    notes: list[str] = []
    libs: list[Path] = []

    # What nvcc will accept, before asking for a compiler -- so that a machine with
    # several toolsets installed is configured with one this toolkit can use rather
    # than with the newest and a workaround.
    limit = nvcc_msvc_limit(nvcc) if target.kind == "cuda" and nvcc is not None else None
    compiler = _find_host_compiler(required=target.kind != "rocm", max_msc_ver=limit)
    if target.kind == "cuda" and compiler is not None and compiler.kind != "msvc":
        # Refused rather than warned about. nvcc on Windows drives cl.exe and nothing
        # else, so naming gcc gets as far as `enable_language(CUDA)` and then fails
        # inside CMake's compiler-identification step -- and because nvcc picks a cl
        # off PATH when it is not told which to use, an x86 Developer Prompt hands it
        # the 32-bit one and `cudafe++` dies with an access violation instead of
        # saying what is wrong. Better to say it here.
        raise GgufError(
            f"a CUDA build needs MSVC as nvcc's host compiler, and the compiler found here "
            f"is {compiler.label}.\n"
            "  nvcc on Windows drives cl.exe only. Install the \"Desktop development with\n"
            "  C++\" workload in the Visual Studio Installer if it is not there; if it is,\n"
            "  something is stopping vcvarsall.bat from running -- try a plain cmd.exe\n"
            "  rather than a Developer Command Prompt, which this looks for and reuses.\n"
            "  Or build the CPU variant of the GGUF backend (--gguf-device cpu)."
        )

    # A Windows HIP build is already described: `_describe_environment` names ROCm's
    # clang for both languages with the Ninja generator, which is llama.cpp's
    # documented configuration. What it still wants from the host is MSVC's headers
    # and libraries -- that clang targets `x86_64-pc-windows-msvc` and links the
    # Microsoft C runtime like everything else on the platform. Not fatal when there
    # is no MSVC: the HIP SDK may carry enough of one, and the link says so if not.
    if target.kind == "rocm":
        if compiler is not None and compiler.kind == "msvc":
            env.update(_msvc_variables(compiler))
            env["PATH"] = _without_foreign_toolchains(env.get("PATH", os.environ.get("PATH", "")))
            notes.append(f"CRT headers and libraries from {compiler.label}")
        return env, args, notes, libs

    args.append(f"-DCMAKE_C_COMPILER={_posix(compiler.path)}")
    cxx = compiler.path
    if compiler.kind == "msvc":
        env["CMAKE_GENERATOR"] = "Ninja"
        env.update(_msvc_variables(compiler))
        # vcvarsall inherits this process's PATH and prepends to it, so MSYS2's
        # cmake and gcc are still on the front of what it hands back.
        env["PATH"] = _without_foreign_toolchains(env.get("PATH", os.environ.get("PATH", "")))
        args.append(f"-DCMAKE_CXX_COMPILER={_posix(compiler.path)}")
        if shutil.which("ninja") is None:
            notes.append("no ninja on PATH; the isolated build environment installs one")
        if not (os.environ.get("INCLUDE") and os.environ.get("LIB")):
            notes.append("a hand-run install needs a Developer Command Prompt for cl.exe's "
                         "INCLUDE and LIB; --install sets them itself")
    else:
        plus = {"gcc": "g++", "clang": "clang++"}.get(compiler.kind, "c++")
        candidate = compiler.path.with_name(f"{plus}.exe")
        cxx = candidate if candidate.is_file() else compiler.path
        args.append(f"-DCMAKE_CXX_COMPILER={_posix(cxx)}")
        make = _make_program(compiler.path.parent)
        if make is not None:
            env["CMAKE_GENERATOR"] = "MinGW Makefiles"
            args.append(f"-DCMAKE_MAKE_PROGRAM={_posix(make)}")
        else:
            # No make of any kind, next to the compiler or on PATH. Ninja instead --
            # scikit-build-core installs one rather than going looking.
            env["CMAKE_GENERATOR"] = "Ninja"
            notes.append("no mingw32-make or make found; building with Ninja instead")
        env["PATH"] = os.pathsep.join(
            [str(compiler.path.parent), os.environ.get("PATH", "")]
        )
        # MinGW-w64 gates its headers on _WIN32_WINNT and defaults it below Windows 8,
        # so `CreateFile2` -- which llama.cpp's bundled cpp-httplib calls -- is simply
        # not declared and the build stops there. MSVC's headers have no such gate,
        # which is why upstream never trips over it. 0x0A00 is Windows 10, the oldest
        # Windows that still gets updates.
        winnt = "-DWINVER=0x0A00 -D_WIN32_WINNT=0x0A00"
        env["CFLAGS"] = winnt
        env["CXXFLAGS"] = winnt
        # Not `-static-libgcc -static-libstdc++`, which is the obvious way to make the
        # result loadable on a machine with no MinGW: llama.cpp builds a chain of DLLs,
        # and a static libgcc leaves the first of them exporting `_Unwind_Resume` --
        # ld generates the import library from every symbol in the DLL, unwinder
        # included -- so the next one links both that import library and its own
        # libgcc_eh.a and stops at *multiple definition of `_Unwind_Resume'*.
        # `--allow-multiple-definition` gets past it and is worse: two unwinders and
        # two sets of libstdc++ statics across a DLL boundary, which is a crash while
        # a model loads rather than an error at build time. So the runtime stays shared
        # and the DLLs are shipped instead.
        libs = _runtime_libraries(compiler.path.parent)
        if libs:
            notes.append("ships " + ", ".join(dll.name for dll in libs)
                         + " beside the build (--install copies them)")
        else:
            notes.append(f"no MinGW runtime DLLs found in {compiler.path.parent}; the "
                         "built library will only load where that compiler is on PATH")
        notes.append(f"{compiler.kind} builds llama.cpp here; MSVC is what a release wants")
    # Last, with the environment those two need already in `env`: both probes have to
    # run the compilers the way CMake will, or they prove something about a different
    # configuration. The C++ one first, because it is the cheaper of the two and a
    # compiler that cannot compile C++ has nothing to say about CUDA.
    _verify_cxx(compiler, cxx, env)
    if target.kind == "cuda" and compiler.kind == "msvc" and nvcc is not None:
        env.update(_cuda_host_compiler(nvcc, compiler, limit, env, notes))
    notes.append(f"generator: {env['CMAKE_GENERATOR']}, compiler: {compiler.label}")
    return env, args, notes, libs


def prepare(
    target: Target,
    *,
    openblas_mode: str = "auto",
    cpu_baseline: str = "avx2",
    cuda_arch: str | None = None,
    hip_arch: str | None = None,
) -> Build:
    """Work out the arguments, checking that this machine can honour them.

    Raises rather than falling back. A CUDA distribution whose llama.cpp is
    CPU-only is the failure this whole module is arranged around: it works, it is
    ten times slower, and nothing says so.

    Everything is compiled here, from the sdist, for exactly this device -- there
    is deliberately no prebuilt-wheel path. A wheel's linkage is its publisher's
    choice, and a statically linked CUDA backend makes the whole package
    unimportable without a driver, CPU included. Compiling keeps the one
    property a distribution needs: a missing driver degrades to CPU instead.
    """
    nvcc: Path | None = None
    rocm: Path | None = None
    toolkit: tuple[int, int] | None = None
    notes: list[str] = []

    if target.kind == "cuda":
        nvcc = find_nvcc(target.cuda)
        if nvcc is None:
            raise GgufError(
                f"--device {target.device} wants a CUDA llama.cpp, and there is no nvcc on this "
                "machine.\n"
                "  torch arrives prebuilt, but llama-cpp-python is compiled here, so a CUDA\n"
                "  build needs the CUDA Toolkit installed:\n"
                "      https://developer.nvidia.com/cuda-downloads\n"
                "  Or build without the GGUF backend (--no-gguf), or build its CPU variant\n"
                "  into this distribution deliberately (--gguf-device cpu)."
            )
        toolkit = cuda_toolkit(nvcc)
        print(f"      nvcc: {nvcc}" + (f"  (CUDA {toolkit[0]}.{toolkit[1]})" if toolkit else ""))
        if toolkit is None:
            notes.append(f"could not read a version out of {nvcc}; gating on {target.device}'s")
            toolkit = target.cuda
        elif target.cuda and toolkit[0] != target.cuda[0]:
            # The libraries ggml links -- cublas, cudart -- are the ones torch
            # ships in the same tree, and those are compatible within a major
            # version and not across one.
            raise GgufError(
                f"the CUDA toolkit here is {toolkit[0]}.{toolkit[1]}, but --device "
                f"{target.device} bundles a torch built for {target.cuda[0]}.{target.cuda[1]}.\n"
                "  llama.cpp would link against this machine's cublas and cudart while the\n"
                "  distribution ships torch's, and those only match within a major version.\n"
                f"  Install CUDA {target.cuda[0]}.x, or build for --device "
                f"cu{toolkit[0]}{toolkit[1]} instead."
            )
        elif target.cuda and toolkit != target.cuda:
            notes.append(
                f"toolkit is {toolkit[0]}.{toolkit[1]}, torch is for "
                f"{target.cuda[0]}.{target.cuda[1]}; same major version, so they interoperate"
            )

    if target.kind == "metal" and not IS_MACOS:
        raise GgufError(
            f"--device {target.device} wants a Metal llama.cpp, which only builds on macOS.\n"
            "  Build the macOS distribution on a Mac, or pick this machine's device -- and\n"
            "  if what you wanted was a CPU-only GGUF backend, --gguf-device cpu says so."
        )

    if target.kind == "rocm":
        rocm = find_rocm()
        if rocm is None:
            looked = (
                "$ROCM_PATH, $HIP_PATH, hipconfig and %ProgramFiles%\\AMD\\ROCm"
                if IS_WINDOWS else
                "$ROCM_PATH, hipconfig, /opt/rocm and /opt/rocm-*"
            )
            raise GgufError(
                f"--device {target.device} wants a HIP llama.cpp, and no ROCm installation was "
                "found.\n"
                f"  Looked at {looked}.\n"
                "  Install ROCm (https://rocm.docs.amd.com/), or build without the GGUF\n"
                "  backend (--no-gguf), or build its CPU variant (--gguf-device cpu)."
            )
        version = rocm_version(rocm)
        print(f"      ROCm: {rocm}" + (f"  ({version[0]}.{version[1]})" if version else ""))
        if version is not None and version < MIN_ROCM:
            raise GgufError(
                f"ROCm {version[0]}.{version[1]} at {rocm} is too old for ggml's HIP backend, "
                f"which needs {MIN_ROCM[0]}.{MIN_ROCM[1]} or newer."
            )
        if IS_WINDOWS:
            # A Windows HIP build is llama.cpp's documented configuration, but not
            # the default one: it needs ROCm's clang for both languages and the
            # Ninja generator. `_describe_environment` asks for both; what it
            # cannot do is conjure a clang that the HIP SDK did not install.
            llvm = _rocm_llvm(rocm)
            if llvm is None or not (llvm / "clang++.exe").is_file():
                raise GgufError(
                    f"the ROCm at {rocm} has no clang++ to compile HIP with.\n"
                    "  Looked under llvm\\bin and lib\\llvm\\bin. The HIP SDK component of the\n"
                    "  Windows ROCm installer is what provides it -- a driver-only install has\n"
                    "  the runtime and not the compiler.\n"
                    "  Or build the CPU variant of the GGUF backend (--gguf-device cpu)."
                )
            notes.append(f"HIP on Windows: clang from {llvm}, Ninja generator")
            if shutil.which("ninja") is None:
                notes.append("no ninja on PATH; the isolated build environment will install one")

    # Asked for on GPU targets too, not just `cpu`. What a CUDA or HIP build gets out
    # of it is the same CPU backend a cpu-only build gets, running the ops the device
    # has no kernel for and the layers that did not fit -- so an OpenBLAS already
    # downloaded into `nc/` for a CPU build is used here as well, without asking
    # again. Metal is left out: ggml links Accelerate on Apple hardware.
    openblas = resolve_openblas(openblas_mode) if target.kind != "metal" else None

    args = cmake_args(
        target,
        openblas=openblas,
        cpu_baseline=cpu_baseline,
        cuda_arch=cuda_arch,
        hip_arch=hip_arch,
        toolkit=toolkit,
        rocm_root=rocm,
        nvcc=nvcc,
    )

    version = locked_version()
    # Left unpinned when the lock is silent so that `--print-args` still answers;
    # `install` is where an unpinned spec is refused.
    spec = f"{DISTRIBUTION}=={version}" if version else DISTRIBUTION

    host_env, host_args, host_notes, host_libs = _host_toolchain(target, nvcc)
    args += host_args
    notes += host_notes

    build = Build(target=target, args=args, value=cmake_args_value(args), spec=spec,
                  openblas=openblas, notes=notes, runtime_libs=host_libs)
    if nvcc is not None:
        build.notes.append(f"nvcc={nvcc}")
    # The device's needs win over the host's: a Windows HIP build names ROCm's clang
    # for both languages, and must not have the host's cl put back over it.
    build.env = {**host_env, **_describe_environment(target, nvcc, rocm)}
    return build


def _force_remove_llama(python: Path, env: dict[str, str] | None = None) -> None:
    """Delete whatever `uv pip uninstall` left of a previous llama install.

    Only the package directory and its own `.dist-info`: anything else in
    site-packages belongs to somebody else. Not fatal if there is nothing, or
    if a removal fails -- the install below is the real check, and its error
    names the wheel rather than this cleanup.
    """
    remnants: list[Path] = []
    lib = _package_lib_dir(python, env)
    if lib is not None:
        remnants.append(lib.parent)
    dist_info = _dist_info_dir(python, env)
    if dist_info is not None:
        remnants.append(dist_info)

    seen: set[Path] = set()
    for path in remnants:
        if path in seen:
            continue
        seen.add(path)
        try:
            kind = "directory" if path.is_dir() else "file"
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            print(f"      removed leftover {path.name} ({kind}) from a half-removed install")
        except OSError as exc:
            print(f"      warning: could not remove leftover {path}: {exc}")


def install(python: Path, build: Build, *, env: dict[str, str] | None = None) -> Build:
    """``CMAKE_ARGS="..." uv pip install --no-cache-dir llama-cpp-python==<locked>``.

    Deliberately `uv pip install` and not `uv sync --extra gguf`. The extra is
    declarative, and a declarative install is free to hand back a wheel it built
    earlier -- uv's cache key does not include ``CMAKE_ARGS``, because uv has no
    way to know that an environment variable decided what is inside the wheel. So
    a CPU build's llama.cpp gets reused for a CUDA build, and the result is a
    distribution that loads a model and segmentation faults.

    ``--no-cache-dir`` is what makes that impossible rather than merely unlikely:
    uv neither reads nor writes its cache, so every install compiles llama.cpp
    again from the sdist with the arguments in front of it. It costs the compile
    time on every build, and it is the only setting under which the contents of
    the wheel are known.

    The version is the one uv.lock resolved -- see `locked_version`. Nothing else
    is added to the command.
    """
    uv = shutil.which("uv")
    if not uv:
        raise GgufError("uv not found. Install with: pip install uv")

    if "==" not in build.spec:
        raise GgufError(
            f"uv.lock names no {DISTRIBUTION} version.\n"
            "  The version installed is the locked one, not whatever resolves today --\n"
            "  otherwise two builds of the same commit compile different llama.cpp\n"
            "  releases, and the one that segmentation faults is not reproducible.\n"
            "  Add it to the `gguf` extra in pyproject.toml and run `uv lock`, or build\n"
            "  without the GGUF backend (--no-gguf)."
        )

    # A version that is already there is a requirement uv considers satisfied, so
    # it would audit the environment and install nothing -- leaving whatever some
    # earlier build compiled, for whatever device that was. Only reachable with a
    # reused build venv (`--keep-venv`); removed rather than worked around with
    # flags on the install itself.
    present = _installed_version(python, env)
    if present is not None:
        print(f"      {DISTRIBUTION} {present} is already installed here; removing it first", flush=True)
        remove = [uv, "pip", "uninstall", "--python", str(python), DISTRIBUTION]
        print(f"  $ {' '.join(remove)}", flush=True)
        sys.stdout.flush()
        result = subprocess.run(remove, cwd=str(PROJECT_ROOT), env=dict(env or os.environ))
        if result.returncode != 0 or _installed_version(python, env) is not None:
            # uv's uninstall is not atomic on Windows: a DLL held by a scan, or
            # a dist-info it half-removed, leaves the package importable (or
            # half-importable) and the install below then fails reading the
            # previous install's metadata. What is still on disk is removed
            # directly instead.
            _force_remove_llama(python, env)

    cmd = [uv, "pip", "install", "--python", str(python), "--no-cache-dir", build.spec]

    environment = dict(env or os.environ)
    # `prepare` put the generator and the compiler's own environment in here, so
    # this overrides the inherited PATH and INCLUDE rather than deferring to them.
    environment.update(build.env)
    environment["CMAKE_ARGS"] = build.value
    # llama-cpp-python's own switch from before it used scikit-build-core. Still
    # read by its setup, and harmless where it is not.
    environment["FORCE_CMAKE"] = "1"
    environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(_jobs(build.target))
    # Release, said out loud. It is already the default for scikit-build-core's
    # `cmake.build-type`, but a default is a thing that can change under us, and a
    # Debug llama.cpp is an unusably slow one -- `/Od` and no autovectorisation
    # through every kernel in ggml. Not `-DCMAKE_BUILD_TYPE` in `CMAKE_ARGS`:
    # scikit-build-core owns that variable and answers an attempt to set it there
    # with `Unsupported CMAKE_ARGS ignored`. `SKBUILD_<SETTING>` is the supported
    # spelling, and it is what ends up on the configure line.
    environment["SKBUILD_CMAKE_BUILD_TYPE"] = "Release"
    # Nothing in the build venv should be picked up by the compile; the only
    # thing this needs from it is where to install.
    environment.pop("CMAKE_TOOLCHAIN_FILE", None)
    print(f'  $ CMAKE_ARGS="{build.value}"', flush=True)

    print(f"  $ {' '.join(cmd)}", flush=True)
    print(f"      building with {environment['CMAKE_BUILD_PARALLEL_LEVEL']} parallel jobs", flush=True)

    sys.stdout.flush()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=environment)
    if result.returncode != 0:
        raise GgufError(
            f"{DISTRIBUTION} failed to build (exit {result.returncode}).\n"
            f"  The arguments were: {build.value}\n"
            "  Everything llama.cpp needs beyond a C++ compiler is device-specific: nvcc\n"
            "  for CUDA, ROCm's clang for HIP, and a CMake that can find them. If the\n"
            "  output above ends in a CMake error rather than a compile error, that is\n"
            "  where to look.\n"
            "  --no-gguf builds the rest of the distribution without this."
        )

    _copy_runtime_libraries(python, build, env=env)
    _repair_top_level(python, env=env)
    return build


def _copy_runtime_libraries(python: Path, build: Build, env: dict[str, str] | None = None) -> None:
    """Put the runtime libraries the build needs beside the libraries that need them.

    Two sets, for the same reason. A MinGW build links libgcc, libstdc++ and
    libwinpthread dynamically -- see `_host_toolchain` for why it must not link
    them statically. And a BLAS build links an OpenBLAS import library, whose DLL
    lives wherever OpenBLAS was found and is on nobody's search path. Both have to
    be somewhere the loader will look. ``llama_cpp/lib/`` is: the package calls
    `os.add_dll_directory` on it before it opens anything, and `bundle` copies the
    directory wholesale, so the distribution inherits them without knowing they
    are there.

    OpenBLAS matters here and not only in `bundle`, even though `bundle` copies it
    too, because `verify` runs in between. Leaving it to `bundle` means the import
    check fails on a layout the shipped tree would have got right -- a build that
    aborts complaining `libllama.dll` cannot be found when the library is there and
    the only absentee is a dependency two steps down. Copying it now also makes
    `verify` a check on the layout the distribution actually ships.

    Nothing here is fatal. The import in `verify` is the real check, and its error
    names the DLL the loader could not find.
    """
    openblas_runtime = build.openblas.runtime if build.openblas is not None else ()
    cuda_runtime = _cuda_runtime(python, build, env)
    if build.target.kind == "cuda" and not cuda_runtime:
        # Not fatal here -- `verify` is where a backend that cannot load is
        # refused, and it has the loader's own account of what is missing. Said
        # anyway, because this is the step that could have prevented it, and
        # because the usual cause is a fixable mismatch: a cu118 GGUF backend in
        # an environment whose torch is cu130 has nowhere to get a CUDA 11
        # runtime from, and the toolkit is the other half of the answer.
        print(f"      warning: no CUDA {build.target.cuda[0] if build.target.cuda else '?'} "
              f"runtime was found to put in {LIB_DIR}/.")
        print("               Neither torch nor the toolkit in this environment has one, so "
              "the backend")
        print("               will look for cudart and cuBLAS on the machine it runs on.")
    if not build.runtime_libs and not openblas_runtime and not cuda_runtime:
        return
    lib = _package_lib_dir(python, env)
    if lib is None:
        return

    def copy(source: Path) -> bool:
        """`source` into `lib`, under its own name. Symlinks resolved."""
        try:
            # A SONAME symlink has to arrive as a real file under the link's own
            # name: the library is asked for by that name, and a dangling link
            # ships nothing. Same rule as `bundle`.
            shutil.copy2(source.resolve() if source.is_symlink() else source, lib / source.name)
        except OSError as exc:
            print(f"      warning: could not copy {source.name} into {lib}: {exc}")
            return False
        return True

    copied = [dll.name for dll in build.runtime_libs if copy(dll)]
    if copied:
        print(f"      compiler runtime into {LIB_DIR}/: {', '.join(copied)}")

    # Skip anything the compiler runtime already put there: MinGW-built OpenBLAS
    # distributions ship their own libgcc/libwinpthread, and the compiler's own
    # are the ones that match what was just compiled.
    already = {name.lower() for name in copied}
    blas = [path.name for path in openblas_runtime
            if path.name.lower() not in already and copy(path)]
    if blas:
        print(f"      OpenBLAS runtime into {LIB_DIR}/: {', '.join(blas)}")

    cuda = [path.name for path in cuda_runtime
            if not (lib / path.name).is_file() and copy(path)]
    if cuda:
        print(f"      CUDA runtime into {LIB_DIR}/: {', '.join(cuda)}")


#: What ggml-cuda loads at run time, as the linker spells it. cudart is the runtime
#: itself; cuBLAS is what ggml hands the matrix multiplications to, and cuBLASLt is
#: cuBLAS's own. Nothing else in the toolkit is linked -- no cuSPARSE, no cuSOLVER,
#: no cuDNN. See ggml/src/ggml-cuda/CMakeLists.txt: `CUDA::cudart CUDA::cublas`.
#:
#: Two spellings because the two loaders name libraries differently, and both lists
#: are prefixes: the version is part of the file name on both platforms and is
#: matched separately, against the CUDA the build was made for.
_CUDA_RUNTIME_WINDOWS = ("cudart64_", "cublas64_", "cublaslt64_")
_CUDA_RUNTIME_LINUX = ("libcudart.so", "libcublas.so", "libcublaslt.so")

#: What a copied CUDA library is allowed to drag in behind it. cuBLAS has needed
#: libnvJitLink since 12.0 and may need more later, so the dependency graph is read
#: rather than listed -- but only these families are followed, so a stray DT_NEEDED
#: on libc or libstdc++ does not pull the system out from under the distribution.
_CUDA_FAMILIES = ("libcu", "libnv")

#: The one library in those families that must never be shipped, whatever needs it.
#: ``libcuda.so.1`` is the driver API. It comes with the NVIDIA display driver, is
#: matched to the driver and the card, and the copy in the toolkit is a link stub
#: with no implementation in it at all. Shipping either one puts a library on the
#: user's machine that shadows their real driver, which is a distribution that
#: loads and then cannot see a GPU -- strictly worse than one that fails to load.
_CUDA_DRIVER = ("libcuda.so", "nvcuda.dll")


def _elf_needed(path: Path) -> list[str]:
    """The ``DT_NEEDED`` names of an ELF64 shared object. Empty if it is not one.

    What a library needs beside it is written down inside it, and reading that
    is the difference between shipping the dependency graph and shipping a list
    someone maintained by hand until they stopped. Kept next to `_pe_import_names`
    below so both readers live in one module -- build.py uses that one too.

    Deliberately tolerant. Anything unreadable, truncated, 32-bit, big-endian or
    simply not ELF answers "nothing needed", because every caller treats that as
    "copy this one file and no more" -- which is the previous behaviour, not a
    failure.
    """
    try:
        blob = path.read_bytes()
    except OSError:
        return []
    # 0x7f E L F, 64-bit (EI_CLASS 2), little-endian (EI_DATA 1). Nothing this
    # ships on is anything else, and guessing at a format is worse than skipping.
    if len(blob) < 64 or blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        return []

    def u16(at: int) -> int:
        return int.from_bytes(blob[at:at + 2], "little")

    def u64(at: int) -> int:
        return int.from_bytes(blob[at:at + 8], "little")

    phoff, phentsize, phnum = u64(0x20), u16(0x36), u16(0x38)
    if phoff <= 0 or phentsize < 56 or phnum <= 0:
        return []
    if phoff + phentsize * phnum > len(blob):
        return []

    loads: list[tuple[int, int, int]] = []  # (vaddr, filesz, offset)
    dynamic: tuple[int, int] | None = None  # (offset, filesz)
    for index in range(phnum):
        header = phoff + index * phentsize
        kind = int.from_bytes(blob[header:header + 4], "little")
        offset, vaddr, filesz = u64(header + 0x08), u64(header + 0x10), u64(header + 0x20)
        if kind == 1:  # PT_LOAD
            loads.append((vaddr, filesz, offset))
        elif kind == 2:  # PT_DYNAMIC
            dynamic = (offset, filesz)
    if dynamic is None:
        return []

    def to_offset(vaddr: int) -> int | None:
        """A virtual address as a position in the file, through the load map."""
        for start, size, offset in loads:
            if start <= vaddr < start + size:
                return vaddr - start + offset
        return None

    # One pass over .dynamic: DT_NEEDED holds an offset into a string table whose
    # own address is a later entry, so the names cannot be read until the end.
    offset, size = dynamic
    needed_offsets: list[int] = []
    strtab_vaddr: int | None = None
    for at in range(offset, min(offset + size, len(blob) - 15), 16):
        tag, value = u64(at), u64(at + 8)
        if tag == 0:  # DT_NULL
            break
        if tag == 1:  # DT_NEEDED
            needed_offsets.append(value)
        elif tag == 5:  # DT_STRTAB
            strtab_vaddr = value
    if strtab_vaddr is None:
        return []
    strtab = to_offset(strtab_vaddr)
    if strtab is None:
        return []

    names = []
    for entry in needed_offsets:
        start = strtab + entry
        if not 0 <= start < len(blob):
            continue
        end = blob.find(b"\0", start)
        if end < 0:
            continue
        name = blob[start:end].decode("utf-8", "replace")
        if name:
            names.append(name)
    return names


def _pe_import_names(path: Path) -> list[str]:
    """The DLL names in a PE file's import directory. Empty if it is not one.

    The Windows counterpart to `_elf_needed`, and here for the same reason: what
    a library needs beside it is written down inside it. Deliberately tolerant
    for the same reason too -- anything unreadable answers "nothing needed".

    Only the load-time import directory, not delay-load: a delay-loaded backend
    fails gracefully at runtime, which is the dynamic case below, so reading it
    here would misreport a graceful build as a static one.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return []
    if data[:2] != b"MZ":
        return []

    try:
        head = struct.unpack_from("<I", data, 0x3C)[0]
        if data[head:head + 4] != b"PE\0\0":
            return []
        sections = struct.unpack_from("<H", data, head + 6)[0]
        opt_size = struct.unpack_from("<H", data, head + 20)[0]
        opt = head + 24
        magic = struct.unpack_from("<H", data, opt)[0]
        count_at = opt + (108 if magic == 0x20B else 92)
        if struct.unpack_from("<I", data, count_at)[0] < 2:
            return []
        imports_rva = struct.unpack_from("<I", data, count_at + 4 + 8)[0]
        if not imports_rva:
            return []

        table = []
        base = opt + opt_size
        for index in range(sections):
            entry = base + index * 40
            v_size, v_addr, r_size, r_addr = struct.unpack_from("<IIII", data, entry + 8)
            table.append((v_addr, max(v_size, r_size), r_addr))

        def offset_of(rva: int) -> int | None:
            for v_addr, size, r_addr in table:
                if v_addr <= rva < v_addr + size:
                    return r_addr + (rva - v_addr)
            return None

        cursor = offset_of(imports_rva)
        if cursor is None:
            return []

        names = []
        while True:
            descriptor = data[cursor:cursor + 20]
            if len(descriptor) < 20 or not any(descriptor):
                break
            at = offset_of(struct.unpack_from("<I", descriptor, 12)[0])
            if at is not None and at < len(data):
                names.append(data[at:data.index(b"\0", at)].decode("ascii", "replace"))
            cursor += 20
        return names
    except Exception:  # noqa: BLE001 - an unreadable header is not a build failure
        return []


def _cuda_major(name: str) -> str | None:
    """The CUDA major version a runtime library's file name carries, if any.

    ``cudart64_13.dll`` and ``libcudart.so.13.0.96`` both say 13. The point is not
    the number but the comparison: an environment can hold more than one CUDA --
    a cu118 torch beside a CUDA 13 toolkit is a supported combination of this
    script's own flags -- and a cudart from one beside a cuBLAS from the other is
    a distribution that loads on nobody's machine.
    """
    match = re.search(r"_(\d+)\.dll$", name) or re.search(r"\.so\.(\d+)", name)
    if not match:
        return None
    digits = match.group(1)
    if name.lower().startswith("cudart64_") and len(digits) > 2:
        # Windows' cudart carries the full version without dots: cudart64_110.dll
        # is CUDA 11.x, so the major is the first two digits. Without this the
        # 11.x cudart never matches a cu118 build and is silently left out of the
        # runtime the distribution ships -- and `_missing_runtime` stays blind to
        # it for the same reason.
        return digits[:2]
    return digits


def _cuda_library_dirs(python: Path, env: dict[str, str] | None, build: Build) -> list[Path]:
    """Where a CUDA runtime might be found, best source first.

    torch first, and not the toolkit, for two reasons. It is already in this
    environment and already being shipped, so the libraries are on disk whether or
    not a toolkit is; and they are the ones the distribution runs against anyway.

    The toolkit is the fallback, and it is the one that matters for
    ``--device cpu --gguf-device cu130``: a CPU torch ships no CUDA at all, so
    without it that build produces a GGUF backend with nothing to load.
    """
    code, text = _capture(
        [str(python), "-c",
         "import importlib.util as u, sys; s = u.find_spec('torch');"
         " sys.stdout.write(list(s.submodule_search_locations)[0] if s else '')"],
        env=env,
    )
    dirs: list[Path] = []
    if code == 0 and text.strip():
        torch = Path(text.strip().splitlines()[-1].strip())
        dirs.append(torch / "lib")
        # On Linux torch does not carry CUDA itself: it depends on NVIDIA's own
        # wheels, which unpack to `nvidia/<component>/lib` beside it. Windows
        # torch bundles everything in `torch/lib`, so this finds nothing there and
        # costs one failed glob.
        dirs += sorted(path for path in (torch.parent / "nvidia").glob("*/lib") if path.is_dir())

    root = (build.env or {}).get("CUDA_PATH") or (env or os.environ).get("CUDA_PATH")
    if root:
        base = Path(root)
        dirs += [base / "bin", base / "lib64", base / "lib" / "x64", base / "lib"]
    return [path for path in dirs if path.is_dir()]


def _cuda_runtime(python: Path, build: Build, env: dict[str, str] | None) -> list[Path]:
    """The CUDA runtime libraries a CUDA build needs beside it.

    `llama_cpp` puts one directory on the loader's path -- its own ``lib/`` -- and
    then opens ``libllama``, which pulls in ``ggml-cuda``, which needs cudart and
    cuBLAS. Those are not in the wheel. On a development machine they resolve off
    PATH or off the toolkit's absolute path baked into the RPATH, so nothing
    appears to be missing; on a machine with no toolkit the loader finds nothing
    and reports only that ``libllama`` could not be loaded.

    ``lib/`` is the only directory that works, on either platform, and it is worth
    saying why the obvious alternative does not. The libraries are already in the
    tree -- torch ships them -- and on Windows that is enough, because the entry
    point calls `os.add_dll_directory` on ``torch/lib`` before anything is
    imported. On Linux there is no such call and no equivalent: what ggml-cuda
    finds is fixed at link time, and llama-cpp-python's own CMakeLists pins it to
    ``$ORIGIN`` per target, overriding anything passed in ``CMAKE_ARGS``. So on
    Linux the files are copied even though a copy already exists a few directories
    away. Several hundred megabytes, and the alternative is a backend that does
    not load.
    """
    if build.target.kind != "cuda":
        return []
    prefixes = _CUDA_RUNTIME_WINDOWS if IS_WINDOWS else _CUDA_RUNTIME_LINUX
    if not IS_WINDOWS and not IS_LINUX:
        return []

    #: Every candidate by file name, first source winning, so torch's copy is
    #: preferred over the toolkit's and the two are never mixed.
    index: dict[str, Path] = {}
    pattern = "*.dll" if IS_WINDOWS else "*.so*"
    for directory in _cuda_library_dirs(python, env, build):
        for path in sorted(directory.glob(pattern)):
            if path.is_file() and not path.name.lower().startswith(_CUDA_DRIVER):
                index.setdefault(path.name.lower(), path)

    wanted = str(build.target.cuda[0]) if build.target.cuda else None

    def seeded(name: str) -> bool:
        if not name.startswith(prefixes):
            return False
        major = _cuda_major(name)
        if major is None:
            # `libcublas.so` with no version is the development symlink, pointing
            # at the versioned file that is already a candidate. Copying it would
            # be the same several hundred megabytes twice under two names.
            return IS_WINDOWS
        return wanted is None or major == wanted

    seeds = [name for name in index if seeded(name)]

    # Close over what the seeds themselves need. cuBLAS has needed libnvJitLink
    # since CUDA 12, and that is exactly the kind of fact that changes in a point
    # release, so it is read out of the libraries rather than written down here.
    chosen: dict[str, Path] = {}
    queue = list(seeds)
    while queue:
        name = queue.pop()
        if name in chosen:
            continue
        path = index.get(name)
        if path is None:
            continue
        chosen[name] = path
        if IS_WINDOWS:
            continue
        for dependency in _elf_needed(path):
            lowered = dependency.lower()
            if (lowered.startswith(_CUDA_FAMILIES)
                    and not lowered.startswith(_CUDA_DRIVER)
                    and lowered not in chosen):
                queue.append(lowered)

    return sorted(chosen.values(), key=lambda path: path.name.lower())


#: Finds the installed distribution's `.dist-info`. Runs in the build venv, because
#: that is the environment whose metadata is being asked about.
_DIST_INFO_PROBE = r"""
import sys
from importlib.metadata import distribution
from pathlib import Path

name = sys.argv[1]
dist = distribution(name)

# `_path` is private, and is what every pip or uv install has. The fallback covers
# anything that is not a plain directory install: `locate_file` is public and gives
# the directory the distribution went into, and the name before the first dash is
# the project name with its separators flattened -- which is what the installer
# wrote there.
path = getattr(dist, "_path", None)
if path is None:
    key = name.replace("-", "_").replace(".", "_").lower()
    found = [
        candidate
        for candidate in Path(str(dist.locate_file(""))).glob("*.dist-info")
        if candidate.name.split("-")[0].replace("-", "_").replace(".", "_").lower() == key
    ]
    path = found[0] if found else None

sys.stdout.write(str(path) if path is not None else "")
"""


def _dist_info_dir(python: Path, env: dict[str, str] | None = None) -> Path | None:
    """The installed distribution's `.dist-info` directory, asked of the venv itself.

    Asked rather than assembled: the directory name carries the version, and the
    version is whatever `prepare` resolved.
    """
    code, text = _capture([str(python), "-c", _DIST_INFO_PROBE, DISTRIBUTION], env=env)
    if code != 0 or not text.strip():
        return None
    path = Path(text.strip().splitlines()[-1].strip())
    return path if path.is_dir() else None


def _repair_top_level(python: Path, env: dict[str, str] | None = None) -> None:
    """Write the `top_level.txt` that this distribution does not ship.

    Nuitka has to know which package a distribution's metadata belongs to, so that
    `--include-distribution-metadata` cannot bundle a version string for something
    that was never compiled in. It reads `top_level.txt` to find out, and falls
    back to guessing from the file list when that is missing -- taking the first
    top-level directory it sees there.

    Since 0.3.34 the guess is wrong. The project builds with `uv_build` now and
    installs CMake's whole install tree next to the package, so its file list
    begins `bin/ggml-base.dll`, `include/ggml.h`, `lib/libllama.dll.a` -- three
    directories that are data, sort ahead of `llama_cpp`, and are not importable.
    Nuitka picks `bin`, waits until the module set is complete, finds no such
    module and stops the build with

        Error, including metadata for distribution 'llama_cpp_python'
        without including related package 'bin'.

    which names neither the real package nor anything a caller can act on. One
    line of the metadata upstream omitted settles it, and settles it in the right
    place: `llama_cpp` is genuinely this distribution's only importable top-level
    package, so the file says what is true rather than working around anything.

    Not fatal on its own. If it cannot be written the build proceeds and Nuitka
    reports the above, so the warning here is what connects the two.
    """
    dist_info = _dist_info_dir(python, env)
    if dist_info is None:
        print(f"      warning: cannot find {DISTRIBUTION}'s .dist-info, so its"
              " top_level.txt is left alone")
        return

    top_level = dist_info / "top_level.txt"
    try:
        # Rewritten only when it does not already name the package: a future
        # release that ships its own correct one must be left as it is.
        current = top_level.read_text(encoding="utf-8") if top_level.is_file() else ""
        if PACKAGE in current.split():
            return
        top_level.write_text(f"{PACKAGE}\n", encoding="utf-8")
    except OSError as exc:
        print(f"      warning: could not write {top_level}: {exc}")
        return

    print(f"      wrote {top_level.name} naming {PACKAGE}"
          " (the distribution ships none, and Nuitka guesses wrong without it)")

    # Read back through the same steps Nuitka takes, and confirm the name that
    # comes out is the package. Not Nuitka's own function -- that calls its module
    # locator, which raises outside a compile -- but the rest of its first branch:
    # a line per name, '/' meaning '.', anything not importable dropped. What this
    # catches is the file being written and still not answering the question, which
    # is the only way this can fail quietly.
    code, text = _capture(
        [str(python), "-c",
         "import importlib.util as u, sys;"
         f" names = open({str(top_level)!r}, encoding='utf-8').read().splitlines();"
         " ok = [n.replace('/', '.') for n in names if n.strip()];"
         " ok = [n for n in ok if u.find_spec(n) is not None];"
         " sys.stdout.write(ok[0] if ok else '')"],
        env=env,
    )
    chosen = text.strip().splitlines()[-1].strip() if code == 0 and text.strip() else ""
    if chosen != PACKAGE:
        print(f"      warning: {top_level.name} still does not resolve to '{PACKAGE}'"
              f" (got '{chosen or 'nothing'}').\n"
              "               The compile will stop on that. Either the package is not"
              " importable in this\n"
              f"               venv, or drop {DISTRIBUTION} from"
              " OPTIONAL_DISTRIBUTION_METADATA in build.py.")


# ── Proving what was built ────────────────────────────────────────────────────

#: Asks the freshly installed llama_cpp what it is. Tolerant: every answer is
#: optional, because the point is to report what was built, and an upstream
#: rename must not fail a build that is otherwise fine.
_VERIFY_PROBE = r"""
import json, sys

#: An upper bound on a registry the caller is about to walk. See `_registry`.
_REGISTRY_LIMIT = 64

answer = {"version": None, "info": "", "gpu_offload": None, "lib": "", "error": None,
          "note": "", "backends": None, "devices": None}


def _shared_libraries(base):
    '''ggml's shared libraries in `base`, in no particular order.'''
    found = []
    for pattern in ("ggml*.dll", "libggml*.dylib", "libggml*.so", "libggml*.so.*"):
        for path in sorted(base.glob(pattern)):
            # Import libraries and static archives are not loadable. On Windows
            # `libggml-base.dll.a` sits in the same directory as the DLLs.
            if path.name.endswith((".a", ".lib")) or not path.is_file():
                continue
            found.append(path)
    return found


def _registry(base):
    '''The backend registry names ggml reports, or None if it cannot be asked.

    The registry rather than the device list, deliberately. A CUDA build on a
    machine with no card registers the backend and enumerates zero devices, so
    devices would describe the builder's hardware where this describes the build.

    Every handle is tried for every symbol because they are split across two
    libraries -- `reg_count` and `reg_get` are exported by ggml, `reg_name` by
    ggml-base -- and which library holds which is not a thing to depend on.
    llama_cpp's own handle goes first, for a build that linked ggml into libllama
    and so has no separate ggml library to open.
    '''
    import ctypes

    handles = []
    try:
        import llama_cpp.llama_cpp as core

        if getattr(core, "_lib", None) is not None:
            handles.append(core._lib)
    except Exception:
        pass
    for path in _shared_libraries(base):
        try:
            handles.append(ctypes.CDLL(str(path)))
        except OSError:
            pass
    if not handles:
        return None, None

    def resolve(name):
        for handle in handles:
            try:
                return getattr(handle, name)
            except AttributeError:
                continue
        return None

    count = resolve("ggml_backend_reg_count")
    get = resolve("ggml_backend_reg_get")
    label = resolve("ggml_backend_reg_name")
    backends = None
    if count is not None and get is not None and label is not None:
        count.restype = ctypes.c_size_t
        get.restype = ctypes.c_void_p
        get.argtypes = [ctypes.c_size_t]
        label.restype = ctypes.c_char_p
        label.argtypes = [ctypes.c_void_p]
        backends = []
        # Bounded. `ggml_backend_reg_get` calls GGML_ASSERT on an index past the
        # end, and a GGML_ASSERT is an abort() -- no exception to catch, no JSON on
        # stdout, and a build that fails with nothing to say. A wrong count from a
        # library that did not initialise is exactly the case this runs in.
        for index in range(min(count(), _REGISTRY_LIMIT)):
            entry = get(index)
            if not entry:
                continue
            raw = label(entry)
            if raw:
                backends.append(raw.decode("utf-8", "replace"))

    dev_count = resolve("ggml_backend_dev_count")
    dev_get = resolve("ggml_backend_dev_get")
    dev_name = resolve("ggml_backend_dev_name")
    dev_desc = resolve("ggml_backend_dev_description")
    devices = None
    if dev_count is not None and dev_get is not None and dev_name is not None:
        dev_count.restype = ctypes.c_size_t
        dev_get.restype = ctypes.c_void_p
        dev_get.argtypes = [ctypes.c_size_t]
        dev_name.restype = ctypes.c_char_p
        dev_name.argtypes = [ctypes.c_void_p]
        if dev_desc is not None:
            dev_desc.restype = ctypes.c_char_p
            dev_desc.argtypes = [ctypes.c_void_p]
        devices = []
        for index in range(min(dev_count(), _REGISTRY_LIMIT)):
            entry = dev_get(index)
            if not entry:
                continue
            name = dev_name(entry) or b""
            text = (dev_desc(entry) or b"") if dev_desc is not None else b""
            devices.append([name.decode("utf-8", "replace"), text.decode("utf-8", "replace")])
    return backends, devices


try:
    import llama_cpp
except Exception as exc:
    answer["error"] = "%s: %s" % (type(exc).__name__, exc)
else:
    answer["version"] = getattr(llama_cpp, "__version__", None)
    try:
        answer["lib"] = str(getattr(llama_cpp.llama_cpp, "_base_path", ""))
    except Exception:
        pass
    try:
        raw = llama_cpp.llama_print_system_info()
        answer["info"] = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    except Exception as exc:
        answer["note"] = "llama_print_system_info: %s: %s" % (type(exc).__name__, exc)
    try:
        answer["gpu_offload"] = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        pass
    try:
        from pathlib import Path

        if answer["lib"]:
            answer["backends"], answer["devices"] = _registry(Path(answer["lib"]))
    except Exception as exc:
        answer["note"] = (answer["note"] + " " if answer["note"] else "") + (
            "backend registry: %s: %s" % (type(exc).__name__, exc)
        )

json.dump(answer, sys.stdout)
# Explicit, because what runs after this is ggml's static destructors, and a CUDA
# build with no driver has been seen to take the process down in them. An
# unflushed buffer there is a build that fails with no reason given, on a machine
# where the reason is "there is no GPU here" and the answer was already written.
sys.stdout.flush()
"""

compile(_VERIFY_PROBE, "<llama-verify>", "exec")

#: What each device has to show up as. `BACKEND_NAMES` is what ggml calls the
#: backend in its registry; more than one spelling where llama.cpp has used more
#: than one (the HIP build has answered to both). `BACKEND_MARKER` is the token to
#: look for in `llama_print_system_info` when the registry cannot be read.
#:
#: Both are lists of registries rather than devices, so a CUDA build says so even
#: on a machine with no card in it -- which is what makes this a check on the build
#: rather than on the builder's hardware.
BACKEND_NAMES = {
    "cuda": ("CUDA",),
    "rocm": ("ROCm", "HIP", "CUDA"),
    "metal": ("Metal",),
}

#: The BLAS backend is not in `BACKEND_NAMES` because no device targets it: it is
#: an accelerator for a CPU build, checked separately and never fatally.
BLAS_BACKEND = "BLAS"

BACKEND_MARKER = {"cuda": "CUDA", "rocm": "ROCm", "metal": "Metal"}


def _has_backend(info: str, token: str) -> bool:
    """Whether `info` reports `token` as a backend that is actually there.

    Both spellings llama.cpp has used: a section header (`CUDA : ARCHS = ...`)
    and the older flag form (`BLAS = 1`). The flag form is why this cannot be a
    substring test -- a CPU-only build says `BLAS = 0`.

    Only a fallback. `llama_print_system_info` lists a backend only if it exposes
    a `ggml_backend_get_features` proc address, and ggml-blas registers a NULL
    one -- so a perfectly good OpenBLAS build is invisible here, and the registry
    (`_registered`) is what should be asked first.
    """
    if re.search(rf"(^|\|)\s*{re.escape(token)}\b\s*:", info, re.IGNORECASE):
        return True
    return bool(re.search(rf"\b{re.escape(token)}\s*=\s*1\b", info, re.IGNORECASE))


def _registered(backends: object, names: tuple[str, ...] | list[str]) -> bool | None:
    """Whether any of `names` is in the registry `backends`.

    None when there is no registry to consult -- a missing answer, not a negative
    one, and the difference decides whether the caller may fail the build.
    """
    if not isinstance(backends, list):
        return None
    reported = {str(name).strip().lower() for name in backends}
    return any(name.lower() in reported for name in names)


def _import_env(env: dict[str, str] | None) -> dict[str, str]:
    """`env` with any GPU root `llama_cpp` would trip over taken back out.

    `llama_cpp`'s import runs, on Windows, ``os.add_dll_directory`` over four
    directories -- ``CUDA_PATH/bin``, ``CUDA_PATH/lib``, ``HIP_PATH/bin`` and
    ``HIP_PATH/lib`` -- without checking that any of them are there, and
    `os.add_dll_directory` raises ``FileNotFoundError`` when they are not. So the
    package does not import at all, with an error naming a directory instead of a
    library. A CUDA whose toolkit has half gone away is enough to cause it: an
    uninstall that keeps the runtime leaves ``include`` and ``lib`` behind, deletes
    ``bin``, and leaves ``CUDA_PATH`` and the PATH entry pointing at what is no
    longer there.

    Both subdirectories are checked, not just ``bin``, because the import needs
    both and fails on the first one missing. A directory that does not exist
    contributes no DLLs, so dropping the variable costs nothing and is what lets
    the import happen. Done for the probe's environment rather than by repairing
    the machine's, which is not this script's to change.
    """
    merged = dict(env if env is not None else os.environ)
    for key in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDAToolkit_ROOT", "HIP_PATH", "ROCM_PATH"):
        root = merged.get(key)
        if not root:
            continue
        absent = [name for name in ("bin", "lib") if not (Path(root) / name).is_dir()]
        if absent:
            merged.pop(key)
            print(f"      {key} is {root}, which has no {'/'.join(absent)} -- dropped for the "
                  f"import check, which {PACKAGE} would otherwise fail outright")
    return merged


def _driver_present() -> bool | None:
    """Whether this machine has an NVIDIA driver. None when it cannot be told.

    The driver API is a shared library that ships with the display driver and
    with nothing else, so its presence is the question. It is asked of the loader
    rather than by looking for a file, because on Linux the file's directory is a
    distribution's business -- ``/usr/lib/x86_64-linux-gnu``, ``/usr/lib64``, a
    NixOS store path -- and the loader is the thing that knows.

    The toolkit's own copy is deliberately not counted. CUDA installs a *stub*
    ``libcuda.so`` under ``lib64/stubs``, which exists so a machine with no driver
    can still link; it has no implementation in it, and a build that found it and
    concluded there was a driver would be wrong in the one direction that
    matters. Nothing puts ``stubs`` on the default search path, so asking the
    loader for the SONAME -- ``libcuda.so.1``, which the stub is not called --
    keeps it out.
    """
    if IS_WINDOWS:
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        return (system32 / "nvcuda.dll").is_file()
    if not IS_LINUX:
        return None
    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        pass
    # A container often has the driver mounted in without a cache entry for it, so
    # a failed dlopen is suggestive rather than conclusive. The device nodes are
    # the second opinion: they are what the driver creates and nothing else does.
    if Path("/dev/nvidiactl").exists() or Path("/proc/driver/nvidia/version").exists():
        return True
    return False


def _load_failure_hint(build: Build) -> str:
    """The one cause of a CUDA load failure that nothing here can put right.

    The CUDA driver API -- ``nvcuda.dll``, ``libcuda.so.1`` -- ships with the
    NVIDIA display driver and with nothing else: not with the toolkit, not with
    torch, and it cannot be copied from another machine, because it is matched to
    the driver and the GPU. ggml links it directly (``CUDA::cuda_driver``, for the
    virtual memory API), so a machine with no NVIDIA GPU can build a CUDA
    llama.cpp and never load it, and the loader's only complaint is that
    ``libllama`` could not be found. Worth naming, because every other reading of
    that message sends the reader looking for a library that is in fact already
    there.

    Both platforms, and that is the point of it. Windows was the only one checked,
    which meant the same driverless machine -- a CI runner, a container, a laptop
    with an AMD card -- produced a warning and a distribution on Windows and a
    hard failure on Linux, against a build script whose whole purpose is to
    produce CUDA distributions on machines that cannot run them.
    """
    if build.target.kind != "cuda" or _driver_present() is not False:
        return ""
    library = "nvcuda.dll" if IS_WINDOWS else "libcuda.so.1"
    return (
        f"No {library} could be loaded here, which means no NVIDIA display driver is\n"
        "installed. That is the CUDA driver API: it comes with the driver and with\n"
        "nothing else, and cannot be shipped, so the CUDA backend will not load on this\n"
        "machine however it was built. The build itself is unaffected -- the result runs\n"
        "on a machine that has an NVIDIA GPU. Install the driver to check it here, or\n"
        "build for the device this machine has (--gguf-device cpu, or rocm for AMD).\n"
    )


#: The shared library each GPU backend is compiled into. Its presence is the one
#: statement about the build that survives the library failing to load: ggml links
#: its backends into libggml (``GGML_BACKEND_DL`` is off), so the file is there if
#: and only if the backend was compiled, whatever the loader then makes of it.
_BACKEND_LIBRARY = {"cuda": "ggml-cuda", "rocm": "ggml-hip", "metal": "ggml-metal"}


def _backend_library(python: Path, build: Build, env: dict[str, str] | None) -> Path | None:
    """The compiled backend library for this target, if it was built at all."""
    stem = _BACKEND_LIBRARY.get(build.target.kind)
    if stem is None:
        return None
    lib = _package_lib_dir(python, env)
    if lib is None:
        return None
    for candidate in (f"{stem}.dll", f"lib{stem}.so", f"lib{stem}.dylib", f"{stem}.so"):
        path = lib / candidate
        if path.is_file():
            return path
    return None


def _backend_static_linkage(python: Path, build: Build, env: dict[str, str] | None) -> str:
    """How the GPU backend is linked: ``"static"``, ``"dynamic"`` or ``"unknown"``.

    ``"static"`` means one of the package's own libraries names the backend
    library in its load-time imports (PE import directory, ELF DT_NEEDED) --
    ``ggml.dll`` importing ``ggml-cuda.dll``, which imports ``nvcuda.dll``. On a
    machine without that driver the package cannot import at all, CPU included:
    the loader fails the whole chain before any code runs. Prebuilt wheels are
    commonly built this way.

    ``"dynamic"`` means the backend library is there but nothing load-links it,
    so it can only be reached at runtime (dlopen) -- a missing driver degrades
    to CPU instead. A source build through this module comes out this way.

    ``"unknown"`` means it could not be told: not a GPU target, the backend
    library was never built, or something was unreadable. Callers keep their
    previous behavior then.

    Read by extension rather than by platform, so a Windows tree reports the
    same answer checked on Linux and the unit tests do not depend on where they
    run. Import libraries (``.lib``) and static archives (``.a``) are skipped:
    they are never loaded.
    """
    backend = _backend_library(python, build, env)
    if backend is None:
        return "unknown"
    try:
        entries = [entry for entry in backend.parent.iterdir() if entry.is_file()]
    except OSError:
        return "unknown"

    wanted = backend.name.lower()
    scanned = 0
    for entry in entries:
        if entry.name.lower() == wanted:
            continue
        lowered = entry.name.lower()
        if lowered.endswith((".lib", ".a")):
            continue
        if lowered.endswith(".dll"):
            imports = _pe_import_names(entry)
        elif ".so" in lowered:
            imports = _elf_needed(entry)
        else:
            # A format with no reader here (a macOS .dylib): skipping it is not
            # evidence of dynamic loading.
            continue
        scanned += 1
        if any(name.lower() == wanted for name in imports):
            return "static"
    # Nothing scannable is not an answer either -- a directory of .dylibs says
    # nothing about how they reach each other.
    return "dynamic" if scanned else "unknown"


def _missing_runtime(python: Path, build: Build, env: dict[str, str] | None) -> list[str]:
    """Runtime libraries this build needs beside it that are not beside it.

    Asked before the driver excuse is accepted, and that is the whole reason it
    exists. On a machine with no NVIDIA driver every CUDA load failure looks
    identical -- Windows reports error 126 and names nothing at all -- so a
    missing cudart, a cuBLAS from the wrong CUDA and an absent driver are one
    message. Taking the excuse without this check means those first two ship.
    """
    lib = _package_lib_dir(python, env)
    if lib is None:
        return []
    wanted = [path.name for path in _cuda_runtime(python, build, env)]
    if build.openblas is not None:
        wanted += [path.name for path in build.openblas.runtime]
    if IS_WINDOWS:
        # `_register_dll_directories` puts torch/lib on the search path in the
        # shipped tree, and `bundle` leaves the copies there rather than doubling
        # them, so a name that is in torch/lib is not missing.
        elsewhere = {path.name.lower() for path in (lib.parent.parent / "torch" / "lib").glob("*.dll")}
    else:
        elsewhere = set()
    return sorted({name for name in wanted
                   if not (lib / name).is_file() and name.lower() not in elsewhere})


def verify(python: Path, build: Build, env: dict[str, str] | None = None) -> None:
    """Import the built package and insist it is what was asked for."""
    proc = subprocess.run([str(python), "-c", _VERIFY_PROBE], capture_output=True,
                          env=_import_env(env))
    try:
        answer = json.loads(proc.stdout.decode("utf-8", "replace"))
    except ValueError as exc:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise GgufError(
            f"could not ask the build environment about {PACKAGE} (exit {proc.returncode}, "
            f"{exc}).\n"
            + ("".join(f"    {line}\n" for line in detail[-8:]) or "    no output")
        ) from exc

    if answer.get("error"):
        missing_driver = _load_failure_hint(build)
        # Three things have to hold before "no driver here" is accepted as the
        # reason. There is no driver; the backend was actually compiled, so this
        # is not a CUDA build that quietly came out CPU-only; and nothing it needs
        # beside it is absent. Any of those failing and the error is a real one
        # wearing the same message.
        built = _backend_library(python, build, env)
        missing = _missing_runtime(python, build, env)
        if missing_driver and built is not None and not missing:
            if _backend_static_linkage(python, build, env) == "static":
                # Not the graceful case above -- the backend is linked into the
                # load chain (ggml.dll names ggml-cuda.dll in its own imports),
                # so without a driver this package cannot import at all, CPU
                # included. This module compiles the backend to load at runtime
                # and degrade to CPU, so reaching here means something changed
                # that arrangement -- report it, loudly, rather than shipping a
                # distribution whose CPU fallback silently needs a driver.
                # Still not a build failure: on a machine with a driver this
                # distribution works. But it must not be mistaken for something
                # that was checked, or for something whose CPU fallback works
                # without one.
                print("      warning: the CUDA backend is linked statically, so without "
                      "an NVIDIA driver")
                print("               this package cannot import at all here -- CPU fallback "
                      "included -- and it was not checked.")
                for line in missing_driver.rstrip().splitlines():
                    print(f"      {line.strip()}")
                print(f"      What was checked: {built.name} is in {LIB_DIR}/, with "
                      "everything it links except the driver, so on a machine with "
                      "an NVIDIA GPU this distribution works.")
                print("      For CPU inference on a driverless machine, build the CPU "
                      "backend (--gguf-device cpu).")
                build.notes.append("statically linked CUDA backend: importable only where an "
                                   "NVIDIA driver is installed (CPU fallback included); never "
                                   "imported here")
                return
            # Not a build failure. Nothing is wrong with the library: this host has no
            # NVIDIA driver, so it cannot load a CUDA one, and no arrangement of the
            # build would change that. Refusing here would mean a CUDA distribution
            # could only ever be built on a machine with an NVIDIA GPU, which is not
            # a requirement of the distribution -- only of running it.
            print("      warning: the CUDA backend cannot be loaded on this machine, so it "
                  "was not checked.")
            for line in missing_driver.rstrip().splitlines():
                print(f"      {line.strip()}")
            print(f"      What was checked: {built.name} was compiled and is in {LIB_DIR}/, "
                  "with everything it links.")
            build.notes.append("unverified: no NVIDIA driver here to load the CUDA backend "
                               "with. It was compiled and is complete, but was never imported")
            return
        reasons = []
        if missing_driver and built is None:
            reasons.append(
                f"  The backend library itself is not there. A {build.target.label} build\n"
                f"  produces {_BACKEND_LIBRARY.get(build.target.kind)}, and {LIB_DIR}/ has no such\n"
                "  file, so CMake configured this as a CPU-only build -- look for\n"
                "  'CUDA Toolkit found' in the configure output above, and for the error that\n"
                "  came after it if it is there."
            )
        if missing:
            reasons.append(
                f"  Not beside the library, and needed by it: {', '.join(missing)}.\n"
                f"  `_copy_runtime_libraries` puts these in {LIB_DIR}/ and something stopped it."
            )
        raise GgufError(
            f"{PACKAGE} does not import in the build environment: {answer['error']}.\n"
            + ("\n".join(reasons) + "\n" if reasons else
               "  It was just installed, so this is the shared library failing to load rather\n"
               "  than a missing package -- a library it was linked against that the loader\n"
               "  cannot find. Usually a CUDA or ROCm runtime; on a BLAS build, OpenBLAS's own\n"
               f"  library, which `_copy_runtime_libraries` puts in {LIB_DIR}/ beside it.\n")
            + f"  Everything in that directory has to resolve from inside it: {PACKAGE} adds\n"
            "  only that one directory to the search path."
        )

    info = answer.get("info") or ""
    build.system_info = info
    backends = answer.get("backends")
    print(f"      {PACKAGE} {answer.get('version') or '?'} from {answer.get('lib') or '?'}")
    if answer.get("note"):
        print(f"      note: {answer['note']}")
    if isinstance(backends, list) and backends:
        print(f"        backends: {', '.join(backends)}")
    for name, description in answer.get("devices") or []:
        print(f"        device {name}: {description}" if description else f"        device {name}")
    if info:
        for line in info.strip().splitlines():
            print(f"        {line.strip()}")

    names = BACKEND_NAMES.get(build.target.kind)
    marker = BACKEND_MARKER.get(build.target.kind)
    if names:
        # The registry is authoritative. The string is a fallback for a future
        # llama.cpp that stops exporting these symbols, and reports nothing
        # rather than a negative when it cannot answer either.
        present = _registered(backends, names)
        if present is None:
            present = _has_backend(info, marker) if info and marker else None
        if present is False:
            reported = ", ".join(backends) if isinstance(backends, list) and backends else None
            raise GgufError(
                f"{PACKAGE} was built for {build.target.label}, and the library reports no "
                f"{marker} backend.\n"
                f"  What it reports is: {reported or info.strip()[:400] or 'nothing'}\n"
                "  A GPU distribution with a CPU-only llama.cpp runs, and runs the GGUF\n"
                "  translators an order of magnitude slower with nothing in the log to say so,\n"
                "  which is why this is fatal. The arguments used were:\n"
                f"      {build.value}"
            )
        if present is None:
            print(f"      note: this build cannot report its backends, so {marker} is unverified")

    if build.openblas is not None:
        # Not fatal either way: BLAS is prompt-processing speed and nothing else,
        # and a build that quietly did without it is still a correct build.
        blas = _registered(backends, (BLAS_BACKEND,))
        if blas is None and info:
            blas = _has_backend(info, BLAS_BACKEND) or None
        if blas is False:
            print("      warning: OpenBLAS was found and passed to CMake, and the library reports")
            print("               no BLAS backend. Check the configure output above.")
        elif blas is None:
            print("      note: this build cannot report its backends, so BLAS is unverified")

    if build.target.is_gpu and answer.get("gpu_offload") is False:
        # Expected on a machine with the toolkit and no card: this asks the
        # runtime what devices exist, not what the build supports.
        print("      note: no GPU visible here, so offload could not be exercised")


# ── Shipping it ───────────────────────────────────────────────────────────────

def _package_lib_dir(python: Path, env: dict[str, str] | None = None) -> Path | None:
    """Where the installed llama_cpp keeps its shared libraries."""
    code, text = _capture(
        # `find_spec` rather than `import`: importing opens the shared library, and
        # this is also called before the DLLs that library needs are in place.
        [str(python), "-c",
         "import importlib.util as u, sys; s = u.find_spec('llama_cpp');"
         " sys.stdout.write(list(s.submodule_search_locations)[0] if s else '')"],
        env=env,
    )
    if code != 0 or not text.strip():
        return None
    lib = Path(text.strip().splitlines()[-1].strip()) / LIB_DIR
    return lib if lib.is_dir() else None


def bundle(dist: Path, python: Path, build: Build, env: dict[str, str] | None = None) -> None:
    """Copy llama.cpp's libraries, and OpenBLAS, into the compiled tree.

    Nuitka does not do this and cannot be asked to: `llama_cpp` is pure Python
    that opens its library with ctypes, and ``--include-package-data`` is
    documented as covering data files rather than DLLs. So the same treatment the
    C runtime gets (see `bundle_runtime_dlls` in build.py) -- find what is
    missing, copy it in.

    The layout is not negotiable. `llama_cpp/llama_cpp.py` computes its library
    directory from ``__file__`` at import time, so it has to be
    ``<dist>/llama_cpp/lib/``, and on Windows that is also the directory it adds
    to the DLL search path, which is what lets the copies here resolve each
    other.
    """
    source = _package_lib_dir(python, env)
    if source is None:
        raise GgufError(
            f"{PACKAGE} is installed but has no {LIB_DIR}/ directory, so there is nothing to "
            "ship.\n  That is a build that produced no shared library; look at the install "
            "output above."
        )

    target = dist / PACKAGE / LIB_DIR
    target.mkdir(parents=True, exist_ok=True)

    #: Everything already anywhere in the tree, by name. A CUDA build's llama.cpp
    #: links the same cublas and cudart torch ships, and CMake copies them in
    #: beside its own libraries -- half a gigabyte of files that are already
    #: here. Skipped rather than copied, which is safe *on Windows* because the
    #: entry point registers both directories on the DLL search path before
    #: anything is imported (`_register_dll_directories` in
    #: packaging/fox_reader_main.py). Each package does register its own
    #: directory on import, but only its own and only when it is imported, and
    #: the GGUF translator never imports torch.
    have = {path.name.lower(): path for path in dist.rglob("*") if path.is_file()}

    #: The exception, and it is a platform difference rather than a special case.
    #: There is no `add_dll_directory` on Linux and nothing that does its job: an
    #: ELF finds its dependencies through the RUNPATH recorded at link time, and
    #: llama-cpp-python's CMakeLists pins that to ``$ORIGIN`` per target. So the
    #: only directory ``libggml-cuda.so`` will ever search is the one it is in,
    #: and a copy elsewhere in the tree is a copy it cannot reach. These are the
    #: files `_copy_runtime_libraries` put here for exactly that reason; skipping
    #: them because torch has its own would undo it.
    pinned: set[str] = set()
    if not IS_WINDOWS:
        pinned = {path.name.lower() for path in _cuda_runtime(python, build, env)}
        pinned |= {path.name.lower() for path in
                   (build.openblas.runtime if build.openblas is not None else ())}

    copied, skipped = [], []
    for entry in sorted(source.iterdir()):
        if not entry.is_file():
            continue
        existing = have.get(entry.name.lower())
        if existing is not None and existing.parent != target and entry.name.lower() not in pinned:
            skipped.append((entry.name, existing.relative_to(dist)))
            continue
        shutil.copy2(entry, target / entry.name)
        copied.append(entry.name)

    for name in copied:
        size = (target / name).stat().st_size / (1024 * 1024)
        print(f"      + {PACKAGE}/{LIB_DIR}/{name}  ({size:.1f} MB)")
    for name, where in skipped:
        print(f"      = {name} already in {where.parent}")

    if not any((target / name).is_file() for name in copied) and not skipped:
        raise GgufError(f"nothing was copied out of {source}, so the GGUF backend cannot load")

    if build.openblas is not None:
        for path in build.openblas.runtime:
            destination = target / path.name
            if destination.is_file():
                continue
            if path.is_symlink():
                # Copy what it points at under the link's own name: the library
                # is asked for by SONAME, and a dangling link ships nothing.
                shutil.copy2(path.resolve(), destination)
            else:
                shutil.copy2(path, destination)
            size = destination.stat().st_size / (1024 * 1024)
            print(f"      + {PACKAGE}/{LIB_DIR}/{destination.name}  ({size:.1f} MB, OpenBLAS)")


# ── Standalone ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", type=str.lower,
                        help="the device to generate arguments for (cpu, cu129, rocm72, macos)")
    parser.add_argument("--openblas", default="auto", choices=["auto", "download", "off"],
                        help="what to do about OpenBLAS (default: auto -- use it if installed,"
                             " otherwise download it into nc/ without asking; off builds without it)")
    parser.add_argument("--cpu-baseline", default="avx2", choices=sorted(CPU_BASELINES),
                        help="the x86-64 instruction set floor (default: avx2)")
    parser.add_argument("--cuda-arch", default=None,
                        help="override CMAKE_CUDA_ARCHITECTURES, e.g. '75-real;86-real'")
    parser.add_argument("--hip-arch", default=None, help="override the gfx list, e.g. 'gfx1100'")
    parser.add_argument("--print-args", action="store_true",
                        help="print the CMAKE_ARGS this device would build with and stop")
    parser.add_argument("--find-openblas", action="store_true",
                        help="report the OpenBLAS this machine has and stop")
    parser.add_argument("--install", action="store_true",
                        help="install llama-cpp-python with these arguments, into --python")
    parser.add_argument("--python", default=None, metavar="PATH",
                        help="the interpreter to install into (default: the one running this)")
    args = parser.parse_args(argv)

    try:
        if args.find_openblas:
            found, passed_over = find_openblas()
            for line in passed_over:
                print(f"passed over {line}")
            print(found.summary if found else "no usable OpenBLAS found")
            print(f"  {found.include}\n  {found.library}" if found else f"  looked under {NC_DIR}")
            return 0 if found else 1

        # Resolved before anything else happens because preparing a build can
        # download an OpenBLAS, and a typo in --python should not cost that.
        python = Path(args.python) if args.python else Path(sys.executable)
        if (args.install or args.python) and not python.is_file():
            raise GgufError(
                f"no interpreter at {python}.\n"
                "  --python takes the python of the environment to install into, which for\n"
                "  a uv project is .venv/Scripts/python.exe (Windows) or .venv/bin/python."
            )

        target = parse_device(args.device)
        try:
            build = prepare(
                target,
                openblas_mode=args.openblas,
                cpu_baseline=args.cpu_baseline,
                cuda_arch=args.cuda_arch,
                hip_arch=args.hip_arch,
            )
        except GgufError as exc:
            if not args.print_args:
                raise
            # Looking at the arguments is worth doing on a machine that could not
            # build them -- that is most of why one would look. The check that
            # stopped this is the same one a build would fail on, so it is said
            # in full, and then the answer is given anyway.
            print(f"this machine cannot build for {target.label}:\n  {exc}\n")
            plain = cmake_args(
                target,
                cpu_baseline=args.cpu_baseline,
                cuda_arch=args.cuda_arch,
                hip_arch=args.hip_arch,
                toolkit=target.cuda,
            )
            print(f"unfiltered, for {target.label} ({target.device}):")
            print(f'CMAKE_ARGS="{cmake_args_value(plain)}"')
            return 1
        print(f"{target.label} ({target.device}), {build.spec}")
        print(f'CMAKE_ARGS="{build.value}"')
        generator = build.env.get("CMAKE_GENERATOR")
        if generator:
            # Part of the answer, not decoration: without it scikit-build-core picks
            # NMake, which is not on PATH outside a Developer Command Prompt.
            print(f'CMAKE_GENERATOR="{generator}"')
        for name in ("CFLAGS", "CXXFLAGS", "LDFLAGS", "CUDAFLAGS"):
            # Also part of the answer -- a MinGW build needs the Windows version
            # macros to see CreateFile2 at all, and a CUDA build needs nvcc told to
            # keep the environment it was handed. PATH and the MSVC variables are not
            # printed: they are this machine's, not the recipe's.
            if build.env.get(name):
                print(f'{name}="{build.env[name]}"')
        for note in build.notes:
            print(f"  note: {note}")

        if args.install:
            print(f"\ninstalling into {python}")
            build = install(python, build)
            verify(python, build)
            return 0

        if not args.print_args:
            print("\nNothing was installed. --install does that, into the interpreter running\n"
                  "this or the one given by --python; packaging/build.py does it for a\n"
                  "distribution. --print-args says 'arguments only' explicitly.")
    except GgufError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
