#!/usr/bin/env python3
"""Fox Reader build script.

Compiles the backend to machine code with Nuitka, embeds the frontend inside the
binary, builds the C launcher, and stages the result as::

    fox-reader/
      launcher(.exe)   the only thing the user starts
      LICENSE/NOTICE/TERMS.md  legal: AGPL text, attributions, terms of use
      bin/             the compiled backend and everything it links against
      cache/           empty; scratch, cleared at startup and at shutdown
      config/          fox_config.yaml, and nothing else
      fonts/           the user's fonts; the fallbacks live in the binary
      models/          empty; weights are downloaded on first run

Nuitka rather than PyInstaller for one reason: PyInstaller ships ``.pyc`` files
that decompile cleanly, so the "compiled" build hands the source to anyone who
unzips it. Nuitka turns Python into C and then into the platform's object code.
The frontend gets the same treatment -- ``packaging/embed_assets.py`` turns
``frontend/`` into a module of byte constants that is compiled in, so there is no
``frontend/`` folder in a shipped build either.

Two compile scopes:

*Default.* ``fox_reader`` and the entry point are compiled to C; third-party
packages are included as bytecode. Five to fifteen minutes, and the code that is
ours -- which is the code worth protecting -- is the code that gets compiled.

*``--deep-compile``.* Everything Nuitka can compile, compiles. Hours, hundreds of
thousands of C functions, and a real chance of tripping over one package's
introspection. Worth it for a release build; not for iterating.

Usage::

    python packaging/build.py                      # cpu, default scope
    python packaging/build.py --device cu129       # bundle a CUDA torch
    python packaging/build.py --deep-compile       # compile the dependencies too
    python packaging/build.py --skip-frontend      # reuse a built frontend/static
    python packaging/build.py --clean              # discard previous artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# GitHub Actions (and any other pipe) leaves stdout block-buffered, while the
# subprocesses this script drives (uv, Nuitka, ...) inherit the raw file
# descriptor and stream past Python's buffer. The result is inverted logs: the
# child's downloads appear first and every print() above flushes at exit.
# Line-buffer so each line reaches the log before the child output after it.
# Locally this changes nothing: a TTY is line-buffered already.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
del _stream

# pyrefly: ignore [missing-import]
import llamacpp  # noqa: E402

# pyrefly: ignore [missing-import]
from toolchain import LAUNCHER_NAME, ToolchainError, build_launcher, find_compiler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SRC_DIR = PROJECT_ROOT / "src"

VENV_DIR = BUILD_DIR / ".build-venv"

#: Everything this script generates lives here, so `--clean` is one rmtree and
#: `.gitignore` is one line.
WORK_DIR = BUILD_DIR / "build"
#: Where the Nuitka entry point and the generated asset module are assembled.
#: Nuitka puts the main script's directory on the module search path, which is
#: what makes `fox_reader_assets` importable from the entry.
ENTRY_DIR = WORK_DIR / "entry"
#: Nuitka's output: `<here>/fox_reader_main.dist/` and `.build/`.
NUITKA_OUT = WORK_DIR / "nuitka"
#: The finished tree, before it is archived.
STAGE_DIR = WORK_DIR / "stage"

ENTRY_SOURCE = BUILD_DIR / "fox_reader_main.py"
EMBED_SCRIPT = BUILD_DIR / "embed_assets.py"

#: What the launcher looks for in `bin/`. Must stay in step with
#: `BACKEND_NAMES` in packaging/launcher/launcher.c.
BACKEND_NAME = "fox-reader.exe" if os.name == "nt" else "fox-reader"

STAGE_NAME = "fox-reader"

OS_ENV = os.environ.copy()
OS_ENV["UV_PROJECT_ENVIRONMENT"] = str(VENV_DIR)

TOTAL_STEPS = 14
_step = 0


class BuildError(RuntimeError):
    """Anything that should stop the build with a readable message."""


# ── Packages Nuitka has to be told about ──────────────────────────────────────
# Carried over from the PyInstaller build, where each of these was found by a
# build that failed without it. Every name is checked against the build venv
# before it is passed to Nuitka (see `_present`), because a package that renames
# or drops a submodule between versions would otherwise turn into a hard build
# failure on a flag that is only there as insurance.

#: Data files that live inside the package and are read at runtime -- config
#: yaml, vocabularies, compiled kernels. PyInstaller's `--collect-data`.
PACKAGE_DATA = (
    "torch",
    "torchvision",
    "paddlex",
    "paddleocr",
    "pypdfium2",
    "cv2",
    "transformers",
    "sentencepiece",
    "huggingface_hub",
)

#: Whole packages to include that may not be installed at all.
#:
#: `llama_cpp` is the GGUF backend. It is not in the locked dependency set -- it
#: is compiled against this machine's CUDA/ROCm/BLAS by `install_llama_cpp`
#: after the sync, which is the only way to get a llama.cpp that matches the
#: device (see packaging/llamacpp.py) -- so `--no-gguf`, or a build host with no
#: toolchain, leaves it absent and everything below has to cope with that. The
#: import that reaches it is inside a function
#: (`translate.machine_translation.gemma_e4b_q8_llamacpp._import_llama_cpp`), and
#: naming the package here rather than leaning on that keeps the pieces it only
#: reaches lazily -- the chat-format handlers, the tokenizer shims -- in the
#: bundle too.
OPTIONAL_PACKAGES = ("llama_cpp",)

#: Distributions whose *metadata* is read at runtime, by
#: `importlib.metadata.version` and friends. Without these the package imports
#: fine and then raises PackageNotFoundError deep inside a version check.
#:
#: Each of these ships twice: compiled into the binary by
#: `--include-distribution-metadata`, and as its real `.dist-info` directory
#: copied into `bin/`. The second is not redundant -- see `bundle_metadata`,
#: which is where the reason lives.
DISTRIBUTION_METADATA = (
    "torch",
    "torchvision",
    "paddlex",
    "paddleocr",
    "pypdfium2",
    "transformers",
    # transformers checks this version at `import transformers`, but only when
    # the package is importable -- which it always is here, since OCR loads
    # through `device_map`. Without it the backend dies on its first import.
    "accelerate",
    "huggingface_hub",
    # Xet is huggingface_hub's chunked transfer backend, and `is_xet_available()`
    # decides whether to use it by reading this metadata -- not by importing the
    # module, which is bundled either way. Without it every model download prints
    # "Xet Storage is enabled for this repo, but the 'hf_xet' package is not
    # installed" and falls back to plain HTTP.
    "hf-xet",
    "safetensors",
    "tokenizers",
    "sentencepiece",
    "numpy",
    # opencv-contrib-python, not opencv-python: the two install the same `cv2`
    # module and only one of them is a dependency here. Naming both would make
    # the metadata step look for a distribution that was never installed.
    "opencv-contrib-python",
    "pillow",
    "pyyaml",
    "regex",
    "requests",
    "tqdm",
    "packaging",
    "filelock",
    # Nothing here asks for an email address. FastAPI does: one field in its own
    # `openapi.models` is annotated `EmailStr`, and building that model's schema
    # runs pydantic's `import_email_validator`, which checks the *version* right
    # after the import. So `import fastapi` reads this metadata, and without it
    # the backend dies on its first import rather than at some later feature.
    #
    # The other `importlib.metadata` readers in the dependency tree are all
    # reachable but harmless, which is why they are not listed: `optree` and
    # `websockets` catch PackageNotFoundError (and websockets' is behind
    # `if not released`), attrs/click/markupsafe/itsdangerous only look when
    # something reads their deprecated `__version__`, and pydantic's own lookup
    # needs a pydantic-core mismatch on 3.13+ to be reached at all.
    "email-validator",
)

#: The same, for distributions that are only sometimes installed. Anything here
#: that is missing is skipped without comment, so this list is not a promise the
#: staged build has to keep -- which is the whole difference from the list above.
#:
#: llama-cpp-python is here rather than there because `--no-gguf` is a supported
#: build. Nothing loads it by metadata either: `llama_cpp.__version__` is a
#: literal in its `__init__.py`, not an `importlib.metadata` lookup. It ships so
#: that the version the user is running is answerable -- by the settings page,
#: and by a bug report.
OPTIONAL_DISTRIBUTION_METADATA = ("llama-cpp-python",)

#: Spellings that something in the dependency tree passes to
#: `importlib.metadata.version`, where that spelling is not the name the
#: distribution gives itself.
#:
#: A real install does not care: `importlib.metadata` compares names with the
#: dashes, underscores and case flattened out (PEP 503), so
#: `version("huggingface-hub")` finds the project that calls itself
#: `huggingface_hub`. A compiled build does care, which is the whole subject of
#: `bundle_metadata`. Listed here so the build can prove the lookups still
#: resolve rather than leaving it to be discovered at a user's first OCR run.
#:
#: transformers' own list is read from the build venv on top of this one, so a
#: version of it that checks something new fails the build instead of the app.
METADATA_LOOKUPS = (
    # transformers, at `import transformers`. The first two are in
    # DISTRIBUTION_METADATA under the names their metadata declares -- and those
    # names are `huggingface_hub` and `PyYAML`, neither of which is what is asked
    # for here. `accelerate` needs no normalisation, but it is transformers'
    # fatal lookup for the `device_map` path, so it is pinned here too:
    # removing it from DISTRIBUTION_METADATA must fail in a second, not fifteen
    # minutes into a build.
    "huggingface-hub",
    "pyyaml",
    "accelerate",
    "accelerate",
    # huggingface_hub, deciding whether Xet is available. Its distribution calls
    # itself `hf-xet`; the lookup uses the module spelling.
    "hf_xet",
    # huggingface_hub again, reporting what it found in its user agent. Harmless
    # when it answers wrongly, but it is one normalisation away from correct.
    "Pillow",
)

#: Imported by name at runtime, so nothing static can see them. uvicorn's `auto`
#: modules pick an implementation with an import inside a function; scipy.signal
#: is reached through one of paddle's lazy loaders.
HIDDEN_MODULES = (
    "fox_reader.app",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "scipy.signal",
)

#: The OCR model code. transformers resolves these from the model's config at
#: load time -- the string is in a downloaded json file, so there is no import
#: anywhere for Nuitka to follow.
OCR_PACKAGES = tuple(
    f"transformers.models.{name}"
    for name in (
        "pp_ocrv5_server_det",
        "pp_ocrv5_server_rec",
        "pp_ocrv5_mobile_det",
        "pp_ocrv5_mobile_rec",
        "pp_ocrv6_small_det",
        "pp_ocrv6_small_rec",
        "pp_ocrv6_medium_det",
        "pp_ocrv6_tiny_rec",
    )
)

#: Never worth compiling or including as bytecode: build tooling that is in the
#: venv to make the build happen, not to be part of it.
NOT_A_DEPENDENCY = {
    "nuitka",
    "pip",
    "pkginfo",
    "wheel",
    "pytest",
    "_pytest",
    "py",
    "mypy",
    "mypy_extensions",
    "mypyc",
    "ruff",
    "pyinstaller",
    "PyInstaller",
    "fox_reader",
    "fox_reader_assets",
}

#: Nuitka plugins to turn on when the installed version advertises them and does
#: not already have them on. Each is a package that needs more than
#: import-following to work standalone.
#:
#: Not `torch`: Nuitka 4 split that plugin into `torch-hub` (scans Torch Hub
#: repositories) and `torch-jit` (keeps the source torch.jit would inspect), and
#: this build does neither. The old single name is kept so an older Nuitka still
#: gets what it needs; on 4.x it simply does not match anything.
WANTED_PLUGINS = ("torch", "multiprocessing")

#: CUDA runtime libraries, removed from a CPU build. torch ships them whether or
#: not the wheel can use them, and they are most of the download.
CUDA_PATTERNS = (
    "*torch_cuda*", "*c10_cuda*",
    "*libcusparse*", "*libcurand*", "*libcudnn*",
    "*libcublasLt*", "*libcublas*", "*libcupti*",
    "*libcufft*", "*libcudart*", "*libnv*", "*libnccl*",
    "*cublas*", "*cudnn*", "*cusparse*", "*curand*",
    "*cufft*", "*cupti*", "*cusolver*", "*cusolverMg*",
    "*nccl*", "*nvrtc*", "*nvJitLink*",
)

#: Windows will not accept a command line longer than 32,767 characters, and the
#: bytecode flags are the only part of ours that scales with the environment.
#: Measured against a full install: 162 third-party top-levels come to about
#: 15,800 characters, so this is a backstop rather than a limit anyone will hit.
COMMAND_LIMIT = 30_000


# ── Small helpers ─────────────────────────────────────────────────────────────

def step(title: str) -> None:
    global _step
    _step += 1
    # flush=True: must be visible before the subprocess output that follows,
    # even where the line-buffering above could not be applied.
    print(f"[{_step}/{TOTAL_STEPS}] {title}", flush=True)


def _read_version() -> str:
    """The project version, read without importing anything."""
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise BuildError("could not find a version in pyproject.toml")
    return match.group(1)


def _file_version(version: str) -> str:
    """A version Windows will accept in a resource: exactly four numbers."""
    parts = re.findall(r"\d+", version)[:4]
    parts += ["0"] * (4 - len(parts))
    return ".".join(parts)


#: What a device extra is called. `cpu`, `macos`, anything CUDA, anything ROCm --
#: as opposed to `dev`, `gguf`, `bin-build` and `segmentation`, which are also
#: extras and are not a choice of torch build.
_DEVICE_EXTRA = re.compile(r"^(?:cpu|macos|cu\d+.*|rocm.*)$")

#: The devices as they stood when this was written, used only if pyproject cannot
#: be read for them. `--help` staying useful is worth more than being strict here.
_DEVICE_FALLBACK = ("cpu", "macos", "cu118", "cu126", "cu129", "cu129_win", "cu130", "rocm71", "rocm72")


def _device_choices() -> list[str]:
    """The device extras pyproject declares, in the order it declares them.

    Read rather than hard-coded so that adding `cu131 = [...]` to pyproject is all
    it takes for `--device cu131` to work -- the CUDA arguments llama.cpp needs are
    derived from the number too (see packaging/llamacpp.py), so nothing else in the
    build has a list of versions in it either.
    """
    try:
        text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return list(_DEVICE_FALLBACK)

    block = re.search(
        r"^\[project\.optional-dependencies\]\s*$(.*?)(?=^\[)", text, re.MULTILINE | re.DOTALL
    )
    if not block:
        return list(_DEVICE_FALLBACK)

    found = [
        name
        for name in re.findall(r"^([A-Za-z0-9._-]+)\s*=", block.group(1), re.MULTILINE)
        if _DEVICE_EXTRA.match(name.lower())
    ]
    # `cpu` is the default, so its absence means this parse found the wrong thing.
    return found if "cpu" in found else list(_DEVICE_FALLBACK)


def _platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        return "windows-x64"
    if system == "darwin":
        mac_arch = "macOS-arm64" if machine in ("arm64", "aarch64") else "macOS-x64"
        print(f"{mac_arch} Detected. You need to perform the ritual of compiling from source.")
        sys.exit(0)
    if system == "linux":
        return "linux-x64"
    raise BuildError(f"Unsupported: {system} {machine}")


def _find_uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise BuildError("uv not found. Install with: pip install uv")
    return uv


def _flush_stdio() -> None:
    """Push buffered prints out before a child writes to the same log.

    With line buffering every print already flushed on its newline; this is
    the backstop for a stdout that could not be reconfigured (replaced
    stream, odd embedder): without it the `  $ ...` line sits in the buffer
    while the command it names runs, and the log reads backwards.
    """
    try:
        sys.stdout.flush()
    except (AttributeError, ValueError):
        pass
    try:
        sys.stderr.flush()
    except (AttributeError, ValueError):
        pass


def _run(cmd: list[str], **kwargs) -> None:
    """Run a command, raising on failure."""
    if platform.system() == "Windows" and cmd:
        resolved = shutil.which(cmd[0])
        if resolved:
            cmd = [resolved] + cmd[1:]
    print(f"  $ {' '.join(cmd)}", flush=True)
    cwd = str(kwargs.pop("cwd", PROJECT_ROOT))
    _flush_stdio()
    subprocess.run(cmd, cwd=cwd, check=True, **kwargs)


def _capture(cmd: list[str], timeout: int = 300) -> tuple[int, str]:
    """Run a command for its output. Never raises on a non-zero exit."""
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    out = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
    return proc.returncode, out


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _rmtree(directory: Path, *, attempts: int = 5) -> None:
    """`shutil.rmtree`, on Windows, where a plain rmtree is not reliable.

    Two things break it here, and neither is our doing.

    A **read-only file** cannot be unlinked on Windows even by its owner, and the
    venv is full of candidates: uv populates it by hardlinking out of its cache,
    and a hardlink shares the attributes of the file it points at. So the
    read-only bit is cleared and the operation retried, which is what every other
    packaging tool ends up doing.

    A **file still mapped into a process** cannot be unlinked either, and it stays
    that way for a moment after the process is gone -- Windows releases the section
    when the last handle closes, not when the process exits. Antivirus scanning a
    DLL that was just written holds one just as effectively, and a `.pyd` from a
    build that was interrupted seconds ago is the common case. So the sweep is
    retried a few times with a widening pause, which is enough for a lock that is
    on its way out and no help at all for one that is not -- and the difference
    between those two is the whole point of the message at the end.
    """
    def clear_readonly(func, path, exc):
        # Anything this cannot fix is re-raised, for the retry loop to sit out or
        # the message below to explain.
        os.chmod(path, stat.S_IWRITE)
        func(path)

    for attempt in range(attempts):
        try:
            shutil.rmtree(directory, onexc=clear_readonly)
            return
        except OSError as exc:
            if attempt == attempts - 1:
                held = getattr(exc, "filename", None) or directory
                raise BuildError(
                    f"could not remove {directory}: {exc.strerror or exc}\n"
                    f"  Something still holds {held}.\n"
                    "  Close anything running out of that folder -- an editor with the\n"
                    "  tree open, a running build, an antivirus scan mid-file -- and try\n"
                    "  again. --keep-venv builds without removing the environment."
                ) from exc
            time.sleep(0.5 * (attempt + 1))


def _fresh(directory: Path) -> Path:
    if directory.exists():
        _rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def _tree_size(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


# ── Asking the build venv what it actually has ────────────────────────────────

_PROBE = r"""
import json, os, re, sys
import importlib.util as iu
import importlib.metadata as im

request = json.loads(sys.stdin.read())


def normalize(name):
    # PEP 503. This is how importlib.metadata compares two names, and the reason
    # `version("huggingface-hub")` resolves a project named `huggingface_hub`.
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def dist_info_dir(dist, name):
    # The `.dist-info` directory this distribution was installed from, if it has
    # one. `_path` is what importlib.metadata's own PathDistribution carries, and
    # covers every wheel install; the scan below is for anything that does not.
    path = getattr(dist, "_path", None)
    if path is not None:
        try:
            candidate = os.fspath(path)
        except TypeError:
            candidate = None
        if (candidate and candidate.endswith((".dist-info", ".egg-info"))
                and os.path.isdir(candidate)):
            return candidate

    try:
        base = os.fspath(dist.locate_file(""))
        entries = sorted(os.listdir(base))
    except Exception:
        return ""

    wanted = normalize(name)
    for entry in entries:
        if not entry.endswith((".dist-info", ".egg-info")):
            continue
        # `hf_xet-1.6.0.dist-info` -> `hf_xet`. A wheel writes the name with
        # underscores and joins it to the version with the first dash, so
        # splitting there is exact rather than a guess.
        stem = entry.rsplit(".", 1)[0].split("-")[0]
        full = os.path.join(base, entry)
        if normalize(stem) == wanted and os.path.isdir(full):
            return full
    return ""


def resolves(name):
    # Whether `name` imports to a real file, as opposed to not at all or to a
    # namespace package. Nuitka rejects a --include-module it cannot locate, and
    # a namespace package has nothing to locate.
    if not name:
        return False
    for part in name.split("."):
        if not part.isidentifier() or part == "__init__":
            return False
    try:
        spec = iu.find_spec(name)
    except Exception:
        return False
    return spec is not None and spec.origin not in (None, "namespace")


def prune(names):
    # Keep the plain top-level names and drop the submodules, which Nuitka never
    # asks for: `top_level.txt` order is preserved, so its first entry -- the only
    # one Nuitka checks -- is still first here. sentencepiece is the reason this
    # exists: it lists four of its own submodules alongside itself, two of them
    # protobuf message modules that would drag protobuf into the bundle to satisfy
    # a requirement nothing actually made. A distribution that offers nothing but
    # submodules keeps them, since then one of those is what Nuitka will want.
    resolved = [n for n in names if resolves(n)]
    plain = [n for n in resolved if "." not in n]
    return plain or resolved


def top_levels(dist):
    # The importable top-level names a distribution provides.
    #
    # This mirrors Nuitka's own derivation (getDistributionTopLevelPackageNames):
    # top_level.txt if it says anything usable, otherwise the first path element
    # of every file the distribution installed. Nuitka needs only the *first*
    # name -- it is the one it insists on finding in the module graph before it
    # will bundle the metadata -- but the whole set is returned, because matching
    # its ordering exactly is more fragile than covering it.
    names = []

    def add(name):
        if name and name not in names:
            names.append(name)

    try:
        text = dist.read_text("top_level.txt")
    except Exception:
        text = None
    if text:
        for line in text.splitlines():
            add(line.strip().replace("\\", "/").replace("/", "."))
        found = prune(names)
        # Some wheels list a `__dummy__` here and nothing else, which is why this
        # only counts as an answer if something in it actually imports.
        if found:
            return found
        names = []

    try:
        files = dist.files or []
    except Exception:
        files = []
    for entry in files:
        path = str(entry).replace("\\", "/")
        if path.startswith("."):
            continue
        head, _, remainder = path.partition("/")
        if head.endswith((".dist-info", ".egg-info")) or head == "__pycache__":
            continue
        if not remainder:
            # A file directly in site-packages: a module only if it looks like one.
            for suffix in (".py", ".pyc", ".pyd", ".so", ".pyi"):
                if head.endswith(suffix):
                    head = head[: -len(suffix)].split(".")[0]
                    break
            else:
                continue
        add(head)
    return prune(names)


modules = []
for name in request["modules"]:
    if resolves(name):
        modules.append(name)

dists = []
dist_names = []
dist_modules = []
dist_info = []
for name in request["dists"]:
    try:
        found = im.distribution(name)
    except Exception:
        continue
    dists.append(name)
    # Nuitka matches a distribution name literally; importlib.metadata does not.
    # `im.distribution("pyyaml")` happily resolves the project that calls itself
    # `PyYAML`, so the requested spelling can be one Nuitka then warns about and
    # cannot look up. Ask the metadata what its own name is.
    try:
        real = (found.metadata["Name"] or "").strip()
    except Exception:
        real = ""
    dist_names.append(real or name)
    dist_info.append(dist_info_dir(found, real or name))
    for module in top_levels(found):
        if module not in dist_modules:
            dist_modules.append(module)

paths = []
if request["sites"]:
    import sysconfig
    got = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        value = got.get(key)
        if value and value not in paths:
            paths.append(value)

json.dump({"modules": modules, "dists": dists, "dist_names": dist_names,
           "dist_modules": dist_modules, "dist_info": dist_info, "sites": paths},
          sys.stdout)
"""

# The probe is source held in a string, so a mistake in it would surface as a
# confusing subprocess failure minutes into a build -- and, because the string is
# triple-quoted, a docstring written inside it would end the string early and
# break this module instead. Compile it here so either mistake is immediate.
compile(_PROBE, "<probe>", "exec")


#: Asks the build venv which OpenCV it has, and whether the contrib half is real.
#: Tolerant of every broken state on purpose -- a hollow cv2 has to come back as
#: data to be repaired, not as a traceback.
_OPENCV_PROBE = r"""
import importlib.metadata as im, json, sys


def norm(name):
    return (name or "").strip().lower().replace("_", "-")


installed = sorted({
    norm(dist.metadata["Name"]) for dist in im.distributions()
    if norm(dist.metadata["Name"]).startswith("opencv")
})

version, contrib, error = None, {}, None
try:
    import cv2
except Exception as exc:
    error = "%s: %s" % (type(exc).__name__, exc)
else:
    # Missing on a cv2 whose binaries were carried off by the other
    # distribution's uninstall, which leaves the package importable and empty.
    version = getattr(cv2, "__version__", None)
    # Presence proves nothing either: a namespace folder the other distribution
    # left behind imports perfectly and exports not one symbol. So count them.
    for name in ("xphoto", "ximgproc", "bgsegm"):
        module = getattr(cv2, name, None)
        contrib[name] = 0 if module is None else len([n for n in dir(module) if not n.startswith("_")])

json.dump({"dists": installed, "version": version, "contrib": contrib, "error": error}, sys.stdout)
"""

compile(_OPENCV_PROBE, "<opencv-probe>", "exec")

#: The contrib namespaces worth insisting on. `xphoto` is the one that matters --
#: `xphoto.inpaint` is a clean method the app offers -- and the other two come
#: from the same wheel, so all three being hollow together is the signature being
#: looked for rather than three separate problems.
_CONTRIB_MODULES = ("xphoto", "ximgproc", "bgsegm")


def _opencv_state(python: Path) -> dict:
    """What the build venv's cv2 actually is. Raises only if it cannot be asked."""
    proc = subprocess.run([str(python), "-c", _OPENCV_PROBE], capture_output=True, env=_build_env())
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace"))
    except ValueError as exc:
        raise BuildError(
            f"Could not ask the build environment about OpenCV ({exc}).\n"
            + (proc.stderr.decode("utf-8", "replace").strip() or "  no output")
        ) from exc


def verify_opencv(python: Path, repair_cmd: list[str] | None = None) -> None:
    """Insist the one OpenCV present is a working contrib build, repairing once.

    Both distributions unpack into a single shared ``cv2`` package, and that has
    two consequences worth checking for. Install both and whichever wrote last
    owns the shared modules while the loser's namespace folders stay behind, so
    ``cv2.xphoto`` imports cleanly and holds nothing -- inpainting then degrades
    at runtime with no import error anywhere to point at it. Remove one of them
    afterwards and its uninstall takes the shared binaries with it, leaving the
    survivor a hollow package that still satisfies ``import cv2``.

    The second is repairable and worth repairing rather than reporting, because
    it is the normal state of a venv that has just been synced onto this
    lockfile from an older one.
    """
    print("      Checking OpenCV...")
    state = _opencv_state(python)

    intact = state["version"] and not any(not state["contrib"].get(m) for m in _CONTRIB_MODULES)
    if not intact and repair_cmd:
        print("        cv2 is incomplete; reinstalling opencv-contrib-python...", flush=True)
        print(f"  $ {' '.join(repair_cmd)}", flush=True)
        _flush_stdio()
        subprocess.run(repair_cmd, env=OS_ENV)
        state = _opencv_state(python)

    if state["dists"] != ["opencv-contrib-python"]:
        raise BuildError(
            "The build environment has the wrong OpenCV: "
            + (", ".join(state["dists"]) or "none at all") + ".\n"
            "  Only opencv-contrib-python belongs here; the two distributions share one\n"
            "  cv2/ directory and overwrite each other. pyproject overrides the plain\n"
            "  one away, so a lockfile older than that override is the likely cause:\n"
            "      uv lock"
        )

    if state["error"] or not state["version"]:
        raise BuildError(
            "cv2 does not work in the build environment: "
            + (state["error"] or "it imports but has no __version__") + ".\n"
            "  Reinstall it over the top of whatever is there:\n"
            "      uv sync --reinstall-package opencv-contrib-python"
        )

    hollow = sorted(m for m in _CONTRIB_MODULES if not state["contrib"].get(m))
    if hollow:
        raise BuildError(
            f"cv2 {state['version']} is installed but its contrib modules are empty "
            f"({', '.join(hollow)}).\n"
            "  That is what opencv-python having been unpacked over the top of\n"
            "  opencv-contrib-python looks like. Reinstall the contrib build:\n"
            "      uv sync --reinstall-package opencv-contrib-python"
        )
    print(f"      cv2 {state['version']} with contrib.")


def _build_env() -> dict[str, str]:
    """The environment the build venv's Python is run in.

    ``fox_reader`` from the checkout and ``fox_reader_assets`` from the entry
    directory, whether or not the venv installed the project. Both the probe and
    Nuitka itself need this, and they need the *same* one -- a probe that cannot
    see ``fox_reader`` would quietly drop the flags that cover it.
    """
    env = OS_ENV.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_DIR), str(ENTRY_DIR), *([existing] if existing else [])]
    )
    return env


def _probe(python: Path, modules: tuple[str, ...] = (), dists: tuple[str, ...] = (),
           sites: bool = False) -> dict:
    """Which of these names the build venv can resolve.

    A flag naming a module that is not installed is a hard error in Nuitka, so
    every name this script would pass is checked first. That is what makes the
    same flag lists survive a torch or transformers upgrade instead of turning
    into a build that fails on a submodule that was renamed upstream.
    """
    payload = json.dumps({"modules": list(modules), "dists": list(dists), "sites": sites})
    try:
        proc = subprocess.run(
            [str(python), "-c", _PROBE],
            input=payload.encode("utf-8"),
            capture_output=True,
            env=_build_env(),
            timeout=900,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BuildError(f"could not inspect the build environment: {exc}") from exc

    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise BuildError(f"could not inspect the build environment:\n{detail}")

    try:
        return json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise BuildError(f"the environment probe returned nothing usable: {exc}") from exc


def _report_missing(kind: str, wanted: tuple[str, ...], found: list[str]) -> None:
    missing = [name for name in wanted if name not in found]
    if missing:
        # Said out loud rather than swallowed: a name going missing is usually an
        # upstream rename, and the flag it was covering is no longer covering it.
        print(f"      note: {len(missing)} {kind} not installed, skipped: {', '.join(missing)}")


#: File suffixes that make a directory (or a lone file) an importable Python
#: module or package. `_third_party_top_levels` uses these to tell real
#: packages apart from the data trees installers leave beside them.
_PYTHON_SUFFIXES = (".py", ".pyc", ".pyo", ".pyd", ".so")


def _holds_python(directory: Path) -> bool:
    """Whether `directory` looks like something Python could import.

    Since llama-cpp-python 0.3.34 the install drops CMake's whole install tree
    -- `bin/` of DLLs, `include/` of headers, `lib/` of import libraries --
    next to the package in site-packages. Each of those is a top-level
    directory that is not a module, and handing one to Nuitka as a
    `--noinclude-custom-mode` name is at best noise and at worst a build that
    fails on a flag. A real package carries Python at its top (an `__init__`
    or a module file) or at worst one level down (a namespace package such as
    `google/` whose subdirectories hold the code), and none of those trees do.
    """
    try:
        children = list(directory.iterdir())
    except OSError:
        return False
    for child in children:
        if child.is_file() and (child.name.startswith("__init__.py")
                                or child.suffix in _PYTHON_SUFFIXES):
            return True
    for child in children:
        if not child.is_dir():
            continue
        try:
            grandchildren = list(child.iterdir())
        except OSError:
            continue
        if any(g.is_file() and (g.name.startswith("__init__.py")
                                or g.suffix in _PYTHON_SUFFIXES)
               for g in grandchildren):
            return True
    return False


def _third_party_top_levels(python: Path, site_paths: list[str]) -> list[str]:
    """Top-level names in the build venv's site-packages.

    These are the packages that get shipped as bytecode in the default scope --
    the ones whose compilation would cost hours and protect somebody else's code.

    Directories that hold no Python (a `bin/` of DLLs, an `include/` of headers
    -- see `_holds_python`) are skipped rather than named: Nuitka can do nothing
    with a bytecode flag for something that is not a module.
    """
    names: set[str] = set()
    for raw in site_paths:
        site = Path(raw)
        if not site.is_dir():
            continue
        for entry in site.iterdir():
            name = entry.name
            if entry.is_dir():
                if name.endswith((".dist-info", ".egg-info", ".data", ".libs", ".dylibs")):
                    continue
                if not _holds_python(entry):
                    continue
            elif entry.is_file() and entry.suffix in _PYTHON_SUFFIXES:
                name = entry.stem
            else:
                continue
            if not name.isidentifier() or name.startswith("__"):
                continue
            if name in NOT_A_DEPENDENCY:
                continue
            names.add(name)
    return sorted(names)


# ── What the installed Nuitka understands ─────────────────────────────────────

class Nuitka:
    """The options and plugins the installed Nuitka advertises.

    Every flag is checked against `--help` before it is used. Nuitka renames
    options between major versions -- `--include-distribution-metadata` did not
    always exist, and the mode names for `--noinclude-custom-mode` have moved --
    so a build script that hard-codes its flags works until it silently doesn't.
    Here a missing option is either fatal (if the build cannot mean anything
    without it) or reported and skipped.
    """

    def __init__(self, python: Path) -> None:
        self.python = python

        code, version = _capture([str(python), "-m", "nuitka", "--version"], timeout=300)
        if code != 0:
            raise BuildError(
                "nuitka is not usable in the build environment.\n"
                f"  {' '.join([str(python), '-m', 'nuitka', '--version'])} said:\n"
                f"{version.strip()}\n"
                "Check that pyproject.toml's `bin-build` extra installs it."
            )
        self.version = version.strip().splitlines()[0].strip() if version.strip() else "unknown"

        code, help_text = _capture([str(python), "-m", "nuitka", "--help"], timeout=300)
        if code != 0 or not help_text.strip():
            raise BuildError("nuitka --help produced nothing; the install is broken")
        self.options = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", help_text))

        code, plugin_text = _capture([str(python), "-m", "nuitka", "--plugin-list"], timeout=300)
        self.plugins: set[str] = set()
        #: Plugins Nuitka switches on by itself. `--plugin-list` marks them
        #: `[auto-enabled]`, and asking for one again earns a warning.
        self.always_on: set[str] = set()
        if code == 0:
            for line in plugin_text.splitlines():
                match = re.match(r"^\s{0,6}([a-z][a-z0-9-]{2,})\s{2,}(\S.*)$", line)
                if match:
                    self.plugins.add(match.group(1))
                    if "[auto-enabled]" in match.group(2):
                        self.always_on.add(match.group(1))

        self.skipped: list[str] = []

    def has(self, option: str) -> bool:
        return option in self.options

    def require(self, option: str) -> None:
        if not self.has(option):
            raise BuildError(
                f"this Nuitka ({self.version}) does not accept {option}, which the build "
                "cannot do without. Pin a version that does, or update packaging/build.py."
            )

    def add(self, cmd: list[str], option: str, value: str | None = None, *,
            required: bool = False, note: str = "") -> bool:
        """Append `option` (with `value`) if Nuitka advertises it."""
        if not self.has(option):
            if required:
                self.require(option)
            self.skipped.append(f"{option}{f'  ({note})' if note else ''}")
            return False
        cmd.append(option if value is None else f"{option}={value}")
        return True

    def add_each(self, cmd: list[str], option: str, values, *, note: str = "") -> bool:
        """Append `option=v` for each value, or record one skip for all of them."""
        values = list(values)
        if not values:
            return True
        if not self.has(option):
            self.skipped.append(f"{option}  ({note or f'{len(values)} values'})")
            return False
        cmd.extend(f"{option}={value}" for value in values)
        return True

    def plugin(self, cmd: list[str], name: str) -> bool:
        option = self.first("--enable-plugins", "--enable-plugin")
        if option is None or name not in self.plugins:
            return False
        if name in self.always_on:
            # Enabling it anyway is not harmless: Nuitka answers with a warning,
            # and a build log that cries wolf is one nobody reads the rest of.
            return False
        cmd.append(f"{option}={name}")
        return True

    def disable_plugin(self, cmd: list[str], name: str) -> bool:
        """Turn a plugin off, if this Nuitka has it to turn off."""
        option = self.first("--disable-plugins", "--disable-plugin")
        if option is None or name not in self.plugins:
            return False
        cmd.append(f"{option}={name}")
        return True

    def module_parameter(self, cmd: list[str], name: str, value: str) -> bool:
        """Answer one of the questions Nuitka's options-nanny would otherwise ask."""
        if not self.has("--module-parameter"):
            return False
        cmd.append(f"--module-parameter={name}={value}")
        return True

    def first(self, *options: str) -> str | None:
        """The first of `options` this Nuitka advertises, newest spelling first.

        Nuitka renames options across major versions and keeps the old name
        working as an undocumented alias -- `--standalone` became `--mode=`,
        `--enable-plugin` became `--enable-plugins`. Probing for one name alone
        concludes the option is gone when it has only moved, so every renamed
        option is looked up through its whole history here.
        """
        for option in options:
            if self.has(option):
                return option
        return None

    def mode_standalone(self, cmd: list[str]) -> None:
        """Ask for a standalone folder build, whatever this Nuitka calls it."""
        if self.has("--mode"):
            cmd.append("--mode=standalone")  # Nuitka 4 and later
        elif self.has("--standalone"):
            cmd.append("--standalone")  # Nuitka 2 and 3
        else:
            raise BuildError(
                f"this Nuitka ({self.version}) advertises neither --mode nor --standalone, "
                "and the build cannot do without a standalone mode. Pin a version that "
                "does, or update packaging/build.py."
            )


# ── Build steps ───────────────────────────────────────────────────────────────

#: Builds `node_modules` by copying rather than by symlinking. Windows only lets
#: an unprivileged process create a directory symlink with Developer Mode on --
#: without it the call fails with EPERM, though junctions still work -- and pnpm
#: links each package's own dependencies with real symlinks. So on a stock
#: Windows box `install` dies partway through linking with a bare
#: "UNKNOWN: unknown error, symlink ..." naming whichever package it reached
#: first, which differs run to run and reads like anything but a permission
#: problem. Only used as a retry: hoisting gives up the isolated linker's
#: strictness about undeclared dependencies, which costs nothing for a package
#: that is built and never published, but there is no reason to give it up on a
#: machine that does not need to.
_PNPM_HOISTED = "--node-linker=hoisted"

#: `pnpm run <script>` re-checks node_modules and silently runs `install` itself
#: when it disagrees -- with default settings, which on Windows means re-entering
#: the symlink step that cannot work and failing the build immediately after the
#: install that just succeeded. Nothing is lost by turning it off here: `install`
#: ran seconds earlier, on the line above.
_PNPM_NO_DEPS_CHECK = "--config.verify-deps-before-run=false"

#: The one file the frontend build exists to produce. Checked afterwards because
#: a zero exit code is not proof: point the bundler at a config that matches no
#: entry and it reports success having written nothing, and the next step would
#: then embed whatever stale copy happened to be lying there.
_FRONTEND_BUNDLE = FRONTEND_DIR / "static" / "app.js"


def _pnpm_failed(what: str, exc: Exception) -> BuildError:
    """A pnpm exit code, explained."""
    return BuildError(
        f"{what} failed ({exc}).\n"
        "  If the output mentioned a symlink, it is the Windows permission: only a\n"
        "  process running with Developer Mode on can create a directory symlink.\n"
        f"  The build already retries `install` with {_PNPM_HOISTED}, which copies\n"
        "  instead, so reaching this means something else is wrong -- check the pnpm\n"
        "  output above rather than the exit code.\n"
        f"  If {FRONTEND_DIR / 'static'} is already current, --skip-frontend builds\n"
        "  with what is there."
    )


def build_frontend() -> None:
    """Build the TypeScript frontend via pnpm."""
    step("Building frontend...")

    before = _FRONTEND_BUNDLE.stat().st_mtime if _FRONTEND_BUNDLE.exists() else None

    try:
        _run(["pnpm", "install"], cwd=FRONTEND_DIR)
    except subprocess.CalledProcessError:
        print("      install failed; retrying with a copied node_modules...")
        try:
            _run(["pnpm", "install", _PNPM_HOISTED], cwd=FRONTEND_DIR)
        except subprocess.CalledProcessError as exc:
            raise _pnpm_failed("pnpm install", exc) from exc

    # Deliberately `pnpm run build` rather than invoking tsc and the bundler here:
    # how the frontend is built belongs in frontend/package.json, and spelling it
    # out again in this file is one more thing to drift.
    try:
        _run(["pnpm", _PNPM_NO_DEPS_CHECK, "run", "build"], cwd=FRONTEND_DIR)
    except subprocess.CalledProcessError as exc:
        raise _pnpm_failed("pnpm run build", exc) from exc

    if not _FRONTEND_BUNDLE.exists():
        raise BuildError(
            f"pnpm run build succeeded without writing {_FRONTEND_BUNDLE}.\n"
            "  Check frontend/vite.config.ts still emits there, and that the build\n"
            "  script in frontend/package.json is the one that produces the bundle."
        )
    if before is not None and _FRONTEND_BUNDLE.stat().st_mtime == before:
        # Untouched, so what follows would embed the previous build's output while
        # reporting this one's. Say so rather than quietly shipping it.
        raise BuildError(
            f"pnpm run build left {_FRONTEND_BUNDLE.name} untouched.\n"
            "  The bundle predates this build, so it would be embedded stale. Delete\n"
            f"  {_FRONTEND_BUNDLE} and build again, or pass --skip-frontend to embed\n"
            "  it deliberately."
        )

    print(f"      Frontend built ({_FRONTEND_BUNDLE.stat().st_size / 1024:.0f} KB).\n")


def create_build_venv(device: str) -> Path:
    """An isolated venv with the project, its device extra, and Nuitka."""
    uv = _find_uv()

    if VENV_DIR.exists() and _venv_python().exists():
        step("Reusing existing build environment.")
    else:
        step("Creating build environment...")
        if VENV_DIR.exists():
            _rmtree(VENV_DIR)
        _run([uv, "venv", str(VENV_DIR)])

    python = _venv_python()

    # Synced even when the venv is being reused. `uv sync` costs a couple of
    # seconds when nothing has changed, and skipping it means a venv populated
    # from an older lockfile is what gets compiled -- so a dependency since
    # removed from the lock is still sitting there to be found and bundled. It is
    # declarative, so this also *un*installs anything the lock no longer names.
    install_args = [uv, "sync", "--locked", "--extra", device, "--extra", "bin-build"]
    print(f"  $ {' '.join(install_args)}", flush=True)
    _flush_stdio()
    result = subprocess.run(install_args, env=OS_ENV)
    if result.returncode != 0:
        # By far the most likely cause, and the message uv gives for it is easy to
        # skim past: `bin-build` installs nuitka now, and a uv.lock written when it
        # installed pyinstaller does not satisfy `--locked`.
        raise BuildError(
            f"uv sync failed (exit {result.returncode}).\n"
            "  If it reported that the lockfile is out of date, refresh it once:\n"
            "      uv lock\n"
            "  pyproject's `bin-build` extra installs nuitka rather than pyinstaller,\n"
            "  so a lockfile predating that change will not satisfy --locked."
        )

    verify_opencv(python, repair_cmd=[*install_args, "--reinstall-package", "opencv-contrib-python"])
    print("      Build environment ready.\n")
    return python


def compile_launcher(python: Path) -> tuple[Path, str]:
    """Build the C launcher. Returns the binary and the compiler's kind.

    Deliberately before Nuitka: a machine with no C compiler should find that out
    in two seconds, not after a twenty-minute compile of everything else. The kind
    is handed on so Nuitka can be pointed at the same toolchain rather than going
    looking for a second one.

    ``python`` is the build venv, which is where Nuitka lives -- so if this machine
    has no compiler at all, the toolchain can ask Nuitka to download the MinGW64 it
    would have downloaded for itself a step later anyway.
    """
    step("Compiling launcher...")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out = WORK_DIR / LAUNCHER_NAME
    try:
        compiler = find_compiler(python=python)
        built = build_launcher(out, compiler=compiler)
    except ToolchainError as exc:
        raise BuildError(str(exc)) from exc
    print()
    return built, compiler.kind


def embed_frontend(python: Path) -> Path:
    """Generate the asset module next to the entry point.

    Run with the build venv's interpreter rather than this one: the generator
    needs brotli, which is a dependency of the project and not of whatever
    Python happens to be running this script.
    """
    step("Embedding frontend and fallback fonts...")
    _fresh(ENTRY_DIR)

    module = ENTRY_DIR / "fox_reader_assets.py"
    _run([
        str(python), str(EMBED_SCRIPT),
        "--root", str(PROJECT_ROOT),
        "--out", str(module),
    ])

    if not module.is_file():
        raise BuildError("the asset generator reported success but wrote no module")

    entry = ENTRY_DIR / ENTRY_SOURCE.name
    shutil.copy2(ENTRY_SOURCE, entry)
    print(f"      {module.name}  ({module.stat().st_size:,} bytes of source)\n")
    return entry


def _effective_openblas_mode(args: argparse.Namespace) -> str:
    """The OpenBLAS mode the build runs with. `--no-blas` wins over `--openblas`.

    A helper rather than inline logic so it stays testable: both spellings mean
    "no BLAS", and everything else passes through untouched.
    """
    if getattr(args, "no_blas", False):
        return "off"
    return getattr(args, "openblas", "auto")


def install_llama_cpp(python: Path, args: argparse.Namespace, compiler_kind: str) -> llamacpp.Build | None:
    """Compile llama-cpp-python into the build venv, for this build's device.

    Deliberately not `uv sync --extra gguf`. llama-cpp-python publishes no wheels
    -- every install of it compiles llama.cpp from source -- so what the binary can
    actually do is decided entirely by the `CMAKE_ARGS` it was compiled with, and a
    sync has nowhere to put them. It gets worse than "no GPU support": uv keys its
    built-wheel cache on the sdist, not on `CMAKE_ARGS`, so a sync hands a CUDA
    build the CPU wheel from yesterday's CPU build, or the reverse, and a wheel
    whose ggml does not match the libraries beside it segfaults on the first model
    load however small the model is.

    So it is installed here, after the sync -- which is `--locked` and would
    uninstall it -- with the arguments packaging/llamacpp.py generates for the
    device, at the version uv.lock pins, and with `--no-cache-dir` so that uv has
    no cached wheel to reach for in the first place.

    Returns the `Build` describing what was installed, for `bundle_llama_cpp`, or
    None when there is no GGUF backend in this build.
    """
    step("Building the GGUF backend (llama-cpp-python)...")

    if args.no_gguf:
        print("      Skipped: --no-gguf.")
        print("      Everything else is unaffected; GGUF translation models will not\n"
              "      load in the distribution this produces.\n")
        return None

    device = args.gguf_device or args.device
    if device != args.device:
        print(f"      Device: {device} (overriding --device {args.device})")

    try:
        target = llamacpp.parse_device(device)

        # nvcc on Windows drives the host compiler itself and only supports MSVC.
        # A warning rather than an error: the launcher's toolchain preferring MinGW
        # does not prove MSVC is absent, and the compile below says so plainly
        # enough if it is.
        if (target.kind == "cuda" and platform.system() == "Windows"
                and compiler_kind == "gcc"):
            print("      warning: this machine's C compiler is MinGW gcc, and nvcc on Windows\n"
                  "               only drives MSVC. If the compile below fails on the host\n"
                  "               compiler, install the Visual Studio C++ workload, or build\n"
                  "               the CPU variant with --gguf-device cpu.")

        build = llamacpp.prepare(
            target,
            openblas_mode=_effective_openblas_mode(args),
            cpu_baseline=args.cpu_baseline,
            cuda_arch=args.cuda_arch,
            hip_arch=args.hip_arch,
        )
        build = llamacpp.install(python, build, env=_build_env())
        llamacpp.verify(python, build, env=_build_env())
    except llamacpp.GgufError as exc:
        raise BuildError(
            f"{exc}\n"
            "  --no-gguf builds everything else and leaves the GGUF backend out."
        ) from exc

    for note in build.notes:
        print(f"      note: {note}")
    print()
    return build


def _bytecode_flags(nuitka: Nuitka, python: Path, site_paths: list[str]) -> list[str]:
    """The default scope: third-party packages included, but not compiled."""
    if not nuitka.has("--noinclude-custom-mode"):
        nuitka.skipped.append(
            "--noinclude-custom-mode  (dependencies will be compiled -- expect a long build)"
        )
        return []

    names = _third_party_top_levels(python, site_paths)
    if not names:
        print("      note: found no third-party packages to keep as bytecode")
        return []

    # Both the package and everything under it. `torch` alone would leave the
    # ~3,000 modules of `torch.*` to be compiled, which is the entire cost.
    flags = [f"--noinclude-custom-mode={name}:bytecode" for name in names]
    flags += [f"--noinclude-custom-mode={name}.*:bytecode" for name in names]

    size = sum(len(flag) + 1 for flag in flags)
    if size > COMMAND_LIMIT:
        # Only reachable with an implausibly large venv, but a silently truncated
        # flag list would show up as a build that takes six hours for no visible
        # reason, so it says what it dropped.
        print(f"      note: {size:,} characters of flags is too many; dropping submodule patterns")
        flags = [f"--noinclude-custom-mode={name}:bytecode" for name in names]
        while flags and sum(len(flag) + 1 for flag in flags) > COMMAND_LIMIT:
            dropped = flags.pop()
            print(f"      note: dropped {dropped} -- it will be compiled")

    print(f"      {len(names)} third-party packages will be included as bytecode")
    return flags


#: What a guarded plugin-trigger read looks like in Nuitka's own source. The
#: crash this works around (see `_trigger_cache_workaround`) is an *unguarded*
#: ``trigger_module.getCompilationMode()`` / ``fake_module.getCompilationMode()``
#: -- `UncompiledPythonModule` has no such method, only `CompiledPythonModule`
#: does -- so any of these nearby means the installed Nuitka handles the
#: bytecode-cached case and needs no workaround.
_TRIGGER_GUARD_TOKENS = (
    "isUncompiledPythonModule",
    "isCompiledPythonModule",
    "hasattr",
    "getattr(",
)

#: The two call sites that must both be guarded. Line 977 (trigger modules) is
#: the one observed in the wild (`anyio.to_process-postLoad`, built by the
#: multiprocessing plugin for a submodule our `--noinclude-custom-mode=anyio.*`
#: keeps as bytecode); line ~1136 (fake modules) is the same shape one branch
#: over and goes down with the same cache hit.
_TRIGGER_RISK_SITES = (
    "trigger_module.getCompilationMode()",
    "fake_module.getCompilationMode()",
)


def _classify_trigger_handling(source: str) -> str:
    """How the installed Nuitka reads compilation modes off generated modules.

    Returns ``"guarded"`` only when every ``trigger_module``/``fake_module``
    ``getCompilationMode()`` read in ``nuitka/plugins/Plugins.py`` has a guard
    token within a few lines of it; ``"unguarded"`` when such a read is present
    without one; ``"unknown"`` when the source shows neither (a Nuitka that
    renamed everything, or source that could not be read -- the caller passes
    ``""`` then). ``"unknown"`` is deliberately *not* ``"guarded"``: failing
    safe means one slower build, failing open means the crash below.
    """
    lines = (source or "").splitlines()
    if not lines:
        return "unknown"
    hits = [
        index
        for index, line in enumerate(lines)
        if any(site in line for site in _TRIGGER_RISK_SITES)
    ]
    if not hits:
        return "unknown"
    for index in hits:
        window = "\n".join(lines[max(0, index - 8):index + 9])
        if not any(token in window for token in _TRIGGER_GUARD_TOKENS):
            return "unguarded"
    return "guarded"


def _installed_nuitka_plugins_source(python: Path) -> str:
    """The text of the installed Nuitka's ``plugins/Plugins.py``, or ``""``.

    Read from disk rather than imported: this module is parsed, never executed,
    and a build script that only wants one predicate out of Nuitka should not
    care what importing all of Nuitka does on this machine.
    """
    code, out = _capture(
        [str(python), "-c",
         "import nuitka.plugins.Plugins as _P; print(_P.__file__)"],
        timeout=300,
    )
    if code != 0:
        return ""
    path = Path(out.strip().splitlines()[-1]) if out.strip() else None
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _trigger_cache_workaround(nuitka: Nuitka, python: Path, extra: list[str]) -> list[str]:
    """Flags keeping plugin-generated modules compilable, or nothing.

    Nuitka 4.2.1 crashes mid-optimisation on a warm bytecode cache::

        File ".../nuitka/plugins/Plugins.py", line 977, in _createTriggerLoadedModule
            if trigger_module.getCompilationMode() == "bytecode":
        AttributeError: 'UncompiledPythonModule' object has no attribute 'getCompilationMode'

    The chain needs three links, all present here: a plugin that injects load
    code for a dotted submodule (the multiprocessing plugin does for
    ``anyio.to_process``, a uvicorn dependency), a ``pkg.*:bytecode`` flag
    covering that dotted trigger name (``_bytecode_flags`` emits one per
    third-party package), and a cache hit for the trigger from a previous
    build. The first build populates the cache and passes; every rebuild after
    it dies at ``354/557 modules - anyio``. It is not caused by anything the
    trigger's parent package changed -- any rebuild with a warm cache trips it.

    ``--disable-cache=bytecode`` breaks only the third link: triggers are then
    always built as compiled modules (which do have ``getCompilationMode``),
    at the cost of re-analysing bytecode modules on every build. Applied
    unless the installed Nuitka provably guards the read -- a future release
    with the guard keeps its cache -- and unless the user already passed their
    own ``--disable-cache``.
    """
    if any(arg == "--disable-cache" or arg.startswith("--disable-cache=") for arg in extra):
        print("      note: leaving --disable-cache to the user's own flag")
        return []

    handling = _classify_trigger_handling(_installed_nuitka_plugins_source(python))
    if handling == "guarded":
        return []
    if handling == "unknown":
        print("      note: could not inspect the installed Nuitka for the "
              "trigger-module fix; disabling its bytecode cache to be safe")
    else:
        print(f"      note: Nuitka {nuitka.version} reads plugin trigger modules "
              f"without an uncompiled guard; disabling its bytecode cache")
    cmd: list[str] = []
    if nuitka.add(cmd, "--disable-cache", "bytecode",
                  note="Nuitka reads bytecode-cached triggers as compiled and crashes"):
        print("      bytecode cache disabled (plugin triggers always compile; "
              "rebuilds are a little slower instead of crashing)")
    else:
        print("      warning: this Nuitka has the trigger bug and does not accept "
              "--disable-cache; the build may fail in _createTriggerLoadedModule. "
              "Upgrade Nuitka past the fix, or clear its module cache and build once.")
    return cmd


def _version_info(nuitka: Nuitka, cmd: list[str], version: str) -> None:
    """Windows resource fields, so the binary is not anonymous in Task Manager."""
    if platform.system() != "Windows":
        return
    nuitka.add(cmd, "--company-name", "Fox Reader", note="version info")
    nuitka.add(cmd, "--product-name", "Fox Reader")
    nuitka.add(cmd, "--file-description", "Fox Reader backend")
    nuitka.add(cmd, "--file-version", _file_version(version))
    nuitka.add(cmd, "--product-version", _file_version(version))

    for candidate in (
        BUILD_DIR / "launcher" / "fox.ico",
        FRONTEND_DIR / "static" / "favicon.ico",
        FRONTEND_DIR / "static" / "icon.ico",
    ):
        if candidate.is_file():
            nuitka.add(cmd, "--windows-icon-from-ico", str(candidate))
            break


def _warn_if_disk_tight(device: str) -> None:
    """Say so before the compile if the disk looks too full for what follows.

    A CUDA torch plus Nuitka's output, the staged copy and the archive need room
    several times over: the same gigabytes are written three times before they
    are archived. Running out mid-build surfaces as a bare OSError from deep
    inside a copy, so a warning up front -- never a refusal, the measurement is
    rough -- is worth the two lines.
    """
    try:
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
    except OSError:
        return
    need_gb = 12 if device != "cpu" else 6
    if free_gb < need_gb:
        print(f"      warning: {free_gb:.1f} GB free beside the checkout; a {device} build"
              f" wants roughly {need_gb} GB before it starts.")
        print("        Free some space or the build may die halfway with a bare write error.")


#: Shared-library suffixes that must never travel as `--include-package-data`.
#:
#: Nuitka documents `--include-package-data` as "non-DLL, non-extension modules"
#: only, and its filter ignores `*.so`, `*.dll`, `*.dylib` and `*.pyd`. What it
#: does not ignore is a *versioned* Linux shared library: `torch/lib/libgomp.so.1`
#: (shipped by the torch 2.13 CPU wheel, and the reason Linux builds die while
#: Windows ones pass -- a `.dll` has no version suffix to slip through) ends with
#: `.1`, not `.so`, so it is collected as a data file. The dll-files plugin finds
#: the same file through dependency scanning and wants it as a DLL, and the build
#: stops with::
#:
#:     FATAL: Error, data file to be placed in distribution as
#:     'torch/lib/libgomp.so.1' conflicts with dll 'torch/lib/libgomp.so.1'.
#:
#: The fix is to keep shared libraries on the DLL side, which does dependency and
#: rpath handling a plain data copy does not, and exclude them from the data side.
#: The patterns below are matched against the *target* filename (e.g.
#: `torch/lib/libgomp.so.1`), where fnmatch `*` also crosses `/`, so a bare
#: `*.so.*` already covers any nesting. Per-package spellings (`torch/*.so.*`)
#: are emitted as well: they cost nothing, and they keep working if a future
#: Nuitka ever requires the package prefix its own documentation shows.
_DATA_FILE_SHARED_LIB_SUFFIXES = ("*.so", "*.so.*", "*.so*", "*.dll", "*.dylib", "*.pyd")


def _shared_lib_data_exclusions() -> list[str]:
    """`--noinclude-data-files` patterns keeping shared libs off the data side.

    Global spellings plus one set per package that can carry binaries
    (`PACKAGE_DATA` and the lazily-loaded `OPTIONAL_PACKAGES`). Patterns for a
    package that is not installed simply never match, so the list is stable from
    build to build and does not depend on the environment probe. Redundancy is
    deliberate: `*.so*` already covers `*.so` and `*.so.*`, but one pattern
    failing to match on some Nuitka version must not reintroduce the FATAL.
    """
    patterns = list(_DATA_FILE_SHARED_LIB_SUFFIXES)
    for package in (*PACKAGE_DATA, *OPTIONAL_PACKAGES):
        for suffix in _DATA_FILE_SHARED_LIB_SUFFIXES:
            patterns.append(f"{package}/{suffix}")
    # Stable order for a reproducible command line and readable logs.
    return sorted(set(patterns))


def _exclude_shared_libs_from_data(nuitka: Nuitka, cmd: list[str]) -> None:
    """Exclude shared libraries from `--include-package-data`, keeping the DLL copy.

    Must run after the `--include-package-data` flags are added and before Nuitka
    runs: order on the command line does not matter, only that both are present.
    A Nuitka without `--noinclude-data-files` is left to fail with Nuitka's own
    message rather than a second guess -- every Nuitka this build supports has the
    option, so "skipped" here means a genuinely old one.
    """
    patterns = _shared_lib_data_exclusions()
    before = len(cmd)
    if nuitka.add_each(cmd, "--noinclude-data-files", patterns,
                       note="shared libraries travel as DLLs, not data files"):
        added = len(cmd) - before
        print(f"      excluding {added} shared-library patterns from package data")
    else:
        print("      warning: this Nuitka does not accept --noinclude-data-files; "
              "a versioned .so (e.g. torch/lib/libgomp.so.1) may fail the build "
              "as data-file-vs-dll conflict")


def run_nuitka(python: Path, entry: Path, deep: bool, extra: list[str], compiler_kind: str) -> Path:
    """Compile the backend. Returns the `.dist` directory."""
    step("Compiling backend with Nuitka...")

    nuitka = Nuitka(python)
    print(f"      Nuitka {nuitka.version}, {len(nuitka.options)} options advertised")

    for required in ("--output-dir", "--include-package", "--include-module"):
        nuitka.require(required)

    probe = _probe(
        python,
        modules=HIDDEN_MODULES + OCR_PACKAGES + PACKAGE_DATA + OPTIONAL_PACKAGES,
        dists=DISTRIBUTION_METADATA + OPTIONAL_DISTRIBUTION_METADATA,
        sites=True,
    )
    site_paths: list[str] = probe["sites"]
    present_modules: list[str] = probe["modules"]
    present_dists: list[str] = probe["dists"]
    # The same distributions under the names their own metadata gives them, which
    # is what Nuitka matches on. `pyyaml` is installed and resolvable and still
    # not what PyYAML calls itself.
    present_dist_names: list[str] = probe.get("dist_names") or present_dists
    #: The packages those distributions provide. Nuitka refuses to bundle metadata
    #: for a distribution whose own top-level package is not in the module graph,
    #: and being a dependency is not enough to put it there: PyYAML's first
    #: top-level name is the `_yaml` compatibility shim and torch's is `functorch`,
    #: neither of which anything imports.
    present_dist_modules: list[str] = probe.get("dist_modules") or []

    _report_missing("modules", HIDDEN_MODULES + OCR_PACKAGES + PACKAGE_DATA, present_modules)
    _report_missing("distributions", DISTRIBUTION_METADATA, present_dists)

    # The optional ones get a line either way rather than a "not installed" note:
    # absent is a normal outcome here, and which backends a build ended up with is
    # worth being able to read off the log afterwards.
    optional = [name for name in OPTIONAL_PACKAGES if name in present_modules]
    absent = [name for name in OPTIONAL_PACKAGES if name not in present_modules]
    if optional:
        print(f"      optional packages: {', '.join(optional)}")
    if absent:
        print(f"      optional packages not installed: {', '.join(absent)}")

    if not site_paths:
        raise BuildError("the build environment reported no site-packages directory")

    _fresh(NUITKA_OUT)

    cmd = [str(python), "-m", "nuitka"]
    nuitka.mode_standalone(cmd)
    if nuitka.has("--assume-yes-for-downloads"):
        cmd.append("--assume-yes-for-downloads")
    else:
        nuitka.skipped.append("--assume-yes-for-downloads  (a download may prompt)")

    cmd += [f"--output-dir={NUITKA_OUT}"]
    nuitka.add(cmd, "--output-filename", BACKEND_NAME, note="the binary keeps Nuitka's default name")
    nuitka.add(cmd, "--jobs", str(max(1, os.cpu_count() or 1)), note="C compilation will be serial")
    nuitka.add(cmd, "--report", str(WORK_DIR / "nuitka-report.xml"), note="no build report")

    # Reuse the compiler this script already found, so Nuitka does not go looking
    # for a second one -- or download one it does not need.
    if platform.system() == "Windows":
        if compiler_kind == "msvc":
            nuitka.add(cmd, "--msvc", "latest")
        elif compiler_kind == "gcc":
            nuitka.add(cmd, "--mingw64")

    for name in WANTED_PLUGINS:
        if nuitka.plugin(cmd, name):
            print(f"      plugin: {name}")

    # torch.jit is never used -- nothing in fox_reader scripts or traces a model.
    # Left unanswered this is only a warning, but it is one Nuitka repeats on
    # every build, and answering it is also what keeps torch's JIT machinery out
    # of the bundle.
    if nuitka.module_parameter(cmd, "torch-disable-jit", "yes"):
        print("      torch: JIT disabled")

    # Ours, and the only thing here that is compiled in either scope.
    cmd.append("--include-package=fox_reader")
    # The generated asset module: reached by name from `fox_reader.assets`, so
    # nothing static can see the import.
    cmd.append("--include-module=fox_reader_assets")

    nuitka.add_each(cmd, "--include-module", [m for m in HIDDEN_MODULES if m in present_modules],
                    note="hidden imports")
    nuitka.add_each(cmd, "--include-package", [m for m in OCR_PACKAGES if m in present_modules],
                    note="OCR model packages")
    nuitka.add_each(cmd, "--include-package-data", [m for m in PACKAGE_DATA if m in present_modules],
                    note="package data files")
    _exclude_shared_libs_from_data(nuitka, cmd)
    nuitka.add_each(cmd, "--include-package", optional, note="optional packages")
    nuitka.add_each(cmd, "--include-distribution-metadata", present_dist_names,
                    note="distribution metadata")
    nuitka.add_each(cmd, "--include-module", present_dist_modules,
                    note="packages the bundled metadata describes")

    if deep:
        print("      --deep-compile: dependencies will be compiled too. This takes hours.")
    else:
        cmd += _bytecode_flags(nuitka, python, site_paths)

    # Unconditional (both scopes): stdlib stays bytecode even under
    # --deep-compile, so a stdlib trigger hits the same bug there.
    cmd += _trigger_cache_workaround(nuitka, python, extra)

    _version_info(nuitka, cmd, _read_version())

    cmd += extra
    cmd.append(str(entry))

    if nuitka.skipped:
        print("      Nuitka does not accept, so not used:")
        for item in nuitka.skipped:
            print(f"        {item}")

    # The full command is ~16,000 characters, nearly all of it bytecode flags.
    # It goes to a file so a failed build can be reproduced by hand.
    record = WORK_DIR / "nuitka-command.txt"
    record.write_text("\n".join(cmd), encoding="utf-8")
    _print_command(cmd)
    print(f"      (full command: {record})", flush=True)

    started = time.monotonic()
    _flush_stdio()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_build_env())
    if result.returncode != 0:
        raise BuildError(
            f"Nuitka failed (exit {result.returncode}). "
            f"The report, if one was written, is at {WORK_DIR / 'nuitka-report.xml'}."
        )

    elapsed = time.monotonic() - started
    dist = _find_dist()
    print(f"      Compiled in {elapsed / 60:.1f} min -> {dist.name}\n")
    return dist


def _print_command(cmd: list[str]) -> None:
    """The command, with the bytecode flags collapsed to a count."""
    shown, collapsed = [], 0
    for arg in cmd:
        if arg.startswith("--noinclude-custom-mode="):
            collapsed += 1
            continue
        shown.append(arg)
    print(f"  $ {' '.join(shown)}", flush=True)
    if collapsed:
        print(f"    + {collapsed} x --noinclude-custom-mode=...:bytecode", flush=True)


def _find_dist() -> Path:
    """Nuitka's standalone output directory.

    Globbed rather than assumed: the directory is named after the entry file, and
    `--output-filename` renames the binary inside it but not the directory, so
    the two names differ and only one of them is predictable.
    """
    candidates = sorted(NUITKA_OUT.glob("*.dist"))
    if not candidates:
        raise BuildError(f"Nuitka wrote no .dist directory under {NUITKA_OUT}")
    if len(candidates) > 1:
        raise BuildError(f"expected one .dist directory, found {len(candidates)}: {NUITKA_OUT}")
    return candidates[0]


def purge_cuda_libs(dist: Path) -> None:
    """Drop the CUDA runtime from a CPU build."""
    step("Purging CUDA libraries...")
    before = _tree_size(dist)

    removed = 0
    for pattern in CUDA_PATTERNS:
        for found in dist.rglob(pattern):
            if found.is_file():
                found.unlink()
                removed += 1

    saved = (before - _tree_size(dist)) / (1024 * 1024)
    print(f"      Removed {removed} files ({saved:.0f} MB).\n")


#: Set to any value to skip `prune_duplicated_libs` below. The tree then ships
#: as Nuitka produced it -- twice the size on Linux, but byte-identical to a
#: build from before the pruning existed. The escape hatch if a future
#: dependency ever loads one of the pruned paths in a way the checks here do
#: not cover.
_PRUNE_SKIP_ENV = "FOX_BUILD_KEEP_DUP_LIBS"

#: torch subtrees that are never read at runtime. `test/` is the C++ and Python
#: test suite (dozens of executables, each dynamically linked against
#: libtorch_cpu -- which is also what drags duplicate top-level copies into the
#: tree), `include/` is the C++ headers, needed only to *compile* extensions.
#: Nothing in `src/` JIT-compiles (`cpp_extension`, `torch.jit` and
#: `torch.compile` are all absent, and the build passes
#: `--module-parameter=torch-disable-jit=yes`), so both go.
_TORCH_JUNK_DIRS = ("torch/test", "torch/include")

#: What `torch/bin/` keeps. Everything else in there is a test executable
#: (`test_*`, `*Test`), the protobuf compiler (`protoc*`, build-time only) or
#: JIT upgrader data (`*.ptl*`, `upgrader_models/`). `torch_shm_manager` stays:
#: it is the one helper torch itself may spawn for shared memory.
_TORCH_BIN_KEEP = frozenset({"torch_shm_manager", "torch_shm_manager.exe"})

#: A shared-library filename, split into its versioned base and the rest:
#: `libllama.so.0.1.0` -> (`libllama.so`, `.0.1.0`), `libmtmd.so.SOVERSION` ->
#: (`libmtmd.so`, `.SOVERSION`), `libtorch_cpu.so` -> (`libtorch_cpu.so`, None).
#: Names that are not versioned-library-shaped give None and never group.
_LIB_VERSIONED = re.compile(r"^(?P<base>.+\.so)(?P<ver>\..+)?$")


def _lib_version_base(name: str) -> str | None:
    """The grouping key for version-alias copies of one library, or None."""
    match = _LIB_VERSIONED.match(name)
    if match is None:
        return None
    return match.group("base")


def _torch_bin_prunable(names: list[str]) -> list[str]:
    """Filenames in `torch/bin/` that are safe to delete. Pure, for tests."""
    return sorted(name for name in names if name not in _TORCH_BIN_KEEP)


def _torch_lib_prunable(name: str) -> bool:
    """Whether a file directly in `torch/lib/` is test-only scaffolding.

    Matches `libtorchbind_test.so` / `libjitbackend_test.so`: linked only by
    the `torch/bin/test_*` executables, which are pruned above. The underscore
    keeps `latest`-shaped names safe -- only a `test` prefix or a `_test`
    infix counts.
    """
    low = name.lower()
    return low.startswith("test") or "_test" in low


def _pick_shadow_twin(top_name: str, twins: list[Path]) -> Path | None:
    """The nested copy a top-level shadow should link to, or None to keep.

    Exactly one twin means unambiguous: every loader path that resolved to the
    top-level file still resolves (through the symlink) to package-structured
    bytes whose own RUNPATH already covers their dependencies -- the same bytes
    the package's own loaders use today. Zero twins means the file is unique
    (a transitive dependency like the hashed `libopenblas-*.so`); several means
    ambiguous. Both keep the file.
    """
    if len(twins) != 1:
        return None
    return twins[0]


def _hash_file(path: Path) -> str:
    """SHA-256 of a file, streamed so a 400 MB `.so` is not one read."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_instead(link: Path, target: Path) -> bool:
    """Replace `link` with a relative symlink to `target`. Keeps `link` on failure.

    `os.symlink` on Windows needs a privilege a normal user has not got, and any
    filesystem can refuse a link for its own reasons, so a failure is a kept
    file and a warning, never a half-deleted tree: the link is created under a
    temporary name and moved into place, so `link` is untouched until the move.
    """
    try:
        rel = os.path.relpath(target, link.parent)
    except (OSError, ValueError):
        return False
    tmp = link.parent / (link.name + ".tmp-link")
    try:
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(rel, tmp)
        os.replace(tmp, link)
    except OSError:
        try:
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False
    return True


def prune_duplicated_libs(dist: Path, *, allow_links: bool = os.name != "nt") -> None:
    """Delete what never runs and link what Nuitka copied twice.

    On Linux the tree ships every shared library twice: once under its package
    (`torch/lib/libtorch_cpu.so`, where the package's own RUNPATH finds it --
    the layout Windows ships *only*, and works) and once flat at the top level
    (where Nuitka's dependency scan puts it for referrers like torchvision).
    Both copies are live, so deleting either outright breaks one world; a
    relative symlink from the flat copy to the packaged one keeps every
    existing lookup path resolving while the bytes ship once. Version triplets
    (`libllama.so`, `.so.0`, `.so.0.20.0` -- symlinks in the wheel that arrive
    as three full copies) collapse the same way after a hash check.

    Deletions are the conservative half: `torch/test`, `torch/include`, the
    test executables and protobuf compiler in `torch/bin`, and `*test*` stubs
    in `torch/lib` -- none imported, spawned or dlopened by anything the app
    runs (verified against `src/`; `torch_shm_manager` is deliberately kept).
    """
    step("Pruning duplicated libraries...")
    if os.environ.get(_PRUNE_SKIP_ENV):
        print(f"      Skipped: {_PRUNE_SKIP_ENV} is set.\n")
        return

    removed = 0
    removed_bytes = 0
    linked = 0
    linked_bytes = 0
    warnings: list[str] = []

    def _delete(path: Path) -> None:
        nonlocal removed, removed_bytes
        try:
            removed_bytes += path.stat().st_size
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        except OSError as exc:
            warnings.append(f"{path.name}: could not remove ({exc.strerror or exc})")

    # ── 1. torch scaffolding that never runs ──────────────────────────────
    for rel in _TORCH_JUNK_DIRS:
        target = dist / rel
        if target.is_dir():
            _delete(target)
    torch_bin = dist / "torch" / "bin"
    if torch_bin.is_dir():
        try:
            entries = [p.name for p in torch_bin.iterdir()]
        except OSError:
            entries = []
        for name in _torch_bin_prunable(entries):
            _delete(torch_bin / name)
    torch_lib = dist / "torch" / "lib"
    if torch_lib.is_dir():
        try:
            lib_entries = list(torch_lib.iterdir())
        except OSError:
            lib_entries = []
        for entry in lib_entries:
            if entry.is_file() and not entry.is_symlink() and _torch_lib_prunable(entry.name):
                _delete(entry)

    # ── 2. nested version triplets: one real file, the rest links ─────────
    # `allow_links` is a parameter rather than a second `os.name` read so tests
    # can exercise the linking paths on Windows, where creating a symlink
    # needs a privilege the test runner has not got.
    if allow_links:
        groups: dict[tuple[str, str], list[Path]] = {}
        try:
            nested_libs = [p for p in dist.rglob("*.so*")
                           if p.is_file() and not p.is_symlink() and _lib_version_base(p.name)]
        except OSError:
            nested_libs = []
        for path in nested_libs:
            base = _lib_version_base(path.name)
            if base is not None:
                groups.setdefault((str(path.parent), base), []).append(path)
        for (_, _base), members in sorted(groups.items()):
            if len(members) < 2:
                continue
            if len({m.stat().st_size for m in members}) != 1:
                continue
            try:
                if len({_hash_file(m) for m in members}) != 1:
                    continue
            except OSError as exc:
                warnings.append(f"{members[0].name}: could not hash ({exc.strerror or exc})")
                continue
            # Any member works -- the bytes are identical -- so keep the
            # shortest name and link the longer version aliases to it.
            keep = sorted(members, key=lambda m: (len(m.name), m.name))[0]
            for member in members:
                if member == keep:
                    continue
                size = member.stat().st_size
                if _link_instead(member, keep):
                    linked += 1
                    linked_bytes += size
                else:
                    warnings.append(f"{member.name}: could not link, kept the copy")
    else:
        print("      version-alias linking skipped on Windows (needs symlinks).")

    # ── 3. flat top-level shadows of packaged libraries ───────────────────
    if allow_links:
        try:
            nested_by_name: dict[str, list[Path]] = {}
            for path in dist.rglob("*"):
                if path.parent == dist or not path.is_file() or path.is_symlink():
                    continue
                nested_by_name.setdefault(path.name, []).append(path)
            top_level = [p for p in dist.iterdir()
                         if p.is_file() and not p.is_symlink()
                         and re.search(r"\.so(\.|$)", p.name)]
        except OSError:
            nested_by_name = {}
            top_level = []
        for top in sorted(top_level):
            twin = _pick_shadow_twin(top.name, nested_by_name.get(top.name, []))
            if twin is None:
                if len(nested_by_name.get(top.name, [])) > 1:
                    warnings.append(f"{top.name}: {len(nested_by_name[top.name])} nested copies, kept")
                continue
            # Point at the real file rather than through an alias created in
            # step 2 above: one hop instead of two, and the link stays valid
            # even if the alias is ever removed.
            seen = {twin}
            while twin.is_symlink():
                try:
                    twin = twin.parent / os.readlink(twin)
                except OSError:
                    break
                if twin in seen:
                    break
                seen.add(twin)
            size = top.stat().st_size
            if _link_instead(top, twin):
                linked += 1
                linked_bytes += size
            else:
                warnings.append(f"{top.name}: could not link, kept the copy")
    else:
        print("      top-level shadow linking skipped on Windows (needs symlinks).")

    saved_mb = (removed_bytes + linked_bytes) / (1024 * 1024)
    print(f"      Removed {removed} files, linked {linked} duplicates ({saved_mb:.0f} MB).\n")
    for warning in warnings[:8]:
        print(f"      warning: {warning}")
    if len(warnings) > 8:
        print(f"      warning: ... and {len(warnings) - 8} more")


def bundle_llama_cpp(dist: Path, python: Path, build: llamacpp.Build | None) -> None:
    """Copy llama.cpp's shared libraries into the compiled tree.

    Nuitka does not do this. `--include-package-data` collects data files and
    skips binaries, and nothing follows a ctypes load, so a compiled tree gets
    `llama_cpp/` without the one thing it exists to load. The copy has to land in
    `llama_cpp/lib/`, which is where the package looks -- next to its own
    `__file__`.

    Before `bundle_runtime_dlls`, not after: these arrive with imports of their
    own (the MSVC runtime, and for a CUDA build the CUDA runtime), and that step
    is what reads the tree's imports and satisfies them.
    """
    step("Bundling the GGUF backend's libraries...")
    if build is None:
        print("      Skipped: no GGUF backend in this build.\n")
        return

    try:
        llamacpp.bundle(dist, python, build, env=_build_env())
    except llamacpp.GgufError as exc:
        raise BuildError(str(exc)) from exc
    print()


#: Import-name prefixes of the Microsoft C/C++ runtime. These are not part of
#: Windows -- they arrive with the "Visual C++ Redistributable", which a clean
#: machine has never had installed -- so a standalone build has to carry them.
#: Nuitka finds `vcruntime140` on its own and, at least for this dependency set,
#: misses `msvcp140` (sentencepiece, pyclipper, torchvision) and
#: `msvcp140_atomic_wait` (torch). Rather than hard-code that list and watch it
#: rot, the build reads the imports out of what it actually produced.
_CRT_PREFIXES = ("msvcp", "vcruntime", "concrt", "vcomp", "msvcr")

#: Ships with Windows itself. Copying it would be both pointless and wrong.
_CRT_SYSTEM = frozenset({"msvcrt.dll"})


#: The DLL names in a PE file's import directory. Lives in `llamacpp`, because
#: that module needs the same reader for its backend-linkage check -- one
#: implementation rather than two copies to keep in step. Empty for anything
#: that is not a PE file or cannot be walked: a missing import table is a
#: normal thing for a resource-only DLL, not an error worth stopping a build
#: over.
_pe_imports = llamacpp._pe_import_names


def _crt_search_path(dist: Path) -> list[Path]:
    """Where to look for the redistributable runtime, best source first.

    The Visual Studio redist folder comes first because those files are the ones
    Microsoft actually licenses for redistribution. System32 is the fallback
    every other packaging tool uses. The wheels come last and are the reason this
    works at all on a machine with no redistributable and no Visual Studio:
    delvewheel vendors a real `msvcp140.dll` into `numpy.libs` and friends under
    a hashed name, and those folders are already inside the tree being fixed.
    """
    arch = {"AMD64": "x64", "ARM64": "arm64", "x86": "x86"}.get(platform.machine(), "x64")
    found: list[Path] = []

    def add(path: Path) -> None:
        if path.is_dir() and path not in found:
            found.append(path)

    for env in ("ProgramFiles(x86)", "ProgramFiles", "ProgramW6432"):
        root = os.environ.get(env)
        if not root:
            continue
        for redist in sorted(Path(root).glob(f"Microsoft Visual Studio/*/*/VC/Redist/MSVC/*/{arch}/*.CRT"),
                             reverse=True):
            add(redist)

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    add(system_root / ("System32" if arch in ("x64", "arm64") else "SysWOW64"))

    for libs in sorted(dist.glob("*.libs")):
        add(libs)
    return found


#: What a Linux tree is allowed to expect from the machine it runs on: the C
#: library and the handful of things that come with it and with the graphics
#: stack. Copying any of these is actively wrong -- glibc's loader and its
#: `libc.so.6` are one unit and must match, and `libcuda.so.1` belongs to the
#: installed driver -- so they are named as system dependencies rather than
#: bundled.
_ELF_SYSTEM = ("libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so",
               "libutil.so", "libresolv.so", "ld-linux", "libgcc_s.so",
               "libstdc++.so", "libcuda.so", "libGL.so", "libGLX.so", "libEGL.so",
               "libX11", "libxcb", "libdrm", "libnvidia", "libgomp.so",
               "libcrypt.so", "libz.so")


def _report_elf_dependencies(dist: Path) -> None:
    """Say what a Linux build will need from the machine it is unpacked on.

    The Windows half of this copies, because the Visual C++ runtime is not part
    of Windows and a clean machine has never had it. Nothing equivalent is true
    here: every library below is either part of glibc -- which cannot be
    relocated, its loader and its `libc.so.6` being one unit -- or part of the
    installed driver. So this reports instead of copying.

    Worth the scan anyway, because the alternative is finding out from a user.
    An ELF records what it needs by SONAME with no path, and the loader answers
    from RUNPATH, `ld.so.cache` and the default directories in that order; a name
    that is neither in the tree nor on the list below is one the build picked up
    from a development machine, and it will be missing on a plain one. That
    failure arrives as `error while loading shared libraries` before any of our
    code runs, which is exactly as informative as it sounds.
    """
    have = {path.name for path in dist.rglob("*") if path.is_file()}
    wanted: dict[str, set[str]] = {}
    scanned = 0
    for path in dist.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if ".so" not in path.name and path.suffix not in ("", ".bin"):
            continue
        needed = llamacpp._elf_needed(path)
        if not needed:
            continue
        scanned += 1
        for name in needed:
            if name in have or name.startswith(_ELF_SYSTEM):
                continue
            wanted.setdefault(name, set()).add(path.name)

    print(f"      Scanned {scanned} ELF binaries; glibc and the driver come from the system.")
    for name in sorted(wanted):
        users = sorted(wanted[name])
        print(f"      ! {name} is neither in the tree nor a system library"
              f" -- needed by {', '.join(users[:4])}")
    if wanted:
        print("        The build will only run where those are already installed.")
    print()


def bundle_runtime_dlls(dist: Path) -> None:
    """Copy in any C++ runtime the compiled tree imports but does not carry.

    Without this the build runs on the machine that made it -- which has the
    redistributable, that is why it could compile -- and fails on a clean one
    with a dialog about a missing DLL before any of our code gets to run.
    """
    step("Bundling the C++ runtime...")
    if os.name != "nt":
        _report_elf_dependencies(dist)
        return

    #: Every filename anywhere in the tree. A wheel that vendors its own copy in
    #: a `.libs` folder has already solved this for itself, so a name that is
    #: present anywhere counts as satisfied rather than missing.
    have = {path.name.lower() for path in dist.rglob("*") if path.is_file()}

    wanted: dict[str, set[str]] = {}
    scanned = 0
    for path in dist.rglob("*"):
        if path.suffix.lower() not in (".exe", ".dll", ".pyd") or not path.is_file():
            continue
        scanned += 1
        for name in _pe_imports(path):
            low = name.lower()
            if not low.startswith(_CRT_PREFIXES) or low in _CRT_SYSTEM or low in have:
                continue
            wanted.setdefault(low, set()).add(path.name)

    if not wanted:
        print(f"      Scanned {scanned} binaries; the runtime is already complete.\n")
        return

    sources = _crt_search_path(dist)
    copied, missing = [], []
    for name in sorted(wanted):
        for folder in sources:
            candidate = folder / name
            if not candidate.is_file():
                # delvewheel renames its copies, so match on the stem instead.
                matches = sorted(folder.glob(f"{name[:-4]}-*.dll"))
                if not matches:
                    continue
                candidate = matches[0]
            shutil.copy2(candidate, dist / name)
            copied.append((name, len(wanted[name]), folder))
            break
        else:
            missing.append((name, sorted(wanted[name])))

    print(f"      Scanned {scanned} binaries.")
    for name, users, folder in copied:
        print(f"      + {name}  ({users} importer{'s' if users != 1 else ''}) from {folder}")
    for name, users in missing:
        print(f"      ! {name} not found -- needed by {', '.join(users[:4])}")
    if missing:
        print("        The build will only run where the Visual C++ Redistributable")
        print("        is installed. Install it on this machine and build again.")
    print()


#: Asks the build venv which distributions its packages will look up by name at
#: runtime, and under which spelling. Only transformers is asked, because it is
#: the only one whose lookups are fatal: they run at `import transformers`, and a
#: miss raises there rather than degrading into a wrong answer somewhere later.
#:
#: Everything is filtered to what the venv actually has. transformers checks its
#: optional dependencies only when they are importable -- `accelerate` is in its
#: list, is not installed here, and is skipped at runtime for exactly that reason
#: -- so demanding metadata for it would fail a build that is entirely correct.
#:
#: Tolerant on purpose. transformers' list is a private name inside somebody
#: else's package, so it is allowed to move; when it does, the build reports what
#: it could not read and falls back to METADATA_LOOKUPS rather than failing on the
#: introspection itself.
_LOOKUP_PROBE = r"""
import json, sys
import importlib.metadata as im

request = json.loads(sys.stdin.read())
names, notes = [], []


def add(name):
    if not name or name in names:
        return
    # Not installed means not shipped, and also means nothing will look it up:
    # every one of these checks is guarded by an availability test upstream.
    try:
        im.distribution(name)
    except Exception:
        return
    names.append(name)


for name in request["static"]:
    add(name)

try:
    from transformers.dependency_versions_check import pkgs_to_check_at_runtime
    from transformers.dependency_versions_table import deps
except Exception as exc:
    notes.append("could not read transformers' version checks (%s: %s)"
                 % (type(exc).__name__, exc))
else:
    for pkg in pkgs_to_check_at_runtime:
        # `python` is compared against sys.version_info, not looked up. Anything
        # absent from `deps` is not checked at all.
        if pkg == "python" or pkg not in deps:
            continue
        add(pkg)

json.dump({"names": names, "notes": notes}, sys.stdout)
"""

compile(_LOOKUP_PROBE, "<lookup-probe>", "exec")


def _normalize_dist(name: str) -> str:
    """A distribution name with case, dashes and underscores flattened (PEP 503)."""
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def _required_lookups(python: Path) -> list[str]:
    """Every spelling the compiled build has to be able to resolve."""
    fallback = list(METADATA_LOOKUPS)

    try:
        proc = subprocess.run([str(python), "-c", _LOOKUP_PROBE],
                              input=json.dumps({"static": fallback}).encode("utf-8"),
                              capture_output=True, env=_build_env(), timeout=900)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"      note: could not ask transformers what it checks ({exc})")
        return fallback

    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        print("      note: could not ask transformers what it checks "
              f"({detail[-1] if detail else 'no output'})")
        return fallback

    try:
        answer = json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        print("      note: the runtime-lookup probe returned nothing usable")
        return fallback

    for note in answer.get("notes", []):
        print(f"      note: {note}")
    return answer.get("names") or fallback


def bundle_metadata(dist: Path, python: Path) -> None:
    """Copy the real `.dist-info` directories into the compiled tree.

    `--include-distribution-metadata` already bundles this metadata, in the sense
    that it compiles it into the binary. That is not enough, and cannot be made
    enough: Nuitka keeps what it embedded in a dict keyed by the name each
    distribution gives *itself*, and reads it back with a plain dictionary get. No
    normalisation, on either side. So huggingface_hub's metadata goes in under
    exactly that spelling, and

        importlib.metadata.version("huggingface-hub")

    -- the call `import transformers` makes, with the dash, before anything else
    happens -- misses the dict, falls through to the real `importlib.metadata`,
    finds no `.dist-info` anywhere in the tree, and raises PackageNotFoundError.
    The traceback surfaces eleven frames deep in paddlex building a text detector,
    a long way from the flag that was supposed to prevent it. `pyyaml` (installed
    as `PyYAML`) and `hf_xet` (installed as `hf-xet`) are the same bug; the first
    is a second startup failure standing behind the first, the second is why model
    downloads announce that Xet is unavailable and fall back to HTTP.

    Copying the directories in fixes all of them at once, because the fallback is
    the real implementation and the real implementation normalises. It runs before
    staging, so `bin/` carries them.

    What ships is still the curated DISTRIBUTION_METADATA list, plus whichever of
    OPTIONAL_DISTRIBUTION_METADATA is installed, and nothing else.
    """
    step("Bundling distribution metadata...")

    probe = _probe(python, dists=DISTRIBUTION_METADATA + OPTIONAL_DISTRIBUTION_METADATA)
    found: list[str] = probe["dists"]
    names: list[str] = probe.get("dist_names") or found
    sources: list[str] = probe.get("dist_info") or []

    shipped: dict[str, str] = {}
    copied: list[str] = []
    without: list[str] = []

    for index, requested in enumerate(found):
        real = names[index] if index < len(names) else requested
        source = Path(sources[index]) if index < len(sources) and sources[index] else None
        if source is None or not source.is_dir():
            without.append(real)
            continue

        target = dist / source.name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
        shipped[_normalize_dist(real)] = source.name
        copied.append(real)

    print(f"      Copied {len(copied)} of {len(found)} .dist-info "
          f"director{'y' if len(copied) == 1 else 'ies'} into {dist.name}/")
    if without:
        # Not fatal on its own: the embedded copy still answers a lookup that uses
        # the distribution's own spelling. Whether anything that matters was lost
        # with it is what the check below decides.
        print(f"      note: no .dist-info found for: {', '.join(without)}")

    lookups = _required_lookups(python)
    missing = [name for name in lookups if _normalize_dist(name) not in shipped]
    if missing:
        raise BuildError(
            "the compiled build could not resolve this metadata:\n"
            + "".join(f"    importlib.metadata.version({name!r})\n" for name in missing)
            + "  Each name has to match a distribution in DISTRIBUTION_METADATA once\n"
            "  dashes, underscores and case are flattened. Add whichever distribution\n"
            "  provides it and build again -- reaching a user, this is a startup\n"
            "  failure, not a warning."
        )
    print(f"      {len(lookups)} runtime lookups resolve.\n")


def stage_layout(dist: Path, launcher: Path) -> Path:
    """Assemble the tree the user unzips."""
    step("Staging the distribution...")
    root = _fresh(STAGE_DIR) / STAGE_NAME
    root.mkdir(parents=True)

    # symlinks=True: `prune_duplicated_libs` replaces duplicate `.so` files with
    # relative symlinks, and the default would materialise them back into full
    # copies here. `make_archive` below stores symlinks as symlinks, so the
    # archive stays small and extraction restores the links.
    shutil.copytree(dist, root / "bin", symlinks=True)
    shutil.copy2(launcher, root / LAUNCHER_NAME)
    if os.name != "nt":
        (root / LAUNCHER_NAME).chmod(0o755)

    # Legal files at the root, next to the launcher users actually see:
    # LICENSE (AGPL §4/§6: every binary conveyance must include the license
    # text), NOTICE (Apache-2.0 §4d attributions plus the model/dataset credits
    # that must travel with distributions), TERMS.md (terms of use). A missing
    # file fails the build rather than shipping a non-compliant archive.
    for legal in ("LICENSE", "NOTICE", "TERMS.md"):
        source = PROJECT_ROOT / legal
        if not source.is_file():
            raise BuildError(f"{source} is missing — refusing to stage a build without it")
        shutil.copy2(source, root / legal)

    binary = root / "bin" / BACKEND_NAME
    if not binary.is_file():
        # Without --output-filename, Nuitka names the binary after the entry
        # file. The launcher looks for it by name, so it is renamed here rather
        # than teaching the launcher a fourth candidate to guess at.
        fallback = root / "bin" / (ENTRY_SOURCE.stem + (".exe" if os.name == "nt" else ""))
        if fallback.is_file():
            fallback.rename(binary)
            print(f"      renamed {fallback.name} -> {BACKEND_NAME}")
        else:
            found = sorted(
                p.name for p in (root / "bin").glob("*")
                if p.is_file() and p.suffix in ("", ".exe")
            )
            raise BuildError(
                f"{BACKEND_NAME} is not in {dist}, and neither is {fallback.name}.\n"
                f"  The launcher looks for it by name. Candidates: {', '.join(found[:12]) or 'none'}"
            )

    # config/: the one file the user is expected to edit, and the one the
    # launcher reads to find the host and port. settings.yaml and
    # user_endpoints.yaml are deliberately absent -- their managers write
    # defaults on first load, and shipping them would overwrite a user's
    # choices on every upgrade.
    config = root / "config"
    config.mkdir()
    source_config = PROJECT_ROOT / "config" / "fox_config.yaml"
    if not source_config.is_file():
        raise BuildError(f"{source_config} is missing")
    shutil.copy2(source_config, config / "fox_config.yaml")

    # fonts/: the user's, and it ships empty. ComicMono and the symbol fallback
    # are inside the binary as the typesetter's faces, so their licences have
    # to travel with the build even though the files they cover are not visible
    # here.
    fonts = root / "fonts"
    fonts.mkdir()
    licence = PROJECT_ROOT / "fonts" / "LICENSE-ComicMono"
    if licence.is_file():
        shutil.copy2(licence, fonts / licence.name)
    else:
        print(f"      warning: {licence} is missing, and the embedded font needs it")
    symbol_licence = PROJECT_ROOT / "fonts" / "OFL.txt"
    if symbol_licence.is_file():
        shutil.copy2(symbol_licence, fonts / symbol_licence.name)
    else:
        print(f"      warning: {symbol_licence} is missing, and the embedded symbol font needs it")

    # Both empty on purpose. cache/ is scratch, cleared at startup and shutdown;
    # models/ is filled by the first-run download, which is why the weights are
    # not in the binary.
    (root / "cache").mkdir()
    (root / "models").mkdir()

    total = _tree_size(root) / (1024 * 1024)
    print(f"      {root}  ({total:.0f} MB)")
    for entry in sorted(root.iterdir()):
        marker = "/" if entry.is_dir() else ""
        print(f"        {entry.name}{marker}")
    print()
    return root


def archive_stem(version: str, device: str, gguf_device: str | None, platform_tag: str) -> str:
    """The archive's name, without the extension.

    Every build of this project produces a tree with the same layout and a wildly
    different set of shared libraries in it, and the old name -- ``fox-reader-
    windows-x64`` -- said nothing about which. Two builds in the same ``dist/``
    overwrote each other, and a downloaded archive could not be told apart from
    any other. So the name carries what a person choosing between them needs: the
    version, the torch backend, and whether GGUF is in there at all.

    ``gguf_device`` is None when the backend was left out. When it matches
    ``device`` -- the usual case, since it defaults to it -- the word ``gguf`` is
    enough; naming the device twice reads as though they were different. It is
    spelled out only when it really is different, which is the build
    (``--device cpu --gguf-device cu130``) whose name would otherwise lie::

        fox-reader-v0.1.0-cpu-windows-x64.zip
        fox-reader-v0.1.0-cpu-gguf-windows-x64.zip
        fox-reader-v0.1.0-cu130-gguf-windows-x64.zip
        fox-reader-v0.1.0-cpu-gguf-cu130-windows-x64.zip
    """
    parts = ["fox-reader", f"v{version}", device]
    if gguf_device is not None:
        parts.append("gguf")
        if gguf_device != device:
            parts.append(gguf_device)
    parts.append(platform_tag)
    # A version can be `0.1.0rc1+local.1`, and `+` is not a filename anywhere
    # worth having. Everything outside the safe set becomes a dash.
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(parts))


def package_output(root: Path, version: str, device: str, gguf_device: str | None) -> Path:
    """Archive the staged tree."""
    platform_tag = _platform_tag()
    step(f"Packaging {platform_tag}...")
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    base = DIST_DIR / archive_stem(version, device, gguf_device, platform_tag)
    fmt = "zip" if platform_tag == "windows-x64" else "gztar"
    # `make_archive` appends to a zip that is already there, so a rebuild would
    # accumulate the union of every tree ever staged under this name.
    for stale in DIST_DIR.glob(f"{base.name}.*"):
        stale.unlink(missing_ok=True)
    made = shutil.make_archive(str(base), fmt, str(root.parent), root.name)
    archive = Path(made)

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"      Output: {archive} ({size_mb:.1f} MB)\n")
    return archive


def cleanup(keep_venv: bool) -> None:
    """Remove the build venv. Nuitka's caches stay, so a rebuild is cheaper."""
    step("Cleaning up...")
    if keep_venv:
        print("      Keeping the build environment.\n")
        return
    if VENV_DIR.exists():
        _rmtree(VENV_DIR)
    print("      Done.\n")


def clean_all() -> None:
    """Remove everything this script has ever written."""
    for directory in (WORK_DIR, VENV_DIR, BUILD_DIR / "dist", BUILD_DIR / "build_tmp"):
        if directory.is_dir():
            _rmtree(directory)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Fox Reader distribution")
    parser.add_argument("--clean", action="store_true", help="discard previous build artifacts first")
    parser.add_argument("--skip-frontend", action="store_true",
                        help="skip the pnpm build and embed the existing frontend/static")
    parser.add_argument("--keep-venv", action="store_true",
                        help="keep the build venv, so the next build skips dependency resolution")
    parser.add_argument(
        "--deep-compile",
        action="store_true",
        help="compile the dependencies as well, not just fox_reader. Hours, not minutes.",
    )
    parser.add_argument(
        "--nuitka-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="pass an extra argument straight to Nuitka (repeatable)",
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        choices=_device_choices(),
        default="cpu",
        help="torch backend to bundle (default: cpu)",
    )

    gguf = parser.add_argument_group(
        "GGUF backend (llama-cpp-python)",
        "Compiled from source against this machine's toolchain, once per build. "
        "The defaults produce a binary that runs on any x86-64 machine, not just "
        "this one.",
    )
    gguf.add_argument("--no-gguf", action="store_true",
                      help="leave llama-cpp-python out; GGUF models then will not load")
    gguf.add_argument("--gguf-device", type=str.lower, metavar="DEVICE",
                      help="build llama.cpp for a different device than --device (e.g. cpu)")
    gguf.add_argument("--openblas", type=str.lower, choices=["auto", "download", "off"],
                      default="auto",
                      help="BLAS for ggml's CPU backend: use what is installed, otherwise "
                           "download it into nc/ without asking (auto, the default), "
                           "download without asking, or build without it (off)")
    gguf.add_argument("--no-blas", action="store_true",
                      help="build without OpenBLAS (same as --openblas off)")
    gguf.add_argument("--cpu-baseline", type=str.lower, choices=["avx2", "avx", "sse42"],
                      default="avx2",
                      help="oldest x86-64 CPU the GGUF backend must run on (default: avx2, "
                           "Haswell and later)")
    gguf.add_argument("--cuda-arch", metavar="LIST",
                      help="override CMAKE_CUDA_ARCHITECTURES, e.g. '75-real;86-real'")
    gguf.add_argument("--hip-arch", metavar="LIST",
                      help="override the ROCm GPU targets, e.g. 'gfx1100;gfx1030'")

    args = parser.parse_args()

    # Checked here rather than at the step that uses it: `--gguf-device cu125`
    # should be rejected in the first second, not after the frontend build and a
    # dependency sync. `--device` cannot be misspelled -- argparse has the choices
    # -- but it can name a torch that does not exist for this platform.
    if args.gguf_device and not args.no_gguf:
        try:
            llamacpp.parse_device(args.gguf_device)
        except llamacpp.GgufError as exc:
            parser.error(str(exc))

    if args.device.startswith("rocm") and platform.system() != "Linux":
        # The extras resolve on every platform, because a lock has to; what they
        # resolve *to* off Linux is the ordinary PyPI torch, since AMD publishes
        # no ROCm wheels for Windows or macOS. Silently shipping a CPU torch under
        # a ROCm device name is worse than refusing.
        parser.error(
            f"--device {args.device} is Linux only: torch's ROCm wheels are built for "
            "linux_x86_64 and\n"
            "nothing else, so this would quietly bundle the CPU torch instead.\n"
            "  llama.cpp's HIP backend does build on Windows, so an AMD GPU can still be\n"
            f"  used for GGUF models: --device cpu --gguf-device {args.device}."
        )

    platform_tag = _platform_tag()
    version = _read_version()
    gguf_device = None if args.no_gguf else (args.gguf_device or args.device)
    blas_mode = "left out (--no-gguf)" if args.no_gguf else _effective_openblas_mode(args)
    print("=== Fox Reader Build ===")
    print(f"Platform:  {platform_tag}")
    print(f"Project:   {PROJECT_ROOT}")
    print(f"Version:   {version}")
    print(f"Device:    {args.device}")
    print(f"GGUF:      {'left out (--no-gguf)' if args.no_gguf else gguf_device}")
    print(f"BLAS:      {blas_mode}")
    print(f"Scope:     {'everything (--deep-compile)' if args.deep_compile else 'fox_reader + entry'}")
    print(f"Archive:   {archive_stem(version, args.device, gguf_device, platform_tag)}")
    print()

    if args.clean:
        clean_all()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_frontend:
        step("Skipping the frontend build.")
        print()
    else:
        build_frontend()

    python = create_build_venv(args.device)
    launcher, compiler_kind = compile_launcher(python)
    entry = embed_frontend(python)
    gguf_build = install_llama_cpp(python, args, compiler_kind)
    _warn_if_disk_tight(args.device)
    dist = run_nuitka(python, entry, args.deep_compile, list(args.nuitka_arg), compiler_kind)

    if args.device == "cpu":
        purge_cuda_libs(dist)
    else:
        step(f"Keeping CUDA libraries ({args.device}).")
        print()

    # After the purge, so a CPU distribution with a deliberately CUDA GGUF backend
    # (`--device cpu --gguf-device cu129`) keeps the libraries that backend needs.
    bundle_llama_cpp(dist, python, gguf_build)
    bundle_runtime_dlls(dist)
    # After every library is in place (Nuitka, the CUDA purge above, the GGUF
    # bundle): it only removes relocatable copies, never unique files.
    prune_duplicated_libs(dist)
    bundle_metadata(dist, python)

    root = stage_layout(dist, launcher)
    archive = package_output(root, version, args.device, gguf_device)
    cleanup(args.keep_venv)

    print("=== Build Complete ===")
    print(f"  {archive}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        print(f"\nerror: {' '.join(str(part) for part in exc.cmd)} failed (exit {exc.returncode})",
              file=sys.stderr)
        raise SystemExit(1) from exc
