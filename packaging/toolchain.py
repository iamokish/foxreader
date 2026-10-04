"""Finds a C compiler and builds the launcher.

The launcher is one C file with no dependencies beyond libc and the OS, so this
does not need a build system -- it needs a compiler and four flags. What it does
need is to find one on a machine nobody prepared, which on Windows means Visual
Studio's environment batch file, and to fall back to the MinGW that Nuitka
downloads for itself rather than telling the user to go and install something.

Search order, most to least preferred:

  Windows   an already-configured MSVC (a *64-bit* Developer Prompt) -> MSVC via
            vswhere -> clang -> gcc -> the MinGW under Nuitka's cache -> ask
            Nuitka to download that MinGW
  Linux     gcc -> clang -> cc

MSVC first on Windows because ``/MT`` links the C runtime statically, so the
launcher needs no redistributable installed to run -- which is the whole point of
a launcher. MinGW gets ``-static`` for the same reason.

Every candidate is made to compile *and link* a small program before it is
accepted. This is not paranoia for its own sake: a real machine this was tested
on had a gcc that answered ``--version`` perfectly well but shipped without
binutils, so it could not assemble a single instruction. Finding that out here
costs two seconds; finding it out after Nuitka has run costs a quarter of an
hour, and the error it gives ("CreateProcess: No such file or directory") names
nothing that would tell you why.

Standalone::

    python packaging/toolchain.py --out dist/launcher.exe
    python packaging/toolchain.py --which
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Same pipe problem as packaging/build.py: under CI stdout is block-buffered,
# so the compiler lines below would flush after the compile they describe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
del _stream

PACKAGING_DIR = Path(__file__).resolve().parent
LAUNCHER_SOURCE = PACKAGING_DIR / "launcher" / "launcher.c"

IS_WINDOWS = os.name == "nt"

#: What the built launcher is called. It sits at the root of the distribution,
#: which is the only thing the user is meant to double-click.
LAUNCHER_NAME = "launcher.exe" if IS_WINDOWS else "launcher"

#: The smoke test. Small, but it pulls in the same two things the launcher needs
#: that a "int main(){return 0;}" would not prove: libc and the socket library.
#: If this links, the toolchain is complete.
SMOKE_SOURCE = """\
#ifdef _WIN32
#  include <winsock2.h>
#else
#  include <sys/socket.h>
#endif
#include <stdio.h>

int main(void)
{
#ifdef _WIN32
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d) == 0)
        WSACleanup();
#else
    int s = socket(AF_INET, SOCK_STREAM, 0);
    (void)s;
#endif
    printf("ok\\n");
    return 0;
}
"""


class ToolchainError(RuntimeError):
    """No usable compiler, or the compile failed."""


@dataclass
class Compiler:
    """A compiler that has been found and, if it needed one, configured."""

    kind: str  #: "msvc", "gcc" or "clang"
    path: Path
    label: str
    #: The full environment to run it in. MSVC is useless without the INCLUDE and
    #: LIB that its batch file sets; the others inherit ours.
    env: dict[str, str] | None = field(default=None, repr=False)
    #: What ``-dumpmachine`` said, when it was asked. For MSVC, which has no such
    #: option, the triple implied by the architecture it was configured for.
    target: str | None = None
    #: The architecture MSVC builds for -- "x64", "x86", "arm64" -- read back out of
    #: cl.exe's banner rather than assumed from what it was asked for. None elsewhere.
    arch: str | None = None
    #: MSVC's ``_MSC_VER``: 1944 for compiler 19.44. What every version check in
    #: every third-party header is written against, CUDA's host_config.h included.
    msc_ver: int | None = None
    #: The toolset ``-vcvars_ver`` was pinned to, if it was: "14.39". None means
    #: vcvarsall chose, which is the newest installed.
    toolset: str | None = None


# ── Windows: Visual Studio ────────────────────────────────────────────────────

def _vswhere() -> Path | None:
    # Installed with any Visual Studio since 2017, always at this fixed path --
    # which is the entire reason vswhere exists.
    base = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles")
    if not base:
        return None
    candidate = Path(base) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    return candidate if candidate.is_file() else None


def _vs_install_paths() -> list[Path]:
    exe = _vswhere()
    if exe is None:
        return []

    queries = [
        # A VS with the C++ toolset, which is the only kind we can use...
        ["-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        # ...and then anything at all, in case the component id differs on an
        # older or Preview install. The vcvars check below is the real gate.
        ["-latest", "-prerelease", "-products", "*", "-property", "installationPath"],
    ]

    found: list[Path] = []
    for query in queries:
        try:
            proc = subprocess.run([str(exe), *query], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if line and Path(line).is_dir() and Path(line) not in found:
                found.append(Path(line))
    return found


def _decode_console(raw: bytes) -> str:
    return raw.decode("utf-8", "replace")


def msc_ver_of_toolset(name: str) -> int | None:
    """The ``_MSC_VER`` a toolset directory name implies, or None if unrecognised.

    ``VC/Tools/MSVC/14.44.35207`` is compiler 19.44, which is ``_MSC_VER`` 1944.
    Toolsets before 14.10 are excluded: they predate ``-vcvars_ver`` and cannot be
    asked for by version anyway.
    """
    match = re.fullmatch(r"14\.(\d+)\.\d+", name.strip())
    if match is None:
        return None
    minor = int(match.group(1))
    return 1900 + minor if minor >= 10 else None


def _msvc_toolsets(install: Path) -> list[tuple[int, str]]:
    """The compiler toolsets an install carries, newest first: ``[(1944, "14.44")]``.

    Usually one. The Visual Studio installer offers older ones as individual
    components precisely so that something with a version ceiling -- a CUDA
    toolkit, most often -- can still be built, and this is how they are found.
    """
    root = install / "VC" / "Tools" / "MSVC"
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    found: dict[int, str] = {}
    for entry in entries:
        version = msc_ver_of_toolset(entry.name)
        if version is not None and entry.is_dir():
            found.setdefault(version, ".".join(entry.name.split(".")[:2]))
    return sorted(found.items(), reverse=True)


#: Everything a Developer Command Prompt sets that makes vcvarsall.bat decide it has
#: already run. It short-circuits on VSCMD_VER, so invoked from inside one of those
#: prompts it prints nothing, changes nothing and exits 0 -- and asking for x64 from
#: the Start menu's x86 prompt then yields the x86 compiler, silently. The rest go
#: too so that what comes back is one toolset's environment and not two merged.
_VS_STATE = ("VSCMD_VER", "VSCMD_ARG_TGT_ARCH", "VSCMD_ARG_HOST_ARCH", "VSCMD_ARG_app_plat",
             "__VSCMD_PREINIT_PATH", "__VSCMD_script_err_count", "VSINSTALLDIR",
             "VCINSTALLDIR", "VCToolsVersion", "VCToolsInstallDir", "VCToolsRedistDir",
             "WindowsSDKVersion", "WindowsSdkDir", "INCLUDE", "LIB", "LIBPATH")


def _clean_vs_env() -> dict[str, str]:
    """This environment with any Developer Command Prompt's mark on it removed.

    PATH keeps whatever that prompt prepended -- harmless, since vcvarsall puts the
    toolset it configures in front of it, and `_find_msvc` reads the architecture back
    out of cl.exe rather than trusting the order.
    """
    return {key: value for key, value in os.environ.items() if key not in _VS_STATE}


def _msvc_env(vcvarsall: Path, target: str, toolset: str | None = None) -> dict[str, str] | None:
    """The environment left behind by vcvarsall.bat, or None if it failed.

    There is no way to ask MSVC for its include and library paths; the batch file
    sets forty environment variables and cl.exe reads them. So it is run in a
    subshell and the result is dumped and parsed.

    ``toolset`` pins the compiler version -- ``"14.39"`` for the v14.39 build tools
    -- for when something downstream has a ceiling on what it accepts. Left None,
    vcvarsall picks the newest installed. Asking for one that is not installed is
    not an error here: the batch file fails, this returns None, and the caller
    moves on to the next candidate.
    """
    version = f" -vcvars_ver={toolset}" if toolset else ""
    # chcp so `set` prints UTF-8 rather than the console's OEM code page -- otherwise
    # any non-ASCII in a path (a user's name, most often) comes back mangled and the
    # compile fails on a path that looks fine in the log. Joined with `&` and not
    # `&&`: where chcp cannot be run at all the worst case is a mangled path, and
    # that must not turn into "no compiler found" for the whole machine.
    inner = f'chcp 65001 >nul 2>&1 & call "{vcvarsall}" {target}{version} >nul && set'
    try:
        # One string, and not the argument list that would be the obvious way to
        # write this. subprocess quotes any list argument containing spaces and
        # escapes the quotes *inside* it as \" -- while cmd /s strips only the
        # outermost pair, so the batch file's own quotes survive into its name and
        # `call` reports that it "is not recognized as an internal or external
        # command". Every MSVC discovery then fails on every machine whose Visual
        # Studio is under "Program Files", which is all of them.
        # /d skips any autorun script; /s takes the rest of the line verbatim.
        proc = subprocess.run(f'cmd /d /s /c "{inner}"', capture_output=True, timeout=180,
                              env=_clean_vs_env())
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None

    env: dict[str, str] = {}
    for line in _decode_console(proc.stdout).splitlines():
        key, sep, value = line.partition("=")
        if sep and key and "\x00" not in key:
            env[key] = value

    # If INCLUDE is missing the batch file ran but did nothing useful, and cl
    # would fail later with a confusing "cannot open stdio.h".
    if "INCLUDE" not in env or "LIB" not in env:
        return None
    return env


#: cl.exe's banner: "Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35228
#: for x64". The version is what ``_MSC_VER`` will be, and the trailing word is
#: what it builds for -- the only trustworthy way to tell an x64 cl from the x86
#: one, since both live under the same install and answer to the same name.
_CL_BANNER = re.compile(r"Version\s+(\d+)\.(\d+)\.\S+\s+for\s+(\S+)", re.IGNORECASE)


def _cl_identity(cl: Path, env: dict[str, str] | None) -> tuple[int | None, str | None]:
    """``(_MSC_VER, architecture)`` from cl.exe's own banner, best effort.

    Run with no arguments cl prints the banner and a usage message and exits
    non-zero, so the exit code says nothing and only the text is read. Both streams
    are searched: which one carries the banner has moved between releases.
    """
    try:
        proc = subprocess.run([str(cl)], capture_output=True, timeout=60, env=env)
    except (OSError, subprocess.SubprocessError):
        return None, None
    match = _CL_BANNER.search(_decode_console(proc.stdout) + _decode_console(proc.stderr))
    if match is None:
        return None, None
    return int(match.group(1)) * 100 + int(match.group(2)), match.group(3).lower()


#: The triple MSVC builds for, by architecture. MSVC answers no ``-dumpmachine``, so
#: without this the word-size check in `find_compiler` has nothing to look at -- and
#: a 32-bit cl accepted as the host compiler is precisely the mistake worth catching.
_MSVC_TRIPLES = {
    "x64": "x86_64-pc-windows-msvc",
    "x86": "i686-pc-windows-msvc",
    "arm64": "aarch64-pc-windows-msvc",
    "arm": "arm-pc-windows-msvc",
}


def _msvc_host_arch() -> str:
    """The architecture MSVC should be configured to build for on this machine."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return "x64" if machine in ("amd64", "x86_64") else "x86"


def _find_msvc(*, max_msc_ver: int | None = None) -> Compiler | None:
    """MSVC, configured to build for this machine's own architecture.

    ``max_msc_ver`` is an exclusive ceiling on ``_MSC_VER``. CUDA's headers refuse a
    host compiler newer than the toolkit was released against, so a CUDA build asks
    for a toolset below its limit and an install that carries an older one is
    configured to use it. Finding nothing compatible is not fatal here: the newest
    is returned with `Compiler.msc_ver` filled in, and the caller decides whether to
    override the check or to give up with something actionable to say.
    """
    wanted = _msvc_host_arch()

    # Already inside a Developer Command Prompt: use it as-is rather than paying
    # three seconds to run a batch file that would set the same variables. But only
    # a *usable* one. The Start menu's plain "Developer Command Prompt for VS 2022"
    # configures the 32-bit compiler -- only "x64 Native Tools Command Prompt" sets
    # up the 64-bit one -- so an environment taken on trust here is how a build
    # started from that prompt ends up compiling llama.cpp for x86 and failing in
    # CMake before it reads a source file. So it is checked rather than assumed.
    if os.environ.get("INCLUDE") and os.environ.get("LIB"):
        existing = shutil.which("cl")
        if existing:
            msc_ver, arch = _cl_identity(Path(existing), None)
            fits = max_msc_ver is None or msc_ver is None or msc_ver < max_msc_ver
            # An unreadable banner is not held against it: better a compiler whose
            # details are unknown than none at all, and `_verify` is still ahead.
            if arch in (None, wanted) and fits:
                return _msvc_compiler(
                    Path(existing), arch or wanted,
                    f"MSVC {arch or wanted} (from the current environment)", msc_ver=msc_ver,
                )

    for install in _vs_install_paths():
        vcvarsall = install / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        if not vcvarsall.is_file():
            continue

        # None is "whichever vcvarsall thinks is newest", which is the common case
        # and one subshell. A ceiling turns that into a shortlist of the installed
        # toolsets under it, newest first, with None last so that a machine carrying
        # nothing compatible still gets a compiler and a diagnosis.
        candidates: list[str | None] = [None]
        if max_msc_ver is not None:
            candidates = [name for version, name in _msvc_toolsets(install)
                          if version < max_msc_ver] + [None]

        for toolset in candidates:
            env = _msvc_env(vcvarsall, wanted, toolset)
            if env is None:
                continue
            cl = shutil.which("cl", path=env.get("PATH", ""))
            if not cl:
                continue
            msc_ver, arch = _cl_identity(Path(cl), env)
            if arch is not None and arch != wanted:
                continue
            label = f"MSVC {wanted} ({install.name}"
            label += f", toolset {toolset})" if toolset else ")"
            return _msvc_compiler(Path(cl), arch or wanted, label, env, msc_ver, toolset)

    return None


def _msvc_compiler(cl: Path, arch: str, label: str, env: dict[str, str] | None = None,
                   msc_ver: int | None = None, toolset: str | None = None) -> Compiler:
    return Compiler("msvc", cl, label, env, target=_MSVC_TRIPLES.get(arch),
                    arch=arch, msc_ver=msc_ver, toolset=toolset)


def _nuitka_gcc_roots() -> list[Path]:
    """Everywhere Nuitka might have put a downloaded MinGW."""
    roots: list[Path] = []
    # An explicit cache location wins, and CI sets it often enough to matter.
    override = os.environ.get("NUITKA_CACHE_DIR")
    if override:
        roots += [Path(override) / "downloads" / "gcc", Path(override) / "DOWNLOADS" / "gcc"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        base = Path(local) / "Nuitka" / "Nuitka"
        roots += [base / "Cache" / "DOWNLOADS" / "gcc", base / "Cache" / "downloads" / "gcc", base / "gcc"]
    return roots


def _find_nuitka_mingw() -> Compiler | None:
    """The MinGW Nuitka downloads for itself.

    Worth looking for: on a machine set up to build this project at all, Nuitka
    has usually already fetched a working gcc, and using it saves telling the
    user to install Visual Studio for the sake of one 40 KB executable.
    """
    for root in _nuitka_gcc_roots():
        if not root.is_dir():
            continue
        # Newest first: the version is in the directory name, and a lexical sort
        # is close enough when the alternative is picking arbitrarily.
        for gcc in sorted(root.rglob("gcc.exe"), reverse=True):
            if gcc.parent.name.lower() == "bin" and gcc.is_file():
                return Compiler("gcc", gcc, f"MinGW gcc (Nuitka's, {gcc.parent.parent.parent.name})")
    return None


def _download_nuitka_mingw(python: Path | None) -> Compiler | None:
    """Ask Nuitka to fetch its MinGW64, then use it.

    Nuitka carries a downloader for a known-good MinGW64, but only runs it while
    it is actually compiling -- so the way to trigger it is to compile something.
    A three-line throwaway does. That is slow, half a minute plus a 100 MB
    download, which is why this is the last thing tried; but on a Windows machine
    with no compiler at all it is the difference between "go and install Visual
    Studio, then come back" and a build that simply works.
    """
    interpreter = python or Path(sys.executable)

    probe = subprocess.run(
        [str(interpreter), "-c", "import nuitka"], capture_output=True, timeout=120
    )
    if probe.returncode != 0:
        return None

    print("  no compiler found; asking Nuitka for its MinGW64 (this downloads ~100 MB)...")

    with tempfile.TemporaryDirectory(prefix="fox-mingw-") as tmp:
        scratch = Path(tmp)
        hello = scratch / "fox_toolchain_probe.py"
        hello.write_text("print('ok')\n", encoding="utf-8")

        cmd = [
            str(interpreter), "-m", "nuitka",
            "--mingw64",
            "--assume-yes-for-downloads",
            "--no-progressbar",
            f"--output-dir={scratch}",
            str(hello),
        ]
        try:
            proc = subprocess.run(cmd, cwd=scratch, capture_output=True, timeout=2400)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  could not run Nuitka to fetch MinGW64: {exc}")
            return None

    if proc.returncode != 0:
        # Not necessarily a problem. Nuitka downloads the toolchain first and only
        # then compiles, so its own compile can fail for reasons that say nothing
        # about whether gcc works -- and the launcher needs far less from a
        # compiler than Nuitka's C backend does. So look for the gcc anyway and
        # let the caller's smoke test have the final word; only report if there is
        # nothing there.
        found = _find_nuitka_mingw()
        if found is not None:
            return found
        tail = _decode_console(proc.stderr).strip().splitlines()[-6:]
        print(f"  Nuitka could not set up MinGW64 (exit {proc.returncode})")
        for line in tail:
            print(f"    {line}")
        return None

    found = _find_nuitka_mingw()
    if found is None:
        print("  Nuitka finished but no gcc.exe turned up in its cache")
    return found


# ── Discovery ─────────────────────────────────────────────────────────────────

def _find_on_path(name: str, kind: str) -> Compiler | None:
    found = shutil.which(name)
    if not found:
        return None
    return Compiler(kind, Path(found), f"{name} (on PATH)")


def _dumpmachine(compiler: Compiler) -> str | None:
    """The target triple gcc or clang was built for."""
    if compiler.kind == "msvc":
        return None
    try:
        proc = subprocess.run(
            [str(compiler.path), "-dumpmachine"],
            capture_output=True,
            timeout=60,
            env=compiler.env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return _decode_console(proc.stdout).strip() or None


def _host_is_64bit() -> bool:
    return platform.machine().lower() in ("amd64", "x86_64", "arm64", "aarch64")


def _target_is_64bit(triple: str | None) -> bool | None:
    """True, False, or None when the triple says nothing recognisable."""
    if not triple:
        return None
    low = triple.lower()
    if any(tag in low for tag in ("x86_64", "amd64", "aarch64", "arm64")):
        return True
    if any(tag in low for tag in ("i386", "i486", "i586", "i686", "mingw32")):
        return False
    return None


def _verify(compiler: Compiler) -> str | None:
    """Compile and link the smoke test. Returns None on success, else the reason.

    The whole point is to reject a compiler that cannot finish the job, so this
    runs the same command shape as the real build -- static CRT, socket library
    and all -- and insists on an executable at the end of it.
    """
    with tempfile.TemporaryDirectory(prefix="fox-cc-check-") as tmp:
        scratch = Path(tmp)
        source = scratch / "smoke.c"
        source.write_text(SMOKE_SOURCE, encoding="utf-8")
        out = scratch / ("smoke.exe" if IS_WINDOWS else "smoke")

        try:
            proc = subprocess.run(
                _command(compiler, source, out, scratch),
                cwd=scratch,
                env=compiler.env,
                capture_output=True,
                timeout=600,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"could not be run ({exc})"

        if proc.returncode != 0:
            detail = (_decode_console(proc.stderr).strip()
                      or _decode_console(proc.stdout).strip()
                      or "no output")
            first = next((ln.strip() for ln in detail.splitlines() if ln.strip()), "no output")
            return f"cannot build a program (exit {proc.returncode}: {first})"

        if not out.is_file():
            return "reported success but produced no executable"

    return None


def find_compiler(*, allow_download: bool = True, python: Path | None = None,
                  max_msc_ver: int | None = None) -> Compiler:
    """The best *working* compiler, or raise with something actionable.

    ``python`` is the interpreter to look for Nuitka in when it comes to that;
    build.py passes its build venv, where Nuitka is certain to be installed.
    ``max_msc_ver`` asks MSVC for a toolset below that ``_MSC_VER``, for a caller
    whose real compiler -- nvcc -- will not accept a newer one.
    """
    finders = (
        [lambda: _find_msvc(max_msc_ver=max_msc_ver),
         lambda: _find_on_path("clang", "clang"), lambda: _find_on_path("gcc", "gcc"),
         _find_nuitka_mingw]
        if IS_WINDOWS
        else [lambda: _find_on_path("gcc", "gcc"), lambda: _find_on_path("clang", "clang"),
              lambda: _find_on_path("cc", "gcc")]
    )
    if IS_WINDOWS and allow_download:
        finders.append(lambda: _download_nuitka_mingw(python))

    rejected: list[str] = []
    #: A compiler that works but builds for the wrong word size. Usable -- a
    #: 32-bit launcher runs fine on 64-bit Windows -- but not what anyone wants
    #: shipped, so the search carries on and only comes back to it at the end.
    wrong_width: Compiler | None = None
    seen: set[Path] = set()

    def report() -> None:
        """Say what was passed over. Only once the outcome is known, so a build
        log does not carry a scary-looking 'skipping' line for a machine where
        the very next candidate worked perfectly."""
        for line in rejected:
            print(f"  passed over {line}")

    for finder in finders:
        try:
            compiler = finder()
        except Exception as exc:  # noqa: BLE001 - a broken candidate must not stop the search
            rejected.append(f"a candidate that could not be inspected: {exc}")
            continue
        if compiler is None or compiler.path in seen:
            continue
        seen.add(compiler.path)

        # MSVC answers no `-dumpmachine`; `_find_msvc` fills in the triple for the
        # architecture it configured, and asking again would only erase it.
        if compiler.target is None:
            compiler.target = _dumpmachine(compiler)
        reason = _verify(compiler)
        if reason is not None:
            rejected.append(f"{compiler.label}: {reason}")
            continue

        width = _target_is_64bit(compiler.target)
        if width is False and _host_is_64bit():
            if wrong_width is None:
                wrong_width = compiler
            rejected.append(f"{compiler.label}: builds 32-bit ({compiler.target}) on a 64-bit host")
            continue

        report()
        return compiler

    if wrong_width is not None:
        rejected = [line for line in rejected if not line.startswith(wrong_width.label + ":")]
        report()
        print(f"  warning: using {wrong_width.label}, which builds 32-bit ({wrong_width.target}).")
        print("           The launcher will still run, but a 64-bit compiler is preferable.")
        return wrong_width

    detail = "".join(f"\n  passed over {line}" for line in rejected)
    if IS_WINDOWS:
        raise ToolchainError(
            "no working C compiler found. Install either:\n"
            "  - Visual Studio Build Tools, with the 'Desktop development with C++' workload\n"
            "    (https://visualstudio.microsoft.com/downloads/, under 'All downloads')\n"
            "  - or MinGW-w64, with gcc on PATH -- note that MinGW.org's plain 'gcc'\n"
            "    package installs no assembler or linker and cannot build anything;\n"
            "    use https://winlibs.com/ or MSYS2's mingw-w64-x86_64-gcc instead."
            + detail
        )
    raise ToolchainError(
        "no working C compiler found. Install gcc or clang (e.g. `apt install build-essential`)."
        + detail
    )


# ── Compiling ─────────────────────────────────────────────────────────────────

def _command(compiler: Compiler, source: Path, out: Path, scratch: Path) -> list[str]:
    if compiler.kind == "msvc":
        return [
            str(compiler.path),
            "/nologo",
            "/O2",          # speed; the file is small enough that size is moot
            "/MT",          # static CRT: no redistributable needed to run this
            "/W3",
            "/DNDEBUG",
            "/D_CRT_SECURE_NO_WARNINGS",
            str(source),
            f"/Fo{scratch}\\",   # keep the .obj out of the output directory
            f"/Fe{out}",
            "/link",
            "/INCREMENTAL:NO",
            "ws2_32.lib",
        ]

    # gcc and clang take the same flags for everything here.
    cmd = [
        str(compiler.path),
        "-O2",
        "-Wall",
        "-Wextra",
        "-DNDEBUG",
        "-s",  # strip: smaller, and nothing here is worth leaving symbols for
        str(source),
        "-o",
        str(out),
    ]
    if IS_WINDOWS:
        # -static so the launcher does not need libgcc or the MinGW runtime DLLs
        # sitting next to it. ws2_32 for the sockets.
        cmd += ["-static", "-lws2_32"]
    return cmd


def build_launcher(
    out: Path,
    source: Path = LAUNCHER_SOURCE,
    compiler: Compiler | None = None,
    python: Path | None = None,
) -> Path:
    """Compile ``source`` to ``out``. Returns ``out``."""
    if not source.is_file():
        raise ToolchainError(f"launcher source is missing: {source}")

    if compiler is None:
        compiler = find_compiler(python=python)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        # A stale binary left behind by a failed compile would look like success.
        out.unlink()

    with tempfile.TemporaryDirectory(prefix="fox-launcher-") as tmp:
        scratch = Path(tmp)
        cmd = _command(compiler, source, out, scratch)

        print(f"  compiler: {compiler.label}")
        print(f"  {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd,
                cwd=scratch,
                env=compiler.env,
                capture_output=True,
                timeout=600,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolchainError(f"could not run the compiler: {exc}") from exc

        stdout = _decode_console(proc.stdout).strip()
        stderr = _decode_console(proc.stderr).strip()

        # Warnings are worth seeing even on success -- this is C, and a warning
        # here is usually a real portability problem on someone else's toolchain.
        for stream in (stdout, stderr):
            for line in stream.splitlines():
                if line.strip():
                    print(f"    {line}")

        if proc.returncode != 0:
            raise ToolchainError(f"the launcher failed to compile (exit {proc.returncode})")

    if not out.is_file():
        raise ToolchainError(f"the compiler reported success but produced no {out.name}")

    if not IS_WINDOWS:
        out.chmod(0o755)

    print(f"  built {out.name}  ({out.stat().st_size:,} bytes)")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=PACKAGING_DIR / "build" / LAUNCHER_NAME,
        help="where to write the launcher",
    )
    parser.add_argument("--source", type=Path, default=LAUNCHER_SOURCE, help="the C file to compile")
    parser.add_argument(
        "--which",
        action="store_true",
        help="report the compiler that would be used and stop",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="do not fall back to downloading Nuitka's MinGW64 (Windows)",
    )
    args = parser.parse_args(argv)

    try:
        compiler = find_compiler(allow_download=not args.no_download)
        if args.which:
            target = f"  [{compiler.target}]" if compiler.target else ""
            print(f"{compiler.label}{target}")
            return 0
        build_launcher(args.out.resolve(), args.source.resolve(), compiler)
    except ToolchainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
