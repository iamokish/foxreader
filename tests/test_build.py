"""The build domain's entry point: the packaging pipeline, and what it produces.

A build cannot be exercised by importing it, and running one takes five to fifteen
minutes, so this domain owns no ``tests/test_*.py`` module. It has three halves
instead.

**Static.** ``packaging/build.py`` is read, never imported and never run: its
constants are pulled out of the syntax tree and held to the invariants that make
the difference between a build that works and one that fails at a user's first OCR
run. Cheap enough to run every time, which matters, because every one of these
was learned from something that shipped broken:

* every spelling in ``METADATA_LOOKUPS`` resolves to a distribution in
  ``DISTRIBUTION_METADATA``. Nuitka keys embedded metadata by the name each
  distribution declares and looks it up with an exact dict hit, so
  ``version("huggingface-hub")`` -- what ``import transformers`` calls before
  anything else happens -- misses a record filed as ``huggingface_hub`` and
  raises. The build proves the same thing dynamically against its own venv; this
  says so in a second, before the fifteen minutes.
* ``TOTAL_STEPS`` still matches the ``step()`` calls, or the progress counter
  lies.

**Unit.** ``packaging/llamacpp.py`` decides what llama.cpp is compiled with, and
unlike the rest of the build that decision is a pure function of the device name.
So it is imported and called directly: a CPU build must not be handed
``GGML_CUDA``, a portable build must not be handed ``GGML_NATIVE=ON``, and a CUDA
version this file has never heard of must still produce a plausible architecture
list -- the user adds those to pyproject, so nothing may hard-code the ones that
exist today.

**Live.** The scripts under ``tests/dist/``, which need something already built.
``cvcheck.py`` needs only the checkout and runs every time. ``scan_meta.py`` reads
every module in ``.venv`` looking for metadata lookups, which is a minute of work,
so it waits for ``FOX_TEST_SLOW=1``. The rest drive the compiled distribution --
they start real backends, kill every ``fox-reader.exe`` on the machine, and
download and re-download model weights -- so they are skipped unless
``FOX_TEST_DIST=1`` is set, and skipped anyway when there is no staged build.

``tests/dist/`` also holds two files that are not checks: ``hold_lock.py``, which
another test drives, and ``probe_payload.py``, which is only meaningful once
Nuitka has compiled it. They are declared as helpers in :mod:`tests.suites` so the
accounting knows about them without pretending they pass or fail.
"""
import ast
import importlib.util
import re
import shlex
import stat
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import suites

DOMAIN = suites.BUILD
PACKAGING_DIR = suites.REPO_ROOT / "packaging"
BUILD_PY = PACKAGING_DIR / "build.py"
LLAMACPP_PY = PACKAGING_DIR / "llamacpp.py"

pytestmark = pytest.mark.skipif(
    suites.child_run(),
    reason=f"running under a main ({suites.CHILD_ENV} is set)",
)

_UNKNOWN = object()


def _normalize(name: str) -> str:
    """PEP 503: what ``importlib.metadata`` compares, and Nuitka does not."""
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def _literal(node: ast.AST):
    """A literal, or a set/frozenset/tuple/list wrapped around one."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("frozenset", "set", "tuple", "list")
        and len(node.args) == 1
        and not node.keywords
    ):
        inner = _literal(node.args[0])
        if inner is _UNKNOWN:
            return _UNKNOWN
        return {"frozenset": frozenset, "set": set, "tuple": tuple, "list": list}[node.func.id](inner)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _UNKNOWN


@pytest.fixture(scope="module")
def build_source() -> ast.Module:
    if not BUILD_PY.is_file():
        pytest.skip(f"no {BUILD_PY}")
    return ast.parse(BUILD_PY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def constants(build_source: ast.Module) -> dict:
    """Module-level constants of ``packaging/build.py``, read without importing it.

    Importing would work today and is not worth the risk: a build script is
    allowed to look at the machine it is on while it loads, and a test that only
    wants four tuples should not care.
    """
    found = {}
    for node in build_source.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.isupper():
            continue
        value = _literal(node.value)
        if value is not _UNKNOWN:
            found[target.id] = value
    return found


def _require(constants: dict, name: str):
    assert name in constants, (
        f"{name} is gone from packaging/build.py, or is no longer a literal this can "
        f"read. Found: {', '.join(sorted(constants))}"
    )
    return constants[name]


def _all_metadata(constants: dict) -> list[str]:
    """Every distribution the build may ship metadata for, promised or optional."""
    return [
        *_require(constants, "DISTRIBUTION_METADATA"),
        *_require(constants, "OPTIONAL_DISTRIBUTION_METADATA"),
    ]


# ── static: the metadata rules ────────────────────────────────────────────────

def test_every_metadata_lookup_is_shipped(constants):
    """Each spelling something asks for resolves to a distribution that ships.

    Against DISTRIBUTION_METADATA alone, deliberately: the optional list is
    allowed to be absent from a build, so a lookup that only it satisfies is a
    lookup that raises whenever that backend was left out.
    """
    lookups = _require(constants, "METADATA_LOOKUPS")
    shipped = {_normalize(name) for name in _require(constants, "DISTRIBUTION_METADATA")}

    missing = [name for name in lookups if _normalize(name) not in shipped]
    assert not missing, (
        "these lookups resolve to nothing that ships:\n"
        + "".join(f"    importlib.metadata.version({name!r})\n" for name in missing)
        + "  Nuitka files metadata under the name the distribution declares and reads it\n"
        "  back with an exact dict lookup -- nothing normalises -- so a compiled build\n"
        "  raises PackageNotFoundError here while a real install would not. Add the\n"
        "  distribution that provides each name to DISTRIBUTION_METADATA."
    )


def test_distribution_metadata_has_no_duplicate_spellings(constants):
    """Two spellings of one distribution would make the build ask for it twice."""
    names = _all_metadata(constants)
    seen: dict[str, str] = {}
    duplicates = []
    for name in names:
        key = _normalize(name)
        if key in seen:
            duplicates.append(f"{seen[key]} and {name}")
        else:
            seen[key] = name
    assert not duplicates, f"DISTRIBUTION_METADATA names the same distribution twice: {duplicates}"


def test_optional_metadata_is_not_also_required(constants):
    """A distribution is either promised or optional, and cannot be both.

    Both lists go to the same probe and the same copy loop, so a name in both is
    copied twice and, worse, reads as a promise the staged-build check enforces --
    which would make ``--no-gguf`` fail that check for a backend it deliberately
    left out.
    """
    required = {_normalize(name) for name in _require(constants, "DISTRIBUTION_METADATA")}
    optional = {_normalize(name) for name in _require(constants, "OPTIONAL_DISTRIBUTION_METADATA")}

    both = sorted(required & optional)
    assert not both, (
        f"{', '.join(both)} is in DISTRIBUTION_METADATA and "
        "OPTIONAL_DISTRIBUTION_METADATA.\n  Optional means a build without it still "
        "passes; required means it must be there. Pick one."
    )


# ── static: the progress counter ──────────────────────────────────────────────

def test_total_steps_matches_the_step_calls(build_source, constants):
    """``TOTAL_STEPS`` is what the user counts against, so it has to be right.

    The accounting: every ``step()`` at function-body level runs on every build.
    The nested ones are branches -- either a substitute for one of those ("Skipping
    the frontend build." instead of building it) or one half of a pair where
    exactly one side runs (the build venv, reused or created). Only the pair adds a
    step, and there is one of them.
    """
    calls = [
        node
        for node in ast.walk(build_source)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "step"
    ]
    unconditional = [node for node in calls if node.col_offset == 4]
    nested = [node for node in calls if node.col_offset > 4]

    assert len(nested) == 4, (
        f"packaging/build.py has {len(nested)} conditional step() calls, not 4 "
        f"(lines {sorted(node.lineno for node in nested)}).\n"
        "  Work out whether each new one *adds* a step or *replaces* one, then update "
        "this test\n  and TOTAL_STEPS together -- the progress counter is wrong either "
        "way otherwise."
    )
    total = _require(constants, "TOTAL_STEPS")
    assert total == len(unconditional) + 1, (
        f"TOTAL_STEPS is {total}, but there are {len(unconditional)} unconditional "
        f"step() calls plus one\n  for the reuse-or-create pair, which comes to "
        f"{len(unconditional) + 1}. The build would count to the wrong number."
    )


# ── unit: what llama.cpp gets compiled with ───────────────────────────────────

@pytest.fixture(scope="module")
def llama() -> object:
    """``packaging/llamacpp.py``, imported from its path.

    ``packaging/`` is not a package and is not on the path, so this loads the file
    directly. It has no import-time side effects -- everything that looks at the
    machine is behind a function -- so importing it is safe here even though
    ``build.py`` next to it is only ever parsed.
    """
    if not LLAMACPP_PY.is_file():
        pytest.skip(f"no {LLAMACPP_PY}")

    spec = importlib.util.spec_from_file_location("fox_packaging_llamacpp", LLAMACPP_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered *before* executing it: the module defines dataclasses, and
    # `dataclasses` resolves a class's module through `sys.modules[cls.__module__]`
    # to read its globals. Without this the class body raises AttributeError on
    # None, which is a confusing way to find out about a two-line fixture bug.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    yield module
    sys.modules.pop(spec.name, None)


def _args(llama, device: str, **kwargs) -> dict[str, str]:
    """``cmake_args`` for a device, as a dict of option to value.

    ``machine`` is pinned so the result does not depend on the machine running the
    tests -- on arm64 the x86 ISA options are left out entirely, which is correct
    there and would make these assertions meaningless.
    """
    kwargs.setdefault("machine", "x86_64")
    args = llama.cmake_args(llama.parse_device(device), **kwargs)
    out = {}
    for arg in args:
        assert arg.startswith("-D"), f"{arg!r} is not a -D argument"
        name, _, value = arg[2:].partition("=")
        out[name] = value
    return out


@pytest.mark.parametrize("device, kind, cuda, rocm", [
    ("cpu", "cpu", None, None),
    ("macos", "metal", None, None),
    ("metal", "metal", None, None),
    ("cu118", "cuda", (11, 8), None),
    ("cu126", "cuda", (12, 6), None),
    ("cu129_win", "cuda", (12, 9), None),
    ("cu130", "cuda", (13, 0), None),
    # A two-digit minor, which is the only shape a CUDA 12.10 could have.
    ("cu1210", "cuda", (12, 10), None),
    ("rocm71", "rocm", None, (7, 1)),
    ("rocm7.2", "rocm", None, (7, 2)),
    ("hip-6.4", "rocm", None, (6, 4)),
])
def test_parse_device_reads_the_version_out_of_the_name(llama, device, kind, cuda, rocm):
    """No lookup table: a torch extra added to pyproject next year must just work."""
    target = llama.parse_device(device)
    assert (target.kind, target.cuda, target.rocm) == (kind, cuda, rocm)


@pytest.mark.parametrize("device", ["", "cu", "gpu", "cuda12", "vulkan", "sycl", "rock"])
def test_parse_device_refuses_what_it_cannot_read(llama, device):
    """The one thing this module exists to prevent is a silent downgrade to CPU."""
    with pytest.raises(llama.GgufError):
        llama.parse_device(device)


def test_no_device_is_built_natively(llama):
    """``GGML_NATIVE=ON`` bakes in this machine's CPU. Every device says OFF."""
    for device in ("cpu", "cu129", "rocm72", "macos"):
        assert _args(llama, device, toolkit=(12, 9))["GGML_NATIVE"] == "OFF"


def test_cpu_build_has_no_gpu_backend(llama):
    """A CPU distribution that quietly links CUDA would not run at all."""
    args = _args(llama, "cpu")
    for option in ("GGML_CUDA", "GGML_HIP", "GGML_METAL"):
        assert args.get(option, "OFF") == "OFF", f"{option} is on for a CPU build"


@pytest.mark.parametrize("baseline, expected_avx2", [("avx2", "ON"), ("avx", "OFF"), ("sse42", "OFF")])
def test_cpu_baseline_pins_every_isa_option(llama, baseline, expected_avx2):
    """Named options on, everything else off -- not ggml's defaults.

    With ``GGML_NATIVE=OFF`` ggml still turns AVX2 and friends on by itself, so
    ``--cpu-baseline sse42`` only means anything if the rest are pinned OFF.
    """
    args = _args(llama, "cpu", cpu_baseline=baseline)
    assert args["GGML_AVX2"] == expected_avx2
    assert args["GGML_SSE42"] == "ON", "every baseline here is at least SSE4.2"
    for option in llama.CPU_ISA_OPTIONS:
        assert args[option] in ("ON", "OFF"), f"{option} was left to ggml"


def test_unknown_cpu_baseline_is_an_error(llama):
    with pytest.raises(llama.GgufError):
        _args(llama, "cpu", cpu_baseline="avx512")


def test_cuda_build_asks_for_cuda_and_named_architectures(llama):
    args = _args(llama, "cu129", toolkit=(12, 9))
    assert args["GGML_CUDA"] == "ON"
    assert args["CMAKE_CUDA_ARCHITECTURES"], "no architectures: the build would be for nothing"
    assert args.get("GGML_HIP", "OFF") == "OFF"


@pytest.mark.parametrize("version", [(11, 7), (11, 8), (12, 0), (12, 6), (12, 9), (13, 0), (14, 2), (15, 0)])
def test_cuda_architectures_are_plausible_for_any_version(llama, version):
    """Including versions that do not exist yet -- the user adds those to pyproject.

    Every entry has to be something nvcc would accept: a number, optionally with
    the ``a`` of an architecture-specific target, optionally ``-real`` or
    ``-virtual``. And at least one has to be ``-virtual``, because that is the PTX
    a GPU newer than the toolkit falls back to.
    """
    entries = llama.cuda_architectures(version)
    assert entries, f"CUDA {version} produced no architectures"
    for entry in entries:
        assert re.fullmatch(r"\d{2,3}a?(?:-real|-virtual)?", entry), f"nvcc would reject {entry!r}"
    assert any(entry.endswith("-virtual") for entry in entries), (
        "no PTX in the list, so a GPU newer than this toolkit has nothing to fall back to"
    )


def test_cuda_13_drops_the_architectures_it_removed(llama):
    """CUDA 13 dropped everything below Turing; asking for 50 is a hard nvcc error."""
    for entry in llama.cuda_architectures((13, 0)):
        assert int(re.match(r"\d+", entry).group()) >= 75, f"CUDA 13 cannot build {entry}"


def test_rocm_build_names_the_targets_every_way_hip_reads_them(llama):
    """The variable ggml reads has changed name twice; all three are set."""
    args = _args(llama, "rocm72")
    assert args["GGML_HIP"] == "ON"
    assert args.get("GGML_CUDA", "OFF") == "OFF"
    targets = {args.get("GPU_TARGETS"), args.get("AMDGPU_TARGETS"), args.get("CMAKE_HIP_ARCHITECTURES")}
    assert len(targets) == 1 and None not in targets, f"the gfx list is not consistent: {targets}"


def test_explicit_architectures_are_passed_through(llama):
    """``--cuda-arch``/``--hip-arch``: the escape hatch has to actually escape."""
    assert _args(llama, "cu129", toolkit=(12, 9),
                 cuda_arch="75-real;86-real")["CMAKE_CUDA_ARCHITECTURES"] == "75-real;86-real"
    assert _args(llama, "rocm72", hip_arch="gfx1100")["GPU_TARGETS"] == "gfx1100"


def test_metal_build_embeds_its_shader_library(llama):
    """Unembedded, the shaders are a file next to a dylib that ships without it."""
    args = _args(llama, "macos")
    assert args["GGML_METAL"] == "ON"
    assert args["GGML_METAL_EMBED_LIBRARY"] == "ON"


def test_the_cmake_args_value_survives_being_split_again(llama):
    """scikit-build-core reads ``CMAKE_ARGS`` with ``shlex.split`` in POSIX mode.

    So the string this builds has to come back out as the same list it went in as.
    An OpenBLAS under "C:/Program Files/..." is the case that breaks a naive join:
    unquoted, half the path becomes a separate argument and CMake is handed a
    directory that does not exist.
    """
    args = llama.cmake_args(
        llama.parse_device("cpu"),
        machine="x86_64",
        openblas=llama.OpenBlas(
            root=Path("C:/Program Files/OpenBLAS"),
            include=Path("C:/Program Files/OpenBLAS/include"),
            library=Path("C:/Program Files/OpenBLAS/lib/libopenblas.lib"),
            runtime=(),
            label="test",
        ),
    )
    assert shlex.split(llama.cmake_args_value(args)) == args


def test_blas_is_only_asked_for_when_there_is_a_blas(llama):
    """A missing OpenBLAS degrades to no BLAS; ggml's is a FATAL_ERROR otherwise."""
    args = _args(llama, "cpu", openblas=None)
    assert args.get("GGML_BLAS", "OFF") == "OFF"
    assert "BLAS_INCLUDE_DIRS" not in args


def _capture_install(llama, monkeypatch, tmp_path, build, installed=None, repairs=None):
    """Run ``install`` with every subprocess faked; return the calls it made.

    `_repair_top_level` is stubbed rather than allowed to run. It probes the
    environment with a subprocess of its own, which is not one of the commands
    these tests are about -- and counting it would make an assertion about the uv
    command line fail for a reason that has nothing to do with uv. Pass `repairs`
    to see that it was reached.
    """
    calls: list[tuple[list[str], dict]] = []
    repaired = repairs if repairs is not None else []

    # A constant `_installed_version` looks "still installed" after the
    # uninstall, which would send `install` into `_force_remove_llama` and its
    # real probes. That path has its own tests below; here it is a no-op so
    # these stay assertions about the uv command line.
    monkeypatch.setattr(llama, "_force_remove_llama", lambda *a, **k: None)

    class Done:
        returncode = 0
        # `_capture` reads these. Nothing here should reach it, but an addition to
        # `install` that does deserves a stub answer rather than an AttributeError
        # from inside a helper.
        stdout = b""
        stderr = b""

    monkeypatch.setattr(llama.shutil, "which", lambda name: "uv" if name == "uv" else None)
    monkeypatch.setattr(llama, "_installed_version", lambda *a, **k: installed)
    monkeypatch.setattr(llama, "_repair_top_level", lambda *a, **k: repaired.append(a))
    monkeypatch.setattr(
        llama.subprocess, "run",
        lambda cmd, **kw: (calls.append((list(cmd), dict(kw.get("env") or {}))), Done())[1],
    )
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    llama.install(python, build, env={})
    return python, calls


def _cpu_build(llama, spec="llama-cpp-python==0.3.35"):
    return llama.Build(
        target=llama.parse_device("cpu"),
        args=["-DGGML_NATIVE=OFF"],
        value="-DGGML_NATIVE=OFF",
        spec=spec,
    )


def test_install_runs_exactly_the_documented_command(llama, monkeypatch, tmp_path):
    """``CMAKE_ARGS="..." uv pip install --no-cache-dir llama-cpp-python==<locked>``.

    Nothing else. ``--no-cache-dir`` is the whole guarantee: uv neither reads nor
    writes its built-wheel cache, so the wheel installed is the one just compiled
    with the arguments in the environment. Any extra cache-shaped flag here would
    mean some path through this function can reuse a wheel built for a different
    device, which is the segmentation fault this module exists to prevent.
    """
    build = _cpu_build(llama)
    python, calls = _capture_install(llama, monkeypatch, tmp_path, build)

    assert len(calls) == 1, f"nothing was installed already, so one command only: {calls}"
    cmd, env = calls[0]
    assert cmd == [
        "uv", "pip", "install", "--python", str(python), "--no-cache-dir",
        "llama-cpp-python==0.3.35",
    ]
    assert env["CMAKE_ARGS"] == build.value


def test_install_repairs_the_distribution_metadata_afterwards(llama, monkeypatch, tmp_path):
    """The install is not finished when uv returns.

    What it leaves behind has no `top_level.txt`, and Nuitka needs one to know
    whose metadata `--include-distribution-metadata` is bundling -- so a compile
    that follows a successful install stops with an error naming ``bin``. The
    repair belongs to installing, not to compiling, because this is the only point
    where the distribution has just been written and is known to be the one the
    compile will see.
    """
    repairs: list = []
    _capture_install(llama, monkeypatch, tmp_path, _cpu_build(llama), repairs=repairs)

    assert repairs, "install must repair the metadata it just wrote, or Nuitka stops on it"


def test_install_replaces_a_package_already_in_the_environment(llama, monkeypatch, tmp_path):
    """An already-satisfied requirement is one uv audits and skips.

    Which would leave whatever an earlier build compiled, for whatever device that
    was. Removed first rather than by adding ``--reinstall-package`` to the install.
    """
    build = _cpu_build(llama)
    python, calls = _capture_install(llama, monkeypatch, tmp_path, build, installed="0.3.35")

    assert len(calls) == 2, f"expected an uninstall then an install: {calls}"
    assert calls[0][0] == ["uv", "pip", "uninstall", "--python", str(python), "llama-cpp-python"]
    assert calls[1][0][-1] == "llama-cpp-python==0.3.35"
    assert "--no-cache-dir" in calls[1][0]


def _install_with(llama, monkeypatch, tmp_path, build, installed, uninstall_rc=0):
    """Run `install` with uninstall/install exit codes controlled separately."""
    forced: list[bool] = []
    uninstalls: list[list[str]] = []

    class Done:
        returncode = 0

    class Failed:
        returncode = 1

    monkeypatch.setattr(llama.shutil, "which", lambda name: "uv" if name == "uv" else None)
    monkeypatch.setattr(llama, "_installed_version", lambda *a, **k: installed)
    monkeypatch.setattr(llama, "_force_remove_llama", lambda *a, **k: forced.append(True))
    monkeypatch.setattr(llama, "_repair_top_level", lambda *a, **k: None)

    def fake_run(cmd, **kw):
        if "uninstall" in cmd:
            uninstalls.append(list(cmd))
            return Failed() if uninstall_rc else Done()
        return Done()

    monkeypatch.setattr(llama.subprocess, "run", fake_run)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    llama.install(python, build, env={})
    return forced, uninstalls


def test_a_failed_uninstall_is_finished_by_hand(llama, monkeypatch, tmp_path):
    """uv's uninstall is not atomic on Windows: a half-removed install left in
    place makes the install below fail reading the old metadata. A nonzero
    exit is finished by hand."""
    forced, uninstalls = _install_with(llama, monkeypatch, tmp_path,
                                       _cpu_build(llama), installed="0.3.35",
                                       uninstall_rc=1)

    assert uninstalls, "expected an uninstall attempt first"
    assert forced, ("the failed uninstall was not finished by hand, so the install "
                    "below reads a half-removed previous install")


def test_a_half_removed_install_is_finished_by_hand(llama, monkeypatch, tmp_path):
    """Exit code 0 is not proof either: still importable afterwards means the
    uninstall did nothing, and installing over it is the same failure."""
    forced, _ = _install_with(llama, monkeypatch, tmp_path,
                              _cpu_build(llama), installed="0.3.35",
                              uninstall_rc=0)

    assert forced, ("the package is still there after the uninstall, so the install "
                    "below would read its metadata")


def test_a_clean_environment_needs_no_force(llama, monkeypatch, tmp_path):
    """Nothing installed, nothing forced: the happy path gains no extra step."""
    forced, uninstalls = _install_with(llama, monkeypatch, tmp_path,
                                       _cpu_build(llama), installed=None)

    assert not uninstalls and not forced


def test_force_remove_deletes_only_its_own_dirs(llama, monkeypatch, tmp_path):
    """The package directory and its own `.dist-info` go; a neighbour stays."""
    lib = tmp_path / "site-packages" / "llama_cpp" / "lib"
    lib.mkdir(parents=True)
    (lib / "ggml-cuda.dll").write_bytes(b"")
    dist_info = tmp_path / "site-packages" / "llama_cpp_python-0.3.35.dist-info"
    dist_info.mkdir(parents=True)
    neighbour = tmp_path / "site-packages" / "torch"
    neighbour.mkdir()
    monkeypatch.setattr(llama, "_package_lib_dir", lambda *a, **k: lib)
    monkeypatch.setattr(llama, "_dist_info_dir", lambda *a, **k: dist_info)

    llama._force_remove_llama(Path("python"))

    assert not (tmp_path / "site-packages" / "llama_cpp").exists()
    assert not dist_info.exists()
    assert neighbour.is_dir(), "a neighbour package must never be touched"


def test_install_refuses_a_version_the_lock_does_not_pin(llama, monkeypatch, tmp_path):
    """Two builds of one commit must not compile two different llama.cpp releases."""
    build = _cpu_build(llama, spec="llama-cpp-python")
    with pytest.raises(llama.GgufError, match="uv.lock"):
        _capture_install(llama, monkeypatch, tmp_path, build)


def test_the_version_installed_is_the_one_uv_lock_pins(llama):
    """The spec comes out of uv.lock, not from resolving afresh at build time."""
    version = llama.locked_version()
    if version is None:
        pytest.skip("uv.lock does not name llama-cpp-python")
    assert re.fullmatch(r"\d+(\.\d+)*", version), version


def test_rocm_clang_is_found_in_either_layout(llama, tmp_path):
    """Linux ROCm keeps clang at ``llvm/bin``; the Windows HIP SDK at ``lib/llvm/bin``."""
    linux = tmp_path / "opt-rocm"
    (linux / "llvm" / "bin").mkdir(parents=True)
    windows = tmp_path / "hip-sdk"
    (windows / "lib" / "llvm" / "bin").mkdir(parents=True)

    assert llama._rocm_llvm(linux) == linux / "llvm" / "bin"
    assert llama._rocm_llvm(windows) == windows / "lib" / "llvm" / "bin"
    assert llama._rocm_llvm(tmp_path / "not-rocm") is None


def test_windows_hip_build_asks_for_ninja_and_rocm_clang(llama, tmp_path, monkeypatch):
    """llama.cpp's documented Windows HIP build, which MSVC cannot do.

    The Visual Studio generator cannot drive a HIP compile and MSVC cannot compile
    the device code, so both languages are pointed at ROCm's own clang. Skipped off
    Windows: what the environment should be is decided by the build host, and the
    ``.exe`` names only exist there.
    """
    if not llama.IS_WINDOWS:
        pytest.skip("the Windows HIP configuration is only produced on Windows")

    root = tmp_path / "ROCm" / "7.2"
    llvm = root / "lib" / "llvm" / "bin"
    llvm.mkdir(parents=True)
    for name in ("clang.exe", "clang++.exe"):
        (llvm / name).write_text("", encoding="utf-8")

    env = llama._describe_environment(llama.parse_device("rocm72"), None, root)
    assert env["CMAKE_GENERATOR"] == "Ninja"
    assert env["CXX"] == str(llvm / "clang++.exe")
    assert env["CC"] == str(llvm / "clang.exe")
    assert env["HIPCXX"] == str(llvm / "clang++.exe")
    assert env["ROCM_PATH"] == str(root) and env["HIP_PATH"] == str(root)


def test_the_openblas_import_library_matches_the_linker(llama, tmp_path):
    """The Windows OpenBLAS release ships one import library per linker.

    ``lib/libopenblas.lib`` is MSVC's and ``lib/libopenblas.dll.a`` is MinGW's, and
    they sit in the same directory. Picking by filesystem order rather than by
    compiler is a mistake CMake cannot catch -- it checks the file exists, nothing
    more -- so it surfaces as an undefined ``cblas_sgemm`` at the point ggml's BLAS
    backend links, long after the configuration that chose wrong.
    """
    if not llama.IS_WINDOWS:
        pytest.skip("only the Windows release ships both forms")

    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libopenblas.lib").write_bytes(b"")
    (lib / "libopenblas.dll.a").write_bytes(b"")

    assert llama._link_library(tmp_path, gnu=True).name == "libopenblas.dll.a"
    assert llama._link_library(tmp_path, gnu=False).name == "libopenblas.lib"


def test_either_import_library_alone_is_still_usable(llama, tmp_path):
    """A preference, not a requirement: one form is better than no BLAS at all."""
    if not llama.IS_WINDOWS:
        pytest.skip("only the Windows release ships both forms")

    msvc_only = tmp_path / "msvc" / "lib"
    msvc_only.mkdir(parents=True)
    (msvc_only / "libopenblas.lib").write_bytes(b"")
    assert llama._link_library(msvc_only.parent, gnu=True).name == "libopenblas.lib"

    gnu_only = tmp_path / "gnu" / "lib"
    gnu_only.mkdir(parents=True)
    (gnu_only / "libopenblas.dll.a").write_bytes(b"")
    assert llama._link_library(gnu_only.parent, gnu=False).name == "libopenblas.dll.a"

    assert llama._link_library(tmp_path / "nothing-here", gnu=False) is None


def test_openblas_dll_travels_into_the_package_lib_dir(llama, monkeypatch, tmp_path):
    """The OpenBLAS DLL has to be copied next to libllama before `verify` runs.

    `bundle` copies it too, but `verify` happens in between, so leaving it to
    `bundle` means the import check fails on a layout the shipped tree would have
    got right. What that looks like from the outside is a build aborting because
    ``libllama.dll`` cannot be found, when libllama is there and the only absentee
    is a dependency two steps down -- ``ggml-blas.dll`` asks for OpenBLAS, and
    nothing but this directory is on the search path.
    """
    lib = tmp_path / "site-packages" / "llama_cpp" / "lib"
    lib.mkdir(parents=True)
    monkeypatch.setattr(llama, "_package_lib_dir", lambda *a, **k: lib)

    blas = tmp_path / "openblas" / "bin"
    blas.mkdir(parents=True)
    (blas / "libopenblas.dll").write_bytes(b"blas")
    runtime = tmp_path / "mingw" / "bin"
    runtime.mkdir(parents=True)
    (runtime / "libstdc++-6.dll").write_bytes(b"stdc++")

    openblas = llama.OpenBlas(
        root=blas.parent, include=blas.parent, library=blas.parent / "libopenblas.dll.a",
        runtime=(blas / "libopenblas.dll",), label="test",
    )
    build = llama.Build(
        target=llama.parse_device("cpu"), args=[], value="", spec="x",
        openblas=openblas, notes=[], runtime_libs=[runtime / "libstdc++-6.dll"],
    )
    llama._copy_runtime_libraries(Path(sys.executable), build)

    assert (lib / "libopenblas.dll").read_bytes() == b"blas"
    assert (lib / "libstdc++-6.dll").read_bytes() == b"stdc++"


def test_the_compilers_own_runtime_wins_over_openblas_copy(llama, monkeypatch, tmp_path):
    """A MinGW-built OpenBLAS ships its own libgcc, and it is the wrong one.

    Whatever the compiler just built llama.cpp against is the copy that has to be
    there; a second one from an unrelated toolchain overwriting it is a way to get
    a library that loads and then misbehaves.
    """
    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setattr(llama, "_package_lib_dir", lambda *a, **k: lib)

    blas = tmp_path / "openblas"
    blas.mkdir()
    (blas / "libgcc_s_seh-1.dll").write_bytes(b"openblas-copy")
    compiler = tmp_path / "mingw"
    compiler.mkdir()
    (compiler / "libgcc_s_seh-1.dll").write_bytes(b"compiler-copy")

    openblas = llama.OpenBlas(
        root=blas, include=blas, library=blas / "libopenblas.dll.a",
        runtime=(blas / "libgcc_s_seh-1.dll",), label="test",
    )
    build = llama.Build(
        target=llama.parse_device("cpu"), args=[], value="", spec="x",
        openblas=openblas, notes=[], runtime_libs=[compiler / "libgcc_s_seh-1.dll"],
    )
    llama._copy_runtime_libraries(Path(sys.executable), build)

    assert (lib / "libgcc_s_seh-1.dll").read_bytes() == b"compiler-copy"


def test_a_registry_that_cannot_be_read_is_not_a_missing_backend(llama):
    """Three answers, not two: yes, no, and *could not ask*.

    Collapsing the third into "no" would fail a good build the moment a future
    llama.cpp stops exporting the symbols the probe reads, and collapsing it into
    "yes" would ship a CPU-only library into a CUDA distribution. Both are worse
    than saying so.
    """
    assert llama._registered(["CUDA", "CPU"], ("CUDA",)) is True
    assert llama._registered(["CPU"], ("CUDA",)) is False
    assert llama._registered([], ("CUDA",)) is False
    assert llama._registered(None, ("CUDA",)) is None
    assert llama._registered("CUDA", ("CUDA",)) is None

    # Case and stray whitespace come from a C string, not from a promise.
    assert llama._registered([" cuda "], ("CUDA",)) is True


def test_the_hip_build_answers_to_more_than_one_name(llama):
    """llama.cpp's HIP backend has registered as ROCm and, historically, as CUDA.

    Insisting on one spelling would fail a working ROCm build on an upstream rename,
    and the check exists to catch a *CPU-only* library in a GPU distribution -- which
    every accepted spelling still catches.
    """
    for name in ("ROCm", "HIP", "CUDA"):
        assert llama._registered([name], llama.BACKEND_NAMES["rocm"]) is True
    assert llama._registered(["CPU", "BLAS"], llama.BACKEND_NAMES["rocm"]) is False


def test_blas_is_checked_against_the_registry_not_the_info_string(llama):
    """`llama_print_system_info` cannot see the BLAS backend, and never could.

    It lists a backend only if that backend exposes a `ggml_backend_get_features`
    proc address, and ggml-blas registers a NULL one. So a perfectly good OpenBLAS
    build reports ``CPU : ...`` and nothing else -- which, checked against the
    string, is a warning on every BLAS build ever produced.
    """
    cpu_only_info = "CPU : SSE3 = 1 | AVX2 = 1 | LLAMAFILE = 1 |"
    assert llama._has_backend(cpu_only_info, llama.BLAS_BACKEND) is False
    assert llama._registered(["BLAS", "CPU"], (llama.BLAS_BACKEND,)) is True

    # The old flag spelling still has to work, for the fallback path.
    assert llama._has_backend("BLAS = 1 |", llama.BLAS_BACKEND) is True
    assert llama._has_backend("BLAS = 0 |", llama.BLAS_BACKEND) is False


def test_the_verify_probe_is_valid_python_and_reads_the_registry(llama):
    """It runs in the *target* interpreter as ``-c``, so nothing here parses it.

    A syntax error would surface as unparsable JSON from a subprocess, reported as
    the build environment failing to answer -- so it is compiled here instead. The
    module compiles it at import too; this also pins what it looks at.
    """
    compile(llama._VERIFY_PROBE, "<probe>", "exec")

    for symbol in ("ggml_backend_reg_count", "ggml_backend_reg_get", "ggml_backend_reg_name"):
        assert symbol in llama._VERIFY_PROBE, f"the probe no longer reads {symbol}"
    # Docstrings inside the probe must not use the quoting the probe itself uses.
    assert '"""' not in llama._VERIFY_PROBE


def test_the_dist_info_probe_is_valid_python(llama):
    """It runs in the build venv as ``-c``, so nothing else parses it.

    A syntax error would come back as an empty answer, reported as the `.dist-info`
    being unfindable -- a warning about the wrong thing entirely.
    """
    compile(llama._DIST_INFO_PROBE, "<probe>", "exec")

    # The distribution name arrives as an argument rather than interpolated, so a
    # name with a quote in it cannot end up as code.
    assert "sys.argv[1]" in llama._DIST_INFO_PROBE
    assert llama.DISTRIBUTION not in llama._DIST_INFO_PROBE


def test_the_missing_top_level_file_is_written_naming_the_package(llama, monkeypatch, tmp_path):
    """Nuitka reads `top_level.txt` to learn whose metadata it is being asked to bundle.

    llama-cpp-python ships none, and installs CMake's install tree beside the
    package -- so the file list Nuitka falls back to guessing from starts with
    ``bin/``, ``include/`` and ``lib/``, which are data and sort ahead of
    ``llama_cpp``. It takes the first, then stops the build because no module
    called ``bin`` was compiled. The line written here is what the file would say
    if upstream shipped it.
    """
    dist_info = tmp_path / "llama_cpp_python-0.3.35.dist-info"
    dist_info.mkdir()
    monkeypatch.setattr(llama, "_dist_info_dir", lambda *a, **k: dist_info)
    monkeypatch.setattr(llama, "_capture", lambda *a, **k: (0, llama.PACKAGE))

    llama._repair_top_level(Path(sys.executable))

    assert (dist_info / "top_level.txt").read_text(encoding="utf-8").split() == [llama.PACKAGE]


def test_a_top_level_file_that_already_names_the_package_is_left_alone(llama, monkeypatch, tmp_path):
    """A release that ships its own correct one must not be rewritten.

    The check is that the package is named, not that the file matches what would
    have been written: a file listing more than one top-level name is still right,
    and replacing it would drop the others.
    """
    dist_info = tmp_path / "llama_cpp_python-9.9.9.dist-info"
    dist_info.mkdir()
    original = f"{llama.PACKAGE}\nllama_cpp_extras\n"
    (dist_info / "top_level.txt").write_text(original, encoding="utf-8")
    monkeypatch.setattr(llama, "_dist_info_dir", lambda *a, **k: dist_info)
    monkeypatch.setattr(llama, "_capture", lambda *a, **k: pytest.fail("must not verify a no-op"))

    llama._repair_top_level(Path(sys.executable))

    assert (dist_info / "top_level.txt").read_text(encoding="utf-8") == original


def test_a_top_level_file_naming_the_wrong_thing_is_replaced(llama, monkeypatch, tmp_path):
    """Whatever names the build venv inherited, only the package answers the question."""
    dist_info = tmp_path / "llama_cpp_python-0.3.35.dist-info"
    dist_info.mkdir()
    (dist_info / "top_level.txt").write_text("bin\ninclude\nlib\n", encoding="utf-8")
    monkeypatch.setattr(llama, "_dist_info_dir", lambda *a, **k: dist_info)
    monkeypatch.setattr(llama, "_capture", lambda *a, **k: (0, llama.PACKAGE))

    llama._repair_top_level(Path(sys.executable))

    assert (dist_info / "top_level.txt").read_text(encoding="utf-8").split() == [llama.PACKAGE]


def test_a_top_level_file_that_does_not_resolve_is_reported(llama, monkeypatch, tmp_path, capsys):
    """Writing the file is not the same as the file answering the question.

    The only way this fails quietly is the name being written and still not
    resolving, so the read-back is checked and its failure named. Not fatal:
    Nuitka reports it too, and this is what connects the two messages.
    """
    dist_info = tmp_path / "llama_cpp_python-0.3.35.dist-info"
    dist_info.mkdir()
    monkeypatch.setattr(llama, "_dist_info_dir", lambda *a, **k: dist_info)
    # Nothing importable came back -- what `find_spec` returning None looks like.
    monkeypatch.setattr(llama, "_capture", lambda *a, **k: (0, ""))

    llama._repair_top_level(Path(sys.executable))

    out = capsys.readouterr().out
    assert "warning" in out and "does not resolve" in out
    assert "OPTIONAL_DISTRIBUTION_METADATA" in out


def test_a_missing_dist_info_is_a_warning_not_a_failure(llama, monkeypatch, capsys):
    """The install is already done by this point, and the metadata is a nicety.

    Nothing about the built library depends on it, so an unfindable `.dist-info`
    must not undo a compile -- it just has to say so, because the Nuitka error it
    leads to names neither this distribution's package nor anything actionable.
    """
    monkeypatch.setattr(llama, "_dist_info_dir", lambda *a, **k: None)

    llama._repair_top_level(Path(sys.executable))

    out = capsys.readouterr().out
    assert "warning" in out and "dist-info" in out


def test_a_compiler_that_cannot_build_cpp_is_rejected_before_the_build(llama, monkeypatch):
    """llama.cpp is C++, and ``toolchain.py`` only ever proved the compiler does C.

    The two are not the same claim. A gcc whose libstdc++ headers it cannot reach
    compiles every ``.c`` in ggml and fails on the first ``.cpp`` -- twenty minutes
    in, under a hundred lines of make output naming a header rather than a cause.
    So one three-line C++ file is compiled first.
    """
    compiler = SimpleNamespace(kind="gcc", label="MinGW gcc (test)", path=Path("g++"), env=None)
    monkeypatch.delenv(llama.SKIP_PROBE_ENV, raising=False)

    monkeypatch.setattr(llama, "_capture", lambda *a, **k: (0, ""))
    llama._verify_cxx(compiler, Path("g++"), {})  # a working compiler says nothing

    monkeypatch.setattr(llama, "_capture", lambda *a, **k: (1, "g++: internal compiler error"))
    monkeypatch.setattr(llama, "_diagnose_missing_header", lambda *a, **k: None)
    with pytest.raises(llama.GgufError) as raised:
        llama._verify_cxx(compiler, Path("g++"), {})

    message = str(raised.value)
    assert "MinGW gcc (test)" in message
    assert "internal compiler error" in message
    assert "--no-gguf" in message


def test_the_probe_can_be_switched_off(llama, monkeypatch):
    """An escape hatch for the case where the probe is wrong and the build is not."""
    compiler = SimpleNamespace(kind="gcc", label="gcc", path=Path("g++"), env=None)
    monkeypatch.setattr(llama, "_capture", lambda *a, **k: (1, "boom"))
    monkeypatch.setenv(llama.SKIP_PROBE_ENV, "1")

    llama._verify_cxx(compiler, Path("g++"), {})


def test_a_header_past_max_path_is_named_as_the_reason(llama, monkeypatch, tmp_path):
    """The one compiler failure worth diagnosing rather than quoting.

    gcc joins a search directory to the header name without tidying the ``..``
    segments out, so the path it opens is far longer than the file's real one -- and
    gcc.exe has no long-path manifest, so at 261 characters it reports a header that
    is demonstrably there as missing. Quoting *No such file or directory* sends the
    reader off reinstalling a toolchain that was never broken, so the length is
    measured and named instead.

    The limit is moved rather than the path made long: creating a real 261-character
    path needs long paths enabled on the host, and what is under test is the
    comparison, not Windows.
    """
    header = "bits/c++config.h"
    search = tmp_path / "toolchain" / "include"
    (search / "bits").mkdir(parents=True)
    (search / header).write_text("", encoding="utf-8")
    joined = f"{search.as_posix()}/{header}"

    monkeypatch.setattr(llama, "_cxx_search_dirs", lambda *a, **k: [search.as_posix()])

    # Over the limit, and the file is really there: that pair is the diagnosis.
    monkeypatch.setattr(llama, "MAX_PATH", len(joined) - 5)
    detail = llama._diagnose_missing_header(Path("g++"), {}, header)
    assert detail is not None
    assert str(len(joined)) in detail and str(llama.MAX_PATH) in detail
    assert header in detail and joined in detail

    # Under it, this is an ordinary missing header and must not be explained away.
    monkeypatch.setattr(llama, "MAX_PATH", len(joined) + 50)
    assert llama._diagnose_missing_header(Path("g++"), {}, header) is None

    # Over the limit but genuinely absent is also not this failure.
    monkeypatch.setattr(llama, "MAX_PATH", len(joined) - 5)
    assert llama._diagnose_missing_header(Path("g++"), {}, "bits/not-here.h") is None


def test_the_include_search_list_is_read_the_way_gcc_prints_it(llama, monkeypatch):
    """Verbatim, ``..`` segments and all: the untidied path is the one that is long."""
    printed = (
        "ignoring nonexistent directory \"/nowhere\"\n"
        "#include \"...\" search starts here:\n"
        "#include <...> search starts here:\n"
        " C:/tc/bin/../lib/gcc/x86_64-w64-mingw32/15.2.0/../../../../include/c++/15.2.0\n"
        " C:/tc/include\n"
        "End of search list.\n"
        "GNU C++17 (GCC) version 15.2.0\n"
    )
    monkeypatch.setattr(llama, "_capture", lambda *a, **k: (0, printed))

    dirs = llama._cxx_search_dirs(Path("g++"), {})
    assert dirs == [
        "C:/tc/bin/../lib/gcc/x86_64-w64-mingw32/15.2.0/../../../../include/c++/15.2.0",
        "C:/tc/include",
    ]

    monkeypatch.setattr(llama, "_capture", lambda *a, **k: (1, "g++: not found"))
    assert llama._cxx_search_dirs(Path("g++"), {}) == []


def test_the_probe_source_needs_the_standard_library(llama):
    """A C probe proves nothing here: the failure is libstdc++, not the compiler."""
    assert "#include <string>" in llama._CXX_PROBE
    assert "#include <vector>" in llama._CXX_PROBE
    assert "std::" in llama._CXX_PROBE


# ── unit: what gets shipped beside the backend ────────────────────────────────

@pytest.fixture(scope="module")
def build_module() -> object:
    """``packaging/build.py``, imported rather than parsed.

    The `constants` fixture above reads this file without importing it, and says
    why: a build script is allowed to look at the machine while it loads. It does
    not -- the only statements outside a function are the ``sys.path`` insert that
    lets it find ``llamacpp``, and three ``compile()`` calls over its own probe
    sources -- and `archive_stem` and `_rmtree` are real functions whose behaviour
    cannot be asserted from a syntax tree. So this imports it, and the `constants`
    tests keep not needing to.
    """
    if not BUILD_PY.is_file():
        pytest.skip(f"no {BUILD_PY}")

    spec = importlib.util.spec_from_file_location("fox_packaging_build", BUILD_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    yield module
    sys.modules.pop(spec.name, None)


def test_the_archive_name_says_what_is_in_the_archive(build_module):
    """Two builds in one ``dist/`` used to be one file, and the second one won.

    ``fox-reader-windows-x64.zip`` named the platform and nothing else, so a CPU
    build and a CUDA build of the same version produced the same filename and
    silently overwrote each other -- and there is no way to tell afterwards which
    one is on disk.
    """
    stem = build_module.archive_stem
    assert stem("0.1.0", "cpu", None, "windows-x64") == "fox-reader-v0.1.0-cpu-windows-x64"
    assert stem("0.1.0", "cpu", "cpu", "windows-x64") == "fox-reader-v0.1.0-cpu-gguf-windows-x64"
    assert stem("0.1.0", "cu130", "cu130", "windows-x64") == "fox-reader-v0.1.0-cu130-gguf-windows-x64"
    # `--gguf-device` differing from `--device` is the one case the word "gguf"
    # cannot cover on its own, so the backend's own device is named too.
    assert stem("0.1.0", "cpu", "cu130", "windows-x64") == "fox-reader-v0.1.0-cpu-gguf-cu130-windows-x64"
    assert stem("0.1.0", "cpu", None, "linux-x64") == "fox-reader-v0.1.0-cpu-linux-x64"


def test_the_archive_name_is_a_filename_on_every_platform(build_module):
    """A local version is ``0.1.0rc1+local.1``, and ``+`` and ``!`` are not filenames."""
    name = build_module.archive_stem("0.1.0rc1+local.1", "cu130", "cu130", "linux-x64")
    assert name == "fox-reader-v0.1.0rc1-local.1-cu130-gguf-linux-x64"
    assert not (set(name) & set('<>:"/\\|?*+ ')), f"{name!r} is not a filename on Windows"


def test_a_read_only_venv_does_not_stop_a_clean(build_module, tmp_path):
    """``--clean`` died on this: uv hardlinks out of its cache, and hardlinks
    inherit the cached file's read-only bit, which Windows will not unlink."""
    tree = tmp_path / "venv" / "Lib" / "site-packages" / "cv2"
    tree.mkdir(parents=True)
    victim = tree / "cv2.pyd"
    victim.write_bytes(b"\x00")
    victim.chmod(stat.S_IREAD)

    build_module._rmtree(tmp_path / "venv")
    assert not (tmp_path / "venv").exists()


def test_a_held_file_is_reported_and_not_swallowed(build_module, tmp_path):
    """The other half. A lock that does not clear has to be an error naming the
    file -- ``ignore_errors`` would leave half a venv and build against it."""
    tree = tmp_path / "venv"
    tree.mkdir()
    victim = tree / "held.pyd"
    with victim.open("wb") as handle:
        handle.write(b"\x00")
        if sys.platform != "win32":
            pytest.skip("only Windows refuses to unlink an open file")
        with pytest.raises(build_module.BuildError) as caught:
            build_module._rmtree(tree, attempts=2)
    assert "held.pyd" in str(caught.value), "the error does not say which file is held"
    assert "--keep-venv" in str(caught.value), "the error offers no way forward"


def test_family_specific_architectures_survive_the_nvcc_filter(llama, monkeypatch):
    """``--list-gpu-arch`` prints ``compute_120``; the entry is ``120a-real``.

    Comparing them unstripped dropped every ``a`` target on every toolkit that
    has one, and dropped it silently: the build succeeds, the CUDA backend is
    there, and no Blackwell-consumer kernel is in it.
    """
    monkeypatch.setattr(llama, "_nvcc_known_arch", lambda nvcc: {"75", "80", "90", "120"})
    kept, dropped = llama.filter_cuda_architectures(
        ["75-real", "120a-real", "121a-real", "120-virtual"], Path("nvcc"))
    assert "120a-real" in kept, "120a-real was dropped even though compute_120 is known"
    assert "120-virtual" in kept
    assert dropped == ["121a-real"], f"expected only 121a to go, got {dropped}"


def test_the_arch_filter_still_drops_what_nvcc_cannot_build(llama, monkeypatch):
    """The filter has to keep filtering: 50 on CUDA 13 is a hard nvcc error."""
    monkeypatch.setattr(llama, "_nvcc_known_arch", lambda nvcc: {"75", "80"})
    kept, dropped = llama.filter_cuda_architectures(["50-real", "75-real"], Path("nvcc"))
    assert kept == ["75-real"] and dropped == ["50-real"]


def test_an_empty_filter_result_means_the_filter_is_wrong(llama, monkeypatch):
    """Nothing kept is not a build for no architectures -- it is a bad arch list."""
    monkeypatch.setattr(llama, "_nvcc_known_arch", lambda nvcc: {"nonesuch"})
    kept, dropped = llama.filter_cuda_architectures(["120a-real"], Path("nvcc"))
    assert kept == ["120a-real"] and dropped == []


def test_an_nvcc_that_cannot_be_asked_filters_nothing(llama, monkeypatch):
    """No answer is not the same as an empty answer."""
    monkeypatch.setattr(llama, "_nvcc_known_arch", lambda nvcc: set())
    assert llama.filter_cuda_architectures(["50-real"], Path("nvcc")) == (["50-real"], [])
    assert llama.filter_cuda_architectures(["50-real"], None) == (["50-real"], [])


def test_the_gnu_ceiling_is_read_out_of_the_toolkit(llama, tmp_path):
    """``#if __GNUC__ > 15``: the last GCC accepted, not the first refused.

    A table here would be wrong about the next toolkit, and this is the same file
    that raises ``#error -- unsupported GNU version!``.
    """
    header = tmp_path / "include" / "crt" / "host_config.h"
    header.parent.mkdir(parents=True)
    header.write_text("#if __GNUC__ > 15\n#error -- unsupported GNU version!\n#endif\n")
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.write_text("")
    assert llama.nvcc_gnu_limit(nvcc) == 15
    # A toolkit with no such header, which is what an unreadable one has to be:
    # None, so the caller leaves the host compiler alone rather than guessing.
    assert llama.nvcc_gnu_limit(tmp_path / "elsewhere" / "bin" / "nvcc") is None


def test_a_cuda_library_is_matched_by_its_own_major(llama):
    """Shipping CUDA 13's cuBLAS with a cu118 build is a load failure at runtime,
    and on Windows the message names nothing at all."""
    assert llama._cuda_major("cublas64_13.dll") == "13"
    assert llama._cuda_major("libcublas.so.12") == "12"
    assert llama._cuda_major("libcudart.so.13.0.48") == "13"
    # Windows' cudart carries the full version without dots: cudart64_110.dll is
    # CUDA 11.x, and torch's cu118 wheels ship exactly that name. Reading it as
    # major "110" would leave the 11.x cudart out of every cu118 distribution.
    assert llama._cuda_major("cudart64_110.dll") == "11"
    assert llama._cuda_major("cudart64_13.dll") == "13"
    # The unversioned development symlink, which is not a file to ship.
    assert llama._cuda_major("libcublas.so") is None
    assert llama._cuda_major("cudart64.dll") is None


def test_the_elf_reader_finds_what_a_library_links(llama, tmp_path):
    """The Linux half of `_pe_imports`. There is no ``add_dll_directory`` there:
    an ELF finds its dependencies through the RUNPATH recorded at link time, so
    what a library needs has to be read out of the library."""
    needed = ["libcudart.so.13", "libcublas.so.13", "libc.so.6"]
    path = tmp_path / "libggml-cuda.so"
    path.write_bytes(_elf64(needed))
    assert llama._elf_needed(path) == needed


def test_the_elf_reader_is_silent_about_what_it_cannot_read(llama, tmp_path):
    """It runs over every file in a staged tree, so a text file, a truncated
    library or a 32-bit one is an empty answer and not a failed build."""
    for name, blob in (("notelf.txt", b"hello"),
                       ("truncated.so", b"\x7fELF\x02\x01\x01" + b"\x00" * 20),
                       ("empty.so", b"")):
        target = tmp_path / name
        target.write_bytes(blob)
        assert llama._elf_needed(target) == [], f"{name} should read as nothing"
    assert llama._elf_needed(tmp_path / "absent.so") == []


def test_the_driver_is_never_shipped(llama):
    """``libcuda.so.1``/``nvcuda.dll`` come from the installed driver and are
    version-locked to the kernel module. Shipping one is how a working machine
    stops working -- and ``libcuda`` starts with ``libcu``, so the DT_NEEDED
    closure would have taken it."""
    assert "libcuda.so" in llama._CUDA_DRIVER
    assert "nvcuda.dll" in llama._CUDA_DRIVER
    for name in llama._CUDA_DRIVER:
        assert not name.startswith(tuple(llama._CUDA_RUNTIME_WINDOWS) + tuple(llama._CUDA_RUNTIME_LINUX)), (
            f"{name} is both a driver and a runtime prefix, so one of the two lists is wrong"
        )


def test_the_backend_library_is_named_for_every_gpu_target(llama):
    """`verify` accepts "no driver here" only when the backend was really built,
    which needs a filename per target -- and a target missing from this map would
    make that check pass by default."""
    for kind in ("cuda", "rocm", "metal"):
        assert kind in llama._BACKEND_LIBRARY, f"{kind} builds have no library name to look for"
    assert "cpu" not in llama._BACKEND_LIBRARY, "a CPU build has no backend library"


def test_the_no_driver_excuse_needs_the_library_to_be_there(llama, monkeypatch, tmp_path):
    """The excuse is load-bearing: without it a CUDA build could only ever be made
    on a machine with an NVIDIA GPU. So it is fenced -- a CUDA build that came out
    CPU-only fails here rather than shipping."""
    build = SimpleNamespace(
        target=llama.Target(kind="cuda", device="cu130", cuda=(13, 0)),
        openblas=None, notes=[],
    )
    monkeypatch.setattr(llama, "_driver_present", lambda: False)
    monkeypatch.setattr(llama, "_missing_runtime", lambda *a, **k: [])
    monkeypatch.setattr(llama, "_backend_library", lambda *a, **k: None)
    monkeypatch.setattr(llama.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=b'{"error": "DLL load failed while importing llama_cpp"}', stderr=b""))

    with pytest.raises(llama.GgufError) as caught:
        llama.verify(tmp_path / "python", build)
    assert "ggml-cuda" in str(caught.value), (
        "the error does not say the backend library is the thing that is missing"
    )
    assert not build.notes, "a failed verify must not leave an 'unverified' note behind"


def test_the_no_driver_excuse_needs_the_runtime_to_be_complete(llama, monkeypatch, tmp_path):
    """The other fence. On a driverless machine a missing cudart, a cuBLAS from the
    wrong CUDA and an absent driver are one indistinguishable message."""
    build = SimpleNamespace(
        target=llama.Target(kind="cuda", device="cu130", cuda=(13, 0)),
        openblas=None, notes=[],
    )
    monkeypatch.setattr(llama, "_driver_present", lambda: False)
    monkeypatch.setattr(llama, "_missing_runtime", lambda *a, **k: ["cudart64_13.dll"])
    monkeypatch.setattr(llama, "_backend_library", lambda *a, **k: tmp_path / "ggml-cuda.dll")
    monkeypatch.setattr(llama.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=b'{"error": "DLL load failed while importing llama_cpp"}', stderr=b""))

    with pytest.raises(llama.GgufError) as caught:
        llama.verify(tmp_path / "python", build)
    assert "cudart64_13.dll" in str(caught.value)
    assert not build.notes


def test_a_driverless_machine_can_still_build_cuda(llama, monkeypatch, tmp_path, capsys):
    """And when both fences hold, the build is allowed through with a note.

    This is the case the whole arrangement is for: the backend compiled, everything
    it links is beside it, and this host has no NVIDIA driver to load it with. No
    arrangement of the build would change that.
    """
    build = SimpleNamespace(
        target=llama.Target(kind="cuda", device="cu130", cuda=(13, 0)),
        openblas=None, notes=[],
    )
    monkeypatch.setattr(llama, "_driver_present", lambda: False)
    monkeypatch.setattr(llama, "_missing_runtime", lambda *a, **k: [])
    monkeypatch.setattr(llama, "_backend_library", lambda *a, **k: tmp_path / "ggml-cuda.dll")
    monkeypatch.setattr(llama.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=b'{"error": "DLL load failed while importing llama_cpp"}', stderr=b""))

    llama.verify(tmp_path / "python", build)
    assert any("unverified" in note for note in build.notes), (
        "the build was accepted without recording that it was never imported"
    )
    assert "ggml-cuda.dll" in capsys.readouterr().out, (
        "the warning does not say what was checked in place of the import"
    )


def test_a_present_driver_makes_a_load_failure_a_real_failure(llama, monkeypatch, tmp_path):
    """No excuse at all when the driver is there -- then the library is the problem."""
    build = SimpleNamespace(
        target=llama.Target(kind="cuda", device="cu130", cuda=(13, 0)),
        openblas=None, notes=[],
    )
    monkeypatch.setattr(llama, "_driver_present", lambda: True)
    monkeypatch.setattr(llama, "_missing_runtime", lambda *a, **k: [])
    monkeypatch.setattr(llama, "_backend_library", lambda *a, **k: tmp_path / "ggml-cuda.dll")
    monkeypatch.setattr(llama.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=b'{"error": "DLL load failed while importing llama_cpp"}', stderr=b""))

    with pytest.raises(llama.GgufError):
        llama.verify(tmp_path / "python", build)
    assert not build.notes


def _lib_tree(tmp_path, *names):
    """A fake `llama_cpp/lib/` holding empty files with real backend names.

    The import-table readers are stubbed in these tests, so the files only have
    to be there to be found -- what matters is the names, which is also what the
    detector matches on.
    """
    lib = tmp_path / "llama_cpp" / "lib"
    lib.mkdir(parents=True)
    for name in names:
        (lib / name).write_bytes(b"")
    return lib


def test_a_load_linked_backend_reports_static(llama, monkeypatch, tmp_path):
    """`ggml.dll` naming `ggml-cuda.dll` in its imports is the prebuilt-wheel
    shape: without a driver the package cannot import at all, CPU included."""
    lib = _lib_tree(tmp_path, "ggml.dll", "ggml-cuda.dll", "cublas64_11.dll")
    monkeypatch.setattr(llama, "_package_lib_dir", lambda *a, **k: lib)
    monkeypatch.setattr(llama, "_pe_import_names",
                        lambda p: ["ggml-cuda.dll", "KERNEL32.dll"]
                        if p.name == "ggml.dll" else [])
    build = SimpleNamespace(
        target=llama.Target(kind="cuda", device="cu118", cuda=(11, 8)))

    assert llama._backend_static_linkage(Path("python"), build, None) == "static"


def test_a_runtime_loaded_backend_reports_dynamic(llama, monkeypatch, tmp_path):
    """The backend library present but named by nothing at load time is the
    source-build shape: a missing driver degrades to CPU instead."""
    lib = _lib_tree(tmp_path, "ggml.dll", "ggml-cuda.dll")
    monkeypatch.setattr(llama, "_package_lib_dir", lambda *a, **k: lib)
    monkeypatch.setattr(llama, "_pe_import_names", lambda p: ["KERNEL32.dll"])
    monkeypatch.setattr(llama, "_elf_needed", lambda p: [])
    build = SimpleNamespace(
        target=llama.Target(kind="cuda", device="cu130", cuda=(13, 0)))

    assert llama._backend_static_linkage(Path("python"), build, None) == "dynamic"


def test_linkage_is_unknown_without_a_backend_file(llama, monkeypatch, tmp_path):
    """No backend library, no statement: a CPU-only misbuild fails in `verify`
    on the missing file, not here."""
    lib = _lib_tree(tmp_path, "ggml.dll")
    monkeypatch.setattr(llama, "_package_lib_dir", lambda *a, **k: lib)
    build = SimpleNamespace(
        target=llama.Target(kind="cuda", device="cu130", cuda=(13, 0)))

    assert llama._backend_static_linkage(Path("python"), build, None) == "unknown"


def test_linkage_is_unknown_off_gpu(llama, monkeypatch, tmp_path):
    """A CPU target has no backend library to be linked either way."""
    lib = _lib_tree(tmp_path, "ggml-cpu.dll")
    monkeypatch.setattr(llama, "_package_lib_dir", lambda *a, **k: lib)
    build = SimpleNamespace(target=llama.Target(kind="cpu", device="cpu"))

    assert llama._backend_static_linkage(Path("python"), build, None) == "unknown"


def test_static_linkage_says_cpu_needs_a_driver_too(llama, monkeypatch, tmp_path, capsys):
    """The driverless allowance has two shapes now. A dynamically loaded backend
    is merely unverified; a statically linked one cannot run here at all, CPU
    included -- and shipping it anyway (for a machine with a driver) must say
    exactly that rather than sounding like the graceful case."""
    build = SimpleNamespace(
        target=llama.Target(kind="cuda", device="cu118", cuda=(11, 8)),
        openblas=None, notes=[],
    )
    monkeypatch.setattr(llama, "_driver_present", lambda: False)
    monkeypatch.setattr(llama, "_missing_runtime", lambda *a, **k: [])
    monkeypatch.setattr(llama, "_backend_library", lambda *a, **k: tmp_path / "ggml-cuda.dll")
    monkeypatch.setattr(llama, "_backend_static_linkage", lambda *a, **k: "static")
    monkeypatch.setattr(llama.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=b'{"error": "DLL load failed while importing llama_cpp"}', stderr=b""))

    llama.verify(tmp_path / "python", build)  # allowed: a driver machine runs this fine

    out = capsys.readouterr().out
    assert "statically" in out, "the warning does not name the linkage"
    assert "CPU" in out, "the warning does not say the CPU fallback is affected"
    assert any("statically linked" in note for note in build.notes), (
        "the build was accepted without recording that CPU needs a driver too"
    )


def test_nvcc_gets_a_gcc_it_accepts_or_is_told_to_stop_checking(llama, monkeypatch, tmp_path):
    """Linux moves GCC every six months and CUDA moves its ceiling once a year, so
    a default compiler nvcc refuses is the ordinary case on a current distribution.

    The versioned compiler is named for C and C++ as well as for CUDA. Naming it
    for nvcc alone puts two libstdc++ ABIs in one library, which is a crash while
    a model loads rather than an error at configure time.
    """
    monkeypatch.setattr(llama, "nvcc_gnu_limit", lambda nvcc: 15)
    monkeypatch.setattr(llama, "_gcc_major", lambda compiler: 16)
    monkeypatch.setattr(llama.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name.endswith("-15") else None)

    notes: list[str] = []
    env, args = llama._posix_cuda_host_compiler(tmp_path / "nvcc", notes)
    assert env == {}, "a compiler was found, so nothing needed to be waved through"
    assert sorted(args) == [
        "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15",
        "-DCMAKE_CXX_COMPILER=/usr/bin/g++-15",
        "-DCMAKE_C_COMPILER=/usr/bin/gcc-15",
    ]
    assert any("gcc-15" in note for note in notes), "the substitution was not reported"


def test_an_acceptable_gcc_is_left_alone(llama, monkeypatch, tmp_path):
    monkeypatch.setattr(llama, "nvcc_gnu_limit", lambda nvcc: 15)
    monkeypatch.setattr(llama, "_gcc_major", lambda compiler: 13)
    notes: list[str] = []
    assert llama._posix_cuda_host_compiler(tmp_path / "nvcc", notes) == ({}, [])
    assert notes, "the check ran and said nothing about it"


def test_no_usable_gcc_falls_back_rather_than_failing(llama, monkeypatch, tmp_path):
    """``-allow-unsupported-compiler`` is what upstream llama.cpp tells people to
    use, and it usually works. Through ``CUDAFLAGS`` rather than
    ``CMAKE_CUDA_FLAGS``, which CMake overwrites at ``enable_language(CUDA)``."""
    monkeypatch.setattr(llama, "nvcc_gnu_limit", lambda nvcc: 15)
    monkeypatch.setattr(llama, "_gcc_major", lambda compiler: 16)
    monkeypatch.setattr(llama.shutil, "which", lambda name: None)
    monkeypatch.delenv("CUDAFLAGS", raising=False)

    notes: list[str] = []
    env, args = llama._posix_cuda_host_compiler(tmp_path / "nvcc", notes)
    assert env == {"CUDAFLAGS": "-allow-unsupported-compiler"}
    assert args == [], "no compiler was found, so none may be named"
    assert any("install g++-15" in note for note in notes), "the fix is not suggested"


def test_a_clang_host_compiler_is_not_measured_against_the_gcc_ceiling(llama):
    """clang 21 and GCC 21 are years apart, so the comparison means nothing."""
    assert llama._gcc_major("this-compiler-does-not-exist") is None


def test_only_cuda_needs_a_host_compiler_chosen_for_it(llama, tmp_path):
    """A CPU or Metal build on POSIX needs nothing told to it at all."""
    for kind in ("cpu", "metal"):
        target = llama.Target(kind=kind, device=kind)
        assert llama._posix_host_toolchain(target, tmp_path / "nvcc") == ({}, [], [], [])
    cuda = llama.Target(kind="cuda", device="cu130", cuda=(13, 0))
    assert llama._posix_host_toolchain(cuda, None) == ({}, [], [], [])


def _elf64(needed: list[str]) -> bytes:
    """A minimal ELF64 little-endian shared object with a DT_NEEDED list.

    Enough of one for `_elf_needed`: an identification header, one PT_LOAD
    covering the whole file at vaddr 0 -- which makes every virtual address equal
    to its file offset -- and a PT_DYNAMIC pointing at a dynamic section whose
    DT_STRTAB addresses the string table at the end. Nothing else in it is
    meaningful, and it is deliberately not loadable.
    """
    strtab = b"\x00" + b"".join(name.encode() + b"\x00" for name in needed)
    offsets, cursor = [], 1
    for name in needed:
        offsets.append(cursor)
        cursor += len(name) + 1

    ehsize, phentsize, phnum = 64, 56, 2
    phoff = ehsize
    dyn_off = phoff + phentsize * phnum
    dyn_size = (len(needed) + 2) * 16          # NEEDED per name, then STRTAB, then NULL
    strtab_off = dyn_off + dyn_size

    dynamic = b"".join(struct.pack("<qQ", 1, off) for off in offsets)
    dynamic += struct.pack("<qQ", 5, strtab_off) + struct.pack("<qQ", 0, 0)
    assert len(dynamic) == dyn_size

    total = strtab_off + len(strtab)
    ident = b"\x7fELF" + bytes([2, 1, 1, 0, 0]) + b"\x00" * 7
    header = ident + struct.pack(
        "<HHIQQQIHHHHHH",
        3, 62, 1,          # e_type ET_DYN, e_machine x86-64, e_version
        0, phoff, 0, 0,    # e_entry, e_phoff, e_shoff, e_flags
        ehsize, phentsize, phnum, 64, 0, 0,
    )
    load = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, total, total, 0x1000)
    dyn = struct.pack("<IIQQQQQQ", 2, 6, dyn_off, dyn_off, dyn_off,
                      dyn_size, dyn_size, 8)
    body = header + load + dyn + dynamic + strtab
    assert len(body) == total, f"built {len(body)} bytes, said {total}"
    return body


def test_a_linux_tree_reports_what_it_expects_from_the_machine(build_module, tmp_path, capsys):
    """glibc and the driver are the system's, and copying either is wrong.

    So the Linux side reports rather than bundles -- and the report is the point:
    a SONAME that is neither in the tree nor part of a base system is one the
    build picked up from a development machine, and a plain one fails with
    ``error while loading shared libraries`` before any of our code runs.
    """
    lib = tmp_path / "libggml-cuda.so"
    lib.write_bytes(_elf64(["libc.so.6", "libcuda.so.1", "libcudart.so.13",
                            "libsomeones-dev-box.so.4"]))
    (tmp_path / "libcudart.so.13").write_bytes(b"\x00")

    build_module._report_elf_dependencies(tmp_path)
    out = capsys.readouterr().out

    assert "libsomeones-dev-box.so.4" in out, "the one library that will be missing is not named"
    assert "libggml-cuda.so" in out, "the report does not say what needs it"
    for expected in ("libc.so.6", "libcuda.so.1", "libcudart.so.13"):
        assert f"! {expected}" not in out, (
            f"{expected} was flagged; it is either the system's or already in the tree"
        )


def test_the_driver_and_the_c_library_are_never_bundled_on_linux(build_module):
    """`libc.so.6` cannot be relocated -- it and the loader are one unit -- and
    `libcuda.so.1` is version-locked to the installed kernel module."""
    for name in ("libc.so.6", "ld-linux-x86-64.so.2", "libcuda.so.1", "libstdc++.so.6"):
        assert name.startswith(build_module._ELF_SYSTEM), f"{name} is not treated as the system's"


# ── OpenBLAS is automatic; the toolkit is chosen, not inherited ──────────────

def test_no_blas_wins_over_openblas(build_module):
    """Both spellings mean "no BLAS", and `--no-blas` is the one that wins."""
    assert build_module._effective_openblas_mode(
        SimpleNamespace(no_blas=True, openblas="auto")) == "off"
    assert build_module._effective_openblas_mode(
        SimpleNamespace(no_blas=True, openblas="download")) == "off"
    assert build_module._effective_openblas_mode(
        SimpleNamespace(no_blas=False, openblas="auto")) == "auto"
    assert build_module._effective_openblas_mode(
        SimpleNamespace(no_blas=False, openblas="off")) == "off"
    assert build_module._effective_openblas_mode(SimpleNamespace()) == "auto"


def test_resolve_openblas_off_touches_nothing(llama, monkeypatch):
    """`off` short-circuits before any scan or download."""
    monkeypatch.setattr(llama, "find_openblas",
                        lambda: pytest.fail("must not scan the machine"))
    monkeypatch.setattr(llama, "download_openblas",
                        lambda: pytest.fail("must not download"))
    assert llama.resolve_openblas("off") is None


def test_openblas_resolution_is_non_interactive(llama):
    """The build runs unattended (CI, scheduled builds), so there is no prompt
    left to answer: `auto` fetches without asking and `off` skips."""
    assert not hasattr(llama, "_ask"), (
        "resolve_openblas must not prompt; auto fetches instead of asking"
    )


def test_cuda_environment_names_its_own_toolkit(llama, tmp_path):
    """Machines accumulate half-removed toolkits -- an uninstall that keeps
    `include` and `lib` while deleting `bin`, with CUDA_PATH still pointing at
    it. The environment handed to the compile must name the nvcc that was found,
    so that a stale inherited value loses at merge time in `install`."""
    nvcc = tmp_path / "v13.0" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    (tmp_path / "v13.0" / "include").mkdir(parents=True)
    target = llama.Target(kind="cuda", device="cu130", cuda=(13, 0))

    env = llama._describe_environment(target, nvcc, None)

    assert env["CUDACXX"] == str(nvcc)
    assert env["CUDA_PATH"] == str(tmp_path / "v13.0")


def test_install_trees_are_not_packages(build_module, tmp_path):
    """llama-cpp-python installs CMake's `bin/` of DLLs, `include/` of headers
    and `lib/` of import libraries next to the package in site-packages. None
    of those is a module, and naming one in a Nuitka bytecode flag is at best
    noise -- while a real package and a namespace package must survive."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ggml-cuda.dll").write_bytes(b"")
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "ggml.h").write_bytes(b"")
    (tmp_path / "lib" / "cmake").mkdir(parents=True)
    (tmp_path / "lib" / "ggml-base.lib").write_bytes(b"")
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    namespace = tmp_path / "google" / "auth"
    namespace.mkdir(parents=True)
    (namespace / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "lonely.py").write_text("", encoding="utf-8")

    names = build_module._third_party_top_levels(Path("python"), [str(tmp_path)])

    for real in ("torch", "google", "lonely"):
        assert real in names, f"{real} is a package and must be shipped as bytecode"
    for data in ("bin", "include", "lib"):
        assert data not in names, f"{data}/ holds no Python and must not reach Nuitka"


# ── Nuitka plugin-trigger / bytecode-cache workaround ─────────────────────────
# Nuitka 4.2.1 dies on a warm bytecode cache inside _createTriggerLoadedModule:
# a plugin trigger for a dotted submodule (multiprocessing plugin's
# `anyio.to_process-postLoad`) matches our `anyio.*:bytecode` flag, comes back
# from the cache as UncompiledPythonModule, and has no getCompilationMode.
# First build populates the cache and passes; every rebuild after it crashes.
# These pin the detector and the flag wiring without running a build.

_TRIGGER_BAD = """\
        if trigger_module.getCompilationMode() == "bytecode":
            trigger_module.setSourceCode(code)
"""

_TRIGGER_GOOD = """\
        if (not trigger_module.isUncompiledPythonModule()
                and trigger_module.getCompilationMode() == "bytecode"):
            trigger_module.setSourceCode(code)
"""

_FAKE_BAD = """\
                if fake_module.getCompilationMode() == "bytecode":
                    fake_module.setSourceCode(fake_module_description.source_code)
"""

# Far enough apart that the ±8-line guard windows cannot see each other, as in
# the real file where the two sites are ~160 lines apart.
_FAR_APART = "\n".join(f"    # filler {index}" for index in range(30)) + "\n"


def _nuitka_stub(build_module, options=("--disable-cache", "--output-dir")):
    """A real ``Nuitka`` helper object without running its ``--help`` probes."""
    stub = build_module.Nuitka.__new__(build_module.Nuitka)
    stub.version = "4.2.1-test"
    stub.options = set(options)
    stub.plugins = set()
    stub.always_on = set()
    stub.skipped = []
    return stub


def test_trigger_handling_sees_the_known_bad_shape(build_module):
    assert build_module._classify_trigger_handling(_TRIGGER_BAD + _FAKE_BAD) == "unguarded"


def test_trigger_handling_accepts_a_guarded_nuitka(build_module):
    guarded_fake = _FAKE_BAD.replace(
        "if fake_module.getCompilationMode()",
        'if hasattr(fake_module, "getCompilationMode") and fake_module.getCompilationMode()',
    )
    source = _TRIGGER_GOOD + _FAR_APART + guarded_fake
    assert build_module._classify_trigger_handling(source) == "guarded"


def test_trigger_handling_one_bad_site_poisons_the_verdict(build_module):
    source = _TRIGGER_GOOD + _FAR_APART + _FAKE_BAD
    assert build_module._classify_trigger_handling(source) == "unguarded"


@pytest.mark.parametrize("source", ["", "class Nuitka: pass\n"])
def test_trigger_handling_unknown_when_it_cannot_tell(build_module, source):
    assert build_module._classify_trigger_handling(source) == "unknown"


def test_workaround_adds_the_flag_on_vulnerable_nuitka(build_module, monkeypatch, tmp_path):
    monkeypatch.setattr(
        build_module, "_installed_nuitka_plugins_source",
        lambda _python: _TRIGGER_BAD + _FAKE_BAD,
    )
    stub = _nuitka_stub(build_module)
    assert build_module._trigger_cache_workaround(stub, tmp_path / "python", []) == [
        "--disable-cache=bytecode"
    ]


def test_workaround_stays_quiet_on_a_fixed_nuitka(build_module, monkeypatch, tmp_path):
    guarded_fake = _FAKE_BAD.replace(
        "if fake_module.getCompilationMode()",
        'if hasattr(fake_module, "getCompilationMode") and fake_module.getCompilationMode()',
    )
    monkeypatch.setattr(
        build_module, "_installed_nuitka_plugins_source",
        lambda _python: _TRIGGER_GOOD + guarded_fake,
    )
    stub = _nuitka_stub(build_module)
    assert build_module._trigger_cache_workaround(stub, tmp_path / "python", []) == []
    assert stub.skipped == []


def test_workaround_yields_to_the_users_own_flag(build_module, monkeypatch, tmp_path):
    def explode(_python):
        raise AssertionError("the installed Nuitka must not even be inspected")

    monkeypatch.setattr(build_module, "_installed_nuitka_plugins_source", explode)
    stub = _nuitka_stub(build_module)
    assert build_module._trigger_cache_workaround(
        stub, tmp_path / "python", ["--disable-cache=all"]) == []


def test_workaround_warns_when_the_flag_does_not_exist(build_module, monkeypatch, tmp_path):
    """An ancient Nuitka without --disable-cache cannot be steered; say so."""
    monkeypatch.setattr(
        build_module, "_installed_nuitka_plugins_source",
        lambda _python: _TRIGGER_BAD + _FAKE_BAD,
    )
    stub = _nuitka_stub(build_module, options=("--output-dir",))
    assert build_module._trigger_cache_workaround(stub, tmp_path / "python", []) == []
    assert any("--disable-cache" in item for item in stub.skipped)


def test_unreadable_nuitka_source_is_not_an_error(build_module, tmp_path):
    assert build_module._installed_nuitka_plugins_source(
        tmp_path / "definitely-not-a-python") == ""


def test_run_nuitka_wires_the_trigger_workaround(build_source):
    """The call itself, read from the syntax tree: flags mean nothing if the
    compile entry point never asks for them."""
    for node in ast.walk(build_source):
        if isinstance(node, ast.FunctionDef) and node.name == "run_nuitka":
            wanted = {
                getattr(child.func, "id", None) or getattr(child.func, "attr", None)
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, (ast.Name, ast.Attribute))
            }
            assert "_trigger_cache_workaround" in wanted, (
                "run_nuitka no longer calls _trigger_cache_workaround -- "
                "a warm-cache rebuild will crash in Nuitka's trigger handling"
            )
            return
    raise AssertionError("run_nuitka is gone from packaging/build.py")


# ── duplicated-library pruning ──────────────────────────────────────────────
# Linux ships every shared library twice (flat at the top level for Nuitka's
# dependency scan, and under its package where the package's own RUNPATH finds
# it), plus version triplets as three full copies, plus torch's test suite and
# headers. These pin the pure helpers and the file handling without running a
# build; symlink creation itself is faked or skipped on Windows, where the
# privilege to make one is absent.

def test_lib_version_base_groups_aliases(build_module):
    assert build_module._lib_version_base("libllama.so.0.1.0") == "libllama.so"
    assert build_module._lib_version_base("libllama.so.0") == "libllama.so"
    assert build_module._lib_version_base("libllama.so") == "libllama.so"
    assert build_module._lib_version_base("libmtmd.so.SOVERSION") == "libmtmd.so"
    assert build_module._lib_version_base("libtorch_cpu.so") == "libtorch_cpu.so"
    assert build_module._lib_version_base("torch_cpu.dll") is None
    assert build_module._lib_version_base("version.json") is None


def test_torch_bin_prune_keeps_the_shm_manager(build_module):
    names = ["FileStoreTest", "test_api", "protoc", "protoc-3.13.0.0",
             "script_module_v4.ptl", "upgrader_models", "torch_shm_manager"]
    prunable = build_module._torch_bin_prunable(names)
    assert "torch_shm_manager" not in prunable
    assert sorted(prunable) == sorted(set(names) - {"torch_shm_manager"})


@pytest.mark.parametrize("name, expected", [
    ("libtorchbind_test.so", True),
    ("libjitbackend_test.so", True),
    ("test_shim", True),
    ("libtorch_cpu.so", False),
    ("libtorch.so", False),
    # Contains "test" but neither as a prefix nor after an underscore.
    ("liblatest.so", False),
    ("contest.so", False),
])
def test_torch_lib_prune_matches_only_test_scaffolding(build_module, name, expected):
    assert build_module._torch_lib_prunable(name) is expected


def test_shadow_twin_needs_exactly_one(build_module, tmp_path):
    twin = tmp_path / "torch" / "lib" / "libfoo.so"
    assert build_module._pick_shadow_twin("libfoo.so", [twin]) == twin
    assert build_module._pick_shadow_twin("libfoo.so", []) is None
    assert build_module._pick_shadow_twin("libfoo.so", [twin, tmp_path / "other"]) is None


def _prune_tree(tmp_path):
    """A miniature `dist/`: torch scaffolding plus one shadowed library."""
    dist = tmp_path / "dist"
    (dist / "torch" / "test").mkdir(parents=True)
    (dist / "torch" / "test" / "basic").write_bytes(b"x")
    (dist / "torch" / "include").mkdir(parents=True)
    (dist / "torch" / "include" / "h.h").write_bytes(b"x")
    bin_dir = dist / "torch" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "test_api").write_bytes(b"x")
    (bin_dir / "torch_shm_manager").write_bytes(b"x")
    (dist / "torch" / "lib").mkdir(parents=True)
    (dist / "torch" / "lib" / "libfoo.so").write_bytes(b"nested")
    (dist / "libfoo.so").write_bytes(b"top")
    (dist / "libunique.so").write_bytes(b"only-top")
    return dist


def test_prune_removes_torch_scaffolding_and_keeps_shadows_without_links(
        build_module, tmp_path):
    """Deletions run everywhere; without `allow_links` shadows stay files."""
    dist = _prune_tree(tmp_path)
    build_module.prune_duplicated_libs(dist, allow_links=False)
    assert not (dist / "torch" / "test").exists()
    assert not (dist / "torch" / "include").exists()
    assert not (dist / "torch" / "bin" / "test_api").exists()
    assert (dist / "torch" / "bin" / "torch_shm_manager").is_file()
    assert (dist / "libfoo.so").read_bytes() == b"top"
    assert (dist / "libunique.so").read_bytes() == b"only-top"


def test_prune_links_shadows_when_allowed(build_module, tmp_path, monkeypatch):
    """Linking, with `os.symlink` faked: the top file becomes the twin's path."""
    dist = _prune_tree(tmp_path)

    def fake_symlink(target, link):
        Path(link).write_text(f"link->{target}", encoding="utf-8")

    monkeypatch.setattr(build_module.os, "symlink", fake_symlink)
    build_module.prune_duplicated_libs(dist, allow_links=True)

    expected = "link->" + build_module.os.path.relpath(
        dist / "torch" / "lib" / "libfoo.so", dist)
    assert (dist / "libfoo.so").read_text(encoding="utf-8") == expected
    # Unique files are never touched, whatever the flag says.
    assert (dist / "libunique.so").read_bytes() == b"only-top"


def test_prune_skip_env_keeps_everything(build_module, tmp_path, monkeypatch):
    monkeypatch.setenv(build_module._PRUNE_SKIP_ENV, "1")
    dist = _prune_tree(tmp_path)
    build_module.prune_duplicated_libs(dist, allow_links=False)
    assert (dist / "torch" / "test" / "basic").is_file()
    assert (dist / "libfoo.so").read_bytes() == b"top"


def test_link_failure_keeps_the_original(build_module, tmp_path, monkeypatch):
    """A refused symlink is a kept file, not a deleted one."""
    link = tmp_path / "a.so"
    link.write_bytes(b"orig")
    target = tmp_path / "b.so"
    target.write_bytes(b"real")

    def refuse(_target, _link):
        raise OSError("privilege")

    monkeypatch.setattr(build_module.os, "symlink", refuse)
    assert build_module._link_instead(link, target) is False
    assert link.read_bytes() == b"orig"


@pytest.mark.skipif(sys.platform == "win32", reason="needs real symlinks")
def test_link_instead_replaces_with_a_relative_link(build_module, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    real = sub / "real.so"
    real.write_bytes(b"bytes")
    link = tmp_path / "link.so"
    link.write_bytes(b"old")
    assert build_module._link_instead(link, real) is True
    assert link.is_symlink()
    assert build_module.os.readlink(link) == "sub/real.so"
    assert link.read_bytes() == b"bytes"


def test_main_prunes_after_every_library_is_staged(build_source):
    """The call itself, read from the syntax tree like the trigger workaround."""
    for node in ast.walk(build_source):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            wanted = {
                getattr(child.func, "id", None) or getattr(child.func, "attr", None)
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, (ast.Name, ast.Attribute))
            }
            assert "prune_duplicated_libs" in wanted, (
                "main no longer calls prune_duplicated_libs -- "
                "Linux ships every .so twice again"
            )
            return
    raise AssertionError("main is gone from packaging/build.py")


# ── the staged build, when there is one ───────────────────────────────────────

def test_staged_build_ships_the_metadata_it_promised(constants):
    """Read-only: what is in ``bin/`` against what ``build.py`` said would be.

    No environment variable on this one. It touches nothing, takes no time, and it
    is the fastest way to tell whether the tree on disk was built before or after
    the metadata fix.
    """
    bin_dir = suites.STAGE_DIR / "bin"
    if not bin_dir.is_dir():
        pytest.skip(f"no staged build at {suites.STAGE_DIR}; run packaging/build.py first")

    on_disk = {
        _normalize(path.name.rsplit(".dist-info", 1)[0].rsplit("-", 1)[0])
        for path in bin_dir.glob("*.dist-info")
    }
    assert on_disk, (
        f"{bin_dir} carries no .dist-info directories at all. Nuitka writes none of "
        "its own,\n  so this tree was staged by a build that predates bundle_metadata "
        "-- every runtime\n  metadata lookup in it raises PackageNotFoundError."
    )

    promised = {_normalize(name) for name in _require(constants, "DISTRIBUTION_METADATA")}
    missing = sorted(promised - on_disk)
    assert not missing, f"the staged build is missing metadata for: {', '.join(missing)}"

    optional = {_normalize(name) for name in _require(constants, "OPTIONAL_DISTRIBUTION_METADATA")}
    extra = sorted(on_disk - promised - optional)
    assert not extra, (
        f"the staged build ships uncurated metadata for: {', '.join(extra)}"
    )


# ── the domain's own test files ───────────────────────────────────────────────

def test_accounting():
    """Every test file under ``tests/`` is claimed by exactly one domain."""
    suites.check_accounting()


def test_modules_collect():
    """Skipped: a build has no pytest module. The static checks above are its unit tests."""
    suites.check_modules_collect(DOMAIN)


def test_helpers_exist():
    """The files declared as helpers are there, since a check reaches for them by path."""
    missing = [
        helper.path for helper in DOMAIN.helpers if not (Path(suites.TESTS_DIR) / helper.path).is_file()
    ]
    assert not missing, f"declared helpers are missing: {', '.join(missing)}"


@pytest.mark.parametrize("script", suites.script_params(DOMAIN))
def test_script(script):
    """One build script. The ones needing a compiled distribution skip without FOX_TEST_DIST=1."""
    suites.check_script(script)
