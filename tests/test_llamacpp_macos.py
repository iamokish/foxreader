"""Portable Metal build policy and the failures a Mac release must reject."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def llama(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "packaging"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("fox_llama_macos_tests", directory / "llamacpp.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _build(llama):
    return llama.Build(target=llama.parse_device("macos"), args=[], value="metal", spec="llama-cpp-python==0.3.36")


def test_metal_target_does_not_inherit_the_builders_isa(llama):
    expected = llama.cmake_args(llama.parse_device("macos"), machine="arm64")
    assert expected == llama.cmake_args(llama.parse_device("macos"), machine="x86_64")
    flags = dict(option[2:].split("=", 1) for option in expected)
    assert flags["GGML_NATIVE"] == "OFF"
    assert flags["GGML_CPU_ARM_ARCH"] == "armv8.2-a+fp16+dotprod"
    for language in ("C", "CXX", "OBJC", "OBJCXX", "ASM"):
        assert flags[f"CMAKE_{language}_FLAGS"] == "-march=armv8.2-a+fp16+dotprod"
    assert flags["CMAKE_OSX_ARCHITECTURES"] == "arm64"
    assert flags["CMAKE_OSX_DEPLOYMENT_TARGET"] == "14.0"
    assert flags["GGML_CPU_ALL_VARIANTS"] == "OFF"
    assert not set(flags).intersection(llama.CPU_ISA_OPTIONS)
    assert not any("native" in value.lower() for value in flags.values())
    assert flags["GGML_METAL_EMBED_LIBRARY"] == "ON"
    assert flags["GGML_ACCELERATE"] == "ON"
    assert flags["GGML_BLAS"] == "OFF"
    assert flags["GGML_OPENMP"] == "OFF"
    assert flags["CMAKE_INSTALL_RPATH"] == "@loader_path"


def test_metal_prepare_uses_apple_tools_and_never_enters_other_build_paths(llama, monkeypatch):
    environment = {"CC": "/Apple SDK/bin/clang", "CXX": "/Apple SDK/bin/clang++",
                   "SDKROOT": "/Apple SDK/MacOSX.sdk", "CFLAGS": ""}
    monkeypatch.setattr(llama._macos_policy(), "preflight", lambda: environment.copy())
    monkeypatch.setattr(llama, "locked_version", lambda: "0.3.36")

    def unexpected(*args, **kwargs):
        pytest.fail("Metal entered the CPU/CUDA host/OpenBLAS build path")

    monkeypatch.setattr(llama, "resolve_openblas", unexpected)
    monkeypatch.setattr(llama, "_host_toolchain", unexpected)
    build = llama.prepare(llama.parse_device("macos"))
    assert build.env["CMAKE_GENERATOR"] == "Ninja"
    assert build.env["CC"] == environment["CC"]
    assert build.openblas is None
    assert build.spec == "llama-cpp-python==0.3.36"
    import shlex

    assert shlex.split(build.value) == build.args


@pytest.mark.parametrize("system,machine,bits,version", [
    ("Darwin", "x86_64", 64, [3, 12]),
    ("Darwin", "arm64", 64, [3, 13]),
    ("Darwin", "arm64", 32, [3, 12]),
    ("Windows", "arm64", 64, [3, 12]),
])
def test_metal_install_rejects_wrong_venv_before_touching_it(llama, monkeypatch, system, machine, bits, version):
    monkeypatch.setattr(llama, "_capture", lambda *a, **k: (0, json.dumps([system, machine, bits, version])))
    with pytest.raises(llama.GgufError, match="native arm64 Python 3.12"):
        llama._verify_metal_python(Path("python"))


def test_metal_installs_from_source_even_if_a_wheel_appears(llama, monkeypatch):
    calls = []
    checked = []
    monkeypatch.setattr(llama.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(llama, "_verify_metal_python", lambda *a: checked.append(a))
    monkeypatch.setattr(llama, "_installed_version", lambda *a: None)
    monkeypatch.setattr(llama, "_copy_runtime_libraries", lambda *a, **k: None)
    monkeypatch.setattr(llama, "_repair_top_level", lambda *a, **k: None)
    monkeypatch.setattr(llama.subprocess, "run", lambda cmd, **kwargs: (
        calls.append((cmd, kwargs)) or SimpleNamespace(returncode=0)))
    build = _build(llama)
    build.env = {"CFLAGS": "-arch arm64", "MACOSX_DEPLOYMENT_TARGET": "14.0"}
    llama.install(Path("python"), build, env={"CFLAGS": "-mcpu=native", "CMAKE_TOOLCHAIN_FILE": "old.cmake"})
    assert checked
    cmd, kwargs = calls[0]
    assert "--no-cache-dir" in cmd
    assert cmd[-2:] == ["--no-binary", "llama-cpp-python"]
    assert kwargs["env"]["CFLAGS"] == "-arch arm64"
    assert "CMAKE_TOOLCHAIN_FILE" not in kwargs["env"]
    assert kwargs["env"]["SKBUILD_CMAKE_BUILD_TYPE"] == "Release"


@pytest.mark.parametrize("answer,exit_code", [
    ({"error": "no Metal backend"}, 0),
    ({"error": "could not initialize shader library"}, 0),
    ({"initialized": False}, 0),
    ({"initialized": True}, 1),
    ({}, 0),
])
def test_metal_verification_never_accepts_an_unknown_or_failed_backend(llama, monkeypatch, answer, exit_code):
    monkeypatch.setattr(llama.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps(answer), stderr="Metal diagnostic", returncode=exit_code))
    with pytest.raises(llama.GgufError, match="Metal llama.cpp verification failed"):
        llama.verify(Path("python"), _build(llama))


def test_metal_probe_cannot_be_rescued_by_build_directory_or_external_libraries(llama, monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=json.dumps({"initialized": True, "error": None, "info": "CPU Metal",
                                                 "version": "0.3.36", "device": "Apple M1",
                                                 "gpu_available": True, "cpu_initialized": True,
                                                 "embedded_sources": True}),
                               stderr="", returncode=0)

    monkeypatch.setattr(llama.subprocess, "run", run)
    build = _build(llama)
    llama.verify(Path("python"), build, env={"DYLD_LIBRARY_PATH": "/brew/lib", "LLAMA_CPP_LIB_PATH": "old",
                                          "GGML_METAL_PATH_RESOURCES": "/source", "PATH": "/usr/bin"})
    cmd, kwargs = calls[0]
    assert cmd[:3] == ["python", "-I", "-c"]
    assert Path(kwargs["cwd"]) != llama.PROJECT_ROOT
    assert kwargs["env"] == {"PATH": "/usr/bin"}
    assert kwargs["timeout"] == 300
    assert build.system_info == "CPU Metal"


def test_metal_bundle_keeps_its_ctypes_library_directory_complete(llama, monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in ("libllama.dylib", "libggml.dylib", "libggml-base.dylib", "libggml-metal.dylib"):
        (source / name).write_bytes(b"package copy")
        (dist / name).write_bytes(b"Nuitka copy")
    monkeypatch.setattr(llama, "_package_lib_dir", lambda *a: source)
    monkeypatch.setattr(llama, "_cuda_runtime", lambda *a: [])
    llama.bundle(dist, Path("python"), _build(llama))
    for path in source.iterdir():
        assert (dist / "llama_cpp" / "lib" / path.name).read_bytes() == b"package copy"


@pytest.mark.parametrize("allowed", [None, "0", "1"])
def test_headless_probe_requires_explicit_opt_in(llama, monkeypatch, allowed, capsys):
    answer = {"initialized": False, "error": None, "gpu_available": False,
              "cpu_initialized": True, "embedded_sources": True}
    monkeypatch.setattr(llama.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps(answer), stderr="", returncode=0))
    env = {} if allowed is None else {"FOX_BUILD_ALLOW_NO_GPU": allowed}
    if allowed == "1":
        assert llama.run_metal_probe(Path("python"), env=env) == answer
        assert "SKIPPED Metal GPU execution" in capsys.readouterr().out
    else:
        with pytest.raises(llama.GgufError, match="explicit FOX_BUILD_ALLOW_NO_GPU"):
            llama.run_metal_probe(Path("python"), env=env)


@pytest.mark.parametrize("change", [
    {"gpu_available": None}, {"embedded_sources": False}, {"cpu_initialized": False},
    {"gpu_available": True}, {"error": "corrupt library"},
])
def test_headless_opt_in_never_hides_incomplete_or_failed_checks(llama, monkeypatch, change):
    answer = {"initialized": False, "error": None, "gpu_available": False,
              "cpu_initialized": True, "embedded_sources": True, **change}
    monkeypatch.setattr(llama.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps(answer), stderr="", returncode=0))
    with pytest.raises(llama.GgufError, match="verification failed"):
        llama.run_metal_probe(Path("python"), env={"FOX_BUILD_ALLOW_NO_GPU": "1"})


def test_relocated_probe_checks_the_requested_library_directory(llama, monkeypatch, tmp_path):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps({
            "initialized": False, "error": None, "gpu_available": False,
            "cpu_initialized": True, "embedded_sources": True,
        }), stderr="", returncode=0)

    monkeypatch.setattr(llama.subprocess, "run", run)
    library = tmp_path / "relocated app" / "bin" / "llama_cpp" / "lib"
    llama.run_metal_probe(Path("python"), env={"FOX_BUILD_ALLOW_NO_GPU": "1"}, libdir=library)
    assert commands[0][-1] == str(library.resolve())
    assert commands[0][3] == llama._METAL_VERIFY_PROBE


@pytest.mark.parametrize("output", ["", "[]", "null", "not JSON"])
def test_probe_crashes_and_malformed_results_preserve_native_diagnostics(llama, monkeypatch, output):
    monkeypatch.setattr(llama.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=output, stderr="dyld: missing symbol in libggml-metal", returncode=-6))
    with pytest.raises(llama.GgufError, match="exit -6") as error:
        llama.run_metal_probe(Path("python"), env={"FOX_BUILD_ALLOW_NO_GPU": "1"})
    assert "dyld: missing symbol" in str(error.value)


def test_installed_metal_version_must_match_the_lock(llama, monkeypatch):
    monkeypatch.setattr(llama, "run_metal_probe", lambda *a, **k: {"version": "0.3.35"})
    with pytest.raises(llama.GgufError, match="does not match llama-cpp-python==0.3.36"):
        llama.verify(Path("python"), _build(llama))


@pytest.mark.parametrize("gpu_available,failure,registered_metal", [
    (True, None, 1), (True, None, 41), (False, None, 1),
    (False, "source", 1), (True, "gpu", 41), (False, "apple", 1),
    (True, "native_registry", 1), (True, "registry_name", 1),
    (True, "registry", 41), (True, "offload", 41), (True, "devices", 41),
])
@pytest.mark.parametrize("registry_name", [b"MTL", b"Metal"])
@pytest.mark.parametrize("relocated", [False, True])
def test_metal_probe_executes_native_checks(llama, monkeypatch, capsys, tmp_path,
                                          gpu_available, failure, registered_metal, registry_name, relocated):
    """Execute the subprocess code against the pinned ABI, including a headless VM."""
    initialized = []
    freed = []
    shader_symbols = []
    shader = b"#include <metal_stdlib>\nkernel void example() {}"

    class Function:
        def __init__(self, call):
            self.call = call

        def __call__(self, *args):
            return self.call(*args)

    def initialize(device, params):
        assert params is None
        initialized.append(device)
        return 0 if failure == "gpu" else device + 100

    def find_registry(name):
        # ggml_backend_reg_by_name returns NULL for an unknown name. The
        # locked 0.3.36 Metal backend is named MTL; an assumed Metal fails.
        if failure == "registry":
            return 0
        return {registry_name: registered_metal, b"CPU": 2}.get(name, 0)

    def device_count(registry):
        # The direct unversioned library may be another loaded image. All
        # execution checks must use libllama's globally registered instance.
        assert registry == registered_metal
        return 0 if failure == "devices" else 1

    def get_device(registry, index):
        assert registry == registered_metal and index == 0
        return registry + 10

    symbols = SimpleNamespace(**{name: Function(call) for name, call in {
        "ggml_backend_reg_by_name": find_registry,
        "ggml_backend_reg_name": lambda registry: b"" if failure == "registry_name" else registry_name,
        "ggml_backend_reg_dev_count": device_count,
        "ggml_backend_reg_dev_get": get_device,
        "ggml_backend_dev_description": lambda device: b"Apple M1",
        "ggml_backend_dev_init": initialize,
        "ggml_backend_free": lambda backend: freed.append(backend),
        "ggml_backend_cpu_init": lambda: 112,
        "ggml_backend_metal_reg": lambda: 0 if failure == "native_registry" else 1,
        "ggml_backend_metal_init": lambda: 111,
        "llama_backend_init": lambda: None,
        "llama_backend_free": lambda: None,
        "llama_print_system_info": lambda: b"CPU Metal",
        "llama_supports_gpu_offload": lambda: failure != "offload",
    }.items()})

    class Char:
        @staticmethod
        def in_dll(library, symbol):
            assert library is symbols
            shader_symbols.append(symbol)
            if failure == "source" and symbol == "ggml_metallib_misc_start":
                raise ValueError("missing embedded shader symbol")
            return 1000 if symbol.endswith("_start") else 1000 + len(shader)

    def apple_device():
        if failure == "apple":
            raise OSError("Metal framework probe failed")
        return 1 if gpu_available else 0

    apple = SimpleNamespace(MTLCreateSystemDefaultDevice=Function(apple_device))
    fake_ctypes = SimpleNamespace(
        CDLL=lambda path: apple if "Metal.framework" in path else symbols,
        c_void_p=llama.ctypes.c_void_p, c_char_p=llama.ctypes.c_char_p,
        c_size_t=llama.ctypes.c_size_t, c_bool=llama.ctypes.c_bool,
        c_char=Char, addressof=lambda value: value, string_at=lambda address, length: shader,
    )
    core = SimpleNamespace(_base_path=str(tmp_path), _lib=symbols)
    package = SimpleNamespace(__version__="0.3.36", llama_cpp=core,
                              llama_backend_init=lambda: None,
                              llama_supports_gpu_offload=lambda: True,
                              llama_print_system_info=lambda: b"CPU Metal")
    monkeypatch.setitem(sys.modules, "llama_cpp", package)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", core)
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setattr(sys, "argv", ["-c", str(tmp_path)] if relocated else ["-c"])
    exec(llama._METAL_VERIFY_PROBE, {})
    answer = json.loads(capsys.readouterr().out)
    if failure:
        assert answer["error"]
        assert answer["initialized"] is False
        if failure in {"native_registry", "registry_name", "registry", "offload", "devices"}:
            assert initialized == []
        if failure == "native_registry":
            assert "no native backend registry" in answer["error"]
        elif failure == "registry_name":
            assert "empty backend registry name" in answer["error"]
        elif failure == "registry":
            assert repr(registry_name.decode()) in answer["error"]
            assert "is not registered" in answer["error"]
    else:
        assert answer["error"] is None
        assert answer["embedded_sources"] is True
        assert answer["version"] == (None if relocated else "0.3.36")
        assert answer["cpu_initialized"] is True
        assert answer["gpu_available"] is gpu_available
        assert answer["initialized"] is gpu_available
        assert answer["registry_name"] == (registry_name.decode() if gpu_available else None)
        assert initialized == ([registered_metal + 10] if gpu_available else [])
        assert freed == ([112, registered_metal + 110] if gpu_available else [112])
        assert len(shader_symbols) == 68  # all 34 embedded source regions
