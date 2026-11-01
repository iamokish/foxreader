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
                                                 "version": "0.3.36", "device": "Apple M1"}),
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


def test_metal_probe_initializes_and_frees_both_registered_backends(llama, monkeypatch, capsys, tmp_path):
    """Execute the subprocess code against the pinned ABI, catching ctypes drift."""
    initialized = []
    freed = []

    class Function:
        def __init__(self, call):
            self.call = call

        def __call__(self, *args):
            return self.call(*args)

    def initialize(device, params):
        assert params is None
        initialized.append(device)
        return device + 100

    symbols = SimpleNamespace(**{name: Function(call) for name, call in {
        "ggml_backend_reg_by_name": lambda name: {b"Metal": 1, b"CPU": 2}[name],
        "ggml_backend_reg_dev_count": lambda registry: 1,
        "ggml_backend_reg_dev_get": lambda registry, index: registry + 10,
        "ggml_backend_dev_description": lambda device: b"Apple M1",
        "ggml_backend_dev_init": initialize,
        "ggml_backend_free": lambda backend: freed.append(backend),
    }.items()})
    core = SimpleNamespace(_base_path=str(tmp_path), _lib=symbols)
    package = SimpleNamespace(__version__="0.3.36", llama_cpp=core,
                              llama_backend_init=lambda: None,
                              llama_supports_gpu_offload=lambda: True,
                              llama_print_system_info=lambda: b"CPU Metal")
    monkeypatch.setitem(sys.modules, "llama_cpp", package)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", core)
    monkeypatch.setattr(llama.ctypes, "CDLL", lambda path: symbols)
    exec(llama._METAL_VERIFY_PROBE, {})
    answer = json.loads(capsys.readouterr().out)
    assert answer["error"] is None
    assert answer["initialized"] is True
    assert initialized == [12, 11]
    assert freed == [112, 111]
