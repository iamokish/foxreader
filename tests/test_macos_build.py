"""Mac packaging invariants exercised without an Apple toolchain."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("fox_macos_build_tests", PACKAGING / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mac(build):
    return build.macos


@pytest.mark.parametrize("system,machine,expected", [
    ("Windows", "AMD64", "windows-x64"), ("Linux", "x86_64", "linux-x64"),
    ("Darwin", "arm64", "macos-arm64"),
])
def test_platform_tags(build, monkeypatch, system, machine, expected):
    monkeypatch.setattr(build.platform, "system", lambda: system)
    monkeypatch.setattr(build.platform, "machine", lambda: machine)
    assert build._platform_tag() == expected


@pytest.mark.parametrize("system,machine", [
    ("Darwin", "x86_64"), ("Windows", "ARM64"), ("Linux", "aarch64"),
])
def test_rejects_unsupported_architectures(build, monkeypatch, system, machine):
    monkeypatch.setattr(build.platform, "system", lambda: system)
    monkeypatch.setattr(build.platform, "machine", lambda: machine)
    with pytest.raises(build.BuildError):
        build._platform_tag()


@pytest.mark.parametrize("argv", [
    ["--device", "cpu"], ["--device", "cu129"], ["--gguf-device", "cpu"],
    ["--gguf-device", "cu129"], ["--cuda-arch", "86-real"],
    ["--nuitka-arg=--macos-target-arch=x86_64"], ["--nuitka-arg=--macos-target-version=15.0"],
    ["--nuitka-arg=--target-arch=native"],
])
def test_invalid_mac_configuration_fails_before_work(build, monkeypatch, argv):
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sys, "argv", ["build.py", *argv])
    monkeypatch.setattr(build, "create_build_venv", lambda *_: pytest.fail("must reject before build"))
    with pytest.raises(SystemExit) as error:
        build.main()
    assert error.value.code == 2


def test_mac_default_is_metal(build, monkeypatch):
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(sys, "argv", ["build.py", "--no-gguf", "--skip-frontend"])
    monkeypatch.setattr(build.macos, "preflight", lambda: {})
    captured = []

    def stop_after_device(device):
        captured.append(device)
        raise build.BuildError("stop before build")

    monkeypatch.setattr(build, "create_build_venv", stop_after_device)
    with pytest.raises(build.BuildError, match="stop before build"):
        build.main()
    assert captured == ["macos"]


def test_preflight_pins_environment(mac, monkeypatch):
    monkeypatch.setattr(mac.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mac.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(mac.platform, "mac_ver", lambda: ("15.2", (), ""))
    monkeypatch.setenv("CFLAGS", "-mcpu=apple-m4 -march=native")
    monkeypatch.setenv("CCFLAGS", "-mcpu=apple-m4")
    monkeypatch.setenv("MACOSX_DEPLOYMENT_TARGET", "15.2")
    monkeypatch.setattr(mac, "_run", lambda cmd: "15.0" if cmd[-1] == "--show-sdk-version"
                        else "/Applications/Xcode App/" + cmd[-1])
    env = mac.preflight()
    assert env["MACOSX_DEPLOYMENT_TARGET"] == "14.0"
    assert env["CC"].endswith("/clang")
    assert env["ARCHFLAGS"] == "-arch arm64"
    assert "native" not in env["CFLAGS"] and "apple-m4" not in env["CFLAGS"]
    assert env["SKBUILD_CMAKE_ARGS"] == ""
    assert env["CCFLAGS"] == ""


def test_venv_python_deployment_target_is_checked_before_nuitka(mac, monkeypatch):
    state = {"machine": "arm64", "python": [3, 12], "bits": 64,
             "mps_built": True, "mps_available": True}
    monkeypatch.setattr(mac.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(state)))
    inspected = []
    monkeypatch.setattr(mac, "_inspect", lambda path: inspected.append(path))
    python = Path("build-venv/bin/python")
    mac.verify_environment(python, {})
    assert inspected == [python.resolve()]


def test_preflight_rejects_old_os(mac, monkeypatch):
    monkeypatch.setattr(mac.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mac.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(mac.platform, "mac_ver", lambda: ("13.6", (), ""))
    with pytest.raises(mac.MacOSBuildError, match="14.0"):
        mac.preflight()


@pytest.mark.parametrize("change", [
    {"machine": "x86_64"}, {"python": [3, 13]}, {"bits": 32},
    {"mps_built": False}, {"mps_available": False},
])
def test_bad_venv_is_rejected(mac, monkeypatch, change):
    state = {"machine": "arm64", "python": [3, 12], "bits": 64,
             "mps_built": True, "mps_available": True, **change}
    monkeypatch.setattr(mac.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(state)))
    with pytest.raises(mac.MacOSBuildError):
        mac.verify_environment(Path("python"), {})


def _commands(deps=(), *, install_id=None, rpaths=(), minimum="14.0", target="1"):
    blocks = [f"cmd LC_BUILD_VERSION\ncmdsize 32\nplatform {target}\nminos {minimum}\nsdk 15.0"]
    if install_id:
        blocks.append(f"cmd LC_ID_DYLIB\ncmdsize 48\nname {install_id} (offset 24)")
    for name in deps:
        blocks.append(f"cmd LC_LOAD_DYLIB\ncmdsize 48\nname {name} (offset 24)")
    for rpath in rpaths:
        blocks.append(f"cmd LC_RPATH\ncmdsize 32\npath {rpath} (offset 12)")
    return "\n".join(f"Load command {i}\n{block}\n" for i, block in enumerate(blocks))


def test_load_commands_preserve_spaces_and_separate_install_id(mac):
    info = mac._load_commands(_commands(
        ["/System/Library/Frameworks/Metal.framework/Versions/A/Metal", "@rpath/a library.dylib"],
        install_id="/venv/lib/Python", rpaths=["@loader_path/a directory"],
    ))
    assert info.install_id == "/venv/lib/Python"
    assert len(info.dependencies) == 2
    assert info.dependencies[-1] == "@rpath/a library.dylib"
    assert info.rpaths == ["@loader_path/a directory"]
    assert info.minimum_os == "14.0"


def test_inspect_rejects_newer_deployment_target(mac, monkeypatch):
    monkeypatch.setattr(mac, "_run", lambda cmd: _commands(minimum="15.0") if "otool" in cmd else "")
    with pytest.raises(mac.MacOSBuildError, match="above the 14.0"):
        mac._inspect(Path("library.dylib"))


def test_inspect_rejects_ios_slice(mac):
    with pytest.raises(mac.MacOSBuildError, match="other than macOS"):
        mac._load_commands(_commands(target="2"))


def _binary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 40)
    return path


def test_dependency_lookup_never_uses_host_library(mac, tmp_path):
    root = tmp_path / "dist"
    binary = _binary(root / "bin" / "fox-reader")
    external = _binary(tmp_path / "homebrew" / "libbad.dylib")
    with pytest.raises(mac.MacOSBuildError, match="missing bundled library"):
        mac._dependency_target(binary, external.as_posix(), root, {binary: mac.MachO()})


def test_dependency_lookup_rejects_ambiguous_libraries(mac, tmp_path):
    binary = _binary(tmp_path / "bin" / "fox-reader")
    first = _binary(tmp_path / "bin" / "a" / "duplicate.dylib")
    second = _binary(tmp_path / "bin" / "b" / "duplicate.dylib")
    infos = {path: mac.MachO() for path in (binary, first, second)}
    with pytest.raises(mac.MacOSBuildError, match="ambiguous"):
        mac._dependency_target(binary, "@rpath/duplicate.dylib", tmp_path, infos)


def test_multiple_loaders_can_resolve_the_same_relocated_library(mac, tmp_path):
    binary = _binary(tmp_path / "bin" / "fox-reader")
    library = _binary(tmp_path / "bin" / "torch" / "lib" / "shared.dylib")
    infos = {path: mac.MachO() for path in (binary, library)}
    names = mac._library_names(tmp_path, infos)
    for _ in range(2):
        assert mac._dependency_target(binary, "@rpath/shared.dylib", tmp_path, infos, names) == library


def test_finalize_relocates_before_signing(mac, tmp_path, monkeypatch):
    root = tmp_path / "relocated app"
    launcher = _binary(root / "launcher")
    backend = _binary(root / "bin" / "fox-reader")
    library = _binary(root / "bin" / "llama_cpp" / "lib" / "libllama.dylib")
    host_name = "/builder/venv/libllama.dylib"
    infos = {
        launcher: _commands(["/usr/lib/libSystem.B.dylib"]),
        backend: _commands([host_name], rpaths=["/builder/venv", "@loader_path/llama_cpp/lib"]),
        library: _commands(["/usr/lib/libSystem.B.dylib"], install_id=host_name),
    }
    calls = []

    def run(cmd):
        calls.append(cmd)
        path = Path(cmd[-1])
        if "otool" in cmd:
            return infos[path]
        if "install_name_tool" in cmd:
            if "-change" in cmd:
                i = cmd.index("-change")
                infos[path] = infos[path].replace(cmd[i + 1], cmd[i + 2])
        return ""

    monkeypatch.setattr(mac, "_run", run)
    mac.finalize(root)
    assert "@loader_path/llama_cpp/lib/libllama.dylib" in infos[backend]
    modifications = [i for i, cmd in enumerate(calls) if "install_name_tool" in cmd]
    signatures = [i for i, cmd in enumerate(calls) if "--sign" in cmd]
    assert max(modifications) < min(signatures)
    changed_backend = next(cmd for cmd in calls if "install_name_tool" in cmd and cmd[-1] == str(backend))
    assert "-delete_rpath" in changed_backend
    assert "@loader_path/llama_cpp/lib" not in changed_backend
    assert len(signatures) == 3


def test_finalize_does_not_sign_incomplete_bundle(mac, tmp_path, monkeypatch):
    _binary(tmp_path / "launcher")
    _binary(tmp_path / "bin" / "fox-reader")
    calls = []

    def run(cmd):
        calls.append(cmd)
        return _commands(["/opt/homebrew/lib/missing.dylib"]) if "otool" in cmd else ""

    monkeypatch.setattr(mac, "_run", run)
    with pytest.raises(mac.MacOSBuildError, match="missing"):
        mac.finalize(tmp_path)
    assert not any("codesign" in cmd for cmd in calls)


@pytest.mark.parametrize("system,compiler", [
    ("Darwin", "clang"), ("Windows", "msvc"), ("Linux", "gcc"),
])
def test_nuitka_command_uses_supported_standalone_options(build, tmp_path, monkeypatch, system, compiler):
    """Exercise real option probing and command assembly without compiling.

    The relevant options are from Nuitka 4.2.1's OptionParsing.py. In
    particular, it has no --macos-target-version: accepting every requested
    option in the fake would conceal a failure before compilation even starts.
    """
    supported = {
        "--mode", "--assume-yes-for-downloads", "--output-dir", "--output-filename",
        "--include-package", "--include-module", "--include-package-data",
        "--include-distribution-metadata", "--noinclude-data-files", "--jobs", "--report",
        "--clang", "--macos-target-arch", "--msvc", "--mingw64", "--enable-plugins",
        "--module-parameter", "--company-name", "--product-name", "--file-description",
        "--file-version", "--product-version", "--windows-icon-from-ico",
    }
    probes = []

    def capture(cmd, timeout=300):
        probes.append(cmd[-1])
        responses = {
            "--version": "4.2.1",
            "--help": "\n".join(sorted(supported)),
            "--plugin-list": "multiprocessing  Multiprocessing support",
        }
        assert cmd[-1] in responses, f"unexpected external probe: {cmd}"
        return 0, responses[cmd[-1]]

    work = tmp_path / "work with spaces"
    nuitka_out = work / "nuitka"
    expected_dist = nuitka_out / "fox_reader_main.dist"
    python = tmp_path / "venv with spaces" / "bin" / "python"
    entry = work / "entry" / "fox_reader_main.py"
    environment = {"MACOSX_DEPLOYMENT_TARGET": "14.0"} if system == "Darwin" else {}
    compiled = []

    def compile_command(cmd, *, cwd, env):
        compiled.append(list(cmd))
        assert cwd == str(build.PROJECT_ROOT)
        assert env == environment
        expected_dist.mkdir()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build.platform, "system", lambda: system)
    monkeypatch.setattr(build, "WORK_DIR", work)
    monkeypatch.setattr(build, "NUITKA_OUT", nuitka_out)
    monkeypatch.setattr(build, "BACKEND_NAME", "fox-reader.exe" if system == "Windows" else "fox-reader")
    monkeypatch.setattr(build, "_capture", capture)
    monkeypatch.setattr(build, "_probe", lambda *a, **k: {
        "sites": [str(tmp_path / "site-packages")], "modules": ["torch"], "dists": [],
    })
    monkeypatch.setattr(build, "_bytecode_flags", lambda *a: [])
    monkeypatch.setattr(build, "_trigger_cache_workaround", lambda *a: [])
    monkeypatch.setattr(build, "_build_env", lambda: environment)
    monkeypatch.setattr(build.subprocess, "run", compile_command)

    assert build.run_nuitka(python, entry, False, [], compiler) == expected_dist
    assert probes == ["--version", "--help", "--plugin-list"]
    assert len(compiled) == 1
    cmd = compiled[0]
    assert cmd[:3] == [str(python), "-m", "nuitka"]
    assert cmd[-1] == str(entry)
    assert cmd.count("--mode=standalone") == 1
    assert all(arg.split("=", 1)[0] in supported for arg in cmd if arg.startswith("--"))
    assert "--include-package=fox_reader" in cmd
    assert "--include-module=fox_reader_assets" in cmd
    assert (work / "nuitka-command.txt").read_text(encoding="utf-8").splitlines() == cmd
    mac_flags = [arg for arg in cmd if arg.startswith("--macos-")]
    if system == "Darwin":
        assert "--clang" in cmd
        assert mac_flags == ["--macos-target-arch=arm64"]
        assert not any(arg.startswith(("--msvc", "--mingw64")) for arg in cmd)
    else:
        assert mac_flags == []
        assert "--clang" not in cmd
        assert ("--msvc=latest" in cmd) == (system == "Windows")
