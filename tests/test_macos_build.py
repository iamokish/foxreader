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


@pytest.mark.parametrize("no_window", [False, True])
def test_build_launcher_receives_window_variant(build, monkeypatch, tmp_path, no_window):
    monkeypatch.setattr(build, "WORK_DIR", tmp_path / "work")
    compiler = SimpleNamespace(kind="fixture")
    monkeypatch.setattr(build, "find_compiler", lambda **_: compiler)
    calls = []

    def compile_fixture(output, **kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr(build, "build_launcher", compile_fixture)
    launcher, kind = build.compile_launcher(tmp_path / "python", no_window=no_window)
    assert launcher == build.WORK_DIR / build.LAUNCHER_NAME
    assert kind == "fixture"
    assert calls == [{"compiler": compiler, "no_window": no_window}]


@pytest.mark.parametrize("no_window", [False, True])
def test_mac_staging_always_finalizes_binaries_and_signs_only_present_app(
    build, monkeypatch, tmp_path, no_window,
):
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build, "STAGE_DIR", tmp_path / "stage")
    dist = tmp_path / "backend.dist"
    dist.mkdir()
    (dist / build.BACKEND_NAME).write_bytes(b"backend fixture")
    launcher = tmp_path / build.LAUNCHER_NAME
    launcher.write_bytes(b"launcher fixture")
    finalized = []
    codesigned = []
    monkeypatch.setattr(build.macos, "finalize", finalized.append)
    monkeypatch.setattr(build.macos, "_run", codesigned.append)

    root = build.stage_layout(dist, launcher, no_window=no_window)

    assert finalized == [root]
    assert (root / "bin" / build.BACKEND_NAME).read_bytes() == b"backend fixture"
    if no_window:
        assert codesigned == []
        assert not (root / "Fox Reader.app").exists()
        assert (root / "Fox Reader.command").is_file()
    else:
        assert codesigned == [
            ["xcrun", "codesign", "--force", "--sign", "-", str(root / "Fox Reader.app")],
            ["xcrun", "codesign", "--verify", "--strict", str(root / "Fox Reader.app")],
        ]


@pytest.mark.parametrize("gguf_device", [None, "cpu", "cu130"])
@pytest.mark.parametrize("platform_tag", ["windows-x64", "linux-x64", "macos-arm64"])
def test_no_window_archive_has_distinct_suffix(build, gguf_device, platform_tag):
    native = build.archive_stem("0.2.2", "cpu", gguf_device, platform_tag)
    terminal = build.archive_stem("0.2.2", "cpu", gguf_device, platform_tag, no_window=True)
    assert terminal == native + "-no-window"


def test_no_window_packaging_preserves_existing_native_archive(build, monkeypatch, tmp_path):
    monkeypatch.setattr(build, "DIST_DIR", tmp_path / "dist")
    monkeypatch.setattr(build, "_platform_tag", lambda: "windows-x64")
    root = tmp_path / "fox-reader"
    root.mkdir()
    (root / "fixture.txt").write_text("native contents", encoding="utf-8")
    native = build.package_output(root, "0.2.2", "cpu", None)
    original = native.read_bytes()
    (root / "fixture.txt").write_text("no-window contents", encoding="utf-8")

    terminal = build.package_output(root, "0.2.2", "cpu", None, no_window=True)

    assert terminal.name == "fox-reader-v0.2.2-cpu-windows-x64-no-window.zip"
    assert native.read_bytes() == original
    assert terminal.read_bytes() != original


@pytest.mark.parametrize("no_window", [False, True])
def test_window_variant_propagates_through_main_and_metadata(build, monkeypatch, tmp_path, no_window):
    monkeypatch.setattr(build.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build.platform, "machine", lambda: "x86_64")
    argv = ["build.py", "--no-gguf", "--skip-frontend"]
    monkeypatch.setattr(sys, "argv", argv + (["--no-window"] if no_window else []))
    monkeypatch.setattr(build, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(build, "DIST_DIR", tmp_path / "dist")
    root = tmp_path / "staged"
    root.mkdir()
    propagated = []

    def compile_fixture(python, **kwargs):
        propagated.append(("compile", kwargs))
        return tmp_path / "launcher", "fixture"

    def stage_fixture(dist, launcher, **kwargs):
        propagated.append(("stage", kwargs))
        return root

    def package_fixture(root, version, device, gguf_device, **kwargs):
        propagated.append(("package", kwargs))
        return tmp_path / "archive.tar.gz"

    monkeypatch.setattr(build, "compile_launcher", compile_fixture)
    monkeypatch.setattr(build, "stage_layout", stage_fixture)
    monkeypatch.setattr(build, "package_output", package_fixture)
    for name in (
        "create_build_venv", "embed_frontend", "install_llama_cpp", "_warn_if_disk_tight",
        "run_nuitka", "purge_cuda_libs", "bundle_llama_cpp", "bundle_runtime_dlls",
        "prune_duplicated_libs", "bundle_metadata", "cleanup",
    ):
        monkeypatch.setattr(build, name, lambda *_, **__: None)
    monkeypatch.setattr(build, "finalize_release_files", lambda archive, split: [archive])

    assert build.main() == 0
    assert propagated == [(name, {"no_window": no_window}) for name in ("compile", "stage", "package")]
    assert json.loads((root / "build-info.json").read_text(encoding="utf-8"))["no_window"] is no_window


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


def test_hosted_builder_checks_mps_wheel_and_python_without_a_gpu(mac, monkeypatch, capsys):
    state = {"machine": "arm64", "python": [3, 12], "bits": 64,
             "mps_built": True, "mps_available": False, "gpu_available": False}
    monkeypatch.setattr(mac.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(state)))
    inspected = []
    monkeypatch.setattr(mac, "_inspect", lambda path: inspected.append(path))
    python = Path("build-venv/bin/python")
    mac.verify_environment(python, {"FOX_BUILD_ALLOW_NO_GPU": "1"})
    assert inspected == [python.resolve()]
    assert "GPU execution not tested" in capsys.readouterr().out


@pytest.mark.parametrize("change", [
    {"machine": "x86_64"}, {"python": [3, 13]}, {"mps_built": False},
    {"gpu_available": True}, {"gpu_available": None},
])
def test_hosted_flag_does_not_accept_a_broken_build_environment(mac, monkeypatch, change):
    state = {"machine": "arm64", "python": [3, 12], "bits": 64,
             "mps_built": True, "mps_available": False, "gpu_available": False, **change}
    monkeypatch.setattr(mac.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(state)))
    with pytest.raises(mac.MacOSBuildError):
        mac.verify_environment(Path("python"), {"FOX_BUILD_ALLOW_NO_GPU": "1"})


def test_hosted_flag_does_not_hide_a_gpu_execution_failure(mac, monkeypatch):
    def gpu_failure(*args, **kwargs):
        raise mac.subprocess.CalledProcessError(1, ["python"], stderr="MPS allocation failed")

    monkeypatch.setattr(mac.subprocess, "run", gpu_failure)
    with pytest.raises(mac.MacOSBuildError, match="MPS allocation failed"):
        mac.verify_environment(Path("python"), {"FOX_BUILD_ALLOW_NO_GPU": "1"})


@pytest.fixture
def entry():
    spec = importlib.util.spec_from_file_location("fox_macos_entry_tests", PACKAGING / "fox_reader_main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("gpu_available", [False, True])
def test_compiled_torch_probe_exercises_the_available_devices(entry, monkeypatch, capsys, gpu_available):
    calls = []
    monkeypatch.setattr(entry, "_macos_gpu_available", lambda: gpu_available)

    class Tensor:
        def __add__(self, other):
            assert other == 1
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [2.0, 2.0]

    def ones(count, device="cpu"):
        assert count == 2
        calls.append(device)
        return Tensor()

    torch = SimpleNamespace(ones=ones, backends=SimpleNamespace(mps=SimpleNamespace(
        is_built=lambda: True, is_available=lambda: gpu_available,
    )))
    monkeypatch.setitem(sys.modules, "torch", torch)
    entry._verify_macos_runtime()
    assert calls == (["cpu", "mps"] if gpu_available else ["cpu"])
    assert json.loads(capsys.readouterr().out) == {
        "mps_built": True, "mps_available": gpu_available, "cpu_ok": True, "mps_ok": gpu_available,
        "gpu_available": gpu_available,
    }


def test_compiled_torch_probe_rejects_non_mps_wheel_even_on_hosted_runner(entry, monkeypatch):
    monkeypatch.setenv("FOX_BUILD_ALLOW_NO_GPU", "1")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(
        is_built=lambda: False,
    ))))
    with pytest.raises(RuntimeError, match="without MPS support"):
        entry._verify_macos_runtime()


def test_compiled_probe_exits_before_starting_the_app(entry, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "argv", ["fox-reader", "--verify-macos-runtime"])
    monkeypatch.setattr(entry, "_mark_frozen", lambda: None)
    monkeypatch.setattr(entry, "_force_utf8_output", lambda: None)
    monkeypatch.setattr(entry, "_register_dll_directories", lambda: None)
    monkeypatch.setattr(entry, "_verify_macos_runtime", lambda: calls.append("verified"))
    entry.main()
    assert calls == ["verified"]


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


@pytest.mark.parametrize("architectures", [("arm64",), ("x86_64", "arm64"), ("x86_64",)])
def test_inspect_obeys_lipo_verify_arch_argument_contract(mac, monkeypatch, architectures):
    binary = Path("bundle with spaces") / "Python"
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["xcrun", "lipo"]:
            # Apple's -verify_arch consumes all remaining arguments as
            # architecture names; the input filename must precede it.
            verify_at = cmd.index("-verify_arch")
            assert cmd[2:verify_at] == [str(binary)]
            requested = cmd[verify_at + 1:]
            assert requested == ["arm64"]
            missing = set(requested) - set(architectures)
            return SimpleNamespace(returncode=int(bool(missing)), stdout="",
                                   stderr="input file does not contain arm64" if missing else "")
        assert cmd[:2] == ["xcrun", "otool"]
        return SimpleNamespace(returncode=0, stdout=_commands(), stderr="")

    monkeypatch.setattr(mac.subprocess, "run", run)
    if "arm64" in architectures:
        assert mac._inspect(binary).minimum_os == "14.0"
        assert len(calls) == 2
    else:
        with pytest.raises(mac.MacOSBuildError, match="does not contain arm64"):
            mac._inspect(binary)
        assert len(calls) == 1


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
@pytest.mark.parametrize("keep_dup_libs", [False, True])
def test_nuitka_command_uses_supported_standalone_options(build, tmp_path, monkeypatch, system, compiler,
                                                        keep_dup_libs):
    """Exercise real option probing and command assembly without compiling.

    The relevant options are from Nuitka 4.2.1's OptionParsing.py. In
    particular, it has no --macos-target-version: accepting every requested
    option in the fake would conceal a failure before compilation even starts.
    """
    # Exercise the actual hub dependency: filelock exposes this class lazily,
    # so a working import in the venv alone does not put it in a frozen build.
    from huggingface_hub.utils._fixes import SoftFileLock

    soft_lock_module = SoftFileLock.__module__
    supported = {
        "--mode", "--assume-yes-for-downloads", "--output-dir", "--output-filename",
        "--include-package", "--include-module", "--include-package-data",
        "--include-distribution-metadata", "--noinclude-data-files", "--jobs", "--report",
        "--clang", "--macos-target-arch", "--msvc", "--mingw64", "--enable-plugins",
        "--module-parameter", "--company-name", "--product-name", "--file-description",
        "--file-version", "--product-version", "--windows-icon-from-ico", "--windows-console-mode",
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
    if keep_dup_libs:
        monkeypatch.setenv("FOX_BUILD_KEEP_DUP_LIBS", "1")
    else:
        monkeypatch.delenv("FOX_BUILD_KEEP_DUP_LIBS", raising=False)
    monkeypatch.setattr(build, "WORK_DIR", work)
    monkeypatch.setattr(build, "NUITKA_OUT", nuitka_out)
    monkeypatch.setattr(build, "BACKEND_NAME", "fox-reader.exe" if system == "Windows" else "fox-reader")
    monkeypatch.setattr(build, "_capture", capture)
    monkeypatch.setattr(build, "_probe", lambda *a, **k: {
        "sites": [str(tmp_path / "site-packages")], "modules": ["torch", soft_lock_module], "dists": [],
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
    assert f"--include-module={soft_lock_module}" in cmd
    assert ("--windows-console-mode=disable" in cmd) == (system == "Windows")
    assert (work / "nuitka-command.txt").read_text(encoding="utf-8").splitlines() == cmd
    mac_flags = [arg for arg in cmd if arg.startswith("--macos-")]
    early_torch_exclusions = {
        "--noinclude-data-files=torch/include/*", "--noinclude-data-files=torch/test/*",
    }
    if system == "Darwin":
        assert "--clang" in cmd
        assert mac_flags == ["--macos-target-arch=arm64"]
        assert not any(arg.startswith(("--msvc", "--mingw64")) for arg in cmd)
    else:
        assert mac_flags == []
        assert "--clang" not in cmd
        assert ("--msvc=latest" in cmd) == (system == "Windows")
    if system == "Darwin" and not keep_dup_libs:
        assert early_torch_exclusions <= set(cmd)
        # Match Nuitka's target-path fnmatch semantics, including nested files.
        # Runtime resources and the shared-memory helper must stay eligible.
        import fnmatch

        patterns = [arg.split("=", 1)[1] for arg in early_torch_exclusions]
        for path in ("torch/include/ATen/ops/add.h", "torch/include/torch/csrc/api/include/torch.h",
                     "torch/test/test_api"):
            assert any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
        for path in ("torch/lib/libtorch_cpu.dylib", "torch/lib/libtorch_python.dylib",
                     "torch/bin/torch_shm_manager", "torch/__init__.py", "torch/_inductor/script.ld"):
            assert not any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
    else:
        assert early_torch_exclusions.isdisjoint(cmd)


@pytest.fixture
def mac_smoke():
    source = Path(__file__).resolve().parent / "dist" / "smoke_macos.py"
    spec = importlib.util.spec_from_file_location("fox_macos_dist_smoke_tests", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mac_smoke_removes_host_library_overrides_but_keeps_validation_policy(mac_smoke, monkeypatch):
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/build/host/libs")
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", "/build/host/llama.dylib")
    monkeypatch.setenv("GGML_METAL_PATH_RESOURCES", "/build/source")
    monkeypatch.setenv("PYTHONPATH", "/build/venv")
    monkeypatch.setenv("FOX_READER_MODELS_DIR", "/models")
    monkeypatch.setenv("FOX_BUILD_ALLOW_NO_GPU", "1")
    env = mac_smoke._clean_env()
    assert not any(name.startswith(("DYLD_", "PYTHON", "GGML_")) for name in env)
    assert "LLAMA_CPP_LIB_PATH" not in env
    assert "FOX_READER_MODELS_DIR" not in env
    assert env["FOX_BUILD_ALLOW_NO_GPU"] == "1"
    assert env["FOX_READER_CLI"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"


@pytest.mark.parametrize("mps_available", [False, True])
def test_mac_smoke_settings_must_match_independent_compiled_probe(mac_smoke, mps_available):
    expected = "mps" if mps_available else "cpu"
    devices = {
        "options": [{"id": "cpu", "available": True}, {"id": "mps", "available": mps_available}],
        "resolved": {slot: expected for slot in ("paddleocr", "bubble", "textseg")},
    }
    mac_smoke._check_devices(devices, mps_available=mps_available)
    with pytest.raises(RuntimeError, match="disagrees"):
        mac_smoke._check_devices(devices, mps_available=not mps_available)
    devices["resolved"]["textseg"] = "cpu" if mps_available else "mps"
    with pytest.raises(RuntimeError, match="did not select"):
        mac_smoke._check_devices(devices, mps_available=mps_available)


def test_mac_smoke_executes_the_relocated_torch_binary(mac_smoke, monkeypatch, tmp_path):
    calls = []
    state = {"mps_built": True, "mps_available": False, "gpu_available": False, "cpu_ok": True, "mps_ok": False}

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(state), stderr="")

    monkeypatch.setattr(mac_smoke.subprocess, "run", run)
    root = tmp_path / "relocated app"
    scratch = tmp_path / "unrelated cwd"
    env = {"FOX_BUILD_ALLOW_NO_GPU": "1"}
    assert mac_smoke._torch_smoke(root, env, scratch) == state
    assert calls[0][0] == [str(root / "bin" / "fox-reader"), "--verify-macos-runtime"]
    assert calls[0][1]["cwd"] == scratch
    assert calls[0][1]["env"] == env
    with pytest.raises(RuntimeError, match="no usable Apple GPU"):
        mac_smoke._torch_smoke(root, {}, scratch)


@pytest.mark.parametrize("change", [
    {"mps_built": False}, {"cpu_ok": False}, {"mps_available": None},
    {"gpu_available": None}, {"gpu_available": True},
    {"mps_available": True, "gpu_available": True, "mps_ok": False},
])
def test_mac_smoke_no_gpu_permission_cannot_hide_a_broken_torch_build(mac_smoke, monkeypatch, tmp_path, change):
    state = {"mps_built": True, "mps_available": False, "gpu_available": False,
             "cpu_ok": True, "mps_ok": False, **change}
    monkeypatch.setattr(mac_smoke.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps(state), stderr=""))
    with pytest.raises(RuntimeError, match="compiled Torch smoke"):
        mac_smoke._torch_smoke(tmp_path, {"FOX_BUILD_ALLOW_NO_GPU": "1"}, tmp_path)


def test_mac_smoke_reuses_metal_validation_against_relocated_libraries(mac_smoke, monkeypatch, tmp_path):
    calls = []
    state = {"gpu_available": False, "cpu_initialized": True}

    def probe(python, **kwargs):
        calls.append((python, kwargs))
        return state

    monkeypatch.setattr(mac_smoke.llamacpp, "run_metal_probe", probe, raising=False)
    env = {"FOX_BUILD_ALLOW_NO_GPU": "1"}
    assert mac_smoke._metal_smoke(tmp_path, env) is state
    assert calls == [(Path(sys.executable), {"env": env, "libdir": tmp_path / "bin" / "llama_cpp" / "lib"})]


@pytest.mark.parametrize("gpu_available", [False, True])
def test_mac_smoke_summary_distinguishes_validation_from_gpu_execution(mac_smoke, tmp_path, gpu_available):
    summary = tmp_path / "summary.md"
    mac_smoke._write_summary(
        {"GITHUB_STEP_SUMMARY": str(summary)},
        torch_state={"mps_available": gpu_available}, metal_state={"initialized": gpu_available},
    )
    text = summary.read_text(encoding="utf-8")
    assert "embedded source, CPU initialization" in text
    if gpu_available:
        assert "| MPS tensor execution | Passed |" in text
        assert "| Metal initialization and shader compilation | Passed |" in text
        assert "Not run" not in text
    else:
        assert text.count("Not run: no GPU exposed") == 2
        assert "| MPS tensor execution | Passed |" not in text


def test_mac_smoke_summary_marks_omitted_gguf_backend(mac_smoke, tmp_path):
    summary = tmp_path / "summary.md"
    mac_smoke._write_summary(
        {"GITHUB_STEP_SUMMARY": str(summary)}, torch_state={"mps_available": False}, metal_state=None,
    )
    assert summary.read_text(encoding="utf-8").count("Not included (--no-gguf)") == 2
