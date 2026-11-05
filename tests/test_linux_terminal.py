"""Real Linux PTYs verify terminal handoff without opening a desktop terminal."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux terminal/Unix socket handoff")


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.025)
    raise AssertionError("Timed out waiting for terminal handoff evidence")


@pytest.fixture(scope="module")
def terminal_binary(tmp_path_factory):
    directory = tmp_path_factory.mktemp("linux-terminal-build")
    spec = importlib.util.spec_from_file_location("fox_terminal_test_toolchain", ROOT / "packaging/toolchain.py")
    toolchain = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(ROOT / "packaging"))
        imports.setitem(sys.modules, spec.name, toolchain)
        spec.loader.exec_module(toolchain)
        try:
            compiler = toolchain.find_compiler(allow_download=False)
        except toolchain.ToolchainError as exc:
            pytest.skip(f"native compiler unavailable: {exc}")
        output = directory / "terminal-host"
        result = subprocess.run(
            [str(compiler.path), "-Wall", "-Wextra", str(ROOT / "tests/terminal_host.c"),
             str(ROOT / "packaging/launcher/terminal_linux.c"), "-o", str(output)],
            cwd=directory, env=compiler.env, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return output


@pytest.fixture
def terminal_installation(terminal_binary, tmp_path):
    root = tmp_path / "terminal path ; $with spaces 日本"
    commands = root / "commands"
    commands.mkdir(parents=True)
    binary = root / "launcher ; literal"
    shutil.copy2(terminal_binary, binary)
    env = {**os.environ, "PATH": str(commands), "FOX_TEST_STARTED": str(root / "started"),
           "FOX_TEST_EVIDENCE": str(root)}
    # The private command provides a real controlling PTY without a GUI. Its
    # interpreter link also works when the Python installation path has spaces.
    (commands / "python3").symlink_to(sys.executable)
    emulator = commands / "xdg-terminal-exec"
    emulator.write_text("#!/usr/bin/env python3\n" + textwrap.dedent('''\
        import json
        import os
        import pty
        import select
        import socket
        import subprocess
        import sys
        import time
        from pathlib import Path

        root = Path(os.environ['FOX_TEST_EVIDENCE'])
        (root / 'terminal-pid').write_text(str(os.getpid()))
        arguments = sys.argv[1:]
        if arguments and arguments[0] == '--':
            arguments = arguments[1:]
        socket_path = Path(arguments[-1])
        (root / 'terminal-arguments.json').write_text(json.dumps(arguments))
        (root / 'socket-mode').write_text(oct(socket_path.parent.stat().st_mode & 0o777))
        mode = os.environ.get('FOX_TEST_TERMINAL_BEHAVIOR', 'pty')
        if mode == 'silent-success':
            sys.exit(0)
        if mode == 'no-tty':
            sys.exit(subprocess.call(arguments, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if mode == 'waiting-peer':
            peer = socket.socket(socket.AF_UNIX)
            peer.connect(str(socket_path))
            (root / 'peer-connected').touch()
            time.sleep(30)
            sys.exit(4)
        pid, master = pty.fork()
        if pid == 0:
            os.execv(arguments[0], arguments)
        try:
            while True:
                ended, status = os.waitpid(pid, os.WNOHANG)
                if ended:
                    (root / 'terminal-exited').write_text(str(os.waitstatus_to_exitcode(status)))
                    sys.exit(os.waitstatus_to_exitcode(status))
                readable, _, _ = select.select([master], [], [], 0.1)
                if readable:
                    try:
                        os.read(master, 4096)
                    except OSError:
                        pass
        finally:
            os.close(master)
        '''), encoding="utf-8")
    emulator.chmod(0o755)
    try:
        yield root, binary, env
    finally:
        # A failed cancellation assertion must not leave its delayed fake
        # emulator alive. Only the process running this exact fixture qualifies.
        marker = root / "terminal-pid"
        if marker.exists():
            pid = int(marker.read_text())
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                if os.fsencode(emulator) in command:
                    os.kill(pid, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass


def run_handoff(binary, root, env, timeout=15):
    return subprocess.run([str(binary), "--handoff"], cwd=root, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout, start_new_session=True)


def test_handoff_requires_a_visible_successor_and_preserves_arguments(terminal_installation):
    root, binary, env = terminal_installation
    result = run_handoff(binary, root, env)
    assert result.returncode == 0, result.stdout + result.stderr
    wait_for(lambda: (root / "started").exists())
    wait_for(lambda: (root / "terminal-exited").exists())
    assert (root / "terminal-exited").read_text() == "0"
    arguments = json.loads((root / "terminal-arguments.json").read_text())
    assert arguments[:2] == [str(binary), "--terminal-manager"]
    socket_path = Path(arguments[2])
    assert (root / "socket-mode").read_text() == "0o700"
    assert not socket_path.exists() and not socket_path.parent.exists()


def test_missing_terminal_exits_without_starting_a_successor(terminal_installation):
    root, binary, env = terminal_installation
    (root / "commands/xdg-terminal-exec").unlink()
    result = run_handoff(binary, root, env)
    assert result.returncode == 3
    assert not (root / "started").exists()


def test_terminal_without_a_controlling_tty_cannot_authorize_startup(terminal_installation):
    root, binary, env = terminal_installation
    env["FOX_TEST_TERMINAL_BEHAVIOR"] = "no-tty"
    result = run_handoff(binary, root, env)
    assert result.returncode == 3
    assert not (root / "started").exists()
    arguments = json.loads((root / "terminal-arguments.json").read_text())
    assert not Path(arguments[-1]).parent.exists()


def test_a_terminal_exit_code_alone_does_not_authorize_startup(terminal_installation):
    root, binary, env = terminal_installation
    env["FOX_TEST_TERMINAL_BEHAVIOR"] = "silent-success"
    result = run_handoff(binary, root, env)
    assert result.returncode == 3
    assert not (root / "started").exists()
    arguments = json.loads((root / "terminal-arguments.json").read_text())
    assert not Path(arguments[-1]).parent.exists()


def test_cancellation_while_a_peer_connects_does_not_authorize_startup(terminal_installation):
    root, binary, env = terminal_installation
    env["FOX_TEST_TERMINAL_BEHAVIOR"] = "waiting-peer"
    process = subprocess.Popen([str(binary), "--handoff"], cwd=root, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        wait_for(lambda: (root / "peer-connected").exists())
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=3) == 3
        assert not (root / "started").exists()
        arguments = json.loads((root / "terminal-arguments.json").read_text())
        assert not Path(arguments[-1]).parent.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_tty_descriptors_without_a_controlling_terminal_are_insufficient(terminal_installation):
    import pty

    root, binary, env = terminal_installation
    master, slave = pty.openpty()
    try:
        result = subprocess.run([str(binary), "--available"], cwd=root, env=env, stdin=slave,
                                stdout=slave, stderr=slave, start_new_session=True, timeout=5)
        assert result.returncode == 3
    finally:
        os.close(slave)
        os.close(master)


def test_stale_handoff_socket_cannot_start_a_successor(terminal_installation):
    root, binary, env = terminal_installation
    stale = root / "stale"
    stale.mkdir(mode=0o700)
    result = subprocess.run([str(root / "commands/xdg-terminal-exec"), "--", str(binary),
                             "--terminal-manager", str(stale / "control")],
                            cwd=root, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, start_new_session=True, timeout=5)
    assert result.returncode == 3
    assert not (root / "started").exists()
