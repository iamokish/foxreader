"""Which test file belongs to which domain, and how the ones pytest cannot collect are run.

Three files are the suite's entry points -- :mod:`tests.test_backend`,
:mod:`tests.test_frontend` and :mod:`tests.test_build` -- and each verifies the
test files of its own domain. "Verify" means two different things, because there
are two kinds of test file here:

* **modules** -- ``tests/test_*.py``, which pytest already collects and runs. A
  main does not re-run them by default; that would double the suite for nothing.
  It asks pytest to *collect* them in a child process, which is what catches the
  failures a rename or a bad import cause: a module that no longer exists, one
  that raises on import, one that quietly stopped holding any tests. Set
  ``FOX_TEST_FULL=1`` and the main runs them for real, so
  ``pytest tests/test_backend.py`` becomes "run the whole backend domain".

* **scripts** -- ``tests/manual/*`` and ``tests/dist/*``, which pytest never
  collects: they are standalone programs that print their own checks and exit
  non-zero on failure. Nothing ran them, so nothing noticed when three of them
  broke. The mains run them, one pytest case each, in dependency order.

Every domain also asserts that nothing in ``tests/`` is unaccounted for -- in
either direction. Add a test file and forget to list it here and the suite fails
rather than silently ignoring it; list one that no longer exists and it fails the
same way.

Cost tiers, because a test suite nobody waits for is a test suite nobody runs:

* default -- everything that needs only the checkout. Roughly twenty-five seconds.
* ``FOX_TEST_SLOW=1`` -- the one script that reads every module in ``.venv`` to
  see what it asks metadata for. A minute warm, several cold. Worth running after
  a dependency upgrade, not on every save.
* ``FOX_TEST_DIST=1`` -- the scripts that drive the *compiled* distribution in
  ``packaging/build/stage``. These start real backends, kill every
  ``fox-reader.exe`` on the machine, download model weights and delete and
  re-download one of them. Minutes to tens of minutes, and a network. Skipped
  without the variable, and skipped anyway when there is no staged build.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
MANUAL_DIR = TESTS_DIR / "manual"
DIST_DIR = TESTS_DIR / "dist"

#: What ``packaging/build.py`` lays down. Absent until someone runs a build.
STAGE_DIR = REPO_ROOT / "packaging" / "build" / "stage" / "fox-reader"

#: Set in every child process the mains spawn. A main skips itself when it sees
#: it, so a child that is handed a directory instead of a file cannot recurse.
CHILD_ENV = "FOX_TEST_CHILD"
#: Run each domain's pytest modules for real instead of only collecting them.
FULL_ENV = "FOX_TEST_FULL"
#: Run the scripts that take a minute or more on their own. Read the module docstring.
SLOW_ENV = "FOX_TEST_SLOW"
#: Run the scripts that drive the compiled distribution. Read the module docstring.
DIST_ENV = "FOX_TEST_DIST"

#: Files in ``tests/`` that are not test files.
NOT_A_TEST = frozenset({"__init__.py", "conftest.py", "suites.py"})

_SCRIPT_SUFFIXES = frozenset({".py", ".mjs"})


class MissingTool(Exception):
    """A script needs an interpreter that is not on PATH."""


class PrerequisiteFailed(Exception):
    """A script another script depends on did not pass, so this one cannot mean anything."""


@dataclass(frozen=True)
class Script:
    """A test file pytest cannot collect: a program that exits 0 when it passes."""

    #: Path relative to ``tests/``, with forward slashes.
    path: str
    #: What it pins down. Shown when it fails, so make it worth reading.
    why: str
    #: Scripts that must run first, by path. Run on demand, once per session.
    needs: tuple[str, ...] = ()
    runner: str = "python"
    timeout: int = 600
    #: Takes a minute or more by itself, and so the env variable.
    slow: bool = False
    #: Needs the compiled distribution in ``STAGE_DIR``, and so the env variable.
    dist: bool = False
    #: Uses Windows APIs or tools and cannot run anywhere else.
    windows_only: bool = False
    #: Non-empty when the script is known to fail. It still runs, and still
    #: reports, but the suite stays green -- the point is that it stays visible.
    xfail: str = ""

    @property
    def file(self) -> Path:
        return TESTS_DIR / self.path

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Helper:
    """A file under ``tests/`` that a check uses but that is not itself a check."""

    path: str
    why: str

    @property
    def file(self) -> Path:
        return TESTS_DIR / self.path


@dataclass(frozen=True)
class Domain:
    """One main's territory."""

    name: str
    #: pytest modules this domain owns, without the ``.py``.
    modules: tuple[str, ...] = ()
    scripts: tuple[Script, ...] = ()
    helpers: tuple[Helper, ...] = field(default=())


# ── the frontend: what the browser is handed ──────────────────────────────────
# Templates, the bundle a compiled build serves them from, the fonts offered to
# the page, and the inline scripts run under node against a stub DOM. The two
# ``render_*`` scripts that write to $TMPDIR are prerequisites rather than
# curiosities: the node harnesses lift the real inline script out of the real
# rendered page, so nothing here can drift away from the template it tests.
FRONTEND = Domain(
    name="frontend",
    modules=(
        "test_assets",
        "test_fonts",
    ),
    scripts=(
        Script(
            path="manual/render_index_page.py",
            why="index.html renders, its inline script parses, and the notice sits "
                "outside #nocontextmenu",
        ),
        Script(
            path="manual/render_pages.py",
            why="every template renders with balanced tags, defined CSS variables and "
                "a back crumb",
        ),
        Script(
            path="manual/render_settings_page.py",
            why="the settings page renders; writes the script the node harness reads",
        ),
        Script(
            path="manual/smoke_settings_devices.mjs",
            why="the settings page's device picker, run against a stub DOM",
            needs=("manual/render_settings_page.py",),
            runner="node",
        ),
        Script(
            path="manual/render_user_endpoints_page.py",
            why="the user-endpoints page renders; writes the script the node harness reads",
        ),
        Script(
            path="manual/smoke_user_endpoints_editor.mjs",
            why="the endpoint editor's key generation and payload building, in node",
            needs=("manual/render_user_endpoints_page.py",),
            runner="node",
        ),
        Script(
            path="manual/render_setup_page.py",
            why="the setup wizard renders with the licence card and the optional section; "
                "writes the script the node harness reads",
        ),
        Script(
            # The backend refuses a download for any model whose notice is
            # unaccepted, so this card is the only way one starts. That makes it
            # the one inline script where a silent break is a compliance
            # problem, not a cosmetic one.
            path="manual/smoke_setup_terms.mjs",
            why="the per-model licence gate cannot be bypassed and each acceptance is "
                "recorded before its download, in node",
            needs=("manual/render_setup_page.py",),
            runner="node",
        ),
    ),
)

# ── the backend: what answers a request ───────────────────────────────────────
BACKEND = Domain(
    name="backend",
    modules=(
        "test_bubble",
        "test_clean_caps",
        "test_clean_methods",
        "test_clean_tuning",
        "test_clean_unetplusplus",
        "test_config",
        "test_device_vl_slot",
        "test_folder_workspace",
        "test_models",
        "test_mtl_characters",
        "test_mtl_context",
        "test_ocr_engine",
        "test_ocr_routes",
        "test_setup_downloads",
        "test_session_service",
        "test_system_routes",
        "test_typeset_layout",
    ),
    scripts=(
        Script(
            path="manual/smoke_device.py",
            why="device selection and the memory report, without blocking on a probe",
        ),
        Script(
            path="manual/smoke_device_settings_api.py",
            why="the device settings API round-trips and survives a bad value",
        ),
        Script(
            path="manual/smoke_ml_memory_route.py",
            why="the ML memory route reports what is loaded",
        ),
        Script(
            path="manual/smoke_user_endpoints.py",
            why="user endpoint storage, encryption at rest and validation",
        ),
        Script(
            path="manual/smoke_custom_endpoint_http.py",
            why="a custom endpoint's real HTTP request shape",
            timeout=900,
        ),
        Script(
            # The one cross-domain dependency in here, and the reason it is worth
            # the wiring: this feeds keys and a payload the *page* built to the
            # real Fernet and the real model, so a browser-side change that stops
            # producing something the backend accepts fails here rather than in a
            # user's saved endpoint.
            path="manual/smoke_endpoint_crypto_uuid.py",
            why="keys and payloads minted by the page are accepted by the real backend",
            needs=("manual/smoke_user_endpoints_editor.mjs",),
        ),
        Script(
            path="manual/smoke_mtl_memory.py",
            why="the MTL model's memory accounting",
            xfail="3 checks fail on the current tree; nothing ran this script, so it "
                  "went unnoticed. Registered as expected-fail to keep it visible "
                  "instead of excluded.",
        ),
        Script(
            path="manual/smoke_mtl_defaults.py",
            why="which MTL model is offered by default",
            xfail="3 checks fail on the current tree, same story as smoke_mtl_memory.",
        ),
        Script(
            path="manual/smoke_gemma_gguf.py",
            why="the Gemma GGUF loader and its quantisation choices",
            xfail="imports GEMMA_E4B_MODEL_DIR, which is now GEMMA_E4B_Q6_MODEL_DIR -- "
                  "the constant was renamed and this script was never run again.",
        ),
    ),
)

# ── the build: the pipeline, and what it produces ─────────────────────────────
# No pytest module of its own: a build cannot be exercised by importing it. The
# static half lives in test_build.py, which reads packaging/build.py without
# running it; the rest is these scripts, which need something already built.
BUILD = Domain(
    name="build",
    scripts=(
        Script(
            path="dist/cvcheck.py",
            why="opencv-contrib is the installed variant and its contrib modules are real",
        ),
        Script(
            path="dist/scan_meta.py",
            why="reports every distribution-metadata lookup in the bundled tree -- the "
                "scan that found the huggingface-hub startup crash",
            slow=True,
            timeout=900,
        ),
        Script(
            path="dist/pescan.py",
            why="every MSVC runtime DLL the staged binaries import is bundled next to them",
            dist=True,
            windows_only=True,
        ),
        Script(
            path="dist/repro.py",
            why="the staged backend starts and keeps running; prints the tail of its output",
            dist=True,
            windows_only=True,
            timeout=600,
        ),
        Script(
            path="dist/verify_metadata.py",
            why="the compiled backend starts with models present, and re-downloads one "
                "removed behind a running server",
            dist=True,
            windows_only=True,
            timeout=1800,
        ),
        Script(
            path="dist/verify_download.py",
            why="first-run model download through the compiled binary, over Xet rather "
                "than the HTTP fallback",
            dist=True,
            windows_only=True,
            timeout=2400,
        ),
        Script(
            path="dist/smoke_dist.py",
            why="the launcher: one backend, fallback fonts, a second launcher attaches, "
                "closing the owner stops the backend and clears cache/",
            dist=True,
            windows_only=True,
            timeout=1200,
        ),
        Script(
            path="dist/time_shutdown.py",
            why="a graceful shutdown finishes well inside the launcher's grace window",
            dist=True,
            windows_only=True,
            timeout=900,
        ),
    ),
    helpers=(
        Helper(
            path="dist/hold_lock.py",
            why="holds .fox-reader.lock the way launcher.c does, for a peer-launcher "
                "case; takes a path and a duration on argv",
        ),
        Helper(
            path="dist/probe_payload.py",
            why="compiled by Nuitka by hand when embedded metadata is in question. "
                "Meaningless run as a script -- the answer is different once compiled.",
        ),
    ),
)

DOMAINS = (BACKEND, BUILD, FRONTEND)

#: Every main, so each domain can leave the others out of its own accounting.
MAINS = frozenset(f"test_{domain.name}.py" for domain in DOMAINS)

SCRIPTS: dict[str, Script] = {
    script.path: script for domain in DOMAINS for script in domain.scripts
}
HELPERS: dict[str, Helper] = {
    helper.path: helper for domain in DOMAINS for helper in domain.helpers
}


# ── running things ────────────────────────────────────────────────────────────

#: One result per script per session. Two domains can want the same script -- the
#: endpoint editor harness is the frontend's own test and the backend's
#: prerequisite -- and running it twice would only make the suite slower.
_RESULTS: dict[str, subprocess.CompletedProcess[str]] = {}


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env[CHILD_ENV] = "1"
    # These scripts print box drawing and the odd non-ASCII name. Without this a
    # child on Windows dies in cp1252 halfway through its own output, which reads
    # as a failing check rather than as an encoding accident.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _command(script: Script) -> list[str]:
    if script.runner == "node":
        node = shutil.which("node")
        if node is None:
            raise MissingTool("node is not on PATH")
        return [node, str(script.file)]
    return [sys.executable, "-u", str(script.file)]


def run_script(script: Script) -> subprocess.CompletedProcess[str]:
    """Run ``script`` and whatever it needs, at most once each, and return the result."""
    cached = _RESULTS.get(script.path)
    if cached is not None:
        return cached

    for path in script.needs:
        prerequisite = SCRIPTS[path]
        result = run_script(prerequisite)
        if result.returncode != 0:
            raise PrerequisiteFailed(
                f"{script.name} needs {prerequisite.name}, which exited "
                f"{result.returncode}:\n{tail(result)}"
            )

    if not script.file.is_file():
        raise FileNotFoundError(script.file)

    completed = subprocess.run(
        _command(script),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_child_env(),
        timeout=script.timeout,
    )
    _RESULTS[script.path] = completed
    return completed


def tail(result: subprocess.CompletedProcess[str], lines: int = 25) -> str:
    """The end of a script's output, indented, for a failure message."""
    text = (result.stdout or "") + (result.stderr or "")
    kept = text.replace("\r\n", "\n").rstrip().splitlines()[-lines:]
    return "\n".join(f"    {line}" for line in kept) or "    (no output)"


def run_pytest(arguments: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    """Run pytest in a child, marked as a child so the mains skip themselves."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *arguments],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_child_env(),
        timeout=timeout,
    )


def collect(domain: Domain) -> tuple[subprocess.CompletedProcess[str], dict[str, int]]:
    """Ask pytest to collect the domain's modules, and count what it found per module."""
    paths = [f"tests/{name}.py" for name in domain.modules]
    result = run_pytest(["--collect-only", "-q", *paths], timeout=600)

    found: dict[str, int] = {name: 0 for name in domain.modules}
    for line in (result.stdout or "").splitlines():
        if "::" not in line:
            continue
        stem = Path(line.split("::", 1)[0].replace("\\", "/")).stem
        if stem in found:
            found[stem] += 1
    return result, found


# ── accounting ────────────────────────────────────────────────────────────────

def _declared() -> tuple[set[str], set[str]]:
    modules = {f"{name}.py" for domain in DOMAINS for name in domain.modules}
    scripts = set(SCRIPTS) | set(HELPERS)
    return modules, scripts


def _on_disk() -> tuple[set[str], set[str]]:
    modules = {
        path.name
        for path in TESTS_DIR.glob("test_*.py")
        if path.name not in MAINS and path.name not in NOT_A_TEST
    }
    scripts = set()
    for directory in (MANUAL_DIR, DIST_DIR):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in _SCRIPT_SUFFIXES:
                continue
            if "__pycache__" in path.parts or path.name in NOT_A_TEST:
                continue
            scripts.add(path.relative_to(TESTS_DIR).as_posix())
    return modules, scripts


def accounting_problems() -> list[str]:
    """Every test file under ``tests/`` that no domain claims, and every claim with no file.

    Both directions matter. An unclaimed file is a test nobody runs, which is how
    three of the scripts in here came to be broken without anyone knowing. A claim
    with no file behind it is a rename that left this map stale.
    """
    declared_modules, declared_scripts = _declared()
    disk_modules, disk_scripts = _on_disk()

    problems = []
    for name in sorted(disk_modules - declared_modules):
        problems.append(f"tests/{name} is collected by pytest but no domain lists it")
    for name in sorted(declared_modules - disk_modules):
        problems.append(f"a domain lists tests/{name}, which does not exist")
    for name in sorted(disk_scripts - declared_scripts):
        problems.append(f"tests/{name} exists but no domain lists it as a script or a helper")
    for name in sorted(declared_scripts - disk_scripts):
        problems.append(f"a domain lists tests/{name}, which does not exist")

    claimed: dict[str, list[str]] = {}
    for domain in DOMAINS:
        for name in domain.modules:
            claimed.setdefault(f"{name}.py", []).append(domain.name)
        for script in domain.scripts:
            claimed.setdefault(script.path, []).append(domain.name)
    for name, owners in sorted(claimed.items()):
        if len(owners) > 1:
            problems.append(f"tests/{name} is claimed by more than one domain: {', '.join(owners)}")

    return problems


# ── the shared bodies of the three mains ──────────────────────────────────────

def child_run() -> bool:
    """Whether this pytest was started by one of the mains.

    The mains skip themselves when it is true. A child is always given specific
    files, so it should never reach a main at all -- this is the belt to that
    braces, and it means a stray `pytest tests/` from inside a main cannot fork
    forever.
    """
    return bool(os.environ.get(CHILD_ENV))


def check_accounting() -> None:
    problems = accounting_problems()
    assert not problems, "the domain map and tests/ disagree:\n" + "\n".join(
        f"  - {problem}" for problem in problems
    )


def check_modules_collect(domain: Domain) -> None:
    """The domain's pytest modules exist, import, and hold tests."""
    if not domain.modules:
        pytest.skip(f"the {domain.name} domain owns no pytest modules")

    result, found = collect(domain)
    assert result.returncode == 0, (
        f"pytest could not collect the {domain.name} modules:\n{tail(result)}"
    )
    empty = sorted(name for name, count in found.items() if count == 0)
    assert not empty, (
        f"these {domain.name} modules were collected but hold no tests: {', '.join(empty)}\n"
        f"{tail(result)}"
    )


def run_domain_for_real(domain: Domain) -> None:
    """Run the domain's pytest modules in a child. Opt-in: see FULL_ENV."""
    if not os.environ.get(FULL_ENV):
        pytest.skip(f"set {FULL_ENV}=1 to run the {domain.name} modules, not just collect them")
    if not domain.modules:
        pytest.skip(f"the {domain.name} domain owns no pytest modules")

    result = run_pytest(["-q", *[f"tests/{name}.py" for name in domain.modules]])
    assert result.returncode == 0, f"the {domain.name} modules failed:\n{tail(result)}"


def check_script(script: Script) -> None:
    """Run one script and hold it to its exit code."""
    if script.windows_only and os.name != "nt":
        pytest.skip(f"{script.name} uses Windows-only tools")
    if script.slow and not os.environ.get(SLOW_ENV):
        pytest.skip(f"set {SLOW_ENV}=1 to run {script.name}; it reads every module in .venv")
    if script.dist:
        if not os.environ.get(DIST_ENV):
            pytest.skip(f"set {DIST_ENV}=1 to run {script.name} against a compiled build")
        if not STAGE_DIR.is_dir():
            pytest.skip(f"no staged build at {STAGE_DIR}; run packaging/build.py first")

    try:
        result = run_script(script)
    except MissingTool as exc:
        pytest.skip(f"{script.name}: {exc}")
    except PrerequisiteFailed as exc:
        pytest.fail(str(exc))
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name} did not finish inside {script.timeout}s")

    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}.\n"
        f"  it checks: {script.why}\n{tail(result)}"
    )


def script_params(domain: Domain) -> list:
    """One pytest case per script, in the order the domain lists them.

    Returns ``pytest.param`` values -- the type is private to pytest, hence the
    bare ``list``.
    """
    params = []
    for script in domain.scripts:
        marks = []
        if script.xfail:
            # Not strict: a fix should show up as XPASS, not as a new failure.
            marks.append(pytest.mark.xfail(reason=script.xfail, strict=False))
        params.append(pytest.param(script, marks=marks, id=script.name))
    return params
