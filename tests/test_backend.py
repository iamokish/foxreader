"""The backend domain's entry point: what answers a request.

Routes, services, models, config, OCR, cleaning, typesetting, bubble detection --
everything on the Python side of a request. This file does not test any of that
itself. It verifies the test files that do, which is a different job:

* the nine ``tests/test_*.py`` modules the domain owns are collected by pytest,
  import cleanly and still hold tests. Not re-run by default -- ``pytest tests/``
  already runs them, and running them twice buys nothing. ``FOX_TEST_FULL=1``
  turns this file into "run the whole backend domain", which is what you want
  when you run ``pytest tests/test_backend.py`` on its own.

* the nine standalone smoke scripts under ``tests/manual/``, which pytest never
  collects and which nothing therefore ran. Three of them had quietly broken --
  one imports a constant that was renamed. They run here, one case each, and the
  three that fail are marked expected-fail: still run, still reported, and the
  reason is on the case rather than in a comment nobody reads.

The domain map itself is in :mod:`tests.suites`, along with the runner. Adding a
test file to ``tests/`` without listing it there fails ``test_accounting`` in all
three mains, which is the point: an unlisted test file is a test nobody runs.
"""
import pytest

from tests import suites

DOMAIN = suites.BACKEND

pytestmark = pytest.mark.skipif(
    suites.child_run(),
    reason=f"running under a main ({suites.CHILD_ENV} is set)",
)


def test_accounting():
    """Every test file under ``tests/`` is claimed by exactly one domain."""
    suites.check_accounting()


def test_modules_collect():
    """The backend's pytest modules exist, import, and hold tests."""
    suites.check_modules_collect(DOMAIN)


def test_modules_pass():
    """The backend's pytest modules pass. Opt-in with ``FOX_TEST_FULL=1``."""
    suites.run_domain_for_real(DOMAIN)


@pytest.mark.parametrize("script", suites.script_params(DOMAIN))
def test_script(script):
    """One backend smoke script, run in its own process, held to its exit code."""
    suites.check_script(script)
