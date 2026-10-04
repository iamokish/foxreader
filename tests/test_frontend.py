"""The frontend domain's entry point: what the browser is handed.

Templates, the compressed bundle a compiled build serves them out of, the fonts
offered to the page, and the inline scripts -- run under node against a stub DOM.
As with the other two mains, this file verifies the domain's test files rather
than the frontend:

* ``test_assets`` and ``test_fonts`` are collected, import and hold tests.
  ``FOX_TEST_FULL=1`` runs them for real.

* six scripts under ``tests/manual/``, in dependency order, because two of them
  are inputs to the other two. ``render_settings_page.py`` and
  ``render_user_endpoints_page.py`` write the *real* rendered page's inline script
  to $TMPDIR, and the ``.mjs`` harnesses lift it from there and run it in
  ``node:vm``. That indirection is what stops the JS tests from drifting away
  from the template they claim to cover -- and it means running them out of order
  tests a stale file, or nothing at all. The runner in :mod:`tests.suites`
  handles that: each script names what it needs, and each runs once per session
  however many domains ask for it.

The node harnesses skip rather than fail when node is not installed, since a
backend-only checkout is a legitimate way to work here.
"""
import pytest

from tests import suites

DOMAIN = suites.FRONTEND

pytestmark = pytest.mark.skipif(
    suites.child_run(),
    reason=f"running under a main ({suites.CHILD_ENV} is set)",
)


def test_accounting():
    """Every test file under ``tests/`` is claimed by exactly one domain."""
    suites.check_accounting()


def test_modules_collect():
    """The frontend's pytest modules exist, import, and hold tests."""
    suites.check_modules_collect(DOMAIN)


def test_modules_pass():
    """The frontend's pytest modules pass. Opt-in with ``FOX_TEST_FULL=1``."""
    suites.run_domain_for_real(DOMAIN)


@pytest.mark.parametrize("script", suites.script_params(DOMAIN))
def test_script(script):
    """One frontend script -- a template render, or its script run in node."""
    suites.check_script(script)
