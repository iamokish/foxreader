"""Where things live.

Two roots, and conflating them is the bug this module exists to prevent:

* ``PROJECT_ROOT`` -- the user's data: ``config/``, ``fonts/``, ``models/``,
  ``cache/``. In a checkout that is the repository. In a compiled build it is
  the directory *above* ``bin/``, because the shipped tree puts every binary in
  ``bin/`` and everything the user owns beside it::

      launcher            <- starts bin/fox-reader
      bin/                <- the compiled backend and its libraries
      cache/  config/  fonts/  models/

* the code and the frontend, which in a compiled build are *inside* the binary
  and have no directory at all. Ask :mod:`fox_reader.assets` for those; the
  ``FRONTEND_DIR`` constants below are the checkout-only fallback it uses.

The model weights are deliberately not in the binary -- they are ~GB, they are
downloaded by the setup wizard, and re-shipping them on every release would
make the download pointless.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: True in a Nuitka ``--standalone`` build (and in a PyInstaller one, which set
#: the same flag). Nuitka also defines ``__compiled__`` on every compiled
#: module, but that is true of ``--module``/accelerated builds too, where
#: ``sys.executable`` is a plain interpreter and the layout below would not
#: apply -- so ``sys.frozen`` is the one to branch on.
is_frozen = bool(getattr(sys, "frozen", False))

#: Set by the launcher so the backend never has to guess. An explicit answer
#: also covers the case where someone moves the executable out of ``bin/``.
ROOT_ENV = "FOX_READER_ROOT"

_this_dir = Path(__file__).resolve().parent


def _frozen_root() -> Path:
    override = os.environ.get(ROOT_ENV, "").strip()
    if override:
        candidate = Path(override).expanduser()
        try:
            if candidate.is_dir():
                return candidate.resolve()
        except OSError:  # a path the OS refuses to stat is not an answer
            pass
        logger.warning("%s=%s is not a usable directory; ignoring.", ROOT_ENV, override)

    exe_dir = Path(sys.executable).resolve().parent
    # The shipped layout. Anything else -- a build run in place, a binary copied
    # somewhere by hand -- keeps its own directory as the root, which is what
    # the PyInstaller layout did and what a lone executable should do.
    if exe_dir.name.lower() == "bin":
        return exe_dir.parent
    return exe_dir


def _source_root() -> Path:
    # src/fox_reader/utils.py -> repository root
    candidate = _this_dir.parent.parent.parent
    if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
        return candidate
    return Path.cwd()


PROJECT_ROOT = _frozen_root() if is_frozen else _source_root()

# ── User data ─────────────────────────────────────────────────────────────────

CONFIG_DIR = PROJECT_ROOT / "config"
FONT_DIR = PROJECT_ROOT / "fonts"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"

USER_DIRS = (CONFIG_DIR, FONT_DIR, MODELS_DIR, CACHE_DIR)

# ── Frontend, in a checkout ───────────────────────────────────────────────────
# A compiled build has none of these; `fox_reader.assets` reads the embedded
# copies instead and only falls back here.

FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"


def ensure_user_dirs() -> None:
    """Create the four user directories, so the tree looks right on first run.

    Best effort: a read-only install still starts, and every consumer creates
    what it needs on demand anyway.
    """
    for directory in USER_DIRS:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create %s: %s", directory, exc)


def clear_cache() -> int:
    """Empty ``cache/``, keeping the directory. Returns how many entries went.

    Called at shutdown. Everything in there is a recomputable intermediate --
    inpainting masks, chiefly -- so carrying it across runs buys nothing and a
    stale entry from a crashed run is worse than no entry.

    Deliberately paranoid about *what* it deletes: it is a recursive delete
    driven by a path that was derived from ``sys.executable``, so it refuses to
    run unless the directory is really the one named ``cache``, and it never
    follows a symlink out of it.
    """
    directory = CACHE_DIR
    if directory.name != "cache" or not directory.is_dir():
        return 0

    removed = 0
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        logger.debug("Could not list %s: %s", directory, exc)
        return 0

    for entry in entries:
        try:
            # is_dir() follows symlinks, and a symlinked directory must be
            # unlinked rather than walked -- otherwise a link planted in cache/
            # would aim rmtree at whatever it points to.
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            removed += 1
        except OSError as exc:
            # A file another process still has open on Windows, most likely.
            # Not a reason to fail a shutdown.
            logger.debug("Could not remove %s: %s", entry, exc)

    return removed
