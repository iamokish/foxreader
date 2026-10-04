"""The Nuitka entry point.

Nuitka compiles this file and follows everything it imports. It is separate from
``src/fox_reader/__main__.py`` because a compiled build has to settle a few
things before any of the application is imported, and ``python -m fox_reader``
should not pay for any of them.

The build copies this file into a scratch directory next to the generated
``fox_reader_assets.py`` and points Nuitka at the copy, so the asset bundle is
importable from the entry's own directory. See ``packaging/build.py``.
"""
from __future__ import annotations

import os
import sys

#: Directories inside ``bin/`` holding DLLs that other DLLs in the tree link
#: against, registered on the process DLL search path at startup. Windows only;
#: on Linux and macOS the equivalent is an RPATH, which is set at compile time.
#:
#: This exists because the build deduplicates: llama.cpp links the same CUDA
#: runtime torch ships, and rather than carry ~700 MB twice, ``bundle`` in
#: ``packaging/llamacpp.py`` skips any library already somewhere in the tree. The
#: copy that is kept is torch's, in ``torch/lib``, and Windows will not look
#: there for a dependency of ``llama_cpp/lib/llama.dll`` unless it is told to.
#:
#: Each package does register its own directory when it is imported -- that is
#: how the kept copy would be found at all -- but only its own, and only if it
#: is imported first. Nothing guarantees that order: the GGUF translator does
#: not import torch, so a run that never touches a torch-backed feature would
#: reach ``llama_cpp`` with ``torch/lib`` unregistered. Doing it here, once, for
#: every such directory, makes the order irrelevant.
_DLL_DIRECTORIES = ("torch/lib", "llama_cpp/lib")


def _register_dll_directories() -> None:
    """Put the tree's library directories on the DLL search path.

    ``os.add_dll_directory`` is process-wide and additive (it is
    ``AddDllDirectory`` plus ``LOAD_LIBRARY_SEARCH_USER_DIRS``), which is the
    same mechanism torch and llama_cpp use for their own directories, so this
    neither replaces nor conflicts with what they do.

    Silent about everything: a directory that is not there is the normal state of
    a build that left that backend out, and a failure to register one is not
    worth refusing to start over -- the import that needs it will say so itself,
    with a message about the library it could not load.
    """
    if os.name != "nt":
        return
    add = getattr(os, "add_dll_directory", None)
    if add is None:  # Python 3.7 and earlier; not a supported build, but free.
        return

    root = os.path.dirname(os.path.abspath(sys.executable))
    for relative in _DLL_DIRECTORIES:
        path = os.path.join(root, *relative.split("/"))
        if not os.path.isdir(path):
            continue
        try:
            add(path)
        except OSError:
            pass


def _mark_frozen() -> None:
    """Make ``sys.frozen`` true, since ``fox_reader.utils`` branches on it.

    Nuitka's ``--standalone`` sets this itself. It is set again here because the
    consequence of it being missing is not an error but a wrong answer: without
    it ``PROJECT_ROOT`` resolves by looking for ``pyproject.toml`` instead of
    from ``bin/``, and the backend would read config and models out of whatever
    directory it happened to be started from. Cheap insurance against a Nuitka
    version that stops setting it, and a no-op when it already did.

    ``__compiled__`` is the marker Nuitka defines unconditionally; it is checked
    rather than assumed so that running this file with a plain interpreter --
    which is a useful thing to be able to do -- still behaves like a checkout.
    """
    if not getattr(sys, "frozen", False) and "__compiled__" in globals():
        sys.frozen = True  # type: ignore[attr-defined]


def _force_utf8_output() -> None:
    """Stop a log line from being able to kill the process.

    The launcher sets ``PYTHONIOENCODING``, but this binary can also be started
    by hand, and then stdout is the console's code page -- cp1252 on a Western
    Windows install. One log line naming a file with a CJK character in it raises
    UnicodeEncodeError inside logging, which surfaces as a crash whose message is
    about encoding rather than about whatever was actually happening.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            # Not a text stream, or already closed -- neither is worth failing over.
            pass


def main() -> None:
    # Before anything else. On Windows a spawned multiprocessing child re-runs
    # this executable from the top, and freeze_support is what makes it hand over
    # to the worker instead of starting a second web server. Nothing in the app
    # spawns one today, but torch's DataLoader and paddle both can, and the
    # failure mode -- an infinite fan-out of processes each trying to bind the
    # port -- is bad enough to be worth two lines.
    import multiprocessing

    multiprocessing.freeze_support()

    _mark_frozen()
    _force_utf8_output()
    _register_dll_directories()

    # Imported here, not at module scope: everything above has to have run first,
    # because `fox_reader.utils` reads `sys.frozen` at import time to decide where
    # the user's config, models and cache live.
    from fox_reader.__main__ import main as run

    run()


if __name__ == "__main__":
    main()
