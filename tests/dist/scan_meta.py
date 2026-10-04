"""Find runtime distribution-metadata lookups in the bundled dependency tree.

This is the scan that found the reported startup crash. Something in the tree calls
`importlib.metadata.version("huggingface-hub")`; the distribution declares itself
`huggingface_hub`; a real install does not care because `importlib.metadata`
flattens dashes, underscores and case (PEP 503), and a Nuitka build does, because
its embedded metadata is a dict keyed by the declared name and read with an exact
lookup. So the interesting column here is not "who looks something up" but "who
looks something up under a spelling that is not the one the distribution declares".

A report, not a gate. The gate is in `packaging/build.py`: `bundle_metadata` proves
every lookup resolves before it packages anything, and `test_build.py` checks the
curated list statically. This is what you run when a *new* one appears -- after a
dependency upgrade, say -- because a literal string in someone else's source is the
only warning you get.

Fails only if it cannot scan: no site-packages, or a scan that finds nothing at all.
"""
import ast
import collections
import importlib.metadata as im
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Functions that hit distribution metadata with a distribution *name*.
FUNCS = {
    "version", "distribution", "metadata", "requires", "files",
    "get_distribution", "get_provider", "require",
    "PackageNotFoundError",
}
# Packages that are never bundled, so their lookups cannot fire at runtime.
SKIP_TOP = {
    "pip", "pytest", "_pytest", "ruff", "setuptools", "pkg_resources",
    "wheel", "uv", "nuitka", "pylint",
    "coverage", "mypy", "black", "isort", "flake8", "sphinx",
}


def site_packages():
    """The checkout's virtualenv site-packages, on either layout."""
    windows = ROOT / ".venv" / "Lib" / "site-packages"
    if windows.is_dir():
        return windows
    posix = sorted((ROOT / ".venv" / "lib").glob("python3*/site-packages"))
    return posix[-1] if posix else None


def normalize(name):
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


SP = site_packages()
if SP is None:
    print(f"no site-packages under {ROOT / '.venv'}; run uv sync first")
    sys.exit(1)

print(f"scanning {SP}\n")

hits = collections.defaultdict(set)   # dist name -> {"pkg  (file:line)", ...}
for path in SP.rglob("*.py"):
    try:
        rel = path.relative_to(SP)
    except ValueError:
        continue
    top = rel.parts[0].split(".")[0]
    if top in SKIP_TOP or "test" in rel.parts or "tests" in rel.parts:
        continue
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
        if name not in FUNCS:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            val = arg.value.strip()
            # A distribution name, not a path or a dotted module.
            if val and "/" not in val and "\\" not in val and " " not in val:
                hits[val].add(f"{top}  ({rel}:{node.lineno})")

if not hits:
    print("FAIL: found no metadata lookups at all -- the scan is broken, not the tree")
    sys.exit(1)

#: Declared name of every installed distribution, by normalised key.
declared = {}
for dist in im.distributions():
    try:
        real = dist.metadata["Name"]
    except Exception:
        continue
    if real:
        declared.setdefault(normalize(real), real)

mismatched, exact, uninstalled = [], [], []
for asked in sorted(hits):
    real = declared.get(normalize(asked))
    if real is None:
        uninstalled.append(asked)
    elif real == asked:
        exact.append(asked)
    else:
        mismatched.append((asked, real))

print("── asked for under a spelling the distribution does not declare ──")
print("   (these are the ones a compiled build gets wrong)\n")
for asked, real in mismatched:
    print(f"{asked}   -- installed as {real}")
    for src in sorted(hits[asked])[:3]:
        print(f"      {src}")

print(f"\n── asked for by their declared name ({len(exact)}) ──")
print("   " + ", ".join(exact))

print(f"\n── asked for but not installed here ({len(uninstalled)}) ──")
print("   " + ", ".join(uninstalled))

print(f"\nPASSED: scanned the tree, {len(hits)} distinct names, "
      f"{len(mismatched)} of them spelled differently from the install")
