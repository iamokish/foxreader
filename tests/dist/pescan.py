"""Every MSVC runtime DLL the staged binaries import is somewhere the loader will look.

The Visual C++ runtime is not part of Windows. Several of the compiled extensions
link against `msvcp140.dll`, which arrives with the Visual C++ Redistributable --
and a machine that has ever had Visual Studio on it has it, which is exactly why
this is easy to ship broken. So rather than trusting a list, this reads the import
directory back out of every PE file the build produced and asks whether what they
name is there.

"There" is three directories, not one: `bin/`, the importer's own, and the
`<package>.libs/` a delvewheel-repaired wheel brings its private copies in. See
`resolves`. Windows' own `msvcrt.dll` is exempt -- it is in System32 on every
machine and must not be shipped.

The build does the same thing (`bundle_runtime_dlls`); this is the independent
check on its output, parsing the headers directly rather than through whatever the
build used to make its decision.
"""
import collections
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "packaging" / "build" / "stage" / "fox-reader" / "bin"

RUNTIME_PREFIXES = ("msvcp", "vcruntime", "vcomp", "concrt", "msvcr")

# `msvcrt.dll` is Windows' own C runtime. It lives in System32 on every supported
# release, it is not part of any redistributable, and shipping a copy would be worse
# than not: the loader takes the system one anyway. The `msvcr` prefix above is there
# for the versioned ones -- msvcr100, msvcr120 -- which really do have to travel.
SYSTEM = frozenset({"msvcrt.dll"})


def imports(path):
    """Names in a PE file's import directory. Empty on anything unparsable."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    if data[:2] != b"MZ":
        return []
    try:
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return []
        nsec, = struct.unpack_from("<H", data, pe + 6)
        opt_size, = struct.unpack_from("<H", data, pe + 20)
        opt = pe + 24
        magic, = struct.unpack_from("<H", data, opt)
        ndir_off = opt + (108 if magic == 0x20B else 92)
        ndir, = struct.unpack_from("<I", data, ndir_off)
        if ndir < 2:
            return []
        imp_rva, = struct.unpack_from("<I", data, ndir_off + 4 + 8)
        if not imp_rva:
            return []
        secs = []
        base = opt + opt_size
        for i in range(nsec):
            s = base + i * 40
            vsize, vaddr, rsize, raddr = struct.unpack_from("<IIII", data, s + 8)
            secs.append((vaddr, max(vsize, rsize), raddr))

        def to_off(rva):
            for vaddr, size, raddr in secs:
                if vaddr <= rva < vaddr + size:
                    return raddr + (rva - vaddr)
            return None

        out, off = [], to_off(imp_rva)
        if off is None:
            return []
        while True:
            entry = data[off:off + 20]
            if len(entry) < 20 or entry == b"\0" * 20:
                break
            name_rva = struct.unpack_from("<I", entry, 12)[0]
            noff = to_off(name_rva)
            if noff:
                end = data.index(b"\0", noff)
                out.append(data[noff:end].decode("ascii", "replace"))
            off += 20
        return out
    except Exception:
        return []


if not BIN.is_dir():
    print(f"no staged bin/ at {BIN}; run packaging/build.py first")
    sys.exit(1)

need = collections.defaultdict(list)
scanned = 0
for path in BIN.rglob("*"):
    if path.suffix.lower() not in (".dll", ".pyd", ".exe"):
        continue
    scanned += 1
    for name in imports(path):
        low = name.lower()
        if low.startswith(RUNTIME_PREFIXES):
            need[low].append(str(path.relative_to(BIN)))

print(f"scanned {scanned} PE files\n")
if not scanned:
    print("FAIL: nothing to scan -- the staged bin/ holds no PE files")
    sys.exit(1)

_listings: dict[Path, set[str]] = {}


def names_in(directory):
    """Lowercased file names in a directory, read once. Empty if it is not there."""
    if directory not in _listings:
        try:
            _listings[directory] = {p.name.lower() for p in directory.iterdir() if p.is_file()}
        except OSError:
            _listings[directory] = set()
    return _listings[directory]


def resolves(dll, importer):
    """Where the loader would find `dll` for `importer`, or None if nowhere.

    Three places count, and only three:

    * `bin/`, which holds the executable, so it is on the search path for the whole
      tree. This is where `bundle_runtime_dlls` puts things.
    * the importer's own directory, which the loader searches for a module loaded by
      full path.
    * `<top-level-package>.libs/`, where a delvewheel-repaired wheel keeps its private
      hash-suffixed copies -- numpy and pandas each carry their own `msvcp140-<hash>.dll`
      -- and which the package adds to the search path itself before importing anything.

    A copy anywhere else in the tree does not count, however reassuring it looks.
    """
    if dll in names_in(BIN):
        return "bin/"
    local = (BIN / importer).parent
    if dll in names_in(local):
        return f"{local.relative_to(BIN).as_posix()}/"
    parts = Path(importer).parts
    if len(parts) > 1:
        libs = BIN / f"{parts[0]}.libs"
        if dll in names_in(libs):
            return f"{libs.name}/"
    return None


missing = []
for dll in sorted(need):
    importers = sorted(need[dll])
    if dll in SYSTEM:
        print(f"{dll:44} system, ships with Windows   ({len(importers)} importers)")
        continue
    found = {importer: resolves(dll, importer) for importer in importers}
    unresolved = [importer for importer, where in found.items() if where is None]
    places = sorted({where for where in found.values() if where})
    print(f"{dll:44} {'*** MISSING ***' if unresolved else 'OK in ' + ', '.join(places)}"
          f"   ({len(importers)} importers)")
    for who in (unresolved or importers)[:4]:
        print(f"      {who}" + ("" if unresolved else f"   -> {found[who]}"))
    if unresolved:
        missing.append(dll)

if missing:
    print(f"\nFAIL: imported but not bundled: {', '.join(missing)}")
    print("      the build runs on a machine with the redistributable installed and")
    print("      the user's does not, which is the whole failure mode")
    sys.exit(1)

print(f"\nPASSED: every runtime DLL the tree imports ({len(need)}) is somewhere the loader looks")
