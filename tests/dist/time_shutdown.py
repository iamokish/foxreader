"""How long a graceful backend shutdown actually takes, against the launcher's own budget.

Sizing SHUTDOWN_GRACE_MS is a guess without this number. Too generous and the OS
kills the launcher mid-wait and the cache is never cleared; too mean and the
launcher force-kills a backend that was about to finish cleanly on its own.

The two budgets are read out of launcher.c rather than written down here, so
changing one of them there is what moves this test:

  SHUTDOWN_GRACE_MS  the normal path -- the user asked to quit, we can wait
  CLOSE_GRACE_MS     the console close handler, which the OS is timing

A shutdown inside the close budget needs nothing from anyone. Between the two,
the normal path is still clean and the close path force-kills -- which is by
design, and smoke_dist.py checks the outcome of that (backend gone, cache
cleared). Past the shutdown budget, both paths force-kill and this fails.

Runs the backend directly rather than through the launcher, so no shutdown token
is set and POST /api/shutdown is accepted unauthenticated.
"""
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "packaging" / "build" / "stage" / "fox-reader"
BACKEND = STAGE / "bin" / "fox-reader.exe"
LAUNCHER_C = ROOT / "packaging" / "launcher" / "launcher.c"
URL = "http://127.0.0.1:7954"


def budget(name, fallback):
    """Read a #define out of launcher.c, in seconds."""
    try:
        source = LAUNCHER_C.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback
    found = re.search(rf"^#define\s+{name}\s+(\d+)", source, re.MULTILINE)
    return int(found.group(1)) / 1000 if found else fallback


if not BACKEND.is_file():
    print(f"no compiled backend at {BACKEND}; run packaging/build.py first")
    sys.exit(1)

shutdown_grace = budget("SHUTDOWN_GRACE_MS", 20.0)
close_grace = budget("CLOSE_GRACE_MS", 2.5)
print(f"launcher budgets: shutdown {shutdown_grace:.1f}s, console close {close_grace:.1f}s")

proc = subprocess.Popen(
    [str(BACKEND)],
    cwd=str(STAGE),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

start = time.time()
while time.time() - start < 240:
    if proc.poll() is not None:
        print(f"backend exited on its own, rc={proc.returncode}")
        sys.exit(1)
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=2) as response:
            if response.status == 200:
                break
    except Exception:
        pass
    time.sleep(0.5)
else:
    print("never became healthy")
    proc.kill()
    sys.exit(1)

print(f"healthy after {time.time() - start:.1f}s")

cache = STAGE / "cache"
cache.mkdir(exist_ok=True)
(cache / "probe.bin").write_bytes(b"x" * 1024)
(cache / "nested").mkdir(exist_ok=True)
(cache / "nested" / "deep.bin").write_bytes(b"y" * 512)
print(f"cache entries before: {len(list(cache.rglob('*')))}")

asked = time.time()
try:
    request = urllib.request.Request(f"{URL}/api/shutdown", method="POST", data=b"")
    with urllib.request.urlopen(request, timeout=10) as response:
        print(f"POST /api/shutdown -> {response.status} {response.read().decode()[:120]}")
except Exception as exc:
    print(f"shutdown request failed: {exc}")

try:
    code = proc.wait(timeout=180)
except subprocess.TimeoutExpired:
    print("backend never exited; killing it")
    proc.kill()
    raise SystemExit(1) from None

took = time.time() - asked
print(f"backend exited rc={code} {took:.2f}s after the POST")
print(f"cache entries after: {len(list(cache.rglob('*')))}")

if took > shutdown_grace:
    print(f"\nFAIL: {took:.2f}s is past the launcher's {shutdown_grace:.1f}s grace, so even a "
          "normal quit force-kills it")
    sys.exit(1)

margin = "inside the console close budget too" if took <= close_grace else (
    f"over the {close_grace:.1f}s console close budget, so that path force-kills by design")
print(f"\nPASSED: graceful shutdown in {took:.2f}s -- {margin}")
