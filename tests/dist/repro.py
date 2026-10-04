"""The staged backend starts and keeps running -- the minimal form of the reported bug.

This is the smallest thing that used to fail: run `bin/fox-reader.exe` in the
staged tree and watch it die with `PackageNotFoundError: No package metadata was
found for huggingface-hub` before it ever bound a port. No wizard, no download, no
launcher -- just the binary and whatever is in `models/`.

verify_metadata.py covers this too, as its second section, but it takes fifteen
minutes and deletes a model on the way. When a build is suspect this is the thirty
second answer, and its output is the tail of the backend's own log.
"""
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "packaging" / "build" / "stage" / "fox-reader"
BACKEND = STAGE / "bin" / "fox-reader.exe"
URL = "http://127.0.0.1:7954"
LOG = Path(tempfile.gettempdir()) / "fox-reader-tests" / "repro.log"
DEADLINE = 300

if not BACKEND.is_file():
    print(f"no compiled backend at {BACKEND}; run packaging/build.py first")
    sys.exit(1)

LOG.parent.mkdir(parents=True, exist_ok=True)

models = sorted(p.name for p in (STAGE / "models").iterdir()) if (STAGE / "models").is_dir() else []
print(f"models present: {models or '(none -- the crash needs at least one)'}")

for name in ("fox-reader.exe", "launcher.exe"):
    subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True)
time.sleep(1)

# To a file rather than a pipe: paddle is chatty at startup, and a pipe nobody is
# draining while this polls the health endpoint would fill and stall the backend
# in the middle of the thing being measured.
handle = LOG.open("w", encoding="utf-8", errors="replace")
proc = subprocess.Popen([str(BACKEND)], cwd=str(STAGE), stdout=handle,
                        stderr=subprocess.STDOUT)

started = time.time()
healthy = False
while time.time() - started < DEADLINE:
    if proc.poll() is not None:
        break
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=2) as response:
            if response.status == 200:
                healthy = True
                break
    except Exception:
        pass
    time.sleep(0.5)

if healthy:
    print(f"healthy after {time.time() - started:.1f}s")
    proc.terminate()
    try:
        proc.wait(timeout=40)
    except subprocess.TimeoutExpired:
        proc.kill()
    handle.close()
    print(f"\nPASSED: the backend started and served  (output: {LOG})")
    sys.exit(0)

if proc.poll() is None:
    proc.kill()
    print(f"--- never became healthy inside {DEADLINE}s, still running ---")
else:
    print(f"--- exited rc={proc.returncode} before serving ---")

handle.close()
print("\n".join(LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-45:]))
sys.exit(1)
