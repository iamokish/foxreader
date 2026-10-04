"""Prove the Xet half of the fix, through the compiled binary, the way a user hits it.

`is_xet_available()` asks `importlib.metadata.version("hf_xet")`; the distribution
installs itself as `hf-xet`. Nuitka's embedded metadata is keyed by the name the
distribution gives itself and read back with an exact dict lookup, so that call used
to miss, and huggingface_hub announced that Xet was unavailable and downloaded over
plain HTTP instead.

Order matters here. The setup wizard's worker downloads what `check_models_exist`
found missing *at startup*, so the model has to be absent before the backend starts
-- deleting it behind a running server leaves that list empty and the download
silently no-ops. This is the real first-run sequence: models/bubble missing, backend
starts, POST /api/setup/download, watch what it says.

So this deletes `models/bubble` itself rather than asking to be handed a tree that
already lacks it. It is the condition under test, it makes the script idempotent,
and it means running it after verify_metadata.py -- which leaves the model in place
-- still tests a first run. The download puts back what it removed.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "packaging" / "build" / "stage" / "fox-reader"
BACKEND = STAGE / "bin" / "fox-reader.exe"
URL = "http://127.0.0.1:7954"
LOG = Path(tempfile.gettempdir()) / "fox-reader-tests" / "verify_download.log"
BUBBLE = STAGE / "models" / "bubble"

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(label)


def post(path, payload):
    request = urllib.request.Request(
        f"{URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, json.loads(response.read() or b"{}")


def get(path, timeout=15):
    with urllib.request.urlopen(f"{URL}{path}", timeout=timeout) as response:
        return response.status, response.read()


if not BACKEND.is_file():
    print(f"no compiled backend at {BACKEND}; run packaging/build.py first")
    sys.exit(1)

LOG.parent.mkdir(parents=True, exist_ok=True)

subprocess.run(["taskkill", "/F", "/IM", "fox-reader.exe"], capture_output=True)
time.sleep(1)

if BUBBLE.exists():
    shutil.rmtree(BUBBLE)
    print(f"-- removed {BUBBLE.name} so this is a first run --", flush=True)
check("models/bubble absent before startup (so the wizard will fetch it)",
      not BUBBLE.exists(), "could not remove it" if BUBBLE.exists() else "")

print("\n-- starting the compiled backend --", flush=True)
handle = LOG.open("w", encoding="utf-8", errors="replace")
proc = subprocess.Popen([str(BACKEND)], cwd=str(STAGE), stdout=handle,
                        stderr=subprocess.STDOUT)

up = False
started = time.time()
while time.time() - started < 300:
    if proc.poll() is not None:
        break
    try:
        if get("/api/setup/status")[0] == 200:
            up = True
            break
    except Exception:
        pass
    time.sleep(0.5)

check("backend serving with a required model missing", up,
      f"{time.time() - started:.1f}s" if up else f"exited rc={proc.returncode}")

if not up:
    handle.close()
    print("\n--- last 30 lines ---")
    print("\n".join(LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
    proc.kill()
    sys.exit(1)

_, raw = get("/api/setup/status")
state = json.loads(raw)
print(f"   stage={state.get('stage')} non_mtl_required={state.get('non_mtl_required')}", flush=True)
check("the wizard knows bubble is required", "bubble" in (state.get("non_mtl_required") or []),
      str(state.get("non_mtl_required")))

try:
    status, body = post("/api/setup/download", {})
    print(f"   POST /api/setup/download -> {status} {body}", flush=True)
    check("download started", status == 200 and body.get("status") == "started", str(body))
except urllib.error.HTTPError as exc:
    check("download started", False, f"HTTP {exc.code}: {exc.read()[:200]}")
except Exception as exc:
    check("download started", False, repr(exc))

deadline = time.time() + 1200
last = ""
done = False
while time.time() < deadline:
    try:
        _, raw = get("/api/setup/status")
        state = json.loads(raw)
    except Exception:
        time.sleep(2)
        continue

    progress = state.get("progress") or {}
    line = (f"active={state.get('active')} model={state.get('current_model')} "
            + " ".join(f"{k}={v.get('files_done')}/{v.get('files_total')}"
                       for k, v in sorted(progress.items())))
    if line != last:
        print(f"   {line}", flush=True)
        last = line

    if state.get("error"):
        check("download completed without error", False, str(state["error"]))
        break
    if not state.get("active") and progress:
        done = all(item.get("done") for item in progress.values())
        break
    time.sleep(2)

size = (sum(f.stat().st_size for f in BUBBLE.rglob("*") if f.is_file()) / 1e6
        if BUBBLE.is_dir() else 0)
check("bubble downloaded", done and BUBBLE.is_dir() and size > 1,
      f"{size:.0f} MB" if BUBBLE.is_dir() else "missing")

proc.terminate()
try:
    proc.wait(timeout=40)
except subprocess.TimeoutExpired:
    proc.kill()
handle.close()

output = LOG.read_text(encoding="utf-8", errors="replace")
lines = output.splitlines()

xet = [line for line in lines if "hf_xet" in line or "Xet Storage" in line
       or "hf_transfer" in line]
check("no Xet fallback warning during the download", not xet,
      xet[0][:160] if xet else "")

missing_meta = [line for line in lines
                if "PackageNotFoundError" in line or "distribution was not found" in line]
check("no metadata lookup failed anywhere in the run", not missing_meta,
      missing_meta[0][:160] if missing_meta else "")

tracebacks = [line for line in lines if "Traceback (most recent call last)" in line]
check("no traceback in the backend output", not tracebacks, f"{len(tracebacks)} found")

print(f"\n(backend output: {LOG})")
print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
