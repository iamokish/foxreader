"""Check the compiled build starts with models present, and re-downloads a removed one.

Two things, both of which used to fail:

  1. The reported crash. paddlex imports transformers while building a text
     detector, transformers checks `huggingface-hub`'s version at import, and the
     lookup missed Nuitka's exact-keyed metadata dict -- so the backend died
     before serving anything. Only reproducible with models/ populated, which is
     why a health check on a fresh tree never caught it.

  2. Removing a model while the server runs and asking the wizard to fetch it
     again. The worker's list came from a cache filled in at startup, so it was
     empty here: the POST answered "started" and downloaded nothing, forever.

The Xet half of the metadata fix is covered by verify_download.py, which has to
start with the model already absent to exercise the first-run path.

Not a quiet test. It kills every fox-reader.exe on the machine, deletes
`models/bubble` from the staged build and downloads it again, which is why
`tests/test_build.py` only runs it when FOX_TEST_DIST=1 is set.
"""
import json
import re
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
LOG = Path(tempfile.gettempdir()) / "fox-reader-tests" / "verify_metadata.log"

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
subprocess.run(["taskkill", "/F", "/IM", "launcher.exe"], capture_output=True)
time.sleep(1)

# ── 1. what actually shipped ──────────────────────────────────────────────────
shipped = sorted(p.name for p in (STAGE / "bin").glob("*.dist-info"))
print(f"shipped .dist-info: {len(shipped)}")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


names = {normalize(name.rsplit(".", 1)[0].split("-")[0]) for name in shipped}
check("huggingface_hub metadata shipped", "huggingface-hub" in names)
check("hf_xet metadata shipped", "hf-xet" in names)
check("pyyaml metadata shipped", "pyyaml" in names)
check("pillow metadata shipped", "pillow" in names)

models = STAGE / "models"
present = sorted(p.name for p in models.iterdir()) if models.is_dir() else []
check("models are present (required to reproduce the crash)", bool(present), str(present))

# ── 2. the backend starts ─────────────────────────────────────────────────────
print("\n-- starting the compiled backend --", flush=True)
handle = LOG.open("w", encoding="utf-8", errors="replace")
proc = subprocess.Popen([str(BACKEND)], cwd=str(STAGE), stdout=handle,
                        stderr=subprocess.STDOUT)

healthy = False
started = time.time()
while time.time() - started < 300:
    if proc.poll() is not None:
        break
    try:
        if get("/api/health")[0] == 200:
            healthy = True
            break
    except Exception:
        pass
    time.sleep(0.5)

check("backend healthy with models present", healthy,
      f"{time.time() - started:.1f}s" if healthy else f"exited rc={proc.returncode}")

if not healthy:
    handle.close()
    print("\n--- last 30 lines of backend output ---")
    print("\n".join(LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
    proc.kill()
    sys.exit(1)

# ── 3. remove a model behind the running server, then ask for it back ─────────
bubble = models / "bubble"
if bubble.is_dir():
    shutil.rmtree(bubble)
    print(f"\n-- deleted {bubble.name} while the server runs, asking for it again --",
          flush=True)

started_ok = False
try:
    status, body = post("/api/setup/download", {})
    print(f"   POST /api/setup/download -> {status} {body}", flush=True)
    started_ok = status == 200 and body.get("status") == "started"
    check("download started", started_ok, str(body))
except urllib.error.HTTPError as exc:
    check("download started", False, f"HTTP {exc.code}: {exc.read()[:200]}")
except Exception as exc:
    check("download started", False, repr(exc))

# Short deadline on purpose: the old failure mode was a download that reported
# itself started and then sat idle, so waiting a quarter of an hour to notice
# would only hide it.
deadline = time.time() + 600
last = ""
done = False
while started_ok and time.time() < deadline:
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
    if not state.get("active") and not progress:
        check("the worker was given something to do", False,
              "reported started with an empty work list")
        break
    time.sleep(2)

size = (sum(f.stat().st_size for f in bubble.rglob("*") if f.is_file()) / 1e6
        if bubble.is_dir() else 0)
check("model removed mid-run was downloaded again", done and size > 1,
      f"{size:.0f} MB" if bubble.is_dir() else "missing")

# ── 4. what the run reported ─────────────────────────────────────────────────
proc.terminate()
try:
    proc.wait(timeout=40)
except subprocess.TimeoutExpired:
    proc.kill()
handle.close()

lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()

xet = [line for line in lines if "hf_xet" in line or "Xet Storage" in line]
check("no Xet fallback warning", not xet, xet[0][:160] if xet else "")

missing_meta = [line for line in lines
                if "PackageNotFoundError" in line or "distribution was not found" in line]
check("no metadata lookup failed anywhere in the run", not missing_meta,
      missing_meta[0][:160] if missing_meta else "")

check("no traceback in the backend output",
      not [line for line in lines if "Traceback (most recent call last)" in line])

print(f"\n(backend output: {LOG})")
print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
