"""Smoke-test the freshly built distribution, end to end.

Everything here is one process on purpose. The launcher's behaviour is timing
sensitive -- grace ladders, poll intervals, a close handler on a five second
guillotine -- and driving it from separate shell commands puts my own latency
inside the measurements.

Checks, in order:
  1. launcher.exe starts the backend and /api/health answers 200
  2. fonts/ holds zero fonts, so /api/fonts serves the two embedded ComicMono
     faces -- the fallback rule, verified through the compiled binary
  3. a second launcher reports the running instance and starts no second backend
  4. closing that second launcher leaves the backend alone (it did not start it)
  5. closing the owning launcher stops the backend and clears cache/
  6. nothing is left running

Kills every fox-reader.exe and launcher.exe on the machine to start from a known
state, which is why `tests/test_build.py` gates it behind FOX_TEST_DIST=1.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "packaging" / "build" / "stage" / "fox-reader"
LAUNCHER = STAGE / "launcher.exe"
URL = "http://127.0.0.1:7954"
CREATE_NEW_CONSOLE = 0x00000010

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(label)


def _pids(image):
    """PIDs of a running image, via WMI so the path is unambiguous."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Get-CimInstance Win32_Process -Filter \"Name='{image}'\" | "
         "Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True,
    ).stdout
    return sorted(int(line) for line in out.split() if line.strip().isdigit())


def backends():
    return _pids("fox-reader.exe")


def launchers():
    return _pids("launcher.exe")


def get(path, timeout=5):
    with urllib.request.urlopen(f"{URL}{path}", timeout=timeout) as response:
        return response.status, response.read()


def start(label):
    print(f"\n-- starting launcher ({label}) --", flush=True)
    return subprocess.Popen(
        [str(LAUNCHER), "--no-browser"],
        cwd=str(STAGE),
        creationflags=CREATE_NEW_CONSOLE,
    )


def close_window(pid):
    """Ask the process to close, the way the console's X button does.

    `taskkill` without /F posts a close request rather than terminating, so the
    launcher runs its CTRL_CLOSE_EVENT handler -- which is the path being tested.
    """
    return subprocess.run(["taskkill", "/PID", str(pid)], capture_output=True, text=True)


if not LAUNCHER.is_file():
    print(f"no launcher at {LAUNCHER}; run packaging/build.py first")
    sys.exit(1)

# 0. clean slate
for pid in backends() + launchers():
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
time.sleep(1)
print(f"starting state: backends={backends()} launchers={launchers()}", flush=True)

# 1. start and become healthy
owner = start("owner")
deadline = time.time() + 300
healthy = False
while time.time() < deadline:
    if owner.poll() is not None:
        break
    try:
        if get("/api/health")[0] == 200:
            healthy = True
            break
    except Exception:
        pass
    time.sleep(0.5)

check("backend healthy", healthy, f"after {300 - int(deadline - time.time())}s")
if not healthy:
    print("cannot continue without a healthy backend", flush=True)
    for pid in backends() + launchers():
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    sys.exit(1)

first_backends = backends()
check("exactly one backend", len(first_backends) == 1, str(first_backends))

# 2. the fallback fonts, through the compiled binary
fonts_in_folder = [p.name for p in (STAGE / "fonts").iterdir()
                   if p.suffix.lower() in (".ttf", ".otf")]
check("fonts/ holds zero font files", fonts_in_folder == [],
      str(sorted(p.name for p in (STAGE / "fonts").iterdir())))

status, body = get("/api/fonts", timeout=30)
payload = json.loads(body)
served = payload if isinstance(payload, list) else payload.get("fonts", payload)
names = sorted(f.get("font_filename") or f.get("name") or "?" for f in served)
check("/api/fonts answers 200", status == 200)
check("two fallback faces served", len(served) == 2, str(names))
check("they are the ComicMono pair",
      names == ["ComicMono-Bold.ttf", "ComicMono.ttf"], str(names))
check("no cache file written for embedded faces",
      not (STAGE / "fonts" / ".fonts_cache.json").exists())

# 3. a second launcher must reuse, not duplicate
cache = STAGE / "cache"
cache.mkdir(exist_ok=True)
(cache / "probe.bin").write_bytes(b"x" * 4096)
(cache / "nested").mkdir(exist_ok=True)
(cache / "nested" / "deep.bin").write_bytes(b"y" * 256)
before_cache = len(list(cache.rglob("*")))

second = start("second, should attach")
time.sleep(6)
check("second launcher still running (attached, not exited)", second.poll() is None)
check("still exactly one backend", backends() == first_backends, str(backends()))

# 4. closing the non-owner must leave the backend alone
close_window(second.pid)
try:
    second.wait(timeout=20)
except subprocess.TimeoutExpired:
    check("second launcher exited on close", False, "still running")
else:
    check("second launcher exited on close", True, f"rc={second.returncode}")
time.sleep(2)
check("backend survived the non-owner closing", backends() == first_backends, str(backends()))
try:
    check("health still 200 after non-owner closed", get("/api/health")[0] == 200)
except Exception as exc:
    check("health still 200 after non-owner closed", False, repr(exc))

# 5. closing the owner must stop the backend and clear cache/
print(f"\n-- closing the owner (cache has {before_cache} entries) --", flush=True)
asked = time.time()
close_window(owner.pid)
try:
    owner.wait(timeout=30)
    print(f"  owner exited rc={owner.returncode} after {time.time() - asked:.2f}s", flush=True)
except subprocess.TimeoutExpired:
    check("owner exited on close", False, "still running after 30s")

gone_deadline = time.time() + 30
while time.time() < gone_deadline and backends():
    time.sleep(0.5)
elapsed = time.time() - asked
check("backend stopped", not backends(), f"{elapsed:.2f}s after the close request")

after_cache = len(list(cache.rglob("*"))) if cache.exists() else 0
check("cache/ cleared", after_cache == 0, f"{before_cache} -> {after_cache}")

# 6. nothing left behind
time.sleep(1)
check("no orphaned backend", backends() == [], str(backends()))
check("no orphaned launcher", launchers() == [], str(launchers()))

print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}", flush=True)
sys.exit(1 if failures else 0)
