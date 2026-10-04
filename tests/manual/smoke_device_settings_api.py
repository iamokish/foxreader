"""Drive the device settings API without touching the real config or models.

The routes are exercised through httpx's ASGI transport against a bare FastAPI
app carrying just the state the settings routes read, so no model has to be
downloaded and the user's settings.yaml is left alone. A stubbed torch stands in
for hardware, the same way smoke_device.py does it.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import shutil
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

GIB = 1024**3


def fake_torch(gpus: list[tuple[str, int]]) -> types.ModuleType:
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(
        is_available=lambda: bool(gpus),
        device_count=lambda: len(gpus),
        get_device_properties=lambda index: types.SimpleNamespace(
            name=gpus[index][0], total_memory=gpus[index][1]
        ),
        get_device_name=lambda index: gpus[index][0],
    )
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    module.version = types.SimpleNamespace(cuda="12.6")
    return module


# Installed before device.py is imported, so the very first probe sees the cards.
sys.modules["torch"] = fake_torch([("RTX 4060", 8 * GIB), ("RTX 4090", 24 * GIB)])

import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from fox_reader import device  # noqa: E402
from fox_reader.routes import settings as settings_routes  # noqa: E402
from fox_reader.settings import SettingsManager  # noqa: E402

failures = 0

# An override sitting in the ambient environment wins over everything saved
# below, which would make the thread checks depend on the shell they ran from.
os.environ.pop(device.MTL_THREADS_ENV, None)


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures

    if not condition:
        failures += 1

    print(f"{'ok  ' if condition else 'FAIL'} {label}" + (f" -- {detail}" if detail else ""))


root = pathlib.Path(tempfile.mkdtemp()) / "config"
settings_mgr = SettingsManager(root)
settings_mgr.load()

# What a real startup does, in the same order: resolve, then hand the resolution
# to app.state so the routes can tell saved from running.
resolution = device.configure(settings_mgr.get_devices())

app = FastAPI()
app.include_router(settings_routes.router)
app.state.settings = settings_mgr
app.state.mtl_dir = root / "mtl"
app.state.devices = resolution

client = httpx.AsyncClient(
    transport=httpx.ASGITransport(app=app),
    base_url="http://test",
)


class Sync:
    """httpx's ASGI transport is async-only; this keeps the checks flat."""

    def get(self, url: str) -> httpx.Response:
        return asyncio.run(client.get(url))

    def put(self, url: str, json: object) -> httpx.Response:
        return asyncio.run(client.put(url, json=json))

    def close(self) -> None:
        asyncio.run(client.aclose())


client_sync = Sync()

print("--- GET /api/settings ---")
payload = client_sync.get("/api/settings").json()

check("devices block is present", "devices" in payload, str(sorted(payload)))
devices = payload["devices"]

check("api_tokens still there", "api_tokens" in payload)
check("languages still there", "languages" in payload)
check("slots are the five settable ones",
      devices["slots"] == ["paddleocr", "paddleocr_vl", "bubble", "translator", "textseg"],
      str(devices["slots"]))
check("auto is what is saved", set(devices["selection"].values()) == {"auto"}, str(devices["selection"]))
check("auto resolved to the bigger card", set(devices["resolved"].values()) == {"cuda:1"}, str(devices["resolved"]))
check("active shows the native spellings", devices["active"]["paddleocr"] == "gpu:1", str(devices["active"]))
check("paddleocr_vl active is a torch spelling", devices["active"]["paddleocr_vl"] == "cuda:1", str(devices["active"]))
check("textseg active is a torch spelling", devices["active"]["textseg"] == "cuda:1", str(devices["active"]))
check("backend is derived", devices["backend"] == "cuda", devices["backend"])
check("gpu_name is the winner", devices["gpu_name"] == "RTX 4090", str(devices["gpu_name"]))
check("device_count", devices["device_count"] == 2, str(devices["device_count"]))
check("bubble rides along as a torch string", devices["active"]["bubble"] == "cuda:1", str(devices["active"]))
check("no restart needed yet", devices["restart_required"] is False)
check("running is in ids, comparable with resolved", devices["running"] == devices["resolved"], str(devices["running"]))
check("no fallbacks", devices["fallbacks"] == [], str(devices["fallbacks"]))

check(
    "gpu:1 vs cuda:1 does not read as a pending restart",
    devices["active"]["paddleocr"] != devices["resolved"]["paddleocr"]
    and devices["restart_required"] is False,
    f'active={devices["active"]["paddleocr"]} resolved={devices["resolved"]["paddleocr"]}',
)

labels = [option["label"] for option in devices["options"]]
check("four options offered", len(labels) == 4, str(labels))
check("auto names the winner", labels[0].startswith("Auto"), labels[0])
check("cards carry their memory", "24.0 GB" in labels[3], labels[3])

print("\n--- the thread picker rides along with the devices ---")
threads = devices["mtl_threads"]

check("the block is there", isinstance(threads, dict), str(threads))
check("auto until someone chooses", threads["value"] == "auto", str(threads["value"]))
check(
    "the ceiling is the physical core count, which is what the page offers",
    threads["max"] == device.physical_cores(),
    str(threads["max"]),
)
check("and it is never zero, or the picker would be empty", threads["max"] >= 1, str(threads["max"]))
check(
    "auto is previewed as a number rather than the word",
    isinstance(threads["auto"], int) and threads["auto"] >= 1,
    str(threads["auto"]),
)
check("which is what the saved value comes to", threads["resolved"] == threads["auto"], str(threads))
check("no env override to report", threads["override"] is None, str(threads["override"]))
check(
    "the translator resolved onto a card here, so the lower cap applies",
    threads["auto"] <= device.MAX_GPU_THREADS,
    f'{threads["auto"]} on {devices["resolved"]["translator"]}',
)

print("\n--- PUT /api/settings/devices ---")
response = client_sync.put("/api/settings/devices", json={"bubble": "cuda:0", "translator": "cpu"})
check("accepted", response.status_code == 200, str(response.status_code))

devices = response.json()["devices"]
check("bubble saved", devices["selection"]["bubble"] == "cuda:0", devices["selection"]["bubble"])
check("translator saved", devices["selection"]["translator"] == "cpu", devices["selection"]["translator"])
check("paddleocr untouched", devices["selection"]["paddleocr"] == "auto", devices["selection"]["paddleocr"])
check("textseg untouched", devices["selection"]["textseg"] == "auto", devices["selection"]["textseg"])
check("a restart is now required", devices["restart_required"] is True)
check("the running process is unchanged", device.FOX_DEVICE.bubble == "cuda:1", device.FOX_DEVICE.bubble)
check("and active still reports the running value", devices["active"]["bubble"] == "cuda:1", devices["active"]["bubble"])
check("running still points at the old card", devices["running"]["bubble"] == "cuda:1", devices["running"]["bubble"])
check("a slot that did not change is not flagged", devices["running"]["paddleocr"] == devices["resolved"]["paddleocr"])
check(
    "and the thread preview follows the translator onto the cpu",
    devices["mtl_threads"]["auto"] == device.thread_count(on_gpu=False, preference="auto"),
    f'{devices["mtl_threads"]["auto"]} on {devices["resolved"]["translator"]}',
)

check("it reached the yaml", "bubble: cuda:0" in (root / "settings.yaml").read_text(encoding="utf8"))

reloaded = SettingsManager(root)
reloaded.load()
check("and survives a reload", reloaded.get_devices()["translator"] == "cpu", str(reloaded.get_devices()))

print("\n--- loose spellings the picker will never send, but a curl might ---")
for sent, saved in [("GPU:1", "cuda:1"), ("cuda", "cuda:0"), ("1", "cuda:1"), ("auto", "auto")]:
    body = client_sync.put("/api/settings/devices", json={"bubble": sent}).json()
    check(f"{sent!r} normalises to {saved!r}", body["devices"]["selection"]["bubble"] == saved, body["devices"]["selection"]["bubble"])

print("\n--- saving a thread count ---")
response = client_sync.put("/api/settings/devices", json={"mtl_threads": 4})
check("accepted on the same endpoint as the slots", response.status_code == 200, str(response.status_code))

threads = response.json()["devices"]["mtl_threads"]
check("the count is saved", threads["value"] == 4, str(threads["value"]))
check(
    "and previewed as itself rather than as the heuristic",
    threads["resolved"] == min(4, device.usable_cpus()),
    str(threads["resolved"]),
)
check("it reached the yaml", "mtl_threads: 4" in (root / "settings.yaml").read_text(encoding="utf8"))

reloaded = SettingsManager(root)
reloaded.load()
check("and survives a reload as a number", reloaded.get_mtl_threads() == 4, repr(reloaded.get_mtl_threads()))

# The picker is a <select>, so its value arrives as a string.
threads = client_sync.put("/api/settings/devices", json={"mtl_threads": "6"}).json()["devices"]["mtl_threads"]
check("a string from the picker becomes a count", threads["value"] == 6, repr(threads["value"]))
check(
    "and the yaml holds a number, not a quoted one",
    "mtl_threads: 6" in (root / "settings.yaml").read_text(encoding="utf8"),
)

threads = client_sync.put("/api/settings/devices", json={"mtl_threads": "auto"}).json()["devices"]["mtl_threads"]
check("auto goes back", threads["value"] == "auto", repr(threads["value"]))
check("as the word", "mtl_threads: auto" in (root / "settings.yaml").read_text(encoding="utf8"))
check("and the preview agrees with it again", threads["resolved"] == threads["auto"], str(threads))

body = client_sync.put(
    "/api/settings/devices",
    json={"paddleocr": "cpu", "mtl_threads": 2},
).json()["devices"]
check("a slot and a count save together", body["selection"]["paddleocr"] == "cpu", str(body["selection"]))
check("both of them", body["mtl_threads"]["value"] == 2, str(body["mtl_threads"]["value"]))

print("\n--- textseg diverges from paddleocr ---")
body = client_sync.put(
    "/api/settings/devices",
    json={"paddleocr": "cpu", "textseg": "cuda:1"},
).json()["devices"]
check("ocr pinned to the cpu", body["selection"]["paddleocr"] == "cpu", str(body["selection"]))
check("cleaning pinned to the card", body["selection"]["textseg"] == "cuda:1", str(body["selection"]))
check("and they resolve apart", body["resolved"]["paddleocr"] == "cpu" and body["resolved"]["textseg"] == "cuda:1", str(body["resolved"]))
check("it reached the yaml", "textseg: cuda:1" in (root / "settings.yaml").read_text(encoding="utf8"))

print("\n--- paddleocr_vl diverges from paddleocr ---")
body = client_sync.put(
    "/api/settings/devices",
    json={"paddleocr_vl": "cuda:0"},
).json()["devices"]
check("vl pinned to card 0", body["selection"]["paddleocr_vl"] == "cuda:0", str(body["selection"]))
check("and resolves in torch spelling", body["resolved"]["paddleocr_vl"] == "cuda:0", str(body["resolved"]))
check("classic ocr untouched", body["selection"]["paddleocr"] == "cpu", str(body["selection"]))
check("it reached the yaml", "paddleocr_vl: cuda:0" in (root / "settings.yaml").read_text(encoding="utf8"))

reloaded = SettingsManager(root)
reloaded.load()
check("and survives a reload", reloaded.get_devices()["paddleocr_vl"] == "cuda:0", str(reloaded.get_devices()))

print("\n--- rejections ---")
for body, reason in [
    ({"bubble": "rubbish"}, "not a device"),
    ({"gpu": "cuda:0"}, "not a slot"),
    ({"mtl_threads": "many"}, "not a thread count"),
    ({"mtl_threads": 0}, "fewer than one thread"),
    ({"mtl_threads": True}, "a yaml-style yes"),
    ({}, "nothing to do"),
    ([], "not an object"),
]:
    response = client_sync.put("/api/settings/devices", json=body)
    check(f"{reason} -> 400", response.status_code == 400, str(response.status_code))
    check("  with a message", bool(response.json().get("error")), json.dumps(response.json())[:90])

check(
    "the empty-body message names the thread field too",
    "mtl_threads" in client_sync.put("/api/settings/devices", json={}).json()["error"],
    client_sync.put("/api/settings/devices", json={}).json()["error"],
)

check(
    "a rejected write changed nothing",
    SettingsManager(root).load().devices.bubble == "auto",
    str(SettingsManager(root).load().devices.bubble),
)
check(
    "nor the thread count",
    SettingsManager(root).load().devices.mtl_threads == 2,
    repr(SettingsManager(root).load().devices.mtl_threads),
)

print("\n--- a card that is no longer here ---")
settings_mgr.update_devices({"paddleocr": "cuda:5"})
devices = client_sync.get("/api/settings").json()["devices"]

check("the preference is kept", devices["selection"]["paddleocr"] == "cuda:5", devices["selection"]["paddleocr"])
check("but it resolves to the cpu", devices["resolved"]["paddleocr"] == "cpu", devices["resolved"]["paddleocr"])
check("and it is explained", any("cuda:5" in line for line in devices["fallbacks"]), str(devices["fallbacks"]))
check("textseg is unaffected", devices["resolved"]["textseg"] == "cuda:1", str(devices["resolved"]))
check("still on disk unmodified", "cuda:5" in (root / "settings.yaml").read_text(encoding="utf8"))

client_sync.close()
shutil.rmtree(root.parent, ignore_errors=True)

print(f"\n{failures} FAILURE(S)" if failures else "\nall settings API checks passed")
sys.exit(1 if failures else 0)
