"""Exercise GET /ml/memory, GET /ml/control/status and the refusal path over real HTTP.

The route is mounted on a bare FastAPI app with a stub translate service rather
than the real one: routes/translate.py imports nothing heavy, so this checks the
HTTP contract -- the path, the JSON shape, and that the refusal sentence reaches
the client -- without pulling in torch or paddleocr.

What matters here is that the shape the inline script in index.html reads
(data.ok, data.issues[].message) is the shape every branch of the route returns,
including the two hand-written empty-dict literals.
"""

from __future__ import annotations

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _stub_model_libs() -> None:
    """The route needs none of these; local_mtl's exception class does.

    The GGUF translators import llama.cpp lazily, so only torch needs a
    stand-in here. None of them are exercised.
    """
    for name, attrs in [
        ("torch", {"inference_mode": lambda *a, **k: (lambda fn: fn)}),
    ]:
        if name in sys.modules:
            continue

        module = types.ModuleType(name)

        for attr, value in attrs.items():
            setattr(module, attr, value)

        sys.modules[name] = module


_stub_model_libs()

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fox_reader.routes.translate import router  # noqa: E402

failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures

    if condition:
        print(f"ok   {label}")
        return

    failures += 1
    print(f"FAIL {label}" + (f" -- {detail}" if detail else ""))


SHORTFALL = (
    "Rosetta 4B-2511 needs 6800 MiB of VRAM on GPU 0 (RTX 4060), but only "
    "3072 MiB is free. Close whatever else is using the GPU, choose another "
    "device under Settings → Compute Devices, or pick a model that fits."
)

REPORT = {
    "enabled": True,
    "device": "cuda:0",
    "ok": False,
    "checks": [
        {
            "model": "rosetta-4b-2511",
            "model_name": "Rosetta 4B-2511",
            "device": "cuda:0",
            "pool": "vram",
            "required_mib": 6800,
            "available_mib": 3072,
            "fits": False,
            "measured": True,
            "message": SHORTFALL,
            "languages": ["japanese", "korean"],
        }
    ],
    "issues": [],
}
REPORT["issues"] = [REPORT["checks"][0]]

EMPTY_KEYS = {"enabled", "device", "ok", "checks", "issues"}


class Stub:
    """Stands in for TranslateService."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        report=REPORT,
        raises: Exception | None = None,
        status: dict | None = None,
    ):
        self.mtl_enabled = enabled
        self.report = report
        self.raises = raises
        self.status = (
            status
            if status is not None
            else {"loaded": False, "lang": None, "model_id": None, "languages": []}
        )
        self.loaded: list[str] = []

    def ml_memory(self):
        if self.raises is not None:
            raise self.raises
        if not self.mtl_enabled:
            return {"enabled": False, "device": "", "ok": True, "checks": [], "issues": []}
        return self.report

    def ml_status(self):
        if self.raises is not None:
            raise self.raises
        return self.status

    def ml_load(self, lang):
        if self.raises is not None:
            raise self.raises
        self.loaded.append(lang)


def client(stub: Stub) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.translate = stub
    return TestClient(app, raise_server_exceptions=False)


print("--- GET /ml/memory, a model that does not fit ---")
res = client(Stub()).get("/ml/memory")
check("200", res.status_code == 200, str(res.status_code))

data = res.json()
check("the path is /ml/memory, no prefix", res.status_code == 200)
check("ok is false", data["ok"] is False)
check("one issue", len(data["issues"]) == 1)
check("carrying the sentence", data["issues"][0]["message"] == SHORTFALL)
check("the arrow survives the JSON round-trip", "→" in data["issues"][0]["message"])
check("and the languages ride along", data["checks"][0]["languages"] == ["japanese", "korean"])
check("every key the script reads is present", EMPTY_KEYS <= set(data))

print("\n--- GET /ml/memory with MTL disabled ---")
data = client(Stub(enabled=False)).get("/ml/memory").json()
check("ok is true, so no banner", data["ok"] is True)
check("enabled is false", data["enabled"] is False)
check("same keys as the real report", set(data) == EMPTY_KEYS, str(sorted(set(data) ^ EMPTY_KEYS)))

print("\n--- GET /ml/memory when the probe blows up ---")
data = client(Stub(raises=RuntimeError("driver exploded"))).get("/ml/memory").json()
check("still 200, still the same shape", set(data) == EMPTY_KEYS)
check("and ok, so the page stays quiet", data["ok"] is True)

print("\n--- the shapes agree ---")
disabled = client(Stub(enabled=False)).get("/ml/memory").json()
failed = client(Stub(raises=RuntimeError("x"))).get("/ml/memory").json()
real = client(Stub()).get("/ml/memory").json()
check("both empty literals are identical", disabled == failed)
check("and are a subset of the real report's keys", set(disabled) <= set(real))
check(
    "types line up",
    all(type(disabled[k]) is type(real[k]) for k in disabled),
    str({k: (type(disabled[k]).__name__, type(real[k]).__name__) for k in disabled}),
)

print("\n--- POST /ml/control/load surfaces the refusal verbatim ---")
from fox_reader.translate.local_mtl import InsufficientMemoryError, MemoryCheck  # noqa: E402

check_obj = MemoryCheck(
    model="rosetta-4b-2511",
    model_name="Rosetta 4B-2511",
    device="cuda:0",
    pool="vram",
    required_mib=6800,
    available_mib=3072,
    fits=False,
    measured=True,
    message=SHORTFALL,
)

res = client(Stub(raises=InsufficientMemoryError(check_obj))).post(
    "/ml/control/load", json={"lang": "japanese"}
)
body = res.json()
check("200", res.status_code == 200, str(res.status_code))
check("status false", body["status"] is False)
check("the message is the sentence, unwrapped", body["message"] == SHORTFALL, body["message"])

print("\n--- POST /translate/ml, same refusal ---")


class RaisingStub(Stub):
    async def ml_async(self, text, lang, context=None, character_info=None, context_character_links=None, meta_id=None):
        raise InsufficientMemoryError(check_obj)


body = client(RaisingStub()).post(
    "/translate/ml", json={"text": "テスト", "source_lang": "japanese"}
).json()
check("no crash", body["translated"] == "")
check("the sentence is still readable after the prefix", SHORTFALL in body["error"], body["error"])
check("even though it is prefixed", body["error"].startswith("ML Inference Error: "), body["error"])

print("\n--- a healthy load still works ---")
stub = Stub()
body = client(stub).post("/ml/control/load", json={"lang": "korean"}).json()
check("status true", body["status"] is True)
check("and it reached the service", stub.loaded == ["korean"], str(stub.loaded))

print("\n--- GET /ml/control/status, a model still loaded ---")
STATUS_KEYS = {"loaded", "lang", "model_id", "languages"}
res = client(
    Stub(
        status={
            "loaded": True,
            "lang": "japanese",
            "model_id": "vntl-llama3-8b-v2",
            "languages": ["japanese"],
        }
    )
).get("/ml/control/status")
check("200", res.status_code == 200, str(res.status_code))
data = res.json()
check("loaded is true", data["loaded"] is True)
check("the load language rides along", data["lang"] == "japanese", str(data))
check("the model id rides along", data["model_id"] == "vntl-llama3-8b-v2", str(data))
check("and the supported languages", data["languages"] == ["japanese"], str(data))
check("exactly the documented keys", set(data) == STATUS_KEYS, str(sorted(set(data) ^ STATUS_KEYS)))

print("\n--- GET /ml/control/status, nothing loaded ---")
data = client(Stub()).get("/ml/control/status").json()
check(
    "the honest empty shape",
    data == {"loaded": False, "lang": None, "model_id": None, "languages": []},
    str(data),
)

print("\n--- GET /ml/control/status when the service blows up ---")
data = client(Stub(raises=RuntimeError("driver exploded"))).get("/ml/control/status").json()
check("still 200, still the empty shape", data["loaded"] is False and set(data) == STATUS_KEYS, str(data))

print("\n--- LocalMTLManager.status without loading gigabytes ---")
from fox_reader.translate.local_mtl import LocalMTLManager  # noqa: E402


class FakeModel:
    @classmethod
    def model_id(cls):
        return "vntl-llama3-8b-v2"

    @classmethod
    def model_langs(cls):
        return ["japanese"]


class BrokenModel:
    def model_id(self):
        raise RuntimeError("half-built")

    def model_langs(self):
        raise RuntimeError("half-built")


mgr = LocalMTLManager(mtl_dir=pathlib.Path("."), settings=None)
check(
    "unloaded reads as not loaded",
    mgr.status() == {"loaded": False, "lang": None, "model_id": None, "languages": []},
    str(mgr.status()),
)

mgr.model = FakeModel()
mgr.current_lang = "japanese"
state = mgr.status()
check("a model reads as loaded", state["loaded"] is True, str(state))
check("with its load language", state["lang"] == "japanese", str(state))
check("its id", state["model_id"] == "vntl-llama3-8b-v2", str(state))
check("and its languages", state["languages"] == ["japanese"], str(state))

mgr.model = BrokenModel()
state = mgr.status()
check("a half-built model still answers", state["loaded"] is True, str(state))
check("with None for what it could not say", state["model_id"] is None, str(state))
check("and no languages", state["languages"] == [], str(state))

print(f"\n{failures} FAILURE(S)" if failures else "\nthe /ml/memory route and the refusal path are wired correctly")
sys.exit(1 if failures else 0)
