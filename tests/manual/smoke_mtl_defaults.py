"""Check that the MTL default model is settled against what is on disk.

The bug this covers: a fresh settings.yaml is written with every MTL default
null, and nothing ever filled them in except opening the settings page -- so on
a machine with the models downloaded the translator still refused every
language with "No local MTL model is configured", and the page showed a
selection that was not actually saved until it was clicked.

Nothing here needs torch or a real model: "downloaded" is a marker file, so a
temp directory with the right `_completed` files in it is the whole fixture.
Also drives GET /api/settings over HTTP, since that is the other caller and it
must not rewrite the file on every request.
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import yaml  # noqa: E402

from fox_reader.constants import (  # noqa: E402
    MTL_COMPLETION_MARKER,
    MTL_MODELS,
    mtl_model_downloaded,
    mtl_models_for_language,
)
from fox_reader.settings import SUPPORTED_MTL_LANGUAGES, SettingsManager  # noqa: E402

Q8 = "gemma-4-e4b-q8-uncensored"
Q6 = "gemma-4-e4b-q6-uncensored"
VNTL = "vntl-llama3-8b-v2"

failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures

    if condition:
        print(f"ok   {label}")
        return

    failures += 1
    print(f"FAIL {label}" + (f" -- {detail}" if detail else ""))


TMP = pathlib.Path(tempfile.mkdtemp(prefix="fox-mtl-defaults-"))


def mtl_dir(*model_ids: str) -> pathlib.Path:
    """An MTL directory in which exactly these models look downloaded."""
    root = TMP / ("dir-" + ("-".join(model_ids) or "empty"))

    for model in MTL_MODELS:
        dest = root / model["dest"]
        dest.mkdir(parents=True, exist_ok=True)

        if model["id"] in model_ids:
            (dest / MTL_COMPLETION_MARKER).touch()

    return root


def manager(name: str, saved: dict | None = None) -> SettingsManager:
    """A settings manager over its own config dir, optionally pre-seeded."""
    config_root = TMP / f"config-{name}"
    config_root.mkdir(parents=True, exist_ok=True)

    if saved is not None:
        (config_root / "settings.yaml").write_text(
            yaml.safe_dump(saved, sort_keys=False), encoding="utf-8"
        )

    mgr = SettingsManager(config_root)
    mgr.load()
    return mgr


def on_disk(mgr: SettingsManager) -> dict:
    return yaml.safe_load(mgr.settings_path.read_text(encoding="utf-8")) or {}


# --- the helpers ------------------------------------------------------------

print("--- the shared predicate ---")
# Built from MTL_MODELS rather than from a list of ids, so that adding a model
# does not quietly leave this fixture describing an install that is missing it.
everything = mtl_dir(*[model["id"] for model in MTL_MODELS])
none = mtl_dir()

check("a marker means downloaded", mtl_model_downloaded(everything, MTL_MODELS[0]) is True)
check("a bare directory does not", mtl_model_downloaded(none, MTL_MODELS[0]) is False)
check("an id works as well as an entry", mtl_model_downloaded(everything, Q8) is True)
check("an unknown id is not downloaded", mtl_model_downloaded(everything, "who-knows") is False)
check("nor is a missing dest", mtl_model_downloaded(everything, {"id": "x"}) is False)
check("a str path is accepted", mtl_model_downloaded(str(everything), Q8) is True)

print("\n--- models per language ---")
japanese = [model["id"] for model in mtl_models_for_language("japanese")]
korean = [model["id"] for model in mtl_models_for_language("korean")]

check("three serve japanese", japanese == [Q8, Q6, VNTL], str(japanese))
check("two serve korean", korean == [Q8, Q6], str(korean))
check("MTL_MODELS order is the preference order", japanese[0] == Q8, str(japanese))
check(
    "the quants come before the single-language model",
    japanese[:2] == [Q8, Q6] and korean[0] == Q8,
    str((japanese, korean)),
)
check(
    "case and space do not matter",
    [m["id"] for m in mtl_models_for_language("  Korean ")] == [Q8, Q6],
)
check("an unknown language has none", mtl_models_for_language("klingon") == [])
check("and neither does an empty one", mtl_models_for_language("") == [])


# --- a fresh install --------------------------------------------------------

print("\n--- a fresh settings.yaml, nothing downloaded ---")
mgr = manager("fresh-empty")
check("starts null", all(v is None for v in mgr.get_defaults().values()), str(mgr.get_defaults()))

resolved = mgr.resolve_mtl_defaults(none)
check("stays null", all(v is None for v in resolved.values()), str(resolved))
check("every language is answered", set(resolved) == set(SUPPORTED_MTL_LANGUAGES), str(sorted(resolved)))

print("\n--- a fresh settings.yaml with every model downloaded (the reported bug) ---")
mgr = manager("fresh-both")
check("starts null", all(v is None for v in mgr.get_defaults().values()))

resolved = mgr.resolve_mtl_defaults(everything)
check("japanese takes the first downloaded model", resolved["japanese"] == Q8, str(resolved))
check("korean takes the first one that serves it", resolved["korean"] == Q8, str(resolved))
check("and chinese too", resolved["chinese"] == Q8, str(resolved))

check("the manager agrees", mgr.get_defaults() == resolved, str(mgr.get_defaults()))
check("and it reached the file", on_disk(mgr)["mtl_defaults"] == resolved, str(on_disk(mgr)))

print("\n--- only one multi-language model downloaded ---")
mgr = manager("q6-only")
resolved = mgr.resolve_mtl_defaults(mtl_dir(Q6))
check("japanese falls to it", resolved["japanese"] == Q6, str(resolved))
check("as do the others", resolved["korean"] == Q6 and resolved["chinese"] == Q6)

print("\n--- only the single-language model downloaded ---")
mgr = manager("vntl-only")
resolved = mgr.resolve_mtl_defaults(mtl_dir(VNTL))
check("japanese takes it", resolved["japanese"] == VNTL, str(resolved))
check("korean has nothing", resolved["korean"] is None, str(resolved))
check("and chinese has nothing", resolved["chinese"] is None, str(resolved))


# --- repairing a saved value ------------------------------------------------

print("\n--- a saved choice that is still downloaded is kept ---")
mgr = manager(
    "keep-choice",
    {"mtl_defaults": {"japanese": Q6, "korean": Q6, "chinese": Q6}},
)
resolved = mgr.resolve_mtl_defaults(everything)
check("japanese keeps the Q6, not the first entry", resolved["japanese"] == Q6, str(resolved))

print("\n--- a saved model that has been deleted is replaced ---")
mgr = manager(
    "deleted",
    {"mtl_defaults": {"japanese": Q6, "korean": Q6, "chinese": Q6}},
)
resolved = mgr.resolve_mtl_defaults(mtl_dir(VNTL))
check("japanese moves to what is there", resolved["japanese"] == VNTL, str(resolved))
check("korean goes back to null", resolved["korean"] is None, str(resolved))
check("and the file was rewritten", on_disk(mgr)["mtl_defaults"]["korean"] is None)

print("\n--- a saved model that never existed ---")
mgr = manager("bogus", {"mtl_defaults": {"japanese": "not-a-model"}})
resolved = mgr.resolve_mtl_defaults(everything)
check("is replaced by a real one", resolved["japanese"] == Q8, str(resolved))

print("\n--- a saved model that does not serve that language ---")
mgr = manager("mismatch", {"mtl_defaults": {"korean": VNTL}})
resolved = mgr.resolve_mtl_defaults(everything)
check("korean is corrected to the Q8", resolved["korean"] == Q8, str(resolved))


# --- it must not churn the file ---------------------------------------------

print("\n--- nothing to change means nothing is written ---")
mgr = manager("no-churn")
mgr.resolve_mtl_defaults(everything)

stamp = mgr.settings_path.stat().st_mtime_ns
text = mgr.settings_path.read_text(encoding="utf-8")

for _ in range(3):
    mgr.resolve_mtl_defaults(everything)

check("the file was not touched again", mgr.settings_path.stat().st_mtime_ns == stamp)
check("and is byte-identical", mgr.settings_path.read_text(encoding="utf-8") == text)


# --- the other blocks survive ----------------------------------------------

print("\n--- the rest of settings.yaml is left alone ---")
mgr = manager(
    "other-blocks",
    {
        "mtl_defaults": {"japanese": None, "korean": None, "chinese": None},
        "devices": {"paddleocr": "cuda:1", "bubble": "cpu", "translator": "auto"},
        "deepl_api_token": "deepl-secret",
        "jpdb_api_token": "jpdb-secret",
    },
)
mgr.resolve_mtl_defaults(everything)
saved = on_disk(mgr)

check("the defaults were filled in", saved["mtl_defaults"]["japanese"] == Q8, str(saved["mtl_defaults"]))
check("the pinned device survived", saved["devices"]["paddleocr"] == "cuda:1", str(saved["devices"]))
check(
    "and so did all three tokens",
    (
        saved["deepl_api_token"] == "deepl-secret"
        and saved["jpdb_api_token"] == "jpdb-secret"
    ),
    str({k: v for k, v in saved.items() if k.endswith("_token")}),
)


# --- what the manager will now hand the translator --------------------------

print("\n--- the translator can resolve a model for every language ---")
mgr = manager("translator-view")
mgr.resolve_mtl_defaults(everything)

for language in SUPPORTED_MTL_LANGUAGES:
    model_id = mgr.get_default_model(language)
    check(f"{language} -> {model_id}", bool(model_id))
    check(
        f"and {model_id} really serves {language}",
        model_id in [model["id"] for model in mtl_models_for_language(language)],
    )


# --- over HTTP --------------------------------------------------------------

print("\n--- GET /api/settings ---")


def _stub_heavy() -> None:
    """The settings route imports device, which imports torch if it is there."""
    if "torch" not in sys.modules:
        module = types.ModuleType("torch")
        module.cuda = types.SimpleNamespace(
            is_available=lambda: False, device_count=lambda: 0
        )
        module.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        module.version = types.SimpleNamespace(cuda=None)
        sys.modules["torch"] = module


_stub_heavy()

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fox_reader.routes.settings import router  # noqa: E402

mgr = manager("http")
check("null before the request", all(v is None for v in mgr.get_defaults().values()))

app = FastAPI()
app.include_router(router)
app.state.settings = mgr
app.state.mtl_dir = everything
app.state.devices = None

client = TestClient(app, raise_server_exceptions=False)
body = client.get("/api/settings").json()

check("200", "languages" in body, str(sorted(body))[:200])
check("the payload reports the resolved default", body["defaults"]["japanese"] == Q8, str(body["defaults"]))
check("and it is saved, not just displayed", mgr.get_default_model("japanese") == Q8)
check("the file agrees with the page", on_disk(mgr)["mtl_defaults"] == body["defaults"])

selected = [
    model["id"]
    for model in body["languages"]["japanese"]["models"]
    if model["selected"]
]
check("exactly one japanese model is marked selected", selected == [Q8], str(selected))
check(
    "and the Q8 is still selected under korean, where it is the default",
    [m["id"] for m in body["languages"]["korean"]["models"] if m["selected"]] == [Q8],
    str([m["id"] for m in body["languages"]["korean"]["models"] if m["selected"]]),
)
check("korean knows it has one", body["languages"]["korean"]["has_downloaded_model"] is True)

stamp = mgr.settings_path.stat().st_mtime_ns

for _ in range(3):
    client.get("/api/settings")

check("repeated GETs do not rewrite the file", mgr.settings_path.stat().st_mtime_ns == stamp)

print("\n--- GET /api/settings with nothing downloaded ---")
mgr = manager("http-empty")
app.state.settings = mgr
app.state.mtl_dir = none

body = TestClient(app, raise_server_exceptions=False).get("/api/settings").json()
check("every default is null", all(v is None for v in body["defaults"].values()), str(body["defaults"]))
check("and the page says so", body["languages"]["japanese"]["has_downloaded_model"] is False)
check("nothing is marked selected", not any(m["selected"] for m in body["languages"]["japanese"]["models"]))

shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{failures} FAILURE(S)" if failures else "\nMTL defaults are resolved against what is downloaded")
sys.exit(1 if failures else 0)
