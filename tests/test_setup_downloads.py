"""The unified setup downloader: byte progress, completion, the licence gate.

huggingface_hub is stubbed, so no network is touched and no gigabyte ever
moves: the fake hub drives the real ``tqdm_class`` adapter the way the real
one does (open with total/initial, update per chunk, close per file).
"""

import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader.routes import setup as setup_mod


def _bind(ctx):
    setup_mod._dl_ctx.current = ctx


def _unbind():
    setup_mod._dl_ctx.current = None


def _ctx(state=None, lock=None, key="m:0"):
    return {"state": state if state is not None else {"progress": {}},
            "lock": lock if lock is not None else threading.Lock(),
            "key": key}


def test_adapter_reports_bytes_and_survives_garbage():
    state, lock = {"progress": {}}, threading.Lock()
    ctx = _ctx(state, lock)
    _bind(ctx)
    try:
        bar = setup_mod._HubByteProgress(total=1000, initial=200, desc="a.gguf")
        bar.update(300)
        bar.update(-50)  # hub rewinds this far on a Range-ignored retry
        bar.update("junk")  # never let a bad chunk abort the download
        bar.close()
    finally:
        _unbind()
    entry = state["progress"]["m:0"]
    assert entry["current_bytes"] == 450
    assert entry["current_total"] == 1000
    assert entry["bytes_done"] == 450
    assert entry["bytes_total"] == 1000
    assert entry["current_file"] == "a.gguf"


def test_adapter_without_context_is_inert():
    _unbind()
    bar = setup_mod._HubByteProgress(total=10, initial=0, desc="x")
    bar.update(5)
    bar.close()
    with bar:
        bar.update(5)


def test_adapter_unknown_total_stays_countable():
    state, lock = {"progress": {}}, threading.Lock()
    _bind(_ctx(state, lock))
    try:
        bar = setup_mod._HubByteProgress(total=None, initial=0, desc="v.json")
        bar.update(100)
        bar.close()
    finally:
        _unbind()
    entry = state["progress"]["m:0"]
    assert entry["current_total"] == 0
    assert entry["bytes_done"] == 100


class _FakeApi:
    def __init__(self, files):
        self._files = files

    def list_repo_files(self, repo):
        assert repo == "org/model"
        return [".gitattributes", *self._files]


def _install_fake_hub(monkeypatch, payload, fail_on=()):
    """Stub hf_hub_download: write `payload` bytes per file via the real adapter."""
    import huggingface_hub

    calls = []

    class FakeApi(_FakeApi):
        pass

    def fake_download(*, repo_id, filename, local_dir, tqdm_class=None, **kw):
        from pathlib import Path

        calls.append(filename)
        assert tqdm_class is not None, "byte progress must be requested"
        if filename in fail_on:
            raise ConnectionError("simulated network failure")
        total = len(payload)
        bar = tqdm_class(total=total, initial=0, desc=filename)
        with bar:
            for offset in range(0, total, 4096):
                bar.update(min(4096, total - offset))
        (Path(local_dir) / filename).write_bytes(payload)
        return str(Path(local_dir) / filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: FakeApi(["a.gguf", "b.gguf"]))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return calls


def _model():
    return {"id": "fake-model", "name": "Fake", "repo": "org/model",
            "dest": "fake/model", "files": ["a.gguf", "b.gguf"],
            "isRequired": False}


def test_run_downloads_files_bytes_and_marker(tmp_path, monkeypatch):
    payload = b"x" * 10000
    _install_fake_hub(monkeypatch, payload)
    state = {"current_model": None, "progress": {}}
    lock = threading.Lock()
    setup_mod._run_model_downloads([("m:0", _model(), tmp_path)], lock=lock, state=state)

    entry = state["progress"]["m:0"]
    assert entry["done"] is True and entry["downloaded"] is True
    assert entry["files_done"] == 2 and entry["files_total"] == 2
    assert entry["bytes_done"] == 2 * len(payload)
    assert entry["bytes_total"] == 2 * len(payload)
    assert "bytes_base" in entry  # internal until serialised…
    assert "bytes_base" not in setup_mod._public_progress(state["progress"])["m:0"]  # …then stripped
    from fox_reader.constants import MTL_COMPLETION_MARKER

    assert (tmp_path / "fake/model" / MTL_COMPLETION_MARKER).is_file()
    assert (tmp_path / "fake/model/a.gguf").read_bytes() == payload


def test_run_skips_completed_models(tmp_path, monkeypatch):
    from fox_reader.constants import MTL_COMPLETION_MARKER

    dest = tmp_path / "fake/model"
    dest.mkdir(parents=True)
    (dest / MTL_COMPLETION_MARKER).touch()
    for name in ("a.gguf", "b.gguf"):
        (dest / name).write_bytes(b"x")
    calls = _install_fake_hub(monkeypatch, b"x")
    state = {"current_model": None, "progress": {}}
    setup_mod._run_model_downloads([("m:0", _model(), tmp_path)],
                                   lock=threading.Lock(), state=state)
    assert calls == []
    assert state["progress"]["m:0"]["done"] is True


def test_run_refetches_when_the_marker_is_stale(tmp_path, monkeypatch):
    """A marker left by a previous file set must not pass as downloaded.

    Renaming a weight (the .pth -> .safetensors conversions this project has
    already been through) would otherwise leave the app loading a file that is
    not there instead of fetching the one that is now named.
    """
    from fox_reader.constants import MTL_COMPLETION_MARKER

    dest = tmp_path / "fake/model"
    dest.mkdir(parents=True)
    (dest / MTL_COMPLETION_MARKER).touch()
    (dest / "old-name.gguf").write_bytes(b"x")
    calls = _install_fake_hub(monkeypatch, b"x")
    state = {"current_model": None, "progress": {}}
    setup_mod._run_model_downloads([("m:0", _model(), tmp_path)],
                                   lock=threading.Lock(), state=state)
    assert calls == ["a.gguf", "b.gguf"]
    assert state["progress"]["m:0"]["done"] is True


def test_suffix_file_patterns_still_count_as_downloaded(tmp_path):
    """``files`` is also the hub's suffix filter, so both readings must agree."""
    from fox_reader.constants import MTL_COMPLETION_MARKER, model_downloaded

    dest = tmp_path / "fake/model"
    dest.mkdir(parents=True)
    (dest / MTL_COMPLETION_MARKER).touch()
    model = {**_model(), "files": [".gguf"]}

    assert model_downloaded(tmp_path, model) is False  # marker but nothing to load
    (dest / "whatever-it-is-called.gguf").write_bytes(b"x")
    assert model_downloaded(tmp_path, model) is True


def test_run_names_the_model_in_errors(tmp_path, monkeypatch):
    _install_fake_hub(monkeypatch, b"x", fail_on=("a.gguf",))
    state = {"current_model": None, "progress": {}}
    try:
        setup_mod._run_model_downloads([("m:0", _model(), tmp_path)],
                                       lock=threading.Lock(), state=state)
    except RuntimeError as exc:
        assert "Fake" in str(exc) and "a.gguf" in str(exc)
    else:
        raise AssertionError("a failed file must raise")


def _setup_client(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_mod, "_optional_models", lambda categories=None: [
        {"id": "paddleocr-vl-1.6", "name": "VL", "repo": "org/model",
         "dest": "vl/dir", "files": [".gguf"], "type": "gguf",
         "category": "paddleocr_vl", "size_on_disk": "1.8 GiB", "note": "n"},
    ])
    _install_fake_hub(monkeypatch, b"y" * 100)
    # The worker writes under MODELS_DIR; point the module's binding at tmp so
    # every path -- serialisers, pending checks, worker -- agrees on one root.
    # Patched before any status call so "downloaded" starts False.
    monkeypatch.setattr(setup_mod, "MODELS_DIR", tmp_path)
    # Acceptance must land in tmp too: a test may not write the real user's
    # licence record, and must not read one either.
    from fox_reader import model_terms

    monkeypatch.setattr(model_terms, "_store", model_terms.TermsStore(tmp_path))
    app = FastAPI()
    app.include_router(setup_mod.router)
    return TestClient(app)


def _accept(client, *ids):
    resp = client.post("/api/setup/terms/accept", json={"ids": list(ids)})
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _wait_idle(client, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get("/api/setup/optional").json()
        if not payload["active"]:
            return payload
        time.sleep(0.1)
    raise AssertionError("optional download did not finish in time")


def test_optional_endpoints_and_folded_status(tmp_path, monkeypatch):
    client = _setup_client(monkeypatch, tmp_path)

    status = client.get("/api/setup/status").json()
    assert "optional" in status, "optional block must ride the main status"
    assert "ocr_vl" in status, "the VL alias must keep working"
    assert status["optional"]["models"][0]["size_on_disk"] == "1.8 GiB"
    assert status["optional"]["downloaded"] is False
    assert status["optional"]["models"][0]["terms_accepted"] is False

    alias = client.get("/api/setup/ocr-vl").json()
    assert alias["models"] == status["optional"]["models"]

    _accept(client, "paddleocr-vl-1.6")
    resp = client.post("/api/setup/optional/download", json={})
    assert resp.status_code == 200, resp.json()

    final = _wait_idle(client)
    assert final["downloaded"] is True
    assert final["error"] is None
    entry = final["progress"]["paddleocr-vl-1.6"]
    assert entry["done"] is True
    assert entry["bytes_done"] == 200  # 2 files x 100 bytes


def test_optional_download_conflicts_while_active(monkeypatch, tmp_path):
    client = _setup_client(monkeypatch, tmp_path)
    _accept(client, "paddleocr-vl-1.6")
    with setup_mod._optional_lock:
        setup_mod._optional_state["active"] = True
    try:
        resp = client.post("/api/setup/optional/download", json={})
        assert resp.status_code == 409
        assert client.post("/api/setup/ocr-vl/download").status_code == 409
    finally:
        with setup_mod._optional_lock:
            setup_mod._optional_state["active"] = False


# ── The licence gate ─────────────────────────────────────────────────────────

def test_download_refused_until_terms_accepted(tmp_path, monkeypatch):
    """The gate is the server's, not the page's: a direct POST is refused too."""
    client = _setup_client(monkeypatch, tmp_path)

    resp = client.post("/api/setup/optional/download", json={})
    assert resp.status_code == 403
    assert resp.json()["terms_required"] == ["paddleocr-vl-1.6"]

    # The alias route must not be the way around it.
    assert client.post("/api/setup/ocr-vl/download").status_code == 403

    _accept(client, "paddleocr-vl-1.6")
    assert client.post("/api/setup/optional/download", json={}).status_code == 200
    _wait_idle(client)


def test_terms_document_is_served_and_recorded(tmp_path, monkeypatch):
    client = _setup_client(monkeypatch, tmp_path)

    resp = client.get("/api/setup/terms", params={"id": "bubble-segmentation"})
    assert resp.status_code == 200
    document = resp.json()["document"]
    assert resp.json()["accepted"] is False
    assert document["version"]
    # The lineage the user has to see before this one downloads.
    headings = " ".join(section["heading"] for section in document["sections"])
    assert "Manga109" in headings and "Apache" in headings

    assert client.get("/api/setup/terms", params={"id": "nope"}).status_code == 404

    body = _accept(client, "bubble-segmentation")
    assert body["accepted"] == ["bubble-segmentation"]
    assert body["terms"]["bubble-segmentation"]["accepted"] is True
    assert client.get("/api/setup/terms", params={"id": "bubble-segmentation"}).json()["accepted"] is True


def test_terms_accept_rejects_junk(tmp_path, monkeypatch):
    client = _setup_client(monkeypatch, tmp_path)

    assert client.post("/api/setup/terms/accept", json={"ids": []}).status_code == 400
    assert client.post("/api/setup/terms/accept", json={"ids": "x"}).status_code == 400
    assert client.post("/api/setup/terms/accept", json={}).status_code == 400
    # One unknown id fails the whole call rather than silently recording the
    # rest: a partial acceptance the caller did not ask for is worse than an
    # error it can see.
    resp = client.post(
        "/api/setup/terms/accept",
        json={"ids": ["bubble-segmentation", "made-up"]},
    )
    assert resp.status_code == 400
    assert client.get("/api/setup/terms").json()["terms"]["bubble-segmentation"]["accepted"] is False


def test_optional_download_takes_explicit_ids(tmp_path, monkeypatch):
    client = _setup_client(monkeypatch, tmp_path)
    _accept(client, "paddleocr-vl-1.6")

    resp = client.post("/api/setup/optional/download", json={"ids": ["made-up"]})
    assert resp.status_code == 400

    resp = client.post("/api/setup/optional/download", json={"ids": []})
    assert resp.status_code == 400

    resp = client.post(
        "/api/setup/optional/download",
        json={"ids": ["paddleocr-vl-1.6"]},
    )
    assert resp.status_code == 200, resp.json()
    assert _wait_idle(client)["downloaded"] is True


#: The three models Fox Reader cannot start without, in `HF_BASE_MODELS` order.
#: Spelled out here rather than derived, so the set cannot drift quietly: this
#: list is the promise, and the assertions below check the code against it.
REQUIRED_IDS = ["ppocrv6-det", "ppocrv6-rec", "ppocrv5-rec-korean"]


def _mark_downloaded(root, model):
    """Leave `model` under `root` the way a finished download does.

    The completion marker plus every file the entry names -- `model_downloaded`
    wants both, so a marker on its own would not read as downloaded.
    """
    from fox_reader.constants import MTL_COMPLETION_MARKER

    model_dir = root / model["dest"]
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in model.get("files") or ():
        (model_dir / name).write_bytes(b"weights")
    (model_dir / MTL_COMPLETION_MARKER).write_text("done")


class _StubConfig:
    """The one thing `check_models_exist` asks its config for."""

    def __init__(self, mtl_dir):
        self._mtl_dir = mtl_dir

    def resolve_mtl_dir(self, _root):
        return self._mtl_dir


def test_only_paddleocr_is_required(tmp_path, monkeypatch):
    """Bubble and Text Seg are capabilities now, not a condition of starting."""
    assert setup_mod._required_non_mtl_categories() == ["paddleocr"]
    assert set(setup_mod._optional_non_mtl_categories()) == {
        "bubble", "misc", "paddleocr_vl",
    }

    from fox_reader.constants import HF_BASE_MODELS

    required_ids = [
        model["id"]
        for category in setup_mod._required_non_mtl_categories()
        for model in HF_BASE_MODELS[category]
    ]
    assert required_ids == REQUIRED_IDS


def test_the_required_flag_is_set_on_exactly_those_three_ids():
    """The flag itself, by id -- what the category helpers only see in bulk.

    `isRequired` is reduced per category with `any`, and then every model in a
    required category is demanded. Two ways that goes wrong without anything
    failing: the flag moves onto a capability model, or an optional model joins
    a required category and inherits a block it was never meant to cause. So
    this addresses them by id, which is how the rest of the app names them --
    each capability switches off on its own model alone (the Bubble Capture
    button, the Text Seg clean method).
    """
    from fox_reader.constants import BASE_MODEL_INDEX, HF_BASE_MODELS

    flagged = {
        model_id
        for model_id, model in BASE_MODEL_INDEX.items()
        if model.get("isRequired", False)
    }
    assert flagged == set(REQUIRED_IDS)
    # Named outright, not just absent from the set above: these two are the
    # ~1 GiB a user may never want, and re-requiring either one is the exact
    # regression this file exists to catch.
    assert BASE_MODEL_INDEX["bubble-segmentation"]["isRequired"] is False
    assert BASE_MODEL_INDEX["text-segmentation"]["isRequired"] is False

    # A category is all-or-nothing, which is what lets the helpers reduce the
    # flag with `any` and then go on to demand every model in the category.
    for category, models in HF_BASE_MODELS.items():
        if category == "mtl":
            continue
        flags = {bool(model.get("isRequired", False)) for model in models}
        assert len(flags) == 1, f"{category} mixes required and optional models"


def test_the_app_starts_with_only_the_paddleocr_trio_on_disk(tmp_path, monkeypatch):
    """The startup gate itself, against a disk.

    `check_models_exist` is what decides whether the app opens normally or
    redirects to /setup (`app.state.needs_setup`). The category lists imply
    this; nothing checked it. A fresh install that fetched the three required
    models and skipped both capabilities has to pass.
    """
    from fox_reader.constants import HF_BASE_MODELS, MTL_COMPLETION_MARKER

    monkeypatch.setattr(setup_mod, "MODELS_DIR", tmp_path)
    # A copy, so the stage bookkeeping this performs stays inside the test.
    monkeypatch.setattr(setup_mod, "_download_state", dict(setup_mod._download_state))
    config = _StubConfig(tmp_path / "mtl")

    assert setup_mod.check_models_exist(config) is False
    assert setup_mod._incomplete_models == ["paddleocr"]

    for model in HF_BASE_MODELS["paddleocr"]:
        _mark_downloaded(tmp_path, model)

    # Nothing else is on disk: no bubble weights, no Text Seg, no VL, no MTL.
    assert setup_mod.check_models_exist(config) is True
    # The list the setup page's downloader works from. A capability model here
    # would be fetched at first launch whether or not the user asked for it.
    assert setup_mod._incomplete_models == []
    assert setup_mod._required_models_ready() is True

    # And losing one of the three closes the gate again -- the trio is a floor,
    # not a preference. OCR is what every other feature stands on.
    (tmp_path / HF_BASE_MODELS["paddleocr"][1]["dest"] / MTL_COMPLETION_MARKER).unlink()
    assert setup_mod._required_models_ready() is False
    assert setup_mod.check_models_exist(config) is False


def test_every_model_has_terms():
    """A model with no notice could never be downloaded. Catch it here."""
    from fox_reader import model_terms
    from fox_reader.constants import BASE_MODEL_INDEX

    assert not set(BASE_MODEL_INDEX) - set(model_terms.DOCUMENTS)
