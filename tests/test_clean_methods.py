"""Text clean's capability report: what the entry panel may offer on this machine.

Text Seg is an optional model now -- only the three PaddleOCR models are needed
to run Fox Reader -- so ``textseg`` is a method the app routinely ships without
the weights for. The panel greys the option out and refuses the selection
(``frontend/src/entryOptions.ts`` reads ``available``, sets ``option.disabled``
and relabels it ``Text Seg (unavailable)``), and that one flag is the only signal
it gets. Which makes these the assertions behind a control rather than a label:

* :func:`seg.is_available` answers from the disk, so a model that was never
  downloaded reads unavailable with nothing else configured, and a model
  downloaded while the app runs reads available without a restart;
* the report lists every method every time -- an id the panel cannot find has no
  option to disable, and the select renders blank for an entry that stored it;
* the ids and their order match ``CLEAN_METHODS``, the route's pre-model fallback
  list included, so the duplicated literal cannot drift from the real one;
* answering loads no model, because the panel asks as it opens;
* a saved project that stored ``textseg`` while the weights are gone leaves the
  page untouched and says why, rather than erasing the whole region.

No case here reads the real model directory or trusts the real environment: the
two weight paths and the spec lookup are patched every time, so a dev machine
with Text Seg downloaded runs the same assertions as a fresh install.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader.clean import CLEAN_METHODS, METHOD_KNOBS, seg
from fox_reader.routes import inpaint as inpaint_routes
from fox_reader.services.clean_service import CleanJob, CleanService

#: Everything a healthy install can import. A case passes a subset to describe a
#: broken one -- which is a different failure from a missing model, and says so.
BACKEND_MODULES = ("torch", "safetensors")


def install(monkeypatch, tmp_path, *, safetensors=False, legacy=False,
            present=BACKEND_MODULES, raises=()):
    """Point ``seg`` at ``tmp_path`` and say which backend modules import.

    The weight paths are module constants resolved from ``MODELS_DIR`` at import
    time, so they are patched rather than the models directory being faked.
    ``find_spec`` is patched too: the answer must not change with whatever this
    venv happens to have installed. Names outside :data:`BACKEND_MODULES` are
    delegated to the real lookup, so an unrelated lazy import still works.

    Returns the model directory, so a case can create a weight file later and
    prove the report follows the disk rather than a start-up snapshot.
    """
    model_dir = tmp_path / "manga-text-segmentation"
    model_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(seg, "MODEL_DIR", model_dir)
    monkeypatch.setattr(seg, "MODEL_PATH", model_dir / "model.safetensors")
    monkeypatch.setattr(seg, "LEGACY_PATH", model_dir / "model.pth")
    if safetensors:
        (model_dir / "model.safetensors").write_bytes(b"not really weights")
    if legacy:
        (model_dir / "model.pth").write_bytes(b"not really weights")

    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in raises:
            # What a half-removed distribution does: the name resolves far
            # enough to be looked up and then the lookup itself fails.
            raise ValueError(f"{name}.__spec__ is not set")
        if name in BACKEND_MODULES:
            return object() if name in present else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    return model_dir


def no_model_loads(monkeypatch):
    """Make any attempt to build or run the network a failure, not a stall."""
    def boom(*_args, **_kwargs):
        raise AssertionError("the capability report loaded the model")

    monkeypatch.setattr(seg, "load_model", boom)
    monkeypatch.setattr(seg, "detect_at", boom)


def ocr_with(detector):
    """The one attribute ``available_methods`` asks the OCR engine for."""
    return type("StubOCR", (), {"detector": detector})()


# ---------------------------------------------------------------- the disk gate

class TestSegAvailability:
    """:func:`seg.is_available` -- the only thing that decides the grey-out."""

    def test_a_model_that_was_never_downloaded_names_the_directory(
            self, monkeypatch, tmp_path):
        model_dir = install(monkeypatch, tmp_path)

        ok, why = seg.is_available()

        assert ok is False
        # The reason is shown to the user verbatim, under the Method select, so
        # it has to say where the app looked.
        assert str(model_dir) in why

    def test_safetensors_weights_are_enough(self, monkeypatch, tmp_path):
        install(monkeypatch, tmp_path, safetensors=True)

        assert seg.is_available() == (True, "")

    def test_a_fresh_clone_carrying_only_model_pth_counts(
            self, monkeypatch, tmp_path):
        # model.pth is torch's own format, so the safetensors reader is not
        # needed for it. Refusing here would report a method as broken when it
        # is merely slower to load.
        install(monkeypatch, tmp_path, legacy=True, present=("torch",))

        assert seg.is_available() == (True, "")

    def test_a_directory_is_not_a_weight_file(self, monkeypatch, tmp_path):
        model_dir = install(monkeypatch, tmp_path)
        (model_dir / "model.safetensors").mkdir()

        ok, _ = seg.is_available()

        assert ok is False

    def test_weights_without_their_reader_is_a_broken_install(
            self, monkeypatch, tmp_path):
        model_dir = install(monkeypatch, tmp_path, safetensors=True,
                            present=("torch",))

        ok, why = seg.is_available()

        assert ok is False
        assert "safetensors" in why
        # There is no optional extra to point at: this is a damaged install, and
        # the remedy differs from "download the model".
        assert "reinstall" in why
        assert str(model_dir) not in why

    def test_no_torch_does_not_blame_the_model(self, monkeypatch, tmp_path):
        model_dir = install(monkeypatch, tmp_path, safetensors=True,
                            present=("safetensors",))

        ok, why = seg.is_available()

        assert ok is False
        assert "torch" in why
        assert str(model_dir) not in why

    def test_a_spec_lookup_that_raises_counts_as_absent(
            self, monkeypatch, tmp_path):
        install(monkeypatch, tmp_path, safetensors=True, raises=("torch",))

        ok, why = seg.is_available()

        # Not a traceback out of a panel-open request: the method is reported
        # unusable, the other two keep working.
        assert ok is False
        assert "torch" in why


# ------------------------------------------------------------- the report built

class TestAvailableMethods:
    """:meth:`CleanService.available_methods` -- the payload the panel reads."""

    def test_every_method_is_listed_even_when_none_of_them_can_run(
            self, monkeypatch, tmp_path):
        install(monkeypatch, tmp_path)
        no_model_loads(monkeypatch)

        report = CleanService(None).available_methods()

        # Order matters as much as membership: the panel renders the options in
        # this order, and an id it cannot find has no option to disable.
        assert [m["id"] for m in report] == list(CLEAN_METHODS)
        for info in report:
            assert set(info) == {"id", "label", "available", "reason", "knobs"}
            assert isinstance(info["available"], bool)
            assert isinstance(info["reason"], str)
            assert info["label"]
        by_id = {m["id"]: m for m in report}
        # Selected Region needs no model at all -- which is why it is the
        # default, and why a machine with nothing downloaded can still clean.
        assert by_id["region"]["available"] is True
        assert by_id["region"]["reason"] == ""
        assert by_id["textseg"]["available"] is False
        assert by_id["ppocr"]["available"] is False

    def test_a_download_while_the_app_runs_flips_the_flag(
            self, monkeypatch, tmp_path):
        model_dir = install(monkeypatch, tmp_path)
        no_model_loads(monkeypatch)
        service = CleanService(None)

        before = {m["id"]: m for m in service.available_methods()}["textseg"]
        assert before["available"] is False
        assert before["reason"]

        # The optional models are downloaded from the setup page, which the user
        # can reach with the app already running. Nothing restarts, so the
        # report has to be read off the disk on every ask, not cached at
        # start-up.
        (model_dir / "model.safetensors").write_bytes(b"not really weights")

        after = {m["id"]: m for m in service.available_methods()}["textseg"]
        assert after["available"] is True
        assert after["reason"] == ""

    @pytest.mark.parametrize("detector, available", [
        (object(), True),
        (None, False),
    ])
    def test_ppocr_follows_the_loaded_detector(self, monkeypatch, tmp_path,
                                               detector, available):
        install(monkeypatch, tmp_path)
        no_model_loads(monkeypatch)

        report = CleanService(ocr_with(detector)).available_methods()
        ppocr = {m["id"]: m for m in report}["ppocr"]

        assert ppocr["available"] is available
        assert bool(ppocr["reason"]) is not available

    def test_an_unavailable_method_still_carries_its_knobs(
            self, monkeypatch, tmp_path):
        install(monkeypatch, tmp_path)
        no_model_loads(monkeypatch)

        report = CleanService(None).available_methods()

        for info in report:
            # Lists, not tuples: this is handed to the panel as JSON, and the
            # catalogue lives here so a method whose tuning changed cannot
            # advertise a switch the backend stopped reading.
            assert isinstance(info["knobs"], list)
            assert info["knobs"] == list(METHOD_KNOBS[info["id"]])
        # An older project may have stored `textseg`, and the panel renders the
        # tweak row for whatever is stored. Dropping the knobs along with the
        # availability would take those controls away mid-project.
        assert {m["id"]: m for m in report}["textseg"]["knobs"] == [
            "glow", "speed", "tta", "tile"]

    def test_answering_loads_no_model(self, monkeypatch, tmp_path):
        install(monkeypatch, tmp_path, safetensors=True)
        no_model_loads(monkeypatch)
        cached = seg._model

        # The panel asks as it opens, so this has to be cheap: a disk check and
        # a spec lookup. `no_model_loads` turns a 216 MB build into a failure.
        report = CleanService(ocr_with(object())).available_methods()

        assert {m["id"]: m for m in report}["textseg"]["available"] is True
        # Compared against what was there, not against None: another module in
        # the same session is allowed to have loaded it.
        assert seg._model is cached


# -------------------------------------------------------------------- the route

def app_for(clean):
    app = FastAPI()
    app.include_router(inpaint_routes.router)
    app.state.clean = clean
    return app


def methods_from(clean):
    response = TestClient(app_for(clean)).get("/api/clean/methods")
    assert response.status_code == 200
    return response.json()


class TestCleanMethodsRoute:
    """``GET /api/clean/methods`` -- asked once per panel open."""

    def test_the_panel_can_open_before_the_models_are_built(self):
        # Start-up, or a build with no models at all. The panel still has to
        # open; `region` genuinely works without them.
        payload = methods_from(None)
        by_id = {m["id"]: m for m in payload["methods"]}

        assert by_id["region"]["available"] is True
        assert by_id["textseg"]["available"] is False
        assert by_id["textseg"]["reason"]
        assert by_id["ppocr"]["available"] is False

    def test_the_pre_model_list_cannot_drift_from_the_real_one(
            self, monkeypatch, tmp_path):
        install(monkeypatch, tmp_path)
        no_model_loads(monkeypatch)

        early = methods_from(None)["methods"]
        built = CleanService(None).available_methods()

        # Two sources for one list. They may disagree about availability and
        # about the wording of a reason; they may not disagree about which
        # methods exist, what they are called, or which knobs they read.
        assert [m["id"] for m in early] == [m["id"] for m in built]
        assert [m["label"] for m in early] == [m["label"] for m in built]
        assert [m["knobs"] for m in early] == [m["knobs"] for m in built]

    def test_the_report_is_passed_through_untouched(self):
        asked = []
        reported = [
            {"id": "region", "label": "Selected Region", "available": True,
             "reason": "", "knobs": []},
            {"id": "ppocr", "label": "PaddleOCR", "available": True,
             "reason": "", "knobs": []},
            {"id": "textseg", "label": "Text Seg", "available": False,
             "reason": "the Text Seg model is missing from /models/text-seg",
             "knobs": ["glow", "speed", "tta", "tile"]},
        ]

        class StubClean:
            def available_methods(self):
                asked.append(1)
                return reported

        payload = methods_from(StubClean())

        assert payload["methods"] == reported
        # Once per request, not once per method: this runs in the threadpool and
        # each call stats the model directory.
        assert asked == [1]

    def test_the_payload_never_offers_a_default_it_did_not_list(self):
        payload = methods_from(None)

        # A default that is not in the list renders the select blank, which is
        # how a fill that this OpenCV build cannot run used to look.
        assert payload["default_fill"] in payload["fills"]
        assert payload["default_speed"] in [s["value"] for s in payload["speeds"]]
        assert payload["default_tile"] in [t["value"] for t in payload["tiles"]]


# ------------------------------------------------------- the saved-project path

def test_a_project_that_stored_textseg_without_the_weights_is_left_alone(
        monkeypatch, tmp_path):
    """The failure mode that matters once the model is optional.

    ``entry.clean`` is persisted. A project saved on a machine that had Text Seg
    opens on one that does not, and the panel can only grey out what the user
    picks *next* -- what is already stored still reaches the backend.
    """
    install(monkeypatch, tmp_path)

    def unavailable(*_args, **_kwargs):
        raise seg.SegUnavailable("the Text Seg model is missing")

    monkeypatch.setattr(seg, "detect_at", unavailable)
    monkeypatch.setattr(seg, "load_model", unavailable)

    page = np.full((40, 40, 3), 200, np.uint8)
    job = CleanJob(polygon=[(5, 5), (30, 5), (30, 30), (5, 30)],
                   method="textseg")

    out, plan = CleanService(None).clean(page, [job])

    # Not a 500, and not the whole region erased either: removing everything
    # inside the box would silently do something the user did not ask for.
    assert out is page
    assert plan.notes and "Text Seg" in plan.notes[0]
