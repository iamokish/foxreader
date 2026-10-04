import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader.routes import ocr as ocr_routes


class StubOCR:
    def __init__(self):
        self.calls = []

    def predict(self, *, lang, image, grayscale):
        self.calls.append({
            "lang": lang,
            "grayscale": grayscale,
            "size": image.size,
            "mode": image.mode,
            "thread": threading.current_thread().name,
        })
        return "recognized text"


def _threadpool_spy(monkeypatch):
    """Record what gets offloaded, and run it on a real worker thread.

    Running the callable somewhere other than the calling thread is the point:
    it lets the test tell decode-on-the-event-loop from decode-in-the-pool by
    comparing thread names, rather than by naming the callable.
    """
    hops = []

    async def run_in_threadpool(func, *args, **kwargs):
        hops.append(func)
        box = {}

        def _run():
            box["value"] = func(*args, **kwargs)

        worker = threading.Thread(target=_run, name="fox-test-worker")
        worker.start()
        worker.join()
        return box["value"]

    monkeypatch.setattr(ocr_routes, "run_in_threadpool", run_in_threadpool)
    return hops


def _app(sample_image, ocr):
    app = FastAPI()
    app.include_router(ocr_routes.router)
    app.state.session = type("Session", (), {"current_dir": str(sample_image.parent)})()
    app.state.ocr = ocr
    return app


def test_crop_ocr_runs_prediction_in_threadpool(sample_image, monkeypatch):
    hops = _threadpool_spy(monkeypatch)
    ocr = StubOCR()

    response = TestClient(_app(sample_image, ocr)).post(
        "/api/ocr_crop",
        json={
            "filename": sample_image.name,
            "x": 0,
            "y": 0,
            "width": 20,
            "height": 20,
            "lang": "en",
            "isGrayScale": False,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"text": "recognized text"}

    # One hop, not two: opening the page, clamping the box against its real
    # dimensions, cropping and converting all belong on the same side of the
    # boundary as the model call. `Image.open` is lazy, so a route that
    # offloaded only `predict` would still decode on the event loop.
    assert len(hops) == 1
    assert [call["thread"] for call in ocr.calls] == ["fox-test-worker"]

    # The box was clamped and the crop converted before the model saw it.
    assert ocr.calls[0]["size"] == (20, 20)
    assert ocr.calls[0]["mode"] == "RGB"
    assert ocr.calls[0]["lang"] == "en"
    assert ocr.calls[0]["grayscale"] is False


def test_crop_ocr_clamps_the_box_to_the_page(sample_image, monkeypatch):
    """A box running off the edge is trimmed, not an error.

    `sample_image` is 100x100; the request asks for a 400x400 region at (60,60).
    """
    _threadpool_spy(monkeypatch)
    ocr = StubOCR()

    response = TestClient(_app(sample_image, ocr)).post(
        "/api/ocr_crop",
        json={
            "filename": sample_image.name,
            "x": 60,
            "y": 60,
            "width": 400,
            "height": 400,
            "lang": "en",
            "isGrayScale": False,
        },
    )

    assert response.status_code == 200
    assert ocr.calls[0]["size"] == (40, 40)


def test_freeform_ocr_runs_the_whole_composite_in_threadpool(sample_image, monkeypatch):
    """The freeform path is page-sized, so none of it may sit on the loop."""
    hops = _threadpool_spy(monkeypatch)
    ocr = StubOCR()

    response = TestClient(_app(sample_image, ocr)).post(
        "/api/ocr_freeform",
        json={
            "filename": sample_image.name,
            "points": [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 50, "y": 40}],
            "lang": "en",
            "isGrayScale": False,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"text": "recognized text"}
    assert len(hops) == 1
    assert [call["thread"] for call in ocr.calls] == ["fox-test-worker"]

    # Cropped to the polygon's bounding box, and flattened to RGB. 41x31, not
    # 40x30: `polygon(outline=255)` paints the boundary pixels too, and
    # `getbbox()` reports an exclusive right/lower edge.
    assert ocr.calls[0]["size"] == (41, 31)
    assert ocr.calls[0]["mode"] == "RGB"


def test_freeform_ocr_rejects_a_missing_file(sample_image, monkeypatch):
    """Guarded before any decode, so a bad filename costs nothing."""
    hops = _threadpool_spy(monkeypatch)
    ocr = StubOCR()

    response = TestClient(_app(sample_image, ocr)).post(
        "/api/ocr_freeform",
        json={
            "filename": "does-not-exist.png",
            "points": [{"x": 0, "y": 0}, {"x": 9, "y": 0}, {"x": 9, "y": 9}],
            "lang": "en",
            "isGrayScale": False,
        },
    )

    assert response.status_code == 404
    assert hops == []
    assert ocr.calls == []


def test_freeform_ocr_rejects_a_closed_folder(sample_image, monkeypatch):
    hops = _threadpool_spy(monkeypatch)
    ocr = StubOCR()
    app = _app(sample_image, ocr)
    app.state.session = type("Session", (), {"current_dir": ""})()

    response = TestClient(app).post(
        "/api/ocr_freeform",
        json={
            "filename": sample_image.name,
            "points": [{"x": 0, "y": 0}, {"x": 9, "y": 0}, {"x": 9, "y": 9}],
            "lang": "en",
            "isGrayScale": False,
        },
    )

    assert response.status_code == 400
    assert hops == []
    assert ocr.calls == []
