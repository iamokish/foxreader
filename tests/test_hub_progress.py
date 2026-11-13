"""Real Hub entrypoints with local transports: no models or network required."""
from __future__ import annotations

import gzip
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader import hub_progress as progress_mod
from fox_reader.routes import setup


@pytest.fixture(autouse=True)
def progress_scope(monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(progress_mod.download_context, "current", None, raising=False)
    # Optional agent detection adds a request depending on the machine's cache.
    # These transport fixtures exercise model downloads, not that telemetry.
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_TELEMETRY", True)


@pytest.fixture
def clock(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(progress_mod, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


def _bind(samples):
    progress = progress_mod.FileProgress(lambda count, total, phase: samples.append((count, total, phase)))
    progress_mod.download_context.current = progress
    return progress


def _client(monkeypatch, handler, *, response_hooks=()):
    from huggingface_hub.utils import _http

    client = httpx.Client(transport=httpx.MockTransport(handler), event_hooks={"response": list(response_hooks)})
    monkeypatch.setattr(_http, "_GLOBAL_CLIENT", client)
    progress_mod.observe_hub_client()
    return client


class _Stream(httpx.SyncByteStream):
    def __init__(self, chunks, *, after_chunk=None, fail=None):
        self.chunks, self.after_chunk, self.fail = chunks, after_chunk, fail
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            yield chunk
            if self.after_chunk:
                self.after_chunk()
        if self.fail:
            raise self.fail

    def close(self):
        self.closed = True


def test_real_hub_download_reports_small_chunks_before_ten_mib_buffer(tmp_path, monkeypatch):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    samples, intermediate, existing_hooks = [], [], []
    progress = _bind(samples)
    chunk = b"model bytes" * 8192
    stream = _Stream([chunk] * 3, after_chunk=lambda: intermediate.append(samples[-1]))

    def handle(request):
        headers = {"Content-Length": str(len(chunk) * 3), "ETag": '"test-model"', "X-Repo-Commit": "a" * 40}
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        assert request.method == "GET"
        return httpx.Response(200, headers=headers, stream=stream)

    with (
        _client(monkeypatch, handle, response_hooks=[lambda response: existing_hooks.append(response.request.method)]),
        disable_progress_bars(),
    ):
        progress_mod.observe_hub_client()  # registering twice must not double-count
        destination = hf_hub_download("org/model", "weights.bin", local_dir=tmp_path,
                                      tqdm_class=progress_mod.HubByteProgress)
    assert 0 < intermediate[0][0] < len(chunk) * 3 < 10 * 1024 * 1024
    assert [item[0] for item in intermediate] == [len(chunk), len(chunk) * 2, len(chunk) * 3]
    assert samples[-1] == (len(chunk) * 3, len(chunk) * 3, "finalizing")
    assert Path(destination).read_bytes() == chunk * 3
    assert existing_hooks == ["HEAD", "GET"]
    assert stream.closed
    progress.finish()


@pytest.mark.parametrize("resume_status", [200, 206])
def test_real_http_retry_resumes_or_restarts_without_double_counting(monkeypatch, resume_status):
    from huggingface_hub import file_download

    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    monkeypatch.setattr(file_download, "time", SimpleNamespace(sleep=lambda _: None))
    samples, requests, intermediate = [], [], []
    progress = _bind(samples)
    prefix = b"a" * (10 * 1024 * 1024)
    tail = b"b" * 65536
    size = len(prefix) + len(tail)

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            # Hub commits the first 10 MiB, then loses an incomplete buffered
            # chunk. The observer sees it, so the next attempt must rewind.
            return httpx.Response(200, headers={"Content-Length": str(size)}, stream=_Stream(
                [prefix, b"discarded"], fail=httpx.ReadTimeout("fixture interrupted"),
            ))
        assert request.headers["Range"] == f"bytes={len(prefix)}-"
        headers = {"Content-Length": str(size if resume_status == 200 else len(tail))}
        if resume_status == 206:
            headers["Content-Range"] = f"bytes {len(prefix)}-{size - 1}/{size}"
        chunks = [b"c" * len(prefix), tail] if resume_status == 200 else [tail]
        return httpx.Response(resume_status, headers=headers, stream=_Stream(
            chunks, after_chunk=lambda: intermediate.append(samples[-1][0]),
        ))

    output = io.BytesIO()
    with _client(monkeypatch, handle):
        file_download.http_get("https://fixture.invalid/model", output, expected_size=size,
                               tqdm_class=progress_mod.HubByteProgress)
    assert output.getvalue() == (b"c" * len(prefix) if resume_status == 200 else prefix) + tail
    assert samples[-1] == (size, size, "finalizing")
    assert max(intermediate) == size
    assert any(count == (0 if resume_status == 200 else len(prefix)) for count, _, _ in samples[3:])
    progress.finish()


def test_retry_on_replacement_client_without_observer_falls_back_to_hub_progress(monkeypatch):
    from huggingface_hub import file_download
    from huggingface_hub.utils import _http

    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    monkeypatch.setattr(file_download, "time", SimpleNamespace(sleep=lambda _: None))
    samples, intermediate = [], []
    progress = _bind(samples)
    prefix, tail = b"b" * (10 * 1024 * 1024), b"c" * 65536
    size = len(prefix) + len(tail)

    def replacement_response(request):
        assert "Range" not in request.headers  # failed bytes were still buffered
        return httpx.Response(200, headers={"Content-Length": str(size)}, stream=_Stream(
            [prefix, tail], after_chunk=lambda: intermediate.append(samples[-1][0]),
        ))

    with httpx.Client(transport=httpx.MockTransport(replacement_response)) as replacement:
        def first_response(request):
            return httpx.Response(200, headers={"Content-Length": str(size)}, stream=_Stream(
                [b"discarded"], fail=httpx.ReadTimeout("fixture interrupted"),
                after_chunk=lambda: monkeypatch.setattr(_http, "_GLOBAL_CLIENT", replacement),
            ))

        output = io.BytesIO()
        with _client(monkeypatch, first_response):
            file_download.http_get("https://fixture.invalid/model", output, expected_size=size,
                                   tqdm_class=progress_mod.HubByteProgress)
        assert replacement.event_hooks["response"] == []
    assert intermediate[0] == len(prefix) < size
    assert samples[-1] == (size, size, "finalizing")
    assert output.getvalue() == prefix + tail
    progress.finish()


@pytest.mark.parametrize("encoding", ["identity", "gzip"])
def test_unknown_length_and_compressed_http_keep_correct_saved_byte_counts(monkeypatch, encoding):
    from huggingface_hub import file_download

    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    payload = b"decoded model data" * 1000
    wire = gzip.compress(payload) if encoding == "gzip" else payload
    samples = []
    progress = _bind(samples)

    def handle(request):
        return httpx.Response(200, headers={"Content-Encoding": encoding}, stream=_Stream([wire]))

    output = io.BytesIO()
    with _client(monkeypatch, handle):
        file_download.http_get("https://fixture.invalid/model", output, tqdm_class=progress_mod.HubByteProgress)
    assert output.getvalue() == payload
    assert samples[-1] == (len(payload), 0, "finalizing")
    progress.finish()


def test_real_xet_entrypoint_accepts_native_thread_progress_and_ignores_late_callbacks(monkeypatch):
    from huggingface_hub import file_download
    from huggingface_hub.utils import _xet

    # Importing native hf_xet initializes its logging. Substitute only this
    # module and session; exercise Hub's real Xet-to-tqdm callback bridge.
    native = ModuleType("hf_xet")
    native.__spec__ = ModuleSpec("hf_xet", loader=None)
    native.XetFileInfo = lambda *args: args
    monkeypatch.setitem(sys.modules, "hf_xet", native)
    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    samples, partial, callbacks = [], [], []
    progress = _bind(samples)

    class Group:
        def __init__(self, callback):
            self.callback = callback
            callbacks.append(callback)

        def __enter__(self):
            return self

        def start_download_file(self, *_args):
            def report():
                for count in (25, 60, 100):
                    self.callback(SimpleNamespace(total_bytes_completed=count), {})
                    partial.append(samples[-1][0])
            worker = threading.Thread(target=report)
            worker.start()
            worker.join(5)
            assert not worker.is_alive()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(_xet, "get_xet_session", lambda: SimpleNamespace(
        new_file_download_group=lambda **kwargs: Group(kwargs["progress_callback"]),
    ))
    file_download.xet_get(incomplete_path=Path("unused"), headers={}, expected_size=100,
                          xet_file_data=SimpleNamespace(refresh_route="unused", file_hash="0" * 64),
                          tqdm_class=progress_mod.HubByteProgress)
    assert partial == [25, 60, 100]
    progress.finish(100)
    before = list(samples)
    callbacks[0](SimpleNamespace(total_bytes_completed=200), {})
    assert samples == before


def test_native_bridge_reports_wire_bytes_when_assembled_bytes_stay_zero(monkeypatch):
    from huggingface_hub import file_download
    from huggingface_hub.utils import _xet

    native = ModuleType("hf_xet")
    native.__spec__ = ModuleSpec("hf_xet", loader=None)
    native.XetFileInfo = lambda *args: args
    monkeypatch.setitem(sys.modules, "hf_xet", native)
    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    state = {"progress": {"model": setup._fresh_progress_entry(1)}}
    progress = setup._file_progress(state, threading.Lock(), "model", state["progress"]["model"])
    progress_mod.download_context.current = progress
    partial, callbacks = [], []

    def report(transferred, assembled=0):
        return SimpleNamespace(total_bytes_completed=assembled, total_transfer_bytes_completed=transferred,
                               total_transfer_bytes=100)

    class Group:
        def __init__(self, callback):
            self.callback = callback
            callbacks.append(callback)

        def __enter__(self):
            return self

        def start_download_file(self, *_args):
            def send_progress():
                for count in (25, 60, 100):
                    self.callback(report(count), {})
                    partial.append(setup._public_progress(state["progress"])["model"])
            thread = threading.Thread(target=send_progress)
            thread.start()
            thread.join(5)
            assert not thread.is_alive()

        def __exit__(self, *_args):
            self.callback(report(100, assembled=140), {})

    session = SimpleNamespace(new_file_download_group=lambda **kwargs: Group(kwargs["progress_callback"]))
    monkeypatch.setattr(_xet, "get_xet_session", lambda: session)
    progress_mod.observe_hub_client()
    installed = _xet.get_xet_session
    progress_mod.observe_hub_client()
    assert _xet.get_xet_session is installed
    file_download.xet_get(incomplete_path=Path("unused"), headers={}, expected_size=140,
                          xet_file_data=SimpleNamespace(refresh_route="unused", file_hash="0" * 64),
                          tqdm_class=progress_mod.HubByteProgress)
    assert [entry["transfer_bytes"] for entry in partial] == [25, 60, 100]
    assert all(entry["current_bytes"] == 0 and entry["current_total"] == 140 for entry in partial)
    assert partial[0]["phase"] == "downloading" and partial[-1]["phase"] == "finalizing"
    assert all(entry["transfer_total"] == 100 and entry["done"] is False for entry in partial)
    # Hub's original callback still delivers the eventual assembled file size.
    assert state["progress"]["model"]["current_bytes"] == 140
    progress.finish(140)
    before = setup._public_progress(state["progress"])
    callbacks[0](report(200, assembled=280), {})
    assert setup._public_progress(state["progress"]) == before
    progress_mod.download_context.current = None
    assert _xet.get_xet_session() is session


def test_native_proxy_preserves_arguments_callback_result_errors_and_other_methods():
    samples, received = [], []
    progress = _bind(samples)
    group = object()

    def create_group(*args, **kwargs):
        received.append((args, kwargs))
        return group

    session = SimpleNamespace(new_file_download_group=create_group, status=lambda: "running")
    proxy = progress_mod._XetProgressSession(session, progress)
    callbacks = []

    def callback(report, items):
        callbacks.append((report, items))
        return "preserved result"

    assert proxy.new_file_download_group("endpoint", custom_headers={"test": "header"},
                                         progress_callback=callback, progress_interval_ms=200) is group
    args, kwargs = received[-1]
    assert args == ("endpoint",) and kwargs["custom_headers"] == {"test": "header"}
    assert kwargs["progress_interval_ms"] == 200
    report = SimpleNamespace(total_transfer_bytes_completed=25, total_transfer_bytes=100)
    assert kwargs["progress_callback"](report, {"item": 1}) == "preserved result"
    assert callbacks == [(report, {"item": 1})]
    assert progress.transfer_snapshot() == {"transfer_bytes": 25, "transfer_total": 100}
    assert proxy.status() == "running"

    def fail(*_args):
        raise ValueError("original callback failure")

    proxy.new_file_download_group(progress_callback=fail)
    with pytest.raises(ValueError, match="original callback failure"):
        received[-1][1]["progress_callback"](report, {})
    # Unknown/positional callback shapes are forwarded unchanged.
    assert proxy.new_file_download_group("endpoint", callback) is group
    assert received[-1] == (("endpoint", callback), {})


def test_native_transfer_totals_can_grow_without_claiming_completion(monkeypatch):
    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    samples = []
    progress = _bind(samples)
    progress.bar_started(140, 0)
    progress.native_transfer(SimpleNamespace(total_transfer_bytes_completed=100, total_transfer_bytes=100))
    assert samples[-1] == (0, 140, "finalizing")
    progress.native_transfer(SimpleNamespace(total_transfer_bytes_completed=110, total_transfer_bytes=200))
    assert samples[-1] == (0, 140, "downloading")
    assert progress.transfer_snapshot() == {"transfer_bytes": 110, "transfer_total": 200}
    progress.native_transfer(SimpleNamespace())  # older native report shape
    assert progress.transfer_snapshot() == {"transfer_bytes": 110, "transfer_total": 200}


def test_optional_observer_registration_cannot_break_a_custom_hub_client(monkeypatch):
    from huggingface_hub import utils

    monkeypatch.setattr(utils, "get_session", lambda: SimpleNamespace(event_hooks=None))
    progress_mod.observe_hub_client()


def test_native_transfer_counters_are_cleared_for_the_next_file_and_model_completion(tmp_path, monkeypatch):
    import huggingface_hub

    root = tmp_path / "model"
    root.mkdir()
    for name in ("a.bin", "b.bin"):
        (root / name).write_bytes(b"cached model data")
    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: object())
    monkeypatch.setattr(setup, "_discover_model_files", lambda *_args: ["a.bin", "b.bin"])
    state, old = {"progress": {}}, []
    report = SimpleNamespace(total_transfer_bytes_completed=10, total_transfer_bytes=20)

    def download(filename, **_kwargs):
        progress = progress_mod.download_context.current
        if filename == "a.bin":
            old.append(progress)
            progress.native_transfer(report)
            assert state["progress"]["model"]["transfer_bytes"] == 10
        else:
            old[0].native_transfer(report)  # delayed callback from the previous file
            assert progress.transfer_snapshot() == {}
            assert "transfer_bytes" not in state["progress"]["model"]
            assert "transfer_total" not in state["progress"]["model"]
        return str(root / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    model = {"id": "model", "repo": "fixture/model", "dest": "model", "files": ["a.bin", "b.bin"]}
    setup._run_model_downloads([("model", model, tmp_path)], lock=threading.Lock(), state=state)
    assert state["progress"]["model"]["done"] is True
    assert "transfer_bytes" not in state["progress"]["model"]
    assert "transfer_total" not in state["progress"]["model"]


def test_shared_native_session_isolates_simultaneous_groups_and_late_transfer_callbacks(monkeypatch):
    from huggingface_hub.utils import _xet

    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    session = SimpleNamespace(new_file_download_group=lambda **kwargs: SimpleNamespace(
        callback=kwargs["progress_callback"],
    ))
    monkeypatch.setattr(_xet, "get_xet_session", lambda: session)
    progress_mod._observe_xet_progress()
    states = {name: {"progress": {"model": setup._fresh_progress_entry(1)}} for name in ("a", "b")}
    locks = {name: threading.Lock() for name in states}

    def group_for(name):
        state = states[name]
        progress = setup._file_progress(state, locks[name], "model", state["progress"]["model"])
        progress_mod.download_context.current = progress
        try:
            group = _xet.get_xet_session().new_file_download_group(progress_callback=lambda *_args: None)
            return progress, group
        finally:
            progress_mod.download_context.current = None

    barrier = threading.Barrier(2)

    def notify(group, scale):
        assert getattr(progress_mod.download_context, "current", None) is None
        barrier.wait(5)
        for count in (10, 20, 30):
            group.callback(SimpleNamespace(total_transfer_bytes_completed=count * scale,
                                           total_transfer_bytes=50 * scale), {})

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = workers.submit(group_for, "a"), workers.submit(group_for, "b")
        progress_a, group_a = first.result(timeout=5)
        progress_b, group_b = second.result(timeout=5)
        first, second = workers.submit(notify, group_a, 1), workers.submit(notify, group_b, 10)
        first.result(timeout=5)
        second.result(timeout=5)
    assert states["a"]["progress"]["model"]["transfer_bytes"] == 30
    assert states["b"]["progress"]["model"]["transfer_bytes"] == 300

    progress_a.finish()
    states["a"]["progress"]["model"] = setup._fresh_progress_entry(1)
    next_progress, next_group = group_for("a")
    next_group.callback(SimpleNamespace(total_transfer_bytes_completed=5, total_transfer_bytes=60), {})
    group_a.callback(SimpleNamespace(total_transfer_bytes_completed=999, total_transfer_bytes=1000), {})
    group_b.callback(SimpleNamespace(total_transfer_bytes_completed=350, total_transfer_bytes=500), {})
    assert states["a"]["progress"]["model"]["transfer_bytes"] == 5
    assert states["a"]["progress"]["model"]["transfer_total"] == 60
    assert states["b"]["progress"]["model"]["transfer_bytes"] == 350
    next_progress.finish()
    progress_b.finish()


def test_publication_is_throttled_but_boundaries_are_flushed(clock):
    samples = []
    progress = _bind(samples)
    bar = progress_mod.HubByteProgress(total=100)
    for _ in range(20):
        clock[0] += 0.01
        bar.update(1)
    assert samples == [(0, 100, "downloading")]
    clock[0] += 0.1
    bar.update(1)
    assert samples[-1] == (21, 100, "downloading")
    bar.update(1)
    bar.close()
    assert samples[-1] == (22, 100, "finalizing")
    progress.finish()


@pytest.mark.parametrize("replace_job", [False, True])
def test_late_progress_cannot_replace_another_file_or_job(replace_job):
    state = {"progress": {"model": setup._fresh_progress_entry(2)}}
    lock = threading.Lock()
    old = setup._file_progress(state, lock, "model", state["progress"]["model"])
    if replace_job:
        state["progress"] = {"model": setup._fresh_progress_entry(2)}
    current = setup._file_progress(state, lock, "model", state["progress"]["model"])
    current.bar_started(100, 25)
    old.bar_started(999, 500)
    old.finish(999)
    public = setup._public_progress(state["progress"])["model"]
    assert public["current_bytes"] == 25 and public["current_total"] == 100
    assert "_file_token" not in public


def test_failed_stream_closes_and_unbound_requests_do_not_change_progress(monkeypatch):
    from huggingface_hub import file_download

    samples = []
    progress = _bind(samples)
    failed = _Stream([b"partial"], fail=ValueError("fixture failure"))
    with _client(monkeypatch, lambda request: httpx.Response(200, stream=failed)) as client:
        with pytest.raises(ValueError, match="fixture failure"):
            file_download.http_get("https://fixture.invalid/model", io.BytesIO(),
                                   tqdm_class=progress_mod.HubByteProgress)
        assert failed.closed
        assert samples[-1][2] == "downloading"
        progress.finish()
        before = list(samples)
        progress_mod.download_context.current = None
        with pytest.raises(ValueError, match="fixture failure"):
            client.get("https://fixture.invalid/unrelated")
        assert samples == before


def test_cached_hub_returns_without_callbacks_use_saved_file_sizes(tmp_path, monkeypatch):
    import huggingface_hub

    from fox_reader.constants import MTL_COMPLETION_MARKER

    root = tmp_path / "model"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"cached weights")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: object())
    monkeypatch.setattr(setup, "_discover_model_files", lambda *_args: ["weights.bin"])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_kwargs: str(root / "weights.bin"))
    state = {"progress": {}}
    model = {"id": "model", "repo": "fixture/model", "dest": "model", "files": ["weights.bin"]}
    setup._run_model_downloads([("model", model, tmp_path)], lock=threading.Lock(), state=state)
    assert state["progress"]["model"]["bytes_done"] == len(b"cached weights")
    assert state["progress"]["model"]["phase"] == "complete"
    assert (root / MTL_COMPLETION_MARKER).exists()
    assert progress_mod.download_context.current is None


def test_shared_client_keeps_concurrent_downloads_isolated(monkeypatch):
    from huggingface_hub import file_download

    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    barrier = threading.Barrier(2)
    samples = {"a": [], "b": []}
    payloads = {"a": b"a" * 32768, "b": b"b" * 98304}

    class ParallelStream(httpx.SyncByteStream):
        def __init__(self, name):
            self.name = name

        def __iter__(self):
            barrier.wait(5)
            yield payloads[self.name]
            assert samples[self.name][-1][0] == len(payloads[self.name])
            barrier.wait(5)
            yield payloads[self.name]

    def handle(request):
        name = request.url.path.strip("/")
        return httpx.Response(200, headers={"Content-Length": str(len(payloads[name]) * 2)},
                              stream=ParallelStream(name))

    def download(name):
        progress = _bind(samples[name])
        output = io.BytesIO()
        try:
            progress_mod.observe_hub_client()
            file_download.http_get(f"https://fixture.invalid/{name}", output,
                                   tqdm_class=progress_mod.HubByteProgress)
            return output.getvalue()
        finally:
            progress.finish()
            progress_mod.download_context.current = None

    with _client(monkeypatch, handle), ThreadPoolExecutor(max_workers=2) as workers:
        first, second = workers.submit(download, "a"), workers.submit(download, "b")
        assert first.result(timeout=10) == payloads["a"] * 2
        assert second.result(timeout=10) == payloads["b"] * 2
    assert samples["a"][-1][0] == len(payloads["a"]) * 2
    assert samples["b"][-1][0] == len(payloads["b"]) * 2


def test_success_without_a_saved_file_cannot_create_completion_marker(tmp_path, monkeypatch):
    import huggingface_hub

    from fox_reader.constants import MTL_COMPLETION_MARKER

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: object())
    monkeypatch.setattr(setup, "_discover_model_files", lambda *_args: ["weights.bin"])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_kwargs: str(tmp_path / "missing"))
    state = {"progress": {}}
    model = {"id": "model", "repo": "fixture/model", "dest": "model", "files": ["weights.bin"]}
    with pytest.raises(RuntimeError, match="weights.bin"):
        setup._run_model_downloads([("model", model, tmp_path)], lock=threading.Lock(), state=state)
    assert state["progress"]["model"]["done"] is False
    assert not (tmp_path / "model" / MTL_COMPLETION_MARKER).exists()
    assert progress_mod.download_context.current is None


def test_setup_status_exposes_real_hub_bytes_while_a_file_is_still_downloading(tmp_path, monkeypatch):
    from fox_reader import model_terms
    from fox_reader.constants import MTL_COMPLETION_MARKER

    monkeypatch.setattr(progress_mod, "_PUBLISH_INTERVAL", 0)
    key = "fixture-required-model"
    model = {"id": key, "repo": "fixture/model", "dest": "model", "files": ["weights.bin"]}
    marker = tmp_path / "model" / MTL_COMPLETION_MARKER
    monkeypatch.setattr(setup, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(setup, "_MTL_DIR", tmp_path / "mtl")
    monkeypatch.setattr(setup, "_download_state", {
        "active": True, "stage": "non_mtl", "current_model": None,
        "progress": {}, "error": None, "complete": False,
    })
    monkeypatch.setattr(setup, "_discover_model_files", lambda *_args: ["weights.bin"])
    monkeypatch.setattr(setup, "_required_models_ready", marker.exists)
    monkeypatch.setattr(setup, "_serialize_required_models", lambda *_args: [])
    monkeypatch.setattr(setup, "_get_mtl_models", lambda *_args: [])
    monkeypatch.setattr(setup, "_all_mtl_models_downloaded", lambda *_args: False)
    monkeypatch.setattr(setup, "_optional_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(model_terms, "state", lambda: {})

    chunk = b"fixture model bytes" * 8192
    size = len(chunk) * 2
    partial, release = threading.Event(), threading.Event()

    class PausedStream(httpx.SyncByteStream):
        def __iter__(self):
            yield chunk
            # Hub has consumed this raw chunk into its own buffer, but has
            # neither flushed its 10 MiB chunk nor completed the saved file.
            partial.set()
            assert release.wait(10), "test did not release the paused HTTP stream"
            yield chunk

    def handle(request):
        headers = {"Content-Length": str(size), "ETag": '"fixture-model"', "X-Repo-Commit": "b" * 40}
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        assert request.method == "GET"
        return httpx.Response(200, headers=headers, stream=PausedStream())

    app = FastAPI()
    app.include_router(setup.router)
    worker = threading.Thread(target=setup._download_models_task, args=(tmp_path, [(key, model)], "non_mtl"))
    with _client(monkeypatch, handle), TestClient(app) as client:
        worker.start()
        try:
            assert partial.wait(10), "real Hub download did not reach the partial-data barrier"
            status = client.get("/api/setup/status").json()
            assert status["active"] is True and status["current_model"] == key
            entry = status["progress"][key]
            assert entry["current_file"] == "weights.bin"
            assert 0 < entry["current_bytes"] == len(chunk) < entry["current_total"] == size < 10 * 1024 * 1024
            assert entry["bytes_done"] == len(chunk) and entry["bytes_total"] == 0
            assert entry["phase"] == "downloading"
            assert entry["files_done"] == 0 and entry["done"] is False and entry["downloaded"] is False
            assert not marker.exists()
        finally:
            release.set()
            worker.join(10)
        assert not worker.is_alive()
        final = client.get("/api/setup/status").json()
        assert final["active"] is False and final["error"] is None
        assert final["progress"][key]["phase"] == "complete"
        assert final["progress"][key]["done"] is True
        assert final["progress"][key]["bytes_done"] == size
        assert marker.exists()
        assert (tmp_path / "model" / "weights.bin").read_bytes() == chunk * 2
