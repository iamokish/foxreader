"""System routes — session identity, readiness, and the shutdown endpoint.

The shutdown endpoint is the launcher's only clean way to stop the backend, and
it is reachable by anything that can open a socket to the loopback port -- which
on a desktop includes a web page in the user's own browser, since a cross-origin
POST is sent even though its reply cannot be read. Hence the two gates covered
here.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader.routes.system import SHUTDOWN_TOKEN_ENV, SHUTDOWN_TOKEN_HEADER, router


def _client(host: str = "127.0.0.1") -> tuple[TestClient, str]:
    app = FastAPI()
    app.include_router(router)
    app.state.session_id = "test-session-id"
    return TestClient(app, client=(host, 51234)), "test-session-id"


class TestSystemRoutes:
    def test_session_endpoint(self):
        client, expected = _client()
        res = client.get("/api/session")
        assert res.status_code == 200
        assert res.json() == {"session_id": expected}

    def test_health_endpoint(self):
        client, expected = _client()
        res = client.get("/api/health")
        assert res.status_code == 200
        # The marker is what lets the launcher tell Fox Reader from whatever else
        # might have claimed the port before it started.
        assert res.json() == {"app": "fox-reader", "session_id": expected}


class TestShutdown:
    @pytest.fixture(autouse=True)
    def _no_real_shutdown(self, monkeypatch):
        """Record the stop request instead of stopping the test process."""
        calls: list[float] = []
        monkeypatch.setattr(
            "fox_reader.runtime.stop",
            lambda delay=0.25: calls.append(delay) or True,
        )
        return calls

    def test_loopback_accepted(self):
        client, _ = _client()
        res = client.post("/api/shutdown")
        assert res.status_code == 200
        assert res.json() == {"status": "closing", "graceful": True}

    def test_non_loopback_refused(self):
        client, _ = _client(host="203.0.113.7")
        res = client.post("/api/shutdown")
        assert res.status_code == 403

    def test_token_required_when_set(self, monkeypatch):
        monkeypatch.setenv(SHUTDOWN_TOKEN_ENV, "s3cret")
        client, _ = _client()

        assert client.post("/api/shutdown").status_code == 403
        assert client.post("/api/shutdown", headers={SHUTDOWN_TOKEN_HEADER: "wrong"}).status_code == 403
        assert client.post("/api/shutdown", headers={SHUTDOWN_TOKEN_HEADER: "s3cret"}).status_code == 200

    def test_no_token_configured_is_open_to_loopback(self, monkeypatch):
        monkeypatch.delenv(SHUTDOWN_TOKEN_ENV, raising=False)
        client, _ = _client()
        assert client.post("/api/shutdown").status_code == 200

    def test_get_is_not_a_shutdown(self):
        """A link or a prefetch must not be able to close the app."""
        client, _ = _client()
        assert client.get("/api/shutdown").status_code == 405
