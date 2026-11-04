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


class TestFrontendDiagnostics:
    @pytest.fixture
    def client(self):
        from fox_reader.routes.diagnostics import router as diagnostics_router

        app = FastAPI()
        app.include_router(diagnostics_router)
        return TestClient(app, base_url="http://127.0.0.1:7954", client=("127.0.0.1", 51234))

    @staticmethod
    def headers():
        return {"Origin": "http://127.0.0.1:7954", "X-Fox-Reader-Client": "frontend"}

    def test_local_event_is_redacted_and_single_line(self, client, monkeypatch):
        messages = []
        monkeypatch.setattr("fox_reader.routes.diagnostics.logger.log", lambda *args: messages.append(args))
        response = client.post("/api/frontend-log", headers=self.headers(), json=[{
            "level": "error", "message": "api_key=private\nfailed https://host/path?token=hidden",
            "source": "/static/app.js:2:3",
        }])
        assert response.status_code == 204
        assert len(messages) == 1
        assert "private" not in messages[0][1] and "hidden" not in messages[0][1]
        assert "\\n" in messages[0][1] and "\n" not in messages[0][1]

    @pytest.mark.parametrize("headers", [
        {},
        {"Origin": "https://attacker.example", "X-Fox-Reader-Client": "frontend"},
        {"Origin": "null", "X-Fox-Reader-Client": "frontend"},
        {"Sec-Fetch-Site": "cross-site", "X-Fox-Reader-Client": "frontend"},
    ])
    def test_cross_origin_and_forms_rejected(self, client, headers):
        assert client.post("/api/frontend-log", headers=headers, json=[{
            "level": "error", "message": "bad",
        }]).status_code == 403

    def test_rebinding_host_rejected(self, client):
        headers = {"Host": "attacker.example", "Origin": "http://attacker.example",
                   "X-Fox-Reader-Client": "frontend"}
        assert client.post("/api/frontend-log", headers=headers, json=[]).status_code == 403

    def test_non_loopback_client_rejected(self):
        from fox_reader.routes.diagnostics import router as diagnostics_router

        app = FastAPI()
        app.include_router(diagnostics_router)
        remote = TestClient(app, base_url="http://127.0.0.1:7954", client=("203.0.113.7", 1234))
        assert remote.post("/api/frontend-log", headers=self.headers(), json=[]).status_code == 403

    @pytest.mark.parametrize("payload", [[], {}, [{"level": "debug", "message": "a"}],
                                         [{"level": "error", "message": {"token": "secret"}}],
                                         [{"level": "error", "message": "a", "body": "private"}]])
    def test_invalid_payloads_rejected(self, client, payload):
        assert client.post("/api/frontend-log", headers=self.headers(), json=payload).status_code == 400

    def test_bounded_body_and_events(self, client):
        headers = {**self.headers(), "Content-Type": "application/json"}
        assert client.post("/api/frontend-log", headers=headers, content=b" " * 25000).status_code == 413
        events = [{"level": "error", "message": "error"}] * 10
        for _ in range(6):
            assert client.post("/api/frontend-log", headers=headers, json=events).status_code == 204
        assert client.post("/api/frontend-log", headers=headers, json=events).status_code == 429


class TestFileDiagnostics:
    def test_rotating_files_and_missing_stdio(self, tmp_path, monkeypatch):
        import logging
        import sys

        from fox_reader import logging_setup

        loggers = [logging.getLogger(name) for name in ("", "fox_reader", "fox_reader.frontend",
                                                       "uvicorn", "uvicorn.error", "uvicorn.access")]
        saved = [(logger, logger.handlers[:], logger.level, logger.propagate) for logger in loggers]
        monkeypatch.setattr(logging_setup, "_configured", None)
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", None)
        try:
            assert logging_setup.configure_logging(tmp_path)
            count = len(logging.getLogger().handlers)
            assert logging_setup.configure_logging(tmp_path)
            assert len(logging.getLogger().handlers) == count
            sys.stderr.write("token=secret no console\n")
            logging.getLogger("uvicorn.error").info("server ready")
            logging.getLogger("fox_reader.frontend").error("browser failed")
            backend = (tmp_path / "backend.log").read_text(encoding="utf-8")
            frontend = (tmp_path / "frontend.log").read_text(encoding="utf-8")
            assert "no console" in backend and "secret" not in backend
            assert "server ready" in backend and "browser failed" not in backend
            assert "browser failed" in frontend
            handler = next(h for h in logging.getLogger().handlers
                           if getattr(h, "_fox_reader_handler", False))
            handler.maxBytes = 100
            for _ in range(10):
                logging.getLogger("fox_reader").warning("x" * 80)
            assert len(list(tmp_path.glob("backend.log*"))) <= 4
        finally:
            for logger, handlers, level, propagate in saved:
                for handler in logger.handlers:
                    if handler not in handlers:
                        handler.close()
                logger.handlers = handlers
                logger.setLevel(level)
                logger.propagate = propagate
