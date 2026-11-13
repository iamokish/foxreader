"""The endpoint editor reports failures without requiring local translators.

Request preparation and result handling are real; external HTTP traffic is
replaced or blocked so these tests never contact a translation service.
"""

import copy
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader.routes.user_endpoints import router
from fox_reader.translate import custom_endpoint
from fox_reader.user_endpoints import UserEndpoint

AUTH_SECRET = "test-authorization-secret"
UPSTREAM_SECRET = "upstream-error-private-detail"
EXCEPTION_SECRET = "unexpected-exception-private-detail"
ROUTE_LOGGER = "fox_reader.routes.user_endpoints"


def make_endpoint(**overrides):
    return UserEndpoint.model_validate({
        "id": "custom-failure",
        "name": "Custom",
        "hostname": "example.invalid/translate",
        "languages": ["japanese", "korean"],
        "language_codes": {"japanese": "ja", "korean": "ko"},
        "headers": {"Authorization": f"Bearer {AUTH_SECRET}"},
        **overrides,
    })


@pytest.fixture
def endpoint_api(monkeypatch):
    store = SimpleNamespace(endpoint=make_endpoint())
    store.get = lambda selector: store.endpoint if selector == store.endpoint.id else None
    calls = SimpleNamespace(
        prepared=[], sent=[], outcome=httpx.Response(200, json={"text": "Translated."}),
    )
    real_prepare = custom_endpoint.prepare_request
    real_send = custom_endpoint._send

    def prepare(*args, **kwargs):
        request = real_prepare(*args, **kwargs)
        calls.prepared.append(copy.deepcopy(request))
        return request

    async def send(_endpoint, request):
        calls.sent.append(copy.deepcopy(request))
        if isinstance(calls.outcome, Exception):
            raise calls.outcome
        return calls.outcome

    monkeypatch.setattr(custom_endpoint, "prepare_request", prepare)
    monkeypatch.setattr(custom_endpoint, "_send", send)
    monkeypatch.setattr(custom_endpoint, "_retry_delay", lambda *_args: 0)
    app = FastAPI()
    app.state.user_endpoints = store
    app.include_router(router)
    with TestClient(app) as client:
        yield SimpleNamespace(app=app, client=client, store=store, calls=calls, real_send=real_send)


@pytest.mark.parametrize("service_state", ["missing", "none"])
def test_saved_endpoint_test_works_without_translation_service(endpoint_api, service_state):
    if service_state == "none":
        endpoint_api.app.state.translate = None

    response = endpoint_api.client.post("/api/user_endpoints/custom-failure/test", json={
        "text": "New text", "language": "korean",
    })

    assert response.status_code == 200
    result = response.json()
    assert result["status"] is True and result["code"] == 200
    assert result["translated"] == "Translated."
    assert result["sent"] == "New text" and result["language"] == "korean"
    assert len(endpoint_api.calls.prepared) == len(endpoint_api.calls.sent) == 1
    assert result["request"] == endpoint_api.calls.prepared[0] == endpoint_api.calls.sent[0]
    assert json.loads(result["request"]["body"]) == {
        "text": "New text", "src_lang": "ko", "target_lang": "en",
    }


@pytest.mark.parametrize("store_state", ["missing", "none"])
def test_missing_endpoint_store_returns_structured_unavailable_response(endpoint_api, store_state):
    if store_state == "none":
        endpoint_api.app.state.user_endpoints = None
    else:
        del endpoint_api.app.state.user_endpoints

    response = endpoint_api.client.post("/api/user_endpoints/custom-failure/test", json={})

    assert response.status_code == 503
    result = response.json()
    assert result["status"] is False and result["code"] == 503
    assert result["error"] == result["message"] and result["message"]
    assert result["request"] == {}
    assert endpoint_api.calls.prepared == endpoint_api.calls.sent == []


def test_unknown_endpoint_returns_structured_not_found_response(endpoint_api):
    response = endpoint_api.client.post("/api/user_endpoints/missing/test", json={})

    assert response.status_code == 404
    result = response.json()
    assert result["status"] is False and result["code"] == 404
    assert result["error"] == result["message"] and result["message"]
    assert result["request"] == {}
    assert endpoint_api.calls.prepared == endpoint_api.calls.sent == []


@pytest.mark.parametrize(("outcome", "code", "message_part", "attempts"), [
    pytest.param(httpx.ConnectError("Connection failed"), 503, "Could not reach", 3, id="connection"),
    pytest.param(httpx.ConnectError(""), 503, "Check the hostname", 3, id="connection-without-detail"),
    pytest.param(httpx.ReadTimeout("Read timed out"), 504, "Timed out", 3, id="timeout"),
    pytest.param(httpx.UnsupportedProtocol(UPSTREAM_SECRET), 400, "URL is invalid", 1, id="unsupported-protocol"),
    pytest.param(httpx.LocalProtocolError(UPSTREAM_SECRET), 400, "request is invalid", 1, id="invalid-http-fields"),
    pytest.param(httpx.Response(403, json={"error": UPSTREAM_SECRET}), 403, UPSTREAM_SECRET, 1, id="http-error"),
    pytest.param(httpx.Response(503, json={"error": UPSTREAM_SECRET}), 503, UPSTREAM_SECRET, 3, id="http-retries"),
    pytest.param(
        httpx.Response(200, text=f"<html>{UPSTREAM_SECRET}</html>"), 502, "invalid JSON", 1, id="invalid-json",
    ),
    pytest.param(httpx.Response(200, json={"other": UPSTREAM_SECRET}), 502, "Translated Text", 1, id="missing-text"),
])
def test_runtime_failures_preserve_details_and_actual_request(
    endpoint_api, caplog, outcome, code, message_part, attempts,
):
    endpoint_api.store.endpoint = make_endpoint(retries=2)
    endpoint_api.calls.outcome = outcome
    caplog.set_level(logging.WARNING, logger=ROUTE_LOGGER)

    response = endpoint_api.client.post("/api/user_endpoints/custom-failure/test", json={
        "text": "New text", "language": "japanese",
    })

    assert response.status_code == 200
    result = response.json()
    assert result["status"] is False and result["code"] == code
    assert message_part in result["message"]
    assert result["sent"] == "New text" and result["language"] == "japanese"
    assert result["translated"] == "" and result["alternatives"] == []
    assert len(endpoint_api.calls.prepared) == 1
    assert len(endpoint_api.calls.sent) == attempts
    assert all(request == result["request"] for request in endpoint_api.calls.sent)
    assert result["request"] == endpoint_api.calls.prepared[0]
    records = [record for record in caplog.records if record.name == ROUTE_LOGGER]
    assert len(records) == 1 and records[0].levelno == logging.WARNING
    assert "custom-failure" in records[0].getMessage() and str(code) in records[0].getMessage()
    assert AUTH_SECRET not in caplog.text and UPSTREAM_SECRET not in caplog.text
    assert records[0].exc_info is None


def test_unexpected_exception_returns_generic_error_and_logs_no_secret_or_traceback(endpoint_api, caplog):
    endpoint_api.calls.outcome = RuntimeError(EXCEPTION_SECRET)
    caplog.set_level(logging.ERROR, logger=ROUTE_LOGGER)

    response = endpoint_api.client.post("/api/user_endpoints/custom-failure/test", json={"text": "New text"})

    assert response.status_code == 500
    result = response.json()
    assert result["status"] is False and result["code"] == 500
    assert result["error"] == result["message"] and result["message"]
    assert EXCEPTION_SECRET not in response.text
    assert len(endpoint_api.calls.prepared) == len(endpoint_api.calls.sent) == 1
    assert result["request"] == endpoint_api.calls.sent[0]
    records = [record for record in caplog.records if record.name == ROUTE_LOGGER]
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    assert "custom-failure" in records[0].getMessage() and "RuntimeError" in records[0].getMessage()
    assert AUTH_SECRET not in caplog.text and EXCEPTION_SECRET not in caplog.text
    assert records[0].exc_info is None and records[0].stack_info is None
    assert "Traceback" not in caplog.text


@pytest.mark.parametrize("overrides", [
    pytest.param({"hostname": "[invalid]/translate"}, id="invalid-url"),
    pytest.param({"headers": {"X-Name": "\u3053\u3093\u306b\u3061\u306f"}}, id="non-ascii-header"),
])
def test_http_configuration_errors_are_reported_without_retry_or_network(endpoint_api, monkeypatch, overrides):
    endpoint_api.store.endpoint = make_endpoint(retries=2, **overrides)
    transport_calls = []
    clients = []
    real_client = httpx.AsyncClient

    def handle(request):
        transport_calls.append(request)
        return httpx.Response(200, json={"text": "Should not be sent"})

    def client(**kwargs):
        instance = real_client(**kwargs, transport=httpx.MockTransport(handle))
        clients.append(instance)
        return instance

    async def send(endpoint, request):
        endpoint_api.calls.sent.append(copy.deepcopy(request))
        return await endpoint_api.real_send(endpoint, request)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(custom_endpoint, "_send", send)

    response = endpoint_api.client.post("/api/user_endpoints/custom-failure/test", json={"text": "New text"})

    assert response.status_code == 200
    result = response.json()
    assert result["status"] is False and result["code"] == 400
    assert "invalid" in result["message"].lower() and "Check" in result["message"]
    assert len(endpoint_api.calls.prepared) == len(endpoint_api.calls.sent) == len(clients) == 1
    assert result["request"] == endpoint_api.calls.sent[0]
    assert transport_calls == []
