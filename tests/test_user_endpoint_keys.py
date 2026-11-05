"""Saved keys can be revealed explicitly while routine responses omit them."""

from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader.routes.user_endpoints import router
from fox_reader.user_endpoints import UserEndpointStore


@pytest.fixture
def key_api(tmp_path):
    app = FastAPI()
    app.state.user_endpoints = UserEndpointStore(tmp_path)
    app.include_router(router)
    with TestClient(app) as client:
        yield SimpleNamespace(app=app, client=client, config_root=tmp_path)


def create_endpoint(api, key, encoding="fernet", endpoint_id="saved"):
    payload = {
        "id": endpoint_id, "name": "Saved", "hostname": "example.invalid",
        "languages": ["japanese"],
        "encryption": {"enabled": bool(key), "key": key, "key_encoding": encoding},
    }
    response = api.client.post("/api/user_endpoints", json=payload)
    assert response.status_code == 201
    assert "key" not in response.json()["encryption"]
    assert response.json()["encryption"]["has_key"] is bool(key)
    return response.json()


@pytest.mark.parametrize("encoding", ["fernet", "passphrase"])
def test_saved_key_is_revealed_after_reload_but_not_in_routine_responses(key_api, encoding):
    key = Fernet.generate_key().decode() if encoding == "fernet" else "test shared phrase"
    create_endpoint(key_api, key, encoding)
    key_api.app.state.user_endpoints = UserEndpointStore(key_api.config_root)

    listing = key_api.client.get("/api/user_endpoints")
    assert listing.status_code == 200
    assert "key" not in listing.json()["endpoints"][0]["encryption"]
    assert key not in listing.text

    config_path = key_api.app.state.user_endpoints.path
    original_bytes = config_path.read_bytes()
    original_mtime = config_path.stat().st_mtime_ns
    revealed = key_api.client.get("/api/user_endpoints/saved/key")
    assert revealed.status_code == 200
    assert revealed.json() == {"key": key, "key_encoding": encoding}
    assert revealed.headers["cache-control"] == "no-store"
    assert config_path.read_bytes() == original_bytes
    assert config_path.stat().st_mtime_ns == original_mtime


def test_reveal_returns_all_rotation_keys_in_saved_order(key_api):
    key = " ".join(Fernet.generate_key().decode() for _ in range(3))
    create_endpoint(key_api, key)
    response = key_api.client.get("/api/user_endpoints/saved/key")
    assert response.status_code == 200
    assert response.json()["key"] == key


def test_reveal_selects_only_the_requested_endpoint(key_api):
    first, second = (Fernet.generate_key().decode() for _ in range(2))
    create_endpoint(key_api, first, endpoint_id="first")
    create_endpoint(key_api, second, endpoint_id="second")
    assert key_api.client.get("/api/user_endpoints/first/key").json()["key"] == first
    assert key_api.client.get("/api/user_endpoints/second/key").json()["key"] == second


def test_viewing_then_saving_blank_preserves_a_concurrently_replaced_key(key_api):
    original, replacement = (Fernet.generate_key().decode() for _ in range(2))
    edit = create_endpoint(key_api, original)
    assert key_api.client.get("/api/user_endpoints/saved/key").json()["key"] == original

    rotated = {**edit, "encryption": {**edit["encryption"], "key": replacement}}
    changed = key_api.client.put("/api/user_endpoints/saved", json=rotated)
    assert changed.status_code == 200
    assert "key" not in changed.json()["encryption"]

    edit["name"] = "Edited"
    edit["encryption"]["key"] = ""
    response = key_api.client.put("/api/user_endpoints/saved", json=edit)
    assert response.status_code == 200
    assert "key" not in response.json()["encryption"]
    assert key_api.client.get("/api/user_endpoints/saved/key").json()["key"] == replacement


@pytest.mark.parametrize("initial_encoding", ["fernet", "passphrase"])
def test_unchanged_reveal_preserves_rotated_key_and_encoding_together(key_api, initial_encoding):
    initial = Fernet.generate_key().decode() if initial_encoding == "fernet" else "original test phrase"
    replacement_encoding = "passphrase" if initial_encoding == "fernet" else "fernet"
    replacement = (
        Fernet.generate_key().decode() if replacement_encoding == "fernet" else "replacement test phrase"
    )
    edit = create_endpoint(key_api, initial, initial_encoding)
    assert key_api.client.get("/api/user_endpoints/saved/key").json()["key"] == initial

    rotated = {
        **edit,
        "encryption": {**edit["encryption"], "key": replacement, "key_encoding": replacement_encoding},
    }
    assert key_api.client.put("/api/user_endpoints/saved", json=rotated).status_code == 200
    edit["encryption"]["key"] = ""
    assert key_api.client.put("/api/user_endpoints/saved", json=edit).status_code == 200
    assert key_api.client.get("/api/user_endpoints/saved/key").json() == {
        "key": replacement, "key_encoding": replacement_encoding,
    }


def test_explicit_replacement_can_change_key_encoding(key_api):
    edit = create_endpoint(key_api, Fernet.generate_key().decode())
    edit["encryption"].update(key="replacement test phrase", key_encoding="passphrase")
    assert key_api.client.put("/api/user_endpoints/saved", json=edit).status_code == 200
    assert key_api.client.get("/api/user_endpoints/saved/key").json() == {
        "key": "replacement test phrase", "key_encoding": "passphrase",
    }


@pytest.mark.parametrize("state", ["missing", "keyless", "disabled", "deleted"])
def test_missing_keys_return_uncached_errors_without_secrets(key_api, state):
    secret = Fernet.generate_key().decode()
    if state != "missing":
        saved = create_endpoint(key_api, "" if state == "keyless" else secret)
        if state == "disabled":
            saved["encryption"]["enabled"] = False
            response = key_api.client.put("/api/user_endpoints/saved", json=saved)
            assert response.status_code == 200
        elif state == "deleted":
            assert key_api.client.delete("/api/user_endpoints/saved").status_code == 200

    response = key_api.client.get("/api/user_endpoints/saved/key")
    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    assert "error" in response.json() and "key" not in response.json()
    assert secret not in response.text


def test_unconfigured_store_returns_an_uncached_error(key_api):
    key_api.app.state.user_endpoints = None
    response = key_api.client.get("/api/user_endpoints/saved/key")
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"error": "User endpoints are not configured"}
