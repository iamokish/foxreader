"""Endpoint names are globally unique on writes, while legacy names survive load."""

import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
import yaml
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader.routes.user_endpoints import router
from fox_reader.user_endpoints import UserEndpoint, UserEndpointStore


def definition(endpoint_id, name, language="japanese", **overrides):
    return {
        "id": endpoint_id, "name": name, "hostname": "example.invalid/translate",
        "languages": [language], **overrides,
    }


@pytest.fixture
def name_api(tmp_path):
    app = FastAPI()
    app.state.user_endpoints = UserEndpointStore(tmp_path)
    app.include_router(router)
    with TestClient(app) as client:
        yield SimpleNamespace(app=app, store=app.state.user_endpoints, client=client)


def create(api, via, payload):
    if via == "store":
        return api.store.create(payload)
    response = api.client.post("/api/user_endpoints", json=payload)
    assert response.status_code == 201, response.text
    return api.store.get(payload["id"])


def reject_write(api, via, operation, payload):
    original = copy.deepcopy(payload)
    if via == "store":
        with pytest.raises(ValueError, match="(?i)name") as caught:
            if operation == "create":
                api.store.create(payload)
            else:
                api.store.update(payload["id"], payload)
        message = str(caught.value)
    else:
        response = (
            api.client.post("/api/user_endpoints", json=payload) if operation == "create" else
            api.client.put(f"/api/user_endpoints/{payload['id']}", json=payload)
        )
        assert response.status_code == 400, response.text
        message = response.json()["error"]
        assert "name" in message.lower()
    assert payload == original
    return message


def snapshot(store):
    return (store.all(), store.as_dict(), store.path.read_bytes(), store.path.stat().st_mtime_ns)


@pytest.mark.parametrize("via", ["store", "api"])
@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize("existing, conflicting", [
    ("Reader", "Reader"), ("Reader", "reader"), (" Reader ", "\tREADER\n"),
    ("Reader", "\u00a0reader\u2003"), ("Maße", " MASSE "), ("Σ", "ς"),
])
def test_names_conflict_after_strip_and_casefold_across_languages(name_api, via, operation, existing, conflicting):
    api = name_api
    first = create(api, via, definition("first", existing, "japanese"))
    assert first.name == existing.strip()
    if operation == "update":
        create(api, via, definition("second", "Other", "korean"))
    before = snapshot(api.store)
    reject_write(api, via, operation, definition("second", conflicting, "korean"))
    assert snapshot(api.store) == before


@pytest.mark.parametrize("via", ["store", "api"])
def test_distinct_names_can_share_a_language_and_surrounding_space_is_trimmed(name_api, via):
    first = create(name_api, via, definition("first", " First "))
    second = create(name_api, via, definition("second", "Second"))
    assert [first.name, second.name] == ["First", "Second"]
    assert len(name_api.store.all()) == 2


@pytest.mark.parametrize("via", ["store", "api"])
@pytest.mark.parametrize("name", ["Reader", " reader ", "\tREADER\n"])
def test_self_edit_can_keep_or_recase_its_own_name(name_api, via, name):
    api = name_api
    original = create(api, via, definition("self", "Reader"))
    api.store.set_active("japanese", "self")
    payload = {**original.public_dict(), "name": name, "timeout": 31}
    if via == "store":
        api.store.update("self", payload)
    else:
        response = api.client.put("/api/user_endpoints/self", json=payload)
        assert response.status_code == 200, response.text
    updated = api.store.get("self")
    assert updated.name == name.strip() and updated.timeout == 31
    assert len(api.store.all()) == 1
    assert api.store.as_dict()["active_by_language"] == {"japanese": ["self"]}


def test_name_in_use_is_global_and_can_exclude_only_the_endpoint_being_edited(name_api):
    store = name_api.store
    store.create(definition("first", "Maße", "japanese"))
    store.create(definition("second", "Other", "korean"))
    before = snapshot(store)
    assert store.name_in_use(" MASSE ") is True
    assert store.name_in_use("masse", replacing=None) is True
    assert store.name_in_use("masse", replacing="first") is False
    assert store.name_in_use("masse", replacing="second") is True
    assert store.name_in_use("masse", replacing="missing") is True
    assert store.name_in_use("Unused") is False
    assert snapshot(store) == before


@pytest.mark.parametrize("via", ["store", "api"])
@pytest.mark.parametrize("operation", ["create", "update"])
def test_duplicate_failure_cannot_mutate_disk_activation_or_encryption_keys(name_api, via, operation):
    api = name_api
    first_key, second_key, replacement_key = (Fernet.generate_key().decode() for _ in range(3))
    for endpoint_id, name, language, key in (
        ("first", "Taken", "japanese", first_key), ("second", "Other", "korean", second_key),
    ):
        api.store.create(definition(endpoint_id, name, language, encryption={"enabled": True, "key": key}))
        api.store.set_active(language, endpoint_id)
    before = snapshot(api.store)
    payload = definition(
        "third" if operation == "create" else "second", " tAKEN ", "chinese", hostname="changed.invalid",
        encryption={"enabled": True, "key": replacement_key},
    )
    message = reject_write(api, via, operation, payload)
    assert snapshot(api.store) == before
    assert api.store.get("first").encryption.key == first_key
    assert api.store.get("second").encryption.key == second_key
    assert api.store.get("third") is None
    for key in (first_key, second_key, replacement_key):
        assert key not in message
    reloaded = UserEndpointStore(api.store.config_root)
    assert reloaded.all() == before[0]
    assert reloaded.as_dict() == before[1]


@pytest.mark.parametrize("via", ["store", "api"])
def test_deleting_endpoint_releases_name_without_reusing_activation(name_api, via):
    api = name_api
    create(api, via, definition("first", "Maße"))
    api.store.set_active("japanese", "first")
    if via == "store":
        api.store.delete("first")
    else:
        assert api.client.delete("/api/user_endpoints/first").status_code == 200
    assert api.store.name_in_use(" MASSE ") is False
    replacement = create(api, via, definition("replacement", "MASSE", "korean"))
    assert replacement.name == "MASSE"
    assert api.store.as_dict()["active_by_language"] == {}


@pytest.mark.parametrize("via", ["store", "api"])
def test_legacy_duplicates_keep_names_ids_keys_and_activation_but_edits_require_unique_name(name_api, via):
    api = name_api
    keys = [Fernet.generate_key().decode(), Fernet.generate_key().decode()]
    records = [
        UserEndpoint.model_validate(definition(endpoint_id, name, encryption={"enabled": True, "key": key}))
        for endpoint_id, name, key in zip(("legacy-a", "legacy-b"), ("Maße", "MASSE"), keys)
    ]
    persisted = {
        "endpoints": [endpoint.model_dump(mode="json") for endpoint in records],
        "active_by_language": {"japanese": ["legacy-a", "legacy-b"]},
    }
    api.store.path.write_text(yaml.safe_dump(persisted, sort_keys=False), encoding="utf-8")
    api.store.load()
    assert api.store.all() == records
    assert yaml.safe_load(api.store.path.read_text(encoding="utf-8")) == persisted
    api.store.load()
    assert api.store.all() == records
    assert api.store.as_dict()["active_by_language"] == persisted["active_by_language"]
    assert api.store.name_in_use("masse", replacing="legacy-a") is True
    assert api.store.name_in_use("masse", replacing="legacy-b") is True

    before = snapshot(api.store)
    for endpoint in records:
        reject_write(api, via, "update", {**endpoint.public_dict(), "timeout": 31})
        assert snapshot(api.store) == before
    renamed = {**records[0].public_dict(), "name": "Fresh", "timeout": 31}
    if via == "store":
        api.store.update("legacy-a", renamed)
        api.store.update("legacy-b", records[1].public_dict())
    else:
        assert api.client.put("/api/user_endpoints/legacy-a", json=renamed).status_code == 200
        assert api.client.put("/api/user_endpoints/legacy-b", json=records[1].public_dict()).status_code == 200
    assert [endpoint.name for endpoint in api.store.all()] == ["Fresh", "MASSE"]
    assert [endpoint.encryption.key for endpoint in api.store.all()] == keys
    assert api.store.as_dict()["active_by_language"] == persisted["active_by_language"]


def test_import_warns_about_global_name_conflict_without_saving_and_rename_allows_save(name_api):
    api = name_api
    key = Fernet.generate_key().decode()
    api.store.create(definition("saved", "Maße", encryption={"enabled": True, "key": key}))
    api.store.set_active("japanese", "saved")
    before = snapshot(api.store)
    draft = definition("unused", " MASSE ", "korean")
    draft.pop("id")
    response = api.client.post("/api/user_endpoints/import", json={
        "yaml": yaml.safe_dump({"version": 1, "endpoint": draft}),
    })
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    assert result["endpoint"]["name"] == "MASSE"
    assert result["requires_key"] is False
    assert any("name" in warning.lower() and "before saving" in warning.lower() for warning in result["warnings"])
    assert "id" not in result["endpoint"] and "key" not in result["endpoint"]["encryption"]
    assert key not in response.text
    assert snapshot(api.store) == before
    payload = {**result["endpoint"], "id": "new"}
    reject_write(api, "api", "create", payload)
    assert snapshot(api.store) == before
    payload["name"] = "Fresh"
    assert api.client.post("/api/user_endpoints", json=payload).status_code == 201
    assert api.store.get("new").name == "Fresh"
    assert api.store.get("saved").encryption.key == key
    assert api.store.as_dict()["active_by_language"] == {"japanese": ["saved"]}


def test_import_of_unused_name_has_no_name_conflict_warning(name_api):
    name_api.store.create(definition("saved", "Taken"))
    draft = definition("unused", "Fresh")
    draft.pop("id")
    response = name_api.client.post("/api/user_endpoints/import", json={
        "yaml": yaml.safe_dump({"version": 1, "endpoint": draft}),
    })
    assert response.status_code == 200
    assert not any("name" in warning.lower() for warning in response.json()["warnings"])
    assert len(name_api.store.all()) == 1


def test_concurrent_creates_claim_a_casefolded_name_atomically(name_api):
    store = name_api.store
    names = ["Maße", "MASSE", " masse ", "maße", "Masse", " Maße ", "mAsSe", " MASSE "]
    start = Barrier(len(names))

    def attempt(index):
        start.wait(timeout=10)
        try:
            endpoint = store.create(definition(f"racer-{index}", names[index]))
        except ValueError as exc:
            assert "name" in str(exc).lower()
            return None
        return endpoint

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        outcomes = list(pool.map(attempt, range(len(names))))
    winners = [endpoint for endpoint in outcomes if endpoint is not None]
    assert len(winners) == 1
    assert store.all() == winners
    assert store.as_dict()["active_by_language"] == {}
    persisted = yaml.safe_load(store.path.read_text(encoding="utf-8"))
    assert persisted["endpoints"] == [winners[0].model_dump(mode="json")]
    assert UserEndpointStore(store.config_root).all() == winners
