"""Portable YAML is bounded, strict, secret-free and never saves a draft."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
import yaml
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from fox_reader.endpoint_transfer import MAX_REQUEST_BYTES, MAX_TRANSFER_BYTES, MAX_YAML_DEPTH, MAX_YAML_NODES
from fox_reader.routes.user_endpoints import import_user_endpoint, router
from fox_reader.user_endpoints import MAX_ENDPOINTS_PER_LANGUAGE, UserEndpoint, UserEndpointStore


def settings(**overrides):
    return {
        "name": "Share", "hostname": "example.invalid/translate", "languages": ["japanese"],
        **overrides,
    }


def document(value=None):
    return yaml.safe_dump({"version": 1, "endpoint": settings() if value is None else value}, sort_keys=False)


def import_yaml(api, value=None):
    return api.client.post("/api/user_endpoints/import", json={"yaml": document(value)})


@pytest.fixture
def transfer_api(tmp_path):
    store = UserEndpointStore(tmp_path)
    key = Fernet.generate_key().decode()
    store.create(settings(id="saved", name="Saved", encryption={"enabled": True, "key": key}))
    store.set_active("japanese", "saved")
    app = FastAPI()
    app.state.user_endpoints = store
    app.include_router(router)
    with TestClient(app) as client:
        yield SimpleNamespace(app=app, client=client, store=store, key=key)


def assert_error(response, phrase=None):
    assert response.status_code == 400, response.text
    assert response.headers["cache-control"] == "no-store"
    data = response.json()
    assert isinstance(data["error"], str) and data["error"]
    assert data["suggestions"] and all(isinstance(item, str) for item in data["suggestions"])
    if phrase:
        assert phrase.lower() in data["error"].lower(), response.text
    return data


def test_export_import_do_not_save_activate_contact_api_or_borrow_keys(transfer_api, monkeypatch):
    from fox_reader.translate import custom_endpoint

    api = transfer_api
    original = api.store.as_dict()
    original_bytes = api.store.path.read_bytes()
    original_mtime = api.store.path.stat().st_mtime_ns

    def forbidden(*args, **kwargs):
        pytest.fail("YAML transfer must have no save, activation, key borrowing or HTTP side effects")

    for name in ("save", "create", "update", "set_active", "with_stored_secret"):
        monkeypatch.setattr(api.store, name, forbidden)
    for name in ("prepare_request", "_send", "translate_by_custom_endpoint"):
        monkeypatch.setattr(custom_endpoint, name, forbidden)
    exported = api.client.get("/api/user_endpoints/saved/export")
    assert exported.status_code == 200
    assert exported.headers["cache-control"] == "no-store"
    assert set(exported.json()) == {"yaml"}
    assert api.key not in exported.text
    portable = yaml.safe_load(exported.json()["yaml"])
    assert set(portable) == {"version", "endpoint"} and portable["version"] == 1
    assert "id" not in portable["endpoint"] and "active_by_language" not in portable
    assert "key" not in portable["endpoint"]["encryption"]
    assert "has_key" not in portable["endpoint"]["encryption"]
    assert portable["endpoint"]["request_schema"]["children"]["text"] == {"type": "string", "source": "text"}

    imported = api.client.post("/api/user_endpoints/import", json=exported.json())
    assert imported.status_code == 200
    assert imported.headers["cache-control"] == "no-store"
    data = imported.json()
    assert set(data) == {"endpoint", "requires_key", "warnings"}
    assert data["requires_key"] is True and data["warnings"]
    assert "id" not in data["endpoint"]
    assert "key" not in data["endpoint"]["encryption"]
    assert "has_key" not in data["endpoint"]["encryption"]
    assert api.key not in imported.text and "storedKeyPresent" not in imported.text
    assert api.store.as_dict() == original
    assert api.store.path.read_bytes() == original_bytes
    assert api.store.path.stat().st_mtime_ns == original_mtime
    assert api.store.get("saved").encryption.key == api.key


@pytest.mark.parametrize("body_format", ["json", "form", "query", "text"])
def test_all_transport_settings_and_scalar_schemas_round_trip(transfer_api, body_format):
    api = transfer_api
    value = settings(
        id="roundtrip", host_scheme="http", port=8443, method="PATCH", http=2, doseq=body_format in {"form", "query"},
        timeout=42.5, body_format=body_format, response_format="text", query={"q": "日本語", "blank": ""},
        languages=["japanese", "korean"], language_codes={"japanese": "ja", "korean": "ko"}, target_language="fr",
        text_join="\n", retries=3, max_text_length=9000, headers={"X-Value": "literal <SOURCE_LANG>"},
        response_schema={"type": "string", "source": "text"},
    )
    if body_format == "text":
        value["request_schema"] = {"type": "string", "source": "text"}
    original = api.store.create(value)
    exported = api.client.get("/api/user_endpoints/roundtrip/export")
    assert exported.status_code == 200
    imported = api.client.post("/api/user_endpoints/import", json=exported.json())
    assert imported.status_code == 200
    assert imported.json()["endpoint"] == original.model_dump(
        mode="json", exclude={"id": True, "encryption": {"key"}},
    )
    assert imported.json()["requires_key"] is False


def test_nested_schemas_uuid_repeat_and_static_values_round_trip(transfer_api):
    value = settings(
        id="nested", text_join=" / ",
        request_schema={"type": "list", "items": [
            {"type": "string", "source": "text"},
            {"type": "dict", "children": {
                "nonce": {"type": "uuid", "value": "uuid4"},
                "empty": {"type": "string", "value": ""},
                "nothing": {"type": "null"},
                "false": {"type": "bool", "value": False},
                "zero": {"type": "int", "value": 0},
            }},
        ]},
        response_schema={"type": "list", "each": {"type": "dict", "children": {
            "answer": {"type": "string", "source": "text"},
            "alternatives": {"type": "list", "source": "alt"},
        }}},
    )
    original = transfer_api.store.create(value)
    exported = transfer_api.client.get("/api/user_endpoints/nested/export")
    imported = transfer_api.client.post("/api/user_endpoints/import", json=exported.json())
    assert imported.status_code == 200
    assert imported.json()["endpoint"] == original.model_dump(
        mode="json", exclude={"id": True, "encryption": {"key"}},
    )


@pytest.mark.parametrize("encoding", ["fernet", "passphrase"])
@pytest.mark.parametrize("directions", [(True, False), (False, True), (True, True)])
def test_encrypted_draft_retains_settings_but_save_requires_own_key(transfer_api, encoding, directions):
    api = transfer_api
    encryption = {
        "enabled": True, "encrypt_request": directions[0], "decrypt_response": directions[1],
        "key_encoding": encoding, "max_age": 180,
    }
    imported = import_yaml(api, settings(name="New", encryption=encryption))
    assert imported.status_code == 200
    result = imported.json()
    assert result["requires_key"] is True and result["warnings"]
    assert result["endpoint"]["encryption"] == encryption
    draft = {**result["endpoint"], "id": "new"}
    failed_save = api.client.post("/api/user_endpoints", json=draft)
    assert failed_save.status_code == 400 and "key is required" in failed_save.json()["error"]
    assert api.store.get("new") is None
    own_key = Fernet.generate_key().decode() if encoding == "fernet" else "a private imported endpoint phrase"
    draft["encryption"]["key"] = own_key
    assert api.client.post("/api/user_endpoints", json=draft).status_code == 201
    assert api.store.get("new").encryption.key == own_key
    assert api.store.get("saved").encryption.key == api.key
    assert api.store.as_dict()["active_by_language"] == {"japanese": ["saved"]}


@pytest.mark.parametrize("encryption", [
    {"enabled": True, "encrypt_request": False, "decrypt_response": False},
    {"enabled": True, "key_encoding": "unsupported"},
    {"enabled": True, "max_age": 29},
    {"enabled": True, "max_age": 86401},
])
def test_missing_key_exception_does_not_skip_other_encryption_rules(transfer_api, encryption):
    assert_error(import_yaml(transfer_api, settings(encryption=encryption)))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("secret", ["", "top-secret-private-key", None, {"nested": "top-secret-private-key"}])
def test_import_rejects_key_even_when_disabled_or_empty_without_echoing_it(transfer_api, enabled, secret, caplog):
    response = import_yaml(transfer_api, settings(encryption={"enabled": enabled, "key": secret}))
    data = assert_error(response, "keys cannot be imported")
    assert "encryption.key" in data["suggestions"][0]
    assert "top-secret-private-key" not in response.text + caplog.text
    assert "line " in data["error"] and "column " in data["error"]


@pytest.mark.parametrize("extra", [
    {"id": "saved"}, {"active": True}, {"active_by_language": {"japanese": ["saved"]}}, {"typo": 1},
    {"encryption": {"has_key": True}}, {"encryption": {"typo": 1}}, {"storedKeyPresent": True},
    {"request_schema": {"type": "string", "source": "text", "typo": 1}},
    {"request_schema": {"type": "dict", "children": {"text": {"type": "string", "source": "text", "typo": 1}}}},
    {"request_schema": {"type": "list", "items": [{"type": "string", "source": "text", "typo": 1}]}},
    {"response_schema": {"type": "list", "each": {"type": "string", "source": "text", "typo": 1}}},
])
def test_import_rejects_unknown_fields_ids_and_activation_at_all_config_levels(transfer_api, extra):
    data = assert_error(import_yaml(transfer_api, settings(**extra)), "non-portable")
    assert "line " in data["error"] and "column " in data["error"]


@pytest.mark.parametrize("value", [
    {"version": 2, "endpoint": settings()}, {"version": True, "endpoint": settings()},
    {"version": "1", "endpoint": settings()}, {"endpoint": settings()},
    {"version": 1, "endpoint": [settings()]}, {"version": 1, "endpoints": [settings()]},
    {"version": 1, "endpoint": settings(), "active_by_language": {}},
])
def test_import_requires_one_versioned_endpoint(transfer_api, value):
    assert_error(transfer_api.client.post("/api/user_endpoints/import", json={"yaml": yaml.safe_dump(value)}))


@pytest.mark.parametrize("value, phrase", [
    ("", "mapping"), ("private-scalar-secret", "mapping"), ("[]", "mapping"),
    ("version: 1\nendpoint: {name: 'secret-unclosed", "syntax"),
    ("version: 1\nversion: 1\nendpoint: {}", "duplicate"),
    ("version: 1\nendpoint: {name: A, name: B}", "duplicate"),
    ("version: 1\nendpoint: &shared {name: A}\ncopy: *shared", "aliases"),
    ("version: 1\nendpoint: &recursive [*recursive]", "aliases"),
    ("version: 1\nendpoint: !private-secret {}", "tag"),
    ("version: 1\nendpoint: !!python/object/apply:os.system [private-secret]", "tag"),
    ("version: 1\nendpoint: {}\n---\nversion: 1\nendpoint: {}", "one YAML document"),
    ("version: 1\nendpoint: {true: secret}", "keys must be strings"),
    ("version: 1\nendpoint: {<<: {name: Secret}}", "tag"),
    ("version: 1\nendpoint: {timeout: .inf}", "finite"),
    ("version: 1\nendpoint: {timeout: .nan}", "finite"),
    ("version: 1\nendpoint: {name: 2026-01-01}", "tag"),
    ("version: 1\nendpoint: {doseq: !!bool private-secret}", "Invalid YAML"),
    ("version: 1\nendpoint: !!map [private-secret]", "Invalid YAML"),
    ("version: 1\nendpoint: {timeout: !!float private-secret}", "Invalid YAML"),
    ("version: 1\nendpoint: {port: !!int private-secret}", "Invalid YAML"),
])
def test_yaml_rejections_are_actionable_and_never_echo_input(transfer_api, value, phrase, caplog):
    response = transfer_api.client.post("/api/user_endpoints/import", json={"yaml": value})
    assert_error(response, phrase)
    for secret in ("private-scalar-secret", "secret-unclosed", "private-secret"):
        assert secret not in response.text + caplog.text


@pytest.mark.parametrize("value", [
    "version: 1\nendpoint: {doseq: !!bool private-secret}",
    "version: 1\nendpoint: {name: private-secret\x00}",
    "version: 1\nendpoint: {name: 'private-secret}",
])
def test_scalar_and_syntax_errors_report_position_without_source(transfer_api, value):
    response = transfer_api.client.post("/api/user_endpoints/import", json={"yaml": value})
    data = assert_error(response)
    assert "line 2, column " in data["error"]
    assert "private-secret" not in response.text


@pytest.mark.parametrize("value, phrase", [
    ("version: 1\nendpoint: " + "[" * MAX_YAML_DEPTH + "0" + "]" * MAX_YAML_DEPTH, "deeply"),
    ("version: 1\nendpoint: [" + ",".join("0" for _ in range(MAX_YAML_NODES)) + "]", "nodes"),
    ("#" + "x" * MAX_TRANSFER_BYTES, "128 KiB"),
    ("#" + "界" * (MAX_TRANSFER_BYTES // 3 + 1), "128 KiB"),
], ids=["depth", "nodes", "ascii-bytes", "unicode-bytes"])
def test_yaml_limits_are_enforced(transfer_api, value, phrase):
    assert_error(transfer_api.client.post("/api/user_endpoints/import", json={"yaml": value}), phrase)


def test_exact_yaml_byte_limit_is_accepted(transfer_api):
    value = document()
    value += "#" + "x" * (MAX_TRANSFER_BYTES - len(value.encode("utf-8")) - 1)
    response = transfer_api.client.post("/api/user_endpoints/import", json={"yaml": value})
    assert response.status_code == 200


@pytest.mark.parametrize("body", [None, [], {}, {"yaml": None}, {"yaml": 1}, {"yaml": "", "extra": True}])
def test_import_json_contract(transfer_api, body):
    assert_error(transfer_api.client.post("/api/user_endpoints/import", content=json.dumps(body)))


@pytest.mark.parametrize("body", [b"{bad-json-private-secret", b'"\xff"', b"[" * 2000],
                         ids=["syntax", "encoding", "nesting"])
def test_malformed_json_is_a_safe_400(transfer_api, body):
    response = transfer_api.client.post("/api/user_endpoints/import", content=body)
    assert_error(response, "valid JSON")
    assert "private-secret" not in response.text


def test_request_limit_stops_reading_chunks_even_without_content_length(transfer_api):
    reads = []

    async def receive():
        reads.append(True)
        assert len(reads) <= 3, "oversized request was read past the bound"
        return {"type": "http.request", "body": b" " * (MAX_REQUEST_BYTES // 2), "more_body": True}

    request = Request({"type": "http", "headers": [], "app": transfer_api.app}, receive)
    response = asyncio.run(import_user_endpoint(request))
    assert response.status_code == 400
    assert "too large" in json.loads(response.body)["error"]
    assert response.headers["cache-control"] == "no-store"
    assert len(reads) == 3


@pytest.mark.parametrize("extra, phrase", [
    ({"name": "toolong"}, "6"), ({"port": 65536}, "65535"), ({"timeout": 0}, "greater"),
    ({"method": "TRACE"}, "method"), ({"languages": []}, "language"),
    ({"max_text_length": 99}, "100"), ({"retries": 6}, "5"),
    ({"request_schema": {"type": "string", "value": "literal"}}, "exactly one Text"),
    ({"response_schema": {"type": "string", "value": "literal"}}, "Translated Text"),
    ({"request_schema": {"type": "int", "source": "text"}}, "string node"),
    ({"body_format": "form", "request_schema": {"type": "string", "source": "text"}}, "dict at the root"),
    ({"request_schema": {"type": "list", "each": {"type": "string", "source": "text"}}}, "Repeat templates"),
])
def test_import_uses_existing_endpoint_and_schema_rules(transfer_api, extra, phrase):
    data = assert_error(import_yaml(transfer_api, settings(**extra)), phrase)
    assert "line " in data["error"] and "column " in data["error"]


def test_validation_errors_do_not_echo_arbitrary_schema_names(transfer_api):
    response = import_yaml(transfer_api, settings(body_format="form", request_schema={
        "type": "dict", "children": {"private-secret-child": {
            "type": "dict", "children": {"text": {"type": "string", "source": "text"}},
        }},
    }))
    assert_error(response, "cannot nest an object")
    assert "private-secret-child" not in response.text


@pytest.mark.parametrize("extra", [
    {"timeout": "inf"},
    {"request_schema": {"type": "list", "items": [
        {"type": "string", "source": "text"}, {"type": "float", "value": "nan"},
    ]}},
])
def test_non_finite_strings_cannot_become_null_in_drafts_or_exports(transfer_api, extra):
    assert_error(import_yaml(transfer_api, settings(**extra)), "finite")
    transfer_api.store.create(settings(id="nonfinite", **extra))
    assert_error(transfer_api.client.get("/api/user_endpoints/nonfinite/export"), "finite")


def test_schema_numeric_overflow_is_an_actionable_400(transfer_api):
    response = import_yaml(transfer_api, settings(request_schema={"type": "list", "items": [
        {"type": "string", "source": "text"}, {"type": "float", "value": 10 ** 400},
    ]}))
    assert_error(response, "Invalid endpoint value")


def test_full_language_is_a_warning_until_save(transfer_api):
    api = transfer_api
    for index in range(MAX_ENDPOINTS_PER_LANGUAGE - 1):
        api.store.create(settings(id=f"full-{index}", name=f"Full{index}"))
    imported = import_yaml(api)
    assert imported.status_code == 200
    assert any("limit" in warning for warning in imported.json()["warnings"])
    assert len(api.store.all()) == MAX_ENDPOINTS_PER_LANGUAGE
    response = api.client.post("/api/user_endpoints", json={**imported.json()["endpoint"], "id": "one-more"})
    assert response.status_code == 400
    assert len(api.store.all()) == MAX_ENDPOINTS_PER_LANGUAGE


@pytest.mark.parametrize("store_state", ["missing", "none"])
def test_import_needs_no_store_and_export_unavailability_is_uncached(transfer_api, store_state):
    api = transfer_api
    if store_state == "missing":
        del api.app.state.user_endpoints
    else:
        api.app.state.user_endpoints = None
    assert import_yaml(api).status_code == 200
    response = api.client.get("/api/user_endpoints/saved/export")
    assert response.status_code == 503 and response.headers["cache-control"] == "no-store"


def test_export_missing_endpoint_is_uncached(transfer_api):
    response = transfer_api.client.get("/api/user_endpoints/missing/export")
    assert response.status_code == 404 and response.headers["cache-control"] == "no-store"
    assert transfer_api.key not in response.text


def test_oversized_export_returns_actionable_error(transfer_api):
    transfer_api.store.create(settings(id="large", headers={"X-Large": "x" * MAX_TRANSFER_BYTES}))
    assert_error(transfer_api.client.get("/api/user_endpoints/large/export"), "128 KiB")


@pytest.mark.parametrize("hostname", [
    "http://example.invalid", "https://example.invalid/path", " HTTPS://example.invalid ",
    "ftp://example.invalid", "custom+transport://example.invalid", "//example.invalid", "mailto:somewhere",
])
def test_schemes_are_rejected_on_new_model_save_preview_and_import(transfer_api, hostname):
    value = settings(id="new", hostname=hostname)
    with pytest.raises(ValidationError, match="host_scheme"):
        UserEndpoint.model_validate(value)
    for method, route in (("post", ""), ("put", "/saved"), ("post", "/preview")):
        response = getattr(transfer_api.client, method)("/api/user_endpoints" + route, json=value)
        assert response.status_code == 400 and "host_scheme" in response.json()["error"]
    value.pop("id")
    data = assert_error(import_yaml(transfer_api, value), "scheme")
    assert "host_scheme" in data["suggestions"][0]


@pytest.mark.parametrize("hostname, port, suffix", [
    ("example.invalid/path", None, "example.invalid/path"),
    ("localhost", 8080, "localhost:8080"),
    ("example.invalid:9000/path", 8080, "example.invalid:9000/path"),
    ("[::1]/path", 8080, "[::1]:8080/path"),
    ("[::1]:9000/path", None, "[::1]:9000/path"),
    ("example.invalid/http://embedded", None, "example.invalid/http://embedded"),
])
@pytest.mark.parametrize("scheme", ["http", "https"])
def test_selected_scheme_constructs_url(hostname, port, suffix, scheme):
    endpoint = UserEndpoint.model_validate(settings(id="url", hostname=hostname, port=port, host_scheme=scheme))
    assert endpoint.build_url() == f"{scheme}://{suffix}"


@pytest.mark.parametrize("prefix, selected", [("http", "https"), ("HTTPS", "http")])
def test_load_migrates_stored_full_urls_preserving_effective_target_id_key_and_activation(tmp_path, prefix, selected):
    key = Fernet.generate_key().decode()
    original = {
        "endpoints": [
            settings(id="legacy", hostname=f" {prefix}://example.invalid/path ", host_scheme=selected, port=8080,
                     encryption={"enabled": True, "key": key}),
            settings(id="plain", hostname="other.invalid", host_scheme=selected),
        ],
        "active_by_language": {"japanese": ["legacy", "plain"]},
    }
    path = tmp_path / "user_endpoints.yaml"
    path.write_text(yaml.safe_dump(original), encoding="utf-8")
    store = UserEndpointStore(tmp_path)
    migrated = store.get("legacy")
    assert migrated.hostname == "example.invalid/path"
    assert migrated.host_scheme == prefix.lower()
    assert migrated.build_url() == f"{prefix.lower()}://example.invalid:8080/path"
    assert migrated.encryption.key == key
    assert store.get("plain").host_scheme == selected
    assert store.as_dict()["active_by_language"] == original["active_by_language"]
    persisted = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert persisted["endpoints"][0]["hostname"] == migrated.hostname
    assert persisted["endpoints"][0]["encryption"]["key"] == key
    reloaded = UserEndpointStore(tmp_path)
    assert reloaded.as_dict() == store.as_dict()
    assert reloaded.get("legacy").encryption.key == key
    # The legacy input remains invalid everywhere except loading old storage.
    with pytest.raises(ValidationError, match="host_scheme"):
        store.create(copy.deepcopy(original["endpoints"][0]))
