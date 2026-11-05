"""Exercise custom translation metadata through the real routes and service.

Only the external HTTP send is replaced; schema rendering, normalization,
encryption and the editor's test/preview paths run as they do in the app.
"""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader import endpoint_crypto
from fox_reader.routes.translate import router as translate_router
from fox_reader.routes.user_endpoints import router as editor_router
from fox_reader.services.translate_service import TranslateService
from fox_reader.translate import custom_endpoint
from fox_reader.user_endpoints import UserEndpoint, UserEndpointStore


def make_endpoint(**overrides):
    return UserEndpoint.model_validate({
        "id": "custom",
        "name": "Custom",
        "hostname": "example.invalid/translate",
        "languages": ["japanese"],
        "request_schema": {
            "type": "dict",
            "children": {
                "text": {"type": "string", "source": "text"},
                "context": {"type": "list", "source": "context"},
                "character_info": {"type": "list", "source": "character_info"},
                "context_character_links": {"type": "list", "source": "context_character_links"},
                "meta_id": {"type": "string", "source": "meta_id"},
            },
        },
        **overrides,
    })


@pytest.fixture
def custom_api(monkeypatch):
    store = SimpleNamespace(endpoint=make_endpoint())
    store.get = lambda selector: store.endpoint if selector == store.endpoint.id else None
    store.with_stored_secret = lambda _selector, draft: draft
    service = TranslateService.__new__(TranslateService)
    service._user_endpoints = store
    sent = []

    async def send(_endpoint, request):
        sent.append(copy.deepcopy(request))
        return httpx.Response(200, json={"text": "Translated."})

    monkeypatch.setattr(custom_endpoint, "_send", send)
    app = FastAPI()
    app.state.translate = service
    app.state.user_endpoints = store
    app.include_router(translate_router)
    app.include_router(editor_router)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, store=store, sent=sent)


def translation_input(**overrides):
    return {
        "text": "New line",
        "source_lang": "custom",
        "lang_group": "japanese",
        "context": [["Earlier line", "Earlier translation"]],
        "character_info": [
            {"meta_id": "a", "name_en": "Aiko", "gender": "female"},
            {"meta_id": "b", "name_en": "Ken", "gender": "male"},
        ],
        "context_character_links": ["a"],
        "meta_id": "b",
        **overrides,
    }


def translate(api, payload):
    response = api.client.post("/translate/custom", json=payload)
    assert response.status_code == 200
    assert response.json()["error"] is None
    assert response.json()["translated"] == "Translated."
    return json.loads(api.sent[-1]["body"])


def test_new_speaker_survives_route_service_and_external_request(custom_api):
    sent = translate(custom_api, translation_input())
    assert sent["meta_id"] == "b"
    assert sent["context_character_links"] == ["a"]
    assert [character["meta_id"] for character in sent["character_info"]] == ["a", "b"]


@pytest.mark.parametrize("context_fields", [
    {"context": [], "context_character_links": []},
    {"context": None, "context_character_links": None},
])
def test_current_speaker_does_not_require_history(custom_api, context_fields):
    sent = translate(custom_api, translation_input(**context_fields))
    assert sent["context"] == sent["context_character_links"] == []
    assert sent["meta_id"] == "b"
    assert len(sent["character_info"]) == 2


@pytest.mark.parametrize("speaker", [None, "", "deleted", ["b"], {"id": "b"}])
def test_unassigned_or_invalid_current_speaker_is_null(custom_api, speaker):
    sent = translate(custom_api, translation_input(meta_id=speaker))
    assert sent["meta_id"] is None
    assert sent["context_character_links"] == ["a"]


def test_missing_speaker_and_characters_do_not_invent_metadata(custom_api):
    sent = translate(custom_api, {"text": "New line", "source_lang": "custom"})
    assert sent == {
        "text": "New line", "context": [], "character_info": [],
        "context_character_links": [], "meta_id": None,
    }


def test_filtering_and_trimming_keep_links_aligned_and_clear_unknown_ids(custom_api):
    pairs = [[f"s{i}", f"e{i}"] for i in range(14)]
    pairs[7] = ["", "skip this pair"]
    links = ["a" if i % 2 else "b" for i in range(14)]
    links[10] = "deleted"
    sent = translate(custom_api, translation_input(context=pairs, context_character_links=links))
    kept = [i for i in range(14) if i != 7][-10:]
    assert sent["context"] == [pairs[i] for i in kept]
    assert sent["context_character_links"] == [None if i == 10 else links[i] for i in kept]
    assert sent["meta_id"] == "b"


def test_malformed_roster_clears_both_kinds_of_speaker_links(custom_api):
    sent = translate(custom_api, translation_input(character_info="invalid roster"))
    assert sent["character_info"] == []
    assert sent["context_character_links"] == [None]
    assert sent["meta_id"] is None


def test_nested_renamed_nodes_receive_current_and_historical_speakers(custom_api):
    custom_api.store.endpoint = make_endpoint(request_schema={
        "type": "dict", "children": {
            "characters": {"type": "list", "source": "character_info"},
            "turns": {"type": "list", "items": [{"type": "dict", "children": {
                "content": {"type": "string", "source": "text"},
                "speaker": {"type": "string", "source": "meta_id"},
                "history": {"type": "list", "source": "context"},
                "speakers": {"type": "list", "source": "context_character_links"},
            }}]},
        },
    })
    sent = translate(custom_api, translation_input())
    assert sent["turns"][0] == {
        "content": "New line", "speaker": "b",
        "history": [["Earlier line", "Earlier translation"]], "speakers": ["a"],
    }


def test_legacy_schema_does_not_emit_links_to_unsent_context(custom_api):
    definition = custom_api.store.endpoint.model_dump()
    del definition["request_schema"]["children"]["context"]
    custom_api.store.endpoint = UserEndpoint.model_validate(definition)
    sent = translate(custom_api, translation_input())
    assert "context" not in sent
    assert sent["context_character_links"] == []
    assert sent["meta_id"] == "b"


def test_legacy_schema_survives_store_reload(tmp_path):
    store = UserEndpointStore(tmp_path)
    definition = make_endpoint().model_dump()
    del definition["request_schema"]["children"]["context"]
    del definition["request_schema"]["children"]["meta_id"]
    created = store.create(definition)
    reloaded = UserEndpointStore(tmp_path)
    reloaded.load()
    assert reloaded.get(created.id) is not None


def test_send_test_and_preview_use_complete_matching_sample_metadata(custom_api):
    preview = custom_api.client.post("/api/user_endpoints/preview", json={
        "endpoint": custom_api.store.endpoint.model_dump(),
    })
    assert preview.status_code == 200
    assert custom_api.sent == []
    response = custom_api.client.post("/api/user_endpoints/custom/test", json={})
    assert response.status_code == 200 and response.json()["status"] is True
    assert response.json()["request"] == custom_api.sent[-1]
    sent = json.loads(custom_api.sent[-1]["body"])
    assert json.loads(preview.json()["request"]["body"]) == sent
    assert sent["context"] == custom_endpoint.PREVIEW_CONTEXT
    assert sent["character_info"] == custom_endpoint.PREVIEW_CHARACTER_INFO
    assert sent["context_character_links"] == custom_endpoint.PREVIEW_CONTEXT_CHARACTER_LINKS
    assert sent["meta_id"] == custom_endpoint.PREVIEW_META_ID


@pytest.mark.parametrize("metadata", [
    {"context": []},
    {"character_info": None},
    {"meta_id": None},
    {"context": None, "character_info": None, "context_character_links": None, "meta_id": None},
])
def test_explicit_empty_test_metadata_never_gets_sample_values(custom_api, metadata):
    response = custom_api.client.post("/api/user_endpoints/custom/test", json=metadata)
    assert response.status_code == 200 and response.json()["status"] is True
    sent = json.loads(custom_api.sent[-1]["body"])
    assert sent["context"] == sent["character_info"] == sent["context_character_links"] == []
    assert sent["meta_id"] is None
    preview = custom_api.client.post("/api/user_endpoints/preview", json={
        "endpoint": custom_api.store.endpoint.model_dump(), **metadata,
    })
    assert preview.status_code == 200
    assert json.loads(preview.json()["request"]["body"]) == sent


def test_supplied_test_dialogue_never_acquires_sample_speakers(custom_api):
    response = custom_api.client.post("/api/user_endpoints/custom/test", json={
        "context": [["Actual source", "Actual translation"]],
    })
    assert response.json()["status"] is True
    sent = json.loads(custom_api.sent[-1]["body"])
    assert sent["context"] == [["Actual source", "Actual translation"]]
    assert sent["character_info"] == []
    assert sent["context_character_links"] == [None]
    assert sent["meta_id"] is None


def test_send_test_forwards_explicit_speaker_and_roster(custom_api):
    response = custom_api.client.post("/api/user_endpoints/custom/test", json=translation_input())
    assert response.json()["status"] is True
    sent = json.loads(custom_api.sent[-1]["body"])
    assert sent["meta_id"] == "b"
    assert sent["context_character_links"] == ["a"]


def test_encrypted_test_reports_exact_request_including_generated_uuid(custom_api, monkeypatch):
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode()
    definition = custom_api.store.endpoint.model_dump()
    definition["request_schema"]["children"]["request_id"] = {"type": "uuid"}
    definition["encryption"] = {"enabled": True, "key": key, "decrypt_response": False}
    custom_api.store.endpoint = UserEndpoint.model_validate(definition)
    real_prepare = custom_endpoint.prepare_request
    preparations = []

    def prepare(*args, **kwargs):
        request = real_prepare(*args, **kwargs)
        preparations.append(request)
        return request

    monkeypatch.setattr(custom_endpoint, "prepare_request", prepare)
    response = custom_api.client.post("/api/user_endpoints/custom/test", json=translation_input())
    assert response.json()["status"] is True
    assert len(preparations) == 1
    assert response.json()["request"] == custom_api.sent[-1]
    sent = json.loads(custom_api.sent[-1]["body"])
    assert sent["request_id"]
    assert sent["meta_id"] == "b" and sent["context_character_links"] == ["a"]
    assert [character["meta_id"] for character in sent["character_info"]] == ["a", "b"]
    assert endpoint_crypto.decrypt(sent["text"], secret=key, encoding="fernet") == "New line"
    assert endpoint_crypto.decrypt(sent["character_info"][0]["name_en"], secret=key, encoding="fernet") == "Aiko"
    assert endpoint_crypto.decrypt(sent["context"][0][0], secret=key, encoding="fernet") == "Earlier line"


def test_endpoints_without_metadata_nodes_keep_their_original_payload(custom_api):
    definition = custom_api.store.endpoint.model_dump(exclude={"request_schema"})
    custom_api.store.endpoint = UserEndpoint.model_validate(definition)
    sent = translate(custom_api, translation_input())
    assert sent == {"text": "New line", "src_lang": "japanese", "target_lang": "en"}
