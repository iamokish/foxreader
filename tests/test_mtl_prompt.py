"""Prompt and request regressions, without allocating native model weights."""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fox_reader.constants import MTL_MODELS, mtl_model
from fox_reader.translate.context import MAX_CONTEXT_PAIRS, normalize_context_with_links
from fox_reader.translate.local_mtl import LocalMTLManager
from fox_reader.translate.machine_translation.gemma_e4b_llamacpp import GemmaE4BQ6Translator, GemmaE4BQ8Translator
from fox_reader.translate.machine_translation.vntl_llama3_8b_llamacpp import VntlLlamaQ5Translator
from fox_reader.translate.prompt import build_system_prompt, build_translation_messages


class Tokenizer:
    def tokenize(self, value, **kwargs):
        return list(value.decode("utf-8"))


ROSTER = [
    {"meta_id": "opaque-a", "name_ja": "愛子", "name_en": "Aiko", "gender": "female"},
    {"meta_id": "opaque-b", "name_ja": "健", "name_en": "Ken", "alias_ja": "先生", "alias_en": "Teacher"},
]


def test_normalization_keeps_links_when_rows_are_filtered_and_capped():
    pairs = [[f"s{i}", f"e{i}"] for i in range(MAX_CONTEXT_PAIRS + 3)]
    pairs.insert(2, ["", "bad row"])
    links = [f"speaker{i}" for i in range(len(pairs))]
    normalized, speakers = normalize_context_with_links(pairs, links)
    assert normalized[0] == ["s3", "e3"]
    assert speakers == links[4:]
    assert normalize_context_with_links([["a", "A"], ["b", "B"]], ["known"]) == (
        [["a", "A"], ["b", "B"]], ["known", None],
    )


def test_characters_work_without_context_and_opaque_ids_never_render():
    system = build_system_prompt(character_info=ROSTER, meta_id="opaque-b")
    assert "Aiko" in system and "Ken" in system
    assert "Teacher" in system and "先生" in system
    assert "female" in system
    assert "Speaker of the new text: Character 2." in system
    assert "Conversation context" not in system
    assert "opaque-" not in system
    assert "Do not add names, speaker labels" in system
    assert "健 -> Ken" in system
    assert "先生 -> Teacher" in system
    assert "never translate names by their dictionary meaning" in system


def test_reference_labels_stay_with_valid_history_rows():
    system = build_system_prompt(
        [["", "invalid"], ["健です。", "I'm Ken."]], "japanese", ROSTER,
        ["opaque-a", "opaque-b"], "does-not-exist",
    )
    assert "Speaker for pair 1: Character 2" in system
    assert "Speaker of the new text" not in system
    assert "invalid" not in system


def test_bare_text_does_not_invent_a_roster_or_context():
    messages = build_translation_messages(Tokenizer(), "こんにちは。")
    assert messages[1] == {"role": "user", "content": "こんにちは。"}
    assert "Character reference" not in messages[0]["content"]
    assert "Conversation context" not in messages[0]["content"]
    assert messages[0]["content"] == build_system_prompt()


@pytest.mark.parametrize("use_context,use_characters", [(True, False), (False, True), (True, True)])
def test_reference_data_does_not_teach_quoted_output(use_context, use_characters):
    system = build_system_prompt(
        [["こんにちは。", "Hello."]] if use_context else None, "japanese",
        ROSTER if use_characters else None, ["opaque-a"], "opaque-a",
    )
    assert '"' not in system
    assert "Do not add quotation marks around the translation" in system
    assert "Preserve quotation marks present in the source" in system
    if use_context:
        assert "English: Hello." in system
    if use_characters:
        assert "English name: Aiko" in system
        assert "愛子 -> Aiko" in system


def test_irrelevant_roster_is_trimmed_before_recent_dialogue():
    irrelevant = [{"meta_id": f"other{i}", "name_en": f"Unrelated{i}" + "x" * 80} for i in range(40)]
    def count(messages):
        return sum(len(message["content"]) for message in messages)
    messages = build_translation_messages(
        Tokenizer(), "健、待って。", [["愛子です。", "I'm Aiko."]], "japanese",
        [*ROSTER, *irrelevant], ["opaque-a"], "opaque-b",
        n_ctx=2100, max_new_tokens=100, prompt_token_count=count,
    )
    system = messages[0]["content"]
    assert "I'm Aiko." in system and "Aiko" in system and "Ken" in system
    assert "Speaker of the new text" in system
    assert "Unrelated39" not in system
    assert count(messages) + 100 <= 2100
    assert messages[1]["content"] == "健、待って。"


def test_oldest_history_is_trimmed_with_its_speaker():
    messages = build_translation_messages(
        Tokenizer(), "Current", [["old" * 250, "old translation" * 60], ["recent", "newest"]],
        "japanese", ROSTER, ["opaque-a", "opaque-b"], "opaque-b",
        n_ctx=2150, max_new_tokens=100,
    )
    system = messages[0]["content"]
    assert "old translation" not in system
    assert "English: newest" in system
    assert "Speaker for pair 1: Character 2" in system
    assert "Speaker for pair 1: Character 1" not in system


def test_oversized_source_is_rejected_without_truncation():
    with pytest.raises(RuntimeError, match="too long.*context window"):
        build_translation_messages(Tokenizer(), "long source" * 1000)


def test_character_from_dropped_history_does_not_displace_newer_dialogue():
    def count(messages):
        return sum(len(message["content"]) for message in messages)

    roster = [
        {"meta_id": "old", "name_en": "Discarded" + "x" * 90, "name_ja": "y" * 100, "alias_en": "z" * 100},
        {"meta_id": "current", "name_en": "Ken"},
    ]
    recent = [["recent", "newest"]]
    # Enough for the recent dialogue and its speaker, but not the older cast.
    wanted = build_system_prompt(recent, "japanese", roster[1:], ["current"], "current")
    source = "Current"
    window = len(wanted) + len(source) + 100 + 4
    messages = build_translation_messages(
        Tokenizer(), source, [["old" * 300, "old English" * 80], *recent], "japanese",
        roster, ["old", "current"], "current",
        n_ctx=window, max_new_tokens=100, prompt_token_count=count,
    )
    assert "newest" in messages[0]["content"]
    assert "Discarded" not in messages[0]["content"]
    assert count(messages) + 104 <= window


@pytest.mark.parametrize("cls", [GemmaE4BQ8Translator, GemmaE4BQ6Translator])
def test_gemma_passes_characters_and_preserves_existing_sampling(cls):
    translator = cls.__new__(cls)
    translator._lock = threading.RLock()
    translator.n_ctx = 8192
    calls = []
    model = Tokenizer()

    def complete(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "Hello, Ken."}}]}

    model.create_chat_completion = complete
    translator.model = model
    assert translator.translate("こんにちは、健。", character_info=ROSTER, meta_id="opaque-a") == "Hello, Ken."
    sent = calls[0]
    assert "Aiko" in sent["messages"][0]["content"]
    assert "Speaker of the new text: Character 1." in sent["messages"][0]["content"]
    assert sent["temperature"] == 1.0 and sent["top_p"] == 0.95 and sent["top_k"] == 64


@pytest.mark.parametrize("model_id", [model["id"] for model in MTL_MODELS])
def test_manager_forwards_aligned_metadata_and_plain_text(model_id):
    manager = LocalMTLManager(Path("."))
    manager._load_locked = lambda lang: None
    calls = []
    manager.model = SimpleNamespace(
        model_id=lambda: model_id,
        translate=lambda text, **kwargs: calls.append((text, kwargs)) or "translated",
    )
    manager.translate("japanese", "hello", [["", "bad"], ["b", "B"]], ROSTER,
                      ["opaque-a", "opaque-b"], "opaque-a")
    assert calls[0][1]["context"] == [["b", "B"]]
    assert calls[0][1]["context_character_links"] == ["opaque-b"]
    assert calls[0][1]["meta_id"] == "opaque-a"
    manager.translate("japanese", "only text")
    assert calls[1][1]["context"] == []
    assert calls[1][1]["character_info"] == []
    assert calls[1][1]["meta_id"] is None


def test_manager_does_not_retry_inference_errors_without_metadata():
    manager = LocalMTLManager(Path("."))
    manager._load_locked = lambda lang: None
    calls = []

    def broken(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("native inference failed")

    manager.model = SimpleNamespace(model_id=lambda: "gemma-4-e4b-q8-uncensored", translate=broken)
    with pytest.raises(TypeError, match="native inference"):
        manager.translate("japanese", "text", character_info=ROSTER)
    assert len(calls) == 1


def test_registry_matches_catalog_and_vntl_quant_requires_exact_file(tmp_path):
    manager = LocalMTLManager(tmp_path)
    assert {cls.model_id() for cls in manager.available_translators} == {model["id"] for model in MTL_MODELS}
    entry = mtl_model(VntlLlamaQ5Translator.model_id())
    path = tmp_path / entry["dest"] / entry["files"][0]
    path.parent.mkdir(parents=True)
    (path.parent / "wrong-quant.gguf").touch()
    with pytest.raises(FileNotFoundError, match="Q5_K_M"):
        VntlLlamaQ5Translator._resolve_weights(tmp_path)
    path.touch()
    assert VntlLlamaQ5Translator._resolve_weights(tmp_path) == path
    assert VntlLlamaQ5Translator.model_langs() == ["japanese"]


@pytest.mark.parametrize("endpoint", ["ml", "custom"])
def test_http_keeps_context_and_speaker_links_aligned(endpoint):
    from fox_reader.models.responses import TranslationResult
    from fox_reader.routes.translate import router

    calls = []

    async def translate(*args):
        calls.append(args)
        return TranslationResult(
            code=200, id=1, data="English", source_lang="japanese", target_lang="en", method="test",
        )

    app = FastAPI()
    app.include_router(router)
    app.state.translate = SimpleNamespace(ml_async=translate, custom=translate)
    with TestClient(app) as client:
        response = client.post(f"/translate/{endpoint}", json={
            "text": "text", "source_lang": "japanese", "context": [["", "drop"], ["b", "B"]],
            "context_character_links": ["opaque-a", "opaque-b"],
        })
    assert response.json()["translated"] == "English"
    offset = 0 if endpoint == "ml" else 1
    # Custom endpoints normalize later, after resolving their configured schema.
    # The route must preserve both raw arrays; test_custom_endpoint_context checks
    # their filtering and alignment in the actual outgoing endpoint request.
    expected_context = [["b", "B"]] if endpoint == "ml" else [["", "drop"], ["b", "B"]]
    expected_links = ["opaque-b"] if endpoint == "ml" else ["opaque-a", "opaque-b"]
    assert calls[0][2 + offset] == expected_context
    assert calls[0][4 + offset] == expected_links
