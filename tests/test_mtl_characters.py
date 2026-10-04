"""Unit tests for character metadata: store, CSV, capability, request, prompt.

Memory-only by design: nothing here writes to disk, and the VNTL prompt is
exercised with a stub tokenizer so no 8.5 GiB GGUF is needed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fox_reader.translate import characters as chars
from fox_reader.translate.characters import CharacterStore, parse_csv, to_csv
from fox_reader.translate.context import model_supports_characters, model_supports_context


@pytest.fixture()
def fresh_store(monkeypatch):
    store = CharacterStore()
    monkeypatch.setattr(chars, "store", store)
    store.clear()
    yield store
    store.clear()


def test_gender_normalises_to_male_female_or_none():
    assert chars._clean_gender("Male") == "male"
    assert chars._clean_gender("  FEMALE ") == "female"
    assert chars._clean_gender("") is None
    assert chars._clean_gender("nonbinary") is None
    assert chars._clean_gender(None) is None


def test_meta_id_is_stripped_never_sanitised():
    assert chars.normalise_meta_id("  abc  ") == "abc"
    assert chars.normalise_meta_id("") is None
    assert chars.normalise_meta_id("  ") is None
    assert chars.normalise_meta_id(7) == "7"
    assert chars.normalise_meta_id(["a"]) is None
    # Full-width space survives: byte-exact on purpose.
    assert chars.normalise_meta_id("山田　太郎") == "山田　太郎"


def test_add_generates_meta_id_and_validates(fresh_store):
    created = fresh_store.add({"name_en": "Aiko", "name_ja": "愛子", "gender": "female"})
    assert created["meta_id"]
    assert created["name_en"] == "Aiko"
    assert created["gender"] == "female"
    with pytest.raises(ValueError):
        fresh_store.add({"name_en": "   "})
    with pytest.raises(ValueError):
        fresh_store.add("nope")


def test_add_ids_are_unique(fresh_store):
    ids = {fresh_store.add({"name_en": f"N{i}"})["meta_id"] for i in range(10)}
    assert len(ids) == 10


def test_update_is_partial_and_clears_on_blank(fresh_store):
    created = fresh_store.add({"name_en": "Aiko", "gender": "female"})
    updated = fresh_store.update(created["meta_id"], {"name_ja": "愛子"})
    assert updated["name_en"] == "Aiko"
    assert updated["name_ja"] == "愛子"
    cleared = fresh_store.update(created["meta_id"], {"name_en": "  "})
    assert cleared["name_en"] is None
    assert cleared["name_ja"] == "愛子"
    with pytest.raises(KeyError):
        fresh_store.update("missing", {"name_en": "x"})


def test_delete_and_clear(fresh_store):
    one = fresh_store.add({"name_en": "A"})
    two = fresh_store.add({"name_en": "B"})
    assert fresh_store.delete(one["meta_id"]) is True
    assert fresh_store.delete("missing") is False
    assert [c["name_en"] for c in fresh_store.list()] == ["B"]
    assert fresh_store.clear() == 1
    assert fresh_store.list() == []
    assert two["meta_id"]  # unused, keeps linters honest about two existing


def test_csv_roundtrip_without_meta_id(fresh_store):
    fresh_store.add({"name_en": "Aiko", "name_ja": "愛子", "gender": "female", "alias_en": "Sis", "alias_ja": "お姉ちゃん"})
    fresh_store.add({"name_ja": "先生", "gender": "male"})
    text = fresh_store.export_csv()
    assert text.splitlines()[0] == "name_en,name_ja,gender,alias_en,alias_ja"
    assert "meta_id" not in text.splitlines()[0]
    rows = parse_csv(text)
    assert len(rows) == 2
    assert rows[0]["name_en"] == "Aiko"
    assert rows[0]["gender"] == "female"
    # Fresh ids on import: the CSV carries none.
    imported = fresh_store.import_csv(text)
    assert len(imported) == 2
    assert all(c["meta_id"] for c in imported)


def test_csv_skips_blank_rows_and_normalises_gender():
    text = "name_en,name_ja,gender,alias_en,alias_ja\nAiko,,F,,\n,,,\nKen,,male,,\n"
    rows = parse_csv(text)
    assert [r["name_en"] for r in rows] == ["Aiko", "Ken"]
    assert rows[0]["gender"] is None  # "F" is non-standard
    assert rows[1]["gender"] == "male"
    assert parse_csv("") == []
    assert parse_csv("﻿name_en,name_ja,gender,alias_en,alias_ja\nA,,,,\n")[0]["name_en"] == "A"


def test_to_csv_quotes_commas():
    text = to_csv([{"name_en": "A, B", "name_ja": None, "gender": None, "alias_en": None, "alias_ja": None}])
    assert '"A, B"' in text
    assert parse_csv(text)[0]["name_en"] == "A, B"


def test_normalize_character_info_accepts_all_shapes():
    by_list = chars.normalize_character_info([{"meta_id": "a", "name_en": "A"}])
    assert set(by_list) == {"a"}
    by_single = chars.normalize_character_info({"meta_id": "b", "name_en": "B"})
    assert set(by_single) == {"b"}
    by_map = chars.normalize_character_info({"c": {"name_en": "C"}})
    assert set(by_map) == {"c"}
    assert chars.normalize_character_info(None) == {}
    assert chars.normalize_character_info("nope") == {}
    # Duplicates keep the first.
    dup = chars.normalize_character_info(
        [{"meta_id": "a", "name_en": "First"}, {"meta_id": "a", "name_en": "Second"}]
    )
    assert dup["a"]["name_en"] == "First"


def test_normalize_links_pads_and_truncates():
    assert chars.normalize_character_links(None, 2) == [None, None]
    assert chars.normalize_character_links(["a"], 2) == ["a", None]
    assert chars.normalize_character_links(["a", "b", "c"], 2) == ["a", "b"]
    assert chars.normalize_character_links("nope", 2) == [None, None]


def test_capability_flags():
    assert model_supports_context("vntl-llama3-8b-v2") is True
    assert model_supports_characters("vntl-llama3-8b-v2") is True
    assert model_supports_characters("gemma-4-e4b-q8-uncensored") is False
    assert model_supports_characters("unknown-model") is False
    assert model_supports_characters(None) is False


def test_translation_request_carries_characters():
    from fox_reader.models.requests import TranslationRequest

    req = TranslationRequest(
        text="hi",
        source_lang="japanese",
        context=[["a", "b"]],
        character_info=[{"meta_id": "a", "name_en": "A"}],
        context_character_links=["a"],
        meta_id="a",
    )
    assert req.character_info == [{"meta_id": "a", "name_en": "A"}]
    assert req.context_character_links == ["a"]
    assert req.meta_id == "a"
    # Old pages keep working: everything defaults to None.
    bare = TranslationRequest(text="hi", source_lang="japanese")
    assert bare.character_info is None
    assert bare.context_character_links is None
    assert bare.meta_id is None


def test_vntl_model_registered_for_japanese():
    from fox_reader.constants import mtl_model
    from fox_reader.translate.local_mtl import LocalMTLManager

    entry = mtl_model("vntl-llama3-8b-v2")
    assert entry is not None
    assert entry["languages"] == ["japanese"]
    mgr = LocalMTLManager(mtl_dir=Path("."), settings=None)
    ids = [cls.model_id() for cls in mgr._compatible_translators("japanese")]
    assert "vntl-llama3-8b-v2" in ids
    assert "vntl-llama3-8b-v2" not in [cls.model_id() for cls in mgr._compatible_translators("korean")]


class _StubLLM:
    """One token per character, so prompts read back verbatim."""

    def tokenize(self, data, add_bos=True, special=False):
        return [ord(ch) for ch in data.decode("utf-8")]

    def close(self):
        pass


def _vntl_without_weights():
    from fox_reader.translate.machine_translation import vntl_llama3_8b_llamacpp as vntl

    tr = vntl.VntlLlamaTranslator.__new__(vntl.VntlLlamaTranslator)
    import threading

    tr.model = _StubLLM()
    tr._role_tokens = {role: [ord(c) for c in role] for role in vntl._ROLES}
    tr._load_lock = threading.RLock()
    return vntl, tr


def test_vntl_prompt_starts_with_bos_and_ends_open():
    vntl, tr = _vntl_without_weights()
    prepared = vntl._InputValidator.prepare(
        "今日は早いね。",
        [("先生、おはよう。", "Good morning.")],
        [{"meta_id": "a", "name_en": "Aiko", "name_ja": "愛子", "gender": "female"}],
        ["a"],
        "a",
    )
    ids, dropped = tr._build_prompt(prepared)
    assert dropped == 0
    assert ids[0] == vntl.TOK_BEGIN_OF_TEXT
    assert ids[-3:] == [vntl.TOK_END_HEADER, vntl.NEWLINE, vntl.NEWLINE]
    assert ids.count(vntl.TOK_START_HEADER) == 3 + 2 * 1
    assert ids.count(vntl.TOK_EOT) == 2 + 2 * 1


def test_vntl_trimming_drops_oldest_first():
    vntl, tr = _vntl_without_weights()
    pairs = [(f"日本語{i}", f"English line number {i}") for i in range(30)]
    prepared = vntl._InputValidator.prepare("短い。", pairs, None, None, None)
    ids, dropped = tr._build_prompt(prepared)
    assert dropped >= 0
    assert len(ids) <= vntl.N_CTX - tr._answer_reserve()
    if dropped:
        # Newest survives: the last pair's English is still in the prompt.
        text = "".join(chr(t) for t in ids if t < 128000)
        assert "English line number 29" in text


def test_vntl_parse_reports_english_speaker():
    vntl, tr = _vntl_without_weights()
    prepared = vntl._InputValidator.prepare(
        "今日は早いね。",
        [],
        [{"meta_id": "a", "name_en": "Aiko", "name_ja": "愛子"}],
        [],
        "a",
    )
    assert tr._parse("[愛子]: 早いね。", prepared, 0, 10) == "[Aiko]: 早いね。"
    assert tr._parse("ただ早い。", prepared, 0, 10) == "[Aiko]: ただ早い。"
    assert tr._parse("", prepared, 0, 10) == ""


def test_vntl_parse_collapses_stacked_duplicate_tags():
    vntl, tr = _vntl_without_weights()
    prepared = vntl._InputValidator.prepare(
        "泣いているのか?",
        [],
        [
            {"meta_id": "m", "name_en": "Mikasa", "name_ja": "ミカサ"},
            {"meta_id": "e", "name_en": "Eren", "name_ja": "エレン"},
        ],
        [],
        "m",
    )
    assert tr._parse("[Mikasa]: [Mikasa]: Are you crying?", prepared, 0, 10) == "[Mikasa]: Are you crying?"
    assert (
        tr._parse("[Mikasa]: [Mikasa]: [Mikasa]: Are you crying?", prepared, 0, 10)
        == "[Mikasa]: Are you crying?"
    )
    # The Japanese echo the model mirrors is normalised to English too.
    assert tr._parse("[ミカサ]: [Mikasa]: Are you crying?", prepared, 0, 10) == "[Mikasa]: Are you crying?"
    # Untouched content that merely starts with brackets keeps working.
    assert tr._parse("[Eren]: Has your hair grown?", prepared, 0, 10) == "[Eren]: Has your hair grown?"


def test_vntl_parse_returns_plain_without_characters():
    vntl, tr = _vntl_without_weights()
    # No roster, no speaker, no links: the Characters switch is off, so the
    # model inventing "[Mikasa]:" from its own priors must not leak a prefix.
    prepared = vntl._InputValidator.prepare("今日は早いね。", [[ "a", "b"]], None, None, None)
    assert tr._parse("[Mikasa]: You're early.", prepared, 0, 10) == "You're early."
    assert tr._parse("[Mikasa]: [Mikasa]: You're early.", prepared, 0, 10) == "You're early."
    assert tr._parse("You're early.", prepared, 0, 10) == "You're early."


def test_character_routes_crud(fresh_store):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from fox_reader.routes.characters import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/api/characters").json() == {"characters": []}
    created = client.post("/api/characters", json={"name_en": "Aiko", "gender": "female"}).json()["character"]
    assert created["meta_id"]
    assert client.get("/api/characters").json()["characters"][0]["name_en"] == "Aiko"
    updated = client.put(f"/api/characters/{created['meta_id']}", json={"name_ja": "愛子"}).json()["character"]
    assert updated["name_ja"] == "愛子"
    assert updated["name_en"] == "Aiko"
    assert client.delete(f"/api/characters/{created['meta_id']}").json() == {"status": True}
    assert client.get("/api/characters").json() == {"characters": []}
    assert client.post("/api/characters", json={"name_en": "   "}).status_code == 400

    imported = client.post(
        "/api/characters/import",
        json={"csv": "name_en,name_ja,gender,alias_en,alias_ja\nAiko,愛子,female,,\n"},
    ).json()
    assert imported["added"] == 1
    assert imported["characters"][0]["name_en"] == "Aiko"
    assert imported["characters"][0]["meta_id"]
    assert "Aiko" in client.get("/api/characters/export").text
    assert client.delete("/api/characters").json()["removed"] == 1


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
