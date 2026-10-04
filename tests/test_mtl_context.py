"""Unit tests for translation context: validation, capability, prompt block."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fox_reader.models.requests import TranslationRequest
from fox_reader.translate.context import (
    MAX_CONTEXT_PAIRS,
    model_supports_context,
    normalize_context,
    render_context_block,
    source_name,
)


def test_normalize_keeps_valid_pairs_trimmed():
    assert normalize_context([["  a  ", "  b "], ["c", "d"]]) == [["a", "b"], ["c", "d"]]


def test_normalize_drops_anything_unusable():
    assert normalize_context(None) == []
    assert normalize_context("nope") == []
    assert normalize_context([["a", ""], ["", "b"], ["only"], ["a", "b", "c"], None, 42, [["x"], "y"]]) == []


def test_normalize_keeps_the_most_recent_pairs_up_to_the_cap():
    pairs = [[f"s{i}", f"e{i}"] for i in range(MAX_CONTEXT_PAIRS + 3)]
    result = normalize_context(pairs)
    assert len(result) == MAX_CONTEXT_PAIRS
    assert result[0] == ["s3", "e3"]
    assert result[-1] == [f"s{MAX_CONTEXT_PAIRS + 2}", f"e{MAX_CONTEXT_PAIRS + 2}"]


def test_normalize_truncates_an_overlong_side():
    long = "x" * 5000
    [[source, english]] = normalize_context([[long, "ok"]])
    assert english == "ok"
    assert 0 < len(source) <= 1000


def test_model_supports_context_only_for_gguf_models():
    assert model_supports_context("gemma-4-e4b-q8-uncensored") is True
    assert model_supports_context("gemma-4-e4b-q6-uncensored") is True
    assert model_supports_context("vntl-llama3-8b-v2") is True
    assert model_supports_context("unknown-model") is False
    assert model_supports_context(None) is False
    assert model_supports_context("") is False


def test_source_name_falls_back_to_original():
    assert source_name("japanese") == "Japanese"
    assert source_name(" KOREAN ") == "Korean"
    assert source_name("klingon") == "Original"
    assert source_name(None) == "Original"


def test_render_block_is_empty_without_pairs():
    assert render_context_block([]) == ""
    assert render_context_block([["", ""]]) == ""


def test_render_block_numbers_oldest_first_and_flattens():
    block = render_context_block([["a\nb", "A B"], ["c", "C"]], "japanese")
    assert '1. Japanese: "a b"' in block
    assert '2. Japanese: "c"' in block
    assert block.index('"a b"') < block.index('"c"')
    assert "Do NOT translate" in block
    assert "Translate ONLY the new" in block


def test_render_block_uses_original_for_unknown_languages():
    assert '1. Original: "a"' in render_context_block([["a", "b"]], "klingon")


def test_translation_request_defaults_to_no_context():
    req = TranslationRequest(text="hi", source_lang="japanese")
    assert req.context is None


def test_translation_request_accepts_context():
    req = TranslationRequest(text="hi", source_lang="japanese", context=[["a", "b"]])
    assert req.context == [["a", "b"]]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
