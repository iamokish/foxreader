"""Context choices respect catalogue and on-disk GGUF limits without loading weights."""

from __future__ import annotations

import io
import os
import struct
from pathlib import Path

import pytest

from fox_reader.constants import MTL_MODELS
from fox_reader.translate import context_window as window

GEMMA = "gemma-4-e4b-q6-uncensored"
QWEN = "qwen3.5-4b-q8-0-uncensored"
VNTL = "vntl-llama3-8b-v2-q5-k-m"


def _string(value: str, endian: str = "<") -> bytes:
    text = value.encode("utf-8")
    return struct.pack(endian + "Q", len(text)) + text


def _field(key: str, kind: int, value: object, endian: str = "<") -> bytes:
    prefix = _string(key, endian) + struct.pack(endian + "I", kind)
    if kind == 8:
        return prefix + _string(value, endian)
    if kind in (4, 5, 10, 11):
        return prefix + struct.pack(endian + {4: "I", 5: "i", 10: "Q", 11: "q"}[kind], value)
    return prefix + value


def _gguf(fields: list[bytes], *, endian: str = "<", version: int = 3) -> bytes:
    return b"GGUF" + struct.pack(endian + "IQQ", version, 123, len(fields)) + b"".join(fields)


def _context(limit: int, *, architecture: str = "gemma4", endian: str = "<") -> bytes:
    return _gguf([
        _field("general.architecture", 8, architecture, endian),
        _field(f"{architecture}.context_length", 4, limit, endian),
    ], endian=endian)


@pytest.mark.parametrize("entry", MTL_MODELS, ids=lambda entry: entry["id"])
def test_catalogue_limits_offer_only_supported_choices(entry):
    model_id = entry["id"]
    if model_id.startswith("vntl-"):
        assert window.model_context_limit(model_id) == 8192
        assert window.context_size_options(model_id) == [8192]
        assert window.resolve_context_size(model_id, 16384) == 8192
    else:
        expected = 131072 if model_id.startswith("gemma-") else 262144
        assert window.model_context_limit(model_id) == expected
        assert window.context_size_options(model_id) == [8192, 16384]
        assert window.resolve_context_size(model_id, 16384) == 16384
    assert window.resolve_context_size(model_id) == 8192


@pytest.mark.parametrize("requested", [True, False, "8192", 8192.0, None, 4096, 32768, -1])
def test_context_size_choices_are_strict(requested):
    with pytest.raises(ValueError, match="8192 or 16384"):
        window.resolve_context_size(GEMMA, requested)


@pytest.mark.parametrize("helper", [
    window.model_context_limit, window.context_size_options, window.resolve_context_size,
])
def test_unknown_models_are_rejected(helper):
    with pytest.raises(ValueError, match="Unknown local translation model"):
        helper("unknown-model")


@pytest.mark.parametrize("endian", ["<", ">"])
def test_actual_gguf_can_only_lower_catalogue_limit(tmp_path, endian):
    path = tmp_path / "model.gguf"
    path.write_bytes(_context(8192, endian=endian))
    assert window.model_context_limit(GEMMA, path) == 8192
    assert window.context_size_options(GEMMA, path) == [8192]
    assert window.resolve_context_size(GEMMA, 16384, path) == 8192

    path = tmp_path / "larger.gguf"
    path.write_bytes(_context(262144, endian=endian))
    assert window.model_context_limit(GEMMA, path) == 131072
    assert window.model_context_limit(VNTL, path) == 8192


def test_actual_limit_between_choices_or_below_minimum(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(_context(12000))
    assert window.context_size_options(GEMMA, path) == [8192]
    assert window.resolve_context_size(GEMMA, 16384, path) == 8192
    path = tmp_path / "smaller.gguf"
    path.write_bytes(_context(4096))
    assert window.context_size_options(GEMMA, path) == []
    with pytest.raises(ValueError, match="minimum 8192"):
        window.resolve_context_size(GEMMA, 8192, path)


def test_metadata_order_and_unrelated_context_lengths(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(_gguf([
        _field("vision.context_length", 4, 512),
        _field("qwen35.context_length", 10, 16384),
        _field("general.architecture", 8, "qwen35"),
    ], version=2))
    assert window.model_context_limit(QWEN, path) == 16384


def test_probe_skips_unrelated_values_and_arrays(tmp_path):
    path = tmp_path / "model.gguf"
    numbers = struct.pack("<IQ3I", 4, 3, 1, 2, 3)
    strings = struct.pack("<IQ", 8, 2) + _string("one") + _string("two")
    path.write_bytes(_gguf([
        _field("custom.string", 8, "unused metadata"),
        _field("custom.numbers", 9, numbers),
        _field("custom.strings", 9, strings),
        _field("general.architecture", 8, "gemma4"),
        _field("gemma4.context_length", 4, 16384),
    ]))
    assert window.model_context_limit(GEMMA, path) == 16384


@pytest.mark.parametrize("fields", [
    [_field("general.architecture", 8, "gemma4")],
    [_field("general.architecture", 8, "gemma4"), _field("vision.context_length", 4, 262144)],
    [_field("general.architecture", 8, "gemma4"), _field("gemma4.context_length", 8, "16384")],
    [_field("general.architecture", 8, "gemma4"), _field("gemma4.context_length", 5, -1)],
    [_field("general.architecture", 8, "gemma4"), _field("gemma4.context_length", 4, 0)],
    [_field("unknown.type", 999, b"")],
    [_field("oversized.array", 9, struct.pack("<IQ", 4, 2**63))],
    [_field("oversized.strings", 9, struct.pack("<IQ", 8, window._MAX_ARRAY_STRINGS + 1))],
])
def test_recognized_gguf_with_unknown_or_malformed_metadata_is_conservative(tmp_path, fields):
    path = tmp_path / "model.gguf"
    path.write_bytes(_gguf(fields))
    assert window.model_context_limit(GEMMA, path) == 8192
    assert window.resolve_context_size(GEMMA, 16384, path) == 8192


@pytest.mark.parametrize("content", [b"", b"GGUF", b"not a GGUF"])
def test_invalid_files_remain_native_loader_errors_with_catalogue_fallback(tmp_path, content):
    path = tmp_path / "model.gguf"
    path.write_bytes(content)
    assert window.context_size_options(GEMMA, path) == [8192, 16384]
    assert window.context_size_options(GEMMA, tmp_path / "missing.gguf") == [8192, 16384]


def test_probe_stops_before_vocabulary_or_tensor_payload(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    known_fields = [
        _field("general.architecture", 8, "gemma4"),
        _field("gemma4.context_length", 4, 131072),
    ]
    prefix = _gguf(known_fields)
    data = _gguf([*known_fields, _field("tokenizer.ggml.tokens", 9, b"malformed huge vocabulary")])
    path.write_bytes(data + b"tensor data")
    reads = []

    class HeaderOnly(io.BytesIO):
        def read(self, count=-1):
            assert 0 <= count <= 1024
            assert self.tell() + count <= len(prefix), "probe touched the vocabulary or tensors"
            reads.append(count)
            return super().read(count)

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: HeaderOnly(data))
    assert window.model_context_limit(GEMMA, path) == 131072
    assert sum(reads) == len(prefix)


def test_cached_probe_invalidates_when_weights_are_replaced(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    path.write_bytes(_context(8192))
    original_open = Path.open
    probes = []

    def track_open(opened_path, mode="r", *args, **kwargs):
        if mode == "rb":
            probes.append(opened_path)
        return original_open(opened_path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)
    assert window.model_context_limit(GEMMA, path) == 8192
    assert window.model_context_limit(GEMMA, path) == 8192
    assert len(probes) == 1
    previous = path.stat()
    path.write_bytes(_context(16384))
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
    assert window.model_context_limit(GEMMA, path) == 16384
    assert len(probes) == 2
    with path.open("ab") as stream:
        stream.write(b"different size")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
    assert window.model_context_limit(GEMMA, path) == 16384
    assert len(probes) == 3
