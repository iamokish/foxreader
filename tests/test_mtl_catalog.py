"""Download and licence isolation for local translation quantisations."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fox_reader import model_terms
from fox_reader.constants import MTL_COMPLETION_MARKER, MTL_MODELS, mtl_model, mtl_model_downloaded


@pytest.mark.parametrize(
    ("model_id", "repo", "filename"),
    [
        (
            "vntl-llama3-8b-v2-q5-k-m",
            "lmg-anon/vntl-llama3-8b-v2-gguf",
            "vntl-llama3-8b-v2-hf-q5_k_m.gguf",
        ),
        (
            "qwen3.5-9b-q6-k-uncensored",
            "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
            "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf",
        ),
        (
            "qwen3.5-9b-q8-0-uncensored",
            "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
            "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf",
        ),
        (
            "qwen3.5-4b-q8-0-uncensored",
            "HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive",
            "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf",
        ),
    ],
)
def test_requested_download_uses_only_selected_text_weights(model_id, repo, filename):
    from fox_reader.routes.setup import _discover_model_files, _serialize_mtl_model

    model = mtl_model(model_id)
    assert model is not None
    assert model["repo"] == repo

    # Repositories contain several quants and a separate vision encoder. Only
    # the requested text GGUF should consume download space and need a marker.
    siblings = [filename, "mmproj-vision-BF16.gguf", "another-quant.gguf", "README.md"]
    api = SimpleNamespace(list_repo_files=lambda selected_repo: siblings if selected_repo == repo else [])
    assert _discover_model_files(api, model) == [filename]
    assert model["files"] == [filename]
    assert model["memory_estimated"] is True
    assert "estimate" in model["cpu_ram"]
    assert "estimate" in model["gpu_vram"]

    # The estimates remain labelled after serialisation to the setup page.
    visible = _serialize_mtl_model(model, Path("unused"), {})
    assert visible["cpu_ram"] == model["cpu_ram"]
    assert visible["gpu_vram"] == model["gpu_vram"]


@pytest.mark.parametrize(
    ("first_id", "second_id"),
    [
        ("vntl-llama3-8b-v2", "vntl-llama3-8b-v2-q5-k-m"),
        ("qwen3.5-9b-q6-k-uncensored", "qwen3.5-9b-q8-0-uncensored"),
    ],
)
def test_quants_have_independent_completion_and_acceptance(tmp_path, first_id, second_id):
    first = mtl_model(first_id)
    second = mtl_model(second_id)
    assert first["dest"] != second["dest"]
    model_dir = tmp_path / first["dest"]
    model_dir.mkdir(parents=True)
    (model_dir / first["files"][0]).write_bytes(b"test weights")
    (model_dir / MTL_COMPLETION_MARKER).touch()
    assert mtl_model_downloaded(tmp_path, first_id)
    assert not mtl_model_downloaded(tmp_path, second_id)

    terms = model_terms.TermsStore(tmp_path / "config")
    assert terms.unaccepted([first_id, second_id]) == [first_id, second_id]
    terms.accept([first_id])
    assert terms.is_accepted(first_id)
    assert terms.unaccepted([first_id, second_id]) == [second_id]


def test_all_local_downloads_have_distinct_storage_and_complete_notices():
    assert len({model["id"] for model in MTL_MODELS}) == len(MTL_MODELS)
    assert len({model["dest"] for model in MTL_MODELS}) == len(MTL_MODELS)
    for model in MTL_MODELS:
        doc = model_terms.document(model["id"])
        assert doc is not None
        assert doc["version"] and doc["accept_label"] and doc["sections"]
        assert any(link["url"] == f"https://huggingface.co/{model['repo']}" for link in doc["links"])
        if model["id"].startswith("qwen3.5-"):
            assert doc["licence"] == "Apache License 2.0"
        elif model["id"].startswith("vntl-"):
            assert "Llama 3 Community License" in doc["licence"]
