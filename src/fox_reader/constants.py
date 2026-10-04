"""Shared constants for model repositories and paths."""
from __future__ import annotations

from pathlib import Path

from fox_reader.utils import MODELS_DIR

# ── OCR engine selection ──────────────────────────────────────────────────────
#: The two text-recognition engines the settings page can offer. ``paddleocr``
#: is the classic PP-OCRv6 detect-then-recognise pipeline that is always
#: available once the required weights are downloaded. ``paddleocr-vl`` is the
#: PaddleOCR-VL-1.6 vision-language GGUF, which needs the ``gguf`` extra
#: (llama-cpp-python) *and* its own weights before it can be selected.
OCR_ENGINE_PADDLEOCR = "paddleocr"
OCR_ENGINE_PADDLEOCR_VL = "paddleocr-vl"
SUPPORTED_OCR_ENGINES: tuple[str, ...] = (OCR_ENGINE_PADDLEOCR, OCR_ENGINE_PADDLEOCR_VL)
DEFAULT_OCR_ENGINE = OCR_ENGINE_PADDLEOCR

# ── PaddleOCR-VL model paths ──────────────────────────────────────────────────
PADDLEOCR_VL_REPO = "PaddlePaddle/PaddleOCR-VL-1.6-GGUF"
#: Lives beside the classic weights: one `paddleocr` folder for everything OCR.
PADDLEOCR_VL_MODEL_DIR = "paddleocr/PaddleOCR-VL-1.6-GGUF"
#: Pre-move location. Existing installs carry weights here; see
#: `fox_reader.ocr_vl.migrate_legacy_vl_dir`, which moves them once instead
#: of forcing a ~1.8 GiB re-download. Never add new files here.
PADDLEOCR_VL_LEGACY_MODEL_DIR = "paddleocr-vl/PaddleOCR-VL-1.6-GGUF"
PADDLEOCR_VL_MODEL_FILE = "PaddleOCR-VL-1.6-GGUF.gguf"
PADDLEOCR_VL_MMPROJ_FILE = "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
#: Jinja chat template shipped beside the GGUFs upstream. Inference reads the
#: template embedded in the GGUF metadata, so this file is bundle completeness
#: (and a future override hook), not a runtime dependency.
PADDLEOCR_VL_CHAT_TEMPLATE_FILE = "chat_template.jinja"

# ── MISC model paths ──────────────────────────────────────────────────────────
TEXT_SEG_MODEL_DIR = "misc/manga-text-segmentation-safetensors"

# ── Bubble model paths ────────────────────────────────────────────────────────
BUBBLE_MODEL_DIR = "bubble"
BUBBLE_MODEL_FILE = "model.safetensors"

# ── MTL model paths ──────────────────────────────────────────────────────────
GEMMA_E4B_Q8_MODEL_DIR = "mtl/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8"
GEMMA_E4B_Q8_MODEL_FILE = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
GEMMA_E4B_Q6_MODEL_DIR = "mtl/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6"
GEMMA_E4B_Q6_MODEL_FILE = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"
VNTL_LLAMA3_MODEL_DIR = "mtl/vntl-llama3-8b-v2-gguf"
VNTL_LLAMA3_MODEL_FILE = "vntl-llama3-8b-v2-hf-q8_0.gguf"

# Model metadata shown to users on the setup page.
#
# Fill in any "Not specified" values with the exact figures/details you want
# users to see. Keeping these values here makes adding another MTL model easy.
#
# `cpu_ram_mib` and `gpu_vram_mib` are the measured peak footprint of a loaded
# model. They are the figures the loader gates on (see
# fox_reader.translate.local_mtl), so they are recorded as numbers; the
# `cpu_ram` / `gpu_vram` strings the setup page prints are derived from them at
# the bottom of this file rather than written out, so the two cannot drift.
MTL_MODELS: list[dict] = [
    {
        "id": "gemma-4-e4b-q8-uncensored",
        "name": "Gemma 4 E4B Q8 Uncensored (Aggressive)",
        "repo": "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
        "dest": GEMMA_E4B_Q8_MODEL_DIR,
        "files": [GEMMA_E4B_Q8_MODEL_FILE],
        "type": "gguf",
        "size_on_disk": "7.7 GiB",
        "cpu_ram_mib": 7000,
        "gpu_vram_mib": 6800,
        "languages": ["japanese", "chinese", "korean"],
        "note": "Not specified",
    },
    {
        "id": "gemma-4-e4b-q6-uncensored",
        "name": "Gemma 4 E4B Q6 Uncensored (Aggressive)",
        "repo": "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
        "dest": GEMMA_E4B_Q6_MODEL_DIR,
        "files": [GEMMA_E4B_Q6_MODEL_FILE],
        "type": "gguf",
        "size_on_disk": "5.9 GiB",
        "cpu_ram_mib": 6000,
        "gpu_vram_mib": 5800,
        "languages": ["japanese", "chinese", "korean"],
        "note": "Not specified",
    },
    {
        "id": "vntl-llama3-8b-v2",
        "name": "VNTL Llama3 8B v2 Q8 (context + characters)",
        "repo": "lmg-anon/vntl-llama3-8b-v2-gguf",
        "dest": VNTL_LLAMA3_MODEL_DIR,
        "files": [VNTL_LLAMA3_MODEL_FILE],
        "type": "gguf",
        "size_on_disk": "8.5 GiB",
        "cpu_ram_mib": 9000,
        "gpu_vram_mib": 8800,
        "languages": ["japanese"],
        "note": "Japanese-English VN dialogue; uses conversation context and character metadata when provided.",
    },
]

def _gib(mib: int | None) -> str:
    """A MiB figure as the GiB string the setup page shows."""
    if not mib:
        return "Not specified"

    return f"{mib / 1024:.1f} GiB"


for _model in MTL_MODELS:
    _model.setdefault("cpu_ram", _gib(_model.get("cpu_ram_mib")))
    _model.setdefault("gpu_vram", _gib(_model.get("gpu_vram_mib")))

del _model


def mtl_model(model_id: str) -> dict | None:
    """The MTL_MODELS entry with this id, or None when there is no such model."""
    return next(
        (model for model in MTL_MODELS if model.get("id") == model_id),
        None,
    )


#: Written into a model's destination directory only after every file has
#: landed, so this marker -- not the directory, which exists as soon as the
#: first file starts -- is what says a model is usable.
MTL_COMPLETION_MARKER = "_completed"


def model_downloaded(base_dir: Path | str, model: dict | None) -> bool:
    """Whether every file of `model` has landed under `base_dir`.

    The one definition of "downloaded", so the setup page, the settings page,
    the startup capability checks and the download worker cannot disagree.

    The marker alone is not enough: when a model's file list changes (e.g. a
    weight conversion renames the file), a stale marker from the previous file
    set must not read as downloaded -- the app would then try to load a file
    that is not there instead of fetching it. So a model naming files
    additionally requires each of them on disk.

    ``files`` doubles as a suffix filter when the downloader asks the hub what
    to fetch (see ``_discover_model_files``), so an entry may be a whole
    repo-relative path or just an extension. Exact paths are checked with one
    stat each -- the case every shipped model is in -- and only an entry that
    does not resolve falls back to scanning for something ending in it, which
    keeps the two readings of ``files`` from disagreeing.
    """
    if not model:
        return False

    dest = model.get("dest")

    if not dest:
        return False

    model_dir = Path(base_dir) / dest

    if not (model_dir / MTL_COMPLETION_MARKER).exists():
        return False

    wanted = [str(name) for name in model.get("files") or () if name]
    missing = [name for name in wanted if not (model_dir / name).is_file()]

    if not missing:
        return True

    try:
        present = [
            path.relative_to(model_dir).as_posix()
            for path in model_dir.rglob("*")
            if path.is_file()
        ]
    except OSError:
        return False

    return all(
        any(path.endswith(name) for path in present)
        for name in missing
    )


def mtl_model_downloaded(mtl_dir: Path | str, model: dict | str) -> bool:
    """Whether this MTL model's weights are on disk and finished downloading.

    Takes either an MTL_MODELS entry or an id.
    """
    entry = mtl_model(model) if isinstance(model, str) else model

    return model_downloaded(mtl_dir, entry)


def mtl_models_for_language(language: str) -> list[dict]:
    """Every MTL model that can translate `language`, in MTL_MODELS order.

    The order is the preference order: whoever needs to choose a model for a
    language without being told which one takes the first entry that is
    downloaded.
    """
    wanted = str(language or "").strip().lower()

    return [
        model
        for model in MTL_MODELS
        if wanted
        and wanted in {str(item).strip().lower() for item in model.get("languages") or ()}
    ]


#: Every non-MTL model Fox Reader can fetch, grouped by the category the setup
#: page shows them under.
#:
#: ``isRequired`` means one thing only: the app refuses to start without it.
#: That is the PaddleOCR trio and nothing else -- OCR is the floor every other
#: feature stands on, so there is no useful app without it. Bubble detection and
#: Text Segmentation are *capabilities*: each one's absence switches off exactly
#: its own control (the Bubble Capture button, the Text Seg clean method) and
#: leaves the rest of the app working, so making them block first launch only
#: cost users ~1 GiB they may never use.
#:
#: ``id`` is the stable identity: progress keys, per-model licence acceptance
#: and the startup capability checks all address models by it. It must never be
#: reused for different weights, because an acceptance record keyed by it would
#: carry over to them.
HF_BASE_MODELS: dict[str, list[dict]] = {
    "bubble": [
        {
            "id": "bubble-segmentation",
            "name": "Speech Bubble Segmentation",
            "repo": "iamokish/manga-bubble-segmentation-pytorch",
            "dest": BUBBLE_MODEL_DIR,
            "files": [BUBBLE_MODEL_FILE],
            "isRequired": False,
            "size_on_disk": "Not specified",
            "note": "Powers the Bubble Capture button. Without it that button is hidden.",
        }
    ],
    "paddleocr": [
        {
            "id": "ppocrv6-det",
            "name": "PaddleOCRv6 Detection",
            "repo": "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
            "dest": "paddleocr/PP-OCRv6_medium_det_safetensors",
            "isRequired": True,
        },
        {
            "id": "ppocrv6-rec",
            "name": "PaddleOCRv6 Recognition",
            "repo": "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
            "dest": "paddleocr/PP-OCRv6_medium_rec_safetensors",
            "isRequired": True,
        },
        {
            "id": "ppocrv5-rec-korean",
            "name": "PaddleOCRv5 Recognition [Korean]",
            "repo": "PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors",
            "dest": "paddleocr/korean_PP-OCRv5_mobile_rec_safetensors",
            "isRequired": True,
        },
    ],
    "misc": [
        {
            "id": "text-segmentation",
            "name": "Text Segmentation",
            "repo": "iamokish/manga-text-segmentation-safetensors",
            "dest": TEXT_SEG_MODEL_DIR,
            "isRequired": False,
            "size_on_disk": "Not specified",
            "note": "Adds the Text Seg method to Background text clean. Without it that method stays greyed out.",
        },
    ],
    # Optional OCR engine. ``isRequired`` is False on purpose: the app must
    # start and run the classic PaddleOCR pipeline without these weights, and
    # the VL engine is only offered once both GGUFs have landed (see
    # fox_reader.ocr_vl.vl_model_downloaded). The chat template rides along
    # for bundle completeness; installs predating it fetch it lazily (see
    # fox_reader.ocr_vl.ensure_chat_template).
    "paddleocr_vl": [
        {
            "id": "paddleocr-vl-1.6",
            "name": "PaddleOCR-VL-1.6 (GGUF)",
            "repo": PADDLEOCR_VL_REPO,
            "dest": PADDLEOCR_VL_MODEL_DIR,
            "files": [PADDLEOCR_VL_MODEL_FILE, PADDLEOCR_VL_MMPROJ_FILE, PADDLEOCR_VL_CHAT_TEMPLATE_FILE],
            "isRequired": False,
            "type": "gguf",
            "size_on_disk": "1.8 GiB",
            "note": "Vision-language OCR via llama.cpp. Needs the gguf extra.",
        },
    ],
    # MTL is intentionally kept separate from the required model list.
    "mtl": MTL_MODELS,
}

#: Every model by id, built once. Two models sharing an id would make licence
#: acceptance ambiguous, so that is a hard error at import rather than a
#: mystery at download time.
BASE_MODEL_INDEX: dict[str, dict] = {}

for _category, _models in HF_BASE_MODELS.items():
    for _entry in _models:
        _entry_id = _entry.get("id")

        if not _entry_id:
            raise RuntimeError(f"Model in category {_category!r} has no id: {_entry.get('name')!r}")

        if _entry_id in BASE_MODEL_INDEX:
            raise RuntimeError(f"Duplicate model id: {_entry_id!r}")

        # Stamped on the entry rather than looked up later: whoever holds a model
        # often needs to know which group it belongs to (the setup page groups by
        # it), and a reverse index would be a second thing to keep in step.
        _entry.setdefault("category", _category)
        BASE_MODEL_INDEX[_entry_id] = _entry

del _category, _models, _entry, _entry_id


def base_model(model_id: str) -> dict | None:
    """Any model entry by id -- required, optional or MTL -- or None."""
    return BASE_MODEL_INDEX.get(model_id)


def base_model_downloaded(model_id: str) -> bool:
    """Whether a *non-MTL* model's weights are on disk under MODELS_DIR.

    The startup capability check: whether to build BubbleService, whether the
    Bubble Capture button is worth rendering. MTL models live under the
    configurable MTL directory instead -- use `mtl_model_downloaded` for those,
    which takes that directory explicitly.
    """
    return model_downloaded(MODELS_DIR, base_model(model_id))
