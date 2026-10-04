"""Per-model Terms of Use and licence notices, and the record of accepting them.

Why this module exists: NOTICE and TERMS.md state every model's terms, but a
document shipped beside the launcher is not a decision anyone made. Weights are
fetched from third-party vendors on the user's behalf, and some of them carry
real restrictions -- academic-only training-data lineage, Google's Gemma Terms,
Meta's Llama 3 Community License. So each model's terms are shown, and accepted,
before its download is allowed to start. The acceptance is recorded here, server
side, and :mod:`fox_reader.routes.setup` refuses to download anything whose
terms have not been accepted.

Two deliberate choices:

* **The text lives in Python, not in a file.** ``packaging/embed_assets.py``
  embeds ``frontend/`` into the binary; ``packaging/build.py`` copies NOTICE and
  TERMS.md to the *archive root*. A compiled build therefore cannot read either
  one, so a notice that loaded them would be empty exactly where it matters.
  Compiled code is the only thing guaranteed to be there.

* **The version is the content hash.** Editing a notice changes its version,
  which retires the old acceptance automatically. Nobody has to remember to bump
  a number, and an acceptance can never refer to text the user did not see.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: Bumped only if the *file format* changes. The per-document content hash is
#: what retires acceptances when the text changes.
STORE_VERSION = 1
STORE_FILENAME = "model_terms.json"


# ── Shared notice blocks ──────────────────────────────────────────────────────
# Several models share a lineage, so the blocks are written once and referenced
# by every document that needs them. One copy also means one place to correct.

_APACHE_2 = {
    "heading": "Weights licence — Apache License 2.0",
    "paragraphs": [
        "The publisher releases these weights under the Apache License 2.0: you "
        "may use, modify and redistribute them, including commercially, provided "
        "you keep the licence, copyright and attribution notices intact and state "
        "any changes you make.",
        "Apache 2.0 is offered without warranty or condition of any kind. The "
        "publisher is not liable for what the weights produce.",
    ],
    "bullets": [],
}

_NO_DATASET = {
    "heading": "Fox Reader has never touched any dataset",
    "paragraphs": [
        "Fox Reader ships no manga, no datasets and no weights. It did not train "
        "these weights, has no copy of the data behind them, and downloads none "
        "of that data. This download is an agreement between you and the model's "
        "publisher; Fox Reader only performs the transfer you asked for.",
        "The training-data terms below are reproduced because they travel with "
        "the weights and therefore reach you. Fox Reader neither grants nor can "
        "grant rights it does not hold.",
    ],
    "bullets": [],
}

_MANGA109 = {
    "heading": "Training data — Manga109 (academic, non-commercial)",
    "paragraphs": [
        "Permission to use Manga109 was granted by the authors of each work in "
        "it. The dataset may be used solely for academic purposes.",
    ],
    "bullets": [
        "Permitted uses: running experiments; printing the works as part of an "
        "academic paper; recording such a paper to a digital library; use within "
        "digital media such as demo videos presenting academic results.",
        "Redistribution of any part of the dataset to third parties is "
        "forbidden.",
        "Attribution: note the author's permission as “courtesy of [Author's "
        "Name]” (or “© [Author's Name]”), note that the work "
        "was cited from Manga109, and cite the related papers.",
        "Proper use: academic purposes by non-commercial organizations. The data "
        "shall not be transferred to a third party, and the author's name must be "
        "included as above.",
        "Disclaimer: the distributors and the manga authors are not liable for "
        "any claim or damages arising from use of the dataset.",
    ],
}

_MANGA109_S = {
    "heading": "Training data — Manga109-s (87 books, commercial use of results)",
    "paragraphs": [
        "Manga109-s is the 87-book subset available for commercial use. Its "
        "conditions are narrower than they first look: what may be used "
        "commercially is the *results*, not the images.",
    ],
    "bullets": [
        "Permitted uses: machine-learning and image-processing experiments; "
        "printing images in an academic paper; recording such a paper to a "
        "digital library; use in academic demo videos and other digital media; "
        "commercial use of results or portions of results.",
        "Redistribution of any part of the dataset to third parties is "
        "forbidden.",
        "When publishing results — including pre-trained models — the "
        "use of Manga109-s must be indicated clearly.",
        "Selling manga images together with results is forbidden.",
        "Direct copies or modifications of the images must not be treated as "
        "products, whether free or paid.",
        "For all of the uses above, when publishing whole pages (or "
        "modifications of whole pages), the total must not exceed 20% of the "
        "entire volume, per volume.",
        "Attribution and disclaimer are as for Manga109, naming Manga109-s.",
    ],
}

_MS92 = {
    "heading": "Training data — MS92/MangaSegmentation",
    "paragraphs": [
        "A custom licence behind a gated download. The images are Copyrighted by "
        "Minshan Xie. Academic *and* commercial use are permitted, provided the "
        "credit “Copyrighted by Minshan Xie” appears in any "
        "publication, reproduction, redistribution or derivative of the images.",
    ],
    "bullets": [],
}

_MANGA109_COMMERCIAL_WARNING = {
    "heading": "What this means for commercial use",
    "paragraphs": [
        "A paid or commercial Fox Reader offering must exclude the "
        "Manga109-lineage weights unless you obtain separate rights. Running "
        "them yourself on manga you hold rights to is a different question from "
        "shipping them inside a product you sell — and the second one is "
        "the restricted one.",
        "See NOTICE §5. A commercial licence for Fox Reader's own code never "
        "covers third-party weights, data, fonts or libraries.",
    ],
    "bullets": [],
}

_YOUR_RESPONSIBILITY = {
    "heading": "Your side of this",
    "paragraphs": [
        "Fox Reader does not inspect your pages, verify ownership, meter how much "
        "of a volume you publish, or phone home. Nothing in the pipeline gates on "
        "any of the above. These terms are stated, not enforced — "
        "compliance is yours (TERMS.md §§3–4).",
    ],
    "bullets": [],
}

_PADDLE_LINK = {
    "label": "PaddleOCR (PaddlePaddle)",
    "url": "https://github.com/PaddlePaddle/PaddleOCR",
}
_MANGA109_LINK = {
    "label": "Manga109 project — terms and citations",
    "url": "https://manga109.github.io/manga109-project-website/en/index.html",
}
_MS92_LINK = {
    "label": "MS92/MangaSegmentation dataset card",
    "url": "https://huggingface.co/datasets/MS92/MangaSegmentation",
}


def _paddle_doc(model_id: str, name: str, repo: str, extra: list[dict] | None = None) -> dict:
    """One of the Apache-2.0 PaddleOCR documents.

    The three OCR weights differ only by name and repository, so writing the
    notice out three times would be three places for it to drift.
    """
    return {
        "id": model_id,
        "name": name,
        "licence": "Apache License 2.0 — no additional restriction",
        "lede": (
            "PaddleOCR weights published by PaddlePaddle under the Apache License "
            "2.0. This is the most permissive model in Fox Reader: no "
            "academic-only clause, no acceptable-use policy, no restriction on "
            "commercial use."
        ),
        "sections": [_APACHE_2, *(extra or []), _YOUR_RESPONSIBILITY],
        "links": [
            {"label": f"Model card — {repo}", "url": f"https://huggingface.co/{repo}"},
            _PADDLE_LINK,
        ],
        "accept_label": (
            "I have read the Apache 2.0 terms above and accept them for this model."
        ),
    }


# ── Documents ─────────────────────────────────────────────────────────────────
# Keyed by the stable model id in fox_reader.constants.HF_BASE_MODELS. A model
# with no document here cannot be downloaded (see `document` below): failing
# closed is the only safe direction for a licence gate.

_DOCUMENTS: tuple[dict, ...] = (
    _paddle_doc(
        "ppocrv6-det",
        "PaddleOCRv6 Detection",
        "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
    ),
    _paddle_doc(
        "ppocrv6-rec",
        "PaddleOCRv6 Recognition",
        "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
    ),
    _paddle_doc(
        "ppocrv5-rec-korean",
        "PaddleOCRv5 Recognition [Korean]",
        "PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors",
    ),
    _paddle_doc(
        "paddleocr-vl-1.6",
        "PaddleOCR-VL-1.6 (GGUF)",
        "PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
        extra=[{
            "heading": "Runtime note",
            "paragraphs": [
                "These GGUFs run through llama.cpp, which Fox Reader only has "
                "when it was installed with the ``gguf`` extra. Without it the "
                "weights download and sit unused until the extra is installed; "
                "the classic PaddleOCR pipeline keeps working either way.",
            ],
            "bullets": [],
        }],
    ),
    {
        "id": "bubble-segmentation",
        "name": "Speech Bubble Segmentation",
        "licence": (
            "Apache 2.0 (weights) · Manga109 / Manga109-s / MS92 "
            "(training data) — read both"
        ),
        "lede": (
            "The weights card declares Apache 2.0, but these weights were trained "
            "on MS92/MangaSegmentation and Manga109. The Manga109 lineage is "
            "academic and non-commercial, and a permissive weights licence does "
            "not lift a restriction on the data behind it. Read the dataset terms "
            "before assuming commercial rights."
        ),
        "sections": [
            _APACHE_2,
            _NO_DATASET,
            _MS92,
            _MANGA109,
            _MANGA109_S,
            _MANGA109_COMMERCIAL_WARNING,
            _YOUR_RESPONSIBILITY,
        ],
        "links": [
            {
                "label": "Model card — iamokish/manga-bubble-segmentation-pytorch",
                "url": "https://huggingface.co/iamokish/manga-bubble-segmentation-pytorch",
            },
            _MS92_LINK,
            _MANGA109_LINK,
        ],
        "accept_label": (
            "I have read the Apache 2.0, MS92 and Manga109 / Manga109-s terms "
            "above and accept them for this model."
        ),
    },
    {
        "id": "text-segmentation",
        "name": "Text Segmentation",
        "licence": (
            "MIT (research code) · weights declare no licence — treat as "
            "research / academic only"
        ),
        "lede": (
            "The sharpest edge of any model here. The research code is MIT "
            "(juvian/Manga-Text-Segmentation, del Gobbo & Matuk Herrera, ECCV 2020 "
            "Workshops), but the weight file itself declares no licence upstream. "
            "Until the publisher says otherwise, treat these weights as "
            "research/academic only — and the Manga109 lineage below applies "
            "to them as well."
        ),
        "sections": [
            {
                "heading": "Research code — MIT License",
                "paragraphs": [
                    "The official implementation is MIT-licensed: permissive, with "
                    "the copyright and permission notice to be kept. The label "
                    "masks are published via Zenodo (doi:10.5281/zenodo.4511796).",
                ],
                "bullets": [],
            },
            {
                "heading": "The weight file declares no licence",
                "paragraphs": [
                    "An MIT research repository does not make its published "
                    "checkpoints MIT. This weight file carries no licence "
                    "statement at all, so there is no permission to rely on "
                    "beyond research use. Do not build a product on it without "
                    "asking the publisher.",
                ],
                "bullets": [],
            },
            _NO_DATASET,
            _MANGA109,
            _MANGA109_S,
            _MANGA109_COMMERCIAL_WARNING,
            {
                "heading": "Citation",
                "paragraphs": [
                    "Academic or public write-ups of results from these weights "
                    "must cite del Gobbo & Matuk Herrera, ECCV 2020 Workshops "
                    "(doi:10.5281/zenodo.4511796), and the Manga109 papers listed "
                    "in NOTICE §6.",
                ],
                "bullets": [],
            },
            _YOUR_RESPONSIBILITY,
        ],
        "links": [
            {
                "label": "Model card — iamokish/manga-text-segmentation-safetensors",
                "url": "https://huggingface.co/iamokish/manga-text-segmentation-safetensors",
            },
            {
                "label": "Research code — juvian/Manga-Text-Segmentation (MIT)",
                "url": "https://github.com/juvian/Manga-Text-Segmentation",
            },
            _MANGA109_LINK,
        ],
        "accept_label": (
            "I understand these weights carry no licence of their own, accept the "
            "Manga109 / Manga109-s terms, and will treat them as research only."
        ),
    },
    {
        "id": "gemma-4-e4b-q8-uncensored",
        "name": "Gemma 4 E4B Q8 Uncensored (Aggressive)",
        "licence": "Google Gemma Terms of Use — custom, acceptance required",
        "lede": (
            "Gemma weights are not open-source-licensed. They are distributed "
            "under Google's custom Gemma Terms of Use, which you must accept, "
            "which forbid certain uses outright, and which you must pass on to "
            "anyone you redistribute the weights to."
        ),
        "sections": [
            {
                "heading": "Gemma Terms of Use",
                "paragraphs": [
                    "Use of these weights is governed by Google's Gemma Terms of "
                    "Use together with the Gemma Prohibited Use Policy. Read both "
                    "before downloading — they are the licence, not a summary "
                    "of one.",
                ],
                "bullets": [
                    "Acceptance is required: downloading and using the weights "
                    "means agreeing to the Terms.",
                    "The Prohibited Use Policy applies and restricts what you may "
                    "generate.",
                    "Any distribution you make of these weights, or of a model "
                    "derived from them, must carry the same Terms downstream.",
                ],
            },
            {
                "heading": "“Uncensored” removes no obligation",
                "paragraphs": [
                    "This is a third-party fine-tune with safety behaviour "
                    "removed. Removing a model's refusals does not remove a single "
                    "term it is licensed under: the Gemma Terms and the Prohibited "
                    "Use Policy apply in full, and the model's willingness to "
                    "produce something is not permission to produce it.",
                    "Fox Reader's own prohibited uses apply on top (TERMS.md "
                    "§6), including illegal content — CSAM is reported.",
                ],
                "bullets": [],
            },
            _YOUR_RESPONSIBILITY,
        ],
        "links": [
            {"label": "Gemma Terms of Use", "url": "https://ai.google.dev/gemma/terms"},
            {
                "label": "Gemma Prohibited Use Policy",
                "url": "https://ai.google.dev/gemma/prohibited_use_policy",
            },
            {
                "label": "Model card — HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
                "url": "https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
            },
        ],
        "accept_label": (
            "I accept Google's Gemma Terms of Use and Prohibited Use Policy for "
            "these weights."
        ),
    },
    {
        "id": "vntl-llama3-8b-v2",
        "name": "VNTL Llama3 8B v2 Q8",
        "licence": "Meta Llama 3 Community License — custom, conditions apply",
        "lede": (
            "A Llama 3 derivative, so Meta's Llama 3 Community License travels "
            "with it. Commercial use is generally allowed, but attribution, the "
            "Acceptable Use Policy and a monthly-active-user threshold all attach."
        ),
        "sections": [
            {
                "heading": "Llama 3 Community License",
                "paragraphs": [
                    "Not an OSI-approved open-source licence. The obligations that "
                    "catch people out:",
                ],
                "bullets": [
                    "Attribution: a distribution built on these weights must state "
                    "it is “Built with Meta Llama 3”, and a derivative "
                    "model's name must begin with “Llama 3”.",
                    "The Llama 3 Acceptable Use Policy applies to what you "
                    "generate.",
                    "Redistribution must carry the licence and the use "
                    "restrictions with it.",
                    "Above 700 million monthly active users you need a separate "
                    "licence from Meta before using these weights.",
                ],
            },
            {
                "heading": "What this model is for",
                "paragraphs": [
                    "A Japanese→English visual-novel dialogue translator. It "
                    "reads conversation context and character metadata when Fox "
                    "Reader provides them. Machine translation is often wrong — "
                    "verify anything that matters (TERMS.md §8).",
                ],
                "bullets": [],
            },
            _YOUR_RESPONSIBILITY,
        ],
        "links": [
            {
                "label": "Meta Llama 3 Community License",
                "url": "https://llama.meta.com/llama3/license/",
            },
            {
                "label": "Llama 3 Acceptable Use Policy",
                "url": "https://llama.meta.com/llama3/use-policy/",
            },
            {
                "label": "Model card — lmg-anon/vntl-llama3-8b-v2-gguf",
                "url": "https://huggingface.co/lmg-anon/vntl-llama3-8b-v2-gguf",
            },
        ],
        "accept_label": (
            "I accept the Meta Llama 3 Community License and Acceptable Use "
            "Policy for these weights."
        ),
    },
)

# The Q6 Gemma is the same weights at a different quantisation, so it carries
# the same notice under its own id -- acceptance is per model, and these are two
# downloads.
_GEMMA_Q6 = {
    **next(doc for doc in _DOCUMENTS if doc["id"] == "gemma-4-e4b-q8-uncensored"),
    "id": "gemma-4-e4b-q6-uncensored",
    "name": "Gemma 4 E4B Q6 Uncensored (Aggressive)",
}


def _fingerprint(document: dict) -> str:
    """A short stable hash of everything the user is shown.

    Keyed by content, so editing a notice retires every acceptance of the old
    text without anyone having to bump a version by hand. ``sort_keys`` makes it
    independent of dict ordering; the id is excluded because it is identity, not
    content.
    """
    payload = {key: value for key, value in document.items() if key != "version"}

    return blake2b(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8"),
        digest_size=8,
    ).hexdigest()


#: Every document by model id, with its content-hash version attached. Built
#: once at import: the text is static, and hashing it per request would be work
#: for no reason.
DOCUMENTS: dict[str, dict] = {}

for _doc in (*_DOCUMENTS, _GEMMA_Q6):
    DOCUMENTS[_doc["id"]] = {**_doc, "version": _fingerprint(_doc)}

del _doc


# Every downloadable model needs a notice, or the licence gate makes it
# permanently undownloadable -- a confusing way to find out that adding a model
# meant adding its terms too. Caught here, at first import, rather than by a
# user staring at a button that will not work. Imported late so the module-level
# documents above are already built.
from fox_reader.constants import BASE_MODEL_INDEX  # noqa: E402  (cycle-free: constants never imports this module)

if _missing := sorted(set(BASE_MODEL_INDEX) - set(DOCUMENTS)):
    raise RuntimeError(
        "Models with no terms notice (add one to fox_reader.model_terms): "
        + ", ".join(_missing)
    )

del _missing


def document(model_id: str) -> dict | None:
    """The notice for this model id, or None when there is none."""
    return DOCUMENTS.get(model_id)


def version(model_id: str) -> str:
    """This notice's content version, or "" when the model has no notice."""
    doc = DOCUMENTS.get(model_id)

    return str(doc["version"]) if doc else ""


# ── Acceptance record ─────────────────────────────────────────────────────────

class TermsStore:
    """Which model notices this installation has accepted, and at what version.

    One small JSON file beside the rest of the user's configuration. Server side
    on purpose: acceptance gates a server action (fetching weights), so checking
    it in the browser would be decoration. It also survives a cleared browser
    profile, which an acceptance of licence terms should.

    Reads are served from memory after the first load; writes are atomic
    (temp file plus ``os.replace``) so an interrupted write cannot leave a
    truncated record that reads as "nothing accepted".
    """

    def __init__(self, config_dir: Path | str) -> None:
        self._path = Path(config_dir) / STORE_FILENAME
        self._lock = threading.RLock()
        self._accepted: dict[str, str] | None = None

    # -- internals --

    def _load_locked(self) -> dict[str, str]:
        if self._accepted is not None:
            return self._accepted

        accepted: dict[str, str] = {}

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = None
        except (OSError, ValueError) as exc:
            # A corrupt record must not read as "everything accepted", and must
            # not crash setup either. Treating it as empty re-asks, which is the
            # safe direction.
            logger.warning("Unreadable model-terms record (%s): re-asking.", exc)
            raw = None

        if isinstance(raw, dict):
            entries = raw.get("accepted")

            if isinstance(entries, dict):
                for model_id, entry in entries.items():
                    recorded = (
                        entry.get("version") if isinstance(entry, dict) else entry
                    )

                    if isinstance(model_id, str) and isinstance(recorded, str):
                        accepted[model_id] = recorded

        self._accepted = accepted

        return accepted

    def _write_locked(self, accepted: dict[str, str], stamps: dict[str, str]) -> None:
        payload = {
            "version": STORE_VERSION,
            "accepted": {
                model_id: {
                    "version": recorded,
                    "accepted_at": stamps.get(model_id, ""),
                }
                for model_id, recorded in sorted(accepted.items())
            },
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(f"{self._path.name}.tmp")

        try:
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _stamps_locked(self) -> dict[str, str]:
        """The accepted_at timestamps already on disk, so a write keeps them."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

        entries = raw.get("accepted") if isinstance(raw, dict) else None

        if not isinstance(entries, dict):
            return {}

        return {
            model_id: str(entry.get("accepted_at") or "")
            for model_id, entry in entries.items()
            if isinstance(model_id, str) and isinstance(entry, dict)
        }

    # -- public --

    def is_accepted(self, model_id: str) -> bool:
        """Whether this model's *current* notice text has been accepted.

        A model with no notice is never accepted: a licence gate that passes
        models it knows nothing about is not a gate.
        """
        wanted = version(model_id)

        if not wanted:
            return False

        with self._lock:
            return self._load_locked().get(model_id) == wanted

    def unaccepted(self, model_ids: Iterable[str]) -> list[str]:
        """Those of `model_ids` that may not be downloaded yet, in order."""
        with self._lock:
            accepted = self._load_locked()

            return [
                model_id
                for model_id in model_ids
                if not version(model_id) or accepted.get(model_id) != version(model_id)
            ]

    def accept(self, model_ids: Iterable[str]) -> list[str]:
        """Record acceptance of each model's current notice. Returns what stuck.

        Unknown ids are ignored rather than rejected: they cannot be downloaded
        anyway (`unaccepted` reports a model with no notice as unaccepted
        forever), so failing the whole call over one would only make the page
        harder to recover.
        """
        stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        recorded: list[str] = []

        with self._lock:
            accepted = dict(self._load_locked())
            stamps = self._stamps_locked()

            for model_id in model_ids:
                wanted = version(model_id)

                if not wanted:
                    logger.warning("Ignoring acceptance of unknown model: %s", model_id)
                    continue

                accepted[model_id] = wanted
                stamps[model_id] = stamp
                recorded.append(model_id)

            if recorded:
                self._write_locked(accepted, stamps)
                self._accepted = accepted

        return recorded

    def state(self) -> dict[str, dict[str, Any]]:
        """Per-model acceptance for the setup page: ``{id: {accepted, version}}``.

        Deliberately tiny: this rides in the status payload the page polls once a
        second, so the notice text itself is fetched separately and only when a
        card is actually opened.
        """
        with self._lock:
            accepted = self._load_locked()

        return {
            model_id: {
                "accepted": accepted.get(model_id) == doc["version"],
                "version": doc["version"],
            }
            for model_id, doc in DOCUMENTS.items()
        }


_store: TermsStore | None = None
_store_lock = threading.Lock()


def store() -> TermsStore:
    """The process-wide acceptance record, created on first use."""
    global _store

    with _store_lock:
        if _store is None:
            from fox_reader.utils import CONFIG_DIR

            _store = TermsStore(CONFIG_DIR)

        return _store


# Module-level shorthands for the process-wide store. Call sites read better for
# it, and nothing outside this module has to know the singleton exists.

def is_accepted(model_id: str) -> bool:
    """Whether this model's current notice has been accepted."""
    return store().is_accepted(model_id)


def unaccepted(model_ids: Iterable[str]) -> list[str]:
    """Those of `model_ids` that may not be downloaded yet, in order."""
    return store().unaccepted(model_ids)


def accept(model_ids: Iterable[str]) -> list[str]:
    """Record acceptance of each model's current notice. Returns what stuck."""
    return store().accept(model_ids)


def state() -> dict[str, dict[str, Any]]:
    """Per-model acceptance for the setup page: ``{id: {accepted, version}}``."""
    return store().state()
