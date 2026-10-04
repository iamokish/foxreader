"""Setup page routes — first-launch model download wizard."""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from fox_reader import assets
from fox_reader.config import FoxConfig
from fox_reader.utils import MODELS_DIR, PROJECT_ROOT

router = APIRouter(tags=["setup"])
templates = assets.templates()

# ── Download state ────────────────────────────────────────────────────────────

_download_state: dict[str, Any] = {
    "active": False,
    "stage": "non_mtl",  # non_mtl -> mtl_selection -> mtl -> complete
    "current_model": None,
    "progress": {},  # model id -> progress
    "error": None,
    "complete": False,
}

_download_lock = threading.Lock()
_incomplete_models: list[str] = []
_HF_MODELS: dict[str, list[dict]] = {}
_MTL_DIR = Path("models")
CATEGORY_NAME_MAP: dict[str, str] = {
    "bubble": "Bubble Detection",
    "paddleocr": "Text Detection & Recognition",
    "misc": "Miscellaneous",
    "paddleocr_vl": "OCR Vision-Language (optional)",
}

# Optional model downloads run beside the wizard, never inside it: the wizard's
# stage machine (non_mtl -> mtl_selection -> mtl -> complete) must not gain a
# branch for weights nobody is required to fetch. Bubble segmentation, Text
# Segmentation and PaddleOCR-VL all come through here -- each one switches on
# exactly one capability and the app runs without any of them. Both jobs share
# the worker in `_run_model_downloads` below; only the state dicts differ.
_optional_state: dict[str, Any] = {
    "active": False,
    "current_model": None,
    # model id -> {done, files_done, files_total, current_file, downloaded,
    #              bytes_done, bytes_total, current_bytes, current_total}.
    # The byte fields are new: file counts alone leave a ~900 MiB GGUF sitting
    # at "0/2 files" for minutes. Old readers ignore the extra keys.
    "progress": {},
    "error": None,
}
_optional_lock = threading.Lock()

#: Which download (job, model, file) a hub progress bar belongs to. Set by the
#: worker around each `hf_hub_download` call; the bar itself only sees bytes.
_dl_ctx = threading.local()


def _safe_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


class _HubByteProgress:
    """A tqdm-compatible sink forwarding hub byte progress into setup state.

    Passed as ``tqdm_class`` to ``hf_hub_download`` so the battle-tested hub
    machinery (resume, retries, Xet) stays untouched while the setup page gets
    real bytes instead of file counts. Only the interface matters here —
    ``__init__(total, initial, desc)``, ``update(n)``, ``close()`` and the
    context-manager protocol — so a hub upgrade that stops calling it degrades
    to file-count progress rather than breaking the download.

    Never raises and never prints: this runs on the download thread, where an
    exception would abort the file and stdout goes nowhere useful.
    """

    def __init__(self, total: Any = None, initial: Any = 0, desc: Any = None, **_kwargs: Any) -> None:
        self._total = _safe_int(total)
        self._count = _safe_int(initial)
        ctx = getattr(_dl_ctx, "current", None)
        self._ctx = dict(ctx) if isinstance(ctx, dict) else None
        if self._ctx is not None:
            try:
                _progress_file_started(self._ctx, self._total, self._count, desc)
            except Exception:
                pass

    def update(self, n: Any) -> None:
        try:
            step = int(n or 0)
        except (TypeError, ValueError):
            return
        self._count = max(0, self._count + step)
        if self._ctx is not None:
            try:
                _progress_file_advanced(self._ctx, self._count, self._total)
            except Exception:
                pass

    def close(self) -> None:
        if self._ctx is not None:
            try:
                _progress_file_advanced(self._ctx, self._count, self._total)
            except Exception:
                pass

    def __enter__(self) -> _HubByteProgress:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _progress_entry(state: dict[str, Any], key: str) -> dict[str, Any]:
    return state["progress"].setdefault(key, {
        "done": False,
        "files_done": 0,
        "files_total": 0,
        "current_file": "",
        "downloaded": False,
        "bytes_done": 0,
        "bytes_total": 0,
        "current_bytes": 0,
        "current_total": 0,
    })


def _progress_file_started(ctx: dict[str, Any], total: int, initial: int, desc: Any) -> None:
    """A hub bar opened for the worker's current file: publish its total."""
    state, lock, key = ctx["state"], ctx["lock"], ctx["key"]
    with lock:
        entry = _progress_entry(state, key)
        entry["current_total"] = total
        entry["current_bytes"] = min(initial, total) if total else initial
        entry["bytes_total"] = _safe_int(entry.get("bytes_base", 0)) + total
        entry["bytes_done"] = _safe_int(entry.get("bytes_base", 0)) + entry["current_bytes"]
        if isinstance(desc, str) and desc and not entry.get("current_file"):
            entry["current_file"] = desc


def _progress_file_advanced(ctx: dict[str, Any], count: int, total: int) -> None:
    """A hub bar moved: publish bytes without touching file counts."""
    state, lock, key = ctx["state"], ctx["lock"], ctx["key"]
    with lock:
        entry = _progress_entry(state, key)
        if total:
            entry["current_total"] = total
            entry["bytes_total"] = _safe_int(entry.get("bytes_base", 0)) + total
        entry["current_bytes"] = count
        entry["bytes_done"] = _safe_int(entry.get("bytes_base", 0)) + count

_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_INVALID_CHARS = re.compile(r'[<>:"|?*\x00]')

# ── Helpers ───────────────────────────────────────────────────────────────────

def _required_non_mtl_categories() -> list[str]:
    """Categories the app refuses to start without: the PaddleOCR trio.

    Derived from ``isRequired``, so flipping a model there is the only edit
    needed to move it between required and optional.
    """
    from fox_reader.constants import HF_BASE_MODELS

    return [
        category
        for category, models in HF_BASE_MODELS.items()
        if category != "mtl" and any(model.get("isRequired", False) for model in models)
    ]


def _optional_non_mtl_categories() -> list[str]:
    """The non-MTL categories the app runs happily without.

    Bubble segmentation, Text Segmentation and PaddleOCR-VL. Each absence
    switches off one control and nothing else, so they are offered rather than
    demanded -- and the complement of the required set, so a category can never
    fall into neither list or both.
    """
    from fox_reader.constants import HF_BASE_MODELS

    required = set(_required_non_mtl_categories())

    return [
        category
        for category in HF_BASE_MODELS
        if category != "mtl" and category not in required
    ]


def _model_completed(base_dir: Path, model: dict) -> bool:
    """Whether every file of `model` has landed under `base_dir`.

    A thin alias for the one definition in `fox_reader.constants` -- kept
    because this module calls it on every status poll and the short name reads
    better at the call sites.
    """
    from fox_reader.constants import model_downloaded

    return model_downloaded(base_dir, model)


def _serialize_required_models(accepted: dict[str, dict] | None = None) -> list[dict]:
    """Required non-MTL models with on-disk and terms-acceptance state.

    `accepted` is the store's acceptance map, passed in by callers that already
    have it so one status poll takes the store lock once rather than once per
    model.
    """
    from fox_reader.constants import HF_BASE_MODELS

    if accepted is None:
        from fox_reader import model_terms

        accepted = model_terms.state()

    result: list[dict] = []
    for category in _required_non_mtl_categories():
        models = HF_BASE_MODELS[category]
        items = []
        for model in models:
            model_id = model["id"]
            items.append({
                "id": model_id,
                "name": model["name"],
                "category": category,
                "downloaded": _model_completed(MODELS_DIR, model),
                "terms_accepted": bool(accepted.get(model_id, {}).get("accepted")),
            })
        result.append({
            "id": category,
            "name": CATEGORY_NAME_MAP[category],
            "items": items,
            "downloaded": all(item["downloaded"] for item in items),
        })
    return result


def _normalize_languages(value: Any) -> list[str]:
    """Return model languages as a lowercase list of strings."""
    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple)):
        return ["not specified"]

    languages: list[str] = []
    for language in value:
        text = str(language).strip().lower()
        if text:
            languages.append(text)

    return languages or ["not specified"]


def _serialize_mtl_model(model: dict, mtl_dir: Path, accepted: dict[str, dict]) -> dict:
    return {
        "id": model["id"],
        "name": model["name"],
        "type": model.get("type", "Not specified"),
        "size_on_disk": model.get("size_on_disk", "Not specified"),
        "cpu_ram": model.get("cpu_ram", "Not specified"),
        "gpu_vram": model.get("gpu_vram", "Not specified"),
        "languages": _normalize_languages(model.get("languages")),
        "note": model.get("note", "Not specified"),
        "downloaded": _model_completed(mtl_dir, model),
        "terms_accepted": bool(accepted.get(model["id"], {}).get("accepted")),
    }


def _all_mtl_models_downloaded(mtl_dir: Path) -> bool:
    from fox_reader.constants import MTL_MODELS

    return bool(MTL_MODELS) and all(
        _model_completed(mtl_dir, model)
        for model in MTL_MODELS
    )


def _get_mtl_models(mtl_dir: Path, accepted: dict[str, dict] | None = None) -> list[dict]:
    from fox_reader.constants import MTL_MODELS

    if accepted is None:
        from fox_reader import model_terms

        accepted = model_terms.state()

    return [_serialize_mtl_model(model, mtl_dir, accepted) for model in MTL_MODELS]


def _resolve_mtl_models(selected_ids: list[str]) -> list[dict]:
    from fox_reader.constants import MTL_MODELS

    by_id = {model["id"]: model for model in MTL_MODELS}
    selected: list[dict] = []

    for model_id in selected_ids:
        model = by_id.get(model_id)
        if model is None:
            raise ValueError(f"Unknown MTL model: {model_id}")
        selected.append(model)

    return selected


# ── Licence gate ─────────────────────────────────────────────────────────────

def _terms_refusal(models: list[dict]) -> JSONResponse | None:
    """A 403 naming every model whose terms are unaccepted, or None to proceed.

    The gate, and the only one that counts: a browser can be driven directly, so
    checking acceptance in the page would be decoration. Every route that starts
    a download calls this, which is why it takes the resolved model entries
    rather than a stage name -- there is nothing to forget to pass.

    A model with no notice at all is refused too (see
    `fox_reader.model_terms.is_accepted`). Failing closed is the only safe
    direction for a licence check, and the import-time coverage check in that
    module means this can only fire for a model someone forgot to write terms
    for -- in which case refusing is correct.
    """
    from fox_reader import model_terms

    blocked = model_terms.unaccepted([model["id"] for model in models])

    if not blocked:
        return None

    names = {model["id"]: model.get("name", model["id"]) for model in models}

    return JSONResponse(
        {
            "error": (
                "Accept the terms for "
                + ", ".join(names.get(model_id, model_id) for model_id in blocked)
                + " before downloading."
            ),
            "terms_required": blocked,
        },
        status_code=403,
    )


# ── Model completeness ───────────────────────────────────────────────────────

def check_models_exist(config: FoxConfig) -> bool:
    """Return whether all required non-MTL models are downloaded.

    MTL remains optional. When required models are ready, the normal
    application can start, while /setup can still be opened to manage MTL
    model downloads.
    """
    global _incomplete_models, _HF_MODELS, _MTL_DIR

    from fox_reader.constants import HF_BASE_MODELS

    _MTL_DIR = config.resolve_mtl_dir(PROJECT_ROOT)
    _incomplete_models = []
    _HF_MODELS = {}

    for category in _required_non_mtl_categories():
        models = HF_BASE_MODELS[category]
        category_complete = all(
            _model_completed(MODELS_DIR, model) for model in models
        )

        if not category_complete:
            _incomplete_models.append(category)
            _HF_MODELS[category] = models

    required_ready = not _incomplete_models

    with _download_lock:
        if not _download_state["active"]:
            _download_state["stage"] = (
                "mtl_selection" if required_ready else "non_mtl"
            )
            _download_state["complete"] = False
        _download_state["error"] = None

    return required_ready


def _required_models_ready() -> bool:
    from fox_reader.constants import HF_BASE_MODELS

    for category in _required_non_mtl_categories():
        for model in HF_BASE_MODELS.get(category, []):
            if not _model_completed(MODELS_DIR, model):
                return False
    return True


def _derive_stage() -> str:
    with _download_lock:
        if _download_state["active"]:
            return _download_state["stage"]

    # A direct /setup visit is also a model-management/setup page after the
    # required models are installed. The persistent setup-complete marker
    # controls normal application startup, but must not hide MTL choices here.
    if not _required_models_ready():
        return "non_mtl"

    return "mtl_selection"


# ── Download worker ───────────────────────────────────────────────────────────

def _discover_model_files(api, model: dict) -> list[str]:
    raw_files = model.get("files")
    repo_files = api.list_repo_files(model["repo"])

    files: list[str] = []
    for repo_file in repo_files:
        if repo_file.startswith("."):
            continue
        if not raw_files or repo_file.endswith(tuple(raw_files)):
            files.append(repo_file)

    return files


def _fresh_progress_entry(files_total: int) -> dict[str, Any]:
    """A progress entry with file counts and byte counters.

    ``bytes_base`` is internal (bytes of files finished this run) and is
    stripped before anything is sent to a client.
    """
    return {
        "done": False,
        "files_done": 0,
        "files_total": files_total,
        "current_file": "",
        "downloaded": False,
        "bytes_done": 0,
        "bytes_total": 0,
        "current_bytes": 0,
        "current_total": 0,
        "bytes_base": 0,
    }


def _public_progress(progress: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Progress safe to serialise: the same entries minus internal fields."""
    return {
        key: {k: v for k, v in entry.items() if k != "bytes_base"}
        for key, entry in progress.items()
    }


def _run_model_downloads(
    models_to_download: list[tuple[str, dict, Path]],
    *,
    lock: threading.Lock,
    state: dict[str, Any],
) -> None:
    """Download every ``(key, model, base_dir)`` triple, reporting progress.

    Shared by the wizard stages and the optional PaddleOCR-VL job: the only
    difference between them is which state dict is passed in. Each file goes
    through ``hf_hub_download`` with a byte-reporting ``tqdm_class`` so the
    page sees MiBs moving, not just file counts; file counts remain the
    fallback for servers that reply without a length and for hub paths that
    never instantiate the bar.

    Raises the first failure with the model named, so the route can store an
    error the page can show. The completion marker is touched only after all
    of a model's files have landed, which is what makes a re-run skip it.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    from fox_reader.constants import MTL_COMPLETION_MARKER

    # Our adapter prints nothing; this only quiets the hub's own bars on hub
    # paths that never see our tqdm_class.
    disable_progress_bars()
    api = HfApi()

    for model_key, model, base_dir in models_to_download:
        base_dir.mkdir(parents=True, exist_ok=True)
        model_dest = base_dir / model["dest"]
        completed_file = model_dest / MTL_COMPLETION_MARKER

        # Never re-download a model that has already completed. The marker
        # alone is not trusted: a stale one from a previous file set (weight
        # conversion, renamed file) must fall through and fetch what is
        # actually missing.
        if _model_completed(base_dir, model):
            with lock:
                state["current_model"] = model_key
                state["progress"][model_key] = {
                    **_fresh_progress_entry(0),
                    "done": True,
                    "downloaded": True,
                }
            continue
        try:
            completed_file.unlink(missing_ok=True)
        except OSError:
            pass

        try:
            files = _discover_model_files(api, model)
        except Exception as exc:
            raise RuntimeError(f"{model.get('name', model_key)}: could not list files ({exc})") from exc

        with lock:
            state["current_model"] = model_key
            state["progress"][model_key] = _fresh_progress_entry(len(files))

        model_dest.mkdir(parents=True, exist_ok=True)
        completed_bytes = 0

        for index, filename in enumerate(files, start=1):
            with lock:
                entry = state["progress"][model_key]
                entry["current_file"] = filename
                entry["files_done"] = index - 1
                entry["current_bytes"] = 0
                entry["current_total"] = 0
                entry["bytes_base"] = completed_bytes
                entry["bytes_done"] = completed_bytes

            _dl_ctx.current = {"state": state, "lock": lock, "key": model_key}
            try:
                hf_hub_download(
                    repo_id=model["repo"],
                    filename=filename,
                    local_dir=str(model_dest),
                    tqdm_class=_HubByteProgress,
                )
            except Exception as exc:
                raise RuntimeError(f"{model.get('name', model_key)}: {filename}: {exc}") from exc
            finally:
                _dl_ctx.current = None

            with lock:
                entry = state["progress"][model_key]
                # The bar's last count is the file's true size, even when the
                # server never sent a length up front.
                finished = max(_safe_int(entry.get("current_bytes")), _safe_int(entry.get("current_total")))
                completed_bytes += finished
                entry["files_done"] = index
                entry["bytes_base"] = completed_bytes
                entry["bytes_done"] = completed_bytes
                entry["bytes_total"] = max(_safe_int(entry.get("bytes_total")), completed_bytes)
                entry["current_file"] = ""
                entry["current_bytes"] = 0
                entry["current_total"] = 0

        completed_file.touch()

        with lock:
            entry = state["progress"][model_key]
            entry["done"] = True
            entry["downloaded"] = True
            entry["current_file"] = ""
            entry["current_bytes"] = 0
            entry["current_total"] = 0
            if entry["bytes_total"] <= 0:
                entry["bytes_total"] = entry["bytes_done"]


def _download_models_task(
    mtl_dir: Path,
    models_to_download: list[tuple[str, dict]],
    stage: str,
) -> None:
    """Background task for one setup wizard stage."""
    global _download_state

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if stage == "mtl":
        mtl_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        (key, model, mtl_dir if stage == "mtl" else MODELS_DIR)
        for key, model in models_to_download
    ]

    try:
        _run_model_downloads(entries, lock=_download_lock, state=_download_state)
    except Exception as e:
        with _download_lock:
            _download_state["error"] = str(e)
            _download_state["active"] = False
            _download_state["current_model"] = None
            _download_state["complete"] = False
        return

    with _download_lock:
        _download_state["active"] = False
        _download_state["current_model"] = None

        # Downloading models is not the same thing as finishing setup. Both
        # stages return to the selection screen so the user can see what
        # landed and either download more or explicitly finish.
        _download_state["stage"] = "mtl_selection"
        _download_state["complete"] = False


# ── Path validation ──────────────────────────────────────────────────────────

def validate_output_calc(user_path: str) -> dict[str, Path | str]:
    path = Path(user_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve(strict=False)

    for part in path.parts:
        if part in (path.anchor, "", os.sep):
            continue

        if _INVALID_CHARS.search(part):
            return {
                "status": "failed",
                "error": f"Invalid character in path component: {part!r}",
            }

        if part.endswith((" ", ".")):
            return {
                "status": "failed",
                "error": f"Invalid trailing space/dot: {part!r}",
            }

        stem = part.rstrip(" .").split(".")[0].upper()
        if stem in _WINDOWS_RESERVED:
            return {
                "status": "failed",
                "error": f"Reserved Windows filename: {part!r}",
            }

    current = Path(path.anchor)

    for part in path.parts[1:]:
        current /= part

        if not current.exists() and not current.is_symlink():
            break

        if current.is_symlink():
            return {"status": "failed", "error": f"Symlink not allowed: {current}"}

    if path.is_symlink():
        return {"status": "failed", "error": "Target path is a symlink."}

    if path.exists() and not path.is_dir():
        return {"status": "failed", "error": "Target path already exists as a file."}

    parent = path.parent

    if not parent.exists():
        return {"status": "failed", "error": "Parent directory does not exist."}

    if parent.is_symlink():
        return {"status": "failed", "error": "Parent directory is a symlink."}

    if not parent.is_dir():
        return {"status": "failed", "error": "Parent is not a directory."}

    if not os.access(parent, os.R_OK | os.W_OK | os.X_OK):
        return {
            "status": "failed",
            "error": f"Insufficient permissions for: {path}",
        }

    return {"status": "success", "path": path}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    global _MTL_DIR

    config = request.app.state.config
    _MTL_DIR = config.resolve_mtl_dir(PROJECT_ROOT)

    if not _download_state["active"]:
        _download_state["stage"] = _derive_stage()
        _download_state["complete"] = False

    return templates.TemplateResponse(
        request=request,
        name="setup.html",
        context={
            "mtl_dir": str(config.mtl_dir),
        },
    )


@router.get("/api/setup/status")
async def setup_status(request: Request):
    from fox_reader import model_terms

    mtl_dir = _MTL_DIR
    derived_stage = _derive_stage()

    with _download_lock:
        if not _download_state["active"]:
            _download_state["stage"] = derived_stage
            _download_state["complete"] = False

        state = {
            "active": _download_state["active"],
            "stage": _download_state["stage"],
            "current_model": _download_state["current_model"],
            "progress": _public_progress(_download_state["progress"]),
            "error": _download_state["error"],
            "complete": _download_state["complete"],
        }

    # Read once and thread it through: this route is polled every second, and
    # every serializer below needs the same answer.
    accepted = model_terms.state()

    state["required_models"] = _serialize_required_models(accepted)
    state["mtl_models"] = _get_mtl_models(mtl_dir, accepted)
    state["all_mtl_downloaded"] = _all_mtl_models_downloaded(mtl_dir)
    state["non_mtl_required"] = _incomplete_models
    # Acceptance state only -- versions and booleans. The notice text itself is
    # fetched from /api/setup/terms when a card is opened, so this payload stays
    # small enough to poll.
    state["terms"] = accepted
    # Optional downloads ride along for display; the wizard stage machine
    # never waits on them.
    state["optional"] = _optional_payload(accepted)
    state["ocr_vl"] = _optional_payload(accepted, categories=("paddleocr_vl",))
    return JSONResponse(state)


@router.post("/api/setup/download")
async def setup_download(request: Request):
    global _MTL_DIR

    body = await request.json()

    with _download_lock:
        if _download_state["active"]:
            return JSONResponse(
                {"error": "Download already in progress"},
                status_code=409,
            )

    # Reconcile the wizard stage from the actual filesystem and completion
    # marker before processing the button click.
    current_stage = _derive_stage()

    with _download_lock:
        _download_state["stage"] = current_stage
        _download_state["complete"] = current_stage == "complete"

    # Stage 1: required non-MTL models.
    if current_stage == "non_mtl":
        if _required_models_ready():
            with _download_lock:
                _download_state["stage"] = "mtl_selection"
                _download_state["complete"] = False
            return JSONResponse({
                "status": "mtl_selection",
                "stage": "mtl_selection",
                "message": "Required models are already complete.",
                "required_models": _serialize_required_models(),
                "mtl_models": _get_mtl_models(_MTL_DIR),
                "all_mtl_downloaded": _all_mtl_models_downloaded(_MTL_DIR),
                "optional": _optional_payload(),
                "ocr_vl": _optional_payload(categories=("paddleocr_vl",)),
            })

        from fox_reader.constants import HF_BASE_MODELS

        mtl_dir = _MTL_DIR

        # Read from the constants and the filesystem rather than _HF_MODELS.
        # That cache is filled in by `check_models_exist` at startup, so a model
        # removed since then -- deleted to force a re-download, say -- leaves it
        # empty, and this would report a started download that had nothing to
        # do.
        pending = [
            model
            for category in _required_non_mtl_categories()
            for model in HF_BASE_MODELS.get(category, [])
            if not _model_completed(MODELS_DIR, model)
        ]

        # Before the thread, not inside it: a refusal has to reach the caller as
        # a status code, and a licence check that happens after the download
        # starts has not checked anything.
        if refusal := _terms_refusal(pending):
            return refusal

        with _download_lock:
            models_to_download: list[tuple[str, dict]] = [
                (model["id"], model) for model in pending
            ]

            _download_state["active"] = True
            _download_state["error"] = None
            _download_state["complete"] = False
            _download_state["progress"] = {}
            _download_state["current_model"] = None
            _download_state["stage"] = "non_mtl"

        thread = threading.Thread(
            target=_download_models_task,
            args=(mtl_dir, models_to_download, "non_mtl"),
            daemon=True,
        )
        thread.start()

        return JSONResponse({"status": "started", "stage": "non_mtl"})

    # Stage 2: selected MTL models.
    if current_stage == "mtl_selection":
        mtl_dir_raw = body.get("mtl_dir", str(_MTL_DIR))
        selected_ids = body.get("selected_mtl", [])

        if not isinstance(selected_ids, list):
            return JSONResponse(
                {"error": "selected_mtl must be a list"},
                status_code=400,
            )

        try:
            selected_models = _resolve_mtl_models(selected_ids)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        mtl_dir_rep = validate_output_calc(str(mtl_dir_raw))
        if (
            mtl_dir_rep["status"] != "success"
            or not isinstance(mtl_dir_rep.get("path"), Path)
        ):
            return JSONResponse({"error": mtl_dir_rep["error"]}, status_code=400)

        mtl_dir = mtl_dir_rep["path"]
        _MTL_DIR = mtl_dir

        # Already-downloaded models are excluded from the worker.
        selected_models = [
            model for model in selected_models
            if not _model_completed(mtl_dir, model)
        ]

        if not selected_models:
            with _download_lock:
                _download_state["stage"] = "complete"
                _download_state["complete"] = True
                _download_state["active"] = False
                _download_state["error"] = None
                _download_state["current_model"] = None
                _download_state["progress"] = {}
            return JSONResponse({"status": "complete", "stage": "complete"})

        # Gemma and VNTL carry the strictest terms of anything here -- Google's
        # Gemma Terms and Meta's Llama 3 Community License, both of which
        # require acceptance by their own wording.
        if refusal := _terms_refusal(selected_models):
            return refusal

        models_to_download = [
            (model["id"], model)
            for model in selected_models
        ]

        with _download_lock:
            _download_state["active"] = True
            _download_state["error"] = None
            _download_state["complete"] = False
            _download_state["progress"] = {}
            _download_state["current_model"] = None
            _download_state["stage"] = "mtl"

        thread = threading.Thread(
            target=_download_models_task,
            args=(mtl_dir, models_to_download, "mtl"),
            daemon=True,
        )
        thread.start()

        return JSONResponse({"status": "started", "stage": "mtl"})

    return JSONResponse(
        {"error": f"Setup is not ready for downloads in stage: {current_stage}"},
        status_code=409,
    )


@router.post("/api/setup/finish")
async def setup_finish():
    """Finish setup without downloading any MTL models."""
    with _download_lock:
        if _download_state["active"]:
            return JSONResponse(
                {"error": "Download already in progress"},
                status_code=409,
            )

        if _incomplete_models:
            return JSONResponse(
                {"error": "Required non-MTL models are not fully downloaded."},
                status_code=409,
            )

        _download_state["stage"] = "complete"
        _download_state["complete"] = True
        _download_state["error"] = None

    return JSONResponse({"status": "complete"})


@router.post("/api/setup/close")
async def setup_close(request: Request):
    """Finish the setup wizard and terminate the current backend process."""
    with _download_lock:
        if _download_state["active"]:
            return JSONResponse(
                {"error": "A model download is still in progress."},
                status_code=409,
            )

    if not _required_models_ready():
        return JSONResponse(
            {"error": "Required models are not fully downloaded."},
            status_code=409,
        )

    with _download_lock:
        _download_state["stage"] = "complete"
        _download_state["complete"] = True
        _download_state["error"] = None

    # Give the HTTP response a moment to reach the browser, then stop this Fox
    # Reader backend. The setup page no longer restarts it.
    #
    # Through the server object, not a signal: uvicorn then runs the lifespan
    # shutdown, which is what unloads the Text Seg network and empties cache/.
    # `os.kill(SIGTERM)` -- what this used to do -- is TerminateProcess on
    # Windows, so none of that happened there.
    from fox_reader import runtime

    runtime.stop(0.25)

    return JSONResponse({"status": "closing"})


# ── Optional model downloads ─────────────────────────────────────────────────
# Bubble segmentation, Text Segmentation and PaddleOCR-VL. None of them blocks
# startup; each one's absence switches off exactly its own control, which is
# what makes them safe to offer rather than demand.


def _optional_models(categories: tuple[str, ...] | None = None) -> list[dict]:
    """Optional non-MTL model entries, in category then table order."""
    from fox_reader.constants import HF_BASE_MODELS

    wanted = list(categories) if categories else _optional_non_mtl_categories()

    return [
        model
        for category in wanted
        for model in HF_BASE_MODELS.get(category, [])
    ]


def _serialize_optional(
    accepted: dict[str, dict],
    categories: tuple[str, ...] | None = None,
) -> list[dict]:
    """Optional models with on-disk and terms-acceptance state.

    Reads through `_optional_models`, so the list the page shows and the list a
    download iterates are the same list.
    """
    return [
        {
            "id": model["id"],
            "name": model.get("name", model["id"]),
            "category": model.get("category", ""),
            "category_name": CATEGORY_NAME_MAP.get(model.get("category", ""), ""),
            "repo": model.get("repo", ""),
            "dest": model.get("dest", ""),
            "files": list(model.get("files") or []),
            "type": model.get("type", "Not specified"),
            "size_on_disk": model.get("size_on_disk", "Not specified"),
            "note": model.get("note", ""),
            "downloaded": _model_completed(MODELS_DIR, model),
            "terms_accepted": bool(accepted.get(model["id"], {}).get("accepted")),
        }
        for model in _optional_models(categories)
    ]


def _optional_task(models_to_download: list[tuple[str, dict]], base_dir: Path) -> None:
    """Background download for the optional weights.

    Same worker as the wizard stages, pointed at the optional state dict: the
    stage machine never sees this job, and this job never touches the stages.
    ``base_dir`` travels with the job (rather than being re-read from the module
    global) so the pending check in the route and the worker below can never
    disagree about where the weights live.
    """
    entries = [(key, model, base_dir) for key, model in models_to_download]
    try:
        _run_model_downloads(entries, lock=_optional_lock, state=_optional_state)
    except Exception as exc:
        with _optional_lock:
            _optional_state["error"] = str(exc)
            _optional_state["active"] = False
            _optional_state["current_model"] = None
        return
    with _optional_lock:
        _optional_state["active"] = False
        _optional_state["current_model"] = None
        _optional_state["error"] = None


def _optional_payload(
    accepted: dict[str, dict] | None = None,
    categories: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """The optional-models block: folded into /api/setup/status and aliased.

    `categories` narrows it, which is how the deprecated VL-only alias keeps
    reporting exactly what it used to.
    """
    if accepted is None:
        from fox_reader import model_terms

        accepted = model_terms.state()

    with _optional_lock:
        payload = {
            "active": _optional_state["active"],
            "current_model": _optional_state["current_model"],
            "progress": _public_progress(_optional_state["progress"]),
            "error": _optional_state["error"],
        }
    payload["models"] = _serialize_optional(accepted, categories)
    payload["downloaded"] = bool(payload["models"]) and all(m["downloaded"] for m in payload["models"])
    return payload


def _start_optional_download(models: list[dict]) -> JSONResponse:
    """Start (or refuse) an optional download of `models`. One code path.

    Shared by the general route and the VL alias so the licence gate, the
    already-complete short-circuit and the busy check cannot differ between
    them.
    """
    with _optional_lock:
        if _optional_state["active"]:
            return JSONResponse({"error": "Download already in progress"}, status_code=409)

    pending = [model for model in models if not _model_completed(MODELS_DIR, model)]

    if not pending:
        return JSONResponse({"status": "complete", "downloaded": True})

    if refusal := _terms_refusal(pending):
        return refusal

    with _optional_lock:
        # Re-checked under the lock: two clicks a few milliseconds apart would
        # otherwise both pass the check above and run the same download twice.
        if _optional_state["active"]:
            return JSONResponse({"error": "Download already in progress"}, status_code=409)

        _optional_state["active"] = True
        _optional_state["current_model"] = None
        _optional_state["error"] = None
        _optional_state["progress"] = {}

    thread = threading.Thread(
        target=_optional_task,
        args=([(model["id"], model) for model in pending], MODELS_DIR),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"status": "started"})


@router.get("/api/setup/optional")
async def optional_status():
    """The optional-models block on its own; also rides in /api/setup/status."""
    return JSONResponse(_optional_payload())


@router.post("/api/setup/optional/download")
async def optional_download(request: Request):
    """Download the optional models named in ``ids``, or every one of them.

    Per-model selection is the point: a user who wants the Bubble Capture button
    but not a 1.8 GiB OCR engine should be able to say so.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    requested = body.get("ids") if isinstance(body, dict) else None
    available = _optional_models()

    if requested is None:
        selected = available
    elif not isinstance(requested, list):
        return JSONResponse({"error": "ids must be a list"}, status_code=400)
    else:
        by_id = {model["id"]: model for model in available}
        selected = []

        for model_id in requested:
            model = by_id.get(model_id) if isinstance(model_id, str) else None

            if model is None:
                return JSONResponse(
                    {"error": f"Unknown optional model: {model_id}"},
                    status_code=400,
                )

            selected.append(model)

        if not selected:
            return JSONResponse({"error": "No models selected."}, status_code=400)

    return _start_optional_download(selected)


@router.get("/api/setup/ocr-vl")
async def ocr_vl_status():
    """Deprecated alias: the same block now rides in /api/setup/status."""
    return JSONResponse(_optional_payload(categories=("paddleocr_vl",)))


@router.post("/api/setup/ocr-vl/download")
async def ocr_vl_download():
    """Deprecated alias: downloads the VL weights via the shared optional job."""
    return _start_optional_download(_optional_models(("paddleocr_vl",)))


# ── Model terms and licences ─────────────────────────────────────────────────


@router.get("/api/setup/terms")
async def setup_terms(request: Request):
    """One model's terms notice, or every notice's acceptance state.

    Fetched per model on demand rather than folded into /api/setup/status: the
    full text of nine notices on a one-second poll would be absurd, and a card
    nobody opens costs nothing this way.
    """
    from fox_reader import model_terms

    model_id = request.query_params.get("id")

    if not model_id:
        return JSONResponse({"terms": model_terms.state()})

    document = model_terms.document(model_id)

    if document is None:
        return JSONResponse({"error": f"No terms for model: {model_id}"}, status_code=404)

    return JSONResponse({
        "document": document,
        "accepted": model_terms.is_accepted(model_id),
    })


@router.post("/api/setup/terms/accept")
async def setup_terms_accept(request: Request):
    """Record that the user read and accepted the notices for ``ids``.

    Returns the acceptance state afterwards so the page never has to guess at
    what stuck -- an id it asked about and does not get back is one the server
    declined, which the page should surface rather than paper over.
    """
    from fox_reader import model_terms

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected a JSON body."}, status_code=400)

    ids = body.get("ids") if isinstance(body, dict) else None

    if isinstance(ids, str):
        ids = [ids]

    if not isinstance(ids, list) or not ids:
        return JSONResponse({"error": "ids must be a non-empty list"}, status_code=400)

    unknown = [
        model_id
        for model_id in ids
        if not isinstance(model_id, str) or model_terms.document(model_id) is None
    ]

    if unknown:
        return JSONResponse(
            {"error": "Unknown model: " + ", ".join(str(item) for item in unknown)},
            status_code=400,
        )

    try:
        recorded = model_terms.accept(ids)
    except OSError as exc:
        # Acceptance that was not written down is not acceptance: say so rather
        # than let the page enable a download the next poll will refuse.
        return JSONResponse(
            {"error": f"Could not save your acceptance: {exc}"},
            status_code=500,
        )

    return JSONResponse({"accepted": recorded, "terms": model_terms.state()})
