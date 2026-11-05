"""Setup page routes — first-launch model download wizard."""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from fox_reader import assets
from fox_reader.config import FoxConfig
from fox_reader.hub_progress import FileProgress, observe_hub_client
from fox_reader.hub_progress import HubByteProgress as _HubByteProgress
from fox_reader.hub_progress import download_context as _dl_ctx
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
_storage_apply_generation = 0
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

def _file_progress(state: dict[str, Any], lock: Any, key: str, entry: dict[str, Any]) -> FileProgress:
    """Bind a callback to this exact job, row and file; never reuse its identity."""
    token = entry["_file_token"] = object()
    completed = entry["bytes_base"]

    def publish(count: int, total: int, phase: str) -> None:
        transfer = progress.transfer_snapshot()
        with lock:
            if state["progress"].get(key) is not entry or entry.get("_file_token") is not token:
                return
            entry.update(
                current_bytes=count, current_total=total, bytes_done=completed + count, phase=phase, **transfer,
            )

    progress = FileProgress(publish)
    return progress

_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_INVALID_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')

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


def _derive_idle_stage() -> str:
    """Derive the selection stage without taking a snapshot of the running job."""
    # A direct /setup visit is also a model-management/setup page after the
    # required models are installed. The persistent setup-complete marker
    # controls normal application startup, but must not hide MTL choices here.
    if not _required_models_ready():
        return "non_mtl"

    return "mtl_selection"


def _derive_stage() -> str:
    with _download_lock:
        if _download_state["active"]:
            return _download_state["stage"]
    return _derive_idle_stage()


# ── Download worker ───────────────────────────────────────────────────────────

def _discover_model_files(api, model: dict) -> list[str]:
    """Resolve every configured filename/suffix before downloading any files."""
    patterns = tuple(model.get("files") or ())
    repo_files = api.list_repo_files(model["repo"])

    files: list[str] = []
    for repo_file in repo_files:
        if repo_file.startswith("."):
            continue
        if not patterns or repo_file.endswith(patterns):
            files.append(repo_file)

    missing = [pattern for pattern in patterns if not any(name.endswith(pattern) for name in files)]
    if missing:
        raise ValueError(f"required model files not found in repository: {', '.join(missing)}")
    if not files:
        raise ValueError("repository contains no downloadable model files")

    return files


def _fresh_progress_entry(files_total: int) -> dict[str, Any]:
    """A progress entry with file counts and byte counters.

    ``bytes_base`` is internal (bytes of files finished this run) and is
    stripped before anything is sent to a client.
    """
    return {
        "done": False,
        "phase": "preparing",
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
        key: {k: v for k, v in entry.items() if k not in ("bytes_base", "_file_token")}
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
    page sees bytes moving, including when the server provides no length.
    A response-stream observer adds finer updates for HTTP without changing
    Hub's chunking, retries, or cache. Xet's native transfer counters are kept
    separate from assembled file bytes, which can stay at zero until assembly.
    Phases describe the current file; its byte total never stands in for the
    unknown sizes of remaining model files.

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
                    "phase": "complete",
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
                entry.pop("transfer_bytes", None)
                entry.pop("transfer_total", None)
                entry["phase"] = "preparing"
                entry["bytes_base"] = completed_bytes
                entry["bytes_done"] = completed_bytes
                progress = _file_progress(state, lock, model_key, entry)

            previous = getattr(_dl_ctx, "current", None)
            _dl_ctx.current = progress
            try:
                observe_hub_client()
                downloaded_path = hf_hub_download(
                    repo_id=model["repo"],
                    filename=filename,
                    local_dir=str(model_dest),
                    tqdm_class=_HubByteProgress,
                )
                # Cached files may never open a progress bar. A successful Hub
                # return and the saved file, not a byte counter, prove completion.
                finished = Path(downloaded_path).stat().st_size
                progress.finish(finished)
            except Exception as exc:
                raise RuntimeError(f"{model.get('name', model_key)}: {filename}: {exc}") from exc
            finally:
                progress.finish()
                _dl_ctx.current = previous

            with lock:
                entry = state["progress"][model_key]
                completed_bytes += finished
                entry["files_done"] = index
                entry["bytes_base"] = completed_bytes
                entry["bytes_done"] = completed_bytes
                entry.pop("_file_token", None)

        completed_file.touch()

        with lock:
            entry = state["progress"][model_key]
            entry["done"] = True
            entry["downloaded"] = True
            entry["phase"] = "complete"
            entry["current_file"] = ""
            entry["current_bytes"] = 0
            entry["current_total"] = 0
            entry.pop("transfer_bytes", None)
            entry.pop("transfer_total", None)
            entry["bytes_total"] = completed_bytes


def _download_models_task(
    mtl_dir: Path,
    models_to_download: list[tuple[str, dict]],
    stage: str,
) -> None:
    """Background task for one setup wizard stage."""
    global _download_state

    entries = [
        (key, model, mtl_dir if stage == "mtl" else MODELS_DIR)
        for key, model in models_to_download
    ]

    try:
        # The shared worker creates its destination directories here, so a
        # filesystem failure clears the active state like any download error.
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

def _storage_paths(user_path: str) -> tuple[Path, Path]:
    """Return a portable configured spelling and its absolute destination."""
    if not isinstance(user_path, str):
        raise ValueError("Enter a folder path.")
    value = user_path.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    if not value:
        raise ValueError("Enter a folder path or choose the default location.")
    if "\x00" in value:
        raise ValueError("The folder path contains an invalid character.")
    path = Path(value).expanduser()
    if os.name == "nt":
        if path.drive and not path.is_absolute():
            raise ValueError("Use a full drive path, such as C:\\Models, instead of a drive-relative path.")
        for part in path.parts:
            if part in (path.anchor, "", ".", ".."):
                continue
            if _INVALID_CHARS.search(part):
                raise ValueError(f"Invalid character in folder name: {part!r}")
            if part.endswith((" ", ".")):
                raise ValueError(f"Folder names cannot end in a space or dot: {part!r}")
            if part.split(".")[0].upper() in _WINDOWS_RESERVED:
                raise ValueError(f"Reserved Windows folder name: {part!r}")
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve(strict=False)
    # The default stays relative so moving a portable build does not orphan its models.
    configured = Path("models") if resolved == (PROJECT_ROOT / "models").resolve() else path
    return configured, resolved


def _inspect_storage(user_path: str) -> dict[str, Any]:
    """Validate without creating folders or changing configuration."""
    from fox_reader.fsaccess import _probe_writable

    report: dict[str, Any] = {"ok": False, "path": "", "error": "", "exists": False, "can_create": False}
    try:
        _, path = _storage_paths(user_path)
        report["path"] = str(path)
        report["exists"] = path.exists()
        if report["exists"] and not path.is_dir():
            raise ValueError("This path points to a file. Choose a folder instead.")
        target = path if report["exists"] else path.parent
        if not target.exists():
            raise ValueError("The parent folder does not exist. Choose an existing folder or a new folder inside one.")
        if not target.is_dir():
            raise ValueError("The parent path is a file, not a folder.")
        # Use the exact host-native path here. fsaccess.describe_dir also
        # normalizes backslashes/quotes, which are valid POSIX filename characters.
        with os.scandir(target):
            pass
        if not os.access(target, os.X_OK) or not _probe_writable(str(target)):
            raise ValueError("Fox Reader cannot write to this folder. Choose a writable location.")
        report.update(ok=True, can_create=not report["exists"])
        try:
            report["free_bytes"] = shutil.disk_usage(target).free
        except OSError:
            pass
    except (OSError, RuntimeError, ValueError) as exc:
        report["error"] = str(exc) or "This folder could not be checked."
    return report


def validate_output_calc(user_path: str) -> dict[str, Path | str]:
    """Compatibility wrapper shared with setup's original download entry point."""
    report = _inspect_storage(user_path)
    if not report["ok"]:
        return {"status": "failed", "error": report["error"]}
    return {"status": "success", "path": Path(report["path"])}


def _store_mtl_location(request: Request, user_path: str, path: Path, *, preserve_existing: bool = False) -> str:
    """Called under the download lock; publish only after persistence succeeds."""
    global _MTL_DIR
    configured, resolved = _storage_paths(user_path)
    if resolved != path:
        raise ValueError("The folder changed while it was being checked. Try again.")
    manager = getattr(request.app.state, "config_manager", None)
    if manager is None:
        # Older embedded clients can still download into their existing root.
        # A different location must never look saved when there is no config owner.
        if path != _MTL_DIR.resolve():
            raise RuntimeError("The storage configuration is unavailable. Restart Fox Reader and try again.")
        return str(path)
    config = manager.config
    if config is None:
        raise RuntimeError("The storage configuration has not loaded. Restart Fox Reader and try again.")
    if preserve_existing and config.resolve_mtl_dir(PROJECT_ROOT).resolve() == path:
        # Download requests identify the applied folder by absolute path; that
        # must not rewrite an already-saved portable relative preference.
        configured = config.mtl_dir
    if config.mtl_dir != configured:
        config = manager.update(mtl_dir=configured)
    request.app.state.config = config
    request.app.state.mtl_dir = path
    _MTL_DIR = path
    return str(config.mtl_dir)


async def _storage_request_path(request: Request) -> str | JSONResponse:
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "Send a folder path as JSON."}, status_code=400)
    if not isinstance(body, dict) or not isinstance(body.get("path"), str):
        return JSONResponse({"error": "path must be a folder path string."}, status_code=400)
    return body["path"]


@router.post("/api/setup/storage/inspect")
async def inspect_setup_storage(request: Request):
    path = await _storage_request_path(request)
    if isinstance(path, JSONResponse):
        return path
    return JSONResponse(await asyncio.to_thread(_inspect_storage, path))


@router.post("/api/setup/storage")
async def apply_setup_storage(request: Request):
    global _storage_apply_generation
    value = await _storage_request_path(request)
    if isinstance(value, JSONResponse):
        return value
    with _download_lock:
        if _download_state["active"] or getattr(request.app.state, "setup_closing", False):
            return JSONResponse(
                {"error": "Wait until the current download finishes before changing its location."}, status_code=409,
            )
        _storage_apply_generation += 1
        generation = _storage_apply_generation
    report = await asyncio.to_thread(_inspect_storage, value)
    if not report["ok"]:
        return JSONResponse(report, status_code=400)
    path = Path(report["path"])
    with _download_lock:
        if _download_state["active"] or getattr(request.app.state, "setup_closing", False):
            return JSONResponse(
                {"error": "The storage location cannot change while downloading or closing."}, status_code=409,
            )
        # A timed-out request can finish inspection after a retry has already
        # saved a newer choice. Never let that older request overwrite it.
        if generation != _storage_apply_generation:
            return JSONResponse(
                {"error": "A newer storage location request replaced this one. Check the current location."},
                status_code=409,
            )
        try:
            configured = _store_mtl_location(request, value, path)
        except (OSError, RuntimeError, ValueError) as exc:
            return JSONResponse({"error": f"Could not save the storage location: {exc}"}, status_code=503)
        # These entries describe the previous job's destination. Its completed
        # flags must not make a different folder look populated in the UI.
        _download_state["progress"] = {}
        _download_state["current_model"] = None
        _download_state["error"] = None
        _download_state["complete"] = False
    return JSONResponse({
        "path": str(path), "mtl_dir": configured,
        "mtl_models": _get_mtl_models(path),
        "all_mtl_downloaded": _all_mtl_models_downloaded(path),
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    global _MTL_DIR

    stage = _derive_idle_stage()
    with _download_lock:
        config = request.app.state.config
        if not _download_state["active"]:
            _MTL_DIR = config.resolve_mtl_dir(PROJECT_ROOT).resolve()
            _download_state["stage"] = stage
            _download_state["complete"] = False
        resolved = str(_MTL_DIR)

    return templates.TemplateResponse(
        request=request,
        name="setup.html",
        context={
            "mtl_dir": str(config.mtl_dir),
            "mtl_default_dir": str((PROJECT_ROOT / "models").resolve()),
            "mtl_resolved_dir": resolved,
        },
    )


@router.get("/api/setup/status")
async def setup_status(request: Request):
    from fox_reader import model_terms

    # A worker can finish while readiness is checked. Only read its active
    # stage in the same locked snapshot as `active`, never before that snapshot.
    derived_stage = _derive_idle_stage()

    with _download_lock:
        mtl_dir = _MTL_DIR
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
            "mtl_dir": str(mtl_dir),
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

        if not isinstance(mtl_dir_raw, str):
            return JSONResponse({"error": "mtl_dir must be a folder path string."}, status_code=400)
        if not isinstance(selected_ids, list):
            return JSONResponse(
                {"error": "selected_mtl must be a list"},
                status_code=400,
            )

        try:
            selected_models = _resolve_mtl_models(selected_ids)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        mtl_dir_rep = await asyncio.to_thread(validate_output_calc, mtl_dir_raw)
        if (
            mtl_dir_rep["status"] != "success"
            or not isinstance(mtl_dir_rep.get("path"), Path)
        ):
            return JSONResponse({"error": mtl_dir_rep["error"]}, status_code=400)

        mtl_dir = mtl_dir_rep["path"]

        # Already-downloaded models are excluded from the worker.
        selected_models = [
            model for model in selected_models
            if not _model_completed(mtl_dir, model)
        ]

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
            # Inspection runs off-thread. Another tab may have started a job
            # while it was pending, so claim the destination and job together.
            if _download_state["active"] or getattr(request.app.state, "setup_closing", False):
                return JSONResponse({"error": "A download is active or Fox Reader is closing."}, status_code=409)
            try:
                _store_mtl_location(request, mtl_dir_raw, mtl_dir, preserve_existing=True)
            except (OSError, RuntimeError, ValueError) as exc:
                return JSONResponse({"error": f"Could not save the storage location: {exc}"}, status_code=503)
            if not selected_models:
                _download_state["stage"] = "complete"
                _download_state["complete"] = True
                _download_state["active"] = False
                _download_state["error"] = None
                _download_state["current_model"] = None
                _download_state["progress"] = {}
                return JSONResponse({"status": "complete", "stage": "complete"})
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
        request.app.state.setup_closing = True

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
