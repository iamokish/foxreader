from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from fox_reader import assets
from fox_reader.constants import MTL_MODELS, mtl_model_downloaded
from fox_reader.device import AUTO_THREADS, CPU
from fox_reader.device import SLOTS as DEVICE_SLOTS
from fox_reader.device import physical_cores, thread_count, thread_override
from fox_reader.device import snapshot as device_snapshot
from fox_reader.settings import SUPPORTED_MTL_LANGUAGES

router = APIRouter(tags=["settings"])
templates = assets.templates()

#: The one device-section field that is not a slot: how many CPU threads the
#: GGUF translator gets. Accepted on the same endpoint as the slots because the
#: page has one picker grid and saves from it the same way.
THREADS_FIELD = "mtl_threads"


def _normalize_languages(value) -> list[str]:
    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple)):
        return []

    return [
        text
        for language in value
        if (text := str(language).strip().lower())
    ]


def _serialize_available_models(mtl_dir: Path) -> list[dict]:
    return [
        {
            "id": model["id"],
            "name": model["name"],
            "type": model.get("type", "Not specified"),
            "size_on_disk": model.get("size_on_disk", "Not specified"),
            "languages": _normalize_languages(model.get("languages")),
            "downloaded": mtl_model_downloaded(mtl_dir, model),
        }
        for model in MTL_MODELS
    ]


def _build_settings_payload(request: Request) -> dict:
    settings_mgr = request.app.state.settings
    mtl_dir = request.app.state.mtl_dir
    models = _serialize_available_models(mtl_dir)

    # Startup already did this, but a model can be downloaded or deleted while
    # the app is running. Re-settling here keeps the page showing a value that
    # is saved rather than one that only takes effect once it is clicked.
    defaults = settings_mgr.resolve_mtl_defaults(mtl_dir)

    languages = {}

    for language in SUPPORTED_MTL_LANGUAGES:
        # Copied per language: `models` is one list of shared dicts, and a
        # model that serves several languages would otherwise carry whichever
        # `selected` the last language happened to write -- Gemma showing as
        # selected under Japanese because it is the Korean default.
        compatible = [
            dict(model)
            for model in models
            if language in model["languages"]
        ]
        downloaded = [model for model in compatible if model["downloaded"]]
        selected_model = defaults.get(language)

        for model in compatible:
            model["selected"] = (
                model["downloaded"] and model["id"] == selected_model
            )

        languages[language] = {
            "default_model": selected_model,
            "models": compatible,
            "has_downloaded_model": bool(downloaded),
        }

    return {
        "languages": languages,
        "defaults": defaults,
        "api_tokens": settings_mgr.get_api_tokens(),
        "translate_status": settings_mgr.get_translate_status(),
        "devices": _build_devices_payload(request),
        "ocr": _build_ocr_payload(request),
    }


def _build_ocr_payload(request: Request) -> dict:
    """What the settings page needs for the OCR engine section.

    Never raises: the page must render even when OCR was never constructed
    (models not ready) or the VL probe fails. `running` is what the live
    OCRService was built with; `selected` is what is saved and takes effect
    on restart.
    """
    settings_mgr = request.app.state.settings
    try:
        selected = settings_mgr.get_ocr_engine()
    except Exception:
        from fox_reader.constants import DEFAULT_OCR_ENGINE

        selected = DEFAULT_OCR_ENGINE
    try:
        from fox_reader.ocr import describe_ocr_engines

        payload = describe_ocr_engines(selected)
    except Exception:
        from fox_reader.constants import DEFAULT_OCR_ENGINE

        payload = {"selected": selected, "default": DEFAULT_OCR_ENGINE, "engines": []}
    ocr_service = getattr(request.app.state, "ocr", None)
    running = None
    active = None
    if ocr_service is not None:
        try:
            running = ocr_service.requested_engine
        except Exception:
            running = None
        try:
            active = ocr_service.active_engine
        except Exception:
            active = None
        try:
            live = ocr_service.engine_info()
            # Live availability wins over a fresh probe: the service has
            # already attempted (and possibly failed) the VL load.
            if isinstance(live, dict) and live.get("engines"):
                payload = live
                payload["selected"] = selected
        except Exception:
            pass
    payload["running"] = running
    payload["active"] = active
    payload["restart_required"] = bool(running is not None and running != selected)
    return payload


def _build_devices_payload(request: Request) -> dict:
    """What the settings page needs for the Compute Devices section.

    `selection` is what is saved, `resolved` is what that means on this machine,
    and `running` is what the process actually started with. The three differ
    whenever a choice has been saved but not yet restarted into, which is
    exactly what the page has to tell the user.

    `running` is in device ids so it can be compared with `resolved`; `active`
    carries the same devices in each consumer's own spelling (``gpu:1`` against
    ``cuda:1``), which is what belongs in a bug report but cannot be compared.
    """
    settings_mgr = request.app.state.settings
    saved = settings_mgr.get_devices()
    payload = device_snapshot(saved)

    state = getattr(request.app.state, "devices", None)
    running = {
        slot: (state.resolved.get(slot, "") if state is not None else "")
        for slot in payload["slots"]
    }

    payload["running"] = running
    payload["restart_required"] = any(
        running[slot] and running[slot] != payload["resolved"][slot]
        for slot in payload["slots"]
    )
    payload[THREADS_FIELD] = _build_threads_payload(request, payload)

    return payload


def _build_threads_payload(request: Request, devices: dict) -> dict:
    """The thread picker: what is saved, the ceiling, and what it will mean.

    `auto` and `resolved` are both previews rather than settings — the first is
    what the heuristic would decide, the second what the saved value comes to
    once it is clamped to the CPUs this process may use. They are taken against
    the device the translator resolved to, because a fully offloaded model needs
    far fewer threads and a preview that ignored that would be wrong on a GPU.
    """
    settings_mgr = request.app.state.settings
    value = settings_mgr.get_mtl_threads()
    on_gpu = devices["resolved"]["translator"] != CPU

    return {
        "value": value,
        "max": physical_cores(),
        "auto": thread_count(on_gpu=on_gpu, preference=AUTO_THREADS),
        "resolved": thread_count(on_gpu=on_gpu, preference=value),
        # An env override wins over anything saved here, so the page has to be
        # able to say so rather than show a number that is not being used.
        "override": thread_override(),
    }


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={},
    )


@router.get("/api/settings")
async def get_settings(request: Request):
    return JSONResponse(_build_settings_payload(request))


@router.put("/api/settings/devices")
async def update_devices(request: Request):
    body = await request.json()

    if not isinstance(body, dict):
        return JSONResponse({"error": "Expected an object"}, status_code=400)

    changes = {
        slot: body[slot]
        for slot in DEVICE_SLOTS
        if slot in body
    }
    threads_changed = THREADS_FIELD in body

    if not changes and not threads_changed:
        return JSONResponse(
            {
                "error": (
                    "Expected at least one of: "
                    + ", ".join((*DEVICE_SLOTS, THREADS_FIELD))
                )
            },
            status_code=400,
        )

    settings_mgr = request.app.state.settings

    try:
        if changes:
            settings_mgr.update_devices(changes)

        if threads_changed:
            settings_mgr.update_mtl_threads(body[THREADS_FIELD])
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # The devices are saved but deliberately not applied: the models bound
    # theirs when they were constructed, so a new choice takes effect on the next
    # launch. The thread count is the exception -- LocalMTLManager reads it again
    # before every load, so it applies as soon as a model is next loaded.
    return JSONResponse(_build_settings_payload(request))


@router.put("/api/settings/mtl-default")
async def update_mtl_default(request: Request):
    body = await request.json()

    language = body.get("language")
    model_id = body.get("model_id")

    if not isinstance(language, str):
        return JSONResponse({"error": "language is required"}, status_code=400)

    language = language.strip().lower()
    if language not in SUPPORTED_MTL_LANGUAGES:
        return JSONResponse({"error": "Unsupported language"}, status_code=400)

    if model_id is not None and not isinstance(model_id, str):
        return JSONResponse(
            {"error": "model_id must be a string or null"},
            status_code=400,
        )

    settings_mgr = request.app.state.settings
    mtl_dir = request.app.state.mtl_dir

    compatible_downloaded = [
        model
        for model in MTL_MODELS
        if language in _normalize_languages(model.get("languages"))
        and mtl_model_downloaded(mtl_dir, model)
    ]

    # None is only valid when no downloaded compatible model exists.
    if model_id is None:
        if compatible_downloaded:
            return JSONResponse(
                {
                    "error": (
                        "A downloaded model is available for this language. "
                        "Select one of the available models."
                    )
                },
                status_code=400,
            )
    else:
        target = next(
            (model for model in MTL_MODELS if model["id"] == model_id),
            None,
        )

        if target is None:
            return JSONResponse({"error": "Unknown MTL model"}, status_code=400)

        if language not in _normalize_languages(target.get("languages")):
            return JSONResponse(
                {"error": "Selected model does not support this language"},
                status_code=400,
            )

        if not mtl_model_downloaded(mtl_dir, target):
            return JSONResponse(
                {"error": "Selected MTL model is not downloaded"},
                status_code=400,
            )

    settings_mgr.update_default_model(language, model_id)
    return JSONResponse(_build_settings_payload(request))


def _is_unreachable_message(message: str) -> bool:
    """Whether a validation message means 'no network', not 'bad token'."""
    lowered = (message or "").lower()
    return (
        "could not reach" in lowered
        or "timed out" in lowered
        or "timeout" in lowered
    )


@router.put("/api/settings/api-tokens")
async def update_api_tokens(request: Request):
    """Save DeepL/JPDB tokens, validating non-empty ones first.

    This is the endpoint ``settings.html`` already calls — it existed on the
    page but had no backend, so saving silently 404'd. Validation is on save,
    as requested:

    - Empty string clears that token (always allowed, no network call).
    - A non-empty token is checked live: JPDB via ``POST /api/v1/ping``,
      DeepL via ``GET /v2/usage`` (no quota consumed; equivalent to the
      こんにちは -> Hello test translation).
    - If any non-empty token is invalid, nothing is saved and the response is
      400 with per-field ``details``. Unreachable validators are 503 instead,
      so a network blip is not reported as a bad key.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected a JSON object"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"error": "Expected an object"}, status_code=400)

    allowed = ("deepl_api_token", "jpdb_api_token")
    changes: dict[str, str] = {}
    for key in allowed:
        if key not in body:
            continue
        value = body[key]
        if value is None:
            value = ""
        if not isinstance(value, str):
            return JSONResponse(
                {"error": f"{key} must be a string"}, status_code=400
            )
        changes[key] = value.strip()

    if not changes:
        return JSONResponse(
            {"error": "Expected at least one of: " + ", ".join(allowed)},
            status_code=400,
        )

    # Validate non-empty tokens concurrently before persisting anything.
    import asyncio as _asyncio

    async def _check(key: str, token: str) -> tuple[str, bool, str]:
        if not token:
            return key, True, ""
        try:
            if key == "jpdb_api_token":
                from fox_reader.translate.jpdb_api import validate_jpdb_token

                ok, msg = await validate_jpdb_token(token)
            else:
                from fox_reader.translate.deepl_api import validate_deepl_token

                ok, msg = await validate_deepl_token(token)
        except Exception as exc:  # never let a validator crash the save
            return key, False, f"Could not validate {key}: {exc}"
        return key, ok, msg

    results = await _asyncio.gather(
        *[_check(key, token) for key, token in changes.items()]
    )

    invalid: dict[str, str] = {}
    unreachable: dict[str, str] = {}
    for key, ok, msg in results:
        if ok:
            continue
        if _is_unreachable_message(msg):
            unreachable[key] = msg
        else:
            invalid[key] = msg or f"Invalid {key}."

    if unreachable:
        return JSONResponse(
            {
                "error": "Could not validate token(s): "
                + "; ".join(f"{k}: {v}" for k, v in unreachable.items()),
                "details": {**invalid, **unreachable},
            },
            status_code=503,
        )
    if invalid:
        return JSONResponse(
            {
                "error": "; ".join(f"{k}: {v}" for k, v in invalid.items()),
                "details": invalid,
            },
            status_code=400,
        )

    settings_mgr = request.app.state.settings
    try:
        settings_mgr.update_api_tokens(**changes)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse(_build_settings_payload(request))


@router.put("/api/settings/ocr-engine")
async def update_ocr_engine(request: Request):
    """Select the text-recognition engine. Saved, applied on restart.

    Accepts {"engine": "paddleocr" | "paddleocr-vl"}. An unavailable VL
    (missing llama-cpp-python or weights) is still saved: the running
    OCRService falls back to classic with a warning, and the settings page
    shows why VL cannot run yet. Only an unknown id is a 400.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected a JSON object"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"error": "Expected an object"}, status_code=400)

    engine = body.get("engine", body.get("ocr_engine"))
    if engine is None:
        return JSONResponse({"error": "engine is required"}, status_code=400)
    if not isinstance(engine, str):
        return JSONResponse({"error": "engine must be a string"}, status_code=400)

    settings_mgr = request.app.state.settings
    try:
        settings_mgr.update_ocr_engine(engine)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse(_build_settings_payload(request))
