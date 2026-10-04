from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from fox_reader.models.requests import LoadMLRequest, TranslationRequest
from fox_reader.translate.context import normalize_context

logger = logging.getLogger(__name__)

router = APIRouter()

ML_TRANSLATION_LIMIT = asyncio.Semaphore(1)


def _empty_response(error: str = "") -> dict:
    return {
        "original": "",
        "translated": "",
        "alt_translated": "",
        "error": error,
    }


def _translate_status(request: Request) -> dict:
    """Which token-gated translators may be offered. Never leaks tokens.

    Tokens are validated on save, so presence means "valid when saved". A key
    revoked afterwards surfaces as a 403 from the translate call itself rather
    than from this flag — checking live here would add a ping/usage round-trip
    to every page load.
    """
    translate = getattr(request.app.state, "translate", None)
    if translate is None:
        return {"deepl": False, "jpdb": False}
    try:
        status = translate.translate_status()
        return {"deepl": bool(status.get("deepl")), "jpdb": bool(status.get("jpdb"))}
    except Exception:
        pass
    # Fall back to the stored tokens directly (e.g. in tests where the
    # service is stubbed but settings exist).
    try:
        tokens = request.app.state.settings.get_api_tokens()
        return {
            "deepl": bool((tokens.get("deepl_api_token") or "").strip()),
            "jpdb": bool((tokens.get("jpdb_api_token") or "").strip()),
        }
    except Exception:
        return {"deepl": False, "jpdb": False}


@router.get("/api/translate/status", response_class=JSONResponse)
async def translate_status(request: Request):
    """For the reader page: hide DeepL/JPDB buttons unless their token is set."""
    return JSONResponse(content=_translate_status(request))


@router.post("/translate/deepl", response_class=JSONResponse)
async def translate_deepl(request: Request, payload: TranslationRequest):
    text = payload.text.strip()
    if not text:
        return JSONResponse(content=_empty_response("No text provided!"))

    translate = request.app.state.translate
    result = await translate.deepl(text, payload.source_lang.strip().lower())
    resp = result.to_dict()
    error = f"Error: {resp.get('message', 'Unknown')}" if result.code != 200 else None
    return JSONResponse(content={
        "original": text,
        "translated": resp.get("data", ""),
        "alt_translated": "\n".join(resp.get("alternatives", [])),
        "error": error,
    })


@router.post("/translate/jpdb", response_class=JSONResponse)
async def translate_jpdb(request: Request, payload: TranslationRequest):
    text = payload.text.strip()
    if not text:
        return JSONResponse(content=_empty_response("No text provided!"))

    translate = request.app.state.translate
    result = await translate.jpdb(text, payload.source_lang.strip().lower())
    resp = result.to_dict()
    error = f"Error: {resp.get('message', 'Unknown')}" if result.code != 200 else None
    return JSONResponse(content={
        "original": text,
        "translated": resp.get("data", ""),
        "alt_translated": "",
        "error": error,
    })


@router.post("/translate/custom", response_class=JSONResponse)
async def translate_custom(request: Request, payload: TranslationRequest):
    text = payload.text.strip()
    if not text:
        return JSONResponse(content=_empty_response("No text provided!"))

    translate = request.app.state.translate
    result = await translate.custom(
        text,
        payload.source_lang.strip(),
        (payload.lang_group or "").strip().lower(),
        normalize_context(payload.context),
        getattr(payload, "character_info", None),
        getattr(payload, "context_character_links", None),
    )
    resp = result.to_dict()
    error = f"Error: {resp.get('message', 'Unknown')}" if result.code != 200 else None

    return JSONResponse(content={
        "original": text,
        "translated": resp.get("data", ""),
        "alt_translated": "\n".join(resp.get("alternatives", [])),
        "error": error,
    })


@router.post("/translate/ml", response_class=JSONResponse)
async def translate_ml(request: Request, payload: TranslationRequest):
    translate = request.app.state.translate

    text = payload.text.strip()
    if not text:
        return JSONResponse(content=_empty_response("No text provided!"))

    try:
        async with ML_TRANSLATION_LIMIT:
            context = normalize_context(payload.context)
            result = await translate.ml_async(
                text,
                payload.source_lang.strip().lower(),
                context,
                getattr(payload, "character_info", None),
                getattr(payload, "context_character_links", None),
                getattr(payload, "meta_id", None),
            )
        resp = result.to_dict()
        return JSONResponse(content={
            "original": text,
            "translated": resp.get("data", ""),
            "alt_translated": "",
            "error": None,
        })
    except Exception as e:
        return JSONResponse(content=_empty_response(f"ML Inference Error: {e}"))


@router.get("/ml/memory")
async def ml_memory(request: Request):
    """Whether the configured local models fit in RAM/VRAM, without loading.

    Never an error: this only feeds a warning banner, and a probe that fails is
    not worth breaking the reader page over.
    """
    translate = request.app.state.translate
    empty = {"enabled": False, "device": "", "ok": True, "checks": [], "issues": []}

    try:
        # psutil and the torch device probes are synchronous, and the CUDA probe
        # in particular can take a moment on a cold driver.
        return await run_in_threadpool(translate.ml_memory)
    except Exception as exc:
        logger.warning("Could not check the MTL memory requirements: %s", exc)
        return empty


@router.get("/ml/control/status")
async def ml_status(request: Request):
    """Which local model (if any) is still loaded from before a page refresh.

    The reader page asks once on startup so its MTL switch and language
    select match the backend instead of resetting to off. Never an error:
    anything unreadable reads as "nothing loaded", and the page simply stays
    as it was.
    """
    empty: dict[str, object] = {"loaded": False, "lang": None, "model_id": None, "languages": []}

    try:
        return await run_in_threadpool(request.app.state.translate.ml_status)
    except Exception as exc:
        logger.warning("Could not read the loaded MTL model status: %s", exc)
        return empty


@router.post("/ml/control/load")
async def load_ml(request: Request, req: LoadMLRequest):
    translate = request.app.state.translate
    try:
        # Reads gigabytes of weights off disk and onto the device. Run on the
        # event loop it would hold up every other request for the whole load,
        # `GET /api/health` included -- which is exactly when the UI polls it,
        # since the user just asked for a model.
        await run_in_threadpool(translate.ml_load, req.lang)
        return {"status": True, "message": "Model loaded successfully", "lang": req.lang}
    except Exception as e:
        return {"status": False, "message": str(e)}


@router.post("/ml/control/unload")
async def unload_model(request: Request):
    translate = request.app.state.translate
    try:
        # Freeing a native/VRAM allocation blocks too, and `LocalMTLManager`
        # serialises this against an in-flight translation internally.
        await run_in_threadpool(translate.ml_unload)
        return {"status": True, "message": "Model unloaded successfully"}
    except Exception as e:
        return {"status": False, "message": str(e)}
