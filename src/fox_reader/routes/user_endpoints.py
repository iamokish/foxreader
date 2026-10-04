from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

from fox_reader import assets, endpoint_crypto
from fox_reader.endpoint_schema import DEFAULT_UUID_VARIANT, UUID_VARIANTS
from fox_reader.user_endpoints import (
    DEFAULT_MAX_TEXT_LENGTH,
    KNOWN_LANGUAGES,
    MAX_ACTIVE_PER_LANGUAGE,
    MAX_ENDPOINTS_PER_LANGUAGE,
    MAX_MAX_TEXT_LENGTH,
    MAX_TOKEN_MAX_AGE,
    MIN_MAX_TEXT_LENGTH,
    MIN_TOKEN_MAX_AGE,
    NAME_MAX_LENGTH,
    UserEndpoint,
)

router = APIRouter(tags=["user-endpoints"])
templates = assets.templates()

SAMPLE_TEXT = "こんにちは"


def _error_message(exc: Exception) -> str:
    """Turn a validation failure into something worth showing in the editor."""
    if not isinstance(exc, ValidationError):
        return str(exc) or exc.__class__.__name__

    messages: list[str] = []

    for error in exc.errors():
        location = ".".join(
            str(part)
            for part in error.get("loc", ())
            if part != "__root__"
        )

        message = str(error.get("msg", "")).removeprefix("Value error, ")
        message = f"{location}: {message}" if location else message

        if message and message not in messages:
            messages.append(message)

    return "; ".join(messages) or "Invalid endpoint"


async def _body(request: Request) -> dict[str, Any]:
    """Read a JSON object body, tolerating an empty or malformed one."""
    try:
        data = await request.json()
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def _language_of(body: dict[str, Any]) -> str:
    return str(body.get("language") or "").strip().lower()


@router.get("/user_endpoints", response_class=HTMLResponse)
async def user_endpoints_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="user_endpoints.html",
        context={
            "known_languages": list(KNOWN_LANGUAGES),
            "max_per_language": MAX_ENDPOINTS_PER_LANGUAGE,
            "max_active_per_language": MAX_ACTIVE_PER_LANGUAGE,
            "name_max_length": NAME_MAX_LENGTH,
            "text_length_min": MIN_MAX_TEXT_LENGTH,
            "text_length_max": MAX_MAX_TEXT_LENGTH,
            "text_length_default": DEFAULT_MAX_TEXT_LENGTH,
            "uuid_variants": list(UUID_VARIANTS),
            "uuid_default": DEFAULT_UUID_VARIANT,
            "key_encodings": list(endpoint_crypto.KEY_ENCODINGS),
            "key_length": endpoint_crypto.KEY_TEXT_LENGTH,
            "max_keys": endpoint_crypto.MAX_KEYS,
            "encryption_available": endpoint_crypto.available(),
            "token_max_age_min": MIN_TOKEN_MAX_AGE,
            "token_max_age_max": MAX_TOKEN_MAX_AGE,
        },
    )


@router.get("/api/user_endpoints")
async def get_user_endpoints(request: Request):
    return JSONResponse(request.app.state.user_endpoints.as_dict())


@router.get("/api/user_endpoints/active")
async def get_active_endpoints(request: Request):
    """The endpoints the reader turns into translator buttons."""
    store = request.app.state.user_endpoints

    if store is None:
        return JSONResponse({"active": {}})

    return JSONResponse({
        "active": store.active_summary(),
        "max_active_per_language": MAX_ACTIVE_PER_LANGUAGE,
    })


@router.post("/api/user_endpoints/preview")
async def preview_user_endpoint(request: Request):
    """Validate a draft endpoint and show the request it would send."""
    from fox_reader.translate.custom_endpoint import describe_request

    body = await _body(request)
    draft = body.get("endpoint") if "endpoint" in body else body

    if not isinstance(draft, dict):
        return JSONResponse({"error": "endpoint must be an object"}, status_code=400)

    # An edit form has no copy of the stored encryption key, so borrow it for
    # the preview rather than failing on the blank field.
    store = getattr(request.app.state, "user_endpoints", None)

    if store is not None and draft.get("id"):
        draft = store.with_stored_secret(str(draft["id"]), draft)

    try:
        endpoint = UserEndpoint.model_validate(draft)
    except Exception as exc:
        return JSONResponse({"error": _error_message(exc)}, status_code=400)

    text = str(body.get("text") or SAMPLE_TEXT)
    language = _language_of(body)

    try:
        request_preview = describe_request(
            endpoint,
            text=text,
            source_lang=language,
        )
    except Exception as exc:
        return JSONResponse({"error": _error_message(exc)}, status_code=400)

    return JSONResponse({
        "request": request_preview,
        "request_example": endpoint.request_schema.example(),
        "response_example": endpoint.response_schema.example(),
    })


@router.post("/api/user_endpoints")
async def create_user_endpoint(request: Request):
    try:
        endpoint = request.app.state.user_endpoints.create(await _body(request))
        return JSONResponse(endpoint.public_dict(), status_code=201)
    except Exception as exc:
        return JSONResponse({"error": _error_message(exc)}, status_code=400)


@router.put("/api/user_endpoints/{endpoint_id}")
async def update_user_endpoint(request: Request, endpoint_id: str):
    try:
        endpoint = request.app.state.user_endpoints.update(
            endpoint_id,
            await _body(request),
        )
        return JSONResponse(endpoint.public_dict())
    except KeyError:
        return JSONResponse({"error": "Endpoint not found"}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": _error_message(exc)}, status_code=400)


@router.delete("/api/user_endpoints/{endpoint_id}")
async def delete_user_endpoint(request: Request, endpoint_id: str):
    try:
        request.app.state.user_endpoints.delete(endpoint_id)
        return JSONResponse({"status": True})
    except KeyError:
        return JSONResponse({"error": "Endpoint not found"}, status_code=404)


@router.post("/api/user_endpoints/{endpoint_id}/test")
async def test_user_endpoint(request: Request, endpoint_id: str):
    """Send one real request through a saved endpoint."""
    from fox_reader.translate.custom_endpoint import describe_request

    store = request.app.state.user_endpoints
    endpoint = store.get(endpoint_id)

    if endpoint is None:
        return JSONResponse({"error": "Endpoint not found"}, status_code=404)

    body = await _body(request)
    text = str(body.get("text") or SAMPLE_TEXT).strip() or SAMPLE_TEXT
    language = _language_of(body)

    if language not in endpoint.languages:
        language = endpoint.languages[0]

    try:
        # Explicit empties: this preview describes the real request sent
        # below, which carries no history -- unlike the editor preview, which
        # fills the lists with samples to show their shape.
        request_preview = describe_request(
            endpoint,
            text=text,
            source_lang=language,
            context=[],
            character_info=[],
            context_character_links=[],
        )
    except Exception as exc:
        return JSONResponse({"error": _error_message(exc)}, status_code=400)

    result = await request.app.state.translate.custom(text, endpoint_id, language)

    return JSONResponse({
        "status": result.code == 200,
        "code": result.code,
        "language": language,
        "sent": text,
        "translated": result.data,
        "alternatives": result.alternatives,
        "message": result.message,
        "request": request_preview,
    })


@router.put("/api/user_endpoints/{endpoint_id}/active")
async def activate_endpoint(request: Request, endpoint_id: str):
    body = await _body(request)
    language = _language_of(body)

    if not language:
        return JSONResponse({"error": "language is required"}, status_code=400)

    try:
        active = request.app.state.user_endpoints.set_active(
            language,
            endpoint_id,
            True,
        )
    except KeyError:
        return JSONResponse({"error": "Endpoint not found"}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": _error_message(exc)}, status_code=400)

    return JSONResponse({
        "status": True,
        "language": language,
        "endpoint_id": endpoint_id,
        "active": active,
        "max_active_per_language": MAX_ACTIVE_PER_LANGUAGE,
    })


@router.delete("/api/user_endpoints/{endpoint_id}/active")
async def deactivate_endpoint(request: Request, endpoint_id: str):
    body = await _body(request)
    language = _language_of(body)

    if not language:
        language = str(request.query_params.get("language") or "").strip().lower()

    if not language:
        return JSONResponse({"error": "language is required"}, status_code=400)

    try:
        active = request.app.state.user_endpoints.set_active(
            language,
            endpoint_id,
            False,
        )
    except KeyError:
        return JSONResponse({"error": "Endpoint not found"}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": _error_message(exc)}, status_code=400)

    return JSONResponse({
        "status": True,
        "language": language,
        "endpoint_id": endpoint_id,
        "active": active,
    })
