"""Official DeepL API client (v2).

Replaces the previous reverse-engineered oneshot/extension endpoint with the
documented API: https://developers.deepl.com/docs/getting-started/quickstart

- Free plans use ``https://api-free.deepl.com``, Pro uses
  ``https://api.deepl.com``. A free key ends with ``:fx`` — the same rule the
  official ``deepl-python`` client uses — so the base URL is derived from the
  token itself.
- Auth is ``Authorization: DeepL-Auth-Key <token>`` on every call.
- Validation is ``GET /v2/usage`` (no quota consumed). Translating
  ``こんにちは`` JA -> EN is equivalent, but usage is cheaper, so validation
  uses usage.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

from fox_reader.models.responses import TranslationResult
from fox_reader.translate.lang_map import (
    DEEPL_SOURCE_LANGS,
    DEEPL_TARGET_LANGS,
    deepl_source_for_fox_language,
    get_supported_langs,
    normalize_fox_language,
    resolve_lang,
)

# =============================================================================
# Constants
# =============================================================================

DEEPL_FREE_BASE = "https://api-free.deepl.com"
DEEPL_PRO_BASE = "https://api.deepl.com"

_REQUEST_TIMEOUT = 20.0
_VALIDATE_TIMEOUT = 10.0

#: Official per-request body cap for /v2/translate.
_MAX_BODY_BYTES = 128 * 1024

_USER_AGENT = "FoxReader/1.0"


# =============================================================================
# Base URL selection
# =============================================================================


def is_free_token(token: str | None) -> bool:
    """Whether a DeepL key belongs to the free plan (ends with ``:fx``)."""
    return bool(token) and token.strip().endswith(":fx")


def deepl_base_url(token: str | None) -> str:
    """Pick the official base URL for a token. Free -> api-free, else api."""
    return DEEPL_FREE_BASE if is_free_token(token) else DEEPL_PRO_BASE


def _auth_headers(token: str) -> dict:
    return {
        "Authorization": f"DeepL-Auth-Key {token.strip()}",
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
    }


# =============================================================================
# Public helpers (kept for compatibility)
# =============================================================================


def get_supported_target_langs() -> str:
    return get_supported_langs(DEEPL_TARGET_LANGS)


def get_supported_source_langs() -> str:
    return get_supported_langs(DEEPL_SOURCE_LANGS)


def resolve_target_lang(code: str) -> str:
    return resolve_lang(code, DEEPL_TARGET_LANGS, "target_lang")


def resolve_source_lang(code: str) -> str:
    if not code or code.strip().lower() == "auto":
        return ""
    # Fox language groups ("japanese") are accepted alongside DeepL codes.
    try:
        return deepl_source_for_fox_language(code)
    except ValueError:
        pass
    return resolve_lang(code, DEEPL_SOURCE_LANGS, "source_lang")


def _error(code: int, request_id: int, message: str) -> TranslationResult:
    return TranslationResult(
        code=code,
        id=request_id,
        data="",
        source_lang="",
        target_lang="",
        method="",
        message=message,
    )


def _deepl_error_message(status: int, payload: dict | list | str | None) -> str:
    """Best-effort message from a DeepL error body, which is {"message": ...}."""
    detail = ""
    if isinstance(payload, dict):
        msg = payload.get("message")
        if isinstance(msg, str) and msg.strip():
            detail = msg.strip()
    if status == 403:
        base = "Invalid DeepL API token (403). Check the key and its plan."
        if detail and "invalid" not in detail.lower():
            return f"{base} {detail}"
        return base
    if detail:
        return detail
    if status == 456:
        return "DeepL character limit reached (456/quota exceeded)."
    if status == 429:
        return "Too many DeepL requests (429). Slow down and retry."
    if status == 413:
        return "DeepL request too large (413)."
    if status == 400:
        return "Bad DeepL request (400). Check source/target languages."
    if status == 404:
        return "DeepL endpoint not found (404)."
    return f"DeepL request failed with status code: {status}"


# =============================================================================
# Validation (for Settings -> Save with validation)
# =============================================================================


async def validate_deepl_token(token: str | None) -> tuple[bool, str]:
    """Check a DeepL key via the official ``GET /v2/usage`` endpoint.

    Returns (is_valid, message). Empty tokens are never valid here — the
    settings route treats empty as "clear this token" without calling this.
    ``456 quota exceeded`` still means the key itself is correct, so it counts
    as valid (translation will report quota until it resets).
    """
    token = (token or "").strip()
    if not token:
        return False, "DeepL API token is empty."

    url = f"{deepl_base_url(token)}/v2/usage"
    try:
        async with httpx.AsyncClient(timeout=_VALIDATE_TIMEOUT) as client:
            response = await client.get(url, headers=_auth_headers(token))
    except httpx.TimeoutException:
        return False, f"DeepL validation timed out after {_VALIDATE_TIMEOUT}s."
    except httpx.HTTPError as exc:
        logger.debug("DeepL validation transport error: %s", exc)
        return False, f"Could not reach DeepL to validate the token: {exc}"

    if response.status_code == 200:
        return True, "DeepL token is valid."
    if response.status_code == 456:
        return True, "DeepL token is valid but its character limit is reached."
    if response.status_code in (401, 403):
        return False, "Invalid DeepL API token (rejected with 403)."

    try:
        payload = response.json()
    except Exception:
        payload = None
    return False, _deepl_error_message(response.status_code, payload)


async def test_deepl_translation(
    token: str | None,
    text: str = "こんにちは",
    source_lang: str = "JA",
    target_lang: str = "EN-US",
) -> tuple[bool, str]:
    """Translate a tiny sample (こんにちは -> Hello) to prove the key works.

    Kept as the literal "konnichiwa to hello" check from the requirements.
    Prefer :func:`validate_deepl_token` on save (it costs no quota); this is
    for manual testing and returns (ok, translated_text_or_error).
    """
    result = await translate_by_deepl(
        source_lang, target_lang, text, api_token=token
    )
    if result.code == 200:
        return True, result.data
    return False, result.message or f"DeepL test failed ({result.code})."


# =============================================================================
# Translation
# =============================================================================


async def translate_by_deepl(
    source_lang: str,
    target_lang: str,
    text: str,
    api_token: Optional[str] = None,
) -> TranslationResult:
    """Translate via the official DeepL ``POST /v2/translate`` endpoint."""
    request_id = int(time.time() * 1000)

    if not text or not text.strip():
        return _error(404, request_id, "No text to translate")

    token = (api_token or "").strip()
    if not token:
        return _error(
            401,
            request_id,
            "DeepL API token is not configured. Add one in Settings.",
        )

    # The reader always translates *to* English; Fox groups ("japanese") and
    # DeepL codes ("JA") are both accepted as the source.
    try:
        if not target_lang or not target_lang.strip():
            resolved_target = "EN-US"
        else:
            try:
                resolved_target = resolve_target_lang(target_lang.strip())
            except Exception:
                # Be lenient: "english"/"en" mean US English here.
                if target_lang.strip().lower() in ("en", "english"):
                    resolved_target = "EN-US"
                else:
                    raise
        if not source_lang or source_lang.strip().lower() in ("", "auto"):
            resolved_source = ""
        else:
            try:
                resolved_source = deepl_source_for_fox_language(source_lang)
            except ValueError:
                resolved_source = resolve_source_lang(source_lang.strip())
                # Source codes carry no region ("EN", not "EN-US").
                if resolved_source in ("EN-US", "EN-GB"):
                    resolved_source = "EN"
    except Exception as exc:
        # resolve_lang raises HTTPException with .detail; ValueError otherwise.
        detail = getattr(exc, "detail", None) or str(exc)
        return _error(400, request_id, str(detail))

    # Fox target is always English; normalise the region for the API.
    if resolved_target.upper() in ("EN", "ENGLISH"):
        resolved_target = "EN-US"

    body: dict = {"text": [text], "target_lang": resolved_target}
    if resolved_source:
        body["source_lang"] = resolved_source

    import json as _json

    if len(_json.dumps(body).encode("utf-8")) > _MAX_BODY_BYTES:
        return _error(
            413,
            request_id,
            f"text exceeds DeepL request size limit ({_MAX_BODY_BYTES // 1024} KiB)",
        )

    url = f"{deepl_base_url(token)}/v2/translate"
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.post(url, headers=_auth_headers(token), json=body)
    except httpx.TimeoutException:
        return _error(
            504, request_id, f"upstream DeepL request timed out after {_REQUEST_TIMEOUT}s"
        )
    except httpx.HTTPError as exc:
        logger.debug("DeepL transport error: %s", exc)
        return _error(503, request_id, f"Could not reach DeepL: {exc}")

    if response.status_code != 200:
        try:
            payload = response.json()
        except Exception:
            payload = None
        # Map the official error codes to TranslationResult codes 1:1 so the
        # settings validator and the reader show the same wording.
        code = response.status_code
        if code not in (400, 401, 403, 404, 413, 429, 456, 500, 503):
            code = 503
        return _error(code, request_id, _deepl_error_message(response.status_code, payload))

    try:
        data = response.json()
    except Exception as exc:
        return _error(503, request_id, f"Invalid DeepL response: {exc}")

    translations = data.get("translations", []) if isinstance(data, dict) else []
    if not translations or not isinstance(translations[0], dict) or not translations[0].get("text"):
        return _error(503, request_id, "DeepL returned no translation.")

    first = translations[0]
    detected = first.get("detected_source_language", "") or resolved_source or source_lang
    try:
        fox_group = normalize_fox_language(source_lang)
        src_display = {"japanese": "JA", "chinese": "ZH", "korean": "KO"}[fox_group]
    except ValueError:
        src_display = str(detected).upper() or source_lang.upper()

    return TranslationResult(
        code=200,
        id=request_id,
        data=first["text"],
        source_lang=src_display,
        target_lang="EN",
        method="DeepL",
    )
