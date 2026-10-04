"""Official JPDB API client (https://jpdb.io API v1).

Replaces the previous ``jpdb.io/search`` HTML scraper with the documented
endpoints:

- ``POST /api/v1/ping`` — no body, ``Authorization: Bearer <token>``.
  ``200 {}`` means the key is valid, ``403 {"error": "bad_key"}`` means it is
  not. Used by Settings to validate on save.
- ``POST /api/v1/ja2en`` — ``{"text": "...", "context": ["", ""]}``.
  Returns ``{"text": "<english>", "is_truncated": false}``.

Although the docs say "Japanese to English", the endpoint also translates
Chinese/Korean input (verified: ``你好`` -> ``Hello``), so the reader keeps
the JPDB button enabled for all three language groups and always sends the
text to ``ja2en`` with an empty context pair as instructed.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

from fox_reader.models.responses import TranslationResult
from fox_reader.translate.lang_map import normalize_fox_language

# =============================================================================
# Constants
# =============================================================================

JPDB_BASE = "https://jpdb.io/api/v1"
_PING_PATH = "ping"
_JA2EN_PATH = "ja2en"

_TIMEOUT = 20.0
_VALIDATE_TIMEOUT = 10.0

#: Sent on every ja2en call, per the requirements (no surrounding sentences).
_EMPTY_CONTEXT: list[str] = ["", ""]


def _auth_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "FoxReader/1.0",
    }


def _error(code: int, request_id: int, message: str, src: str, tgt: str) -> TranslationResult:
    return TranslationResult(
        code=code,
        id=request_id,
        data="",
        source_lang=src,
        target_lang=tgt,
        method="",
        message=message,
    )


def _jpdb_error_message(status: int, payload: object) -> str:
    """Human wording for JPDB's ``{"error": ..., "error_message": ...}``."""
    code = ""
    detail = ""
    if isinstance(payload, dict):
        err = payload.get("error")
        msg = payload.get("error_message")
        if isinstance(err, str):
            code = err
        if isinstance(msg, str):
            detail = msg

    if code == "bad_key" or status == 403:
        return "Invalid JPDB API token (bad_key)."
    if code == "text_too_long":
        return "JPDB rejected the text as too long (text_too_long)."
    if code == "bad_request":
        return f"Bad JPDB request (bad_request): {detail or 'body did not match schema'}"
    if code == "too_many_requests" or status == 429:
        return "Too many JPDB requests (429). Slow down and retry."
    if code == "api_unavailable" or status == 503:
        return "JPDB API is unavailable (maintenance). Try again later."
    if code == "not_found" or status == 404:
        return "JPDB endpoint not found (404)."
    if detail:
        return f"JPDB error ({code or status}): {detail}"
    if code:
        return f"JPDB error: {code} (HTTP {status})"
    return f"JPDB request failed with status code: {status}"


# =============================================================================
# Validation (for Settings -> Save with validation)
# =============================================================================


async def validate_jpdb_token(token: str | None) -> tuple[bool, str]:
    """Check a JPDB key via the official ``POST /api/v1/ping`` endpoint."""
    token = (token or "").strip()
    if not token:
        return False, "JPDB API token is empty."

    url = f"{JPDB_BASE}/{_PING_PATH}"
    try:
        # Ping takes no body; an empty POST keeps both the real API and the
        # Stoplight mock happy.
        async with httpx.AsyncClient(timeout=_VALIDATE_TIMEOUT) as client:
            response = await client.post(url, headers=_auth_headers(token))
    except httpx.TimeoutException:
        return False, f"JPDB validation timed out after {_VALIDATE_TIMEOUT}s."
    except httpx.HTTPError as exc:
        logger.debug("JPDB validation transport error: %s", exc)
        return False, f"Could not reach JPDB to validate the token: {exc}"

    if response.status_code == 200:
        return True, "JPDB token is valid."
    if response.status_code == 403:
        return False, "Invalid JPDB API token (bad_key)."

    try:
        payload = response.json()
    except Exception:
        payload = None
    return False, _jpdb_error_message(response.status_code, payload)


# =============================================================================
# Translation
# =============================================================================


async def translate_by_jpdb(
    source_lang: str,
    target_lang: str,
    text: str,
    api_token: Optional[str] = None,
) -> TranslationResult:
    """Translate via the official ``POST /api/v1/ja2en`` endpoint."""
    request_id = int(time.time() * 1000)
    src_display = (source_lang or "").strip() or "japanese"
    tgt_display = (target_lang or "").strip() or "english"

    if not text or not text.strip():
        return _error(400, request_id, "text content is required", src_display, tgt_display)

    token = (api_token or "").strip()
    if not token:
        return _error(
            401,
            request_id,
            "JPDB API token is not configured. Add one in Settings.",
            src_display,
            tgt_display,
        )

    # The reader only translates *to* English (ja2en). Anything else is a
    # caller bug, reported as 400 rather than sent to the wrong endpoint.
    if tgt_display.lower() not in ("en", "english"):
        return _error(
            400,
            request_id,
            "Unsupported target language. Only English ('en') is supported.",
            src_display,
            tgt_display,
        )

    # Accept all three Fox groups (plus JA/ZH/KO spellings). The ja2en
    # endpoint handles non-Japanese input too, so unknown spellings still try
    # rather than failing client-side — the server is the authority.
    try:
        fox_group = normalize_fox_language(src_display)
    except ValueError:
        fox_group = None
    if fox_group is not None:
        src_display = fox_group

    url = f"{JPDB_BASE}/{_JA2EN_PATH}"
    body = {"text": text, "context": list(_EMPTY_CONTEXT)}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(url, headers=_auth_headers(token), json=body)
    except httpx.TimeoutException:
        return _error(
            504,
            request_id,
            f"upstream JPDB request timed out after {_TIMEOUT}s",
            src_display,
            tgt_display,
        )
    except httpx.HTTPError as exc:
        logger.debug("JPDB transport error: %s", exc)
        return _error(503, request_id, f"Could not reach JPDB: {exc}", src_display, tgt_display)

    if response.status_code != 200:
        try:
            payload = response.json()
        except Exception:
            payload = None
        message = _jpdb_error_message(response.status_code, payload)
        code = response.status_code
        if code not in (400, 401, 403, 404, 429, 503):
            code = 503
        return _error(code, request_id, message, src_display, tgt_display)

    try:
        data = response.json()
    except Exception as exc:
        return _error(
            503, request_id, f"Invalid JPDB response: {exc}", src_display, tgt_display
        )

    translated = data.get("text", "") if isinstance(data, dict) else ""
    if not isinstance(translated, str):
        return _error(
            503, request_id, "Invalid JPDB response shape.", src_display, tgt_display
        )

    # Empty string is a valid answer (e.g. empty/whitespace input already
    # rejected above; punctuation-only input may genuinely translate to "").
    # The old scraper returned 404 "no translation" here, but the official
    # API returns 200 with "", so pass it through.
    return TranslationResult(
        code=200,
        id=request_id,
        data=translated,
        source_lang=src_display.upper() if fox_group is None else src_display,
        target_lang="EN",
        method="JPDB",
    )
