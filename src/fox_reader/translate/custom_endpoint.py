from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.parse
from typing import Any

import httpx

from fox_reader import endpoint_crypto
from fox_reader.endpoint_schema import apply_placeholders
from fox_reader.models.responses import TranslationResult
from fox_reader.translate._http import decompress_response, parse_json_response
from fox_reader.translate.characters import (
    normalize_character_info,
    normalize_character_links,
)
from fox_reader.translate.context import normalize_context
from fox_reader.user_endpoints import UserEndpoint

logger = logging.getLogger(__name__)

# Statuses worth trying again: throttling and transient server failures.
_RETRY_STATUSES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

_BACKOFF_BASE = 0.4
_BACKOFF_CAP = 4.0

_DEFAULT_CONTENT_TYPES: dict[str, str] = {
    "json": "application/json",
    "form": "application/x-www-form-urlencoded",
    "text": "text/plain; charset=utf-8",
}

_BODY_SNIPPET_LIMIT = 300

#: Preview samples: what ``describe_request`` (the editor/test preview) fills
#: the list sources with when the caller provides none, so the preview shows
#: the real wire shape instead of three empty arrays. The live translate path
#: (``prepare_request`` via ``translate_by_custom_endpoint``) never uses
#: these -- it sends the actual request data, empty when there is none.
PREVIEW_CONTEXT: list[list[str]] = [
    ["こんにちは", "Hello"],
    ["おはようございます", "Good morning"],
]

PREVIEW_CHARACTER_INFO: list[dict[str, Any]] = [
    {
        "meta_id": "a1b2c3d4",
        "name_en": "Aiko",
        "name_ja": "愛子",
        "gender": "female",
        "alias_en": "Sis",
        "alias_ja": "お姉ちゃん",
    },
]

#: One entry per preview context pair, positionally aligned.
PREVIEW_CONTEXT_CHARACTER_LINKS: list[str | None] = ["a1b2c3d4", None]


def _scalar_to_str(value: Any) -> str:
    """Render a JSON scalar the way HTTP APIs expect to read it."""
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float)):
        return str(value)

    # A nested object inside a flat payload: send it as compact JSON rather
    # than Python's repr.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _query_pairs(
    payload: dict[str, Any],
    doseq: bool,
) -> list[tuple[str, str]]:
    """Turn a flat payload into ordered key/value pairs.

    With ``doseq`` a list becomes repeated keys (``dt=t&dt=at``); without it
    the list is sent as a single JSON-ish value, which some APIs expect.
    """
    pairs: list[tuple[str, str]] = []

    for key, value in payload.items():
        if isinstance(value, (list, tuple)):
            if doseq:
                pairs.extend((key, _scalar_to_str(item)) for item in value)
            else:
                pairs.append((key, _scalar_to_str(list(value))))
        else:
            pairs.append((key, _scalar_to_str(value)))

    return pairs


def _encode_pairs(pairs: list[tuple[str, str]]) -> str:
    return urllib.parse.urlencode(pairs, doseq=False)


def _append_query(url: str, encoded: str) -> str:
    if not encoded:
        return url

    return url + ("&" if "?" in url else "?") + encoded


def _body_snippet(content: bytes, encoding: str) -> str:
    try:
        raw = decompress_response(content, encoding)
    except Exception:
        raw = content

    text = raw.decode("utf-8", errors="replace").strip()
    text = " ".join(text.split())

    if len(text) > _BODY_SNIPPET_LIMIT:
        return text[:_BODY_SNIPPET_LIMIT] + "…"

    return text


def _decode_text_response(response: httpx.Response) -> str:
    encoding = response.headers.get("Content-Encoding", "").lower()

    try:
        raw = decompress_response(response.content, encoding)
    except Exception:
        raw = response.content

    charset = response.charset_encoding or "utf-8"

    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


def _encrypt_text(endpoint: UserEndpoint, text: str) -> str:
    """Seal the selection before it can reach the wire."""
    try:
        return endpoint_crypto.encrypt(
            text,
            secret=endpoint.encryption.key,
            encoding=endpoint.encryption.key_encoding,
        )
    except endpoint_crypto.EncryptionError as exc:
        raise ValueError(str(exc)) from exc


def _decrypt_text(endpoint: UserEndpoint, value: str) -> str:
    return endpoint_crypto.decrypt(
        value,
        secret=endpoint.encryption.key,
        encoding=endpoint.encryption.key_encoding,
        max_age=endpoint.encryption.max_age,
    )


def _encrypt_pairs(
    endpoint: UserEndpoint,
    pairs: list[list[str]],
) -> list[list[str]]:
    """Seal every side of every context pair, keeping the pairs shape.

    Each source and each translation is sealed on its own -- the same way
    response fragments are opened one by one -- so the API sees the familiar
    [[token, token], ...] shape and opens each string with the shared key. A
    sealing failure aborts the request rather than leaking one side in
    plaintext next to sealed siblings.
    """
    return [
        [_encrypt_text(endpoint, source), _encrypt_text(endpoint, english)]
        for source, english in pairs
    ]


def _encrypt_characters(
    endpoint: UserEndpoint,
    characters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Seal the identifying fields of every roster entry, keeping its shape.

    Each name/alias is sealed on its own -- the same way response fragments
    are opened one by one -- so the API opens familiar string fields with the
    shared key. ``meta_id`` and ``gender`` stay plaintext on purpose: the ids
    are the structural join keys for ``context_character_links`` (sealing one
    side would break the join the API performs after opening the names), and
    a gender word carries nothing worth a token. A sealing failure aborts the
    request rather than leaking one field in plaintext next to sealed
    siblings.
    """
    sealed: list[dict[str, Any]] = []

    for char in characters:
        entry = dict(char)

        for field in ("name_en", "name_ja", "alias_en", "alias_ja"):
            value = entry.get(field)

            if value is None:
                continue

            entry[field] = _encrypt_text(endpoint, str(value))

        sealed.append(entry)

    return sealed


def _decrypt_alternatives(
    endpoint: UserEndpoint,
    values: list[str],
) -> list[str]:
    """Open a list of alternatives, dropping any that will not open.

    A bad alternative is skipped rather than failing the translation — the main
    text has already been read by this point. Deduplication happens after
    opening, not before: a fresh IV per token means two tokens for the same
    words look nothing alike.
    """
    out: list[str] = []

    for value in values:
        if not value:
            continue

        try:
            opened = _decrypt_text(endpoint, value)
        except endpoint_crypto.EncryptionError as exc:
            logger.debug(
                "Dropping an alternative from %s that would not decrypt: %s",
                endpoint.id,
                exc,
            )
            continue

        if opened and opened not in out:
            out.append(opened)

    return out


def _schema_uses_character_lists(schema: Any) -> bool:
    """Whether a request schema carries either character list node.

    Anything answering ``count_source`` works -- including a schema build
    that predates the list sources (it simply reports zero for both), which
    is what keeps a partially-updated install translating instead of 500ing.
    Never raises: an unaskable schema reads as "no lists".
    """
    try:
        return bool(
            schema.count_source("character_info")
            or schema.count_source("context_character_links")
        )
    except Exception:
        return False


def prepare_request(
    endpoint: UserEndpoint,
    *,
    text: str,
    source_lang: str,
    target_lang: str,
    context: object = None,
    character_info: object = None,
    context_character_links: object = None,
) -> dict[str, Any]:
    """Resolve an endpoint into the exact HTTP request it will send.

    Returned keys: ``method``, ``url``, ``headers``, ``body_format`` and
    ``body`` (a str, or None when the payload rode along in the query
    string). Everything is already placeholder-expanded.

    ``context`` is the earlier [[source, english], ...] pairs. It only
    reaches the wire when the request schema carries a Context node -- other
    endpoints build byte-identical requests with or without it. When the
    endpoint encrypts requests, every side of every pair is sealed with the
    same key as the selection text, so the API opens familiar
    [[token, token], ...] pairs instead of plaintext dialogue.

    ``character_info`` is the roster snapshot ([{meta_id, name_en, name_ja,
    gender, alias_en, alias_ja}, ...]) and ``context_character_links`` one
    meta_id (or None) per context pair. Each only reaches the wire when the
    request schema carries the matching node -- Character Info and Context
    Links respectively -- and malformed rows degrade to empty lists, never to
    a failed request. When the endpoint encrypts requests, the four
    name/alias fields are sealed (see ``_encrypt_characters``); the ids stay
    plaintext so the API can still join links to roster entries.

    When the endpoint encrypts requests, the text is sealed here — once, at the
    top — so every place a selection can surface carries the token instead of
    the words: the body, a ``<TEXT>`` in the hostname, a header, a query
    parameter. A URL is the most heavily logged part of a request, so leaving
    plaintext there would give most of the feature away. Doing it here also
    means the editor's preview shows exactly what goes on the wire.
    """
    if endpoint.encrypts_request:
        text = _encrypt_text(endpoint, text)

    pairs = normalize_context(context)

    try:
        characters = list(normalize_character_info(character_info).values())
    except Exception:
        characters = []

    try:
        links = normalize_character_links(context_character_links, len(pairs))
    except Exception:
        links = [None] * len(pairs)

    if endpoint.encrypts_request and pairs:
        pairs = _encrypt_pairs(endpoint, pairs)

    if endpoint.encrypts_request and characters:
        characters = _encrypt_characters(endpoint, characters)

    if _schema_uses_character_lists(endpoint.request_schema):
        payload = endpoint.request_schema.build(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=pairs,
            character_info=characters,
            context_character_links=links,
        )
    else:
        # Legacy shape: schemas without the new nodes build byte-identical
        # requests with or without them -- and the old keywords are never
        # passed at all, so this also stays working against a schema build
        # that predates the list sources.
        payload = endpoint.request_schema.build(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=pairs,
        )

    url = endpoint.build_url(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    headers = endpoint.resolve_headers(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    static_query = endpoint.resolve_query(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    wire = endpoint.wire_format
    body: str | None = None

    if wire == "query":
        if not isinstance(payload, dict):
            raise ValueError(
                "A query request needs a dict at the root of the request "
                "schema."
            )

        merged: dict[str, Any] = dict(static_query)
        merged.update(payload)

        url = _append_query(url, _encode_pairs(_query_pairs(merged, endpoint.doseq)))

    else:
        if static_query:
            url = _append_query(
                url,
                _encode_pairs(_query_pairs(static_query, False)),
            )

        if wire == "form":
            if not isinstance(payload, dict):
                raise ValueError(
                    "A form request needs a dict at the root of the request "
                    "schema."
                )

            body = _encode_pairs(_query_pairs(payload, endpoint.doseq))

        elif wire == "json":
            body = json.dumps(payload, ensure_ascii=False)

        else:  # text
            body = (
                payload
                if isinstance(payload, str)
                else json.dumps(payload, ensure_ascii=False)
            )

    if body is not None:
        has_content_type = any(
            key.lower() == "content-type" for key in headers
        )

        if not has_content_type:
            headers["Content-Type"] = _DEFAULT_CONTENT_TYPES[wire]

    return {
        "method": endpoint.method,
        "url": url,
        "headers": headers,
        "body_format": wire,
        "body": body,
    }


def describe_request(
    endpoint: UserEndpoint,
    *,
    text: str = "",
    source_lang: str = "",
    target_lang: str | None = None,
    context: object = None,
    character_info: object = None,
    context_character_links: object = None,
) -> dict[str, Any]:
    """A preview of the outgoing request, with no network activity.

    List sources the caller leaves out are filled with the preview samples,
    so the preview shows the shape the API will actually receive (pairs, a
    roster entry, per-pair speaker ids) instead of empty arrays. Explicitly
    passed lists -- including empty ones -- are honoured as given.
    """
    language = (source_lang or "").strip().lower()

    if not language and endpoint.languages:
        language = endpoint.languages[0]

    if context is None:
        context = PREVIEW_CONTEXT

    if character_info is None:
        character_info = PREVIEW_CHARACTER_INFO

    if context_character_links is None:
        context_character_links = PREVIEW_CONTEXT_CHARACTER_LINKS

    return prepare_request(
        endpoint,
        text=text,
        source_lang=endpoint.source_code(language),
        target_lang=target_lang or endpoint.target_language,
        context=context,
        character_info=character_info,
        context_character_links=context_character_links,
    )


def _failure(
    endpoint: UserEndpoint,
    *,
    code: int,
    request_id: int,
    source_lang: str,
    target_lang: str,
    message: str,
) -> TranslationResult:
    return TranslationResult(
        code=code,
        id=request_id,
        data="",
        source_lang=source_lang.upper(),
        target_lang=target_lang.upper(),
        method=endpoint.name,
        message=message,
    )


async def _send(
    endpoint: UserEndpoint,
    request: dict[str, Any],
) -> httpx.Response:
    async with httpx.AsyncClient(
        timeout=endpoint.timeout,
        http2=endpoint.http2,
        follow_redirects=True,
    ) as client:
        return await client.request(
            request["method"],
            request["url"],
            headers=request["headers"] or None,
            content=(
                request["body"].encode("utf-8")
                if request["body"] is not None
                else None
            ),
        )


def _retry_delay(attempt: int, response: httpx.Response | None) -> float:
    if response is not None:
        raw = response.headers.get("Retry-After", "").strip()

        if raw:
            try:
                hinted = float(raw)
            except ValueError:
                hinted = 0.0

            if 0 < hinted <= _BACKOFF_CAP:
                return hinted

    return min(_BACKOFF_BASE * (2**attempt), _BACKOFF_CAP)


async def translate_by_custom_endpoint(
    endpoint: UserEndpoint,
    *,
    text: str,
    source_lang: str,
    target_lang: str | None = None,
    context: object = None,
    character_info: object = None,
    context_character_links: object = None,
) -> TranslationResult:
    request_id = int(time.time() * 1000)

    language = (source_lang or "").strip().lower()
    wire_target = (target_lang or endpoint.target_language).strip() or "en"
    wire_source = endpoint.source_code(language)

    def fail(code: int, message: str) -> TranslationResult:
        return _failure(
            endpoint,
            code=code,
            request_id=request_id,
            source_lang=language,
            target_lang=wire_target,
            message=message,
        )

    if not text or not text.strip():
        return fail(400, "Nothing to translate")

    # Deliberately measured on the plaintext: the guard is about how much text
    # the user selected, not how much it expands to once sealed.
    if len(text) > endpoint.max_text_length:
        return fail(
            413,
            f"Text is {len(text)} characters; "
            f"'{endpoint.name}' accepts at most {endpoint.max_text_length}",
        )

    try:
        request = prepare_request(
            endpoint,
            text=text,
            source_lang=wire_source,
            target_lang=wire_target,
            context=context,
            character_info=character_info,
            context_character_links=context_character_links,
        )
    except Exception as exc:
        return fail(500, f"Could not build the request: {exc}")

    response: httpx.Response | None = None
    last_error: str | None = None

    for attempt in range(endpoint.retries + 1):
        if attempt:
            await asyncio.sleep(_retry_delay(attempt - 1, response))

        try:
            response = await _send(endpoint, request)
        except httpx.TimeoutException:
            response = None
            last_error = (
                f"Timed out after {endpoint.timeout:g}s"
                f" ({endpoint.method} {endpoint.build_url()})"
            )
            continue
        except httpx.HTTPError as exc:
            response = None
            last_error = f"Could not reach the endpoint: {exc}"
            continue

        if response.status_code in _RETRY_STATUSES and attempt < endpoint.retries:
            continue

        break

    if response is None:
        code = 504 if last_error and "Timed out" in last_error else 503
        return fail(code, last_error or "Request failed")

    encoding = response.headers.get("Content-Encoding", "").lower()
    ok = 200 <= response.status_code < 300

    if endpoint.response_format == "text":
        body = _decode_text_response(response).strip()

        if not ok:
            return fail(
                response.status_code,
                f"HTTP {response.status_code}"
                + (f" — {body[:_BODY_SNIPPET_LIMIT]}" if body else ""),
            )

        if not body:
            return fail(502, "Endpoint returned an empty body")

        if endpoint.decrypts_response:
            try:
                body = _decrypt_text(endpoint, body).strip()
            except endpoint_crypto.EncryptionError as exc:
                return fail(502, str(exc))

            if not body:
                return fail(502, "The decrypted response was empty")

        return TranslationResult(
            code=200,
            id=request_id,
            data=body,
            source_lang=language.upper(),
            target_lang=wire_target.upper(),
            method=endpoint.name,
        )

    result = parse_json_response(response.content, encoding, default=None)

    if result is None:
        snippet = _body_snippet(response.content, encoding)

        return fail(
            response.status_code if not ok else 502,
            f"Endpoint returned invalid JSON (HTTP {response.status_code})"
            + (f": {snippet}" if snippet else ""),
        )

    # An API's own error message beats a bare status code, so read it first.
    try:
        error = endpoint.response_schema.extract_error(result)
    except Exception as exc:
        logger.debug("Error extraction failed for %s: %s", endpoint.id, exc)
        error = None

    if error:
        return fail(
            response.status_code if not ok else 502,
            str(error),
        )

    if not ok:
        snippet = _body_snippet(response.content, encoding)

        return fail(
            response.status_code,
            f"HTTP {response.status_code}" + (f" — {snippet}" if snippet else ""),
        )

    try:
        parts = endpoint.response_schema.extract_text_parts(result)
    except Exception as exc:
        return fail(502, f"Could not read the response: {exc}")

    if endpoint.decrypts_response and parts:
        # Each fragment is opened on its own, then joined. That covers both the
        # ordinary case of one sealed field and an endpoint that seals every
        # sentence chunk separately — joining tokens first would break the
        # second one.
        try:
            parts = [_decrypt_text(endpoint, part) for part in parts]
        except endpoint_crypto.EncryptionError as exc:
            return fail(502, str(exc))

    translated = endpoint.text_join.join(parts) if parts else None

    if translated is None or not translated.strip():
        snippet = _body_snippet(response.content, encoding)

        return fail(
            502,
            "The response did not contain the configured Translated Text "
            "node" + (f": {snippet}" if snippet else ""),
        )

    try:
        candidates = endpoint.response_schema.extract_alternatives(result)
    except Exception as exc:
        logger.debug(
            "Alternative extraction failed for %s: %s",
            endpoint.id,
            exc,
        )
        candidates = []

    if endpoint.decrypts_response:
        candidates = _decrypt_alternatives(endpoint, candidates)

    alternatives = [
        item for item in candidates if item and item != translated
    ]

    return TranslationResult(
        code=200,
        id=request_id,
        data=translated,
        source_lang=language.upper(),
        target_lang=wire_target.upper(),
        method=endpoint.name,
        alternatives=alternatives,
    )
