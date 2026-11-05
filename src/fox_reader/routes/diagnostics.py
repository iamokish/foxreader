"""Local browser diagnostics. Never accepts remote pages or unbounded payloads."""
from __future__ import annotations

import ipaddress
import json
import logging
import time
from collections import deque

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from fox_reader.logging_setup import redact
from fox_reader.routes.system import _is_loopback

router = APIRouter(tags=["system"])
logger = logging.getLogger("fox_reader.frontend")
MAX_BODY = 24 * 1024
MAX_BATCH = 10
MAX_EVENTS_PER_MINUTE = 60


def _local_page(request: Request) -> bool:
    if not _is_loopback(request):
        return False
    hostname = request.url.hostname or ""
    try:
        local = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        local = hostname.lower() == "localhost"
    # Fixed custom header also prevents cross-origin simple form POSTs on
    # browsers that omit Origin. CORS preflight is intentionally unsupported.
    origin = request.headers.get("origin")
    return (
        local
        and request.headers.get("x-fox-reader-client") == "frontend"
        and (origin is None or origin == str(request.base_url).rstrip("/"))
        and request.headers.get("sec-fetch-site", "same-origin") in {"same-origin", "none"}
    )


@router.post("/api/frontend-log")
async def frontend_log(request: Request):
    if not _local_page(request):
        return JSONResponse({"error": "Diagnostics are local-only."}, status_code=403)
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        return JSONResponse({"error": "Expected JSON."}, status_code=415)
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > MAX_BODY:
            return JSONResponse({"error": "Diagnostics are too large."}, status_code=413)
        raw.extend(chunk)
    try:
        events = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return JSONResponse({"error": "Invalid diagnostics."}, status_code=400)
    if not isinstance(events, list) or not 1 <= len(events) <= MAX_BATCH:
        return JSONResponse({"error": "Invalid diagnostics batch."}, status_code=400)
    messages: list[tuple[int, str]] = []
    for item in events:
        if not isinstance(item, dict) or set(item) - {"level", "message", "source"}:
            return JSONResponse({"error": "Invalid diagnostic event."}, status_code=400)
        level, message, source = item.get("level"), item.get("message"), item.get("source", "")
        if (level not in ("info", "warn", "error") or not isinstance(message, str)
                or not 1 <= len(message) <= 2000 or not isinstance(source, str) or len(source) > 200):
            return JSONResponse({"error": "Invalid diagnostic event."}, status_code=400)
        # One physical line per event, even when an error contains user input.
        safe = redact(f"{source}: {message}" if source else message)
        safe = safe.replace("\r", "\\r").replace("\n", "\\n").replace("\x00", "")
        messages.append(({"info": logging.INFO, "warn": logging.WARNING, "error": logging.ERROR}[level], safe))
    # This is reached only on the ASGI loop; no await while changing the budget.
    recent = getattr(request.app.state, "frontend_log_times", None)
    if recent is None:
        recent = request.app.state.frontend_log_times = deque()
    now = time.monotonic()
    while recent and recent[0] <= now - 60:
        recent.popleft()
    if len(recent) + len(messages) > MAX_EVENTS_PER_MINUTE:
        return JSONResponse({"error": "Diagnostics rate limit reached."}, status_code=429,
                            headers={"Retry-After": "60"})
    recent.extend([now] * len(messages))

    def write() -> None:
        for level, message in messages:
            logger.log(level, message)

    await run_in_threadpool(write)
    return Response(status_code=204)
