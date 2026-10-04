"""System routes — session identity, readiness, progress, and shutdown."""
from __future__ import annotations

import ipaddress
import logging
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])

#: What ``/api/health`` puts in its answer so the launcher can tell Fox Reader
#: apart from whatever else might have taken the port.
APP_MARKER = "fox-reader"

#: Set by the launcher to a fresh random value per run, and sent back in
#: ``X-Fox-Reader-Token``. Without it the endpoint is open to anything that can
#: reach the loopback port -- including a page in the user's browser, which can
#: POST cross-origin even though it cannot read the reply. A custom header
#: cannot be sent cross-origin without a preflight, so the token closes that.
SHUTDOWN_TOKEN_ENV = "FOX_READER_SHUTDOWN_TOKEN"
SHUTDOWN_TOKEN_HEADER = "x-fox-reader-token"


def _session_id(request: Request) -> dict:
    return {"session_id": request.app.state.session_id}


@router.get("/api/session")
async def get_session(request: Request):
    return _session_id(request)


@router.get("/api/health")
async def get_health(request: Request):
    """Answers as soon as the process is serving, and never does any work.

    The launcher waits on this to decide the backend is up, and treats a 200
    carrying the marker as proof that the port belongs to Fox Reader rather than
    to something else. It answers only after the lifespan startup has finished,
    because uvicorn does not open the listening socket until then -- so a
    successful request here means every model is loaded, not merely that a
    process exists.
    """
    return {"app": APP_MARKER, **_session_id(request)}


@router.get("/api/progress/{name}")
async def get_progress(request: Request, name: str):
    """What a long job is currently doing.

    The jobs themselves run in worker threads and answer with a file, so they
    cannot report through their own response; the UI polls this instead. Reading
    is a dict copy under a lock, so polling it every 400 ms costs nothing and can
    never be what makes the health check late.
    """
    registry = getattr(request.app.state, "progress", None)
    if registry is None:
        return {"name": name, "running": False, "message": "", "lines": [],
                "done": 0, "total": 0, "idle": True}
    return registry.snapshot(name)


def _is_loopback(request: Request) -> bool:
    client = request.client
    if client is None or not client.host:
        return False
    try:
        return ipaddress.ip_address(client.host).is_loopback
    except ValueError:
        # A hostname rather than an address, which only happens behind a proxy
        # that rewrote it. Not something to trust with a shutdown.
        return client.host.lower() == "localhost"


def _token_ok(request: Request) -> bool:
    expected = os.environ.get(SHUTDOWN_TOKEN_ENV, "")
    if not expected:
        return True

    import hmac

    supplied = request.headers.get(SHUTDOWN_TOKEN_HEADER, "")
    return hmac.compare_digest(supplied, expected)


@router.post("/api/shutdown")
async def post_shutdown(request: Request):
    """Stop the backend in order: finish in-flight work, unload, clear cache.

    This is how the launcher closes Fox Reader down when its console window is
    closed. It answers before anything stops, so the caller learns the request
    was accepted rather than having the connection dropped underneath it.
    """
    if not _is_loopback(request):
        logger.warning("Rejected a shutdown request from %s.", request.client)
        return JSONResponse({"error": "Shutdown is local-only."}, status_code=403)

    if not _token_ok(request):
        logger.warning("Rejected a shutdown request with a bad token.")
        return JSONResponse({"error": "Invalid token."}, status_code=403)

    from fox_reader import runtime

    graceful = runtime.stop(0.2)
    return JSONResponse({"status": "closing", "graceful": graceful})
