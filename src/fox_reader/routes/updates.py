"""Local-only, explicit update checks and downloads."""
from __future__ import annotations

import ipaddress
import logging
import threading
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from fox_reader import utils
from fox_reader.routes.system import _is_loopback
from fox_reader.services.updates import SOURCE_UPDATE_MESSAGE, UpdateError, UpdateService
from fox_reader.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)
router = APIRouter(tags=["updates"])
_service_lock = threading.Lock()


def _service(request: Request) -> UpdateService:
    with _service_lock:
        if not hasattr(request.app.state, "updates"):
            from fox_reader import runtime

            request.app.state.updates = UpdateService(PROJECT_ROOT, shutdown=lambda: runtime.shutdown_soon(1.0))
        return request.app.state.updates


def _access(request: Request, mutation: bool = False) -> JSONResponse | None:
    if not utils.is_frozen:
        return JSONResponse({"error": SOURCE_UPDATE_MESSAGE}, status_code=403)
    if not _is_loopback(request):
        return JSONResponse({"error": "Updates are available only on this computer."}, status_code=403)
    try:
        host = urlsplit(str(request.base_url)).hostname or ""
        local_host = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local_host = False
    if not local_host:
        return JSONResponse({"error": "Open Fox Reader using its localhost address to manage updates."},
                            status_code=403)
    # A custom header cannot be sent by another website without an approved
    # preflight. No update route enables cross-origin requests.
    if mutation and request.headers.get("x-fox-reader-update") != "1":
        return JSONResponse({"error": "Missing update request header."}, status_code=403)
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
        return JSONResponse({"error": "Updates require a same-origin request."}, status_code=403)
    return None


@router.get("/api/updates/status")
def status(request: Request):
    if denied := _access(request):
        return denied
    return _service(request).status()


@router.get("/api/updates/check")
def check(request: Request):
    if denied := _access(request, mutation=True):
        return denied
    try:
        return _service(request).check()
    except (OSError, httpx.HTTPError, UpdateError) as error:
        logger.warning("Update check failed: %s", error)
        return JSONResponse({"error": str(error)}, status_code=502)


@router.post("/api/updates/download")
def download(request: Request):
    if denied := _access(request, mutation=True):
        return denied
    try:
        return _service(request).start()
    except UpdateError as error:
        return JSONResponse({"error": str(error)}, status_code=409)


@router.post("/api/updates/cancel")
def cancel(request: Request):
    if denied := _access(request, mutation=True):
        return denied
    return _service(request).cancel()
