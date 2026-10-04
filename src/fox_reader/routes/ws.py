from __future__ import annotations

import logging
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/lifecycle")
async def lifecycle_websocket(websocket: WebSocket):
    await websocket.accept()
    session = websocket.app.state.session
    session.add_tab(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        session.remove_tab(websocket)


@router.get("/api/ping-ui")
async def ping_ui(request: Request):
    """Used by the launcher to check if a tab is already open."""
    session = request.app.state.session
    if session.has_active_tabs():
        for tab in session.get_active_tabs():
            try:
                await tab.send_json({"action": "focus_tab"})
            except Exception as e:
                logger.debug("Failed to send focus_tab: %s", e)
        return {"status": "alerted"}
    return {"status": "none"}
