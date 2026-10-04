from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from fox_reader.models.requests import BubbleRequest, SplitRequest

router = APIRouter()


@router.post("/api/bubble", response_class=JSONResponse)
async def detect_bubbles(request: Request, req: BubbleRequest):
    session = request.app.state.session
    bubble = request.app.state.bubble

    try:
        if not session.current_dir:
            return JSONResponse({"error": "No folder loaded"}, status_code=400)
        if bubble is None:
            return JSONResponse({"error": "Models are not ready"}, status_code=503)

        cache_key = "gray_scale" if req.isGrayScale else "normal"
        cached = session.get_cached_bubble(req.filename, cache_key)
        if cached is not None:
            return JSONResponse(cached)

        file_path = os.path.join(session.current_dir, req.filename)
        if not os.path.exists(file_path):
            return JSONResponse({"error": "File not found"}, status_code=404)

        img = Image.open(file_path)
        response_data = await run_in_threadpool(
            bubble.detect, img, req.isGrayScale
        )
        session.set_cached_bubble(req.filename, cache_key, response_data)

        return JSONResponse(response_data)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/splitbubble")
async def split_bubble(request: Request, req: SplitRequest):
    try:
        region_coords = req.region.to_polygon()
        if len(region_coords) < 3:
            return JSONResponse({"error": "Invalid region data"}, status_code=400)
        if len(req.split_object) < 2:
            return JSONResponse({"error": "Invalid split object data"}, status_code=400)

        split_coords = [{"x": p.x, "y": p.y} for p in req.split_object]

        from fox_reader.services.bubble_service import BubbleService

        # Rasterising a long freehand stroke is milliseconds of pure CPU, but it
        # is still CPU on the event loop, and the health endpoint has to answer
        # while a user is scribbling across a full-page panel.
        result = await run_in_threadpool(
            BubbleService.split, region_coords, split_coords
        )
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
