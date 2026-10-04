from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from starlette.concurrency import run_in_threadpool

from fox_reader.models.requests import CropRequest, FreeformRequest

router = APIRouter()


@router.post("/api/ocr_freeform", response_class=JSONResponse)
async def ocr_freeform(request: Request, req: FreeformRequest):
    session = request.app.state.session
    ocr = request.app.state.ocr

    try:
        if not session.current_dir:
            return JSONResponse({"error": "No folder loaded"}, status_code=400)

        file_path = os.path.join(session.current_dir, req.filename)
        if not os.path.exists(file_path):
            return JSONResponse({"error": "File not found"}, status_code=404)

        polygon = [(p["x"], p["y"]) for p in req.points]

        # Everything below is page-sized: `convert("RGBA")` forces a full decode,
        # then three more page-sized buffers and a composite. Left on the event
        # loop that is tens of milliseconds per call with nothing else able to
        # run -- and Page TL fires one call per bubble, so a 20-bubble page used
        # to stall the health check twenty times over.
        def _read_and_predict() -> str:
            img = Image.open(file_path).convert("RGBA")

            mask = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(polygon, outline=255, fill=255)

            white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img_masked = Image.composite(img, white_bg, mask)

            bbox = mask.getbbox()
            if bbox:
                img_masked = img_masked.crop(bbox)

            return ocr.predict(
                lang=req.lang, image=img_masked.convert("RGB"), grayscale=req.isGrayScale
            )

        result = await run_in_threadpool(_read_and_predict)
        return JSONResponse({"text": result.strip()})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/ocr_crop", response_class=JSONResponse)
async def ocr_crop(request: Request, req: CropRequest):
    session = request.app.state.session
    ocr = request.app.state.ocr

    try:
        if not session.current_dir:
            return JSONResponse({"error": "No folder loaded"}, status_code=400)

        file_path = os.path.join(session.current_dir, req.filename)
        if not os.path.exists(file_path):
            return JSONResponse({"error": "File not found"}, status_code=404)

        # `Image.open` is lazy, but `.crop().convert()` is what actually decodes
        # the page, so the whole read belongs in the threadpool with the model
        # call. Clamping the box needs the real dimensions, which is also a read.
        def _read_and_predict() -> str:
            img = Image.open(file_path)
            left = max(0, req.x)
            top = max(0, req.y)
            right = min(img.width, left + req.width)
            bottom = min(img.height, top + req.height)
            cropped = img.crop((left, top, right, bottom)).convert("RGB")
            return ocr.predict(
                lang=req.lang, image=cropped, grayscale=req.isGrayScale
            )

        result = await run_in_threadpool(_read_and_predict)
        return JSONResponse({"text": result.strip()})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
