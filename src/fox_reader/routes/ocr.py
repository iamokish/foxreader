from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from starlette.concurrency import run_in_threadpool

from fox_reader.models.requests import CropRequest, FreeformRequest

router = APIRouter()


def _payload(reading) -> dict:
    """The response body: the text, plus the colours when any were measured.

    ``colors`` is left out entirely rather than sent as three nulls, so a region
    whose colours could not be read is indistinguishable on the wire from a
    client that did not ask for them -- in both cases the page keeps whatever
    hints it already had and the backend detects the rest, which is what the two
    mean.
    """
    body = {"text": (reading.text or "").strip()}
    if reading.colors.measured:
        body["colors"] = reading.colors.payload()
    return body


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
        def _read_and_predict() -> dict:
            img = Image.open(file_path).convert("RGBA")

            mask = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(polygon, outline=255, fill=255)

            white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img_masked = Image.composite(img, white_bg, mask)

            bbox = mask.getbbox()
            if bbox:
                img_masked = img_masked.crop(bbox)
                # The same crop of the same mask: the colour read has to know
                # which pixels are the selection and which are the white filler
                # the composite put around it, or it measures the filler as the
                # region's background.
                mask = mask.crop(bbox)

            reading = ocr.read(
                lang=req.lang, image=img_masked.convert("RGB"),
                grayscale=req.isGrayScale, colors=req.colors, mask=mask,
            )
            return _payload(reading)

        return JSONResponse(await run_in_threadpool(_read_and_predict))

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
        def _read_and_predict() -> dict:
            img = Image.open(file_path)
            left = max(0, req.x)
            top = max(0, req.y)
            right = min(img.width, left + req.width)
            bottom = min(img.height, top + req.height)
            cropped = img.crop((left, top, right, bottom)).convert("RGB")
            reading = ocr.read(
                lang=req.lang, image=cropped, grayscale=req.isGrayScale,
                colors=req.colors,
            )
            return _payload(reading)

        return JSONResponse(await run_in_threadpool(_read_and_predict))

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
