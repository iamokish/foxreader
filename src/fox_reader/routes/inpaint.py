from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from fox_reader import fsaccess
from fox_reader.models.requests import ProcessImageRequest, SavePreviewRequest

router = APIRouter()

#: Extension -> media type. The old code interpolated the extension straight into
#: `image/{ext}`, which produced `image/.png`; browsers cope with that for a blob
#: but it is wrong, and `jpg` needs mapping either way.
_MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "avif": "image/avif",
}

#: Sent on a 409 so the page can tell the two refusals apart without matching on
#: prose. `exists` is answerable (ask, then retry with overwrite); `stale_preview`
#: is not (the preview has to be re-rendered).
_CODE_EXISTS = "exists"
_CODE_STALE = "stale_preview"


def _source_path(request: Request, filename: str) -> str:
    """The original page. Every render reads from here, whatever is on screen."""
    session = request.app.state.session
    path = fsaccess.safe_join(session.current_dir or "", filename)
    if path is None:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return path


def _dest_path(request: Request, filename: str) -> str:
    """Where the typeset page is written: the destination folder, or the source."""
    session = request.app.state.session
    path = fsaccess.safe_join(session.save_dir or "", filename)
    if path is None:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return path


def _guard_overwrite(target: str, overwrite: bool) -> None:
    """Refuse to replace an existing file unless the page said to.

    Checked here rather than only in the browser, and checked *before* the render
    rather than after: a page opened before the file appeared would otherwise
    overwrite it, and there is no reason to spend forty seconds of typesetting on
    a save that is going to be refused.
    """
    if overwrite or not os.path.exists(target):
        return
    raise HTTPException(
        status_code=409,
        detail=f"{os.path.basename(target)} already exists in the destination folder.",
        headers={"X-Fox-Code": _CODE_EXISTS},
    )


@router.get("/api/clean/methods")
async def clean_methods(request: Request):
    """What text clean can do on this machine, for the entry panel to offer.

    Cheap by construction (see `CleanService.available_methods`): no model is
    loaded, so the entry UI can ask on open without stalling. The tuning
    catalogues come from `fox_reader.clean.tuning`, which imports nothing, so
    asking this question never pulls torch in.

    The fill list is filtered by what this OpenCV build can actually run --
    offering `fsr` on a build whose `cv2.xphoto` is an empty stand-in module got
    the user an `AttributeError` per region and a page that came back untouched
    (see `fox_reader.clean.caps`).

    Speeds and tiles are sent as `{value, label, ...}` records rather than bare
    values: the pass count and the overlap are derived from the same tables the
    detector reads, so the panel cannot show a tile size paired with an overlap
    the backend would not use.
    """
    from fox_reader.clean import (
        DEFAULT_INPAINT_METHOD,
        DEFAULT_SPEED,
        DEFAULT_TILE,
        DEFAULT_TTA,
        INPAINT_METHODS,
        METHOD_KNOBS,
        SPEED_ORDER,
        TILE_ORDER,
        TILE_OVERLAPS,
        fill_available,
        passes_for,
        usable_fills,
    )

    clean = getattr(request.app.state, "clean", None)
    if clean is None:
        # Startup, or a build with no models. The panel still has to open, and
        # `region` genuinely does work without them.
        methods = [
            {"id": "region", "label": "Selected Region", "available": True,
             "reason": ""},
            {"id": "ppocr", "label": "PaddleOCR", "available": False,
             "reason": "the models are not ready"},
            {"id": "textseg", "label": "Text Seg", "available": False,
             "reason": "the models are not ready"},
        ]
        for info in methods:
            info["knobs"] = list(METHOD_KNOBS.get(str(info["id"]), ()))
    else:
        methods = await run_in_threadpool(clean.available_methods)

    fills = usable_fills(INPAINT_METHODS)
    default_fill = (DEFAULT_INPAINT_METHOD
                    if fill_available(DEFAULT_INPAINT_METHOD)
                    else (fills[0] if fills else DEFAULT_INPAINT_METHOD))

    return {
        "methods": methods,
        "fills": fills,
        "default_fill": default_fill,
        "speeds": [
            {"value": name, "label": name.title(),
             # What it costs at most: the inverted pass is skipped outright on a
             # page with no dark ground, so this is an upper bound.
             "passes": passes_for(name, True),
             # Whether this preset has flip TTA for the switch to turn off.
             "tta": passes_for(name, True) != passes_for(name, False)}
            for name in SPEED_ORDER
        ],
        "default_speed": DEFAULT_SPEED,
        "default_tta": DEFAULT_TTA,
        "tiles": [
            {"value": size, "label": "Off" if size == 0 else str(size),
             "overlap": TILE_OVERLAPS[size]}
            for size in TILE_ORDER
        ],
        "default_tile": DEFAULT_TILE,
    }


@router.post("/inpaint/generate")
async def process_image(request: Request, payload: ProcessImageRequest):
    inpaint = request.app.state.inpaint
    if inpaint is None:
        raise HTTPException(status_code=503, detail="Models are not ready")

    session = request.app.state.session
    if not session.current_dir:
        raise HTTPException(status_code=400, detail="No folder loaded")

    # Read the original, write the destination copy. Rendering never reads back
    # what it wrote, so typesetting a page twice is the same as typesetting it
    # once -- the lettering cannot stack up on itself.
    image_path = _source_path(request, payload.filename)
    target_path = _dest_path(request, payload.filename)
    _guard_overwrite(target_path, payload.overwrite)

    task = request.app.state.progress.start("generate")
    task.say(f"Typesetting {payload.filename}")

    try:
        # In a worker thread: typesetting a page is seconds of pure CPU work, and
        # on the event loop it would stall every health check for the duration.
        await run_in_threadpool(
            inpaint.generate, image_path, payload.data, task, target_path)
    except Exception as e:
        task.fail(str(e))
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

    task.finish("Saved")
    return {
        "status": "success",
        "message": "Successfully processed items for image.",
        "img_path": target_path,
        "dest": session.save_dir,
        "saved": True,
        "in_place": session.writes_in_place,
    }


@router.post("/inpaint/preview")
async def inpaint_preview(request: Request, payload: ProcessImageRequest):
    inpaint = request.app.state.inpaint
    if inpaint is None:
        raise HTTPException(status_code=503, detail="Models are not ready")

    image_path = _source_path(request, payload.filename)
    ext = os.path.splitext(payload.filename)[-1].strip(".").lower()

    task = request.app.state.progress.start("preview")
    task.say(f"Rendering a preview of {payload.filename}")

    try:
        artifact = await run_in_threadpool(
            inpaint.preview, image_path, payload.data, payload.filename, task)
    except Exception as e:
        task.fail(str(e))
        raise HTTPException(
            status_code=500, detail=f"An error occurred during preview: {e}"
        )

    task.finish("Preview ready", token=artifact.token)
    return FileResponse(
        path=artifact.path,
        media_type=_MEDIA_TYPES.get(ext, "application/octet-stream"),
        filename=f"preview.{ext or 'png'}",
        # The token is how "Save this image" tells the backend it means the image
        # it is looking at, and not whatever a re-render would produce.
        headers={"X-Preview-Token": artifact.token},
    )


@router.post("/inpaint/save-preview")
async def save_preview(request: Request, payload: SavePreviewRequest):
    """Write the previewed image into the destination, without rendering again."""
    inpaint = request.app.state.inpaint
    if inpaint is None:
        raise HTTPException(status_code=503, detail="Models are not ready")

    session = request.app.state.session
    if not session.current_dir:
        raise HTTPException(status_code=400, detail="No folder loaded")

    target_path = _dest_path(request, payload.filename)
    _guard_overwrite(target_path, payload.overwrite)

    try:
        saved = await run_in_threadpool(
            inpaint.save_preview, payload.token, payload.filename, target_path)
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e),
                            headers={"X-Fox-Code": _CODE_STALE})
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not save: {e}")

    return {"status": "success", "message": "Saved the previewed image.",
            "img_path": saved, "dest": session.save_dir, "saved": True,
            "in_place": session.writes_in_place}
