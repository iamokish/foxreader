from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from PIL import Image

from fox_reader import assets, fsaccess
from fox_reader.models.requests import FolderInspectRequest, FolderRequest

router = APIRouter()
templates = assets.templates()

#: Kept as a module name for anything that imported it; the list itself lives in
#: `fsaccess` so listing a folder and counting its images agree by construction.
_VALID_EXTS = fsaccess.IMAGE_EXTS


def _measure_dimensions(directory: str, filenames: list[str]) -> list[list[int]]:
    """Header-only reads of image dimensions (index-aligned with filenames).

    Unreadable/corrupt files yield [0, 0].
    """
    dims: list[list[int]] = []
    for filename in filenames:
        path = os.path.join(directory, filename)
        try:
            with Image.open(path) as img:
                width, height = img.size
                dims.append([width, height])
        except Exception:
            dims.append([0, 0])
    return dims


def _saved_pages(source: str, dest: str, files: list[str]) -> list[str]:
    """Which of *files* already have a typeset copy in *dest*.

    Empty when the two directories are the same: there is no separate saved copy
    to switch to, and claiming otherwise would show the user an Original/Saved
    toggle whose two sides are one file.
    """
    if not dest or fsaccess.same_dir(source, dest):
        return []
    try:
        present = set(os.listdir(dest))
    except OSError:
        return []
    return [name for name in files if name in present]


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if request.app.state.needs_setup:
        return RedirectResponse(url="/setup", status_code=302)
    config = request.app.state.config
    try:
        translate_status = request.app.state.settings.get_translate_status()
    except Exception:
        translate_status = {"deepl": False, "jpdb": False}
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "initial_theme": config.theme,
            "initial_layout": config.layout,
            "deepl_available": bool(translate_status.get("deepl")),
            "jpdb_available": bool(translate_status.get("jpdb")),
            # The service, not the weights on disk: it is None when the model is
            # absent *and* when present weights failed to load, and either way
            # there is nothing behind the Bubble Capture button. Read from the
            # state the routes themselves use, so the button and the endpoint it
            # calls can never disagree.
            "bubble_available": getattr(request.app.state, "bubble", None) is not None,
        },
    )


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Browsers ask the site root for this regardless of what the HTML declares,
    # so it is served here as well as from the /static mount.
    return assets.static_response("favicon.ico")


@router.get("/api/fonts")
async def get_fonts_list(request: Request):
    """Returns all font metadata and raw bytes pre-encoded in base64 from RAM."""
    fonts = request.app.state.fonts.get_all()
    return [
        {
            "font_name": f["font_name"],
            "font_filename": f["font_filename"],
            "font_format": f["format"],
            "font_data_uri": f["data_uri"],
        }
        for f in fonts
    ]


@router.post("/api/folder/inspect", response_class=JSONResponse)
async def inspect_folder(request: Request, request_data: FolderInspectRequest):
    """What the picker needs to know about a source/destination pair.

    Read-only, and safe to call on every keystroke (the page debounces): nothing
    here creates a directory. The write test does touch the filesystem -- a probe
    file created and removed -- because that is the only answer that holds on
    Windows; see `fox_reader.fsaccess`.
    """
    workspaces = getattr(request.app.state, "workspaces", None)

    source_raw = request_data.source or ""
    source = fsaccess.resolve_dir(source_raw)
    source_report = await asyncio.to_thread(
        fsaccess.describe_dir, source_raw, count_images=True
    )

    suggested = fsaccess.default_dest(source) if source else ""
    remembered = workspaces.dest_for(source) if (workspaces and source) else None

    dest_raw = request_data.dest
    if dest_raw is None or not dest_raw.strip():
        dest_raw = remembered or suggested

    dest = fsaccess.resolve_dir(dest_raw or "")
    dest_report = (
        await asyncio.to_thread(fsaccess.describe_dir, dest_raw)
        if dest
        else fsaccess.DirReport(path="", reason="Enter a destination folder.")
    )

    return JSONResponse({
        "status": "success",
        "source": source_report.to_dict(),
        "dest": dest_report.to_dict(),
        "default_dest": suggested,
        "remembered_dest": remembered or "",
        "same_dir": bool(source and dest and fsaccess.same_dir(source, dest)),
    })


@router.get("/api/folder/state", response_class=JSONResponse)
async def folder_state(request: Request):
    """The folder pair currently loaded, plus what to prefill the picker with."""
    session = request.app.state.session
    workspaces = getattr(request.app.state, "workspaces", None)

    source = session.current_dir or ""
    dest = session.save_dir or ""
    files = fsaccess.list_images(source) if source else []

    return JSONResponse({
        "status": "success",
        "source": source,
        "dest": dest,
        "same_dir": session.writes_in_place if source else False,
        "saved": _saved_pages(source, dest, files),
        "last_source": (workspaces.last_source() if workspaces else None) or "",
        "default_dest_name": fsaccess.DEFAULT_DEST_NAME,
    })


@router.post("/api/load_folder", response_class=JSONResponse)
async def load_folder(request: Request, request_data: FolderRequest):
    session = request.app.state.session
    workspaces = getattr(request.app.state, "workspaces", None)

    source = fsaccess.resolve_dir(request_data.path)

    if not os.path.isdir(source):
        # The wording every existing caller checks for. Kept verbatim.
        return JSONResponse(
            {"status": "error", "error": "Invalid Path"}, status_code=400
        )

    source_report = await asyncio.to_thread(
        fsaccess.describe_dir, source, probe_write=False
    )
    if not source_report.readable:
        return JSONResponse(
            {"status": "error",
             "error": source_report.reason or "This folder cannot be read."},
            status_code=400,
        )

    # Explicit beats remembered beats the default subfolder. A caller that knows
    # nothing about destinations therefore still lands where it did last time,
    # or in <source>/fox_tled on a folder that has never been loaded.
    requested = (request_data.dest or "").strip()
    if not requested and workspaces is not None:
        requested = workspaces.dest_for(source) or ""
    if not requested:
        requested = fsaccess.default_dest(source)

    dest = fsaccess.resolve_dir(requested)
    same = fsaccess.same_dir(source, dest)

    if same:
        # Saving in place is a supported choice, and the source is already known
        # good; no probe, no creation.
        dest_report = source_report
    elif request_data.create_dest:
        dest_report = await asyncio.to_thread(fsaccess.ensure_dir, dest)
    else:
        dest_report = await asyncio.to_thread(fsaccess.describe_dir, dest)

    if not same and not (dest_report.is_dir and dest_report.writable):
        return JSONResponse(
            {"status": "error",
             "error": dest_report.reason
                      or f"Cannot save into {dest_report.path or dest}."},
            status_code=400,
        )

    dest = dest_report.path or dest

    session.set_workspace(source, dest)
    session.clear_bubble_cache()

    files = await asyncio.to_thread(fsaccess.list_images, source)
    dimensions = await asyncio.to_thread(_measure_dimensions, source, files)
    saved = await asyncio.to_thread(_saved_pages, source, dest, files)

    if workspaces is not None:
        workspaces.remember(source, dest)

    # `status`, `files` and `dimensions` are unchanged; everything else is added.
    return JSONResponse(
        {"status": "success", "files": files, "dimensions": dimensions,
         "source": source, "dest": dest, "same_dir": same, "saved": saved},
        status_code=200,
    )


@router.get("/img_serve/{filename}")
async def serve_image(request: Request, filename: str, variant: str = "original"):
    """Serve one page, from the source folder or from the destination.

    `variant=saved` asks for the typeset copy. A page with no typeset copy yet
    falls back to the original rather than 404-ing: the switch is a view state
    that outlives a re-render, and a blank viewer would look like data loss.
    """
    session = request.app.state.session
    source = session.current_dir
    if not source:
        return JSONResponse({"error": "No folder loaded"}, status_code=400)

    candidates = [source]
    if variant == "saved":
        dest = session.save_dir
        if dest and dest != source:
            candidates.insert(0, dest)

    for directory in candidates:
        # The filename comes from the browser. `safe_join` rejects anything with a
        # path in it, which is what kept `../../secrets.png` out of a FileResponse.
        file_path = fsaccess.safe_join(directory, filename)
        if file_path is None:
            return JSONResponse({"error": "Invalid filename"}, status_code=400)
        if os.path.isfile(file_path):
            return FileResponse(file_path)

    return JSONResponse({"error": "File not found"}, status_code=404)
