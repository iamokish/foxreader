"""FastAPI application — initialization, middleware, and router mounting."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request

from fox_reader import assets
from fox_reader.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: The model id whose weights the Bubble Capture button needs.
BUBBLE_MODEL_ID = "bubble-segmentation"


def _build_bubble():
    """BubbleService, or None when the app has to do without it.

    Bubble detection is a capability, not a requirement: its weights are
    optional on the setup page, so the service may legitimately have nothing to
    load. Returning None here is what switches the Bubble Capture button off
    (routes/folder.py reads it), and every route that uses the service already
    guards on it.

    Construction is also wrapped, because weights can be present and still
    unloadable -- a truncated download, a device that went away between the
    disk check and the load. Either way the rest of the app starts.
    """
    from fox_reader.constants import base_model_downloaded

    if not base_model_downloaded(BUBBLE_MODEL_ID):
        logger.info("Bubble weights are not downloaded; Bubble Capture is off.")
        return None

    try:
        from fox_reader.services.bubble_service import BubbleService

        return BubbleService()
    except Exception:  # noqa: BLE001 - one optional model must not stop startup
        logger.warning(
            "Bubble weights are present but would not load; "
            "Bubble Capture is off.",
            exc_info=True,
        )
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    from fox_reader.config import ConfigManager
    from fox_reader.device import configure as configure_devices
    from fox_reader.device import configure_threads
    from fox_reader.routes.setup import check_models_exist
    from fox_reader.services.font_service import FontService
    from fox_reader.services.progress import ProgressRegistry
    from fox_reader.services.session_service import SessionService
    from fox_reader.settings import SettingsManager
    from fox_reader.user_endpoints import UserEndpointStore
    from fox_reader.utils import CONFIG_DIR
    from fox_reader.workspaces import WorkspaceStore

    # Also cleared on the way out, which is the normal path. This is for the
    # abnormal one: a hard kill or a power cut leaves scratch files behind and
    # nothing else ever removes them. Cheap, and deliberately here rather than
    # later -- cache/ holds a handful of recomputable .npy intermediates, and no
    # model has loaded yet, so it is nowhere near the readiness path.
    try:
        from fox_reader.utils import clear_cache

        clear_cache()
    except Exception:  # noqa: BLE001 - a stale cache must not stop startup
        logger.debug("Startup cache clear failed.", exc_info=True)

    fox_config_mgr = ConfigManager(CONFIG_DIR)
    config = fox_config_mgr.load()

    settings_mgr = SettingsManager(CONFIG_DIR)
    settings_mgr.load()

    user_endpoints = UserEndpointStore(CONFIG_DIR)

    # Which destination folder each source scan was last typeset into, so the
    # folder picker can offer it back instead of asking again every session.
    workspaces = WorkspaceStore(CONFIG_DIR)

    mtl_dir = config.resolve_mtl_dir(PROJECT_ROOT)

    # A default model is only meaningful once its weights are on disk, and
    # nothing writes one at download time -- so a fresh settings.yaml holds
    # nulls even on a machine with every model fetched, and the translator
    # refuses every language until someone opens the settings page and picks
    # one by hand. Settle it here instead, before anything reads it.
    settings_mgr.resolve_mtl_defaults(mtl_dir)

    # Before any model is built: the OCR engines and the translator bind their
    # device at construction, so the saved choice has to be resolved against
    # the hardware actually present first. A slot pointing at a card that is
    # not here comes back on the CPU, with the preference left as written.
    device_state = configure_devices(settings_mgr.get_devices())

    # The GGUF translator is the one model that picks its own thread count, so
    # the saved preference is applied here too. LocalMTLManager re-applies it
    # before every load -- this is for the log line at startup, and for anything
    # that builds a translator without going through the manager.
    configure_threads(settings_mgr.get_mtl_threads())

    app.state.config = config
    app.state.settings = settings_mgr
    app.state.user_endpoints = user_endpoints
    app.state.workspaces = workspaces
    app.state.mtl_dir = mtl_dir
    app.state.devices = device_state
    app.state.session_id = str(uuid4())
    app.state.session = SessionService()
    app.state.fonts = FontService()

    # Long jobs (typesetting, text clean) run in worker threads and answer with
    # an image, so they narrate through here and the UI polls /api/progress.
    app.state.progress = ProgressRegistry()

    ready_flag: bool = check_models_exist(config)

    if ready_flag:
        from fox_reader.services.clean_service import CleanService
        from fox_reader.services.inpaint_service import InpaintService
        from fox_reader.services.ocr_service import OCRService
        from fox_reader.services.translate_service import TranslateService

        # Settings travel with the service so predict() honours the saved OCR
        # engine. A selected-but-broken VL falls back to classic at predict
        # time (see OCRService), never here, so startup cannot fail on it.
        app.state.ocr = OCRService(settings_mgr)
        app.state.translate = TranslateService(
            config,
            settings_mgr,
            user_endpoints,
        )
        # CleanService borrows the OCR detector rather than building its own, and
        # loads the Text Seg network only if an entry actually asks for it. Its
        # Text Seg method reports itself unavailable when those weights are
        # absent (see fox_reader.clean.seg.is_available), so the service is
        # built either way and the one method switches itself off.
        app.state.clean = CleanService(app.state.ocr)
        app.state.inpaint = InpaintService(app.state.clean)
        app.state.bubble = _build_bubble()
        app.state.needs_setup = False
    else:
        app.state.bubble = None
        app.state.ocr = None
        app.state.translate = None
        app.state.clean = None
        app.state.inpaint = None
        app.state.needs_setup = True

    yield

    # The Text Seg network is the one model built outside the service graph, so
    # it is the one that has to be told to let go of its device.
    try:
        from fox_reader.clean import seg

        seg.unload()
    except Exception:  # noqa: BLE001 - shutdown must not raise
        pass

    # Everything in cache/ is a recomputable intermediate, so carrying it into
    # the next run buys nothing -- and an entry left behind by a run that ended
    # badly is worse than none. This is why the shutdown path is `should_exit`
    # rather than a signal: on Windows the old SIGTERM never reached here.
    try:
        from fox_reader.utils import clear_cache

        removed = clear_cache()
        if removed:
            logger.info("Cleared %d cache entr%s.", removed, "y" if removed == 1 else "ies")
    except Exception:  # noqa: BLE001 - shutdown must not raise
        logger.debug("Cache clear failed.", exc_info=True)


app = FastAPI(lifespan=lifespan)


# ── Middleware ──────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/font"):
        return response
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ── Static files ───────────────────────────────────────────────────────────────
# Served out of the binary in a compiled build and off disk in a checkout;
# either way the mount is named "static", which is what every template's
# `url_for('static', path=...)` resolves against.

assets.mount_static(app)


# ── Routers ────────────────────────────────────────────────────────────────────

from fox_reader.routes import bubble, folder, inpaint, ocr, setup, system, translate, ws  # noqa: E402
from fox_reader.routes import characters as characters_routes  # noqa: E402
from fox_reader.routes import settings as settings_routes  # noqa: E402
from fox_reader.routes import user_endpoints as user_endpoints_routes  # noqa: E402

app.include_router(setup.router)
app.include_router(ws.router)
app.include_router(folder.router)
app.include_router(system.router)
app.include_router(ocr.router)
app.include_router(bubble.router)
app.include_router(translate.router)
app.include_router(characters_routes.router)
app.include_router(inpaint.router)
app.include_router(settings_routes.router)
app.include_router(user_endpoints_routes.router)
