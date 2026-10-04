"""A handle on the running server, so the app can ask itself to stop.

The old way out was ``os.kill(os.getpid(), SIGTERM)``. On POSIX uvicorn catches
that and shuts down in order; on Windows :func:`os.kill` maps everything except
the two console events to ``TerminateProcess``, so the process vanished mid-
request -- no lifespan shutdown, no ``seg.unload()``, and nothing to clear the
cache the launcher is now expected to leave empty.

Setting ``should_exit`` on the server object instead is the same graceful path on
both platforms: uvicorn polls that flag every 100 ms, stops accepting, lets
in-flight requests finish and then runs lifespan shutdown. It needs a reference
to the object, which only the entry point has -- hence this module.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_server: Any | None = None
_timer: threading.Timer | None = None


def register_server(server: Any) -> None:
    """Record the :class:`uvicorn.Server` this process is running."""
    global _server
    with _lock:
        _server = server


def unregister_server() -> None:
    global _server
    with _lock:
        _server = None


def is_registered() -> bool:
    with _lock:
        return _server is not None


def request_shutdown() -> bool:
    """Ask the server to stop gracefully. False if there is no server to ask.

    A False here is not a failure to handle quietly -- it means the process was
    started some other way (``uvicorn fox_reader.app:app``, a test client) and
    the caller has to fall back to a signal.
    """
    with _lock:
        server = _server

    if server is None:
        logger.warning("Shutdown requested but no server is registered.")
        return False

    logger.info("Shutdown requested; stopping the server.")
    server.should_exit = True
    return True


def shutdown_soon(delay: float = 0.25) -> bool:
    """Same, once the current response has had time to reach the client.

    Returns whether a server was registered, so the caller can answer the
    request honestly before anything actually stops.
    """
    global _timer

    with _lock:
        if _server is None:
            logger.warning("Shutdown requested but no server is registered.")
            return False
        # Idempotent: a browser that double-fires the request must not leave two
        # timers behind.
        if _timer is not None and _timer.is_alive():
            return True

        timer = threading.Timer(max(0.0, delay), request_shutdown)
        timer.daemon = True
        _timer = timer

    timer.start()
    return True


def terminate_hard(delay: float = 0.25) -> None:
    """Last resort for a process nobody registered a server for.

    Reached when the app was started some other way -- ``uvicorn
    fox_reader.app:app``, a test harness -- so nothing is going to run the
    lifespan shutdown. The one part of it the user can see, an emptied
    ``cache/``, is therefore done by hand first.
    """
    import os
    import signal

    from fox_reader.utils import clear_cache

    try:
        clear_cache()
    except Exception:  # noqa: BLE001 - already on the way out
        logger.debug("Cache clear failed on hard shutdown.", exc_info=True)

    def _terminate() -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Timer(max(0.0, delay), _terminate)
    timer.daemon = True
    timer.start()


def stop(delay: float = 0.25) -> bool:
    """Stop this process. Returns whether it will be a graceful stop.

    The two callers -- ``POST /api/shutdown`` and the setup wizard's close
    button -- both want the response delivered before anything shuts down, and
    both want the graceful path when there is one.
    """
    if shutdown_soon(delay):
        return True
    terminate_hard(delay)
    return False
