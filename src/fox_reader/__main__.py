import logging
import os
import sys

#: Overrides the reader's own log level. INFO by default: the lines that matter
#: most when a local model will not load are logged there — the memory check, and
#: the line the GGUF translator writes immediately before a call that can end the
#: process without raising anything Python can catch.
LOG_LEVEL_ENV = "FOX_READER_LOG_LEVEL"

def _configure_logging() -> None:
    """Give the reader's own loggers somewhere to go.

    uvicorn configures handlers for its own three loggers and nothing else, so
    without this everything under ``fox_reader`` falls through to logging's
    lastResort handler — which prints WARNING and above, and drops exactly the
    lines that explain a failed model load.

    The level is raised on ``fox_reader`` alone rather than on the root: at INFO
    the root logger would also let through every library in the process, and
    huggingface_hub and PIL between them have plenty to say.
    """
    wanted = os.environ.get(LOG_LEVEL_ENV, "").strip().upper() or "INFO"
    level = logging.getLevelNamesMapping().get(wanted, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))

    root = logging.getLogger()
    root.addHandler(handler)

    if root.level == logging.NOTSET:
        root.setLevel(logging.WARNING)

    logging.getLogger("fox_reader").setLevel(level)


def main():
    _configure_logging()

    from fox_reader.config import ConfigManager
    from fox_reader.utils import CONFIG_DIR, ensure_user_dirs

    # The shipped tree is created by the installer, but a first run from a bare
    # directory should still produce config/ fonts/ models/ cache/ rather than
    # having each of them appear whenever something happens to need one.
    ensure_user_dirs()

    fox_config_mgr = ConfigManager(CONFIG_DIR)
    config = fox_config_mgr.load()

    import uvicorn

    from fox_reader import runtime
    from fox_reader.app import app

    # Built explicitly rather than through `uvicorn.run`, for the server object:
    # `/api/shutdown` and the setup wizard stop the process by setting
    # `should_exit` on it, which is the only way to get an ordered shutdown --
    # and therefore a cleared cache -- on Windows as well as POSIX. Passing the
    # app object rather than "fox_reader.app:app" also keeps the import string
    # machinery, and its sys.path assumptions, out of a compiled build.
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.host,
            port=config.port,
            reload=False,
        )
    )
    runtime.register_server(server)

    try:
        server.run()
    finally:
        runtime.unregister_server()


if __name__ == "__main__":
    main()
