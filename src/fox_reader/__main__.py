from fox_reader.logging_setup import LOG_LEVEL_ENV as LOG_LEVEL_ENV
from fox_reader.logging_setup import configure_logging


def _configure_logging() -> None:
    """Configure bounded files before importing models or starting the server."""
    configure_logging()


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
            log_config=None,
            access_log=False,
        )
    )
    runtime.register_server(server)

    try:
        server.run()
    finally:
        runtime.unregister_server()


if __name__ == "__main__":
    main()
