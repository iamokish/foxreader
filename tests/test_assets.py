"""The embedded frontend.

A compiled build has no ``frontend/`` folder: the build generates a module of
compressed byte constants and Nuitka compiles it in. None of that can be
exercised by importing a real bundle here, so these install a synthetic one --
built by the same generator the build uses -- and check that the app serves from
it identically: templates render, ``url_for('static', ...)`` still resolves,
brotli payloads go out compressed when the client says it can read them and
decompressed when it cannot.
"""
import importlib
import importlib.util
import random
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from fox_reader import assets


def _load_generator():
    """The build-time generator, by path -- packaging/ is not an installed package."""
    path = Path(__file__).resolve().parent.parent / "packaging" / "embed_assets.py"
    spec = importlib.util.spec_from_file_location("_embed_assets_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embed_assets = _load_generator()

#: A stand-in for the real PNGs: 1 KiB that brotli cannot shrink.
_NOISE = random.Random(0).randbytes(1024)


def _reset_caches() -> None:
    # Both are lru_cached, and the cache would otherwise carry one test's answers
    # into the next.
    assets._bundle.cache_clear()
    assets.templates.cache_clear()
    assets._decoded.clear()
    # And drop the module itself. monkeypatch.delitem records nothing when the key
    # was absent to begin with -- which it always is here -- so undoing the fixture
    # leaves the synthetic bundle sitting in sys.modules. Clearing the lru_cache
    # then achieves nothing: the next _bundle() re-imports, finds the stale entry
    # and hands back a bundle that no test asked for. Anything that asks whether a
    # bundle exists (tests/test_fonts.py does) sees the wrong answer.
    sys.modules.pop(assets.BUNDLE_MODULE, None)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A generated bundle module, installed as if Nuitka had compiled it in."""
    page = "<html><body>{{ greeting }} <img src=\"{{ url_for('static', path='logo.png') }}\"></body></html>"
    source = embed_assets.render_module(
        templates={"page.html": page.encode("utf-8")},
        static={
            # Compresses well, so it is stored brotli and negotiated.
            "app.js": ("application/javascript", b"console.log('hi');" * 200),
            # Incompressible, like the real PNGs -- brotli makes it bigger, so it
            # is stored as-is. Seeded, so the bundle is the same every run.
            "logo.png": ("image/png", _NOISE),
        },
        fonts={"ComicMono.ttf": b"\x00\x01\x00\x00not-a-real-font"},
    )

    (tmp_path / f"{assets.BUNDLE_MODULE}.py").write_text(source, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv(assets.DISK_ASSETS_ENV, raising=False)

    _reset_caches()
    yield importlib.import_module(assets.BUNDLE_MODULE)
    _reset_caches()


@pytest.fixture
def client(bundle):
    app = FastAPI()
    assets.mount_static(app)

    @app.get("/page")
    async def page(request: Request):  # noqa: ANN202
        return assets.templates().TemplateResponse(
            request=request, name="page.html", context={"greeting": "hello"}
        )

    return TestClient(app)


class TestGeneratedModule:
    def test_declares_the_format_the_reader_expects(self, bundle):
        assert bundle.FORMAT == assets.BUNDLE_FORMAT
        assert embed_assets.FORMAT == assets.BUNDLE_FORMAT

    def test_generator_and_reader_agree_on_the_bundled_fonts(self):
        """Two copies of the list; a build that embedded one and looked up the
        other would ship a fallback nothing ever reads."""
        assert embed_assets.BUNDLED_FONTS == assets.BUNDLED_FONTS
        assert embed_assets.SYMBOL_FONT == assets.SYMBOL_FONT

    def test_compresses_only_where_it_wins(self, bundle):
        assert bundle.STATIC["app.js"][2] == "br"
        # The real PNGs and the woff2 come out larger under brotli, so storing
        # them compressed would cost a decompression pass to save nothing.
        assert bundle.STATIC["logo.png"][2] == "raw"

    def test_chunks_stay_under_the_c_string_limit(self, bundle):
        for section in (bundle.TEMPLATES, bundle.STATIC, bundle.FONTS):
            for name, entry in section.items():
                assert entry[0], name
                for chunk in entry[0]:
                    assert len(chunk) <= embed_assets.CHUNK_SIZE, name

    def test_round_trips(self, bundle):
        assert assets.static_bytes("logo.png") == _NOISE
        assert assets.static_bytes("app.js") == b"console.log('hi');" * 200
        assert assets.bundled_font_bytes("ComicMono.ttf") == b"\x00\x01\x00\x00not-a-real-font"

    def test_multi_chunk_payload_rejoins(self):
        """Anything past CHUNK_SIZE is split, so the join path is the normal one
        for the real assets -- app.js alone is four chunks."""
        big = random.Random(1).randbytes(embed_assets.CHUNK_SIZE * 2 + 17)
        chunks, raw_size, encoding = embed_assets.encode(big)
        assert len(chunks) == 3
        assert encoding == "raw"
        assert assets._decode((chunks, raw_size, encoding, "image/png"), "test") == big

    def test_truncated_payload_is_caught(self):
        """The raw size is stored so a payload that did not survive the round
        trip fails loudly instead of being served half-empty."""
        chunks, raw_size, encoding = embed_assets.encode(b"x" * 100)
        with pytest.raises(ValueError):
            assets._decode((chunks, raw_size + 1, encoding, "text/plain"), "test")

    def test_reports_a_bundle(self, bundle):
        assert assets.has_bundle() is True
        assert assets.template_names() == ["page.html"]
        assert assets.static_names() == ["app.js", "logo.png"]
        assert assets.bundled_font_names() == ["ComicMono.ttf"]

    def test_unknown_names_are_absent_rather_than_falling_back_to_disk(self, bundle):
        """With a bundle present, disk is not consulted -- a stale frontend/ next
        to a compiled binary must not be able to shadow the embedded copy."""
        assert assets.static_bytes("style.css") is None
        assert assets.template_source("index.html") is None


class TestContentTypes:
    def test_javascript_is_not_left_to_the_registry(self):
        """mimetypes answers from the registry on Windows, where .js has come
        back as text/plain -- which the browser then refuses to execute."""
        assert embed_assets.content_type("app.js") == "application/javascript"

    def test_every_shipped_asset_has_a_known_type(self):
        root = Path(__file__).resolve().parent.parent
        for directory in (root / "frontend" / "static", root / "frontend" / "templates"):
            for path in directory.rglob("*"):
                # _skip takes the path relative to the asset root, the same way
                # the collector calls it.
                if not path.is_file() or embed_assets._skip(path.relative_to(directory)):
                    continue
                assert embed_assets.content_type(path.name) != "application/octet-stream", path.name

    def test_build_litter_is_not_shipped(self):
        """Vite drops a .vite/ manifest into static/. It is not an asset."""
        assert embed_assets._skip(Path(".vite/license.md"))
        assert embed_assets._skip(Path(".vite/manifest.json"))
        assert not embed_assets._skip(Path("app.js"))
        assert not embed_assets._skip(Path("fonts/ComicMono.ttf"))

    def test_extensionless_files_are_text(self):
        # frontend/static/LICENSE-material-icons.
        assert embed_assets.content_type("LICENSE-material-icons").startswith("text/plain")


class TestCollection:
    def test_reads_the_checkout(self):
        root = Path(__file__).resolve().parent.parent
        templates, static, fonts = embed_assets.collect(root)

        assert "index.html" in templates
        assert "app.js" in static and static["app.js"][0] == "application/javascript"
        # The fallback faces plus the typesetting-only symbol face: the rest of
        # fonts/ is the user's. The symbol face travels in the same FONTS
        # section but is never offered to the frontend (see
        # TestSymbolFallback below).
        assert tuple(sorted(fonts)) == tuple(sorted((*embed_assets.BUNDLED_FONTS, embed_assets.SYMBOL_FONT)))
        # The font cache is machine-specific and must never be embedded.
        assert ".fonts_cache.json" not in fonts

    def test_missing_directory_is_an_error_not_an_empty_bundle(self, tmp_path):
        with pytest.raises(embed_assets.AssetError):
            embed_assets.collect(tmp_path)


class TestTemplates:
    def test_renders_and_resolves_url_for(self, client):
        res = client.get("/page")
        assert res.status_code == 200
        assert "hello" in res.text
        # The mount is named "static", so url_for keeps working unchanged.
        assert "/static/logo.png" in res.text


class TestStaticServing:
    def test_brotli_passed_through_when_accepted(self, client, bundle):
        res = client.get("/static/app.js", headers={"Accept-Encoding": "br"})
        assert res.status_code == 200
        assert res.headers["content-encoding"] == "br"
        assert res.headers["vary"] == "Accept-Encoding"
        # httpx decodes br for us, so this is the real payload either way.
        assert res.content == b"console.log('hi');" * 200
        assert res.headers["content-type"].startswith("application/javascript")

    def test_decompressed_when_not_accepted(self, client):
        res = client.get("/static/app.js", headers={"Accept-Encoding": "identity"})
        assert res.status_code == 200
        assert "content-encoding" not in res.headers
        assert res.content == b"console.log('hi');" * 200

    def test_raw_asset_is_never_marked_compressed(self, client):
        res = client.get("/static/logo.png", headers={"Accept-Encoding": "br"})
        assert res.status_code == 200
        assert "content-encoding" not in res.headers
        assert res.headers["content-type"] == "image/png"

    def test_missing_is_404(self, client):
        assert client.get("/static/nope.css").status_code == 404

    def test_mount_prefix_is_stripped_from_the_lookup(self):
        """Starlette's Mount leaves the full request path in scope['path'] and
        records the matched prefix in root_path. Reading the wrong one does not
        fail loudly -- every lookup misses and the entire frontend 404s."""
        assert assets._sub_path({"path": "/static/app.js", "root_path": "/static"}) == "app.js"
        assert assets._sub_path({"path": "/static/sub/x.js", "root_path": "/static"}) == "sub/x.js"
        # Unmounted, and a prefix that is not actually a prefix, both survive.
        assert assets._sub_path({"path": "/app.js", "root_path": ""}) == "app.js"
        assert assets._sub_path({"path": "/app.js", "root_path": "/other"}) == "app.js"

    def test_traversal_has_nowhere_to_go(self, client):
        """Encoded so the client cannot normalise it away before it is sent. In a
        bundle lookups are dict keys, so there is no path to walk."""
        res = client.get("/static/..%2F..%2Fconfig%2Ffox_config.yaml")
        assert res.status_code == 404

    def test_head_reports_length_without_a_body(self, client):
        res = client.head("/static/logo.png")
        assert res.status_code == 200
        assert res.content == b""
        assert res.headers["content-length"] == str(len(_NOISE))

    def test_other_methods_refused(self, client):
        res = client.post("/static/logo.png")
        assert res.status_code == 405
        assert res.headers["allow"] == "GET, HEAD"


class TestDiskFallback:
    def test_no_bundle_reads_the_repository(self, monkeypatch):
        _reset_caches()
        try:
            assert assets.has_bundle() is False
            # The checkout's real files, so a dev run is unaffected by any of this.
            assert "index.html" in assets.template_names()
            assert assets.static_bytes("site.webmanifest") is not None
        finally:
            _reset_caches()

    def test_env_override_ignores_a_present_bundle(self, bundle, monkeypatch):
        monkeypatch.setenv(assets.DISK_ASSETS_ENV, "1")
        assets._bundle.cache_clear()
        assets.templates.cache_clear()
        assets._decoded.clear()
        assert assets.has_bundle() is False

    def test_disk_reads_cannot_escape_the_static_directory(self, monkeypatch):
        """The disk branch does touch the filesystem, so containment is checked
        there rather than left to the caller."""
        monkeypatch.delitem(sys.modules, assets.BUNDLE_MODULE, raising=False)
        _reset_caches()
        try:
            assert assets.static_bytes("../templates/index.html") is None
            assert assets.static_bytes("../../config/fox_config.yaml") is None
        finally:
            _reset_caches()

    def test_wrong_format_is_ignored(self, tmp_path, monkeypatch):
        path = tmp_path / f"{assets.BUNDLE_MODULE}.py"
        path.write_text("FORMAT = 99\nTEMPLATES = {}\nSTATIC = {}\nFONTS = {}\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delenv(assets.DISK_ASSETS_ENV, raising=False)
        _reset_caches()
        try:
            assert assets.has_bundle() is False
        finally:
            _reset_caches()


class TestSymbolFallback:
    """The inpainting-only face: embedded, never listed, reachable by name."""

    def test_symbol_is_hidden_from_the_frontend_list_but_reachable(self, tmp_path, monkeypatch):
        source = embed_assets.render_module(
            templates={"page.html": b"<html></html>"},
            static={"app.js": ("application/javascript", b"x")},
            fonts={
                "ComicMono.ttf": b"\x00\x01\x00\x00not-a-real-font",
                embed_assets.SYMBOL_FONT: b"\x00\x01\x00\x00not-a-real-symbol",
            },
        )
        (tmp_path / f"{assets.BUNDLE_MODULE}.py").write_text(source, encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delenv(assets.DISK_ASSETS_ENV, raising=False)
        _reset_caches()
        try:
            assert assets.has_bundle() is True
            # Hidden from the frontend-facing list...
            assert embed_assets.SYMBOL_FONT not in assets.bundled_font_names()
            assert assets.bundled_font_names() == ["ComicMono.ttf"]
            # ...but readable through the dedicated accessor.
            assert assets.symbol_font_bytes() == b"\x00\x01\x00\x00not-a-real-symbol"
            assert assets.symbol_font_bytes(embed_assets.SYMBOL_FONT) == b"\x00\x01\x00\x00not-a-real-symbol"
            # Anything else is refused so this cannot become a generic loader.
            assert assets.symbol_font_bytes("ComicMono.ttf") is None
        finally:
            _reset_caches()

    def test_disk_symbol_is_hidden_from_bundled_names(self):
        # Without a bundle the disk fallback still hides the symbol face from
        # the list while serving its bytes on demand.
        _reset_caches()
        try:
            assert assets.has_bundle() is False
            assert assets.SYMBOL_FONT not in assets.bundled_font_names()
            raw = assets.symbol_font_bytes()
            # The checkout ships the face; a machine without it gets None, not
            # an exception.
            assert raw is None or isinstance(raw, bytes)
        finally:
            _reset_caches()
