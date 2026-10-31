"""Which fonts the app offers, and when the bundled fallbacks stand in.

The rule is simple and worth pinning down, because getting it wrong is invisible
until someone ships: the ``fonts/`` folder is counted, and the two embedded
ComicMono faces are used only when that count is zero. Nothing else in the folder
votes -- ``.fonts_cache.json`` is a cache, ``LICENSE-ComicMono`` is a licence, and
a folder holding only those is an empty folder as far as this is concerned.

There is no bundle module installed here, so ``assets.bundled_font_*`` reads the
checkout's own ``fonts/`` for the fallbacks -- which is the disk-side equivalent
of what a compiled build has compiled in.
"""
import json
import shutil
from pathlib import Path

import pytest

from fox_reader import assets, fonts

REPO_FONTS = Path(__file__).resolve().parent.parent / "fonts"


def _names(payloads: list[dict]) -> set[str]:
    return {p["font_filename"] for p in payloads}


@pytest.fixture
def real_font() -> Path:
    """A font that actually parses. The checkout's ComicMono is the only one that
    is guaranteed to be there, so it doubles as the sample."""
    source = REPO_FONTS / "ComicMono.ttf"
    if not source.is_file():
        pytest.skip("the checkout has no fonts/ComicMono.ttf to copy")
    return source


class TestWhatCountsAsAFont:
    def test_only_font_extensions(self, tmp_path):
        (tmp_path / "Real.ttf").write_bytes(b"")
        (tmp_path / "Also.OTF").write_bytes(b"")
        (tmp_path / ".fonts_cache.json").write_text("{}", encoding="utf-8")
        (tmp_path / "LICENSE-ComicMono").write_text("licence", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

        assert [p.name for p in fonts._font_files(tmp_path)] == ["Also.OTF", "Real.ttf"]

    def test_a_directory_called_something_ttf_is_not_a_font(self, tmp_path):
        """Plausible on a machine where someone unzipped a font pack: the folder
        would be counted, the fallbacks suppressed, and no font would load."""
        (tmp_path / "FontPack.ttf").mkdir()
        assert fonts._font_files(tmp_path) == []

    def test_an_unreadable_folder_counts_as_zero_rather_than_raising(self, tmp_path):
        assert fonts._font_files(tmp_path / "does-not-exist") == []


class TestFallbackSelection:
    def test_the_fallback_is_exactly_the_two_comicmono_faces(self):
        assert assets.BUNDLED_FONTS == ("ComicMono.ttf", "ComicMono-Bold.ttf")

    def test_symbol_constant_agrees_across_modules(self):
        import importlib.util

        path = Path(__file__).resolve().parent.parent / "packaging" / "embed_assets.py"
        spec = importlib.util.spec_from_file_location("_embed_assets_symbol_check", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert fonts.SYMBOL_FONT_FILENAME == assets.SYMBOL_FONT == module.SYMBOL_FONT

    def test_empty_folder_falls_back(self, tmp_path):
        assert _names(fonts.getFonts(tmp_path)) == set(assets.BUNDLED_FONTS)

    def test_folder_holding_only_the_cache_and_the_licence_falls_back(self, tmp_path):
        """The two files the user named. A populated cache index in particular must
        not look like a populated folder -- it is keyed by files that are gone."""
        (tmp_path / ".fonts_cache.json").write_text(
            json.dumps({"Ghost.ttf": {"signature": "Ghost.ttf_1_1", "payload": {}}}),
            encoding="utf-8",
        )
        (tmp_path / "LICENSE-ComicMono").write_text("licence", encoding="utf-8")

        assert _names(fonts.getFonts(tmp_path)) == set(assets.BUNDLED_FONTS)

    def test_one_real_font_suppresses_the_fallback(self, tmp_path, real_font):
        shutil.copy(real_font, tmp_path / "MyFont.ttf")

        result = fonts.getFonts(tmp_path)
        assert _names(result) == {"MyFont.ttf"}

    def test_a_corrupt_font_still_leaves_the_user_with_something(self, tmp_path, caplog):
        """Zero *loadable* fonts is also zero fonts. The alternative -- honouring
        the file count and returning nothing -- offers no face at all and drops
        the typesetter to Pillow's bitmap default."""
        (tmp_path / "broken.ttf").write_bytes(b"not a font at all")

        with caplog.at_level("WARNING"):
            result = fonts.getFonts(tmp_path)

        assert _names(result) == set(assets.BUNDLED_FONTS)
        assert "corrupt or unreadable" in caplog.text

    def test_a_good_font_beside_a_corrupt_one_is_enough(self, tmp_path, real_font):
        shutil.copy(real_font, tmp_path / "MyFont.ttf")
        (tmp_path / "broken.ttf").write_bytes(b"not a font at all")

        assert _names(fonts.getFonts(tmp_path)) == {"MyFont.ttf"}


class TestSymbolExclusion:
    """The inpainting-only face is never a frontend font."""

    def test_symbol_alone_does_not_count_as_a_font(self, tmp_path, real_font):
        """A folder holding only the symbol face is an empty folder: the
        ComicMono fallbacks still stand in and the symbol is not offered."""
        shutil.copy(real_font, tmp_path / fonts.SYMBOL_FONT_FILENAME)
        assert _names(fonts.getFonts(tmp_path)) == set(assets.BUNDLED_FONTS)

    def test_symbol_beside_a_real_font_is_hidden(self, tmp_path, real_font):
        shutil.copy(real_font, tmp_path / "MyFont.ttf")
        shutil.copy(real_font, tmp_path / fonts.SYMBOL_FONT_FILENAME)

        assert _names(fonts.getFonts(tmp_path)) == {"MyFont.ttf"}


class TestPerGlyphWiring:
    """Every payload ships the preview's half of the symbol-fallback rule.

    The render resizes the fallback per primary face (`typeset.symbol_scale_for`)
    and splits mixed lines by `_needs_symbol` -- which needs to know what each
    face covers. The preview receives the exact same rule as numbers: the
    calibration factor on every selectable face, the _fallback_ face's own
    cmap as ranges, and the hidden payload carries the face the browser must
    render those characters with instead of a system font.
    """

    def test_payload_carries_scale_and_coverage(self, tmp_path, real_font):
        shutil.copy(real_font, tmp_path / "MyFont.ttf")
        payload = fonts.getFonts(tmp_path)[0]
        assert isinstance(payload["symbol_scale"], float)
        from fox_reader.typeset import symbol_scale_for

        assert payload["symbol_scale"] == symbol_scale_for("MyFont.ttf")
        coverage = payload["coverage"]
        assert isinstance(coverage, list)
        assert all(type(cp) is int for cp in coverage)
        assert len(coverage) >= 2 and len(coverage) % 2 == 0
        # A latin face covers the letters the test text needs.
        pairs = list(zip(coverage[0::2], coverage[1::2]))
        assert any(a <= ord("M") <= b for a, b in pairs)
        # ... and no heart, or the fallback would never trigger for it.
        assert not any(a <= 0x2661 <= b for a, b in pairs)
        # The per-size metrics table rides along, straight from the face.
        from fox_reader.typeset import line_metrics_for

        assert payload["line_metrics"] == line_metrics_for("MyFont.ttf")
        pairs = list(zip(payload["line_metrics"][0::2], payload["line_metrics"][1::2]))
        assert pairs and all(a + b > 0 for a, b in pairs)

    def test_fallback_payload_is_hidden_and_ranged(self):
        payload = fonts.symbol_fallback_payload()
        if payload is None:
            pytest.skip("symbol face not present in this checkout")
        assert payload["hidden"] is True
        assert payload["font_filename"] == fonts.SYMBOL_FONT_FILENAME
        assert payload["symbol_scale"] == 1.0
        coverage = payload["coverage"]
        pairs = list(zip(coverage[0::2], coverage[1::2]))
        for char in "♡★☆✿❤▲■✱❣✓":
            assert any(a <= ord(char) <= b for a, b in pairs), char

    def test_fallback_payload_caches_until_the_bytes_change(self, monkeypatch):
        from fox_reader import assets as assets_module

        if fonts.symbol_fallback_payload() is None:
            pytest.skip("symbol face not present in this checkout")

        raw_a = b"raw symbol bytes A"

        def raw_bytes_one(*a, **k):
            return raw_a

        monkeypatch.setattr(assets_module, "symbol_font_bytes", raw_bytes_one)
        fonts._SYMBOL_FALLBACK = None
        fonts._SYMBOL_FALLBACK_SIG = None
        first = fonts.symbol_fallback_payload()

        monkeypatch.setattr(assets_module, "symbol_font_bytes", lambda *a, **k: raw_a)
        assert fonts.symbol_fallback_payload() is first, "same bytes must hit the cache"

        monkeypatch.setattr(assets_module, "symbol_font_bytes", lambda *a, **k: None)
        assert fonts.symbol_fallback_payload() is None, "no bytes -> no payload"

    def test_fallback_payload_survives_unsanitizable_bytes(self, monkeypatch):
        from fox_reader import assets as assets_module

        monkeypatch.setattr(assets_module, "symbol_font_bytes", lambda *a, **k: b"not a font")
        fonts._SYMBOL_FALLBACK = None
        fonts._SYMBOL_FALLBACK_SIG = None
        try:
            assert fonts.symbol_fallback_payload() is None
        finally:
            fonts._SYMBOL_FALLBACK = None
            fonts._SYMBOL_FALLBACK_SIG = None

    def test_a_failed_face_is_not_reparsed_on_every_call(self, monkeypatch):
        """The signature gate comes first, so a *cached failure* is a hit too.

        The old check (`_SYMBOL_FALLBACK is not None and sig matches`) kept
        rebuilding -- and re-warning -- an unusable face on every single call,
        because the None it cached failed its own gate.
        """
        from fox_reader import assets as assets_module

        monkeypatch.setattr(assets_module, "symbol_font_bytes", lambda *a, **k: b"not a font")
        fonts._SYMBOL_FALLBACK = None
        fonts._SYMBOL_FALLBACK_SIG = None
        try:
            calls = {"n": 0}
            original = fonts._sanitize_and_get_bytes

            def counting(stream, *a, **k):
                calls["n"] += 1
                return original(stream, *a, **k)

            monkeypatch.setattr(fonts, "_sanitize_and_get_bytes", counting)
            assert fonts.symbol_fallback_payload() is None
            assert fonts.symbol_fallback_payload() is None
            assert fonts.symbol_fallback_payload() is None
            assert calls["n"] == 1, "a failed face must not be re-parsed per call"
        finally:
            fonts._SYMBOL_FALLBACK = None
            fonts._SYMBOL_FALLBACK_SIG = None

    def test_fresh_bytes_retry_after_a_failure(self, monkeypatch):
        """A new signature invalidates the cached failure, as it should."""
        from fox_reader import assets as assets_module

        real = assets_module.symbol_font_bytes
        monkeypatch.setattr(assets_module, "symbol_font_bytes", lambda *a, **k: b"junk one")
        fonts._SYMBOL_FALLBACK = None
        fonts._SYMBOL_FALLBACK_SIG = None
        try:
            assert fonts.symbol_fallback_payload() is None
            if real() is None:
                pytest.skip("symbol face not present in this checkout")
            monkeypatch.setattr(assets_module, "symbol_font_bytes", lambda *a, **k: real())
            # The *real* face's bytes differ from the junk hash, so the cached
            # failure must be dropped and the face derived.
            refreshed = fonts.symbol_fallback_payload()
            assert refreshed is not None, "changed bytes must invalidate the failure"
        finally:
            fonts._SYMBOL_FALLBACK = None
            fonts._SYMBOL_FALLBACK_SIG = None

    def test_valid_payload_accepts_the_metrics_table(self, tmp_path, real_font):
        shutil.copy(real_font, tmp_path / "MyFont.ttf")
        payload = dict(fonts.getFonts(tmp_path)[0])
        assert fonts._valid_payload(payload)
        # An empty table is a legitimate "unmeasurable face" answer.
        assert fonts._valid_payload({**payload, "line_metrics": []})
        # Odd length / non-ints / bools are not a table the preview can walk.
        for bad in (
            [5, 2, 4],  # odd length
            [5, "2"],  # non-int
            [True, False],  # bool is not int here
            "5 2",  # not even a list
            None,  # missing
        ):
            assert not fonts._valid_payload({**payload, "line_metrics": bad}), bad

    def test_old_cache_entries_without_the_wiring_reparse(self, tmp_path, real_font, monkeypatch):
        """A `symbol_scale`-less payload is an old cache entry, not a valid
        one: it is re-parsed once (the deliberate upgrade cost) and gains the
        wiring in the same pass."""
        payload = fonts._primary_payload(real_font.read_bytes(), "MyFont.ttf", {
            "family_name": "My Font", "subfamily": "Regular",
            "bold": False, "italic": False,
            "units_per_em": 1800, "ascent": 1500, "descent": 300,
        })
        sanitized = payload.get("coverage")
        # The shape without the wiring:
        legacy = {
            "signature": "MyFont.ttf_1_1",
            "payload": {
                "data_uri": payload["data_uri"], "format": payload["format"],
                "font_name": payload["font_name"],
                "font_filename": "MyFont.ttf", "family_name": "My Font",
                "bold": False, "italic": False,
                "unitsPerEm": 1800, "ascent": 1500, "descent": 300,
            },
        }
        (tmp_path / ".fonts_cache.json").write_text(json.dumps({"MyFont.ttf": legacy}), encoding="utf-8")
        # And the file itself must be there, so the re-parse is possible:
        shutil.copy(real_font, tmp_path / "MyFont.ttf")

        calls = {"sanitised": 0}
        original = fonts._sanitize_and_get_bytes

        def counting(stream, *a, **k):
            calls["sanitised"] += 1
            return original(stream, *a, **k)

        monkeypatch.setattr(fonts, "_sanitize_and_get_bytes", counting)
        out = fonts.getFonts(tmp_path)[0]
        assert calls["sanitised"] == 1, "a pre-wiring cache entry must re-parse once"
        assert out["symbol_scale"] == payload["symbol_scale"]
        assert sanitized is not None and out["coverage"] == sanitized
        # The upgrade also gains the metrics table in the same pass.
        from fox_reader.typeset import line_metrics_for

        assert out["line_metrics"] == line_metrics_for("MyFont.ttf")


class TestCache:
    def test_the_index_is_written_and_reused(self, tmp_path, real_font):
        shutil.copy(real_font, tmp_path / "MyFont.ttf")

        first = fonts.getFonts(tmp_path)
        cache = tmp_path / ".fonts_cache.json"
        assert cache.is_file()
        assert "MyFont.ttf" in json.loads(cache.read_text(encoding="utf-8"))

        # Second pass takes the cached payload; sanitising is the expensive part
        # and must not run again for an unchanged file.
        def explode(*_args, **_kwargs):
            raise AssertionError("re-sanitised a file the cache already covered")

        original = fonts._sanitize_and_get_bytes
        fonts._sanitize_and_get_bytes = explode
        try:
            second = fonts.getFonts(tmp_path)
        finally:
            fonts._sanitize_and_get_bytes = original

        assert second == first

    def test_a_corrupt_index_is_discarded_not_fatal(self, tmp_path, real_font):
        shutil.copy(real_font, tmp_path / "MyFont.ttf")
        (tmp_path / ".fonts_cache.json").write_text("{ this is not json", encoding="utf-8")

        assert _names(fonts.getFonts(tmp_path)) == {"MyFont.ttf"}

    @pytest.mark.parametrize(
        "text",
        [
            "5",
            '"a string"',
            '["MyFont.ttf"]',
            '{"MyFont.ttf": "not an entry"}',
            '{"MyFont.ttf": null}',
            '{"MyFont.ttf": [1, 2]}',
        ],
        ids=["number", "string", "list", "entry-string", "entry-null", "entry-list"],
    )
    def test_valid_json_of_the_wrong_shape_is_discarded_not_fatal(self, tmp_path, real_font, text):
        """Unparseable JSON was already survivable; this is JSON that parses
        perfectly and is not a mapping of mappings. The file lives in the folder
        users are invited to drop things into, so every shape here is reachable,
        and each one used to reach a reader that assumed the shape and raised."""
        shutil.copy(real_font, tmp_path / "MyFont.ttf")
        (tmp_path / ".fonts_cache.json").write_text(text, encoding="utf-8")

        assert _names(fonts.getFonts(tmp_path)) == {"MyFont.ttf"}

    def test_a_matching_signature_with_a_broken_payload_is_re_read(self, tmp_path, real_font):
        """The dangerous shape, because the signature still matches and so the
        payload is handed out without a second look. Everything downstream reads
        it with ``.get`` and then calls ``.lower()`` on what comes back, which is
        an AttributeError rather than a missing font."""
        shutil.copy(real_font, tmp_path / "MyFont.ttf")
        fonts.getFonts(tmp_path)

        cache_path = tmp_path / ".fonts_cache.json"
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["MyFont.ttf"]["payload"] = {"font_name": None, "family_name": 7}
        cache_path.write_text(json.dumps(cached), encoding="utf-8")

        result = fonts.getFonts(tmp_path)
        assert _names(result) == {"MyFont.ttf"}
        assert isinstance(result[0]["data_uri"], str)
        assert result[0]["data_uri"].startswith("data:")

    def test_a_changed_file_is_re_read(self, tmp_path, real_font):
        """The signature is name plus mtime plus size, so replacing the file with a
        different one under the same name must not serve the old payload."""
        target = tmp_path / "MyFont.ttf"
        shutil.copy(real_font, target)
        fonts.getFonts(tmp_path)

        bold = REPO_FONTS / "ComicMono-Bold.ttf"
        if not bold.is_file():
            pytest.skip("the checkout has no fonts/ComicMono-Bold.ttf")
        shutil.copy(bold, target)

        result = fonts.getFonts(tmp_path)
        assert _names(result) == {"MyFont.ttf"}
        assert result[0]["bold"] is True


class TestFolderCreation:
    def test_a_missing_folder_is_created_and_falls_back(self, tmp_path):
        """A fresh install has no fonts/ at all until something asks for it."""
        target = tmp_path / "fonts"
        result = fonts.getFonts(target)

        assert target.is_dir()
        assert _names(result) == set(assets.BUNDLED_FONTS)

    def test_a_font_dir_that_cannot_be_created_falls_back_instead_of_raising(self, tmp_path, caplog):
        """A path already occupied by a file stands in for the general case -- a
        read-only install, a revoked ACL. ``getFonts`` is called from
        ``FontService.__init__``, which runs in the app's lifespan, so an OSError
        escaping from here took the whole server's startup down with it."""
        target = tmp_path / "fonts"
        target.write_text("not a directory", encoding="utf-8")

        with caplog.at_level("WARNING"):
            result = fonts.getFonts(target)

        assert _names(result) == set(assets.BUNDLED_FONTS)
        assert "Cannot create font directory" in caplog.text
