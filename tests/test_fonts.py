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
