"""`FoxConfig` and the `ConfigManager` that reads and writes it.

Two things here are load-bearing beyond "does validation work":

* A config file that fails validation is replaced with defaults, silently. That
  is the right call at startup -- refusing to boot over a stray character helps
  nobody -- but it means every path that can put a bad value into the model is a
  path that can lose *all* of the user's settings. `TestUpdate` pins the
  atomicity that stops it.
* The launcher parses `host` and `port` out of the saved YAML in C, before any
  Python exists to ask. `TestSavedFileShape` pins the flat `key: value` layout
  that parser relies on, and the `user_config.yaml`-over-`fox_config.yaml`
  precedence it has to reproduce.
"""
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fox_reader.config import ConfigManager, FoxConfig


class TestFoxConfigDefaults:
    def test_defaults(self):
        cfg = FoxConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 7954
        assert cfg.mtl_dir == Path("models")
        assert cfg.theme == "dark"
        assert cfg.layout == "basic"

    def test_custom_values(self):
        cfg = FoxConfig(
            host="0.0.0.0",
            port=8080,
            mtl_dir=Path("/data/mtl"),
            theme="light",
            layout="reader",
        )
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080
        assert cfg.mtl_dir == Path("/data/mtl")
        assert cfg.theme == "light"
        assert cfg.layout == "reader"

    def test_frozen(self):
        cfg = FoxConfig()
        with pytest.raises(ValidationError):
            cfg.host = "0.0.0.0"  # type: ignore[misc]

    def test_extra_fields_ignored(self):
        """Fields dropped by a future version must not make old files unloadable."""
        cfg = FoxConfig.model_validate({"host": "1.2.3.4", "low_memory_ocr": True})
        assert cfg.host == "1.2.3.4"
        assert not hasattr(cfg, "low_memory_ocr")


class TestHostAndPort:
    def test_whitespace_stripped(self):
        assert FoxConfig(host="  1.2.3.4  ").host == "1.2.3.4"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_rejected(self, value):
        with pytest.raises(ValidationError):
            FoxConfig(host=value)

    @pytest.mark.parametrize("value", [0, -1, 65536, 70000])
    def test_port_out_of_range(self, value):
        with pytest.raises(ValidationError):
            FoxConfig(port=value)

    @pytest.mark.parametrize("value", [1, 7954, 65535])
    def test_port_bounds_inclusive(self, value):
        assert FoxConfig(port=value).port == value

    def test_port_from_string(self):
        """YAML quoting a number must not break the launcher's agreement with it."""
        assert FoxConfig(port="8080").port == 8080


class TestThemeAndLayout:
    @pytest.mark.parametrize("value,expected", [("dark", "dark"), ("  DARK  ", "dark"), ("Light", "light")])
    def test_theme_normalised(self, value, expected):
        assert FoxConfig(theme=value).theme == expected

    @pytest.mark.parametrize("value", ["purple", "", "auto"])
    def test_theme_rejected(self, value):
        with pytest.raises(ValidationError):
            FoxConfig(theme=value)

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("basic", "basic"),
            (" Default ", "default"),
            ("READER", "reader"),
            ("Translation", "translation"),
        ],
    )
    def test_layout_normalised(self, value, expected):
        assert FoxConfig(layout=value).layout == expected

    @pytest.mark.parametrize("value", ["grid", "", "compact"])
    def test_layout_rejected(self, value):
        with pytest.raises(ValidationError):
            FoxConfig(layout=value)


class TestResolveMtlDir:
    def test_relative_is_joined_to_root(self):
        cfg = FoxConfig(mtl_dir=Path("m"))
        assert cfg.resolve_mtl_dir(Path("/root")) == Path("/root") / "m"

    def test_absolute_is_returned_as_is(self, tmp_path):
        absolute = tmp_path / "elsewhere"
        cfg = FoxConfig(mtl_dir=absolute)
        assert cfg.resolve_mtl_dir(Path("/root")) == absolute

    def test_default_resolves_under_root(self):
        assert FoxConfig().resolve_mtl_dir(Path("/root")) == Path("/root") / "models"

    def test_string_becomes_path(self):
        assert FoxConfig(mtl_dir="a/b").mtl_dir == Path("a/b")


class TestSavedFileShape:
    """The saved file is also a C parser's input; its shape is part of the API."""

    def test_dump_is_json_safe(self):
        """`mtl_dir` has to leave as a string, or yaml.safe_dump refuses a Path."""
        assert FoxConfig().model_dump(mode="json") == {
            "host": "127.0.0.1",
            "port": 7954,
            "mtl_dir": "models",
            "theme": "dark",
            "layout": "basic",
        }

    def test_saved_yaml_is_flat_and_ordered(self, tmp_path):
        mgr = ConfigManager(tmp_path)
        mgr.load()

        lines = [
            line for line in mgr.config_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        # One `key: value` per line, in field order, no nesting and no anchors --
        # the launcher scans for `host:`/`port:` with a line-oriented reader.
        assert lines == [
            "host: 127.0.0.1",
            "port: 7954",
            "mtl_dir: models",
            "theme: dark",
            "layout: basic",
        ]

    def test_round_trip(self, tmp_path):
        mgr = ConfigManager(tmp_path)
        mgr.load()
        mgr.update(theme="light", layout="reader")

        reloaded = ConfigManager(tmp_path).load()
        assert reloaded.theme == "light"
        assert reloaded.layout == "reader"


class TestConfigManagerLoad:
    def test_missing_file_is_created_with_defaults(self, tmp_path):
        mgr = ConfigManager(tmp_path)
        cfg = mgr.load()

        assert cfg == FoxConfig()
        assert mgr.config_path == tmp_path / "fox_config.yaml"
        assert mgr.config_file_class == "default"
        assert mgr.config_path.exists()

    def test_missing_root_is_created(self, tmp_path):
        root = tmp_path / "nested" / "config"
        cfg = ConfigManager(root).load()

        assert cfg == FoxConfig()
        assert (root / "fox_config.yaml").exists()

    def test_valid_file_is_honoured(self, tmp_path):
        (tmp_path / "fox_config.yaml").write_text(
            "host: 0.0.0.0\nport: 3000\ntheme: light\n", encoding="utf-8"
        )
        cfg = ConfigManager(tmp_path).load()

        assert (cfg.host, cfg.port, cfg.theme) == ("0.0.0.0", 3000, "light")
        # Absent keys fall back to defaults rather than failing the load.
        assert cfg.layout == "basic"

    def test_user_config_wins(self, tmp_path):
        (tmp_path / "fox_config.yaml").write_text("port: 3000\n", encoding="utf-8")
        (tmp_path / "user_config.yaml").write_text("port: 4000\n", encoding="utf-8")

        mgr = ConfigManager(tmp_path)
        cfg = mgr.load()

        assert cfg.port == 4000
        assert mgr.config_file_class == "user"
        assert mgr.config_path == tmp_path / "user_config.yaml"

    @pytest.mark.parametrize(
        "contents",
        [
            ": invalid: yaml: {{{{",  # unparseable
            "port: 70000\n",  # out of range
            "theme: purple\n",  # not an allowed value
            "host: ''\n",  # empty after strip
            "- a\n- b\n",  # a list where a mapping belongs
        ],
        ids=["malformed", "out-of-range", "bad-enum", "empty-host", "not-a-mapping"],
    )
    def test_unusable_file_is_replaced_with_defaults(self, tmp_path, contents):
        path = tmp_path / "fox_config.yaml"
        path.write_text(contents, encoding="utf-8")

        cfg = ConfigManager(tmp_path).load()

        assert cfg == FoxConfig()
        # And the bad file is rewritten, so the next launch starts clean.
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == FoxConfig().model_dump(mode="json")

    def test_empty_file_loads_defaults(self, tmp_path):
        """An empty file is valid input -- every field has a default -- so it
        loads rather than being treated as corrupt, and is left as it is. The
        launcher's own fallbacks match these defaults, so the two still agree on
        host and port with nothing written down."""
        path = tmp_path / "fox_config.yaml"
        path.write_text("", encoding="utf-8")

        assert ConfigManager(tmp_path).load() == FoxConfig()
        assert path.read_text(encoding="utf-8") == ""


class TestUpdate:
    def test_mutable_field(self, tmp_path):
        mgr = ConfigManager(tmp_path)
        mgr.load()

        cfg = mgr.update(theme="light")

        assert cfg.theme == "light"
        assert mgr.config is cfg
        assert yaml.safe_load(mgr.config_path.read_text(encoding="utf-8"))["theme"] == "light"

    @pytest.mark.parametrize("field,value", [("host", "0.0.0.0"), ("port", 9000)])
    def test_immutable_fields_refused(self, tmp_path, field, value):
        """host/port are read once at boot; changing them at runtime would leave
        the launcher probing an address nothing is listening on."""
        mgr = ConfigManager(tmp_path)
        mgr.load()

        with pytest.raises(PermissionError, match=field):
            mgr.update(**{field: value})

    def test_rejected_update_leaves_config_untouched(self, tmp_path):
        """The regression this guards is a total settings wipe, not a bad theme.

        `update()` used to assign the unvalidated copy before validating it, so a
        rejected value stayed in memory. The next `save()` -- from any unrelated
        setting change -- wrote it to disk, and on the following launch `load()`
        rejected the file and reset *every* field to its default.
        """
        mgr = ConfigManager(tmp_path)
        mgr.load()
        mgr.update(layout="reader")

        with pytest.raises(ValidationError):
            mgr.update(theme="purple")

        assert mgr.config is not None
        assert mgr.config.theme == "dark"
        # Everything set before the failure survives, in memory and on disk.
        assert mgr.config.layout == "reader"

        mgr.save()
        reloaded = ConfigManager(tmp_path).load()
        assert (reloaded.theme, reloaded.layout) == ("dark", "reader")

    def test_partial_update_keeps_other_fields(self, tmp_path):
        mgr = ConfigManager(tmp_path)
        mgr.load()
        mgr.update(layout="translation")

        cfg = mgr.update(theme="light")

        assert cfg.layout == "translation"
        assert cfg.host == "127.0.0.1"

    def test_values_are_normalised(self, tmp_path):
        mgr = ConfigManager(tmp_path)
        mgr.load()

        assert mgr.update(theme="  LIGHT  ").theme == "light"

    def test_unknown_field_ignored(self, tmp_path):
        """`model_copy(update=...)` sets unknown names, then the revalidating
        round-trip through `model_dump()` drops them again."""
        mgr = ConfigManager(tmp_path)
        mgr.load()

        cfg = mgr.update(nonexistent="x")

        assert not hasattr(cfg, "nonexistent")

    def test_before_load(self, tmp_path):
        with pytest.raises(RuntimeError):
            ConfigManager(tmp_path).update(theme="light")


class TestSaveAndReset:
    def test_save_before_load(self, tmp_path):
        with pytest.raises(RuntimeError):
            ConfigManager(tmp_path).save()

    def test_reset_without_load(self, tmp_path):
        """Reset is a recovery path, so it must not require a successful load."""
        mgr = ConfigManager(tmp_path)

        cfg = mgr.reset()

        assert cfg == FoxConfig()
        assert mgr.config_path.exists()

    def test_reset_discards_changes(self, tmp_path):
        mgr = ConfigManager(tmp_path)
        mgr.load()
        mgr.update(theme="light", layout="reader")

        cfg = mgr.reset()

        assert (cfg.theme, cfg.layout) == ("dark", "basic")
        assert ConfigManager(tmp_path).load() == FoxConfig()

    def test_reset_writes_to_the_file_it_read(self, tmp_path):
        """With a user_config.yaml present, reset must not resurrect defaults in
        fox_config.yaml and leave the user file overriding them."""
        (tmp_path / "fox_config.yaml").write_text("theme: light\n", encoding="utf-8")
        (tmp_path / "user_config.yaml").write_text("theme: light\n", encoding="utf-8")

        mgr = ConfigManager(tmp_path)
        mgr.reset()

        assert yaml.safe_load((tmp_path / "user_config.yaml").read_text(encoding="utf-8"))["theme"] == "dark"
        assert yaml.safe_load((tmp_path / "fox_config.yaml").read_text(encoding="utf-8"))["theme"] == "light"
