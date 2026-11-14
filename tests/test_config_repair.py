"""Startup reconciliation preserves usable values without creating backups."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import yaml

from fox_reader import config as config_module
from fox_reader.config import ConfigManager, FoxConfig


def _document(**changes):
    return {**FoxConfig().model_dump(mode="json"), **changes}


def _write(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _assert_only_config_files(directory, *filenames):
    assert {path.name for path in directory.iterdir()} == {"fox_config.yaml", *filenames}


def test_user_file_still_creates_missing_default_file(tmp_path):
    user = tmp_path / "user_config.yaml"
    _write(user, _document(port=8123, theme="light"))

    manager = ConfigManager(tmp_path)
    config = manager.load()

    assert (config.port, config.theme) == (8123, "light")
    assert manager.config_path == user
    assert manager.config_file_class == "user"
    assert _read(tmp_path / "fox_config.yaml") == _document()
    _assert_only_config_files(tmp_path, "user_config.yaml")


def test_user_file_replaces_default_file_without_inheriting_its_values(tmp_path):
    default = tmp_path / "fox_config.yaml"
    user = tmp_path / "user_config.yaml"
    _write(default, _document(host="0.0.0.0", port=8123, theme="light"))
    _write(user, {"layout": "vulpes", "future": {"choices": ["keep", 42]}})

    config = ConfigManager(tmp_path).load()

    assert (config.host, config.port, config.theme) == ("127.0.0.1", 7954, "dark")
    assert config.layout == "vulpes"
    assert _read(default)["port"] == 8123
    assert _read(user) == _document(layout="vulpes", future={"choices": ["keep", 42]})
    _assert_only_config_files(tmp_path, "user_config.yaml")


def test_both_files_are_repaired_independently(tmp_path, caplog):
    default = tmp_path / "fox_config.yaml"
    user = tmp_path / "user_config.yaml"
    _write(default, {"port": 8123, "theme": "purple", "old_feature": {"enabled": True}})
    _write(user, {"port": -1, "theme": "light", "layout": "vulpes"})

    with caplog.at_level(logging.INFO, logger="fox_reader.config"):
        config = ConfigManager(tmp_path).load()

    assert (config.port, config.theme, config.layout) == (7954, "light", "vulpes")
    assert _read(default) == _document(port=8123, old_feature={"enabled": True})
    assert _read(user) == _document(theme="light", layout="vulpes")
    _assert_only_config_files(tmp_path, "user_config.yaml")
    assert "fox_config.yaml" in caplog.text
    assert "user_config.yaml" in caplog.text
    assert "theme" in caplog.text and "port" in caplog.text


@pytest.mark.parametrize("filename", ["fox_config.yaml", "user_config.yaml"])
def test_missing_schema_fields_are_written_and_user_data_survives(tmp_path, filename, caplog):
    path = tmp_path / filename
    data = {"port": 8123, "theme": "light", "future": {"unicode": "日本語", "items": [1, None]}}
    _write(path, data)

    with caplog.at_level(logging.INFO, logger="fox_reader.config"):
        config = ConfigManager(tmp_path).load()

    assert (config.port, config.theme, config.mtl_dir) == (8123, "light", Path("models"))
    assert _read(path) == _document(**data)
    _assert_only_config_files(tmp_path, filename)
    assert "mtl_dir" in caplog.text and "layout" in caplog.text


@pytest.mark.parametrize(("field", "value"), [
    ("host", "http://localhost:8123"),
    ("host", "bad host"),
    ("host", "999.999.999.999"),
    ("port", True),
    ("port", 8123.0),
    ("port", 0),
    ("port", 65536),
    ("port", "8123oops"),
    ("mtl_dir", ""),
    ("mtl_dir", "bad\x00path"),
    ("theme", "purple"),
    ("layout", "grid"),
])
def test_invalid_field_does_not_reset_other_fields(tmp_path, caplog, field, value):
    path = tmp_path / "fox_config.yaml"
    values = _document(port=8123, theme="light", layout="vulpes", mtl_dir="my models")
    values.update({field: value, "future": {"names": ["A", "B"], "enabled": False}})
    _write(path, values)

    with caplog.at_level(logging.INFO, logger="fox_reader.config"):
        config = ConfigManager(tmp_path).load()

    expected = {**values, field: _document()[field]}
    assert _read(path) == expected
    assert config.model_dump(mode="json") == {key: expected[key] for key in _document()}
    _assert_only_config_files(tmp_path)
    assert field in caplog.text


def test_multiple_invalid_fields_are_repaired_together(tmp_path):
    path = tmp_path / "fox_config.yaml"
    _write(path, _document(host=None, port=0, theme="light", layout="grid", mtl_dir="models elsewhere"))

    config = ConfigManager(tmp_path).load()

    assert config == FoxConfig(theme="light", mtl_dir="models elsewhere")
    assert _read(path) == _document(theme="light", mtl_dir="models elsewhere")
    _assert_only_config_files(tmp_path)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-16-be"])
def test_bom_encoded_manual_config_is_accepted(tmp_path, encoding):
    path = tmp_path / "fox_config.yaml"
    text = yaml.safe_dump(_document(port=8123, theme="light", mtl_dir="日本語 models"), allow_unicode=True)
    raw = text.encode(encoding)
    if encoding == "utf-16-be":
        raw = b"\xfe\xff" + raw
    path.write_bytes(raw)

    config = ConfigManager(tmp_path).load()

    assert (config.port, config.theme, config.mtl_dir) == (8123, "light", Path("日本語 models"))
    _assert_only_config_files(tmp_path)


def test_normalized_values_are_saved_without_extra_files(tmp_path):
    path = tmp_path / "fox_config.yaml"
    _write(path, _document(host="  localhost  ", port="8123", theme=" LIGHT ", layout=" Vulpes "))

    config = ConfigManager(tmp_path).load()

    assert config == FoxConfig(host="localhost", port=8123, theme="light", layout="vulpes")
    assert _read(path) == _document(host="localhost", port=8123, theme="light", layout="vulpes")
    _assert_only_config_files(tmp_path)


@pytest.mark.parametrize("filename", ["fox_config.yaml", "user_config.yaml"])
@pytest.mark.parametrize("raw", [
    b"host: localhost\nport: [8123\n",
    b"- a\n- b\n",
    b"a scalar\n",
    b"false\n",
    b"host: localhost\n---\nport: 8123\n",
    b"theme: light\n# invalid byte \xff\n",
])
def test_unreadable_content_is_replaced_with_defaults_without_backups(tmp_path, caplog, filename, raw):
    path = tmp_path / filename
    path.write_bytes(raw)

    with caplog.at_level(logging.WARNING, logger="fox_reader.config"):
        config = ConfigManager(tmp_path).load()

    assert config == FoxConfig()
    assert _read(path) == _document()
    _assert_only_config_files(tmp_path, filename)
    assert filename in caplog.text


def test_duplicate_fields_keep_last_value_without_extra_files(tmp_path, caplog):
    path = tmp_path / "fox_config.yaml"
    raw = b"port: 8123\nport: 8124\ntheme: light\n"
    path.write_bytes(raw)

    with caplog.at_level(logging.WARNING, logger="fox_reader.config"):
        config = ConfigManager(tmp_path).load()

    assert config.port == 8124 and config.theme == "light"
    assert _read(path) == _document(port=8124, theme="light")
    _assert_only_config_files(tmp_path)
    assert "fox_config.yaml" in caplog.text and "duplicate" in caplog.text.lower()


def test_valid_complete_file_is_not_rewritten(tmp_path, monkeypatch):
    path = tmp_path / "fox_config.yaml"
    text = "# Keep the user's comments and formatting.\n" + yaml.safe_dump(_document(theme="light"), sort_keys=False)
    path.write_text(text, encoding="utf-8")
    original = path.read_bytes()

    def unexpected_replace(*args, **kwargs):
        pytest.fail("Loading a complete, valid config must not write it")

    monkeypatch.setattr(config_module.os, "replace", unexpected_replace)
    manager = ConfigManager(tmp_path)
    assert manager.load().theme == "light"
    assert manager.load().theme == "light"
    assert path.read_bytes() == original
    _assert_only_config_files(tmp_path)


def test_repaired_file_has_no_further_writes_or_extra_files_on_next_load(tmp_path, monkeypatch):
    path = tmp_path / "fox_config.yaml"
    _write(path, {"port": 8123, "theme": "purple"})
    ConfigManager(tmp_path).load()
    original = path.read_bytes()
    _assert_only_config_files(tmp_path)

    def unexpected_replace(*args, **kwargs):
        pytest.fail("A second load must not repeat an already completed repair")

    monkeypatch.setattr(config_module.os, "replace", unexpected_replace)
    assert ConfigManager(tmp_path).load().port == 8123
    assert path.read_bytes() == original
    _assert_only_config_files(tmp_path)


def test_read_failure_does_not_replace_existing_config(tmp_path, monkeypatch, caplog):
    path = tmp_path / "fox_config.yaml"
    _write(path, _document(port=8123, theme="light"))
    original = path.read_bytes()
    original_open = Path.open

    def fail_read(self, mode="r", *args, **kwargs):
        if self == path and mode in {"r", "rb"}:
            raise PermissionError("configuration is temporarily unreadable")
        return original_open(self, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_read)
        with caplog.at_level(logging.WARNING, logger="fox_reader.config"):
            assert ConfigManager(tmp_path).load() == FoxConfig()

    assert path.read_bytes() == original
    _assert_only_config_files(tmp_path)
    assert "fox_config.yaml" in caplog.text


def test_repair_replace_failure_preserves_original_and_uses_repaired_values(tmp_path, monkeypatch, caplog):
    path = tmp_path / "fox_config.yaml"
    _write(path, _document(port=8123, theme="purple"))
    original = path.read_bytes()
    replace = os.replace

    def fail_config_replace(source, destination):
        if Path(destination) == path:
            raise PermissionError("config is locked")
        return replace(source, destination)

    monkeypatch.setattr(config_module.os, "replace", fail_config_replace)
    with caplog.at_level(logging.WARNING, logger="fox_reader.config"):
        config = ConfigManager(tmp_path).load()

    assert config.port == 8123 and config.theme == "dark"
    assert path.read_bytes() == original
    _assert_only_config_files(tmp_path)
    assert "fox_config.yaml" in caplog.text


@pytest.mark.parametrize("raw", [
    b"port: [8123\n",
    b"theme: purple\n",
    b"theme: light\ntheme: dark\n",
])
def test_repairs_do_not_attempt_to_create_backups(tmp_path, monkeypatch, raw):
    path = tmp_path / "fox_config.yaml"
    path.write_bytes(raw)
    original_os_open = os.open

    def reject_backup_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fsdecode(file).endswith(".bak"):
            pytest.fail("Configuration repairs must not create a backup")
        return original_os_open(file, *args, **kwargs)

    monkeypatch.setattr(os, "open", reject_backup_open)
    assert ConfigManager(tmp_path).load() == FoxConfig()

    assert _read(path) == _document()
    _assert_only_config_files(tmp_path)


@pytest.mark.parametrize("operation", ["update", "reset"])
def test_failed_runtime_write_rolls_back_memory_and_preserves_extra_data(tmp_path, monkeypatch, operation):
    path = tmp_path / "fox_config.yaml"
    _write(path, _document(theme="light", layout="vulpes", future={"keep": [1, 2]}))
    manager = ConfigManager(tmp_path)
    original_config = manager.load()
    original_bytes = path.read_bytes()

    def fail_replace(*args):
        raise PermissionError("config locked")

    monkeypatch.setattr(config_module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        if operation == "update":
            manager.update(theme="dark")
        else:
            manager.reset()

    assert manager.config is original_config
    assert path.read_bytes() == original_bytes
    _assert_only_config_files(tmp_path)


def test_runtime_update_preserves_unknown_nested_data(tmp_path):
    path = tmp_path / "fox_config.yaml"
    extra = {"keep": [1, {"label": "日本語"}], "enabled": None}
    _write(path, _document(future=extra))
    manager = ConfigManager(tmp_path)
    manager.load()

    manager.update(mtl_dir=Path("my models"), theme="light")

    assert _read(path) == _document(mtl_dir="my models", theme="light", future=extra)
    assert ConfigManager(tmp_path).load().mtl_dir == Path("my models")
    _assert_only_config_files(tmp_path)


def test_reset_creates_a_missing_config_directory(tmp_path):
    directory = tmp_path / "nested" / "config"
    manager = ConfigManager(directory)

    assert manager.reset() == FoxConfig()
    assert _read(directory / "fox_config.yaml") == _document()


def test_new_schema_field_is_added_to_both_existing_files(tmp_path, monkeypatch):
    default = tmp_path / "fox_config.yaml"
    user = tmp_path / "user_config.yaml"
    _write(default, _document(port=8123, future={"keep": True}))
    _write(user, _document(theme="light"))

    class UpdatedConfig(FoxConfig):
        new_preference: str = "automatic"

    monkeypatch.setattr(config_module, "FoxConfig", UpdatedConfig)
    config = ConfigManager(tmp_path).load()

    assert config.new_preference == "automatic"
    assert config.theme == "light"
    assert _read(default) == _document(port=8123, future={"keep": True}, new_preference="automatic")
    assert _read(user) == _document(theme="light", new_preference="automatic")
    _assert_only_config_files(tmp_path, "user_config.yaml")


def test_yaml_merge_keeps_explicit_overrides_without_extra_files(tmp_path):
    path = tmp_path / "fox_config.yaml"
    path.write_text(
        "shared: &shared\n  port: 8123\n  theme: light\n<<: *shared\nport: 8124\n",
        encoding="utf-8",
    )

    config = ConfigManager(tmp_path).load()

    assert (config.port, config.theme) == (8124, "light")
    assert _read(path) == _document(port=8124, theme="light", shared={"port": 8123, "theme": "light"})
    _assert_only_config_files(tmp_path)


def test_successive_repairs_keep_latest_valid_values_without_extra_files(tmp_path):
    path = tmp_path / "fox_config.yaml"
    first = b"port: 8123\ntheme: purple\nfuture:\n  keep: 1\n"
    second = b"port: 8124\ntheme: orange\nfuture:\n  keep: 2\n"
    path.write_bytes(first)
    ConfigManager(tmp_path).load()
    assert _read(path) == _document(port=8123, future={"keep": 1})
    _assert_only_config_files(tmp_path)
    path.write_bytes(second)

    assert ConfigManager(tmp_path).load().port == 8124
    assert _read(path) == _document(port=8124, future={"keep": 2})
    _assert_only_config_files(tmp_path)


@pytest.mark.parametrize("operation", ["update", "reset"])
def test_runtime_write_does_not_overwrite_an_external_edit(tmp_path, operation):
    path = tmp_path / "fox_config.yaml"
    manager = ConfigManager(tmp_path)
    original = manager.load()
    _write(path, _document(theme="light", future={"keep": "manual edit"}))
    external = path.read_bytes()

    with pytest.raises(OSError):
        if operation == "update":
            manager.update(layout="vulpes")
        else:
            manager.reset()

    assert manager.config is original
    assert path.read_bytes() == external
    _assert_only_config_files(tmp_path)


def test_failed_repair_cannot_be_overwritten_by_later_runtime_update(tmp_path, monkeypatch):
    path = tmp_path / "fox_config.yaml"
    path.write_bytes(b"port: 8123\ntheme: purple\n")
    original = path.read_bytes()
    manager = ConfigManager(tmp_path)

    def fail_replace(*args):
        raise PermissionError("config locked")

    with monkeypatch.context() as patch:
        patch.setattr(config_module.os, "replace", fail_replace)
        config = manager.load()

    with pytest.raises(OSError):
        manager.update(layout="vulpes")
    assert manager.config is config
    assert path.read_bytes() == original

    manager.load()
    manager.update(layout="vulpes")
    assert _read(path) == _document(port=8123, layout="vulpes")


@pytest.mark.parametrize("text", [
    "{host: localhost, port: 8123, mtl_dir: models, theme: light, layout: basic}\n",
    "'host': localhost\n\"port\": '8123'\nmtl_dir: models\ntheme: light\nlayout: basic\n",
    "host: localhost\nport: 017673\nmtl_dir: models\ntheme: light\nlayout: basic\n",
])
def test_alternative_yaml_address_forms_are_normalized_once(tmp_path, monkeypatch, text):
    path = tmp_path / "fox_config.yaml"
    path.write_text(text, encoding="utf-8")

    config = ConfigManager(tmp_path).load()

    assert (config.host, config.port, config.theme) == ("localhost", 8123, "light")
    saved = path.read_bytes()
    assert saved.startswith(b"host: localhost\nport: 8123\n")
    assert _read(path) == _document(host="localhost", port=8123, theme="light")
    _assert_only_config_files(tmp_path)

    def unexpected_replace(*args):
        pytest.fail("Canonical address fields must not require another startup rewrite")

    monkeypatch.setattr(config_module.os, "replace", unexpected_replace)
    assert ConfigManager(tmp_path).load() == config
    assert path.read_bytes() == saved


def test_missing_directory_creation_failure_uses_defaults_without_crashing(tmp_path, monkeypatch, caplog):
    directory = tmp_path / "unwritable" / "config"
    original_mkdir = Path.mkdir

    def fail_mkdir(self, *args, **kwargs):
        if self == directory:
            raise PermissionError("parent directory is read-only")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    with caplog.at_level(logging.WARNING, logger="fox_reader.config"):
        manager = ConfigManager(directory)
        assert manager.load() == FoxConfig()

    assert not directory.exists()
    assert "config" in caplog.text
    with pytest.raises(OSError):
        manager.update(theme="light")


def test_failed_repair_sync_removes_temporary_file_and_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "fox_config.yaml"
    raw = b"port: [8123\n"
    path.write_bytes(raw)

    def fail_sync(descriptor):
        raise OSError("disk full while syncing repaired config")

    monkeypatch.setattr(config_module.os, "fsync", fail_sync)
    assert ConfigManager(tmp_path).load() == FoxConfig()
    assert path.read_bytes() == raw
    _assert_only_config_files(tmp_path)


@pytest.mark.parametrize("operation", ["update", "reset"])
def test_runtime_config_is_published_only_after_disk_replacement(tmp_path, monkeypatch, operation):
    path = tmp_path / "fox_config.yaml"
    _write(path, _document(theme="light"))
    manager = ConfigManager(tmp_path)
    original = manager.load()
    original_replace = os.replace
    observations = []

    def observe_replace(source, destination):
        observations.append(manager.config)
        original_replace(source, destination)

    monkeypatch.setattr(config_module.os, "replace", observe_replace)
    if operation == "update":
        result = manager.update(layout="vulpes")
    else:
        result = manager.reset()

    assert observations == [original]
    assert manager.config is result
    assert result is not original
    assert _read(path) == result.model_dump(mode="json")
