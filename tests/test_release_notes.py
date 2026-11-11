"""Draft release notes use only the selected version's curated changelog."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "packaging" / "release_notes.py"
spec = importlib.util.spec_from_file_location("fox_release_notes_tests", SOURCE)
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

## [Unreleased]

- A future change.

## [0.3.01]

- A similarly named version.

## [0.3.0-rc.1]

- A prerelease change.

## [0.3.0] - 2026-11-10

### Added

- Reader windows.

### Changed

- Improved setup.

### Fixed

- Saved keys.

## [0.2.2]

- An older change.

[0.3.0]: https://example.test/compare/0.2.2...0.3.0
[0.2.2]: https://example.test/compare/0.2.1...0.2.2
"""
DOWNLOADS = "## What to download\n\nPick your platform.\n"


@pytest.mark.parametrize("tag", ["0.3.0", "v0.3.0"])
def test_selects_exact_version_and_preserves_subsections(tag):
    assert release_notes.changelog_entry(CHANGELOG, tag) == (
        "### Added\n\n- Reader windows.\n\n### Changed\n\n- Improved setup."
        "\n\n### Fixed\n\n- Saved keys."
    )


def test_prerelease_is_distinct_from_stable():
    assert release_notes.changelog_entry(CHANGELOG, "v0.3.0-rc.1") == "- A prerelease change."


def test_last_entry_excludes_changelog_link_footer():
    assert release_notes.changelog_entry(CHANGELOG, "0.2.2") == "- An older change."


@pytest.mark.parametrize("changelog,tag", [
    (CHANGELOG, "0.4.0"),
    ("# Changelog\n\n## [0.3.0]\n\n## [0.2.2]\n\n- Old change.\n", "0.3.0"),
])
def test_missing_or_empty_entry_keeps_download_instructions(changelog, tag):
    assert release_notes.render_release_notes(changelog, tag, DOWNLOADS) == DOWNLOADS


def test_repository_release_notes_include_curated_changes_only():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    notes = release_notes.render_release_notes(changelog, "v0.3.0", DOWNLOADS)
    assert notes.startswith("## What's new\n\n### Added\n")
    assert "Qwen3.5 4B and 9B" in notes
    assert "### Changed" in notes and "### Fixed" in notes
    assert "CUDA 13.2 backend support" not in notes
    assert "## [0.2.2]" not in notes
    assert "[0.3.0]:" not in notes
    assert all(heading not in notes for heading in ("What's Changed", "New Contributors", "## Contributors"))
    assert notes.endswith("\n\n" + DOWNLOADS)


def test_cli_combines_utf8_changelog_and_download_instructions(tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    downloads = tmp_path / "downloads.md"
    output = tmp_path / "release.md"
    changelog.write_text("## [v1.0.0]\n\n### Fixed\n\n- 日本語 — reader.\n", encoding="utf-8-sig")
    downloads.write_text("## What to download\n\nJoin .001, .002, …\n", encoding="utf-8")
    result = subprocess.run([
        sys.executable, str(SOURCE), "--changelog", str(changelog), "--tag", "1.0.0",
        "--download-notes", str(downloads), "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = "## What's new\n\n### Fixed\n\n- 日本語 — reader.\n\n## What to download\n\nJoin .001, .002, …\n"
    assert output.read_bytes() == expected.encode("utf-8")
