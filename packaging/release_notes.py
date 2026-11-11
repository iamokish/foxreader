"""Combine a tagged changelog entry with the release's download instructions.

Only the matching ``## [version]`` entry is included. If that entry is absent or
empty, the draft contains just the download instructions and can be edited before
publication. GitHub's generated change and contributor sections are never used.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def changelog_entry(changelog: str, tag: str) -> str:
    """Extract one version's body, excluding version headings and link footers."""
    version = tag.removeprefix("v")
    lines = []
    collecting = False
    for line in changelog.splitlines():
        if re.match(r"^#{1,2}\s", line):
            if collecting:
                break
            heading = re.fullmatch(r"##[ \t]+\[([^\]]+)\](?:[ \t]+.*)?", line)
            collecting = bool(heading and heading[1].removeprefix("v") == version)
        elif collecting:
            if re.match(r"^\[[^\]]+\]:\s", line):
                break
            lines.append(line)
    return "\n".join(lines).strip()


def render_release_notes(changelog: str, tag: str, download_notes: str) -> str:
    sections = []
    if entry := changelog_entry(changelog, tag):
        sections.append(f"## What's new\n\n{entry}")
    if download_notes.strip():
        sections.append(download_notes.strip())
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changelog", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--download-notes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    notes = render_release_notes(
        args.changelog.read_text(encoding="utf-8-sig"), args.tag,
        args.download_notes.read_text(encoding="utf-8-sig"),
    )
    args.output.write_text(notes, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
