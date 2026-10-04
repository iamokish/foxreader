"""Render settings.html and lift its <script> out for node to check.

Writes the page to $TMPDIR/settings-page.html and the script to
$TMPDIR/settings-page.js. The route passes an empty context, so nothing has to
be supplied here.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]

from jinja2 import Environment, FileSystemLoader  # noqa: E402

env = Environment(
    loader=FileSystemLoader(str(ROOT / "frontend" / "templates")),
    autoescape=True,
)

page = env.get_template("settings.html").render()

blocks = re.findall(r"<script>(.*?)</script>", page, re.S)

if len(blocks) != 1:
    raise SystemExit(f"expected one <script> block, found {len(blocks)}")

out = pathlib.Path(tempfile.gettempdir())
(out / "settings-page.html").write_text(page, encoding="utf-8")
(out / "settings-page.js").write_text(blocks[0], encoding="utf-8")

print(f"page   {len(page)} chars")
print(f"script {len(blocks[0])} chars -> {out / 'settings-page.js'}")

markers = (
    'id="devices"',
    "Compute Devices",
    "device-select",
    "chooseDevice",
    # The thread picker lives in the same section but saves through its own
    # handler, so both halves have to survive a template edit.
    "chooseThreads",
    "threadRow",
    "CPU threads",
    # The section description's own sentence about it. Kept short because the
    # rendered page keeps the template's line breaks, and a longer phrase would
    # match nothing the moment it is re-wrapped.
    "which choose their own",
)

missing = [marker for marker in markers if marker not in page]

for marker in markers:
    print(f"{'ok  ' if marker in page else 'MISSING'} {marker}")

sys.exit(1 if missing else 0)
