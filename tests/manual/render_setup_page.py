"""Render setup.html and lift its main <script> out for node to check.

Writes the page to $TMPDIR/setup-page.html and the script to
$TMPDIR/setup-page.js. The page carries two inline scripts -- a short one in
``<head>`` that decides the copyright notice's first paint before anything is
drawn, and the wizard itself -- so this picks the wizard by what it contains
rather than by position.

The markers below are the contract ``smoke_setup_terms.mjs`` relies on: the
licence card's element ids, the optional-models section, and the two routes a
download goes through. A template edit that renames one of them fails here,
where the reason is obvious, instead of in the node harness as a stub element
that silently answers nothing.
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

page = env.get_template("setup.html").render(mtl_dir="models")

blocks = re.findall(r"<script>(.*?)</script>", page, re.S)
wizard = [body for body in blocks if "function initialize()" in body]

if len(wizard) != 1:
    raise SystemExit(
        f"expected exactly one <script> with the wizard in it, found {len(wizard)} "
        f"of {len(blocks)} blocks"
    )

out = pathlib.Path(tempfile.gettempdir())
(out / "setup-page.html").write_text(page, encoding="utf-8")
(out / "setup-page.js").write_text(wizard[0], encoding="utf-8")

print(f"page   {len(page)} chars")
print(f"script {len(wizard[0])} chars -> {out / 'setup-page.js'}")

markers = (
    # The licence card, its scrolling pane, and the controls that gate on it.
    'id="termsGate"',
    'id="termsBody"',
    'id="termsMore"',
    'id="termsAcceptChk"',
    'id="termsAcceptBtn"',
    'id="termsDeclineBtn"',
    "fox-terms-open",
    # Optional models: the section, the list, and the one button that starts it.
    'id="optionalSection"',
    'id="optionalModels"',
    'id="optionalBtn"',
    "Optional Models",
    "downloadOptional()",
    # The licence link on a model card, and the function behind it.
    "terms-peek",
    "reviewTerms(",
    # The routes. A download that does not pass through the first of these is
    # refused by the backend, so both have to be here.
    "/api/setup/terms/accept",
    "/api/setup/optional/download",
    # Bubble and Text Seg are optional now; nothing may call them required.
    "works without all of them",
)

missing = [marker for marker in markers if marker not in page]

for marker in markers:
    print(f"{'ok  ' if marker in page else 'MISSING'} {marker}")

sys.exit(1 if missing else 0)
