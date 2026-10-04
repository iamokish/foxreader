"""Render index.html and syntax-check the inline script it now carries.

Two things could break silently in that file. A jinja2 error in the new markup
would only show up as a 500 on the reader page, and a typo in the inline
memory-notice script would only show up in a browser console -- app.js is a
committed build artifact, so nothing in the toolchain looks at this script
otherwise.

Also asserts the notice sits *outside* #nocontextmenu: main.tsx clears that
element when the app mounts, so a notice inside it would be deleted on load.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "frontend" / "templates"

failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures

    if condition:
        print(f"ok   {label}")
        return

    failures += 1
    print(f"FAIL {label}" + (f" -- {detail}" if detail else ""))


env = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=StrictUndefined)
env.globals["url_for"] = lambda name, path="": f"/{name}/{path}"

#: A JS literal: a quoted string, a boolean, null, or a number. Anything else
#: in __FOX_CONFIG__ is a bug -- most likely a flag added without the `| lower`
#: filter, which renders a Python ``True`` that the browser reads as an
#: undeclared identifier and throws on.
JS_LITERAL = re.compile(r'^(?:"[^"\\]*"|true|false|null|-?\d+(?:\.\d+)?)$')

rendered = ""

# The context routes/folder.py passes. bubble_available is the one the markup
# branches on, so it is rendered both ways round.
for bubble in (True, False):
    print(f"--- bubble_available={bubble} ---")

    html = env.get_template("index.html").render(
        initial_theme="dark",
        initial_layout="default",
        deepl_available=False,
        jpdb_available=False,
        bubble_available=bubble,
    )

    check("renders", bool(html))

    # Checked as a whole object rather than one named flag: the previous
    # version asserted on `enableMTL`, which was removed from the template and
    # left this failing unnoticed. This version needs no edit when a flag is
    # added, and still catches the failure that matters.
    config = re.search(r"window\.__FOX_CONFIG__\s*=\s*\{(.*?)\};", html, re.DOTALL)
    check("__FOX_CONFIG__ is present", config is not None)

    if config:
        # Values here are booleans and short identifiers (theme, layout names),
        # none of which contain a comma, so splitting on one is enough.
        pairs = [part.strip() for part in config.group(1).split(",") if part.strip()]
        bad = [pair for pair in pairs if not JS_LITERAL.match(pair.partition(":")[2].strip())]

        check("every config value is a JS literal", not bad, "; ".join(bad))
        check(
            "the bubble flag is a JS boolean",
            f"bubbleAvailable: {str(bubble).lower()}" in html,
            config.group(1).strip(),
        )

    # The button is left out of the markup, not disabled: there is nothing
    # behind it without the weights.
    check(
        "the Bubble Capture button follows the flag",
        ('id="bubCapture"' in html) is bubble,
        f'bubCapture present={("id=\"bubCapture\"" in html)}, flag={bubble}',
    )

    # The failsafe for the renderer this template does not control. The Preact
    # toolbar rebuilds the button row at mount time, so a bundle compiled
    # before that gate existed puts the button back on screen -- which is
    # exactly what happened once. style.css hides #bubCapture under this class,
    # so the server has the last word either way.
    check(
        "<body> carries no-bubble when the model is absent",
        ('class="no-bubble"' in html) is not bubble,
        f'no-bubble present={("class=\"no-bubble\"" in html)}, flag={bubble}',
    )

    notice = html.find('id="mtlMemoryNotice"')
    mount = html.find('id="nocontextmenu"')

    check("the notice is present", notice != -1)
    check("the mount point is present", mount != -1)
    check(
        "the notice comes before the mount point, so the app cannot wipe it",
        -1 < notice < mount,
        f"notice at {notice}, mount at {mount}",
    )
    check("no unrendered jinja is left", "{{" not in html and "{%" not in html)

    if bubble:
        rendered = html

# The flags default rather than raise: routes/folder.py is not the only caller
# the template has to survive (an error page, a future embed), and a missing
# flag must read as "off", not as a 500.
print("\n--- with no context at all ---")

bare = env.get_template("index.html").render(initial_theme="dark", initial_layout="default")

check("renders without the optional flags", bool(bare))
check("and the button defaults to hidden", 'id="bubCapture"' not in bare)
check("and <body> defaults to no-bubble", 'class="no-bubble"' in bare)

print("\n--- the style.css failsafe ---")

style = (ROOT / "frontend" / "static" / "style.css").read_text(encoding="utf-8")

check(
    "style.css hides #bubCapture under body.no-bubble",
    re.search(r"body\.no-bubble\s+#bubCapture\s*\{[^}]*display:\s*none", style) is not None,
)

print("\n--- the inline notice script ---")

scripts = re.findall(r"<script>(.*?)</script>", rendered, re.DOTALL)
notice_scripts = [body for body in scripts if "mtlMemoryNotice" in body]

check("exactly one inline script mentions the notice", len(notice_scripts) == 1, str(len(notice_scripts)))

body = notice_scripts[0]
check("it fetches the endpoint", '"/ml/memory"' in body)
check("it reads the ok flag", "data.ok !== false" in body)
check("it writes text, never markup", "textContent" in body and "innerHTML" not in body)
check("and it swallows failures", ".catch(" in body)

path = pathlib.Path(tempfile.gettempdir()) / "fox-index-notice.js"
path.write_text(body, encoding="utf-8")

node = subprocess.run(
    ["node", "--check", str(path)],
    capture_output=True,
    text=True,
)

check("node --check passes", node.returncode == 0, (node.stderr or node.stdout).strip())

print(f"\n{failures} FAILURE(S)" if failures else "\nindex.html renders and its inline script parses")
sys.exit(1 if failures else 0)
