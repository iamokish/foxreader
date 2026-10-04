"""Render every server-side template and check the shared page furniture.

These four pages are plain Jinja2 with inline CSS and inline JS -- nothing in
the toolchain type-checks or lints them, and a broken one only shows up as a 500
or a silently unstyled page in a browser. So: render each with the context its
route actually passes, then check the things that are easy to get wrong by hand.

The checks that matter here:
  * the template renders and leaves no jinja behind
  * <div> and <nav> tags balance (a stray close would swallow later markup)
  * every `var(--x)` the page uses is defined in its own :root
  * the "back to reader" crumb points at / on all three secondary pages
  * setup.html and settings.html share settings.html's palette
"""

from __future__ import annotations

import pathlib
import re
import sys

from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "frontend" / "templates"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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

#: The context each route hands its template, verbatim enough to render.
CONTEXTS: dict[str, dict] = {
    "index.html": {
        "initial_theme": "dark",
        "initial_layout": "default",
        "deepl_available": False,
        "jpdb_available": False,
        "bubble_available": True,
    },
    "setup.html": {"mtl_dir": "models"},
    "settings.html": {},
    "user_endpoints.html": {
        "known_languages": ["japanese", "korean", "chinese"],
        "max_per_language": 20,
        "max_active_per_language": 3,
        "name_max_length": 6,
        "text_length_min": 100,
        "text_length_max": 10000,
        "text_length_default": 2000,
        "uuid_variants": ["uuid1", "uuid2", "uuid4"],
        "uuid_default": "uuid4",
        "key_encodings": ["base64", "raw"],
        "key_length": 44,
        "max_keys": 8,
        "encryption_available": True,
        "token_max_age_min": 0,
        "token_max_age_max": 86400,
    },
}

#: The palette settings.html defines, and setup.html now has to agree with.
PALETTE = {
    "--bg": "#111318",
    "--surface": "#1a1d24",
    "--surface-selected": "#222733",
    "--border": "#303541",
    "--accent": "#e94560",
    "--text": "#f1f3f5",
    "--muted": "#9298a4",
    "--success": "#4ecca3",
}

#: Pages that carry the crumbs nav, and where their first link must go.
CRUMBED = ("setup.html", "settings.html", "user_endpoints.html")

rendered: dict[str, str] = {}


def declared_vars(html: str) -> dict[str, str]:
    """Every custom property defined in a :root block, name -> value.

    Split on `;` rather than matching a terminated declaration: the last one in a
    block is allowed to leave its semicolon off, and requiring it here made this
    report a variable as undeclared when it was sitting right there.
    """
    found: dict[str, str] = {}

    for block in re.findall(r":root\s*\{(.*?)\}", html, re.DOTALL):
        for piece in re.sub(r"/\*.*?\*/", "", block, flags=re.DOTALL).split(";"):
            name, sep, value = piece.partition(":")

            if sep and name.strip().startswith("--"):
                found[name.strip()] = value.strip()

    return found


def rule_bodies(html: str, selector: str) -> list[str]:
    """The declarations of every rule whose selector is exactly `selector`.

    The lookbehind is what makes it "exactly": `body {` also appears at the tail
    of `.mtl-memory-notice body {` and of any class ending in the same letters,
    and slicing the stylesheet up by hand to avoid that was how this file
    previously managed to fail on `.name-row`'s alignment.
    """
    pattern = rf"(?<![\w.#:-]){re.escape(selector)}\s*\{{([^}}]*)\}}"

    return re.findall(pattern, html)


def tag_balance(html: str, tag: str) -> int:
    opens = len(re.findall(rf"<{tag}\b", html, re.IGNORECASE))
    closes = len(re.findall(rf"</{tag}\s*>", html, re.IGNORECASE))
    return opens - closes


for name, context in CONTEXTS.items():
    print(f"--- {name} ---")

    try:
        html = env.get_template(name).render(**context)
    except Exception as exc:  # noqa: BLE001
        check("renders", False, f"{type(exc).__name__}: {exc}")
        continue

    rendered[name] = html

    check("renders", bool(html.strip()))
    check("no unrendered jinja is left", "{{" not in html and "{%" not in html)

    for tag in ("div", "nav", "main", "section", "style", "script", "body"):
        # `<style>` and `<script>` bodies can mention a tag name in a string, so
        # only an imbalance in both directions is worth reporting.
        check(f"<{tag}> tags balance", tag_balance(html, tag) == 0, str(tag_balance(html, tag)))

    declared = declared_vars(html)
    used = set(re.findall(r"var\((--[\w-]+)", html))

    check(
        "every var() it uses is declared in its own :root",
        used <= set(declared),
        str(sorted(used - set(declared))),
    )


print("\n--- the back-to-reader crumb ---")

for name in CRUMBED:
    html = rendered.get(name, "")

    crumbs = re.search(r'<nav class="crumbs">(.*?)</nav>', html, re.DOTALL)
    check(f"{name} has a crumbs nav", crumbs is not None)

    if crumbs is None:
        continue

    links = re.findall(r'<a href="([^"]+)"[^>]*>(.*?)</a>', crumbs.group(1), re.DOTALL)
    check(f"{name}: the first crumb goes to /", bool(links) and links[0][0] == "/", str(links))
    check(f"{name}: and it is labelled for the reader", "Reader" in links[0][1], str(links))
    check(
        f"{name}: every other crumb is a real page",
        all(href in {"/", "/settings", "/setup", "/user_endpoints"} for href, _ in links),
        str([href for href, _ in links]),
    )

    check(f"{name} styles .crumbs", ".crumbs {" in html or ".crumbs{" in html)


print("\n--- setup.html shares settings.html's palette ---")

setup = rendered.get("setup.html", "")
settings = rendered.get("settings.html", "")

setup_vars = declared_vars(setup)
settings_vars = declared_vars(settings)

for name, value in PALETTE.items():
    check(f"settings.html {name} is {value}", settings_vars.get(name) == value, str(settings_vars.get(name)))
    check(f"setup.html {name} matches", setup_vars.get(name) == value, str(setup_vars.get(name)))

# The old hardcoded navy theme, which is what "make it look like settings" meant.
for stale in ("#1a1a2e", "#16213e", "#0f3460", "#1a4a7a", "#e0e0e0", "#274a6c"):
    check(f"setup.html no longer mentions {stale}", stale not in setup)

bodies = rule_bodies(setup, "body")

check("setup.html has a body rule to check", len(bodies) == 1, f"{len(bodies)} matched")
check(
    "and its <body> no longer clips a tall card",
    not any("align-items" in body for body in bodies),
    "body still centres on the cross axis, so an overlong card loses its top",
)
check("the card sits in a shell", 'class="setup-shell"' in setup)
check("which is what carries the width", ".setup-shell" in setup and "margin: auto" in setup)

print(f"\n{failures} FAILURE(S)" if failures else "\nevery template renders and the page furniture is consistent")
sys.exit(1 if failures else 0)
