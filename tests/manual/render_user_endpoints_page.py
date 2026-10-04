"""Render user_endpoints.html and lift its <script> out for node to check.

Writes the page to $TMPDIR/ue-page.html and the script to $TMPDIR/ue-page.js.
Pass `off` to render with encryption unavailable.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jinja2 import Environment, FileSystemLoader  # noqa: E402

from fox_reader import endpoint_crypto  # noqa: E402
from fox_reader.endpoint_schema import DEFAULT_UUID_VARIANT, UUID_VARIANTS  # noqa: E402
from fox_reader.user_endpoints import (  # noqa: E402
    DEFAULT_MAX_TEXT_LENGTH,
    KNOWN_LANGUAGES,
    MAX_ACTIVE_PER_LANGUAGE,
    MAX_ENDPOINTS_PER_LANGUAGE,
    MAX_MAX_TEXT_LENGTH,
    MAX_TOKEN_MAX_AGE,
    MIN_MAX_TEXT_LENGTH,
    MIN_TOKEN_MAX_AGE,
    NAME_MAX_LENGTH,
)

available = endpoint_crypto.available() and "off" not in sys.argv[1:]

env = Environment(
    loader=FileSystemLoader(str(ROOT / "frontend" / "templates")),
    autoescape=True,
)

page = env.get_template("user_endpoints.html").render(
    known_languages=list(KNOWN_LANGUAGES),
    max_per_language=MAX_ENDPOINTS_PER_LANGUAGE,
    max_active_per_language=MAX_ACTIVE_PER_LANGUAGE,
    name_max_length=NAME_MAX_LENGTH,
    text_length_min=MIN_MAX_TEXT_LENGTH,
    text_length_max=MAX_MAX_TEXT_LENGTH,
    text_length_default=DEFAULT_MAX_TEXT_LENGTH,
    uuid_variants=list(UUID_VARIANTS),
    uuid_default=DEFAULT_UUID_VARIANT,
    key_encodings=list(endpoint_crypto.KEY_ENCODINGS),
    key_length=endpoint_crypto.KEY_TEXT_LENGTH,
    max_keys=endpoint_crypto.MAX_KEYS,
    encryption_available=available,
    token_max_age_min=MIN_TOKEN_MAX_AGE,
    token_max_age_max=MAX_TOKEN_MAX_AGE,
)

blocks = re.findall(r"<script>(.*?)</script>", page, re.S)

if len(blocks) != 1:
    raise SystemExit(f"expected one <script> block, found {len(blocks)}")

out = pathlib.Path(tempfile.gettempdir())
(out / "ue-page.html").write_text(page, encoding="utf-8")
(out / "ue-page.js").write_text(blocks[0], encoding="utf-8")

print(f"page   {len(page)} chars  encryption_available={available}")
print(f"script {len(blocks[0])} chars -> {out / 'ue-page.js'}")
