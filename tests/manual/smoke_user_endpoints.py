"""Offline smoke test for the reworked custom-endpoint backend.

Run with:  python tests/manual/smoke_user_endpoints.py
No network access: only schema building, request preparation and store rules.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fox_reader.endpoint_schema import EndpointSchemaNode  # noqa: E402
from fox_reader.translate.custom_endpoint import (  # noqa: E402
    describe_request,
    prepare_request,
)
from fox_reader.user_endpoints import (  # noqa: E402
    MAX_ACTIVE_PER_LANGUAGE,
    MAX_ENDPOINTS_PER_LANGUAGE,
    UserEndpoint,
    UserEndpointStore,
)

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def equals(label: str, actual, expected) -> None:
    check(label, actual == expected, f"\n        got:      {actual!r}\n        expected: {expected!r}")


# ---------------------------------------------------------------------------
print("\n[1] Google translate_a/single — GET + query + doseq + repeat reader")
# ---------------------------------------------------------------------------

google = UserEndpoint.model_validate({
    "id": "google-clone",
    "name": "Google",
    "hostname": "translate.googleapis.com/translate_a/single",
    "method": "GET",
    "body_format": "query",
    "doseq": True,
    "languages": ["japanese", "korean"],
    "language_codes": {"japanese": "ja", "korean": "ko"},
    "target_language": "en",
    "text_join": "",
    "query": {"client": "gtx"},
    "request_schema": {
        "type": "dict",
        "children": {
            "sl": {"type": "string", "source": "src_lang"},
            "tl": {"type": "string", "source": "target_lang"},
            "dt": {
                "type": "list",
                "items": [
                    {"type": "string", "value": "t"},
                    {"type": "string", "value": "at"},
                ],
            },
            "q": {"type": "string", "source": "text"},
        },
    },
    # result[0] is a list of [chunk, original, ...]; result[5][0][2] holds
    # the alternative renderings.
    "response_schema": {
        "type": "list",
        "items": [
            {
                "type": "list",
                "each": {
                    "type": "list",
                    "items": [{"type": "string", "source": "text"}],
                },
            }
        ],
    },
})

prepared = prepare_request(
    google,
    text="こんにちは 世界",
    source_lang=google.source_code("japanese"),
    target_lang=google.target_language,
)

equals("method", prepared["method"], "GET")
equals("body is not sent for query format", prepared["body"], None)
check(
    "url carries static client=gtx first",
    prepared["url"].startswith(
        "https://translate.googleapis.com/translate_a/single?client=gtx&sl=ja&tl=en&dt=t&dt=at&q="
    ),
    prepared["url"],
)
check("text is percent-encoded", "%E3%81%93%E3%82%93" in prepared["url"], prepared["url"])
equals("source_code maps japanese -> ja", google.source_code("japanese"), "ja")
equals("doseq survives validation for query bodies", google.doseq, True)

google_body = [
    [["Hello ", "こんにちは ", None, None], ["world", "世界", None, None]],
    None,
    "ja",
]
equals(
    "chunked text is joined",
    google.response_schema.extract_text(google_body, join_with=google.text_join),
    "Hello world",
)

# ---------------------------------------------------------------------------
print("\n[2] DeepL-style JSON-RPC POST with alternatives")
# ---------------------------------------------------------------------------

deepl = UserEndpoint.model_validate({
    "id": "deepl-clone",
    "name": "DeepL",
    "hostname": "https://oneshot-free.www.deepl.com/v1/translate",
    "method": "POST",
    "languages": ["japanese"],
    "language_codes": {"japanese": "JA"},
    "target_language": "EN",
    "headers": {"X-Src": "<SOURCE_LANG>"},
    "request_schema": {
        "type": "dict",
        "children": {
            "jsonrpc": {"type": "string", "value": "2.0"},
            "method": {"type": "string", "value": "LMT_handle_texts"},
            "id": {"type": "int", "value": 1},
            "params": {
                "type": "dict",
                "children": {
                    "texts": {
                        "type": "list",
                        "items": [
                            {
                                "type": "dict",
                                "children": {
                                    "text": {"type": "string", "source": "text"}
                                },
                            }
                        ],
                    },
                    "lang": {
                        "type": "dict",
                        "children": {
                            "source_lang_user_selected": {
                                "type": "string",
                                "source": "src_lang",
                            },
                            "target_lang": {
                                "type": "string",
                                "source": "target_lang",
                            },
                        },
                    },
                    "quality": {"type": "float", "value": "0.5"},
                    "html": {"type": "bool", "value": False},
                    "nothing": {"type": "null"},
                },
            },
        },
    },
    "response_schema": {
        "type": "dict",
        "children": {
            "error": {
                "type": "dict",
                "children": {
                    "message": {"type": "string", "source": "error"},
                },
            },
            "result": {
                "type": "dict",
                "children": {
                    "texts": {
                        "type": "list",
                        "items": [
                            {
                                "type": "dict",
                                "children": {
                                    "text": {"type": "string", "source": "text"},
                                    "alternatives": {
                                        "type": "list",
                                        "source": "alt",
                                    },
                                },
                            }
                        ],
                    }
                },
            },
        },
    },
})

prepared = prepare_request(
    deepl,
    text="こんにちは",
    source_lang=deepl.source_code("japanese"),
    target_lang=deepl.target_language,
)

equals("json content type is defaulted", prepared["headers"]["Content-Type"], "application/json")
equals("header placeholder expands", prepared["headers"]["X-Src"], "JA")
equals("pasted scheme is honoured", prepared["url"], "https://oneshot-free.www.deepl.com/v1/translate")

body = json.loads(prepared["body"])
equals("nested text node", body["params"]["texts"][0]["text"], "こんにちは")
equals("src lang node", body["params"]["lang"]["source_lang_user_selected"], "JA")
equals("float static", body["params"]["quality"], 0.5)
equals("bool static", body["params"]["html"], False)
equals("null static", body["params"]["nothing"], None)
check("body is not ascii-escaped", "こんにちは" in prepared["body"], prepared["body"])

deepl_ok = {
    "result": {
        "texts": [
            {
                "text": "Hello",
                "alternatives": [{"text": "Hi"}, {"text": "Good day"}],
            }
        ]
    }
}
equals("text extraction", deepl.response_schema.extract_text(deepl_ok), "Hello")
equals(
    "alternatives extraction",
    deepl.response_schema.extract_alternatives(deepl_ok),
    ["Hi", "Good day"],
)
equals("no error on success", deepl.response_schema.extract_error(deepl_ok), None)
equals(
    "error extraction",
    deepl.response_schema.extract_error({"error": {"message": "quota exceeded"}}),
    "quota exceeded",
)

# ---------------------------------------------------------------------------
print("\n[3] Form body, port handling, text response, placeholders in URL")
# ---------------------------------------------------------------------------

form = UserEndpoint.model_validate({
    "id": "libre",
    "name": "Form",
    "hostname": "127.0.0.1/translate",
    "host_scheme": "http",
    "port": 5000,
    "method": "POST",
    "body_format": "form",
    "response_format": "text",
    "languages": ["chinese"],
    "request_schema": {
        "type": "dict",
        "children": {
            "q": {"type": "string", "source": "text"},
            "source": {"type": "string", "source": "src_lang"},
            "flag": {"type": "bool", "value": True},
        },
    },
    "response_schema": {"type": "string", "source": "text"},
})

prepared = prepare_request(form, text="你好", source_lang="zh", target_lang="en")
equals("port lands before the path", prepared["url"], "http://127.0.0.1:5000/translate")
equals(
    "form content type",
    prepared["headers"]["Content-Type"],
    "application/x-www-form-urlencoded",
)
equals("form body", prepared["body"], "q=%E4%BD%A0%E5%A5%BD&source=zh&flag=true")

url_tpl = UserEndpoint.model_validate({
    "id": "path-api",
    "name": "Path",
    "hostname": "api.example.com/v1/<SOURCE_LANG>/<TARGET_LANG>",
    "method": "GET",
    "languages": ["japanese"],
    "request_schema": {
        "type": "dict",
        "children": {"q": {"type": "string", "source": "text"}},
    },
    "response_schema": {"type": "string", "source": "text"},
})
prepared = prepare_request(url_tpl, text="hi", source_lang="ja", target_lang="en")
equals(
    "placeholders inside the hostname",
    prepared["url"],
    "https://api.example.com/v1/ja/en?q=hi",
)
equals("GET defaults to a query body", url_tpl.wire_format, "query")
equals("describe_request needs no language", describe_request(url_tpl)["method"], "GET")

# ---------------------------------------------------------------------------
print("\n[4] Validation guards")
# ---------------------------------------------------------------------------


def rejects(label: str, payload: dict) -> None:
    try:
        UserEndpoint.model_validate(payload)
    except Exception as exc:
        print(f"  ok    {label} -> {str(exc).splitlines()[-1].strip()[:80]}")
        return
    print(f"  FAIL  {label} was accepted")
    failures.append(label)


base = {
    "id": "x",
    "name": "x",
    "hostname": "example.com",
    "languages": ["japanese"],
    "response_schema": {"type": "string", "source": "text"},
}

rejects(
    "two text nodes in a request",
    base
    | {
        "request_schema": {
            "type": "dict",
            "children": {
                "a": {"type": "string", "source": "text"},
                "b": {"type": "string", "source": "text"},
            },
        }
    },
)
rejects(
    "no text node in a request",
    base | {"request_schema": {"type": "dict", "children": {}}},
)
rejects(
    "repeat template in a request",
    base
    | {
        "request_schema": {
            "type": "dict",
            "children": {
                "q": {"type": "string", "source": "text"},
                "xs": {"type": "list", "each": {"type": "string", "value": "a"}},
            },
        }
    },
)
rejects(
    "nested object in a form body",
    base
    | {
        "method": "POST",
        "body_format": "form",
        "request_schema": {
            "type": "dict",
            "children": {
                "q": {"type": "string", "source": "text"},
                "nested": {"type": "dict", "children": {}},
            },
        },
    },
)
rejects(
    "non-dict root for a query body",
    base
    | {
        "method": "GET",
        "request_schema": {
            "type": "list",
            "items": [{"type": "string", "source": "text"}],
        },
    },
)
rejects("unknown method", base | {"method": "TRACE", "request_schema": {"type": "string", "source": "text"}})
rejects("no languages", {**base, "languages": [], "request_schema": {"type": "string", "source": "text"}})
rejects(
    "two error nodes in a response",
    {
        **base,
        "request_schema": {"type": "string", "source": "text"},
        "response_schema": {
            "type": "dict",
            "children": {
                "t": {"type": "string", "source": "text"},
                "e1": {"type": "string", "source": "error"},
                "e2": {"type": "string", "source": "error"},
            },
        },
    },
)

text_only = UserEndpoint.model_validate(
    base | {"method": "POST", "request_schema": {"type": "string", "source": "text"}}
)
equals("POST defaults to a json body", text_only.wire_format, "json")
equals("doseq is forced off for json", text_only.doseq, False)
equals("unsupported language codes are dropped", text_only.language_codes, {})

# ---------------------------------------------------------------------------
print("\n[5] Store: capacity, multi-active, legacy migration")
# ---------------------------------------------------------------------------

equals("max per language", MAX_ENDPOINTS_PER_LANGUAGE, 20)
equals("max active per language", MAX_ACTIVE_PER_LANGUAGE, 3)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    store = UserEndpointStore(root)

    def payload(index: int, languages: list[str]) -> dict:
        return {
            "id": f"ep-{index}",
            "name": f"ep{index}",
            "hostname": "example.com",
            "languages": languages,
            "request_schema": {
                "type": "dict",
                "children": {"q": {"type": "string", "source": "text"}},
            },
            "response_schema": {"type": "string", "source": "text"},
        }

    for index in range(MAX_ENDPOINTS_PER_LANGUAGE):
        store.create(payload(index, ["japanese"]))

    equals("20 endpoints stored", len(store.all()), 20)

    try:
        store.create(payload(99, ["japanese"]))
        check("21st endpoint rejected", False)
    except ValueError as exc:
        check("21st endpoint rejected", "at most 20" in str(exc), str(exc))

    store.create(payload(100, ["korean"]))
    equals("other languages unaffected", len(store.for_language("korean")), 1)

    for index in range(MAX_ACTIVE_PER_LANGUAGE):
        store.set_active("japanese", f"ep-{index}", True)

    equals("three active", store.active_ids("japanese"), ["ep-0", "ep-1", "ep-2"])

    try:
        store.set_active("japanese", "ep-3", True)
        check("fourth activation rejected", False)
    except ValueError as exc:
        check("fourth activation rejected", "already has 3" in str(exc), str(exc))

    store.set_active("japanese", "ep-1", False)
    equals("deactivation", store.active_ids("japanese"), ["ep-0", "ep-2"])
    store.set_active("japanese", "ep-3", True)
    equals("slot freed", store.active_ids("japanese"), ["ep-0", "ep-2", "ep-3"])

    try:
        store.set_active("korean", "ep-0", True)
        check("activating an unsupported language rejected", False)
    except ValueError as exc:
        check(
            "activating an unsupported language rejected",
            "does not support" in str(exc),
            str(exc),
        )

    equals("activation order preserved", [e.id for e in store.active_for_language("japanese")], ["ep-0", "ep-2", "ep-3"])
    equals("active summary languages", sorted(store.active_summary()), ["japanese"])
    equals("active summary size", len(store.active_summary()["japanese"]), 3)

    # Dropping a language drops that activation only.
    store.update("ep-0", payload(0, ["korean"]))
    equals("activation dropped with the language", store.active_ids("japanese"), ["ep-2", "ep-3"])

    store.delete("ep-2")
    equals("activation dropped on delete", store.active_ids("japanese"), ["ep-3"])

    snapshot = store.as_dict()
    equals("limits exposed", snapshot["limits"]["max_active_per_language"], 3)
    # 20 created for japanese, ep-0 moved to korean, ep-2 deleted.
    check("counts exposed", snapshot["counts"]["japanese"]["total"] == 18, snapshot["counts"]["japanese"])
    check(
        "active_languages per endpoint",
        next(e for e in snapshot["endpoints"] if e["id"] == "ep-3")["active_languages"] == ["japanese"],
    )
    check("url exposed", snapshot["endpoints"][0]["url"] == "https://example.com")

    # Reload from disk keeps state.
    reloaded = UserEndpointStore(root)
    equals("state survives a reload", reloaded.active_ids("japanese"), ["ep-3"])

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "user_endpoints.yaml").write_text(
        """
endpoints:
  - id: legacy
    name: Legacy
    hostname: example.com
    languages: [japanese, korean]
    request_schema:
      type: dict
      children:
        q: {type: string, source: text}
    response_schema: {type: string, source: text}
active_by_language:
  japanese: legacy
  korean: [legacy, missing]
  chinese: legacy
""",
        encoding="utf-8",
    )

    migrated = UserEndpointStore(root)
    equals("legacy string migrated to a list", migrated.active_ids("japanese"), ["legacy"])
    equals("unknown ids pruned", migrated.active_ids("korean"), ["legacy"])
    equals("unsupported language pruned", migrated.active_ids("chinese"), [])
    check(
        "migration rewritten to disk",
        "- legacy" in (root / "user_endpoints.yaml").read_text(encoding="utf-8"),
    )

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "user_endpoints.yaml").write_text("endpoints: {oops: true}\n", encoding="utf-8")
    broken = UserEndpointStore(root)
    equals("malformed config does not block startup", broken.all(), [])

# ---------------------------------------------------------------------------
print("\n[6] Context source: opt-in pairs in the request only")
# ---------------------------------------------------------------------------

from fox_reader.user_endpoints import default_request_schema  # noqa: E402

ctx_base = {
    "id": "ctx",
    "name": "Ctx",
    "hostname": "example.com",
    "languages": ["japanese"],
    "response_schema": {"type": "string", "source": "text"},
}

ctx = UserEndpoint.model_validate(ctx_base | {
    "request_schema": {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "context": {"type": "list", "source": "context"},
        },
    },
})

pairs = [["こんにちは", "Hello"], ["元気?", "How are you?"]]

prepared = prepare_request(
    ctx, text="さようなら", source_lang="ja", target_lang="en", context=pairs
)
equals(
    "context pairs ride in the request body",
    json.loads(prepared["body"])["context"],
    [["こんにちは", "Hello"], ["元気?", "How are you?"]],
)

prepared = prepare_request(ctx, text="さようなら", source_lang="ja", target_lang="en")
equals("no context sent, empty array", json.loads(prepared["body"])["context"], [])

plain = UserEndpoint.model_validate(ctx_base | {
    "request_schema": {
        "type": "dict",
        "children": {"text": {"type": "string", "source": "text"}},
    },
})
prepared = prepare_request(
    plain, text="hi", source_lang="ja", target_lang="en", context=pairs
)
equals(
    "schemas without a context node ignore pairs",
    json.loads(prepared["body"]),
    {"text": "hi"},
)

equals(
    "default schema carries no context",
    default_request_schema().count_source("context"),
    0,
)

equals(
    "context previews as a placeholder",
    ctx.request_schema.example()["context"],
    ["<CONTEXT>"],
)


def rejects_context(label: str, request_schema, response_schema, fragment: str) -> None:
    try:
        UserEndpoint.model_validate(ctx_base | {
            "request_schema": request_schema,
            "response_schema": response_schema,
        })
        check(label, False, "validated but should not have")
    except Exception as exc:
        check(label, fragment in str(exc), str(exc)[:200])


text_response = {"type": "string", "source": "text"}

rejects_context(
    "a second context node is rejected",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "c1": {"type": "list", "source": "context"},
            "c2": {"type": "list", "source": "context"},
        },
    },
    text_response,
    "at most one Context",
)

rejects_context(
    "context needs a list node",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "c": {"type": "string", "source": "context"},
        },
    },
    text_response,
    "must use a list node",
)

rejects_context(
    "context array takes no items",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "c": {
                "type": "list",
                "source": "context",
                "items": [{"type": "string", "value": "x"}],
            },
        },
    },
    text_response,
    "consumes the whole array",
)

rejects_context(
    "context is request-only",
    {"type": "string", "source": "text"},
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "c": {"type": "list", "source": "context"},
        },
    },
    "cannot contain a Context node",
)

# ---------------------------------------------------------------------------
print("\n[7] Encrypted context pairs")
# ---------------------------------------------------------------------------

from fox_reader import endpoint_crypto  # noqa: E402

if not endpoint_crypto.available():
    print("  SKIP  cryptography is not installed, context stays plaintext")
else:
    key = endpoint_crypto.generate_key()

    sealed_ep = UserEndpoint.model_validate(ctx_base | {
        "request_schema": {
            "type": "dict",
            "children": {
                "text": {"type": "string", "source": "text"},
                "context": {"type": "list", "source": "context"},
            },
        },
        "encryption": {"enabled": True, "key": key},
    })

    prepared = prepare_request(
        sealed_ep, text="さようなら", source_lang="ja", target_lang="en",
        context=pairs,
    )
    sent = json.loads(prepared["body"])
    check(
        "pairs keep their shape sealed",
        isinstance(sent["context"], list)
        and len(sent["context"]) == 2
        and all(isinstance(pair, list) and len(pair) == 2 for pair in sent["context"])
        and all(side and side not in ("こんにちは", "Hello", "元気?", "How are you?")
                for pair in sent["context"] for side in pair),
        json.dumps(sent["context"], ensure_ascii=False)[:160],
    )
    opened = [
        [endpoint_crypto.decrypt(side, secret=key, encoding="fernet") for side in pair]
        for pair in sent["context"]
    ]
    equals("sealed pairs open back to the originals", opened, pairs)
    check(
        "selection text is sealed alongside",
        sent["text"] != "さようなら" and endpoint_crypto.looks_encrypted(sent["text"]),
        sent["text"][:60],
    )

    sealed_plain = UserEndpoint.model_validate(ctx_base | {
        "request_schema": {
            "type": "dict",
            "children": {"text": {"type": "string", "source": "text"}},
        },
        "encryption": {"enabled": True, "key": key},
    })
    prepared = prepare_request(
        sealed_plain, text="hi", source_lang="ja", target_lang="en", context=pairs
    )
    check(
        "no context node, nothing extra sealed or sent",
        set(json.loads(prepared["body"])) == {"text"},
        prepared["body"][:160],
    )

# ---------------------------------------------------------------------------
print("\n[8] Character Info + Context Links sources: opt-in lists only")
# ---------------------------------------------------------------------------

chars_ep = UserEndpoint.model_validate(ctx_base | {
    "request_schema": {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "character_info": {"type": "list", "source": "character_info"},
            "context_character_links": {"type": "list", "source": "context_character_links"},
        },
    },
})

roster = [
    {"meta_id": "a1", "name_en": "Aiko", "name_ja": "愛子", "gender": "female", "alias_en": "Sis", "alias_ja": None},
    {"meta_id": "b2", "name_en": None, "name_ja": "先生", "gender": "male", "alias_en": None, "alias_ja": None},
]
links = ["a1", None]

prepared = prepare_request(
    chars_ep, text="さようなら", source_lang="ja", target_lang="en",
    context=pairs, character_info=roster, context_character_links=links,
)
sent = json.loads(prepared["body"])
equals(
    "roster rides whole, every field present",
    sent["character_info"],
    [
        {"meta_id": "a1", "name_en": "Aiko", "name_ja": "愛子", "gender": "female", "alias_en": "Sis", "alias_ja": None},
        {"meta_id": "b2", "name_en": None, "name_ja": "先生", "gender": "male", "alias_en": None, "alias_ja": None},
    ],
)
equals("links ride aligned, nulls kept", sent["context_character_links"], ["a1", None])

prepared = prepare_request(chars_ep, text="さようなら", source_lang="ja", target_lang="en")
sent = json.loads(prepared["body"])
equals("no roster sent, empty array", sent["character_info"], [])
equals("no links sent, empty array", sent["context_character_links"], [])

prepared = prepare_request(
    plain, text="hi", source_lang="ja", target_lang="en",
    context=pairs, character_info=roster, context_character_links=links,
)
equals(
    "schemas without the nodes ignore both lists",
    json.loads(prepared["body"]),
    {"text": "hi"},
)

equals(
    "default schema carries no character info",
    default_request_schema().count_source("character_info"),
    0,
)
equals(
    "default schema carries no links",
    default_request_schema().count_source("context_character_links"),
    0,
)
equals(
    "character info previews as a placeholder",
    chars_ep.request_schema.example()["character_info"],
    ["<CHARACTER_INFO>"],
)
equals(
    "links preview as a placeholder",
    chars_ep.request_schema.example()["context_character_links"],
    ["<CONTEXT_LINKS>"],
)


def rejects_lists(label: str, request_schema, response_schema, fragment: str) -> None:
    try:
        UserEndpoint.model_validate(ctx_base | {
            "request_schema": request_schema,
            "response_schema": response_schema,
        })
        check(label, False, "validated but should not have")
    except Exception as exc:
        check(label, fragment in str(exc), str(exc)[:200])


def list_node(source: str, extra: dict | None = None) -> dict:
    node: dict = {"type": "list", "source": source}
    if extra:
        node.update(extra)
    return node


rejects_lists(
    "a second character info node is rejected",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "c1": list_node("character_info"),
            "c2": list_node("character_info"),
        },
    },
    text_response,
    "at most one Character Info",
)

rejects_lists(
    "a second links node is rejected",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "l1": list_node("context_character_links"),
            "l2": list_node("context_character_links"),
        },
    },
    text_response,
    "at most one Context Links",
)

rejects_lists(
    "character info needs a list node",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "c": {"type": "string", "source": "character_info"},
        },
    },
    text_response,
    "must use a list node",
)

rejects_lists(
    "links need a list node",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "l": {"type": "string", "source": "context_character_links"},
        },
    },
    text_response,
    "must use a list node",
)

rejects_lists(
    "character info array takes no items",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "c": list_node("character_info", {"items": [{"type": "string", "value": "x"}]}),
        },
    },
    text_response,
    "consumes the whole array",
)

rejects_lists(
    "links array takes no items",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "l": list_node("context_character_links", {"items": [{"type": "string", "value": "x"}]}),
        },
    },
    text_response,
    "consumes the whole array",
)

rejects_lists(
    "character info is request-only",
    {"type": "string", "source": "text"},
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "c": list_node("character_info"),
        },
    },
    "cannot contain a Character Info node",
)

rejects_lists(
    "links are request-only",
    {"type": "string", "source": "text"},
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "l": list_node("context_character_links"),
        },
    },
    "cannot contain a Context Links node",
)

rejects_lists(
    "character info without links is rejected",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "roster": list_node("character_info"),
        },
    },
    text_response,
    "go together",
)

rejects_lists(
    "links without character info are rejected",
    {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "speakers": list_node("context_character_links"),
        },
    },
    text_response,
    "go together",
)

# Field names are free-form: the sources can sit under any key.
renamed = UserEndpoint.model_validate(ctx_base | {
    "request_schema": {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "roster": list_node("character_info"),
            "speakers": list_node("context_character_links"),
        },
    },
})
prepared = prepare_request(
    renamed, text="hi", source_lang="ja", target_lang="en",
    context=pairs, character_info=roster, context_character_links=links,
)
sent = json.loads(prepared["body"])
equals("roster under a custom key", [c["name_en"] for c in sent["roster"]], ["Aiko", None])
equals("links under a custom key", sent["speakers"], ["a1", None])
equals(
    "paired nodes preview under custom keys",
    (renamed.request_schema.example()["roster"], renamed.request_schema.example()["speakers"]),
    (["<CHARACTER_INFO>"], ["<CONTEXT_LINKS>"]),
)

from fox_reader.translate.custom_endpoint import _schema_uses_character_lists  # noqa: E402


class _Unaskable:
    def count_source(self, source):
        raise RuntimeError("cannot answer")


equals("gate sees paired nodes", _schema_uses_character_lists(chars_ep.request_schema), True)
equals("gate sees renamed nodes", _schema_uses_character_lists(renamed.request_schema), True)
equals("gate is false without the nodes", _schema_uses_character_lists(plain.request_schema), False)
equals("gate is false by default", _schema_uses_character_lists(default_request_schema()), False)
equals("gate never raises", _schema_uses_character_lists(_Unaskable()), False)

# ---------------------------------------------------------------------------
print("\n[9] Encrypted character names open back, ids stay plaintext")
# ---------------------------------------------------------------------------

if not endpoint_crypto.available():
    print("  SKIP  cryptography is not installed, names stay plaintext")
else:
    key = endpoint_crypto.generate_key()

    sealed_chars = UserEndpoint.model_validate(ctx_base | {
        "request_schema": {
            "type": "dict",
            "children": {
                "text": {"type": "string", "source": "text"},
                "character_info": {"type": "list", "source": "character_info"},
                "context_character_links": {"type": "list", "source": "context_character_links"},
            },
        },
        "encryption": {"enabled": True, "key": key},
    })

    prepared = prepare_request(
        sealed_chars, text="さようなら", source_lang="ja", target_lang="en",
        context=pairs, character_info=roster, context_character_links=links,
    )
    sent = json.loads(prepared["body"])
    sealed_names = [
        sent["character_info"][0]["name_en"],
        sent["character_info"][0]["name_ja"],
        sent["character_info"][0]["alias_en"],
    ]
    check(
        "names sealed, ids and gender plaintext",
        all(name and name not in ("Aiko", "愛子", "Sis") for name in sealed_names)
        and sent["character_info"][0]["meta_id"] == "a1"
        and sent["character_info"][0]["gender"] == "female"
        and sent["character_info"][1]["meta_id"] == "b2"
        and sent["context_character_links"] == ["a1", None],
        json.dumps(sent["character_info"], ensure_ascii=False)[:200],
    )
    opened = {
        field: endpoint_crypto.decrypt(sent["character_info"][0][field], secret=key, encoding="fernet")
        for field in ("name_en", "name_ja", "alias_en")
    }
    equals(
        "sealed names open back to the originals",
        opened,
        {"name_en": "Aiko", "name_ja": "愛子", "alias_en": "Sis"},
    )

# ---------------------------------------------------------------------------
print("\n[10] Preview fills the lists with samples, live path does not")
# ---------------------------------------------------------------------------

from fox_reader.translate.custom_endpoint import (  # noqa: E402
    PREVIEW_CHARACTER_INFO,
    PREVIEW_CONTEXT,
    PREVIEW_CONTEXT_CHARACTER_LINKS,
)

full_ep = UserEndpoint.model_validate(ctx_base | {
    "request_schema": {
        "type": "dict",
        "children": {
            "text": {"type": "string", "source": "text"},
            "context": {"type": "list", "source": "context"},
            "character_info": {"type": "list", "source": "character_info"},
            "context_character_links": {"type": "list", "source": "context_character_links"},
        },
    },
})

previewed = describe_request(full_ep, text="こんにちは", source_lang="japanese")
preview_body = json.loads(previewed["body"])
equals("preview shows sample pairs", preview_body["context"], PREVIEW_CONTEXT)
equals("preview shows a sample roster entry", preview_body["character_info"], PREVIEW_CHARACTER_INFO)
equals("preview shows sample links", preview_body["context_character_links"], PREVIEW_CONTEXT_CHARACTER_LINKS)
check(
    "sample links align to sample pairs",
    len(PREVIEW_CONTEXT_CHARACTER_LINKS) == len(PREVIEW_CONTEXT),
    f"{len(PREVIEW_CONTEXT_CHARACTER_LINKS)} links for {len(PREVIEW_CONTEXT)} pairs",
)

explicit = describe_request(
    full_ep, text="こんにちは", source_lang="japanese",
    context=[["a", "b"]], character_info=roster, context_character_links=["x"],
)
explicit_body = json.loads(explicit["body"])
equals("explicit pairs win over samples", explicit_body["context"], [["a", "b"]])
equals("explicit roster wins over samples", explicit_body["character_info"], [
    {"meta_id": "a1", "name_en": "Aiko", "name_ja": "愛子", "gender": "female", "alias_en": "Sis", "alias_ja": None},
    {"meta_id": "b2", "name_en": None, "name_ja": "先生", "gender": "male", "alias_en": None, "alias_ja": None},
])
equals("explicit links win over samples", explicit_body["context_character_links"], ["x"])

emptied = describe_request(
    full_ep, text="こんにちは", source_lang="japanese",
    context=[], character_info=[], context_character_links=[],
)
emptied_body = json.loads(emptied["body"])
equals("explicit empties stay empty", emptied_body["context"], [])
equals("explicit empty roster stays empty", emptied_body["character_info"], [])
equals("explicit empty links stay empty", emptied_body["context_character_links"], [])

live = prepare_request(full_ep, text="こんにちは", source_lang="ja", target_lang="en")
live_body = json.loads(live["body"])
equals("live path sends no sample pairs", live_body["context"], [])
equals("live path sends no sample roster", live_body["character_info"], [])
equals("live path sends no sample links", live_body["context_character_links"], [])

print()
if failures:
    print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
    sys.exit(1)

print("all checks passed")
