"""Live loopback test for the custom-endpoint executor.

Spins up a stdlib HTTP server that imitates several real translation APIs,
then drives `translate_by_custom_endpoint` against it.

Run with:  python tests/manual/smoke_custom_endpoint_http.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fox_reader.translate.custom_endpoint import (  # noqa: E402
    translate_by_custom_endpoint,
)
from fox_reader.user_endpoints import UserEndpoint  # noqa: E402

failures: list[str] = []
hits: dict[str, int] = {}
seen: dict[str, dict] = {}


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def equals(label: str, actual, expected) -> None:
    check(
        label,
        actual == expected,
        f"\n        got:      {actual!r}\n        expected: {expected!r}",
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence the default stderr logging
        pass

    def _record(self, path: str, query: str, body: bytes) -> None:
        hits[path] = hits.get(path, 0) + 1
        seen[path] = {
            "query": query,
            "body": body.decode("utf-8"),
            "content_type": self.headers.get("Content-Type", ""),
            "custom_header": self.headers.get("X-Api-Key", ""),
            "method": self.command,
        }

    def _reply(self, code: int, payload, content_type: str = "application/json") -> None:
        raw = (
            payload.encode("utf-8")
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )

        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _handle(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        self._record(parsed.path, parsed.query, body)

        if parsed.path == "/deepl":
            payload = json.loads(body)
            source = payload["params"]["texts"][0]["text"]
            self._reply(200, {
                "result": {
                    "texts": [{
                        "text": f"[{source}]",
                        "alternatives": [{"text": "alt one"}, {"text": "alt two"}],
                    }]
                }
            })

        elif parsed.path == "/google":
            self._reply(200, [
                [["Hello ", "x", None, None], ["world", "y", None, None]],
                None,
                "ja",
            ])

        elif parsed.path == "/error":
            self._reply(400, {"error": {"message": "invalid api key"}})

        elif parsed.path == "/bare-500":
            self._reply(500, "gateway exploded", content_type="text/plain")

        elif parsed.path == "/flaky":
            if hits[parsed.path] < 3:
                self._reply(503, {"error": {"message": "warming up"}})
            else:
                self._reply(200, {"translation": "third time lucky"})

        elif parsed.path == "/plain":
            self._reply(200, "  Hi there  ", content_type="text/plain; charset=utf-8")

        elif parsed.path == "/not-json":
            self._reply(200, "<html>nope</html>", content_type="text/html")

        elif parsed.path == "/missing-node":
            self._reply(200, {"other": "field"})

        elif parsed.path == "/form":
            fields = urllib.parse.parse_qs(body.decode("utf-8"))
            self._reply(200, {"translation": fields.get("q", [""])[0].upper()})

        elif parsed.path == "/slow":
            time.sleep(1.5)
            self._reply(200, {"translation": "too late"})

        else:
            self._reply(404, {"error": {"message": "no such route"}})

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
PORT = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"test server on 127.0.0.1:{PORT}")


def endpoint(path: str, **overrides) -> UserEndpoint:
    payload = {
        "id": path.strip("/") or "root",
        "name": "Test",
        "hostname": f"127.0.0.1{path}",
        "host_scheme": "http",
        "port": PORT,
        "method": "POST",
        "timeout": 5,
        "languages": ["japanese"],
        "request_schema": {
            "type": "dict",
            "children": {"q": {"type": "string", "source": "text"}},
        },
        "response_schema": {
            "type": "dict",
            "children": {"translation": {"type": "string", "source": "text"}},
        },
    }
    payload.update(overrides)
    return UserEndpoint.model_validate(payload)


def run(ep: UserEndpoint, text: str = "こんにちは"):
    return asyncio.run(
        translate_by_custom_endpoint(ep, text=text, source_lang="japanese")
    )


# ---------------------------------------------------------------------------
print("\n[1] DeepL-shaped JSON POST")
# ---------------------------------------------------------------------------

deepl = endpoint(
    "/deepl",
    language_codes={"japanese": "JA"},
    target_language="EN",
    headers={"X-Api-Key": "secret-<SOURCE_LANG>"},
    request_schema={
        "type": "dict",
        "children": {
            "params": {
                "type": "dict",
                "children": {
                    "texts": {
                        "type": "list",
                        "items": [{
                            "type": "dict",
                            "children": {
                                "text": {"type": "string", "source": "text"},
                            },
                        }],
                    },
                    "lang": {
                        "type": "dict",
                        "children": {
                            "source": {"type": "string", "source": "src_lang"},
                            "target": {"type": "string", "source": "target_lang"},
                        },
                    },
                },
            }
        },
    },
    response_schema={
        "type": "dict",
        "children": {
            "error": {
                "type": "dict",
                "children": {"message": {"type": "string", "source": "error"}},
            },
            "result": {
                "type": "dict",
                "children": {
                    "texts": {
                        "type": "list",
                        "items": [{
                            "type": "dict",
                            "children": {
                                "text": {"type": "string", "source": "text"},
                                "alternatives": {"type": "list", "source": "alt"},
                            },
                        }],
                    }
                },
            },
        },
    },
)

result = run(deepl)
equals("code", result.code, 200)
equals("translated", result.data, "[こんにちは]")
equals("alternatives", result.alternatives, ["alt one", "alt two"])
equals("method label", result.method, "Test")
equals("source lang label", result.source_lang, "JAPANESE")
equals("target lang label", result.target_lang, "EN")
equals("header placeholder reached the server", seen["/deepl"]["custom_header"], "secret-JA")
equals("content type", seen["/deepl"]["content_type"], "application/json")
sent = json.loads(seen["/deepl"]["body"])
equals("src lang on the wire", sent["params"]["lang"]["source"], "JA")
equals("target lang on the wire", sent["params"]["lang"]["target"], "EN")

# ---------------------------------------------------------------------------
print("\n[2] Google-shaped GET with repeated query keys")
# ---------------------------------------------------------------------------

google = endpoint(
    "/google",
    method="GET",
    doseq=True,
    query={"client": "gtx"},
    language_codes={"japanese": "ja"},
    request_schema={
        "type": "dict",
        "children": {
            "sl": {"type": "string", "source": "src_lang"},
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
    response_schema={
        "type": "list",
        "items": [{
            "type": "list",
            "each": {
                "type": "list",
                "items": [{"type": "string", "source": "text"}],
            },
        }],
    },
)

result = run(google)
equals("chunks joined", result.data, "Hello world")
equals("no body on GET", seen["/google"]["body"], "")
equals("server saw a GET", seen["/google"]["method"], "GET")
check(
    "repeated dt keys",
    seen["/google"]["query"] == "client=gtx&sl=ja&dt=t&dt=at&q=%E3%81%93%E3%82%93%E3%81%AB%E3%81%A1%E3%81%AF",
    seen["/google"]["query"],
)

# ---------------------------------------------------------------------------
print("\n[3] Error surfaces")
# ---------------------------------------------------------------------------

result = run(endpoint(
    "/error",
    response_schema={
        "type": "dict",
        "children": {
            "translation": {"type": "string", "source": "text"},
            "error": {
                "type": "dict",
                "children": {"message": {"type": "string", "source": "error"}},
            },
        },
    },
))
equals("upstream error status", result.code, 400)
equals("upstream error message", result.message, "invalid api key")
equals("no data on error", result.data, "")

result = run(endpoint("/bare-500"))
equals("bare 500 status", result.code, 500)
check("bare 500 includes the body", "gateway exploded" in result.message, result.message)

result = run(endpoint("/not-json"))
equals("non-json status", result.code, 502)
check("non-json message", "invalid JSON" in result.message, result.message)
check("non-json snippet", "nope" in result.message, result.message)

result = run(endpoint("/missing-node"))
equals("missing node status", result.code, 502)
check(
    "missing node message",
    "Translated Text" in result.message,
    result.message,
)

result = run(endpoint("/nowhere"))
equals("404 route status", result.code, 404)

# ---------------------------------------------------------------------------
print("\n[4] Retries")
# ---------------------------------------------------------------------------

result = run(endpoint("/flaky", retries=2))
equals("retried until success", result.data, "third time lucky")
equals("three attempts", hits["/flaky"], 3)

hits["/flaky"] = 99  # keeps the handler on its 200 branch
result = run(endpoint("/flaky", retries=0))
equals("no retry when disabled", result.data, "third time lucky")

# ---------------------------------------------------------------------------
print("\n[5] Plain-text responses, form bodies, guards")
# ---------------------------------------------------------------------------

result = run(endpoint(
    "/plain",
    response_format="text",
    response_schema={"type": "string", "source": "text"},
))
equals("plain text body is trimmed", result.data, "Hi there")

result = run(endpoint("/form", body_format="form"), text="hello")
equals("form field round-tripped", result.data, "HELLO")
equals(
    "form content type",
    seen["/form"]["content_type"],
    "application/x-www-form-urlencoded",
)

result = run(endpoint("/slow", timeout=0.4))
equals("timeout status", result.code, 504)
check("timeout message", "Timed out" in result.message, result.message)

result = run(endpoint("/deepl"), text="   ")
equals("blank text rejected", result.code, 400)
equals("blank text message", result.message, "Nothing to translate")

result = run(endpoint("/form", body_format="form", max_text_length=100), text="a" * 101)
equals("length guard status", result.code, 413)
check("length guard message", "at most 100" in result.message, result.message)

result = run(endpoint("/form", body_format="form", max_text_length=100), text="b" * 100)
equals("length guard allows the limit itself", result.data, "B" * 100)

# ---------------------------------------------------------------------------
print("\n[6] Other verbs")
# ---------------------------------------------------------------------------

for method in ("PUT", "PATCH", "DELETE"):
    ep = endpoint("/form", method=method, body_format="form")
    result = run(ep, text=f"via {method}")
    equals(f"{method} works", result.data, f"VIA {method}")
    equals(f"{method} reached the server", seen["/form"]["method"], method)

server.shutdown()

print()
if failures:
    print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
    sys.exit(1)

print("all checks passed")
