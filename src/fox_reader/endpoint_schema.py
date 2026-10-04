from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonType = Literal[
    "dict",
    "list",
    "string",
    "int",
    "float",
    "bool",
    "null",
    "uuid",
]
ValueSource = Literal[
    "static",
    "text",
    "src_lang",
    "target_lang",
    "context",
    "character_info",
    "context_character_links",
    "error",
    "alt",
]

PRIMITIVE_TYPES: frozenset[str] = frozenset(
    {"string", "int", "float", "bool", "null", "uuid"}
)

#: Request-only list sources. Each consumes its whole array (like Context):
#: the node renders the full list and cannot describe its elements.
LIST_ARRAY_SOURCES: frozenset[str] = frozenset(
    {"context", "character_info", "context_character_links"}
)

#: Human names for the list-array sources, for validation errors.
LIST_SOURCE_LABELS: dict[str, str] = {
    "context": "Context",
    "character_info": "Character Info",
    "context_character_links": "Context Links",
}

# ---------------------------------------------------------------------------
# UUID nodes
# ---------------------------------------------------------------------------

#: Variants offered in the editor. A `uuid` node stores its variant name in
#: `value` and mints a fresh identifier on every request.
UUID_VARIANTS: tuple[str, ...] = ("uuid1", "uuid2", "uuid4")

DEFAULT_UUID_VARIANT = "uuid4"

#: DCE 1.1 domain numbers. Only "person" is offered; it is the one that makes
#: sense for a request identifier.
_DCE_DOMAIN_PERSON = 0


def _local_dce_id() -> int:
    """A stable 32-bit local id for the version-2 layout.

    DCE takes this from the caller's POSIX UID. Windows has no equivalent, so
    the account name is hashed into the same 32-bit space instead: stable for
    a given user, which is all this field is for.
    """
    getuid = getattr(os, "getuid", None)

    if getuid is not None:
        return getuid() & 0xFFFFFFFF

    account = os.environ.get("USERNAME") or os.environ.get("USER") or "fox-reader"

    digest = hashlib.blake2s(
        account.encode("utf-8", "replace"), digest_size=4
    ).digest()

    return int.from_bytes(digest, "big")


def _uuid2() -> uuid.UUID:
    """A DCE Security (version 2) UUID in the "person" domain.

    Python's `uuid` module generates versions 1, 3, 4 and 5 — there is no
    `uuid.uuid2()`, because version 2 belongs to DCE 1.1 rather than to the
    UUID RFC and was never added. It is built here from the same definition
    DCE gives: start from a version-1 UUID, overwrite `time_low` with the
    local id, put the domain number in `clock_seq_low`, and set the version
    nibble to 2.

    That definition costs version 2 its uniqueness, which matters here because
    an API is likely to read this as a request id. `time_low` is the only
    fast-moving part of a version-1 timestamp; once the local id takes its
    place, the surviving `time_mid`/`time_hi` bits tick roughly every seven
    minutes, so two identifiers minted back to back would otherwise come out
    identical. The fields DCE leaves free are therefore drawn fresh each call:
    the clock sequence, and the node — a random node with the multicast bit
    set, which RFC 4122 §4.5 provides for exactly the case of not using a
    hardware address. That also keeps the machine's MAC address off the wire,
    which is the point of the surrounding feature.
    """
    fields = list(
        uuid.uuid1(
            node=secrets.randbits(48) | (1 << 40),
            clock_seq=secrets.randbits(14),
        ).fields
    )

    fields[0] = _local_dce_id()  # time_low      <- local id
    fields[4] = _DCE_DOMAIN_PERSON  # clock_seq_low <- domain

    return uuid.UUID(fields=tuple(fields), version=2)


_UUID_GENERATORS: dict[str, Callable[[], uuid.UUID]] = {
    "uuid1": uuid.uuid1,
    "uuid2": _uuid2,
    "uuid4": uuid.uuid4,
}


def generate_uuid(variant: str = DEFAULT_UUID_VARIANT) -> str:
    """A fresh identifier for `variant`, in the usual hyphenated form."""
    generator = _UUID_GENERATORS.get(variant)

    if generator is None:
        raise ValueError(
            f"Unknown UUID variant: {variant!r}. Expected one of: "
            + ", ".join(UUID_VARIANTS)
            + "."
        )

    return str(generator())

REQUEST_SOURCES: tuple[ValueSource, ...] = (
    "static",
    "text",
    "src_lang",
    "target_lang",
    "context",
    "character_info",
    "context_character_links",
)

RESPONSE_SOURCES: tuple[ValueSource, ...] = (
    "static",
    "text",
    "error",
    "alt",
)

# Placeholders may appear inside any static string, header value, query
# parameter or hostname. They are the same tokens the editor preview shows,
# so what a user sees in the preview is what gets sent.
TEXT_PLACEHOLDER = "<TEXT>"
SOURCE_LANG_PLACEHOLDER = "<SOURCE_LANG>"
TARGET_LANG_PLACEHOLDER = "<TARGET_LANG>"


def apply_placeholders(
    value: str,
    *,
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """Substitute the dynamic placeholder tokens inside a static string."""
    if not value or "<" not in value:
        return value

    return (
        value.replace(TEXT_PLACEHOLDER, text)
        .replace(SOURCE_LANG_PLACEHOLDER, source_lang)
        .replace(TARGET_LANG_PLACEHOLDER, target_lang)
    )


def flatten_strings(value: Any) -> list[str]:
    """Collect every non-empty scalar inside an arbitrary JSON value.

    Upstream APIs are inconsistent about whether a field holds a string, a
    list of sentence chunks, or a list of `[text, metadata...]` pairs. Rather
    than failing on the unexpected shape, flatten it and let the caller join.
    """
    if value is None:
        return []

    if isinstance(value, str):
        return [value] if value else []

    if isinstance(value, bool):
        return [str(value).lower()]

    if isinstance(value, (int, float)):
        return [str(value)]

    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(flatten_strings(item))
        return out

    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(flatten_strings(item))
        return out

    return [str(value)]


class EndpointSchemaNode(BaseModel):
    """A typed JSON node used both to build requests and to read responses.

    Structure:
      * ``dict``      — named children
      * ``list``      — positional ``items`` and/or a single ``each`` template
      * primitive     — a typed static ``value`` or a dynamic ``source``
      * ``uuid``      — a freshly generated identifier; ``value`` names the
        variant (``uuid1`` / ``uuid2`` / ``uuid4``). Request-only.

    ``each`` describes "every element of this array looks like this". It only
    applies when reading a response: it is how a translation split across an
    unknown number of sentence chunks is collected back into one string.

    Dynamic sources:
      * request  — exactly one ``text``, at most one ``src_lang`` /
        ``target_lang`` / ``context`` / ``character_info`` /
        ``context_character_links``
      * response — exactly one ``text``, at most one ``error`` / ``alt``
        (``context``, ``character_info`` and ``context_character_links`` are
        request-only: they travel out, never back)

    Static ``string`` values are placeholder-expanded, so a literal
    ``"q=<TEXT>"`` works anywhere a plain string does.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    type: JsonType
    source: ValueSource = "static"
    value: Any = None

    # `dict` uses children; `list` uses items and/or each.
    children: dict[str, "EndpointSchemaNode"] = Field(default_factory=dict)
    items: list["EndpointSchemaNode"] = Field(default_factory=list)
    each: "EndpointSchemaNode | None" = None

    @model_validator(mode="after")
    def validate_structure(self) -> "EndpointSchemaNode":
        if self.type == "dict":
            if self.items:
                raise ValueError("dict nodes cannot contain list items")
            if self.each is not None:
                raise ValueError("dict nodes cannot contain a repeat template")
            if self.source != "static":
                raise ValueError("dict nodes cannot use a dynamic value source")

        elif self.type == "list":
            if self.children:
                raise ValueError("list nodes cannot contain named children")
            if self.source not in {"static", "alt"} | LIST_ARRAY_SOURCES:
                raise ValueError(
                    "list nodes can only use the Alternatives, Context, "
                    "Character Info or Context Links source"
                )
            if self.source == "alt" and (self.items or self.each is not None):
                raise ValueError(
                    "an Alternatives array consumes the whole array and "
                    "cannot also describe its elements"
                )
            if self.source in LIST_ARRAY_SOURCES and (self.items or self.each is not None):
                label = LIST_SOURCE_LABELS[self.source]
                raise ValueError(
                    f"a {label} array consumes the whole array and "
                    "cannot also describe its elements"
                )

        else:
            if self.children:
                raise ValueError("primitive nodes cannot contain dict children")
            if self.items:
                raise ValueError("primitive nodes cannot contain list items")
            if self.each is not None:
                raise ValueError(
                    "primitive nodes cannot contain a repeat template"
                )

        if self.source == "text" and self.type != "string":
            raise ValueError("Text source must use a string node")

        if self.source in {"src_lang", "target_lang"} and self.type != "string":
            raise ValueError(f"{self.source} source must use a string node")

        if self.source == "error" and self.type not in {"string", "null"}:
            raise ValueError("Error source must use a string or null node")

        if self.source == "alt" and self.type not in {"string", "list"}:
            raise ValueError(
                "Alternatives source must use a string or list node"
            )

        if self.source in LIST_ARRAY_SOURCES and self.type != "list":
            raise ValueError(
                f"{LIST_SOURCE_LABELS[self.source]} source must use a list node"
            )

        if self.type == "uuid" and self.source != "static":
            raise ValueError(
                "a UUID node generates its own value and cannot take a "
                "dynamic source"
            )

        if self.type == "string" and self.source == "static":
            if self.value is None:
                object.__setattr__(self, "value", "")
            elif isinstance(self.value, (dict, list)):
                raise ValueError("string value cannot be an object or list")
            elif not isinstance(self.value, str):
                object.__setattr__(self, "value", str(self.value))

        elif self.type == "int" and self.source == "static":
            if isinstance(self.value, bool):
                raise ValueError("int value cannot be bool")
            if self.value is None or (
                isinstance(self.value, str) and self.value.strip() == ""
            ):
                raise ValueError("int value cannot be empty")
            try:
                object.__setattr__(self, "value", int(self.value))
            except (TypeError, ValueError) as exc:
                raise ValueError("int value must be an integer") from exc

        elif self.type == "float" and self.source == "static":
            if isinstance(self.value, bool):
                raise ValueError("float value cannot be bool")
            if self.value is None or (
                isinstance(self.value, str) and self.value.strip() == ""
            ):
                raise ValueError("float value cannot be empty")
            try:
                object.__setattr__(self, "value", float(self.value))
            except (TypeError, ValueError) as exc:
                raise ValueError("float value must be a number") from exc

        elif self.type == "bool" and self.source == "static":
            if not isinstance(self.value, bool):
                raise ValueError(
                    "bool value must be the boolean true or false"
                )

        elif self.type == "null":
            if self.source == "static":
                object.__setattr__(self, "value", None)

        elif self.type == "uuid":
            # `value` is the variant name, not a literal — the identifier
            # itself is minted per request in `build()`.
            variant = self.value

            if variant is None or (
                isinstance(variant, str) and not variant.strip()
            ):
                variant = DEFAULT_UUID_VARIANT
            elif isinstance(variant, str):
                variant = variant.strip().lower()
            else:
                raise ValueError(
                    "UUID value must be a variant name, one of: "
                    + ", ".join(UUID_VARIANTS)
                )

            if variant not in UUID_VARIANTS:
                raise ValueError(
                    "UUID value must be one of: " + ", ".join(UUID_VARIANTS)
                )

            object.__setattr__(self, "value", variant)

        return self

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        text: str,
        source_lang: str,
        target_lang: str,
        context: list[list[str]] | None = None,
        character_info: list[dict[str, Any]] | None = None,
        context_character_links: list[str | None] | None = None,
    ) -> Any:
        if self.source == "text":
            return text
        if self.source == "src_lang":
            return source_lang
        if self.source == "target_lang":
            return target_lang
        if self.source == "context":
            # The earlier [[source, english], ...] pairs, oldest first. A
            # schema without a Context node never sees them -- callers that
            # accept context (custom endpoints) normalize it before calling.
            return [list(pair) for pair in (context or [])]
        if self.source == "character_info":
            # The roster snapshot: [{meta_id, name_en, name_ja, gender,
            # alias_en, alias_ja}, ...]. Copied so the caller's roster cannot
            # be mutated through the built payload. Schemas without a
            # Character Info node never see it.
            return [dict(item) for item in (character_info or [])]
        if self.source == "context_character_links":
            # One meta_id (or None) per context pair, positionally aligned.
            # Schemas without a Context Links node never see it.
            return list(context_character_links or [])
        if self.source in {"error", "alt"}:
            raise ValueError(
                f"{self.source} is a response-only source and cannot be sent"
            )

        if self.type == "dict":
            return {
                key: child.build(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context=context,
                    character_info=character_info,
                    context_character_links=context_character_links,
                )
                for key, child in self.children.items()
            }

        if self.type == "list":
            # `each` is a response-only reader; requests send `items` only.
            return [
                child.build(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context=context,
                    character_info=character_info,
                    context_character_links=context_character_links,
                )
                for child in self.items
            ]

        if self.type == "uuid":
            # A new identifier every time this is built, which is the point of
            # the type — a request id an API can deduplicate on.
            return generate_uuid(
                self.value
                if isinstance(self.value, str) and self.value
                else DEFAULT_UUID_VARIANT
            )

        if self.type == "string":
            return apply_placeholders(
                "" if self.value is None else str(self.value),
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        if self.type == "int":
            return int(self.value)

        if self.type == "float":
            return float(self.value)

        if self.type == "bool":
            return bool(self.value)

        if self.type == "null":
            return None

        raise ValueError(f"Unsupported JSON type: {self.type}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count_source(self, source: ValueSource) -> int:
        count = int(self.source == source)

        if self.type == "dict":
            count += sum(
                child.count_source(source)
                for child in self.children.values()
            )
        elif self.type == "list":
            count += sum(child.count_source(source) for child in self.items)
            if self.each is not None:
                count += self.each.count_source(source)

        return count

    def has_repeat(self) -> bool:
        if self.type == "dict":
            return any(child.has_repeat() for child in self.children.values())

        if self.type == "list":
            if self.each is not None:
                return True
            return any(child.has_repeat() for child in self.items)

        return False

    def has_uuid(self) -> bool:
        if self.type == "uuid":
            return True

        if self.type == "dict":
            return any(child.has_uuid() for child in self.children.values())

        if self.type == "list":
            if self.each is not None and self.each.has_uuid():
                return True
            return any(child.has_uuid() for child in self.items)

        return False

    def find_first_source(
        self,
        source: ValueSource,
    ) -> "EndpointSchemaNode | None":
        if self.source == source:
            return self

        if self.type == "dict":
            for child in self.children.values():
                found = child.find_first_source(source)
                if found:
                    return found

        elif self.type == "list":
            for child in self.items:
                found = child.find_first_source(source)
                if found:
                    return found
            if self.each is not None:
                return self.each.find_first_source(source)

        return None

    # ------------------------------------------------------------------
    # Response reading
    # ------------------------------------------------------------------

    def collect(self, data: Any, source: ValueSource) -> list[Any]:
        """Every value in `data` sitting at a node marked with `source`.

        A marked node is terminal: the value is taken as-is, whatever its
        actual JSON shape turned out to be.
        """
        if self.source == source:
            return [data]

        out: list[Any] = []

        if self.type == "dict" and isinstance(data, dict):
            for key, child in self.children.items():
                if child.count_source(source) == 0:
                    continue
                if key not in data:
                    continue
                out.extend(child.collect(data[key], source))

        elif self.type == "list" and isinstance(data, list):
            for index, child in enumerate(self.items):
                if child.count_source(source) == 0:
                    continue
                if index >= len(data):
                    continue
                out.extend(child.collect(data[index], source))

            if self.each is not None and self.each.count_source(source):
                for element in data:
                    out.extend(self.each.collect(element, source))

        return out

    def extract_text_parts(self, data: Any) -> list[str]:
        """Every fragment marked Translated Text, before they are joined.

        Split out from `extract_text` so a caller that has to transform each
        fragment — decrypt it, say — can do that before the join.
        """
        chunks: list[str] = []

        for value in self.collect(data, "text"):
            chunks.extend(flatten_strings(value))

        return chunks

    def extract_text(self, data: Any, join_with: str = "") -> str | None:
        """Read the node marked Translated Text out of a decoded response."""
        chunks = self.extract_text_parts(data)

        if not chunks:
            return None

        return join_with.join(chunks)

    def extract_error(self, data: Any) -> str | None:
        for value in self.collect(data, "error"):
            if value is None:
                continue

            parts = flatten_strings(value)
            if parts:
                return " ".join(parts)

        return None

    def extract_alternatives(self, data: Any) -> list[str]:
        out: list[str] = []

        for value in self.collect(data, "alt"):
            for part in flatten_strings(value):
                if part not in out:
                    out.append(part)

        return out

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def example(self) -> Any:
        """A representative JSON preview, without any network activity."""
        if self.source == "text":
            return TEXT_PLACEHOLDER
        if self.source == "src_lang":
            return SOURCE_LANG_PLACEHOLDER
        if self.source == "target_lang":
            return TARGET_LANG_PLACEHOLDER
        if self.source == "error":
            return "<ERROR>"
        if self.source == "alt":
            return ["<ALTERNATIVE>"] if self.type == "list" else "<ALTERNATIVE>"
        if self.source == "context":
            return ["<CONTEXT>"]
        if self.source == "character_info":
            return ["<CHARACTER_INFO>"]
        if self.source == "context_character_links":
            return ["<CONTEXT_LINKS>"]

        if self.type == "dict":
            return {key: child.example() for key, child in self.children.items()}

        if self.type == "list":
            preview = [child.example() for child in self.items]
            if self.each is not None:
                preview.append(self.each.example())
                preview.append("<REPEATS>")
            return preview

        if self.type == "string":
            return "" if self.value is None else str(self.value)
        if self.type == "uuid":
            # Shown as a token, like the other per-request values, because the
            # real identifier is only minted when the request is sent.
            variant = (
                self.value
                if isinstance(self.value, str) and self.value
                else DEFAULT_UUID_VARIANT
            )
            return f"<{variant.upper()}>"
        if self.type == "int":
            return int(self.value)
        if self.type == "float":
            return float(self.value)
        if self.type == "bool":
            return bool(self.value)
        if self.type == "null":
            return None

        raise ValueError(f"Unsupported type: {self.type}")


EndpointSchemaNode.model_rebuild()


def validate_request_schema(schema: EndpointSchemaNode) -> None:
    if schema.count_source("text") != 1:
        raise ValueError("Request schema must contain exactly one Text node.")

    if schema.count_source("src_lang") > 1:
        raise ValueError(
            "Request schema can contain at most one Source Language node."
        )

    if schema.count_source("target_lang") > 1:
        raise ValueError(
            "Request schema can contain at most one Target Language node."
        )

    if schema.count_source("context") > 1:
        raise ValueError(
            "Request schema can contain at most one Context node."
        )

    if schema.count_source("character_info") > 1:
        raise ValueError(
            "Request schema can contain at most one Character Info node."
        )

    if schema.count_source("context_character_links") > 1:
        raise ValueError(
            "Request schema can contain at most one Context Links node."
        )

    # The two lists describe one thing -- a roster and which entry each
    # history turn belongs to -- so one without the other is a misconfigured
    # endpoint, not a partial feature. Node keys are free-form; only the
    # sources have to travel together.
    info_count = schema.count_source("character_info")
    links_count = schema.count_source("context_character_links")

    if (info_count == 0) != (links_count == 0):
        raise ValueError(
            "Character Info and Context Links go together: the request "
            "schema must contain both or neither."
        )

    if schema.count_source("error"):
        raise ValueError("Request schema cannot contain an Error node.")

    if schema.count_source("alt"):
        raise ValueError("Request schema cannot contain an Alternatives node.")

    if schema.has_repeat():
        raise ValueError(
            "Repeat templates only apply when reading a response and cannot "
            "be used in a request schema."
        )


def validate_response_schema(schema: EndpointSchemaNode) -> None:
    if schema.count_source("text") != 1:
        raise ValueError(
            "Response schema must contain exactly one Translated Text node."
        )

    if schema.count_source("error") > 1:
        raise ValueError(
            "Response schema can contain at most one Error node."
        )

    if schema.count_source("alt") > 1:
        raise ValueError(
            "Response schema can contain at most one Alternatives node."
        )

    if schema.count_source("src_lang"):
        raise ValueError(
            "Response schema cannot contain a Source Language node."
        )

    if schema.count_source("target_lang"):
        raise ValueError(
            "Response schema cannot contain a Target Language node."
        )

    if schema.count_source("context"):
        raise ValueError(
            "Response schema cannot contain a Context node."
        )

    if schema.count_source("character_info"):
        raise ValueError(
            "Response schema cannot contain a Character Info node."
        )

    if schema.count_source("context_character_links"):
        raise ValueError(
            "Response schema cannot contain a Context Links node."
        )

    if schema.has_uuid():
        raise ValueError(
            "A UUID node generates a value to send and cannot appear in a "
            "response schema."
        )
