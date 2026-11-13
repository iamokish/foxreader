"""Bounded, secret-free transfer of one endpoint as an editable YAML draft."""

from __future__ import annotations

import math
from typing import Any

import yaml
from pydantic import ValidationError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node, SequenceNode

from fox_reader.endpoint_schema import EndpointSchemaNode
from fox_reader.user_endpoints import EndpointEncryption, UserEndpoint

MAX_TRANSFER_BYTES = 128 * 1024
# JSON quoting needs room around the YAML, but the request is bounded before
# parsing too. The decoded YAML is independently limited to 128 KiB of UTF-8.
MAX_REQUEST_BYTES = 256 * 1024
MAX_YAML_DEPTH = 32
MAX_YAML_NODES = 8192
_YAML_TAGS = {f"tag:yaml.org,2002:{kind}" for kind in ("map", "seq", "str", "null", "bool", "int", "float")}


class EndpointTransferError(ValueError):
    """User-facing messages must never quote YAML contents or parser snippets."""

    def __init__(self, message: str, suggestion: str, mark: Any = None):
        if mark is not None:
            message += f" (line {mark.line + 1}, column {mark.column + 1})"
        super().__init__(message)
        self.suggestions = [suggestion]


class _TransferLoader(yaml.SafeLoader):
    def __init__(self, stream: str):
        super().__init__(stream)
        self._depth = 0
        self._nodes = 0

    def compose_node(self, parent: Node | None, index: Any) -> Node:
        event = self.peek_event()
        if isinstance(event, AliasEvent):
            raise EndpointTransferError(
                "YAML aliases are not allowed.", "Replace aliases with explicit values.", event.start_mark,
            )
        if self._depth >= MAX_YAML_DEPTH:
            raise EndpointTransferError(
                "YAML is nested too deeply.", f"Use fewer than {MAX_YAML_DEPTH} nested levels.", event.start_mark,
            )
        self._nodes += 1
        if self._nodes > MAX_YAML_NODES:
            raise EndpointTransferError(
                "YAML contains too many nodes.", "Simplify the schemas and remove unused values.", event.start_mark,
            )
        self._depth += 1
        try:
            node = super().compose_node(parent, index)
        finally:
            self._depth -= 1
        if node.tag not in _YAML_TAGS:
            raise EndpointTransferError(
                "Unsupported YAML tag or scalar type.",
                "Remove custom tags; quote dates and other values that should be strings.", node.start_mark,
            )
        return node

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[str, Any]:
        result = {}
        for key_node, value_node in node.value:
            if key_node.tag != "tag:yaml.org,2002:str":
                raise EndpointTransferError(
                    "YAML mapping keys must be strings.", "Quote mapping keys and remove merge keys.",
                    key_node.start_mark,
                )
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise EndpointTransferError(
                    "Duplicate YAML mapping key.", "Keep only one value for the duplicated key.", key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result

    def construct_object(self, node: Node, deep: bool = False) -> Any:
        try:
            value = super().construct_object(node, deep=deep)
        except EndpointTransferError:
            raise
        except (ValueError, OverflowError, KeyError, TypeError):
            raise EndpointTransferError(
                "Invalid YAML value.", "Correct the scalar value or remove its explicit YAML tag.", node.start_mark,
            ) from None
        if isinstance(value, float) and not math.isfinite(value):
            raise EndpointTransferError(
                "Non-finite numbers are not allowed.", "Use a finite number instead of NaN or infinity.",
                node.start_mark,
            )
        return value


def _marks(node: Node, path: tuple = (), result: dict | None = None) -> dict:
    if result is None:
        result = {}
    result[path] = node.start_mark
    if isinstance(node, MappingNode):
        for key, value in node.value:
            _marks(value, (*path, key.value), result)
    elif isinstance(node, SequenceNode):
        for index, value in enumerate(node.value):
            _marks(value, (*path, index), result)
    return result


def _load(document: str) -> tuple[dict, dict]:
    try:
        size = len(document.encode("utf-8"))
    except UnicodeError:
        raise EndpointTransferError("YAML must be valid UTF-8 text.", "Remove invalid Unicode characters.") from None
    if size > MAX_TRANSFER_BYTES:
        raise EndpointTransferError("YAML exceeds 128 KiB.", "Shorten the YAML to at most 128 KiB of UTF-8 text.")
    loader = None
    try:
        loader = _TransferLoader(document)
        node = loader.get_single_node()
        if not isinstance(node, MappingNode):
            raise EndpointTransferError(
                "YAML must contain one endpoint document as a mapping.",
                "Use version: 1 and an endpoint: mapping, not a scalar or list.",
                node.start_mark if node else None,
            )
        data = loader.construct_document(node)
        return data, _marks(node)
    except yaml.YAMLError as exc:
        multiple = getattr(exc, "context", "") == "expected a single document in the stream"
        mark = getattr(exc, "problem_mark", None)
        if mark is None and isinstance(exc, yaml.reader.ReaderError):
            position = exc.position
            mark = yaml.error.Mark(
                "YAML", position, document.count("\n", 0, position),
                position - document.rfind("\n", 0, position) - 1, None, None,
            )
        raise EndpointTransferError(
            "Only one YAML document is allowed." if multiple else "Invalid YAML syntax.",
            "Keep one version: 1 / endpoint: document." if multiple else
            "Check indentation, quoting, and matching brackets near this position.",
            mark,
        ) from None
    except (ValueError, OverflowError, RecursionError, KeyError, TypeError) as exc:
        if isinstance(exc, EndpointTransferError):
            raise
        raise EndpointTransferError(
            "Invalid YAML value.", "Use ordinary strings, finite numbers, booleans, lists and mappings.",
        ) from None
    finally:
        if loader is not None:
            loader.dispose()


def _check_fields(data: Any, allowed: set[str], path: tuple, marks: dict) -> None:
    if not isinstance(data, dict):
        return  # Let the model report the type error.
    for field in data:
        if field not in allowed:
            if path == ("endpoint", "encryption") and field == "key":
                message = "Encryption keys cannot be imported."
                suggestion = "Remove encryption.key from the YAML and enter your own key in the editor before saving."
            else:
                message = "Unknown or non-portable configuration field."
                suggestion = (
                    "Remove the field at this position. IDs, activation state and key-presence flags are not portable. "
                    "Allowed fields here: " + ", ".join(sorted(allowed)) + "."
                )
            raise EndpointTransferError(message, suggestion, marks.get((*path, field)))


def _check_schema_fields(data: Any, path: tuple, marks: dict) -> None:
    _check_fields(data, set(EndpointSchemaNode.model_fields), path, marks)
    if not isinstance(data, dict):
        return
    children = data.get("children")
    if isinstance(children, dict):
        for name, child in children.items():
            _check_schema_fields(child, (*path, "children", name), marks)
    items = data.get("items")
    if isinstance(items, list):
        for index, child in enumerate(items):
            _check_schema_fields(child, (*path, "items", index), marks)
    if data.get("each") is not None:
        _check_schema_fields(data["each"], (*path, "each"), marks)


def _validation_error(exc: ValidationError, marks: dict) -> EndpointTransferError:
    # No input values, source excerpts, or arbitrary schema-child names are
    # included. Line/column pinpoints the editable field without echoing it.
    error = exc.errors(include_input=False, include_url=False, include_context=False)[0]
    path = ("endpoint", *error["loc"])
    message = error["msg"].removeprefix("Value error, ")
    if "request cannot nest an object under" in message:
        message = "A form or query request cannot nest an object. Use a JSON body instead."
    if "hostname" in error["loc"]:
        suggestion = "Use a hostname/path without http://, https:// or //, and set host_scheme to http or https."
    elif any(name in error["loc"] for name in ("request_schema", "response_schema")) or "schema" in message.lower():
        suggestion = (
            "Edit the request/response schema in the YAML and retry. Each needs exactly one string Text source; "
            "request-only sources belong in the request schema."
        )
    else:
        suggestion = "Correct this endpoint setting in the YAML and retry."
    while path not in marks and path:
        path = path[:-1]
    field = error["loc"][0] if error["loc"] else "endpoint"
    return EndpointTransferError(f"{field}: {message}", suggestion, marks.get(path))


def _check_finite(value: Any, path: tuple = (), marks: dict | None = None) -> None:
    # Some existing schema validators convert string values to floats. Check
    # after validation too, before JSON serialization could turn inf into null.
    if isinstance(value, float) and not math.isfinite(value):
        raise EndpointTransferError(
            "Non-finite numbers are not allowed.", "Use a finite number instead of NaN or infinity.",
            (marks or {}).get(path),
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, (*path, key), marks)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, (*path, index), marks)


def import_endpoint(document: str) -> dict[str, Any]:
    """Validate without a store, a secret, a generated ID or any HTTP calls."""
    data, marks = _load(document)
    _check_fields(data, {"version", "endpoint"}, (), marks)
    if type(data.get("version")) is not int or data["version"] != 1:
        raise EndpointTransferError(
            "Unsupported or missing endpoint YAML version.", "Set version: 1 at the top level.",
            marks.get(("version",), marks[()]),
        )
    settings = data.get("endpoint")
    if not isinstance(settings, dict):
        raise EndpointTransferError(
            "endpoint must be a mapping for exactly one endpoint.", "Put one endpoint's settings under endpoint:.",
            marks.get(("endpoint",), marks[()]),
        )
    _check_fields(settings, set(UserEndpoint.model_fields) - {"id"}, ("endpoint",), marks)
    _check_fields(
        settings.get("encryption"), set(EndpointEncryption.model_fields) - {"key"}, ("endpoint", "encryption"), marks,
    )
    for name in ("request_schema", "response_schema"):
        _check_schema_fields(settings.get(name), ("endpoint", name), marks)
    try:
        endpoint = UserEndpoint.model_validate(
            {**settings, "id": "import-draft"}, context={"allow_missing_encryption_key": True},
        )
    except ValidationError as exc:
        raise _validation_error(exc, marks) from None
    except (ValueError, TypeError, OverflowError, RecursionError):
        # Numeric conversions in older schema validators can raise directly
        # instead of producing a Pydantic error. Keep these user-facing too.
        raise EndpointTransferError(
            "Invalid endpoint value.", "Check scalar types and use numbers within the supported numeric range.",
            marks[("endpoint",)],
        ) from None
    _check_finite(endpoint.model_dump(exclude={"id": True, "encryption": {"key"}}), ("endpoint",), marks)
    requires_key = endpoint.encryption.enabled
    return {
        "endpoint": endpoint.model_dump(mode="json", exclude={"id": True, "encryption": {"key"}}),
        "requires_key": requires_key,
        "warnings": ["Enter your own encryption key before saving this endpoint."] if requires_key else [],
    }


def export_endpoint(endpoint: UserEndpoint) -> str:
    _check_finite(endpoint.model_dump(exclude={"id": True, "encryption": {"key"}}))
    settings = endpoint.model_dump(mode="json", exclude={"id": True, "encryption": {"key"}})
    # Only schemas omit defaults: this keeps every endpoint setting visible
    # while avoiding six mostly empty fields for every schema node.
    for name in ("request_schema", "response_schema"):
        settings[name] = getattr(endpoint, name).model_dump(mode="json", exclude_defaults=True)
    document = yaml.safe_dump({"version": 1, "endpoint": settings}, sort_keys=False, allow_unicode=True)
    # Every successful export must fit the same limits as import, including
    # structural bounds. A large stored endpoint can be simplified first.
    _load(document)
    return document
