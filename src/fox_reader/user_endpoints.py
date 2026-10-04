from __future__ import annotations

import copy
import logging
import re
import threading
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from fox_reader import endpoint_crypto
from fox_reader.endpoint_schema import (
    UUID_VARIANTS,
    EndpointSchemaNode,
    apply_placeholders,
    validate_request_schema,
    validate_response_schema,
)

logger = logging.getLogger(__name__)

# A language can hold a large library of endpoints, but only a few of them
# become translator buttons at once.
MAX_ENDPOINTS_PER_LANGUAGE = 20
MAX_ACTIVE_PER_LANGUAGE = 3

# An endpoint name is rendered as a translator button label alongside the
# built-in engines, so it is kept to their width.
NAME_MAX_LENGTH = 6

# Bounds for the per-endpoint text guard. Anything under a hundred characters
# is too small to be a useful selection, and ten thousand is past the point
# where any of these APIs answers in one request.
MIN_MAX_TEXT_LENGTH = 100
MAX_MAX_TEXT_LENGTH = 10000
DEFAULT_MAX_TEXT_LENGTH = 2000

# Bounds for the optional replay guard on an encrypted response. Fernet
# timestamps are whole seconds and the two clocks are never exactly aligned, so
# anything under half a minute would reject good responses.
MIN_TOKEN_MAX_AGE = 30
MAX_TOKEN_MAX_AGE = 86400

# The languages Fox Reader itself can OCR. Endpoints are not restricted to
# these — the list only drives the editor's checkboxes.
KNOWN_LANGUAGES: tuple[str, ...] = ("japanese", "korean", "chinese")

HTTP_METHODS: frozenset[str] = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE"}
)

# How the built request payload is put on the wire.
#   json  — JSON body
#   form  — application/x-www-form-urlencoded body
#   query — appended to the URL query string (no body)
#   text  — raw body, serialized without a JSON wrapper
BODY_FORMATS: frozenset[str] = frozenset({"json", "form", "query", "text"})

# json — decode and read through the response schema
# text — the whole response body *is* the translation
RESPONSE_FORMATS: frozenset[str] = frozenset({"json", "text"})

# Methods that carry their payload in the query string unless told otherwise.
_QUERY_DEFAULT_METHODS: frozenset[str] = frozenset({"GET", "DELETE"})

# Formats that need a flat-ish object at the root, because there is no way to
# express nesting in a urlencoded key/value pair.
_FLAT_BODY_FORMATS: frozenset[str] = frozenset({"form", "query"})


def default_request_schema() -> EndpointSchemaNode:
    return EndpointSchemaNode.model_validate(
        {
            "type": "dict",
            "children": {
                "text": {
                    "type": "string",
                    "source": "text",
                },
                "src_lang": {
                    "type": "string",
                    "source": "src_lang",
                },
                "target_lang": {
                    "type": "string",
                    "source": "target_lang",
                },
            },
        }
    )


def default_response_schema() -> EndpointSchemaNode:
    return EndpointSchemaNode.model_validate(
        {
            "type": "dict",
            "children": {
                "text": {
                    "type": "string",
                    "source": "text",
                },
                "error": {
                    "type": "string",
                    "source": "error",
                },
            },
        }
    )


class EndpointEncryption(BaseModel):
    """Optional Fernet encryption of the text going to and from an endpoint.

    Off by default. When on, the selection is sealed before it goes on the wire
    and the translation is opened when it comes back, so a proxy, a middlebox
    or an access log sees only opaque tokens. The endpoint holds the same key
    and does the reverse.

    The two directions are separate settings because an endpoint may well
    support only one of them — read a sealed request but answer in plain JSON.

    `key` is write-only. It is persisted so requests can be encrypted, and it
    is stripped from every response the browser can see; see
    `UserEndpoint.public_dict`.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        # A validation failure here would otherwise echo the offending input
        # back in `str(exc)` — which is the key. That string reaches the log
        # file when a stored endpoint fails to load.
        hide_input_in_errors=True,
    )

    enabled: bool = False

    key: str = ""
    key_encoding: str = "fernet"

    encrypt_request: bool = True
    decrypt_response: bool = True

    # Replay guard, in seconds. Off by default: it only works when the two
    # machines' clocks agree.
    max_age: int | None = None

    @field_validator("key", mode="before")
    @classmethod
    def normalize_key(cls, value: Any) -> str:
        if value is None:
            return ""

        if not isinstance(value, str):
            raise ValueError("encryption key must be a string")

        return value.strip()

    @field_validator("key_encoding", mode="before")
    @classmethod
    def normalize_key_encoding(cls, value: Any) -> str:
        value = str(value or "fernet").strip().lower()

        if value not in endpoint_crypto.KEY_ENCODINGS:
            raise ValueError(
                "key_encoding must be one of: "
                + ", ".join(endpoint_crypto.KEY_ENCODINGS)
            )

        return value

    @field_validator("max_age", mode="before")
    @classmethod
    def normalize_max_age(cls, value: Any) -> int | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None

        if isinstance(value, bool):
            raise ValueError("max_age must be a number of seconds")

        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_age must be a whole number of seconds"
            ) from exc

        # Zero or negative reads as "no expiry", which is the default.
        if value <= 0:
            return None

        if not MIN_TOKEN_MAX_AGE <= value <= MAX_TOKEN_MAX_AGE:
            raise ValueError(
                f"max_age must be between {MIN_TOKEN_MAX_AGE} and "
                f"{MAX_TOKEN_MAX_AGE} seconds, or empty to accept any age"
            )

        return value

    @model_validator(mode="after")
    def validate_encryption(self) -> "EndpointEncryption":
        if not self.enabled:
            # A disabled endpoint has no business keeping a secret on disk.
            object.__setattr__(self, "key", "")
            return self

        if not endpoint_crypto.available():
            raise ValueError(
                "Encryption needs the 'cryptography' package, which is not "
                "installed in this environment. Install it, or leave "
                "encryption off for this endpoint."
            )

        if not self.key:
            raise ValueError(
                "An encryption key is required when encryption is on."
            )

        if not (self.encrypt_request or self.decrypt_response):
            raise ValueError(
                "Choose at least one direction to encrypt: the request, the "
                "response, or both."
            )

        try:
            endpoint_crypto.validate_secret(self.key, self.key_encoding)
        except endpoint_crypto.EncryptionError as exc:
            raise ValueError(str(exc)) from exc

        return self


class UserEndpoint(BaseModel):
    # Inputs are hidden from error text for the same reason as on
    # EndpointEncryption: a failure in this model's own validator would echo
    # the whole payload, nested encryption key and all.
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    id: str

    # The name is the translator button label, sitting next to others,
    # so it has to stay short enough not to reflow that row.
    name: str = Field(max_length=NAME_MAX_LENGTH)

    hostname: str

    port: int | None = Field(default=None, ge=1, le=65535)
    method: str = "POST"
    http: int | None = None
    host_scheme: str = "https"
    doseq: bool = False
    timeout: float = Field(default=20.0, gt=0)

    # Transport shape. `body_format` is filled in from `method` when omitted.
    body_format: str | None = None
    response_format: str = "json"

    # Static query parameters, merged with (and overridden by) a `query` body.
    query: dict[str, str] = Field(default_factory=dict)

    languages: list[str] = Field(default_factory=list)

    # Fox language name -> the code this API expects, e.g. japanese -> ja.
    language_codes: dict[str, str] = Field(default_factory=dict)
    target_language: str = "en"

    # Separator used when a response yields several text fragments, which is
    # what a repeat template over sentence chunks produces.
    text_join: str = ""

    retries: int = Field(default=0, ge=0, le=5)

    # Selections longer than this are refused before a request goes out, which
    # keeps a stray whole-page selection from being billed or rate-limited.
    max_text_length: int = Field(
        default=DEFAULT_MAX_TEXT_LENGTH,
        ge=MIN_MAX_TEXT_LENGTH,
        le=MAX_MAX_TEXT_LENGTH,
    )

    request_schema: EndpointSchemaNode = Field(
        default_factory=default_request_schema
    )
    response_schema: EndpointSchemaNode = Field(
        default_factory=default_response_schema
    )

    headers: dict[str, str] = Field(default_factory=dict)

    encryption: EndpointEncryption = Field(
        default_factory=EndpointEncryption
    )

    @field_validator("id", "name", "hostname", mode="before")
    @classmethod
    def normalize_required_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("value must be a string")

        value = value.strip()

        if not value:
            raise ValueError("value cannot be empty")

        return value

    @field_validator("max_text_length", mode="before")
    @classmethod
    def normalize_max_text_length(cls, value: Any) -> Any:
        # Earlier builds stored this as an optional "no limit", and a stale
        # editor page can still post null. Treat a missing value as the
        # default rather than failing the whole endpoint; an out-of-range
        # number is still rejected loudly.
        if value is None or (isinstance(value, str) and not value.strip()):
            return DEFAULT_MAX_TEXT_LENGTH

        return value

    @field_validator("method", mode="before")
    @classmethod
    def normalize_method(cls, value: Any) -> str:
        value = str(value or "POST").strip().upper()

        if value not in HTTP_METHODS:
            raise ValueError(
                "method must be one of: " + ", ".join(sorted(HTTP_METHODS))
            )

        return value

    @field_validator("body_format", mode="before")
    @classmethod
    def normalize_body_format(cls, value: Any) -> str | None:
        if value is None or str(value).strip() == "":
            return None

        value = str(value).strip().lower()

        if value not in BODY_FORMATS:
            raise ValueError(
                "body_format must be one of: " + ", ".join(sorted(BODY_FORMATS))
            )

        return value

    @field_validator("response_format", mode="before")
    @classmethod
    def normalize_response_format(cls, value: Any) -> str:
        value = str(value or "json").strip().lower()

        if value not in RESPONSE_FORMATS:
            raise ValueError(
                "response_format must be one of: "
                + ", ".join(sorted(RESPONSE_FORMATS))
            )

        return value

    @field_validator("host_scheme", mode="before")
    @classmethod
    def normalize_scheme(cls, value: Any) -> str:
        value = str(value or "https").strip().lower()

        if value not in {"http", "https"}:
            raise ValueError("host_scheme must be http or https")

        return value

    @field_validator("languages", mode="before")
    @classmethod
    def normalize_languages(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]

        if not isinstance(value, (list, tuple)):
            raise ValueError("languages must be a list")

        result: list[str] = []

        for language in value:
            language = str(language).strip().lower()

            if language and language not in result:
                result.append(language)

        if not result:
            raise ValueError("endpoint must support at least one language")

        return result

    @field_validator("headers", "query", "language_codes", mode="before")
    @classmethod
    def normalize_string_map(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise ValueError("value must be an object")

        return {
            str(key).strip(): str(item)
            for key, item in value.items()
            if str(key).strip()
        }

    @field_validator("target_language", mode="before")
    @classmethod
    def normalize_target_language(cls, value: Any) -> str:
        value = str(value or "en").strip()
        return value or "en"

    @model_validator(mode="after")
    def validate_schema_config(self) -> "UserEndpoint":
        validate_request_schema(self.request_schema)
        validate_response_schema(self.response_schema)

        if self.body_format is None:
            object.__setattr__(
                self,
                "body_format",
                "query" if self.method in _QUERY_DEFAULT_METHODS else "json",
            )

        if self.body_format in _FLAT_BODY_FORMATS:
            if self.request_schema.type != "dict":
                raise ValueError(
                    f"A {self.body_format} request needs a dict at the root of "
                    "the request schema."
                )

            for key, child in self.request_schema.children.items():
                if child.type == "dict":
                    raise ValueError(
                        f"A {self.body_format} request cannot nest an object "
                        f"under '{key}'. Use a JSON body instead."
                    )

        # doseq only changes how repeated keys are encoded.
        if self.body_format not in _FLAT_BODY_FORMATS:
            object.__setattr__(self, "doseq", False)

        # Language codes are keyed by the languages the endpoint declares.
        object.__setattr__(
            self,
            "language_codes",
            {
                language: code
                for language, code in self.language_codes.items()
                if language.strip().lower() in self.languages
            },
        )

        return self

    @property
    def http2(self) -> bool:
        return self.http == 2

    @property
    def encrypts_request(self) -> bool:
        return self.encryption.enabled and self.encryption.encrypt_request

    @property
    def decrypts_response(self) -> bool:
        return self.encryption.enabled and self.encryption.decrypt_response

    def public_dict(self) -> dict[str, Any]:
        """`model_dump`, minus the encryption key.

        Every browser-facing surface goes through this. The key is write-only:
        it is stored and used, and never sent back out — only whether one is
        set, so the editor can say so without revealing it.
        """
        data = self.model_dump(mode="json")

        block = data.get("encryption")
        block = dict(block) if isinstance(block, dict) else {}

        block.pop("key", None)
        block["has_key"] = bool(self.encryption.key)

        data["encryption"] = block

        return data

    @property
    def wire_format(self) -> str:
        """The resolved body format, never None after validation."""
        return self.body_format or (
            "query" if self.method in _QUERY_DEFAULT_METHODS else "json"
        )

    def source_code(self, language: str) -> str:
        """The code this API expects for one of Fox Reader's languages."""
        language = (language or "").strip().lower()
        return self.language_codes.get(language) or language

    def resolve_headers(
        self,
        *,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> dict[str, str]:
        return {
            key: apply_placeholders(
                value,
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            for key, value in self.headers.items()
        }

    def resolve_query(
        self,
        *,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> dict[str, str]:
        return {
            key: apply_placeholders(
                value,
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            for key, value in self.query.items()
        }

    def build_url(
        self,
        *,
        text: str = "",
        source_lang: str = "",
        target_lang: str = "",
    ) -> str:
        raw = apply_placeholders(
            self.hostname.strip(),
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        ).rstrip("/")

        scheme = self.host_scheme

        if "://" in raw:
            pasted_scheme, raw = raw.split("://", 1)

            if pasted_scheme.lower() in {"http", "https"}:
                scheme = pasted_scheme.lower()

        if self.port is None or re.search(r":\d+(?:/|$)", raw):
            return f"{scheme}://{raw}"

        # The port belongs to the authority, before any path.
        if "/" in raw:
            authority, path = raw.split("/", 1)
            return f"{scheme}://{authority}:{self.port}/{path}"

        return f"{scheme}://{raw}:{self.port}"


class UserEndpointStore:
    def __init__(self, config_root: Path):
        self.config_root = Path(config_root)
        self.path = self.config_root / "user_endpoints.yaml"

        self._lock = threading.RLock()
        self._endpoints: list[UserEndpoint] = []
        self._active_by_language: dict[str, list[str]] = {}

        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_active(raw: Any) -> dict[str, list[str]]:
        """Read `active_by_language`, accepting the legacy single-id format."""
        if not isinstance(raw, dict):
            raise ValueError("active_by_language must be an object")

        result: dict[str, list[str]] = {}

        for language, value in raw.items():
            language = str(language).strip().lower()

            if not language:
                continue

            if value is None:
                continue

            if isinstance(value, str):
                candidates = [value]
            elif isinstance(value, (list, tuple)):
                candidates = list(value)
            else:
                raise ValueError(
                    "active_by_language values must be an id or a list of ids"
                )

            ids: list[str] = []

            for candidate in candidates:
                candidate = str(candidate).strip()

                if candidate and candidate not in ids:
                    ids.append(candidate)

            if ids:
                result[language] = ids

        return result

    def load(self) -> None:
        with self._lock:
            self.config_root.mkdir(parents=True, exist_ok=True)

            if not self.path.exists():
                self._endpoints = []
                self._active_by_language = {}
                self.save()
                return

            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                raw_endpoints = data.get("endpoints", [])

                if not isinstance(raw_endpoints, list):
                    raise ValueError("endpoints must be a list")

                # Validate one at a time: a single stale record (a name from
                # before the length cap, say) must not take the others with it.
                self._endpoints = []

                for item in raw_endpoints:
                    try:
                        self._endpoints.append(UserEndpoint.model_validate(item))
                    except ValidationError as exc:
                        label = (
                            item.get("id")
                            if isinstance(item, dict)
                            else None
                        )

                        logger.warning(
                            "Skipping unusable stored endpoint %r: %s",
                            label or "<unnamed>",
                            exc,
                        )

                self._active_by_language = self._normalize_active(
                    data.get("active_by_language", {})
                )

                self._repair_active_state()

            except (ValidationError, yaml.YAMLError, OSError, ValueError) as exc:
                logger.warning(
                    "Failed to load user endpoints: %s",
                    exc,
                )

                # Keep malformed config from preventing Fox Reader startup.
                self._endpoints = []
                self._active_by_language = {}

            self.save()

    def save(self) -> None:
        with self._lock:
            payload = {
                # This dump deliberately keeps `encryption.key`: the file is
                # where the secret lives. Only browser-facing surfaces redact
                # it, via UserEndpoint.public_dict.
                "endpoints": [
                    endpoint.model_dump(mode="json")
                    for endpoint in self._endpoints
                ],
                "active_by_language": {
                    language: list(ids)
                    for language, ids in self._active_by_language.items()
                },
            }

            with self.path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def all(self) -> list[UserEndpoint]:
        with self._lock:
            return copy.deepcopy(self._endpoints)

    def get(self, endpoint_id: str) -> UserEndpoint | None:
        endpoint_id = (endpoint_id or "").strip()

        if not endpoint_id:
            return None

        with self._lock:
            for endpoint in self._endpoints:
                if endpoint.id == endpoint_id:
                    return copy.deepcopy(endpoint)

        return None

    def for_language(self, language: str) -> list[UserEndpoint]:
        language = (language or "").strip().lower()

        with self._lock:
            return copy.deepcopy([
                endpoint
                for endpoint in self._endpoints
                if language in endpoint.languages
            ])

    def active_ids(self, language: str) -> list[str]:
        language = (language or "").strip().lower()

        with self._lock:
            return list(self._active_by_language.get(language, []))

    def active_for_language(self, language: str) -> list[UserEndpoint]:
        """Every active endpoint for a language, in activation order."""
        language = (language or "").strip().lower()

        with self._lock:
            result: list[UserEndpoint] = []

            for endpoint_id in self._active_by_language.get(language, []):
                for endpoint in self._endpoints:
                    if (
                        endpoint.id == endpoint_id
                        and language in endpoint.languages
                    ):
                        result.append(copy.deepcopy(endpoint))
                        break

            return result

    def active_summary(self) -> dict[str, list[dict[str, Any]]]:
        """The minimal shape the reader UI needs to build its buttons."""
        with self._lock:
            summary: dict[str, list[dict[str, Any]]] = {}

            for language in sorted(self._active_by_language):
                entries: list[dict[str, Any]] = []

                for endpoint_id in self._active_by_language[language]:
                    endpoint = next(
                        (
                            item
                            for item in self._endpoints
                            if item.id == endpoint_id
                            and language in item.languages
                        ),
                        None,
                    )

                    if endpoint is None:
                        continue

                    entries.append({
                        "id": endpoint.id,
                        "name": endpoint.name,
                        "url": endpoint.build_url(),
                        "method": endpoint.method,
                    })

                if entries:
                    summary[language] = entries

            return summary

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def create(self, payload: dict[str, Any]) -> UserEndpoint:
        endpoint = UserEndpoint.model_validate(payload)

        with self._lock:
            if any(existing.id == endpoint.id for existing in self._endpoints):
                raise ValueError(
                    f"Endpoint ID already exists: {endpoint.id}"
                )

            self._check_language_limits(endpoint.languages)

            self._endpoints.append(endpoint)
            self.save()

        return copy.deepcopy(endpoint)

    def update(
        self,
        endpoint_id: str,
        payload: dict[str, Any],
    ) -> UserEndpoint:
        with self._lock:
            index = next(
                (
                    index
                    for index, endpoint in enumerate(self._endpoints)
                    if endpoint.id == endpoint_id
                ),
                None,
            )

            if index is None:
                raise KeyError(endpoint_id)

            payload = dict(payload)
            payload["id"] = endpoint_id
            payload = self._preserve_secret(payload, self._endpoints[index])

            endpoint = UserEndpoint.model_validate(payload)

            self._check_language_limits(
                endpoint.languages,
                replacing=endpoint_id,
            )

            self._endpoints[index] = endpoint

            # Dropping a language also drops its activation.
            self._forget_activation(
                endpoint_id,
                keep_languages=endpoint.languages,
            )

            self.save()

            return copy.deepcopy(endpoint)

    def delete(self, endpoint_id: str) -> None:
        with self._lock:
            if not any(
                endpoint.id == endpoint_id
                for endpoint in self._endpoints
            ):
                raise KeyError(endpoint_id)

            self._endpoints = [
                endpoint
                for endpoint in self._endpoints
                if endpoint.id != endpoint_id
            ]

            self._forget_activation(endpoint_id)

            self.save()

    def set_active(
        self,
        language: str,
        endpoint_id: str,
        active: bool = True,
    ) -> list[str]:
        """Activate or deactivate one endpoint for one language."""
        language = (language or "").strip().lower()

        if not language:
            raise ValueError("language is required")

        with self._lock:
            current = list(self._active_by_language.get(language, []))

            if not active:
                if endpoint_id not in current:
                    raise ValueError(
                        "Endpoint is not active for this language"
                    )

                current.remove(endpoint_id)

                if current:
                    self._active_by_language[language] = current
                else:
                    self._active_by_language.pop(language, None)

                self.save()
                return current

            endpoint = self.get(endpoint_id)

            if endpoint is None:
                raise KeyError(endpoint_id)

            if language not in endpoint.languages:
                raise ValueError("Endpoint does not support this language")

            if endpoint_id in current:
                return current

            if len(current) >= MAX_ACTIVE_PER_LANGUAGE:
                raise ValueError(
                    f"'{language}' already has {MAX_ACTIVE_PER_LANGUAGE} "
                    "active endpoints. Deactivate one first."
                )

            current.append(endpoint_id)
            self._active_by_language[language] = current
            self.save()

            return current

    def clear_active(self, language: str) -> None:
        language = (language or "").strip().lower()

        with self._lock:
            self._active_by_language.pop(language, None)
            self.save()

    def with_stored_secret(
        self,
        endpoint_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """A draft payload with the stored encryption key filled back in.

        The editor never receives the key, so validating or previewing a saved
        endpoint would otherwise fail on the blank field.
        """
        endpoint_id = (endpoint_id or "").strip()

        if not endpoint_id:
            return payload

        with self._lock:
            current = next(
                (
                    endpoint
                    for endpoint in self._endpoints
                    if endpoint.id == endpoint_id
                ),
                None,
            )

        if current is None:
            return payload

        return self._preserve_secret(dict(payload), current)

    # ------------------------------------------------------------------
    # Serialization for the management UI
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            endpoints: list[dict[str, Any]] = []

            for endpoint in self._endpoints:
                item = endpoint.public_dict()

                item["url"] = endpoint.build_url()
                item["http2"] = endpoint.http2
                item["active_languages"] = sorted([
                    language
                    for language, ids in self._active_by_language.items()
                    if endpoint.id in ids
                ])

                endpoints.append(item)

            counts: dict[str, dict[str, int]] = {}

            for endpoint in self._endpoints:
                for language in endpoint.languages:
                    bucket = counts.setdefault(
                        language,
                        {"total": 0, "active": 0},
                    )
                    bucket["total"] += 1

            for language, ids in self._active_by_language.items():
                bucket = counts.setdefault(
                    language,
                    {"total": 0, "active": 0},
                )
                bucket["active"] = len(ids)

            return {
                "endpoints": endpoints,
                "active_by_language": {
                    language: list(ids)
                    for language, ids in self._active_by_language.items()
                },
                "counts": counts,
                "limits": {
                    "max_per_language": MAX_ENDPOINTS_PER_LANGUAGE,
                    "max_active_per_language": MAX_ACTIVE_PER_LANGUAGE,
                    "name_max_length": NAME_MAX_LENGTH,
                    "text_length_min": MIN_MAX_TEXT_LENGTH,
                    "text_length_max": MAX_MAX_TEXT_LENGTH,
                    "text_length_default": DEFAULT_MAX_TEXT_LENGTH,
                    "token_max_age_min": MIN_TOKEN_MAX_AGE,
                    "token_max_age_max": MAX_TOKEN_MAX_AGE,
                },
                "uuid_variants": list(UUID_VARIANTS),
                "encryption": {
                    "available": endpoint_crypto.available(),
                    "encodings": list(endpoint_crypto.KEY_ENCODINGS),
                    "key_length": endpoint_crypto.KEY_TEXT_LENGTH,
                    "max_keys": endpoint_crypto.MAX_KEYS,
                },
                "known_languages": list(KNOWN_LANGUAGES),
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _preserve_secret(
        payload: dict[str, Any],
        current: UserEndpoint,
    ) -> dict[str, Any]:
        """Carry a stored encryption key across an update that omits it.

        The key is never sent to the browser, so the edit form necessarily
        posts it back blank. Blank therefore means "leave it alone"; switching
        encryption off is what clears it.
        """
        raw = payload.get("encryption")

        if not isinstance(raw, dict):
            return payload

        block = dict(raw)
        incoming = block.get("key")
        incoming = incoming.strip() if isinstance(incoming, str) else ""

        if not incoming and current.encryption.key:
            block["key"] = current.encryption.key

        payload["encryption"] = block

        return payload

    def _check_language_limits(
        self,
        languages: list[str],
        replacing: str | None = None,
    ) -> None:
        for language in languages:
            count = sum(
                1
                for endpoint in self._endpoints
                if language in endpoint.languages
                and endpoint.id != replacing
            )

            if count >= MAX_ENDPOINTS_PER_LANGUAGE:
                raise ValueError(
                    f"Language '{language}' can have at most "
                    f"{MAX_ENDPOINTS_PER_LANGUAGE} endpoints"
                )

    def _forget_activation(
        self,
        endpoint_id: str,
        keep_languages: list[str] | None = None,
    ) -> None:
        """Drop an endpoint's activations, optionally keeping some languages."""
        keep = set(keep_languages) if keep_languages is not None else None

        for language in list(self._active_by_language):
            if keep is not None and language in keep:
                continue

            ids = [
                item
                for item in self._active_by_language[language]
                if item != endpoint_id
            ]

            if ids:
                self._active_by_language[language] = ids
            else:
                self._active_by_language.pop(language, None)

    def _repair_active_state(self) -> None:
        endpoint_by_id = {
            endpoint.id: endpoint
            for endpoint in self._endpoints
        }

        repaired: dict[str, list[str]] = {}

        for language, ids in self._active_by_language.items():
            kept: list[str] = []

            for endpoint_id in ids:
                endpoint = endpoint_by_id.get(endpoint_id)

                if endpoint is None or language not in endpoint.languages:
                    continue

                if endpoint_id in kept:
                    continue

                kept.append(endpoint_id)

                if len(kept) >= MAX_ACTIVE_PER_LANGUAGE:
                    break

            if kept:
                repaired[language] = kept

        self._active_by_language = repaired
