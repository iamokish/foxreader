"""Per-model context windows, checked before allocating a native GGUF model.

The catalogue records verified training limits. A GGUF may lower that limit,
including renamed weights, but cannot raise it. Reading its header never loads
tensor weights or imports llama.cpp; successful probes are cached by file identity.
"""

from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from fox_reader.constants import mtl_model

DEFAULT_CONTEXT_SIZE = 8192
CONTEXT_SIZE_OPTIONS = (8192, 16384)

# Actual files place architecture/context before the vocabulary. Bounds also
# keep malformed or unusual metadata from turning a settings request into a
# large read, allocation, or an unbounded walk through a string array.
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_METADATA_FIELDS = 4096
_MAX_ARRAY_STRINGS = 65536
_INTEGER_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 10: "Q", 11: "q"}
_SCALAR_TYPES = {**_INTEGER_TYPES, 6: "f", 7: "?", 12: "d"}


class _MetadataReader:
    def __init__(self, stream: BinaryIO, file_size: int):
        self.stream = stream
        self.limit = min(file_size, _MAX_METADATA_BYTES)
        self.endian = "<"
        self.strings_left = _MAX_ARRAY_STRINGS

    def read(self, size: int) -> bytes:
        if size < 0 or self.stream.tell() + size > self.limit:
            raise ValueError("GGUF metadata exceeds probe bounds")
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated GGUF metadata")
        return data

    def number(self, kind: str) -> int:
        return struct.unpack(self.endian + kind, self.read(struct.calcsize(kind)))[0]

    def skip(self, size: int) -> None:
        if size < 0 or self.stream.tell() + size > self.limit:
            raise ValueError("GGUF metadata exceeds probe bounds")
        self.stream.seek(size, 1)

    def string(self, *, maximum: int = 1024) -> str:
        size = self.number("Q")
        if size > maximum:
            raise ValueError("GGUF metadata name is too long")
        return self.read(size).decode("utf-8")

    def skip_value(self, value_type: int) -> None:
        if value_type in _SCALAR_TYPES:
            self.skip(struct.calcsize(_SCALAR_TYPES[value_type]))
        elif value_type == 8:
            self.skip(self.number("Q"))
        elif value_type == 9:
            element_type, count = self.number("I"), self.number("Q")
            if element_type in _SCALAR_TYPES:
                self.skip(count * struct.calcsize(_SCALAR_TYPES[element_type]))
            elif element_type == 8 and count <= self.strings_left:
                self.strings_left -= count
                for _ in range(count):
                    self.skip(self.number("Q"))
            else:
                raise ValueError("Unsupported or oversized GGUF metadata array")
        else:
            raise ValueError("Unknown GGUF metadata type")


@lru_cache(maxsize=64)
def _gguf_context(path: str, modified_ns: int, file_size: int) -> tuple[bool, int | None]:
    """Return (recognized GGUF header, declared context), without tensor I/O.

    Missing/invalid files remain the native loader's error to explain. A valid
    header with unrecognized or unbounded metadata is limited to the default.
    The stat arguments invalidate cached probes when weights are replaced.
    """
    recognized = False
    try:
        with Path(path).open("rb") as stream:
            reader = _MetadataReader(stream, file_size)
            if reader.read(4) != b"GGUF":
                return False, None
            version = reader.read(4)
            if struct.unpack("<I", version)[0] not in (2, 3):
                if struct.unpack(">I", version)[0] not in (2, 3):
                    return True, None
                reader.endian = ">"
            reader.number("Q")  # tensor count: tensor descriptors/data are never needed
            count = reader.number("Q")
            recognized = True
            if count > _MAX_METADATA_FIELDS:
                return True, None
            architecture = None
            limits: dict[str, int] = {}
            for _ in range(count):
                key = reader.string()
                value_type = reader.number("I")
                if key == "general.architecture" and value_type == 8:
                    architecture = reader.string(maximum=128)
                elif key.endswith(".context_length") and value_type in _INTEGER_TYPES:
                    limit = reader.number(_INTEGER_TYPES[value_type])
                    if limit > 0:
                        limits[key.removesuffix(".context_length")] = limit
                else:
                    reader.skip_value(value_type)
                if architecture in limits:
                    return True, limits[architecture]
    except (OSError, ValueError, OverflowError, struct.error):
        pass
    return recognized, None


def model_context_limit(model_id: str, model_path: Path | str | None = None) -> int:
    """Verified catalogue limit, lowered by the actual GGUF when available."""
    entry = mtl_model(model_id)
    if entry is None:
        raise ValueError(f"Unknown local translation model: {model_id}")
    declared = entry.get("max_context_tokens", DEFAULT_CONTEXT_SIZE)
    limit = declared if type(declared) is int and declared > 0 else DEFAULT_CONTEXT_SIZE
    if model_path is None:
        return limit
    try:
        path = Path(model_path).resolve()
        stat = path.stat()
    except OSError:
        return limit
    recognized, actual = _gguf_context(str(path), stat.st_mtime_ns, stat.st_size)
    if actual is not None:
        return min(limit, actual)
    return min(limit, DEFAULT_CONTEXT_SIZE) if recognized else limit


def context_size_options(model_id: str, model_path: Path | str | None = None) -> list[int]:
    """Available UI choices, never above the selected weights' known limit."""
    limit = model_context_limit(model_id, model_path)
    return [size for size in CONTEXT_SIZE_OPTIONS if size <= limit]


def resolve_context_size(
    model_id: str, requested: int = DEFAULT_CONTEXT_SIZE, model_path: Path | str | None = None,
) -> int:
    """Validate a stored/requested size and cap it to these model weights.

    Settings may reject a disallowed choice before saving. Loading rechecks the
    actual file, so replacement weights with a lower limit safely use 8192.
    """
    if type(requested) is not int or requested not in CONTEXT_SIZE_OPTIONS:
        raise ValueError("Context size must be 8192 or 16384 tokens.")
    options = context_size_options(model_id, model_path)
    if not options:
        raise ValueError("These model weights cannot support the minimum 8192-token context window.")
    return min(requested, options[-1])
