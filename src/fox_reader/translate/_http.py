import json
import gzip
import zlib

import brotli


def decompress_response(content: bytes, encoding: str) -> bytes:
    """Decompress HTTP response body based on Content-Encoding header."""
    if "gzip" in encoding:
        return gzip.decompress(content)
    if "deflate" in encoding:
        return zlib.decompress(content)
    if "br" in encoding:
        return brotli.decompress(content)
    return content


def parse_json_response(content: bytes, encoding: str, default=None):
    """Parse JSON from HTTP response, auto-decompressing if needed."""
    try:
        return json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    try:
        content = decompress_response(content, encoding)
        return json.loads(content.decode("utf-8"))
    except Exception:
        return default
