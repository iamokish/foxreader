"""Authenticated encryption for the text sent to a custom endpoint.

The point is narrow: keep the *content* of a selection out of the request body,
so whoever can read the traffic between Fox Reader and a translation endpoint
cannot read what is being translated. The endpoint on the other side holds the
same shared key and reverses it.

Scheme
------
Fernet, from the ``cryptography`` package — AES-128-CBC for confidentiality
with HMAC-SHA256 for authenticity, encrypt-then-MAC, plus a creation timestamp
inside the authenticated token. A Fernet token is URL-safe base64, so it drops
straight into a JSON string, a form field or a query parameter with no extra
escaping.

Fernet is used rather than a hand-rolled construction on purpose: it is a
single well-reviewed primitive with a published token format and independent
implementations in JavaScript, Go, Ruby, PHP and Rust, so the server on the
other end does not have to match a bespoke scheme.

What this protects and what it does not
---------------------------------------
Protected: the plaintext of the selection and of the translation coming back,
against anyone who can observe or store the request body — a proxy, a
TLS-terminating middlebox, a shared host's access logs, an endpoint operator's
own request logs.

Not protected: that a request happened, when, or roughly how large it was
(token length tracks plaintext length). The endpoint itself necessarily sees
the plaintext — it has the key and has to read the text to translate it. This
is a shared-secret scheme, not secrecy from the endpoint.

Keys
----
``fernet`` mode takes the exact 44-character key that ``Fernet.generate_key()``
produces (32 bytes, URL-safe base64). Paste the same string on both sides.

``passphrase`` mode stretches a memorable phrase into that key with scrypt,
using the fixed public salt and cost parameters named below. There is no way to
exchange a random salt — the user types the same words in two places and
nothing else is shared — so the salt is a constant and the cost parameter
carries the weight. A server reproducing the key must use ``SCRYPT_SALT``,
``SCRYPT_N``, ``SCRYPT_R`` and ``SCRYPT_P`` verbatim, then URL-safe-base64 the
resulting 32 bytes.

Rotation
--------
More than one key may be given, whitespace- or comma-separated. The first is
used to encrypt; every key is tried when decrypting. That allows a key to be
changed on the endpoint without a flag day: add the new key in front, keep the
old one until the endpoint has moved over, then drop it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import threading
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Key material
# ---------------------------------------------------------------------------

KEY_ENCODINGS: tuple[str, ...] = ("fernet", "passphrase")

#: Bytes behind a Fernet key: a 16-byte signing key and a 16-byte AES key.
KEY_BYTES = 32

#: Length of the base64 text form, which is what a user pastes.
KEY_TEXT_LENGTH = 44

# Part of the interoperability contract for passphrase mode — a server must use
# these verbatim to derive the same key.
SCRYPT_SALT = b"fox-reader/user-endpoint/v1"
SCRYPT_N = 1 << 14
SCRYPT_R = 8
SCRYPT_P = 1

#: How many keys one endpoint may carry, to bound rotation lists.
MAX_KEYS = 4

# scrypt costs ~16 MiB and tens of milliseconds: fine once, far too slow per
# request. The built cipher is cached against the secret it came from.
_CIPHER_CACHE: OrderedDict[bytes, object] = OrderedDict()
_CIPHER_CACHE_LIMIT = 32
_CIPHER_CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Hardening limits
# ---------------------------------------------------------------------------

# A response is attacker-controlled: a hostile or broken endpoint can return
# anything at all. Cap the work before decoding rather than after.
MAX_TOKEN_CHARS = 400_000
MAX_PLAINTEXT_BYTES = 256 * 1024

# version byte + 8-byte timestamp + 16-byte IV + 32-byte HMAC, before any
# ciphertext. Used only as a cheap floor before handing work to Fernet.
_MIN_TOKEN_BYTES = 57
_FERNET_VERSION = 0x80

_KEY_SPLIT = re.compile(r"[\s,]+")


class EncryptionError(Exception):
    """Base class for every failure in this module."""


class EncryptionUnavailable(EncryptionError):
    """The `cryptography` package is missing, so encryption cannot run."""


class InvalidKeyError(EncryptionError):
    """The configured key material is unusable."""


class DecryptionError(EncryptionError):
    """The token was malformed, or it failed authentication."""


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

_fernet_cache: list = []


def _fernet_classes():
    """`(Fernet, MultiFernet, InvalidToken)`, or None when unavailable."""
    if not _fernet_cache:
        try:
            from cryptography.fernet import Fernet, InvalidToken, MultiFernet
        except Exception:  # pragma: no cover - depends on the environment
            _fernet_cache.append(None)
        else:
            _fernet_cache.append((Fernet, MultiFernet, InvalidToken))

    return _fernet_cache[0]


def available() -> bool:
    """Whether this installation can encrypt at all."""
    return _fernet_classes() is not None


def _require() -> tuple:
    classes = _fernet_classes()

    if classes is None:
        raise EncryptionUnavailable(
            "Encryption needs the 'cryptography' package, which is not "
            "installed in this environment. Install it with "
            "'pip install cryptography', or turn encryption off for this "
            "endpoint."
        )

    return classes


def generate_key() -> str:
    """A fresh random key, ready to paste on both sides."""
    Fernet, _, _ = _require()
    return Fernet.generate_key().decode("ascii")


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------


def split_keys(secret: str) -> list[str]:
    """Every key in a secret, newest first. Order decides which encrypts."""
    if not secret:
        return []

    return [part for part in _KEY_SPLIT.split(secret.strip()) if part]


def _validate_fernet_key(key: str, *, index: int, total: int) -> bytes:
    """Check one pasted key and return its raw bytes."""
    where = "" if total == 1 else f" (key {index + 1} of {total})"

    try:
        raw = base64.urlsafe_b64decode(key.encode("ascii"))
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise InvalidKeyError(
            f"The key is not valid URL-safe base64{where}. Expected the "
            f"{KEY_TEXT_LENGTH}-character string that Fernet.generate_key() "
            "produces."
        ) from exc

    if len(raw) != KEY_BYTES:
        raise InvalidKeyError(
            f"The key must decode to exactly {KEY_BYTES} bytes{where}; this "
            f"one decodes to {len(raw)}. Use a key from "
            "Fernet.generate_key(), or switch to passphrase mode to derive "
            "one from a memorable phrase."
        )

    return raw


def _passphrase_to_key(passphrase: str) -> str:
    raw = hashlib.scrypt(
        passphrase.encode("utf-8"),
        salt=SCRYPT_SALT,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=KEY_BYTES,
        maxmem=64 * 1024 * 1024,
    )

    return base64.urlsafe_b64encode(raw).decode("ascii")


def split_secret(secret: str, encoding: str = "fernet") -> list[str]:
    """The individual secrets inside one key field, newest first.

    Structural checks only. A ``fernet`` part must decode to a 32-byte key; a
    passphrase has nothing to check beyond how many there are, and is
    deliberately *not* stretched here — derivation costs ~60ms and this runs
    on every config load.
    """
    if encoding not in KEY_ENCODINGS:
        raise InvalidKeyError(
            f"Unknown key encoding: {encoding!r}. Expected one of: "
            + ", ".join(KEY_ENCODINGS)
            + "."
        )

    if not secret or not secret.strip():
        raise InvalidKeyError("An encryption key is required.")

    if encoding == "passphrase":
        # A passphrase may legitimately contain spaces, so only a newline or a
        # comma separates one phrase from the next.
        parts = [
            part.strip()
            for part in re.split(r"[\n\r,]+", secret.strip())
            if part.strip()
        ]
        noun = "passphrases"
    else:
        parts = split_keys(secret)
        noun = "keys"

    if not parts:
        raise InvalidKeyError("An encryption key is required.")

    if len(parts) > MAX_KEYS:
        raise InvalidKeyError(
            f"At most {MAX_KEYS} {noun} are allowed; found {len(parts)}."
        )

    if encoding == "fernet":
        total = len(parts)

        for index, key in enumerate(parts):
            _validate_fernet_key(key, index=index, total=total)

    return parts


def resolve_keys(secret: str, encoding: str = "fernet") -> list[str]:
    """The Fernet key strings a secret resolves to, newest first.

    In ``passphrase`` mode each phrase is stretched into a key, so rotation
    works the same way as with raw keys.
    """
    parts = split_secret(secret, encoding)

    if encoding == "passphrase":
        return [_passphrase_to_key(part) for part in parts]

    return parts


def validate_secret(secret: str, encoding: str = "fernet") -> int:
    """Check key material without deriving or building anything.

    Returns the number of keys. Raises `InvalidKeyError` with a message meant
    to be shown in the editor.
    """
    return len(split_secret(secret, encoding))


# ---------------------------------------------------------------------------
# Cipher construction
# ---------------------------------------------------------------------------


def _build_cipher(secret: str, encoding: str):
    Fernet, MultiFernet, _ = _require()

    keys = resolve_keys(secret, encoding)
    ciphers = [Fernet(key.encode("ascii")) for key in keys]

    # MultiFernet encrypts with the first key and decrypts with any of them.
    return ciphers[0] if len(ciphers) == 1 else MultiFernet(ciphers)


def cipher_for(secret: str, encoding: str = "fernet"):
    """A Fernet (or MultiFernet) for this secret, cached.

    The cache exists for passphrase mode, where rebuilding means re-running
    scrypt. It is keyed by a digest of the secret, never by the secret itself.
    """
    _require()

    cache_key = hashlib.sha256(
        encoding.encode("ascii") + b"\x00" + secret.encode("utf-8")
    ).digest()

    with _CIPHER_CACHE_LOCK:
        cached = _CIPHER_CACHE.get(cache_key)

        if cached is not None:
            _CIPHER_CACHE.move_to_end(cache_key)
            return cached

    cipher = _build_cipher(secret, encoding)

    with _CIPHER_CACHE_LOCK:
        _CIPHER_CACHE[cache_key] = cipher
        _CIPHER_CACHE.move_to_end(cache_key)

        while len(_CIPHER_CACHE) > _CIPHER_CACHE_LIMIT:
            _CIPHER_CACHE.popitem(last=False)

    return cipher


# ---------------------------------------------------------------------------
# Token shape
# ---------------------------------------------------------------------------


def looks_encrypted(value: object) -> bool:
    """Whether `value` is shaped like a Fernet token.

    Lets "the endpoint replied in plaintext" be reported differently from "the
    endpoint replied with something that would not decrypt" — different
    mistakes with different fixes.
    """
    if not isinstance(value, str):
        return False

    value = value.strip()

    if not value or len(value) > MAX_TOKEN_CHARS:
        return False

    try:
        raw = base64.urlsafe_b64decode(value.encode("ascii"))
    except (binascii.Error, ValueError, UnicodeEncodeError):
        return False

    return len(raw) >= _MIN_TOKEN_BYTES and raw[0] == _FERNET_VERSION


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encrypt(plaintext: str, *, secret: str, encoding: str = "fernet") -> str:
    """Wrap `plaintext` in a Fernet token."""
    raw = plaintext.encode("utf-8")

    if len(raw) > MAX_PLAINTEXT_BYTES:
        raise EncryptionError(
            f"Refusing to encrypt {len(raw)} bytes; the limit is "
            f"{MAX_PLAINTEXT_BYTES}."
        )

    cipher = cipher_for(secret, encoding)

    try:
        return cipher.encrypt(raw).decode("ascii")
    except EncryptionError:
        raise
    except Exception as exc:
        # Never let a library message carry key material into a response.
        raise EncryptionError("Could not encrypt the text.") from exc


def decrypt(
    token: str,
    *,
    secret: str,
    encoding: str = "fernet",
    max_age: int | None = None,
) -> str:
    """Unwrap and verify a Fernet token.

    `max_age`, in seconds, rejects a token whose embedded timestamp is older
    than that — a replay guard. It is off by default because it makes the
    result depend on the two machines' clocks agreeing. Fernet timestamps have
    whole-second granularity, so the effective cutoff can run up to a second
    past `max_age`; values below a few seconds are not meaningful.
    """
    _, _, InvalidToken = _require()

    if not isinstance(token, str) or not token.strip():
        raise DecryptionError("The endpoint returned an empty response.")

    token = token.strip()

    if len(token) > MAX_TOKEN_CHARS:
        raise DecryptionError(
            f"The encrypted response is {len(token)} characters, past the "
            f"{MAX_TOKEN_CHARS} limit."
        )

    if not looks_encrypted(token):
        raise DecryptionError(
            "The endpoint's reply is not an encrypted token. It is set to "
            "decrypt responses, so the endpoint must encrypt what it sends "
            "back with the same key."
        )

    cipher = cipher_for(secret, encoding)

    try:
        raw = cipher.decrypt(token.encode("ascii"), ttl=max_age or None)
    except InvalidToken as exc:
        raise DecryptionError(
            "The encrypted response failed authentication. The key on each "
            "side probably differs, or the reply was altered in transit."
            + (
                f" It may also have expired: this endpoint rejects tokens "
                f"older than {max_age}s."
                if max_age
                else ""
            )
        ) from exc
    except Exception as exc:
        raise DecryptionError("Could not decrypt the response.") from exc

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DecryptionError(
            "The decrypted response is not valid UTF-8 text."
        ) from exc


# ---------------------------------------------------------------------------
# Self check
# ---------------------------------------------------------------------------


def self_test() -> dict[str, bool]:
    """Round-trip, tamper, wrong-key and rotation checks.

    Raises on the first real failure, so it doubles as a startup assertion if
    one is ever wanted.
    """
    _require()

    results: dict[str, bool] = {}
    sample = "こんにちは、世界 — mixed ascii & 日本語 " * 3

    key = generate_key()
    token = encrypt(sample, secret=key)

    if decrypt(token, secret=key) != sample:
        raise EncryptionError("round trip did not match")

    results["round_trip"] = True

    if not looks_encrypted(token):
        raise EncryptionError("token not recognised by looks_encrypted")

    results["recognised"] = True

    # Flipping one character of the token must fail the HMAC.
    flipped = "A" if token[30] != "A" else "B"
    tampered = token[:30] + flipped + token[31:]

    try:
        decrypt(tampered, secret=key)
    except DecryptionError:
        results["tamper_detected"] = True
    else:
        raise EncryptionError("tampering went undetected")

    try:
        decrypt(token, secret=generate_key())
    except DecryptionError:
        results["wrong_key_rejected"] = True
    else:
        raise EncryptionError("wrong key was accepted")

    # Rotation: a token made under the old key still reads once the new key is
    # in front, and new tokens use the new key.
    old_key = key
    new_key = generate_key()
    rotated = f"{new_key} {old_key}"

    if decrypt(token, secret=rotated) != sample:
        raise EncryptionError("rotation did not accept the old key")

    fresh = encrypt(sample, secret=rotated)

    if decrypt(fresh, secret=new_key) != sample:
        raise EncryptionError("rotation did not encrypt with the new key")

    try:
        decrypt(fresh, secret=old_key)
    except DecryptionError:
        results["rotation"] = True
    else:
        raise EncryptionError("rotation encrypted with the wrong key")

    # Passphrase mode is deterministic across processes.
    phrase = "correct horse battery staple"
    sealed = encrypt(sample, secret=phrase, encoding="passphrase")

    if decrypt(sealed, secret=phrase, encoding="passphrase") != sample:
        raise EncryptionError("passphrase round trip did not match")

    if resolve_keys(phrase, "passphrase") != resolve_keys(phrase, "passphrase"):
        raise EncryptionError("passphrase derivation is not deterministic")

    results["passphrase"] = True

    return results
