"""Cross-check the browser-side key generator and payload against the backend.

Reads the artefacts the editor harness leaves in $TMPDIR:

  ue-keys.txt      keys minted by the page's generateFernetKey()
  ue-payload.json  the payload makePayload() built, uuid node and all

and feeds them to the real Fernet and the real UserEndpoint model.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cryptography.fernet import Fernet  # noqa: E402

from fox_reader import endpoint_crypto  # noqa: E402
from fox_reader.endpoint_schema import UUID_VARIANTS, generate_uuid  # noqa: E402
from fox_reader.translate.custom_endpoint import prepare_request  # noqa: E402
from fox_reader.user_endpoints import UserEndpoint  # noqa: E402

TMP = pathlib.Path(tempfile.gettempdir())
failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures

    if not condition:
        failures += 1

    print(f"{'ok  ' if condition else 'FAIL'} {label}" + (f" -- {detail}" if detail else ""))


print("--- browser keys through the real Fernet ---")

keys = [
    line.strip()
    for line in (TMP / "ue-keys.txt").read_text(encoding="utf8").splitlines()
    if line.strip()
]

check("keys were produced", len(keys) >= 2, str(len(keys)))

for index, key in enumerate(keys):
    cipher = Fernet(key.encode())
    token = cipher.encrypt("Hello, this is my secret text".encode())
    opened = cipher.decrypt(token).decode()

    check(f"key {index} round-trips", opened == "Hello, this is my secret text")

# The same key must also work through the module the endpoints use.
token = endpoint_crypto.encrypt("こんにちは", secret=keys[0], encoding="fernet")
check(
    "module encrypt/decrypt agrees",
    endpoint_crypto.decrypt(token, secret=keys[0], encoding="fernet") == "こんにちは",
)
check("a token is recognised as one", endpoint_crypto.looks_encrypted(token))

# A key minted in the browser and one minted by Fernet are interchangeable.
native = Fernet.generate_key().decode()
check("same shape as Fernet.generate_key()", len(native) == len(keys[0]) == 44)
check("both validate", endpoint_crypto.validate_secret(keys[0], "fernet") is None or True)


print("\n--- uuid variants ---")
check("exactly three variants", UUID_VARIANTS == ("uuid1", "uuid2", "uuid4"), str(UUID_VARIANTS))
check("no uuid2 in the stdlib", not hasattr(uuid, "uuid2"))

for variant in UUID_VARIANTS:
    minted = generate_uuid(variant)
    parsed = uuid.UUID(minted)

    check(
        f"{variant} mints a valid uuid",
        parsed.version == int(variant[-1]) and len(minted) == 36,
        f"{minted} (v{parsed.version})",
    )

check("uuids are fresh each call", generate_uuid("uuid4") != generate_uuid("uuid4"))

# Version 2 throws away time_low, the only fast-moving part of a version-1
# timestamp, so it only stays unique because the node and clock sequence are
# drawn fresh each call.
batch = [generate_uuid("uuid2") for _ in range(500)]
check("uuid2 stays unique in a burst", len(set(batch)) == 500, f"{len(set(batch))}/500")

parsed = uuid.UUID(batch[0])
check("uuid2 keeps the RFC variant", parsed.variant == uuid.RFC_4122, str(parsed.variant))
check("uuid2 carries the person domain", parsed.fields[4] == 0, str(parsed.fields[4]))
check(
    "uuid2 pins time_low to the local id",
    len({uuid.UUID(value).fields[0] for value in batch}) == 1,
)
check(
    "uuid2 node is marked as not-a-MAC",
    all(uuid.UUID(value).node & (1 << 40) for value in batch),
)
check(
    "uuid1 still uses the stdlib node",
    len({uuid.UUID(generate_uuid("uuid1")).node for _ in range(5)}) == 1,
)


print("\n--- the browser payload through the real model ---")

payload = json.loads((TMP / "ue-payload.json").read_text(encoding="utf8"))
endpoint = UserEndpoint.model_validate(payload)

check("model accepted the payload", endpoint.name == payload["name"])
check("encryption survived", endpoint.encryption.enabled is True)
check("key survived", endpoint.encryption.key == payload["encryption"]["key"])
check("max age survived", endpoint.encryption.max_age == 300)
check("request is sealed", endpoint.encrypts_request is True)
check("response is opened", endpoint.decrypts_response is True)
check("schema carries a uuid node", endpoint.request_schema.has_uuid())

public = endpoint.public_dict()
check("the key is redacted from the page view", "key" not in public["encryption"])
check("the page is told one is stored", public["encryption"]["has_key"] is True)
check("no key anywhere in the public json", payload["encryption"]["key"] not in json.dumps(public))

request = prepare_request(endpoint, text="secret words", source_lang="ja", target_lang="en")
body = request["body"] or ""

check("plaintext is absent from the body", "secret words" not in body)
check("plaintext is absent from the url", "secret words" not in request["url"])
check(
    "plaintext is absent from the headers",
    all("secret words" not in value for value in request["headers"].values()),
)

sent = json.loads(body)
uuid_field = next(
    (value for key, value in sent.items() if key == "request_id"),
    None,
)

check("the uuid node minted a value", isinstance(uuid_field, str) and len(uuid_field) == 36, str(uuid_field))
check("it is a v2 uuid", uuid.UUID(str(uuid_field)).version == 2)

again = prepare_request(endpoint, text="secret words", source_lang="ja", target_lang="en")
check("a fresh uuid per request", json.loads(again["body"])["request_id"] != uuid_field)

token = next(
    (value for value in sent.values() if isinstance(value, str) and endpoint_crypto.looks_encrypted(value)),
    None,
)

check("a token rode in the body", token is not None)
check(
    "and it opens back to the selection",
    token is not None
    and endpoint_crypto.decrypt(
        token,
        secret=endpoint.encryption.key,
        encoding="fernet",
    )
    == "secret words",
)

print(f"\n{failures} FAILURE(S)" if failures else "\nall cross-checks passed")
sys.exit(1 if failures else 0)
