/*
 * Manual smoke check for the user_endpoints editor script.
 *
 * The page's JavaScript lives inside the Jinja template, so this renders the
 * template, lifts the <script> out of it and runs it against a stub DOM. That
 * is enough to exercise the parts with real logic in them — the uuid node type
 * and the encryption block — without a browser.
 *
 * Run it after rendering the template to $TMPDIR/ue-page.js:
 *
 *   node tests/manual/smoke_user_endpoints_editor.mjs
 *
 * Not part of the automated suite: it is a hand-run check, like the smoke
 * scripts beside it.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";

const SCRIPT_PATH = path.join(os.tmpdir(), "ue-page.js");
const script = fs.readFileSync(SCRIPT_PATH, "utf8");

/* ------------------------------------------------------------------ *
 * a DOM stub, just deep enough for the page to boot
 * ------------------------------------------------------------------ */

function makeEl(id, extra = {}) {
  const target = {
    id,
    value: "",
    textContent: "",
    innerHTML: "",
    type: "text",
    checked: false,
    disabled: false,
    tagName: "DIV",
    style: {},
    dataset: {},
    classList: { contains: () => false, add() {}, remove() {}, toggle() {} },
    ...extra,
  };

  return new Proxy(target, {
    get(object, prop) {
      if (prop in object) return object[prop];
      if (prop === "closest") return () => null;
      if (prop === "querySelectorAll") return () => [];
      if (prop === "querySelector") return () => null;
      return () => undefined;
    },
    set(object, prop, value) {
      object[prop] = value;
      return true;
    },
  });
}

const els = new Map();

const encFields = Array.from({ length: 5 }, (_, i) => makeEl(`enc-field-${i}`));

const langBoxes = [
  makeEl("lc-english", {
    value: "english",
    checked: true,
    classList: { contains: (name) => name === "lc" },
  }),
];

const document = {
  getElementById(id) {
    if (!els.has(id)) els.set(id, makeEl(id));
    return els.get(id);
  },
  querySelectorAll(selector) {
    if (selector === ".enc-field") return encFields;
    if (selector.includes(".lc")) return langBoxes;
    return [];
  },
  querySelector: () => null,
  createElement: () => makeEl("created"),
  addEventListener() {},
  body: makeEl("body"),
};

class BroadcastChannel {
  addEventListener() {}
  postMessage() {}
  close() {}
}

const sandbox = {
  document,
  BroadcastChannel,
  console,
  crypto: globalThis.crypto,
  btoa,
  CSS: { escape: (value) => value },
  confirm: () => false,
  fetch: async () => ({
    ok: true,
    json: async () => ({
      endpoints: [],
      counts: {},
      active_by_language: {},
      limits: {},
    }),
  }),
  setTimeout,
  clearTimeout,
};

sandbox.window = sandbox;
sandbox.globalThis = sandbox;

/* The page is a plain script, so appending to it lands in the same scope. */
const expose = `
;globalThis.__api = {
  uuidVariant, newNode, toModel, fromModel, example, typesFor,
  TYPES, RESPONSE_TYPES, generateFernetKey, checkKeyMaterial,
  collectEncryption, encryptionSummary, syncEncryptionFields,
  setType, setValue, addField, nodeAt, schemas, makePayload,
  defaultSchema, renderNode,
  get storedKeyPresent() { return storedKeyPresent },
  set storedKeyPresent(value) { storedKeyPresent = value },
  el: (id) => document.getElementById(id),
};
`;

vm.createContext(sandbox);
vm.runInContext(script + expose, sandbox, { filename: "user_endpoints.js" });

const api = sandbox.__api;
const el = api.el;

/* ------------------------------------------------------------------ *
 * assertions
 * ------------------------------------------------------------------ */

let failures = 0;

function check(label, condition, detail = "") {
  if (!condition) failures += 1;
  console.log(`${condition ? "ok  " : "FAIL"} ${label}${detail ? ` -- ${detail}` : ""}`);
}

function throws(label, run, needle = "") {
  try {
    run();
    failures += 1;
    console.log(`FAIL ${label} -- no error raised`);
    return;
  } catch (error) {
    const hit = !needle || error.message.includes(needle);
    if (!hit) failures += 1;
    console.log(`${hit ? "ok  " : "FAIL"} ${label} -- ${error.message.slice(0, 90)}`);
  }
}

console.log("--- boot ---");
check("boot ran resetDialog", el("encEnabled").value === "false");
check("key type defaulted", el("encEncoding").value === "fernet");
check("fields hidden while off", encFields.every((f) => f.style.display === "none"));

console.log("\n--- uuid node model ---");
check("uuid offered on requests", api.typesFor("request").includes("uuid"));
check("uuid withheld from responses", !api.typesFor("response").includes("uuid"));
check("unknown variant falls back", api.uuidVariant("uuid7") === "uuid4");
check("variant normalised", api.uuidVariant(" UUID1 ") === "uuid1");

const node = api.newNode("uuid", "request_id");
check("new node defaults to uuid4", node.value === "uuid4");
check("new node is static", node.source === "static");
check("example is a token", api.example(node) === "<UUID4>");
check(
  "toModel shape",
  JSON.stringify(api.toModel(node)) === '{"type":"uuid","source":"static","value":"uuid4"}',
  JSON.stringify(api.toModel(node)),
);

node.value = "uuid2";
check("uuid2 survives toModel", api.toModel(node).value === "uuid2");
check("uuid2 example", api.example(node) === "<UUID2>");

node.value = "uuid9";
check("toModel repairs a bad variant", api.toModel(node).value === "uuid4");

check(
  "fromModel repairs a bad variant",
  api.fromModel({ type: "uuid", source: "static", value: "nope" }).value === "uuid4",
);
check(
  "fromModel keeps uuid1",
  api.fromModel({ type: "uuid", source: "static", value: "uuid1" }).value === "uuid1",
);

api.schemas.request = api.defaultSchema("request");
api.addField("request", []);

const last = api.schemas.request.children.length - 1;
const lastPath = [`c:${last}`];

api.setType("request", lastPath, "uuid");
check("setType built a uuid node", api.nodeAt("request", lastPath).type === "uuid");

api.setValue("request", lastPath, "uuid1");
check("setValue stored the variant", api.nodeAt("request", lastPath).value === "uuid1");

const rendered = api.renderNode("request", api.nodeAt("request", lastPath), lastPath, "field");
check("value slot is a variant picker", rendered.includes('aria-label="UUID variant"'));
check(
  "all three variants offered",
  ["uuid1", "uuid2", "uuid4"].every((variant) => rendered.includes(`value="${variant}"`)),
);
check("the stored variant is selected", rendered.includes('value="uuid1" selected'));
check("type select offers uuid", rendered.includes('value="uuid"'));

const responseRendered = api.renderNode("response", api.newNode("string", "x"), [], "field");
check("response type select has no uuid", !responseRendered.includes('value="uuid"'));

api.setType("request", lastPath, "string");
api.nodeAt("request", lastPath).source = "text";
api.setType("request", lastPath, "uuid");
check(
  "a dynamic source is dropped when switching to uuid",
  api.nodeAt("request", lastPath).source === "static",
);

console.log("\n--- key generation ---");
const keys = Array.from({ length: 5 }, () => api.generateFernetKey());
check("44 characters", keys.every((key) => key.length === 44), keys[0]);
check("url-safe alphabet", keys.every((key) => /^[A-Za-z0-9_-]{43}=$/.test(key)));
check("all distinct", new Set(keys).size === keys.length);

fs.writeFileSync(path.join(os.tmpdir(), "ue-keys.txt"), keys.join("\n"), "utf8");

console.log("\n--- key material checks ---");
api.checkKeyMaterial(keys[0], "fernet");
console.log("ok   one generated key");
api.checkKeyMaterial(keys.slice(0, 4).join(" "), "fernet");
console.log("ok   four keys, space separated");
api.checkKeyMaterial(keys.slice(0, 3).join(", "), "fernet");
console.log("ok   three keys, comma separated");
throws("five keys rejected", () => api.checkKeyMaterial(keys.join(" "), "fernet"), "At most 4 keys");
throws("short key rejected", () => api.checkKeyMaterial("abc", "fernet"), "44-character");
throws("unpadded key rejected", () => api.checkKeyMaterial(keys[0].slice(0, 43), "fernet"));

api.checkKeyMaterial("correct horse battery staple", "passphrase");
console.log("ok   a phrase with spaces is one passphrase");
api.checkKeyMaterial("new phrase, old phrase", "passphrase");
console.log("ok   two phrases, comma separated");
throws(
  "five passphrases rejected",
  () => api.checkKeyMaterial("a,b,c,d,e", "passphrase"),
  "At most 4 passphrases",
);

console.log("\n--- collectEncryption ---");

function setEnc({
  enabled = "false",
  key = "",
  encoding = "fernet",
  request = "true",
  response = "true",
  maxAge = "",
} = {}) {
  el("encEnabled").value = enabled;
  el("encKey").value = key;
  el("encEncoding").value = encoding;
  el("encRequest").value = request;
  el("encResponse").value = response;
  el("encMaxAge").value = maxAge;
}

setEnc();
let block = api.collectEncryption();
check("off: disabled", block.enabled === false);
check("off: max age null", block.max_age === null);
check("off: blank key", block.key === "");

setEnc({ enabled: "true", key: keys[0], maxAge: "600" });
block = api.collectEncryption();
check("on: enabled", block.enabled === true);
check("on: key carried", block.key === keys[0]);
check("on: encoding carried", block.key_encoding === "fernet");
check("on: both directions", block.encrypt_request === true && block.decrypt_response === true);
check("on: max age carried", block.max_age === 600);

setEnc({ enabled: "true" });
api.storedKeyPresent = false;
throws("on with no key anywhere", () => api.collectEncryption(), "needs a key");

api.storedKeyPresent = true;
block = api.collectEncryption();
check("blank key allowed once one is stored", block.key === "");
api.storedKeyPresent = false;

setEnc({ enabled: "true", key: "not-a-key" });
throws("bad key shape", () => api.collectEncryption(), "44-character");

setEnc({ enabled: "true", key: keys[0], request: "false", response: "false" });
throws("neither direction", () => api.collectEncryption(), "neither direction");

setEnc({ enabled: "true", key: keys[0], maxAge: "5" });
throws("max age under the floor", () => api.collectEncryption(), "between 30 and 86400");

setEnc({ enabled: "true", key: keys[0], maxAge: "999999" });
throws("max age over the cap", () => api.collectEncryption(), "between 30 and 86400");

setEnc({ enabled: "true", key: "words words words", encoding: "passphrase" });
block = api.collectEncryption();
check(
  "a phrase is accepted whole",
  block.key_encoding === "passphrase" && block.key === "words words words",
);

setEnc({ key: "not-a-key", maxAge: "5" });
throws("max age still checked while off", () => api.collectEncryption(), "between 30 and 86400");

setEnc({ key: "not-a-key" });
block = api.collectEncryption();
check("a bad key is ignored while off", block.enabled === false);

console.log("\n--- syncEncryptionFields ---");
setEnc({ enabled: "true", key: keys[0] });
api.storedKeyPresent = false;
api.syncEncryptionFields();
check("fields shown when on", encFields.every((field) => field.style.display === ""));
check("generate offered for fernet", el("encGenerate").style.display === "");
check("fresh hint says it is never returned", el("encKeyHint").innerHTML.includes("never returned"));
check("hint names the rotation cap", el("encKeyHint").innerHTML.includes("Up to 4"));
check("fernet hint names generate_key", el("encEncodingHint").innerHTML.includes("generate_key()"));

api.storedKeyPresent = true;
api.syncEncryptionFields();
check("stored-key hint says leave it empty", el("encKeyHint").innerHTML.includes("Leave the box empty"));

el("encEncoding").value = "passphrase";
api.syncEncryptionFields();
check("generate hidden for a passphrase", el("encGenerate").style.display === "none");
check("passphrase hint names scrypt", el("encEncodingHint").innerHTML.includes("scrypt"));
check("passphrase hint says commas", el("encKeyHint").innerHTML.includes("separated by commas"));

el("encEnabled").value = "false";
api.syncEncryptionFields();
check("fields hidden again", encFields.every((field) => field.style.display === "none"));

console.log("\n--- makePayload ---");
el("name").value = "MyAPI";
el("host").value = "api.example.com/v2/translate";
el("timeout").value = "20";
el("retries").value = "0";
el("maxTextLength").value = "2000";
el("method").value = "POST";
el("scheme").value = "https";
el("bodyFormat").value = "json";
el("responseFormat").value = "json";
el("targetLanguage").value = "en";
el("doseq").value = "false";

api.schemas.request = api.defaultSchema("request");
api.schemas.response = api.defaultSchema("response");
api.addField("request", []);

const index = api.schemas.request.children.length - 1;
const uuidPath = [`c:${index}`];

api.setType("request", uuidPath, "uuid");
api.nodeAt("request", uuidPath).key = "request_id";
api.setValue("request", uuidPath, "uuid2");

setEnc({ enabled: "true", key: keys[1], maxAge: "300" });
api.storedKeyPresent = false;

const payload = api.makePayload();
check("payload carries an encryption block", Boolean(payload.encryption));
check("payload encryption is on", payload.encryption.enabled === true);
check("payload carries the key", payload.encryption.key === keys[1]);
check("payload carries the max age", payload.encryption.max_age === 300);

const uuidChild = payload.request_schema.children.request_id;
check(
  "payload carries the uuid node",
  Boolean(uuidChild) && uuidChild.type === "uuid" && uuidChild.value === "uuid2",
  JSON.stringify(uuidChild),
);

fs.writeFileSync(path.join(os.tmpdir(), "ue-payload.json"), JSON.stringify(payload), "utf8");

console.log("\n--- encryptionSummary ---");
check("nothing configured", api.encryptionSummary({}) === "Off");
check("explicitly off", api.encryptionSummary({ encryption: { enabled: false } }) === "Off");
check(
  "both directions",
  api.encryptionSummary({
    encryption: {
      enabled: true,
      encrypt_request: true,
      decrypt_response: true,
      key_encoding: "fernet",
    },
  }) === "On · text sent + reply · fernet",
);
check(
  "one direction with an age",
  api.encryptionSummary({
    encryption: {
      enabled: true,
      encrypt_request: true,
      key_encoding: "passphrase",
      max_age: 300,
    },
  }) === "On · text sent · passphrase · 300s max age",
);

console.log(failures ? `\n${failures} FAILURE(S)` : "\nall editor checks passed");
process.exit(failures ? 1 : 0);
