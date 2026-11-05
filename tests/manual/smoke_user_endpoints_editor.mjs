/*
 * Manual smoke check for the user_endpoints editor script.
 *
 * The page's JavaScript lives inside the Jinja template, so this renders the
 * template, lifts the <script> out of it and runs it against a stub DOM. That
 * is enough to exercise schema editing, saved keys and async feedback/test
 * behavior without a browser.
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
  const listeners = new Map();
  const target = {
    id,
    value: "",
    textContent: "",
    innerHTML: "",
    type: "text",
    checked: false,
    disabled: false,
    open: false,
    modalShows: 0,
    tagName: "DIV",
    style: {},
    dataset: {},
    classList: { contains: () => false, add() {}, remove() {}, toggle() {} },
    addEventListener(type, listener) {
      const handlers = listeners.get(type) || [];
      handlers.push(listener);
      listeners.set(type, handlers);
    },
    dispatch(type, event = {}) {
      for (const listener of listeners.get(type) || []) {
        listener({ target: this, preventDefault() {}, ...event });
      }
    },
    showModal() { this.open = true; this.modalShows += 1; },
    close() { this.open = false; this.dispatch("close"); },
    getBoundingClientRect() { return { left: 100, top: 100, right: 500, bottom: 400 }; },
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
  AbortController,
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
  defaultSchema, renderNode, dynamicNode, setSource, availableSources,
  validateEditor, updatePreview,
  revealKey, fillGeneratedKey, resetDialog, openEdit, closeDialog,
  status, closeFeedback, testEndpoint, renderCards,
  get state() { return state },
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

console.log("\n--- character and context schema ---");
api.schemas.request = api.defaultSchema("request");
const rosterNode = api.dynamicNode("list", "roster", "character_info");
api.schemas.request.children.push(rosterNode);
api.validateEditor("request");
console.log("ok   a roster works without history or speaker links");

const speakerNode = api.newNode("string", "speaker");
api.schemas.request.children.push(speakerNode);
const speakerPath = [`c:${api.schemas.request.children.length - 1}`];
check("current speaker offered on request strings", api.availableSources("request", speakerNode).includes("meta_id"));
check("current speaker withheld from arrays", !api.availableSources("request", api.newNode("list")).includes("meta_id"));
check("current speaker withheld from responses", !api.availableSources("response", speakerNode).includes("meta_id"));
api.setSource("request", speakerPath, "meta_id");
api.validateEditor("request");
console.log("ok   current speaker and roster work without context");
check("current speaker preview identifies its dynamic value", api.example(speakerNode) === "<CURRENT_SPEAKER>");
check(
  "current speaker survives model round trip under a custom key",
  api.fromModel(api.toModel(api.schemas.request)).children.some((child) => child.key === "speaker" && child.source === "meta_id"),
);
check(
  "selected current speaker remains available on its own node",
  api.availableSources("request", speakerNode).includes("meta_id"),
);
const duplicateNode = api.newNode("string", "other_speaker");
api.schemas.request.children.push(duplicateNode);
const duplicatePath = [`c:${api.schemas.request.children.length - 1}`];
check("current speaker is withheld from a second node", !api.availableSources("request", duplicateNode).includes("meta_id"));
api.setSource("request", duplicatePath, "meta_id");
check("duplicate current speaker selection is refused", duplicateNode.source === "static");
duplicateNode.source = "meta_id";
throws("duplicate current speaker is rejected on save", () => api.validateEditor("request"), "at most one Current Speaker");
api.schemas.request.children.pop();

api.schemas.request.children.push(api.dynamicNode("list", "speakers", "context_character_links"));
api.validateEditor("request");
api.updatePreview("request");
check("legacy links without context show a warning", el("requestWarning").textContent.includes("links will be empty"));
api.schemas.request.children.push(api.dynamicNode("list", "history", "context"));
api.updatePreview("request");
check("adding context clears the legacy warning", el("requestWarning").textContent === "");
api.schemas.request.children = api.schemas.request.children.filter((child) => child !== rosterNode);
throws("history links require a roster", () => api.validateEditor("request"), "Context Links requires a Character Info");
api.schemas.request.children = api.schemas.request.children.filter((child) => child.source !== "context_character_links");
throws("current speaker requires a roster", () => api.validateEditor("request"), "Current Speaker requires a Character Info");
api.schemas.response = api.defaultSchema("response");
api.schemas.response.children.push(api.dynamicNode("string", "speaker", "meta_id"));
throws("current speaker is request-only", () => api.validateEditor("response"), "cannot contain a Current Speaker");
api.schemas.request = api.defaultSchema("request");
api.schemas.response = api.defaultSchema("response");
api.updatePreview("request");

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
check("fresh hint explains how to reveal after saving", el("encKeyHint").innerHTML.includes("use Show to reveal"));
check("hint names the rotation cap", el("encKeyHint").innerHTML.includes("Up to 4"));
check("fernet hint names generate_key", el("encEncodingHint").innerHTML.includes("generate_key()"));

api.storedKeyPresent = true;
api.syncEncryptionFields();
check("stored-key hint explains preserving the key and type", el("encKeyHint").innerHTML.includes("Leave the box empty") && el("encKeyHint").innerHTML.includes("reveal or enter the key first"));

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

console.log("\n--- saved key Show/Hide ---");
// Let the page's initial async list fetch finish before installing fixtures.
await new Promise((resolve) => setTimeout(resolve, 0));
api.state.endpoints = ["saved", "other"].map((id) => ({
  id, name: id, hostname: "example.invalid", languages: [],
  encryption: { enabled: true, has_key: true, key_encoding: "fernet" },
}));
let revealCalls = [];
const keyResponse = (value = keys[0], encoding = "fernet") => ({
  ok: true, json: async () => ({ key: value, key_encoding: encoding }),
});
sandbox.fetch = async (url, options) => {
  revealCalls.push({ url, options });
  return keyResponse();
};
api.openEdit("saved");
check("reopen leaves saved key blank and hidden", el("encKey").value === "" && el("encKey").type === "password");
check("opening does not fetch the saved key", revealCalls.length === 0);
const revealing = api.revealKey();
await api.revealKey();
check("pending reveal disables Show and deduplicates requests", el("encReveal").disabled && revealCalls.length === 1);
await revealing;
check("Show loads saved key visibly", el("encKey").value === keys[0] && el("encKey").type === "text");
check("Show fetch targets only current endpoint without caching", revealCalls[0].url === "/api/user_endpoints/saved/key" && revealCalls[0].options.cache === "no-store");
check("loaded key can be hidden", el("encReveal").textContent === "Hide" && !el("encReveal").disabled);
check("unchanged revealed key is not resubmitted", api.collectEncryption().key === "");
await api.revealKey();
check("Hide masks the loaded key", el("encKey").type === "password" && el("encKey").value === keys[0]);
await api.revealKey();
check("repeated Show reuses this dialog's loaded key", el("encKey").type === "text" && revealCalls.length === 1);
el("encEncoding").value = "passphrase";
el("dialog").dispatch("change", { target: el("encEncoding") });
check("changing encoding submits the revealed value explicitly", api.collectEncryption().key === keys[0]);
api.closeDialog();
check("close clears the revealed value and masks input", el("encKey").value === "" && el("encKey").type === "password");
api.openEdit("saved");
await api.revealKey();
check("a reopened dialog retrieves a fresh saved key", revealCalls.length === 2);
el("dialog").dispatch("cancel");
check("Escape clears the revealed key", el("encKey").value === "" && el("encKey").type === "password");
await api.revealKey();
el("dialog").close();
check("native close also clears the revealed key", el("encKey").value === "" && el("encKey").type === "password");

api.openEdit("saved");
el("encKey").value = keys[1];
el("dialog").dispatch("input", { target: el("encKey") });
const beforeLocal = revealCalls.length;
await api.revealKey();
await api.revealKey();
check("typed replacement shows and hides without fetching", revealCalls.length === beforeLocal && el("encKey").type === "password");
check("typed replacement is submitted", api.collectEncryption().key === keys[1]);
api.fillGeneratedKey();
const generatedKey = el("encKey").value;
await api.revealKey();
await api.revealKey();
check("generated key stays local and is submitted", revealCalls.length === beforeLocal && api.collectEncryption().key === generatedKey);

for (const [label, fetchResult] of [
  ["HTTP failure", async () => ({ ok: false, json: async () => ({ error: "Key not found" }) })],
  ["network failure", async () => { throw new Error("Connection failed"); }],
  ["invalid JSON", async () => ({ ok: true, json: async () => { throw new Error("Invalid JSON"); } })],
  ["null response", async () => ({ ok: true, json: async () => null })],
]) {
  api.openEdit("saved");
  sandbox.fetch = fetchResult;
  await api.revealKey();
  check(`${label} keeps key hidden and permits retry`, el("encKey").value === "" && el("encKey").type === "password" && !el("encReveal").disabled);
  check(`${label} is shown in the dialog`, Boolean(el("dialogStatus").textContent));
}

// Resolving aborted requests anyway exercises stale-response checks even if
// a transport or test double ignores AbortSignal.
for (const [label, interrupt] of [
  ["typed replacement", () => {
    el("encKey").value = keys[1];
    el("dialog").dispatch("input", { target: el("encKey") });
  }],
  ["generated replacement", () => api.fillGeneratedKey()],
  ["encoding change", () => {
    el("encEncoding").value = "passphrase";
    el("dialog").dispatch("change", { target: el("encEncoding") });
  }],
  ["encryption disabled", () => {
    el("encEnabled").value = "false";
    el("dialog").dispatch("change", { target: el("encEnabled") });
  }],
  ["dialog close", () => api.closeDialog()],
  ["Escape", () => el("dialog").dispatch("cancel")],
  ["native close", () => el("dialog").close()],
  ["same endpoint reopened", () => { api.closeDialog(); api.openEdit("saved"); }],
  ["another endpoint opened", () => api.openEdit("other")],
]) {
  api.openEdit("saved");
  let finish;
  let signal;
  sandbox.fetch = (_url, options) => {
    signal = options.signal;
    return new Promise((resolve) => { finish = resolve; });
  };
  const pending = api.revealKey();
  interrupt();
  const expectedValue = el("encKey").value;
  const expectedEncoding = el("encEncoding").value;
  check(`${label} cancels pending reveal`, signal.aborted);
  finish(keyResponse());
  await pending;
  check(`${label} is not overwritten by the stale response`, el("encKey").value === expectedValue && el("encEncoding").value === expectedEncoding);
  check(`${label} leaves the reveal button usable`, !el("encReveal").disabled);
}

api.openEdit("saved");
const pendingResponses = [];
sandbox.fetch = () => new Promise((resolve) => pendingResponses.push(resolve));
const oldReveal = api.revealKey();
api.closeDialog();
api.openEdit("other");
const newReveal = api.revealKey();
el("dialog").dispatch("close");
check("queued old close does not cancel a reopened dialog", el("dialog").open && el("encReveal").disabled);
pendingResponses[0](keyResponse(keys[0]));
await oldReveal;
check("old request completion leaves the new reveal pending", el("encKey").value === "" && el("encReveal").disabled);
pendingResponses[1](keyResponse(keys[1]));
await newReveal;
check("only the new endpoint's key becomes visible", el("encKey").value === keys[1] && !el("encReveal").disabled);

api.openEdit("saved");
sandbox.fetch = async () => keyResponse("saved passphrase", "passphrase");
await api.revealKey();
check("revealed key uses its saved encoding", el("encKey").value === "saved passphrase" && el("encEncoding").value === "passphrase");
check("unchanged revealed passphrase remains server-managed", api.collectEncryption().key === "");
api.closeDialog();

console.log("\n--- visible page feedback ---");
const feedback = el("resultDialog");
const firstShow = feedback.modalShows;
api.status("Endpoint saved.");
check("page feedback opens a native modal", feedback.open && feedback.modalShows === firstShow + 1);
check("page feedback states the outcome", el("resultMessage").textContent === "Endpoint saved.");
el("resultBody").scrollTop = 180;
api.status("<b>Activation failed</b>", true);
check("new feedback reuses the open modal", feedback.modalShows === firstShow + 1);
check("replacement feedback scrolls its message into view", el("resultBody").scrollTop === 0);
check("error text is displayed literally", el("resultMessage").textContent === "<b>Activation failed</b>" && el("resultMessage").className.includes("error"));
const closeButton = makeEl("close-feedback", {
  closest: (selector) => selector === '[data-act="close-result"]' ? closeButton : null,
});
feedback.dispatch("click", { target: closeButton });
check("Close dismisses page feedback", !feedback.open);

api.status("An endpoint message.");
feedback.dispatch("pointerdown", { clientX: 150, clientY: 150 });
feedback.dispatch("click", { clientX: 150, clientY: 150 });
check("clicking inside the dialog keeps it open", feedback.open);
feedback.dispatch("pointerdown", { clientX: 150, clientY: 150 });
feedback.dispatch("click", { clientX: 50, clientY: 50 });
check("dragging from inside to outside does not dismiss feedback", feedback.open);
feedback.dispatch("pointerdown", { clientX: 50, clientY: 50 });
feedback.dispatch("pointercancel");
feedback.dispatch("click", { clientX: 50, clientY: 50 });
check("a cancelled pointer gesture cannot dismiss feedback", feedback.open);
feedback.dispatch("pointerdown", { clientX: 50, clientY: 50 });
feedback.dispatch("click", { clientX: 50, clientY: 50 });
check("clicking outside dismisses feedback", !feedback.open);
api.status("Escape can close this.");
let escapePrevented = false;
feedback.dispatch("cancel", { preventDefault() { escapePrevented = true; } });
check("native Escape dismissal remains enabled", !escapePrevented);
feedback.close();

console.log("\n--- Send test feedback and request lifecycle ---");
const testButtons = ["saved", "other"].map((id) => makeEl(`test-${id}`, {
  dataset: { card: "test", id },
}));
el("list").querySelectorAll = (selector) => selector === '[data-card="test"]' ? testButtons : [];
const testResponse = (body, status = 200, statusText = "OK") => ({
  ok: status >= 200 && status < 300,
  status,
  statusText,
  text: async () => typeof body === "string" ? body : JSON.stringify(body),
});
let finishTest;
let testCalls = 0;
sandbox.fetch = () => {
  testCalls += 1;
  return new Promise((resolve) => { finishTest = resolve; });
};
const pendingTest = api.testEndpoint("saved");
await api.testEndpoint("saved");
await api.testEndpoint("other");
check("duplicate and competing tests share one pending request", testCalls === 1);
check("test progress is visible and explains dismissal", feedback.open && el("resultTitle").textContent === "Testing endpoint" && el("resultMessage").textContent.includes("result will appear"));
check("test buttons are disabled while a request is pending", testButtons.every((button) => button.disabled));
check("the current endpoint's test button shows progress", testButtons[0].textContent === "Testing…");
api.renderCards();
check("rerendered cards retain the pending-test guard", (el("list").innerHTML.match(/data-card="test"[^>]* disabled/g) || []).length === 2);
api.closeFeedback();
el("resultBody").scrollTop = 350;
await api.testEndpoint("saved");
check("dismissing progress does not start another test", !feedback.open && testCalls === 1);
finishTest(testResponse({
  status: true, code: 200, sent: "こんにちは", language: "japanese",
  translated: "Hello", alternatives: ["Hi"],
  request: { body: '{"text":"actual prepared token"}' },
}));
await pendingTest;
check("a dismissed pending test still delivers its result", feedback.open && el("resultTitle").textContent === "Test succeeded");
check("reopened test results start at the visible message", el("resultBody").scrollTop === 0);
check("the result retains translation and actual request details", el("resultOut").textContent.includes("Hello") && el("resultOut").textContent.includes("actual prepared token"));
check("completed tests reenable every test button", testButtons.every((button) => !button.disabled && button.textContent === "Send test"));

for (const [label, fetchResult, expectedMessage, expectedDetail] of [
  ["structured HTTP error", async () => testResponse({ status: false, code: 404, message: "Endpoint no longer exists", error: "legacy fallback" }, 404, "Not Found"), "Endpoint no longer exists", "status:     404"],
  ["HTTP 200 endpoint failure", async () => testResponse({ status: false, code: 503, message: "Could not reach the endpoint", request: { url: "https://example.invalid/test" } }), "Could not reach the endpoint", "https://example.invalid/test"],
  ["error-only HTTP response", async () => testResponse({ error: "Service unavailable" }, 503, "Service Unavailable"), "Service unavailable", "HTTP status: 503"],
  ["non-JSON HTTP error", async () => testResponse("<html>Bad gateway</html>", 502, "Bad Gateway"), "HTTP 502 Bad Gateway", "<html>Bad gateway</html>"],
  ["non-JSON successful HTTP response", async () => testResponse("not JSON"), "unexpected response", "not JSON"],
  ["empty HTTP response", async () => testResponse("", 503, "Service Unavailable"), "empty response", "HTTP status: 503"],
  ["null JSON response", async () => testResponse(null), "unexpected response", "null"],
  ["array JSON response", async () => testResponse([]), "unexpected response", "[]"],
  ["network failure", async () => { throw new Error("Connection refused"); }, "Connection refused", ""],
  ["response read failure", async () => ({ ok: true, status: 200, text: async () => { throw new Error("Connection closed"); } }), "Connection closed", ""],
]) {
  sandbox.fetch = fetchResult;
  const showsBefore = feedback.modalShows;
  await api.testEndpoint("saved");
  check(`${label} stays in the same visible result modal`, feedback.open && feedback.modalShows === showsBefore && el("resultTitle").textContent !== "Test succeeded");
  check(`${label} shows a useful message`, el("resultMessage").textContent.includes(expectedMessage), el("resultMessage").textContent);
  check(`${label} preserves available details without undefined values`, el("resultOut").textContent.includes(expectedDetail) && !el("resultOut").textContent.includes("undefined"));
  check(`${label} allows another test`, testButtons.every((button) => !button.disabled));
}

api.status("Endpoint deleted.");
check("ordinary feedback clears test-only content", el("resultOut").textContent === "" && el("resultHint").hidden);
api.status("");
check("clearing feedback closes the modal", !feedback.open);

console.log(failures ? `\n${failures} FAILURE(S)` : "\nall editor checks passed");
process.exit(failures ? 1 : 0);
