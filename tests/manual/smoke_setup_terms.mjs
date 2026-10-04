/**
 * Run the setup wizard's own script against a stub DOM.
 *
 * The script is the one lifted by render_setup_page.py, evaluated in node:vm
 * with just enough of a browser around it. What is being checked is the thing
 * nothing else in the toolchain looks at: the per-model licence gate, and the
 * optional-models section it guards.
 *
 * The licence gate is not a decoration. /api/setup/download and
 * /api/setup/optional/download both answer 403 + `terms_required` for a model
 * whose notice is unaccepted, so this card is the only way a download starts.
 * That makes four properties worth more than the rest:
 *
 *   * accept is unreachable until the notice has been scrolled to its end
 *   * every acceptance is POSTed, per model, before the download POST
 *   * declining -- by button or by Escape -- cancels the download entirely
 *   * an acceptance the server did not confirm is not treated as an acceptance
 *
 * The stub is deliberately narrow and deliberately loud: `querySelectorAll`
 * knows only the selectors the page actually uses and throws on anything else.
 * A stub that answered an unknown selector with `[]` would turn a template edit
 * into a silently passing test, which is the specific failure this file exists
 * to prevent.
 *
 * Run render_setup_page.py first.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";

const SCRIPT = path.join(os.tmpdir(), "setup-page.js");
const PAGE = path.join(os.tmpdir(), "setup-page.html");

let failures = 0;

function check(label, condition, detail = "") {
  if (!condition) failures += 1;
  console.log(`${condition ? "ok  " : "FAIL"} ${label}${detail ? ` -- ${detail}` : ""}`);
}

/** Let the sandbox's pending promises settle. */
const tick = () => new Promise((resolve) => setImmediate(resolve));

// --- a DOM that is only as real as the script needs -------------------------

function makeClassList(el) {
  const names = new Set();

  return {
    add: (...cls) => cls.forEach((c) => names.add(c)),
    remove: (...cls) => cls.forEach((c) => names.delete(c)),
    contains: (cls) => names.has(cls),
    toggle(cls, force) {
      const on = force === undefined ? !names.has(cls) : Boolean(force);
      if (on) names.add(cls);
      else names.delete(cls);
      return on;
    },
    get length() {
      return names.size;
    },
    toString: () => [...names].join(" "),
    __names: names,
    __owner: el,
  };
}

//: The only selectors the page asks an element for, and what each one means.
//: Unknown selectors throw: see the file header.
const SELECTORS = {
  "input[type=checkbox]:not([disabled])": (el) => el.type === "checkbox" && !el.disabled,
  "input[type=checkbox]:not([disabled]):checked": (el) =>
    el.type === "checkbox" && !el.disabled && el.checked,
};

const GATE_FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])';

/** Every element the page can reach by id, whether declared or rendered. */
const byId = new Map();

/**
 * The ids the template actually declares, read off the rendered page.
 *
 * This is what lets getElementById return `null` the way a browser does. A stub
 * that invented an element for every id asked of it would hide two real bugs:
 * an `els` entry pointing at an id the template renamed, and
 * captureOptionalSelections reading a checkbox that is not on the page (which
 * guards with `if (!cb) return` precisely because the answer can be null).
 */
const DECLARED = new Set(
  [...fs.readFileSync(PAGE, "utf8").matchAll(/\bid="([^"]+)"/g)].map(([, id]) => id),
);

function makeElement(id, extra = {}) {
  const el = {
    id,
    textContent: "",
    value: "",
    type: "text",
    disabled: false,
    checked: false,
    style: {},
    attributes: {},
    //: Children parsed out of innerHTML. Only <input> is modelled: it is the
    //: only descendant the page reads back (see SELECTORS).
    children: [],
    //: What a browser would give a laid-out, visible element. focusablesIn
    //: filters on it, so `null` here means "not focusable".
    offsetParent: {},
    scrollTop: 0,
    scrollHeight: 0,
    clientHeight: 0,
    __handlers: Object.create(null),
    __focused: 0,

    get innerHTML() {
      return el.__html;
    },
    set innerHTML(html) {
      el.__html = String(html);
      // Drop the previous children from the id registry first: a re-render
      // replaces them, and a stale entry would let getElementById hand back a
      // node that is no longer on the page.
      el.children.forEach((child) => {
        if (byId.get(child.id) === child) byId.delete(child.id);
      });
      el.children = parseInputs(el.__html);
      el.children.forEach((child) => byId.set(child.id, child));
    },

    addEventListener(type, handler) {
      (el.__handlers[type] || (el.__handlers[type] = [])).push(handler);
    },
    removeEventListener(type, handler) {
      const list = el.__handlers[type] || [];
      const at = list.indexOf(handler);
      if (at !== -1) list.splice(at, 1);
    },
    setAttribute(name, value) {
      el.attributes[name] = String(value);
    },
    getAttribute: (name) => (name in el.attributes ? el.attributes[name] : null),
    focus() {
      el.__focused += 1;
      document.activeElement = el;
    },
    contains: (node) => node === el || el.children.indexOf(node) !== -1 || (el.__focusables || []).indexOf(node) !== -1,
    querySelectorAll(selector) {
      if (selector === GATE_FOCUSABLE) return el.__focusables || [];

      const match = SELECTORS[selector];
      if (!match) {
        throw new Error(
          `the stub DOM was asked for an unknown selector: ${selector}\n` +
            "  teach SELECTORS in smoke_setup_terms.mjs what it means, or fix the template",
        );
      }
      return el.children.filter(match);
    },
  };

  el.__html = "";
  el.classList = makeClassList(el);
  Object.assign(el, extra);
  return el;
}

/**
 * Lift the <input> tags out of a rendered card list.
 *
 * Quoted attribute values are stripped before the bare ones are read: the
 * markup carries `onchange="...this.checked)"` on every box, and a naive
 * /\bchecked\b/ over the whole tag reads that as a ticked checkbox.
 */
function parseInputs(html) {
  return [...html.matchAll(/<input\b([^>]*)>/g)].map(([, attrs]) => {
    const quoted = (name) => {
      const found = new RegExp(`\\b${name}="([^"]*)"`).exec(attrs);
      return found ? found[1] : "";
    };
    const bare = attrs.replace(/[\w-]+\s*=\s*"[^"]*"/g, " ");

    return makeElement(quoted("id"), {
      type: quoted("type") || "text",
      // Attribute to property once, at parse time, exactly as a browser does
      // for the *initial* state. From then on the property is what counts,
      // which is what makes toggleOptionalModel's `cb.checked = !cb.checked`
      // mean anything here.
      checked: /\bchecked\b/.test(bare),
      disabled: /\bdisabled\b/.test(bare),
    });
  });
}

const documentElement = makeElement("html");

const document = {
  documentElement,
  body: makeElement("body"),
  activeElement: null,
  getElementById(id) {
    if (byId.has(id)) return byId.get(id);
    if (!DECLARED.has(id)) return null;
    byId.set(id, makeElement(id));
    return byId.get(id);
  },
  // Only the nodes the stub knows about are "on the page". Anything else is
  // detached, which is the case closeTermsGate guards its focus restore with.
  contains: (node) => Boolean(node) && byId.get(node.id) === node,
  addEventListener(type, handler) {
    (document.__handlers[type] || (document.__handlers[type] = [])).push(handler);
  },
  __handlers: Object.create(null),
};

/** Dispatch `type` at `target`, synchronously, and report what it did. */
function fire(target, type, event = {}) {
  const record = { prevented: false, ...event, type };
  record.preventDefault = () => {
    record.prevented = true;
  };
  (target.__handlers[type] || []).forEach((handler) => handler(record));
  return record;
}

// Every request the page makes, and the replies it gets back.
const calls = [];
let routes = {};

const fetchStub = async (url, options = {}) => {
  const call = {
    url,
    method: options.method || "GET",
    body: options.body ? JSON.parse(options.body) : null,
  };
  calls.push(call);

  const route = Object.keys(routes)
    // Longest prefix wins, so `/api/setup/terms/accept` is not answered by the
    // handler for `/api/setup/terms`.
    .sort((a, b) => b.length - a.length)
    .find((prefix) => url.startsWith(prefix));

  if (!route) return { ok: false, status: 404, json: async () => ({ error: `no stub for ${url}` }) };

  const answer = (typeof routes[route] === "function" ? routes[route](call) : routes[route]) || {};
  return {
    ok: answer.ok !== false,
    status: answer.status || (answer.ok === false ? 500 : 200),
    json: async () => {
      if (answer.throws) throw new Error("not json");
      return answer.data || {};
    },
  };
};

const resizeObservers = [];

const sandbox = {
  document,
  console,
  fetch: fetchStub,
  JSON,
  Math,
  String,
  Object,
  Boolean,
  Number,
  Array,
  Set,
  Map,
  Error,
  Promise,
  RegExp,
  encodeURIComponent,
  // A no-op that still hands back an id: pollOnce re-arms itself with this, and
  // a stub that ran the callback would recurse until the stack gave out.
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  // Synchronous: openTermsGate defers the first read-state check to after
  // layout, and there is no layout here to wait for.
  requestAnimationFrame: (fn) => {
    fn();
    return 1;
  },
  // `inert` is present on the prototype, so the branch that uses it runs. That
  // is the interesting case: closeTermsGate has to *restore* it rather than
  // clear it, or closing a licence card would un-inert the wizard while the
  // copyright notice is still up.
  HTMLElement: { prototype: { inert: false } },
  ResizeObserver: class {
    constructor(callback) {
      this.callback = callback;
      resizeObservers.push(this);
    }
    observe(target) {
      this.target = target;
    }
    disconnect() {}
  },
  window: {
    location: { protocol: "http:", host: "localhost:8000" },
    localStorage: {
      // Already acknowledged: the copyright notice is a separate gate with its
      // own tests, and it must not be on screen while the licence card is.
      getItem: () => "1",
      setItem: () => {},
    },
    addEventListener(type, handler) {
      (sandbox.window.__handlers[type] || (sandbox.window.__handlers[type] = [])).push(handler);
    },
    __handlers: Object.create(null),
  },
  WebSocket: class {
    constructor() {
      this.onmessage = null;
    }
    close() {}
  },
};

sandbox.globalThis = sandbox;

const source =
  fs.readFileSync(SCRIPT, "utf8") +
  `
;globalThis.__api = {
  requireTerms, reviewTerms, renderTermsDocument, updateTermsReadState,
  renderOptional, downloadOptional, toggleOptionalModel, handleOptionalCheckboxChange,
  initCopyrightGate, initTermsGate, settleTerms, onGateKeydown,
  OPTIONAL_BLURB,
  isTermsOpen: () => termsOpen,
  setGateMode: (mode) => { gateMode = mode; },
  el: (id) => document.getElementById(id),
};`;

vm.runInContext(source, vm.createContext(sandbox), { filename: "setup-page.js" });

const api = sandbox.__api;

// The keydown listener lives on `document` and is wired here, so Escape is
// tested through the same path a real key press takes.
api.initCopyrightGate();
api.initTermsGate();

const els = {
  gate: api.el("termsGate"),
  card: api.el("termsCard"),
  step: api.el("termsStep"),
  name: api.el("termsModelName"),
  licence: api.el("termsLicence"),
  body: api.el("termsBody"),
  more: api.el("termsMore"),
  error: api.el("termsError"),
  chk: api.el("termsAcceptChk"),
  acceptLabel: api.el("termsAcceptLabel"),
  accept: api.el("termsAcceptBtn"),
  decline: api.el("termsDeclineBtn"),
  shell: api.el("setupShell"),
  optionalSection: api.el("optionalSection"),
  optionalModels: api.el("optionalModels"),
  optionalBtn: api.el("optionalBtn"),
  errorMsg: api.el("errorMsg"),
};

// What the focus trap has to cycle between.
els.gate.__focusables = [els.decline, els.accept];

check(
  "every element the gate needs is in the rendered page",
  Object.entries(els).every(([, el]) => el && typeof el.focus === "function"),
);

// --- the notices the stub server serves -------------------------------------

const BUBBLE = {
  id: "bubble-segmentation",
  name: "Bubble Segmentation",
  licence: "Apache 2.0 (weights) · Manga109 (training data)",
  lede: "Read this before downloading.",
  sections: [
    {
      heading: "Training data",
      paragraphs: ["Trained on MS92/MangaSegmentation and Manga109."],
      bullets: ["Redistribution of any part of the dataset is forbidden.", "Academic purposes only."],
    },
  ],
  links: [
    { label: "Manga109", url: "https://manga109.github.io/manga109-project-website/en/index.html" },
    { label: "Not a link", url: "javascript:alert(1)" },
  ],
  accept_label: "I accept the Manga109 terms for this model.",
  version: "a1b2c3d4",
};

const TEXTSEG = {
  id: "text-segmentation",
  name: "Text Segmentation",
  licence: "MIT (research code) · Manga109 (training data)",
  lede: "Read this too.",
  sections: [{ heading: "Code", paragraphs: ["juvian/Manga-Text-Segmentation is MIT."] }],
  links: [],
  version: "e5f6a7b8",
};

const DOCS = { [BUBBLE.id]: BUBBLE, [TEXTSEG.id]: TEXTSEG };

/** Install the stub routes. `accepted` lists ids the server already has. */
function serve({ accepted = [], acceptReply, statusOk = false } = {}) {
  const accept = new Set(accepted);

  routes = {
    "/api/setup/status": { ok: statusOk, data: {} },
    "/api/setup/terms/accept": (call) => {
      if (acceptReply) return acceptReply(call);
      call.body.ids.forEach((id) => accept.add(id));
      return { data: { accepted: call.body.ids } };
    },
    "/api/setup/terms": (call) => {
      const id = /[?&]id=([^&]+)/.exec(call.url);
      if (id) {
        const doc = DOCS[decodeURIComponent(id[1])];
        return doc ? { data: { document: doc } } : { ok: false, data: { error: "unknown model" } };
      }
      const state = {};
      accept.forEach((name) => {
        state[name] = { accepted: true, version: (DOCS[name] || {}).version };
      });
      return { data: { terms: state } };
    },
    "/api/setup/optional/download": { data: { ok: true } },
  };

  return accept;
}

/** The scroll geometry of a notice that does not fit its pane. */
function makeScrollable() {
  els.body.scrollHeight = 1200;
  els.body.clientHeight = 400;
  els.body.scrollTop = 0;
}

function scrollToEnd() {
  els.body.scrollTop = els.body.scrollHeight - els.body.clientHeight;
  fire(els.body, "scroll");
}

/** Tick the box the way a user does, through the change listener. */
function tickAcceptBox() {
  els.chk.checked = true;
  fire(els.chk, "change");
}

const postsTo = (url) => calls.filter((call) => call.url === url && call.method === "POST");

function reset() {
  calls.length = 0;
  els.body.scrollTop = 0;
  els.chk.checked = false;
  els.card.__focused = 0;
  document.activeElement = null;
  showErrors();
}

function showErrors() {
  els.errorMsg.textContent = "";
}

// ---------------------------------------------------------------------------
console.log("\n--- nothing left to ask ---");

serve({ accepted: [BUBBLE.id, TEXTSEG.id] });
reset();

let result = await api.requireTerms([BUBBLE.id, TEXTSEG.id], els.optionalBtn);

check("an all-accepted queue resolves true", result === true);
check("the card never opens", api.isTermsOpen() === false);
check("and nothing is written down", postsTo("/api/setup/terms/accept").length === 0);
check(
  "the state is read once, and no notice is fetched",
  calls.length === 1 && calls[0].url.startsWith("/api/setup/terms") && !calls[0].url.includes("id="),
  JSON.stringify(calls.map((c) => c.url)),
);

// ---------------------------------------------------------------------------
console.log("\n--- one unaccepted notice ---");

serve({ accepted: [TEXTSEG.id] });
reset();
makeScrollable();

const trigger = els.optionalBtn;
let pending = api.requireTerms([TEXTSEG.id, BUBBLE.id], trigger);
await tick();

check("the card opens", api.isTermsOpen() === true);
check("the page behind it is inert", els.shell.inert === true);
check("and scroll-locked", documentElement.classList.contains("fox-terms-open"));
check("focus lands on the card, not the accept button", els.card.__focused === 1);
check(
  "only the unaccepted notice is fetched",
  calls.filter((call) => call.url.includes("id=")).length === 1,
  JSON.stringify(calls.map((c) => c.url)),
);
check("the step label is bare for a single notice", els.step.textContent === "Licence", els.step.textContent);
check("the model is named", els.name.textContent === "Bubble Segmentation", els.name.textContent);
check("the licence is named", els.licence.textContent.includes("Apache 2.0"), els.licence.textContent);
check("its own accept wording is used", els.acceptLabel.textContent === BUBBLE.accept_label, els.acceptLabel.textContent);

let html = els.body.innerHTML;
check("the lede is shown", html.includes("Read this before downloading."));
check("the heading is shown", html.includes("<h3>Training data</h3>"));
check("the paragraphs are shown", html.includes("MS92/MangaSegmentation"));
check("every bullet is shown", (html.match(/<li>/g) || []).length === 3, html.match(/<li>/g)?.length + " <li>");
check("the http link survives", html.includes('href="https://manga109.github.io/manga109-project-website/en/index.html"'));
check("and is safe to open", html.includes('rel="noopener noreferrer"'));
check("the javascript: link is dropped", !html.includes("javascript:"));

check("accept is unreachable", els.accept.disabled === true);
check("the box cannot even be ticked", els.chk.disabled === true);
check('"keep reading" is visible', !els.more.classList.contains("hidden"));
check("and announced", els.more.getAttribute("aria-hidden") === "false");

// A tick the user cannot reach must not be honoured.
els.chk.checked = true;
fire(els.chk, "change");
check("ticking it early is undone", els.chk.checked === false);
check("and accept stays unreachable", els.accept.disabled === true);

scrollToEnd();
check("reaching the end unlocks the box", els.chk.disabled === false);
check('and hides "keep reading"', els.more.classList.contains("hidden"));
check("but accept still needs the tick", els.accept.disabled === true);

els.body.scrollTop = 0;
fire(els.body, "scroll");
check("scrolling back up does not re-lock it", els.chk.disabled === false);

tickAcceptBox();
check("tick plus end-of-notice enables accept", els.accept.disabled === false);

fire(els.accept, "click");
await tick();
result = await pending;

check("the queue resolves true", result === true);
let posts = postsTo("/api/setup/terms/accept");
check("exactly one acceptance is written down", posts.length === 1, String(posts.length));
check(
  "and it names the one model",
  posts.length === 1 && JSON.stringify(posts[0].body) === JSON.stringify({ ids: [BUBBLE.id] }),
  JSON.stringify(posts[0]?.body),
);
check("the card closes", api.isTermsOpen() === false);
check("the scroll lock is released", !documentElement.classList.contains("fox-terms-open"));
check("the page is interactive again", els.shell.inert === false);
check("and focus goes back to what opened it", document.activeElement === trigger);

// ---------------------------------------------------------------------------
console.log("\n--- two notices in a row ---");

serve();
reset();
makeScrollable();

pending = api.requireTerms([BUBBLE.id, TEXTSEG.id], els.optionalBtn);
await tick();

check("the first notice is numbered", els.step.textContent === "Licence 1 of 2", els.step.textContent);
check("and it is the first id asked for", els.name.textContent === "Bubble Segmentation", els.name.textContent);

scrollToEnd();
tickAcceptBox();
fire(els.accept, "click");
await tick();

check("the card stays open between notices", api.isTermsOpen() === true);
check("the second notice is numbered", els.step.textContent === "Licence 2 of 2", els.step.textContent);
check("and it is the second model", els.name.textContent === "Text Segmentation", els.name.textContent);
check("the new notice has to be read too", els.accept.disabled === true);
check("its tick is cleared", els.chk.checked === false);
check("and its scroll position is reset", els.body.scrollTop === 0);

scrollToEnd();
tickAcceptBox();
fire(els.accept, "click");
await tick();
result = await pending;

check("the queue resolves true", result === true);
posts = postsTo("/api/setup/terms/accept");
check(
  "one acceptance per model, in card order",
  JSON.stringify(posts.map((call) => call.body.ids)) === JSON.stringify([[BUBBLE.id], [TEXTSEG.id]]),
  JSON.stringify(posts.map((call) => call.body.ids)),
);
check("and the card is closed now", api.isTermsOpen() === false);

// ---------------------------------------------------------------------------
console.log("\n--- declining ---");

serve();
reset();
makeScrollable();

pending = api.requireTerms([BUBBLE.id, TEXTSEG.id], els.optionalBtn);
await tick();

scrollToEnd();
tickAcceptBox();
fire(els.accept, "click");
await tick();
fire(els.decline, "click");
await tick();
result = await pending;

check("declining resolves false", result === false);
check("the card closes", api.isTermsOpen() === false);
check("the page is interactive again", els.shell.inert === false);
posts = postsTo("/api/setup/terms/accept");
check(
  "the acceptance already given is kept",
  JSON.stringify(posts.map((call) => call.body.ids)) === JSON.stringify([[BUBBLE.id]]),
  JSON.stringify(posts.map((call) => call.body.ids)),
);

console.log("\n--- Escape is a decline ---");

serve();
reset();
makeScrollable();

pending = api.requireTerms([BUBBLE.id], els.optionalBtn);
await tick();
scrollToEnd();
tickAcceptBox();

let event = fire(document, "keydown", { key: "Escape" });
await tick();
result = await pending;

check("Escape resolves false, even with accept enabled", result === false);
check("and the browser does not also act on the key", event.prevented === true);
check("nothing is written down", postsTo("/api/setup/terms/accept").length === 0);

console.log("\n--- Tab stays inside the card ---");

serve();
reset();
makeScrollable();

pending = api.requireTerms([BUBBLE.id], els.optionalBtn);
await tick();

document.activeElement = els.accept;
event = fire(document, "keydown", { key: "Tab", shiftKey: false });
check("Tab off the last control wraps to the first", document.activeElement === els.decline);
check("and the default is suppressed", event.prevented === true);

document.activeElement = els.decline;
event = fire(document, "keydown", { key: "Tab", shiftKey: true });
check("Shift+Tab off the first wraps to the last", document.activeElement === els.accept);
check("and the default is suppressed", event.prevented === true);

fire(document, "keydown", { key: "Escape" });
await tick();
await pending;

// ---------------------------------------------------------------------------
console.log("\n--- an acceptance the server does not confirm ---");

let acceptCalls = 0;

serve({
  acceptReply: (call) => {
    acceptCalls += 1;
    // First a 500, then an OK that echoes nothing back, then the truth.
    if (acceptCalls === 1) return { ok: false, data: { error: "disk is full" } };
    if (acceptCalls === 2) return { data: { accepted: [] } };
    return { data: { accepted: call.body.ids } };
  },
});
reset();
makeScrollable();

pending = api.requireTerms([BUBBLE.id], els.optionalBtn);
await tick();
scrollToEnd();
tickAcceptBox();

fire(els.accept, "click");
await tick();

check("a failed save keeps the card open", api.isTermsOpen() === true);
check("and says so", els.error.textContent.includes("disk is full"), els.error.textContent);
check("the error is visible", !els.error.classList.contains("hidden"));
check("the notice has not advanced", els.name.textContent === "Bubble Segmentation");

fire(els.accept, "click");
await tick();

check("an unconfirmed save is also a failure", api.isTermsOpen() === true);
check(
  "and is not reported as success",
  els.error.textContent.includes("did not record"),
  els.error.textContent,
);

fire(els.accept, "click");
await tick();
result = await pending;

check("the retry gets through", result === true);
check("the card closes", api.isTermsOpen() === false);
check("three attempts, three POSTs", postsTo("/api/setup/terms/accept").length === 3);

// ---------------------------------------------------------------------------
console.log("\n--- a notice too short to scroll ---");

serve();
reset();
els.body.scrollHeight = 300;
els.body.clientHeight = 400;
els.body.scrollTop = 0;

pending = api.requireTerms([TEXTSEG.id], els.optionalBtn);
await tick();

check("the box unlocks without a scroll event", els.chk.disabled === false);
check('and "keep reading" is hidden', els.more.classList.contains("hidden"));
check("accept still waits for the tick", els.accept.disabled === true);

tickAcceptBox();
fire(els.accept, "click");
await tick();
await pending;

console.log("\n--- a reflow that removes the scrollbar ---");

serve();
reset();
makeScrollable();

pending = api.requireTerms([TEXTSEG.id], els.optionalBtn);
await tick();

check("locked while it overflows", els.chk.disabled === true);

// A webfont lands and the text re-wraps shorter: the end is now reachable
// without scrolling, and no scroll event will ever fire again.
els.body.scrollHeight = 350;
check("still locked until something re-measures", els.chk.disabled === true);

check("a ResizeObserver is watching the notice", resizeObservers.length === 1, String(resizeObservers.length));
check("and it is watching the right element", resizeObservers[0].target === els.body);
resizeObservers[0].callback();
check("a reflow unlocks it", els.chk.disabled === false);

fire(document, "keydown", { key: "Escape" });
await tick();
await pending;

serve();
reset();
makeScrollable();
pending = api.requireTerms([TEXTSEG.id], els.optionalBtn);
await tick();
els.body.scrollHeight = 350;
fire(sandbox.window, "resize");
check("so does a window resize", els.chk.disabled === false);
fire(document, "keydown", { key: "Escape" });
await tick();
await pending;

// ---------------------------------------------------------------------------
console.log("\n--- hostile notice text ---");

const NASTY = {
  id: "text-segmentation",
  name: '<script>alert("name")</script>',
  licence: '<img src=x onerror=alert(1)>',
  lede: "5 > 3 && 2 < 4",
  sections: [{ heading: "<b>bold</b>", paragraphs: ['"quoted"'], bullets: ["<i>italic</i>"] }],
  links: [{ label: "<em>label</em>", url: "https://example.com/?a=1&b=2" }],
};

serve();
reset();
els.body.scrollHeight = 300;
els.body.clientHeight = 400;
api.renderTermsDocument(NASTY, 1, 1);
html = els.body.innerHTML;

check("a script tag in the name is text, not markup", els.name.textContent === NASTY.name);
check("the body escapes <", !html.includes("<b>bold</b>") && html.includes("&lt;b&gt;bold&lt;/b&gt;"));
check("and quotes", html.includes("&quot;quoted&quot;"));
check("and ampersands", html.includes("5 &gt; 3 &amp;&amp; 2 &lt; 4"));
check("a link label is escaped too", html.includes("&lt;em&gt;label&lt;/em&gt;"));
check("but its href is left usable", html.includes('href="https://example.com/?a=1&amp;b=2"'));

// ---------------------------------------------------------------------------
console.log("\n--- the optional models section ---");

function optionalBlock({ active = false, downloaded = [], accepted = [], current = null } = {}) {
  const done = new Set(downloaded);
  const ok = new Set(accepted);

  return {
    active,
    current_model: current,
    progress: {},
    models: [BUBBLE, TEXTSEG].map((doc) => ({
      id: doc.id,
      name: doc.name,
      type: "segmentation",
      size_on_disk: "Not specified",
      downloaded: done.has(doc.id),
      terms_accepted: ok.has(doc.id),
    })),
  };
}

serve();
reset();
api.renderOptional(optionalBlock());

let boxes = els.optionalModels.children;
check("the stub found both checkboxes", boxes.length === 2, String(boxes.length));
check("a card per model", (els.optionalModels.innerHTML.match(/class="model /g) || []).length === 2);
check("both are ticked by default", boxes.every((box) => box.checked));
check("neither is locked", boxes.every((box) => !box.disabled));
check("the section is offered", !els.optionalSection.classList.contains("hidden"));
check("the button is offered", !els.optionalBtn.classList.contains("hidden"));
check("the button counts them", els.optionalBtn.textContent === "Download 2 Optional Models", els.optionalBtn.textContent);
check("and is live", els.optionalBtn.disabled === false);
check("each says what it adds", els.optionalModels.innerHTML.includes(api.OPTIONAL_BLURB[BUBBLE.id]));
check(
  "an unaccepted licence offers the notice",
  (els.optionalModels.innerHTML.match(/class="terms-peek"/g) || []).length === 2,
);

api.toggleOptionalModel(TEXTSEG.id);
check("clicking a card unticks it", boxes[1].checked === false);
check("the count follows", els.optionalBtn.textContent === "Download 1 Optional Model", els.optionalBtn.textContent);

// The status poll lands a second later and re-renders everything.
api.renderOptional(optionalBlock());
boxes = els.optionalModels.children;
check("a poll does not re-tick what the user cleared", boxes[1].checked === false);
check("nor untick what they kept", boxes[0].checked === true);
check("and the count is still right", els.optionalBtn.textContent === "Download 1 Optional Model", els.optionalBtn.textContent);

api.toggleOptionalModel(TEXTSEG.id);
check("it can be re-ticked", boxes[1].checked === true);

api.renderOptional(optionalBlock({ downloaded: [BUBBLE.id], accepted: [BUBBLE.id] }));
boxes = els.optionalModels.children;
check("a model already on disk is shown as ticked", boxes[0].checked === true);
check("and locked", boxes[0].disabled === true);
check("so it is not counted", els.optionalBtn.textContent === "Download 1 Optional Model", els.optionalBtn.textContent);
check(
  "an accepted licence is reported, not re-offered",
  (els.optionalModels.innerHTML.match(/class="terms-peek"/g) || []).length === 1,
);

api.renderOptional(optionalBlock({ active: true, current: TEXTSEG.id }));
boxes = els.optionalModels.children;
check("a running job locks every box", boxes.every((box) => box.disabled));
check("and the button", els.optionalBtn.disabled === true);
check("which says what it is doing", els.optionalBtn.textContent === "Downloading…", els.optionalBtn.textContent);

api.renderOptional({ models: [] });
check("no optional models means no section", els.optionalSection.classList.contains("hidden"));

// ---------------------------------------------------------------------------
console.log("\n--- a download goes through the licences ---");

serve();
reset();
makeScrollable();
api.renderOptional(optionalBlock());

// Set the boxes through the real handlers rather than assuming what the last
// render left behind: `selectedOptionalIds` is page-lifetime state, and a test
// that inherited it would be asserting on the previous section's leftovers.
boxes = els.optionalModels.children;
boxes.forEach((box) => {
  if (box.checked !== (box.id === `opt-${BUBBLE.id}`)) {
    api.toggleOptionalModel(box.id.slice("opt-".length));
  }
});

check("one model is ticked", boxes.filter((box) => box.checked).length === 1);
check("and it is the bubble model", boxes[0].checked === true && boxes[1].checked === false);
check("so the button is live", els.optionalBtn.disabled === false, els.optionalBtn.textContent);
calls.length = 0;

let download = api.downloadOptional();
await tick();

check("the licence card comes up first", api.isTermsOpen() === true);
check("no download has started", postsTo("/api/setup/optional/download").length === 0);
check("and only the ticked model's notice is shown", els.name.textContent === "Bubble Segmentation", els.name.textContent);

scrollToEnd();
tickAcceptBox();
fire(els.accept, "click");
await tick();
await download;

posts = postsTo("/api/setup/optional/download");
check("the download starts once", posts.length === 1, String(posts.length));
check(
  "with exactly the ticked ids",
  posts.length === 1 && JSON.stringify(posts[0].body) === JSON.stringify({ ids: [BUBBLE.id] }),
  JSON.stringify(posts[0]?.body),
);
check(
  "and only after the acceptance was written down",
  calls.findIndex((c) => c.url === "/api/setup/terms/accept") <
    calls.findIndex((c) => c.url === "/api/setup/optional/download"),
);

console.log("\n--- declining cancels the download ---");

serve();
reset();
makeScrollable();
api.renderOptional(optionalBlock());
calls.length = 0;

download = api.downloadOptional();
await tick();
check("the licence card is up", api.isTermsOpen() === true);
fire(els.decline, "click");
await tick();
await download;

check("nothing is downloaded", postsTo("/api/setup/optional/download").length === 0);
check("nothing is accepted", postsTo("/api/setup/terms/accept").length === 0);
check("the card is closed", api.isTermsOpen() === false);

// ---------------------------------------------------------------------------
console.log("\n--- the copyright notice behind the card ---");

serve();
reset();
makeScrollable();
api.setGateMode("ack");

pending = api.requireTerms([BUBBLE.id], els.optionalBtn);
await tick();
check("the wizard is inert for both gates", els.shell.inert === true);

fire(document, "keydown", { key: "Escape" });
await tick();
await pending;

check(
  "closing the licence card leaves the wizard inert for the notice still up",
  els.shell.inert === true,
);
api.setGateMode(null);

// ---------------------------------------------------------------------------
console.log("\n--- the notice text is fetched once per model ---");

serve({ accepted: [] });
reset();
makeScrollable();

pending = api.requireTerms([BUBBLE.id], els.optionalBtn);
await tick();
check(
  "a notice seen earlier is served from the page's own cache",
  calls.filter((call) => call.url.includes("id=")).length === 0,
  JSON.stringify(calls.map((c) => c.url)),
);
fire(document, "keydown", { key: "Escape" });
await tick();
await pending;

// ---------------------------------------------------------------------------
console.log("\n--- the server cannot be reached ---");

reset();
routes = {};

result = await api.requireTerms([BUBBLE.id], els.optionalBtn);
check("an unreadable licence state is not an acceptance", result === false);
check("and the card does not open half-rendered", api.isTermsOpen() === false);
check("the page says something", els.errorMsg.textContent.length > 0, els.errorMsg.textContent);

reset();
// An id this page has never fetched, so the notice cache cannot answer for it.
routes = { "/api/setup/terms": { data: { terms: {} } } };

result = await api.requireTerms(["paddleocr-vl-1.6"], els.optionalBtn);
check("a notice that will not load is not an acceptance", result === false);
check("and the card stays shut", api.isTermsOpen() === false);
check("the page says something", els.errorMsg.textContent.length > 0, els.errorMsg.textContent);

console.log(failures ? `\n${failures} FAILURE(S)` : "\nthe licence gate holds and the optional section behaves");
process.exit(failures ? 1 : 0);
