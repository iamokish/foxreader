/**
 * Run the settings page's own script against a stub DOM.
 *
 * The script is the one lifted by render_settings_page.py, evaluated in node:vm
 * with just enough of a browser around it. What is being checked is the Compute
 * Devices section: that each slot gets a picker with the saved option selected,
 * that the hint text distinguishes what was picked from what it resolved to from
 * what the process is running, and that a save posts one slot and re-renders.
 *
 * The thread picker sits in the same grid but is not a slot -- it is a count, it
 * needs no restart, and it must show the saved number even when this machine is
 * too narrow to honour it. Hence the extra row and select every count here
 * allows for.
 *
 * Run render_settings_page.py first.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";

const SCRIPT = path.join(os.tmpdir(), "settings-page.js");

let failures = 0;

function check(label, condition, detail = "") {
  if (!condition) failures += 1;
  console.log(`${condition ? "ok  " : "FAIL"} ${label}${detail ? ` -- ${detail}` : ""}`);
}

// --- a DOM that is only as real as the script needs -------------------------

function makeElement(id) {
  return {
    id,
    innerHTML: "",
    textContent: "",
    className: "",
    value: "",
    type: "text",
    disabled: false,
    style: {},
    addEventListener() {},
  };
}

const elements = new Map();

const document = {
  title: "Settings",
  body: makeElement("body"),
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  },
  addEventListener() {},
};

// Every request the page makes, and the reply it gets back.
const calls = [];
let reply = null;

const fetchStub = async (url, options = {}) => {
  calls.push({
    url,
    method: options.method || "GET",
    body: options.body ? JSON.parse(options.body) : null,
  });

  const answer = typeof reply === "function" ? reply(url, options) : reply;

  return {
    ok: answer.ok !== false,
    json: async () => answer.data,
  };
};

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
  Error,
  Promise,
  setInterval: () => 0,
  clearInterval: () => {},
  setTimeout: (fn) => { fn(); return 0; },
  window: {
    location: { protocol: "http:", host: "localhost:8000" },
    addEventListener() {},
    stop() {},
    close() {},
  },
  WebSocket: class { constructor() { this.onmessage = null; } close() {} },
  BroadcastChannel: class {
    constructor() { this.onmessage = null; }
    postMessage() {}
    close() {}
  },
};

sandbox.globalThis = sandbox;

// A payload shaped exactly like _build_devices_payload's output.
function payload({ selection, resolved, active, running, fallbacks = [], restart = false, gpus = 2, threads } = {}) {
  const options = [
    { id: "auto", kind: "auto", name: "Automatic", label: "Auto — RTX 4090" },
    { id: "cpu", kind: "cpu", name: "CPU", label: "CPU" },
  ];

  if (gpus >= 1) options.push({ id: "cuda:0", kind: "cuda", index: 0, name: "RTX 4060", label: "GPU 0 — RTX 4060 · 8.0 GB" });
  if (gpus >= 2) options.push({ id: "cuda:1", kind: "cuda", index: 1, name: "RTX 4090", label: "GPU 1 — RTX 4090 · 24.0 GB" });

  return {
    languages: {},
    defaults: {},
    api_tokens: {},
    devices: {
      options,
      slots: ["paddleocr", "bubble", "translator"],
      selection: selection || { paddleocr: "auto", bubble: "auto", translator: "auto" },
      resolved: resolved || { paddleocr: "cuda:1", bubble: "cuda:1", translator: "cuda:1" },
      active: active || { paddleocr: "gpu:1", bubble: "cuda:1", translator: "cuda:1" },
      running: running || resolved || { paddleocr: "cuda:1", bubble: "cuda:1", translator: "cuda:1" },
      fallbacks,
      backend: gpus ? "cuda" : "cpu",
      cuda_available: gpus > 0,
      cuda_version: gpus ? "12.6" : null,
      gpu_name: gpus ? "RTX 4090" : null,
      device_count: gpus,
      restart_required: restart,
      // Not a slot, but it is drawn in the same grid, so every count below has
      // to allow for one more row and one more select.
      mtl_threads: threads === undefined
        ? { value: "auto", max: 8, auto: 6, resolved: 6, override: null }
        : threads,
    },
  };
}

//: The thread select's own options: Auto, plus one per physical core.
const THREAD_OPTIONS = 1 + 8;

// The bootstrap line at the end runs load() against the stub fetch; give it
// something to load before evaluating.
reply = { data: payload() };

const source = fs.readFileSync(SCRIPT, "utf8") + `
;globalThis.__api = {
  renderDevices, chooseDevice, chooseThreads, deviceName, load, show,
  SLOT_LABELS, SLOT_HINTS, THREADS_LABEL,
  el: (id) => document.getElementById(id),
};`;

const context = vm.createContext(sandbox);
vm.runInContext(source, context, { filename: "settings-page.js" });

const api = sandbox.__api;
const devicesEl = api.el("devices");

const optionsOf = (html) => [...html.matchAll(/<option value="([^"]*)"([^>]*)>/g)]
  .map(([, id, rest]) => ({ id, selected: rest.includes("selected") }));

console.log("--- every slot gets a picker ---");
api.renderDevices(payload());
let html = devicesEl.innerHTML;

check("three slot rows, plus the thread row", (html.match(/class="device-row"/g) || []).length === 4);
check("and a select for each", (html.match(/class="device-select"/g) || []).length === 4);
check("each names its slot", ["PaddleOCR", "Bubble detection", "Local translation"].every(name => html.includes(name)));
check("options carry the labels", html.includes("GPU 1 — RTX 4090 · 24.0 GB"));
check("auto is selected in each", optionsOf(html).filter(o => o.selected).every(o => o.id === "auto"));
check(
  "four options per slot select, and the thread select's own",
  optionsOf(html).length === 12 + THREAD_OPTIONS,
  String(optionsOf(html).length),
);
check("the summary shows the backend", html.includes("Backend <b>cuda</b>"));
check("and the card count", html.includes("GPUs <b>2</b>"));
check("and the fastest card", html.includes("Fastest <b>RTX 4090</b>"));
check("and the cuda version", html.includes("CUDA <b>12.6</b>"));
check("auto says what it resolved to", html.includes("Currently RTX 4090."), html.match(/Currently [^<]*/)?.[0] || "");
check("no restart note yet", !html.includes("Restart Fox Reader"));
check("gpu:1 vs cuda:1 is not read as a pending restart", !html.includes("until restart"), html.match(/Running on [^<]*/)?.[0] || "");
check("no fallback note", !html.includes("Falling back"));

console.log("\n--- a pinned slot ---");
api.renderDevices(payload({
  selection: { paddleocr: "cuda:0", bubble: "auto", translator: "cpu" },
  resolved: { paddleocr: "cuda:0", bubble: "cuda:1", translator: "cpu" },
}));
html = devicesEl.innerHTML;

const selected = optionsOf(html).filter(o => o.selected).map(o => o.id);
check(
  "each select shows what was saved, the thread one included",
  JSON.stringify(selected) === '["cuda:0","auto","cpu","auto"]',
  JSON.stringify(selected),
);
check("a pinned slot gets no 'currently' line", (html.match(/Currently/g) || []).length === 1);

console.log("\n--- a card that is gone ---");
api.renderDevices(payload({
  selection: { paddleocr: "cuda:5", bubble: "auto", translator: "auto" },
  resolved: { paddleocr: "cpu", bubble: "cuda:1", translator: "cuda:1" },
  active: { paddleocr: "cpu", bubble: "cuda:1", translator: "cuda:1" },
  fallbacks: ["paddleocr: cuda:5 is not available on this machine, using the CPU"],
}));
html = devicesEl.innerHTML;

check("the row explains the fallback", html.includes("cuda:5 is not available — using the CPU"));
check("and the note lists it", html.includes("Falling back to the CPU for:"));
check("with the backend's own wording", html.includes("is not available on this machine"));
check("the missing id is still the selected option", optionsOf(html).some(o => o.selected && o.id === "cuda:5") === false, "cuda:5 is not in the option list, so nothing is selected");

console.log("\n--- saved but not restarted into ---");
api.renderDevices(payload({
  selection: { paddleocr: "auto", bubble: "cuda:0", translator: "auto" },
  resolved: { paddleocr: "cuda:1", bubble: "cuda:0", translator: "cuda:1" },
  active: { paddleocr: "gpu:1", bubble: "cuda:1", translator: "cuda:1" },
  running: { paddleocr: "cuda:1", bubble: "cuda:1", translator: "cuda:1" },
  restart: true,
}));
html = devicesEl.innerHTML;

check("the restart note appears", html.includes("Restart Fox Reader to use the new devices"));
check("and the row says what is running", html.includes("Running on RTX 4090 until restart."), html.match(/Running on [^<]*/)?.[0] || "");
check("only the changed slot says so", (html.match(/until restart/g) || []).length === 1);

console.log("\n--- a machine with no gpu ---");
api.renderDevices(payload({
  selection: { paddleocr: "auto", bubble: "auto", translator: "auto" },
  resolved: { paddleocr: "cpu", bubble: "cpu", translator: "cpu" },
  active: { paddleocr: "cpu", bubble: "cpu", translator: "cpu" },
  gpus: 0,
}));
html = devicesEl.innerHTML;

check("only auto and cpu are offered", optionsOf(html).length === 6 + THREAD_OPTIONS, String(optionsOf(html).length));
check("the backend says cpu", html.includes("Backend <b>cpu</b>"));
check("no card is named", !html.includes("Fastest"));
check("no cuda version", !html.includes("CUDA <b>"));
check("auto explains itself as the cpu", html.includes("Currently CPU."));

console.log("\n--- the thread row ---");
api.renderDevices(payload());
html = devicesEl.innerHTML;

const threadOptions = () => optionsOf(html).slice(12);
// Details only: a regex over the whole section finds a device row's wording
// first, so a failure here would print something from the wrong row.
const threadRowHtml = () => html.slice(html.indexOf(api.THREADS_LABEL));
const inThreadRow = (pattern) => threadRowHtml().match(pattern)?.[0] || "";

check("it is labelled", html.includes(api.THREADS_LABEL), api.THREADS_LABEL);
check("auto says what it would decide", html.includes("Auto — 6 threads"), inThreadRow(/Auto — [^<]*/));
check("one option per physical core", threadOptions().length === THREAD_OPTIONS, String(threadOptions().length));
check(
  "counted from one, not zero",
  JSON.stringify(threadOptions().map(o => o.id)) === '["auto","1","2","3","4","5","6","7","8"]',
  JSON.stringify(threadOptions().map(o => o.id)),
);
check("the singular is respected", html.includes(">1 thread<"), inThreadRow(/>1 thread[s]?</));
check("auto is what is selected", threadOptions().filter(o => o.selected).map(o => o.id).join() === "auto");
check("the hint names the ceiling", html.includes("out of 8 physical cores"), inThreadRow(/out of [^<]*/));
check("and says when it takes effect", html.includes("Applies the next time a model loads."));
check("it does not claim a restart is needed", !html.includes("Restart Fox Reader"));
check("it saves through its own handler", html.includes("chooseThreads(this.value)"));

console.log("\n--- a pinned thread count ---");
api.renderDevices(payload({ threads: { value: 3, max: 8, auto: 6, resolved: 3, override: null } }));
html = devicesEl.innerHTML;

check("the saved count is selected", threadOptions().filter(o => o.selected).map(o => o.id).join() === "3",
  JSON.stringify(threadOptions().filter(o => o.selected)));
check("auto is still offered, with its own figure", html.includes("Auto — 6 threads"));
check("and nothing is said about clamping", !html.includes("can run here"));

console.log("\n--- a count this machine cannot honour ---");
api.renderDevices(payload({ threads: { value: 12, max: 8, auto: 6, resolved: 8, override: null } }));
html = devicesEl.innerHTML;

check("the saved count is offered even though it is off the scale", threadOptions().some(o => o.id === "12"),
  JSON.stringify(threadOptions().map(o => o.id)));
check("so the select is not silently showing auto instead", threadOptions().filter(o => o.selected).map(o => o.id).join() === "12",
  JSON.stringify(threadOptions().filter(o => o.selected)));
check("in order", JSON.stringify(threadOptions().map(o => o.id)).endsWith('"8","12"]'), JSON.stringify(threadOptions().map(o => o.id)));
check("and it says the machine is narrower", html.includes("more than the 8 cores here"), inThreadRow(/That is [^<]*/));
check("and what will really run", html.includes("Only 8 can run here."), inThreadRow(/Only [^<]*/));

console.log("\n--- an environment override ---");
api.renderDevices(payload({ threads: { value: 3, max: 8, auto: 6, resolved: 3, override: 12 } }));
html = devicesEl.innerHTML;

check("the page says the variable is winning", html.includes("FOX_READER_MTL_THREADS=12 is overriding this."),
  inThreadRow(/FOX_READER[^<]*/));
check("but still shows what is saved", threadOptions().filter(o => o.selected).map(o => o.id).join() === "3");

console.log("\n--- a single-core machine ---");
api.renderDevices(payload({ threads: { value: "auto", max: 1, auto: 1, resolved: 1, override: null } }));
html = devicesEl.innerHTML;

check("there is still something to pick", threadOptions().length === 2, String(threadOptions().length));
check("the core is singular", html.includes("out of 1 physical core."), inThreadRow(/out of [^<]*/));
check("and so is the thread", html.includes("Auto — 1 thread<"), inThreadRow(/Auto — [^<]*/));

console.log("\n--- a payload with no threads block ---");
api.renderDevices((() => {
  const data = payload();
  delete data.devices.mtl_threads;
  return data;
})());
html = devicesEl.innerHTML;

check("the slots still render", (html.match(/class="device-row"/g) || []).length === 3);
check("and the thread row is simply absent", !html.includes(api.THREADS_LABEL));

console.log("\n--- a payload with no devices block ---");
api.renderDevices({ languages: {} });
check("renders nothing rather than throwing", devicesEl.innerHTML === "");

console.log("\n--- saving ---");
calls.length = 0;
reply = { data: payload({ selection: { paddleocr: "auto", bubble: "cuda:0", translator: "auto" }, restart: true }) };
await api.chooseDevice("bubble", "cuda:0");

check("one request", calls.length === 1, String(calls.length));
check("to the devices route", calls[0].url === "/api/settings/devices", calls[0].url);
check("as a PUT", calls[0].method === "PUT", calls[0].method);
check("carrying only that slot", JSON.stringify(calls[0].body) === '{"bubble":"cuda:0"}', JSON.stringify(calls[0].body));
check("the section re-rendered", devicesEl.innerHTML.includes("Restart Fox Reader"));
check("and the user is told", api.el("message").textContent.includes("Restart"), api.el("message").textContent);

console.log("\n--- a rejected save ---");
calls.length = 0;
reply = { ok: false, data: { error: "Unknown device: 'rubbish'." } };
await api.chooseDevice("bubble", "rubbish");

check("the error is shown", api.el("message").textContent.includes("Unknown device"), api.el("message").textContent);
check("as an error", api.el("message").className === "error", api.el("message").className);
check("and the page reloads to undo the picker", calls.length === 2 && calls[1].url.startsWith("/api/settings"), JSON.stringify(calls.map(c => c.url)));

console.log("\n--- saving a thread count ---");
calls.length = 0;
reply = { data: payload({ threads: { value: 6, max: 8, auto: 6, resolved: 6, override: null } }) };
await api.chooseThreads("6");

check("one request", calls.length === 1, String(calls.length));
check("to the same route as the slots", calls[0].url === "/api/settings/devices", calls[0].url);
check("as a PUT", calls[0].method === "PUT", calls[0].method);
check("carrying only the count", JSON.stringify(calls[0].body) === '{"mtl_threads":"6"}', JSON.stringify(calls[0].body));
check("the section re-rendered with it selected", devicesEl.innerHTML.includes('value="6" selected'), devicesEl.innerHTML.match(/value="6"[^>]*/)?.[0] || "");
check(
  "and the user is told it needs no restart",
  api.el("message").textContent.includes("next time a model loads"),
  api.el("message").textContent,
);

console.log("\n--- a rejected thread count ---");
calls.length = 0;
reply = { ok: false, data: { error: "Thread count 0 is below one." } };
await api.chooseThreads("0");

check("the error is shown", api.el("message").textContent.includes("below one"), api.el("message").textContent);
check("as an error", api.el("message").className === "error", api.el("message").className);
check("and the page reloads to undo the picker", calls.length === 2 && calls[1].url.startsWith("/api/settings"), JSON.stringify(calls.map(c => c.url)));

console.log("\n--- escaping ---");
api.renderDevices({
  devices: {
    ...payload().devices,
    gpu_name: '<img src=x onerror=alert(1)>',
    fallbacks: ['bubble: <script>alert(2)</script>'],
  },
});
html = devicesEl.innerHTML;

check("a hostile gpu name is escaped", !html.includes("<img") && html.includes("&lt;img"));
check("a hostile fallback line is escaped", !html.includes("<script>") && html.includes("&lt;script&gt;"));

console.log(failures ? `\n${failures} FAILURE(S)` : "\nall settings page checks passed");
process.exit(failures ? 1 : 0);
