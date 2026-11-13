// @vitest-environment node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { JSDOM, VirtualConsole } from "jsdom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Render this page's known Jinja inputs without Python or a second copy of its script.
// Limits match user_endpoints.py, endpoint_schema.py and endpoint_crypto.py.
const templateValues: Record<string, unknown> = {
    known_languages: ["japanese", "korean", "chinese"],
    max_per_language: 20,
    max_active_per_language: 3,
    name_max_length: 6,
    text_length_min: 100,
    text_length_max: 10000,
    text_length_default: 2000,
    uuid_variants: ["uuid1", "uuid2", "uuid4"],
    uuid_default: "uuid4",
    key_encodings: ["fernet", "passphrase"],
    key_length: 44,
    max_keys: 4,
    encryption_available: true,
    token_max_age_min: 30,
    token_max_age_max: 86400,
};
const template = readFileSync(resolve("templates/user_endpoints.html"), "utf8")
    .replace(/\{%\s*for language in known_languages\s*%\}([\s\S]*?)\{%\s*endfor\s*%\}/g, (_, row: string) =>
        (templateValues.known_languages as string[])
            .map((language) =>
                row
                    .replace(/\{\{\s*language\|capitalize\s*\}\}/g, language[0].toUpperCase() + language.slice(1))
                    .replace(/\{\{\s*language\s*\}\}/g, language),
            )
            .join(""),
    )
    .replace(/\{%\s*if not encryption_available\s*%\}[\s\S]*?\{%\s*endif\s*%\}/g, "")
    .replace(/\{\{\s*(\w+)(\|tojson)?\s*\}\}/g, (_, name: string, json: string) => {
        if (!Object.hasOwn(templateValues, name)) throw new Error(`Unknown template input: ${name}`);
        return json ? JSON.stringify(templateValues[name]) : String(templateValues[name]);
    });
if (/\{%|\{\{/.test(template)) throw new Error("Unrendered Jinja in user endpoint fixture");

const API = "/api/user_endpoints";
const OLD_KEY = "saved endpoint passphrase";
const NEW_KEY = "separately supplied import passphrase";
const textNode = () => ({ type: "string", source: "text", value: "" });
const endpointDraft = (overrides: Record<string, unknown> = {}) => ({
    name: "Share",
    hostname: "api.example.invalid/translate",
    host_scheme: "https",
    method: "POST",
    timeout: 20,
    retries: 0,
    max_text_length: 2000,
    body_format: "json",
    response_format: "json",
    languages: ["japanese"],
    language_codes: { japanese: "ja" },
    target_language: "en",
    headers: {},
    query: {},
    encryption: {
        enabled: false,
        key_encoding: "passphrase",
        encrypt_request: true,
        decrypt_response: true,
        max_age: null,
    },
    request_schema: { type: "dict", source: "static", children: { text: textNode() } },
    response_schema: { type: "dict", source: "static", children: { text: textNode() } },
    ...overrides,
});
const importResult = (endpoint = endpointDraft(), requires_key = false, warnings: string[] = []) => ({
    endpoint,
    requires_key,
    warnings,
});
// JSON flow mappings are valid YAML; parsing/validation belongs to the mocked API.
const yamlFor = (endpoint = endpointDraft()) => `version: 1\nendpoint: ${JSON.stringify(endpoint)}\n`;
const exportedYAML = yamlFor();
interface Reply {
    ok: boolean;
    status: number;
    json: () => Promise<unknown>;
}
interface Request {
    path: string;
    method: string;
    body: any;
    signal?: AbortSignal | null;
}
const reply = (body: unknown, status = 200): Reply => ({ ok: status < 400, status, json: async () => body });
function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason: Error) => void;
    const promise = new Promise<T>((done, fail) => {
        resolve = done;
        reject = fail;
    });
    return { promise, resolve, reject };
}

const pages: { dom: JSDOM; errors: Error[] }[] = [];
beforeEach(() => vi.useFakeTimers());
afterEach(() => {
    const errors = pages.flatMap((page) => page.errors);
    for (const { dom } of pages.splice(0)) dom.window.close();
    vi.useRealTimers();
    expect(errors).toEqual([]);
});

function endpointPage() {
    const requests: Request[] = [];
    const errors: Error[] = [];
    const responses = new Map<string, (() => Reply | Promise<Reply>)[]>();
    const saved = {
        ...endpointDraft(),
        id: "saved",
        url: "https://api.example.invalid/translate",
        active_languages: ["japanese"],
        encryption: { ...endpointDraft().encryption, enabled: true, has_key: true },
    };
    const endpoints: any[] = [saved];
    const fetchMock = vi.fn(async (url: string, options: RequestInit = {}) => {
        const path = new URL(url, "http://localhost").pathname;
        const method = options.method || "GET";
        const body = options.body ? JSON.parse(String(options.body)) : undefined;
        requests.push({ path, method, body, signal: options.signal });
        const queued = responses.get(`${method} ${path}`)?.shift();
        if (queued) return queued();
        if (path === API && method === "GET") {
            return reply({
                endpoints: structuredClone(endpoints),
                counts: { japanese: endpoints.length },
                active_by_language: { japanese: ["saved"] },
                limits: { max_per_language: 20, max_active_per_language: 3 },
            });
        }
        if (path === API && method === "POST") {
            endpoints.push({ ...body, active_languages: [] });
            return reply({ endpoint: body });
        }
        if (path === `${API}/saved/key` && method === "GET") {
            return reply({ key: OLD_KEY, key_encoding: "passphrase" });
        }
        if (path === `${API}/saved/export` && method === "GET") return reply({ yaml: exportedYAML });
        throw new Error(`Unexpected endpoint request: ${method} ${path}`);
    });
    const writeText = vi.fn(async (_text: string) => {});
    const virtualConsole = new VirtualConsole();
    virtualConsole.on("jsdomError", (error) => errors.push(error));
    const dom = new JSDOM(template, {
        url: "http://localhost/user_endpoints",
        runScripts: "dangerously",
        pretendToBeVisual: true,
        virtualConsole,
        beforeParse(window) {
            Object.assign(window, {
                fetch: fetchMock,
                TextEncoder,
                BroadcastChannel: class {
                    addEventListener() {}
                    postMessage() {}
                    close() {}
                },
            });
            Object.defineProperty(window.navigator, "clipboard", { configurable: true, value: { writeText } });
            window.HTMLElement.prototype.scrollIntoView = () => {};
            // JSDOM has no modal top layer; retain native open state and queued close events.
            window.HTMLDialogElement.prototype.showModal = function () {
                this.open = true;
            };
            window.HTMLDialogElement.prototype.close = function () {
                if (!this.open) return;
                this.open = false;
                window.setTimeout(() => this.dispatchEvent(new window.Event("close")), 0);
            };
        },
    });
    pages.push({ dom, errors });
    const document = dom.window.document;
    const element = <T extends HTMLElement = HTMLElement>(id: string) => {
        const found = document.getElementById(id);
        expect(found, `#${id}`).not.toBeNull();
        return found as T;
    };
    const button = (id: string) => element<HTMLButtonElement>(id);
    const input = (id: string) => element<HTMLInputElement>(id);
    const dialog = (id: string) => element<HTMLDialogElement>(id);
    const textarea = () => element<HTMLTextAreaElement>("transferText");
    const select = <T extends HTMLElement = HTMLElement>(selector: string) => {
        const found = document.querySelector<T>(selector);
        expect(found, selector).not.toBeNull();
        return found!;
    };
    const edit = (id: string, value: string) => {
        const field = element<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(id);
        field.value = value;
        field.dispatchEvent(new dom.window.Event(field.tagName === "SELECT" ? "change" : "input", { bubbles: true }));
    };
    const card = (action: string) => select<HTMLButtonElement>(`[data-card="${action}"][data-id="saved"]`);
    const flush = () => vi.advanceTimersByTimeAsync(0);
    const ready = async () => {
        await flush();
        expect(errors).toEqual([]);
        expect(card("export").textContent).toContain("Export");
    };
    const queue = (path: string, response: () => Reply | Promise<Reply>, method = "POST") => {
        const key = `${method} ${path}`;
        responses.set(key, [...(responses.get(key) || []), response]);
    };
    const hold = (path: string, body: unknown, phase: "fetch" | "body", method = "POST") => {
        const gate = deferred<void>();
        // Deliberately ignore AbortSignal: stale result guards and timeouts must still work.
        queue(
            path,
            () =>
                phase === "fetch"
                    ? gate.promise.then(() => reply(body))
                    : {
                          ...reply(body),
                          json: async () => {
                              await gate.promise;
                              return body;
                          },
                      },
            method,
        );
        return () => gate.resolve();
    };
    const imports = () => requests.filter((request) => request.path === `${API}/import`);
    const saves = () => requests.filter((request) => request.path === API && request.method === "POST");
    const openImport = (yaml = exportedYAML) => {
        button("importEndpointButton").click();
        edit("transferText", yaml);
    };
    const importDraft = async (result = importResult()) => {
        queue(`${API}/import`, () => reply(result));
        openImport(yamlFor(result.endpoint));
        button("validateImportButton").click();
        await flush();
        expect(dialog("transferDialog").open).toBe(false);
        expect(dialog("dialog").open).toBe(true);
        expect(element("dialogTitle").textContent).toBe("Import Endpoint");
        expect(saves()).toHaveLength(0);
    };
    return {
        dom,
        document,
        element,
        button,
        input,
        dialog,
        textarea,
        select,
        edit,
        card,
        flush,
        ready,
        queue,
        hold,
        imports,
        saves,
        openImport,
        importDraft,
        requests,
        writeText,
    };
}

describe("endpoint YAML transfer", () => {
    it("exports server YAML into a readonly dialog and copies only on an explicit click", async () => {
        const ui = endpointPage();
        await ui.ready();
        ui.card("export").click();
        expect(ui.dialog("transferDialog").open).toBe(true);
        expect(ui.button("copyExportButton").disabled).toBe(true);
        expect(ui.textarea().readOnly).toBe(true);
        expect(ui.button("validateImportButton").hidden).toBe(true);
        await ui.flush();
        expect(ui.textarea().value).toBe(exportedYAML);
        expect(ui.writeText).not.toHaveBeenCalled();
        expect(ui.requests.map(({ method, path }) => `${method} ${path}`)).toEqual([
            `GET ${API}`,
            `GET ${API}/saved/export`,
        ]);
        ui.button("copyExportButton").click();
        await ui.flush();
        expect(ui.writeText).toHaveBeenCalledExactlyOnceWith(exportedYAML);
        expect(ui.element("transferStatus").textContent).toMatch(/copied/i);
        ui.button("closeTransferButton").click();
        expect(ui.dialog("transferDialog").open).toBe(false);
        expect(ui.textarea().value).toBe("");
    });

    it.each(["denied", "unavailable"])("selects YAML for manual copying when the clipboard is %s", async (failure) => {
        const ui = endpointPage();
        await ui.ready();
        if (failure === "denied") ui.writeText.mockRejectedValueOnce(new Error("Permission denied"));
        else Object.defineProperty(ui.dom.window.navigator, "clipboard", { value: undefined });
        ui.card("export").click();
        await ui.flush();
        ui.button("copyExportButton").click();
        await ui.flush();
        expect(ui.textarea().value).toBe(exportedYAML);
        expect(ui.document.activeElement).toBe(ui.textarea());
        expect(ui.textarea().selectionStart).toBe(0);
        expect(ui.textarea().selectionEnd).toBe(exportedYAML.length);
        expect(ui.element("transferStatus").textContent).toMatch(/Ctrl\+C.*Cmd\+C/);
        expect(ui.dialog("transferDialog").open).toBe(true);
    });

    it("keeps invalid YAML editable and displays server suggestions as literal text", async () => {
        const ui = endpointPage();
        await ui.ready();
        const yaml = "endpoint: [broken YAML";
        const error = '<img src=x onerror="window.injected=true"> Invalid YAML';
        const suggestion = '<svg onload="window.injected=true">Use a mapping</svg>';
        ui.queue(`${API}/import`, () => reply({ error, suggestions: [suggestion, "Check indentation."] }, 400));
        ui.openImport(yaml);
        expect(ui.textarea().readOnly).toBe(false);
        expect(ui.button("copyExportButton").hidden).toBe(true);
        ui.button("validateImportButton").click();
        await ui.flush();
        expect(ui.imports()[0].body).toEqual({ yaml });
        expect(ui.textarea().value).toBe(yaml);
        expect(ui.element("transferStatus").textContent).toContain(error);
        expect(ui.element("transferStatus").textContent).toContain(suggestion);
        expect(ui.element("transferStatus").querySelector("img, svg, script")).toBeNull();
        expect(ui.dialog("transferDialog").open).toBe(true);
        expect(ui.dialog("dialog").open).toBe(false);
        expect(ui.button("validateImportButton").disabled).toBe(false);
        expect(ui.saves()).toHaveLength(0);
        ui.queue(`${API}/import`, () => reply(importResult()));
        ui.edit("transferText", exportedYAML);
        ui.button("validateImportButton").click();
        await ui.flush();
        expect(ui.dialog("dialog").open).toBe(true);
    });

    it("keeps the YAML available for retry when the server returns a malformed error response", async () => {
        const ui = endpointPage();
        await ui.ready();
        ui.queue(`${API}/import`, () => reply(null, 400));
        ui.openImport();
        ui.button("validateImportButton").click();
        await ui.flush();
        expect(ui.element("transferStatus").textContent).toBe("Fox Reader returned an invalid response. Try again.");
        expect(ui.textarea().value).toBe(exportedYAML);
        expect(ui.dialog("transferDialog").open).toBe(true);
        expect(ui.dialog("dialog").open).toBe(false);
        expect(ui.button("validateImportButton").disabled).toBe(false);
        expect(ui.saves()).toHaveLength(0);
    });

    it("imports a new unsaved draft, clears a revealed old key, and requires a separately supplied key to save", async () => {
        const ui = endpointPage();
        await ui.ready();
        ui.card("edit").click();
        ui.button("encReveal").click();
        await ui.flush();
        expect(ui.input("encKey").value).toBe(OLD_KEY);
        ui.select<HTMLButtonElement>('#dialog [data-act="close"]').click();
        const start = ui.requests.length;
        const draft = endpointDraft({ encryption: { ...endpointDraft().encryption, enabled: true } });
        const warning = "<b>Enter your own shared key.</b>";
        await ui.importDraft(importResult(draft, true, [warning]));
        expect(ui.input("name").value).toBe("Share");
        expect(ui.input("encKey").value).toBe("");
        expect(ui.input("encKey").type).toBe("password");
        expect(ui.element("dialogStatus").textContent).toContain(warning);
        expect(ui.element("dialogStatus").querySelector("b")).toBeNull();
        ui.button("saveButton").click();
        await ui.flush();
        expect(ui.element("dialogStatus").textContent).toMatch(/needs a passphrase/i);
        expect(ui.dialog("dialog").open).toBe(true);
        expect(ui.saves()).toHaveLength(0);

        ui.edit("encKey", NEW_KEY);
        ui.button("saveButton").click();
        await ui.flush();
        expect(ui.saves()).toHaveLength(1);
        const saved = ui.saves()[0].body;
        expect(saved.id).toMatch(/^[0-9a-f-]{36}$/i);
        expect(saved.id).not.toBe("saved");
        expect(saved.encryption.key).toBe(NEW_KEY);
        expect(saved).not.toHaveProperty("active_languages");
        expect(ui.requests.slice(start).map(({ method, path }) => `${method} ${path}`)).toEqual([
            `POST ${API}/import`,
            `POST ${API}`,
            `GET ${API}`,
        ]);
        expect(JSON.stringify(ui.requests.slice(start))).not.toContain(OLD_KEY);
        expect(ui.dialog("dialog").open).toBe(false);
        expect(ui.element("resultMessage").textContent).toContain("Endpoint saved");
    });

    it("preserves an imported duplicate-name draft after server rejection and saves it after renaming", async () => {
        const ui = endpointPage();
        await ui.ready();
        const duplicateName = "An endpoint with this name already exists. Choose a different name before saving.";
        const draft = endpointDraft({
            name: "sHaRe",
            hostname: "duplicate.example.invalid/translate",
            headers: { "X-Mode": "preview" },
            query: { region: "in" },
        });
        await ui.importDraft(importResult(draft, false, [duplicateName]));
        expect(ui.element("dialogStatus").textContent).toContain(duplicateName);
        ui.edit("targetLanguage", "fr");
        ui.edit("timeout", "35");
        const requestPreview = ui.element("requestPreview").textContent;

        ui.queue(API, () => reply({ error: duplicateName }, 400));
        ui.button("saveButton").click();
        await ui.flush();
        expect(ui.saves()).toHaveLength(1);
        const rejectedPayload = ui.saves()[0].body;
        expect(rejectedPayload.name).toBe(draft.name);
        expect(ui.element("dialogStatus").textContent).toBe(duplicateName);
        expect(ui.dialog("dialog").open).toBe(true);
        expect(ui.button("saveButton").disabled).toBe(false);
        expect(ui.input("name").value).toBe(draft.name);
        expect(ui.input("host").value).toBe(draft.hostname);
        expect(ui.input("targetLanguage").value).toBe("fr");
        expect(ui.input("timeout").value).toBe("35");
        expect(ui.select<HTMLInputElement>('#langPicker .lc[value="japanese"]').checked).toBe(true);
        expect(ui.select<HTMLInputElement>("#headerRows .r-value").value).toBe("preview");
        expect(ui.select<HTMLInputElement>("#queryRows .r-value").value).toBe("in");
        expect(ui.element("requestPreview").textContent).toBe(requestPreview);

        ui.edit("name", "Fresh");
        ui.button("saveButton").click();
        await ui.flush();
        expect(ui.saves()).toHaveLength(2);
        expect(ui.saves()[1].body).toEqual({ ...rejectedPayload, id: expect.any(String), name: "Fresh" });
        expect(ui.saves()[1].body.id).not.toBe("saved");
        expect(ui.dialog("dialog").open).toBe(false);
        expect(ui.element("resultMessage").textContent).toContain("Endpoint saved");
        expect(ui.requests.map(({ method, path }) => `${method} ${path}`)).toEqual([
            `GET ${API}`,
            `POST ${API}/import`,
            `POST ${API}`,
            `POST ${API}`,
            `GET ${API}`,
        ]);
    });

    it("allows cancelling a validated draft without saving or activating it", async () => {
        const ui = endpointPage();
        await ui.ready();
        await ui.importDraft();
        ui.select<HTMLButtonElement>('#dialog [data-act="close"]').click();
        await ui.flush();
        expect(ui.dialog("dialog").open).toBe(false);
        expect(ui.requests.map(({ method, path }) => `${method} ${path}`)).toEqual([
            `GET ${API}`,
            `POST ${API}/import`,
        ]);
    });

    it.each([
        ["timeout", "0", "20", /Timeout/],
        ["port", "65536", "8080", /Port/],
        ["retries", "6", "0", /Retries/],
        ["maxTextLength", "99", "2000", /Max text length/],
        ["encMaxAge", "29", "30", /Reject replies older than/],
    ] as const)("keeps existing %s validation on Save for imported drafts", async (field, invalid, valid, message) => {
        const ui = endpointPage();
        await ui.ready();
        await ui.importDraft();
        ui.edit(field, invalid);
        ui.button("saveButton").click();
        await ui.flush();
        expect(ui.saves()).toHaveLength(0);
        if (field === "maxTextLength" || field === "encMaxAge") {
            // Native number-input constraints reject the click before the submit handler runs.
            expect(ui.input(field).validity.rangeUnderflow).toBe(true);
        } else expect(ui.element("dialogStatus").textContent).toMatch(message);
        ui.edit(field, valid);
        ui.button("saveButton").click();
        await ui.flush();
        expect(ui.saves()).toHaveLength(1);
    });

    it("preserves imported string roots and enforces the Text source requirement before Save", async () => {
        const ui = endpointPage();
        await ui.ready();
        const draft = endpointDraft({
            body_format: "text",
            response_format: "text",
            request_schema: textNode(),
            response_schema: textNode(),
        });
        await ui.importDraft(importResult(draft));
        for (const kind of ["request", "response"]) {
            expect(ui.element<HTMLSelectElement>(`${kind}RootType`).value).toBe("string");
            expect(ui.element<HTMLSelectElement>(`${kind}RootType`).disabled).toBe(false);
            expect(ui.select(`#${kind}Tree .slot-tag`).textContent).toBe("root");
            expect(ui.select<HTMLSelectElement>(`#${kind}Tree [data-act="type"][data-path=""]`).disabled).toBe(true);
            expect(ui.select<HTMLSelectElement>(`#${kind}Tree [data-act="source"][data-path=""]`).value).toBe("text");
        }
        const changeSource = (value: string) => {
            const select = ui.select<HTMLSelectElement>('#requestTree [data-act="source"][data-path=""]');
            select.value = value;
            select.dispatchEvent(new ui.dom.window.Event("change", { bubbles: true }));
        };
        changeSource("static");
        ui.button("saveButton").click();
        await ui.flush();
        expect(ui.saves()).toHaveLength(0);
        expect(ui.element("dialogStatus").textContent).toMatch(/Text/);
        changeSource("text");
        ui.button("saveButton").click();
        await ui.flush();
        expect(ui.saves()).toHaveLength(1);
        expect(ui.saves()[0].body.request_schema).toEqual(textNode());
        expect(ui.saves()[0].body.response_schema).toEqual(textNode());
        expect(ui.saves()[0].body.body_format).toBe("text");
        expect(ui.saves()[0].body.response_format).toBe("text");
    });

    it("retains prototype-like field names in imported schema previews, headers, query and saved JSON", async () => {
        const ui = endpointPage();
        await ui.ready();
        const names = ["__proto__", "constructor", "toString"];
        const fields = Object.fromEntries(
            names.map((name) => [name, { type: "string", source: "static", value: `${name} value` }]),
        );
        const map = Object.fromEntries(names.map((name) => [name, `${name} value`]));
        const schema = { type: "dict", source: "static", children: { text: textNode(), ...fields } };
        await ui.importDraft(
            importResult(endpointDraft({ request_schema: schema, response_schema: schema, headers: map, query: map })),
        );
        for (const kind of ["request", "response"]) {
            const preview = JSON.parse(ui.element(`${kind}Preview`).textContent!);
            for (const name of names) expect(preview[name]).toBe(`${name} value`);
        }
        ui.button("saveButton").click();
        await ui.flush();
        expect(ui.saves()).toHaveLength(1);
        const saved = ui.saves()[0].body;
        expect(saved.headers).toEqual(map);
        expect(saved.query).toEqual(map);
        expect(saved.request_schema).toEqual(schema);
        expect(saved.response_schema).toEqual(schema);
    });

    it("checks the 128 KiB UTF-8 limit before requesting validation", async () => {
        const ui = endpointPage();
        await ui.ready();
        ui.openImport(" \n\t");
        ui.button("validateImportButton").click();
        expect(ui.element("transferStatus").textContent).toMatch(/Paste/);
        const oversized = "é".repeat(65536) + "a";
        expect(oversized.length).toBeLessThan(131072);
        ui.edit("transferText", oversized);
        ui.button("validateImportButton").click();
        expect(ui.element("transferStatus").textContent).toMatch(/128 KiB/);
        expect(ui.imports()).toHaveLength(0);
        const atLimit = "#" + "é".repeat(65535) + "\n";
        ui.queue(`${API}/import`, () =>
            reply({ error: "No endpoint", suggestions: ["Add an endpoint mapping."] }, 400),
        );
        ui.edit("transferText", atLimit);
        ui.button("validateImportButton").click();
        await ui.flush();
        expect(ui.imports()).toHaveLength(1);
        expect(ui.imports()[0].body.yaml).toBe(atLimit);
    });
});

describe("endpoint transfer request lifecycle", () => {
    it.each(["fetch", "body"] as const)(
        "ignores an old import %s after the paste is edited and another validation starts",
        async (phase) => {
            const ui = endpointPage();
            await ui.ready();
            const releaseOld = ui.hold(`${API}/import`, importResult(endpointDraft({ name: "Old" })), phase);
            ui.openImport();
            ui.button("validateImportButton").click();
            ui.button("validateImportButton").click();
            await ui.flush();
            expect(ui.imports()).toHaveLength(1);
            expect(ui.button("validateImportButton").disabled).toBe(true);
            const newer = endpointDraft({ name: "New" });
            ui.edit("transferText", yamlFor(newer));
            expect(ui.imports()[0].signal?.aborted).toBe(true);
            const releaseNew = ui.hold(`${API}/import`, importResult(newer), "body");
            ui.button("validateImportButton").click();
            await ui.flush();
            releaseOld();
            await ui.flush();
            expect(ui.dialog("dialog").open).toBe(false);
            expect(ui.textarea().value).toBe(yamlFor(newer));
            expect(ui.button("validateImportButton").disabled).toBe(true);
            releaseNew();
            await ui.flush();
            expect(ui.input("name").value).toBe("New");
            expect(ui.dialog("dialog").open).toBe(true);
            expect(ui.saves()).toHaveLength(0);
        },
    );

    it.each(["close", "Escape", "native close", "reopen"])("ignores a pending import after %s", async (action) => {
        const ui = endpointPage();
        await ui.ready();
        const release = ui.hold(`${API}/import`, importResult(), "body");
        ui.openImport();
        ui.button("validateImportButton").click();
        await ui.flush();
        if (action === "Escape") {
            const dialog = ui.dialog("transferDialog");
            if (dialog.dispatchEvent(new ui.dom.window.Event("cancel", { cancelable: true }))) dialog.close();
        } else if (action === "native close") ui.dialog("transferDialog").close();
        else ui.button("closeTransferButton").click();
        if (action === "reopen") ui.openImport("# newer paste");
        await ui.flush();
        expect(ui.imports()[0].signal?.aborted).toBe(true);
        release();
        await ui.flush();
        expect(ui.dialog("dialog").open).toBe(false);
        expect(ui.dialog("transferDialog").open).toBe(action === "reopen");
        expect(ui.textarea().value).toBe(action === "reopen" ? "# newer paste" : "");
        expect(ui.button("validateImportButton").disabled).toBe(false);
        expect(ui.saves()).toHaveLength(0);
    });

    it("ignores a closed export while another export is pending, including its queued close event", async () => {
        const ui = endpointPage();
        await ui.ready();
        const releaseOld = ui.hold(`${API}/saved/export`, { yaml: "# old export" }, "fetch", "GET");
        ui.card("export").click();
        ui.button("closeTransferButton").click();
        const releaseNew = ui.hold(`${API}/saved/export`, { yaml: exportedYAML }, "body", "GET");
        ui.card("export").click();
        await ui.flush();
        expect(ui.requests[1].signal?.aborted).toBe(true);
        expect(ui.requests[2].signal?.aborted).toBe(false);
        releaseOld();
        await ui.flush();
        expect(ui.textarea().value).toBe("");
        expect(ui.button("copyExportButton").disabled).toBe(true);
        releaseNew();
        await ui.flush();
        expect(ui.textarea().value).toBe(exportedYAML);
        expect(ui.button("copyExportButton").disabled).toBe(false);
        expect(ui.writeText).not.toHaveBeenCalled();
    });

    it("does not let a late clipboard rejection select or replace a reopened import", async () => {
        const ui = endpointPage();
        await ui.ready();
        ui.card("export").click();
        await ui.flush();
        const clipboard = deferred<void>();
        ui.writeText.mockReturnValueOnce(clipboard.promise);
        ui.button("copyExportButton").click();
        ui.button("closeTransferButton").click();
        ui.openImport("# my new paste");
        ui.textarea().setSelectionRange(3, 3);
        clipboard.reject(new Error("Permission denied"));
        await ui.flush();
        expect(ui.textarea().value).toBe("# my new paste");
        expect(ui.textarea().selectionStart).toBe(3);
        expect(ui.textarea().selectionEnd).toBe(3);
        expect(ui.element("transferStatus").textContent).toBe("");
    });

    it.each(["fetch", "body"] as const)(
        "times out an import whose %s ignores abort and allows a fresh validation",
        async (phase) => {
            const ui = endpointPage();
            await ui.ready();
            const release = ui.hold(`${API}/import`, importResult(endpointDraft({ name: "Old" })), phase);
            ui.openImport();
            ui.button("validateImportButton").click();
            await ui.flush();
            await vi.advanceTimersByTimeAsync(9999);
            expect(ui.button("validateImportButton").disabled).toBe(true);
            await vi.advanceTimersByTimeAsync(1);
            expect(ui.imports()[0].signal?.aborted).toBe(true);
            expect(ui.element("transferStatus").textContent).toMatch(/timed out/i);
            expect(ui.button("validateImportButton").disabled).toBe(false);
            expect(ui.textarea().value).toBe(exportedYAML);
            ui.queue(`${API}/import`, () => reply(importResult(endpointDraft({ name: "Retry" }))));
            ui.button("validateImportButton").click();
            await ui.flush();
            expect(ui.input("name").value).toBe("Retry");
            release();
            await ui.flush();
            expect(ui.input("name").value).toBe("Retry");
            expect(ui.saves()).toHaveLength(0);
        },
    );
});

describe("endpoint hostname and Scheme", () => {
    it.each(["https://", "HTTP://", "ftp://", "https:", "mailto:", "//"])(
        "rejects a %s hostname prefix until it is removed and Scheme is chosen separately",
        async (prefix) => {
            const ui = endpointPage();
            await ui.ready();
            ui.button("addEndpointButton").click();
            ui.edit("name", "Local");
            ui.edit("host", `${prefix}localhost:8080/translate`);
            ui.select<HTMLInputElement>('#langPicker .lc[value="japanese"]').click();
            ui.button("saveButton").click();
            await ui.flush();
            expect(ui.saves()).toHaveLength(0);
            expect(ui.element("dialogStatus").textContent).toMatch(/prefix.*Scheme/i);
            expect(ui.input("host").value).toBe(`${prefix}localhost:8080/translate`);
            ui.edit("scheme", "http");
            ui.button("saveButton").click();
            await ui.flush();
            expect(ui.saves()).toHaveLength(0);
            ui.edit("host", "localhost:8080/translate");
            ui.button("saveButton").click();
            await ui.flush();
            expect(ui.saves()).toHaveLength(1);
            expect(ui.saves()[0].body).toMatchObject({ hostname: "localhost:8080/translate", host_scheme: "http" });
        },
    );
});
