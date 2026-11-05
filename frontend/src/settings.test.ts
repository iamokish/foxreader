// @vitest-environment node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { JSDOM, VirtualConsole } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

const template = readFileSync(resolve("templates/settings.html"), "utf8");
const pages: JSDOM[] = [];
const context = (max = 32768) => ({
    selected: 8192,
    effective: 8192 as number | null,
    max,
    options: max >= 16384 ? [8192, 16384] : [8192],
    running: null as number | null,
    pending: false,
    error: undefined as string | undefined,
});
const gemma = { id: "gemma", name: "Gemma 4 E4B Q8", downloaded: true, size_on_disk: "7.6 GiB", context: context() };
const qwen = { id: "qwen", name: "Qwen3.5 9B Q6_K", downloaded: true, size_on_disk: "7.3 GiB", context: context() };
const vntl = {
    id: "vntl",
    name: "VNTL Llama3 8B v2 Q8",
    downloaded: false,
    size_on_disk: "8.0 GiB",
    context: context(8192),
};

interface Model {
    id: string;
    name: string;
    downloaded: boolean;
    size_on_disk: string;
    context: ReturnType<typeof context>;
}
interface Settings {
    languages: Record<string, { default_model: string; models: Model[] }>;
    api_tokens: Record<string, string>;
}

function payload(): Settings {
    return {
        languages: {
            japanese: { default_model: gemma.id, models: [gemma, qwen, vntl] },
            korean: { default_model: gemma.id, models: [gemma, qwen] },
            chinese: { default_model: qwen.id, models: [gemma, qwen] },
        },
        api_tokens: {},
    };
}

afterEach(() => {
    for (const page of pages.splice(0)) page.window.close();
});

function settingsPage(initial = payload()) {
    let current = structuredClone(initial);
    const requests: { path: string; method: string; body: any }[] = [];
    const held = new Map<string, Promise<void>>();
    const rejected = new Map<string, string>();
    const errors: Error[] = [];
    const fetchMock = vi.fn(async (url: string, options: RequestInit = {}) => {
        const path = new URL(url, "http://localhost").pathname;
        const method = options.method || "GET";
        const body = options.body ? JSON.parse(String(options.body)) : undefined;
        requests.push({ path, method, body });
        const rejection = rejected.get(path);
        if (!rejection && path === "/api/settings/mtl-default") {
            current.languages[body.language].default_model = body.model_id;
        }
        if (!rejection && path === "/api/settings/mtl-context") {
            for (const language of Object.values(current.languages)) {
                for (const model of language.models) {
                    if (model.id !== body.model_id) continue;
                    model.context.selected = body.context_size;
                    model.context.effective = Math.min(body.context_size, model.context.max);
                    model.context.pending =
                        model.context.running !== null && model.context.running !== model.context.effective;
                }
            }
        }
        const response = rejection ? { error: rejection } : structuredClone(current);
        return {
            ok: !rejection,
            json: async () => {
                await held.get(path);
                return response;
            },
        };
    });
    const virtualConsole = new VirtualConsole();
    virtualConsole.on("jsdomError", (error) => errors.push(error));
    const page = new JSDOM(template, {
        url: "http://localhost/settings",
        runScripts: "dangerously",
        pretendToBeVisual: true,
        virtualConsole,
        beforeParse(window) {
            Object.assign(window, {
                fetch: fetchMock,
                WebSocket: class {
                    close() {}
                },
                BroadcastChannel: class {
                    postMessage() {}
                    close() {}
                },
            });
        },
    });
    pages.push(page);
    const document = page.window.document;
    const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
    const tab = (language: string) => element<HTMLButtonElement>(`mtl-tab-${language}`);
    const panel = (language: string) => element(`mtl-panel-${language}`);
    const choice = (language: string, id: string) =>
        Array.from(panel(language).querySelectorAll<HTMLInputElement>("[data-model-choice]")).find(
            (input) => input.value === id,
        )!;
    const saves = () => requests.filter((request) => request.method === "PUT");
    const contextChoice = (language: string, id: string, size: number) =>
        Array.from(panel(language).querySelectorAll<HTMLInputElement>("[data-context-model]")).find(
            (input) => input.dataset.contextModel === id && Number(input.value) === size,
        )!;
    const hold = (path: string) => {
        let release!: () => void;
        held.set(
            path,
            new Promise<void>((resolve) => {
                release = resolve;
            }),
        );
        return () => {
            held.delete(path);
            release();
        };
    };
    const ready = async () => {
        await vi.waitFor(() => expect(document.querySelectorAll('[role="tab"]')).toHaveLength(3));
        expect(errors).toEqual([]);
    };
    return { page, document, element, tab, panel, choice, contextChoice, saves, hold, rejected, ready, errors };
}

describe("settings language tabs and local model defaults", () => {
    it("shows one language's compatible models with clear availability and default badges", async () => {
        const ui = settingsPage();
        await ui.ready();
        expect(ui.tab("japanese").getAttribute("aria-selected")).toBe("true");
        expect(ui.document.querySelectorAll('[role="tabpanel"]:not([hidden])')).toHaveLength(1);
        expect(ui.choice("japanese", gemma.id).checked).toBe(true);
        expect(ui.choice("japanese", vntl.id).disabled).toBe(true);
        expect(ui.choice("japanese", gemma.id).closest("article")?.textContent).toContain("Default");
        expect(ui.choice("japanese", gemma.id).closest("article")?.textContent).toContain("Downloaded");
        expect(ui.choice("japanese", vntl.id).closest("article")?.textContent).toContain("Not downloaded");
        ui.tab("korean").click();
        expect(ui.panel("japanese").hidden).toBe(true);
        expect(ui.panel("korean").hidden).toBe(false);
        expect(ui.panel("korean").textContent).not.toContain("VNTL");
        expect(ui.document.getElementById("deeplToken")).not.toBeNull();
        expect(ui.document.getElementById("devices")).not.toBeNull();
    });

    it("supports arrow, Home and End keys with roving focus", async () => {
        const ui = settingsPage();
        await ui.ready();
        const key = (language: string, key: string) =>
            ui.tab(language).dispatchEvent(new ui.page.window.KeyboardEvent("keydown", { key, bubbles: true }));
        ui.tab("japanese").focus();
        key("japanese", "ArrowRight");
        expect(ui.document.activeElement).toBe(ui.tab("korean"));
        expect(ui.tab("japanese").tabIndex).toBe(-1);
        expect(ui.tab("korean").tabIndex).toBe(0);
        key("korean", "End");
        expect(ui.document.activeElement).toBe(ui.tab("chinese"));
        key("chinese", "Home");
        expect(ui.document.activeElement).toBe(ui.tab("japanese"));
        key("japanese", "ArrowLeft");
        expect(ui.document.activeElement).toBe(ui.tab("chinese"));
    });

    it("saves only the chosen language and preserves that tab", async () => {
        const ui = settingsPage();
        await ui.ready();
        ui.tab("korean").click();
        ui.choice("korean", qwen.id).focus();
        ui.choice("korean", qwen.id).click();
        await vi.waitFor(() => expect(ui.element("message").textContent).toContain("Korean default model saved"));
        expect(ui.saves()).toEqual([
            { path: "/api/settings/mtl-default", method: "PUT", body: { language: "korean", model_id: qwen.id } },
        ]);
        expect(ui.tab("korean").getAttribute("aria-selected")).toBe("true");
        expect(ui.choice("korean", qwen.id).checked).toBe(true);
        expect(ui.choice("japanese", gemma.id).checked).toBe(true);
        expect(ui.choice("korean", qwen.id).disabled).toBe(false);
        expect(ui.document.activeElement).toBe(ui.choice("korean", qwen.id));
    });

    it("keeps tabs usable during a save without allowing overlapping model changes", async () => {
        const ui = settingsPage();
        await ui.ready();
        const release = ui.hold("/api/settings/mtl-default");
        ui.choice("japanese", qwen.id).click();
        expect(ui.element("settings").getAttribute("aria-busy")).toBe("true");
        ui.tab("chinese").click();
        expect(ui.choice("chinese", gemma.id).disabled).toBe(true);
        ui.choice("chinese", gemma.id).click();
        expect(ui.saves()).toHaveLength(1);
        release();
        await vi.waitFor(() => expect(ui.element("settings").getAttribute("aria-busy")).toBe("false"));
        expect(ui.tab("chinese").getAttribute("aria-selected")).toBe("true");
        expect(ui.choice("chinese", gemma.id).disabled).toBe(false);
        expect(ui.choice("japanese", qwen.id).checked).toBe(true);
    });

    it("rolls back rejected choices and never selects missing downloads", async () => {
        const ui = settingsPage();
        await ui.ready();
        ui.choice("japanese", vntl.id).click();
        expect(ui.saves()).toHaveLength(0);
        ui.rejected.set("/api/settings/mtl-default", "Model download was removed.");
        ui.choice("japanese", qwen.id).click();
        await vi.waitFor(() => expect(ui.element("settings").getAttribute("aria-busy")).toBe("false"));
        expect(ui.element("message").textContent).toBe("Model download was removed.");
        expect(ui.choice("japanese", gemma.id).checked).toBe(true);
        expect(ui.choice("japanese", qwen.id).checked).toBe(false);
        expect(ui.choice("japanese", qwen.id).disabled).toBe(false);
    });

    it("does not overwrite a saved default with an older GET response", async () => {
        const ui = settingsPage();
        await ui.ready();
        const release = ui.hold("/api/settings");
        const loading = (ui.page.window as unknown as { load(): Promise<void> }).load();
        ui.choice("japanese", qwen.id).click();
        await vi.waitFor(() => expect(ui.element("message").textContent).toContain("default model saved"));
        release();
        await loading;
        expect(ui.choice("japanese", qwen.id).checked).toBe(true);
    });

    it("treats model names and IDs as data rather than markup or handler code", async () => {
        const data = payload();
        const hostile = { ...qwen, id: "quoted'\"model", name: '<img src=x onerror="window.injected=1">' };
        data.languages.japanese.models.push(hostile);
        const ui = settingsPage(data);
        await ui.ready();
        expect(ui.panel("japanese").querySelector("img")).toBeNull();
        expect(ui.panel("japanese").textContent).toContain(hostile.name);
        ui.choice("japanese", hostile.id).click();
        await vi.waitFor(() => expect(ui.element("message").textContent).toContain("default model saved"));
        expect(ui.saves()[0].body.model_id).toBe(hostile.id);
        expect(ui.errors).toEqual([]);
    });
});

describe("per-model context windows", () => {
    it("saves a model's context across languages without changing any default model", async () => {
        const ui = settingsPage();
        await ui.ready();
        ui.tab("korean").click();
        ui.contextChoice("korean", qwen.id, 16384).focus();
        ui.contextChoice("korean", qwen.id, 16384).click();
        await vi.waitFor(() => expect(ui.element("message").textContent).toContain("Context window saved"));
        expect(ui.saves()).toEqual([
            { path: "/api/settings/mtl-context", method: "PUT", body: { model_id: qwen.id, context_size: 16384 } },
        ]);
        for (const language of ["japanese", "korean", "chinese"]) {
            expect(ui.contextChoice(language, qwen.id, 16384).checked).toBe(true);
            expect(ui.contextChoice(language, gemma.id, 8192).checked).toBe(true);
        }
        expect(ui.choice("korean", gemma.id).checked).toBe(true);
        expect(ui.choice("chinese", qwen.id).checked).toBe(true);
        expect(ui.tab("korean").getAttribute("aria-selected")).toBe("true");
        expect(ui.document.activeElement).toBe(ui.contextChoice("korean", qwen.id, 16384));
        expect(ui.element("message").textContent).toContain("Applies the next time this model is loaded.");
        expect(ui.document.querySelector(".toast")?.textContent).toContain(
            "Applies the next time this model is loaded.",
        );
    });

    it("explains the model limit and prevents selecting unsupported 16K", async () => {
        const ui = settingsPage();
        await ui.ready();
        const choice = ui.contextChoice("japanese", vntl.id, 16384);
        expect(choice.disabled).toBe(true);
        expect(ui.contextChoice("japanese", vntl.id, 8192).checked).toBe(true);
        expect(choice.closest("article")?.textContent).toContain("Model limit: 8,192 tokens.");
        choice.click();
        expect(ui.saves()).toHaveLength(0);
    });

    it("shows the active allocation and pending saved change", async () => {
        const data = payload();
        data.languages.japanese.models = data.languages.japanese.models.map((model) => ({
            ...model,
            context: model.id === qwen.id ? { ...context(), running: 8192 } : model.context,
        }));
        const ui = settingsPage(data);
        await ui.ready();
        ui.contextChoice("japanese", qwen.id, 16384).click();
        await vi.waitFor(() => expect(ui.element("message").textContent).toContain("Context window saved"));
        const card = ui.contextChoice("japanese", qwen.id, 16384).closest("article");
        expect(card?.textContent).toContain("Currently loaded: 8,192 tokens. Change pending.");
        expect(card?.textContent).toContain("Applies the next time this model is loaded.");
    });

    it("keeps an unsupported saved preference visible while explaining the effective cap", async () => {
        const data = payload();
        data.languages.japanese.models = [{ ...vntl, context: { ...context(8192), selected: 16384 } }];
        const ui = settingsPage(data);
        await ui.ready();
        expect(ui.contextChoice("japanese", vntl.id, 16384).checked).toBe(true);
        expect(ui.contextChoice("japanese", vntl.id, 16384).disabled).toBe(true);
        expect(ui.panel("japanese").textContent).toContain("Saved: 16,384; this model will use 8,192 tokens.");
    });

    it("rolls back a failed context save, restores controls, and retains the chosen tab", async () => {
        const ui = settingsPage();
        await ui.ready();
        ui.tab("chinese").click();
        ui.rejected.set("/api/settings/mtl-context", "The model supports at most 8192 tokens.");
        ui.contextChoice("chinese", qwen.id, 16384).click();
        await vi.waitFor(() => expect(ui.element("settings").getAttribute("aria-busy")).toBe("false"));
        expect(ui.element("message").textContent).toContain("at most 8192");
        expect(ui.contextChoice("chinese", qwen.id, 8192).checked).toBe(true);
        expect(ui.contextChoice("chinese", qwen.id, 16384).disabled).toBe(false);
        expect(ui.tab("chinese").getAttribute("aria-selected")).toBe("true");
    });

    it("disables unusable model contexts and shows the backend's reason", async () => {
        const data = payload();
        data.languages.japanese.models = [
            {
                ...vntl,
                context: {
                    ...context(4096),
                    effective: null,
                    options: [],
                    error: "These model weights cannot support the minimum 8192-token context window.",
                },
            },
        ];
        const ui = settingsPage(data);
        await ui.ready();
        expect(ui.contextChoice("japanese", vntl.id, 8192).disabled).toBe(true);
        expect(ui.contextChoice("japanese", vntl.id, 16384).disabled).toBe(true);
        expect(ui.panel("japanese").textContent).toContain("cannot support the minimum 8192-token");
        expect(ui.panel("japanese").textContent).not.toContain("this model will use");
    });
});
