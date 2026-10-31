import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadFontsAndBuildMenu } from "./fonts";
import { state } from "./state";
import type { FontData } from "./state";

const face = (name: string, coverage = [32, 126]): FontData => ({
    font_name: name,
    font_filename: `${name}.ttf`,
    font_format: "truetype",
    font_data_uri: "data:font/ttf;base64,AA==",
    unitsPerEm: 1000,
    ascent: 800,
    descent: 200,
    coverage,
});

describe("font loading", () => {
    beforeEach(() => {
        document.body.innerHTML = '<style id="dynamicFontStyles"></style><div id="fontOptionsContainer"></div>';
        state.currentFontsData = [];
        state.currentSymbolFonts = [];
        state.currentSymbolFont = null;
    });
    afterEach(() => {
        vi.unstubAllGlobals();
        state.currentSymbolFonts = [];
        state.currentSymbolFont = null;
    });

    it("registers every fallback without offering it as text and loads a covered probe", async () => {
        const text = face("Text");
        const fallbacks = [face("First", [0xe001, 0xe001]), face("Second"), face("Third"), face("Noto")];
        const fetcher = vi.fn(async (url: string) => ({
            ok: true,
            status: 200,
            json: async () => (url === "/api/fonts" ? [text] : fallbacks),
        }));
        vi.stubGlobal("fetch", fetcher);
        const load = vi.fn(async () => []);
        Object.defineProperty(document, "fonts", { configurable: true, value: { load } });
        await loadFontsAndBuildMenu();
        expect(state.currentSymbolFonts.map((font) => font.font_name)).toEqual(["First", "Second", "Third", "Noto"]);
        expect(document.querySelectorAll(".font-option-item")).toHaveLength(1);
        const style = document.getElementById("dynamicFontStyles")!.textContent!;
        for (const font of fallbacks) expect(style).toContain(`font-family:'${font.font_name}'`);
        expect(load).toHaveBeenCalledWith("100px 'First'", "\ue001");
    });
});

describe("Settings font controls", () => {
    it("adds, reorders, removes and saves the optional fallbacks", async () => {
        const page = readFileSync("templates/settings.html", "utf8");
        const script = page.match(/<script>([\s\S]*?)<\/script>/)![1];
        const dom = new JSDOM(page, { url: "http://localhost/settings", runScripts: "outside-only" });
        Object.defineProperty(dom.window, "BroadcastChannel", {
            value: class {
                postMessage() {}
                close() {}
            },
        });
        const available = ["Text", "First", "Second", "Third"].map((name) => ({
            font_name: name,
            font_filename: `${name}.ttf`,
            calibrated: true,
            limited_symbol_mapping: name === "First",
        }));
        const saved: string[][] = [];
        dom.window.fetch = (async (url: string, options?: RequestInit) => {
            if (url === "/api/fonts/available")
                return {
                    ok: true,
                    json: async () => [
                        ...available.map((font) => face(font.font_name)),
                        { ...face("Noto"), font_filename: "NotoSansSymbols2-Regular.otf", coverage: [0x2605, 0x2605] },
                    ],
                };
            if (url === "/api/settings/fonts") {
                const selection = JSON.parse(String(options?.body)).symbol_fallbacks as string[];
                saved.push(selection);
                return {
                    ok: true,
                    json: async () => ({
                        fonts: {
                            symbol_fallbacks: selection,
                            active: selection,
                            available,
                        },
                    }),
                };
            }
            return {
                ok: true,
                json: async () => ({
                    fonts: {
                        symbol_fallbacks: [],
                        active: [],
                        available,
                    },
                    languages: {},
                }),
            };
        }) as unknown as typeof dom.window.fetch;
        dom.window.eval(script);
        await new Promise((resolve) => setTimeout(resolve, 0));
        const choose = (index: number, value: string) => {
            dom.window.document.getElementById(`font-trigger-${index}`)!.click();
            const picker = dom.window.document.querySelectorAll(".font-picker")[index];
            const option = Array.from(picker.querySelectorAll<HTMLButtonElement>(".font-option")).find(
                (element) => element.dataset.filename === value,
            )!;
            option.click();
        };
        const click = (selector: string) => {
            const button = dom.window.document.querySelector<HTMLButtonElement>(selector)!;
            const handler = button.getAttribute("onclick");
            if (handler) dom.window.Function(handler).call(button);
            else button.click();
        };
        const selections = () =>
            Array.from(
                dom.window.document.querySelectorAll<HTMLButtonElement>(".font-picker-trigger"),
                (trigger) => trigger.dataset.filename,
            );
        choose(0, "First.ttf");
        choose(1, "Second.ttf");
        choose(2, "Third.ttf");
        expect(selections()).toEqual(["First.ttf", "Second.ttf", "Third.ttf"]);
        expect(dom.window.document.querySelectorAll('.font-option[data-filename="First.ttf"]')).toHaveLength(1);
        expect(
            dom.window.document.querySelector<HTMLElement>('[data-font-face="First.ttf"]')!.style.fontFamily,
        ).toContain("fox-settings-font-1");
        click('[aria-label="Move fallback 1 down"]');
        expect(selections()).toEqual(["Second.ttf", "First.ttf", "Third.ttf"]);
        expect(dom.window.document.getElementById("symbolFallbackFonts")!.textContent).toContain(
            "limited Unicode symbol mappings",
        );
        click("#saveFontsBtn");
        await new Promise((resolve) => setTimeout(resolve, 0));
        expect(saved).toEqual([["Second.ttf", "First.ttf", "Third.ttf"]]);
        expect(dom.window.localStorage.getItem("foxreader:font-settings")).toBeTruthy();
        click('[aria-label="Remove fallback 2"]');
        expect(selections()).toEqual(["Second.ttf", "Third.ttf", ""]);
        expect(dom.window.document.getElementById("symbolFallbackFonts")!.textContent).toContain(
            "Final fallback · always last",
        );
        dom.window.close();
    });
});

describe("character assignments and glyph browsing", () => {
    const windows: JSDOM[] = [];
    afterEach(() => {
        for (const dom of windows.splice(0)) dom.window.close();
    });

    async function controls() {
        const page = readFileSync("templates/settings.html", "utf8");
        const dom = new JSDOM(page, { url: "http://localhost/settings", runScripts: "outside-only" });
        windows.push(dom);
        Object.defineProperty(dom.window, "BroadcastChannel", {
            value: class {
                postMessage() {}
                close() {}
            },
        });
        const available = ["Text", "First", "Second", "Third"].map((name) => ({
            font_name: name,
            font_filename: `${name}.ttf`,
            calibrated: true,
        }));
        const inventory = available.map((font) =>
            face(font.font_name, [32, 126, 0x2605, 0x2605, 0x2661, 0x2661, 0x1f600, 0x1f600]),
        );
        inventory.push({ ...face("Noto"), font_filename: "NotoSansSymbols2-Regular.otf", coverage: [0x2605, 0x2605] });
        const saved: Array<{ symbol_fallbacks: string[]; symbol_characters: Record<string, string> }> = [];
        const glyphRequests: string[] = [];
        const glyphs = [
            { character: "H", codepoint: 0x48, unicode: "U+0048", name: "LATIN CAPITAL LETTER H" },
            { character: "★", codepoint: 0x2605, unicode: "U+2605", name: "BLACK STAR" },
            { character: "♡", codepoint: 0x2661, unicode: "U+2661", name: "WHITE HEART" },
        ];
        dom.window.fetch = (async (url: string, options?: RequestInit) => {
            if (url === "/api/fonts/available") return { ok: true, json: async () => inventory };
            if (url.startsWith("/api/fonts/glyphs?")) {
                glyphRequests.push(url);
                const params = new URL(url, "http://localhost").searchParams;
                const query = (params.get("query") || "").toLowerCase();
                const result = glyphs.filter((glyph) =>
                    `${glyph.character} ${glyph.unicode} ${glyph.name}`.toLowerCase().includes(query),
                );
                return { ok: true, json: async () => ({ glyphs: result, total: result.length, offset: 0, limit: 96 }) };
            }
            if (url === "/api/settings/fonts") {
                const body = JSON.parse(String(options?.body));
                saved.push(body);
                return {
                    ok: true,
                    json: async () => ({
                        fonts: {
                            ...body,
                            active: body.symbol_fallbacks,
                            available,
                            final_fallback: "NotoSansSymbols2-Regular.otf",
                        },
                    }),
                };
            }
            return {
                ok: true,
                json: async () => ({
                    fonts: {
                        symbol_fallbacks: [],
                        active: [],
                        available,
                        symbol_characters: {},
                        final_fallback: "NotoSansSymbols2-Regular.otf",
                    },
                    languages: {},
                }),
            };
        }) as unknown as typeof dom.window.fetch;
        dom.window.eval(page.match(/<script>([\s\S]*?)<\/script>/)![1]);
        await vi.waitFor(() => expect(dom.window.document.getElementById("settingsFontFaces")).not.toBeNull());
        const choose = (index: number, filename: string) => {
            dom.window.document.getElementById(`font-trigger-${index}`)!.click();
            Array.from(
                dom.window.document.querySelectorAll<HTMLButtonElement>(
                    `.font-picker[data-font-index="${index}"] .font-option`,
                ),
            )
                .find((option) => option.dataset.filename === filename)!
                .click();
        };
        const paste = (filename: string, text: string) => {
            const input = Array.from(dom.window.document.querySelectorAll<HTMLInputElement>("[data-font-paste]")).find(
                (element) => element.dataset.fontPaste === filename,
            )!;
            input.value = text;
            input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
            input
                .closest(".font-character-editor")!
                .querySelector<HTMLButtonElement>('[data-font-action="add-characters"]')!
                .click();
        };
        const editor = (filename: string) =>
            Array.from(dom.window.document.querySelectorAll<HTMLElement>("[data-font-editor]")).find(
                (element) => element.dataset.fontEditor === filename,
            )!;
        const chips = (filename: string) =>
            Array.from(
                editor(filename).querySelectorAll<HTMLButtonElement>(".character-chip"),
                (chip) => chip.dataset.character,
            );
        return { dom, choose, paste, editor, chips, saved, glyphRequests };
    }

    it("searches font names, supports keyboard selection and uses each font face", async () => {
        const { dom } = await controls();
        const trigger = dom.window.document.getElementById("font-trigger-0")!;
        trigger.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }));
        const search = dom.window.document.querySelector<HTMLInputElement>(
            '.font-picker[data-font-index="0"] .font-search',
        )!;
        expect(dom.window.document.activeElement).toBe(search);
        search.value = "second";
        search.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
        const options = Array.from(
            dom.window.document.querySelectorAll<HTMLButtonElement>('.font-picker[data-font-index="0"] .font-option'),
        ).filter((option) => !option.hidden);
        expect(options).toHaveLength(1);
        expect(options[0].querySelector<HTMLElement>(".font-display")!.style.fontFamily).toContain(
            "fox-settings-font-2",
        );
        search.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }));
        expect(dom.window.document.activeElement).toBe(options[0]);
        options[0].click();
        expect(dom.window.document.getElementById("font-trigger-0")!.dataset.filename).toBe("Second.ttf");
        dom.window.document.getElementById("font-trigger-0")!.click();
        dom.window.document
            .querySelector<HTMLInputElement>(".font-search")!
            .dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
        expect(dom.window.document.getElementById("font-trigger-0")!.getAttribute("aria-expanded")).toBe("false");
    });

    it("rejects duplicate ownership and overflow, counts emoji once, and saves valid rules", async () => {
        const { dom, choose, paste, editor, chips, saved } = await controls();
        choose(0, "First.ttf");
        choose(1, "Second.ttf");
        choose(2, "Third.ttf");
        paste("First.ttf", "HH ");
        expect(chips("First.ttf")).toEqual(["H"]);
        paste("Second.ttf", "H");
        expect(editor("Second.ttf").textContent).toContain("already assigned to First.ttf");
        expect(chips("Second.ttf")).toEqual([]);
        paste("Second.ttf", "");
        paste("First.ttf", "ABCDEFGHIJKLMNOPQRSTUVWXY");
        expect(chips("First.ttf")).toHaveLength(25);
        paste("First.ttf", "Z");
        expect(editor("First.ttf").textContent).toContain("at most 25");
        paste("First.ttf", "");
        paste("Third.ttf", "😀😀");
        expect(chips("Third.ttf")).toEqual(["😀"]);
        paste("NotoSansSymbols2-Regular.otf", "★");
        expect(chips("NotoSansSymbols2-Regular.otf")).toEqual(["★"]);
        const save = dom.window.document.getElementById("saveFontsBtn")!;
        dom.window.Function(save.getAttribute("onclick")!).call(save);
        await vi.waitFor(() => expect(saved).toHaveLength(1));
        expect(saved[0].symbol_characters).toEqual({
            "First.ttf": "HABCDEFGIJKLMNOPQRSTUVWXY",
            "Third.ttf": "😀",
            "NotoSansSymbols2-Regular.otf": "★",
        });
    });

    it("browses supported glyphs, caches searches, and disables another font's assigned glyph", async () => {
        const { dom, choose, editor, chips, glyphRequests } = await controls();
        choose(0, "First.ttf");
        choose(1, "Second.ttf");
        editor("First.ttf").querySelector<HTMLButtonElement>('[data-font-action="browse-glyphs"]')!.click();
        await vi.waitFor(() => expect(editor("First.ttf").querySelectorAll(".glyph-button")).toHaveLength(3));
        const search = editor("First.ttf").querySelector<HTMLInputElement>("[data-glyph-search]")!;
        search.value = "star";
        search.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
        await vi.waitFor(() => expect(editor("First.ttf").querySelectorAll(".glyph-button")).toHaveLength(1));
        editor("First.ttf").querySelector<HTMLButtonElement>(".glyph-button")!.click();
        expect(chips("First.ttf")).toEqual(["★"]);
        const calls = glyphRequests.length;
        editor("First.ttf").querySelector<HTMLButtonElement>('[data-font-action="browse-glyphs"]')!.click();
        editor("First.ttf").querySelector<HTMLButtonElement>('[data-font-action="browse-glyphs"]')!.click();
        await vi.waitFor(() => expect(editor("First.ttf").querySelectorAll(".glyph-button")).toHaveLength(1));
        expect(glyphRequests).toHaveLength(calls);
        editor("Second.ttf").querySelector<HTMLButtonElement>('[data-font-action="browse-glyphs"]')!.click();
        await vi.waitFor(() => expect(editor("Second.ttf").querySelectorAll(".glyph-button")).toHaveLength(3));
        const star = Array.from(editor("Second.ttf").querySelectorAll<HTMLButtonElement>(".glyph-button")).find(
            (button) => button.dataset.character === "★",
        )!;
        expect(star.disabled).toBe(true);
        editor("First.ttf").querySelector<HTMLButtonElement>(".character-chip")!.click();
        const released = Array.from(editor("Second.ttf").querySelectorAll<HTMLButtonElement>(".glyph-button")).find(
            (button) => button.dataset.character === "★",
        )!;
        expect(released.disabled).toBe(false);
    });

    it("preserves pasted drafts and keyboard focus while choosing or removing glyphs", async () => {
        const { dom, choose, editor, chips } = await controls();
        choose(0, "First.ttf");
        editor("First.ttf").querySelector<HTMLButtonElement>('[data-font-action="browse-glyphs"]')!.click();
        await vi.waitFor(() => expect(editor("First.ttf").querySelectorAll(".glyph-button")).toHaveLength(3));
        const draft = editor("First.ttf").querySelector<HTMLInputElement>("[data-font-paste]")!;
        draft.value = "AB";
        draft.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
        const star = Array.from(editor("First.ttf").querySelectorAll<HTMLButtonElement>(".glyph-button")).find(
            (button) => button.dataset.character === "★",
        )!;
        star.focus();
        star.click();
        expect(chips("First.ttf")).toEqual(["★"]);
        expect(editor("First.ttf").querySelector<HTMLInputElement>("[data-font-paste]")!.value).toBe("AB");
        expect((dom.window.document.activeElement as HTMLElement).dataset.character).toBe("★");
        const chip = editor("First.ttf").querySelector<HTMLButtonElement>(".character-chip")!;
        chip.focus();
        chip.click();
        const input = editor("First.ttf").querySelector<HTMLInputElement>("[data-font-paste]")!;
        expect(dom.window.document.activeElement).toBe(input);
        expect(input.value).toBe("AB");
        input.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
        expect(chips("First.ttf")).toEqual(["A", "B"]);
        expect(dom.window.document.activeElement).toBe(editor("First.ttf").querySelector("[data-font-paste]"));
        expect((dom.window.document.activeElement as HTMLInputElement).value).toBe("");
    });
});
