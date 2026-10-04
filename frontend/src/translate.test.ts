import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderTranslationUI, handleAutoTranslation, translationPanelState, translationPanelText } from "./translate";
import { addNewEntries } from "./entries";
import { state as appState } from "./state";
import * as api from "./api";
import type { CharacterInfo } from "./types";

vi.mock("./api", () => ({ translate: vi.fn() }));

vi.mock("./characters", async (importOriginal) => {
    const actual = await importOriginal<typeof import("./characters")>();
    return {
        ...actual,
        getCharacters: () => mockRoster,
    };
});

let mockRoster: CharacterInfo[] = [];

beforeEach(() => {
    mockRoster = [];
});

function panel(markup: string): HTMLElement {
    document.body.innerHTML = markup;
    return document.getElementById("translatedText")!;
}

describe("the translation panel's state", () => {
    beforeEach(() => {
        document.body.innerHTML = "";
    });

    it("reads the state stamped by renderTranslationUI", () => {
        panel(`<div id="translatedText" class="panel"></div>`);

        renderTranslationUI({ error: "Translator not reachable" });
        expect(translationPanelState()).toBe("error");
        expect(document.getElementById("translatedText")!.contentEditable).toBe("false");

        renderTranslationUI({ translating: true });
        expect(translationPanelState()).toBe("translating");

        renderTranslationUI({ translated: "Hello" });
        expect(translationPanelState()).toBe("text");
        expect(document.getElementById("translatedText")!.contentEditable).toBe("true");
    });

    it("keeps the class the stylesheet paints, alongside the state", () => {
        const p = panel(`<div id="translatedText" class="panel"></div>`);

        renderTranslationUI({ error: "nope" });
        expect(p.className).toBe("panel error");
        renderTranslationUI({ translating: true });
        expect(p.className).toBe("panel translating");
        renderTranslationUI({ translated: "" });
        expect(p.className).toBe("panel translated");
    });

    it("falls back to the class on markup renderTranslationUI has not touched yet", () => {
        // What the Preact layout renders before any translator is picked.
        expect(
            translationPanelState(
                panel(`<div id="translatedText" class="panel error"><strong>Select a translator</strong></div>`),
            ),
        ).toBe("error");
        expect(
            translationPanelState(panel(`<div id="translatedText" class="panel translating">Translating...</div>`)),
        ).toBe("translating");
        expect(translationPanelState(panel(`<div id="translatedText" class="panel translated">Hi</div>`))).toBe("text");
        // A bare panel is not an error: it is text, and there is none yet.
        expect(translationPanelState(panel(`<div id="translatedText" class="panel"></div>`))).toBe("text");
    });

    it("prefers the stamped state over a stale class", () => {
        const p = panel(`<div id="translatedText" class="panel error" data-state="text">Kept</div>`);
        expect(translationPanelState(p)).toBe("text");
        expect(translationPanelText(p)).toBe("Kept");
    });

    it("reads no text out of a panel that is showing a message", () => {
        expect(
            translationPanelText(panel(`<div id="translatedText" class="panel error">Select a translator</div>`)),
        ).toBe("");
        expect(
            translationPanelText(panel(`<div id="translatedText" class="panel translating">Translating...</div>`)),
        ).toBe("");
    });

    it("reads the trimmed text out of a panel that is showing some", () => {
        expect(
            translationPanelText(panel(`<div id="translatedText" class="panel translated">  Hi there\n</div>`)),
        ).toBe("Hi there");
        expect(translationPanelText(panel(`<div id="translatedText" class="panel">  typed  </div>`))).toBe("typed");
    });

    it("treats a layout with no panel as empty text rather than throwing", () => {
        document.body.innerHTML = "";
        expect(translationPanelState()).toBe("text");
        expect(translationPanelText()).toBe("");
        expect(() => renderTranslationUI({ translated: "x" })).not.toThrow();
    });
});

describe("handleAutoTranslation honours the Auto TL switch", () => {
    const translate = vi.mocked(api.translate);

    function autoDom(): void {
        document.body.innerHTML = `
            <textarea id="textArea">こんにちは</textarea>
            <div class="translator-buttons">
                <button class="active-translator" data-endpoint="/translate/x" data-lang="JA" data-lang-group="japanese">X</button>
            </div>
            <div id="translatedText" class="panel"></div>`;
    }

    beforeEach(() => {
        document.body.innerHTML = "";
        vi.clearAllMocks();
        translate.mockResolvedValue({ original: "こんにちは", translated: "hi", alt_translated: "" });
        appState.isAutoTranslateEnabled = true;
    });

    it("translates through the active button while the switch is on", async () => {
        autoDom();
        handleAutoTranslation();
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
    });

    it("stays idle while the switch is off", async () => {
        appState.isAutoTranslateEnabled = false;
        autoDom();
        handleAutoTranslation();
        await new Promise((resolve) => setTimeout(resolve, 20));
        expect(translate).not.toHaveBeenCalled();
    });

    it("still translates an explicit request while the switch is off", async () => {
        appState.isAutoTranslateEnabled = false;
        autoDom();
        handleAutoTranslation({ force: true });
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
    });
});

describe("custom endpoint context", () => {
    const translate = vi.mocked(api.translate);

    function customDom(endpoint: string, lang: string): void {
        document.body.innerHTML = `
            <textarea id="textArea">text</textarea>
            <div class="translator-buttons">
                <button class="active-translator" data-endpoint="${endpoint}" data-lang="${lang}" data-lang-group="japanese">X</button>
            </div>
            <div id="translatedText" class="panel"></div>`;
    }

    beforeEach(() => {
        document.body.innerHTML = "";
        vi.clearAllMocks();
        translate.mockResolvedValue({ original: "", translated: "hi", alt_translated: "" });
        appState.isAutoTranslateEnabled = true;
        appState.isContextEnabled = true;
        appState.isCharactersEnabled = true;
        appState.pageEntriesCache = {};
        appState.currentImageFile = "p.png";
        appState.currentTrackingColorIdx = 0;
        addNewEntries([
            { type: "rectangle", coords: { x: 200, y: 10, w: 40, h: 40 } },
            { type: "rectangle", coords: { x: 10, y: 10, w: 40, h: 40 } },
        ]);
        // RTL: the 200-region is first; it holds the pair, the second is translated.
        const [first, second] = appState.pageEntriesCache["p.png"];
        first.ocr_text = "a";
        first.text = "b";
        appState.currentlySelectedEntryId = second.id;
    });

    it("sends pairs for a custom button while Context is on", async () => {
        customDom("/translate/custom", "ep1");
        handleAutoTranslation();
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
        expect(translate).toHaveBeenCalledWith("/translate/custom", "text", "ep1", "japanese", [["a", "b"]]);
    });

    it("sends nothing for a built-in button", async () => {
        customDom("/translate/deepl", "JA");
        handleAutoTranslation();
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
        expect(translate).toHaveBeenCalledWith("/translate/deepl", "text", "JA", "japanese", undefined);
    });

    it("sends nothing for a custom button while Context is off", async () => {
        appState.isContextEnabled = false;
        customDom("/translate/custom", "ep1");
        handleAutoTranslation();
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
        expect(translate).toHaveBeenCalledWith("/translate/custom", "text", "ep1", "japanese", undefined);
    });

    it("never sends character extras on a built-in button", async () => {
        // Built-in engines have no character fields: the 5-arg shape is the
        // contract, roster or no roster.
        mockRoster = [{ meta_id: "a", name_en: "Aiko" }];
        customDom("/translate/deepl", "JA");
        handleAutoTranslation();
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
        expect(translate.mock.calls[0]).toHaveLength(5);
    });

    it("sends character extras on a custom button while Characters is on", async () => {
        mockRoster = [
            { meta_id: "a", name_en: "Aiko" },
            { meta_id: "b", name_en: "Ken" },
        ];
        const [first, second] = appState.pageEntriesCache["p.png"];
        first.character_id = "a";
        second.character_id = "b";
        appState.isCharactersEnabled = true;
        customDom("/translate/custom", "ep1");
        handleAutoTranslation();
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
        expect(translate).toHaveBeenCalledWith("/translate/custom", "text", "ep1", "japanese", [["a", "b"]], {
            character_info: mockRoster,
            context_character_links: ["a"],
            meta_id: "b",
        });
    });

    it("sends no character extras on custom while Characters is off", async () => {
        mockRoster = [{ meta_id: "a", name_en: "Aiko" }];
        appState.isCharactersEnabled = false;
        customDom("/translate/custom", "ep1");
        handleAutoTranslation();
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
        expect(translate).toHaveBeenCalledWith("/translate/custom", "text", "ep1", "japanese", [["a", "b"]]);
        expect(translate.mock.calls[0]).toHaveLength(5);
    });
});

describe("ML character extras", () => {
    const translate = vi.mocked(api.translate);

    function mlDom(): void {
        document.body.innerHTML = `
            <textarea id="textArea">text</textarea>
            <input type="checkbox" id="enableML" checked />
            <div class="translator-buttons">
                <button class="active-translator" data-endpoint="/translate/ml" data-lang="japanese" data-lang-group="japanese" data-requires-ml="true">MTL</button>
            </div>
            <div id="translatedText" class="panel"></div>`;
    }

    beforeEach(() => {
        document.body.innerHTML = "";
        vi.clearAllMocks();
        translate.mockResolvedValue({ original: "", translated: "hi", alt_translated: "" });
        appState.isAutoTranslateEnabled = true;
        appState.isContextEnabled = true;
        appState.isCharactersEnabled = true;
        appState.pageEntriesCache = {};
        appState.currentImageFile = "p.png";
        appState.currentTrackingColorIdx = 0;
        addNewEntries([
            { type: "rectangle", coords: { x: 200, y: 10, w: 40, h: 40 } },
            { type: "rectangle", coords: { x: 10, y: 10, w: 40, h: 40 } },
        ]);
        const [first, second] = appState.pageEntriesCache["p.png"];
        first.ocr_text = "a";
        first.text = "b";
        appState.currentlySelectedEntryId = second.id;
    });

    it("sends context in the 5-arg shape while the roster is empty", async () => {
        mlDom();
        handleAutoTranslation();
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
        // Empty roster: no character fields ride along, so the call keeps the
        // exact shape older backends (and older tests) expect.
        expect(translate).toHaveBeenCalledWith("/translate/ml", "text", "japanese", "japanese", [["a", "b"]]);
    });

    it("sends no context while Context is off, and still no extras", async () => {
        appState.isContextEnabled = false;
        mlDom();
        handleAutoTranslation();
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
        expect(translate).toHaveBeenCalledWith("/translate/ml", "text", "japanese", "japanese", undefined);
    });

    it("sends no character extras while Characters is off", async () => {
        appState.isCharactersEnabled = false;
        mlDom();
        handleAutoTranslation();
        await vi.waitFor(() => expect(translate).toHaveBeenCalledTimes(1));
        // Context still rides along, but the 6th argument (character_info /
        // links / speaker) is omitted entirely -- the backend then translates
        // without any character data and returns plain text.
        expect(translate).toHaveBeenCalledWith("/translate/ml", "text", "japanese", "japanese", [["a", "b"]]);
        expect(translate.mock.calls[0]).toHaveLength(5);
    });
});
