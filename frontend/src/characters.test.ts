import { beforeEach, describe, expect, it, vi } from "vitest";
import { characterLabel, characterPickerLabel, charactersToCsv, parseCharactersCsv } from "./characters";
import { ensureEntryDefaults } from "./entryOptions";
import { PLAIN_GEOMETRY, state } from "./state";
import type { Entry } from "./state";

vi.mock("./characters", async (importOriginal) => {
    const actual = await importOriginal<typeof import("./characters")>();
    return {
        ...actual,
        getCharacters: () => mockRoster,
    };
});

import { buildTranslationCharacters } from "./translationContext";
import type { CharacterInfo } from "./types";

let mockRoster: CharacterInfo[] = [];
let seq = 0;

function entry(ocr_text: string, text: string, character_id: string | null = null): Entry {
    seq += 1;
    return {
        id: `e${seq}`,
        ocr_text,
        text,
        fontfile: "",
        fontname: "",
        region: { type: "rectangle", coords: { x: 0, y: 0, w: 10, h: 10 } },
        color: "#fff",
        text_align: "center",
        visible: true,
        layer: 1,
        character_id,
        font_size: null,
        font_color: null,
        stroke_width: null,
        stroke_color: null,
        bg_mode: "auto",
        bg_color: null,
        clean: {
            method: "region",
            fill: "hybrid-level",
            glow: true,
            transport: true,
            speed: "fastest",
            tta: true,
            tile: 0,
        },
        ...PLAIN_GEOMETRY,
    };
}

beforeEach(() => {
    vi.restoreAllMocks();
    seq = 0;
    mockRoster = [];
    state.pageEntriesCache = {};
    state.currentImageFile = "p.png";
});

describe("character CSV", () => {
    it("parses the documented columns and skips blank rows", () => {
        const rows = parseCharactersCsv("name_en,name_ja,gender,alias_en,alias_ja\nAiko,愛子,female,Sis,\n\nKen,,,,\n");
        expect(rows).toEqual([
            { name_en: "Aiko", name_ja: "愛子", gender: "female", alias_en: "Sis", alias_ja: "" },
            { name_en: "Ken", name_ja: "", gender: "", alias_en: "", alias_ja: "" },
        ]);
    });

    it("tolerates BOM, casing and whitespace in the header", () => {
        const rows = parseCharactersCsv("﻿ Name_EN , NAME_JA , Gender , Alias_EN , Alias_JA \nAiko,愛子,FEMALE,,\n");
        expect(rows).toEqual([{ name_en: "Aiko", name_ja: "愛子", gender: "female", alias_en: "", alias_ja: "" }]);
    });

    it("drops non-standard genders and caps field length", () => {
        const long = "x".repeat(500);
        const rows = parseCharactersCsv(`name_en,name_ja,gender,alias_en,alias_ja\n${long},,nonbinary,,\n`);
        expect(rows).toHaveLength(1);
        expect(rows[0].gender).toBe("");
        expect(rows[0]?.name_en?.length).toBeLessThanOrEqual(100);
    });

    it("round-trips commas through quoting", () => {
        const text = charactersToCsv([
            { meta_id: "a", name_en: "A, B", name_ja: null, gender: null, alias_en: null, alias_ja: null },
        ]);
        expect(text.split("\n")[0]).toBe("name_en,name_ja,gender,alias_en,alias_ja");
        expect(parseCharactersCsv(text)[0].name_en).toBe("A, B");
    });

    it("returns nothing for empty input", () => {
        expect(parseCharactersCsv("")).toEqual([]);
        expect(parseCharactersCsv("name_en,name_ja,gender,alias_en,alias_ja\n")).toEqual([]);
    });
});

describe("characterLabel", () => {
    it("prefers EN, then JA, then alias, then the id", () => {
        expect(characterLabel({ meta_id: "abcdef", name_en: "Aiko", name_ja: "愛子" })).toBe("Aiko");
        expect(characterLabel({ meta_id: "abcdef", name_en: null, name_ja: "愛子" })).toBe("愛子");
        expect(characterLabel({ meta_id: "abcdef", alias_en: "Sis" })).toBe("Sis");
        expect(characterLabel({ meta_id: "abcdef12" })).toBe("Character abcdef");
    });

    it("stays gender-free: symbols belong to the picker label", () => {
        expect(characterLabel({ meta_id: "e", name_en: "Eren", gender: "male" })).toBe("Eren");
    });
});

describe("characterPickerLabel", () => {
    it("appends the gender symbol when one is known", () => {
        expect(characterPickerLabel({ meta_id: "e", name_en: "Eren", gender: "male" })).toBe("Eren (♂)");
        expect(characterPickerLabel({ meta_id: "m", name_en: "Mikasa", gender: "FEMALE" })).toBe("Mikasa (♀)");
        expect(characterPickerLabel({ meta_id: "a", name_en: "Annie", gender: null })).toBe("Annie");
        expect(characterPickerLabel({ meta_id: "x", name_en: "Ken", gender: "other" })).toBe("Ken");
    });
});

describe("entry speaker defaults", () => {
    it("back-fills missing character_id as none", () => {
        const bare = { id: "x" } as unknown as Entry;
        expect(ensureEntryDefaults(bare).character_id).toBeNull();
    });

    it("keeps a stored speaker and clears garbage", () => {
        expect(ensureEntryDefaults(entry("a", "b", "abc")).character_id).toBe("abc");
        const dirty = entry("a", "b");
        (dirty as unknown as Record<string, unknown>).character_id = 42;
        expect(ensureEntryDefaults(dirty).character_id).toBeNull();
    });
});

describe("buildTranslationCharacters", () => {
    it("links earlier speakers and names the target speaker", () => {
        mockRoster = [
            { meta_id: "a", name_en: "Aiko" },
            { meta_id: "b", name_en: "Ken" },
        ];
        const first = entry("おはよう", "Morning", "a");
        const skipped = entry("ざわざわ", "", "b");
        const second = entry("元気?", "How are you?", "b");
        const target = entry("さようなら", "", "a");
        state.pageEntriesCache["p.png"] = [first, skipped, second, target];

        const payload = buildTranslationCharacters(target.id);
        expect(payload.character_info).toEqual(mockRoster);
        // Skipped (untranslated) entries contribute no link.
        expect(payload.context_character_links).toEqual(["a", "b"]);
        expect(payload.meta_id).toBe("a");
    });

    it("reads unknown ids as no speaker", () => {
        mockRoster = [{ meta_id: "a", name_en: "Aiko" }];
        const first = entry("a", "b", "deleted-id");
        const target = entry("c", "", "also-gone");
        state.pageEntriesCache["p.png"] = [first, target];

        const payload = buildTranslationCharacters(target.id);
        expect(payload.context_character_links).toEqual([null]);
        expect(payload.meta_id).toBeNull();
        expect(payload.character_info).toEqual(mockRoster);
    });

    it("is empty without a roster, a page, or a match", () => {
        const target = entry("a", "");
        state.pageEntriesCache["p.png"] = [target];
        expect(buildTranslationCharacters(target.id)).toEqual({
            character_info: [],
            context_character_links: [],
            meta_id: null,
        });
        expect(buildTranslationCharacters(null)).toEqual({
            character_info: [],
            context_character_links: [],
            meta_id: null,
        });
        state.currentImageFile = "";
        expect(buildTranslationCharacters(target.id).character_info).toEqual([]);
    });
});
