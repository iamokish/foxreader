import { beforeEach, describe, expect, it, vi } from "vitest";
import { MAX_CONTEXT_PAIRS, buildTranslationContext } from "./translationContext";
import { PLAIN_GEOMETRY, state } from "./state";
import type { Entry } from "./state";

let seq = 0;

function entry(ocr_text: string, text: string): Entry {
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
    state.pageEntriesCache = {};
    state.currentImageFile = "p.png";
});

describe("buildTranslationContext", () => {
    it("collects earlier pairs in order, skipping the untranslated", () => {
        const a = entry("おはよう", "Good morning");
        const skipped = entry("ざわざわ", "");
        const b = entry("元気？", "How are you?");
        const target = entry("さようなら", "");
        state.pageEntriesCache["p.png"] = [a, skipped, b, target];
        expect(buildTranslationContext(target.id)).toEqual([
            ["おはよう", "Good morning"],
            ["元気？", "How are you?"],
        ]);
    });

    it("yields nothing for the first entry", () => {
        const target = entry("a", "");
        state.pageEntriesCache["p.png"] = [target];
        expect(buildTranslationContext(target.id)).toEqual([]);
    });

    it("excludes unchecked (hidden) predecessors", () => {
        const hidden = entry("かくれた", "Hidden");
        hidden.visible = false;
        const shown = entry("みえる", "Visible");
        const target = entry("つぎ", "");
        state.pageEntriesCache["p.png"] = [hidden, shown, target];
        expect(buildTranslationContext(target.id)).toEqual([["みえる", "Visible"]]);
    });

    it("yields nothing without a target, a page, or a match", () => {
        state.pageEntriesCache["p.png"] = [entry("a", "A")];
        expect(buildTranslationContext(null)).toEqual([]);
        expect(buildTranslationContext("missing")).toEqual([]);
        state.currentImageFile = "";
        expect(buildTranslationContext("e1")).toEqual([]);
    });

    it("trims sides and drops blanks", () => {
        const a = entry("  おはよう  ", "  Good morning\n");
        const blank = entry("   ", "   ");
        const target = entry("x", "");
        state.pageEntriesCache["p.png"] = [a, blank, target];
        expect(buildTranslationContext(target.id)).toEqual([["おはよう", "Good morning"]]);
    });

    it(`caps at the newest ${10} pairs`, () => {
        const entries = Array.from({ length: MAX_CONTEXT_PAIRS + 4 }, (_, i) => entry(`s${i}`, `e${i}`));
        const target = entry("t", "");
        state.pageEntriesCache["p.png"] = [...entries, target];
        const pairs = buildTranslationContext(target.id);
        expect(pairs).toHaveLength(MAX_CONTEXT_PAIRS);
        expect(pairs[0]).toEqual(["s4", "e4"]);
        expect(pairs[pairs.length - 1]).toEqual([`s${MAX_CONTEXT_PAIRS + 3}`, `e${MAX_CONTEXT_PAIRS + 3}`]);
    });
});
