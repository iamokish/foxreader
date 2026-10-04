/**
 * The per-entry options panel.
 *
 * These cover the four controls that were reported as dead -- font size, font
 * colour, outline width, outline/background colour -- plus the row packing that
 * replaced one-control-per-line. Every case here failed before the fix:
 * the numeric controls shipped as sliders that were `disabled` whenever the value
 * was `null` (which is every entry's starting state), and the colour popover
 * compared two `undefined`s to decide it was already open, so the first click
 * closed a popover that had never been built.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { buildEntryOptions, defaultCleanOptions, ensureEntryDefaults, normalizeHex } from "./entryOptions";
import type { OptionsContext } from "./entryOptions";
import {
    state,
    MIN_FONT_SIZE,
    MAX_FONT_SIZE,
    MIN_STROKE_WIDTH,
    MAX_STROKE_WIDTH,
    MIN_WORD_SPACING,
    MAX_WORD_SPACING,
    MIN_LINE_SPACING,
    MAX_LINE_SPACING,
    MIN_FONT_SCALE,
    MAX_FONT_SCALE,
    MAX_TEXT_ANGLE,
    PLAIN_GEOMETRY,
} from "./state";
import type { Entry } from "./state";
import type { CharacterInfo } from "./types";

vi.mock("./characters", async (importOriginal) => {
    const actual = await importOriginal<typeof import("./characters")>();
    return {
        ...actual,
        getCharacters: () => mockRoster,
    };
});

let mockRoster: CharacterInfo[] = [];

function makeEntry(over: Partial<Entry> = {}): Entry {
    return {
        id: "e1",
        ocr_text: "",
        text: "",
        fontfile: "a.ttf",
        fontname: "'A', sans-serif",
        region: {
            type: "polygon",
            coords: [
                { x: 0, y: 0 },
                { x: 20, y: 0 },
                { x: 20, y: 20 },
            ],
        },
        color: "#00FF66",
        text_align: "center",
        visible: true,
        layer: 1,
        font_size: null,
        font_color: null,
        stroke_width: null,
        stroke_color: null,
        bg_mode: "auto",
        bg_color: null,
        clean: defaultCleanOptions(),
        ...PLAIN_GEOMETRY,
        ...over,
    };
}

function makeCtx(entry: Entry): OptionsContext {
    return { entries: [entry], refresh: vi.fn(), rebuild: vi.fn(), refreshAll: vi.fn() };
}

/** The panel, mounted -- popovers anchor to `<body>`, so it has to be in the DOM. */
function mount(entry: Entry, ctx = makeCtx(entry)): HTMLElement {
    const panel = buildEntryOptions(entry, ctx);
    document.body.appendChild(panel);
    return panel;
}

const rowsOf = (panel: HTMLElement): HTMLElement[] =>
    Array.from(panel.querySelectorAll<HTMLElement>(".eo-grid > .eo-row"));

const labelsOf = (row: HTMLElement): string[] =>
    Array.from(row.querySelectorAll(".eo-label")).map((n) => n.textContent ?? "");

beforeEach(() => {
    document.body.innerHTML = "";
    mockRoster = [];
    state.currentFontsData = [
        { font_filename: "a.ttf", font_name: "Alpha", font_data_uri: "", font_format: "truetype" },
        { font_filename: "b.ttf", font_name: "Beta", font_data_uri: "", font_format: "truetype" },
    ];
});

describe("normalizeHex", () => {
    it("expands the short form and lower-cases", () => {
        expect(normalizeHex("#F0C")).toBe("#ff00cc");
        expect(normalizeHex("FF00CC")).toBe("#ff00cc");
        expect(normalizeHex("  #ff00cc  ")).toBe("#ff00cc");
    });

    it("rejects anything that is not a hex colour", () => {
        for (const bad of ["", "#", "#12", "#12345", "#1234567", "red", "#gggggg"]) {
            expect(normalizeHex(bad)).toBeNull();
        }
    });
});

describe("row packing", () => {
    it("puts layer with align and speaker, and the outline on the font row", () => {
        const rows = rowsOf(mount(makeEntry()));

        expect(labelsOf(rows[0])).toEqual(["Layer", "Align", "Speaker"]);
        // The family select is the one wide control in the panel and it used to
        // sit alone, with the outline on a row of its own below it.
        expect(labelsOf(rows[1])).toEqual(["Font", "Outline"]);
        // Geometry in the order it is applied: fitted first, then post-fit.
        expect(labelsOf(rows[2])).toEqual(["Spacing", "Scale"]);
        expect(labelsOf(rows[3])).toEqual(["Shift", "Rotate"]);
        // BG stays last: the two rows above it are the ones that deliberately
        // leave it alone, so they read as adjustments to the text above them.
        expect(labelsOf(rows[4])).toEqual(["BG"]);
        expect(rows).toHaveLength(5);

        const fields = Array.from(rows[1].querySelectorAll<HTMLElement>(".eo-field"));
        // Font family, font size, font colour.
        expect(fields[0].querySelectorAll("select")).toHaveLength(2);
        expect(fields[0].querySelectorAll(".eo-color-btn")).toHaveLength(1);
        // Outline width and outline colour, hugging their content so the family
        // select keeps the leftover width.
        expect(fields[1].querySelectorAll("select")).toHaveLength(1);
        expect(fields[1].querySelectorAll(".eo-color-btn")).toHaveLength(1);
        expect(fields[1].classList.contains("eo-field-hug")).toBe(true);
    });

    it("sizes the font group by the panel, not by the longest family name", () => {
        // A select is as wide as its widest option, so installing one long family
        // name used to move the point where the outline group wrapped -- it needed
        // a panel half again as wide before the two shared a line. The basis is the
        // group's own minimum instead, and it still grows into leftover width.
        const groups = Array.from(rowsOf(mount(makeEntry()))[1].querySelectorAll<HTMLElement>(".eo-group"));
        expect(groups[0].classList.contains("eo-group-fluid")).toBe(true);
        expect(groups[1].classList.contains("eo-group-fluid")).toBe(false);
    });

    it("keeps each label with its own fields, so a group cannot break in half", () => {
        // A wrapping flex row whose labels and fields are bare siblings can break
        // between them: "Outline" ended one line with its two controls stranded on
        // the next. Each group is its own flex item now.
        const groups = Array.from(rowsOf(mount(makeEntry()))[1].querySelectorAll<HTMLElement>(".eo-group"));
        expect(groups).toHaveLength(2);
        for (const group of groups) {
            expect(group.querySelectorAll(".eo-label")).toHaveLength(1);
            expect(group.querySelectorAll(".eo-field")).toHaveLength(1);
        }
    });

    it("adds the background colour beside the mode select, not below it", () => {
        const rows = rowsOf(mount(makeEntry({ bg_mode: "color", bg_color: "#ffffff" })));
        const bgRow = rows[rows.length - 1];
        expect(labelsOf(bgRow)).toEqual(["BG"]);
        expect(bgRow.querySelectorAll("select")).toHaveLength(1);
        expect(bgRow.querySelectorAll(".eo-color-btn")).toHaveLength(1);
    });
});

describe("speaker select", () => {
    const speakerOf = (panel: HTMLElement): HTMLSelectElement | null =>
        panel.querySelector<HTMLSelectElement>("select.eo-speaker");

    it("sits on the layer row beside align, defaulting to none", () => {
        const panel = mount(makeEntry());
        const row = rowsOf(panel)[0];
        expect(labelsOf(row)).toContain("Speaker");
        const select = speakerOf(panel);
        expect(select).not.toBeNull();
        expect(select?.value).toBe("");
        expect(Array.from(select?.options ?? []).map((o) => o.text)).toEqual(["None"]);
    });

    it("lists the roster and writes the pick onto the entry", () => {
        mockRoster = [
            { meta_id: "a", name_en: "Aiko", name_ja: "愛子", gender: "female", alias_en: null, alias_ja: null },
            { meta_id: "b", name_en: null, name_ja: "先生", gender: "male", alias_en: null, alias_ja: null },
        ];
        const entry = makeEntry();
        const ctx = makeCtx(entry);
        const panel = mount(entry, ctx);
        const select = speakerOf(panel)!;
        expect(Array.from(select.options).map((o) => o.text)).toEqual(["None", "Aiko (♀)", "先生 (♂)"]);

        select.value = "b";
        select.dispatchEvent(new Event("change", { bubbles: true }));
        expect(entry.character_id).toBe("b");
        // A value change repaints the summary, it does not rebuild the panel.
        expect(ctx.refresh).toHaveBeenCalled();
        expect(ctx.rebuild).not.toHaveBeenCalled();
    });

    it("reads a deleted character as none", () => {
        mockRoster = [{ meta_id: "a", name_en: "Aiko" }];
        const panel = mount(makeEntry({ character_id: "gone" }));
        expect(speakerOf(panel)?.value).toBe("");
    });

    it("shows the gender symbol next to speaker names", () => {
        mockRoster = [
            { meta_id: "e", name_en: "Eren", name_ja: "エレン", gender: "male", alias_en: "Titan", alias_ja: null },
            { meta_id: "m", name_en: "Mikasa", name_ja: "ミカサ", gender: "female", alias_en: null, alias_ja: null },
            { meta_id: "u", name_en: "Annie", name_ja: null, gender: null, alias_en: null, alias_ja: null },
        ];
        const panel = mount(makeEntry());
        const texts = Array.from(speakerOf(panel)!.options).map((o) => o.text);
        expect(texts).toEqual(["None", "Eren (♂)", "Mikasa (♀)", "Annie"]);
    });
});

describe("text clean options", () => {
    /** Rows of the clean block. It hangs off the panel, not the main grid. */
    const cleanRowsOf = (panel: HTMLElement): HTMLElement[] =>
        Array.from(panel.querySelectorAll<HTMLElement>(".eo-clean > .eo-row"));

    const tweakRow = (panel: HTMLElement): HTMLElement => {
        const found = cleanRowsOf(panel).find((r) => labelsOf(r)[0] === "Tweaks");
        expect(found).toBeDefined();
        return found as HTMLElement;
    };

    const togglesOf = (row: HTMLElement): string[] =>
        Array.from(row.querySelectorAll<HTMLElement>(".eo-check")).map((n) => n.textContent?.trim() ?? "");

    const openClean = (over: Partial<Entry> = {}) => {
        const entry = makeEntry({ bg_mode: "clean", ...over });
        const ctx = makeCtx(entry);
        return { entry, ctx, panel: mount(entry, ctx) };
    };

    beforeEach(() => {
        // No backend here, so `cleanCapabilities` serves its fallback synchronously
        // and never caches a real answer that would leak into the next case.
        vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    });

    it("puts the clean method and its fill on one row", () => {
        const cleanRows = cleanRowsOf(openClean().panel);
        expect(labelsOf(cleanRows[0])).toEqual(["Method", "Fill"]);
        expect(cleanRows[0].querySelectorAll("select")).toHaveLength(2);
    });

    it("fits every detector tweak on one row", () => {
        const row = tweakRow(openClean({ clean: { ...defaultCleanOptions(), method: "textseg" } }).panel);
        // Speed and tile, then the switches -- one line, five controls.
        expect(row.querySelectorAll("select")).toHaveLength(2);
        expect(togglesOf(row)).toEqual(["Glow", "TTA", "Transport"]);
        // Five controls do not fit a narrow panel; the switches drop to a second
        // line as a block instead of being squeezed into a column.
        expect(row.querySelector(".eo-field")?.classList.contains("eo-field-wrap")).toBe(true);
    });

    it("offers only the knobs the chosen method reads", () => {
        // Selected Region runs no detector, so a speed or a tile size would be a
        // control that does nothing. Transport is not a detector knob -- it gates
        // the reconstruction stage, which every method goes through.
        const region = tweakRow(openClean().panel);
        expect(region.querySelectorAll("select")).toHaveLength(0);
        expect(togglesOf(region)).toEqual(["Transport"]);

        // Box coverage has no halo for Glow to grow over, either.
        const ppocr = tweakRow(openClean({ clean: { ...defaultCleanOptions(), method: "ppocr" } }).panel);
        expect(togglesOf(ppocr)).toEqual(["Transport"]);
    });

    it("pairs each tile size with the overlap the backend would use", () => {
        const { entry, ctx, panel } = openClean({
            clean: { ...defaultCleanOptions(), method: "textseg" },
        });
        const tile = tweakRow(panel).querySelectorAll("select")[1] as HTMLSelectElement;

        expect(Array.from(tile.options).map((o) => o.value)).toEqual(["0", "256", "512", "1024", "2048"]);
        expect(tile.options[0].textContent).toBe("Off");
        expect(tile.options[1].title).toBe("256 px tiles, 48 px overlap");
        expect(tile.options[3].title).toBe("1024 px tiles, 192 px overlap");

        tile.value = "512";
        tile.dispatchEvent(new Event("change"));
        // A number, not the string the select carries: it is sent as `tile`.
        expect(entry.clean.tile).toBe(512);
        expect(ctx.rebuild).toHaveBeenCalled();
    });

    it("names what each speed costs, and rebuilds when it changes", () => {
        const { entry, ctx, panel } = openClean({
            clean: { ...defaultCleanOptions(), method: "textseg" },
        });
        const speed = tweakRow(panel).querySelectorAll("select")[0] as HTMLSelectElement;

        expect(Array.from(speed.options).map((o) => o.value)).toEqual(["best", "fast", "fastest", "single"]);
        expect(speed.value).toBe("fastest");
        expect(speed.options[0].title).toBe("24 forward passes at most");
        expect(speed.options[3].title).toBe("1 forward pass at most");

        speed.value = "best";
        speed.dispatchEvent(new Event("change"));
        expect(entry.clean.speed).toBe("best");
        // Structural: whether TTA has anything to act on depends on the preset.
        expect(ctx.rebuild).toHaveBeenCalled();
    });

    it("dims TTA under a preset that does no flip averaging, without hiding it", () => {
        const fastest = tweakRow(openClean({ clean: { ...defaultCleanOptions(), method: "textseg" } }).panel);
        const dimmed = fastest.querySelectorAll<HTMLElement>(".eo-check")[1];
        expect(dimmed.textContent?.trim()).toBe("TTA");
        expect(dimmed.classList.contains("eo-check-inert")).toBe(true);
        // Still clickable: the value is stored and does apply on Best.
        expect((dimmed.querySelector("input") as HTMLInputElement).disabled).toBe(false);

        const best = tweakRow(
            openClean({
                clean: { ...defaultCleanOptions(), method: "textseg", speed: "best" },
            }).panel,
        );
        const live = best.querySelectorAll<HTMLElement>(".eo-check")[1];
        expect(live.classList.contains("eo-check-inert")).toBe(false);
    });

    it("stores each switch on the entry", () => {
        const { entry, panel } = openClean({
            clean: { ...defaultCleanOptions(), method: "textseg" },
        });
        const boxes = tweakRow(panel).querySelectorAll<HTMLInputElement>(".eo-check input");
        for (const box of boxes) {
            box.checked = false;
            box.dispatchEvent(new Event("change"));
        }
        expect(entry.clean.glow).toBe(false);
        expect(entry.clean.tta).toBe(false);
        expect(entry.clean.transport).toBe(false);
    });

    it("back-fills tuning an older project never wrote", () => {
        // `entry.clean` is persisted, so one saved before the tuning existed
        // arrives without it -- and a select with no matching option renders blank.
        const entry = makeEntry({
            clean: { method: "textseg", fill: "telea", glow: false, transport: false } as never,
        });
        ensureEntryDefaults(entry);

        expect(entry.clean).toEqual({
            method: "textseg",
            fill: "telea",
            glow: false,
            transport: false,
            speed: "fastest",
            tta: true,
            tile: 0,
        });
    });

    it("repairs a tuning field of the wrong type rather than trusting it", () => {
        const entry = makeEntry({
            clean: { ...defaultCleanOptions(), tile: Number.NaN, speed: 3 } as never,
        });
        ensureEntryDefaults(entry);
        expect(entry.clean.tile).toBe(0);
        expect(entry.clean.speed).toBe("fastest");
    });
});

describe("numeric controls", () => {
    it("offers Auto plus every font size, and is not disabled on a fresh entry", () => {
        const entry = makeEntry();
        const ctx = makeCtx(entry);
        const rows = rowsOf(mount(entry, ctx));
        const size = rows[1].querySelectorAll("select")[1] as HTMLSelectElement;

        expect(size.disabled).toBe(false);
        expect(size.value).toBe("");
        expect(size.options).toHaveLength(MAX_FONT_SIZE - MIN_FONT_SIZE + 2);
        expect(size.options[0].textContent).toBe("Auto");
        expect(size.options[1].value).toBe(String(MIN_FONT_SIZE));
        expect(size.options[size.options.length - 1].value).toBe(String(MAX_FONT_SIZE));

        size.value = "24";
        size.dispatchEvent(new Event("change"));
        expect(entry.font_size).toBe(24);
        expect(ctx.refresh).toHaveBeenCalled();

        size.value = "";
        size.dispatchEvent(new Event("change"));
        expect(entry.font_size).toBeNull();
    });

    it("offers Auto plus outline widths 0..8", () => {
        const entry = makeEntry();
        const rows = rowsOf(mount(entry, makeCtx(entry)));
        // Third select on the font row: family, size, then the outline width.
        const width = rows[1].querySelectorAll("select")[2] as HTMLSelectElement;

        expect(width.disabled).toBe(false);
        expect(width.options).toHaveLength(MAX_STROKE_WIDTH - MIN_STROKE_WIDTH + 2);
        expect(width.options[1].value).toBe(String(MIN_STROKE_WIDTH));
        expect(width.options[width.options.length - 1].value).toBe(String(MAX_STROKE_WIDTH));

        width.value = "5";
        width.dispatchEvent(new Event("change"));
        expect(entry.stroke_width).toBe(5);
    });

    it("shows an explicit size as the selected option", () => {
        const entry = makeEntry({ font_size: 30, stroke_width: 3 });
        const selects = rowsOf(mount(entry, makeCtx(entry)))[1].querySelectorAll("select");
        expect((selects[1] as HTMLSelectElement).value).toBe("30");
        expect((selects[2] as HTMLSelectElement).value).toBe("3");
    });
});

describe("typesetting geometry", () => {
    /** Row 2: Spacing (W, L) then Scale (X, Y). All four are selects. */
    const spacing = (panel: HTMLElement): HTMLSelectElement[] =>
        Array.from(rowsOf(panel)[2].querySelectorAll<HTMLSelectElement>("select"));

    /** Row 3: Shift (X, Y) then Rotate (X, Y, Z). All five are number boxes. */
    const transform = (panel: HTMLElement): HTMLInputElement[] =>
        Array.from(rowsOf(panel)[3].querySelectorAll<HTMLInputElement>("input"));

    /** A decoded page behind the overlay, which jsdom will not give us for free. */
    const withPage = (w: number, h: number): void => {
        const img = document.createElement("img");
        img.id = "mainImage";
        Object.defineProperty(img, "naturalWidth", { value: w });
        Object.defineProperty(img, "naturalHeight", { value: h });
        document.body.appendChild(img);
    };

    const set = (control: HTMLSelectElement | HTMLInputElement, value: string): void => {
        control.value = value;
        control.dispatchEvent(new Event("change"));
    };

    it("starts neutral -- auto spacing, no stretch, no shift, no rotation", () => {
        const panel = mount(makeEntry());
        expect(spacing(panel).map((s) => s.value)).toEqual(["", "", "1.0", "1.0"]);
        expect(transform(panel).map((i) => i.value)).toEqual(["0", "0", "0", "0", "0"]);
    });

    it("offers every tenth, with Auto only on the two spacings", () => {
        const [words, lines, scaleX, scaleY] = spacing(mount(makeEntry()));
        const tenths = (lo: number, hi: number) => Math.round(hi * 10) - Math.round(lo * 10) + 1;

        expect(words.options).toHaveLength(tenths(MIN_WORD_SPACING, MAX_WORD_SPACING) + 1);
        expect(words.options[0].textContent).toBe("Auto");
        expect(words.options[1].value).toBe(MIN_WORD_SPACING.toFixed(1));
        expect(lines.options).toHaveLength(tenths(MIN_LINE_SPACING, MAX_LINE_SPACING) + 1);
        expect(lines.options[0].textContent).toBe("Auto");

        // The scales have a neutral value instead of an auto mode, so no "Auto".
        for (const scale of [scaleX, scaleY]) {
            expect(scale.options).toHaveLength(tenths(MIN_FONT_SCALE, MAX_FONT_SCALE));
            expect(scale.options[0].value).toBe(MIN_FONT_SCALE.toFixed(1));
            expect(scale.options[scale.options.length - 1].value).toBe(MAX_FONT_SCALE.toFixed(1));
        }
        // One decimal place throughout, and every option reachable -- an option
        // list built by adding 0.1 repeatedly drifts to "0.7000000000000001",
        // which `select.value` matches against nothing.
        expect(Array.from(scaleX.options).every((o) => /^\d\.\d$/.test(o.value))).toBe(true);
        expect(scaleX.options[2].textContent).toBe("0.7×");
    });

    it("writes a pick onto the entry and repaints without rebuilding", () => {
        const entry = makeEntry();
        const ctx = makeCtx(entry);
        const [words, lines, scaleX, scaleY] = spacing(mount(entry, ctx));

        set(words, "1.5");
        set(lines, "0.8");
        set(scaleX, "2.0");
        set(scaleY, "0.5");
        expect(entry.word_spacing).toBe(1.5);
        expect(entry.line_spacing).toBe(0.8);
        expect(entry.font_scale_x).toBe(2);
        expect(entry.font_scale_y).toBe(0.5);

        // Geometry only changes where the ink lands, so the panel stays put.
        expect(ctx.refresh).toHaveBeenCalled();
        expect(ctx.rebuild).not.toHaveBeenCalled();

        set(words, "");
        expect(entry.word_spacing).toBeNull();
    });

    it("commits a typed shift and rotation, clamped to the range", () => {
        const entry = makeEntry();
        const ctx = makeCtx(entry);
        const [shiftX, shiftY, angleX, , angleZ] = transform(mount(entry, ctx));

        set(shiftX, "-40");
        set(shiftY, "12");
        expect(entry.shift_x).toBe(-40);
        expect(entry.shift_y).toBe(12);

        set(angleZ, "45");
        expect(entry.angle_z).toBe(45);
        // Out of range settles on the bound, and the box shows what was stored
        // rather than the number that was typed at it.
        set(angleX, "900");
        expect(entry.angle_x).toBe(MAX_TEXT_ANGLE);
        expect(angleX.value).toBe(String(MAX_TEXT_ANGLE));
        set(angleX, "-900");
        expect(entry.angle_x).toBe(-MAX_TEXT_ANGLE);
        expect(ctx.rebuild).not.toHaveBeenCalled();
    });

    it("rounds a fractional entry and settles an emptied box on zero", () => {
        const entry = makeEntry();
        const [shiftX] = transform(mount(entry, makeCtx(entry)));

        set(shiftX, "7.6");
        expect(entry.shift_x).toBe(8);
        expect(shiftX.value).toBe("8");

        set(shiftX, "");
        expect(entry.shift_x).toBe(0);
        expect(shiftX.value).toBe("0");
    });

    it("bounds the shift by the page when one is on screen", () => {
        withPage(600, 900);
        const entry = makeEntry();
        const [shiftX, shiftY] = transform(mount(entry, makeCtx(entry)));
        expect(shiftX.max).toBe("600");
        expect(shiftY.max).toBe("900");
        expect(shiftX.min).toBe("-600");

        // The renderer clamps again, to the page edge rather than to the number:
        // it is the only place that knows where the tilted, spun block lands.
        set(shiftX, "99999");
        expect(entry.shift_x).toBe(600);
    });

    it("shows a stored geometry as the selected options", () => {
        const entry = makeEntry({
            word_spacing: 2.4,
            line_spacing: 1.2,
            font_scale_x: 1.7,
            font_scale_y: 0.6,
            shift_x: -18,
            angle_y: -33,
        });
        const panel = mount(entry, makeCtx(entry));
        expect(spacing(panel).map((s) => s.value)).toEqual(["2.4", "1.2", "1.7", "0.6"]);
        expect(transform(panel).map((i) => i.value)).toEqual(["-18", "0", "0", "-33", "0"]);
    });

    it("fills and clamps geometry an older entry does not have", () => {
        // An entry cached before these options existed, or restored from a build
        // whose ranges were wider. Neither may reach the renderer as-is.
        const stale = { layer: 1, text_align: "center" } as unknown as Entry;
        ensureEntryDefaults(stale);
        expect(stale.word_spacing).toBeNull();
        expect(stale.line_spacing).toBeNull();
        expect(stale.font_scale_x).toBe(1);
        expect(stale.font_scale_y).toBe(1);
        expect([stale.shift_x, stale.shift_y, stale.angle_x, stale.angle_y, stale.angle_z]).toEqual([0, 0, 0, 0, 0]);

        const wild = makeEntry({
            word_spacing: 99,
            line_spacing: -5,
            font_scale_x: 4,
            font_scale_y: 0.01,
            angle_z: 400,
            shift_x: 3.7,
        });
        ensureEntryDefaults(wild);
        expect(wild.word_spacing).toBe(MAX_WORD_SPACING);
        expect(wild.line_spacing).toBe(MIN_LINE_SPACING);
        expect(wild.font_scale_x).toBe(MAX_FONT_SCALE);
        expect(wild.font_scale_y).toBe(MIN_FONT_SCALE);
        expect(wild.angle_z).toBe(MAX_TEXT_ANGLE);
        expect(wild.shift_x).toBe(4);
    });

    it("snaps a multiplier to the tenth the picker can show", () => {
        // Otherwise the control reads 1.1 while the renderer is given 1.05, and
        // the two disagree for as long as nobody touches the select.
        const entry = makeEntry({ word_spacing: 1.05, font_scale_y: 1.449 });
        ensureEntryDefaults(entry);
        expect(entry.word_spacing).toBe(1.1);
        expect(entry.font_scale_y).toBe(1.4);
        expect(spacing(mount(entry, makeCtx(entry))).map((s) => s.value)).toEqual(["1.1", "", "1.0", "1.4"]);
    });
});

describe("colour control", () => {
    /** Click a colour button. Row 1 has two: font colour, then outline colour. */
    const openFor = (panel: HTMLElement, rowIdx: number, btnIdx = 0): HTMLElement => {
        const buttons = rowsOf(panel)[rowIdx].querySelectorAll<HTMLElement>(".eo-color-btn");
        const button = buttons[btnIdx];
        button.click();
        return button;
    };

    it("opens its popover on the very first click", () => {
        const panel = mount(makeEntry());
        expect(document.querySelector(".eo-pop")).toBeNull();
        openFor(panel, 1);
        expect(document.querySelector(".eo-pop")).not.toBeNull();
    });

    it("closes again when its own button is clicked a second time", () => {
        const panel = mount(makeEntry());
        const button = openFor(panel, 1);
        button.click();
        expect(document.querySelector(".eo-pop")).toBeNull();
    });

    it("writes the chosen swatch to the entry and closes", () => {
        const entry = makeEntry();
        const ctx = makeCtx(entry);
        const panel = mount(entry, ctx);
        openFor(panel, 1);
        const swatch = document.querySelector(".eo-pop-grid .eo-pop-swatch") as HTMLElement;
        swatch.click();

        expect(entry.font_color).toBe(swatch.title.toLowerCase());
        expect(ctx.refresh).toHaveBeenCalled();
        expect(document.querySelector(".eo-pop")).toBeNull();
    });

    it("accepts a typed hex on Enter and normalises it", () => {
        const entry = makeEntry();
        const panel = mount(entry, makeCtx(entry));
        openFor(panel, 1, 1);
        const hexInput = document.querySelector(".eo-pop-hex") as HTMLInputElement;

        hexInput.value = "#f0c";
        hexInput.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
        expect(entry.stroke_color).toBe("#ff00cc");
    });

    it("marks a malformed hex instead of committing it", () => {
        const entry = makeEntry();
        const panel = mount(entry, makeCtx(entry));
        openFor(panel, 1, 1);
        const hexInput = document.querySelector(".eo-pop-hex") as HTMLInputElement;

        hexInput.value = "not-a-colour";
        hexInput.dispatchEvent(new Event("input"));
        expect(hexInput.classList.contains("is-bad")).toBe(true);
        hexInput.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
        expect(entry.stroke_color).toBeNull();
        expect(document.querySelector(".eo-pop")).not.toBeNull();
    });

    it("offers Auto, and Auto clears the value", () => {
        const entry = makeEntry({ font_color: "#dc2626" });
        const panel = mount(entry, makeCtx(entry));
        openFor(panel, 1);
        (document.querySelector(".eo-pop .eo-chip") as HTMLElement).click();
        expect(entry.font_color).toBeNull();
    });

    it("carries the current value in the button tooltip, since the compact form has no readout", () => {
        const entry = makeEntry({ font_color: "#dc2626", stroke_color: "#0f0" });
        const panel = mount(entry, makeCtx(entry));
        const buttons = rowsOf(panel)[1].querySelectorAll<HTMLElement>(".eo-color-btn");

        expect(buttons[0].classList.contains("is-compact")).toBe(true);
        expect(buttons[0].title).toBe("Font colour: #DC2626");
        // Two swatches share the row now, so the tooltip is what tells them apart.
        expect(buttons[1].title).toBe("Outline colour: #0F0");
    });
});
