/**
 * The Method select, against what the backend says it can run.
 *
 * Text Seg is an optional model now -- only the three PaddleOCR models are
 * needed to run Fox Reader -- so `textseg` is a method the panel routinely has
 * to offer on a machine that cannot run it. It greys the option out on exactly
 * one signal: `available` from /api/clean/methods (tests/test_clean_methods.py
 * covers the other end). Three things have to hold for that to be a control
 * rather than a label:
 *
 *   * the option is `disabled`, so it cannot be picked;
 *   * it says so in the label and in the tooltip, so the reason is reachable;
 *   * a project that already stored it still opens, with the select marked and
 *     the reason spelled out underneath.
 *
 * Its own file because `cleanCapabilities` caches the backend's answer for the
 * life of the module, and these cases need different answers -- so each one
 * drops the module registry and re-imports. `entryOptions.test.ts` deliberately
 * keeps the opposite setup (a rejecting fetch, hence the fallback), and a
 * `vi.resetModules()` there would hand its later cases a second copy of
 * `state`, which is not the one its top-level import holds.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import type { CleanMethodsResponse } from "./core/types";
import type { Entry } from "./state";

type Method = CleanMethodsResponse["methods"][number];

const REGION: Method = {
    id: "region",
    label: "Selected Region",
    available: true,
    reason: "",
    knobs: [],
};
const PPOCR: Method = {
    id: "ppocr",
    label: "PaddleOCR",
    available: true,
    reason: "",
    knobs: [],
};
const TEXTSEG: Method = {
    id: "textseg",
    label: "Text Seg",
    available: true,
    reason: "",
    knobs: ["glow", "speed", "tta", "tile"],
};
/** What the backend sends when the optional model was never downloaded. */
const MISSING = "the Text Seg model is missing from /models/manga-text-segmentation";

/** The whole report from a machine that never downloaded the optional model. */
const WITHOUT_WEIGHTS: Method[] = [REGION, PPOCR, { ...TEXTSEG, available: false, reason: MISSING }];

/** What a project saved where the model *was* downloaded carries in `entry.clean`. */
const storedTextseg = (speed: Entry["clean"]["speed"] = "fastest"): Entry["clean"] => ({
    method: "textseg",
    fill: "hybrid-level",
    glow: true,
    transport: true,
    speed,
    tta: true,
    tile: 0,
});

const REST = {
    fills: ["hybrid-level", "hybrid", "telea"],
    default_fill: "hybrid-level",
    speeds: [
        { value: "best", label: "Best", passes: 24, tta: true },
        { value: "fastest", label: "Fastest", passes: 3, tta: false },
    ],
    default_speed: "fastest",
    default_tta: true,
    tiles: [{ value: 0, label: "Off", overlap: 192 }],
    default_tile: 0,
};

/**
 * Build the panel for `over` against a backend reporting `methods`.
 *
 * The capability cache is primed through the real fetch path first -- a panel
 * built before the answer lands shows the fallback, which is a different case --
 * and `onReady` is what the production code uses to know it arrived, so nothing
 * here waits on a timer. Pass `null` for a backend that cannot be reached.
 */
async function panelWith(methods: Method[] | null, over: Partial<Entry> = {}) {
    vi.resetModules();

    const fetched: string[] = [];
    vi.stubGlobal(
        "fetch",
        vi.fn(async (url: string) => {
            fetched.push(String(url));
            if (methods === null) throw new Error("offline");
            if (String(url) !== "/api/clean/methods") throw new Error(`unexpected ${url}`);
            return { ok: true, status: 200, json: async () => ({ methods, ...REST }) };
        }),
    );

    const options = await import("./entryOptions");
    // From the same fresh registry: the module under test reads this copy of
    // `state`, and a top-level import here would hold the previous one.
    const { state, PLAIN_GEOMETRY } = await import("./state");
    state.currentFontsData = [
        { font_filename: "a.ttf", font_name: "Alpha", font_data_uri: "", font_format: "truetype" },
    ];

    if (methods !== null) {
        await new Promise<void>((resolve) => {
            options.cleanCapabilities(() => resolve());
        });
    }

    const entry = {
        id: "e1",
        ocr_text: "",
        text: "",
        fontfile: "a.ttf",
        fontname: "'A', sans-serif",
        region: { type: "rectangle", coords: { x: 0, y: 0, w: 20, h: 20 } },
        color: "#00FF66",
        text_align: "center",
        visible: true,
        layer: 1,
        font_size: null,
        font_color: null,
        stroke_width: null,
        stroke_color: null,
        bg_mode: "clean",
        bg_color: null,
        clean: options.defaultCleanOptions(),
        ...PLAIN_GEOMETRY,
        ...over,
    } as Entry;

    const ctx = { entries: [entry], refresh: vi.fn(), rebuild: vi.fn(), refreshAll: vi.fn() };
    const panel = options.buildEntryOptions(entry, ctx);
    document.body.appendChild(panel);

    const box = panel.querySelector(".eo-clean") as HTMLElement;
    const rows = Array.from(panel.querySelectorAll<HTMLElement>(".eo-clean > .eo-row"));
    const labelled = (label: string) =>
        rows.find((r) => r.querySelector(".eo-label")?.textContent === label) as HTMLElement;
    return {
        entry,
        ctx,
        fetched,
        box,
        rows,
        /** The Tweaks row: the knobs the chosen method reads. */
        tweaks: () => labelled("Tweaks"),
        method: box.querySelector("select") as HTMLSelectElement,
        /** The reason line the panel draws under an unavailable selection. */
        warning: box.querySelector(".eo-warn-text") as HTMLElement | null,
    };
}

const optionFor = (select: HTMLSelectElement, id: string) =>
    Array.from(select.options).find((o) => o.value === id) as HTMLOptionElement;

afterEach(() => {
    document.body.innerHTML = "";
    vi.unstubAllGlobals();
});

describe("clean method availability", () => {
    it("greys out Text Seg and refuses the selection when the model is missing", async () => {
        const { method, warning } = await panelWith(WITHOUT_WEIGHTS);

        const textseg = optionFor(method, "textseg");
        // Disabled is the part that matters: the option stays listed, so a user
        // who has read about the method can see it exists and why it is off.
        expect(textseg.disabled).toBe(true);
        expect(textseg.textContent).toBe("Text Seg (unavailable)");
        expect(textseg.title).toBe(MISSING);

        // Only that one. The two that need no optional model stay live.
        expect(optionFor(method, "region").disabled).toBe(false);
        expect(optionFor(method, "ppocr").disabled).toBe(false);
        // Nothing is wrong with the current selection, so no warning yet.
        expect(method.classList.contains("eo-warn")).toBe(false);
        expect(warning).toBeNull();
    });

    it("offers it normally once the weights are there", async () => {
        const { method, warning } = await panelWith([REGION, PPOCR, TEXTSEG]);

        const textseg = optionFor(method, "textseg");
        expect(textseg.disabled).toBe(false);
        expect(textseg.textContent).toBe("Text Seg");
        expect(textseg.title).toBe("");
        expect(warning).toBeNull();
    });

    it("explains itself when a saved project already chose it", async () => {
        // `entry.clean` is persisted: a project saved where Text Seg was
        // downloaded opens where it is not, and the select shows a value the
        // user can no longer pick.
        const { method, warning } = await panelWith(WITHOUT_WEIGHTS, { clean: storedTextseg() });

        expect(method.value).toBe("textseg");
        expect(method.classList.contains("eo-warn")).toBe(true);
        // The reason, in full, under the row -- a tooltip on a disabled option
        // is not reachable for the option that is already selected.
        expect(warning?.textContent).toBe(MISSING);
    });

    it("keeps the tweak row for a stored method it cannot run", async () => {
        // The knobs travel with the method whether or not it is available, so
        // the controls an older project set do not vanish from under it.
        // Best, so the TTA switch has something to act on: the point is that the
        // row is drawn at all, not which preset it was saved with.
        const { tweaks } = await panelWith(WITHOUT_WEIGHTS, { clean: storedTextseg("best") });

        const row = tweaks();
        // Speed and tile, plus the three switches.
        expect(row.querySelectorAll("select")).toHaveLength(2);
        expect(Array.from(row.querySelectorAll<HTMLElement>(".eo-check")).map((c) => c.textContent?.trim())).toEqual([
            "Glow",
            "TTA",
            "Transport",
        ]);
    });

    it("asks the backend once for a panel opened twice", async () => {
        const { fetched } = await panelWith([REGION, PPOCR, TEXTSEG]);
        const options = await import("./entryOptions");

        options.cleanCapabilities();
        options.cleanCapabilities();

        expect(fetched).toEqual(["/api/clean/methods"]);
    });

    it("serves the fallback when the backend cannot be reached, with Text Seg off", async () => {
        // A cold start, or a backend that is not up yet: the panel is built
        // synchronously from whatever is cached, because waiting on a fetch
        // would make the collapsible feel broken. Fail-closed is the only safe
        // default -- offering a method the backend has not confirmed invites a
        // clean that cannot run.
        const { method } = await panelWith(null);

        expect(optionFor(method, "textseg").disabled).toBe(true);
        expect(optionFor(method, "ppocr").disabled).toBe(true);
        expect(optionFor(method, "region").disabled).toBe(false);
        expect(method.value).toBe("region");
    });
});
