import { beforeEach, describe, expect, it } from "vitest";
import {
    buildSpreads,
    clampPan,
    contentSize,
    DEFAULT_SPREAD_GAP,
    DEFAULT_VERTICAL_GAP,
    DEFAULT_VIEW_SETTINGS,
    fitScale,
    gutterWidth,
    loadViewSettings,
    MAX_SPREAD_GAP,
    MAX_VERTICAL_GAP,
    MAX_ZOOM,
    MIN_ZOOM,
    normalizeViewSettings,
    orderForDisplay,
    pagesPerStep,
    rowSize,
    saveViewSettings,
    spreadIndexOf,
    stepZoom,
    wheelZoom,
} from "./viewSettings";

describe("normalizeViewSettings", () => {
    it("fills in the defaults for an empty object", () => {
        expect(normalizeViewSettings({})).toEqual({ ...DEFAULT_VIEW_SETTINGS });
    });

    it.each([null, undefined, 7, "single", [], { pageView: {} }])("survives junk input %j", (raw) => {
        const settings = normalizeViewSettings(raw);
        expect(settings.pageView).toBe("single");
        expect(settings.verticalGap).toBe(DEFAULT_VERTICAL_GAP);
        expect(settings.spreadGap).toBe(DEFAULT_SPREAD_GAP);
    });

    it("keeps every view the picker offers", () => {
        for (const view of ["single", "double", "vertical", "compare"] as const) {
            expect(normalizeViewSettings({ pageView: view }).pageView).toBe(view);
        }
    });

    it("falls back to single for a view this build does not have", () => {
        // A profile written by a newer build, or a hand-edited value.
        expect(normalizeViewSettings({ pageView: "carousel" }).pageView).toBe("single");
    });

    it("clamps the gaps into their own ranges", () => {
        expect(normalizeViewSettings({ verticalGap: 999 }).verticalGap).toBe(MAX_VERTICAL_GAP);
        expect(normalizeViewSettings({ verticalGap: -4 }).verticalGap).toBe(0);
        expect(normalizeViewSettings({ spreadGap: 999 }).spreadGap).toBe(MAX_SPREAD_GAP);
        expect(normalizeViewSettings({ spreadGap: -1 }).spreadGap).toBe(0);
    });

    it("rounds a fractional gap, so the slider and the CSS agree", () => {
        expect(normalizeViewSettings({ verticalGap: 4.6 }).verticalGap).toBe(5);
    });

    it("reads a non-numeric gap as the minimum rather than NaN", () => {
        expect(normalizeViewSettings({ verticalGap: "wide" }).verticalGap).toBe(0);
    });

    it("keeps the pairing rules off when they were turned off", () => {
        const settings = normalizeViewSettings({ coverAlone: false, wideAlone: false });
        expect(settings.coverAlone).toBe(false);
        expect(settings.wideAlone).toBe(false);
    });
});

describe("loadViewSettings", () => {
    beforeEach(() => localStorage.clear());

    it("is the defaults with nothing stored", () => {
        expect(loadViewSettings()).toEqual({ ...DEFAULT_VIEW_SETTINGS });
    });

    it("round-trips what was saved", () => {
        const settings = { ...DEFAULT_VIEW_SETTINGS, pageView: "double" as const, spreadGap: 24, pinned: true };
        saveViewSettings(settings);
        expect(loadViewSettings()).toEqual(settings);
    });

    it("falls back to the defaults on unparseable JSON", () => {
        localStorage.setItem("fox-reader-vulpes-view", "{not json");
        expect(loadViewSettings()).toEqual({ ...DEFAULT_VIEW_SETTINGS });
    });
});

describe("buildSpreads", () => {
    it("has nothing to show for an empty folder", () => {
        expect(buildSpreads(0)).toEqual([]);
        expect(buildSpreads(-3)).toEqual([]);
    });

    it("stands the cover alone and pairs the rest as the print edition does", () => {
        // 2-3, 4-5 ... is the printing; pairing from page one would put every
        // spread one page out of step.
        expect(buildSpreads(6, { coverAlone: true, wideAlone: false })).toEqual([[0], [1, 2], [3, 4], [5]]);
    });

    it("pairs from the first page when the cover rule is off", () => {
        expect(buildSpreads(6, { coverAlone: false, wideAlone: false })).toEqual([
            [0, 1],
            [2, 3],
            [4, 5],
        ]);
    });

    it("leaves a trailing odd page on its own", () => {
        expect(buildSpreads(5, { coverAlone: false, wideAlone: false })).toEqual([[0, 1], [2, 3], [4]]);
    });

    it("gives a landscape page the stage to itself", () => {
        // Aspects are height/width: 0.7 is wider than tall, i.e. already a spread.
        const aspects = [1.4, 1.4, 0.7, 1.4, 1.4];
        expect(buildSpreads(5, { coverAlone: false, wideAlone: true, aspects })).toEqual([[0, 1], [2], [3, 4]]);
    });

    it("does not pair a portrait page with the wide one that follows it", () => {
        const aspects = [1.4, 0.7, 1.4];
        expect(buildSpreads(3, { coverAlone: false, wideAlone: true, aspects })).toEqual([[0], [1], [2]]);
    });

    it("pairs a wide page like any other when the rule is off", () => {
        const aspects = [1.4, 0.7, 1.4, 1.4];
        expect(buildSpreads(4, { coverAlone: false, wideAlone: false, aspects })).toEqual([
            [0, 1],
            [2, 3],
        ]);
    });

    it("treats a page of unknown aspect as portrait", () => {
        // Guessing "wide" would split a pair and reflow the book the moment the
        // image decoded and the real aspect arrived.
        expect(buildSpreads(4, { coverAlone: false, wideAlone: true, aspects: [0, 0, 0, 0] })).toEqual([
            [0, 1],
            [2, 3],
        ]);
    });

    it("covers every page exactly once, whatever the rules", () => {
        const aspects = [0.6, 1.4, 1.4, 0.8, 1.4, 1.4, 1.4];
        for (const coverAlone of [true, false]) {
            for (const wideAlone of [true, false]) {
                const flat = buildSpreads(7, { coverAlone, wideAlone, aspects }).flat();
                expect(flat).toEqual([0, 1, 2, 3, 4, 5, 6]);
            }
        }
    });
});

describe("spreadIndexOf", () => {
    const spreads = [[0], [1, 2], [3, 4]];

    it("finds the spread holding a page", () => {
        expect(spreadIndexOf(spreads, 0)).toBe(0);
        expect(spreadIndexOf(spreads, 2)).toBe(1);
        expect(spreadIndexOf(spreads, 4)).toBe(2);
    });

    it("reports -1 for a page in none of them", () => {
        expect(spreadIndexOf(spreads, 9)).toBe(-1);
    });
});

describe("orderForDisplay", () => {
    it("puts the earlier page on the right for right-to-left manga", () => {
        expect(orderForDisplay([2, 3], "rtl")).toEqual([3, 2]);
    });

    it("puts the earlier page on the left for left-to-right comics", () => {
        expect(orderForDisplay([2, 3], "ltr")).toEqual([2, 3]);
    });

    it("leaves a lone page alone either way", () => {
        expect(orderForDisplay([5], "rtl")).toEqual([5]);
        expect(orderForDisplay([5], "ltr")).toEqual([5]);
    });

    it("does not mutate its input", () => {
        const spread = [2, 3];
        orderForDisplay(spread, "rtl");
        expect(spread).toEqual([2, 3]);
    });
});

describe("contentSize", () => {
    it("is nothing when no page has been measured", () => {
        expect(contentSize([])).toEqual({ width: 0, height: 0 });
        expect(contentSize([{ width: 0, height: 0 }])).toEqual({ width: 0, height: 0 });
    });

    it("sums the widths and takes the tallest height", () => {
        // Unequal pages are not reconciled: each keeps its aspect and the row
        // centres them, rather than stretching the short one to match.
        expect(
            contentSize([
                { width: 800, height: 1200 },
                { width: 700, height: 1000 },
            ]),
        ).toEqual({
            width: 1500,
            height: 1200,
        });
    });

    it("ignores a pane whose image has not decoded yet", () => {
        expect(
            contentSize([
                { width: 800, height: 1200 },
                { width: 0, height: 0 },
            ]),
        ).toEqual({
            width: 800,
            height: 1200,
        });
    });
});

describe("gutterWidth / contentSize / rowSize", () => {
    it("has no gutter for fewer than two panes", () => {
        expect(gutterWidth(8, 0)).toBe(0);
        expect(gutterWidth(8, 1)).toBe(0);
    });

    it("puts one gutter between two panes", () => {
        expect(gutterWidth(8, 2)).toBe(8);
        expect(gutterWidth(8, 3)).toBe(16);
    });

    it("folds the gutter into the content, so one transform can scale the row", () => {
        // The gutter scales with the pages, the way the gap between two facing pages
        // of a book does. Holding it at a fixed number of screen pixels instead
        // would mean re-laying out every page on every frame of a zoom.
        expect(
            contentSize(
                [
                    { width: 800, height: 1200 },
                    { width: 700, height: 1000 },
                ],
                8,
            ),
        ).toEqual({
            width: 1508,
            height: 1200,
        });
    });

    it("adds no gutter to a single page", () => {
        expect(contentSize([{ width: 800, height: 1200 }], 8)).toEqual({ width: 800, height: 1200 });
    });

    it("drops the gutter of a pane that has not decoded yet", () => {
        expect(
            contentSize(
                [
                    { width: 800, height: 1200 },
                    { width: 0, height: 0 },
                ],
                8,
            ),
        ).toEqual({
            width: 800,
            height: 1200,
        });
    });

    it("scales the whole row, gutter included", () => {
        expect(rowSize({ width: 1008, height: 1400 }, 2)).toEqual({ width: 2016, height: 2800 });
    });

    it("is nothing for unmeasured content", () => {
        expect(rowSize({ width: 0, height: 0 }, 2)).toEqual({ width: 0, height: 0 });
    });
});

describe("fitScale", () => {
    const viewport = { width: 1000, height: 800 };
    const content = { width: 2000, height: 3200 };

    it("shows the whole page for the page fit", () => {
        // The smaller of the two ratios: 0.5 by width, 0.25 by height.
        expect(fitScale("page", viewport, content)).toBe(0.25);
    });

    it("fills the width for the width fit", () => {
        expect(fitScale("width", viewport, content)).toBe(0.5);
    });

    it("fills the height for the height fit", () => {
        expect(fitScale("height", viewport, content)).toBe(0.25);
    });

    it("is exactly 1 for the actual-size fit, whatever the sizes", () => {
        expect(fitScale("actual", viewport, content)).toBe(1);
        expect(fitScale("actual", { width: 0, height: 0 }, content)).toBe(1);
    });

    it("is 1 while nothing has been measured, rather than Infinity or 0", () => {
        expect(fitScale("page", { width: 0, height: 0 }, content)).toBe(1);
        expect(fitScale("page", viewport, { width: 0, height: 0 })).toBe(1);
    });

    it("clamps a wildly small page into the zoom range", () => {
        expect(fitScale("width", { width: 10000, height: 10000 }, { width: 1, height: 1 })).toBe(MAX_ZOOM);
    });

    it("clamps a wildly large page into the zoom range", () => {
        expect(fitScale("page", { width: 10, height: 10 }, { width: 100000, height: 100000 })).toBe(MIN_ZOOM);
    });
});

describe("clampPan", () => {
    it("holds a row that fits exactly centred", () => {
        expect(clampPan(1000, 600, 250)).toBe(0);
        expect(clampPan(1000, 1000, -80)).toBe(0);
    });

    it("lets a larger row be dragged until its edge meets the viewport's", () => {
        // 400px of overhang, so 200 either side of centre.
        expect(clampPan(1000, 1400, 120)).toBe(120);
        expect(clampPan(1000, 1400, 900)).toBe(200);
        expect(clampPan(1000, 1400, -900)).toBe(-200);
    });
});

describe("stepZoom / wheelZoom", () => {
    it("steps by a proportion, so a notch is the same visual step at any zoom", () => {
        expect(stepZoom(1, 0.1)).toBeCloseTo(1.1);
        expect(stepZoom(4, 0.1)).toBeCloseTo(4.4);
    });

    it("stays inside the zoom range", () => {
        expect(stepZoom(MAX_ZOOM, 0.5)).toBe(MAX_ZOOM);
        expect(stepZoom(MIN_ZOOM, -0.5)).toBe(MIN_ZOOM);
    });

    it("zooms in on a wheel up and out on a wheel down", () => {
        expect(wheelZoom(1, -100)).toBeGreaterThan(1);
        expect(wheelZoom(1, 100)).toBeLessThan(1);
    });

    it("keeps a trackpad's small deltas small", () => {
        expect(wheelZoom(1, -4)).toBeCloseTo(1.004, 3);
    });
});

describe("pagesPerStep", () => {
    it("moves two pages in the double view and one everywhere else", () => {
        expect(pagesPerStep("double")).toBe(2);
        expect(pagesPerStep("single")).toBe(1);
        expect(pagesPerStep("vertical")).toBe(1);
        expect(pagesPerStep("compare")).toBe(1);
    });
});
