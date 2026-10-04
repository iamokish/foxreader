import { beforeEach, describe, expect, it, vi } from "vitest";
import {
    getReadingDirection,
    normalizeDirection,
    setReadingDirection,
    sortRegionsByReadingOrder,
} from "./readingOrder";
import type { Region } from "./types";

function rect(cx: number, cy: number, w = 40, h = 30): Region {
    return { type: "rectangle", coords: { x: cx - w / 2, y: cy - h / 2, w, h } };
}

function item(id: string, cx: number, cy: number, w = 40, h = 30) {
    return { id, region: rect(cx, cy, w, h) };
}

const ids = <T extends { id: string }>(items: T[]): string[] => items.map((i) => i.id);

beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
});

describe("reading direction preference", () => {
    it("defaults to RTL for manga", () => {
        expect(getReadingDirection()).toBe("rtl");
    });

    it("keeps ltr and rtl, folds anything else to rtl", () => {
        expect(normalizeDirection("ltr")).toBe("ltr");
        expect(normalizeDirection("rtl")).toBe("rtl");
        expect(normalizeDirection("sideways")).toBe("rtl");
        expect(normalizeDirection(null)).toBe("rtl");
    });

    it("persists the choice", () => {
        setReadingDirection("ltr");
        expect(getReadingDirection()).toBe("ltr");
        expect(localStorage.getItem("fox-reader-direction")).toBe("ltr");
    });
});

describe("sortRegionsByReadingOrder", () => {
    it("reads rows top to bottom, right to left in RTL", () => {
        const items = [item("bl", 10, 100), item("tr", 90, 10), item("tl", 10, 10), item("br", 90, 100)];
        expect(ids(sortRegionsByReadingOrder(items, "rtl"))).toEqual(["tr", "tl", "br", "bl"]);
    });

    it("reads rows top to bottom, left to right in LTR", () => {
        const items = [item("br", 90, 100), item("tr", 90, 10), item("bl", 10, 100), item("tl", 10, 10)];
        expect(ids(sortRegionsByReadingOrder(items, "ltr"))).toEqual(["tl", "tr", "bl", "br"]);
    });

    it("absorbs pixel jitter into the same row", () => {
        const items = [item("b", 90, 12), item("a", 10, 10), item("c", 50, 80)];
        expect(ids(sortRegionsByReadingOrder(items, "rtl"))).toEqual(["b", "a", "c"]);
        expect(ids(sortRegionsByReadingOrder(items, "ltr"))).toEqual(["a", "b", "c"]);
    });

    it("does not chain a staircase into one row", () => {
        const items = [item("a", 90, 10), item("b", 70, 24), item("c", 50, 38)];
        // Steps of 14px with 30px-tall boxes (tolerance 15): a+b share a row,
        // c's distance from the row mean (17) breaks it into the next row.
        expect(ids(sortRegionsByReadingOrder(items, "rtl"))).toEqual(["a", "b", "c"]);
    });

    it("is stable for identical boxes", () => {
        const items = [item("a", 10, 10), item("b", 10, 10)];
        expect(ids(sortRegionsByReadingOrder(items, "rtl"))).toEqual(["a", "b"]);
    });

    it("sinks degenerate regions to the end", () => {
        const good = item("good", 10, 10);
        const flat = { id: "flat", region: rect(90, 10, 0, 30) };
        expect(ids(sortRegionsByReadingOrder([flat, good], "rtl"))).toEqual(["good", "flat"]);
    });

    it("does not mutate the input", () => {
        const items = [item("b", 90, 10), item("a", 10, 10)];
        sortRegionsByReadingOrder(items, "rtl");
        expect(ids(items)).toEqual(["b", "a"]);
    });

    it("handles an empty list", () => {
        expect(sortRegionsByReadingOrder([], "rtl")).toEqual([]);
    });
});
