import { describe, expect, it } from "vitest";
import { normalizeLayout } from "./LayoutContext";

describe("normalizeLayout", () => {
    it.each(["basic", "vulpes"])('keeps supported layout "%s"', (layout) => {
        expect(normalizeLayout(layout)).toBe(layout);
    });

    it("carries the retired Default layout over to Vulpes", () => {
        // Default *was* the resizable workspace with a reader, and Vulpes is its
        // successor, so anyone sitting on it keeps the kind of layout they chose
        // instead of being dropped into the legacy split view.
        expect(normalizeLayout("default")).toBe("vulpes");
    });

    it.each(["reader", "translation", "unexpected", null, undefined, 7, {}])("falls back to Basic for %j", (layout) => {
        expect(normalizeLayout(layout)).toBe("basic");
    });
});
