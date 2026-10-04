import { describe, expect, it } from "vitest";
import { normalizeLayout } from "./LayoutContext";

describe("normalizeLayout", () => {
    it.each(["reader", "translation", "unexpected", null])("falls back to Basic for retired layout %j", (layout) => {
        expect(normalizeLayout(layout)).toBe("basic");
    });

    it.each(["basic", "default"])('keeps supported layout "%s"', (layout) => {
        expect(normalizeLayout(layout)).toBe(layout);
    });
});
