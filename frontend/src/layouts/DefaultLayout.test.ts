import { describe, expect, it } from "vitest";
import { getGalleryResizeDelta } from "./DefaultLayout";

describe("getGalleryResizeDelta", () => {
    it("grows the bottom gallery when its splitter is dragged upward", () => {
        expect(getGalleryResizeDelta(-25)).toBe(25);
    });

    it("shrinks the bottom gallery when its splitter is dragged downward", () => {
        expect(getGalleryResizeDelta(25)).toBe(-25);
    });
});
