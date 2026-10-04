import { beforeEach, describe, expect, it } from "vitest";
import { currentPageUrl, getClampedZoom, markPageSaved, pageUrl } from "./viewer";
import { isViewingSaved, state } from "./state";

describe("getClampedZoom", () => {
    it("changes zoom by the requested amount within its bounds", () => {
        expect(getClampedZoom(1, 0.2, 0.5, 1.5)).toBe(1.2);
    });

    it("does not zoom below the minimum or above the maximum", () => {
        expect(getClampedZoom(0.5, -1, 0.5, 1.5)).toBe(0.5);
        expect(getClampedZoom(1.5, 1, 0.5, 1.5)).toBe(1.5);
    });
});

describe("pageUrl", () => {
    beforeEach(() => {
        state.viewVariant = "original";
        state.sameDir = false;
        state.savedFiles = new Set();
    });

    it("escapes the filename", () => {
        // `1+2 (a#b).png` used to reach the server as a different name, or get
        // cut off at the `#`.
        expect(pageUrl("1+2 (a#b).png", 7)).toBe("/img_serve/1%2B2%20(a%23b).png?t=7");
    });

    it("asks for the original by default", () => {
        state.savedFiles = new Set(["p1.png"]);
        expect(pageUrl("p1.png", 1)).toBe("/img_serve/p1.png?t=1");
    });

    it("asks for the saved copy when the switch is on that side", () => {
        state.viewVariant = "saved";
        state.savedFiles = new Set(["p1.png"]);
        expect(pageUrl("p1.png", 1)).toBe("/img_serve/p1.png?t=1&variant=saved");
    });

    it("does not ask for a saved copy of a page that has none", () => {
        state.viewVariant = "saved";
        state.savedFiles = new Set(["p1.png"]);
        expect(pageUrl("p2.png", 1)).toBe("/img_serve/p2.png?t=1");
    });

    it("never asks for a variant when the destination is the source", () => {
        // Both sides would be the same file, so the parameter would be a lie.
        state.viewVariant = "saved";
        state.sameDir = true;
        state.savedFiles = new Set(["p1.png"]);
        expect(pageUrl("p1.png", 1)).toBe("/img_serve/p1.png?t=1");
    });
});

describe("currentPageUrl", () => {
    beforeEach(() => {
        document.body.innerHTML = "";
        state.viewVariant = "original";
        state.sameDir = false;
        state.savedFiles = new Set();
        state.currentFiles = ["p1.png", "p2.png"];
        state.currentImageFile = "p1.png";
    });

    it("gives the same URL for the same bytes", () => {
        // The buster used to be `Date.now()`, read at the moment of every
        // render, so a repaint produced a URL the browser had never seen and
        // re-downloaded a gallery that had not changed a pixel.
        expect(currentPageUrl("p1.png")).toBe(currentPageUrl("p1.png"));
    });

    it("leaves the original's URL alone when the page is saved elsewhere", () => {
        // The original file was not touched -- only a copy in the destination
        // was written -- so the copy on screen is still current.
        const before = currentPageUrl("p1.png");
        markPageSaved("p1.png");
        expect(currentPageUrl("p1.png")).toBe(before);
    });

    it("moves the saved side's URL every time that page is written", () => {
        // Twice in a row, which Preview-then-Save does inside one millisecond:
        // a timestamp would give both the same token and serve the stale copy.
        markPageSaved("p1.png");
        state.viewVariant = "saved";
        const first = currentPageUrl("p1.png");
        markPageSaved("p1.png");
        expect(currentPageUrl("p1.png")).not.toBe(first);
    });

    it("does not move any other page's URL", () => {
        state.viewVariant = "saved";
        const other = currentPageUrl("p2.png");
        markPageSaved("p1.png");
        expect(currentPageUrl("p2.png")).toBe(other);
    });

    it("moves the URL at once when the page itself was overwritten", () => {
        // Saving in place: there is no second copy, and the bytes behind the
        // one URL there is have just changed.
        state.sameDir = true;
        const before = currentPageUrl("p1.png");
        markPageSaved("p1.png");
        expect(currentPageUrl("p1.png")).not.toBe(before);
    });
});

describe("isViewingSaved", () => {
    beforeEach(() => {
        state.viewVariant = "saved";
        state.sameDir = false;
        state.savedFiles = new Set(["p1.png"]);
        state.currentImageFile = "p1.png";
    });

    it("is true on a saved page with the switch on the saved side", () => {
        expect(isViewingSaved()).toBe(true);
    });

    it("is false while the switch is on the original side", () => {
        state.viewVariant = "original";
        expect(isViewingSaved()).toBe(false);
    });

    it("is false on a page that has no saved copy", () => {
        // The preference is sticky across pages, so an unsaved page must still be
        // fully editable -- it is showing the original either way.
        state.currentImageFile = "p2.png";
        expect(isViewingSaved()).toBe(false);
    });

    it("is false when the destination is the source", () => {
        state.sameDir = true;
        expect(isViewingSaved()).toBe(false);
    });
});
