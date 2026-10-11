import { beforeEach, describe, expect, it, vi } from "vitest";
import {
    applyImageFit,
    currentPageUrl,
    getClampedZoom,
    markPageSaved,
    pageUrl,
    renderGallery,
    updateStageZoomReadout,
    variantPageUrl,
    zoomViewerBy,
} from "./viewer";
import { isViewingSaved, state } from "./state";
import { off, on } from "./eventbus";

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

describe("variantPageUrl", () => {
    beforeEach(() => {
        document.body.innerHTML = "";
        state.viewVariant = "original";
        state.sameDir = false;
        state.savedFiles = new Set(["p1.png"]);
        state.currentFiles = ["p1.png", "p2.png"];
        state.currentImageFile = "p1.png";
    });

    it("names the copy it wants, whichever side the switch is on", () => {
        // What the before/after comparison needs: it shows both copies at once, so
        // it cannot read the choice off a switch that has one position.
        for (const variant of ["original", "saved"] as const) {
            state.viewVariant = variant;
            expect(variantPageUrl("p1.png", "original")).not.toContain("variant=saved");
            expect(variantPageUrl("p1.png", "saved")).toContain("variant=saved");
        }
    });

    it("still refuses to ask for a saved copy that does not exist", () => {
        expect(variantPageUrl("p2.png", "saved")).not.toContain("variant=saved");
    });

    it("still refuses when the destination is the source", () => {
        state.sameDir = true;
        expect(variantPageUrl("p1.png", "saved")).not.toContain("variant=saved");
    });

    it("moves only the saved side's URL when that page is written", () => {
        const beforeOriginal = variantPageUrl("p1.png", "original");
        const beforeSaved = variantPageUrl("p1.png", "saved");
        markPageSaved("p1.png");
        // The original file was not touched; the copy in the destination was.
        expect(variantPageUrl("p1.png", "original")).toBe(beforeOriginal);
        expect(variantPageUrl("p1.png", "saved")).not.toBe(beforeSaved);
    });
});

describe("markPageSaved announcing the save", () => {
    beforeEach(() => {
        document.body.innerHTML = "";
        state.sameDir = false;
        state.savedFiles = new Set();
        state.currentFiles = ["p1.png"];
        state.currentImageFile = "p1.png";
        state.viewVariant = "original";
    });

    it("emits page:saved, so the Saved badge and the comparison can react", () => {
        const seen = vi.fn();
        on("page:saved", seen);
        markPageSaved("p1.png");
        off("page:saved", seen);
        expect(seen).toHaveBeenCalledWith("p1.png");
    });

    it("has already recorded the page by the time it announces it", () => {
        // A listener that asks "does this page have a saved copy?" must get yes.
        let sawSaved = false;
        const check = () => {
            sawSaved = state.savedFiles.has("p1.png");
        };
        on("page:saved", check);
        markPageSaved("p1.png");
        off("page:saved", check);
        expect(sawSaved).toBe(true);
    });
});

describe("renderGallery", () => {
    beforeEach(() => {
        document.body.innerHTML = `<div id="gallery"></div>`;
        state.savedFiles = new Set();
        state.sameDir = false;
        state.viewVariant = "original";
        state.aspectCache = {};
    });

    it("seeds the aspect cache from the dimensions the folder load returned", () => {
        // Two things read it before any image has decoded: the strip's slot-height
        // estimates, and the double-page view's "a landscape page is already a
        // spread" rule.
        renderGallery(["p1.png", "p2.png"], [
            [800, 1200],
            [1600, 1200],
        ]);
        expect(state.aspectCache["p1.png"]).toBeCloseTo(1.5);
        expect(state.aspectCache["p2.png"]).toBeCloseTo(0.75);
    });

    it("clears aspects from the previous folder", () => {
        // Keyed by filename, and two folders can hold the same name for different
        // artwork -- a stale 0.6 would wrongly split a spread.
        state.aspectCache = { "p1.png": 0.6, "gone.png": 2 };
        renderGallery(["p1.png"], [[800, 1200]]);
        expect(state.aspectCache["p1.png"]).toBeCloseTo(1.5);
        expect(state.aspectCache["gone.png"]).toBeUndefined();
    });

    it("records no aspect for a page whose dimensions are unknown", () => {
        // `buildSpreads` reads a missing aspect as portrait, which is the safe guess.
        renderGallery(["p1.png", "p2.png"], [[800, 1200]]);
        expect(state.aspectCache["p2.png"]).toBeUndefined();
    });

    it("ignores degenerate dimensions rather than storing Infinity", () => {
        renderGallery(["p1.png"], [[0, 1200]]);
        expect(state.aspectCache["p1.png"]).toBeUndefined();
    });
});

describe("updateStageZoomReadout", () => {
    const DASH = "—";

    beforeEach(() => {
        document.body.innerHTML = "";
        state.zoomLevel = 1;
    });

    function footerWith(text: string): HTMLElement {
        document.body.innerHTML = `<footer><button class="vx-statusbar__zoom" type="button">${text}</button></footer>`;
        return document.querySelector<HTMLElement>(".vx-statusbar__zoom")!;
    }

    it("paints the live zoom as a percent", () => {
        footerWith("—");
        state.zoomLevel = 0.42;
        updateStageZoomReadout();
        expect(document.querySelector(".vx-statusbar__zoom")?.textContent).toBe("42%");
    });

    it("shows a dash when there is no zoom yet", () => {
        footerWith("99%");
        state.zoomLevel = 0;
        updateStageZoomReadout();
        expect(document.querySelector(".vx-statusbar__zoom")?.textContent).toBe(DASH);
    });

    it("is a no-op without the footer, which is Basic's case", () => {
        // No `.vx-statusbar__zoom` in the document: must not throw.
        expect(() => updateStageZoomReadout()).not.toThrow();
    });

    it("skips the write when the text is already right", () => {
        // Trackpads send dozens of ticks per gesture while the rounded percent
        // barely moves; each redundant write would cost a style recalc.
        const btn = footerWith("110%");
        state.zoomLevel = 1.1;
        let writes = 0;
        const desc = Object.getOwnPropertyDescriptor(Node.prototype, "textContent")!;
        Object.defineProperty(btn, "textContent", {
            configurable: true,
            get: desc.get,
            set(v: string | null) {
                writes++;
                desc.set!.call(this, v);
            },
        });
        updateStageZoomReadout();
        updateStageZoomReadout();
        expect(writes).toBe(0);
        expect(btn.textContent).toBe("110%");
    });
});

describe("zoom readout across the silent paths", () => {
    beforeEach(() => {
        document.body.innerHTML = `
            <div id="imageContainer"><img id="mainImage"></div>
            <footer><button class="vx-statusbar__zoom" type="button">—</button></footer>`;
        document.body.classList.remove("workspace-reader");
        state.zoomLevel = 1;
        state.fitZoomLevel = 1;
        state.isHorizontalFitEnabled = false;
        state.stripMode = false;
        state.panX = 0;
        state.panY = 0;
        const container = document.getElementById("imageContainer")!;
        const img = document.getElementById("mainImage") as HTMLImageElement;
        Object.defineProperty(container, "clientWidth", { value: 800, configurable: true });
        Object.defineProperty(container, "clientHeight", { value: 600, configurable: true });
        Object.defineProperty(img, "naturalWidth", { value: 400, configurable: true });
        Object.defineProperty(img, "naturalHeight", { value: 600, configurable: true });
    });

    const readout = (): string => document.querySelector(".vx-statusbar__zoom")?.textContent ?? "";

    it("follows button and wheel zoom, which emit nothing", () => {
        // The footer only re-renders on page/image/entry events; the wheel and
        // the +/- buttons go through `zoomWorkAround` with no announcement.
        zoomViewerBy(0.1);
        expect(state.zoomLevel).toBeCloseTo(1.1, 10);
        expect(readout()).toBe("110%");
    });

    it("follows a refit, which splitter drags and panel toggles trigger", () => {
        // `refitViewer` recomputes the fit straight onto `state.zoomLevel`,
        // again with no announcement.
        state.zoomLevel = 2;
        const container = document.getElementById("imageContainer")!;
        const img = document.getElementById("mainImage") as HTMLImageElement;
        applyImageFit(container, img);
        expect(state.zoomLevel).toBeCloseTo(1, 10);
        expect(readout()).toBe("100%");
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
