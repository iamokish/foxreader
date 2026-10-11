import { beforeEach, describe, expect, it } from "vitest";
import { state } from "../state";
import { buildPanes } from "./Pager";
import { stepTarget } from "./Reader";
import { buildSpreads } from "./viewSettings";

/** The gallery shape `pageNav` reads: `n` thumbnails with one marked active. */
function mountGallery(n: number, active: number): void {
    document.body.innerHTML = `<div id="gallery"></div>`;
    const gallery = document.getElementById("gallery")!;
    for (let i = 0; i < n; i++) {
        const item = document.createElement("div");
        item.className = `gallery-item-container${i === active ? " active" : ""}`;
        gallery.appendChild(item);
    }
}

function loadFolder(files: string[], active = 0): void {
    state.currentFiles = files;
    state.currentImageFile = files[active] ?? "";
    state.aspectCache = {};
    state.savedFiles = new Set();
    state.sameDir = false;
    state.viewVariant = "original";
    mountGallery(files.length, active);
}

const PAGES = ["p1.png", "p2.png", "p3.png", "p4.png", "p5.png"];

beforeEach(() => {
    localStorage.clear();
    document.body.innerHTML = "";
    state.currentFiles = [];
    state.currentImageFile = "";
    state.aspectCache = {};
    state.savedFiles = new Set();
    state.sameDir = false;
    state.viewVariant = "original";
});

describe("buildPanes — single", () => {
    it("is the page on screen, alone", () => {
        loadFolder(PAGES, 2);
        const panes = buildPanes("single", { coverAlone: true, wideAlone: true });
        expect(panes).toHaveLength(1);
        expect(panes[0].pageNumber).toBe(3);
        expect(panes[0].url).toContain("p3.png");
        expect(panes[0].label).toBeNull();
    });

    it("is empty with no folder loaded", () => {
        expect(buildPanes("single", { coverAlone: true, wideAlone: true })).toEqual([]);
    });
});

describe("buildPanes — double", () => {
    it("shows the cover alone when the rule is on", () => {
        loadFolder(PAGES, 0);
        const panes = buildPanes("double", { coverAlone: true, wideAlone: true });
        expect(panes.map((p) => p.pageNumber)).toEqual([1]);
    });

    it("puts the earlier page on the right for right-to-left manga", () => {
        localStorage.setItem("fox-reader-direction", "rtl");
        loadFolder(PAGES, 1);
        const panes = buildPanes("double", { coverAlone: true, wideAlone: true });
        expect(panes.map((p) => p.pageNumber)).toEqual([3, 2]);
    });

    it("puts the earlier page on the left for left-to-right comics", () => {
        localStorage.setItem("fox-reader-direction", "ltr");
        loadFolder(PAGES, 1);
        const panes = buildPanes("double", { coverAlone: true, wideAlone: true });
        expect(panes.map((p) => p.pageNumber)).toEqual([2, 3]);
    });

    it("shows a landscape page on its own", () => {
        localStorage.setItem("fox-reader-direction", "ltr");
        loadFolder(PAGES, 2);
        // Aspect is height/width, so below 1 is wider than tall.
        state.aspectCache = { "p3.png": 0.7 };
        const panes = buildPanes("double", { coverAlone: false, wideAlone: true });
        expect(panes.map((p) => p.pageNumber)).toEqual([3]);
    });

    it("pairs the page the cursor is on even when it is the right half", () => {
        localStorage.setItem("fox-reader-direction", "ltr");
        loadFolder(PAGES, 2);
        // With the cover alone, page 3 (index 2) is the left of the 2-3 spread.
        expect(buildPanes("double", { coverAlone: true, wideAlone: true }).map((p) => p.pageNumber)).toEqual([2, 3]);
    });
});

describe("buildPanes — compare", () => {
    it("asks for the original on the left and the saved copy on the right", () => {
        loadFolder(PAGES, 1);
        state.savedFiles = new Set(["p2.png"]);
        const panes = buildPanes("compare", { coverAlone: true, wideAlone: true });
        expect(panes).toHaveLength(2);
        expect(panes[0].url).not.toContain("variant=saved");
        expect(panes[1].url).toContain("variant=saved");
        expect(panes[0].label).toBe("Before · original");
        expect(panes[1].label).toBe("After · saved");
    });

    it("shows both sides regardless of which copy the switch is on", () => {
        // The Original/Saved switch has one position; a comparison needs both, so
        // it must not be read from `state.viewVariant`.
        loadFolder(PAGES, 1);
        state.savedFiles = new Set(["p2.png"]);
        state.viewVariant = "saved";
        const panes = buildPanes("compare", { coverAlone: true, wideAlone: true });
        expect(panes[0].url).not.toContain("variant=saved");
        expect(panes[1].url).toContain("variant=saved");
    });

    it("says so, rather than showing a blank, for a page with nothing typeset yet", () => {
        loadFolder(PAGES, 1);
        const panes = buildPanes("compare", { coverAlone: true, wideAlone: true });
        expect(panes[1].label).toBe("After · not saved yet");
        // Never asks the server for a variant that does not exist.
        expect(panes[1].url).not.toContain("variant=saved");
    });

    it("does not ask for a variant when the destination is the source", () => {
        loadFolder(PAGES, 1);
        state.sameDir = true;
        state.savedFiles = new Set(["p2.png"]);
        const panes = buildPanes("compare", { coverAlone: true, wideAlone: true });
        expect(panes[1].url).not.toContain("variant=saved");
    });

    it("gives the two panes distinct keys, so neither reuses the other's image", () => {
        loadFolder(PAGES, 1);
        state.savedFiles = new Set(["p2.png"]);
        const panes = buildPanes("compare", { coverAlone: true, wideAlone: true });
        expect(panes[0].key).not.toBe(panes[1].key);
    });
});

describe("stepTarget", () => {
    it("moves one page in the single view", () => {
        expect(stepTarget("single", 2, 5, 1, [])).toBe(3);
        expect(stepTarget("single", 2, 5, -1, [])).toBe(1);
    });

    it("wraps at both ends", () => {
        expect(stepTarget("single", 4, 5, 1, [])).toBe(0);
        expect(stepTarget("single", 0, 5, -1, [])).toBe(4);
    });

    it("moves a whole spread in the double view", () => {
        const spreads = buildSpreads(6, { coverAlone: true, wideAlone: false });
        // [[0], [1,2], [3,4], [5]] — from page 2 (index 1) forward is index 3.
        expect(stepTarget("double", 1, 6, 1, spreads)).toBe(3);
        expect(stepTarget("double", 2, 6, 1, spreads)).toBe(3);
    });

    it("lands on the first page of the spread it moves back into", () => {
        const spreads = buildSpreads(6, { coverAlone: true, wideAlone: false });
        expect(stepTarget("double", 3, 6, -1, spreads)).toBe(1);
    });

    it("steps one page across a spread that stands alone", () => {
        // A cover is a spread of one, so leaving it moves a single page.
        const spreads = buildSpreads(6, { coverAlone: true, wideAlone: false });
        expect(stepTarget("double", 0, 6, 1, spreads)).toBe(1);
    });

    it("wraps around the spread list, not the page list", () => {
        const spreads = buildSpreads(6, { coverAlone: true, wideAlone: false });
        // Last spread is [5]; forward wraps to the cover.
        expect(stepTarget("double", 5, 6, 1, spreads)).toBe(0);
    });

    it("falls back to a single page when the spread table has no entry", () => {
        expect(stepTarget("double", 2, 5, 1, [])).toBe(3);
    });

    it("reports -1 with nothing loaded", () => {
        expect(stepTarget("single", -1, 0, 1, [])).toBe(-1);
    });
});
