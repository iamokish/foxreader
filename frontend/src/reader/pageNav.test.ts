import { beforeEach, describe, expect, it, vi } from "vitest";
import { state } from "../state";
import { currentPageIndex, gotoPage, pageCount, wrapIndex } from "./pageNav";

/**
 * A gallery of `n` thumbnails, with `active` marked -- the shape `renderGallery`
 * produces, which is what `pageNav` reads. The click handler is the real one's
 * contract: move `.active`, then select.
 */
function mountGallery(n: number, active = 0): HTMLElement[] {
    document.body.innerHTML = `<div id="gallery"></div>`;
    const gallery = document.getElementById("gallery")!;
    const items: HTMLElement[] = [];
    for (let i = 0; i < n; i++) {
        const item = document.createElement("div");
        item.className = "gallery-item-container";
        if (i === active) item.classList.add("active");
        item.scrollIntoView = vi.fn();
        item.onclick = () => {
            for (const el of items) el.classList.remove("active");
            item.classList.add("active");
            state.currentImageFile = state.currentFiles[i] ?? "";
        };
        gallery.appendChild(item);
        items.push(item);
    }
    return items;
}

beforeEach(() => {
    document.body.innerHTML = "";
    state.currentFiles = [];
    state.currentImageFile = "";
});

describe("wrapIndex", () => {
    it("wraps past the end back to the start", () => {
        expect(wrapIndex(5, 5)).toBe(0);
        expect(wrapIndex(6, 5)).toBe(1);
    });

    it("wraps before the start back to the end", () => {
        expect(wrapIndex(-1, 5)).toBe(4);
        expect(wrapIndex(-6, 5)).toBe(4);
    });

    it("leaves an in-range index alone", () => {
        expect(wrapIndex(3, 5)).toBe(3);
    });

    it("reports -1 for an empty folder", () => {
        expect(wrapIndex(0, 0)).toBe(-1);
    });
});

describe("pageCount", () => {
    it("counts the thumbnails", () => {
        mountGallery(4);
        expect(pageCount()).toBe(4);
    });

    it("falls back to the file list before the gallery exists", () => {
        state.currentFiles = ["a.png", "b.png"];
        expect(pageCount()).toBe(2);
    });
});

describe("currentPageIndex", () => {
    it("reads the active thumbnail", () => {
        mountGallery(4, 2);
        expect(currentPageIndex()).toBe(2);
    });

    it("falls back to the file list when no thumbnail is active", () => {
        // The gap between a folder loading and the first thumbnail being marked.
        state.currentFiles = ["a.png", "b.png", "c.png"];
        state.currentImageFile = "b.png";
        mountGallery(3, -1);
        expect(currentPageIndex()).toBe(1);
    });

    it("reports -1 with nothing loaded", () => {
        expect(currentPageIndex()).toBe(-1);
    });
});

describe("gotoPage", () => {
    it("clicks the thumbnail, so `.active` moves with the page", () => {
        // `navigateGallery` and `updateReaderPageIndicator` both read `.active`;
        // calling `selectImage` directly would leave it on the page just left.
        state.currentFiles = ["a.png", "b.png", "c.png"];
        const items = mountGallery(3, 0);
        gotoPage(2);
        expect(items[2].classList.contains("active")).toBe(true);
        expect(items[0].classList.contains("active")).toBe(false);
        expect(state.currentImageFile).toBe("c.png");
    });

    it("wraps rather than clamping, matching the arrow keys", () => {
        state.currentFiles = ["a.png", "b.png"];
        const items = mountGallery(2, 1);
        gotoPage(2);
        expect(items[0].classList.contains("active")).toBe(true);
    });

    it("scrolls the thumbnail into view", () => {
        state.currentFiles = ["a.png", "b.png", "c.png"];
        const items = mountGallery(3, 0);
        gotoPage(2);
        expect(items[2].scrollIntoView).toHaveBeenCalled();
    });

    it("does nothing at all with no gallery", () => {
        expect(() => gotoPage(1)).not.toThrow();
    });
});
