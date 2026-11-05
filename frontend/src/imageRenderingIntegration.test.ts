import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { applyReaderTransform, applyImageFit, renderGallery } from "./viewer";
import { disposeImageTransform } from "./imageTransform";
import { state } from "./state";

let img: HTMLImageElement;
let container: HTMLElement;

beforeEach(() => {
    vi.useFakeTimers();
    document.body.innerHTML =
        '<div id="gallery"></div><div id="imageContainer"><img id="mainImage" src="/img_serve/page.png?t=1"></div>';
    document.body.className = "";
    container = document.getElementById("imageContainer")!;
    img = document.getElementById("mainImage") as HTMLImageElement;
    Object.defineProperties(container, { clientWidth: { value: 800 }, clientHeight: { value: 600 } });
    Object.defineProperties(img, {
        naturalWidth: { value: 1070 },
        naturalHeight: { value: 1600 },
        clientWidth: { value: 1070 },
        clientHeight: { value: 1600 },
    });
    state.svgOverlayElement = null;
    state.stripMode = false;
    state.isHorizontalFitEnabled = false;
    state.zoomLevel = 0.375;
});

afterEach(() => {
    disposeImageTransform(img);
    delete window.__FOX_CONFIG__;
    delete window.foxReaderNativeFullscreen;
    // Retire the gallery manager before the next test replaces its DOM.
    renderGallery([]);
    vi.useRealTimers();
    document.body.innerHTML = "";
    state.svgOverlayElement = null;
});

describe("viewer rendering policy and overlay coordinates", () => {
    it.each([
        [1, false, false],
        [1, true, true],
        [2, false, true],
        [2, true, true],
        [3, false, false],
        [3, true, false],
    ] as const)("mode %s, native %s keeps the image and overlay aligned", (mode, native, enabled) => {
        window.__FOX_CONFIG__ = { imageRenderQuality: mode };
        if (native) window.foxReaderNativeFullscreen = () => {};
        applyReaderTransform(container, img, 0.375, -10.25, 12.5);
        const svg = state.svgOverlayElement!;
        expect(img.style.transform).toBe("translate3d(189.125px, 12.5px, 0) scale(0.375)");
        expect(svg.style.transform).toBe(img.style.transform);
        expect(vi.getTimerCount()).toBe(enabled ? 1 : 0);
        vi.advanceTimersByTime(160);
        expect(img.style.transform).toBe(
            enabled ? "translate(189.125px, 12.5px) scale(0.375)" : "translate3d(189.125px, 12.5px, 0) scale(0.375)",
        );
        expect(svg.style.transform).toBe(img.style.transform);
        expect(svg.getAttribute("viewBox")).toBe("0 0 1070 1600");
        expect(svg.style.width).toBe("1070px");
        expect(svg.style.height).toBe("1600px");
        expect(img.getAttribute("src")).toBe("/img_serve/page.png?t=1");
        expect(state.zoomLevel).toBe(0.375);
    });

    it("settles work-mode fitting with the same overlay geometry", () => {
        window.__FOX_CONFIG__ = { imageRenderQuality: 2 };
        applyImageFit(container, img);
        const moving = img.style.transform;
        vi.advanceTimersByTime(160);
        expect(moving).toBe("translate3d(199.375px, 0px, 0) scale(0.375)");
        expect(img.style.transform).toBe("translate(199.375px, 0px) scale(0.375)");
        expect(state.svgOverlayElement!.style.transform).toBe(img.style.transform);
        expect(state.svgOverlayElement!.getAttribute("viewBox")).toBe("0 0 1070 1600");
    });

    it.each([1, 3] as const)("a browser in mode %s keeps original gallery URLs", (mode) => {
        window.__FOX_CONFIG__ = { imageRenderQuality: mode };
        state.viewVariant = "original";
        state.sameDir = false;
        state.savedFiles = new Set();
        renderGallery(["page.png", "next.png"]);
        const urls = Array.from(document.querySelectorAll<HTMLImageElement>(".gallery-item"), (el) =>
            el.getAttribute("src"),
        );
        expect(urls).toHaveLength(2);
        expect(urls.every((url) => url?.startsWith("/img_serve/"))).toBe(true);
        expect(vi.getTimerCount()).toBe(0);
    });
});
