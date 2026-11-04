import { act, cleanup, fireEvent, render } from "@testing-library/preact";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Pager } from "./Pager";
import { state } from "../state";

const quality = vi.hoisted(() => ({ enabled: true }));
vi.mock("../imageRenderQuality", () => ({
    imageRenderQualityEnabled: () => quality.enabled,
    isNativeViewer: () => true,
}));

const props = {
    view: "single" as const,
    fit: "page" as const,
    gap: 8,
    coverAlone: true,
    wideAlone: true,
    revision: 0,
};

function loadImage(container: Element): HTMLImageElement {
    const image = container.querySelector(".vx-pane__img") as HTMLImageElement;
    Object.defineProperty(image, "naturalWidth", { value: 1200, configurable: true });
    Object.defineProperty(image, "naturalHeight", { value: 1800, configurable: true });
    fireEvent.load(image);
    return image;
}

function settle(): void {
    act(() => {
        vi.advanceTimersByTime(250);
    });
}

beforeEach(() => {
    vi.useFakeTimers();
    quality.enabled = true;
    localStorage.clear();
    state.currentFiles = ["page.png"];
    state.currentImageFile = "page.png";
    state.aspectCache = { "page.png": 1.5 };
    state.savedFiles = new Set();
    state.sameDir = false;
    state.viewVariant = "original";
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(function (this: HTMLElement) {
        return this.classList.contains("vx-pager") ? 900 : 0;
    });
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
        return this.classList.contains("vx-pager") ? 700 : 0;
    });
});

afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.clearAllTimers();
    vi.useRealTimers();
});

describe("pager rendering quality", () => {
    it("keeps page dimensions and original URLs while settling the row", () => {
        const { container } = render(<Pager {...props} />);
        const image = loadImage(container);
        const row = container.querySelector(".vx-pager__row") as HTMLElement;
        expect(row.style.transform).toBe(`translate3d(0px, 0px, 0) scale(${700 / 1800})`);
        settle();
        expect(row.style.transform).toBe(`translate(0px, 0px) scale(${700 / 1800})`);
        expect(row.style.willChange).toBe("auto");
        expect(image.style.width).toBe("1200px");
        expect(image.style.height).toBe("1800px");
        expect(image.getAttribute("src")).toContain("/img_serve/page.png");
    });

    it("does not restore the GPU transform during unrelated Preact renders", () => {
        const { container, rerender } = render(<Pager {...props} />);
        loadImage(container);
        settle();
        const row = container.querySelector(".vx-pager__row") as HTMLElement;
        const resting = row.style.transform;
        rerender(<Pager {...props} onZoomChange={vi.fn()} />);
        expect(row.style.transform).toBe(resting);
        expect(row.style.willChange).toBe("auto");
    });

    it("promotes wheel movement and settles at the new zoom without resizing the page", () => {
        const { container } = render(<Pager {...props} />);
        const image = loadImage(container);
        settle();
        const row = container.querySelector(".vx-pager__row") as HTMLElement;
        const stage = container.querySelector(".vx-pager") as HTMLElement;
        vi.spyOn(stage, "getBoundingClientRect").mockReturnValue({
            left: 0,
            top: 0,
            width: 900,
            height: 700,
            right: 900,
            bottom: 700,
            x: 0,
            y: 0,
            toJSON: () => ({}),
        });
        fireEvent.wheel(stage, { deltaY: -100, clientX: 450, clientY: 350 });
        const moving = row.style.transform;
        expect(moving).toMatch(/^translate3d\(0px, 0px, 0\) scale\(/);
        expect(row.style.willChange).toBe("transform");
        settle();
        expect(row.style.transform).toBe(moving.replace("translate3d(0px, 0px, 0)", "translate(0px, 0px)"));
        expect(row.style.willChange).toBe("auto");
        expect(image.style.width).toBe("1200px");
        expect(image.style.height).toBe("1800px");
    });

    it("preserves the existing GPU path in disabled viewers", () => {
        quality.enabled = false;
        const { container } = render(<Pager {...props} />);
        loadImage(container);
        settle();
        const row = container.querySelector(".vx-pager__row") as HTMLElement;
        expect(row.style.transform).toBe(`translate3d(0px, 0px, 0) scale(${700 / 1800})`);
        expect(row.style.willChange).toBe("");
    });

    it("cancels pending work when the row is removed or the pager unmounts", () => {
        const { container, rerender, unmount } = render(<Pager {...props} />);
        loadImage(container);
        const previous = container.querySelector(".vx-pager__row") as HTMLElement;
        state.currentFiles = [];
        state.currentImageFile = "";
        rerender(<Pager {...props} revision={1} />);
        expect(container.querySelector(".vx-pager__row")).toBeNull();
        settle();
        expect(previous.style.transform).toBe("");
        expect(previous.style.willChange).toBe("");

        state.currentFiles = ["next.png"];
        state.currentImageFile = "next.png";
        rerender(<Pager {...props} revision={2} />);
        loadImage(container);
        const next = container.querySelector(".vx-pager__row") as HTMLElement;
        unmount();
        settle();
        expect(next.style.transform).toBe("");
        expect(next.style.willChange).toBe("");
    });
});
