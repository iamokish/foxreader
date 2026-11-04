import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { applyImageTransform, disposeImageTransform } from "./imageTransform";

const quality = vi.hoisted(() => ({ enabled: true }));
vi.mock("./imageRenderQuality", () => ({ imageRenderQualityEnabled: () => quality.enabled }));

let image: HTMLImageElement;

beforeEach(() => {
    vi.useFakeTimers();
    quality.enabled = true;
    image = document.createElement("img");
    image.style.width = "1070px";
    image.style.height = "1600px";
    image.style.transformOrigin = "0 0";
    document.body.appendChild(image);
});

afterEach(() => {
    disposeImageTransform(image);
    image.remove();
    vi.clearAllTimers();
    vi.useRealTimers();
});

describe("image transform rasterization", () => {
    it("settles with exactly the same fractional placement and scale, then synchronizes the overlay", () => {
        const settled = vi.fn(() => {
            expect(image.style.transform).toBe("translate(-12.25px, 31.125px) scale(0.3)");
            expect(image.style.willChange).toBe("auto");
        });
        applyImageTransform(image, -12.25, 31.125, 0.3, settled);
        expect(image.style.transform).toBe("translate3d(-12.25px, 31.125px, 0) scale(0.3)");
        expect(image.style.willChange).toBe("transform");
        vi.advanceTimersByTime(149);
        expect(settled).not.toHaveBeenCalled();
        vi.advanceTimersByTime(1);
        expect(settled).toHaveBeenCalledOnce();
        expect(image.style.width).toBe("1070px");
        expect(image.style.height).toBe("1600px");
        expect(image.style.transformOrigin).toBe("0 0");
    });

    it("coalesces a gesture and settles only the newest transform", () => {
        const stale = vi.fn();
        const latest = vi.fn();
        applyImageTransform(image, 1, 2, 0.3, stale);
        vi.advanceTimersByTime(100);
        applyImageTransform(image, -58.75, 42.5, 0.8, latest);
        vi.advanceTimersByTime(100);
        expect(image.style.transform).toBe("translate3d(-58.75px, 42.5px, 0) scale(0.8)");
        expect(stale).not.toHaveBeenCalled();
        expect(latest).not.toHaveBeenCalled();
        expect(vi.getTimerCount()).toBe(1);
        vi.advanceTimersByTime(50);
        expect(image.style.transform).toBe("translate(-58.75px, 42.5px) scale(0.8)");
        expect(latest).toHaveBeenCalledOnce();
        expect(stale).not.toHaveBeenCalled();
    });

    it("accepts CSSOM normalization without skipping settling or promoting an unchanged view", () => {
        let serialized = "";
        Object.defineProperty(image.style, "transform", {
            configurable: true,
            get: () => serialized,
            set: (value: string) => {
                serialized = value
                    .replace(", 0) scale", ", 0px) scale")
                    .replace("translate(0px, 0px)", "translate(0px)");
            },
        });
        applyImageTransform(image, 0, 0, 0.5);
        expect(image.style.transform).toBe("translate3d(0px, 0px, 0px) scale(0.5)");
        vi.advanceTimersByTime(200);
        expect(image.style.transform).toBe("translate(0px) scale(0.5)");
        expect(image.style.willChange).toBe("auto");
        applyImageTransform(image, 0, 0, 0.5);
        expect(vi.getTimerCount()).toBe(0);
    });

    it("leaves baseline rendering and layer hints untouched in disabled viewers", () => {
        quality.enabled = false;
        image.style.willChange = "opacity";
        const settled = vi.fn();
        applyImageTransform(image, 5, 6, 0.5, settled);
        expect(image.style.transform).toBe("translate3d(5px, 6px, 0) scale(0.5)");
        expect(image.style.willChange).toBe("opacity");
        expect(vi.getTimerCount()).toBe(0);
        vi.advanceTimersByTime(500);
        expect(settled).not.toHaveBeenCalled();
    });

    it("restores the original hint if disabled before the gesture settles", () => {
        image.style.willChange = "opacity";
        const settled = vi.fn();
        applyImageTransform(image, 0, 0, 1, settled);
        quality.enabled = false;
        vi.advanceTimersByTime(200);
        expect(image.style.transform).toBe("translate3d(0px, 0px, 0) scale(1)");
        expect(image.style.willChange).toBe("opacity");
        expect(settled).not.toHaveBeenCalled();
        expect(vi.getTimerCount()).toBe(0);
    });

    it("returns a settled element to baseline when its next update is disabled", () => {
        applyImageTransform(image, 0, 0, 0.5);
        vi.advanceTimersByTime(200);
        quality.enabled = false;
        applyImageTransform(image, 10, 20, 0.75);
        expect(image.style.transform).toBe("translate3d(10px, 20px, 0) scale(0.75)");
        expect(image.style.willChange).toBe("");
        expect(vi.getTimerCount()).toBe(0);
    });

    it("does not rasterize again when a settled view is repainted without moving", () => {
        applyImageTransform(image, 0, 0, 0.5);
        vi.advanceTimersByTime(200);
        applyImageTransform(image, 0, 0, 0.5);
        expect(image.style.transform).toBe("translate(0px, 0px) scale(0.5)");
        expect(image.style.willChange).toBe("auto");
        expect(vi.getTimerCount()).toBe(0);
    });

    it("does not overwrite a later independent transform", () => {
        const settled = vi.fn();
        applyImageTransform(image, 0, 0, 0.5, settled);
        image.style.transform = "none";
        vi.advanceTimersByTime(200);
        expect(image.style.transform).toBe("none");
        expect(image.style.willChange).toBe("");
        expect(settled).not.toHaveBeenCalled();
    });

    it("skips removed images and cancels explicitly disposed ones", () => {
        const settled = vi.fn();
        applyImageTransform(image, 0, 0, 0.5, settled);
        image.remove();
        vi.advanceTimersByTime(200);
        expect(settled).not.toHaveBeenCalled();
        expect(image.style.transform).toBe("translate3d(0px, 0px, 0) scale(0.5)");
        document.body.appendChild(image);
        applyImageTransform(image, 5, 6, 0.75, settled);
        disposeImageTransform(image);
        expect(vi.getTimerCount()).toBe(0);
        vi.advanceTimersByTime(200);
        expect(settled).not.toHaveBeenCalled();
        expect(image.style.transform).toBe("translate3d(5px, 6px, 0) scale(0.75)");
    });
});
