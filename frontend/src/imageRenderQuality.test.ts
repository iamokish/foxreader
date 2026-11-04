import { afterEach, describe, expect, it, vi } from "vitest";
import { imageRenderQualityEnabled, isNativeViewer } from "./imageRenderQuality";

afterEach(() => {
    delete window.__FOX_CONFIG__;
    delete window.foxReaderNativeFullscreen;
    vi.unstubAllGlobals();
});

describe("image rendering scope", () => {
    it.each([
        [undefined, false, false],
        [undefined, true, true],
        [1, false, false],
        [1, true, true],
        [2, false, true],
        [2, true, true],
        [3, false, false],
        [3, true, false],
    ] as const)("mode %s, native %s enables improvements: %s", (mode, native, expected) => {
        window.__FOX_CONFIG__ = { imageRenderQuality: mode };
        if (native) window.foxReaderNativeFullscreen = () => {};
        expect(isNativeViewer()).toBe(native);
        expect(imageRenderQualityEnabled()).toBe(expected);
    });

    it("does not identify an ordinary browser by its OS or Chromium globals", () => {
        vi.stubGlobal("chrome", { webview: {} });
        expect(imageRenderQualityEnabled()).toBe(false);
    });

    it("keeps improvements off for an unexpected injected mode", () => {
        window.__FOX_CONFIG__ = { imageRenderQuality: 99 as 1 };
        window.foxReaderNativeFullscreen = () => {};
        expect(imageRenderQualityEnabled()).toBe(false);
    });
});
