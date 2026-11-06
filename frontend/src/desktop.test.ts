import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installDesktopIntegration, toggleFullscreen } from "../static/desktop.js";

describe("desktop integration", () => {
    let cleanup: (() => void) | undefined;
    let originalRootClasses: string;
    beforeEach(() => {
        originalRootClasses = document.documentElement.className;
        vi.useFakeTimers();
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ status: 204 }));
        Object.defineProperty(document.documentElement, "requestFullscreen", {
            configurable: true, value: vi.fn().mockResolvedValue(undefined),
        });
    });
    afterEach(() => {
        cleanup?.();
        cleanup = undefined;
        document.documentElement.className = originalRootClasses;
        delete window.foxReaderNativeFullscreen;
        vi.unstubAllGlobals();
        vi.useRealTimers();
    });

    it("uses the native viewer for reader fullscreen", () => {
        const native = vi.fn();
        window.foxReaderNativeFullscreen = native;
        toggleFullscreen();
        expect(native).toHaveBeenCalledOnce();
        expect(document.documentElement.requestFullscreen).not.toHaveBeenCalled();
    });

    it("enables native selection styling without replacing the current theme or other root classes", () => {
        const root = document.documentElement;
        root.classList.add("theme-light", "existing-root-class");
        window.foxReaderNativeFullscreen = vi.fn();

        cleanup = installDesktopIntegration();
        expect(root.classList.contains("fox-native")).toBe(true);
        expect(root.classList.contains("theme-light")).toBe(true);
        expect(root.classList.contains("existing-root-class")).toBe(true);
        expect(window.foxReaderNativeFullscreen).not.toHaveBeenCalled();

        cleanup();
        cleanup = undefined;
        expect(root.classList.contains("fox-native")).toBe(false);
        expect(root.classList.contains("theme-light")).toBe(true);
        expect(root.classList.contains("existing-root-class")).toBe(true);
    });

    it("keeps browser selection behavior even when Chromium globals are present", () => {
        vi.stubGlobal("chrome", { webview: {} });
        cleanup = installDesktopIntegration();
        expect(document.documentElement.classList.contains("fox-native")).toBe(false);
    });

    it("reuses the native integration and restores its marker after cleanup and reinstall", () => {
        window.foxReaderNativeFullscreen = vi.fn();
        cleanup = installDesktopIntegration();
        const firstCleanup = cleanup;
        expect(installDesktopIntegration()).toBe(firstCleanup);
        expect(document.documentElement.classList.contains("fox-native")).toBe(true);

        firstCleanup();
        expect(document.documentElement.classList.contains("fox-native")).toBe(false);
        cleanup = installDesktopIntegration();
        expect(cleanup).not.toBe(firstCleanup);
        expect(document.documentElement.classList.contains("fox-native")).toBe(true);

        cleanup();
        delete window.foxReaderNativeFullscreen;
        cleanup = installDesktopIntegration();
        expect(document.documentElement.classList.contains("fox-native")).toBe(false);
    });

    it("handles Alt+Enter and F11 while typing and ignores held keys", () => {
        cleanup = installDesktopIntegration();
        const input = document.createElement("input");
        document.body.append(input);
        input.focus();
        input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", altKey: true, bubbles: true }));
        window.dispatchEvent(new KeyboardEvent("keydown", { key: "F11", repeat: true }));
        window.dispatchEvent(new KeyboardEvent("keydown", { key: "F11" }));
        expect(document.documentElement.requestFullscreen).toHaveBeenCalledTimes(2);
        input.remove();
    });

    it("installs once and removes listeners on cleanup", () => {
        cleanup = installDesktopIntegration();
        expect(installDesktopIntegration()).toBe(cleanup);
        cleanup();
        cleanup = undefined;
        window.dispatchEvent(new KeyboardEvent("keydown", { key: "F11" }));
        expect(document.documentElement.requestFullscreen).not.toHaveBeenCalled();
    });

    it("persists a bounded batch of redacted errors without serializing objects", async () => {
        cleanup = installDesktopIntegration();
        window.dispatchEvent(new window.ErrorEvent("error", {
            message: "api_key=private failed https://host/path?token=private",
            filename: "http://localhost/static/app.js?token=private", lineno: 5, colno: 2,
        }));
        await vi.advanceTimersByTimeAsync(1000);
        expect(fetch).toHaveBeenCalledOnce();
        const [url, options] = vi.mocked(fetch).mock.calls[0];
        expect(url).toBe("/api/frontend-log");
        const body = JSON.parse(options!.body as string);
        expect(JSON.stringify(body)).not.toContain("private");
        expect(body[1].source).toBe("/static/app.js:5:2");
        expect(options!.headers).toHaveProperty("X-Fox-Reader-Client", "frontend");
    });

    it("never retries failed diagnostics", async () => {
        vi.mocked(fetch).mockRejectedValue(new Error("offline"));
        cleanup = installDesktopIntegration();
        await vi.advanceTimersByTimeAsync(10000);
        expect(fetch).toHaveBeenCalledOnce();
    });
});
