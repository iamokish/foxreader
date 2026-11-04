import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installUpdates } from "../static/updates.js";

const idle = { state: "idle", current_version: "0.2.1", download_supported: true };
const release = {
    available: true, current_version: "0.2.1", version: "0.2.2", compatible: true,
    variant: "cpu-gguf", download_size: 1024, notes: "<img src=x onerror=alert(1)>",
    release_url: "https://github.com/example/fox-reader/releases/tag/v0.2.2",
};

describe("update settings", () => {
    let cleanup: (() => void) | undefined;
    const element = (id: string) => document.getElementById(id)!;
    const settle = async () => { await vi.advanceTimersByTimeAsync(0); };
    beforeEach(() => {
        vi.useFakeTimers();
        document.body.innerHTML = `<section id="updatesPanel">
            <button id="checkUpdateBtn"></button><button id="downloadUpdateBtn"></button>
            <button id="cancelUpdateBtn"></button><p id="updateStatus"></p><p id="updateVersion"></p>
            <progress id="updateProgress"></progress><div id="updateReady"></div>
            <p id="updateArchive"></p><details id="updateDetails"></details><pre id="updateNotes"></pre>
            <a id="updateReleaseLink"></a></section>`;
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => idle }));
    });
    afterEach(() => {
        cleanup?.();
        cleanup = undefined;
        vi.unstubAllGlobals();
        vi.useRealTimers();
        document.body.innerHTML = "";
    });

    it("reads local status without checking GitHub at startup", async () => {
        cleanup = installUpdates();
        await settle();
        expect(fetch).toHaveBeenCalledOnce();
        expect(vi.mocked(fetch).mock.calls[0][0]).toBe("/api/updates/status");
        expect(element("updateVersion").textContent).toContain("0.2.1");
        expect(element("downloadUpdateBtn").hidden).toBe(true);
    });

    it("checks explicitly and treats release notes as plain text", async () => {
        cleanup = installUpdates();
        await settle();
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => release } as Response);
        element("checkUpdateBtn").click();
        await settle();
        expect(vi.mocked(fetch).mock.calls[1][0]).toBe("/api/updates/check");
        expect(vi.mocked(fetch).mock.calls[1][1]!.headers).toHaveProperty("X-Fox-Reader-Update", "1");
        expect(element("updateNotes").textContent).toBe(release.notes);
        expect(element("updateNotes").querySelector("img")).toBeNull();
        expect(element("downloadUpdateBtn").hidden).toBe(false);
    });

    it("offers no incompatible download and rejects unsafe release links", async () => {
        cleanup = installUpdates();
        await settle();
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({
            ...release, compatible: false, release_url: "javascript:alert(1)", reason: "No matching build.",
        }) } as Response);
        element("checkUpdateBtn").click();
        await settle();
        expect(element("downloadUpdateBtn").hidden).toBe(true);
        expect(element("updateReleaseLink").hidden).toBe(true);
        expect(element("updateStatus").textContent).toBe("No matching build.");
    });

    it("restores progress after navigation and cancels with the required header", async () => {
        const downloading = { ...idle, state: "downloading", downloaded: 50, total: 100, message: "Downloading…" };
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => downloading } as Response);
        cleanup = installUpdates();
        await settle();
        expect(element("cancelUpdateBtn").hidden).toBe(false);
        expect((element("updateProgress") as HTMLProgressElement).value).toBe(50);
        vi.mocked(fetch).mockResolvedValue({ ok: true, json: async () => ({ ...idle, state: "cancelled" }) } as Response);
        element("cancelUpdateBtn").click();
        await settle();
        expect(vi.mocked(fetch).mock.calls[1][0]).toBe("/api/updates/cancel");
        expect(vi.mocked(fetch).mock.calls[1][1]).toMatchObject({
            method: "POST", headers: { "X-Fox-Reader-Update": "1" },
        });
        expect(element("updateStatus").textContent).toBe("Download cancelled.");
        expect(element("cancelUpdateBtn").hidden).toBe(true);
    });

    it("displays only a verified completed archive as ready", async () => {
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({
            ...idle, state: "complete", archive: "/updates/fox-reader-v0.2.2.zip",
        }) } as Response);
        cleanup = installUpdates();
        await settle();
        expect(element("updateReady").hidden).toBe(false);
        expect(element("updateArchive").textContent).toBe("/updates/fox-reader-v0.2.2.zip");
    });
});
