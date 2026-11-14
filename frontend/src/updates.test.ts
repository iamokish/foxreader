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
            <p id="updateInstallHint" hidden>Save your work first. Fox Reader will close.</p>
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

    it("does not start update requests when the source settings page omits the updater", async () => {
        document.body.innerHTML = '<main><h1>Settings</h1><div id="settings"></div></main>';
        cleanup = installUpdates();
        await vi.advanceTimersByTimeAsync(60_000);
        expect(fetch).not.toHaveBeenCalled();
        expect(vi.getTimerCount()).toBe(0);
        expect(document.getElementById("updatesPanel")).toBeNull();
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
        expect(element("downloadUpdateBtn").textContent).toBe("Download update");
        expect(element("updateInstallHint").hidden).toBe(true);
    });

    it("downloads and installs through the existing action, then stops polling after handoff", async () => {
        const installed = { ...idle, install_supported: true };
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => installed } as Response);
        cleanup = installUpdates();
        await settle();
        expect(element("downloadUpdateBtn").textContent).toBe("Download and install");
        expect(element("updateInstallHint").hidden).toBe(false);
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => release } as Response);
        element("checkUpdateBtn").click();
        await settle();
        vi.mocked(fetch)
            .mockResolvedValueOnce({ ok: true, json: async () => ({ ...installed, state: "downloading" }) } as Response)
            .mockResolvedValueOnce({ ok: true, json: async () => ({ ...installed, state: "preparing", message: "Checking files" }) } as Response);
        element("downloadUpdateBtn").click();
        await settle();
        expect(vi.mocked(fetch).mock.calls[2][0]).toBe("/api/updates/download");
        expect(vi.mocked(fetch).mock.calls[2][1]!.method).toBe("POST");
        expect(element("cancelUpdateBtn").hidden).toBe(false);
        expect(element("cancelUpdateBtn").textContent).toBe("Cancel update");
        expect(element("updateStatus").textContent).toBe("Checking files");
        expect(element("updateProgress").hasAttribute("value")).toBe(false);
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({ ...installed, state: "handoff" }) } as Response);
        await vi.advanceTimersByTimeAsync(1000);
        expect(element("cancelUpdateBtn").hidden).toBe(true);
        expect((element("checkUpdateBtn") as HTMLButtonElement).disabled).toBe(true);
        expect(element("updateStatus").textContent).toContain("Updater");
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({ ...installed, state: "installing" }) } as Response);
        await vi.advanceTimersByTimeAsync(1000);
        const requests = vi.mocked(fetch).mock.calls.length;
        await vi.advanceTimersByTimeAsync(60_000);
        expect(fetch).toHaveBeenCalledTimes(requests);
        expect(element("updateStatus").textContent).toContain("Start Fox Reader yourself");
        expect(element("updateReady").hidden).toBe(true);
    });

    it("keeps checking an unconfirmed handoff after a lost request so failures remain recoverable", async () => {
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({
            ...idle, install_supported: true, state: "handoff",
        }) } as Response);
        cleanup = installUpdates();
        await settle();
        vi.mocked(fetch).mockRejectedValueOnce(new Error("connection interrupted"));
        await vi.advanceTimersByTimeAsync(1000);
        expect(element("updateStatus").textContent).toContain("Continue in the Fox Reader Updater");
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({
            ...idle, install_supported: true, state: "error", message: "The updater could not open safely.",
        }) } as Response);
        await vi.advanceTimersByTimeAsync(2000);
        expect(element("updateStatus").textContent).toBe("The updater could not open safely.");
        expect((element("checkUpdateBtn") as HTMLButtonElement).disabled).toBe(false);
        await vi.advanceTimersByTimeAsync(60_000);
        expect(fetch).toHaveBeenCalledTimes(3);
        expect(element("cancelUpdateBtn").hidden).toBe(true);
    });

    it("reconciles a lost download response with the backend's active update", async () => {
        cleanup = installUpdates();
        await settle();
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => release } as Response);
        element("checkUpdateBtn").click();
        await settle();
        vi.mocked(fetch)
            .mockRejectedValueOnce(new Error("download response interrupted"))
            .mockResolvedValueOnce({ ok: true, json: async () => ({
                ...idle, install_supported: true, state: "preparing", message: "Preparing verified files",
            }) } as Response);
        element("downloadUpdateBtn").click();
        await settle();
        expect(vi.mocked(fetch).mock.calls[3][0]).toBe("/api/updates/status");
        expect(element("updateStatus").textContent).toBe("Preparing verified files");
        expect(element("cancelUpdateBtn").hidden).toBe(false);
        expect((element("checkUpdateBtn") as HTMLButtonElement).disabled).toBe(true);
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({
            ...idle, install_supported: true, state: "installing",
        }) } as Response);
        await vi.advanceTimersByTimeAsync(1000);
        expect(element("updateStatus").textContent).toContain("Updater window");
        expect(element("cancelUpdateBtn").hidden).toBe(true);
    });

    it("continues polling preparation and restores controls when preparation fails", async () => {
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({
            ...idle, install_supported: true, state: "preparing",
        }) } as Response);
        cleanup = installUpdates();
        await settle();
        vi.mocked(fetch).mockRejectedValueOnce(new Error("temporary connection failure"));
        await vi.advanceTimersByTimeAsync(1000);
        expect(element("updateStatus").textContent).toContain("Waiting for the backend");
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({
            ...idle, install_supported: true, state: "error", message: "Not enough disk space.",
        }) } as Response);
        await vi.advanceTimersByTimeAsync(2000);
        expect(element("updateStatus").textContent).toBe("Not enough disk space.");
        expect((element("checkUpdateBtn") as HTMLButtonElement).disabled).toBe(false);
        expect(element("cancelUpdateBtn").hidden).toBe(true);
    });

    it("bounds reconnect attempts when shutdown hides the final installing status", async () => {
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({
            ...idle, install_supported: true, state: "handoff",
        }) } as Response);
        cleanup = installUpdates();
        await settle();
        vi.mocked(fetch).mockRejectedValue(new Error("backend closed"));
        await vi.advanceTimersByTimeAsync(60_000);
        const requests = vi.mocked(fetch).mock.calls.length;
        expect(requests).toBeGreaterThan(2);
        await vi.advanceTimersByTimeAsync(60_000);
        expect(fetch).toHaveBeenCalledTimes(requests);
        expect(element("updateStatus").textContent).toContain("Continue in the Fox Reader Updater");
        expect(vi.getTimerCount()).toBe(0);
    });

    it("refreshes status if cancellation races with the final handoff", async () => {
        vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({
            ...idle, install_supported: true, state: "preparing",
        }) } as Response);
        cleanup = installUpdates();
        await settle();
        vi.mocked(fetch)
            .mockResolvedValueOnce({ ok: false, json: async () => ({ error: "Already handing off" }) } as Response)
            .mockResolvedValueOnce({ ok: true, json: async () => ({ ...idle, state: "installing", install_supported: true }) } as Response);
        element("cancelUpdateBtn").click();
        await settle();
        expect(vi.mocked(fetch).mock.calls[1][0]).toBe("/api/updates/cancel");
        expect(vi.mocked(fetch).mock.calls[2][0]).toBe("/api/updates/status");
        expect(element("cancelUpdateBtn").hidden).toBe(true);
        expect(element("updateStatus").textContent).toContain("Updater window");
        await vi.advanceTimersByTimeAsync(60_000);
        expect(fetch).toHaveBeenCalledTimes(3);
    });
});
