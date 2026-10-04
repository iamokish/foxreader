import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { initHealthMonitor } from "../core/session";

vi.mock("../components/common", () => ({
    showBackendOffline: vi.fn(),
    showBackendRestarted: vi.fn(),
    hideBackendStatus: vi.fn(),
}));

import { showBackendOffline, showBackendRestarted, hideBackendStatus } from "../components/common";

function mockFetchOnce(data: { session_id: string } | null): void {
    if (data === null) {
        vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new Error("network down"));
        return;
    }
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => data,
    } as Response);
}

async function flush(timers = 0): Promise<void> {
    if (timers > 0) {
        await vi.advanceTimersByTimeAsync(timers);
    } else {
        await vi.advanceTimersByTimeAsync(0);
    }
}

describe("HealthMonitor", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.restoreAllMocks();
        vi.clearAllMocks();
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it("stays online when session matches on poll", async () => {
        mockFetchOnce({ session_id: "s1" });
        mockFetchOnce({ session_id: "s1" });

        const stop = initHealthMonitor();
        await flush(0);
        expect(showBackendOffline).not.toHaveBeenCalled();
        expect(showBackendRestarted).not.toHaveBeenCalled();

        await flush(15_000);
        expect(showBackendOffline).not.toHaveBeenCalled();
        expect(showBackendRestarted).not.toHaveBeenCalled();
        stop();
    });

    it("shows restart overlay when session changes", async () => {
        mockFetchOnce({ session_id: "s1" });
        mockFetchOnce({ session_id: "s2" });

        const stop = initHealthMonitor();
        await flush(0);
        expect(showBackendRestarted).not.toHaveBeenCalled();

        await flush(15_000);
        expect(showBackendRestarted).toHaveBeenCalledOnce();
        stop();
    });

    it("shows offline overlay when health fetch fails", async () => {
        mockFetchOnce({ session_id: "s1" });
        mockFetchOnce(null);

        const stop = initHealthMonitor();
        await flush(0);

        await flush(15_000);
        expect(showBackendOffline).toHaveBeenCalledOnce();
        expect(showBackendRestarted).not.toHaveBeenCalled();
        stop();
    });

    it("shows offline overlay when initial session fetch fails", async () => {
        mockFetchOnce(null);
        mockFetchOnce({ session_id: "s1" });

        const stop = initHealthMonitor();
        await flush(0);
        expect(showBackendOffline).toHaveBeenCalledOnce();

        await flush(15_000);
        expect(hideBackendStatus).toHaveBeenCalledOnce();
        stop();
    });

    it("recovers and hides overlay when backend returns with matching session", async () => {
        mockFetchOnce({ session_id: "s1" });
        mockFetchOnce(null);
        mockFetchOnce({ session_id: "s1" });

        const stop = initHealthMonitor();
        await flush(0);
        await flush(15_000);
        expect(showBackendOffline).toHaveBeenCalledOnce();

        await flush(15_000);
        expect(hideBackendStatus).toHaveBeenCalledOnce();
        stop();
    });

    it("shows restart overlay when backend returns with new session after offline", async () => {
        mockFetchOnce({ session_id: "s1" });
        mockFetchOnce(null);
        mockFetchOnce({ session_id: "s2" });

        const stop = initHealthMonitor();
        await flush(0);
        await flush(15_000);
        expect(showBackendOffline).toHaveBeenCalledOnce();

        await flush(15_000);
        expect(showBackendRestarted).toHaveBeenCalledOnce();
        stop();
    });
});
