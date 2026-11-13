import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { initHealthMonitor } from "../core/session";

vi.mock("../components/common/BackendStatusOverlay", () => ({
    showBackendOffline: vi.fn(),
    showBackendRestarted: vi.fn(),
    hideBackendStatus: vi.fn(),
    disposeBackendStatusOverlay: vi.fn(),
}));
import {
    showBackendOffline, showBackendRestarted, hideBackendStatus,
    disposeBackendStatusOverlay,
} from "../components/common/BackendStatusOverlay";

function response(data: unknown = { session_id: "s1" }): Response {
    return { ok: true, json: async () => data } as Response;
}
function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((done) => { resolve = done; });
    return { promise, resolve };
}
const flush = (ms = 0) => vi.advanceTimersByTimeAsync(ms);
const retry = () => vi.mocked(showBackendOffline).mock.calls.at(-1)![0]!();

describe("reader and template health monitor", () => {
    let stop: (() => void) | undefined;
    beforeEach(() => {
        vi.useFakeTimers();
        vi.clearAllMocks();
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response()));
        document.body.innerHTML = '<dialog id="backendStatusOverlay"></dialog>';
    });
    afterEach(() => {
        stop?.();
        stop = undefined;
        document.body.innerHTML = "";
        vi.unstubAllGlobals();
        vi.useRealTimers();
    });

    it("reads the session once, then polls health every 15 seconds without caching", async () => {
        stop = initHealthMonitor();
        await flush();
        expect(vi.mocked(fetch).mock.calls[0][0]).toBe("/api/session");
        await flush(14_999);
        expect(fetch).toHaveBeenCalledOnce();
        await flush(1);
        await flush(15_000);
        expect(vi.mocked(fetch).mock.calls.map(([url]) => url)).toEqual([
            "/api/session", "/api/health", "/api/health",
        ]);
        for (const [, options] of vi.mocked(fetch).mock.calls) {
            expect(options).toMatchObject({ cache: "no-store", credentials: "same-origin" });
            expect(options!.signal).toBeInstanceOf(AbortSignal);
        }
        expect(showBackendOffline).not.toHaveBeenCalled();
        expect(showBackendRestarted).not.toHaveBeenCalled();
    });

    it("recovers the unchanged session after a temporary outage", async () => {
        vi.mocked(fetch).mockResolvedValueOnce(response()).mockRejectedValueOnce(new Error("offline"));
        stop = initHealthMonitor();
        await flush(15_000);
        expect(showBackendOffline).toHaveBeenCalledOnce();
        await flush(15_000);
        expect(hideBackendStatus).toHaveBeenCalledOnce();
        expect(showBackendRestarted).not.toHaveBeenCalled();
    });

    it("uses the first healthy identity after startup fails", async () => {
        vi.mocked(fetch).mockRejectedValueOnce(new Error("offline"));
        stop = initHealthMonitor();
        await flush();
        expect(showBackendOffline).toHaveBeenCalledOnce();
        retry();
        await flush();
        expect(vi.mocked(fetch).mock.calls[1][0]).toBe("/api/health");
        expect(hideBackendStatus).toHaveBeenCalledOnce();
        expect(showBackendRestarted).not.toHaveBeenCalled();
    });

    it("detects a restart after an outage and latches it through later failures and identities", async () => {
        vi.mocked(fetch)
            .mockResolvedValueOnce(response())
            .mockRejectedValueOnce(new Error("offline"))
            .mockResolvedValueOnce(response({ session_id: "s2" }))
            .mockRejectedValueOnce(new Error("offline again"))
            .mockResolvedValueOnce(response({ session_id: "s2" }))
            .mockResolvedValueOnce(response({ session_id: "s1" }));
        stop = initHealthMonitor();
        await flush(30_000);
        expect(showBackendRestarted).toHaveBeenCalledOnce();
        await flush(45_000);
        expect(showBackendRestarted).toHaveBeenCalledOnce();
        expect(showBackendOffline).toHaveBeenCalledOnce();
        expect(hideBackendStatus).not.toHaveBeenCalled();
    });

    it.each([null, {}, { session_id: "" }, { session_id: "  " }, { session_id: 17 }])(
        "treats an invalid identity as offline: %j", async (data) => {
            vi.mocked(fetch).mockResolvedValueOnce(response(data));
            stop = initHealthMonitor();
            await flush();
            expect(showBackendOffline).toHaveBeenCalledOnce();
            await flush(15_000);
            expect(hideBackendStatus).toHaveBeenCalledOnce();
            expect(showBackendRestarted).not.toHaveBeenCalled();
        },
    );

    it.each([
        { ok: false, status: 503, json: async () => ({ session_id: "wrong" }) },
        { ok: true, json: async () => { throw new SyntaxError("invalid JSON"); } },
    ])("treats HTTP and JSON failures as offline", async (badResponse) => {
        vi.mocked(fetch).mockResolvedValueOnce(badResponse as Response);
        stop = initHealthMonitor();
        await flush();
        expect(showBackendOffline).toHaveBeenCalledOnce();
    });

    it("coalesces rapid Retry clicks and schedules the next poll from completion", async () => {
        const pending = deferred<Response>();
        vi.mocked(fetch).mockRejectedValueOnce(new Error("offline")).mockReturnValueOnce(pending.promise);
        stop = initHealthMonitor();
        await flush();
        await flush(14_000);
        retry();
        retry();
        await flush(1_000); // The old 15s poll must not race the pending retry.
        retry();
        expect(fetch).toHaveBeenCalledTimes(2);
        pending.resolve(response());
        await flush();
        await flush(14_999);
        expect(fetch).toHaveBeenCalledTimes(2);
        await flush(1);
        expect(fetch).toHaveBeenCalledTimes(3);
    });

    it.each(["fetch", "body"])("times out a hung %s and ignores its late result after recovery", async (part) => {
        const pending = deferred<any>();
        vi.mocked(fetch).mockImplementationOnce(() => part === "fetch"
            ? pending.promise
            : Promise.resolve({ ok: true, json: () => pending.promise } as Response));
        stop = initHealthMonitor();
        const signal = vi.mocked(fetch).mock.calls[0][1]!.signal!;
        await flush(4_999);
        expect(showBackendOffline).not.toHaveBeenCalled();
        await flush(1);
        expect(signal.aborted).toBe(true);
        expect(showBackendOffline).toHaveBeenCalledOnce();
        retry();
        await flush();
        expect(hideBackendStatus).toHaveBeenCalledOnce();
        const stale = { session_id: "stale" };
        pending.resolve(part === "fetch" ? response(stale) : stale);
        await flush();
        await flush(15_000);
        expect(showBackendRestarted).not.toHaveBeenCalled();
        expect(showBackendOffline).toHaveBeenCalledOnce();
    });

    it("keeps polling after a timeout even without Retry", async () => {
        vi.mocked(fetch).mockReturnValueOnce(new Promise(() => {}));
        stop = initHealthMonitor();
        await flush(20_000);
        expect(fetch).toHaveBeenCalledTimes(2);
        expect(hideBackendStatus).toHaveBeenCalledOnce();
    });

    it.each(["startup", "poll"])("aborts a pending %s on stop and ignores late completion", async (phase) => {
        const pending = deferred<Response>();
        if (phase === "poll") vi.mocked(fetch).mockResolvedValueOnce(response());
        vi.mocked(fetch).mockReturnValueOnce(pending.promise);
        stop = initHealthMonitor();
        await flush(phase === "poll" ? 15_000 : 0);
        const calls = vi.mocked(fetch).mock.calls.length;
        const signal = vi.mocked(fetch).mock.calls.at(-1)![1]!.signal!;
        stop();
        stop();
        expect(signal.aborted).toBe(true);
        pending.resolve(response({ session_id: "late" }));
        await flush(60_000);
        expect(fetch).toHaveBeenCalledTimes(calls);
        expect(showBackendOffline).not.toHaveBeenCalled();
        expect(showBackendRestarted).not.toHaveBeenCalled();
        expect(disposeBackendStatusOverlay).toHaveBeenCalledOnce();
        expect(vi.getTimerCount()).toBe(0);
    });

    it("installs once and starts cleanly after disposal", async () => {
        stop = initHealthMonitor();
        expect(initHealthMonitor()).toBe(stop);
        await flush();
        expect(fetch).toHaveBeenCalledOnce();
        const oldStop = stop;
        stop();
        vi.mocked(fetch).mockResolvedValue(response({ session_id: "new page" }));
        stop = initHealthMonitor();
        oldStop(); // Old cleanup must not dispose a new monitor.
        await flush(15_000);
        expect(vi.mocked(fetch).mock.calls.map(([url]) => url)).toEqual([
            "/api/session", "/api/session", "/api/health",
        ]);
        expect(showBackendRestarted).not.toHaveBeenCalled();
    });

    it("aborts on pagehide and preserves the original session across a history restore", async () => {
        const pending = deferred<Response>();
        vi.mocked(fetch).mockResolvedValueOnce(response()).mockReturnValueOnce(pending.promise);
        stop = initHealthMonitor();
        await flush(15_000);
        const signal = vi.mocked(fetch).mock.calls[1][1]!.signal!;
        window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }));
        expect(signal.aborted).toBe(true);
        await flush(60_000);
        expect(fetch).toHaveBeenCalledTimes(2);
        vi.mocked(fetch).mockResolvedValue(response({ session_id: "s2" }));
        window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true }));
        window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true }));
        await flush();
        expect(fetch).toHaveBeenCalledTimes(3);
        expect(vi.mocked(fetch).mock.calls[2][0]).toBe("/api/health");
        expect(showBackendRestarted).toHaveBeenCalledOnce();
        pending.resolve(response());
        await flush();
        expect(hideBackendStatus).not.toHaveBeenCalled();
    });

    it("removes page lifecycle listeners on disposal", async () => {
        stop = initHealthMonitor();
        await flush();
        stop();
        window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }));
        window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true }));
        await flush(60_000);
        expect(fetch).toHaveBeenCalledOnce();
        expect(vi.getTimerCount()).toBe(0);
    });
});
