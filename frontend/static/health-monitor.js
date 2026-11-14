/* Shared by the bundled reader and the unbundled setup/settings pages. */
const POLL_INTERVAL_MS = 15_000;
const REQUEST_TIMEOUT_MS = 5_000;

/**
 * Keep the first known session for the lifetime of this page. A restart makes
 * its state stale even if that server subsequently goes offline or changes ID
 * again; only an explicit page refresh can make it current.
 */
export function createHealthMonitor({ online, offline, restarted }) {
    let sessionId = null;
    let status = "online";
    let initial = true;
    let stopped = false;
    let paused = false;
    let timer = null;
    let pending = null;
    let announceAfterResume = false;

    function transition(next) {
        if (status === "restarted") next = status;
        if (status === next && !announceAfterResume) return;
        announceAfterResume = false;
        status = next;
        if (next === "offline") offline(retry);
        else if (next === "restarted") restarted();
        else online();
    }

    async function check() {
        if (stopped || paused || pending) return;
        clearTimeout(timer);
        timer = null;
        const request = { controller: new AbortController(), cancel: null };
        pending = request;
        const url = initial ? "/api/session" : "/api/health";
        initial = false;
        let deadline;
        // Racing also bounds a stalled response body or a transport that does
        // not settle its promise on abort. Late results never change page state.
        const timeout = new Promise((_, reject) => {
            request.cancel = () => {
                clearTimeout(deadline);
                request.controller.abort();
                reject(new Error("Health request cancelled or timed out"));
            };
            deadline = setTimeout(request.cancel, REQUEST_TIMEOUT_MS);
        });
        try {
            const identity = await Promise.race([
                (async () => {
                    const response = await fetch(url, {
                        cache: "no-store",
                        credentials: "same-origin",
                        signal: request.controller.signal,
                    });
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const data = await response.json();
                    if (typeof data?.session_id !== "string" || !data.session_id.trim()) {
                        throw new Error("Missing session identity");
                    }
                    return data.session_id;
                })(),
                timeout,
            ]);
            if (pending !== request || stopped || paused) return;
            if (sessionId === null) sessionId = identity;
            transition(identity === sessionId ? "online" : "restarted");
        } catch {
            if (pending === request && !stopped && !paused) transition("offline");
        } finally {
            clearTimeout(deadline);
            if (pending === request) {
                pending = null;
                if (!stopped && !paused) timer = setTimeout(check, POLL_INTERVAL_MS);
            }
        }
    }

    function retry() {
        void check();
    }

    function pause() {
        paused = true;
        clearTimeout(timer);
        timer = null;
        const request = pending;
        pending = null;
        request?.cancel();
    }

    retry();
    return {
        retry,
        pause,
        resume() {
            if (stopped || !paused) return;
            paused = false;
            // The updater hides warnings while it closes the backend. Publish
            // the next result even if a failed handoff left health unchanged.
            announceAfterResume = true;
            retry();
        },
        stop() {
            stopped = true;
            pause();
        },
    };
}

let installedCleanup = null;

/** Install once per page without participating in its websocket/session lock. */
export function installHealthMonitor(callbacks, anchor, onDispose = () => {}) {
    if (installedCleanup) return installedCleanup;
    const monitor = createHealthMonitor(callbacks);
    let updatePaused = false;
    const pagehide = () => monitor.pause();
    const pageshow = () => { if (!updatePaused) monitor.resume(); };
    const updateState = (event) => {
        const state = event.detail?.state;
        if (state === "handoff" || state === "installing") {
            updatePaused = true;
            monitor.pause();
            callbacks.online();
        } else if (updatePaused) {
            updatePaused = false;
            monitor.resume();
        }
    };
    window.addEventListener("pagehide", pagehide);
    window.addEventListener("pageshow", pageshow);
    window.addEventListener("fox-reader:update-state", updateState);
    updateState({ detail: { state: document.getElementById("updatesPanel")?.dataset.updateState } });
    // Existing duplicate-tab handling replaces the body. Do not cover that
    // screen with a newly created health dialog or keep polling behind it.
    const observer = new MutationObserver(() => {
        if (anchor && !anchor.isConnected) cleanup();
    });
    if (anchor) observer.observe(document.body, { childList: true, subtree: true });
    let disposed = false;
    function cleanup() {
        if (disposed) return;
        disposed = true;
        monitor.stop();
        observer.disconnect();
        window.removeEventListener("pagehide", pagehide);
        window.removeEventListener("pageshow", pageshow);
        window.removeEventListener("fox-reader:update-state", updateState);
        installedCleanup = null;
        onDispose();
    }
    installedCleanup = cleanup;
    return cleanup;
}
