import { getHealth, getSession } from "./api";
import { showBackendOffline, showBackendRestarted, hideBackendStatus } from "../components/common";

const POLL_INTERVAL_MS = 15_000;

type BackendStatus = "online" | "offline" | "restarted";

let storedSessionId: string | null = null;
let status: BackendStatus = "online";
let pollTimer: ReturnType<typeof setInterval> | null = null;

async function checkHealth(): Promise<void> {
    let sessionId: string | null = null;
    try {
        sessionId = (await getHealth()).session_id;
    } catch {
        sessionId = null;
    }

    if (sessionId === null) {
        if (status !== "offline") {
            status = "offline";
            showBackendOffline(() => {
                void checkHealth();
            });
        }
        return;
    }

    if (storedSessionId === null) {
        storedSessionId = sessionId;
    }

    if (sessionId === storedSessionId) {
        if (status === "offline") {
            status = "online";
            hideBackendStatus();
        }
    } else {
        status = "restarted";
        storedSessionId = sessionId;
        showBackendRestarted();
    }
}

export function initHealthMonitor(): () => void {
    void (async () => {
        try {
            storedSessionId = (await getSession()).session_id;
            status = "online";
        } catch {
            status = "offline";
            showBackendOffline(() => {
                void checkHealth();
            });
        }
        if (pollTimer === null) {
            pollTimer = setInterval(() => void checkHealth(), POLL_INTERVAL_MS);
        }
    })();

    return () => {
        if (pollTimer !== null) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    };
}
