export function initWebSocket(): WebSocket {
    const socket = new WebSocket(`ws://${window.location.host}/ws/lifecycle`);
    let flashInterval: ReturnType<typeof setInterval> | null = null;

    socket.onmessage = (event: MessageEvent) => {
        const data = JSON.parse(event.data) as { action: string };
        if (data.action === "focus_tab") {
            if (flashInterval) clearInterval(flashInterval);
            const originalTitle = document.title;
            let isFlashed = false;
            flashInterval = setInterval(() => {
                document.title = isFlashed ? "Fox Reader (Active)" : originalTitle;
                isFlashed = !isFlashed;
            }, 500);
            window.addEventListener(
                "focus",
                () => {
                    if (flashInterval) clearInterval(flashInterval);
                    document.title = originalTitle;
                    flashInterval = null;
                },
                { once: true },
            );
        }
    };

    return socket;
}

export function initSessionLock(socket: WebSocket): void {
    const channel = new BroadcastChannel("fox_reader_session_lock");
    const tabId = Math.random().toString(36).substring(2, 15);
    let isPrimary = true;
    let dominanceInterval: ReturnType<typeof setInterval> | null = null;

    channel.onmessage = (event: MessageEvent) => {
        let data: { action: string; ownerId?: string };
        try {
            data = JSON.parse(event.data);
        } catch {
            return;
        }

        if (data.action === "ping_session_owner" && isPrimary) {
            channel.postMessage(JSON.stringify({ action: "session_owner_ack", ownerId: tabId }));
        }

        if (data.action === "session_owner_ack" && data.ownerId !== tabId) {
            if (isPrimary) {
                isPrimary = false;
                triggerDuplicateBlockMode(socket, channel, dominanceInterval);
            }
        }
    };

    channel.postMessage(JSON.stringify({ action: "ping_session_owner" }));

    dominanceInterval = setInterval(() => {
        if (isPrimary) {
            channel.postMessage(JSON.stringify({ action: "session_owner_ack", ownerId: tabId }));
        } else {
            clearInterval(dominanceInterval!);
        }
    }, 1000);
}

function triggerDuplicateBlockMode(
    socket: WebSocket,
    channel: BroadcastChannel,
    interval: ReturnType<typeof setInterval> | null,
): void {
    window.stop();
    if (socket) socket.close();
    if (interval) clearInterval(interval);

    document.body.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;background-color:#2b2827;color:#fff;font-family:sans-serif;text-align:center;padding:20px;box-sizing:border-box;">
            <h1 style="color:#e74c3c;margin-bottom:10px;font-size:2rem;">Access Denied</h1>
            <p style="font-size:16px;max-width:500px;color:#aaa;line-height:1.5;margin-bottom:20px;">
                Fox Reader is already running in another browser window or tab.
                Please return to your active window.
            </p>
            <p style="font-size:12px;color:#777;">You can safely close this tab.</p>
        </div>
    `;
}
