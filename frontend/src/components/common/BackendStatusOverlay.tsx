let statusMode: "offline" | "restarted" | null = null;
let retryHandler: (() => void) | null = null;

export function showBackendOffline(retry?: () => void): void {
    statusMode = "offline";
    retryHandler = retry ?? null;
    const overlay = document.getElementById("backendStatusOverlay");
    const title = document.getElementById("backendStatusTitle");
    const message = document.getElementById("backendStatusMessage");
    const actionBtn = document.getElementById("backendStatusActionBtn");
    if (title) title.textContent = "Backend offline";
    if (message) message.textContent = "Could not reach the server. Waiting for it to come back...";
    if (actionBtn) {
        actionBtn.textContent = "Retry";
        actionBtn.style.display = "";
    }
    if (overlay) overlay.style.display = "flex";
}

export function showBackendRestarted(): void {
    statusMode = "restarted";
    const overlay = document.getElementById("backendStatusOverlay");
    const title = document.getElementById("backendStatusTitle");
    const message = document.getElementById("backendStatusMessage");
    const actionBtn = document.getElementById("backendStatusActionBtn");
    if (title) title.textContent = "Backend restarted";
    if (message) message.textContent = "The server restarted. Please refresh to reload the app.";
    if (actionBtn) {
        actionBtn.textContent = "Refresh";
        actionBtn.style.display = "";
    }
    if (overlay) overlay.style.display = "flex";
}

export function hideBackendStatus(): void {
    statusMode = null;
    const overlay = document.getElementById("backendStatusOverlay");
    if (overlay) overlay.style.display = "none";
}

export function isBackendStatusVisible(): boolean {
    const overlay = document.getElementById("backendStatusOverlay");
    return overlay ? overlay.style.display === "flex" : false;
}

export function initBackendStatusOverlay(): void {
    document.getElementById("backendStatusActionBtn")?.addEventListener("click", () => {
        if (statusMode === "restarted") {
            window.location.reload();
        } else if (statusMode === "offline") {
            retryHandler?.();
        }
    });
}

export function BackendStatusOverlay() {
    return (
        <div
            id="backendStatusOverlay"
            style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: var(--bg-overlay); z-index: 99999; align-items: center; justify-content: center; box-sizing: border-box;"
        >
            <div
                id="backendStatusContent"
                style="background: var(--bg-secondary); padding: 30px; border-radius: 8px; border: 1px solid var(--border-default); box-shadow: var(--shadow-lg); min-width: 320px; max-width: 480px; text-align: center; box-sizing: border-box;"
            >
                <h2
                    id="backendStatusTitle"
                    style="color: var(--text-primary); margin: 0 0 12px 0; font-size: 20px;"
                ></h2>
                <p
                    id="backendStatusMessage"
                    style="color: var(--text-secondary); font-size: 14px; margin: 0 0 20px 0; line-height: 1.5;"
                ></p>
                <button id="backendStatusActionBtn" class="btn-primary" style="padding: 8px 20px; cursor: pointer;">
                    Retry
                </button>
            </div>
        </div>
    );
}
