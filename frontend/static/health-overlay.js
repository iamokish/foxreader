import { installHealthMonitor } from "./health-monitor.js";

let binding = null;

export function initBackendStatusOverlay() {
    const dialog = document.getElementById("backendStatusOverlay");
    if (binding?.dialog === dialog) return;
    disposeBackendStatusOverlay();
    if (!dialog) return;
    const title = dialog.querySelector("#backendStatusTitle");
    const message = dialog.querySelector("#backendStatusMessage");
    const action = dialog.querySelector("#backendStatusActionBtn");
    let mode = null;
    let retry = null;

    function present() {
        if (dialog.isConnected && !dialog.open) dialog.showModal();
    }
    function click() {
        if (mode === "restarted") window.location.reload();
        else if (mode === "offline") retry?.();
    }
    const cancel = (event) => event.preventDefault();
    // Native modality makes background controls inert, but keyboard events
    // still reach document-level listeners (including capture handlers). Keep
    // setup's licence decisions and the reader's other dialogs out of these
    // keys while preserving native focus traversal and desktop accelerators.
    const guardKeys = (event) => {
        if (dialog.open && dialog.isConnected && (event.key === "Escape" || event.key === "Tab")) {
            event.stopPropagation();
        }
    };
    action.addEventListener("click", click);
    dialog.addEventListener("cancel", cancel);
    window.addEventListener("keydown", guardKeys, true);
    // An in-flight endpoint operation can open its result dialog after us.
    // Re-enter the top layer then, without closing or rebuilding that editor.
    const observer = new MutationObserver((records) => {
        if (!dialog.isConnected) {
            disposeBackendStatusOverlay();
            return;
        }
        if (mode && records.some(({ target }) =>
            target !== dialog && target instanceof HTMLDialogElement && target.open)) {
            if (dialog.open) dialog.close();
            present();
        }
    });

    binding = {
        dialog,
        show(next, handler) {
            mode = next;
            retry = handler;
            const offline = mode === "offline";
            title.textContent = offline ? "Backend offline" : "Backend restarted";
            message.textContent = offline
                ? "Could not reach the server. Waiting for it to come back..."
                : "The server restarted. Please refresh to reload the app. Unsaved changes will be lost.";
            action.textContent = offline ? "Retry" : "Refresh";
            observer.observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["open"] });
            present();
        },
        hide() {
            mode = null;
            retry = null;
            observer.disconnect();
            if (dialog.open) dialog.close();
        },
        dispose() {
            this.hide();
            action.removeEventListener("click", click);
            dialog.removeEventListener("cancel", cancel);
            window.removeEventListener("keydown", guardKeys, true);
        },
    };
}

export function showBackendOffline(retry) {
    initBackendStatusOverlay();
    binding?.show("offline", retry);
}

export function showBackendRestarted() {
    initBackendStatusOverlay();
    binding?.show("restarted");
}

export function hideBackendStatus() {
    binding?.hide();
}

export function isBackendStatusVisible() {
    return Boolean(binding?.dialog.isConnected && binding.dialog.open);
}

export function disposeBackendStatusOverlay() {
    binding?.dispose();
    binding = null;
}

/** The templates already load desktop-entry.js; no template scripts are replaced. */
export function installPageHealth() {
    if (!/^\/(setup|settings|user_endpoints)\/?$/.test(window.location.pathname)) return;
    if (document.querySelector(".session-lock-screen, main.lock")) return;
    if (!document.getElementById("foxHealthStyles")) {
        const style = document.createElement("link");
        style.id = "foxHealthStyles";
        style.rel = "stylesheet";
        style.href = "/static/health.css";
        document.head.append(style);
    }
    let dialog = document.getElementById("backendStatusOverlay");
    if (!dialog) {
        dialog = document.createElement("dialog");
        dialog.id = "backendStatusOverlay";
        dialog.className = "fox-backend-status";
        dialog.setAttribute("aria-labelledby", "backendStatusTitle");
        dialog.setAttribute("aria-describedby", "backendStatusMessage");
        dialog.innerHTML = `
            <h2 id="backendStatusTitle"></h2>
            <p id="backendStatusMessage" data-selectable></p>
            <button id="backendStatusActionBtn" type="button" autofocus>Retry</button>`;
        document.body.append(dialog);
    }
    initBackendStatusOverlay();
    return installHealthMonitor({
        online: hideBackendStatus,
        offline: showBackendOffline,
        restarted: showBackendRestarted,
    }, dialog, () => {
        disposeBackendStatusOverlay();
        dialog.remove();
    });
}
