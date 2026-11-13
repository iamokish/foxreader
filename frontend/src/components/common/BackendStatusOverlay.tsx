import { useLayoutEffect } from "preact/hooks";
import {
    initBackendStatusOverlay, disposeBackendStatusOverlay,
} from "../../../static/health-overlay.js";
import "../../../static/health.css";

export {
    showBackendOffline, showBackendRestarted, hideBackendStatus,
    isBackendStatusVisible, initBackendStatusOverlay, disposeBackendStatusOverlay,
} from "../../../static/health-overlay.js";

export function BackendStatusOverlay() {
    useLayoutEffect(() => {
        initBackendStatusOverlay();
        return disposeBackendStatusOverlay;
    }, []);
    return (
        <dialog
            id="backendStatusOverlay"
            class="fox-backend-status"
            aria-labelledby="backendStatusTitle"
            aria-describedby="backendStatusMessage"
        >
            <h2 id="backendStatusTitle"></h2>
            <p id="backendStatusMessage" data-selectable></p>
            <button id="backendStatusActionBtn" type="button" autoFocus>Retry</button>
        </dialog>
    );
}
