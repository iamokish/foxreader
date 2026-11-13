import { installHealthMonitor } from "../../static/health-monitor.js";
import {
    showBackendOffline, showBackendRestarted, hideBackendStatus,
    disposeBackendStatusOverlay,
} from "../components/common/BackendStatusOverlay";

export function initHealthMonitor(): () => void {
    return installHealthMonitor({
        offline: showBackendOffline,
        restarted: showBackendRestarted,
        online: hideBackendStatus,
    }, document.getElementById("backendStatusOverlay"), disposeBackendStatusOverlay);
}
