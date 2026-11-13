declare module "*static/health-monitor.js" {
    export interface HealthCallbacks {
        online(): void;
        offline(retry: () => void): void;
        restarted(): void;
    }

    export interface HealthMonitor {
        retry(): void;
        pause(): void;
        resume(): void;
        stop(): void;
    }

    export function createHealthMonitor(callbacks: HealthCallbacks): HealthMonitor;
    export function installHealthMonitor(
        callbacks: HealthCallbacks,
        anchor: HTMLElement | null,
        onDispose?: () => void,
    ): () => void;
}

declare module "*static/health-overlay.js" {
    export function initBackendStatusOverlay(): void;
    export function showBackendOffline(retry?: () => void): void;
    export function showBackendRestarted(): void;
    export function hideBackendStatus(): void;
    export function isBackendStatusVisible(): boolean;
    export function disposeBackendStatusOverlay(): void;
    export function installPageHealth(): (() => void) | undefined;
}
