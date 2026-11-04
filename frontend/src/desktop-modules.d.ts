declare module "*static/desktop.js" {
    export function toggleFullscreen(): void;
    export function installDesktopIntegration(): () => void;
}

declare module "*static/updates.js" {
    export function installUpdates(): () => void;
}

interface Window {
    foxReaderNativeFullscreen?: () => void;
}
