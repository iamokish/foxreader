/**
 * Reads the initial theme/layout preferences injected by the backend into
 * `window.__FOX_CONFIG__` (see frontend/templates/index.html and
 * src/fox_reader/routes/folder.py). Used as a fallback source when no
 * explicit user preference exists yet in localStorage.
 */

export interface InitialFoxConfig {
    theme?: string;
    layout?: string;
    deeplAvailable?: boolean;
    jpdbAvailable?: boolean;
    /**
     * Whether the bubble-segmentation model is loaded. Optional, and read as
     * "available" when absent: a cached page served before the flag existed
     * must keep its Bubble Capture button rather than lose it silently.
     */
    bubbleAvailable?: boolean;
}

declare global {
    interface Window {
        __FOX_CONFIG__?: InitialFoxConfig;
    }
}

export function getInitialFoxConfig(): InitialFoxConfig {
    try {
        return window.__FOX_CONFIG__ ?? {};
    } catch {
        return {};
    }
}
