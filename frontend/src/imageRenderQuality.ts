import { getInitialFoxConfig } from "./core/initialConfig";

/** All three native hosts inject this capability before page scripts execute. */
export function isNativeViewer(): boolean {
    return typeof window !== "undefined" && typeof window.foxReaderNativeFullscreen === "function";
}

/** Scope belongs to this viewer, not to the OS or the backend's launch mode. */
export function imageRenderQualityEnabled(): boolean {
    const mode = getInitialFoxConfig().imageRenderQuality ?? 1;
    return mode === 2 || (mode === 1 && isNativeViewer());
}
