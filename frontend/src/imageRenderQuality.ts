import { getInitialFoxConfig } from "./core/initialConfig";
import { isNativeViewer } from "../static/desktop.js";

export { isNativeViewer };

/** Scope belongs to this viewer, not to the OS or the backend's launch mode. */
export function imageRenderQualityEnabled(): boolean {
    const mode = getInitialFoxConfig().imageRenderQuality ?? 1;
    return mode === 2 || (mode === 1 && isNativeViewer());
}
