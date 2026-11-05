import { imageRenderQualityEnabled } from "./imageRenderQuality";

const SETTLE_MS = 150;

interface TransformState {
    timer: number;
    previousWillChange: string;
    resting?: string;
    serializedResting?: string;
}

// Settled elements are weakly held. The only strong reference from a timer lasts
// until the gesture settles; owners can release it immediately on replacement.
const transforms = new WeakMap<HTMLElement, TransformState>();

/** Cancel pending rasterization and restore the element's original layer hint. */
export function disposeImageTransform(element: HTMLElement): void {
    const state = transforms.get(element);
    if (!state) return;
    window.clearTimeout(state.timer);
    element.style.willChange = state.previousWillChange;
    transforms.delete(element);
}

/**
 * Keep gestures on the compositor, then let the browser rasterize a stationary
 * image without a forced GPU layer. Both forms have identical image coordinates;
 * natural dimensions and the caller's zoom/pan arithmetic remain untouched.
 *
 * The caller still synchronizes overlays after this immediate write. onSettled
 * mirrors the later 2D write without another layout read or animation frame.
 */
export function applyImageTransform(
    element: HTMLElement,
    x: number,
    y: number,
    scale: number,
    onSettled?: () => void,
): void {
    const moving = `translate3d(${x}px, ${y}px, 0) scale(${scale})`;
    if (!imageRenderQualityEnabled()) {
        disposeImageTransform(element);
        element.style.transform = moving;
        return;
    }

    const resting = `translate(${x}px, ${y}px) scale(${scale})`;
    let state = transforms.get(element);
    // Publishing an unchanged zoom label or image size must not promote an
    // already settled image and rasterize it again.
    if (state && !state.timer && state.resting === resting && element.style.transform === state.serializedResting)
        return;
    if (!state) {
        state = { timer: 0, previousWillChange: element.style.willChange };
        transforms.set(element, state);
    }
    window.clearTimeout(state.timer);
    element.style.willChange = "transform";
    element.style.transform = moving;
    // CSSOM may normalize a unitless zero to 0px. Compare what the engine wrote,
    // not the source string, when detecting a later independent transform.
    const serializedMoving = element.style.transform;

    const pending = state;
    const timer = window.setTimeout(() => {
        if (transforms.get(element) !== pending || pending.timer !== timer) return;
        pending.timer = 0;
        // A removed/replaced view or a later independent transform must not be
        // rewritten by a stale gesture. Mode changes also restore baseline CSS.
        if (!element.isConnected || !imageRenderQualityEnabled() || element.style.transform !== serializedMoving) {
            disposeImageTransform(element);
            return;
        }
        element.style.transform = resting;
        element.style.willChange = "auto";
        pending.resting = resting;
        pending.serializedResting = element.style.transform;
        onSettled?.();
    }, SETTLE_MS);
    state.timer = timer;
}
