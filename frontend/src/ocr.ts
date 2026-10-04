import { state } from "./state";
import * as api from "./api";
import { updateCursor, suspendVariantSwitch } from "./viewer";
import { showNotify, showLoading, hideLoading, setTextAreaLoading } from "./ui";
import { handleAutoTranslation } from "./translate";
import type { Point, PolygonRegion } from "./types";

/**
 * Every toolbar action that takes over the canvas.
 *
 * While one of them is running the others are disabled: two captures sharing
 * `window.onmouseup` and the draw canvas would each undo the other's cleanup,
 * and the second one to finish would leave the canvas swallowing clicks.
 */
const CAPTURE_BUTTON_IDS = ["rectCapture", "freeCapture", "bubCapture", "PageCapture"] as const;

let captureLock = false;

const captureButton = (id: string): HTMLButtonElement | null => document.getElementById(id) as HTMLButtonElement | null;

/**
 * Take the capture lock, or refuse.
 *
 * `activeId` is left enabled so it can keep its `active-capture` glow -- the
 * disabled style greys buttons out, which would hide it. Re-clicking it is
 * harmless: every entry point starts here and bails on a taken lock.
 */
export function beginCapture(activeId?: string): boolean {
    if (captureLock) {
        showNotify("⚠️ Finish the capture in progress first.");
        return false;
    }
    captureLock = true;
    // The switch floats over the top-right of the page, which is somewhere a
    // capture stroke may well need to start.
    suspendVariantSwitch(true);
    for (const id of CAPTURE_BUTTON_IDS) {
        const btn = captureButton(id);
        if (!btn) continue;
        if (id === activeId) btn.classList.add("active-capture");
        else btn.disabled = true;
    }
    return true;
}

export function endCapture(): void {
    captureLock = false;
    suspendVariantSwitch(false);
    for (const id of CAPTURE_BUTTON_IDS) {
        const btn = captureButton(id);
        if (!btn) continue;
        btn.disabled = false;
        btn.classList.remove("active-capture");
    }
}

export const isCapturing = (): boolean => captureLock;

/**
 * Make region shapes ignore the pointer while a draw capture is in flight.
 *
 * The overlay root itself is already `pointer-events:none`, but every region
 * group opts back in as its own hit target -- so without this, starting a
 * drag on top of a bubble selects its entry, and the click that ends the
 * drag then clobbers the editor with the entry's text. Hiding is not an
 * option: the reader draws *around* the bubbles they can still see. Called
 * on every entry path of the three draw flows below, and lifted on every
 * exit path (including aborts), so a failed start can never leave the page
 * deaf to its own regions.
 */
function suspendRegionShapes(): void {
    document.body.classList.add("fox-capturing");
}

function resumeRegionShapes(): void {
    document.body.classList.remove("fox-capturing");
}

export async function startCapture(): Promise<void> {
    if (!beginCapture("rectCapture")) return;
    state.currentlySelectedEntryId = null;
    state.isCaptureMode = true;
    updateCursor();
    suspendRegionShapes();

    const container = document.getElementById("imageContainer");
    const selector = document.getElementById("cropSelector");
    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    const lang = (document.getElementById("languageSelect") as HTMLSelectElement)?.value ?? "";
    let startX: number, startY: number;

    if (!container || !selector || !mainImg) {
        state.isCaptureMode = false;
        updateCursor();
        resumeRegionShapes();
        endCapture();
        return;
    }

    container.onmousedown = (e: MouseEvent) => {
        const rect = container.getBoundingClientRect();
        startX = e.clientX - rect.left + container.scrollLeft;
        startY = e.clientY - rect.top + container.scrollTop;
        selector.style.display = "block";
        selector.style.left = startX + "px";
        selector.style.top = startY + "px";
        selector.style.width = "0px";
        selector.style.height = "0px";

        container.onmousemove = (moveE: MouseEvent) => {
            const mRect = container.getBoundingClientRect();
            const curX = moveE.clientX - mRect.left + container.scrollLeft;
            const curY = moveE.clientY - mRect.top + container.scrollTop;
            selector.style.width = Math.abs(curX - startX) + "px";
            selector.style.height = Math.abs(curY - startY) + "px";
            selector.style.left = Math.min(curX, startX) + "px";
            selector.style.top = Math.min(curY, startY) + "px";
        };
    };

    window.onmouseup = async () => {
        endCapture();
        container.onmousedown = null;
        container.onmousemove = null;
        window.onmouseup = null;
        // A click that never became a drag still ends the capture, so the cursor
        // and the mode flag have to be reset on both paths.
        state.isCaptureMode = false;
        updateCursor();
        resumeRegionShapes();

        if (selector.style.display === "block") {
            const rect = selector.getBoundingClientRect();
            const imgRect = mainImg.getBoundingClientRect();
            const cropData = {
                filename: state.currentImageFile,
                x: Math.round((rect.left - imgRect.left) / state.zoomLevel),
                y: Math.round((rect.top - imgRect.top) / state.zoomLevel),
                width: Math.round(rect.width / state.zoomLevel),
                height: Math.round(rect.height / state.zoomLevel),
                lang,
                isGrayScale: state.isGrayScaleEnabled,
            };
            state.lastCapturedRegion = {
                type: "rectangle",
                coords: { x: cropData.x, y: cropData.y, w: cropData.width, h: cropData.height },
            };
            selector.style.display = "none";

            if (cropData.width < 5 || cropData.height < 5) return;

            setTextAreaLoading(true);
            try {
                const data = await api.ocrCrop(cropData);
                const textArea = document.getElementById("textArea") as HTMLTextAreaElement | null;
                if (textArea) textArea.value = data.text;
                handleAutoTranslation();
            } catch {
                showNotify("❌ OCR Failed");
            } finally {
                setTextAreaLoading(false);
            }
        }
    };
}

/**
 * Draw a freehand cut and return it, or `null` if the user gave up.
 *
 * No toolbar button glows here: the split is initiated from an entry, so the
 * entry's own split button carries the active state (see `entries.ts`). Escape
 * cancels, which matters because a stroke that is never finished would otherwise
 * leave the draw canvas over the page and the toolbar disabled.
 */
export function splitCapture(): Promise<PolygonRegion | null> {
    return new Promise((resolve) => {
        if (!beginCapture()) {
            resolve(null);
            return;
        }
        state.isCaptureMode = true;
        updateCursor();
        suspendRegionShapes();

        const container = document.getElementById("imageContainer");
        const canvas = document.getElementById("drawCanvas") as HTMLCanvasElement | null;
        const ctx = canvas?.getContext("2d");
        const img = document.getElementById("mainImage") as HTMLImageElement | null;

        if (!container || !canvas || !ctx || !img) {
            state.isCaptureMode = false;
            updateCursor();
            resumeRegionShapes();
            endCapture();
            resolve(null);
            return;
        }

        img.draggable = false;
        img.style.userSelect = "none";
        img.style.pointerEvents = "none";
        container.style.overflow = "hidden";
        container.style.touchAction = "none";

        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        canvas.style.position = "absolute";
        canvas.style.top = container.scrollTop + "px";
        canvas.style.left = container.scrollLeft + "px";
        canvas.style.display = "block";
        canvas.style.pointerEvents = "auto";
        canvas.style.zIndex = "9000";
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        let settled = false;
        /** The one way out: every exit path restores the page and resolves once. */
        const finish = (result: PolygonRegion | null) => {
            if (settled) return;
            settled = true;
            state.isDrawing = false;
            state.isCaptureMode = false;
            canvas.onmousedown = null;
            canvas.onmousemove = null;
            window.onmouseup = null;
            window.removeEventListener("keydown", onKey, true);
            img.style.pointerEvents = "auto";
            container.style.overflow =
                state.isHorizontalFitEnabled || state.zoomLevel > state.fitZoomLevel ? "auto" : "hidden";
            container.style.touchAction = "auto";
            canvas.style.display = "none";
            canvas.style.pointerEvents = "none";
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            updateCursor();
            resumeRegionShapes();
            endCapture();
            resolve(result);
        };

        const onKey = (event: KeyboardEvent) => {
            if (event.key !== "Escape") return;
            event.preventDefault();
            finish(null);
        };
        window.addEventListener("keydown", onKey, true);

        canvas.onmousedown = (e: MouseEvent) => {
            e.stopPropagation();
            e.preventDefault();
            state.isDrawing = true;
            state.lassoPoints = [];
            ctx.beginPath();
            ctx.strokeStyle = "#ffcc00";
            ctx.lineWidth = 2;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            ctx.setLineDash([5, 5]);
        };

        canvas.onmousemove = (e: MouseEvent) => {
            if (!state.isDrawing) return;
            e.stopPropagation();
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            // Sub-pixel moves add points the backend has to rasterise and the
            // overlay has to redraw without changing the shape of the cut.
            const last = state.lassoPoints[state.lassoPoints.length - 1];
            if (last && Math.abs(last.x - x) < 1 && Math.abs(last.y - y) < 1) return;
            state.lassoPoints.push({ x, y });
            ctx.lineTo(x, y);
            ctx.stroke();
        };

        window.onmouseup = () => {
            if (!state.isDrawing) return;
            if (state.lassoPoints.length < 5) {
                finish(null);
                return;
            }
            const imgRect = img.getBoundingClientRect();
            const canvasRect = canvas.getBoundingClientRect();
            const realPoints: Point[] = state.lassoPoints.map((p) => ({
                x: Math.round((p.x + canvasRect.left - imgRect.left) / state.zoomLevel),
                y: Math.round((p.y + canvasRect.top - imgRect.top) / state.zoomLevel),
            }));
            state.lastCapturedRegion = { type: "polygon", coords: realPoints };
            finish({ type: "polygon", coords: realPoints });
        };
    });
}

export async function startFreeCapture(): Promise<void> {
    if (!beginCapture("freeCapture")) return;
    state.currentlySelectedEntryId = null;
    state.isCaptureMode = true;
    updateCursor();
    suspendRegionShapes();

    const container = document.getElementById("imageContainer");
    const canvas = document.getElementById("drawCanvas") as HTMLCanvasElement | null;
    const ctx = canvas?.getContext("2d");
    const img = document.getElementById("mainImage") as HTMLImageElement | null;
    const lang = (document.getElementById("languageSelect") as HTMLSelectElement)?.value ?? "";

    if (!container || !canvas || !ctx || !img) {
        state.isCaptureMode = false;
        updateCursor();
        resumeRegionShapes();
        endCapture();
        return;
    }

    img.draggable = false;
    img.style.userSelect = "none";
    img.style.pointerEvents = "none";
    container.style.overflow = "hidden";
    container.style.touchAction = "none";

    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    canvas.style.position = "absolute";
    canvas.style.top = container.scrollTop + "px";
    canvas.style.left = container.scrollLeft + "px";
    canvas.style.display = "block";
    canvas.style.pointerEvents = "auto";
    canvas.style.zIndex = "9000";
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    canvas.onmousedown = (e: MouseEvent) => {
        e.stopPropagation();
        e.preventDefault();
        state.isDrawing = true;
        state.lassoPoints = [];
        ctx.beginPath();
        ctx.strokeStyle = "#ffcc00";
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.setLineDash([5, 5]);
    };

    canvas.onmousemove = (e: MouseEvent) => {
        if (!state.isDrawing) return;
        e.stopPropagation();
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        state.lassoPoints.push({ x, y });
        ctx.lineTo(x, y);
        ctx.stroke();
    };

    window.onmouseup = async () => {
        if (state.isCaptureMode && state.isDrawing) {
            state.isDrawing = false;
            window.onmouseup = null;
            endCapture();
            img.style.pointerEvents = "auto";

            if (state.isHorizontalFitEnabled || state.zoomLevel > state.fitZoomLevel) {
                container.style.overflow = "auto";
            } else {
                container.style.overflow = "hidden";
            }
            container.style.touchAction = "auto";
            canvas.style.display = "none";
            canvas.style.pointerEvents = "none";

            if (state.lassoPoints.length < 5) {
                state.isCaptureMode = false;
                updateCursor();
                resumeRegionShapes();
                return;
            }

            const imgRect = img.getBoundingClientRect();
            const realPoints: Point[] = state.lassoPoints.map((p) => {
                const canvasRect = canvas.getBoundingClientRect();
                const screenX = p.x + canvasRect.left;
                const screenY = p.y + canvasRect.top;
                return {
                    x: Math.round((screenX - imgRect.left) / state.zoomLevel),
                    y: Math.round((screenY - imgRect.top) / state.zoomLevel),
                };
            });

            state.lastCapturedRegion = { type: "polygon", coords: realPoints };
            state.isCaptureMode = false;
            updateCursor();
            resumeRegionShapes();

            setTextAreaLoading(true);
            try {
                const data = await api.ocrFreeform({
                    filename: state.currentImageFile,
                    points: realPoints,
                    lang,
                    isGrayScale: state.isGrayScaleEnabled,
                });
                const textArea = document.getElementById("textArea") as HTMLTextAreaElement | null;
                if (textArea) textArea.value = data.text;
                handleAutoTranslation();
            } catch {
                showNotify("❌ Freeform OCR Failed");
            } finally {
                setTextAreaLoading(false);
            }
        }
    };
}

export async function bubbleCapture(): Promise<void> {
    if (!state.currentImageFile) {
        showNotify("⚠️ No image loaded!");
        return;
    }
    if (!beginCapture("bubCapture")) return;
    try {
        showLoading("Scanning...");
        const data = await api.bubbleDetect({
            filename: state.currentImageFile,
            isGrayScale: state.isGrayScaleEnabled,
        });
        if (Array.isArray(data) && data.length > 0) {
            const { addNewEntries, renderCurrentPageEntries } = await import("./entries");
            addNewEntries(data);
            renderCurrentPageEntries();
        } else {
            showNotify("⚠️ No bubbles detected");
        }
    } catch {
        showNotify("❌ Bubble detection failed");
    } finally {
        endCapture();
        hideLoading();
    }
}
