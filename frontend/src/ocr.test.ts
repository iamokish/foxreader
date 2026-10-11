/**
 * Draw captures must not talk to the regions underneath the cursor.
 *
 * The overlay root ignores the pointer, but every region group opts back in
 * as its own hit target -- so starting a drag on top of a bubble used to
 * select its entry, and the click ending the drag then overwrote the editor
 * with the entry's own text. While a draw capture runs, `ocr.ts` therefore
 * raises `fox-capturing` on the body, under which the stylesheet suspends
 * hit-testing for the whole overlay (leaving it visible to draw around).
 */

/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve as resolvePath } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { bubbleCapture, endCapture, normalizeBubblePadding, splitCapture, startCapture, startFreeCapture } from "./ocr";
import { state } from "./state";

vi.mock("./api", () => ({
    ocrCrop: vi.fn(async () => ({ text: "captured" })),
    ocrFreeform: vi.fn(async () => ({ text: "captured free" })),
    bubbleDetect: vi.fn(async () => []),
}));

import { ocrCrop, ocrFreeform, bubbleDetect } from "./api";

const stylesheet = readFileSync(resolvePath(process.cwd(), "static/style.css"), "utf8");

const CAPTURING = "fox-capturing";

function dom(): void {
    document.body.innerHTML = `
        <div id="imageContainer">
            <img id="mainImage" src="">
            <div id="cropSelector" style="display:none"></div>
            <canvas id="drawCanvas" style="display:none"></canvas>
        </div>
        <select id="languageSelect"><option value="japanese">Japanese</option></select>
        <textarea id="textArea"></textarea>
        <div id="translatedText" class="panel"></div>`;
}

/**
 * The canvas 2d prototype, without naming the `HTMLCanvasElement` global
 * (the lint config does not know DOM constructor globals in test files).
 */
function canvasPrototype(): { getContext: (...args: never[]) => unknown } {
    return Object.getPrototypeOf(document.createElement("canvas"));
}

function fakeCtx(): CanvasRenderingContext2D {
    return {
        beginPath: vi.fn(),
        clearRect: vi.fn(),
        lineTo: vi.fn(),
        stroke: vi.fn(),
        setLineDash: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
}

const flush = async (): Promise<void> => {
    await new Promise((resolve) => setTimeout(resolve, 0));
    await new Promise((resolve) => setTimeout(resolve, 0));
};

beforeEach(() => {
    document.body.innerHTML = "";
    document.body.classList.remove(CAPTURING);
    window.onmouseup = null;
    state.isCaptureMode = false;
    state.isDrawing = false;
    state.lassoPoints = [];
    state.zoomLevel = 1.0;
    state.currentImageFile = "p.png";
    state.currentlySelectedEntryId = null;
});

afterEach(() => {
    // Belt and braces: no test may leak the capture lock (or the flag) into
    // the next one. `window.onmouseup` is driven by direct invocation below
    // because this jsdom version does not deliver dispatched events to
    // `window.on*` property handlers -- element-level `on*` handlers, and the
    // real browser for both, are unaffected.
    endCapture();
    window.onmouseup = null;
    document.body.classList.remove(CAPTURING);
    vi.clearAllMocks();
});

/**
 * Run the installed `window.onmouseup` handler, whatever a step set up.
 *
 * Dispatched `mouseup` events do reach `addEventListener` handlers in this
 * jsdom version, but not `window.onmouseup = ...` assignments -- while every
 * real browser fires both. Invoking the property directly tests exactly the
 * cleanup the production code runs on mouse-up.
 */
async function fireWindowMouseUp(): Promise<void> {
    const handler = window.onmouseup as unknown as (() => unknown) | null;
    await handler?.();
}

describe("the capturing flag", () => {
    it("suspends region hit-testing in CSS while set", () => {
        expect(stylesheet).toMatch(
            /body\.fox-capturing\s+#regionSvgOverlay\s+\.entry-shape[\s\S]*?pointer-events:\s*none\s*!important/s,
        );
    });
});

describe("startCapture (rectangle)", () => {
    it("raises the flag on start and lowers it when the drag ends", async () => {
        dom();
        await startCapture();
        expect(document.body.classList.contains(CAPTURING)).toBe(true);

        const container = document.getElementById("imageContainer")!;
        container.dispatchEvent(new window.MouseEvent("mousedown", { clientX: 10, clientY: 20, bubbles: true }));
        expect((document.getElementById("cropSelector") as HTMLElement).style.display).toBe("block");

        await fireWindowMouseUp();
        await flush();

        expect(document.body.classList.contains(CAPTURING)).toBe(false);
        expect(state.isCaptureMode).toBe(false);
        // A zero-size box in jsdom never reaches OCR -- the point here is the
        // capture ended cleanly, not what it read.
        expect(ocrCrop).not.toHaveBeenCalled();
    });

    it("never leaves the flag up when there is nothing to draw on", async () => {
        document.body.innerHTML = "";
        await startCapture();
        expect(document.body.classList.contains(CAPTURING)).toBe(false);
        expect(state.isCaptureMode).toBe(false);
    });
});

describe("startFreeCapture (lasso)", () => {
    it("lowers the flag again after a finished stroke", async () => {
        dom();
        vi.spyOn(canvasPrototype(), "getContext").mockReturnValue(fakeCtx());
        try {
            await startFreeCapture();
            expect(document.body.classList.contains(CAPTURING)).toBe(true);

            const canvas = document.getElementById("drawCanvas")!;
            canvas.dispatchEvent(new window.MouseEvent("mousedown", { clientX: 5, clientY: 5, bubbles: true }));
            for (let i = 0; i < 6; i++) {
                canvas.dispatchEvent(
                    new window.MouseEvent("mousemove", { clientX: 10 + i * 3, clientY: 10 + i * 2, bubbles: true }),
                );
            }
            window.dispatchEvent(new window.MouseEvent("mouseup", { bubbles: true }));
            await fireWindowMouseUp();
            await flush();

            expect(ocrFreeform).toHaveBeenCalledTimes(1);
            expect(document.body.classList.contains(CAPTURING)).toBe(false);
            expect(state.isCaptureMode).toBe(false);
        } finally {
            vi.restoreAllMocks();
        }
    });

    it("maps canvas points through the live canvas origin", async () => {
        // A Vulpes-like stage: the canvas sits far from the viewport origin, so
        // dropping its offset (as a zero rect would) shifts every point up-left.
        dom();
        vi.spyOn(canvasPrototype(), "getContext").mockReturnValue(fakeCtx());
        try {
            await startFreeCapture();
            const canvas = document.getElementById("drawCanvas") as HTMLCanvasElement;
            const img = document.getElementById("mainImage") as HTMLImageElement;
            const liveRect = (left: number, top: number, width: number, height: number): DOMRect =>
                ({ left, top, width, height, right: left + width, bottom: top + height, x: left, y: top, toJSON: () => ({}) }) as DOMRect;
            // Like a real browser: hidden elements report a zero rect.
            vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() =>
                canvas.style.display === "none"
                    ? liveRect(0, 0, 0, 0)
                    : liveRect(165, 48, 800, 600),
            );
            vi.spyOn(img, "getBoundingClientRect").mockReturnValue(liveRect(432, 48, 400, 600));
            state.zoomLevel = 0.5;

            canvas.dispatchEvent(new window.MouseEvent("mousedown", { clientX: 492, clientY: 108, bubbles: true }));
            for (let i = 1; i <= 6; i++) {
                canvas.dispatchEvent(
                    new window.MouseEvent("mousemove", { clientX: 492 + i * 10, clientY: 108, bubbles: true }),
                );
            }
            await fireWindowMouseUp();
            await flush();

            expect(ocrFreeform).toHaveBeenCalledTimes(1);
            const sent = vi.mocked(ocrFreeform).mock.calls[0][0];
            // client 502 -> canvas-local 337 -> image (337 + 165 - 432) / 0.5.
            expect(sent.points[0]).toEqual({ x: 140, y: 120 });
        } finally {
            vi.restoreAllMocks();
        }
    });

    it("lowers the flag when the canvas cannot start", async () => {
        dom();
        // No canvas 2d context in jsdom: the flow aborts before drawing.
        await startFreeCapture();
        expect(document.body.classList.contains(CAPTURING)).toBe(false);
        expect(state.isCaptureMode).toBe(false);
    });
});

describe("normalizeBubblePadding", () => {
    it("keeps values inside 0-10", () => {
        expect(normalizeBubblePadding(0)).toBe(0);
        expect(normalizeBubblePadding(5)).toBe(5);
        expect(normalizeBubblePadding(10)).toBe(10);
    });

    it("clamps instead of refusing", () => {
        expect(normalizeBubblePadding(-3)).toBe(0);
        expect(normalizeBubblePadding(15)).toBe(10);
    });

    it("falls back on garbage, like the backend does", () => {
        expect(normalizeBubblePadding(null)).toBe(5);
        expect(normalizeBubblePadding(undefined)).toBe(5);
        expect(normalizeBubblePadding("")).toBe(5);
        expect(normalizeBubblePadding("abc")).toBe(5);
        expect(normalizeBubblePadding(Number.NaN)).toBe(5);
    });

    it("reads numeric strings and floors fractions", () => {
        expect(normalizeBubblePadding("7")).toBe(7);
        expect(normalizeBubblePadding(4.7)).toBe(4);
    });
});

describe("bubbleCapture", () => {
    it("sends the slider padding with the request", async () => {
        dom();
        state.currentImageFile = "p.png";
        state.bubblePadding = 8;
        await bubbleCapture();
        expect(bubbleDetect).toHaveBeenCalledTimes(1);
        expect(vi.mocked(bubbleDetect).mock.calls[0][0]).toEqual({
            filename: "p.png",
            isGrayScale: false,
            padding: 8,
        });
    });

    it("names the backend's reason instead of a dead-end toast", async () => {
        dom();
        state.currentImageFile = "p.png";
        vi.mocked(bubbleDetect).mockRejectedValueOnce(new Error("CUDA out of memory"));
        await bubbleCapture();
        const toasts = Array.from(document.querySelectorAll(".toast-notification"))
            .map((t) => t.textContent ?? "")
            .join(" | ");
        expect(toasts).toContain("Bubble detection failed");
        expect(toasts).toContain("CUDA out of memory");
    });
});

describe("splitCapture (entry cut)", () => {
    it("lowers the flag when Escape cancels the cut", async () => {
        dom();
        vi.spyOn(canvasPrototype(), "getContext").mockReturnValue(fakeCtx());
        try {
            const pending = splitCapture();
            expect(document.body.classList.contains(CAPTURING)).toBe(true);

            window.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
            await expect(pending).resolves.toBeNull();
            expect(document.body.classList.contains(CAPTURING)).toBe(false);
            expect(state.isCaptureMode).toBe(false);
        } finally {
            vi.restoreAllMocks();
        }
    });
});
