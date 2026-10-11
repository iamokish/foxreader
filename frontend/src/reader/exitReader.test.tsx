/**
 * Returning from the reader must leave the page laid out for work mode.
 *
 * The two modes place the page by different mechanics. Reader mode keeps its own
 * pan in `state.panX/panY` and writes it into `#mainImage`'s transform; work mode
 * ignores that pan and centres the page, letting the container scroll for any
 * overflow. So a page the reader left dragged off-centre carries a transform work
 * mode never asked for, and sits outside the stage.
 *
 * This was layout-specific in a way that made it easy to miss: Vulpes re-fits by
 * accident, because leaving the reader un-hides its chrome and the `ResizeObserver`
 * on its stage fires. Basic has no observer, so the stale transform survived. Both
 * layouts are asserted here for that reason.
 */

import { act, render } from "@testing-library/preact";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LayoutProvider } from "../layouts/LayoutContext";
import { ThemeProvider } from "../themes/ThemeContext";
import { WorkspaceProvider } from "../workspace";
import { BasicLayout } from "../layouts/BasicLayout";
import { VulpesLayout } from "../layouts/VulpesLayout";
import { ReaderViewProvider } from "./ReaderViewContext";
import { state } from "../state";

/** A page big enough that work mode's fit is a real scale, not 1. */
const NATURAL = { width: 1200, height: 1800 };

function renderLayout(layout: "vulpes" | "basic", mode: "work" | "reader") {
    localStorage.setItem("fox-reader-layout", layout);
    localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode }));
    return render(
        <LayoutProvider>
            <ThemeProvider>
                <WorkspaceProvider>
                    <ReaderViewProvider>{layout === "vulpes" ? <VulpesLayout /> : <BasicLayout />}</ReaderViewProvider>
                </WorkspaceProvider>
            </ThemeProvider>
        </LayoutProvider>,
    );
}

const mainImage = () => document.getElementById("mainImage") as HTMLImageElement;
const container = () => document.getElementById("imageContainer") as HTMLElement;

/**
 * Give the stage and the page real dimensions.
 *
 * jsdom lays nothing out, so `clientWidth` and `naturalWidth` are 0 and every fit
 * path bails before it does anything. These are the two inputs the transform maths
 * actually reads.
 */
function measurable(): void {
    for (const [prop, value] of [
        ["clientWidth", 900],
        ["clientHeight", 700],
    ] as const) {
        Object.defineProperty(container(), prop, { value, configurable: true });
    }
    for (const [prop, value] of [
        ["naturalWidth", NATURAL.width],
        ["naturalHeight", NATURAL.height],
    ] as const) {
        Object.defineProperty(mainImage(), prop, { value, configurable: true });
    }
    mainImage().style.display = "block";
}

/** Let the `requestAnimationFrame` the exit path defers to run. */
function nextFrame(): Promise<void> {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/**
 * Drain the work the mount schedules, before the test does anything.
 *
 * Entering the reader queues a `setTimeout(refitViewer, 0)`. Left pending, it fires
 * during the test's own `await` -- after the mode has already been switched -- and
 * re-fits the page for reasons that have nothing to do with the exit path. That
 * made an earlier version of these tests pass against the unfixed code.
 */
function settle(): Promise<void> {
    return act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 1));
        await nextFrame();
    });
}

/**
 * The translate offsets of the page's transform.
 *
 * `null` when there is no `translate3d` to read, rather than a zero: a missing
 * transform must not look like a centred one, or an assertion that the page was
 * re-centred passes on a page that was never laid out at all.
 */
function translation(): { x: number; y: number } | null {
    const match = /translate3d\((-?[\d.]+)px,\s*(-?[\d.]+)px/.exec(mainImage().style.transform);
    return match ? { x: Number(match[1]), y: Number(match[2]) } : null;
}

/**
 * Where work mode puts this page, from the same arithmetic `applyWorkTransform`
 * uses: fit the whole page into the stage, then centre whichever axis has room.
 */
const WORK_FIT = Math.min(900 / NATURAL.width, 700 / NATURAL.height);
const WORK_CENTRE = {
    x: Math.max(0, (900 - NATURAL.width * WORK_FIT) / 2),
    y: Math.max(0, (700 - NATURAL.height * WORK_FIT) / 2),
};

beforeEach(() => {
    localStorage.clear();
    document.body.innerHTML = "";
    document.body.className = "";
    document.body.removeAttribute("style");
    state.currentFiles = ["p1.png"];
    state.currentImageFile = "p1.png";
    state.stripMode = false;
    state.panX = 0;
    state.panY = 0;
    state.zoomLevel = 1;
    state.isHorizontalFitEnabled = false;
    vi.stubGlobal(
        "fetch",
        vi.fn(() => Promise.resolve({ ok: false, json: () => Promise.resolve({}) })),
    );
});

afterEach(() => {
    vi.unstubAllGlobals();
});

describe.each(["basic", "vulpes"] as const)("%s layout: returning from the reader", (layout) => {
    it("re-centres a page the reader left dragged off to one side", async () => {
        const { container: root } = renderLayout(layout, "reader");
        measurable();
        await settle();

        // What a drag in the reader leaves behind: a pan, written into the transform.
        act(() => {
            state.panX = -480;
            state.panY = -260;
            mainImage().style.transform = "translate3d(-480px, -260px, 0) scale(1)";
        });

        // Leave the reader the way the UI does -- through the mode, not by poking
        // the DOM -- so the real exit path is what runs.
        act(() => {
            (window as unknown as { setWorkspaceMode: (m: "work") => void }).setWorkspaceMode("work");
        });
        await act(async () => {
            await nextFrame();
        });

        void root;
        // The exact placement work mode computes, not merely "not negative": a page
        // with no transform at all would satisfy a sign check while being just as
        // broken as one still carrying the reader's pan.
        const offsets = translation();
        expect(offsets).not.toBeNull();
        expect(offsets!.x).toBeCloseTo(WORK_CENTRE.x, 1);
        expect(offsets!.y).toBeCloseTo(WORK_CENTRE.y, 1);
    });

    it("drops the reader's pan from the shared state", async () => {
        renderLayout(layout, "reader");
        measurable();
        await settle();
        act(() => {
            state.panX = -480;
            state.panY = -260;
        });

        act(() => {
            (window as unknown as { setWorkspaceMode: (m: "work") => void }).setWorkspaceMode("work");
        });
        await act(async () => {
            await nextFrame();
        });

        expect(state.panX).toBe(0);
        expect(state.panY).toBe(0);
    });

    it("re-derives the work-mode fit rather than keeping the reader's zoom", async () => {
        renderLayout(layout, "reader");
        measurable();
        await settle();
        act(() => {
            // A zoom only the reader would have set.
            state.zoomLevel = 3;
        });

        act(() => {
            (window as unknown as { setWorkspaceMode: (m: "work") => void }).setWorkspaceMode("work");
        });
        await act(async () => {
            await nextFrame();
        });

        // The whole page fitted into 900x700 is bounded by height here.
        expect(state.zoomLevel).toBeCloseTo(700 / NATURAL.height, 5);
    });

    it("leaves the page alone while the reader is still open", async () => {
        renderLayout(layout, "reader");
        measurable();
        await settle();

        act(() => {
            state.panX = -480;
            mainImage().style.transform = "translate3d(-480px, 0px, 0) scale(1)";
        });

        await act(async () => {
            await nextFrame();
        });

        // Still in reader mode: the pan is the reader's to keep.
        expect(state.panX).toBe(-480);
        expect(translation()?.x).toBe(-480);
    });
});
