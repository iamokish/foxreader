/**
 * The reader's idle behaviour: a still pointer hides everything drawn over the page.
 *
 * Driven by a single `setTimeout` reset on each input event, so there is no polling
 * and no animation frame running while the reader sits still. These tests use fake
 * timers because the delay is three seconds -- the point is *which* events reset it
 * and what the class does, not how long it waits.
 */

import { act, fireEvent, render } from "@testing-library/preact";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LayoutProvider } from "../layouts/LayoutContext";
import { ThemeProvider } from "../themes/ThemeContext";
import { WorkspaceProvider } from "../workspace";
import { BasicLayout } from "../layouts/BasicLayout";
import { ReaderViewProvider } from "./ReaderViewContext";
import { state } from "../state";

/** Comfortably past the 3s delay. */
const PAST_IDLE = 4000;

/**
 * Advance the clock and let Preact render.
 *
 * `advanceTimersByTime` is synchronous, but the `setIdle` it causes is flushed by
 * Preact on a microtask -- so without `act` the assertion runs a tick before the
 * class is on the element.
 */
async function idleFor(ms: number): Promise<void> {
    await act(async () => {
        vi.advanceTimersByTime(ms);
    });
}

/** Fire an input event on `window` and let the resulting render settle. */
async function input(name: string): Promise<void> {
    await act(async () => {
        fireEvent(window, new Event(name, { bubbles: true }));
    });
}

/** Fire a keydown for a named key on `window`. */
async function keyInput(key: string): Promise<void> {
    await act(async () => {
        fireEvent(window, new KeyboardEvent("keydown", { key, bubbles: true }));
    });
}

function renderReader() {
    localStorage.setItem("fox-reader-layout", "basic");
    localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
    return render(
        <LayoutProvider>
            <ThemeProvider>
                <WorkspaceProvider>
                    <ReaderViewProvider>
                        <BasicLayout />
                    </ReaderViewProvider>
                </WorkspaceProvider>
            </ThemeProvider>
        </LayoutProvider>,
    );
}

/**
 * The reader surface.
 *
 * Typed against `Element`, not `HTMLElement`: `render()`'s `container` is an
 * `Element`, and `classList` is on the base interface anyway. Demanding the
 * narrower type buys nothing here and the call sites would have to cast.
 */
const surface = (container: Element): HTMLElement => container.querySelector<HTMLElement>(".vx-reader")!;

beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
    document.body.innerHTML = "";
    document.body.className = "";
    document.body.removeAttribute("style");
    state.currentFiles = [];
    state.currentImageFile = "";
    state.stripMode = false;
    vi.stubGlobal(
        "fetch",
        vi.fn(() => Promise.resolve({ ok: false, json: () => Promise.resolve({}) })),
    );
});

afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
});

describe("reader idle", () => {
    it("starts visible", () => {
        const { container } = renderReader();
        expect(surface(container).classList.contains("is-idle")).toBe(false);
    });

    it("hides itself once nothing has been touched for a few seconds", async () => {
        const { container } = renderReader();
        await idleFor(PAST_IDLE);
        expect(surface(container).classList.contains("is-idle")).toBe(true);
    });

    it("marks the body too, so the cursor goes over the strip and the stage as well", async () => {
        // The pointer is over `#strip-scroller` or `#imageContainer` in the
        // continuous view, and neither is inside `.vx-reader`.
        const { container } = renderReader();
        await idleFor(PAST_IDLE);
        expect(document.body.classList.contains("is-reader-idle")).toBe(true);
        expect(surface(container).classList.contains("is-idle")).toBe(true);
    });

    it.each(["pointermove", "pointerdown", "wheel", "keydown"] as const)(
        "comes straight back on %s",
        async (eventName) => {
            const { container } = renderReader();
            await idleFor(PAST_IDLE);
            expect(surface(container).classList.contains("is-idle")).toBe(true);

            await input(eventName);
            expect(surface(container).classList.contains("is-idle")).toBe(false);
            expect(document.body.classList.contains("is-reader-idle")).toBe(false);
        },
    );

    it.each(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"] as const)(
        "leaves a hidden UI hidden on %s",
        async (key) => {
            // Page-turn keys are for reading, not for reaching: flipping pages
            // with the chrome gone must not summon it.
            const { container } = renderReader();
            await idleFor(PAST_IDLE);
            expect(surface(container).classList.contains("is-idle")).toBe(true);

            await keyInput(key);
            expect(surface(container).classList.contains("is-idle")).toBe(true);
            expect(document.body.classList.contains("is-reader-idle")).toBe(true);
        },
    );

    it("still wakes on any other key", async () => {
        const { container } = renderReader();
        await idleFor(PAST_IDLE);
        expect(surface(container).classList.contains("is-idle")).toBe(true);

        await keyInput("x");
        expect(surface(container).classList.contains("is-idle")).toBe(false);
        expect(document.body.classList.contains("is-reader-idle")).toBe(false);
    });

    it("does not let arrow keys restart the timer while visible", async () => {
        // Symmetric with the above: arrows are not input, so the hide deadline
        // set by the last real input stands.
        const { container } = renderReader();
        await keyInput("ArrowRight");
        await idleFor(PAST_IDLE);
        expect(surface(container).classList.contains("is-idle")).toBe(true);
    });

    it("does not hide while the pointer keeps moving", async () => {
        const { container } = renderReader();
        for (let i = 0; i < 4; i++) {
            await idleFor(2000);
            await input("pointermove");
        }
        expect(surface(container).classList.contains("is-idle")).toBe(false);
    });

    it("hides again after the input stops", async () => {
        const { container } = renderReader();
        await idleFor(PAST_IDLE);
        await input("pointermove");
        expect(surface(container).classList.contains("is-idle")).toBe(false);
        await idleFor(PAST_IDLE);
        expect(surface(container).classList.contains("is-idle")).toBe(true);
    });

    it("never hides the editing workspace, however still the pointer is", async () => {
        // Hiding the toolbar and the editor column mid-edit would be hostile; this is
        // reader-mode behaviour only.
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "work" }));
        const { container } = render(
            <LayoutProvider>
                <ThemeProvider>
                    <WorkspaceProvider>
                        <ReaderViewProvider>
                            <BasicLayout />
                        </ReaderViewProvider>
                    </WorkspaceProvider>
                </ThemeProvider>
            </LayoutProvider>,
        );
        await idleFor(PAST_IDLE);
        expect(container.querySelector(".vx-reader")).toBeNull();
        expect(document.body.classList.contains("is-reader-idle")).toBe(false);
    });

    it("leaves no body class behind when the reader unmounts", async () => {
        const { unmount } = renderReader();
        await idleFor(PAST_IDLE);
        expect(document.body.classList.contains("is-reader-idle")).toBe(true);
        unmount();
        expect(document.body.classList.contains("is-reader-idle")).toBe(false);
    });
});
