import { act, cleanup, render } from "@testing-library/preact";
import { useLayoutEffect, useRef } from "preact/hooks";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VulpesPanelsProvider, useVulpesPanels } from "./VulpesPanelsContext";
import type { VerticalGeometry } from "./verticalGeometry";
import { useVerticalPanelLimits } from "./useVerticalPanelLimits";

const STORAGE_KEY = "fox-reader-vulpes-panels";
let panels: ReturnType<typeof useVulpesPanels>;

function Shell({ geometry }: { geometry: VerticalGeometry }) {
    panels = useVulpesPanels();
    useLayoutEffect(() => panels.setVerticalGeometry(geometry), [geometry]);
    return (
        <>
            <section
                data-panel="entries"
                ref={(element) => panels.registerPanel("entries", element)}
                style={{ height: `${panels.getPanelSize("entries")}px` }}
            />
            <section
                data-panel="ocr"
                ref={(element) => panels.registerPanel("ocr", element)}
                style={{ height: `${panels.getPanelSize("ocr")}px` }}
            />
        </>
    );
}

function Fixture({ geometry }: { geometry: VerticalGeometry }) {
    return (
        <VulpesPanelsProvider>
            <Shell geometry={geometry} />
        </VulpesPanelsProvider>
    );
}

function measured(editorHeight = 1000, captureAvailable = 600, captureMinimum = 90): VerticalGeometry {
    return {
        editor: { height: editorHeight, splitter: 2 },
        capture: { available: captureAvailable, minimum: captureMinimum },
    };
}

function element(id: "entries" | "ocr"): HTMLElement {
    return document.querySelector<HTMLElement>(`[data-panel="${id}"]`)!;
}

function height(id: "entries" | "ocr"): number {
    return parseFloat(element(id).style.height);
}

function resize(id: "entries" | "ocr", delta: number): void {
    act(() => {
        panels.resizePanel(id, delta);
        panels.commitResize();
    });
}

function nextFrame(): Promise<void> {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

beforeEach(() => {
    localStorage.clear();
    document.body.className = "";
});

afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
});

describe("Vulpes vertical panel limits", () => {
    it("gives Translators 40% of the measured Editor and maintains that share on window resize", () => {
        const save = vi.spyOn(Storage.prototype, "setItem");
        render(<Fixture geometry={measured()} />);
        expect(height("entries")).toBe(598);
        expect(1000 - 2 - height("entries")).toBe(400);
        expect(height("ocr")).toBe(150);

        act(() => panels.setVerticalGeometry(measured(800)));
        expect(height("entries")).toBe(478);
        expect(800 - 2 - height("entries")).toBe(320);
        expect(panels.getPanelSize("entries")).toBe(478);
        expect(save).not.toHaveBeenCalled();
    });

    it("bounds splitter dragging to 15–70% for Translators and saves the selected share", () => {
        render(<Fixture geometry={measured()} />);
        resize("entries", 10000);
        expect(height("entries")).toBe(848);
        expect(1000 - 2 - height("entries")).toBe(150);
        expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).entries.translationFraction).toBeCloseTo(0.15);

        resize("entries", -10000);
        expect(height("entries")).toBe(298);
        expect(1000 - 2 - height("entries")).toBe(700);
        expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).entries.translationFraction).toBeCloseTo(0.7);

        act(() => panels.setVerticalGeometry(measured(2000)));
        expect(2000 - 2 - height("entries")).toBe(1400);
    });

    it("keeps fractional-pixel divider geometry inside both Translator percentage bounds", () => {
        const editor = { height: 999.5, splitter: 1.25 };
        render(<Fixture geometry={{ editor }} />);
        resize("entries", 10000);
        const smallestTranslation = editor.height - editor.splitter - height("entries");
        expect(smallestTranslation / editor.height).toBeGreaterThanOrEqual(0.15);
        expect(smallestTranslation / editor.height).toBeLessThan(0.152);

        resize("entries", -10000);
        const largestTranslation = editor.height - editor.splitter - height("entries");
        expect(largestTranslation / editor.height).toBeLessThanOrEqual(0.7);
        expect(largestTranslation / editor.height).toBeGreaterThan(0.698);
    });

    it("migrates a legacy pixel size once in memory and restores its ratio after saving and reloading", () => {
        const original = JSON.stringify({ entries: { visible: true, size: 300 } });
        localStorage.setItem(STORAGE_KEY, original);
        const save = vi.spyOn(Storage.prototype, "setItem");
        const view = render(<Fixture geometry={measured(800)} />);
        expect(height("entries")).toBe(300);
        expect(panels.getPanel("entries").translationFraction).toBeCloseTo(0.6225);
        expect(localStorage.getItem(STORAGE_KEY)).toBe(original);
        expect(save).not.toHaveBeenCalled();

        act(() => panels.setVerticalGeometry(measured(1600)));
        expect(height("entries")).toBe(602);
        expect(save).not.toHaveBeenCalled();
        act(() => panels.togglePanel("ocr"));
        expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).entries.translationFraction).toBeCloseTo(0.6225);

        view.unmount();
        render(<Fixture geometry={measured(1000)} />);
        expect(height("entries")).toBe(376);
        expect(panels.getPanelSize("entries")).toBe(376);
    });

    it("uses a saved Translator ratio instead of stale pixels and clamps out-of-range ratios", () => {
        localStorage.setItem(
            STORAGE_KEY,
            JSON.stringify({ entries: { visible: true, size: 9000, translationFraction: 0.55 } }),
        );
        const view = render(<Fixture geometry={measured()} />);
        expect(height("entries")).toBe(448);
        view.unmount();

        localStorage.setItem(
            STORAGE_KEY,
            JSON.stringify({ entries: { visible: true, size: 9000, translationFraction: 9 } }),
        );
        const upper = render(<Fixture geometry={measured()} />);
        expect(height("entries")).toBe(298);
        upper.unmount();

        localStorage.setItem(
            STORAGE_KEY,
            JSON.stringify({ entries: { visible: true, size: 0, translationFraction: -1 } }),
        );
        render(<Fixture geometry={measured()} />);
        expect(height("entries")).toBe(848);
    });

    it("resets to 40% Translators and a 150px capture preference using current geometry", () => {
        render(<Fixture geometry={measured()} />);
        resize("entries", 100);
        resize("ocr", 80);
        act(() => panels.setVerticalGeometry(measured(800, 300)));
        act(() => panels.resetPanels());

        expect(height("entries")).toBe(478);
        expect(height("ocr")).toBe(120);
        expect(panels.getPanel("ocr").size).toBe(150);
        expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).entries.translationFraction).toBe(0.4);
        act(() => panels.setVerticalGeometry(measured(800, 600)));
        expect(height("ocr")).toBe(150);
    });

    it("caps saved capture heights at 40%, preserves the preference, and starts dragging from the displayed size", () => {
        const original = JSON.stringify({ ocr: { visible: true, size: 900 } });
        localStorage.setItem(STORAGE_KEY, original);
        render(<Fixture geometry={measured()} />);
        expect(height("ocr")).toBe(240);
        resize("ocr", 100);
        expect(localStorage.getItem(STORAGE_KEY)).toBe(original);

        act(() => panels.setVerticalGeometry(measured(1000, 3000)));
        expect(height("ocr")).toBe(900);
        act(() => panels.setVerticalGeometry(measured()));
        resize("ocr", -10);
        expect(height("ocr")).toBe(230);
        expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).ocr.size).toBe(230);
    });

    it("prioritizes the measured one-line capture minimum when a short window cannot fit the percentage cap", () => {
        render(<Fixture geometry={measured(1000, 100, 97.2)} />);
        expect(height("ocr")).toBe(98);
        resize("ocr", -10000);
        expect(height("ocr")).toBe(98);

        act(() => panels.setVerticalGeometry({ capture: { available: 600, minimum: 110 } }));
        resize("ocr", -10000);
        expect(height("ocr")).toBe(110);
        expect(height("entries")).toBe(598);
    });

    it("retains valid geometry while hidden or incomplete measurements arrive", () => {
        render(<Fixture geometry={measured()} />);
        act(() =>
            panels.setVerticalGeometry({
                editor: { height: 1, splitter: 2 },
                capture: { available: NaN, minimum: 90 },
            }),
        );
        expect(height("entries")).toBe(598);
        expect(height("ocr")).toBe(150);

        act(() => panels.setVerticalGeometry({ capture: { available: 250, minimum: 90 } }));
        expect(height("entries")).toBe(598);
        expect(height("ocr")).toBe(100);
        act(() => panels.togglePanel("editor"));
        expect(height("entries")).toBe(598);
        expect(height("ocr")).toBe(100);
    });

    it("coalesces live row changes and persists once when the drag ends", async () => {
        render(<Fixture geometry={measured()} />);
        const save = vi.spyOn(Storage.prototype, "setItem");
        const writes: Record<string, number> = { entries: 0, ocr: 0 };
        for (const id of ["entries", "ocr"] as const) {
            const panel = element(id);
            Object.defineProperty(panel, "style", {
                value: new Proxy(panel.style, {
                    set(target, property, value) {
                        if (property === "height") writes[id]++;
                        return Reflect.set(target, property, value);
                    },
                }),
                configurable: true,
            });
        }

        act(() => {
            for (let i = 0; i < 8; i++) {
                panels.resizePanel("entries", 1);
                panels.resizePanel("ocr", 1);
            }
        });
        expect(writes).toEqual({ entries: 0, ocr: 0 });
        expect(save).not.toHaveBeenCalled();
        await nextFrame();
        expect(writes).toEqual({ entries: 1, ocr: 1 });
        expect(height("entries")).toBe(606);
        expect(height("ocr")).toBe(158);
        expect(save).not.toHaveBeenCalled();
        act(() => panels.commitResize());
        expect(save).toHaveBeenCalledOnce();
        const stored = JSON.parse(localStorage.getItem(STORAGE_KEY)!);
        expect(stored.entries.translationFraction).toBeCloseTo(0.392);
        expect(stored.ocr.size).toBe(158);
    });
});

class GeometryObserver {
    static current: GeometryObserver;
    observed = new Set<Element>();
    observe = vi.fn((target: Element) => this.observed.add(target));
    disconnect = vi.fn(() => this.observed.clear());
    private callback: ResizeObserverCallback;

    constructor(callback: ResizeObserverCallback) {
        this.callback = callback;
        GeometryObserver.current = this;
    }

    changed(target: Element, width = 400): void {
        if (!this.observed.has(target)) return;
        this.callback(
            [{ target, contentRect: new DOMRect(0, 0, width, 100) }] as ResizeObserverEntry[],
            this as unknown as ResizeObserver,
        );
    }
}

function MeasurementFixture({ onGeometry }: { onGeometry: (geometry: VerticalGeometry) => void }) {
    const bodyRef = useRef<HTMLDivElement>(null);
    useVerticalPanelLimits(bodyRef, true, onGeometry);
    return (
        <div ref={bodyRef}>
            <div class="vx-editor__body">
                <section />
                <div class="ws-splitter--horizontal" />
                <section />
            </div>
            <main class="vx-canvas">
                <div class="vx-tools" />
                <div class="ws-splitter--horizontal" />
                <div class="vx-capture" style={{ borderTop: "1px solid", borderBottom: "1px solid" }}>
                    <section
                        class="vx-section"
                        style={{ borderTop: "1px solid", borderBottom: "1px solid", rowGap: "4px" }}
                    >
                        <header class="vx-section__head" style={{ border: "0" }} />
                        <div class="vx-section__body" style={{ border: "0" }}>
                            <div class="vx-capture__body" style={{ padding: "8px 0", border: "0" }}>
                                <div class="textarea-wrapper" style={{ padding: "2px 0", border: "0" }}>
                                    <textarea
                                        id="textArea"
                                        style={{
                                            lineHeight: "48px",
                                            padding: "6px 0",
                                            border: "1px solid",
                                            minHeight: "0",
                                            margin: "0",
                                        }}
                                    />
                                </div>
                            </div>
                        </div>
                    </section>
                </div>
            </main>
        </div>
    );
}

function measuredElements() {
    const dimensions = { editor: 800, canvas: 700, tools: 50 };
    vi.stubGlobal("ResizeObserver", GeometryObserver);
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
        let height = 100;
        if (this.matches(".vx-editor__body")) height = dimensions.editor;
        if (this.matches(".vx-canvas")) height = dimensions.canvas;
        if (this.matches(".vx-tools")) height = dimensions.tools;
        if (this.matches(".vx-section__head")) height = 30;
        if (this.matches(".ws-splitter--horizontal")) height = this.parentElement?.matches(".vx-canvas") ? 1 : 2;
        return new DOMRect(0, 0, 400, height);
    });
    return dimensions;
}

describe("Vulpes vertical geometry measurement", () => {
    it("waits for Vulpes styles, then measures dividers and the textarea line with its wrappers and section chrome", async () => {
        measuredElements();
        const onGeometry = vi.fn();
        render(<MeasurementFixture onGeometry={onGeometry} />);
        expect(onGeometry).not.toHaveBeenCalled();
        document.body.classList.add("layout-vulpes");
        await nextFrame();

        expect(onGeometry).toHaveBeenCalledExactlyOnceWith({
            editor: { height: 800, splitter: 2 },
            capture: { available: 649, minimum: 120 },
        });
    });

    it("ignores textarea height-only changes, batches container measurements and cancels work on unmount", async () => {
        const dimensions = measuredElements();
        document.body.classList.add("layout-vulpes");
        const onGeometry = vi.fn();
        const view = render(<MeasurementFixture onGeometry={onGeometry} />);
        const observer = GeometryObserver.current;
        observer.changed(document.querySelector("#textArea")!);
        await nextFrame();
        expect(onGeometry).toHaveBeenCalledOnce();

        dimensions.editor = 900;
        dimensions.tools = 80;
        observer.changed(document.querySelector(".vx-editor__body")!);
        observer.changed(document.querySelector(".vx-tools")!);
        expect(onGeometry).toHaveBeenCalledOnce();
        await nextFrame();
        expect(onGeometry).toHaveBeenCalledTimes(2);
        expect(onGeometry).toHaveBeenLastCalledWith({
            editor: { height: 900, splitter: 2 },
            capture: { available: 619, minimum: 120 },
        });

        dimensions.tools = 100;
        observer.changed(document.querySelector(".vx-tools")!);
        view.unmount();
        await nextFrame();
        expect(observer.disconnect).toHaveBeenCalledOnce();
        expect(onGeometry).toHaveBeenCalledTimes(2);
    });
});
