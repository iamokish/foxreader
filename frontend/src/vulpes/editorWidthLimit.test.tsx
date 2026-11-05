import { act, cleanup, render } from "@testing-library/preact";
import { useRef } from "preact/hooks";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VulpesPanelsProvider, useVulpesPanels } from "./VulpesPanelsContext";
import { useEditorWidthLimit } from "./useEditorWidthLimit";

const STORAGE_KEY = "fox-reader-vulpes-panels";

class Sizes {
    static instances: Sizes[] = [];
    observed = new Set<Element>();
    observe = vi.fn((element: Element) => this.observed.add(element));
    disconnect = vi.fn(() => this.observed.clear());
    private callback: ResizeObserverCallback;

    constructor(callback: ResizeObserverCallback) {
        this.callback = callback;
        Sizes.instances.push(this);
    }

    changed(element: Element): void {
        if (this.observed.has(element)) this.callback([], this as unknown as ResizeObserver);
    }
}

let panels: ReturnType<typeof useVulpesPanels>;
let geometry: { body: number; pages: number; buttons: number; editorVisible: boolean };

function Shell({ enabled }: { enabled: boolean }) {
    panels = useVulpesPanels();
    const bodyRef = useRef<HTMLDivElement>(null);
    useEditorWidthLimit(bodyRef, enabled, panels.setEditorWidthLimit);
    return (
        <div class="vx-body" ref={bodyRef}>
            <aside class="vx-rail" />
            <main class="vx-canvas">
                <div class="vx-tools" style={{ padding: "0 12px", borderLeft: "1px solid", borderRight: "1px solid" }}>
                    <div class="buttons">Capture, OCR, language and options</div>
                </div>
            </main>
            <aside
                class="vx-editor"
                ref={(element) => panels.registerPanel("editor", element)}
                style={{ width: `${panels.getPanel("editor").size}px` }}
            />
        </div>
    );
}

function Fixture({ enabled = true }: { enabled?: boolean }) {
    return (
        <VulpesPanelsProvider>
            <Shell enabled={enabled} />
        </VulpesPanelsProvider>
    );
}

function editor(): HTMLElement {
    return document.querySelector<HTMLElement>(".vx-editor")!;
}

function displayedEditorWidth(): number {
    if (!geometry.editorVisible) return 0;
    const element = editor();
    return Math.min(parseFloat(element.style.width), parseFloat(element.style.maxWidth) || Infinity);
}

function storedEditor(width = 700): string {
    const stored = JSON.stringify({ editor: { visible: true, size: width } });
    localStorage.setItem(STORAGE_KEY, stored);
    return stored;
}

function nextFrame(): Promise<void> {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

beforeEach(() => {
    localStorage.clear();
    Sizes.instances = [];
    geometry = { body: 1400, pages: 100, buttons: 650, editorVisible: true };
    vi.stubGlobal("innerWidth", 2000);
    vi.stubGlobal("ResizeObserver", Sizes);
    // jsdom does not lay out flex columns. Model their rendered widths, including
    // the two 1px splitters and CSS max-width, while using the real provider/hook.
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
        let width = 0;
        if (this.classList.contains("vx-body")) width = geometry.body;
        if (this.classList.contains("vx-rail")) width = geometry.pages;
        if (this.classList.contains("vx-editor")) width = displayedEditorWidth();
        if (this.classList.contains("vx-canvas")) {
            width = Math.max(0, geometry.body - geometry.pages - 2 - displayedEditorWidth());
        }
        if (this.classList.contains("buttons")) width = geometry.buttons;
        return new DOMRect(0, 0, width, 100);
    });
});

afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
});

describe("Vulpes Editor width limit", () => {
    it("reserves the natural toolbar width and insets without fractional overflow", () => {
        geometry.body = 1400.25;
        geometry.buttons = 650.4;
        storedEditor();
        render(<Fixture />);

        expect(editor().style.maxWidth).toBe("621px");
        expect(displayedEditorWidth()).toBe(621);
        const canvasWidth = document.querySelector(".vx-canvas")!.getBoundingClientRect().width;
        expect(canvasWidth).toBeGreaterThanOrEqual(geometry.buttons + 26);
    });

    it("recalculates for window, Pages rail and optional toolbar button changes", async () => {
        render(<Fixture />);
        const observer = Sizes.instances[0]!;
        expect(editor().style.maxWidth).toBe("622px");

        geometry.body = 1300;
        observer.changed(document.querySelector(".vx-body")!);
        await nextFrame();
        expect(editor().style.maxWidth).toBe("522px");

        geometry.pages = 160;
        observer.changed(document.querySelector(".vx-rail")!);
        await nextFrame();
        expect(editor().style.maxWidth).toBe("462px");

        geometry.buttons += 50;
        observer.changed(document.querySelector(".buttons")!);
        await nextFrame();
        expect(editor().style.maxWidth).toBe("412px");

        geometry.pages = 0;
        observer.changed(document.querySelector(".vx-rail")!);
        await nextFrame();
        expect(editor().style.maxWidth).toBe("572px");
    });

    it("keeps the Editor minimum on a narrow-window reload and restores the saved width when enlarged", async () => {
        geometry.body = 600;
        vi.stubGlobal("innerWidth", 600);
        const original = storedEditor();
        render(<Fixture />);
        const save = vi.spyOn(Storage.prototype, "setItem");

        expect(editor().style.maxWidth).toBe("260px");
        expect(displayedEditorWidth()).toBe(260);
        expect(document.querySelector(".vx-canvas")!.getBoundingClientRect().width).toBeLessThan(geometry.buttons);

        geometry.body = 2400;
        vi.stubGlobal("innerWidth", 2400);
        Sizes.instances[0]!.changed(document.querySelector(".vx-body")!);
        await nextFrame();
        expect(editor().style.maxWidth).toBe("720px");
        expect(displayedEditorWidth()).toBe(700);
        expect(panels.getPanel("editor").size).toBe(700);
        expect(localStorage.getItem(STORAGE_KEY)).toBe(original);
        expect(save).not.toHaveBeenCalled();
    });

    it("starts a drag at the displayed cap rather than the wider saved preference", () => {
        const original = storedEditor();
        render(<Fixture />);
        expect(displayedEditorWidth()).toBe(622);

        act(() => {
            panels.resizePanel("editor", 0);
            panels.resizePanel("editor", 10);
            panels.commitResize();
        });
        expect(localStorage.getItem(STORAGE_KEY)).toBe(original);

        act(() => panels.resizePanel("editor", -10));
        expect(localStorage.getItem(STORAGE_KEY)).toBe(original);
        act(() => panels.commitResize());
        expect(displayedEditorWidth()).toBe(612);
        expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).editor.size).toBe(612);

        act(() => {
            panels.resizePanel("editor", 5000);
            panels.commitResize();
        });
        expect(displayedEditorWidth()).toBe(622);
        expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).editor.size).toBe(622);
    });

    it("keeps the cap through visibility updates and resetting panel preferences", () => {
        geometry.body = 1078;
        storedEditor();
        render(<Fixture />);
        expect(displayedEditorWidth()).toBe(300);

        act(() => panels.togglePanel("ocr"));
        expect(editor().style.width).toBe("700px");
        expect(editor().style.maxWidth).toBe("300px");
        expect(displayedEditorWidth()).toBe(300);

        act(() => panels.resetPanels());
        expect(editor().style.width).toBe("360px");
        expect(editor().style.maxWidth).toBe("300px");
        expect(displayedEditorWidth()).toBe(300);
    });

    it("ignores geometry from hidden panels and zero-sized containers", async () => {
        render(<Fixture />);
        const observer = Sizes.instances[0]!;
        geometry.buttons = 0;
        observer.changed(document.querySelector(".buttons")!);
        await nextFrame();
        expect(editor().style.maxWidth).toBe("622px");

        geometry.buttons = 650;
        geometry.editorVisible = false;
        geometry.body = 600;
        observer.changed(document.querySelector(".vx-body")!);
        await nextFrame();
        expect(editor().style.maxWidth).toBe("622px");

        geometry.editorVisible = true;
        geometry.body = 0;
        observer.changed(document.querySelector(".vx-body")!);
        await nextFrame();
        expect(editor().style.maxWidth).toBe("622px");
    });

    it("cancels pending measurements in reader mode and on unmount, then remeasures on return", async () => {
        const view = render(<Fixture />);
        const initialObserver = Sizes.instances[0]!;
        geometry.body = 1200;
        initialObserver.changed(document.querySelector(".vx-body")!);
        view.rerender(<Fixture enabled={false} />);
        expect(initialObserver.disconnect).toHaveBeenCalledOnce();

        window.dispatchEvent(new Event("resize"));
        initialObserver.changed(document.querySelector(".vx-body")!);
        await nextFrame();
        expect(editor().style.maxWidth).toBe("622px");

        view.rerender(<Fixture enabled />);
        expect(editor().style.maxWidth).toBe("422px");
        const currentObserver = Sizes.instances[1]!;
        const detachedEditor = editor();
        geometry.body = 600;
        currentObserver.changed(document.querySelector(".vx-body")!);
        view.unmount();
        expect(currentObserver.disconnect).toHaveBeenCalledOnce();
        const reads = vi.mocked(HTMLElement.prototype.getBoundingClientRect);
        reads.mockClear();
        window.dispatchEvent(new Event("resize"));
        await nextFrame();
        expect(reads).not.toHaveBeenCalled();
        expect(detachedEditor.style.maxWidth).toBe("422px");
    });

    it("still measures window resizes when ResizeObserver is unavailable", async () => {
        vi.stubGlobal("ResizeObserver", undefined);
        render(<Fixture />);
        expect(editor().style.maxWidth).toBe("622px");
        geometry.body = 1200;
        window.dispatchEvent(new Event("resize"));
        await nextFrame();
        expect(editor().style.maxWidth).toBe("422px");
    });

    it("coalesces repeated observer notifications into one measurement per frame", async () => {
        render(<Fixture />);
        const observer = Sizes.instances[0]!;
        const buttons = document.querySelector<HTMLElement>(".buttons")!;
        const read = vi.fn(buttons.getBoundingClientRect.bind(buttons));
        buttons.getBoundingClientRect = read;

        for (let width = 660; width <= 690; width += 10) {
            geometry.buttons = width;
            observer.changed(buttons);
            observer.changed(document.querySelector(".vx-body")!);
            observer.changed(document.querySelector(".vx-rail")!);
        }
        expect(read).not.toHaveBeenCalled();
        expect(editor().style.maxWidth).toBe("622px");
        await nextFrame();
        expect(read).toHaveBeenCalledOnce();
        expect(editor().style.maxWidth).toBe("582px");
    });
});
