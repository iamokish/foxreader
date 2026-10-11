/**
 * The Vulpes shell's contract with the imperative modules.
 *
 * `main.tsx` calls `initPanning`, `initZoom`, `setupImageErrorHandler` and the
 * dropdown wiring *once*, at startup, against elements found by id. Any layout
 * that fails to mount one of those ids -- or mounts it twice -- breaks panning,
 * zooming, capture or OCR in a way no type checks and no amount of CSS review
 * would catch. So the ids are asserted here, by name, exactly once each.
 *
 * Add to `REQUIRED_IDS` whenever the app grows a new one.
 */

import { fireEvent, render } from "@testing-library/preact";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { LayoutProvider } from "./LayoutContext";
import { ThemeProvider } from "../themes/ThemeContext";
import { WorkspaceProvider } from "../workspace";
import { VulpesLayout, invertDelta } from "./VulpesLayout";
import { BasicLayout } from "./BasicLayout";
import { ReaderViewProvider } from "../reader";
import { VX_DEFAULT_PANELS, VX_PANEL_IDS, type VxPanelId } from "../vulpes/VulpesPanelsContext";
import { state } from "../state";

/** Every id the imperative modules look up with `getElementById`. */
const REQUIRED_IDS = [
    // viewer.ts: initPanning, initZoom, applyImageFit, setupImageErrorHandler
    "imageContainer",
    "mainImage",
    "drawCanvas",
    "cropSelector",
    "gallery",
    // viewer.ts: the Original/Saved switch and its swap spinner
    "variantSwitch",
    "variantOriginalBtn",
    "variantSavedBtn",
    "variantLoader",
    "variantLoaderText",
    // ocr.ts / ui.ts
    "textArea",
    "ocrLoader",
    "languageSelect",
    // entries.ts
    "savedEntriesList",
    // translate.ts
    "translatedText",
    // the confirm row -- in the footer for this layout, see item 7
    "confirmActionRow",
    "alignmentCycleBtn",
    "inpaintGenerateBtn",
    "inpaintPreviewBtn",
    "clearPageBtn",
    "confirmTranslationBtn",
    // fonts.ts
    "fontSelect",
    "fontSelectTrigger",
    "fontOptionsContainer",
    "selectedFontLabel",
    // main.tsx: the folder field and every option switch
    "folderPathInput",
    "loadFolderBtn",
    "enableML",
    "grayScaleOCR",
    "horizontalFit",
    "autoOCRresponse",
    "autoTLresponse",
    "liveInpaint",
    "enableContext",
    "enableCharacters",
    // ocr.ts: the capture buttons it enables and disables as a group
    "rectCapture",
    "freeCapture",
    "PageCapture",
] as const;

function renderVulpes() {
    return render(
        <LayoutProvider>
            <ThemeProvider>
                <WorkspaceProvider>
                    <ReaderViewProvider>
                        <VulpesLayout />
                    </ReaderViewProvider>
                </WorkspaceProvider>
            </ThemeProvider>
        </LayoutProvider>,
    );
}

/** The Basic layout, which mounts the same reader in its own colours. */
function renderBasic() {
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

beforeEach(() => {
    localStorage.clear();
    localStorage.setItem("fox-reader-layout", "vulpes");
    document.body.innerHTML = "";
    document.body.className = "";
    // The panel sizes and the page gaps are published as custom properties on the
    // body, so one test's values must not be visible to the next.
    document.body.removeAttribute("style");
    state.currentFiles = [];
    state.currentImageFile = "";
    state.stripMode = false;
    // `TranslatorButtons` asks the backend which translators have tokens. Stubbed
    // so the render is deterministic and nothing hits the network.
    vi.stubGlobal(
        "fetch",
        vi.fn(() => Promise.resolve({ ok: false, json: () => Promise.resolve({}) })),
    );
});

afterEach(() => {
    vi.unstubAllGlobals();
});

describe("invertDelta", () => {
    it("grows a panel when the splitter below it is dragged downward", () => {
        expect(invertDelta(-25)).toBe(25);
        expect(invertDelta(25)).toBe(-25);
    });
});

describe("VulpesLayout DOM contract", () => {
    it.each(REQUIRED_IDS)("mounts #%s exactly once", (id) => {
        const { baseElement } = renderVulpes();
        expect(baseElement.querySelectorAll(`#${id}`)).toHaveLength(1);
    });

    it("renders the three columns and both bars", () => {
        const { container } = renderVulpes();
        expect(container.querySelector(".vx-topbar")).not.toBeNull();
        expect(container.querySelector(".vx-rail")).not.toBeNull();
        expect(container.querySelector(".vx-canvas")).not.toBeNull();
        expect(container.querySelector(".vx-editor")).not.toBeNull();
        expect(container.querySelector(".vx-statusbar")).not.toBeNull();
    });

    it("puts the gallery inside the pages rail, not under the stage", () => {
        const { container } = renderVulpes();
        expect(container.querySelector(".vx-rail #gallery")).not.toBeNull();
    });

    it("gives every resizable boundary a splitter", () => {
        // pages | canvas, ocr, canvas | editor, entries — four by default.
        const { container } = renderVulpes();
        expect(container.querySelectorAll(".ws-splitter").length).toBe(4);
    });
});

describe("VulpesLayout sections", () => {
    /**
     * Where each panel's size is written.
     *
     * An inline style on the element, written once per animation frame during a
     * drag -- never a custom property on `<body>`, which would re-resolve every
     * dependent on every pointer move instead.
     */
    const SIZED: Record<VxPanelId, { selector: string; axis: "width" | "height" }> = {
        pages: { selector: ".vx-rail", axis: "width" },
        editor: { selector: ".vx-editor", axis: "width" },
        ocr: { selector: ".vx-capture", axis: "height" },
        entries: { selector: '[data-section="entries"]', axis: "height" },
    };

    function sizeOf(container: Element, id: VxPanelId): string {
        const { selector, axis } = SIZED[id];
        return (container.querySelector(selector) as HTMLElement).style[axis];
    }

    /** A drag writes once per animation frame, so the assertion has to wait for one. */
    function nextFrame(): Promise<void> {
        return new Promise((resolve) => requestAnimationFrame(() => resolve()));
    }

    it("remembers the stored width of a column", () => {
        localStorage.setItem("fox-reader-vulpes-panels", JSON.stringify({ pages: { visible: true, size: 200 } }));
        const { container } = renderVulpes();
        expect(sizeOf(container, "pages")).toBe("200px");
    });

    it("sizes every section, not only the ones that were stored", () => {
        const { container } = renderVulpes();
        for (const id of VX_PANEL_IDS) {
            expect(sizeOf(container, id), id).toBe(`${VX_DEFAULT_PANELS[id].size}px`);
        }
    });

    it("writes no panel size onto the body", () => {
        // The regression this guards: an inherited custom property here would
        // re-resolve every dependent on every pointer move of a drag.
        const { container } = renderVulpes();
        void container;
        expect(document.body.getAttribute("style") ?? "").not.toContain("--vx-size");
    });

    it("clamps a stored width that is out of range", () => {
        // A window that was much wider last session, or a hand-edited file: a rail
        // 9000px wide would leave no stage and no splitter to drag back.
        localStorage.setItem("fox-reader-vulpes-panels", JSON.stringify({ pages: { visible: true, size: 9000 } }));
        const { container } = renderVulpes();
        expect(sizeOf(container, "pages")).toBe("420px");
    });

    it("survives unparseable stored panel sizes", () => {
        localStorage.setItem("fox-reader-vulpes-panels", "{nope");
        expect(() => renderVulpes()).not.toThrow();
    });

    it("follows the pointer live, and persists only once the drag ends", async () => {
        const { container } = renderVulpes();
        const splitter = container.querySelector(".ws-splitter--vertical")!;

        splitter.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, clientX: 0, clientY: 0 }));
        window.dispatchEvent(new PointerEvent("pointermove", { clientX: 20, clientY: 0, buttons: 1 }));
        await nextFrame();

        expect(sizeOf(container, "pages")).toBe("184px");
        // Nothing written yet: the whole point of deferring it.
        expect(localStorage.getItem("fox-reader-vulpes-panels")).toBeNull();

        window.dispatchEvent(new PointerEvent("pointerup", { clientX: 20, clientY: 0 }));
        expect(JSON.parse(localStorage.getItem("fox-reader-vulpes-panels")!).pages.size).toBe(184);
    });

    it("coalesces many moves in one frame into a single write", async () => {
        // A high-polling-rate mouse reports faster than the screen refreshes; there
        // is no point relaying out more often than it can be seen.
        const { container } = renderVulpes();
        const rail = container.querySelector(".vx-rail") as HTMLElement;
        const splitter = container.querySelector(".ws-splitter--vertical")!;
        splitter.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, clientX: 0, clientY: 0 }));

        let writes = 0;
        const observed = new Proxy(rail.style, {
            set(target, prop, value) {
                if (prop === "width") writes += 1;
                return Reflect.set(target, prop, value);
            },
        });
        Object.defineProperty(rail, "style", { value: observed, configurable: true });

        for (let x = 1; x <= 8; x++) {
            window.dispatchEvent(new PointerEvent("pointermove", { clientX: x, clientY: 0, buttons: 1 }));
        }
        expect(writes).toBe(0);
        await nextFrame();
        expect(writes).toBe(1);

        window.dispatchEvent(new PointerEvent("pointerup", { clientX: 8, clientY: 0 }));
    });

    it("lands the final position even if the drag ends before the frame runs", () => {
        // `commitResize` flushes synchronously, so releasing the mouse never shows a
        // frame of the previous size.
        const { container } = renderVulpes();
        const splitter = container.querySelector(".ws-splitter--vertical")!;
        splitter.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, clientX: 0, clientY: 0 }));
        window.dispatchEvent(new PointerEvent("pointermove", { clientX: 30, clientY: 0, buttons: 1 }));
        window.dispatchEvent(new PointerEvent("pointerup", { clientX: 30, clientY: 0 }));
        expect(sizeOf(container, "pages")).toBe("194px");
    });

    it("refuses to drag a section past its limit", async () => {
        const { container } = renderVulpes();
        const splitter = container.querySelector(".ws-splitter--vertical")!;
        splitter.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, clientX: 0, clientY: 0 }));
        window.dispatchEvent(new PointerEvent("pointermove", { clientX: 5000, clientY: 0, buttons: 1 }));
        await nextFrame();
        window.dispatchEvent(new PointerEvent("pointerup", { clientX: 5000, clientY: 0 }));
        expect(sizeOf(container, "pages")).toBe("420px");
    });

    it("collapses a section without unmounting what is inside it", () => {
        // Collapsing must never take `#textArea` out of the DOM: `ocr.ts` writes to
        // it by id, and an absent element loses the OCR result silently.
        const { container, baseElement } = renderVulpes();
        const toggle = container.querySelector('[data-section="ocr"] .vx-section__toggle')!;
        fireEvent.click(toggle);
        expect(container.querySelector('[data-section="ocr"]')?.classList.contains("is-collapsed")).toBe(true);
        expect(baseElement.querySelectorAll("#textArea")).toHaveLength(1);
    });

    it("gives the entries section no toggle to collapse with", () => {
        const { container } = renderVulpes();
        const section = container.querySelector('[data-section="entries"]')!;
        expect(section.querySelector(".vx-section__toggle")).toBeNull();
        expect(section.querySelector(".vx-section__title")?.textContent).toBe("Entries");
        expect(section.querySelector(".vx-section__note")?.textContent).toBe("text · style · transform");
        expect(section.classList.contains("is-collapsed")).toBe(false);
        // ...while collapsible sections keep theirs.
        expect(container.querySelector('[data-section="ocr"] .vx-section__toggle')).not.toBeNull();
    });

    it("revives entries hidden by a stored value from an older build", () => {
        // `loadPanels` normalises it back: with no toggle there is no way back
        // from a hidden entry list, so one must never load.
        localStorage.setItem("fox-reader-vulpes-panels", JSON.stringify({ entries: { visible: false, size: 300 } }));
        const { container, baseElement } = renderVulpes();
        const section = container.querySelector('[data-section="entries"]')!;
        expect(section.classList.contains("is-collapsed")).toBe(false);
        expect(section.querySelector(".vx-section__toggle")).toBeNull();
        expect(baseElement.querySelectorAll("#savedEntriesList")).toHaveLength(1);
    });

    it("hides the pages rail without unmounting the gallery", () => {
        const { container, baseElement } = renderVulpes();
        const railButton = container.querySelector('[title="Pages rail (H)"]')!;
        fireEvent.click(railButton);
        expect(container.querySelector(".vx-rail")?.classList.contains("is-collapsed")).toBe(true);
        expect(baseElement.querySelectorAll("#gallery")).toHaveLength(1);
    });
});

describe("VulpesLayout reader mode", () => {
    it("draws no reader surface in work mode", () => {
        const { container } = renderVulpes();
        expect(container.querySelector(".vx-reader")).toBeNull();
    });

    it("covers the stage with the reader rather than unmounting it", () => {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        const { container, baseElement } = renderVulpes();
        expect(container.querySelector(".vx-reader")).not.toBeNull();
        // Still exactly one of each: the editor's viewport is covered, not replaced.
        expect(baseElement.querySelectorAll("#imageContainer")).toHaveLength(1);
        expect(baseElement.querySelectorAll("#gallery")).toHaveLength(1);
    });

    it("offers all four page views", () => {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        const { container } = renderVulpes();
        const views = container.querySelectorAll('[aria-label="Page view"] button');
        expect(views).toHaveLength(4);
        expect(Array.from(views).map((b) => b.textContent)).toEqual([
            "crop_portraitSingle",
            "auto_storiesDouble",
            "view_dayVertical",
            "compareCompare",
        ]);
    });

    it("disables Compare until something has been typeset", () => {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        const { container } = renderVulpes();
        const compare = Array.from(container.querySelectorAll<HTMLButtonElement>('[aria-label="Page view"] button')).at(
            -1,
        )!;
        expect(compare.disabled).toBe(true);
    });

    it("offers the gap slider in the continuous view, at the 5px default", () => {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        localStorage.setItem("fox-reader-vulpes-view", JSON.stringify({ pageView: "vertical" }));
        const { container } = renderVulpes();
        const slider = container.querySelector<HTMLInputElement>('.vx-range input[type="range"]')!;
        expect(slider.min).toBe("0");
        expect(slider.max).toBe("10");
        expect(slider.value).toBe("5");
    });

    it("offers the wider gutter slider in the double view, at the 8px default", () => {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        localStorage.setItem("fox-reader-vulpes-view", JSON.stringify({ pageView: "double" }));
        const { container } = renderVulpes();
        const slider = container.querySelector<HTMLInputElement>('.vx-range input[type="range"]')!;
        expect(slider.min).toBe("0");
        expect(slider.max).toBe("40");
        expect(slider.value).toBe("8");
    });

    it("offers no gap at all in the single view, which has nothing to space", () => {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        localStorage.setItem("fox-reader-vulpes-view", JSON.stringify({ pageView: "single" }));
        const { container } = renderVulpes();
        expect(container.querySelector(".vx-range")).toBeNull();
    });

    it("publishes the chosen gap as a custom property the strip inherits", () => {
        // `buildStrip` creates the scroller imperatively, so the gap reaches it by
        // inheritance from <body> rather than by being set on the element.
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        localStorage.setItem("fox-reader-vulpes-view", JSON.stringify({ pageView: "vertical", verticalGap: 7 }));
        renderVulpes();
        expect(document.body.style.getPropertyValue("--strip-gap")).toBe("7px");
    });

    it("records a hand-picked view so webtoon detection stops overriding it", () => {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        const { container } = renderVulpes();
        const double = container.querySelectorAll<HTMLButtonElement>('[aria-label="Page view"] button')[1];
        fireEvent.click(double);
        const stored = JSON.parse(localStorage.getItem("fox-reader-vulpes-view")!);
        expect(stored.pageView).toBe("double");
        expect(stored.pinned).toBe(true);
    });

    it("shows the cover and wide chips only for the double view", () => {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        localStorage.setItem("fox-reader-vulpes-view", JSON.stringify({ pageView: "double" }));
        const { container } = renderVulpes();
        const chips = Array.from(container.querySelectorAll(".vx-chip")).map((c) => c.textContent);
        expect(chips).toContain("Cover");
        expect(chips).toContain("Wide");
    });
});

describe("Vulpes header", () => {
    it("shows the app icon from static/ and no wordmark", () => {
        const { container } = renderVulpes();
        const mark = container.querySelector<HTMLImageElement>(".vx-brand__mark");
        expect(mark).not.toBeNull();
        expect(mark?.tagName).toBe("IMG");
        expect(mark?.getAttribute("src")).toMatch(/^\/static\/.*\.png$/);
        // The title text is gone; the identity lives in the icon and its tooltip.
        expect(container.querySelector(".vx-brand__name")).toBeNull();
        expect(container.querySelector(".vx-brand")?.textContent?.trim()).toBe("");
        expect(container.querySelector(".vx-brand")?.getAttribute("title")).toBe("Fox Reader");
    });

    it("leaves the brand mark unlinked and out of the tab order", () => {
        // A span, not an anchor: no destination, no click, no focus.
        const { container } = renderVulpes();
        const brand = container.querySelector<HTMLElement>(".vx-brand")!;
        expect(brand.tagName).toBe("SPAN");
        expect(brand.getAttribute("href")).toBeNull();
        expect(brand.tabIndex).toBe(-1);
    });
});

describe("Vulpes editor column", () => {
    it("has no column header of its own, but keeps its accessible name", () => {
        // The rail keeps "Pages"; this column's head was removed, so the
        // entries and translation sections below are the only titles.
        const { container } = renderVulpes();
        expect(container.querySelector(".vx-editor__head")).toBeNull();
        expect(container.querySelector(".vx-editor__title")).toBeNull();
        expect(container.querySelector(".vx-editor")?.getAttribute("aria-label")).toBe("Editor");
        expect(container.querySelector(".vx-rail__head")).not.toBeNull();
    });

    it("has no element still calling itself an inspector", () => {
        const { container } = renderVulpes();
        expect(container.querySelector('[class*="inspector"]')).toBeNull();
        expect(container.textContent).not.toContain("Inspector");
    });

    it("keeps the translator picker in the Translators header, not the capture", () => {
        const { container } = renderVulpes();
        expect(container.querySelector(".vx-capture__body .translator-buttons")).toBeNull();
        const head = container.querySelector(".vx-editor__translation > .vx-section__head");
        expect(head?.querySelector(".vx-section__title")?.textContent).toBe("Translators");
        const buttons = head?.querySelector(".translator-buttons");
        expect(buttons).not.toBeNull();
        // The stubbed backend still renders the built-in engines.
        expect(buttons?.querySelectorAll("button").length).toBeGreaterThan(0);
    });
});

describe("Vulpes action row placement", () => {
    it("puts the entry actions in the footer, not the translation section", () => {
        const { container } = renderVulpes();
        const row = container.querySelector("#confirmActionRow");
        expect(row).not.toBeNull();
        expect(row?.closest(".vx-statusbar")).not.toBeNull();
        expect(row?.closest(".vx-editor")).toBeNull();
    });

    it("keeps exactly one of each action id in the document", () => {
        // They are found by `getElementById`: two copies and every caller silently
        // gets whichever comes first.
        const { baseElement } = renderVulpes();
        for (const id of [
            "confirmActionRow",
            "alignmentCycleBtn",
            "inpaintGenerateBtn",
            "inpaintPreviewBtn",
            "clearPageBtn",
            "confirmTranslationBtn",
            "fontSelect",
        ]) {
            expect(baseElement.querySelectorAll(`#${id}`), id).toHaveLength(1);
        }
    });

    it("gives every action both a glyph and a label, so either can be shown", () => {
        const { container } = renderVulpes();
        const buttons = container.querySelectorAll("#confirmActionRow .confirm-action-row__btn");
        expect(buttons.length).toBe(5);
        for (const button of Array.from(buttons)) {
            expect(button.querySelector(".btn-icon")).not.toBeNull();
            expect(button.querySelector(".btn-label")).not.toBeNull();
            // The tooltip is what names the verb once the label is hidden.
            expect(button.getAttribute("title")).toBeTruthy();
        }
    });
});

describe("Basic layout reader", () => {
    it("mounts the same reader the Vulpes layout does", () => {
        localStorage.setItem("fox-reader-layout", "basic");
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        const { container } = renderBasic();
        expect(container.querySelector(".vx-reader")).not.toBeNull();
        expect(container.querySelectorAll('[aria-label="Page view"] button')).toHaveLength(4);
    });

    it("offers a reader toggle beside the folder button", () => {
        localStorage.setItem("fox-reader-layout", "basic");
        const { container } = renderBasic();
        const button = container.querySelector("#readerModeBtn");
        expect(button).not.toBeNull();
        expect(button?.closest(".folder-nav")).not.toBeNull();
        // Directly after Load, not adrift in the tool row.
        expect(button?.previousElementSibling?.id).toBe("loadFolderBtn");
    });

    it("enters reader mode when that toggle is pressed", () => {
        localStorage.setItem("fox-reader-layout", "basic");
        const { container } = renderBasic();
        expect(container.querySelector(".vx-reader")).toBeNull();
        fireEvent.click(container.querySelector("#readerModeBtn")!);
        expect(container.querySelector(".vx-reader")).not.toBeNull();
        // `viewer.ts` reads this class to know which pipeline is live.
        expect(document.body.classList.contains("workspace-reader")).toBe(true);
    });

    it("splits the translation wrapper 65/35 for entries and result", () => {
        localStorage.setItem("fox-reader-layout", "basic");
        const { container } = renderBasic();
        const panel = container.querySelector<HTMLElement>(".translation-wrapper > .panel");
        expect(panel?.style.height).toBe("65%");
        const body = container.querySelector<HTMLElement>("#translationContainer");
        // Remainder flex, never a fixed 35%: 65% + 35% + the 10px gutter would
        // overflow the wrapper by exactly that gutter. (`flexGrow`: the
        // `flex` shorthand serializes expanded as "1 1 0%".)
        expect(body?.style.height).toBe("");
        expect(body?.style.flexGrow).toBe("1");
    });

    it("renders no translation wrapper in Vulpes", () => {
        // The 70/30 split lives in `TranslationPanel`, which Vulpes never mounts.
        const { baseElement } = renderVulpes();
        expect(baseElement.querySelector(".translation-wrapper")).toBeNull();
    });

    it("keeps the action row below the translation wrapper there", () => {
        // Only Vulpes moves it to a footer; Basic has no footer, so the row is
        // the wrapper's next sibling -- out of the tinted box, not in it.
        localStorage.setItem("fox-reader-layout", "basic");
        const { container } = renderBasic();
        const row = container.querySelector("#confirmActionRow");
        expect(row?.closest(".translation-wrapper")).toBeNull();
        expect(row?.previousElementSibling?.classList.contains("translation-wrapper")).toBe(true);
    });

    it("does not render a second reader toggle in Vulpes", () => {
        // Vulpes draws its own, with a label, in the top bar.
        const { baseElement } = renderVulpes();
        expect(baseElement.querySelectorAll("#readerModeBtn")).toHaveLength(0);
        expect(baseElement.querySelector(".vx-topbar__actions .vx-icon-btn")).not.toBeNull();
    });
});

describe("Reader page-turn arrows", () => {
    function renderReader() {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        return renderVulpes();
    }

    it("draws an arrow inside each edge zone", () => {
        const { container } = renderReader();
        const zones = container.querySelectorAll(".vx-reader__zone");
        expect(zones).toHaveLength(2);
        for (const zone of Array.from(zones)) {
            expect(zone.querySelector(".vx-reader__arrow .material-icons")).not.toBeNull();
        }
    });

    it("points the arrows outward, left on the left and right on the right", () => {
        const { container } = renderReader();
        const start = container.querySelector(".vx-reader__zone--start .material-icons");
        const end = container.querySelector(".vx-reader__zone--end .material-icons");
        expect(start?.textContent).toBe("chevron_left");
        expect(end?.textContent).toBe("chevron_right");
    });

    it("labels each zone by what it does, which depends on the reading direction", () => {
        // RTL is the manga default: the left edge is *forward*.
        localStorage.setItem("fox-reader-direction", "rtl");
        const { container } = renderReader();
        expect(container.querySelector(".vx-reader__zone--start")?.getAttribute("aria-label")).toBe("Next page");
        expect(container.querySelector(".vx-reader__zone--end")?.getAttribute("aria-label")).toBe("Previous page");
    });

    it("draws no zones in the continuous view, where the gesture is a scroll", () => {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        localStorage.setItem("fox-reader-vulpes-view", JSON.stringify({ pageView: "vertical" }));
        const { container } = renderVulpes();
        expect(container.querySelector(".vx-reader__zone")).toBeNull();
    });
});

describe("Reader edge veil", () => {
    function renderReader() {
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        return renderVulpes();
    }

    it("renders the pager before the turn zones", () => {
        // `reader.css` drops the veil's live blur during a pan with
        // `.vx-pager.is-panning ~ .vx-reader__zone::before`. A general sibling
        // selector only reaches *later* siblings, so that rule stops working the
        // moment the pager moves after the zones in this tree -- silently, with no
        // type error and no failing CSS test. This is the guard for that.
        const { container } = renderReader();
        const surface = container.querySelector(".vx-reader")!;
        const children = Array.from(surface.children);
        const pager = children.findIndex((el) => el.classList.contains("vx-pager"));
        const firstZone = children.findIndex((el) => el.classList.contains("vx-reader__zone"));
        expect(pager).toBeGreaterThanOrEqual(0);
        expect(firstZone).toBeGreaterThanOrEqual(0);
        expect(pager).toBeLessThan(firstZone);
    });

    it("keeps the pager and the zones as siblings of one parent", () => {
        // The same selector also needs them to share a parent; wrapping either in a
        // div would break it just as quietly.
        const { container } = renderReader();
        const pager = container.querySelector(".vx-pager")!;
        const zones = Array.from(container.querySelectorAll(".vx-reader__zone"));
        expect(zones).toHaveLength(2);
        for (const zone of zones) expect(zone.parentElement).toBe(pager.parentElement);
    });

    it("draws no zones -- and so no veil -- in the continuous view", () => {
        // The continuous view shows the editor's stage through a transparent surface,
        // so a veil there would blur the page itself at the edges.
        localStorage.setItem("fox-reader-workspace", JSON.stringify({ mode: "reader" }));
        localStorage.setItem("fox-reader-vulpes-view", JSON.stringify({ pageView: "vertical" }));
        const { container } = renderVulpes();
        expect(container.querySelector(".vx-reader__zone")).toBeNull();
    });
});
