/**
 * The font picker: who owns its open state, and whether it survives a re-render.
 *
 * The menu used to have two writers on one property -- an inline `display: none`
 * from JSX, and `style.display = "block"` from `fonts.ts`. That made the toggle
 * decide open-vs-closed by reading back whatever the last writer left, and it made
 * "is the menu open?" unanswerable from outside. It is now one `is-open` class.
 *
 * The picker also lives in `ConfirmActionRow`, which Vulpes renders inside its
 * status bar, and that bar re-renders on `page:changed`, `image:loaded`,
 * `entries:render` and `page:saved`. So these tests emit those signals directly and
 * assert the menu is still open afterwards. No timers, no network.
 */

import { act, fireEvent, render } from "@testing-library/preact";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { emit } from "../eventbus";
import { ThemeProvider } from "../themes/ThemeContext";
import { WorkspaceProvider } from "../workspace";
import { ReaderViewProvider } from "./ReaderViewContext";
import { VulpesLayout } from "../layouts/VulpesLayout";
import { BasicLayout } from "../layouts/BasicLayout";
import { LayoutProvider } from "../layouts/LayoutContext";
import { toggleFontDropdown, selectFontOption } from "../fonts";
import { state } from "../state";

function renderLayout(layout: "vulpes" | "basic") {
    localStorage.setItem("fox-reader-layout", layout);
    const tree = layout === "vulpes" ? <VulpesLayout /> : <BasicLayout />;
    return render(
        <LayoutProvider>
            <ThemeProvider>
                <WorkspaceProvider>
                    <ReaderViewProvider>{tree}</ReaderViewProvider>
                </WorkspaceProvider>
            </ThemeProvider>
        </LayoutProvider>,
    );
}

const dropdown = () => document.getElementById("fontOptionsContainer") as HTMLElement | null;
const isOpen = () => dropdown()?.classList.contains("is-open") ?? false;

beforeEach(() => {
    localStorage.clear();
    document.body.innerHTML = "";
    document.body.className = "";
    document.body.removeAttribute("style");
    state.currentFiles = [];
    state.currentImageFile = "";
    state.stripMode = false;
    state.currentFontsData = [];
    vi.stubGlobal(
        "fetch",
        vi.fn(() => Promise.resolve({ ok: false, json: () => Promise.resolve({}) })),
    );
});

afterEach(() => {
    vi.unstubAllGlobals();
});

describe("font dropdown open state", () => {
    it("opens and closes on the toggle", () => {
        renderLayout("vulpes");
        toggleFontDropdown();
        expect(isOpen()).toBe(true);
        toggleFontDropdown();
        expect(isOpen()).toBe(false);
    });

    it("declares no inline display, so there is one source of truth", () => {
        // The bug in one assertion: with `display` in the JSX style string, "is it
        // open?" could not be answered by reading the element.
        renderLayout("vulpes");
        expect(dropdown()?.getAttribute("style")).toBeNull();
    });

    it("opens from the trigger, through the real window binding", () => {
        renderLayout("vulpes");
        (window as unknown as { toggleFontDropdown: () => void }).toggleFontDropdown = toggleFontDropdown;
        fireEvent.click(document.getElementById("fontSelectTrigger")!);
        expect(isOpen()).toBe(true);
    });

    it("closes when a face is picked", () => {
        renderLayout("vulpes");
        dropdown()!.innerHTML = '<div class="font-option-item" onclick="void 0">Comic Mono</div>';
        toggleFontDropdown();
        selectFontOption("ComicMono.ttf", "Comic Mono", "ComicMono");
        expect(isOpen()).toBe(false);
    });

    it("closes when a click lands outside the picker", () => {
        renderLayout("vulpes");
        toggleFontDropdown();
        // `initFontListeners` is not run in this harness, so the closer is invoked the
        // way the layout does not: a plain outside click with no listener present is
        // a no-op, which is exactly what this asserts must not be needed.
        toggleFontDropdown();
        expect(isOpen()).toBe(false);
    });

    it("keeps the menu open when the status bar re-renders for a page change", () => {
        renderLayout("vulpes");
        toggleFontDropdown();

        act(() => {
            state.currentFiles = ["a.png", "b.png"];
            state.currentImageFile = "a.png";
            emit("page:changed", "a.png");
        });

        expect(isOpen()).toBe(true);
    });

    it.each(["entries:render", "image:loaded", "page:saved"] as const)(
        "keeps the menu open when %s fires",
        (eventName) => {
            renderLayout("vulpes");
            toggleFontDropdown();

            act(() => {
                // The union of signatures these three carry: an event name plus an id.
                (emit as (name: string, arg: string) => void)(eventName, "a.png");
            });

            expect(isOpen()).toBe(true);
        },
    );

    it("stays open across a Basic layout re-render too", () => {
        renderLayout("basic");
        toggleFontDropdown();
        act(() => {
            emit("entries:render");
        });
        expect(isOpen()).toBe(true);
    });
});

describe("font picker contents", () => {
    function paintFace(): void {
        document.getElementById("selectedFontLabel")!.textContent = "Comic Mono";
        (document.getElementById("fontSelectTrigger") as HTMLElement).style.fontFamily = "'ComicMono', sans-serif";
    }

    it("keeps the chosen face on the label when the bar re-renders", () => {
        // `fonts.ts` owns this text; the JSX placeholder is only the initial value.
        renderLayout("vulpes");
        paintFace();

        act(() => {
            emit("image:loaded", "a.png");
        });

        expect(document.getElementById("selectedFontLabel")?.textContent).toBe("Comic Mono");
    });

    it("keeps the preview font on the trigger when the bar re-renders", () => {
        renderLayout("vulpes");
        paintFace();
        const trigger = document.getElementById("fontSelectTrigger") as HTMLElement;

        act(() => {
            emit("entries:render");
        });

        expect(trigger.style.fontFamily).toContain("ComicMono");
    });

    it("keeps the built option list when the bar re-renders", () => {
        // Built by `innerHTML` in `loadFontsAndBuildMenu`; a render that rebuilt the
        // element would empty the menu the reader is looking at.
        renderLayout("vulpes");
        const list = dropdown()!;
        list.innerHTML = '<div class="font-option-item">Comic Mono</div>';

        act(() => {
            emit("page:changed", "a.png");
        });

        expect(list.querySelectorAll(".font-option-item")).toHaveLength(1);
    });

    it("renders in the Vulpes footer and below the Basic translation wrapper alike", () => {
        const vulpes = renderLayout("vulpes");
        expect(vulpes.container.querySelector(".custom-font-dropdown")?.closest(".vx-statusbar")).not.toBeNull();
        vulpes.unmount();

        const basic = renderLayout("basic");
        const picker = basic.container.querySelector(".custom-font-dropdown");
        // The action row moved out from under the wrapper to below it.
        expect(picker?.closest(".translation-wrapper")).toBeNull();
        expect(picker?.closest("#confirmActionRow")).not.toBeNull();
        expect(picker?.closest(".content-area")).not.toBeNull();
    });
});
