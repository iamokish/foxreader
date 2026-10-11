import { fireEvent, render } from "@testing-library/preact";
import { beforeEach, describe, expect, it } from "vitest";
import { Toolbar } from "./Toolbar";
import { LayoutProvider } from "../../layouts/LayoutContext";
import { ThemeProvider } from "../../themes/ThemeContext";
import { state } from "../../state";
import type { Region } from "../../types";

function rectRegion(x: number, y: number, w = 40, h = 40): Region {
    return { type: "rectangle", coords: { x, y, w, h } };
}

function renderToolbar() {
    return render(
        <LayoutProvider>
            <ThemeProvider>
                <Toolbar />
            </ThemeProvider>
        </LayoutProvider>,
    );
}

beforeEach(() => {
    localStorage.clear();
    state.pageEntriesCache = {};
    state.currentImageFile = "";
    state.currentTrackingColorIdx = 0;
    document.body.innerHTML = "";
});

describe("Toolbar reading direction", () => {
    it("offers both directions with RTL active by default", () => {
        const { container } = renderToolbar();
        const ltr = container.querySelector("#readingDirLtrBtn");
        const rtl = container.querySelector("#readingDirRtlBtn");
        expect(ltr).not.toBeNull();
        expect(rtl).not.toBeNull();
        expect(rtl?.getAttribute("aria-pressed")).toBe("true");
        expect(ltr?.getAttribute("aria-pressed")).toBe("false");
        expect(rtl?.classList.contains("is-active")).toBe(true);
    });

    it("switching direction persists and resorts the page", async () => {
        state.currentImageFile = "p.png";
        const { addNewEntries } = await import("../../entries");
        // RTL append order: rightmost first.
        addNewEntries([rectRegion(10, 10), rectRegion(200, 10)]);
        const xs = () => state.pageEntriesCache["p.png"].map((e) => (e.region.coords as { x: number }).x);
        expect(xs()).toEqual([200, 10]);

        const { container } = renderToolbar();
        fireEvent.click(container.querySelector("#readingDirLtrBtn")!);

        expect(localStorage.getItem("fox-reader-direction")).toBe("ltr");
        expect(xs()).toEqual([10, 200]);
        expect(container.querySelector("#readingDirLtrBtn")?.getAttribute("aria-pressed")).toBe("true");
    });

    it("clicking the active direction does nothing", async () => {
        state.currentImageFile = "p.png";
        const { addNewEntries } = await import("../../entries");
        addNewEntries([rectRegion(10, 10)]);
        const before = state.pageEntriesCache["p.png"].map((e) => e.id);

        const { container } = renderToolbar();
        fireEvent.click(container.querySelector("#readingDirRtlBtn")!);

        expect(localStorage.getItem("fox-reader-direction")).toBeNull();
        expect(state.pageEntriesCache["p.png"].map((e) => e.id)).toEqual(before);
    });
});

describe("Toolbar folder row", () => {
    it("reads Load on the folder button with the reader toggle right after it", () => {
        localStorage.setItem("fox-reader-layout", "basic");
        const { container } = renderToolbar();
        const nav = container.querySelector(".folder-nav");
        expect(nav).not.toBeNull();
        const load = container.querySelector("#loadFolderBtn");
        expect(load?.textContent).toMatch(/Load/);
        expect(load?.textContent).not.toMatch(/Load Folder/);
        expect(load?.querySelector(".material-icons")?.textContent).toBe("folder_open");
        const reader = container.querySelector("#readerModeBtn");
        expect(reader?.closest(".folder-nav")).not.toBeNull();
        expect(reader?.previousElementSibling?.id).toBe("loadFolderBtn");
        // Icon-only: glyph plus accessible name, no visible words.
        expect(reader?.querySelector(".material-icons")).not.toBeNull();
        expect(reader?.textContent).not.toMatch(/Reader/);
        expect(reader?.getAttribute("aria-label")).toBe("Reader");
    });
});

describe("Toolbar characters", () => {
    it("offers an icon-only Characters button next to Options and an enabled toggle", () => {
        const { container } = renderToolbar();
        const options = container.querySelector("#settingsBtn");
        const button = container.querySelector("#charactersBtn");
        expect(options).not.toBeNull();
        expect(button).not.toBeNull();
        expect(button?.querySelector(".material-icons")?.textContent).toBe("group");
        expect(button?.textContent).not.toMatch(/Characters/);
        expect(button?.getAttribute("aria-label")).toBe("Characters");
        const toggle = container.querySelector("#enableCharacters") as HTMLInputElement | null;
        expect(toggle).not.toBeNull();
        expect(toggle?.checked).toBe(true);
    });

    it("opening the Characters button shows the roster modal", () => {
        const { container } = renderToolbar();
        fireEvent.click(container.querySelector("#charactersBtn")!);
        expect(document.body.querySelector(".ch-modal")).not.toBeNull();
    });

    it("offers a bubble padding slider from 0 to 10, defaulting to 5px", () => {
        const { container } = renderToolbar();
        const slider = container.querySelector("#bubblePadding") as HTMLInputElement | null;
        expect(slider).not.toBeNull();
        expect(slider?.getAttribute("min")).toBe("0");
        expect(slider?.getAttribute("max")).toBe("10");
        expect(slider?.getAttribute("step")).toBe("1");
        expect(slider?.value).toBe("5");
        expect(slider?.getAttribute("aria-label")).toBeTruthy();
        expect(container.querySelector("#bubblePaddingValue")?.textContent?.trim()).toBe("5px");
    });
});
