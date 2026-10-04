import { fireEvent, render } from "@testing-library/preact";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppearanceModal } from "./AppearanceModal";
import { LayoutProvider } from "../../layouts/LayoutContext";
import { ThemeProvider } from "../../themes/ThemeContext";
import { Toolbar } from "../Toolbar/Toolbar";

function renderModal(open = true, onClose = vi.fn()) {
    const utils = render(
        <LayoutProvider>
            <ThemeProvider>
                <AppearanceModal open={open} onClose={onClose} />
            </ThemeProvider>
        </LayoutProvider>,
    );
    return { ...utils, onClose };
}

beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove("theme-dark", "theme-light");
    document.body.style.overflow = "";
    vi.restoreAllMocks();
});

describe("AppearanceModal", () => {
    it("renders nothing when closed", () => {
        const { container } = renderModal(false);
        expect(container.querySelector("#appearanceOverlay")).toBeNull();
    });

    it("renders a dialog with layout and theme pickers when open", () => {
        const { container } = renderModal(true);
        const dialog = container.querySelector('[role="dialog"]');
        expect(dialog).not.toBeNull();
        expect(dialog?.getAttribute("aria-label")).toBe("Appearance settings");

        const layouts = container.querySelectorAll('[role="radiogroup"][aria-label="Layout"] [role="radio"]');
        const themes = container.querySelectorAll('[role="radiogroup"][aria-label="Theme"] [role="radio"]');
        expect(layouts.length).toBe(2);
        expect(themes.length).toBe(2);
        expect(container.querySelector('[data-layout="basic"]')).not.toBeNull();
        expect(container.querySelector('[data-layout="default"]')).not.toBeNull();
        expect(container.querySelector('[data-theme="light"]')).not.toBeNull();
        expect(container.querySelector('[data-theme="dark"]')).not.toBeNull();
    });

    it("marks the active layout and theme as checked", () => {
        localStorage.setItem("fox-reader-layout", "default");
        localStorage.setItem("fox-reader-theme", "light");
        const { container } = renderModal(true);
        expect(container.querySelector('[data-layout="default"]')?.getAttribute("aria-checked")).toBe("true");
        expect(container.querySelector('[data-layout="basic"]')?.getAttribute("aria-checked")).toBe("false");
        expect(container.querySelector('[data-theme="light"]')?.getAttribute("aria-checked")).toBe("true");
        expect(container.querySelector('[data-theme="dark"]')?.getAttribute("aria-checked")).toBe("false");
    });

    it("switches theme live without closing", () => {
        const onClose = vi.fn();
        const { container } = renderModal(true, onClose);
        fireEvent.click(container.querySelector('[data-theme="light"]')!);
        expect(localStorage.getItem("fox-reader-theme")).toBe("light");
        expect(document.documentElement.classList.contains("theme-light")).toBe(true);
        expect(onClose).not.toHaveBeenCalled();
        expect(container.querySelector("#appearanceOverlay")).not.toBeNull();
    });

    it("persists the picked layout (the app reload follows in the browser)", () => {
        const { container } = renderModal(true);
        fireEvent.click(container.querySelector('[data-layout="default"]')!);
        expect(localStorage.getItem("fox-reader-layout")).toBe("default");
    });

    it("closes on Escape", () => {
        const { onClose } = renderModal(true);
        fireEvent.keyDown(document, { key: "Escape" });
        expect(onClose).toHaveBeenCalledTimes(1);
    });

    it("closes on backdrop click but not on card click", () => {
        const { container, onClose } = renderModal(true);
        fireEvent.click(container.querySelector(".appearance-card")!);
        expect(onClose).not.toHaveBeenCalled();
        fireEvent.click(container.querySelector("#appearanceOverlay")!);
        expect(onClose).toHaveBeenCalledTimes(1);
    });

    it("closes via the close button", () => {
        const { container, onClose } = renderModal(true);
        fireEvent.click(container.querySelector("#appearanceCloseBtn")!);
        expect(onClose).toHaveBeenCalledTimes(1);
    });
});

describe("Toolbar appearance entry", () => {
    it("shows an Appearance row with current values instead of standalone switchers", () => {
        const { container } = render(
            <LayoutProvider>
                <ThemeProvider>
                    <Toolbar />
                </ThemeProvider>
            </LayoutProvider>,
        );
        expect(container.querySelector("#layoutSwitcherBtn")).toBeNull();
        expect(container.querySelector("button.theme-toggle")).toBeNull();

        const btn = container.querySelector("#appearanceSettingsBtn");
        expect(btn).not.toBeNull();
        expect(btn?.textContent).toContain("Appearance");
        expect(btn?.textContent).toContain("Basic");
    });

    it("opens the overlay when the Appearance row is clicked", () => {
        const { container } = render(
            <LayoutProvider>
                <ThemeProvider>
                    <Toolbar />
                </ThemeProvider>
            </LayoutProvider>,
        );
        expect(container.querySelector("#appearanceOverlay")).toBeNull();
        fireEvent.click(container.querySelector("#appearanceSettingsBtn")!);
        expect(container.querySelector("#appearanceOverlay")).not.toBeNull();
    });
});
