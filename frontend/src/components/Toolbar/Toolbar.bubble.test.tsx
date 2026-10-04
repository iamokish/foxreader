import { render } from "@testing-library/preact";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LayoutProvider } from "../../layouts/LayoutContext";
import { ThemeProvider } from "../../themes/ThemeContext";

/**
 * The Bubble Capture button follows the bubble model.
 *
 * The weights are optional now -- only the three PaddleOCR models are required
 * to run Fox Reader -- so the backend reports whether it built a bubble service
 * (`bubble_available` in routes/folder.py, mirrored into `window.__FOX_CONFIG__`
 * by index.html) and both renderers of this button gate on it. The static
 * markup half is covered by tests/manual/render_index_page.py; this is the
 * Preact half.
 *
 * Its own file because the component reads the flag once at module load, which
 * means switching the flag means dropping the module registry -- and a
 * `vi.resetModules()` in Toolbar.test.tsx would hand that file's later tests a
 * second copy of `state`, which is not the one its top-level import holds.
 */
async function renderWithConfig(config: Window["__FOX_CONFIG__"]) {
    vi.resetModules();

    if (config === undefined) delete window.__FOX_CONFIG__;
    else window.__FOX_CONFIG__ = config;

    const { Toolbar } = await import("./Toolbar");

    return render(
        <LayoutProvider>
            <ThemeProvider>
                <Toolbar />
            </ThemeProvider>
        </LayoutProvider>,
    );
}

afterEach(() => {
    delete window.__FOX_CONFIG__;
    localStorage.clear();
    document.body.innerHTML = "";
});

describe("Toolbar bubble capture", () => {
    it("offers the button when the backend loaded the weights", async () => {
        const { container } = await renderWithConfig({ bubbleAvailable: true });

        expect(container.querySelector("#bubCapture")).not.toBeNull();
        // The other capture modes do not depend on the model, so a missing
        // bubble button must not be read as a missing toolbar.
        expect(container.querySelector("#rectCapture")).not.toBeNull();
        expect(container.querySelector("#freeCapture")).not.toBeNull();
    });

    it("leaves it out entirely when the model is not downloaded", async () => {
        const { container } = await renderWithConfig({ bubbleAvailable: false });

        // Left out rather than disabled: there is nothing behind it without the
        // weights, and a disabled button only invites a click.
        expect(container.querySelector("#bubCapture")).toBeNull();
        expect(container.querySelector("#rectCapture")).not.toBeNull();
        expect(container.querySelector("#freeCapture")).not.toBeNull();
    });

    it("keeps the button when the flag is missing", async () => {
        // A page cached from before the flag existed. Losing a working button
        // to a stale cache is worse than showing one the backend already
        // guards, so absent reads as available.
        const { container } = await renderWithConfig({});

        expect(container.querySelector("#bubCapture")).not.toBeNull();
    });

    it("and when there is no config at all", async () => {
        const { container } = await renderWithConfig(undefined);

        expect(container.querySelector("#bubCapture")).not.toBeNull();
    });
});
