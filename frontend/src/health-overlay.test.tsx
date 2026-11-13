import { readFileSync } from "node:fs";
import { render } from "@testing-library/preact";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
    disposeBackendStatusOverlay, hideBackendStatus, initBackendStatusOverlay,
    installPageHealth, isBackendStatusVisible, showBackendOffline, showBackendRestarted,
} from "../static/health-overlay.js";
import { BackendStatusOverlay } from "./components/common/BackendStatusOverlay";
import { AppearanceModal } from "./components/Appearance/AppearanceModal";

const response = (session_id = "s1") => ({ ok: true, json: async () => ({ session_id }) } as Response);
const settle = () => vi.advanceTimersByTimeAsync(0);
const dialog = () => document.getElementById("backendStatusOverlay") as HTMLDialogElement;
const action = () => document.getElementById("backendStatusActionBtn") as HTMLButtonElement;
const originalShowModal = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, "showModal");
const originalClose = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, "close");

describe("shared backend health dialog", () => {
    let stop: (() => void) | undefined;
    let unmount: (() => void) | undefined;
    let stack: HTMLDialogElement[];
    beforeEach(() => {
        vi.useFakeTimers();
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response()));
        window.history.replaceState({}, "", "/settings");
        document.body.innerHTML = '<input id="draft" value="unsaved settings">';
        stack = [];
        // jsdom has no top layer. Model modal order and focus restoration here;
        // the browser's actual showModal/close API must be used by the UI.
        const focus = new WeakMap<HTMLDialogElement, HTMLElement>();
        if (!originalShowModal) Object.defineProperty(HTMLDialogElement.prototype, "showModal", { configurable: true, value() {} });
        if (!originalClose) Object.defineProperty(HTMLDialogElement.prototype, "close", { configurable: true, value() {} });
        vi.spyOn(HTMLDialogElement.prototype, "showModal").mockImplementation(function (this: HTMLDialogElement) {
            expect(this.isConnected).toBe(true);
            if (this.open) return;
            focus.set(this, document.activeElement as HTMLElement);
            stack.push(this);
            this.open = true;
            this.querySelector<HTMLElement>("[autofocus], button, input")?.focus();
        });
        vi.spyOn(HTMLDialogElement.prototype, "close").mockImplementation(function (this: HTMLDialogElement) {
            this.open = false;
            stack = stack.filter((item) => item !== this);
            focus.get(this)?.focus();
        });
    });
    afterEach(() => {
        vi.unstubAllGlobals();
        stop?.();
        stop = undefined;
        unmount?.();
        unmount = undefined;
        disposeBackendStatusOverlay();
        document.body.innerHTML = "";
        document.getElementById("foxHealthStyles")?.remove();
        window.history.replaceState({}, "", "/");
        vi.restoreAllMocks();
        if (!originalShowModal) Reflect.deleteProperty(HTMLDialogElement.prototype, "showModal");
        if (!originalClose) Reflect.deleteProperty(HTMLDialogElement.prototype, "close");
        vi.useRealTimers();
    });

    it.each(["/setup", "/settings", "/user_endpoints", "/user_endpoints/"])(
        "%s installs once, retries offline and restores the unsaved form", async (path) => {
            window.history.replaceState({}, "", path);
            vi.mocked(fetch).mockRejectedValueOnce(new Error("offline"));
            const draft = document.getElementById("draft") as HTMLInputElement;
            draft.focus();
            stop = installPageHealth();
            expect(installPageHealth()).toBe(stop);
            await settle();
            expect(fetch).toHaveBeenCalledOnce();
            expect(dialog().open).toBe(true);
            expect(action().textContent).toBe("Retry");
            expect(document.activeElement).toBe(action());
            expect(dialog()).toHaveAccessibleName("Backend offline");
            expect(dialog()).toHaveAccessibleDescription(/Could not reach/);
            expect(document.querySelectorAll("#foxHealthStyles")).toHaveLength(1);

            action().click();
            await settle();
            expect(vi.mocked(fetch).mock.calls[1][0]).toBe("/api/health");
            expect(dialog().open).toBe(false);
            expect(document.getElementById("draft")).toBe(draft);
            expect(draft.value).toBe("unsaved settings");
            expect(document.activeElement).toBe(draft);
        },
    );

    it("keeps the health modal above an open editor and later asynchronous result dialogs", async () => {
        document.body.insertAdjacentHTML("beforeend", '<dialog id="editor"><input value="unsaved endpoint"></dialog><dialog id="result"><button>OK</button></dialog>');
        const editor = document.getElementById("editor") as HTMLDialogElement;
        const result = document.getElementById("result") as HTMLDialogElement;
        editor.showModal();
        const input = editor.querySelector("input")!;
        stop = installPageHealth();
        showBackendOffline(vi.fn());
        expect(stack.at(-1)).toBe(dialog());
        result.showModal();
        await settle();
        expect(stack.at(-1)).toBe(dialog());
        expect(editor.open).toBe(true);
        expect(result.open).toBe(true);
        expect(editor.querySelector("input")).toBe(input);
        expect(input.value).toBe("unsaved endpoint");
        const cancel = new Event("cancel", { cancelable: true });
        dialog().dispatchEvent(cancel);
        expect(cancel.defaultPrevented).toBe(true);
        dialog().click();
        expect(dialog().open).toBe(true);
        hideBackendStatus();
        expect(stack.at(-1)).toBe(result);
    });

    it("refreshes only on an explicit Refresh click and never retries a stale callback", () => {
        stop = installPageHealth();
        const retry = vi.fn();
        const reload = vi.fn();
        showBackendOffline(retry);
        initBackendStatusOverlay();
        initBackendStatusOverlay();
        action().click();
        expect(retry).toHaveBeenCalledOnce();
        vi.stubGlobal("window", { location: { reload } });
        showBackendRestarted();
        expect(isBackendStatusVisible()).toBe(true);
        expect(action().textContent).toBe("Refresh");
        expect(document.getElementById("backendStatusMessage")).toHaveTextContent("Unsaved changes will be lost");
        expect(reload).not.toHaveBeenCalled();
        action().click();
        expect(reload).toHaveBeenCalledOnce();
        expect(retry).toHaveBeenCalledOnce();
        vi.unstubAllGlobals();
    });

    it.each([true, false])("keeps setup gate keyboard handling underneath the health dialog (licence open: %s)", (termsOpen) => {
        // Exercise the template's real document-level handler without starting
        // its unrelated model downloads, storage checks or websocket.
        const template = readFileSync("templates/setup.html", "utf8");
        const source = template.slice(template.indexOf("function onGateKeydown(event)"), template.indexOf("function openGate(mode, trigger)"));
        expect(source).toContain("settleTerms(false)");
        const trapTab = vi.fn();
        const decline = vi.fn();
        const close = vi.fn();
        const onGateKeydown = new Function("termsOpen", "gateMode", "els", "trapTab", "settleTerms", "closeGate",
            source + "\nreturn onGateKeydown;")(termsOpen, "review", {}, trapTab, decline, close);
        document.addEventListener("keydown", onGateKeydown);
        try {
            stop = installPageHealth();
            showBackendOffline(vi.fn());
            for (const key of ["Tab", "Escape"]) {
                const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
                action().dispatchEvent(event);
                // Let the native dialog perform its own keyboard defaults.
                expect(event.defaultPrevented).toBe(false);
            }
            expect(trapTab).not.toHaveBeenCalled();
            expect(decline).not.toHaveBeenCalled();
            expect(close).not.toHaveBeenCalled();
            hideBackendStatus();
            const draft = document.getElementById("draft")!;
            draft.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true }));
            draft.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
            expect(trapTab).toHaveBeenCalledOnce();
            expect(termsOpen ? decline : close).toHaveBeenCalledOnce();
        } finally {
            document.removeEventListener("keydown", onGateKeydown);
        }
    });

    it("keeps Escape from closing the reader's underlying Appearance modal", () => {
        const close = vi.fn();
        unmount = render(<AppearanceModal open onClose={close} />).unmount;
        stop = installPageHealth();
        showBackendOffline(vi.fn());
        action().dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
        expect(close).not.toHaveBeenCalled();
        hideBackendStatus();
        document.getElementById("appearanceCloseBtn")!.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
        expect(close).toHaveBeenCalledOnce();
    });

    it("binds the reader component before DOMContentLoaded and shares the same modal UI", () => {
        window.history.replaceState({}, "", "/");
        unmount = render(<BackendStatusOverlay />).unmount;
        const retry = vi.fn();
        showBackendOffline(retry);
        expect(dialog().tagName).toBe("DIALOG");
        expect(dialog()).toHaveClass("fox-backend-status");
        expect(isBackendStatusVisible()).toBe(true);
        expect(dialog()).toHaveAccessibleName("Backend offline");
        action().click();
        expect(retry).toHaveBeenCalledOnce();
        hideBackendStatus();
        expect(isBackendStatusVisible()).toBe(false);
    });

    it.each(["/", "/index", "/unrelated"])("does not start a second reader monitor on %s", (path) => {
        window.history.replaceState({}, "", path);
        expect(installPageHealth()).toBeUndefined();
        expect(fetch).not.toHaveBeenCalled();
        expect(dialog()).toBeNull();
    });

    it.each(['<main class="session-lock-screen"></main>', '<main class="lock"></main>'])(
        "leaves an existing duplicate-session lock screen alone: %s", (markup) => {
            document.body.innerHTML = markup;
            expect(installPageHealth()).toBeUndefined();
            expect(fetch).not.toHaveBeenCalled();
            expect(document.body.innerHTML).toBe(markup);
        },
    );

    it("stops pending work when the session lock replaces the body and does not recreate the overlay", async () => {
        let resolve!: (value: Response) => void;
        vi.mocked(fetch).mockReturnValueOnce(new Promise((done) => { resolve = done; }));
        stop = installPageHealth();
        const signal = vi.mocked(fetch).mock.calls[0][1]!.signal!;
        document.body.innerHTML = '<main class="lock">Fox Reader is already open</main>';
        await settle();
        expect(signal.aborted).toBe(true);
        resolve(response("s2"));
        await vi.advanceTimersByTimeAsync(60_000);
        expect(fetch).toHaveBeenCalledOnce();
        expect(dialog()).toBeNull();
        expect(document.body.textContent).toBe("Fox Reader is already open");
        expect(vi.getTimerCount()).toBe(0);
    });

    it("disposes action handlers and top-layer observation", async () => {
        stop = installPageHealth();
        const retry = vi.fn();
        showBackendOffline(retry);
        const button = action();
        stop!();
        button.click();
        document.body.insertAdjacentHTML("beforeend", '<dialog id="other"><button>OK</button></dialog>');
        (document.getElementById("other") as HTMLDialogElement).showModal();
        await settle();
        expect(retry).not.toHaveBeenCalled();
        expect(dialog()).toBeNull();
        expect(vi.getTimerCount()).toBe(0);
    });

    it("wires the shared module through every existing template entry", () => {
        const entry = readFileSync("static/desktop-entry.js", "utf8");
        expect(entry).toContain('import { installPageHealth } from "./health-overlay.js"');
        expect(entry).toContain("installPageHealth();");
        for (const name of ["setup", "settings", "user_endpoints"]) {
            const template = readFileSync(`templates/${name}.html`, "utf8");
            expect(template).toContain('src="/static/desktop-entry.js"');
            expect(template).toContain("fox_reader_session_lock");
        }
    });
});
