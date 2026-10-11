import { beforeEach, describe, expect, it, vi } from "vitest";
import { folderCrumbs } from "./VulpesTopBar";
import { setHorizontalFit } from "./VulpesStatusBar";
import { state } from "../state";

describe("folderCrumbs", () => {
    it("is series / chapter from the last two segments", () => {
        expect(folderCrumbs("D:\\Comics\\Some Series\\Volume 3")).toEqual(["Some Series", "Volume 3"]);
    });

    it("handles POSIX separators too", () => {
        expect(folderCrumbs("/home/me/Comics/Series/Ch 12")).toEqual(["Series", "Ch 12"]);
    });

    it("ignores a trailing separator", () => {
        expect(folderCrumbs("D:\\Comics\\Series\\Ch 1\\")).toEqual(["Series", "Ch 1"]);
    });

    it("drops a bare drive letter, which names nothing", () => {
        expect(folderCrumbs("D:\\Comics")).toEqual(["Comics"]);
    });

    it("is a single crumb for a one-segment path", () => {
        expect(folderCrumbs("Comics")).toEqual(["Comics"]);
    });

    it("is empty with no folder, so the bar can show a placeholder", () => {
        expect(folderCrumbs("")).toEqual([]);
        expect(folderCrumbs("\\\\")).toEqual([]);
    });
});

describe("setHorizontalFit", () => {
    beforeEach(() => {
        document.body.innerHTML = "";
        state.isHorizontalFitEnabled = false;
    });

    it("drives the Options switch, so its handler does the work exactly once", () => {
        // `main.tsx` owns the `change` listener that shows the toast and re-fits the
        // page; poking `state` directly would skip both and desync the checkbox.
        document.body.innerHTML = `<input type="checkbox" id="horizontalFit">`;
        const box = document.getElementById("horizontalFit") as HTMLInputElement;
        const onChange = vi.fn();
        box.addEventListener("change", onChange);

        setHorizontalFit(true);

        expect(box.checked).toBe(true);
        expect(onChange).toHaveBeenCalledTimes(1);
    });

    it("does nothing when the switch is already where it is wanted", () => {
        document.body.innerHTML = `<input type="checkbox" id="horizontalFit" checked>`;
        const box = document.getElementById("horizontalFit") as HTMLInputElement;
        const onChange = vi.fn();
        box.addEventListener("change", onChange);

        setHorizontalFit(true);

        expect(onChange).not.toHaveBeenCalled();
    });

    it("falls back to the state flag when the dropdown is not mounted", () => {
        setHorizontalFit(true);
        expect(state.isHorizontalFitEnabled).toBe(true);
    });
});
