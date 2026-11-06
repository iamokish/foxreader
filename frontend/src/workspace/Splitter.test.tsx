import { cleanup, render } from "@testing-library/preact";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Splitter, type SplitterAxis } from "./Splitter";

function pointer(target: EventTarget, type: string, init: PointerEventInit = {}): PointerEvent {
    const event = new PointerEvent(type, {
        bubbles: true,
        cancelable: true,
        pointerId: 7,
        isPrimary: true,
        button: 0,
        buttons: 1,
        ...init,
    });
    target.dispatchEvent(event);
    return event;
}

function selectionStart(target: EventTarget): Event {
    const event = new Event("selectstart", { bubbles: true, cancelable: true });
    target.dispatchEvent(event);
    return event;
}

function fixture(axis: SplitterAxis = "vertical") {
    const onResize = vi.fn();
    const onResizeEnd = vi.fn();
    const view = render(
        <>
            <Splitter axis={axis} onResize={onResize} onResizeEnd={onResizeEnd} />
            <textarea />
            <div contentEditable>Captured words</div>
        </>,
    );
    const handle = view.container.querySelector<HTMLElement>(".ws-splitter")!;
    const textarea = view.container.querySelector("textarea")!;
    const editable = view.container.querySelector<HTMLElement>("[contenteditable]")!;
    // jsdom has PointerEvent but no pointer capture implementation.
    let captured: number | null = null;
    handle.setPointerCapture = vi.fn((id: number) => {
        captured = id;
    });
    handle.hasPointerCapture = vi.fn((id: number) => captured === id);
    handle.releasePointerCapture = vi.fn(() => {
        captured = null;
    });
    return { ...view, handle, textarea, editable, onResize, onResizeEnd };
}

afterEach(() => {
    cleanup();
    document.body.className = "";
    document.body.removeAttribute("style");
    window.getSelection()?.removeAllRanges();
});

describe("Splitter drag selection", () => {
    it.each(["vertical", "horizontal"] as const)("resizes on the %s axis without starting a selection", (axis) => {
        const { handle, onResize, onResizeEnd } = fixture(axis);
        const down = pointer(handle, "pointerdown", { clientX: 10, clientY: 100 });
        const move = pointer(window, "pointermove", { clientX: 30, clientY: 130 });
        pointer(window, "pointermove", { clientX: 25, clientY: 120 });

        expect(down.defaultPrevented).toBe(true);
        expect(move.defaultPrevented).toBe(true);
        expect(handle.setPointerCapture).toHaveBeenCalledWith(7);
        expect(document.body.classList.contains(`ws-resizing--${axis}`)).toBe(true);
        expect(onResize.mock.calls).toEqual(axis === "vertical" ? [[20], [-5]] : [[30], [-10]]);
        expect(onResizeEnd).not.toHaveBeenCalled();

        pointer(window, "pointerup");
        expect(onResizeEnd).toHaveBeenCalledOnce();
        expect(handle.releasePointerCapture).toHaveBeenCalledWith(7);
    });

    it("blocks selection only during a drag and preserves existing text selections and body styles", () => {
        const { handle, textarea, editable } = fixture();
        document.body.className = "layout-vulpes existing-class";
        document.body.style.cssText = "user-select: text; cursor: crosshair; color: red;";
        const originalStyle = document.body.getAttribute("style");
        textarea.value = "Captured words";
        textarea.setSelectionRange(2, 8);
        const range = document.createRange();
        range.selectNodeContents(editable);
        window.getSelection()!.addRange(range);

        for (const target of [textarea, editable]) expect(selectionStart(target).defaultPrevented).toBe(false);
        pointer(handle, "pointerdown");
        for (const target of [textarea, editable]) expect(selectionStart(target).defaultPrevented).toBe(true);
        pointer(window, "pointermove", { clientX: 20 });
        pointer(window, "pointerup");

        for (const target of [textarea, editable]) expect(selectionStart(target).defaultPrevented).toBe(false);
        expect([textarea.selectionStart, textarea.selectionEnd]).toEqual([2, 8]);
        expect(window.getSelection()!.toString()).toBe("Captured words");
        expect(document.body.className).toBe("layout-vulpes existing-class");
        expect(document.body.getAttribute("style")).toBe(originalStyle);
    });

    it.each(["pointerup", "pointercancel", "lostpointercapture", "blur", "unmount"])(
        "restores selection and removes drag handlers after %s",
        (ending) => {
            const { handle, textarea, onResize, onResizeEnd, unmount } = fixture();
            pointer(handle, "pointerdown");
            pointer(window, "pointermove", { clientX: 12 });
            if (ending === "unmount") unmount();
            else if (ending === "blur") window.dispatchEvent(new Event("blur"));
            else pointer(ending === "lostpointercapture" ? handle : window, ending);

            expect(document.body.classList.contains("ws-resizing")).toBe(false);
            expect(document.body.classList.contains("ws-resizing--vertical")).toBe(false);
            expect(handle.releasePointerCapture).toHaveBeenCalledWith(7);
            expect(selectionStart(document.body).defaultPrevented).toBe(false);
            if (ending !== "unmount") expect(selectionStart(textarea).defaultPrevented).toBe(false);
            pointer(window, "pointermove", { clientX: 30 });
            pointer(window, "pointerup");
            window.dispatchEvent(new Event("blur"));
            expect(onResize.mock.calls).toEqual([[12]]);
            expect(onResizeEnd).toHaveBeenCalledTimes(ending === "unmount" ? 0 : 1);
        },
    );

    it("ignores other mouse buttons and unrelated pointers during a drag", () => {
        const { handle, onResize, onResizeEnd } = fixture();
        expect(pointer(handle, "pointerdown", { button: 2 }).defaultPrevented).toBe(false);
        expect(handle.setPointerCapture).not.toHaveBeenCalled();
        expect(document.body.classList.contains("ws-resizing")).toBe(false);

        pointer(handle, "pointerdown");
        expect(pointer(window, "pointermove", { pointerId: 8, clientX: 100 }).defaultPrevented).toBe(false);
        pointer(window, "pointerup", { pointerId: 8 });
        pointer(handle, "lostpointercapture", { pointerId: 8 });
        expect(onResize).not.toHaveBeenCalled();
        expect(onResizeEnd).not.toHaveBeenCalled();
        expect(document.body.classList.contains("ws-resizing")).toBe(true);
        pointer(window, "pointermove", { clientX: 10 });
        pointer(window, "pointerup");
        expect(onResize.mock.calls).toEqual([[10]]);
        expect(onResizeEnd).toHaveBeenCalledOnce();
    });

    it("allows only one splitter to own the drag and permits another after release", () => {
        const first = fixture();
        const second = fixture("horizontal");
        pointer(first.handle, "pointerdown");
        pointer(second.handle, "pointerdown", { pointerId: 8 });
        expect(second.handle.setPointerCapture).not.toHaveBeenCalled();
        pointer(window, "pointerup");

        pointer(second.handle, "pointerdown", { pointerId: 8 });
        pointer(window, "pointermove", { pointerId: 8, clientY: 20 });
        pointer(window, "pointerup", { pointerId: 8 });
        expect(first.onResizeEnd).toHaveBeenCalledOnce();
        expect(second.onResize.mock.calls).toEqual([[20]]);
        expect(second.onResizeEnd).toHaveBeenCalledOnce();
    });
});
