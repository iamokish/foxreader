import type { RefObject } from "preact";
import { useLayoutEffect } from "preact/hooks";

/** Keep room for the capture toolbar before allowing the Editor to grow. */
export function useEditorWidthLimit(
    bodyRef: RefObject<HTMLDivElement>,
    enabled: boolean,
    setLimit: (width: number) => void,
): void {
    useLayoutEffect(() => {
        const body = bodyRef.current;
        if (!enabled || !body) return;
        const canvas = body.querySelector<HTMLElement>(".vx-canvas");
        const editor = body.querySelector<HTMLElement>(".vx-editor");
        const tools = body.querySelector<HTMLElement>(".vx-tools");
        const buttons = tools?.querySelector<HTMLElement>(".buttons");
        const pages = body.querySelector<HTMLElement>(".vx-rail");
        if (!canvas || !editor || !tools || !buttons) return;

        const measure = () => {
            const buttonsWidth = buttons.getBoundingClientRect().width;
            const editorWidth = editor.getBoundingClientRect().width;
            // Reader mode and hidden/unmounted panels have no usable geometry.
            if (buttonsWidth <= 0 || editorWidth <= 0 || body.getBoundingClientRect().width <= 0) return;
            const style = getComputedStyle(tools);
            const horizontalInsets = [
                style.paddingLeft,
                style.paddingRight,
                style.borderLeftWidth,
                style.borderRightWidth,
            ].reduce((total, value) => total + (parseFloat(value) || 0), 0);
            const required = Math.ceil(buttonsWidth + horizontalInsets);
            // This sum excludes the actual Pages rail and splitters, including
            // the compact rail on small windows. It does not depend on the
            // current Editor width, so applying the cap cannot create a loop.
            const available = canvas.getBoundingClientRect().width + editorWidth;
            setLimit(Math.floor(available - required));
        };

        measure();
        let frame = 0;
        const scheduleMeasure = () => {
            if (frame) return;
            frame = requestAnimationFrame(() => {
                frame = 0;
                measure();
            });
        };
        // The row has intrinsic width: font loads and optional buttons resize it,
        // whereas dragging the Editor does not. Measurements stay out of the
        // pointermove path. Defer observer writes to one frame to avoid a resize
        // notification loop when the cap changes the toolbar's scrollbar/height.
        const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(scheduleMeasure);
        for (const el of [body, pages, buttons]) {
            if (el) observer?.observe(el);
        }
        if (!observer) window.addEventListener("resize", scheduleMeasure);
        return () => {
            observer?.disconnect();
            if (frame) cancelAnimationFrame(frame);
            if (!observer) window.removeEventListener("resize", scheduleMeasure);
        };
    }, [bodyRef, enabled, setLimit]);
}
