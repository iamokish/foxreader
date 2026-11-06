import { useCallback, useEffect, useRef } from "preact/hooks";
import "./splitter.css";

export type SplitterAxis = "horizontal" | "vertical";

interface SplitterProps {
    /** "vertical" = drags left/right (resizes column widths). "horizontal" = drags up/down (resizes row heights). */
    axis: SplitterAxis;
    onResize: (deltaPx: number) => void;
    onResizeEnd?: () => void;
    title?: string;
}

export function Splitter({ axis, onResize, onResizeEnd, title }: SplitterProps) {
    const finishDrag = useRef<((commit?: boolean) => void) | null>(null);

    useEffect(() => () => finishDrag.current?.(false), []);

    const onPointerDown = useCallback(
        (e: PointerEvent) => {
            const body = document.body;
            if (e.button !== 0 || finishDrag.current || body.classList.contains("ws-resizing")) return;

            // Cancel the browser's selection gesture before WebKit can extend a
            // textarea/contenteditable selection as the pointer crosses it.
            e.preventDefault();
            const handle = e.currentTarget as HTMLElement;
            const pointerId = e.pointerId;
            let position = axis === "vertical" ? e.clientX : e.clientY;
            const axisClass = `ws-resizing--${axis}`;
            body.classList.add("ws-resizing", axisClass);

            const preventSelection = (event: Event) => event.preventDefault();

            const onMove = (ev: PointerEvent) => {
                if (ev.pointerId !== pointerId) return;
                ev.preventDefault();
                const cur = axis === "vertical" ? ev.clientX : ev.clientY;
                onResize(cur - position);
                position = cur;
            };

            const finish = (commit = true) => {
                if (finishDrag.current !== finish) return;
                finishDrag.current = null;
                body.classList.remove("ws-resizing", axisClass);
                document.removeEventListener("selectstart", preventSelection, true);
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onEnd);
                window.removeEventListener("pointercancel", onEnd);
                window.removeEventListener("blur", onBlur);
                handle.removeEventListener("lostpointercapture", onEnd);
                if (handle.hasPointerCapture?.(pointerId)) handle.releasePointerCapture(pointerId);
                if (commit) onResizeEnd?.();
            };

            const onEnd = (ev: PointerEvent) => {
                if (ev.pointerId === pointerId) finish();
            };
            const onBlur = () => finish();

            finishDrag.current = finish;
            document.addEventListener("selectstart", preventSelection, true);
            window.addEventListener("pointermove", onMove, { passive: false });
            window.addEventListener("pointerup", onEnd);
            window.addEventListener("pointercancel", onEnd);
            window.addEventListener("blur", onBlur);
            handle.addEventListener("lostpointercapture", onEnd);
            handle.setPointerCapture?.(pointerId);
        },
        [axis, onResize, onResizeEnd],
    );

    return <div class={`ws-splitter ws-splitter--${axis}`} title={title} onPointerDown={onPointerDown} />;
}
