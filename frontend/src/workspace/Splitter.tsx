import { useCallback, useRef } from "preact/hooks";

export type SplitterAxis = "horizontal" | "vertical";

interface SplitterProps {
    /** "vertical" = drags left/right (resizes column widths). "horizontal" = drags up/down (resizes row heights). */
    axis: SplitterAxis;
    onResize: (deltaPx: number) => void;
    onResizeEnd?: () => void;
    title?: string;
}

export function Splitter({ axis, onResize, onResizeEnd, title }: SplitterProps) {
    const startPos = useRef(0);
    const dragging = useRef(false);

    const onPointerDown = useCallback(
        (e: PointerEvent) => {
            dragging.current = true;
            startPos.current = axis === "vertical" ? e.clientX : e.clientY;
            document.body.style.userSelect = "none";
            document.body.style.cursor = axis === "vertical" ? "col-resize" : "row-resize";

            const onMove = (ev: PointerEvent) => {
                if (!dragging.current) return;
                const cur = axis === "vertical" ? ev.clientX : ev.clientY;
                onResize(cur - startPos.current);
                startPos.current = cur;
            };

            const onUp = () => {
                dragging.current = false;
                document.body.style.userSelect = "";
                document.body.style.cursor = "";
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                onResizeEnd?.();
            };

            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
        },
        [axis, onResize, onResizeEnd],
    );

    return <div class={`ws-splitter ws-splitter--${axis}`} title={title} onPointerDown={onPointerDown} />;
}
