import { createContext } from "preact";
import { useState, useCallback, useContext, useEffect } from "preact/hooks";
import type { ComponentChildren } from "preact";
import { getInitialFoxConfig } from "../core/initialConfig";

export type Layout = "basic" | "vulpes";

const STORAGE_KEY = "fox-reader-layout";

/** Every layout the body can be in, for the class bookkeeping below. */
const LAYOUT_CLASSES = ["layout-basic", "layout-vulpes", "layout-reader", "layout-translation"];

interface LayoutContextValue {
    layout: Layout;
    setLayout: (layout: Layout) => void;
}

const LayoutContext = createContext<LayoutContextValue>({
    layout: "basic",
    setLayout: () => {},
});

export function useLayout(): LayoutContextValue {
    return useContext(LayoutContext);
}

/**
 * Coerce a stored or server-supplied value into a layout that exists.
 *
 * `default` maps to Vulpes: it *was* the resizable workspace with a reader, and
 * Vulpes is its successor, so anyone sitting on it keeps the kind of layout they
 * chose rather than being dropped into the legacy split view. Everything else
 * unrecognised -- the older `reader` and `translation` layouts, a typo in
 * `fox_config.yaml`, a value from a newer build -- falls back to Basic rather than
 * leaving the app with nothing to render.
 */
export function normalizeLayout(layout: unknown): Layout {
    if (layout === "vulpes" || layout === "default") return "vulpes";
    return "basic";
}

function getInitialLayout(): Layout {
    try {
        const stored = normalizeLayout(localStorage.getItem(STORAGE_KEY));
        if (localStorage.getItem(STORAGE_KEY)) return stored;
    } catch {}

    return normalizeLayout(getInitialFoxConfig().layout);
}

export function LayoutProvider({ children }: { children: ComponentChildren }) {
    const [layout, setLayoutState] = useState<Layout>(getInitialLayout);

    const setLayout = useCallback((newLayout: Layout) => {
        setLayoutState(newLayout);
        try {
            localStorage.setItem(STORAGE_KEY, newLayout);
        } catch {}
        window.location.reload();
    }, []);

    useEffect(() => {
        const body = document.body;
        body.classList.remove(...LAYOUT_CLASSES);
        body.classList.add(`layout-${layout}`);
    }, [layout]);

    return <LayoutContext.Provider value={{ layout, setLayout }}>{children}</LayoutContext.Provider>;
}
