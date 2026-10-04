import { createContext } from "preact";
import { useState, useCallback, useContext, useEffect } from "preact/hooks";
import type { ComponentChildren } from "preact";
import { getInitialFoxConfig } from "../core/initialConfig";

export type Layout = "basic" | "default";

const STORAGE_KEY = "fox-reader-layout";

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

export function normalizeLayout(layout: unknown): Layout {
    return layout === "default" ? "default" : "basic";
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
        body.classList.remove("layout-basic", "layout-default", "layout-reader", "layout-translation");
        body.classList.add(`layout-${layout}`);
    }, [layout]);

    return <LayoutContext.Provider value={{ layout, setLayout }}>{children}</LayoutContext.Provider>;
}
