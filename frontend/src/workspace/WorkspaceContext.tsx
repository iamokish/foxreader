import { createContext } from "preact";
import { useState, useCallback, useContext, useEffect } from "preact/hooks";
import type { ComponentChildren } from "preact";

export type WorkspaceMode = "work" | "reader";

export type PanelId = "viewer" | "gallery" | "ocr" | "entries" | "translation";

export interface PanelState {
    visible: boolean;
    /** Pixel size of the panel's resizable dimension (width for viewer, height otherwise). */
    size: number;
}

interface WorkspaceState {
    mode: WorkspaceMode;
    panels: Record<PanelId, PanelState>;
}

const STORAGE_KEY = "fox-reader-workspace";

const DEFAULT_PANELS: Record<PanelId, PanelState> = {
    viewer: { visible: true, size: 640 },
    gallery: { visible: true, size: 110 },
    ocr: { visible: true, size: 170 },
    entries: { visible: true, size: 230 },
    translation: { visible: true, size: 230 },
};

export const PANEL_IDS: PanelId[] = ["viewer", "gallery", "ocr", "entries", "translation"];

export function resizePanelState<T extends Record<string, PanelState>>(
    panels: T,
    id: keyof T,
    delta: number,
    min: number,
    max: number,
): T {
    const panel = panels[id];
    const size = Math.max(min, Math.min(max, panel.size + delta));
    return { ...panels, [id]: { ...panel, size } };
}

function loadWorkspaceState(): WorkspaceState {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw) as Partial<WorkspaceState>;
            const panels = { ...DEFAULT_PANELS };
            if (parsed.panels) {
                for (const id of PANEL_IDS) {
                    const p = parsed.panels[id];
                    if (p && typeof p.visible === "boolean" && typeof p.size === "number") {
                        panels[id] = { visible: p.visible, size: p.size };
                    }
                }
            }
            const mode: WorkspaceMode = parsed.mode === "reader" ? "reader" : "work";
            return { mode, panels };
        }
    } catch {}
    return { mode: "work", panels: { ...DEFAULT_PANELS } };
}

interface WorkspaceContextValue {
    mode: WorkspaceMode;
    setMode: (mode: WorkspaceMode) => void;
    getPanel: (id: PanelId) => PanelState;
    setPanelSize: (id: PanelId, size: number) => void;
    resizePanel: (id: PanelId, delta: number, min: number, max: number) => void;
    setPanelVisible: (id: PanelId, visible: boolean) => void;
    togglePanel: (id: PanelId) => void;
}

const WorkspaceContext = createContext<WorkspaceContextValue>({
    mode: "work",
    setMode: () => {},
    getPanel: () => DEFAULT_PANELS.viewer,
    setPanelSize: () => {},
    resizePanel: () => {},
    setPanelVisible: () => {},
    togglePanel: () => {},
});

export function useWorkspace(): WorkspaceContextValue {
    return useContext(WorkspaceContext);
}

export function WorkspaceProvider({ children }: { children: ComponentChildren }) {
    const [state, setState] = useState<WorkspaceState>(loadWorkspaceState);

    const persist = useCallback((next: WorkspaceState) => {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
        } catch {}
    }, []);

    const setMode = useCallback(
        (mode: WorkspaceMode) => {
            setState((prev) => {
                const next = { ...prev, mode };
                persist(next);
                return next;
            });
        },
        [persist],
    );

    const setPanelSize = useCallback(
        (id: PanelId, size: number) => {
            setState((prev) => {
                const next = {
                    ...prev,
                    panels: { ...prev.panels, [id]: { ...prev.panels[id], size } },
                };
                persist(next);
                return next;
            });
        },
        [persist],
    );

    const resizePanel = useCallback(
        (id: PanelId, delta: number, min: number, max: number) => {
            setState((prev) => {
                const next = { ...prev, panels: resizePanelState(prev.panels, id, delta, min, max) };
                persist(next);
                return next;
            });
        },
        [persist],
    );

    const setPanelVisible = useCallback(
        (id: PanelId, visible: boolean) => {
            setState((prev) => {
                const next = {
                    ...prev,
                    panels: { ...prev.panels, [id]: { ...prev.panels[id], visible } },
                };
                persist(next);
                return next;
            });
        },
        [persist],
    );

    const togglePanel = useCallback(
        (id: PanelId) => {
            setState((prev) => {
                const next = {
                    ...prev,
                    panels: {
                        ...prev.panels,
                        [id]: { ...prev.panels[id], visible: !prev.panels[id].visible },
                    },
                };
                persist(next);
                return next;
            });
        },
        [persist],
    );

    useEffect(() => {
        document.body.classList.toggle("workspace-reader", state.mode === "reader");
    }, [state.mode]);

    // Expose an imperative escape hatch (keydown handlers in main.tsx etc.)
    useEffect(() => {
        (window as any).setWorkspaceMode = setMode;
        (window as any).togglePanel = togglePanel;
        return () => {
            delete (window as any).setWorkspaceMode;
            delete (window as any).togglePanel;
        };
    }, [setMode, togglePanel]);

    const value: WorkspaceContextValue = {
        mode: state.mode,
        setMode,
        getPanel: (id) => state.panels[id] ?? DEFAULT_PANELS[id],
        setPanelSize,
        resizePanel,
        setPanelVisible,
        togglePanel,
    };

    return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}
