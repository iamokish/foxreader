/**
 * Workspace mode: whether the app is in the editing workspace or the reader.
 *
 * This is the whole of what the retired `WorkspaceContext` still did. It used to
 * carry a fixed set of panel sizes belonging to the Default layout as well; those
 * are gone with that layout, and Vulpes keeps its own in
 * `vulpes/VulpesPanelsContext`. What is left is the one piece of state every layout
 * shares, and the body class the viewer reads to know which pipeline is live
 * (`isReaderMode` in `viewer.ts`).
 *
 * It sits in `src/workspace/` rather than `src/reader/` because it is not the
 * reader's: Basic uses it without a reader at all, and the reader is only one of
 * the two modes it names.
 */

import { createContext } from "preact";
import { useCallback, useContext, useEffect, useMemo, useState } from "preact/hooks";
import type { ComponentChildren } from "preact";

export type WorkspaceMode = "work" | "reader";

const STORAGE_KEY = "fox-reader-workspace";

function loadMode(): WorkspaceMode {
    if (typeof localStorage === "undefined") return "work";
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return "work";
        const parsed = JSON.parse(raw) as { mode?: unknown };
        // The stored object used to hold `panels` too. Reading only `mode` means an
        // existing profile keeps its choice and the stale sizes are ignored.
        return parsed?.mode === "reader" ? "reader" : "work";
    } catch {
        return "work";
    }
}

interface WorkspaceValue {
    mode: WorkspaceMode;
    setMode: (mode: WorkspaceMode) => void;
}

const WorkspaceContext = createContext<WorkspaceValue>({ mode: "work", setMode: () => {} });

export function useWorkspace(): WorkspaceValue {
    return useContext(WorkspaceContext);
}

export function WorkspaceProvider({ children }: { children: ComponentChildren }) {
    const [mode, setModeState] = useState<WorkspaceMode>(loadMode);

    const setMode = useCallback((next: WorkspaceMode) => {
        setModeState(next);
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify({ mode: next }));
        } catch {}
    }, []);

    useEffect(() => {
        document.body.classList.toggle("workspace-reader", mode === "reader");
    }, [mode]);

    // The imperative escape hatch `main.tsx`'s Escape binding uses.
    useEffect(() => {
        (window as unknown as { setWorkspaceMode?: (mode: WorkspaceMode) => void }).setWorkspaceMode = setMode;
        return () => {
            delete (window as unknown as { setWorkspaceMode?: (mode: WorkspaceMode) => void }).setWorkspaceMode;
        };
    }, [setMode]);

    const value = useMemo<WorkspaceValue>(() => ({ mode, setMode }), [mode, setMode]);

    return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}
