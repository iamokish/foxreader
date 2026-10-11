/**
 * Vulpes panel geometry: which sections are open and how big they are.
 *
 * Pages, editor column, captured text and the entry list each have their own size
 * under their own storage key, kept apart from `WorkspaceContext` -- which now
 * holds only the work/reader mode and the `body.workspace-reader` class the viewer
 * reads.
 *
 * ## Why a drag does not go through state
 *
 * `Splitter` reports a delta on every `pointermove`. Routing that through state
 * would re-render the whole shell -- toolbar, stage, entry list, translation panel,
 * status bar -- and write `localStorage` dozens of times a second.
 *
 * So a drag writes the size **straight onto the panel element** and nothing else.
 * State and `localStorage` are touched once, on release (`commitResize`).
 * Visibility still goes through state, because it changes what is rendered rather
 * than how big it is.
 *
 * ## Why not a custom property on <body>
 *
 * That was the first cut. A custom property is inherited, so changing one on
 * `<body>` makes the browser re-resolve every dependent on every pointer move;
 * measured head-to-head against that design at full load (200 thumbnails, 100
 * cards, live refit) both hold 60fps with no long tasks either way. Direct
 * `el.style` writes are kept anyway: per-frame work becomes one element write
 * plus the relayout the drag needs, with no inherited state to reason about.
 *
 * Both halves of that matter: the write is also coalesced into one
 * `requestAnimationFrame`, because a high-polling-rate mouse delivers
 * `pointermove` at 120-240Hz and there is no point recomputing layout more often
 * than the screen can show it.
 *
 * The committed size is *also* rendered as an inline style from JSX, so the first
 * paint is correct and a stored size never flashes at the default. Preact only
 * rewrites it when the committed state changes -- on release -- and by then the
 * value it writes is the one the drag already wrote, so the two never fight.
 */

import { createContext } from "preact";
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { ComponentChildren } from "preact";

/** Which sections of the Vulpes shell are open, and how big they are. */
export interface PanelState {
    visible: boolean;
    /** Pixels along the panel's resizable axis: width for the columns, height for the rows. */
    size: number;
}

/**
 * pages      the thumbnail rail, width
 * editor     the right-hand column, width
 * ocr        the captured-text box under the stage, height
 * entries    the entry list inside the editor, height
 */
export type VxPanelId = "pages" | "editor" | "ocr" | "entries";

export const VX_PANEL_IDS: readonly VxPanelId[] = ["pages", "editor", "ocr", "entries"] as const;

/** The axis each panel is sized along, so a drag writes the right property. */
export const VX_PANEL_AXIS: Record<VxPanelId, "width" | "height"> = {
    pages: "width",
    editor: "width",
    ocr: "height",
    entries: "height",
};

/** Drag limits per section, in CSS px. A width's max is also bounded by the window. */
export const VX_PANEL_LIMITS: Record<VxPanelId, { min: number; max: number }> = {
    pages: { min: 96, max: 420 },
    editor: { min: 260, max: 720 },
    ocr: { min: 72, max: 480 },
    entries: { min: 90, max: 900 },
};

export const VX_DEFAULT_PANELS: Record<VxPanelId, PanelState> = {
    pages: { visible: true, size: 164 },
    editor: { visible: true, size: 360 },
    ocr: { visible: true, size: 150 },
    entries: { visible: true, size: 300 },
};

const STORAGE_KEY = "fox-reader-vulpes-panels";

function clampSize(id: VxPanelId, size: number): number {
    const { min, max } = VX_PANEL_LIMITS[id];
    // A width can never claim the whole window: leave room for the stage, whatever
    // the stored maximum says.
    const ceiling =
        VX_PANEL_AXIS[id] === "width"
            ? Math.max(min, Math.min(max, (typeof window === "undefined" ? 1280 : window.innerWidth) - 320))
            : max;
    if (!Number.isFinite(size)) return VX_DEFAULT_PANELS[id].size;
    return Math.round(Math.max(min, Math.min(ceiling, size)));
}

function loadPanels(): Record<VxPanelId, PanelState> {
    const panels = { ...VX_DEFAULT_PANELS };
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return panels;
        const parsed = JSON.parse(raw) as Partial<Record<VxPanelId, Partial<PanelState>>>;
        for (const id of VX_PANEL_IDS) {
            const stored = parsed?.[id];
            if (!stored) continue;
            panels[id] = {
                visible: typeof stored.visible === "boolean" ? stored.visible : panels[id].visible,
                // Clamped on the way in: a window that was wider last session, or a
                // hand-edited value, must not leave a section off screen with no
                // splitter left to drag it back.
                size: clampSize(id, Number(stored.size)),
            };
        }
        // The entry list cannot collapse (its header has no toggle), so a
        // `visible: false` stored by an older build must not hide it on load.
        // Replaced, not mutated: `panels` shares its untouched rows with
        // `VX_DEFAULT_PANELS` by reference.
        panels.entries = { ...panels.entries, visible: true };
    } catch {
        /* a locked-down profile, or stored JSON from another build */
    }
    return panels;
}

interface VulpesPanelsValue {
    getPanel: (id: VxPanelId) => PanelState;
    /** Hand the provider the element a drag should resize. `null` on unmount. */
    registerPanel: (id: VxPanelId, el: HTMLElement | null) => void;
    /** Live-resize during a drag. Writes the element only; call `commitResize` on release. */
    resizePanel: (id: VxPanelId, delta: number) => void;
    /** Persist the sizes a drag arrived at. */
    commitResize: () => void;
    togglePanel: (id: VxPanelId) => void;
    setPanelVisible: (id: VxPanelId, visible: boolean) => void;
    resetPanels: () => void;
}

const VulpesPanelsContext = createContext<VulpesPanelsValue>({
    getPanel: (id) => VX_DEFAULT_PANELS[id],
    registerPanel: () => {},
    resizePanel: () => {},
    commitResize: () => {},
    togglePanel: () => {},
    setPanelVisible: () => {},
    resetPanels: () => {},
});

export function useVulpesPanels(): VulpesPanelsValue {
    return useContext(VulpesPanelsContext);
}

export function VulpesPanelsProvider({ children }: { children: ComponentChildren }) {
    const [panels, setPanels] = useState<Record<VxPanelId, PanelState>>(loadPanels);

    /** The live sizes a drag is moving, ahead of the committed state. */
    const live = useRef<Record<VxPanelId, number>>({
        pages: panels.pages.size,
        editor: panels.editor.size,
        ocr: panels.ocr.size,
        entries: panels.entries.size,
    });
    /** The elements a drag writes to, by panel. */
    const elements = useRef<Partial<Record<VxPanelId, HTMLElement>>>({});
    /** Panels whose size has moved since the last paint. */
    const dirty = useRef<Set<VxPanelId>>(new Set());
    const frame = useRef(0);

    const persist = useCallback((next: Record<VxPanelId, PanelState>) => {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
        } catch {}
        return next;
    }, []);

    const registerPanel = useCallback((id: VxPanelId, el: HTMLElement | null) => {
        if (el) elements.current[id] = el;
        else delete elements.current[id];
    }, []);

    /** Write one panel's live size to its element. The whole of what a drag costs. */
    const paint = useCallback((id: VxPanelId) => {
        const el = elements.current[id];
        if (el) el.style[VX_PANEL_AXIS[id]] = `${live.current[id]}px`;
    }, []);

    /** Flush every dirty panel now, and drop any frame that was waiting to do it. */
    const flush = useCallback(() => {
        if (frame.current) {
            cancelAnimationFrame(frame.current);
            frame.current = 0;
        }
        for (const id of dirty.current) paint(id);
        dirty.current.clear();
    }, [paint]);

    const resizePanel = useCallback(
        (id: VxPanelId, delta: number) => {
            const next = clampSize(id, live.current[id] + delta);
            if (next === live.current[id]) return;
            live.current[id] = next;
            dirty.current.add(id);
            // One layout per frame, however fast the mouse reports.
            if (frame.current) return;
            frame.current = requestAnimationFrame(() => {
                frame.current = 0;
                for (const pending of dirty.current) paint(pending);
                dirty.current.clear();
            });
        },
        [paint],
    );

    const commitResize = useCallback(() => {
        // Land the final position before the state update, so the element never
        // shows a frame of the previous size.
        flush();
        setPanels((prev) => {
            let changed = false;
            const next = { ...prev };
            for (const id of VX_PANEL_IDS) {
                if (prev[id].size === live.current[id]) continue;
                next[id] = { ...prev[id], size: live.current[id] };
                changed = true;
            }
            return changed ? persist(next) : prev;
        });
    }, [flush, persist]);

    // A committed change -- a reset, a stored value, a visibility toggle -- reaches
    // the element the same way a drag does. Preact also writes the inline style from
    // JSX; this keeps `live` in step so the next drag starts from the right place.
    useEffect(() => {
        for (const id of VX_PANEL_IDS) {
            live.current[id] = panels[id].size;
            const el = elements.current[id];
            if (el) el.style[VX_PANEL_AXIS[id]] = `${panels[id].size}px`;
        }
    }, [panels]);

    useEffect(
        () => () => {
            if (frame.current) cancelAnimationFrame(frame.current);
        },
        [],
    );

    const togglePanel = useCallback(
        (id: VxPanelId) => {
            setPanels((prev) => persist({ ...prev, [id]: { ...prev[id], visible: !prev[id].visible } }));
        },
        [persist],
    );

    const setPanelVisible = useCallback(
        (id: VxPanelId, visible: boolean) => {
            setPanels((prev) =>
                prev[id].visible === visible ? prev : persist({ ...prev, [id]: { ...prev[id], visible } }),
            );
        },
        [persist],
    );

    const resetPanels = useCallback(() => {
        setPanels(() => persist({ ...VX_DEFAULT_PANELS }));
    }, [persist]);

    // The imperative escape hatch the keyboard handler in `main.tsx` uses for `H`
    // (hide/show the page rail).
    useEffect(() => {
        (window as unknown as { vulpesTogglePanel?: (id: VxPanelId) => void }).vulpesTogglePanel = togglePanel;
        return () => {
            delete (window as unknown as { vulpesTogglePanel?: (id: VxPanelId) => void }).vulpesTogglePanel;
        };
    }, [togglePanel]);

    const value = useMemo<VulpesPanelsValue>(
        () => ({
            getPanel: (id) => panels[id] ?? VX_DEFAULT_PANELS[id],
            registerPanel,
            resizePanel,
            commitResize,
            togglePanel,
            setPanelVisible,
            resetPanels,
        }),
        [panels, registerPanel, resizePanel, commitResize, togglePanel, setPanelVisible, resetPanels],
    );

    return <VulpesPanelsContext.Provider value={value}>{children}</VulpesPanelsContext.Provider>;
}
