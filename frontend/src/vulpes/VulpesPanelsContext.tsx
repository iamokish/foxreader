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
import type { VerticalGeometry } from "./verticalGeometry";

/** Which sections of the Vulpes shell are open, and how big they are. */
export interface PanelState {
    visible: boolean;
    /** Pixels along the panel's resizable axis: width for the columns, height for the rows. */
    size: number;
    /** Entries only: the Translator's share of the Editor, authoritative over legacy pixels. */
    translationFraction?: number;
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

/** Drag limits per section, in CSS px. Column maxima also depend on available space. */
export const VX_PANEL_LIMITS: Record<VxPanelId, { min: number; max: number }> = {
    pages: { min: 96, max: 160 },
    editor: { min: 260, max: 720 },
    // Row bounds are measured from their containers; these only sanitize saved pixels.
    ocr: { min: 72, max: Number.MAX_SAFE_INTEGER },
    entries: { min: 0, max: Number.MAX_SAFE_INTEGER },
};

export const VX_TRANSLATION_FRACTIONS = { min: 0.15, max: 0.7, default: 0.4 } as const;

export const VX_DEFAULT_PANELS: Record<VxPanelId, PanelState> = {
    pages: { visible: true, size: 100 },
    editor: { visible: true, size: 360 },
    ocr: { visible: true, size: 150 },
    entries: { visible: true, size: 300, translationFraction: VX_TRANSLATION_FRACTIONS.default },
};

const STORAGE_KEY = "fox-reader-vulpes-panels";

function clampTranslationFraction(value: number): number {
    return Math.max(VX_TRANSLATION_FRACTIONS.min, Math.min(VX_TRANSLATION_FRACTIONS.max, value));
}

/** One sizing path for JSX, window changes, resets and live splitter movement. */
function renderedSize(id: VxPanelId, size: number, fraction: number | undefined, geometry: VerticalGeometry): number {
    if (id === "entries" && geometry.editor) {
        const { height, splitter } = geometry.editor;
        const min = Math.max(0, Math.ceil(height - height * VX_TRANSLATION_FRACTIONS.max - splitter));
        const max = Math.max(min, Math.floor(height - height * VX_TRANSLATION_FRACTIONS.min - splitter));
        const preferred = fraction === undefined ? size : height - height * fraction - splitter;
        return Math.max(min, Math.min(max, Math.round(preferred)));
    }
    if (id === "ocr" && geometry.capture) {
        const min = Math.ceil(geometry.capture.minimum);
        // A complete text line takes priority when a very short window cannot fit both limits.
        const max = Math.max(min, Math.floor(geometry.capture.available * 0.4));
        return Math.max(min, Math.min(max, Math.round(size)));
    }
    return size;
}

function clampSize(id: VxPanelId, size: number): number {
    const { min, max } = VX_PANEL_LIMITS[id];
    // Pages retain a window bound. The Editor uses its measured toolbar limit
    // separately, so loading in a small window preserves a wider saved preference.
    const ceiling =
        id === "pages"
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
            if (id === "entries") {
                const fraction = stored.translationFraction;
                if (typeof fraction === "number" && Number.isFinite(fraction)) {
                    panels.entries.translationFraction = clampTranslationFraction(fraction);
                } else if (!Number.isFinite(Number(stored.size))) {
                    panels.entries.translationFraction = VX_TRANSLATION_FRACTIONS.default;
                }
                // Legacy pixel preferences acquire a fraction once visible geometry is known.
            }
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
    /** Effective size after applying measured row limits, in CSS pixels. */
    getPanelSize: (id: VxPanelId) => number;
    /** Hand the provider the element a drag should resize. `null` on unmount. */
    registerPanel: (id: VxPanelId, el: HTMLElement | null) => void;
    /** Reserve the measured toolbar width without overwriting the user's preferred size. */
    setEditorWidthLimit: (width: number) => void;
    setVerticalGeometry: (geometry: VerticalGeometry) => void;
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
    getPanelSize: (id) => VX_DEFAULT_PANELS[id].size,
    registerPanel: () => {},
    setEditorWidthLimit: () => {},
    setVerticalGeometry: () => {},
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
    const editorWidthLimit = useRef(VX_PANEL_LIMITS.editor.max);
    const verticalGeometry = useRef<VerticalGeometry>({});
    const translationFraction = useRef(panels.entries.translationFraction);
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
        if (el) {
            elements.current[id] = el;
            if (id === "editor") el.style.maxWidth = `${editorWidthLimit.current}px`;
        } else delete elements.current[id];
    }, []);

    const setEditorWidthLimit = useCallback((width: number) => {
        if (!Number.isFinite(width)) return;
        const { min, max } = VX_PANEL_LIMITS.editor;
        const next = Math.max(min, Math.min(max, Math.floor(width)));
        if (next === editorWidthLimit.current) return;
        editorWidthLimit.current = next;
        // Keep the preferred width for a larger window. A separate maximum also
        // constrains JSX/reset writes, without rerendering or saving on resize.
        const el = elements.current.editor;
        if (el) el.style.maxWidth = `${next}px`;
    }, []);

    /** Write one panel's live size to its element. The whole of what a drag costs. */
    const paint = useCallback((id: VxPanelId) => {
        const el = elements.current[id];
        if (el) {
            const size = renderedSize(id, live.current[id], translationFraction.current, verticalGeometry.current);
            const value = `${size}px`;
            if (el.style[VX_PANEL_AXIS[id]] !== value) el.style[VX_PANEL_AXIS[id]] = value;
        }
    }, []);

    const setVerticalGeometry = useCallback(
        (geometry: VerticalGeometry) => {
            const previous = verticalGeometry.current;
            let changed = false;
            if (geometry.editor) {
                const { height, splitter } = geometry.editor;
                if (
                    Number.isFinite(height) &&
                    height > 0 &&
                    Number.isFinite(splitter) &&
                    splitter >= 0 &&
                    splitter < height &&
                    (previous.editor?.height !== height || previous.editor?.splitter !== splitter)
                ) {
                    previous.editor = { height, splitter };
                    changed = true;
                }
            }
            if (geometry.capture) {
                const { available, minimum } = geometry.capture;
                if (
                    Number.isFinite(available) &&
                    available > 0 &&
                    Number.isFinite(minimum) &&
                    minimum > 0 &&
                    (previous.capture?.available !== available || previous.capture?.minimum !== minimum)
                ) {
                    previous.capture = { available, minimum };
                    changed = true;
                }
            }
            if (!changed) return;
            if (previous.editor && translationFraction.current === undefined) {
                const { height, splitter } = previous.editor;
                const size = renderedSize("entries", live.current.entries, undefined, previous);
                const fraction = clampTranslationFraction((height - splitter - size) / height);
                translationFraction.current = fraction;
                // Migrate once in memory. Window resizing never writes preferences.
                setPanels((current) => ({
                    ...current,
                    entries: { ...current.entries, translationFraction: fraction },
                }));
            }
            paint("entries");
            paint("ocr");
        },
        [paint],
    );

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
            const limit = id === "editor" ? editorWidthLimit.current : Infinity;
            // Start from the displayed width, even when a smaller window has
            // temporarily constrained a wider saved preference.
            const current = Math.min(
                limit,
                renderedSize(id, live.current[id], translationFraction.current, verticalGeometry.current),
            );
            const next = Math.min(
                limit,
                renderedSize(id, clampSize(id, current + delta), undefined, verticalGeometry.current),
            );
            if (next === current) return;
            live.current[id] = next;
            if (id === "entries" && verticalGeometry.current.editor) {
                const { height, splitter } = verticalGeometry.current.editor;
                translationFraction.current = clampTranslationFraction((height - splitter - next) / height);
            }
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
                if (
                    prev[id].size === live.current[id] &&
                    (id !== "entries" || prev.entries.translationFraction === translationFraction.current)
                )
                    continue;
                next[id] = { ...prev[id], size: live.current[id] };
                if (id === "entries") next.entries.translationFraction = translationFraction.current;
                changed = true;
            }
            return changed ? persist(next) : prev;
        });
    }, [flush, persist]);

    // A committed change -- a reset, a stored value, a visibility toggle -- reaches
    // the element the same way a drag does. Preact also writes the inline style from
    // JSX; this keeps `live` in step so the next drag starts from the right place.
    useEffect(() => {
        translationFraction.current = panels.entries.translationFraction;
        for (const id of VX_PANEL_IDS) {
            live.current[id] = panels[id].size;
            paint(id);
        }
    }, [panels, paint]);

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
            getPanelSize: (id) =>
                renderedSize(id, panels[id].size, panels.entries.translationFraction, verticalGeometry.current),
            registerPanel,
            setEditorWidthLimit,
            setVerticalGeometry,
            resizePanel,
            commitResize,
            togglePanel,
            setPanelVisible,
            resetPanels,
        }),
        [
            panels,
            registerPanel,
            setEditorWidthLimit,
            setVerticalGeometry,
            resizePanel,
            commitResize,
            togglePanel,
            setPanelVisible,
            resetPanels,
        ],
    );

    return <VulpesPanelsContext.Provider value={value}>{children}</VulpesPanelsContext.Provider>;
}
