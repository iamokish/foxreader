/**
 * The reader's view state, and the one place that owns the strip renderer.
 *
 * Three jobs:
 *
 *   1. hold and persist the {@link ViewSettings} (view, fit, gaps, pairing rules)
 *   2. publish the gaps as CSS custom properties, so the strip -- which is built
 *      imperatively and knows nothing about Preact -- picks them up
 *   3. build and tear down the strip exactly when the continuous view is the one
 *      on screen
 *
 * The third job used to belong to the retired Default layout, which drove it from
 * `state.stripOverride` -- a tri-state its floating Strip button toggled. The
 * reader has a four-way view picker instead, so the continuous view *is* the
 * override, and keeping the tri-state as well would be a second and contradictory
 * source of truth for the same question.
 */

import { createContext } from "preact";
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { ComponentChildren } from "preact";
import { state } from "../state";
import { emit, on, off } from "../eventbus";
import { buildStrip, destroyStrip, refitViewer } from "../viewer";
import { useWorkspace } from "../workspace";
import {
    DEFAULT_VIEW_SETTINGS,
    loadViewSettings,
    saveViewSettings,
    type FitMode,
    type PageView,
    type ViewSettings,
} from "./viewSettings";

interface ReaderViewValue {
    settings: ViewSettings;
    /** Pick a view. Counts as an explicit choice, so webtoon auto-detection stands down. */
    setPageView: (view: PageView) => void;
    setFit: (fit: FitMode) => void;
    /** Patch any subset of the settings without marking the view as hand-picked. */
    update: (patch: Partial<ViewSettings>) => void;
    /** Bumped whenever the pager should re-read the page list or re-fit. */
    revision: number;
}

const ReaderViewContext = createContext<ReaderViewValue>({
    settings: { ...DEFAULT_VIEW_SETTINGS },
    setPageView: () => {},
    setFit: () => {},
    update: () => {},
    revision: 0,
});

export function useReaderView(): ReaderViewValue {
    return useContext(ReaderViewContext);
}

export function ReaderViewProvider({ children }: { children: ComponentChildren }) {
    const { mode } = useWorkspace();
    const [settings, setSettings] = useState<ViewSettings>(loadViewSettings);
    const [revision, setRevision] = useState(0);
    /** Bumped once per folder load, so the strip is rebuilt for the new pages. */
    const [folderToken, setFolderToken] = useState(0);

    const update = useCallback((patch: Partial<ViewSettings>) => {
        setSettings((prev) => {
            const next = { ...prev, ...patch };
            saveViewSettings(next);
            return next;
        });
    }, []);

    const setPageView = useCallback((view: PageView) => {
        setSettings((prev) => {
            // `pinned` is what stops a folder of tall pages overriding the choice
            // on the next load. Picking a view by hand is the moment it is set.
            const next = { ...prev, pageView: view, pinned: true };
            saveViewSettings(next);
            return next;
        });
    }, []);

    const setFit = useCallback((fit: FitMode) => update({ fit }), [update]);

    // ---- gaps as custom properties -------------------------------------------
    //
    // Written onto <body> rather than onto the scroller: the scroller is rebuilt
    // on every view change, so a property set on it would have to be re-applied
    // in lock step. Inheritance does that for free.
    useEffect(() => {
        const root = document.body;
        root.style.setProperty("--vx-vertical-gap", `${settings.verticalGap}px`);
        root.style.setProperty("--vx-spread-gap", `${settings.spreadGap}px`);
        root.style.setProperty("--strip-gap", `${settings.verticalGap}px`);
        return () => {
            // Leaving `--strip-gap` behind would give the *other* layout's strip a
            // gap it never asked for if the user switches layouts without a reload.
            root.style.removeProperty("--strip-gap");
            root.style.removeProperty("--vx-vertical-gap");
            root.style.removeProperty("--vx-spread-gap");
        };
    }, [settings.verticalGap, settings.spreadGap]);

    // ---- a new folder ---------------------------------------------------------
    //
    // `renderGallery` replaces `state.currentFiles` with a fresh array and sets
    // `stripEligible` from the pages' dimensions, then selects page one -- which
    // emits `page:changed`. Comparing the array *identity* is therefore a reliable
    // "a new folder was loaded", without a new event or a change to `viewer.ts`.
    //
    // Two things follow from it: the webtoon guess below, and a strip rebuild. The
    // rebuild is not optional -- `renderGallery` sets `state.stripMode = false`, so
    // a strip left standing from the previous folder would be showing its pages
    // while `isStripMode()` reported there was no strip at all.
    const seenFiles = useRef<readonly string[] | null>(null);
    useEffect(() => {
        const onPageChanged = () => {
            if (seenFiles.current === state.currentFiles) return;
            seenFiles.current = state.currentFiles;
            setFolderToken((n) => n + 1);
            if (settings.pinned || !state.stripEligible) return;
            // Not `setPageView`: this is the app guessing, so it must not count as
            // the reader having chosen and must stay open to being guessed again.
            update({ pageView: "vertical" });
        };
        on("page:changed", onPageChanged);
        return () => off("page:changed", onPageChanged);
    }, [settings.pinned, update]);

    // ---- the strip, built exactly while the continuous view is showing --------
    const wantStrip = mode === "reader" && settings.pageView === "vertical";
    useEffect(() => {
        if (state.stripMode !== wantStrip) {
            state.stripMode = wantStrip;
            emit("strip:changed");
        }
        if (wantStrip) {
            // Torn down first: `buildStrip` returns early when a scroller already
            // exists, so on a folder change the old folder's pages would simply
            // stay on screen.
            destroyStrip();
            // One frame later: the stage is revealed by the same render that set
            // `wantStrip`, and `buildStrip` measures the container to estimate
            // every slot's height.
            const raf = requestAnimationFrame(() => {
                if (state.stripMode) buildStrip();
            });
            return () => cancelAnimationFrame(raf);
        }
        destroyStrip();
        // Back to a `#mainImage` view: the stage it was laid out for may have been
        // a different size (or none at all, while a pager covered it).
        const timer = setTimeout(refitViewer, 0);
        return () => clearTimeout(timer);
    }, [wantStrip, folderToken]);

    // ---- leaving the reader ---------------------------------------------------
    //
    // The two modes express the same page with different mechanics. Reader mode
    // carries its own pan in `state.panX/panY` and writes it into `#mainImage`'s
    // transform (`renderReader`); work mode ignores that pan entirely and centres
    // the page instead, scrolling the container for the overflow
    // (`applyWorkTransform`). So a page the reader left dragged off-centre keeps a
    // transform work mode never asked for, and sits outside the stage.
    //
    // Nothing else covers this. The strip effect above re-fits, but only when
    // `wantStrip` *changes* -- and in the single, double and compare views it is
    // false in both modes, so a mode switch does not re-run it. The Vulpes layout
    // happened to be saved by the `ResizeObserver` on its stage, because leaving the
    // reader un-hides the top bar, footer and rail and the stage therefore changes
    // size; the Basic layout has no such observer and showed the bug.
    //
    // Deferred to the next frame rather than run inline, for an ordering reason:
    // `WorkspaceProvider` owns the `body.workspace-reader` class and sits *above*
    // this provider, so its effect runs after this one. Re-fitting inline would
    // still see `isReaderMode() === true` and re-apply the reader's own layout.
    useEffect(() => {
        if (mode === "reader") return;
        const raf = requestAnimationFrame(() => {
            // The pan belongs to the reader. Clearing it here means re-entering the
            // reader starts centred, and `applyImageFit` does not inherit an offset
            // from a session the reader has already left.
            state.panX = 0;
            state.panY = 0;
            refitViewer();
        });
        return () => cancelAnimationFrame(raf);
    }, [mode]);

    // A page change, a save, or an edit is a reason for the pager and the reader
    // chrome to re-read what they derive from module state. Cheap: a counter they
    // depend on, rather than each of them subscribing to the bus separately.
    useEffect(() => {
        const bump = () => setRevision((n) => n + 1);
        on("page:changed", bump);
        on("image:loaded", bump);
        on("strip:changed", bump);
        // The comparison view's two sides change meaning the moment the second one
        // starts existing, so a save has to reach it.
        on("page:saved", bump);
        return () => {
            off("page:changed", bump);
            off("image:loaded", bump);
            off("strip:changed", bump);
            off("page:saved", bump);
        };
    }, []);

    const value = useMemo<ReaderViewValue>(
        () => ({ settings, setPageView, setFit, update, revision }),
        [settings, setPageView, setFit, update, revision],
    );

    return <ReaderViewContext.Provider value={value}>{children}</ReaderViewContext.Provider>;
}
