/**
 * Reader mode: a full-bleed surface plus the controls that drive it.
 *
 * Covers the editing stage rather than replacing it. `#imageContainer`,
 * `#mainImage` and `#gallery` are mounted exactly once by the layout and must stay
 * mounted for the lifetime of the page -- `initPanning`, `initZoom` and
 * `setupImageErrorHandler` bind to them once at startup in `main.tsx`, and an
 * unmount would silently take panning, zooming and image-retry with it. So the
 * pager is an opaque overlay on top, and the one view that *is* the editor's
 * stage -- the continuous strip, which `buildStrip` builds inside
 * `#imageContainer` -- simply does not draw one.
 *
 * Navigation never calls `selectImage` directly: see `pageNav.ts`.
 *
 * The two stylesheets are imported **here**, not by the layout. `reader.css` is
 * unscoped and shared, and `strip.css` is the renderer the continuous view builds
 * with, so both belong to the reader rather than to whichever layout happens to be
 * on screen. Importing them from the layout was how they ended up missing from one
 * of the two layouts -- and from the reader's own control primitives, which the
 * Vulpes header and footer also use.
 */

import "./reader.css";
import "../workspace/strip.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import { state } from "../state";
import { useWorkspace } from "../workspace";
import { refitViewer, setViewVariant, zoomViewerBy } from "../viewer";
import { getReadingDirection } from "../readingOrder";
import { useReaderView } from "./ReaderViewContext";
import { Pager, type PagerApi, type PagerView } from "./Pager";
import { currentPageIndex, gotoPage, pageCount, wrapIndex } from "./pageNav";
import { setReaderKeyHandler } from "./readerKeys";
import {
    buildSpreads,
    MAX_SPREAD_GAP,
    MAX_VERTICAL_GAP,
    MIN_SPREAD_GAP,
    MIN_VERTICAL_GAP,
    spreadIndexOf,
    type FitMode,
    type PageView,
} from "./viewSettings";

const VIEW_OPTIONS: { view: PageView; icon: string; label: string; hint: string }[] = [
    { view: "single", icon: "crop_portrait", label: "Single", hint: "One page at a time (1)" },
    { view: "double", icon: "auto_stories", label: "Double", hint: "Two facing pages (2)" },
    { view: "vertical", icon: "view_day", label: "Vertical", hint: "Continuous vertical scroll (3)" },
    { view: "compare", icon: "compare", label: "Compare", hint: "Original beside the typeset copy (4)" },
];

const FIT_OPTIONS: { fit: FitMode; label: string; hint: string }[] = [
    { fit: "page", label: "Page", hint: "Fit the whole page" },
    { fit: "width", label: "Width", hint: "Fit the width" },
    { fit: "height", label: "Height", hint: "Fit the height" },
    { fit: "actual", label: "1:1", hint: "Actual pixel size" },
];

function toggleFullscreen(): void {
    if (document.fullscreenElement) {
        document.exitFullscreen().catch(() => {});
    } else {
        document.documentElement.requestFullscreen().catch(() => {});
    }
}

/**
 * The page index one step away, in pages.
 *
 * Single and compare move one page. The double view moves a whole spread, and how
 * many pages that is depends on where in the book it is -- a cover stands alone, a
 * landscape page stands alone -- so the step is read off the spread table rather
 * than assumed to be two.
 */
export function stepTarget(
    view: PageView,
    index: number,
    count: number,
    direction: number,
    spreads: readonly number[][],
): number {
    if (count <= 0) return -1;
    if (view !== "double") return wrapIndex(index + direction, count);
    const at = spreadIndexOf(spreads, index);
    if (at < 0) return wrapIndex(index + direction, count);
    const next = spreads[wrapIndex(at + direction, spreads.length)];
    return next ? next[0] : wrapIndex(index + direction, count);
}

/** How long a still pointer waits before the reader hides its chrome entirely. */
const IDLE_AFTER_MS = 3000;

/** Page-turn keys: exempt from waking the idle UI (see the idle effect). */
const ARROW_KEYS: ReadonlySet<string> = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);

export function Reader() {
    const { mode, setMode } = useWorkspace();
    const { settings, setPageView, setFit, update, revision } = useReaderView();
    const [zoom, setZoom] = useState(1);
    /** Local mirror of `state.viewVariant`, purely to force a re-render on change. */
    const [variant, setVariant] = useState(state.viewVariant);
    /** True while nothing has been touched for `IDLE_AFTER_MS`. */
    const [idle, setIdle] = useState(false);
    const pagerApi = useRef<PagerApi | null>(null);
    const takeApi = useCallback((api: PagerApi | null) => {
        pagerApi.current = api;
    }, []);

    const count = pageCount();
    const index = currentPageIndex();
    const direction = getReadingDirection();

    const spreads = useMemo(
        () =>
            buildSpreads(count, {
                coverAlone: settings.coverAlone,
                wideAlone: settings.wideAlone,
                aspects: state.currentFiles.map((file) => state.aspectCache[file] ?? 0),
            }),
        [count, settings.coverAlone, settings.wideAlone, revision],
    );

    const step = useCallback(
        (delta: number) => {
            const target = stepTarget(settings.pageView, currentPageIndex(), pageCount(), delta, spreads);
            if (target >= 0) gotoPage(target);
        },
        [settings.pageView, spreads],
    );

    const zoomBy = useCallback(
        (delta: number) => {
            // The strip keeps its own zoom inside `viewer.ts`, where the scroller
            // it has to adjust lives; the pager keeps its own, next to the
            // transform it drives. Each is asked in its own terms.
            if (settings.pageView === "vertical") zoomViewerBy(delta);
            else pagerApi.current?.zoomBy(delta);
        },
        [settings.pageView],
    );

    const resetZoom = useCallback(() => {
        if (settings.pageView === "vertical") refitViewer();
        else pagerApi.current?.refit();
    }, [settings.pageView]);

    const pickVariant = useCallback((next: "original" | "saved") => {
        setViewVariant(next);
        setVariant(state.viewVariant);
    }, []);

    // ---- idle ---------------------------------------------------------------
    //
    // A still pointer means the reader is reading, so everything drawn over the
    // page goes: the chrome, the click zones, the comparison captions and the
    // cursor itself (see `reader.css`). Any pointer movement, key press or wheel
    // brings it straight back.
    //
    // Bound to `window` in the capture phase, because the pointer is as often over
    // the strip's scroller, the stage underneath or the pager as it is over
    // anything this component renders -- and a listener on `.vx-reader` would miss
    // all of those. The timer is one `setTimeout`, reset on every event; there is no
    // polling and no rAF loop running while the reader sits still.
    //
    // Deliberately *not* applied to `body.workspace-reader` for the whole app: the
    // editor keeps its chrome no matter how still the pointer is, because hiding the
    // toolbar and the editor panel mid-edit would be hostile.
    useEffect(() => {
        if (mode !== "reader") {
            setIdle(false);
            return;
        }
        let timer = 0;
        const wake = () => {
            clearTimeout(timer);
            setIdle(false);
            timer = window.setTimeout(() => setIdle(true), IDLE_AFTER_MS);
        };
        // Arrow keys turn pages; they are for reading, not for reaching, so as
        // far as idleness is concerned they are not input at all: they neither
        // wake a hidden UI nor restart the timer while it is visible. Every
        // other key still wakes, including a keyless event (synthetic, or a key
        // the platform reports without a name).
        const onInput = (event: Event) => {
            if (ARROW_KEYS.has((event as KeyboardEvent).key)) return;
            wake();
        };
        const events = ["pointermove", "pointerdown", "wheel", "keydown"] as const;
        for (const name of events) window.addEventListener(name, onInput, { capture: true, passive: true });
        wake();
        return () => {
            clearTimeout(timer);
            for (const name of events) window.removeEventListener(name, onInput, { capture: true });
            setIdle(false);
        };
    }, [mode]);

    // The cursor is hidden by a body class as well as by the reader's own: the
    // strip's scroller and `#imageContainer` sit outside `.vx-reader`, and the
    // pointer is over one of them whenever the continuous view is up.
    useEffect(() => {
        document.body.classList.toggle("is-reader-idle", idle);
        return () => document.body.classList.remove("is-reader-idle");
    }, [idle]);

    // ---- keyboard ------------------------------------------------------------
    useEffect(() => {
        if (mode !== "reader") return;
        setReaderKeyHandler((e) => {
            if (e.ctrlKey || e.altKey || e.metaKey) return false;
            switch (e.key.toLowerCase()) {
                case "arrowright":
                case "arrowdown":
                case " ":
                    // The continuous view is scrolled, not paged. `main.tsx` already
                    // routes these to `navigateGallery`, which in strip mode scrolls
                    // smoothly to the next page -- so decline rather than restate it.
                    if (settings.pageView === "vertical") return false;
                    e.preventDefault();
                    step(1);
                    return true;
                case "arrowleft":
                case "arrowup":
                    if (settings.pageView === "vertical") return false;
                    e.preventDefault();
                    step(-1);
                    return true;
                case "1":
                    e.preventDefault();
                    setPageView("single");
                    return true;
                case "2":
                    e.preventDefault();
                    setPageView("double");
                    return true;
                case "3":
                    e.preventDefault();
                    setPageView("vertical");
                    return true;
                case "4":
                    e.preventDefault();
                    setPageView("compare");
                    return true;
                case "s":
                    e.preventDefault();
                    setPageView(settings.pageView === "vertical" ? "single" : "vertical");
                    return true;
                case "0":
                    e.preventDefault();
                    resetZoom();
                    return true;
                case "+":
                case "=":
                    e.preventDefault();
                    zoomBy(0.1);
                    return true;
                case "-":
                    e.preventDefault();
                    zoomBy(-0.1);
                    return true;
                default:
                    // Escape and F are the same in every reader: let `main.tsx`
                    // keep owning them rather than restating them here.
                    return false;
            }
        });
        return () => setReaderKeyHandler(null);
    }, [mode, settings.pageView, step, setPageView, resetZoom, zoomBy]);

    if (mode !== "reader") return null;

    const pagerView = settings.pageView as PagerView;
    const showPager = settings.pageView !== "vertical";
    const currentFile = state.currentFiles[index] ?? "";
    const canCompare = !state.sameDir && state.savedFiles.size > 0;
    const variantUsable = !state.sameDir && state.savedFiles.has(currentFile);
    const pageLabel = count > 0 && index >= 0 ? `${index + 1} / ${count}` : "—";
    const startTurnLabel = direction === "rtl" ? "Next page" : "Previous page";
    const endTurnLabel = direction === "rtl" ? "Previous page" : "Next page";
    const spreadLabel =
        settings.pageView === "double" && index >= 0
            ? `spread ${Math.max(1, spreadIndexOf(spreads, index) + 1)} / ${spreads.length}`
            : null;

    return (
        <div class={`vx-reader vx-reader--${settings.pageView}${idle ? " is-idle" : ""}`}>
            {showPager && (
                <Pager
                    view={pagerView}
                    fit={settings.fit}
                    gap={settings.pageView === "single" ? 0 : settings.spreadGap}
                    coverAlone={settings.coverAlone}
                    wideAlone={settings.wideAlone}
                    revision={revision}
                    onZoomChange={setZoom}
                    onApi={takeApi}
                />
            )}

            {/* Click zones: the page itself is the control for turning it. Absent in
                the continuous view, where the gesture is a scroll. The arrow inside
                is drawn only on hover (see `reader.css`) -- an arrow permanently over
                the edge of the artwork is the thing a reader least wants to see. The
                zone stays clickable either way. */}
            {showPager && (
                <>
                    <button
                        type="button"
                        class="vx-reader__zone vx-reader__zone--start"
                        title={startTurnLabel}
                        aria-label={startTurnLabel}
                        onClick={() => step(direction === "rtl" ? 1 : -1)}
                    >
                        <span class="vx-reader__arrow" aria-hidden="true">
                            <span class="material-icons">chevron_left</span>
                        </span>
                    </button>
                    <button
                        type="button"
                        class="vx-reader__zone vx-reader__zone--end"
                        title={endTurnLabel}
                        aria-label={endTurnLabel}
                        onClick={() => step(direction === "rtl" ? -1 : 1)}
                    >
                        <span class="vx-reader__arrow" aria-hidden="true">
                            <span class="material-icons">chevron_right</span>
                        </span>
                    </button>
                </>
            )}

            <div class="vx-reader__chrome">
                <div class="vx-reader__bar" role="toolbar" aria-label="Reader controls">
                    <div class="vx-seg" role="group" aria-label="Page view">
                        {VIEW_OPTIONS.map((opt) => (
                            <button
                                key={opt.view}
                                type="button"
                                class={`vx-seg__btn${settings.pageView === opt.view ? " is-active" : ""}`}
                                title={opt.hint}
                                aria-pressed={settings.pageView === opt.view}
                                disabled={opt.view === "compare" && !canCompare}
                                onClick={() => setPageView(opt.view)}
                            >
                                <span class="material-icons" aria-hidden="true">
                                    {opt.icon}
                                </span>
                                <span class="vx-seg__label">{opt.label}</span>
                            </button>
                        ))}
                    </div>

                    <span class="vx-reader__sep" />

                    <button type="button" class="vx-icon-btn" title="Previous page" onClick={() => step(-1)}>
                        <span class="material-icons">chevron_left</span>
                    </button>
                    {/* Deliberately not `#readerPageIndicator`: `updateReaderPageIndicator`
                        writes that element's `textContent`, which would wipe out a
                        Preact-rendered child. The label re-renders from `revision`
                        instead, which the view context bumps on every page change. */}
                    <span class="vx-reader__page" title={spreadLabel ?? "Page"}>
                        <span class="vx-reader__page-num">{pageLabel}</span>
                        {spreadLabel && <small class="vx-reader__page-sub">{spreadLabel}</small>}
                    </span>
                    <button type="button" class="vx-icon-btn" title="Next page" onClick={() => step(1)}>
                        <span class="material-icons">chevron_right</span>
                    </button>

                    <span class="vx-reader__sep" />

                    {showPager && (
                        <div class="vx-seg" role="group" aria-label="Fit">
                            {FIT_OPTIONS.map((opt) => (
                                <button
                                    key={opt.fit}
                                    type="button"
                                    class={`vx-seg__btn${settings.fit === opt.fit ? " is-active" : ""}`}
                                    title={opt.hint}
                                    aria-pressed={settings.fit === opt.fit}
                                    onClick={() => setFit(opt.fit)}
                                >
                                    <span class="vx-seg__label">{opt.label}</span>
                                </button>
                            ))}
                        </div>
                    )}

                    <button type="button" class="vx-icon-btn" title="Zoom out (-)" onClick={() => zoomBy(-0.1)}>
                        <span class="material-icons">remove</span>
                    </button>
                    <button type="button" class="vx-icon-btn" title="Reset zoom (0)" onClick={resetZoom}>
                        <span class="material-icons">refresh</span>
                    </button>
                    <button type="button" class="vx-icon-btn" title="Zoom in (+)" onClick={() => zoomBy(0.1)}>
                        <span class="material-icons">add</span>
                    </button>

                    {showPager && <span class="vx-reader__zoom">{Math.round(zoom * 100)}%</span>}

                    <span class="vx-reader__sep" />

                    {/* Gap belongs to whichever view has one: the strip's seam, or
                        the spread's gutter. Compare shares the spread gutter. */}
                    {settings.pageView === "vertical" ? (
                        <label class="vx-range" title="Space between pages in the continuous view">
                            <span class="vx-range__label">Gap</span>
                            <input
                                type="range"
                                min={MIN_VERTICAL_GAP}
                                max={MAX_VERTICAL_GAP}
                                step={1}
                                value={settings.verticalGap}
                                onInput={(e) =>
                                    update({ verticalGap: Number((e.currentTarget as HTMLInputElement).value) })
                                }
                            />
                            <span class="vx-range__value">{settings.verticalGap}px</span>
                        </label>
                    ) : (
                        settings.pageView !== "single" && (
                            <label class="vx-range" title="Gutter between the two pages">
                                <span class="vx-range__label">Gap</span>
                                <input
                                    type="range"
                                    min={MIN_SPREAD_GAP}
                                    max={MAX_SPREAD_GAP}
                                    step={1}
                                    value={settings.spreadGap}
                                    onInput={(e) =>
                                        update({ spreadGap: Number((e.currentTarget as HTMLInputElement).value) })
                                    }
                                />
                                <span class="vx-range__value">{settings.spreadGap}px</span>
                            </label>
                        )
                    )}

                    {settings.pageView === "double" && (
                        <>
                            <button
                                type="button"
                                class={`vx-chip${settings.coverAlone ? " is-active" : ""}`}
                                title="Show the first page on its own, so the rest pair as the print edition does"
                                aria-pressed={settings.coverAlone}
                                onClick={() => update({ coverAlone: !settings.coverAlone })}
                            >
                                Cover
                            </button>
                            <button
                                type="button"
                                class={`vx-chip${settings.wideAlone ? " is-active" : ""}`}
                                title="Give a landscape page the whole stage instead of pairing it"
                                aria-pressed={settings.wideAlone}
                                onClick={() => update({ wideAlone: !settings.wideAlone })}
                            >
                                Wide
                            </button>
                        </>
                    )}

                    {variantUsable && settings.pageView !== "compare" && (
                        <>
                            <span class="vx-reader__sep" />
                            <div class="vx-seg" role="group" aria-label="Page copy">
                                <button
                                    type="button"
                                    class={`vx-seg__btn${variant === "original" ? " is-active" : ""}`}
                                    title="Show the original scan"
                                    aria-pressed={variant === "original"}
                                    onClick={() => pickVariant("original")}
                                >
                                    <span class="vx-seg__label">Original</span>
                                </button>
                                <button
                                    type="button"
                                    class={`vx-seg__btn${variant === "saved" ? " is-active" : ""}`}
                                    title="Show the typeset copy from the destination folder"
                                    aria-pressed={variant === "saved"}
                                    onClick={() => pickVariant("saved")}
                                >
                                    <span class="vx-seg__label">Saved</span>
                                </button>
                            </div>
                        </>
                    )}

                    <span class="vx-reader__sep" />

                    <button type="button" class="vx-icon-btn" title="Toggle fullscreen (F)" onClick={toggleFullscreen}>
                        <span class="material-icons">fullscreen</span>
                    </button>
                    <button
                        type="button"
                        class="vx-icon-btn vx-icon-btn--danger"
                        title="Exit reader (Esc)"
                        onClick={() => setMode("work")}
                    >
                        <span class="material-icons">close</span>
                    </button>
                </div>
            </div>
        </div>
    );
}
