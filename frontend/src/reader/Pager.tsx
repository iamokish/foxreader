/**
 * The pager: single page, double-page spread, and before/after compare.
 *
 * One renderer, three data shapes. A "row" is one or two page images laid side by
 * side with a gutter, scaled as a unit and panned as a unit:
 *
 *   single   the current page, alone
 *   double   the current spread, ordered by reading direction
 *   compare  the current page twice -- the original scan, and the typeset copy
 *
 * Writing them once is the point: fit, zoom, wheel, drag-pan and the clamp that
 * keeps the row on screen exist in a single place, so the three views cannot drift
 * into behaving differently.
 *
 * It owns its own transform rather than reusing `#imageContainer`. That container
 * is the *editor's* viewport -- `initPanning`, `initZoom` and every capture read
 * its scroll offsets and `#mainImage`'s bounding rect to turn a drag into image
 * coordinates -- so a second renderer sharing it would have to keep that maths
 * true for two and three pages at once. Here the editor's stage is simply covered
 * (see `vulpes.css`) and left exactly as it was.
 *
 * ## Why pan and zoom are not component state
 *
 * Each `<img>` is laid out at its *natural* size, once, when it decodes. The live
 * zoom and pan are a single `transform` on the row, written straight to the DOM
 * from the pointer and wheel handlers. So dragging a 4000px page around costs one
 * composited transform per frame: no Preact render, no reflow, and no re-raster of
 * the image.
 *
 * The first cut did the obvious thing instead -- zoom and pan in `useState`, with
 * every pane's `width`/`height` computed from the zoom -- and it was visibly
 * sluggish: each frame re-rendered the subtree, then relaid out and re-rasterised
 * two full-resolution images. The only value that still goes through state is the
 * zoom *percentage* the chrome displays, and that is coalesced to one update per
 * frame.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import { state } from "../state";
import { currentPageUrl, variantPageUrl } from "../viewer";
import { getReadingDirection } from "../readingOrder";
import {
    buildSpreads,
    clampPan,
    contentSize,
    displaySpread,
    fitScale,
    rowSize,
    spreadIndexOf,
    stepZoom,
    wheelZoom,
    type FitMode,
    type PageView,
    type Size,
} from "./viewSettings";
import { currentPageIndex } from "./pageNav";

/** A view the pager can render. The continuous view is the strip's, not the pager's. */
export type PagerView = Exclude<PageView, "vertical">;

/** One image in the row, with the caption the compare view puts under it. */
export interface Pane {
    key: string;
    url: string;
    /** 1-based page number. */
    pageNumber: number;
    /** Caption, for the comparison view. `null` leaves the page uncaptioned. */
    label: string | null;
}

/** The pager's zoom, for the chrome that draws the buttons. */
export interface PagerApi {
    /** Step the zoom by a proportion of itself, keeping the row centred. */
    zoomBy: (delta: number) => void;
    /** Return to the current fit, discarding a hand-set zoom and pan. */
    refit: () => void;
}

interface PagerProps {
    view: PagerView;
    fit: FitMode;
    /** Gutter between panes, unscaled. Unused by `single`, which has one pane. */
    gap: number;
    coverAlone: boolean;
    wideAlone: boolean;
    /** Bumped by the view context when the page, the folder or a save changed. */
    revision: number;
    /** The live zoom, for the chrome's readout. At most one call per frame. */
    onZoomChange?: (zoom: number) => void;
    /** Hands the zoom controls to whoever draws the buttons; `null` on unmount. */
    onApi?: (api: PagerApi | null) => void;
}

/**
 * The panes to show, derived from app state rather than held in the pager.
 *
 * `state.currentFiles` and the active thumbnail are the app's page list and
 * cursor; duplicating either here is how a pager ends up a page behind the
 * gallery. Exported for its tests.
 */
export function buildPanes(view: PagerView, options: { coverAlone: boolean; wideAlone: boolean }): Pane[] {
    const files = state.currentFiles;
    const index = currentPageIndex();
    if (files.length === 0 || index < 0 || !files[index]) return [];

    if (view === "compare") {
        const file = files[index];
        const hasSaved = state.savedFiles.has(file) && !state.sameDir;
        // Both panes always render. A page with no typeset copy shows the original
        // on both sides rather than an empty frame -- `variantPageUrl` refuses to
        // ask for a variant that does not exist -- and the caption says so, which
        // reads correctly as "nothing has been typeset here yet".
        return [
            {
                key: `before:${file}`,
                url: variantPageUrl(file, "original"),
                pageNumber: index + 1,
                label: "Before · original",
            },
            {
                key: `after:${file}`,
                url: variantPageUrl(file, "saved"),
                pageNumber: index + 1,
                label: hasSaved ? "After · saved" : "After · not saved yet",
            },
        ];
    }

    if (view === "double") {
        const spreads = buildSpreads(files.length, {
            coverAlone: options.coverAlone,
            wideAlone: options.wideAlone,
            aspects: files.map((file) => state.aspectCache[file] ?? 0),
        });
        const spread = spreads[spreadIndexOf(spreads, index)] ?? [index];
        return displaySpread(spread).map((i) => ({
            key: `page:${i}:${files[i]}`,
            url: currentPageUrl(files[i]),
            pageNumber: i + 1,
            label: null,
        }));
    }

    return [
        {
            key: `page:${index}:${files[index]}`,
            url: currentPageUrl(files[index]),
            pageNumber: index + 1,
            label: null,
        },
    ];
}

/** How far a press may travel before it counts as a drag rather than a click. */
const PAN_CLICK_SLOP = 4;

export function Pager({ view, fit, gap, coverAlone, wideAlone, revision, onZoomChange, onApi }: PagerProps) {
    const stageRef = useRef<HTMLDivElement>(null);
    const rowRef = useRef<HTMLDivElement>(null);

    /** Natural sizes, keyed by pane, filled in as each image decodes. */
    const [natural, setNatural] = useState<Record<string, Size>>({});
    /** The zoom the chrome displays. Coalesced to one update per frame. */
    const [zoomLabel, setZoomLabel] = useState(1);

    // The live view, as refs: these are read and written by pointer and wheel
    // handlers at frame rate and must not re-render anything.
    const zoomRef = useRef(1);
    const panRef = useRef({ x: 0, y: 0 });
    const viewportRef = useRef<Size>({ width: 0, height: 0 });
    const contentRef = useRef<Size>({ width: 0, height: 0 });
    /** Set once the reader zooms or pans by hand, so a re-fit does not undo it. */
    const touched = useRef(false);
    const labelFrame = useRef(0);

    const direction = getReadingDirection();
    // Derived from module state, which `revision` is the signal for; `direction`
    // decides which side of a spread each page lands on.
    const panes = useMemo(
        () => buildPanes(view, { coverAlone, wideAlone }),
        [view, coverAlone, wideAlone, revision, direction],
    );

    const content = useMemo(
        () =>
            contentSize(
                panes.map((pane) => natural[pane.key] ?? { width: 0, height: 0 }),
                gap,
            ),
        [panes, natural, gap],
    );
    contentRef.current = content;

    /** Write the live zoom and pan to the DOM. The only thing a frame of panning does. */
    const paint = useCallback(() => {
        const row = rowRef.current;
        if (!row) return;
        const { x, y } = panRef.current;
        row.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${zoomRef.current})`;
        // The grab cursor is a promise that there is somewhere to drag to, so it is
        // derived from the live size rather than from state -- which this is
        // deliberately not kept in. A class toggle costs nothing next to the
        // transform write that already happened.
        const stage = stageRef.current;
        if (!stage) return;
        const box = rowSize(contentRef.current, zoomRef.current);
        const vp = viewportRef.current;
        stage.classList.toggle("is-pannable", box.width > vp.width + 1 || box.height > vp.height + 1);
    }, []);

    /** Publish the zoom to the chrome, at most once per frame. */
    const publishZoom = useCallback(() => {
        if (labelFrame.current) return;
        labelFrame.current = requestAnimationFrame(() => {
            labelFrame.current = 0;
            setZoomLabel(zoomRef.current);
        });
    }, []);

    useEffect(() => () => cancelAnimationFrame(labelFrame.current), []);

    useEffect(() => {
        onZoomChange?.(zoomLabel);
    }, [zoomLabel, onZoomChange]);

    /** Clamp the live pan so the row cannot be dragged off the stage. */
    const clampLive = useCallback((next: { x: number; y: number }) => {
        const box = rowSize(contentRef.current, zoomRef.current);
        const vp = viewportRef.current;
        return { x: clampPan(vp.width, box.width, next.x), y: clampPan(vp.height, box.height, next.y) };
    }, []);

    const setLive = useCallback(
        (zoom: number, pan: { x: number; y: number }) => {
            zoomRef.current = zoom;
            panRef.current = pan;
            paint();
            publishZoom();
        },
        [paint, publishZoom],
    );

    /** Apply the current fit. The one path that un-sets `touched`. */
    const refit = useCallback(() => {
        touched.current = false;
        setLive(fitScale(fit, viewportRef.current, contentRef.current), { x: 0, y: 0 });
    }, [fit, setLive]);

    // ---- measurement ---------------------------------------------------------
    useLayoutEffect(() => {
        const el = stageRef.current;
        if (!el) return;
        const measure = () => {
            viewportRef.current = { width: el.clientWidth, height: el.clientHeight };
            // A stage that changed size invalidates both the fit and the pan: re-fit
            // unless the reader has taken over, and otherwise just re-clamp.
            if (touched.current) setLive(zoomRef.current, clampLive(panRef.current));
            else setLive(fitScale(fit, viewportRef.current, contentRef.current), { x: 0, y: 0 });
        };
        measure();
        // A ResizeObserver, not a window listener: the stage also changes size when
        // a splitter is dragged or a section collapsed, and neither resizes the
        // window. (jsdom has no ResizeObserver, hence the fallback.)
        if (typeof ResizeObserver === "undefined") {
            window.addEventListener("resize", measure);
            return () => window.removeEventListener("resize", measure);
        }
        const observer = new ResizeObserver(measure);
        observer.observe(el);
        return () => observer.disconnect();
    }, [fit, setLive, clampLive]);

    // ---- fitting -------------------------------------------------------------
    //
    // A new view, a new page, or an explicit fit always re-fits: what is on screen
    // is not what the old zoom was chosen for, so the fit mode is what sticks
    // rather than the zoom.
    useEffect(() => {
        refit();
    }, [view, fit, revision, refit]);

    // Each image arriving changes the content box, and with it the fit -- but not if
    // the reader has already zoomed in on what had arrived by then.
    useEffect(() => {
        if (touched.current) setLive(zoomRef.current, clampLive(panRef.current));
        else setLive(fitScale(fit, viewportRef.current, contentRef.current), { x: 0, y: 0 });
    }, [content.width, content.height, fit, setLive, clampLive]);

    // ---- panning -------------------------------------------------------------
    const onPointerDown = useCallback(
        (e: PointerEvent) => {
            if (e.button !== 0) return;
            const box = rowSize(contentRef.current, zoomRef.current);
            const vp = viewportRef.current;
            if (box.width <= vp.width + 1 && box.height <= vp.height + 1) return;

            const el = e.currentTarget as HTMLElement;
            const startX = e.clientX;
            const startY = e.clientY;
            const origin = panRef.current;
            let dragging = false;

            const onMove = (ev: PointerEvent) => {
                const dx = ev.clientX - startX;
                const dy = ev.clientY - startY;
                if (!dragging && Math.abs(dx) < PAN_CLICK_SLOP && Math.abs(dy) < PAN_CLICK_SLOP) return;
                if (!dragging) {
                    dragging = true;
                    touched.current = true;
                    el.setPointerCapture?.(ev.pointerId);
                    el.classList.add("is-panning");
                }
                ev.preventDefault();
                // Straight to the DOM: no state, so no render and no reflow.
                panRef.current = clampLive({ x: origin.x + dx, y: origin.y + dy });
                paint();
            };

            const onUp = (ev: PointerEvent) => {
                if (el.hasPointerCapture?.(ev.pointerId)) el.releasePointerCapture(ev.pointerId);
                el.classList.remove("is-panning");
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
            };

            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
        },
        [clampLive, paint],
    );

    // ---- wheel zoom ----------------------------------------------------------
    //
    // Attached by hand rather than via `onWheel`, because it has to be registered
    // non-passive to be allowed to `preventDefault`.
    useEffect(() => {
        const el = stageRef.current;
        if (!el) return;
        const onWheel = (e: WheelEvent) => {
            if (e.deltaY === 0) return;
            e.preventDefault();
            const next = wheelZoom(zoomRef.current, e.deltaY);
            if (next === zoomRef.current) return;
            touched.current = true;
            // Keep whatever is under the cursor on the same pixel of the page. The
            // row is centred, so the pan is an offset from the stage's middle.
            const rect = el.getBoundingClientRect();
            const mx = e.clientX - rect.left - rect.width / 2;
            const my = e.clientY - rect.top - rect.height / 2;
            const factor = next / zoomRef.current;
            const pan = { x: mx - (mx - panRef.current.x) * factor, y: my - (my - panRef.current.y) * factor };
            zoomRef.current = next;
            setLive(next, clampLive(pan));
        };
        el.addEventListener("wheel", onWheel, { passive: false });
        return () => el.removeEventListener("wheel", onWheel);
    }, [clampLive, setLive]);

    // ---- the zoom, handed to the chrome -------------------------------------
    const zoomBy = useCallback(
        (delta: number) => {
            const next = stepZoom(zoomRef.current, delta);
            if (next === zoomRef.current) return;
            touched.current = true;
            // About the middle, which is where a button press aims.
            const factor = next / zoomRef.current;
            zoomRef.current = next;
            setLive(next, clampLive({ x: panRef.current.x * factor, y: panRef.current.y * factor }));
        },
        [clampLive, setLive],
    );

    useEffect(() => {
        onApi?.({ zoomBy, refit });
        return () => onApi?.(null);
    }, [onApi, zoomBy, refit]);

    const onImageLoad = useCallback((key: string, img: HTMLImageElement) => {
        if (!img.naturalWidth || !img.naturalHeight) return;
        setNatural((prev) => {
            const known = prev[key];
            if (known && known.width === img.naturalWidth && known.height === img.naturalHeight) return prev;
            return { ...prev, [key]: { width: img.naturalWidth, height: img.naturalHeight } };
        });
    }, []);

    // One root element for both the empty and the loaded state, on purpose: the
    // `ResizeObserver` and the wheel listener are bound to `stageRef`, and a
    // separate early-return root would swap the node out from under them -- leaving
    // them attached to a detached div, so the first folder loaded into an empty
    // reader would never be measured and never fitted.
    return (
        <div
            class={`vx-pager vx-pager--${view}${panes.length === 0 ? " vx-pager--empty" : ""}`}
            ref={stageRef}
            onPointerDown={onPointerDown}
        >
            {panes.length === 0 ? (
                <div class="vx-pager__empty">
                    <span class="material-icons" aria-hidden="true">
                        auto_stories
                    </span>
                    <p>Load a folder to start reading.</p>
                </div>
            ) : (
                <div
                    class="vx-pager__row"
                    ref={rowRef}
                    style={{
                        gap: `${gap}px`,
                        // Seeded from the live refs so a re-render never reverts the
                        // transform `paint` has written.
                        transform: `translate3d(${panRef.current.x}px, ${panRef.current.y}px, 0) scale(${zoomRef.current})`,
                    }}
                >
                    {panes.map((pane) => {
                        const size = natural[pane.key];
                        return (
                            <figure class="vx-pane" key={pane.key}>
                                {/* Laid out at natural size, once, so a zoom never
                                    relayouts it. Held at zero and hidden until that
                                    size is known: a full-resolution scan with no width
                                    yet would lay out at its intrinsic size for a
                                    frame. */}
                                <img
                                    class="vx-pane__img"
                                    src={pane.url}
                                    alt={`Page ${pane.pageNumber}`}
                                    draggable={false}
                                    decoding="async"
                                    style={
                                        size
                                            ? { width: `${size.width}px`, height: `${size.height}px` }
                                            : { width: 0, height: 0, visibility: "hidden" }
                                    }
                                    onLoad={(e) => onImageLoad(pane.key, e.currentTarget as HTMLImageElement)}
                                />
                            </figure>
                        );
                    })}
                </div>
            )}

            {/* Captions live outside the scaled row on purpose: inside it they would
                be scaled with the artwork, unreadable at a fit and enormous zoomed
                in. One cell per pane, so each label sits under its own page. */}
            {panes.some((pane) => pane.label) && (
                <div class="vx-pager__captions" aria-hidden="true">
                    {panes.map((pane) => (
                        <span class="vx-pager__caption" key={`label:${pane.key}`}>
                            <span class="vx-pane__label">{pane.label}</span>
                        </span>
                    ))}
                </div>
            )}
        </div>
    );
}
