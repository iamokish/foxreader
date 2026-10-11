/**
 * Reader view settings, and the geometry the page views are built from.
 *
 * Four ways to look at a folder:
 *
 *   single    one page, fitted and zoomable          -- the default
 *   double    two facing pages with a gutter         -- print spreads
 *   vertical  continuous scroll, one column          -- webtoons (the strip)
 *   compare   the same page before and after typeset -- original | saved
 *
 * `vertical` is the app's existing strip renderer (`buildStrip` in `viewer.ts`),
 * which already does lazy loading, skeletons and scroll memory; the reader only adds
 * the gap. `single`, `double` and `compare` are one renderer with three data
 * shapes -- see `Pager.tsx` -- so they share a single zoom/pan/fit
 * implementation and cannot drift apart.
 *
 * Everything in here is pure except `loadViewSettings`/`saveViewSettings`, so the
 * pairing and fitting rules are testable without a DOM.
 */

import { getReadingDirection, type ReadingDirection } from "../readingOrder";

export type PageView = "single" | "double" | "vertical" | "compare";

/** How a page (or a spread) is sized to the stage. */
export type FitMode = "page" | "width" | "height" | "actual";

export const PAGE_VIEWS: readonly PageView[] = ["single", "double", "vertical", "compare"] as const;
export const FIT_MODES: readonly FitMode[] = ["page", "width", "height", "actual"] as const;

/** Vertical/continuous gap, in CSS px. The range the slider offers. */
export const MIN_VERTICAL_GAP = 0;
export const MAX_VERTICAL_GAP = 10;
export const DEFAULT_VERTICAL_GAP = 5;

/**
 * Spread gutter, in CSS px. Wider than the vertical range on purpose: a seam
 * between two tiles of one continuous strip wants to be nearly invisible, while
 * a gutter between two facing pages reads better with room to breathe.
 */
export const MIN_SPREAD_GAP = 0;
export const MAX_SPREAD_GAP = 40;
export const DEFAULT_SPREAD_GAP = 8;

export const MIN_ZOOM = 0.05;
export const MAX_ZOOM = 8;

/**
 * Above this height/width ratio a page is a single portrait page; at or below it
 * the page is already a printed double spread and is shown on its own rather
 * than paired with a neighbour. 1.0 is the square: anything landscape qualifies.
 */
export const WIDE_PAGE_ASPECT = 1;

export interface ViewSettings {
    pageView: PageView;
    fit: FitMode;
    /** CSS px between pages in the continuous view. */
    verticalGap: number;
    /** CSS px in the gutter of a double-page spread. */
    spreadGap: number;
    /** Show the first page on its own, so the rest pair up as the print edition does. */
    coverAlone: boolean;
    /** Give a landscape page the whole stage instead of pairing it. */
    wideAlone: boolean;
    /**
     * Whether the reader has picked a view by hand.
     *
     * Until they have, loading a folder of tall pages switches to the continuous
     * view by itself -- the webtoon detection the app already does in
     * `detectStripEligible`. One explicit pick and that stops second-guessing.
     */
    pinned: boolean;
}

export const DEFAULT_VIEW_SETTINGS: Readonly<ViewSettings> = Object.freeze({
    pageView: "single",
    fit: "page",
    verticalGap: DEFAULT_VERTICAL_GAP,
    spreadGap: DEFAULT_SPREAD_GAP,
    coverAlone: true,
    wideAlone: true,
    pinned: false,
});

const STORAGE_KEY = "fox-reader-vulpes-view";

function clamp(value: number, min: number, max: number): number {
    if (!Number.isFinite(value)) return min;
    return Math.max(min, Math.min(max, value));
}

/**
 * Coerce anything at all into usable settings.
 *
 * Stored JSON is the input here, so every field is treated as hostile: a build
 * that drops a view, a hand-edited value out of range, or a half-written object
 * must still produce a reader that opens.
 */
export function normalizeViewSettings(raw: unknown): ViewSettings {
    const input = (raw ?? {}) as Partial<Record<keyof ViewSettings, unknown>>;
    const pageView = PAGE_VIEWS.includes(input.pageView as PageView)
        ? (input.pageView as PageView)
        : DEFAULT_VIEW_SETTINGS.pageView;
    const fit = FIT_MODES.includes(input.fit as FitMode) ? (input.fit as FitMode) : DEFAULT_VIEW_SETTINGS.fit;
    return {
        pageView,
        fit,
        verticalGap: Math.round(
            clamp(Number(input.verticalGap ?? DEFAULT_VERTICAL_GAP), MIN_VERTICAL_GAP, MAX_VERTICAL_GAP),
        ),
        spreadGap: Math.round(clamp(Number(input.spreadGap ?? DEFAULT_SPREAD_GAP), MIN_SPREAD_GAP, MAX_SPREAD_GAP)),
        coverAlone: input.coverAlone === undefined ? DEFAULT_VIEW_SETTINGS.coverAlone : Boolean(input.coverAlone),
        wideAlone: input.wideAlone === undefined ? DEFAULT_VIEW_SETTINGS.wideAlone : Boolean(input.wideAlone),
        pinned: Boolean(input.pinned),
    };
}

export function loadViewSettings(): ViewSettings {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return normalizeViewSettings(raw ? JSON.parse(raw) : null);
    } catch {
        return { ...DEFAULT_VIEW_SETTINGS };
    }
}

export function saveViewSettings(settings: ViewSettings): void {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
        /* a locked-down profile keeps the session's settings only */
    }
}

// ------------------------------------------------------------------- spreads

/**
 * Group page indices into the rows a double-page view shows.
 *
 * Each group is one or two indices in *reading* order -- earlier page first --
 * and `pagesOf` turns that into left/right at paint time. Three rules, each one
 * switchable, applied in this order:
 *
 *   coverAlone  page 1 stands by itself, so 2-3, 4-5 ... land the way they are
 *               printed rather than one page out of step
 *   wideAlone   a landscape page is already a spread and takes the stage alone
 *   otherwise   pair greedily; a trailing odd page ends up alone
 *
 * `aspects` is height/width per index, as `state.aspectCache` holds it. A page
 * whose aspect is not known yet is assumed portrait: guessing "wide" would split
 * a pair and reflow the whole book the moment the image decoded.
 */
export function buildSpreads(
    count: number,
    options: { coverAlone?: boolean; wideAlone?: boolean; aspects?: readonly number[] } = {},
): number[][] {
    const { coverAlone = true, wideAlone = true, aspects } = options;
    const spreads: number[][] = [];
    if (!Number.isFinite(count) || count <= 0) return spreads;

    const isWide = (index: number): boolean => {
        if (!wideAlone) return false;
        const aspect = aspects?.[index];
        return typeof aspect === "number" && aspect > 0 && aspect <= WIDE_PAGE_ASPECT;
    };

    let i = 0;
    while (i < count) {
        if ((i === 0 && coverAlone) || isWide(i)) {
            spreads.push([i]);
            i += 1;
            continue;
        }
        const next = i + 1;
        if (next < count && !isWide(next)) {
            spreads.push([i, next]);
            i += 2;
        } else {
            spreads.push([i]);
            i += 1;
        }
    }
    return spreads;
}

/** Which spread holds `pageIndex`, or -1 when nothing does. */
export function spreadIndexOf(spreads: readonly number[][], pageIndex: number): number {
    for (let i = 0; i < spreads.length; i++) {
        if (spreads[i].includes(pageIndex)) return i;
    }
    return -1;
}

/**
 * Lay a spread out on screen.
 *
 * Reading order is what `buildSpreads` returns; this is the only place it becomes
 * left and right. Manga (RTL, the app's default) puts the earlier page on the
 * right, which is why a spread read in the wrong direction looks like the panels
 * jump backwards across the gutter.
 */
export function orderForDisplay(spread: readonly number[], direction: ReadingDirection): number[] {
    const pages = [...spread];
    return direction === "rtl" ? pages.reverse() : pages;
}

/** The display order for the current reading-direction preference. */
export function displaySpread(spread: readonly number[]): number[] {
    return orderForDisplay(spread, getReadingDirection());
}

// ---------------------------------------------------------------------- fit

export interface Size {
    width: number;
    height: number;
}

/** Total gutter width between `panes` panes, unscaled. Zero for fewer than two. */
export function gutterWidth(gap: number, panes: number): number {
    if (panes < 2) return 0;
    return Math.max(0, gap) * (panes - 1);
}

/**
 * The box a row of pages occupies at scale 1, gutter included.
 *
 * One row, whatever is in it: a single page, a facing pair, or the two panes of a
 * comparison. Pages of unequal size are *not* reconciled -- the row is as wide as
 * the sum and as tall as the tallest, and each page keeps its own aspect, so a
 * spread of a short page beside a long one centres rather than stretching to
 * match. Panes with no measured size yet are skipped: they contribute nothing
 * until their image has decoded, and the gutter shrinks with them.
 *
 * The gutter is part of the content, so it scales with the zoom the way the gap
 * between two facing pages of a physical book does. That is also what lets the
 * pager move the whole row with one `scale()` -- the alternative, a gutter fixed
 * in screen pixels, means re-laying out every page on every frame of a zoom.
 */
export function contentSize(pages: readonly Size[], gap = 0): Size {
    const usable = pages.filter((p) => p.width > 0 && p.height > 0);
    if (usable.length === 0) return { width: 0, height: 0 };
    return {
        width: usable.reduce((sum, p) => sum + p.width, 0) + gutterWidth(gap, usable.length),
        height: usable.reduce((tallest, p) => Math.max(tallest, p.height), 0),
    };
}

/** The box a row occupies on screen at `zoom`. */
export function rowSize(content: Size, zoom: number): Size {
    if (content.width <= 0 || content.height <= 0) return { width: 0, height: 0 };
    return { width: content.width * zoom, height: content.height * zoom };
}

/**
 * The scale that realises `fit` for `content` inside `viewport`.
 *
 * Clamped into the zoom range, so a fit is always a zoom the controls can then
 * step away from and back to. A viewport or content with no area yields 1 rather
 * than Infinity or 0 -- the page is simply not measurable yet, and the next
 * layout pass will ask again.
 */
export function fitScale(fit: FitMode, viewport: Size, content: Size): number {
    if (fit === "actual") return 1;
    if (viewport.width <= 0 || viewport.height <= 0 || content.width <= 0 || content.height <= 0) {
        return 1;
    }
    const byWidth = viewport.width / content.width;
    const byHeight = viewport.height / content.height;
    const scale = fit === "width" ? byWidth : fit === "height" ? byHeight : Math.min(byWidth, byHeight);
    return clamp(scale, MIN_ZOOM, MAX_ZOOM);
}

/**
 * Keep one axis of the content inside the viewport.
 *
 * The same rule the single-page viewer applies (`clampPanAxis` in `viewer.ts`),
 * restated here against a centred row so the pager has no dependency on the
 * editor's viewport: an axis bigger than the viewport may be dragged until its
 * own edge meets the viewport's, an axis that fits may not be dragged at all.
 * Returns a pan offset, not a position.
 */
export function clampPan(viewport: number, content: number, pan: number): number {
    const slack = (content - viewport) / 2;
    if (slack <= 0) return 0;
    return clamp(pan, -slack, slack);
}

/** Step a zoom by a *proportion* of itself, so every notch is the same visual step. */
export function stepZoom(zoom: number, delta: number): number {
    return clamp(zoom * (1 + delta), MIN_ZOOM, MAX_ZOOM);
}

/** The zoom a wheel notch of `deltaY` lands on. Matches the editor's feel. */
export function wheelZoom(zoom: number, deltaY: number): number {
    return clamp(zoom * Math.exp(-deltaY * 0.001), MIN_ZOOM, MAX_ZOOM);
}

/**
 * How far one step of navigation moves, in pages.
 *
 * Single and compare move a page at a time; the double view moves a spread,
 * whose width depends on where it is in the book, so the caller resolves that
 * from `buildSpreads` instead. Exported for the keyboard handler, which has to
 * agree with the buttons.
 */
export function pagesPerStep(view: PageView): number {
    return view === "double" ? 2 : 1;
}
