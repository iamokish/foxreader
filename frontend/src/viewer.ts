import { state, isViewingSaved } from "./state";
import type { PageVariant } from "./state";
import { emit } from "./eventbus";

const STRIP_DETECT_ASPECT = 2.2;
const STRIP_DETECT_SAMPLE = 5;
const STRIP_MIN_ZOOM = 0.25;
const STRIP_MAX_ZOOM = 8;
const STRIP_ADAPTIVE_VIEWPORTS = 1.2;

/**
 * The zoom range for the page on screen, for both non-strip layouts.
 *
 * The two ends are the two fits: the page shown whole is as far out as zooming
 * goes, the page filling the container's width is as far in. Which of the two
 * is the smaller number depends on the page -- a tall scan fits vertically at a
 * smaller zoom than it fits horizontally, a wide one the other way round -- so
 * they are ordered rather than assumed, and a tall page in a narrow viewport
 * cannot come out inverted.
 *
 * A horizontal fit is a single zoom, not a range: it already *is* the upper
 * end, and the point of it is to scroll down a page held at that width. Both
 * bounds are that value, so every zoom entry point clamps back to where it
 * started and falls through.
 */
function zoomBounds(container: HTMLElement, mainImg: HTMLImageElement): { min: number; max: number } {
    if (!mainImg.naturalWidth || !mainImg.naturalHeight) {
        return { min: state.zoomLevel, max: state.zoomLevel };
    }
    const fitWidth = container.clientWidth / mainImg.naturalWidth;
    if (isHorizontalFit()) return { min: fitWidth, max: fitWidth };
    const fitHeight = container.clientHeight / mainImg.naturalHeight;
    return { min: Math.min(fitHeight, fitWidth), max: Math.max(fitHeight, fitWidth) };
}

/** Whether the layout on screen is holding the page at the container's width. */
function isHorizontalFit(): boolean {
    return isReaderMode() ? state.readerFit === "width" : state.isHorizontalFitEnabled;
}

/**
 * The URL for one page, in whichever copy the viewer is currently showing.
 *
 * Every `/img_serve` request goes through here, for two reasons. The filename is
 * escaped -- a page called `1+2 (a#b).png` used to produce a URL the server read
 * as a different name, or as a fragment -- and `variant=saved` is only ever added
 * when there really is a separate saved copy of *this* page, so a stale switch
 * cannot ask for something that does not exist.
 */
export function pageUrl(file: string, buster: number | string): string {
    const url = `/img_serve/${encodeURIComponent(file)}?t=${buster}`;
    const wantsSaved = state.viewVariant === "saved" && !state.sameDir && state.savedFiles.has(file);
    return wantsSaved ? `${url}&variant=saved` : url;
}

/**
 * Cache busting, per page rather than per repaint.
 *
 * `?t=` used to be `Date.now()` read at the moment of every render, so
 * re-pointing the thumbnails -- which flipping the Original/Saved switch does to
 * all of them at once -- produced a brand new URL for every page and a full
 * re-download of a gallery that had not changed a pixel. The token is now a
 * property of the *bytes*: one per folder load, plus a component that moves only
 * when a save rewrites that particular page. Flipping the switch therefore asks
 * for URLs the browser already has, and nothing else is disturbed.
 */
let folderBuster = 0;
const pageBusters = new Map<string, number>();

/**
 * The per-page half of the token: a counter, not a clock.
 *
 * Two saves of the same page inside one millisecond are ordinary -- Preview then
 * Save is a couple of clicks -- and a timestamp would give them the same value,
 * which is the one case where the stale copy must not be served.
 */
let saveTick = 0;

/** The cache-busting token for the copy of `file` that is currently on screen. */
function busterFor(file: string): string {
    // A save writes the *destination* copy, so it only invalidates what a page
    // looks like while the switch is on the saved side -- unless the destination
    // is the source, where the page itself was rewritten and both sides are it.
    const rewritten = state.sameDir || (state.viewVariant === "saved" && state.savedFiles.has(file));
    const bump = rewritten ? pageBusters.get(file) : undefined;
    return bump === undefined ? String(folderBuster) : `${folderBuster}.${bump}`;
}

/** {@link pageUrl} for the copy of `file` on screen, at its current version. */
export function currentPageUrl(file: string): string {
    return pageUrl(file, busterFor(file));
}

/**
 * Point an `<img>` at `url`, and report whether that was actually a change.
 *
 * The comparison is against a stamp rather than `img.src`, which the DOM hands
 * back resolved to an absolute URL: comparing that to the relative form would
 * never match, every call would re-fetch, and the fetch is the thing this exists
 * to avoid.
 */
function setImageSrc(img: HTMLImageElement | null, url: string): boolean {
    if (!img) return false;
    if (img.dataset.foxSrc === url && img.getAttribute("src")) return false;
    img.dataset.foxSrc = url;
    img.src = url;
    return true;
}

let stripScroller: HTMLElement | null = null;
let stripObserver: IntersectionObserver | null = null;
let stripActiveIndex = -1;
let stripAdaptiveApplied = false;

export function ensureSvgOverlayExists(): SVGSVGElement | null {
    if (state.svgOverlayElement) return state.svgOverlayElement;
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("id", "regionSvgOverlay");
    svg.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:5;";
    const mainImg = document.getElementById("mainImage");
    if (mainImg && mainImg.parentNode) {
        mainImg.parentNode.insertBefore(svg, mainImg.nextSibling);
    } else {
        document.getElementById("imageContainer")?.appendChild(svg);
    }
    state.svgOverlayElement = svg;
    return svg;
}

export function syncSvgOverlayDimensions(): void {
    if (isStripMode()) return;
    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    const svg = ensureSvgOverlayExists();
    if (!mainImg || !svg) return;

    svg.setAttribute("width", String(mainImg.clientWidth));
    svg.setAttribute("height", String(mainImg.clientHeight));
    svg.setAttribute("viewBox", `0 0 ${mainImg.naturalWidth} ${mainImg.naturalHeight}`);
    svg.style.width = mainImg.clientWidth + "px";
    svg.style.height = mainImg.clientHeight + "px";
    svg.style.top = mainImg.style.top || "0px";
    svg.style.left = mainImg.style.left || "0px";
    svg.style.transformOrigin = mainImg.style.transformOrigin || "0 0";
    svg.style.transform = mainImg.style.transform;
}

export function updateCursor(): void {
    const container = document.getElementById("imageContainer");
    if (!container) return;
    if (state.isCaptureMode) {
        container.style.cursor = "crosshair";
    } else if (state.isPanning) {
        container.style.cursor = "grabbing";
    } else {
        container.style.cursor = isPannable(container) ? "grab" : "default";
    }
}

/**
 * Whether there is anywhere to pan to -- what the `grab` cursor promises.
 *
 * Work mode asks the container, which is the thing that scrolls. Reader mode
 * has no scrollbars to ask, so the question is put to the transform instead:
 * the page is pannable exactly when it is bigger than the viewport, which is
 * also where {@link clampPanAxis} stops holding it still.
 */
function isPannable(container: HTMLElement): boolean {
    if (isStripMode()) return false;
    if (isReaderMode()) {
        const img = document.getElementById("mainImage") as HTMLImageElement | null;
        if (!img || !img.naturalWidth) return false;
        return (
            img.naturalWidth * state.zoomLevel > container.clientWidth + 1 ||
            img.naturalHeight * state.zoomLevel > container.clientHeight + 1
        );
    }
    return (
        container.scrollWidth > container.clientWidth + 1 || container.scrollHeight > container.clientHeight + 1
    );
}

export function getClampedZoom(current: number, delta: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, current + delta));
}

export function isReaderMode(): boolean {
    return document.body.classList.contains("workspace-reader");
}

export function isStripMode(): boolean {
    return state.stripMode && isReaderMode();
}

function getStripSlot(index: number): HTMLElement | null {
    if (!stripScroller) return null;
    return stripScroller.querySelector<HTMLElement>(`.strip-page[data-index="${index}"]`);
}

function getMedianAspect(): number {
    const values = Object.values(state.aspectCache);
    if (values.length === 0) return 1.4;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function updateGalleryActiveByIndex(index: number): void {
    const items = Array.from(document.querySelectorAll(".gallery-item-container"));
    items.forEach((el, i) => el.classList.toggle("active", i === index));
}

function positionStripOverlay(): void {
    const svg = ensureSvgOverlayExists();
    const slot = getStripSlot(stripActiveIndex);
    if (!svg || !slot) return;
    const img = slot.querySelector<HTMLImageElement>(".strip-img");
    if (!img || img.naturalWidth === 0) {
        svg.setAttribute("viewBox", "0 0 1 1");
        return;
    }
    svg.remove();
    slot.appendChild(svg);
    svg.style.cssText = "position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:5;";
    svg.setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
}

function resetStripOverlayStyle(svg: SVGSVGElement): void {
    svg.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:5;";
}

function updateStripActive(): void {
    if (!stripScroller) return;
    const pages = Array.from(stripScroller.querySelectorAll<HTMLElement>(".strip-page"));
    if (pages.length === 0) return;
    const scrollTop = stripScroller.scrollTop;
    const threshold = scrollTop + (stripScroller.clientHeight || 1) * 0.4;
    let active = 0;
    for (let i = 0; i < pages.length; i++) {
        if (pages[i].offsetTop <= threshold) active = i;
        else break;
    }
    if (active === stripActiveIndex) return;
    stripActiveIndex = active;
    const file = state.currentFiles[active];
    if (file) {
        if (state.currentImageFile) state.stripScroll[state.currentImageFile] = scrollTop;
        const changed = file !== state.currentImageFile;
        state.currentImageFile = file;
        state.currentlySelectedEntryId = null;
        // Scrolling the strip past a page boundary *is* a page change, even
        // though nothing was clicked and no image was loaded to say so.
        if (changed) emit("page:changed", file);
    }
    updateGalleryActiveByIndex(active);
    positionStripOverlay();
    emit("image:loaded", file);
}

function scrollStripTo(index: number, smooth = true): void {
    if (!stripScroller) return;
    const pages = Array.from(stripScroller.querySelectorAll<HTMLElement>(".strip-page"));
    if (pages.length === 0) return;
    const target = Math.max(0, Math.min(pages.length - 1, index));
    stripScroller.scrollTo({ top: pages[target].offsetTop, behavior: smooth ? "smooth" : "auto" });
}

function zoomStripBy(delta: number): void {
    if (!stripScroller) return;
    const next = getClampedZoom(state.stripZoom, delta, STRIP_MIN_ZOOM, STRIP_MAX_ZOOM);
    if (next === state.stripZoom) return;
    const factor = next / state.stripZoom;
    const scrollTop = stripScroller.scrollTop;
    state.stripZoom = next;
    stripScroller.style.setProperty("--strip-zoom", String(next));
    updateStripSlotSizes();
    stripScroller.scrollTop = scrollTop * factor;
}

/** Recomputes every slot's deterministic intrinsic height for the current zoom. */
function updateStripSlotSizes(): void {
    if (!stripScroller) return;
    const container = document.getElementById("imageContainer");
    const width = container?.clientWidth || stripScroller.clientWidth || 1;
    const median = getMedianAspect();
    state.currentFiles.forEach((file, index) => {
        const slot = getStripSlot(index);
        if (!slot) return;
        const aspect = state.aspectCache[file] || median;
        slot.style.containIntrinsicSize = `${Math.round((width * state.stripZoom) / aspect)}px`;
    });
}

function resetStripZoom(): void {
    if (!stripScroller) return;
    const factor = 1 / state.stripZoom;
    state.stripZoom = 1;
    stripScroller.style.setProperty("--strip-zoom", "1");
    updateStripSlotSizes();
    stripScroller.scrollTop *= factor;
}

function getActivePageAspect(): number {
    const file = state.currentFiles[stripActiveIndex];
    if (file && state.aspectCache[file]) return state.aspectCache[file];
    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    if (mainImg && mainImg.naturalWidth > 0) return mainImg.naturalHeight / mainImg.naturalWidth;
    return 0;
}

function applyAdaptiveStripZoomIfNeeded(): void {
    if (stripAdaptiveApplied || !stripScroller) return;
    const container = document.getElementById("imageContainer");
    if (!container || container.clientWidth === 0 || container.clientHeight === 0) return;
    const aspect = getActivePageAspect();
    if (aspect <= 0) return;
    const z = Math.max(
        STRIP_MIN_ZOOM,
        Math.min(1, (STRIP_ADAPTIVE_VIEWPORTS * container.clientHeight) / (container.clientWidth * aspect)),
    );
    stripAdaptiveApplied = true;
    if (z >= state.stripZoom) return;
    const factor = z / state.stripZoom;
    state.stripZoom = z;
    stripScroller.style.setProperty("--strip-zoom", String(z));
    stripScroller.scrollTop *= factor;
}

/** Decides strip eligibility from the first five pages' dimensions. */
function detectStripEligible(dimensions: [number, number][]): boolean {
    const sample = dimensions
        .slice(0, STRIP_DETECT_SAMPLE)
        .map(([w, h]) => (w > 0 && h > 0 ? h / w : 0))
        .filter((aspect) => aspect > 0);
    if (sample.length === 0) return false;
    const tall = sample.filter((aspect) => aspect >= STRIP_DETECT_ASPECT).length;
    return tall / sample.length >= 0.5;
}

export function buildStrip(): void {
    const container = document.getElementById("imageContainer");
    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    if (!container || stripScroller || state.currentFiles.length === 0) return;

    stripAdaptiveApplied = false;

    const scroller = document.createElement("div");
    scroller.id = "strip-scroller";
    scroller.className = "ws-strip-scroller";
    scroller.style.setProperty("--strip-zoom", String(state.stripZoom));
    container.appendChild(scroller);
    stripScroller = scroller;

    const defaultAspect = getMedianAspect();
    const containerWidth = container.clientWidth || 1;

    state.currentFiles.forEach((file, index) => {
        const slot = document.createElement("div");
        slot.className = "strip-page";
        slot.dataset.index = String(index);
        const aspect = state.aspectCache[file] || defaultAspect;
        const estH = Math.round((containerWidth * state.stripZoom) / aspect);
        slot.style.containIntrinsicSize = `${estH}px`;

        const skeleton = document.createElement("div");
        skeleton.className = "strip-page__skeleton";
        skeleton.style.aspectRatio = `1 / ${aspect}`;
        slot.appendChild(skeleton);

        const img = document.createElement("img");
        img.className = "strip-img";
        img.decoding = "async";
        img.draggable = false;
        img.dataset.src = currentPageUrl(file);
        slot.appendChild(img);

        scroller.appendChild(slot);
    });

    if (mainImg) mainImg.style.display = "none";

    stripObserver = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (!entry.isIntersecting) continue;
                const slot = entry.target as HTMLElement;
                const img = slot.querySelector<HTMLImageElement>(".strip-img");
                if (!img || img.src) continue;
                img.src = img.dataset.src || "";
                img.addEventListener("load", () => {
                    slot.classList.add("loaded");
                    const idx = Number(slot.dataset.index || "-1");
                    const file = state.currentFiles[idx];
                    if (file && img.naturalWidth > 0) {
                        const aspect = img.naturalHeight / img.naturalWidth;
                        state.aspectCache[file] = aspect;
                        const sk = slot.querySelector<HTMLElement>(".strip-page__skeleton");
                        if (sk) sk.style.aspectRatio = `1 / ${aspect}`;
                    }
                    if (idx === stripActiveIndex) {
                        positionStripOverlay();
                        applyAdaptiveStripZoomIfNeeded();
                        emit("image:loaded", state.currentImageFile);
                    }
                });
                img.addEventListener("error", () => {
                    slot.classList.add("loaded");
                });
            }
        },
        { root: scroller, rootMargin: "1000px 0px" },
    );

    state.currentFiles.forEach((_, index) => {
        const slot = getStripSlot(index);
        if (slot) stripObserver?.observe(slot);
    });

    stripActiveIndex = state.currentFiles.indexOf(state.currentImageFile);
    if (stripActiveIndex < 0) stripActiveIndex = 0;

    const saved = state.stripScroll[state.currentImageFile];
    if (saved !== undefined) {
        scroller.scrollTop = saved;
    } else {
        const slot = getStripSlot(stripActiveIndex);
        if (slot) scroller.scrollTop = slot.offsetTop;
    }

    scroller.addEventListener("scroll", () => {
        requestAnimationFrame(updateStripActive);
    });

    requestAnimationFrame(() => {
        updateStripActive();
        positionStripOverlay();
        applyAdaptiveStripZoomIfNeeded();
        emit("image:loaded", state.currentImageFile);
    });
}

export function destroyStrip(): void {
    stripObserver?.disconnect();
    stripObserver = null;
    const container = document.getElementById("imageContainer");
    if (stripScroller && stripScroller.parentNode === container) {
        const scrollTop = stripScroller.scrollTop;
        if (state.currentImageFile) state.stripScroll[state.currentImageFile] = scrollTop;
        stripScroller.remove();
    }
    stripScroller = null;
    stripActiveIndex = -1;

    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    if (mainImg) mainImg.style.display = "block";

    const svg = document.getElementById("regionSvgOverlay") as SVGSVGElement | null;
    if (svg && mainImg && svg.parentNode !== mainImg.parentNode) {
        svg.remove();
        mainImg.parentNode?.insertBefore(svg, mainImg.nextSibling);
        resetStripOverlayStyle(svg);
    }
}

export function toggleStripMode(): void {
    const next = !state.stripMode;
    state.stripMode = next;
    state.stripOverride = next ? "on" : "off";
    if (isReaderMode()) {
        if (next) {
            buildStrip();
        } else {
            destroyStrip();
        }
    }
    emit("strip:changed");
}

/**
 * Applies a reader-mode transform: the image is positioned from its top-left,
 * scaled about the origin, with the amount of pan held in state.panX/panY.
 * Centered layout is expressed as (container - image)/2 + pan offsets.
 */
export function applyReaderTransform(
    container: HTMLElement,
    mainImg: HTMLImageElement,
    zoom: number,
    panX: number,
    panY: number,
): void {
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const W = mainImg.naturalWidth;
    const H = mainImg.naturalHeight;
    const px = (cw - W * zoom) / 2 + panX;
    const py = (ch - H * zoom) / 2 + panY;
    mainImg.style.transformOrigin = "0 0";
    mainImg.style.top = "0px";
    mainImg.style.left = "0px";
    mainImg.style.transform = `translate(${px}px, ${py}px) scale(${zoom})`;
    syncSvgOverlayDimensions();
}

/**
 * Keep one axis of the page inside the viewport.
 *
 * Panning is unbounded arithmetic -- `panX + dx`, every frame -- so without this
 * a long drag walks the page off the edge and leaves an empty viewport with no
 * way back but the refit key. The rule is the ordinary one: an axis larger than
 * the viewport may be dragged until its own edge meets the viewport's, and an
 * axis that already fits may not be dragged out of it at all.
 *
 * `base` is where the axis starts with no pan applied, so the return value is
 * again a pan and not a position.
 */
function clampPanAxis(viewport: number, content: number, base: number, pan: number): number {
    const fits = content <= viewport;
    const lo = fits ? 0 : viewport - content;
    const hi = fits ? viewport - content : 0;
    return Math.min(hi, Math.max(lo, base + pan)) - base;
}

/** Store a clamped reader pan and repaint at it. */
function setReaderPan(container: HTMLElement, mainImg: HTMLImageElement, panX: number, panY: number): void {
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const w = mainImg.naturalWidth * state.zoomLevel;
    const h = mainImg.naturalHeight * state.zoomLevel;
    state.panX = clampPanAxis(cw, w, (cw - w) / 2, panX);
    state.panY = clampPanAxis(ch, h, (ch - h) / 2, panY);
    applyReaderTransform(container, mainImg, state.zoomLevel, state.panX, state.panY);
    updateCursor();
}

function renderReader(container: HTMLElement, mainImg: HTMLImageElement): void {
    container.style.overflow = "hidden";
    setReaderPan(container, mainImg, state.panX, state.panY);
}

/**
 * Put the container's overflow back to what the current layout wants.
 *
 * The one owner of that question. Capture turns scrolling off while the canvas
 * is up and has to turn it back on afterwards, and it used to re-derive the rule
 * itself -- in two places, from a test (`isHorizontalFitEnabled || zoom > fit`)
 * that is not what the layout does: reader mode pans by transform and must never
 * scroll, and a page can overflow at the fit itself.
 */
export function syncContainerOverflow(): void {
    const container = document.getElementById("imageContainer");
    if (!container) return;
    // The strip's own scroller is `inset: 0` inside this one and does all the
    // scrolling; the container must not add a second scrollbar of its own.
    if (isStripMode() || isReaderMode()) {
        container.style.overflow = "hidden";
        return;
    }
    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    if (!mainImg || !mainImg.naturalWidth || !mainImg.naturalHeight) {
        container.style.overflow = "hidden";
        return;
    }
    const w = mainImg.naturalWidth * state.zoomLevel;
    const h = mainImg.naturalHeight * state.zoomLevel;
    // A pixel of slack: a fit divides one of these by the other, and the product
    // comes back a hair over often enough to flicker an axis in and out of
    // scrollability for a page that plainly fits.
    container.style.overflow =
        w > container.clientWidth + 1 || h > container.clientHeight + 1 ? "auto" : "hidden";
}

/**
 * Lay the page out for work mode at the current zoom.
 *
 * One rule for both fits and for every zoom either of them lands on: the image
 * is placed from the container's top-left, and an axis smaller than the viewport
 * is centred by an explicit offset instead. Scrolling then works wherever there
 * is something to scroll, which is what the old layout could not do -- centring
 * with `top/left: 50%` and `translate(-50%, -50%)` puts half the image at
 * negative coordinates, where no scroll offset can reach it, so vertical fit had
 * to keep `overflow` off and had nothing for a grab to move.
 *
 * Scroll offsets are deliberately left alone: the callers that change the zoom
 * want to choose their own, and the browser clamps whatever falls out of range.
 */
function applyWorkTransform(container: HTMLElement, mainImg: HTMLImageElement): void {
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const zoom = state.zoomLevel;
    const w = mainImg.naturalWidth * zoom;
    const h = mainImg.naturalHeight * zoom;

    mainImg.style.transformOrigin = "0 0";
    mainImg.style.top = "0px";
    mainImg.style.left = "0px";
    mainImg.style.transform = `translate(${Math.max(0, (cw - w) / 2)}px, ${Math.max(0, (ch - h) / 2)}px) scale(${zoom})`;
    syncContainerOverflow();

    syncSvgOverlayDimensions();
    updateCursor();
}

/**
 * Zoom work mode about a container-local point, keeping that point on the same
 * pixel of the page.
 *
 * The counterpart of {@link zoomReaderAround} for the scrolling layout: what the
 * reader carries as a pan offset is a scroll offset here, so it is the same
 * arithmetic written against `scrollLeft`/`scrollTop`. Without it every zoom
 * step snapped back to the top-left corner, which is what made zooming in on
 * anything below the fold impossible.
 */
function zoomWorkAround(
    container: HTMLElement,
    mainImg: HTMLImageElement,
    mx: number,
    my: number,
    delta: number,
): void {
    const { min, max } = zoomBounds(container, mainImg);
    const z0 = state.zoomLevel;
    const z1 = getClampedZoom(z0, delta, min, max);
    if (z1 === z0) return;

    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const W = mainImg.naturalWidth;
    const H = mainImg.naturalHeight;
    // Where the page starts on screen now, and the image-space point under the
    // cursor because of it.
    const x0 = Math.max(0, (cw - W * z0) / 2) - container.scrollLeft;
    const y0 = Math.max(0, (ch - H * z0) / 2) - container.scrollTop;
    const u = (mx - x0) / z0;
    const v = (my - y0) / z0;

    state.zoomLevel = z1;
    applyWorkTransform(container, mainImg);
    // After the transform, so the scroll range the new size allows already
    // exists; anything outside it the browser clamps for us.
    container.scrollLeft = Math.max(0, (cw - W * z1) / 2) + u * z1 - mx;
    container.scrollTop = Math.max(0, (ch - H * z1) / 2) + v * z1 - my;
}

/**
 * Zoom the viewer by `delta`, about the middle of the viewport.
 *
 * What the buttons and the `+`/`-` keys call. Outside the strip, `delta` is read
 * as a *fraction* of the current zoom rather than an absolute step: a page
 * fitted at 0.25 and one blown up to 4 would otherwise get the same 0.1, which
 * is a 40% jump for the first and imperceptible for the second.
 */
export function zoomViewerBy(delta: number): void {
    if (isStripMode()) {
        zoomStripBy(delta);
        return;
    }

    const container = document.getElementById("imageContainer");
    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    if (!container || !mainImg || !mainImg.naturalHeight || !mainImg.naturalWidth) return;

    const mx = container.clientWidth / 2;
    const my = container.clientHeight / 2;
    const step = state.zoomLevel * delta;
    if (isReaderMode()) {
        zoomReaderAround(container, mainImg, mx, my, step);
    } else {
        // Horizontal fit is not refused here: `zoomBounds` gives it a range of
        // one value, so the step clamps back to where it started and nothing
        // moves.
        zoomWorkAround(container, mainImg, mx, my, step);
    }
}

/** Zooms about a container-local point (mx,my). Reader mode only. */
function zoomReaderAround(
    container: HTMLElement,
    mainImg: HTMLImageElement,
    mx: number,
    my: number,
    delta: number,
): void {
    const z0 = state.zoomLevel;
    const { min, max } = zoomBounds(container, mainImg);
    const z1 = getClampedZoom(z0, delta, min, max);
    if (z1 === z0) return;

    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const W = mainImg.naturalWidth;
    const H = mainImg.naturalHeight;
    // Displayed top-left of the current image.
    const px0 = (cw - W * z0) / 2 + state.panX;
    const py0 = (ch - H * z0) / 2 + state.panY;
    // Image-space coordinate under the cursor.
    const u = (mx - px0) / z0;
    const v = (my - py0) / z0;
    // Where that image point must land so it stays under the cursor at z1.
    const px1 = mx - u * z1;
    const py1 = my - v * z1;

    state.zoomLevel = z1;
    state.panX = px1 - (cw - W * z1) / 2;
    state.panY = py1 - (ch - H * z1) / 2;
    renderReader(container, mainImg);
}

export function renderGallery(files: string[], dimensions: [number, number][] = []): void {
    const gallery = document.getElementById("gallery");
    if (!gallery) return;
    gallery.innerHTML = "";
    folderBuster = Date.now();
    pageBusters.clear();

    state.currentFiles = files;
    state.stripMode = false;
    state.stripOverride = "auto";
    state.stripEligible = detectStripEligible(dimensions);
    state.stripZoom = 1;
    state.stripScroll = {};

    files.forEach((file, index) => {
        const itemDiv = document.createElement("div");
        itemDiv.className = "gallery-item-container";

        const img = document.createElement("img");
        img.className = "gallery-item";
        // A thumbnail is the page itself, served full size, so a folder of two
        // hundred pages was two hundred full downloads before the first one had
        // been looked at. `.gallery-item` is a fixed 80x80 box, so deferring the
        // ones off-screen costs no layout and no reflow when they arrive.
        img.loading = "lazy";
        img.decoding = "async";
        img.draggable = false;
        setImageSrc(img, currentPageUrl(file));

        const label = document.createElement("div");
        label.className = "gallery-index";
        label.innerText = String(index + 1);

        itemDiv.onclick = () => {
            if (!file) return;
            document.querySelectorAll(".gallery-item-container").forEach((el) => el.classList.remove("active"));
            itemDiv.classList.add("active");
            selectImage(file);
        };

        itemDiv.appendChild(img);
        itemDiv.appendChild(label);
        if (state.savedFiles.has(file)) ensureSavedBadge(itemDiv);
        gallery.appendChild(itemDiv);
    });

    updateVariantSwitch();
}

/**
 * Mark one gallery item as having a typeset copy in the destination folder.
 *
 * The badge is independent of the current variant: it says the saved copy
 * *exists*, not that it is what you are looking at.
 */
function ensureSavedBadge(itemDiv: HTMLElement): void {
    if (itemDiv.querySelector(".gallery-saved")) return;
    const badge = document.createElement("span");
    badge.className = "gallery-saved material-icons";
    badge.textContent = "check_circle";
    badge.title = "Saved — a typeset copy of this page is in the destination folder";
    itemDiv.appendChild(badge);
}

/**
 * Record that *file* now has a saved copy, and reflect it in the UI.
 *
 * Nothing is re-fetched that did not change. The write produced new bytes for
 * one page in one folder, so `pageBusters` is bumped for that page and
 * `refreshVariantImages` re-points the images whose URL that actually moves --
 * which, while the switch is on the original side and the destination is not the
 * source, is none of them: the original was not touched.
 */
export function markPageSaved(file: string): void {
    if (!file) {
        updateVariantSwitch();
        return;
    }
    pageBusters.set(file, ++saveTick);
    // Saving in place produces no second copy, so there is nothing to badge and
    // nothing for the Original/Saved switch to switch between -- but the page on
    // screen is now the stale one, which is why the bump above is unconditional.
    if (!state.sameDir) {
        state.savedFiles.add(file);
        const index = state.currentFiles.indexOf(file);
        const itemDiv = index >= 0 ? document.querySelectorAll<HTMLElement>(".gallery-item-container")[index] : null;
        if (itemDiv) ensureSavedBadge(itemDiv);
    }

    refreshVariantImages();
    updateVariantSwitch();
}

/**
 * The controls that change the page. They are locked while the saved copy is on
 * screen: the entries describe the original, and the saved page already has the
 * typeset text baked in, so acting on them there would be meaningless at best.
 */
const EDIT_BUTTON_IDS = [
    "alignmentCycleBtn",
    "inpaintGenerateBtn",
    "inpaintPreviewBtn",
    "clearPageBtn",
    "confirmTranslationBtn",
] as const;

/** Set while a capture owns the canvas -- see `suspendVariantSwitch`. */
let variantSwitchSuspended = false;

/**
 * Hide the switch for the duration of a capture, and bring it back after.
 *
 * `beginCapture`/`endCapture` in `ocr.ts` are the single choke point for every
 * capture (region, free-form, bubble, page and entry split), so wiring it there
 * covers all of them. The switch sits over the top-right of the page, which is
 * exactly where a capture stroke may need to start.
 */
export function suspendVariantSwitch(suspended: boolean): void {
    variantSwitchSuspended = suspended;
    updateVariantSwitch();
}

/** Whether *this* page has a saved copy to switch to. */
function variantSwitchUsable(): boolean {
    return !state.sameDir && state.savedFiles.has(state.currentImageFile);
}

/**
 * Put the read-only lock on, or take it off, to match the current variant.
 *
 * The body class is what hides the entry list and the region outlines and greys
 * the edit buttons out; `disabled` is set as well so the buttons cannot be
 * reached by keyboard either. The previous `disabled` state is remembered, so
 * unlocking never *enables* a button that something else had disabled for its
 * own reasons (`confirmTranslationBtn` with no captured region, say).
 */
function applySavedViewLocks(): void {
    const locked = isViewingSaved();
    document.body.classList.toggle("viewing-saved", locked);

    for (const id of EDIT_BUTTON_IDS) {
        const btn = document.getElementById(id) as HTMLButtonElement | null;
        if (!btn) continue;
        if (locked) {
            if (!btn.dataset.variantLock) btn.dataset.variantLock = btn.disabled ? "off" : "on";
            btn.disabled = true;
        } else if (btn.dataset.variantLock) {
            btn.disabled = btn.dataset.variantLock === "off";
            delete btn.dataset.variantLock;
        }
    }
}

/**
 * Show or hide the Original/Saved switch, and mark which side is active.
 *
 * Shown only when the page being looked at actually has a saved copy -- there is
 * nothing to switch to otherwise -- and never during a capture. `viewVariant`
 * itself is left alone as you move between pages so the choice sticks; it is
 * only forced back to `original` when the whole folder has nothing saved, or the
 * destination *is* the source and the two sides would be one file.
 */
export function updateVariantSwitch(): void {
    if (state.sameDir || state.savedFiles.size === 0) state.viewVariant = "original";

    const wrap = document.getElementById("variantSwitch");
    if (wrap) {
        wrap.style.display = variantSwitchUsable() && !variantSwitchSuspended ? "inline-flex" : "none";
        const original = document.getElementById("variantOriginalBtn");
        const saved = document.getElementById("variantSavedBtn");
        original?.classList.toggle("is-active", state.viewVariant === "original");
        saved?.classList.toggle("is-active", state.viewVariant === "saved");
    }

    applySavedViewLocks();
}

/**
 * Switch which copy of the pages is displayed.
 *
 * Re-fetches what is on screen and nothing else. `selectImage` is called
 * directly rather than clicking the gallery item, and no `.active` class is
 * touched, because that class is what `navigateGallery` reads to know where it
 * is -- driving it from here would move the reading position sideways.
 */
export function setViewVariant(variant: PageVariant): void {
    if (state.viewVariant === variant) return;
    if (variant === "saved" && !variantSwitchUsable()) return;
    state.viewVariant = variant;
    updateVariantSwitch();
    showVariantLoading(variant);
    refreshVariantImages();
    trackVariantLoad();
}

/** How long the swap spinner may stay up before it gives up on the `load` event. */
const VARIANT_LOAD_TIMEOUT_MS = 8000;

/** Bumped per swap, so a slow load cannot dismiss a newer one's spinner. */
let variantLoadToken = 0;

function showVariantLoading(variant: PageVariant): void {
    const loader = document.getElementById("variantLoader");
    if (!loader) return;
    const text = document.getElementById("variantLoaderText");
    if (text) text.textContent = variant === "saved" ? "Loading saved" : "Loading original";
    loader.style.display = "flex";
}

/**
 * Keep the spinner up until the newly pointed page has actually arrived.
 *
 * `addEventListener(..., { once: true })` rather than `onload`, because
 * `selectImage` owns `mainImg.onload` and assigning over it would break the fit
 * and the overlay sync. The timeout is a backstop: a spinner with no way to be
 * dismissed would be worse than showing none at all.
 */
function trackVariantLoad(): void {
    const token = ++variantLoadToken;
    const done = () => {
        if (token !== variantLoadToken) return;
        const loader = document.getElementById("variantLoader");
        if (loader) loader.style.display = "none";
    };

    const img = isStripMode()
        ? (getStripSlot(stripActiveIndex)?.querySelector<HTMLImageElement>(".strip-img") ?? null)
        : (document.getElementById("mainImage") as HTMLImageElement | null);

    // Already decoded (or nothing to wait for): do not flash the spinner.
    if (!img || (img.complete && img.naturalWidth > 0)) {
        done();
        return;
    }
    img.addEventListener("load", done, { once: true });
    img.addEventListener("error", done, { once: true });
    window.setTimeout(done, VARIANT_LOAD_TIMEOUT_MS);
}

/**
 * Re-point every visible page image at the current variant.
 *
 * Every image is *asked*, but only the ones whose URL actually moved are
 * re-assigned -- see {@link setImageSrc} and {@link busterFor}. Switching sides
 * on a folder with three saved pages is three requests, not one per page, and
 * switching back is none at all: those URLs are already in the cache.
 */
export function refreshVariantImages(): void {
    document.querySelectorAll<HTMLElement>(".gallery-item-container").forEach((itemDiv, index) => {
        const file = state.currentFiles[index];
        if (!file) return;
        setImageSrc(itemDiv.querySelector<HTMLImageElement>("img.gallery-item"), currentPageUrl(file));
    });

    document.querySelectorAll<HTMLImageElement>(".strip-img").forEach((img) => {
        const slot = img.closest<HTMLElement>(".strip-page");
        const index = Number(slot?.dataset.index ?? "-1");
        const file = state.currentFiles[index];
        if (!file) return;
        const url = currentPageUrl(file);
        // Slots the observer has not reached yet only carry `dataset.src`; they
        // pick the new URL up when they scroll into view.
        img.dataset.src = url;
        if (img.getAttribute("src")) setImageSrc(img, url);
    });

    if (!isStripMode()) refreshMainImage();
}

/**
 * Show the current variant of the page that is already on screen.
 *
 * Not `selectImage`, which is what this used to call: that is the *page change*
 * path and it resets the scroll, re-derives the fit and re-renders every entry
 * card and region. None of that applies to a variant swap -- it is the same page
 * at the same size, and the zoom and pan the reader had arrived at are theirs.
 * Only the pixels change.
 */
function refreshMainImage(): void {
    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    if (!mainImg || !state.currentImageFile) return;
    if (!setImageSrc(mainImg, currentPageUrl(state.currentImageFile))) return;
    keepViewOnLoad = true;
}

/**
 * Set while the next `load` is a swap of the page already on screen, and the
 * view it arrives into is the user's rather than something to re-derive.
 */
let keepViewOnLoad = false;

/** Re-apply the current zoom and pan, whichever mode owns them. */
function reapplyView(container: HTMLElement, mainImg: HTMLImageElement): void {
    if (isReaderMode()) {
        renderReader(container, mainImg);
        return;
    }
    applyWorkTransform(container, mainImg);
}

export function selectImage(file: string): void {
    const changed = file !== state.currentImageFile;
    state.currentImageFile = file;
    // Whether there is a saved copy to switch to is a per-page question, so this
    // has to be re-asked on every page change -- both branches below included.
    updateVariantSwitch();
    // Announced before anything is loaded, so whatever belongs to the page being
    // left goes with it rather than sitting over the top of the new one.
    if (changed) emit("page:changed", file);
    if (isStripMode()) {
        const idx = state.currentFiles.indexOf(file);
        if (idx >= 0 && idx !== stripActiveIndex) scrollStripTo(idx);
        updateGalleryActiveByIndex(idx >= 0 ? idx : 0);
        updateReaderPageIndicator();
        return;
    }
    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    const container = document.getElementById("imageContainer");
    if (!mainImg || !container) return;

    if (changed) {
        container.scrollLeft = 0;
        container.scrollTop = 0;
        container.style.overflow = "hidden";
        // A different page is always fitted, never restored: the flag belongs to
        // a variant swap on the page we are leaving, and a swap whose load
        // failed would otherwise leave it set for this one.
        keepViewOnLoad = false;
    }

    mainImg.style.display = "block";
    mainImg.onload = function () {
        imageErrorRetries = 0;
        ensureSvgOverlayExists();
        if (keepViewOnLoad) {
            keepViewOnLoad = false;
            reapplyView(container, mainImg);
            return;
        }
        applyImageFit(container, mainImg);
        emit("image:loaded", file);
    };

    if (setImageSrc(mainImg, currentPageUrl(file))) return;

    // Same page, same bytes, already decoded: no `load` is coming, so do here
    // what it would have done. Re-selecting the page on screen is not an idle
    // case -- every window resize and every panel toggle goes through
    // `navigateGallery(0)`, which is a click on the active thumbnail.
    if (mainImg.complete && mainImg.naturalWidth > 0) {
        ensureSvgOverlayExists();
        applyImageFit(container, mainImg);
        emit("image:loaded", file);
    }
}

/**
 * Re-applies the current fit (work or reader mode) against the container's
 * live size. Used on image load, panel resize, and reader mode toggle.
 */
export function applyImageFit(container: HTMLElement, mainImg: HTMLImageElement): void {
    if (isStripMode()) return;
    const containerHeight = container.clientHeight;
    const containerWidth = container.clientWidth;
    const naturalHeight = mainImg.naturalHeight;
    const naturalWidth = mainImg.naturalWidth;
    if (!naturalHeight || !naturalWidth) return;

    if (isReaderMode()) {
        state.zoomLevel =
            state.readerFit === "width" ? containerWidth / naturalWidth : containerHeight / naturalHeight;
        state.fitZoomLevel = state.zoomLevel;
        state.panX = 0;
        state.panY = 0;
        renderReader(container, mainImg);
        return;
    }

    // Horizontal fit fills the width and lets the page run off the bottom --
    // that overhang is the thing being scrolled through. The default fit shows
    // the whole page, so at the fit itself there is nothing to scroll and the
    // layout below turns the container's overflow off again.
    state.fitZoomLevel = state.isHorizontalFitEnabled
        ? containerWidth / naturalWidth
        : Math.min(containerWidth / naturalWidth, containerHeight / naturalHeight);
    state.zoomLevel = state.fitZoomLevel;
    applyWorkTransform(container, mainImg);
}

export function navigateGallery(direction: number): void {
    if (isStripMode()) {
        if (direction === 0) {
            scrollStripTo(stripActiveIndex);
        } else {
            scrollStripTo(stripActiveIndex + direction);
        }
        return;
    }
    const items = Array.from(document.querySelectorAll(".gallery-item-container"));
    if (items.length === 0) return;
    const currentIndex = items.findIndex((item) => item.classList.contains("active"));
    let nextIndex: number;
    if (direction === 1) {
        state.currentlySelectedEntryId = null;
        nextIndex = currentIndex + 1;
        if (nextIndex >= items.length) nextIndex = 0;
    } else if (direction === -1) {
        state.currentlySelectedEntryId = null;
        nextIndex = currentIndex - 1;
        if (nextIndex < 0) nextIndex = items.length - 1;
    } else {
        nextIndex = currentIndex;
    }
    (items[nextIndex] as HTMLElement).click();
    items[nextIndex].scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
}

let imageErrorRetries = 0;
const MAX_IMAGE_RETRIES = 3;

export function setupImageErrorHandler(): void {
    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    if (!mainImg) return;
    mainImg.onerror = () => {
        if (state.currentImageFile && imageErrorRetries < MAX_IMAGE_RETRIES) {
            imageErrorRetries++;
            // A distinct URL, so the retry cannot be answered from the cache
            // that just failed -- and through `setImageSrc`, so the stamp keeps
            // saying what the element is really pointed at.
            setImageSrc(mainImg, `${currentPageUrl(state.currentImageFile)}&retry=${imageErrorRetries}`);
        }
    };
}

/** Movement of the page per pixel of cursor movement while dragging. */
const PAN_SPEED = 1;

/** How far a press may travel before it stops counting as a click. */
const PAN_CLICK_SLOP = 4;

export function initPanning(): void {
    const container = document.getElementById("imageContainer");
    if (!container) return;

    let activePointer: number | null = null;
    let startClientX = 0;
    let startClientY = 0;
    let startScrollLeft = 0;
    let startScrollTop = 0;
    let startPanX = 0;
    let startPanY = 0;
    let dragged = false;

    const stop = () => {
        if (activePointer === null) return;
        if (container.hasPointerCapture?.(activePointer)) container.releasePointerCapture(activePointer);
        activePointer = null;
        state.isPanning = false;
        updateCursor();
    };

    container.addEventListener("pointerdown", (e) => {
        if (state.isCaptureMode || isStripMode()) return;
        // The eyedropper is a press on the page too, and one that must not also
        // move it: a pan would slide the pixel out from under the cursor and
        // then eat the click that was going to sample it.
        if (document.body.classList.contains("eyedropper-active")) return;
        // Touch scrolls the container by itself, and taking the pointer here
        // would replace that with a worse version of it.
        if (e.pointerType === "touch") return;
        // Primary button only. The middle one is the browser's autoscroll and
        // the right one opens the region menu, which a pan starting under the
        // cursor would fight.
        if (e.button !== 0) return;
        if (activePointer !== null) return;
        if (!isPannable(container)) return;

        activePointer = e.pointerId;
        dragged = false;
        startClientX = e.clientX;
        startClientY = e.clientY;
        startScrollLeft = container.scrollLeft;
        startScrollTop = container.scrollTop;
        startPanX = state.panX;
        startPanY = state.panY;
        state.isPanning = true;
        updateCursor();
        // Deliberately no `preventDefault` and no pointer capture here. The
        // first would suppress the compatibility mouse events, and with them
        // every click on a region; the second retargets those same events to
        // this container, so the region under the cursor would never see the
        // click either. Capture is taken below, once this is known to be a drag
        // and the click is going to be thrown away anyway.
    });

    container.addEventListener("pointermove", (e) => {
        if (activePointer !== e.pointerId || !state.isPanning) return;
        // The button was released somewhere we never heard about -- over the
        // browser's own chrome, or in another window. Without this the press
        // stays open and the page follows the cursor with nothing held down.
        if (e.buttons === 0) {
            stop();
            return;
        }
        if (state.isCaptureMode) {
            stop();
            return;
        }
        const dx = (e.clientX - startClientX) * PAN_SPEED;
        const dy = (e.clientY - startClientY) * PAN_SPEED;
        // A press is allowed a few pixels of tremor before it becomes a drag,
        // so clicking a region does not have to be perfectly still.
        if (!dragged && Math.abs(dx) < PAN_CLICK_SLOP && Math.abs(dy) < PAN_CLICK_SLOP) return;
        if (!dragged) {
            dragged = true;
            // From here the drag owns the pointer: it survives leaving the
            // viewport, and the release is delivered even if it happens over
            // another element entirely.
            container.setPointerCapture?.(e.pointerId);
        }
        e.preventDefault();

        if (isReaderMode()) {
            const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
            if (mainImg) setReaderPan(container, mainImg, startPanX + dx, startPanY + dy);
            return;
        }
        // Scroll offsets run the opposite way to the content: dragging right
        // means showing what is further left.
        container.scrollLeft = startScrollLeft - dx;
        container.scrollTop = startScrollTop - dy;
    });

    // On `window`, so a release that lands outside the container still ends the
    // press. Captured or not, the event reaches here: capture retargets it to
    // the container, from which it bubbles.
    const end = (e: PointerEvent) => {
        if (activePointer !== e.pointerId) return;
        const wasDrag = dragged;
        stop();
        if (wasDrag) swallowNextClick();
    };
    window.addEventListener("pointerup", end);
    window.addEventListener("pointercancel", end);
}

/**
 * Drop the click a finished drag is about to produce.
 *
 * A pan that starts on a region ends with `click` on that region -- the event
 * is derived from where the press and the release landed, not from whether
 * anything moved in between -- so without this every drag across the page also
 * selected whatever was under the cursor when it began.
 *
 * Capture phase on `window` so it runs ahead of the region's own listener, and
 * armed for exactly one turn of the event loop: press and release on different
 * elements produce no click at all, and a listener left behind would eat the
 * next real one.
 */
function swallowNextClick(): void {
    const swallow = (event: Event) => {
        event.preventDefault();
        event.stopImmediatePropagation();
    };
    window.addEventListener("click", swallow, true);
    window.setTimeout(() => window.removeEventListener("click", swallow, true), 0);
}

/**
 * A wheel notch as a *proportion* of the current zoom.
 *
 * A fixed step cannot suit both ends. On a page fitted at 0.3 a step of 0.1 is a
 * third of the way in; on one blown up to 3 it is barely visible. Exponentiating
 * makes every notch the same visual step, and keeps a trackpad's small deltas
 * small instead of rounding them up into jumps.
 */
function zoomStep(deltaY: number): number {
    return state.zoomLevel * (Math.exp(-deltaY * 0.001) - 1);
}

export function initZoom(): void {
    const container = document.getElementById("imageContainer");
    if (!container) return;

    container.addEventListener(
        "wheel",
        (e) => {
            if (isStripMode()) {
                // Ctrl is the zoom in the strip; a plain wheel is the read.
                if (e.ctrlKey) {
                    e.preventDefault();
                    zoomStripBy(e.deltaY * -0.001);
                }
                return;
            }

            const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
            if (!mainImg || !mainImg.naturalWidth || !mainImg.naturalHeight) return;

            // A horizontal fit is a fixed zoom (see `zoomBounds`), so there the
            // wheel is the read instead: work mode scrolls the overhang
            // natively, and the reader -- whose overflow is off because it
            // carries its own pan -- is moved by the same amount by hand. Ctrl
            // is no exception; there is no zoom left for it to give. The
            // default fit has nothing to scroll until it is zoomed past the
            // viewport, so there the wheel is the zoom and panning is the drag.
            if (isHorizontalFit()) {
                if (!isReaderMode()) return;
                e.preventDefault();
                setReaderPan(container, mainImg, state.panX, state.panY - e.deltaY);
                return;
            }

            e.preventDefault();
            const rect = container.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            if (isReaderMode()) {
                zoomReaderAround(container, mainImg, mx, my, zoomStep(e.deltaY));
            } else {
                zoomWorkAround(container, mainImg, mx, my, zoomStep(e.deltaY));
            }
        },
        { passive: false },
    );

    const gallery = document.getElementById("gallery");
    if (gallery) {
        gallery.addEventListener(
            "wheel",
            (e) => {
                if (e.deltaY !== 0) {
                    e.preventDefault();
                    gallery.scrollLeft += e.deltaY;
                }
            },
            { passive: false },
        );
    }

    const mainImg = document.getElementById("mainImage");
    if (mainImg) {
        mainImg.addEventListener("dragstart", (e) => e.preventDefault());
    }
}

export function initResize(): void {
    let resizeTimeout: ReturnType<typeof setTimeout>;
    window.addEventListener("resize", () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (state.currentImageFile) {
                navigateGallery(0);
            }
        }, 100);
    });
}

/** Re-fits the current image after a workspace panel has been resized. */
export function refitViewer(): void {
    if (!state.currentImageFile) return;
    if (isStripMode()) {
        resetStripZoom();
        return;
    }
    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    const container = document.getElementById("imageContainer");
    if (!mainImg || !container || mainImg.style.display === "none") return;
    applyImageFit(container, mainImg);
}

/** Updates the reader page indicator (e.g. "3 / 24") from the gallery items. */
export function updateReaderPageIndicator(): void {
    const el = document.getElementById("readerPageIndicator");
    if (!el) return;
    const items = Array.from(document.querySelectorAll(".gallery-item-container"));
    if (items.length === 0) {
        el.textContent = "-";
        return;
    }
    const current = items.findIndex((item) => item.classList.contains("active"));
    el.textContent = `${current < 0 ? 1 : current + 1} / ${items.length}`;
}
