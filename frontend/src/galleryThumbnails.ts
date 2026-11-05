import { imageRenderQualityEnabled, isNativeViewer } from "./imageRenderQuality";

const MAX_DIMENSION = 1024;
const MAX_IN_FLIGHT = 4;
const RESIZE_DELAY_MS = 150;
const LOAD_TIMEOUT_MS = 10000;

interface Thumbnail {
    image: HTMLImageElement;
    original: string;
    visible: boolean;
    failedOriginal: string | null;
    target: Element | null;
    requested: string | null;
    requestedOriginal: string | null;
    timeout: number;
    onLoad: () => void;
    onError: () => void;
}

export interface GalleryThumbnails {
    /** May be called while constructing a detached gallery fragment. */
    update(image: HTMLImageElement, originalUrl: string): void;
    /** Call once after inserting the fragment into the gallery. */
    start(): void;
    /** Coalesce panel, layout or display-density changes. */
    refresh(): void;
    dispose(): void;
}

function setSource(image: HTMLImageElement, url: string): void {
    if (image.dataset.foxSrc === url && image.getAttribute("src") === url) return;
    image.dataset.foxSrc = url;
    image.src = url;
}

function density(): number {
    const value = window.devicePixelRatio;
    return Number.isFinite(value) && value > 0 ? value : 1;
}

function pixels(value: string): number {
    return Number.parseFloat(value) || 0;
}

/** Use the painted content box, not borders, and retain fractional CSS pixels. */
function thumbnailUrl(entry: Thumbnail): string | null {
    const image = entry.image;
    const box = image.getBoundingClientRect();
    const style = getComputedStyle(image);
    const width = Math.round(
        (box.width -
            pixels(style.borderLeftWidth) -
            pixels(style.borderRightWidth) -
            pixels(style.paddingLeft) -
            pixels(style.paddingRight)) *
            density(),
    );
    const height = Math.round(
        (box.height -
            pixels(style.borderTopWidth) -
            pixels(style.borderBottomWidth) -
            pixels(style.paddingTop) -
            pixels(style.paddingBottom)) *
            density(),
    );
    // A hidden or not-yet-laid-out rail is measured again when it becomes visible.
    if (!(width > 0 && height > 0)) return null;
    // Very large rails/displays keep the original instead of receiving a preview
    // with too few pixels, or asking the backend to allocate an unbounded image.
    if (width > MAX_DIMENSION || height > MAX_DIMENSION) return entry.original;
    try {
        const url = new URL(entry.original, window.location.href);
        if (url.origin !== window.location.origin || !url.pathname.startsWith("/img_serve/")) {
            return entry.original;
        }
        url.pathname = url.pathname.replace("/img_serve/", "/api/thumbnail/");
        url.searchParams.set("width", String(width));
        url.searchParams.set("height", String(height));
        url.searchParams.set("fit", style.objectFit === "cover" ? "cover" : "contain");
        url.searchParams.set("viewer", isNativeViewer() ? "native" : "browser");
        return url.pathname + url.search;
    } catch {
        return entry.original;
    }
}

/**
 * One manager per gallery, with one visibility observer and a bounded load queue.
 * The disabled path retains the old original-image URLs and creates no observers,
 * event listeners or timers. Main images never pass through this module.
 */
export function createGalleryThumbnails(gallery: HTMLElement): GalleryThumbnails {
    const enabled = imageRenderQualityEnabled() && typeof IntersectionObserver !== "undefined";
    const entries = new Map<HTMLImageElement, Thumbnail>();
    const targets = new Map<Element, Thumbnail>();
    const visible = new Set<Thumbnail>();
    const queued = new Set<Thumbnail>();
    const active = new Set<Thumbnail>();
    const awaitingSize = new Set<Thumbnail>();
    let started = false;
    let disposed = false;
    let flushing = false;
    let refreshTimer = 0;
    let intersection: IntersectionObserver | null = null;
    let resize: ResizeObserver | null = null;
    let resolution: MediaQueryList | null = null;
    let watchedDensity = 0;

    function release(entry: Thumbnail): void {
        window.clearTimeout(entry.timeout);
        entry.timeout = 0;
        active.delete(entry);
        entry.requested = null;
        entry.requestedOriginal = null;
    }

    function finish(entry: Thumbnail, failed: boolean): void {
        if (disposed || !active.has(entry)) return;
        const requested = entry.requested;
        const original = entry.requestedOriginal;
        release(entry);
        if (failed && original === entry.original && entry.image.getAttribute("src") === requested) {
            entry.failedOriginal = original;
            setSource(entry.image, entry.original);
        }
        // A save, variant switch or resize while loading keeps just the latest
        // desired URL. It does not open more requests until this one completes.
        if (entry.visible) queued.add(entry);
        flush();
    }

    function imageFinished(entry: Thumbnail, failed: boolean): void {
        const image = entry.image;
        // An event queued before a timeout fallback or source replacement must
        // not complete a newer request that is still loading in the same <img>.
        if (!image.complete || (failed ? image.naturalWidth !== 0 : image.naturalWidth === 0)) return;
        if (
            entry.requested &&
            image.currentSrc &&
            image.currentSrc !== new URL(entry.requested, window.location.href).href
        )
            return;
        finish(entry, failed);
    }

    function stopWaitingForSize(entry: Thumbnail): void {
        if (awaitingSize.delete(entry)) resize?.unobserve(entry.image);
    }

    function flush(): void {
        if (disposed || !started || flushing || refreshTimer) return;
        flushing = true;
        try {
            for (const entry of queued) {
                if (active.size >= MAX_IN_FLIGHT) break;
                queued.delete(entry);
                if (!entry.visible || !gallery.contains(entry.image) || active.has(entry)) continue;
                const url = entry.failedOriginal === entry.original ? entry.original : thumbnailUrl(entry);
                if (!url) {
                    // A skipped descendant can gain size without resizing the
                    // gallery. Watch only these visible, unmeasurable images;
                    // the ordinary rail still has one shared resize target.
                    if (resize) {
                        if (!awaitingSize.has(entry)) {
                            awaitingSize.add(entry);
                            resize.observe(entry.image);
                        }
                    } else {
                        setSource(entry.image, entry.original);
                    }
                    continue;
                }
                stopWaitingForSize(entry);
                if (entry.image.getAttribute("src") === url) continue;
                if (url === entry.original) {
                    setSource(entry.image, url);
                    continue;
                }
                active.add(entry);
                entry.requested = url;
                entry.requestedOriginal = entry.original;
                entry.timeout = window.setTimeout(() => finish(entry, true), LOAD_TIMEOUT_MS);
                setSource(entry.image, url);
            }
        } finally {
            flushing = false;
        }
    }

    function observe(entry: Thumbnail): void {
        // Observe the wrapper: content-visibility may skip layout of its image,
        // whereas the wrapper retains its intrinsic slot in the scroll rail.
        const target = entry.image.closest(".gallery-item-container") || entry.image;
        entry.target = target;
        targets.set(target, entry);
        intersection?.observe(target);
    }

    function refresh(): void {
        if (!enabled || !started || disposed) return;
        window.clearTimeout(refreshTimer);
        refreshTimer = window.setTimeout(() => {
            refreshTimer = 0;
            for (const entry of visible) queued.add(entry);
            flush();
        }, RESIZE_DELAY_MS);
    }

    function stopWatchingDensity(): void {
        if (!resolution) return;
        if (typeof resolution.removeEventListener === "function")
            resolution.removeEventListener("change", onDensityChange);
        else resolution.removeListener?.(onDensityChange);
        resolution = null;
    }

    function watchDensity(): void {
        if (typeof window.matchMedia !== "function") return;
        const next = density();
        if (resolution && next === watchedDensity) return;
        stopWatchingDensity();
        watchedDensity = next;
        resolution = window.matchMedia(`(resolution: ${next}dppx)`);
        if (typeof resolution.addEventListener === "function") resolution.addEventListener("change", onDensityChange);
        else resolution.addListener?.(onDensityChange);
    }

    function onDensityChange(): void {
        if (disposed) return;
        watchDensity();
        refresh();
    }

    return {
        update(image, originalUrl) {
            if (disposed) return;
            if (!enabled) {
                setSource(image, originalUrl);
                return;
            }
            let entry = entries.get(image);
            if (!entry) {
                entry = {
                    image,
                    original: originalUrl,
                    visible: false,
                    failedOriginal: null,
                    target: null,
                    requested: null,
                    requestedOriginal: null,
                    timeout: 0,
                    onLoad: () => imageFinished(entry!, false),
                    onError: () => imageFinished(entry!, true),
                };
                entries.set(image, entry);
                image.addEventListener("load", entry.onLoad);
                image.addEventListener("error", entry.onError);
                if (started) observe(entry);
            } else if (entry.original !== originalUrl) {
                entry.original = originalUrl;
                entry.failedOriginal = null;
            }
            if (entry.visible) {
                queued.add(entry);
                flush();
            }
        },
        start() {
            if (!enabled || started || disposed) return;
            started = true;
            intersection = new IntersectionObserver(
                (changes) => {
                    if (disposed) return;
                    for (const change of changes) {
                        const entry = targets.get(change.target);
                        if (!entry) continue;
                        entry.visible = change.isIntersecting;
                        if (entry.visible) {
                            visible.add(entry);
                            queued.add(entry);
                        } else {
                            visible.delete(entry);
                            queued.delete(entry);
                            stopWaitingForSize(entry);
                        }
                    }
                    flush();
                },
                { root: gallery, rootMargin: "200px" },
            );
            for (const entry of entries.values()) observe(entry);
            if (typeof ResizeObserver !== "undefined") {
                resize = new ResizeObserver(refresh);
                resize.observe(gallery);
            }
            watchDensity();
            window.addEventListener("resize", onDensityChange);
        },
        refresh,
        dispose() {
            if (disposed) return;
            disposed = true;
            intersection?.disconnect();
            resize?.disconnect();
            stopWatchingDensity();
            if (started) window.removeEventListener("resize", onDensityChange);
            window.clearTimeout(refreshTimer);
            refreshTimer = 0;
            for (const entry of entries.values()) {
                entry.image.removeEventListener("load", entry.onLoad);
                entry.image.removeEventListener("error", entry.onError);
                release(entry);
            }
            entries.clear();
            targets.clear();
            visible.clear();
            queued.clear();
            awaitingSize.clear();
        },
    };
}
