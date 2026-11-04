import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createGalleryThumbnails } from "./galleryThumbnails";
import type { GalleryThumbnails } from "./galleryThumbnails";

const policy = vi.hoisted(() => ({ mode: 1, native: true }));
vi.mock("./imageRenderQuality", () => ({
    imageRenderQualityEnabled: () => policy.mode === 2 || (policy.mode === 1 && policy.native),
    isNativeViewer: () => policy.native,
}));

class Visibility {
    static instances: Visibility[] = [];
    observe = vi.fn();
    disconnect = vi.fn();
    callback: IntersectionObserverCallback;
    options: IntersectionObserverInit;
    constructor(callback: IntersectionObserverCallback, options: IntersectionObserverInit) {
        this.callback = callback;
        this.options = options;
        Visibility.instances.push(this);
    }
    show(images: HTMLImageElement[], visible = true): void {
        this.callback(
            images.map((image) => ({
                target: image.parentElement,
                isIntersecting: visible,
            })) as unknown as IntersectionObserverEntry[],
            this as unknown as IntersectionObserver,
        );
    }
}

class Sizes {
    static instances: Sizes[] = [];
    observed = new Set<Element>();
    observe = vi.fn((target: Element) => this.observed.add(target));
    unobserve = vi.fn((target: Element) => this.observed.delete(target));
    disconnect = vi.fn(() => this.observed.clear());
    callback: ResizeObserverCallback;
    constructor(callback: ResizeObserverCallback) {
        this.callback = callback;
        Sizes.instances.push(this);
    }
    changed(target?: Element): void {
        if (target && !this.observed.has(target)) return;
        this.callback([], this as unknown as ResizeObserver);
    }
}

let managers: GalleryThumbnails[];
let imageResults: WeakMap<HTMLImageElement, { url: string | null; width: number }>;
let media: {
    change?: () => void;
    addEventListener: ReturnType<typeof vi.fn>;
    removeEventListener: ReturnType<typeof vi.fn>;
}[];

beforeEach(() => {
    vi.useFakeTimers();
    policy.mode = 1;
    policy.native = true;
    Visibility.instances = [];
    Sizes.instances = [];
    managers = [];
    imageResults = new WeakMap();
    media = [];
    document.body.innerHTML = "<div id='gallery'></div>";
    vi.stubGlobal("IntersectionObserver", Visibility);
    vi.stubGlobal("ResizeObserver", Sizes);
    vi.stubGlobal("devicePixelRatio", 1.25);
    vi.stubGlobal(
        "matchMedia",
        vi.fn(() => {
            const item = {
                change: undefined as (() => void) | undefined,
                addEventListener: vi.fn((_event: string, callback: () => void) => {
                    item.change = callback;
                }),
                removeEventListener: vi.fn(),
            };
            media.push(item);
            return item;
        }),
    );
});

afterEach(() => {
    managers.forEach((control) => control.dispose());
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
});

function manager(): GalleryThumbnails {
    const result = createGalleryThumbnails(document.getElementById("gallery")!);
    managers.push(result);
    return result;
}

function thumb(control: GalleryThumbnails, url = "/img_serve/p1.png?t=7", width = 102, height = 152): HTMLImageElement {
    const image = document.createElement("img");
    const result = { url: null as string | null, width: 0 };
    imageResults.set(image, result);
    Object.defineProperties(image, {
        complete: { get: () => result.url === image.getAttribute("src") },
        naturalWidth: { get: () => (result.url === image.getAttribute("src") ? result.width : 0) },
        currentSrc: { get: () => (result.url ? new URL(result.url, window.location.href).href : "") },
    });
    image.loading = "lazy";
    image.style.cssText = "border:1px solid transparent;object-fit:contain";
    vi.spyOn(image, "getBoundingClientRect").mockReturnValue(new window.DOMRect(0, 0, width, height));
    // This is how renderGallery builds its detached fragment: the image has no
    // parent when registered, then its wrapper is inserted before start().
    control.update(image, url);
    const wrapper = document.createElement("div");
    wrapper.className = "gallery-item-container";
    wrapper.append(image);
    document.getElementById("gallery")!.append(wrapper);
    return image;
}

function params(image: HTMLImageElement): URLSearchParams {
    return new URL(image.src).searchParams;
}

function loaded(image: HTMLImageElement): void {
    Object.assign(imageResults.get(image)!, { url: image.getAttribute("src"), width: 100 });
    image.dispatchEvent(new Event("load"));
}

function failed(image: HTMLImageElement): void {
    Object.assign(imageResults.get(image)!, { url: image.getAttribute("src"), width: 0 });
    image.dispatchEvent(new Event("error"));
}

describe("gallery thumbnail policy", () => {
    it.each([
        [1, false],
        [3, true],
        [3, false],
    ])("keeps original rendering without observers or timers for mode %i, native %s", (mode, native) => {
        policy.mode = mode;
        policy.native = native;
        const control = manager();
        const image = thumb(control);
        control.start();
        control.refresh();
        window.dispatchEvent(new Event("resize"));
        expect(image.getAttribute("src")).toBe("/img_serve/p1.png?t=7");
        expect(Visibility.instances).toHaveLength(0);
        expect(Sizes.instances).toHaveLength(0);
        expect(media).toHaveLength(0);
        expect(vi.getTimerCount()).toBe(0);
        expect(image.getBoundingClientRect).not.toHaveBeenCalled();
    });

    it.each([
        [1, true, "native"],
        [2, true, "native"],
        [2, false, "browser"],
    ])("uses previews for mode %i, native %s", (mode, native, viewer) => {
        policy.mode = mode as number;
        policy.native = native as boolean;
        const control = manager();
        const image = thumb(control);
        control.start();
        Visibility.instances[0].show([image]);
        expect(params(image).get("viewer")).toBe(viewer);
    });

    it("falls back to browser-lazy originals when visibility observation is unavailable", () => {
        vi.stubGlobal("IntersectionObserver", undefined);
        const control = manager();
        const image = thumb(control);
        control.start();
        expect(image.getAttribute("src")).toBe("/img_serve/p1.png?t=7");
        expect(image.loading).toBe("lazy");
        expect(Sizes.instances).toHaveLength(0);
        expect(vi.getTimerCount()).toBe(0);
    });
});

describe("gallery preview selection", () => {
    it("waits for insertion and only loads visible or near-visible wrappers", () => {
        const control = manager();
        const first = thumb(control);
        const distant = thumb(control, "/img_serve/p2.png?t=7");
        expect(first.hasAttribute("src")).toBe(false);
        expect(Visibility.instances).toHaveLength(0);
        control.start();
        expect(Visibility.instances[0].options).toEqual({
            root: document.getElementById("gallery"),
            rootMargin: "200px",
        });
        expect(Visibility.instances[0].observe).toHaveBeenCalledWith(first.parentElement);
        Visibility.instances[0].show([first]);
        expect(first.getAttribute("src")).toContain("/api/thumbnail/p1.png?");
        expect(distant.hasAttribute("src")).toBe(false);
        control.refresh();
        vi.advanceTimersByTime(150);
        expect(distant.getBoundingClientRect).not.toHaveBeenCalled();
    });

    it("preserves encoded filenames and revisions while sizing the content box at display density", () => {
        const control = manager();
        const image = thumb(control, "/img_serve/1%2B2%20(a%23b).png?t=123.4&variant=saved", 106, 156);
        image.style.padding = "2px";
        image.style.objectFit = "cover";
        control.start();
        Visibility.instances[0].show([image]);
        expect(new URL(image.src).pathname).toBe("/api/thumbnail/1%2B2%20(a%23b).png");
        expect(Object.fromEntries(params(image))).toEqual({
            t: "123.4",
            variant: "saved",
            width: "125",
            height: "188",
            fit: "cover",
            viewer: "native",
        });
        expect(image.loading).toBe("lazy");
    });

    it("retains fractional sizes and uses contain for the page-shaped rail", () => {
        const control = manager();
        const image = thumb(control, undefined, 102.4, 152.4);
        control.start();
        Visibility.instances[0].show([image]);
        expect(params(image).get("width")).toBe("126");
        expect(params(image).get("height")).toBe("188");
        expect(params(image).get("fit")).toBe("contain");
    });

    it("uses originals for oversized targets and waits for a measurable hidden rail", () => {
        const control = manager();
        const giant = thumb(control, "/img_serve/large.png?t=7", 2000, 3000);
        const hidden = thumb(control, "/img_serve/hidden.png?t=7", 0, 0);
        control.start();
        Visibility.instances[0].show([giant, hidden]);
        expect(giant.getAttribute("src")).toBe("/img_serve/large.png?t=7");
        expect(hidden.hasAttribute("src")).toBe(false);
        vi.mocked(hidden.getBoundingClientRect).mockReturnValue(new window.DOMRect(0, 0, 102, 152));
        Sizes.instances[0].changed();
        vi.advanceTimersByTime(150);
        expect(hidden.getAttribute("src")).toContain("/api/thumbnail/hidden.png");
    });

    it("loads an unskipped image even when the gallery itself never resizes", () => {
        const control = manager();
        const image = thumb(control, "/img_serve/hidden.png?t=7", 0, 0);
        const distant = thumb(control, "/img_serve/distant.png?t=7", 0, 0);
        control.start();
        Visibility.instances[0].show([image]);
        expect(Sizes.instances[0].observed).toEqual(new Set([document.getElementById("gallery"), image]));
        expect(image.hasAttribute("src")).toBe(false);
        vi.mocked(image.getBoundingClientRect).mockReturnValue(new window.DOMRect(0, 0, 102, 152));
        // Notify only the descendant. This cannot wake a manager that observes
        // just the unchanged gallery box.
        Sizes.instances[0].changed(image);
        vi.advanceTimersByTime(150);
        expect(image.getAttribute("src")).toContain("/api/thumbnail/hidden.png");
        expect(Sizes.instances[0].unobserve).toHaveBeenCalledWith(image);
        expect(Sizes.instances[0].observed).toEqual(new Set([document.getElementById("gallery")]));
        expect(distant.hasAttribute("src")).toBe(false);
    });

    it("stops waiting for an unmeasurable image when it leaves the visible rail", () => {
        const control = manager();
        const image = thumb(control, undefined, 0, 0);
        control.start();
        Visibility.instances[0].show([image]);
        expect(Sizes.instances[0].observed.has(image)).toBe(true);
        Visibility.instances[0].show([image], false);
        expect(Sizes.instances[0].unobserve).toHaveBeenCalledWith(image);
        vi.mocked(image.getBoundingClientRect).mockReturnValue(new window.DOMRect(0, 0, 102, 152));
        Sizes.instances[0].changed(image);
        vi.advanceTimersByTime(150);
        expect(image.hasAttribute("src")).toBe(false);
        Visibility.instances[0].show([image]);
        expect(image.getAttribute("src")).toContain("/api/thumbnail/p1.png");
    });

    it("uses the original when an unmeasurable image cannot be watched for size", () => {
        vi.stubGlobal("ResizeObserver", undefined);
        const control = manager();
        const image = thumb(control, undefined, 0, 0);
        control.start();
        Visibility.instances[0].show([image]);
        expect(image.getAttribute("src")).toBe("/img_serve/p1.png?t=7");
        expect(vi.getTimerCount()).toBe(0);
    });

    it("keeps only the latest saved revision for a page outside the visible rail", () => {
        const control = manager();
        const image = thumb(control);
        control.start();
        control.update(image, "/img_serve/p1.png?t=7.1&variant=saved");
        control.update(image, "/img_serve/p1.png?t=7.2&variant=saved");
        expect(image.hasAttribute("src")).toBe(false);
        Visibility.instances[0].show([image]);
        expect(params(image).get("t")).toBe("7.2");
        expect(params(image).get("variant")).toBe("saved");
        loaded(image);
        const src = vi.spyOn(image, "src", "set");
        control.update(image, "/img_serve/p1.png?t=7.2&variant=saved");
        expect(src).not.toHaveBeenCalled();
    });
});

describe("bounded preview work", () => {
    it("limits active loads and advances the visible queue as images finish", () => {
        const control = manager();
        const images = Array.from({ length: 9 }, (_, i) => thumb(control, `/img_serve/${i}.png?t=7`));
        control.start();
        Visibility.instances[0].show(images);
        expect(images.filter((image) => image.hasAttribute("src"))).toHaveLength(4);
        loaded(images[0]);
        expect(images.filter((image) => image.hasAttribute("src"))).toHaveLength(5);
        Visibility.instances[0].show([images[5]], false);
        loaded(images[1]);
        expect(images[5].hasAttribute("src")).toBe(false);
        expect(images[6].hasAttribute("src")).toBe(true);
    });

    it("coalesces resize gestures and updates only the final dimensions", () => {
        const control = manager();
        const image = thumb(control);
        control.start();
        Visibility.instances[0].show([image]);
        loaded(image);
        const src = vi.spyOn(image, "src", "set");
        for (let width = 120; width < 200; width += 10) {
            vi.mocked(image.getBoundingClientRect).mockReturnValue(new window.DOMRect(0, 0, width, 152));
            Sizes.instances[0].changed();
            vi.advanceTimersByTime(30);
        }
        expect(src).not.toHaveBeenCalled();
        vi.advanceTimersByTime(150);
        expect(src).toHaveBeenCalledTimes(1);
        expect(params(image).get("width")).toBe("235");
    });

    it("detects a monitor density change even when CSS viewport dimensions stay unchanged", () => {
        const control = manager();
        const image = thumb(control);
        control.start();
        Visibility.instances[0].show([image]);
        loaded(image);
        vi.stubGlobal("devicePixelRatio", 2);
        media[0].change!();
        vi.advanceTimersByTime(150);
        expect(params(image).get("width")).toBe("200");
        expect(params(image).get("height")).toBe("300");
        expect(media[0].removeEventListener).toHaveBeenCalled();
        expect(window.matchMedia).toHaveBeenLastCalledWith("(resolution: 2dppx)");
    });

    it("waits for resizing to settle before loading newly visible thumbnails", () => {
        const control = manager();
        const image = thumb(control);
        control.start();
        Sizes.instances[0].changed();
        Visibility.instances[0].show([image]);
        expect(image.hasAttribute("src")).toBe(false);
        vi.mocked(image.getBoundingClientRect).mockReturnValue(new window.DOMRect(0, 0, 202, 302));
        Sizes.instances[0].changed();
        vi.advanceTimersByTime(150);
        expect(params(image).get("width")).toBe("250");
    });

    it("does not multiply requests when a loading page is saved or resized repeatedly", () => {
        const control = manager();
        const image = thumb(control);
        control.start();
        Visibility.instances[0].show([image]);
        const src = vi.spyOn(image, "src", "set");
        control.update(image, "/img_serve/p1.png?t=7.1&variant=saved");
        control.update(image, "/img_serve/p1.png?t=7.2&variant=saved");
        vi.mocked(image.getBoundingClientRect).mockReturnValue(new window.DOMRect(0, 0, 202, 302));
        control.refresh();
        vi.advanceTimersByTime(150);
        expect(src).not.toHaveBeenCalled();
        loaded(image);
        expect(src).toHaveBeenCalledTimes(1);
        expect(params(image).get("t")).toBe("7.2");
        expect(params(image).get("width")).toBe("250");
    });

    it("falls back once on error and only retries previews for a changed source", () => {
        const control = manager();
        const image = thumb(control);
        control.start();
        Visibility.instances[0].show([image]);
        failed(image);
        expect(image.getAttribute("src")).toBe("/img_serve/p1.png?t=7");
        const src = vi.spyOn(image, "src", "set");
        image.dispatchEvent(new Event("error"));
        control.refresh();
        vi.advanceTimersByTime(150);
        expect(src).not.toHaveBeenCalled();
        control.update(image, "/img_serve/p1.png?t=7.1");
        expect(image.getAttribute("src")).toContain("/api/thumbnail/p1.png?t=7.1");
    });

    it("releases stalled loads and allows later visible images to proceed", () => {
        const control = manager();
        const images = Array.from({ length: 5 }, (_, i) => thumb(control, `/img_serve/${i}.png?t=7`));
        control.start();
        Visibility.instances[0].show(images);
        expect(images[4].hasAttribute("src")).toBe(false);
        vi.advanceTimersByTime(10000);
        expect(images[0].getAttribute("src")).toBe("/img_serve/0.png?t=7");
        expect(images[4].getAttribute("src")).toContain("/api/thumbnail/4.png");
    });

    it("ignores an abandoned request's late events while a new revision is pending", () => {
        const control = manager();
        const images = Array.from({ length: 5 }, (_, i) => thumb(control, `/img_serve/${i}.png?t=7`));
        control.start();
        Visibility.instances[0].show(images);
        vi.advanceTimersByTime(10000);
        // The first four loads fell back; page five is now loading. Start three
        // updated pages, returning the thumbnail queue to its four-slot limit.
        for (const image of images.slice(0, 3)) {
            control.update(image, "/img_serve/new.png?t=8");
        }
        control.update(images[3], "/img_serve/waiting.png?t=8");
        const pending = images[3].getAttribute("src");
        images[0].dispatchEvent(new Event("load"));
        images[0].dispatchEvent(new Event("error"));
        expect(images[3].getAttribute("src")).toBe(pending);
        expect(params(images[0]).get("t")).toBe("8");
        loaded(images[0]);
        expect(images[3].getAttribute("src")).toContain("/api/thumbnail/waiting.png?t=8");
    });

    it("disposes queued work, observers and timers when a folder is replaced", () => {
        const control = manager();
        const images = Array.from({ length: 5 }, (_, i) => thumb(control, `/img_serve/${i}.png?t=7`));
        control.start();
        Visibility.instances[0].show(images);
        control.refresh();
        control.dispose();
        expect(Visibility.instances[0].disconnect).toHaveBeenCalled();
        expect(Sizes.instances[0].disconnect).toHaveBeenCalled();
        expect(media[0].removeEventListener).toHaveBeenCalled();
        expect(vi.getTimerCount()).toBe(0);
        loaded(images[0]);
        Visibility.instances[0].show([images[4]]);
        window.dispatchEvent(new Event("resize"));
        control.refresh();
        vi.advanceTimersByTime(10000);
        expect(images[4].hasAttribute("src")).toBe(false);
        expect(vi.getTimerCount()).toBe(0);
    });
});
