/**
 * Page navigation, expressed once.
 *
 * The app's source of truth for "which page is on screen" is the `.active` class
 * on a gallery thumbnail: `navigateGallery` reads it to know where it is, and
 * `updateReaderPageIndicator` reads it to label the page. So every jump the reader
 * makes goes through a *click* on the thumbnail rather than calling `selectImage`
 * directly -- the thumbnail's own handler is what moves `.active`, and bypassing
 * it leaves the class pointing at the page the reader just left.
 *
 * `renderGallery` installs that handler (see `viewer.ts`), which also covers the
 * strip view: `selectImage` scrolls the strip to the page instead of swapping
 * `#mainImage`, so the same click is correct in every view.
 */

import { state } from "../state";

function galleryItems(): HTMLElement[] {
    return Array.from(document.querySelectorAll<HTMLElement>(".gallery-item-container"));
}

export function pageCount(): number {
    const items = galleryItems().length;
    return items || state.currentFiles.length;
}

/**
 * The index of the page on screen, or -1 when no folder is loaded.
 *
 * The active thumbnail first, for the reason above; `currentFiles` is the
 * fallback for the moment between a folder loading and the first thumbnail being
 * marked, and for tests with no gallery in the DOM.
 */
export function currentPageIndex(): number {
    const items = galleryItems();
    const active = items.findIndex((item) => item.classList.contains("active"));
    if (active >= 0) return active;
    const byFile = state.currentFiles.indexOf(state.currentImageFile);
    return byFile;
}

/**
 * Wrap an index into `[0, count)`.
 *
 * Wrapping, not clamping, to match `navigateGallery`: paging forward off the last
 * page returns to the first, which is what the arrow keys have always done.
 */
export function wrapIndex(index: number, count: number): number {
    if (count <= 0) return -1;
    return ((index % count) + count) % count;
}

/** Show the page at `index`, wrapping. A no-op when there is no such thumbnail. */
export function gotoPage(index: number): void {
    const items = galleryItems();
    if (items.length === 0) return;
    const target = wrapIndex(index, items.length);
    if (target < 0) return;
    items[target].click();
    items[target].scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
}
