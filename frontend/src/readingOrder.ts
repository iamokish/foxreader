/**
 * Reading order: which way the bubbles on a manga page are read.
 *
 * Manga is right-to-left, top-to-bottom; other comics are left-to-right. The
 * choice lives here (persisted, RTL default) and the geometry below turns it
 * into an order: rows top-to-bottom, entries within a row from the leading
 * edge. Entry *numbers* are derived from that order -- position + 1 in the
 * page list -- so deletes, splits and manual moves can never leave the
 * numbers inconsistent.
 */

import { regionBBox } from "./entries/geometry";
import type { Region } from "./types";

export type ReadingDirection = "ltr" | "rtl";

const STORAGE_KEY = "fox-reader-direction";

export function normalizeDirection(value: unknown): ReadingDirection {
    return value === "ltr" ? "ltr" : "rtl";
}

export function getReadingDirection(): ReadingDirection {
    try {
        return normalizeDirection(localStorage.getItem(STORAGE_KEY));
    } catch {
        return "rtl";
    }
}

export function setReadingDirection(direction: ReadingDirection): ReadingDirection {
    const next = normalizeDirection(direction);
    try {
        localStorage.setItem(STORAGE_KEY, next);
    } catch {
        /* a locked-down profile keeps the session default */
    }
    return next;
}

interface Box {
    cx: number;
    cy: number;
    w: number;
    h: number;
}

function boxOf(region: Region): Box | null {
    let box: { x: number; y: number; w: number; h: number };
    try {
        box = regionBBox(region);
    } catch {
        return null;
    }
    const { x, y, w, h } = box;
    if (![x, y, w, h].every((v) => typeof v === "number" && Number.isFinite(v))) return null;
    if (w <= 0 || h <= 0) return null;
    return { cx: x + w / 2, cy: y + h / 2, w, h };
}

function median(values: number[]): number {
    const clean = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
    if (!clean.length) return 0;
    const mid = clean.length >> 1;
    return clean.length % 2 ? clean[mid] : (clean[mid - 1] + clean[mid]) / 2;
}

/**
 * Order regions the way they are read: rows top-to-bottom, each row from the
 * leading edge (right first for RTL manga, left first for LTR).
 *
 * Rows are bands, not raw y -- detector centres jitter by pixels, and sorting
 * floats directly is what interleaved neighbouring bubbles. The band tolerance
 * is half the median region height, so it tracks the page's own scale. Stable:
 * ties keep their input order. Returns a new array; degenerate regions sink to
 * the end in input order rather than throwing the sort off.
 */
export function sortRegionsByReadingOrder<T extends { region: Region }>(
    items: readonly T[],
    direction: ReadingDirection,
): T[] {
    const rtl = normalizeDirection(direction) === "rtl";
    const placed = items.map((item, idx) => ({ item, idx, box: boxOf(item.region) }));
    const heights = placed.map((p) => p.box?.h ?? 0).filter((h) => h > 0);
    const tolerance = median(heights) * 0.5;

    const byRow = [...placed].sort((a, b) => (a.box ? a.box.cy : Infinity) - (b.box ? b.box.cy : Infinity));

    const rows: { items: typeof placed; meanY: number }[] = [];
    for (const p of byRow) {
        const last = rows[rows.length - 1];
        // Against the row's running mean, not the previous box: a staircase of
        // boxes each a pixel below the last must not chain into one mega-row.
        if (last && p.box && tolerance > 0 && Math.abs(p.box.cy - last.meanY) <= tolerance) {
            last.items.push(p);
            last.meanY += (p.box.cy - last.meanY) / last.items.length;
        } else if (p.box) {
            rows.push({ items: [p], meanY: p.box.cy });
        } else {
            // Degenerate regions sink to the end in input order.
            rows.push({ items: [p], meanY: Infinity });
        }
    }
    rows.sort((a, b) => a.meanY - b.meanY || a.items[0].idx - b.items[0].idx);

    const ordered: typeof placed = [];
    for (const { items: row } of rows) {
        row.sort((a, b) => {
            const ax = a.box ? a.box.cx : rtl ? -Infinity : Infinity;
            const bx = b.box ? b.box.cx : rtl ? -Infinity : Infinity;
            if (ax !== bx) return rtl ? bx - ax : ax - bx;
            const ay = a.box ? a.box.cy : Infinity;
            const by = b.box ? b.box.cy : Infinity;
            return ay - by || a.idx - b.idx;
        });
        ordered.push(...row);
    }
    return ordered.map((p) => p.item);
}
