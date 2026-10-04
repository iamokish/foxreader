/**
 * Split an entry's region by drawing a line across it.
 *
 * Three things went wrong in the old flow and all three are handled here:
 * only polygons could be split, the loading overlay went up *before* the user
 * drew (so it swallowed the strokes), and whatever the backend returned was
 * accepted — including a single piece identical to the source, which replaced a
 * good region with a hidden one and an identical copy.
 */

import { state } from "../state";
import type { Entry } from "../state";
import type { Region } from "../types";
import * as api from "../api";
import { splitCapture } from "../ocr";
import { hideLoading, showLoading, showNotify } from "../ui";
import { isSameRegion, isUsablePiece, nearlyIdentical, regionPoints } from "./geometry";

export interface SplitDeps {
    /** Add the resulting pieces, inheriting the source entry's styling. */
    addEntries: (regions: Region[], inheritFrom: Entry) => void;
    refresh: () => void;
}

/** Whether this entry can still be split. */
export function canSplit(entry: Entry): boolean {
    // Splitting rewrites the geometry the text was fitted to, so once either
    // pass has produced text the region is no longer free to move.
    if (entry.ocr_text || entry.text) return false;
    return regionPoints(entry.region).length >= 3;
}

let busy = false;

/**
 * Toggle the "this is the region being cut" glow on an entry's overlay shape.
 *
 * Looked up by id at both ends rather than held in a variable: the overlay is
 * rebuilt by any render that lands while the cut is being drawn, and a stale node
 * reference would leave the glow stuck on a detached element.
 */
function markSplitTarget(entryId: string, armed: boolean): void {
    document.getElementById(`shape_${entryId}`)?.classList.toggle("split-target", armed);
}

/**
 * Ask the user to draw a cut, send it, and adopt the pieces.
 *
 * `button` gets `.split-active` for the duration — the spec's "the entry split
 * button must show glow", instead of the capture toolbar lighting up. The region
 * itself gets `.split-target`: the button is in the sidebar and the stroke is
 * drawn on the artwork, so the page needs to say which region is about to be cut.
 */
export async function runSplit(entry: Entry, button: HTMLElement | null, deps: SplitDeps): Promise<void> {
    if (busy) return;
    if (!canSplit(entry)) {
        showNotify("⚠️ This region already has text — clear it before splitting.");
        return;
    }
    if (!state.currentImageFile) return;

    const filename = state.currentImageFile;
    busy = true;
    button?.classList.add("split-active");
    markSplitTarget(entry.id, true);
    try {
        // The overlay must stay clickable while the cut is drawn, so nothing is
        // shown until `splitCapture` resolves.
        const cut = await splitCapture();
        if (!cut) return;
        if (state.currentImageFile !== filename) {
            showNotify("⚠️ Page changed — split cancelled.");
            return;
        }

        showLoading("Splitting...");
        let pieces: Region[];
        try {
            pieces = await api.splitBubble({
                filename,
                isGrayScale: state.isGrayScaleEnabled,
                region: entry.region,
                split_object: cut.coords,
            });
        } catch {
            showNotify("❌ Split failed — the server could not process the cut.");
            return;
        }

        const kept = acceptPieces(pieces, entry);
        if (kept.length < 2) {
            // Either the stroke missed, or every piece came back as the region
            // itself. Leaving the entry untouched is the only safe outcome:
            // hiding it and adding a copy looks like data loss.
            showNotify("⚠️ That cut didn't divide the region — nothing changed.");
            return;
        }

        deps.addEntries(kept, entry);
        entry.visible = false;
        deps.refresh();
        showNotify(`✂️ Split into ${kept.length} regions.`);
    } finally {
        busy = false;
        button?.classList.remove("split-active");
        markSplitTarget(entry.id, false);
        hideLoading();
    }
}

/**
 * Drop the pieces that are not real pieces.
 *
 * A backend split can legitimately return the source region (the cut missed),
 * a near-copy of it (the cut only shaved the contour), slivers of
 * rasterisation debris, or duplicates of each other.
 */
function acceptPieces(pieces: Region[] | null, source: Entry): Region[] {
    if (!Array.isArray(pieces)) return [];
    const existing = state.pageEntriesCache[state.currentImageFile] || [];
    const kept: Region[] = [];

    for (const piece of pieces) {
        if (!isUsablePiece(piece)) continue;
        if (isSameRegion(piece, source.region) || nearlyIdentical(piece, source.region)) continue;
        if (kept.some((k) => nearlyIdentical(k, piece))) continue;
        // A region the page already holds would only add a duplicate card.
        if (existing.some((e) => e.id !== source.id && isSameRegion(e.region, piece))) continue;
        kept.push(piece);
    }
    return kept;
}
