/**
 * Entries: the saved-region list, its cards, and the page-level actions.
 *
 * This file is the orchestrator. The pieces that used to make it unreadable now
 * live next door and are wired together here:
 *
 *   entries/geometry.ts  region normalisation, split-dedupe predicates
 *   entries/textFit.ts   the live preview's layout engine (mirrors typeset.py)
 *   entries/shapes.ts    the SVG overlay -- plate, text, clean FX, layer order
 *   entries/split.ts     the draw-a-cut flow
 *   entries/inpaint.ts   progress polling, the preview modal, save
 *   entries/pageTl.ts    whole-page OCR + translate
 *   entryOptions.ts      the per-entry collapsible panel
 *
 * Cards keep list order -- which *is* the reading order, numbered 1..n on both
 * the cards and the overlay -- so the list does not jump around while the user
 * works; the overlay is painted in *layer* order. Those are different orders on
 * purpose.
 */

import { state, WORKBENCH_PALETTE } from "./state";
import type { Entry } from "./state";
import type { DataItem, Point, ProcessImageRequest, RectangleCoords, Region } from "./types";
import { showNotify, setTextAlignmentValue, getTextAlignmentValue, setTextAreaLoading, updateActionRow } from "./ui";
import { getCurrentFontFilename, getCurrentFontName, setSelectedFontByFilename } from "./fonts";
import * as api from "./api";
import { fetchCharacters, subscribeCharacters } from "./characters";
import { renderTranslationUI, handleAutoTranslation, translationPanelState, translationPanelText } from "./translate";
import { showConfirm } from "./components/common";
import {
    buildEntryOptions,
    defaultCleanOptions,
    ensureEntryDefaults,
    getOpenEntry,
    isEntryOpen,
    normalizeLayers,
    setOpenEntry,
    sortByLayer,
    toggleOpenEntry,
    type OptionsContext,
} from "./entryOptions";
import { regionBBox, regionPairs, regionPoints, isSameRegion, cssEscape } from "./entries/geometry";
import {
    buildEntryShape,
    collectPlacedBadges,
    contrastColor,
    planNumberBadges,
    removeEntryShape,
    setShapeHover,
    type NumberBadge,
} from "./entries/shapes";
import {
    getReadingDirection,
    setReadingDirection,
    sortRegionsByReadingOrder,
    type ReadingDirection,
} from "./readingOrder";
import { resetTextMetrics } from "./entries/textFit";
import { canSplit, runSplit } from "./entries/split";
import { closeInpaintModal, runInpaint, saveInpaintPreview as commitPreview } from "./entries/inpaint";
import { pageCapture as runPageTl } from "./entries/pageTl";

export { closeInpaintModal };
export { isSameRegion };

const pageEntries = (): Entry[] => state.pageEntriesCache[state.currentImageFile] || [];

const overlay = (): SVGSVGElement | null => document.querySelector<SVGSVGElement>("#regionSvgOverlay");

// ------------------------------------------------------------------- selection

/**
 * Load an entry into the editor. Reads what the entry holds; runs nothing.
 *
 * Selecting is a navigation gesture -- clicking a region on the page, or its card
 * in the list, to look at it. It used to also OCR an entry that had no text yet
 * and translate one that had no translation, which meant a click on the wrong
 * bubble started work the reader never asked for, and a sweep down the list to
 * find something fired a request per card. The two jobs are the two buttons on
 * the card head now (`ocr-entry-btn` and `tl-entry-btn`), so asking for them is
 * explicit.
 *
 * An entry with no `ocr_text` therefore lands as an empty editor over its region:
 * the region is the confirm target, and the OCR button is the next click.
 *
 * The Auto OCR switch re-arms the old click-to-read behaviour behind an
 * opt-in flag: see `selectEntryAutoOCR`, which the card head and the overlay
 * shapes go through instead of `selectEntry` directly.
 */
export function applySelectedEntryCard(entry: Entry): void {
    ensureEntryDefaults(entry);
    state.currentlySelectedEntryId = entry.id;
    state.lastCapturedRegion = entry.region;
    setTextAlignmentValue(entry.text_align);
    setSelectedFontByFilename(entry.fontfile);

    const textArea = document.getElementById("textArea") as HTMLTextAreaElement | null;
    if (textArea) textArea.value = entry.ocr_text || "";
    renderTranslationUI({
        original: entry.ocr_text || "",
        translated: entry.text || "",
        alt_translated: "",
    });
}

/**
 * OCR this entry's region into the editor, then run the active translator.
 *
 * The translate step is `ocrIntoTextArea`'s, and it is the same one a fresh
 * capture does -- reading a region and then translating what came back is one
 * gesture everywhere else in the app, and re-reading a card's region is that
 * gesture aimed at a region that already exists.
 *
 * The entry is selected first so the region in the editor is the one being read,
 * and so a result landing in the text area belongs to the card that was clicked.
 */
function ocrEntry(entry: Entry, scroll = false): void {
    selectEntry(entry, scroll);

    if (entry.region.type === "polygon") {
        void ocrIntoTextArea(() =>
            api.ocrFreeform({
                filename: state.currentImageFile,
                points: (entry.region.coords as Point[]).map((p) => ({ x: p.x, y: p.y })),
                lang: currentLang(),
                isGrayScale: state.isGrayScaleEnabled,
            }),
        );
    } else {
        const r = entry.region.coords as RectangleCoords;
        void ocrIntoTextArea(() =>
            api.ocrCrop({
                filename: state.currentImageFile,
                x: r.x,
                y: r.y,
                width: r.w,
                height: r.h,
                lang: currentLang(),
                isGrayScale: state.isGrayScaleEnabled,
            }),
        );
    }
}

const currentLang = (): string => (document.getElementById("languageSelect") as HTMLSelectElement | null)?.value ?? "";

/**
 * Drop everything on screen that belonged to the page we just left.
 *
 * The editor, the translation panel and `lastCapturedRegion` are per-page, but
 * nothing reset them on a page turn: the previous page's text stayed in the box
 * and its region stayed the confirm target, so a Confirm aimed at what was on
 * screen wrote the old text onto a region of the old page.
 *
 * Entry cards are deliberately not touched. They are cached per page and
 * re-rendered from `pageEntriesCache` when the new image loads.
 */
export function clearPageWorkspace(): void {
    state.lastCapturedRegion = null;
    state.currentlySelectedEntryId = null;
    const textArea = document.getElementById("textArea") as HTMLTextAreaElement | null;
    if (textArea) textArea.value = "";
    // An OCR still in flight owns the loader; `ocrIntoTextArea` gives it back
    // when it settles, and its result is discarded by the page check there.
    setTextAreaLoading(false);
    renderTranslationUI({ original: "", translated: "", alt_translated: "" });
    updateActionRow();
}

/**
 * Run an OCR call into the text area, with the loader tied to the request.
 *
 * The previous version cleared the loader on the line *after* starting the
 * promise for rectangles and set it *before* the "no page open" guard for
 * polygons, so one flashed and the other stuck on forever.
 */
async function ocrIntoTextArea(request: () => Promise<{ text: string }>): Promise<void> {
    if (!state.currentImageFile) return;
    const page = state.currentImageFile;
    setTextAreaLoading(true);
    try {
        const data = await request();
        // The reader may have turned the page while this was in flight, and the
        // editor now belongs to a different one -- see `clearPageWorkspace`.
        if (state.currentImageFile !== page) return;
        const textArea = document.getElementById("textArea") as HTMLTextAreaElement | null;
        if (textArea) textArea.value = data.text ?? "";
        handleAutoTranslation();
    } catch {
        showNotify("❌ OCR failed for that region.");
    } finally {
        setTextAreaLoading(false);
    }
}

// --------------------------------------------------------------------- adding

/**
 * Add regions to the current page.
 *
 * `force_confirm` folds in whatever is in the editor (the Confirm button);
 * `inheritFrom` copies a source entry's styling onto the new regions, so pieces
 * of a split look like the bubble they came from -- and the pieces take the
 * source's slot, so the reading order survives the cut. `atIndex` overrides
 * that slot explicitly. Anything else is ordered geometrically (per the reading
 * direction) and appended after the entries already on the page.
 */
export function addNewEntries(data: Region[], force_confirm = false, inheritFrom?: Entry, atIndex?: number): void {
    if (!state.currentImageFile) return;
    const entries = (state.pageEntriesCache[state.currentImageFile] ||= []);

    let ocrText = "";
    let tlText = "";
    if (force_confirm) {
        ocrText = (document.getElementById("textArea") as HTMLTextAreaElement | null)?.value.trim() ?? "";
        // Empty unless the panel is showing text -- an error panel's message is
        // not a translation. See `translationPanelText`.
        tlText = translationPanelText();
    }

    const added: Entry[] = [];
    for (const region of data) {
        if (regionPoints(region).length < 3) continue;

        const existing = entries.find((e) => isSameRegion(e.region, region));
        if (existing) {
            // Re-capturing the same bubble is how the user re-runs it, so the
            // text is replaced but every styling choice they made is kept.
            if (force_confirm) {
                existing.ocr_text = ocrText;
                existing.text = tlText;
                existing.fontfile = getCurrentFontFilename();
                existing.fontname = getCurrentFontName();
                existing.text_align = getTextAlignmentValue();
                existing.visible = true;
            }
            continue;
        }

        const color = WORKBENCH_PALETTE[state.currentTrackingColorIdx % WORKBENCH_PALETTE.length];
        state.currentTrackingColorIdx++;

        added.push(
            ensureEntryDefaults({
                id: `entry_${Date.now()}_${crypto.randomUUID()}`,
                ocr_text: ocrText,
                text: tlText,
                fontfile: inheritFrom?.fontfile ?? getCurrentFontFilename(),
                fontname: inheritFrom?.fontname ?? getCurrentFontName(),
                region,
                color,
                text_align: inheritFrom?.text_align ?? getTextAlignmentValue(),
                visible: true,
                layer: inheritFrom?.layer ?? 1,
                // A fresh bubble has no speaker; a split piece keeps its
                // source's (same bubble, same voice).
                character_id: (inheritFrom as { character_id?: string | null } | undefined)?.character_id ?? null,
                font_size: inheritFrom?.font_size ?? null,
                font_color: inheritFrom?.font_color ?? null,
                stroke_width: inheritFrom?.stroke_width ?? null,
                stroke_color: inheritFrom?.stroke_color ?? null,
                bg_mode: inheritFrom?.bg_mode ?? "auto",
                bg_color: inheritFrom?.bg_color ?? null,
                clean: inheritFrom?.clean ? { ...inheritFrom.clean } : defaultCleanOptions(),
                // Geometry follows the same rule as the rest: a split piece
                // inherits it, because the halves of one bubble are still set
                // the same way, and a fresh capture starts neutral.
                word_spacing: inheritFrom?.word_spacing ?? null,
                line_spacing: inheritFrom?.line_spacing ?? null,
                font_scale_x: inheritFrom?.font_scale_x ?? 1,
                font_scale_y: inheritFrom?.font_scale_y ?? 1,
                shift_x: inheritFrom?.shift_x ?? 0,
                shift_y: inheritFrom?.shift_y ?? 0,
                angle_x: inheritFrom?.angle_x ?? 0,
                angle_y: inheritFrom?.angle_y ?? 0,
                angle_z: inheritFrom?.angle_z ?? 0,
            }),
        );
    }
    if (!added.length) return;

    // The batch arrives in detector order, which is no order at all: read it
    // geometrically first, so a bubble sweep lands numbered already.
    const ordered = sortRegionsByReadingOrder(added, getReadingDirection());

    let insertAt = atIndex;
    if (insertAt === undefined && inheritFrom) {
        const found = entries.findIndex((e) => e.id === inheritFrom.id);
        if (found >= 0) insertAt = found;
    }
    insertAt = Math.min(Math.max(insertAt ?? entries.length, 0), entries.length);
    entries.splice(insertAt, 0, ...ordered);
    normalizeLayers(entries);
}

/** This entry's reading number: position in the page list, 1-based. */
export function entryNumber(entry: Entry): number {
    return pageEntries().findIndex((e) => e.id === entry.id) + 1;
}

/**
 * Re-sort the current page into reading order, keeping everything else.
 *
 * The sort is stable and the array is reordered in place, so selection, layer
 * values and styling all stay attached to their entries -- only the numbers
 * (list positions) change. Callers re-render afterwards.
 */
export function resortPageEntries(): void {
    const entries = pageEntries();
    if (entries.length < 2) return;
    const rank = new Map(sortRegionsByReadingOrder(entries, getReadingDirection()).map((e, i) => [e.id, i]));
    entries.sort((a, b) => (rank.get(a.id) ?? 0) - (rank.get(b.id) ?? 0));
}

/** Switch the reading direction and re-number the current page to match. */
export function applyReadingDirection(direction: ReadingDirection): void {
    setReadingDirection(direction);
    if (state.currentImageFile) resortPageEntries();
}

/**
 * Nudge an entry one slot up (`-1`) or down (`+1`) in reading order.
 *
 * Returns whether anything moved. Rendering (and therefore renumbering of
 * cards and overlay badges) is the caller's job -- `renderCurrentPageEntries`
 * rebuilds both from list positions.
 */
export function moveEntry(entry: Entry, delta: -1 | 1): boolean {
    const entries = pageEntries();
    const from = entries.findIndex((e) => e.id === entry.id);
    const to = from + delta;
    if (from < 0 || to < 0 || to >= entries.length) return false;
    const [moved] = entries.splice(from, 1);
    entries.splice(to, 0, moved);
    return true;
}

/**
 * Confirm Entry: store what the editor and the translation panel currently hold.
 *
 * The panel's state decides what "currently hold" means. While a translation is
 * in flight the click is ignored rather than queued -- confirming now would save
 * the placeholder, and the reply lands in the panel a moment later anyway. An
 * error panel confirms as an empty translation, so the region is still saved,
 * with its OCR text, and can be translated afterwards.
 */
export function confirmCurrentTranslation(): void {
    if (!state.lastCapturedRegion) {
        showNotify("⚠️ No region captured -- select one on the page first.");
        return;
    }
    if (translationPanelState() === "translating") {
        showNotify("⏳ Still translating -- confirm when it lands.");
        return;
    }
    addNewEntries([state.lastCapturedRegion], true);
    renderCurrentPageEntries();
}

export async function clearCurrentPageEntries(): Promise<void> {
    if (!state.currentImageFile) return;
    const count = pageEntries().length;
    if (!count) {
        showNotify("⚠️ No entries to clear on this page.");
        return;
    }
    if (await showConfirm(`Are you sure you want to clear all ${count} entries for this page?`)) {
        state.currentlySelectedEntryId = null;
        setOpenEntry(null);
        state.pageEntriesCache[state.currentImageFile] = [];
        renderCurrentPageEntries();
    }
}

// -------------------------------------------------------------------- payloads

/**
 * The page as the backend wants it, bottom layer first.
 *
 * An entry with no text and `bg_mode: "auto"` is skipped: it is a captured
 * region nobody has done anything with yet, and sending it would have the
 * backend paint its detected background over the artwork. Empty text with an
 * explicit background is *not* skipped -- "clean" and a flat colour on an empty
 * region are how a bubble gets erased.
 */
function buildDataItems(): DataItem[] {
    return sortByLayer(
        pageEntries()
            .filter((e) => e.visible)
            .map(ensureEntryDefaults),
    )
        .filter((entry) => true)
        .map((entry) => ({
            og_text: entry.ocr_text || "",
            text: entry.text || "",
            fontfile: entry.fontfile,
            text_align: entry.text_align,
            points: regionPairs(entry.region),
            layer: entry.layer,
            font_size: entry.font_size,
            font_color: entry.font_color,
            stroke_width: entry.stroke_width,
            stroke_color: entry.stroke_color,
            bg_mode: entry.bg_mode,
            bg_color: entry.bg_color,
            clean: entry.bg_mode === "clean" ? (entry.clean ?? null) : null,
            // Typesetting geometry. `ensureEntryDefaults` above has already
            // filled and clamped these, so they are always the nine numbers the
            // backend expects -- neutral on an entry nobody adjusted.
            word_spacing: entry.word_spacing,
            line_spacing: entry.line_spacing,
            font_scale_x: entry.font_scale_x,
            font_scale_y: entry.font_scale_y,
            shift_x: entry.shift_x,
            shift_y: entry.shift_y,
            angle_x: entry.angle_x,
            angle_y: entry.angle_y,
            angle_z: entry.angle_z,
        }));
}

function buildPayload(): ProcessImageRequest | null {
    if (!state.currentImageFile) return null;
    return { filename: state.currentImageFile, data: buildDataItems() };
}

const inpaintDeps = { buildPayload, refresh: () => renderCurrentPageEntries("inpaint") };

export function executeInpaintAction(mode: "preview" | "generate"): Promise<void> {
    return runInpaint(mode, inpaintDeps);
}

/** The preview modal's Save button. */
export function saveInpaintPreview(): void {
    void commitPreview(inpaintDeps);
}

export function pageCapture(): Promise<void> {
    return runPageTl({ refresh: () => renderCurrentPageEntries("pagetl") });
}

// ----------------------------------------------------------------- card update

function selectEntry(entry: Entry, scroll = false, expand = true): void {
    if (state.currentlySelectedEntryId && state.currentlySelectedEntryId !== entry.id) {
        document.getElementById(`card_${state.currentlySelectedEntryId}`)?.classList.remove("highlighted-entry");
        setShapeHover(state.currentlySelectedEntryId, false);
    }
    state.currentlySelectedEntryId = entry.id;
    applySelectedEntryCard(entry);
    // Selecting expands. Done before the card is looked up, not after: opening the
    // panel replaces the card node, so a reference taken any earlier would be the
    // detached one and neither the highlight nor the scroll would land on anything.
    // `expand` is off for the one caller that owns the panel state itself -- see the
    // toggle button, where expanding here would undo the collapse it just asked for.
    if (expand) expandEntryPanel(entry);
    const card = document.getElementById(`card_${entry.id}`);
    if (card) {
        card.classList.add("highlighted-entry");
        if (scroll) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

/**
 * Select, reading the region first when Auto OCR is on and there is
 * something to read.
 *
 * The only funnel for navigation clicks -- the card head and the overlay
 * shapes both come through here instead of `selectEntry` directly. Entries
 * that already hold text are just selected (their text lands in the editor
 * for free), so sweeping the list to find something fires no requests; an
 * entry with no text yet is read on the spot. Whether translation follows
 * the read is the Auto TL switch's job, not this one's.
 */
function selectEntryAutoOCR(entry: Entry, scroll = false): void {
    if (state.isAutoOCREnabled && !entry.ocr_text) {
        ocrEntry(entry, scroll);
    } else {
        selectEntry(entry, scroll);
    }
}

function removeEntry(entry: Entry): void {
    document.getElementById(`card_${entry.id}`)?.remove();
    removeEntryShape(entry.id);
    if (state.currentlySelectedEntryId === entry.id) state.currentlySelectedEntryId = null;
    if (isEntryOpen(entry.id)) setOpenEntry(null);
}

/** Replace one card in place, keeping its position and its open panel. */
function updateEntryCard(entry: Entry): void {
    const old = document.getElementById(`card_${entry.id}`);
    if (!old) return;
    old.replaceWith(buildEntryCard(entry, entryNumber(entry)));
}

function updateEntryShape(entry: Entry): void {
    const svg = overlay();
    if (!svg) return;
    removeEntryShape(entry.id);
    if (!entry.visible) return;
    const badge = planNumberBadges([{ entry, num: entryNumber(entry) }], collectPlacedBadges(svg)).get(entry.id);
    buildEntryShape(entry, svg, shapeHandlers, badge);
}

function updateEntry(entry: Entry): void {
    updateEntryCard(entry);
    updateEntryShape(entry);
}

// ------------------------------------------------------------------ the shapes

const shapeHandlers = {
    onSelect: (entry: Entry) => selectEntryAutoOCR(entry, true),
    onEditInPlace: (entry: Entry, group: SVGGElement) => openInlineEditor(entry, group),
    onHover: (entry: Entry, hovering: boolean) => {
        setShapeHover(entry.id, hovering);
        if (!hovering && state.currentlySelectedEntryId !== entry.id) {
            document.getElementById(`card_${entry.id}`)?.classList.remove("highlighted-entry");
        }
    },
    onContextMenu: (entry: Entry) => hideEntryFromShape(entry),
};

/**
 * Right-click a region to hide it -- the same gesture, and the same result, as
 * right-clicking its card in the list.
 *
 * Hide rather than toggle, unlike the card: the only way to right-click a region
 * is for it to be drawn, and `updateEntryShape` draws nothing for an entry that
 * is already hidden. The checkbox follows because the card is rebuilt from
 * `entry.visible`.
 *
 * Selection is deliberately untouched. Nothing here calls `selectEntry`, so a
 * card that was selected stays selected and one that was not stays that way --
 * hiding a region is not a way of choosing it.
 */
function hideEntryFromShape(entry: Entry): void {
    if (!entry.visible) return;
    entry.visible = false;
    updateEntry(entry);
}

/** Edit an entry's text directly over its bubble. */
function openInlineEditor(entry: Entry, group: SVGGElement): void {
    const container = document.getElementById("imageContainer");
    if (!container) return;
    container.querySelector(".inline-canvas-editor")?.remove();

    const shapeBox = group.getBoundingClientRect();
    const containerBox = container.getBoundingClientRect();
    const textEl = group.querySelector("text");
    const rendered = textEl ? parseFloat(textEl.getAttribute("font-size") ?? "14") : 14;
    // The overlay is in image pixels; the editor is a DOM node in screen pixels.
    const scale = shapeBox.width / Math.max(1, regionBBox(entry.region).w);

    const input = document.createElement("textarea");
    input.className = "inline-canvas-editor";
    input.value = entry.text || "";
    input.style.cssText = [
        "position:absolute",
        `left:${shapeBox.left - containerBox.left + container.scrollLeft}px`,
        `top:${shapeBox.top - containerBox.top + container.scrollTop}px`,
        `width:${shapeBox.width}px`,
        `height:${shapeBox.height}px`,
        `font-size:${Math.max(8, rendered * scale)}px`,
        `text-align:${entry.text_align === "right" ? "right" : entry.text_align === "left" ? "left" : "center"}`,
    ].join(";");
    container.appendChild(input);
    input.focus();
    input.select();

    let closed = false;
    const commit = () => {
        if (closed) return;
        closed = true;
        const value = input.value.trim();
        input.remove();
        if (value === (entry.text || "")) return;
        entry.text = value;
        if (state.lastCapturedRegion === entry.region) {
            // Mirror it into the translation panel through the one writer that
            // owns the panel's state: setting the text alone would leave a panel
            // still marked `error` holding real text, which Confirm then drops.
            renderTranslationUI({ translated: value });
        }
        updateEntry(entry);
    };
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (e) => {
        // Shift+Enter inserts the newline the typesetter honours as a line break.
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            commit();
        }
        if (e.key === "Escape") {
            e.preventDefault();
            closed = true;
            input.remove();
        }
    });
}

// ------------------------------------------------------------------- the cards

const EDIT_ICON_CAPTURED =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 5V2h3"/><path d="M14 5V2h-3"/><path d="M2 11v3h3"/><path d="M14 11v3h-3"/><path d="M5 6.5h6"/><path d="M5 8.5h6"/><path d="M5 10.5h4"/></svg>';
const EDIT_ICON_UNTRANSLATED =
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="m22 22-5-10-5 10"/><path d="M14 18h6"/></svg>';

/**
 * The two work buttons on a card head: read the region, then translate what it read.
 *
 * The OCR button is always there and always looks the same -- it is a verb, and it
 * does the same thing whether or not the region has been read before (a re-read is
 * a normal thing to want: a different language, greyscale on, or a first pass that
 * came back as noise). The Translate button is the one that carries state, because
 * there is nothing to translate before the OCR has landed: it appears with the
 * source text and its icon says whether a translation exists yet.
 */
const canTranslate = (entry: Entry): boolean => Boolean((entry.ocr_text || "").trim());

const tlIconFor = (_entry: Entry): string => EDIT_ICON_UNTRANSLATED;

/** The icon-only chip that says an entry's background is not on auto. */
function bgChipIcon(entry: Entry): string | null {
    if (entry.bg_mode === "color") return "format_color_fill";
    if (entry.bg_mode === "transparent") return "check_box_outline_blank";
    if (entry.bg_mode === "clean") return "auto_fix_high";
    return null;
}

function iconButton(cls: string, glyph: string, title: string): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = cls;
    btn.title = title;
    const span = document.createElement("span");
    span.className = "material-icons";
    span.textContent = glyph;
    btn.appendChild(span);
    return btn;
}

/** Same chrome as `iconButton`, for the inline SVG icons rather than a font glyph. */
function glyphButton(cls: string, svg: string, title: string): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = cls;
    btn.title = title;
    btn.innerHTML = svg;
    return btn;
}

function optionsContext(entry: Entry): OptionsContext {
    return {
        // A getter, not a snapshot: a page change between building the panel and
        // clicking a control would otherwise compute the layer ceiling from the
        // previous page's entries.
        get entries() {
            return pageEntries();
        },
        refresh: () => {
            updateEntryHead(entry);
            updateEntryShape(entry);
        },
        rebuild: () => {
            const panel = document.querySelector(`#card_${cssEscape(entry.id)} .entry-collapse`);
            panel?.replaceWith(buildEntryOptions(entry, optionsContext(entry)));
            updateEntryHead(entry);
            updateEntryShape(entry);
        },
        refreshAll: () => renderCurrentPageEntries("options"),
    };
}

/** Refresh just the summary row, so an open panel keeps its focus and scroll. */
function updateEntryHead(entry: Entry): void {
    const card = document.getElementById(`card_${entry.id}`);
    const head = card?.querySelector(".entry-head");
    if (!card || !head) return;

    const preview = head.querySelector<HTMLElement>(".entry-text-preview");
    if (preview) {
        preview.textContent = entry.text || entry.ocr_text || "";
        preview.style.fontFamily = entry.ocr_text ? entry.fontname : "";
    }
    paintNumberBadge(head.querySelector<HTMLElement>(".entry-layer-badge"), entry, entryNumber(entry));
    const position = entryNumber(entry);
    const total = pageEntries().length;
    const upBtn = head.querySelector<HTMLButtonElement>(".move-entry-up-btn");
    if (upBtn) upBtn.disabled = position <= 1;
    const downBtn = head.querySelector<HTMLButtonElement>(".move-entry-down-btn");
    if (downBtn) downBtn.disabled = position < 1 || position >= total;
    // Only the Translate button is refreshed: the OCR button's icon does not
    // depend on the entry (see `canTranslate`).
    const tlBtn = head.querySelector<HTMLElement>(".tl-entry-btn");
    if (tlBtn) {
        tlBtn.innerHTML = tlIconFor(entry);
        tlBtn.style.display = canTranslate(entry) ? "" : "none";
    }
    const splitBtn = head.querySelector<HTMLElement>(".split-entry-btn");
    if (splitBtn) splitBtn.style.display = canSplit(entry) ? "" : "none";

    const glyph = bgChipIcon(entry);
    let chip = head.querySelector<HTMLElement>(".entry-bg-chip");
    if (!glyph) {
        chip?.remove();
    } else {
        if (!chip) {
            chip = document.createElement("span");
            chip.className = "entry-bg-chip material-icons";
            head.querySelector(".entry-text-preview")?.after(chip);
        }
        chip.textContent = glyph;
        chip.title = `Background: ${entry.bg_mode}`;
    }
}

/**
 * The reading number on a card head: position in the page list, painted in the
 * entry's own outline colour so the card and its region read as one.
 *
 * This deliberately no longer shows the *layer*: stacking order still lives in
 * the options panel, but the badge answers "which bubble is this" -- the
 * number stamped on the overlay shape.
 */
function paintNumberBadge(badge: HTMLElement | null, entry: Entry, num: number): void {
    if (!badge) return;
    badge.textContent = String(num);
    badge.title = `Reading order ${num}`;
    badge.style.background = entry.color;
    badge.style.borderColor = entry.color;
    badge.style.color = contrastColor(entry.color);
}

function buildEntryCard(entry: Entry, num: number): HTMLElement {
    ensureEntryDefaults(entry);

    const card = document.createElement("div");
    card.className = "saved-entry-card";
    card.id = `card_${entry.id}`;
    card.style.borderLeftColor = entry.color;

    const head = document.createElement("div");
    head.className = "entry-head";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "entry-checkbox";
    checkbox.checked = entry.visible;
    checkbox.style.color = entry.color;
    checkbox.title = entry.visible ? "Hide from render" : "Include in render";
    head.appendChild(checkbox);

    const badge = document.createElement("span");
    badge.className = "entry-layer-badge";
    head.appendChild(badge);

    const entries = pageEntries();
    const upBtn = iconButton("move-entry-btn move-entry-up-btn", "arrow_upward", "Move up in reading order");
    upBtn.disabled = num <= 1;
    head.appendChild(upBtn);

    const downBtn = iconButton("move-entry-btn move-entry-down-btn", "arrow_downward", "Move down in reading order");
    downBtn.disabled = num < 1 || num >= entries.length;
    head.appendChild(downBtn);

    paintNumberBadge(badge, entry, num);

    const preview = document.createElement("div");
    preview.className = "entry-text-preview";
    preview.textContent = entry.text || entry.ocr_text || "";
    if (entry.ocr_text) preview.style.fontFamily = entry.fontname;
    head.appendChild(preview);

    const glyph = bgChipIcon(entry);
    if (glyph) {
        const chip = document.createElement("span");
        chip.className = "entry-bg-chip material-icons";
        chip.textContent = glyph;
        chip.title = `Background: ${entry.bg_mode}`;
        head.appendChild(chip);
    }

    const splitBtn = iconButton("split-entry-btn", "content_cut", "Split this region");
    if (!canSplit(entry)) splitBtn.style.display = "none";
    head.appendChild(splitBtn);

    const ocrBtn = glyphButton("ocr-entry-btn edit-entry-btn", EDIT_ICON_CAPTURED, "Read this region (OCR)");
    head.appendChild(ocrBtn);

    const tlBtn = glyphButton("tl-entry-btn edit-entry-btn", tlIconFor(entry), "Translate this entry");
    if (!canTranslate(entry)) tlBtn.style.display = "none";
    head.appendChild(tlBtn);

    const deleteBtn = iconButton("delete-entry-btn", "delete", "Delete this entry");
    head.appendChild(deleteBtn);

    const toggleBtn = iconButton(
        "entry-toggle-btn",
        isEntryOpen(entry.id) ? "expand_less" : "expand_more",
        "Typesetting options",
    );
    head.appendChild(toggleBtn);

    card.appendChild(head);
    if (isEntryOpen(entry.id)) {
        card.classList.add("is-open");
        card.appendChild(buildEntryOptions(entry, optionsContext(entry)));
    }

    // ---- wiring ---------------------------------------------------------
    head.addEventListener("click", (e) => {
        if ((e.target as HTMLElement).closest("button,.entry-checkbox")) return;
        selectEntryAutoOCR(entry);
    });

    card.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        entry.visible = !entry.visible;
        updateEntry(entry);
    });

    checkbox.addEventListener("click", (e) => e.stopPropagation());
    checkbox.addEventListener("change", () => {
        entry.visible = checkbox.checked;
        checkbox.title = entry.visible ? "Hide from render" : "Include in render";
        updateEntryShape(entry);
    });

    toggleBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        // Selected first, and without expanding: this button is the one place that
        // owns the open/closed state, so letting `selectEntry` open the panel too
        // would turn every collapse click into a no-op -- open, then closed again.
        selectEntry(entry, false, false);
        toggleEntryPanel(entry);
    });

    // Reordering rebuilds the list, which drops focus: hand it back to the
    // button just pressed so repeated keyboard nudges keep working.
    const refocus = (cls: string): void => {
        try {
            document.querySelector<HTMLElement>(`#card_${cssEscape(entry.id)} .${cls}`)?.focus();
        } catch {
            /* a detached card: nothing to focus */
        }
    };

    upBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (moveEntry(entry, -1)) {
            renderCurrentPageEntries("move");
            refocus("move-entry-up-btn");
        }
    });

    downBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (moveEntry(entry, 1)) {
            renderCurrentPageEntries("move");
            refocus("move-entry-down-btn");
        }
    });

    splitBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        void runSplit(entry, splitBtn, {
            addEntries: (regions, source) => addNewEntries(regions, false, source),
            refresh: () => renderCurrentPageEntries("split"),
        });
    });

    deleteBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const entries = pageEntries();
        state.pageEntriesCache[state.currentImageFile] = entries.filter((item) => item.id !== entry.id);
        removeEntry(entry);
        // A delete can vacate a layer, and a stack with a hole in it makes every
        // later ceiling calculation wrong.
        normalizeLayers(pageEntries());
        renderCurrentPageEntries("delete");
    });

    // Both work buttons point the eye at the region they are about to act on: the
    // card is in a list, and the bubble it belongs to is somewhere on the page.
    const markTarget = (): void => {
        if (state.currentlySelectedEntryId && state.currentlySelectedEntryId !== entry.id) {
            document.getElementById(`card_${state.currentlySelectedEntryId}`)?.classList.remove("highlighted-entry");
            setShapeHover(state.currentlySelectedEntryId, false);
        }
        setShapeHover(entry.id, true);
    };

    ocrBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        markTarget();
        ocrEntry(entry);
    });

    tlBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        markTarget();
        // Selected first so the editor holds *this* entry's source text: the
        // translators read the text area, not the entry.
        selectEntry(entry);
        // Explicit gesture: translates even while the Auto TL switch is off.
        handleAutoTranslation({ force: true });
    });

    card.addEventListener("mouseenter", () => setShapeHover(entry.id, true));
    card.addEventListener("mouseleave", () => setShapeHover(entry.id, false));

    if (state.currentlySelectedEntryId === entry.id) card.classList.add("highlighted-entry");
    return card;
}

/**
 * Expand this entry's panel, unless it is already the open one.
 *
 * Not `toggleEntryPanel`: this is reached by selecting, and selecting the entry
 * whose panel is open would otherwise close it -- so clicking a region twice, or
 * clicking a card that a region click had already opened, would collapse the very
 * thing the click was asking to see.
 */
function expandEntryPanel(entry: Entry): void {
    if (isEntryOpen(entry.id)) return;
    toggleEntryPanel(entry);
}

/** Open this entry's panel, closing whichever one was open. */
function toggleEntryPanel(entry: Entry): void {
    const previous = getOpenEntry();
    const nowOpen = toggleOpenEntry(entry.id);

    if (previous && previous !== entry.id) {
        const other = pageEntries().find((e) => e.id === previous);
        if (other) updateEntryCard(other);
        else document.getElementById(`card_${previous}`)?.classList.remove("is-open");
    }
    updateEntryCard(entry);
    if (nowOpen) {
        document.getElementById(`card_${entry.id}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

// ------------------------------------------------------------------ rendering

export function renderCurrentPageEntries(_source = ""): void {
    const list = document.getElementById("savedEntriesList");
    if (!list) return;

    const entries = pageEntries();
    entries.forEach(ensureEntryDefaults);
    normalizeLayers(entries);

    // The open panel belongs to an entry that may have been deleted, or to a
    // page the user has navigated away from.
    const open = getOpenEntry();
    if (open && !entries.some((e) => e.id === open)) setOpenEntry(null);

    const scrollTop = list.scrollTop;
    list.replaceChildren();

    const svg = overlay();
    if (svg) {
        // `replaceChildren` also drops the measurer and the shared defs, which
        // are rebuilt on demand by `ensureMeasurer` / `ensureOverlayDefs`.
        svg.replaceChildren();
    }

    // Cards in list order, which *is* the reading order -- the badge on each
    // card and each overlay shape is just its 1-based position.
    const numbers = new Map(entries.map((entry, idx) => [entry.id, idx + 1] as const));
    const badges = svg
        ? planNumberBadges(
              entries.filter((entry) => entry.visible).map((entry) => ({ entry, num: numbers.get(entry.id) ?? 0 })),
          )
        : new Map<string, NumberBadge>();
    for (const entry of entries) list.appendChild(buildEntryCard(entry, numbers.get(entry.id) ?? 0));
    // ...shapes in layer order, because that is what "layer" means in SVG.
    if (svg) {
        for (const entry of sortByLayer(entries)) {
            if (entry.visible) buildEntryShape(entry, svg, shapeHandlers, badges.get(entry.id));
        }
    }

    list.scrollTop = scrollTop;
}

// ------------------------------------------------------------------ listeners

export function initEntriesListeners(): void {
    document.getElementById("inpaintModalOverlay")?.addEventListener("click", (e) => {
        if (!document.getElementById("inpaintModalContent")?.contains(e.target as Node)) closeInpaintModal();
    });

    // The roster lives on the backend; mirror it once at startup (best
    // effort -- an unreachable backend just means an empty roster until the
    // user imports), then repaint the entry selectors on every change.
    void fetchCharacters().catch(() => {
        /* offline backend: selectors stay on "none" until an import */
    });
    subscribeCharacters(() => renderCurrentPageEntries("characters"));

    // Token widths are cached per family. A token measured while the real face
    // was still loading would keep the fallback's width for the session, so the
    // cache is dropped once the fonts settle and the page is re-fitted.
    document.fonts?.ready
        .then(() => {
            resetTextMetrics();
            if (state.currentImageFile && pageEntries().length) renderCurrentPageEntries("fonts");
        })
        .catch(() => {
            /* the browser declined to report; measurements stay as they are */
        });
}
