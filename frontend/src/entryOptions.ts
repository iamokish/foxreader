/**
 * The per-entry options panel -- everything under an entry's collapsible.
 *
 * Kept out of `entries.ts` because it is a self-contained widget kit (layer
 * stepper, colour control with palette / picker / eyedropper, clean settings)
 * rather than part of the entry list's own logic. Everything here is built with
 * plain DOM and CSS classes so it themes from the same custom properties as the
 * rest of the app in both layouts.
 */

import {
    state,
    ALIGNMENT_ICONS,
    COLOR_PALETTE,
    TEXT_ALIGNMENTS,
    MAX_LAYER,
    MIN_FONT_SIZE,
    MAX_FONT_SIZE,
    MIN_STROKE_WIDTH,
    MAX_STROKE_WIDTH,
    MIN_WORD_SPACING,
    MAX_WORD_SPACING,
    MIN_LINE_SPACING,
    MAX_LINE_SPACING,
    MIN_FONT_SCALE,
    MAX_FONT_SCALE,
    MAX_TEXT_ANGLE,
    MAX_TEXT_SHIFT,
} from "./state";
import type { Entry } from "./state";
import type { BgMode, CleanKnob, CleanMethod, CleanMethodsResponse, CleanOptions, TextAlignment } from "./types";
import * as api from "./api";
import { characterPickerLabel, getCharacters } from "./characters";
import { imageBounds } from "./entries/shapes";
import { showNotify } from "./ui";

// ---------------------------------------------------------------- DOM helpers

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
}

function icon(name: string, size = 16): HTMLElement {
    const span = el("span", "material-icons", name);
    span.style.fontSize = `${size}px`;
    return span;
}

function clamp(value: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, value));
}

// ------------------------------------------------------------------- defaults

export function defaultCleanOptions(): CleanOptions {
    return {
        method: "region",
        fill: "hybrid-level",
        glow: true,
        transport: true,
        // Mirrors `fox_reader.clean.tuning`: `fastest` is three forward passes
        // and within a few percent of `best`, which is the right trade for a
        // panel the user can re-run from.
        speed: "fastest",
        tta: true,
        tile: 0,
    };
}

/**
 * Back-fill clean options an older build never wrote.
 *
 * `entry.clean` is persisted with the project, so an entry saved before the
 * detector tuning existed arrives with `speed`, `tta` and `tile` simply absent.
 * The backend would normalise those away, but only after the panel had already
 * rendered a select with no matching option -- which shows as a blank control.
 * Each field is checked for its own type rather than the object for a version,
 * so a partially written one is repaired too.
 */
function fillCleanDefaults(clean: CleanOptions): CleanOptions {
    const fallback = defaultCleanOptions();
    if (typeof clean.method !== "string") clean.method = fallback.method;
    if (typeof clean.fill !== "string") clean.fill = fallback.fill;
    if (typeof clean.glow !== "boolean") clean.glow = fallback.glow;
    if (typeof clean.transport !== "boolean") clean.transport = fallback.transport;
    if (typeof clean.speed !== "string") clean.speed = fallback.speed;
    if (typeof clean.tta !== "boolean") clean.tta = fallback.tta;
    if (typeof clean.tile !== "number" || !Number.isFinite(clean.tile)) clean.tile = fallback.tile;
    return clean;
}

/**
 * Snap a multiplier to the tenth the controls offer, or `null` for "auto".
 *
 * Tenths are the resolution asked for, and snapping here rather than only in the
 * select keeps the stored value and the preview in step: a project saved at 1.05
 * would otherwise show 1.1 in the picker while rendering 1.05.
 */
function fillMultiplier(value: unknown, lo: number, hi: number): number | null {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return Math.round(clamp(value, lo, hi) * 10) / 10;
}

/** The same, for a field that has a neutral value instead of an auto mode. */
function fillScale(value: unknown, lo: number, hi: number): number {
    if (typeof value !== "number" || !Number.isFinite(value)) return 1;
    return Math.round(clamp(value, lo, hi) * 10) / 10;
}

function fillWhole(value: unknown, lo: number, hi: number): number {
    if (typeof value !== "number" || !Number.isFinite(value)) return 0;
    return clamp(Math.round(value), lo, hi);
}

/**
 * Fill in anything an entry is missing.
 *
 * Entries live in an in-memory cache that survives navigation, and a page
 * captured before this panel existed (or restored from an older shape) would
 * otherwise reach the renderer with `undefined` where a colour or a layer is
 * expected. Called on every read path, so there is exactly one place that
 * decides what an unset option means.
 */
export function ensureEntryDefaults(entry: Entry): Entry {
    if (typeof entry.layer !== "number" || !Number.isFinite(entry.layer)) entry.layer = 1;
    entry.layer = clamp(Math.round(entry.layer), 1, MAX_LAYER);
    if (entry.font_size === undefined) entry.font_size = null;
    if (entry.font_color === undefined) entry.font_color = null;
    if (entry.stroke_width === undefined) entry.stroke_width = null;
    if (entry.stroke_color === undefined) entry.stroke_color = null;
    if (!entry.bg_mode) entry.bg_mode = "auto";
    if (entry.bg_color === undefined) entry.bg_color = null;
    if (!entry.clean) entry.clean = defaultCleanOptions();
    else fillCleanDefaults(entry.clean);
    if (!TEXT_ALIGNMENTS.includes(entry.text_align)) entry.text_align = "center";
    // Typesetting geometry, per field rather than per object, for the same
    // reason as `fillCleanDefaults`: an entry written by an older build has
    // none of it, and one restored from a build with a wider range may be out
    // of bounds. The neutral values reproduce the pre-geometry render exactly,
    // so an entry that never touched these is unaffected by them existing.
    // `fox_reader.typeset` clamps again on the way in -- this is for the
    // controls and the live preview, not for trust.
    entry.word_spacing = fillMultiplier(entry.word_spacing, MIN_WORD_SPACING, MAX_WORD_SPACING);
    entry.line_spacing = fillMultiplier(entry.line_spacing, MIN_LINE_SPACING, MAX_LINE_SPACING);
    entry.font_scale_x = fillScale(entry.font_scale_x, MIN_FONT_SCALE, MAX_FONT_SCALE);
    entry.font_scale_y = fillScale(entry.font_scale_y, MIN_FONT_SCALE, MAX_FONT_SCALE);
    entry.shift_x = fillWhole(entry.shift_x, -MAX_TEXT_SHIFT, MAX_TEXT_SHIFT);
    entry.shift_y = fillWhole(entry.shift_y, -MAX_TEXT_SHIFT, MAX_TEXT_SHIFT);
    entry.angle_x = fillWhole(entry.angle_x, -MAX_TEXT_ANGLE, MAX_TEXT_ANGLE);
    entry.angle_y = fillWhole(entry.angle_y, -MAX_TEXT_ANGLE, MAX_TEXT_ANGLE);
    entry.angle_z = fillWhole(entry.angle_z, -MAX_TEXT_ANGLE, MAX_TEXT_ANGLE);
    // Speaker attribution: entries created before characters existed (or
    // restored from an older shape) arrive without the field; "none" is the
    // only safe default -- guessing a speaker is worse than missing a tag.
    if (entry.character_id === undefined) entry.character_id = null;
    if (entry.character_id !== null && entry.character_id !== undefined && typeof entry.character_id !== "string")
        entry.character_id = null;
    if (typeof entry.character_id === "string" && !entry.character_id.trim()) entry.character_id = null;
    return entry;
}

// --------------------------------------------------------------------- layers

/**
 * The highest layer this entry may move to.
 *
 * Layers must stay gap-free: a stack of 1,2,4 has nothing at 3, so "4" says
 * nothing about what is drawn over what. The ceiling is therefore one above the
 * highest layer anyone *else* occupies -- and capped at 10.
 */
export function layerCeiling(entries: Entry[], entry: Entry): number {
    let highest = 0;
    for (const other of entries) {
        if (other === entry || other.id === entry.id) continue;
        highest = Math.max(highest, clamp(Math.round(other.layer ?? 1), 1, MAX_LAYER));
    }
    return clamp(highest + 1, 1, MAX_LAYER);
}

/**
 * Squeeze the occupied layers down to 1..k, preserving their order.
 *
 * Run after any change that can vacate a layer (a move, a delete). Without it,
 * deleting the only entry on layer 2 would leave 1 and 3 with a hole between
 * them, and every subsequent ceiling calculation would be reasoning about a
 * stack that does not exist.
 */
export function normalizeLayers(entries: Entry[]): void {
    const used = Array.from(new Set(entries.map((e) => clamp(Math.round(e.layer ?? 1), 1, MAX_LAYER)))).sort(
        (a, b) => a - b,
    );
    const remap = new Map<number, number>();
    used.forEach((layer, idx) => remap.set(layer, idx + 1));
    for (const entry of entries) {
        const current = clamp(Math.round(entry.layer ?? 1), 1, MAX_LAYER);
        entry.layer = remap.get(current) ?? 1;
    }
}

/** Bottom layer first; entries on the same layer keep their list order. */
export function sortByLayer(entries: Entry[]): Entry[] {
    return entries
        .map((entry, idx) => ({ entry, idx }))
        .sort((a, b) => a.entry.layer - b.entry.layer || a.idx - b.idx)
        .map((pair) => pair.entry);
}

// ------------------------------------------------------------- open/closed set

let openEntryId: string | null = null;

export function isEntryOpen(id: string): boolean {
    return openEntryId === id;
}

/** Which entry's panel is open, if any. */
export function getOpenEntry(): string | null {
    return openEntryId;
}

/** Open one entry's panel, closing whichever was open. Only one at a time. */
export function setOpenEntry(id: string | null): void {
    openEntryId = id;
}

export function toggleOpenEntry(id: string): boolean {
    openEntryId = openEntryId === id ? null : id;
    return openEntryId === id;
}

// ------------------------------------------------------- clean capabilities

let capabilities: CleanMethodsResponse | null = null;
let capabilitiesPending: Promise<CleanMethodsResponse | null> | null = null;

/**
 * What to draw before the backend has answered, and if it never does.
 *
 * The knob lists, the presets and the tile sizes are the same tables
 * `fox_reader.clean.tuning` holds. Duplicating them is deliberate: the panel is
 * built synchronously, so it needs *something* to draw one frame before the
 * fetch lands, and a blank select reads as a broken control. The real answer
 * replaces this wholesale, so the copy can only ever be stale for that frame or
 * on a backend too old to send them.
 */
const FALLBACK_CAPABILITIES: CleanMethodsResponse = {
    methods: [
        { id: "region", label: "Selected Region", available: true, reason: "", knobs: [] },
        {
            id: "ppocr",
            label: "PaddleOCR",
            available: false,
            reason: "not reported by the backend yet",
            knobs: [],
        },
        {
            id: "textseg",
            label: "Text Seg",
            available: false,
            reason: "not reported by the backend yet",
            knobs: ["glow", "speed", "tta", "tile"],
        },
    ],
    fills: ["hybrid-level", "hybrid", "level", "pyramid", "telea", "ns"],
    default_fill: "hybrid-level",
    speeds: [
        { value: "best", label: "Best", passes: 24, tta: true },
        { value: "fast", label: "Fast", passes: 6, tta: false },
        { value: "fastest", label: "Fastest", passes: 3, tta: false },
        { value: "single", label: "Single", passes: 1, tta: false },
    ],
    default_speed: "fastest",
    default_tta: true,
    tiles: [
        { value: 0, label: "Off", overlap: 192 },
        { value: 256, label: "256", overlap: 48 },
        { value: 512, label: "512", overlap: 96 },
        { value: 1024, label: "1024", overlap: 192 },
        { value: 2048, label: "2048", overlap: 192 },
    ],
    default_tile: 0,
};

/**
 * Ask the backend once what clean can do, and remember the answer.
 *
 * The panel is built synchronously from whatever is cached, so the first entry
 * opened after a cold start shows the fallback for a moment and is refreshed by
 * `onReady` when the real answer lands. That is deliberate: waiting on a fetch
 * before drawing the panel would make the collapsible feel broken.
 */
export function cleanCapabilities(onReady?: () => void): CleanMethodsResponse {
    if (!capabilities && !capabilitiesPending) {
        capabilitiesPending = api
            .getCleanMethods()
            .then((data) => {
                capabilities = data;
                return data;
            })
            .catch(() => null)
            .finally(() => {
                capabilitiesPending = null;
                if (capabilities && onReady) onReady();
            });
    }
    return capabilities ?? FALLBACK_CAPABILITIES;
}

export function cleanMethodLabel(method: CleanMethod): string {
    const found = cleanCapabilities().methods.find((m) => m.id === method);
    return found?.label ?? method;
}

// -------------------------------------------------------------- eyedropper

const pixelCanvases = new Map<string, HTMLCanvasElement>();
let pickCleanup: (() => void) | null = null;

function canvasForImage(img: HTMLImageElement): HTMLCanvasElement | null {
    if (!img.complete || !img.naturalWidth) return null;
    const cached = pixelCanvases.get(img.src);
    if (cached) return cached;
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    try {
        ctx.drawImage(img, 0, 0);
    } catch {
        return null;
    }
    // One canvas per page, not per pick: a full-page draw is the expensive part,
    // and the same page gets sampled repeatedly while styling a spread.
    if (pixelCanvases.size > 4) pixelCanvases.clear();
    pixelCanvases.set(img.src, canvas);
    return canvas;
}

const hex = (n: number): string => n.toString(16).padStart(2, "0");

/**
 * Let the user click a pixel on the page and resolve to its colour.
 *
 * Resolves `null` if they cancel (Escape / right click) or the page cannot be
 * sampled. Only one pick can be in flight; starting another cancels the first.
 */
export function pickColorFromImage(): Promise<string | null> {
    pickCleanup?.();
    return new Promise((resolve) => {
        const img = document.getElementById("mainImage") as HTMLImageElement | null;
        const container = document.getElementById("imageContainer");
        if (!img || !container || !img.naturalWidth) {
            showNotify("⚠️ Load a page first, then pick a colour from it.");
            resolve(null);
            return;
        }
        const canvas = canvasForImage(img);
        const ctx = canvas?.getContext("2d", { willReadFrequently: true });
        if (!canvas || !ctx) {
            showNotify("❌ This page cannot be sampled.");
            resolve(null);
            return;
        }

        showNotify("🎨 Click a pixel on the page (Esc to cancel)");
        document.body.classList.add("eyedropper-active");

        // The region overlay sits on top of the image, so while picking it is in the
        // way twice over: it hides the very pixels being chosen between, and its
        // shapes take the click before the `img` listener below can see it, which
        // makes any pixel under a region unsamplable. Gone for the duration, and
        // restored to whatever it was rather than cleared -- `finish` is the single
        // exit, so a cancel, a bad pixel and a successful pick all put it back.
        const regionOverlay = document.querySelector<SVGSVGElement>("#regionSvgOverlay");
        const overlayDisplay = regionOverlay?.style.display ?? "";
        if (regionOverlay) regionOverlay.style.display = "none";

        // While a pick is in flight the page is inert everywhere but the image: every
        // other button would otherwise still take the click that was meant for a
        // pixel. Capture phase on `window`, so this runs ahead of any listener on any
        // node, and `stopImmediatePropagation` rather than `stopPropagation` because
        // handlers sharing the blocked node would still fire otherwise.
        //
        // Pointer and mouse down/up as well as `click`: popovers and drag handles act
        // on the press, not the click, and blocking only `click` would let them move
        // under the pick. Keys are left alone -- `onKey` needs its Escape.
        const guard = (event: Event) => {
            const target = event.target as Node | null;
            if (target && img.contains(target)) return;
            event.preventDefault();
            event.stopImmediatePropagation();
        };
        const guarded = ["pointerdown", "pointerup", "mousedown", "mouseup", "click", "dblclick"];
        for (const type of guarded) window.addEventListener(type, guard, true);

        const finish = (value: string | null) => {
            pickCleanup = null;
            if (regionOverlay) regionOverlay.style.display = overlayDisplay;
            for (const type of guarded) window.removeEventListener(type, guard, true);
            document.body.classList.remove("eyedropper-active");
            img.removeEventListener("click", onClick, true);
            window.removeEventListener("keydown", onKey, true);
            window.removeEventListener("contextmenu", onCancel, true);
            resolve(value);
        };
        pickCleanup = () => finish(null);

        const onClick = (event: MouseEvent) => {
            event.preventDefault();
            event.stopPropagation();
            const rect = img.getBoundingClientRect();
            // The image is scaled by the viewer's zoom, so the on-screen offset has
            // to be divided back out to land on the pixel the user actually saw.
            const x = Math.round(((event.clientX - rect.left) / rect.width) * img.naturalWidth);
            const y = Math.round(((event.clientY - rect.top) / rect.height) * img.naturalHeight);
            if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) {
                finish(null);
                return;
            }
            try {
                const [r, g, b] = ctx.getImageData(x, y, 1, 1).data;
                finish(`#${hex(r)}${hex(g)}${hex(b)}`);
            } catch {
                showNotify("❌ Could not read that pixel.");
                finish(null);
            }
        };
        const onKey = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                event.preventDefault();
                finish(null);
            }
        };
        const onCancel = (event: Event) => {
            event.preventDefault();
            // Stopped here, not just defaulted away. This is registered on `window`
            // in the capture phase, so without it the same right-click would also
            // reach whatever region the pointer happens to be over and hide it --
            // see `onContextMenu` in entries.ts. A cancel cancels and nothing else.
            event.stopPropagation();
            finish(null);
        };

        img.addEventListener("click", onClick, true);
        window.addEventListener("keydown", onKey, true);
        window.addEventListener("contextmenu", onCancel, true);
    });
}

// ----------------------------------------------------------- colour control

let openPopover: HTMLElement | null = null;

function closePopover(): void {
    openPopover?.remove();
    openPopover = null;
}

document.addEventListener(
    "pointerdown",
    (event) => {
        if (!openPopover) return;
        const target = event.target as HTMLElement;
        if (openPopover.contains(target) || target.closest(".eo-color-btn")) return;
        closePopover();
    },
    true,
);

window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && openPopover) closePopover();
});

const HEX_RE = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i;

/** `text` as `#rrggbb`, or `null` if it is not a hex colour. */
export function normalizeHex(text: string): string | null {
    const match = HEX_RE.exec(text.trim());
    if (!match) return null;
    const body = match[1].toLowerCase();
    // `#f0c` and `#ff00cc` are the same colour; store the long form so comparing
    // an entry's colour against a palette swatch is a plain string compare.
    const full = body.length === 3 ? `${body[0]}${body[0]}${body[1]}${body[1]}${body[2]}${body[2]}` : body;
    return `#${full}`;
}

interface ColorControlOptions {
    value: string | null;
    /** Label shown when the value is `null`. */
    autoLabel?: string;
    title?: string;
    /**
     * Swatch only, no hex readout. For rows that already carry two other
     * controls -- the value moves into the tooltip and the popover's hex box.
     */
    compact?: boolean;
    onPick: (value: string | null) => void;
}

/** A swatch button that opens Auto / hex / palette / picker / eyedropper. */
function buildColorControl(opts: ColorControlOptions): HTMLElement {
    const button = el("button", opts.compact ? "eo-color-btn is-compact" : "eo-color-btn");
    button.type = "button";
    const autoLabel = opts.autoLabel ?? "Auto";
    const baseTitle = opts.title ?? "Colour";

    // The popover reads this rather than `opts.value`, which goes stale the moment
    // a colour is chosen -- that is what used to leave the "active" ring on the
    // previously selected swatch after a pick.
    let current = opts.value;

    const swatch = el("span", "eo-swatch");
    const label = el("span", "eo-color-label");
    const paint = (value: string | null) => {
        current = value;
        if (value) {
            swatch.style.background = value;
            swatch.classList.remove("is-auto");
            label.textContent = value.toUpperCase();
        } else {
            swatch.style.background = "";
            swatch.classList.add("is-auto");
            label.textContent = autoLabel;
        }
        button.title = `${baseTitle}: ${value ? value.toUpperCase() : autoLabel}`;
    };
    paint(opts.value);
    button.appendChild(swatch);
    if (!opts.compact) button.appendChild(label);

    const choose = (value: string | null) => {
        paint(value);
        closePopover();
        opts.onPick(value);
    };

    button.addEventListener("click", (event) => {
        event.stopPropagation();
        // Both sides of this used to be read straight off `dataset`, and before the
        // first open both are `undefined` -- so `undefined === undefined` matched,
        // and the very first click "closed" a popover that had never been built.
        // Hence the explicit "is one open, and is it this button's".
        if (openPopover && button.dataset.uid && openPopover.dataset.owner === button.dataset.uid) {
            closePopover();
            return;
        }
        closePopover();

        const uid = Math.random().toString(36).slice(2);
        button.dataset.uid = uid;
        const pop = el("div", "eo-pop");
        pop.dataset.owner = uid;

        const head = el("div", "eo-pop-row");
        const autoBtn = el("button", "eo-chip", autoLabel);
        autoBtn.type = "button";
        autoBtn.title = "Let the backend detect this from the artwork";
        if (current === null) autoBtn.classList.add("active");
        autoBtn.addEventListener("click", () => choose(null));

        const hexInput = el("input", "eo-pop-hex") as HTMLInputElement;
        hexInput.type = "text";
        hexInput.spellcheck = false;
        hexInput.maxLength = 7;
        hexInput.placeholder = "#RRGGBB";
        hexInput.value = current ?? "";
        hexInput.title = "Type a hex colour, then press Enter";
        hexInput.addEventListener("input", () => {
            const text = hexInput.value.trim();
            hexInput.classList.toggle("is-bad", text !== "" && normalizeHex(text) === null);
        });
        hexInput.addEventListener("keydown", (keyEvent) => {
            if (keyEvent.key !== "Enter") return;
            keyEvent.preventDefault();
            const parsed = normalizeHex(hexInput.value);
            // Committing an unparseable value would silently write garbage into the
            // entry, so a bad box just stays marked and waits.
            if (parsed) choose(parsed);
            else hexInput.classList.add("is-bad");
        });
        head.append(autoBtn, hexInput);
        pop.appendChild(head);

        const grid = el("div", "eo-pop-grid");
        for (const color of COLOR_PALETTE) {
            const cell = el("button", "eo-pop-swatch");
            cell.type = "button";
            cell.style.background = color;
            cell.title = color.toUpperCase();
            if (current?.toLowerCase() === color.toLowerCase()) cell.classList.add("active");
            cell.addEventListener("click", () => choose(color));
            grid.appendChild(cell);
        }
        pop.appendChild(grid);

        const tools = el("div", "eo-pop-row");
        const native = el("input", "eo-pop-native") as HTMLInputElement;
        native.type = "color";
        native.value = current ?? "#000000";
        native.title = "Custom colour";
        native.addEventListener("input", () => {
            paint(native.value);
            hexInput.value = native.value;
            hexInput.classList.remove("is-bad");
        });
        native.addEventListener("change", () => choose(native.value));
        const dropper = el("button", "eo-chip");
        dropper.type = "button";
        dropper.title = "Pick a colour from a pixel of the page";
        dropper.append(icon("colorize", 14), document.createTextNode("Pick"));
        dropper.addEventListener("click", async () => {
            closePopover();
            const picked = await pickColorFromImage();
            if (picked) {
                paint(picked);
                opts.onPick(picked);
            }
        });
        tools.append(native, dropper);
        pop.appendChild(tools);

        // Body-anchored, because the entries list scrolls and clips: a popover
        // inside the card would be cut off by the list's own overflow.
        document.body.appendChild(pop);
        openPopover = pop;
        const rect = button.getBoundingClientRect();
        const popRect = pop.getBoundingClientRect();
        let top = rect.bottom + 4;
        let left = rect.left;
        if (top + popRect.height > window.innerHeight) top = Math.max(4, rect.top - popRect.height - 4);
        if (left + popRect.width > window.innerWidth) left = Math.max(4, window.innerWidth - popRect.width - 8);
        pop.style.top = `${top}px`;
        pop.style.left = `${left}px`;
    });

    return button;
}

// --------------------------------------------------------------- small parts

/** One labelled group of controls inside a row. */
interface FieldGroup {
    label: string;
    fields: (Node | null)[];
    /** Long-form label for the tooltip, when the visible one is abbreviated. */
    title?: string;
    /** Size to the content instead of sharing the leftover width. */
    hug?: boolean;
    /**
     * Break the row on the panel's width, not on this group's widest option.
     *
     * For a group holding a select whose options come from data: the font family
     * list is as wide as the longest family name installed, so one long name would
     * decide where the row breaks and push the group beside it onto a second line.
     * The group still grows into whatever width is left over.
     */
    fluid?: boolean;
    /**
     * Let the controls wrap onto a second line instead of being squeezed.
     *
     * For a group with more controls than a narrow panel can hold: without it the
     * last one shrinks to its minimum and, if it is itself a flex box, spills into
     * a ragged column. With it, the controls that no longer fit drop as a block.
     */
    wrap?: boolean;
}

/**
 * One `.eo-row` carrying one or more labelled groups.
 *
 * The panel is only about a quarter of the window wide, so a row per label wasted
 * most of it and pushed the interesting controls off the bottom of the card. The
 * first label keeps the fixed width, so the left edge of the fields still lines up
 * all the way down the panel; later labels size to their own text.
 *
 * Each group is wrapped in its own `.eo-group`, and the wrapping happens between
 * groups rather than inside them. Label and fields as bare siblings of a wrapping
 * flex row can break between each other, which left "Outline" alone at the end of
 * one line and its two controls on the next. Wrapped, a group that no longer fits
 * drops whole.
 */
function packRow(...groups: FieldGroup[]): HTMLElement {
    const wrap = el("div", groups.length > 1 ? "eo-row eo-row-multi" : "eo-row");
    groups.forEach((group, idx) => {
        const cell = el("div", group.hug ? "eo-group eo-group-hug" : "eo-group");
        if (group.fluid) cell.classList.add("eo-group-fluid");
        const label = el("span", idx === 0 ? "eo-label" : "eo-label eo-label-inline", group.label);
        if (group.title) label.title = group.title;
        const field = el("div", group.hug ? "eo-field eo-field-hug" : "eo-field");
        if (group.wrap) field.classList.add("eo-field-wrap");
        for (const node of group.fields) if (node) field.appendChild(node);
        cell.append(label, field);
        wrap.appendChild(cell);
    });
    return wrap;
}

function row(labelText: string, ...fields: (Node | null)[]): HTMLElement {
    return packRow({ label: labelText, fields });
}

interface NumberSelectOptions {
    value: number | null;
    min: number;
    max: number;
    suffix?: string;
    title?: string;
    onChange: (value: number | null) => void;
}

/**
 * A number picker: "Auto" plus every integer in `[min, max]`.
 *
 * This replaced a range slider that could never be moved. The slider was disabled
 * whenever the value was `null`, and every entry starts on Auto, so the only live
 * part of the control was the 12px checkbox next to it -- which read as a broken
 * slider rather than a mode switch. A select cannot get into that state, expresses
 * "let the backend detect it" as just another option, and is narrow enough to sit
 * in a row beside two other controls, which a slider plus its readout is not.
 */
function buildNumberSelect(opts: NumberSelectOptions): HTMLSelectElement {
    const values = [{ value: "", label: "Auto", title: "Detected from the artwork" }];
    for (let n = opts.min; n <= opts.max; n += 1) {
        values.push({ value: String(n), label: `${n}${opts.suffix ?? ""}`, title: "" });
    }
    const current = opts.value === null ? "" : String(clamp(Math.round(opts.value), opts.min, opts.max));
    const select = buildSelect(values, current, (value) => {
        opts.onChange(value === "" ? null : Number(value));
    });
    select.classList.add("eo-num-select");
    if (opts.title) select.title = opts.title;
    return select;
}

interface DecimalSelectOptions {
    value: number | null;
    min: number;
    max: number;
    /** Offer "Auto" (meaning `null`) first, with this as its tooltip. */
    auto?: string;
    title?: string;
    onChange: (value: number | null) => void;
}

/**
 * A multiplier picker: every tenth in `[min, max]`, optionally led by "Auto".
 *
 * Tenths because that is the resolution asked for, and finer steps are not
 * readable at bubble sizes anyway. The options are built by counting *integers*
 * and dividing, never by adding 0.1 repeatedly: a select is matched on its value
 * as a string, and the "0.7000000000000001" that accumulating would produce
 * matches nothing, leaving the control blank at exactly one setting.
 */
function buildDecimalSelect(opts: DecimalSelectOptions): HTMLSelectElement {
    const values = opts.auto ? [{ value: "", label: "Auto", title: opts.auto }] : [];
    for (let tenths = Math.round(opts.min * 10); tenths <= Math.round(opts.max * 10); tenths += 1) {
        const text = (tenths / 10).toFixed(1);
        values.push({ value: text, label: `${text}×`, title: "" });
    }
    const current = opts.value === null ? "" : (Math.round(clamp(opts.value, opts.min, opts.max) * 10) / 10).toFixed(1);
    const select = buildSelect(values, current, (value) => {
        opts.onChange(value === "" ? null : Number(value));
    });
    select.classList.add("eo-num-select");
    if (opts.title) select.title = opts.title;
    return select;
}

interface IntInputOptions {
    value: number;
    min: number;
    max: number;
    title?: string;
    onChange: (value: number) => void;
}

/**
 * A whole-number box, for the ranges a select cannot hold.
 *
 * Shift spans the page and rotation spans 361 degrees, so the `<option>` list
 * that suits font size and layer would be thousands of entries long and unusable
 * with either. A number input types, steps and scrubs in one control instead.
 *
 * Committed on `change` rather than `input`: the spinner and the arrow keys fire
 * it immediately, so stepping still repaints per step, while typing a three-digit
 * angle does not repaint the preview twice on the way to it. Whatever the field
 * ends up holding is written back, so a cleared or nonsense box settles visibly
 * on the value that was actually stored rather than silently disagreeing with it.
 */
function buildIntInput(opts: IntInputOptions): HTMLInputElement {
    const input = el("input", "eo-num-input") as HTMLInputElement;
    const settle = (value: number): number => clamp(Math.round(value), opts.min, opts.max);
    let current = settle(opts.value);
    input.type = "number";
    input.step = "1";
    input.min = String(opts.min);
    input.max = String(opts.max);
    input.value = String(current);
    if (opts.title) input.title = opts.title;
    input.addEventListener("change", () => {
        const raw = Number(input.value);
        const next = input.value.trim() && Number.isFinite(raw) ? settle(raw) : 0;
        input.value = String(next);
        if (next === current) return;
        current = next;
        opts.onChange(next);
    });
    return input;
}

/**
 * A control with a one-letter tag in front of it.
 *
 * Shift, scale and rotation are two or three of the same widget side by side, and
 * at this width there is no room for a label apiece. A `<label>` wrapper gives the
 * tag click-to-focus for free and keeps the pair from breaking across lines.
 */
function taggedField(tag: string, title: string, control: HTMLElement): HTMLElement {
    const wrap = el("label", "eo-axis");
    wrap.title = title;
    wrap.append(el("span", "eo-axis-tag", tag), control);
    return wrap;
}

function buildSelect(
    values: { value: string; label: string; disabled?: boolean; title?: string }[],
    current: string,
    onChange: (value: string) => void,
): HTMLSelectElement {
    const select = el("select", "eo-select") as HTMLSelectElement;
    for (const item of values) {
        const option = el("option") as HTMLOptionElement;
        option.value = item.value;
        option.textContent = item.label;
        if (item.disabled) option.disabled = true;
        if (item.title) option.title = item.title;
        select.appendChild(option);
    }
    select.value = current;
    select.addEventListener("change", () => onChange(select.value));
    return select;
}

/**
 * A labelled checkbox.
 *
 * `note.inert` is for a switch that is real and stored but has nothing to act on
 * at the current settings -- TTA under a preset that does no flip averaging. It
 * stays clickable, because the value is remembered and does apply once the
 * setting that uses it is chosen; it is only dimmed, and `note.title` says why.
 * Hiding it instead would make the option look like it does not exist.
 */

function buildToggle(
    label: string,
    checked: boolean,
    onChange: (value: boolean) => void,
    note?: { title?: string; inert?: boolean },
): HTMLElement {
    const wrap = el("label", note?.inert ? "eo-check eo-check-inert" : "eo-check");
    const box = el("input") as HTMLInputElement;
    box.type = "checkbox";
    box.checked = checked;
    box.addEventListener("change", () => onChange(box.checked));
    wrap.append(box, document.createTextNode(label));
    if (note?.title) wrap.title = note.title;
    return wrap;
}

// -------------------------------------------------------------- the panel

export interface OptionsContext {
    /** All entries on the page -- needed for the layer ceiling. */
    entries: Entry[];
    /**
     * Repaint this entry's live preview and card summary, leaving the panel's DOM
     * alone. Used by every control that only changes a value: rebuilding the panel
     * on each slider commit would drop focus and reset the popover.
     */
    refresh: () => void;
    /** Rebuild this entry's panel -- for changes that add or remove controls. */
    rebuild: () => void;
    /** Re-render the whole list (layer changes reorder the drawing). */
    refreshAll: () => void;
}

function buildLayerRow(entry: Entry, ctx: OptionsContext): HTMLElement {
    const wrap = el("div", "eo-stepper");
    const down = el("button", "eo-step");
    down.type = "button";
    down.appendChild(icon("remove", 14));
    const value = el("span", "eo-layer-val", String(entry.layer));
    const up = el("button", "eo-step");
    up.type = "button";
    up.appendChild(icon("add", 14));

    const ceiling = layerCeiling(ctx.entries, entry);
    down.disabled = entry.layer <= 1;
    up.disabled = entry.layer >= ceiling;
    up.title =
        entry.layer >= ceiling
            ? ceiling >= MAX_LAYER
                ? `Layer ${MAX_LAYER} is the top`
                : `Nothing is on layer ${ceiling} yet, so ${ceiling} is as high as this can go`
            : "Move up";
    down.title = "Move down";

    // The "bottom / over 2" hint used to sit beside the stepper as its own text.
    // It is worth about a third of the row's width, and this row now shares that
    // width with the alignment control, so it moved into the tooltip.
    const paintTitle = () => {
        wrap.title = entry.layer === 1 ? "Layer 1 -- the bottom" : `Layer ${entry.layer}, over ${entry.layer - 1}`;
    };
    paintTitle();

    const move = (delta: number) => {
        const next = clamp(entry.layer + delta, 1, layerCeiling(ctx.entries, entry));
        if (next === entry.layer) return;
        entry.layer = next;
        normalizeLayers(ctx.entries);
        value.textContent = String(entry.layer);
        paintTitle();
        ctx.refreshAll();
    };
    down.addEventListener("click", () => move(-1));
    up.addEventListener("click", () => move(1));

    wrap.append(down, value, up);
    return wrap;
}

function buildAlignRow(entry: Entry, ctx: OptionsContext): HTMLElement {
    const seg = el("div", "eo-seg");
    for (const align of TEXT_ALIGNMENTS) {
        const button = el("button");
        button.type = "button";
        button.title = align;
        button.appendChild(icon(ALIGNMENT_ICONS[align], 15));
        if (entry.text_align === align) button.classList.add("active");
        button.addEventListener("click", () => {
            if (entry.text_align === align) return;
            entry.text_align = align as TextAlignment;
            seg.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
            button.classList.add("active");
            ctx.refresh();
        });
        seg.appendChild(button);
    }
    return seg;
}

/**
 * Who speaks this bubble, for VNTL-style character metadata.
 *
 * A plain value change, not a structural one: picking a speaker only retags
 * future translations, so the panel stays put (`refresh`, not `rebuild`) and
 * the select keeps focus for rapid assignment. A stored id missing from the
 * roster (deleted character, fresh import with new ids) reads as none rather
 * than as a stale selection.
 */
function buildSpeakerSelect(entry: Entry, ctx: OptionsContext): HTMLSelectElement {
    const roster = getCharacters();
    const current = entry.character_id ?? null;
    const known = new Set(roster.map((c) => c.meta_id));
    const select = buildSelect(
        [
            { value: "", label: "None" },
            ...roster.map((char) => ({ value: char.meta_id, label: characterPickerLabel(char) })),
        ],
        current && known.has(current) ? current : "",
        (value) => {
            entry.character_id = value || null;
            ctx.refresh();
        },
    );
    select.classList.add("eo-speaker");
    select.title = "Who speaks this bubble (character metadata for VNTL models)";
    select.setAttribute("aria-label", "Speaker for this entry");
    return select;
}

function buildFontRow(entry: Entry, ctx: OptionsContext): HTMLElement {
    const fonts = state.currentFontsData;
    if (!fonts.length) return el("span", "eo-hint", "no fonts loaded");
    const select = buildSelect(
        fonts.map((font) => ({ value: font.font_filename, label: font.font_name })),
        entry.fontfile,
        (value) => {
            const font = fonts.find((f) => f.font_filename === value);
            if (!font) return;
            entry.fontfile = font.font_filename;
            entry.fontname = `'${font.font_name.replace(/\s+/g, "_")}', sans-serif`;
            select.style.fontFamily = entry.fontname;
            ctx.refresh();
        },
    );
    for (const option of Array.from(select.options)) {
        const font = fonts.find((f) => f.font_filename === option.value);
        if (font) option.style.fontFamily = `'${font.font_name.replace(/\s+/g, "_")}', sans-serif`;
    }
    select.style.fontFamily = entry.fontname || "inherit";
    select.classList.add("eo-font");
    return select;
}

/**
 * Word spacing, line spacing and the X/Y stretch.
 *
 * The half of the geometry that takes part in the automatic font fit: "Auto" on
 * either spacing is the face's own metrics -- exactly what the typesetter did
 * before these existed -- and the scales are neutral at 1.0. Because they are
 * fitted rather than applied afterwards, widening the word gap under an automatic
 * size wraps sooner and settles on a smaller face instead of running out of the
 * bubble, and a 2.0 stretch wraps at half the width and then fills it.
 */
function buildSpacingRow(entry: Entry, ctx: OptionsContext): HTMLElement {
    return packRow(
        {
            label: "Spacing",
            title: "Gap between words and between lines, as multiples of the font's own metrics",
            wrap: true,
            fields: [
                taggedField(
                    "W",
                    "Word spacing, as a multiple of the font's own space",
                    buildDecimalSelect({
                        value: entry.word_spacing,
                        min: MIN_WORD_SPACING,
                        max: MAX_WORD_SPACING,
                        auto: "The font's own space advance",
                        onChange: (value) => {
                            entry.word_spacing = value;
                            ctx.refresh();
                        },
                    }),
                ),
                taggedField(
                    "L",
                    "Line spacing, as a multiple of the font's own line height",
                    buildDecimalSelect({
                        value: entry.line_spacing,
                        min: MIN_LINE_SPACING,
                        max: MAX_LINE_SPACING,
                        auto: "The font's own line height",
                        onChange: (value) => {
                            entry.line_spacing = value;
                            ctx.refresh();
                        },
                    }),
                ),
            ],
        },
        {
            label: "Scale",
            title: "Stretch the lettering. Fitted, so the text still lands inside the region",
            hug: true,
            wrap: true,
            fields: [
                taggedField(
                    "X",
                    "Horizontal stretch",
                    buildDecimalSelect({
                        value: entry.font_scale_x,
                        min: MIN_FONT_SCALE,
                        max: MAX_FONT_SCALE,
                        onChange: (value) => {
                            entry.font_scale_x = value ?? 1;
                            ctx.refresh();
                        },
                    }),
                ),
                taggedField(
                    "Y",
                    "Vertical stretch",
                    buildDecimalSelect({
                        value: entry.font_scale_y,
                        min: MIN_FONT_SCALE,
                        max: MAX_FONT_SCALE,
                        onChange: (value) => {
                            entry.font_scale_y = value ?? 1;
                            ctx.refresh();
                        },
                    }),
                ),
            ],
        },
    );
}

/**
 * The nudge and the rotation -- the half applied after the text is set.
 *
 * Both move the lettering alone: the plate behind it stays where the region is,
 * which is the point of having them. Shift is in image pixels from where the fit
 * put the block, rotation in degrees about the block's own centre, X and Y
 * tilting the plane and Z spinning it clockwise.
 *
 * The shift boxes are bounded by the page, when there is one on screen; the
 * renderer clamps again, because it is the only place that knows where the
 * tilted, spun block actually ends up.
 */
function buildTransformRow(entry: Entry, ctx: OptionsContext): HTMLElement {
    const page = imageBounds();
    const shift = (
        axis: "x" | "y",
        value: number,
        limit: number,
        title: string,
    ): HTMLElement =>
        taggedField(
            axis.toUpperCase(),
            title,
            buildIntInput({
                value,
                min: -limit,
                max: limit,
                onChange: (next) => {
                    if (axis === "x") entry.shift_x = next;
                    else entry.shift_y = next;
                    ctx.refresh();
                },
            }),
        );
    const angle = (tag: string, value: number, title: string, apply: (next: number) => void): HTMLElement =>
        taggedField(
            tag,
            title,
            buildIntInput({
                value,
                min: -MAX_TEXT_ANGLE,
                max: MAX_TEXT_ANGLE,
                onChange: (next) => {
                    apply(next);
                    ctx.refresh();
                },
            }),
        );

    return packRow(
        {
            label: "Shift",
            title: "Move the text -- not the background -- from where it was set, in image pixels",
            wrap: true,
            fields: [
                shift("x", entry.shift_x, page ? page.w : MAX_TEXT_SHIFT, "Rightwards, in image pixels"),
                shift("y", entry.shift_y, page ? page.h : MAX_TEXT_SHIFT, "Downwards, in image pixels"),
            ],
        },
        {
            label: "Rotate",
            title: "Turn the text -- not the background -- about its centre, in degrees",
            hug: true,
            wrap: true,
            fields: [
                angle("X", entry.angle_x, "Tilt about the horizontal axis", (next) => {
                    entry.angle_x = next;
                }),
                angle("Y", entry.angle_y, "Tilt about the vertical axis", (next) => {
                    entry.angle_y = next;
                }),
                angle("Z", entry.angle_z, "Spin clockwise in the page", (next) => {
                    entry.angle_z = next;
                }),
            ],
        },
    );
}

/** The knobs a method reads, from the backend if it said, else from the table. */
function knobsFor(caps: CleanMethodsResponse, method: CleanMethod): CleanKnob[] {
    const info = caps.methods.find((m) => m.id === method);
    if (info?.knobs) return info.knobs;
    const fallback = FALLBACK_CAPABILITIES.methods.find((m) => m.id === method);
    return fallback?.knobs ?? [];
}

/**
 * The detector-tuning line: speed, tile size, and the boolean switches.
 *
 * One row, because it is read as one decision ("how hard should this try"). Which
 * controls appear follows the method's `knobs`, so PaddleOCR does not offer a
 * Glow switch it never reads and Selected Region does not offer a speed for a
 * detector it never runs. Transport is always there: it gates the reconstruction
 * stage's escalation to offset transport, which every method goes through.
 */
function buildTweakRow(entry: Entry, ctx: OptionsContext, caps: CleanMethodsResponse): HTMLElement {
    const clean = entry.clean;
    const knobs = knobsFor(caps, clean.method);
    const fields: (Node | null)[] = [];

    const speeds = caps.speeds?.length ? caps.speeds : FALLBACK_CAPABILITIES.speeds!;
    const preset = speeds.find((s) => s.value === clean.speed);

    if (knobs.includes("speed")) {
        const select = buildSelect(
            speeds.map((s) => ({
                value: s.value,
                label: s.label,
                title: `${s.passes} forward pass${s.passes === 1 ? "" : "es"} at most`,
            })),
            preset ? clean.speed : (caps.default_speed ?? "fastest"),
            (value) => {
                clean.speed = value;
                // Structural: whether the TTA switch has anything to act on
                // depends on the preset, so the row is rebuilt rather than left
                // claiming a pass count it no longer costs.
                ctx.rebuild();
            },
        );
        select.title =
            "How many forward passes to spend" + (preset ? ` -- ${preset.label} costs up to ${preset.passes}` : "");
        select.classList.add("eo-tune-select");
        fields.push(select);
    }

    if (knobs.includes("tile")) {
        const tiles = caps.tiles?.length ? caps.tiles : FALLBACK_CAPABILITIES.tiles!;
        const chosenTile = tiles.find((t) => t.value === clean.tile) ?? tiles[0];
        const select = buildSelect(
            tiles.map((t) => ({
                value: String(t.value),
                label: t.label,
                title:
                    t.value === 0 ? "Predict the whole region at once" : `${t.value} px tiles, ${t.overlap} px overlap`,
            })),
            String(chosenTile.value),
            (value) => {
                clean.tile = Number(value);
                ctx.rebuild();
            },
        );
        select.classList.add("eo-num-select");
        // The overlap is not a separate control: the backend pairs one with each
        // tile size, so showing it here as a fixed fact is the whole truth.
        select.title =
            chosenTile.value === 0
                ? "Tiling off -- the whole region is predicted at once"
                : `Tile size, ${chosenTile.overlap} px overlap`;
        fields.push(select);
    }

    const flags = el("div", "eo-flags");
    if (knobs.includes("glow")) {
        flags.appendChild(
            buildToggle(
                "Glow",
                clean.glow,
                (value) => {
                    clean.glow = value;
                    ctx.refresh();
                },
                { title: "Grow the mask over halos and soft outlines around the lettering" },
            ),
        );
    }
    if (knobs.includes("tta")) {
        const inert = !preset?.tta;
        flags.appendChild(
            buildToggle(
                "TTA",
                clean.tta,
                (value) => {
                    clean.tta = value;
                    ctx.refresh();
                },
                {
                    inert,
                    title: inert
                        ? `Flip averaging -- ${preset?.label ?? "this speed"} does none, so this ` +
                          "only takes effect on Best"
                        : "Average four flipped views of each pass (4x the time)",
                },
            ),
        );
    }
    flags.appendChild(
        buildToggle(
            "Transport",
            clean.transport,
            (value) => {
                clean.transport = value;
                ctx.refresh();
            },
            { title: "Let wide, textured gaps escalate to offset transport instead of diffusion" },
        ),
    );
    fields.push(flags);

    return packRow({
        label: "Tweaks",
        fields,
        title: "How hard the clean should try",
        // Five controls at their widest. On a panel that cannot hold them the
        // switches drop to a second line as a block, rather than the flags box
        // being squeezed into a one-per-line column.
        wrap: true,
    });
}

function buildCleanRows(entry: Entry, ctx: OptionsContext): HTMLElement {
    const box = el("div", "eo-clean");
    const caps = cleanCapabilities(() => ctx.rebuild());
    const clean = entry.clean;

    const methodSelect = buildSelect(
        caps.methods.map((method) => ({
            value: method.id,
            label: method.available ? method.label : `${method.label} (unavailable)`,
            disabled: !method.available,
            title: method.reason,
        })),
        clean.method,
        (value) => {
            clean.method = value as CleanMethod;
            // Structural: each method reads a different set of knobs, so the
            // tweaks row is rebuilt rather than left offering controls the new
            // method ignores.
            ctx.rebuild();
        },
    );
    const chosen = caps.methods.find((m) => m.id === clean.method);
    if (chosen && !chosen.available) methodSelect.classList.add("eo-warn");

    const fills = caps.fills.length ? caps.fills : FALLBACK_CAPABILITIES.fills;
    const fillSelect = buildSelect(
        fills.map((fill) => ({ value: fill, label: fill, title: fill })),
        fills.includes(clean.fill) ? clean.fill : caps.default_fill,
        (value) => {
            clean.fill = value;
            ctx.refresh();
        },
    );
    // Method decides *what* is removed and fill decides *how* the gap is rebuilt;
    // they are read together, so they belong on one line.
    box.appendChild(
        packRow(
            { label: "Method", fields: [methodSelect], title: "Where text-clean gets the shape it removes" },
            { label: "Fill", fields: [fillSelect], title: "How the removed area is reconstructed" },
        ),
    );
    if (chosen && !chosen.available && chosen.reason) {
        box.appendChild(row("", el("span", "eo-hint eo-warn-text", chosen.reason)));
    }

    box.appendChild(buildTweakRow(entry, ctx, caps));
    return box;
}

/** Build the body of `entry`'s collapsible. */
export function buildEntryOptions(entry: Entry, ctx: OptionsContext): HTMLElement {
    ensureEntryDefaults(entry);
    const panel = el("div", "entry-collapse");
    panel.addEventListener("click", (event) => event.stopPropagation());

    const grid = el("div", "eo-grid");

    // Two controls per row wherever they are read together: stacking one per row
    // ran the panel past the bottom of a card in a sidebar this narrow, and left
    // two thirds of every row empty to do it. Speaker sits beside Align -- the
    // row wraps between groups on a very narrow rail, so it drops below rather
    // than crushing the stepper.
    grid.appendChild(
        packRow(
            { label: "Layer", fields: [buildLayerRow(entry, ctx)], hug: true },
            { label: "Align", fields: [buildAlignRow(entry, ctx)], hug: true },
            {
                label: "Speaker",
                title: "Who speaks this bubble (character metadata for VNTL models)",
                fields: [buildSpeakerSelect(entry, ctx)],
            },
        ),
    );

    // Font, size, colour, then outline width and colour -- all one line. The
    // family select was the only wide control in the panel and it sat alone on
    // its row with the outline on the next, which spent two rows on five
    // controls that are all read together while looking at the same glyphs.
    grid.appendChild(
        packRow(
            {
                label: "Font",
                title: "Family, size and colour",
                // Otherwise the longest installed family name, not the panel,
                // decides whether the outline still fits on this line.
                fluid: true,
                fields: [
                    buildFontRow(entry, ctx),
                    buildNumberSelect({
                        value: entry.font_size,
                        min: MIN_FONT_SIZE,
                        max: MAX_FONT_SIZE,
                        title: "Font size, in image pixels",
                        onChange: (value) => {
                            entry.font_size = value;
                            ctx.refresh();
                        },
                    }),
                    buildColorControl({
                        value: entry.font_color,
                        title: "Font colour",
                        compact: true,
                        onPick: (value) => {
                            entry.font_color = value;
                            ctx.refresh();
                        },
                    }),
                ],
            },
            {
                label: "Outline",
                title: "Outline width, in image pixels, and its colour",
                hug: true,
                fields: [
                    buildNumberSelect({
                        value: entry.stroke_width,
                        min: MIN_STROKE_WIDTH,
                        max: MAX_STROKE_WIDTH,
                        title: "Outline width, in image pixels",
                        onChange: (value) => {
                            entry.stroke_width = value;
                            ctx.refresh();
                        },
                    }),
                    buildColorControl({
                        value: entry.stroke_color,
                        title: "Outline colour",
                        compact: true,
                        onPick: (value) => {
                            entry.stroke_color = value;
                            ctx.refresh();
                        },
                    }),
                ],
            },
        ),
    );

    // Typesetting geometry, in the order it is applied: first the spacing and
    // stretch that the fit takes into account, then the nudge and the rotation
    // that happen to the finished block. Both rows sit under Font because they
    // are read as adjustments to it -- and above BG, because neither of them
    // touches the background they are deliberately not moving.
    grid.appendChild(buildSpacingRow(entry, ctx));
    grid.appendChild(buildTransformRow(entry, ctx));

    const bgField: Node[] = [
        buildSelect(
            [
                { value: "auto", label: "Auto" },
                { value: "color", label: "Colour" },
                { value: "transparent", label: "None" },
                { value: "clean", label: "Text Clean" },
            ],
            entry.bg_mode,
            (value) => {
                entry.bg_mode = value as BgMode;
                if (entry.bg_mode === "color" && !entry.bg_color) entry.bg_color = "#ffffff";
                // Structural: switching mode adds or drops the colour swatch and
                // the whole clean block, so the panel itself has to be rebuilt.
                ctx.rebuild();
            },
        ),
    ];
    if (entry.bg_mode === "color") {
        bgField.push(
            buildColorControl({
                value: entry.bg_color,
                title: "Background colour",
                compact: true,
                onPick: (value) => {
                    entry.bg_color = value;
                    ctx.refresh();
                },
            }),
        );
    }
    // "BG" rather than "Background": the label column is 58px and the word is not,
    // and the select right beside it spells the mode out in full anyway.
    grid.appendChild(packRow({ label: "BG", fields: bgField, title: "Region background" }));

    panel.appendChild(grid);
    if (entry.bg_mode === "clean") panel.appendChild(buildCleanRows(entry, ctx));
    return panel;
}
