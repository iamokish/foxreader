import type { TextAlignment, Region, Point, BgMode, CleanOptions, TextGeometry } from "./types";

export type ReaderFitMode = "width" | "height";

/** Which copy of a page is on screen. See `State.viewVariant`. */
export type PageVariant = "original" | "saved";

export interface FontData {
    font_filename: string;
    font_name: string;
    font_data_uri: string;
    font_format: string;
}

/** Layers run 1 (bottom) to 10 (top), with no gaps -- see `layerCeiling`. */
export const MAX_LAYER = 10;

export const MIN_FONT_SIZE = 6;
export const MAX_FONT_SIZE = 72;
export const MIN_STROKE_WIDTH = 0;
export const MAX_STROKE_WIDTH = 32;
export const DEFAULT_STROKE_WIDTH = 2;

/**
 * Per-entry geometry ranges, mirroring the constants of the same names in
 * `fox_reader.typeset` -- which is the authority: the backend clamps every
 * value it is given, so a project saved by a build with a wider range still
 * renders. These bound the controls and the preview.
 *
 * The spacings are offered in tenths, hence `SPACING_STEP`; a `null` spacing
 * is the "Auto" option, meaning the metrics the face itself asks for.
 */
export const MIN_WORD_SPACING = 0;
export const MAX_WORD_SPACING = 3;
export const MIN_LINE_SPACING = 0.5;
export const MAX_LINE_SPACING = 3;
export const MIN_FONT_SCALE = 0.5;
export const MAX_FONT_SCALE = 2;
/** One decimal place, as asked for: the multipliers step in tenths. */
export const SPACING_STEP = 0.1;
/** Degrees, each axis. */
export const MAX_TEXT_ANGLE = 180;
/** A raw guard only -- the real limit is the page edge, applied at draw time. */
export const MAX_TEXT_SHIFT = 100000;

/**
 * Geometry before anyone touches it, i.e. the values that reproduce the render
 * from before these options existed. Frozen and shared: it is the canonical
 * neutral, and `ensureEntryDefaults` settles every entry onto it.
 */
export const PLAIN_GEOMETRY: Readonly<TextGeometry> = Object.freeze({
    word_spacing: null,
    line_spacing: null,
    font_scale_x: 1,
    font_scale_y: 1,
    shift_x: 0,
    shift_y: 0,
    angle_x: 0,
    angle_y: 0,
    angle_z: 0,
});

export interface Entry extends TextGeometry {
    id: string;
    ocr_text: string;
    text: string;
    fontfile: string;
    fontname: string;
    region: Region;
    color: string;
    text_align: TextAlignment;
    visible: boolean;
    detected_bg?: string;
    detected_text_color?: string;

    /**
     * Speaker of this entry, as a character `meta_id` from the roster, or
     * null for "none". Defaults to null for new entries; the selector on the
     * entry card edits it. Survives re-renders; cleared entries drop it with
     * everything else.
     */
    character_id?: string | null;

    /** Stacking order. 1 is drawn first (bottom), 10 last (top). */
    layer: number;

    /** `null` means auto -- the backend detects it from the artwork. */
    font_size: number | null;
    font_color: string | null;
    stroke_width: number | null;
    stroke_color: string | null;

    bg_mode: BgMode;
    bg_color: string | null;
    clean: CleanOptions;

    // ...and the nine `TextGeometry` fields -- word/line spacing, font scale,
    // shift and rotation -- which travel to the backend under the same names.
}

export interface State {
    currentFontsData: FontData[];
    currentlySelectedEntryId: string | null;
    lastCapturedRegion: Region | null;
    pageEntriesCache: Record<string, Entry[]>;
    currentTrackingColorIdx: number;
    svgOverlayElement: SVGSVGElement | null;
    zoomLevel: number;
    currentImageFile: string;
    fitZoomLevel: number;
    readerFit: ReaderFitMode;
    isGrayScaleEnabled: boolean;
    isHorizontalFitEnabled: boolean;
    isAutoOCREnabled: boolean;
    isAutoTranslateEnabled: boolean;
    isLiveInpaintedEnabled: boolean;
    isContextEnabled: boolean;
    /** Whether character metadata rides on MTL requests (default on). */
    isCharactersEnabled: boolean;
    isDrawing: boolean;
    lassoPoints: Point[];
    isPanning: boolean;
    startX: number | undefined;
    startY: number | undefined;
    scrollLeft: number | undefined;
    scrollTop: number | undefined;
    panX: number;
    panY: number;
    isCaptureMode: boolean;
    isEditingText: boolean;
    stripMode: boolean;
    stripZoom: number;
    stripScroll: Record<string, number>;
    currentFiles: string[];
    aspectCache: Record<string, number>;
    stripOverride: "auto" | "on" | "off";
    stripEligible: boolean;

    /** The folder the pages are read from, as the backend resolved it. */
    sourceDir: string;
    /** The folder saves are written to. Equal to `sourceDir` when saving in place. */
    destDir: string;
    /** Whether the two are one folder, i.e. saving overwrites the originals. */
    sameDir: boolean;
    /** Pages with a typeset copy in `destDir`. Always empty when `sameDir`. */
    savedFiles: Set<string>;
    /**
     * Which copy of a page the viewer shows.
     *
     * A view state only: OCR, bubble detection and typesetting always read the
     * original, so flipping this can never change what a render produces.
     */
    viewVariant: PageVariant;
}

export const state: State = {
    currentFontsData: [],
    currentlySelectedEntryId: null,
    lastCapturedRegion: null,
    pageEntriesCache: {},
    currentTrackingColorIdx: 0,
    svgOverlayElement: null,
    zoomLevel: 1.0,
    currentImageFile: "",
    fitZoomLevel: 1.0,
    readerFit: "height",
    isGrayScaleEnabled: false,
    isHorizontalFitEnabled: false,
    isAutoOCREnabled: false,
    isAutoTranslateEnabled: true,
    isLiveInpaintedEnabled: true,
    isContextEnabled: true,
    isCharactersEnabled: true,
    isDrawing: false,
    lassoPoints: [],
    isPanning: false,
    startX: undefined,
    startY: undefined,
    scrollLeft: undefined,
    scrollTop: undefined,
    panX: 0,
    panY: 0,
    isCaptureMode: false,
    isEditingText: false,
    stripMode: false,
    stripZoom: 1,
    stripScroll: {},
    currentFiles: [],
    aspectCache: {},
    stripOverride: "auto",
    stripEligible: false,
    sourceDir: "",
    destDir: "",
    sameDir: false,
    savedFiles: new Set<string>(),
    viewVariant: "original",
};

/**
 * True when what is on screen is the typeset copy from the destination folder
 * rather than the original scan.
 *
 * Deliberately per-page: `viewVariant` is a sticky preference, but a page with
 * no saved copy is always shown -- and edited -- as the original. Editing is
 * blocked while this is true, because the entries and their outlines describe
 * the original page and the saved copy already has the text burned into it.
 */
export function isViewingSaved(): boolean {
    return state.viewVariant === "saved" && !state.sameDir && state.savedFiles.has(state.currentImageFile);
}

export const WORKBENCH_PALETTE: string[] = [
    "#00FF66",
    "#FF007F",
    "#FFCC00",
    "#FF3333",
    "#9933FF",
    "#3385FF",
    "#FF6600",
    "#FF00FF",
    "#CCFF00",
    "#FF9999",
    "#66FFFF",
    "#99FF33",
    "#FF3399",
    "#33FF99",
    "#FF9933",
    "#3399FF",
    "#FF33FF",
    "#99FF99",
    "#FFCC99",
    "#99CCFF",
    "#FF6699",
    "#66FF66",
    "#CC99FF",
    "#FF99CC",
    "#99FFCC",
    "#FFCC66",
    "#6699FF",
    "#FF66CC",
    "#99FF66",
    "#FF9966",
    "#66FFCC",
];

/**
 * The alignments, in the order the toolbar button cycles through them.
 *
 * Three, matching `typeset.TEXT_ALIGNMENTS`. Justify and vertical were dropped:
 * justify needs per-word placement that only reads well in a wide column, and a
 * speech bubble is never one, while vertical is a different typesetting model
 * (one token per line) rather than an alignment.
 */
export const TEXT_ALIGNMENTS: TextAlignment[] = ["center", "right", "left"];

/** Material icon per alignment. */
export const ALIGNMENT_ICONS: Record<TextAlignment, string> = {
    left: "format_align_left",
    center: "format_align_center",
    right: "format_align_right",
};

/**
 * The palette offered for font, outline and background colours.
 *
 * Greys first, then hues at two tones: lettering is nearly always black, white
 * or a flat spot colour, so the shades that actually get used are one row away
 * rather than buried in a 256-swatch grid. Anything else is a click away on the
 * picker or the eyedropper.
 */
export const COLOR_PALETTE: string[] = [
    "#000000",
    "#404040",
    "#808080",
    "#c0c0c0",
    "#ffffff",
    "#7f1d1d",
    "#dc2626",
    "#f97316",
    "#facc15",
    "#fef08a",
    "#14532d",
    "#16a34a",
    "#22d3ee",
    "#0ea5e9",
    "#1e3a8a",
    "#4c1d95",
    "#9333ea",
    "#ec4899",
    "#fbcfe8",
    "#78350f",
];
