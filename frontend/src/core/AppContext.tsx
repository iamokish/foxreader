import { createContext } from "preact";
import { useReducer } from "preact/hooks";
import type { Region, Point, TextAlignment } from "./types";

export interface FontData {
    font_filename: string;
    font_name: string;
    font_data_uri: string;
    font_format: string;
}

export interface Entry {
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
    character_id?: string | null;
}

export type Action =
    | { type: "SET_CURRENT_IMAGE"; payload: string }
    | { type: "SET_ZOOM_LEVEL"; payload: number }
    | { type: "SET_FIT_ZOOM_LEVEL"; payload: number }
    | { type: "SET_CAPTURE_MODE"; payload: boolean }
    | { type: "SET_DRAWING"; payload: boolean }
    | { type: "SET_LASSO_POINTS"; payload: Point[] }
    | { type: "SET_PANNING"; payload: boolean }
    | { type: "SET_START_POSITION"; payload: { x: number; y: number } }
    | { type: "SET_SCROLL_POSITION"; payload: { left: number; top: number } }
    | { type: "SET_GRAYSCALE"; payload: boolean }
    | { type: "SET_HORIZONTAL_FIT"; payload: boolean }
    | { type: "SET_AUTO_OCR"; payload: boolean }
    | { type: "SET_AUTO_TRANSLATE"; payload: boolean }
    | { type: "SET_LIVE_INPAINT"; payload: boolean }
    | { type: "SET_CONTEXT"; payload: boolean }
    | { type: "SET_CHARACTERS_ENABLED"; payload: boolean }
    | { type: "SET_FONTS_DATA"; payload: FontData[] }
    | { type: "SET_SELECTED_ENTRY_ID"; payload: string | null }
    | { type: "SET_LAST_CAPTURED_REGION"; payload: Region | null }
    | { type: "SET_PAGE_ENTRIES"; payload: { page: string; entries: Entry[] } }
    | { type: "ADD_ENTRIES"; payload: { page: string; entries: Entry[] } }
    | { type: "UPDATE_ENTRY"; payload: { page: string; entryId: string; updates: Partial<Entry> } }
    | { type: "DELETE_ENTRY"; payload: { page: string; entryId: string } }
    | { type: "CLEAR_PAGE_ENTRIES"; payload: string }
    | { type: "SET_TRACKING_COLOR_IDX"; payload: number }
    | { type: "SET_SVG_OVERLAY"; payload: SVGSVGElement | null }
    | { type: "SET_CURRENT_FONT"; payload: { filename: string; name: string } }
    | { type: "SET_TEXT_ALIGNMENT"; payload: TextAlignment }
    | { type: "SET_EDITING_TEXT"; payload: boolean };

export interface AppState {
    currentFontsData: FontData[];
    currentlySelectedEntryId: string | null;
    lastCapturedRegion: Region | null;
    pageEntriesCache: Record<string, Entry[]>;
    currentTrackingColorIdx: number;
    svgOverlayElement: SVGSVGElement | null;
    zoomLevel: number;
    currentImageFile: string;
    fitZoomLevel: number;
    isGrayScaleEnabled: boolean;
    isHorizontalFitEnabled: boolean;
    isAutoOCREnabled: boolean;
    isAutoTranslateEnabled: boolean;
    isLiveInpaintedEnabled: boolean;
    isContextEnabled: boolean;
    isCharactersEnabled: boolean;
    isDrawing: boolean;
    lassoPoints: Point[];
    isPanning: boolean;
    startX: number | undefined;
    startY: number | undefined;
    scrollLeft: number | undefined;
    scrollTop: number | undefined;
    isCaptureMode: boolean;
    isEditingText: boolean;
}

export const initialState: AppState = {
    currentFontsData: [],
    currentlySelectedEntryId: null,
    lastCapturedRegion: null,
    pageEntriesCache: {},
    currentTrackingColorIdx: 0,
    svgOverlayElement: null,
    zoomLevel: 1.0,
    currentImageFile: "",
    fitZoomLevel: 1.0,
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
    isCaptureMode: false,
    isEditingText: false,
};

export function appReducer(state: AppState, action: Action): AppState {
    switch (action.type) {
        case "SET_CURRENT_IMAGE":
            return { ...state, currentImageFile: action.payload };
        case "SET_ZOOM_LEVEL":
            return { ...state, zoomLevel: action.payload };
        case "SET_FIT_ZOOM_LEVEL":
            return { ...state, fitZoomLevel: action.payload };
        case "SET_CAPTURE_MODE":
            return { ...state, isCaptureMode: action.payload };
        case "SET_DRAWING":
            return { ...state, isDrawing: action.payload };
        case "SET_LASSO_POINTS":
            return { ...state, lassoPoints: action.payload };
        case "SET_PANNING":
            return { ...state, isPanning: action.payload };
        case "SET_START_POSITION":
            return { ...state, startX: action.payload.x, startY: action.payload.y };
        case "SET_SCROLL_POSITION":
            return { ...state, scrollLeft: action.payload.left, scrollTop: action.payload.top };
        case "SET_GRAYSCALE":
            return { ...state, isGrayScaleEnabled: action.payload };
        case "SET_HORIZONTAL_FIT":
            return { ...state, isHorizontalFitEnabled: action.payload };
        case "SET_AUTO_OCR":
            return { ...state, isAutoOCREnabled: action.payload };
        case "SET_AUTO_TRANSLATE":
            return { ...state, isAutoTranslateEnabled: action.payload };
        case "SET_LIVE_INPAINT":
            return { ...state, isLiveInpaintedEnabled: action.payload };
        case "SET_CONTEXT":
            return { ...state, isContextEnabled: action.payload };
        case "SET_CHARACTERS_ENABLED":
            return { ...state, isCharactersEnabled: action.payload };
        case "SET_FONTS_DATA":
            return { ...state, currentFontsData: action.payload };
        case "SET_SELECTED_ENTRY_ID":
            return { ...state, currentlySelectedEntryId: action.payload };
        case "SET_LAST_CAPTURED_REGION":
            return { ...state, lastCapturedRegion: action.payload };
        case "SET_PAGE_ENTRIES":
            return {
                ...state,
                pageEntriesCache: { ...state.pageEntriesCache, [action.payload.page]: action.payload.entries },
            };
        case "ADD_ENTRIES": {
            const existing = state.pageEntriesCache[action.payload.page] ?? [];
            return {
                ...state,
                pageEntriesCache: {
                    ...state.pageEntriesCache,
                    [action.payload.page]: [...existing, ...action.payload.entries],
                },
            };
        }
        case "UPDATE_ENTRY": {
            const pageEntries = state.pageEntriesCache[action.payload.page] ?? [];
            return {
                ...state,
                pageEntriesCache: {
                    ...state.pageEntriesCache,
                    [action.payload.page]: pageEntries.map((e) =>
                        e.id === action.payload.entryId ? { ...e, ...action.payload.updates } : e,
                    ),
                },
            };
        }
        case "DELETE_ENTRY": {
            const pageEntries = state.pageEntriesCache[action.payload.page] ?? [];
            return {
                ...state,
                pageEntriesCache: {
                    ...state.pageEntriesCache,
                    [action.payload.page]: pageEntries.filter((e) => e.id !== action.payload.entryId),
                },
            };
        }
        case "CLEAR_PAGE_ENTRIES":
            return {
                ...state,
                pageEntriesCache: { ...state.pageEntriesCache, [action.payload]: [] },
                currentlySelectedEntryId: null,
            };
        case "SET_TRACKING_COLOR_IDX":
            return { ...state, currentTrackingColorIdx: action.payload };
        case "SET_SVG_OVERLAY":
            return { ...state, svgOverlayElement: action.payload };
        case "SET_CURRENT_FONT":
            return { ...state };
        case "SET_TEXT_ALIGNMENT":
            return { ...state };
        case "SET_EDITING_TEXT":
            return { ...state, isEditingText: action.payload };
        default:
            return state;
    }
}

export interface AppContextValue {
    state: AppState;
    dispatch: (action: Action) => void;
}

export const AppContext = createContext<AppContextValue>({ state: initialState, dispatch: () => {} });

export function AppProvider({ children }: { children: preact.ComponentChildren }) {
    const [state, dispatch] = useReducer(appReducer, initialState);
    return <AppContext.Provider value={{ state, dispatch }}>{children}</AppContext.Provider>;
}
