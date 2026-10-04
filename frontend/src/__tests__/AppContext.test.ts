import { describe, it, expect } from "vitest";
import { appReducer, initialState } from "../core/AppContext";
import type { AppState, Action } from "../core/AppContext";

describe("appReducer", () => {
    describe("SET_CURRENT_IMAGE", () => {
        it("updates currentImageFile", () => {
            const action: Action = { type: "SET_CURRENT_IMAGE", payload: "/img/test.jpg" };
            const result = appReducer(initialState, action);
            expect(result.currentImageFile).toBe("/img/test.jpg");
        });
    });

    describe("SET_ZOOM_LEVEL", () => {
        it("updates zoomLevel", () => {
            const action: Action = { type: "SET_ZOOM_LEVEL", payload: 2.5 };
            const result = appReducer(initialState, action);
            expect(result.zoomLevel).toBe(2.5);
        });
    });

    describe("SET_FIT_ZOOM_LEVEL", () => {
        it("updates fitZoomLevel", () => {
            const action: Action = { type: "SET_FIT_ZOOM_LEVEL", payload: 1.5 };
            const result = appReducer(initialState, action);
            expect(result.fitZoomLevel).toBe(1.5);
        });
    });

    describe("SET_CAPTURE_MODE", () => {
        it("updates isCaptureMode", () => {
            const action: Action = { type: "SET_CAPTURE_MODE", payload: true };
            const result = appReducer(initialState, action);
            expect(result.isCaptureMode).toBe(true);
        });
    });

    describe("SET_DRAWING", () => {
        it("updates isDrawing", () => {
            const action: Action = { type: "SET_DRAWING", payload: true };
            const result = appReducer(initialState, action);
            expect(result.isDrawing).toBe(true);
        });
    });

    describe("SET_LASSO_POINTS", () => {
        it("updates lassoPoints", () => {
            const points = [
                { x: 10, y: 20 },
                { x: 30, y: 40 },
            ];
            const action: Action = { type: "SET_LASSO_POINTS", payload: points };
            const result = appReducer(initialState, action);
            expect(result.lassoPoints).toEqual(points);
        });
    });

    describe("SET_PANNING", () => {
        it("updates isPanning", () => {
            const action: Action = { type: "SET_PANNING", payload: true };
            const result = appReducer(initialState, action);
            expect(result.isPanning).toBe(true);
        });
    });

    describe("SET_START_POSITION", () => {
        it("updates startX and startY", () => {
            const action: Action = { type: "SET_START_POSITION", payload: { x: 100, y: 200 } };
            const result = appReducer(initialState, action);
            expect(result.startX).toBe(100);
            expect(result.startY).toBe(200);
        });
    });

    describe("SET_SCROLL_POSITION", () => {
        it("updates scrollLeft and scrollTop", () => {
            const action: Action = { type: "SET_SCROLL_POSITION", payload: { left: 50, top: 75 } };
            const result = appReducer(initialState, action);
            expect(result.scrollLeft).toBe(50);
            expect(result.scrollTop).toBe(75);
        });
    });

    describe("SET_GRAYSCALE", () => {
        it("updates isGrayScaleEnabled", () => {
            const action: Action = { type: "SET_GRAYSCALE", payload: true };
            const result = appReducer(initialState, action);
            expect(result.isGrayScaleEnabled).toBe(true);
        });
    });

    describe("SET_HORIZONTAL_FIT", () => {
        it("updates isHorizontalFitEnabled", () => {
            const action: Action = { type: "SET_HORIZONTAL_FIT", payload: true };
            const result = appReducer(initialState, action);
            expect(result.isHorizontalFitEnabled).toBe(true);
        });
    });

    describe("SET_AUTO_OCR", () => {
        it("updates isAutoOCREnabled", () => {
            const action: Action = { type: "SET_AUTO_OCR", payload: true };
            const result = appReducer(initialState, action);
            expect(result.isAutoOCREnabled).toBe(true);
        });
    });

    describe("SET_AUTO_TRANSLATE", () => {
        it("updates isAutoTranslateEnabled", () => {
            const action: Action = { type: "SET_AUTO_TRANSLATE", payload: false };
            const result = appReducer(initialState, action);
            expect(result.isAutoTranslateEnabled).toBe(false);
        });
    });

    describe("SET_LIVE_INPAINT", () => {
        it("updates isLiveInpaintedEnabled", () => {
            const action: Action = { type: "SET_LIVE_INPAINT", payload: false };
            const result = appReducer(initialState, action);
            expect(result.isLiveInpaintedEnabled).toBe(false);
        });
    });

    describe("SET_CONTEXT", () => {
        it("updates isContextEnabled", () => {
            const action: Action = { type: "SET_CONTEXT", payload: false };
            const result = appReducer(initialState, action);
            expect(result.isContextEnabled).toBe(false);
        });
    });

    describe("SET_FONTS_DATA", () => {
        it("updates currentFontsData", () => {
            const fonts = [
                {
                    font_filename: "test.woff2",
                    font_name: "Test Font",
                    font_data_uri: "data:font/woff2",
                    font_format: "woff2",
                },
            ];
            const action: Action = { type: "SET_FONTS_DATA", payload: fonts };
            const result = appReducer(initialState, action);
            expect(result.currentFontsData).toEqual(fonts);
        });
    });

    describe("SET_SELECTED_ENTRY_ID", () => {
        it("updates currentlySelectedEntryId", () => {
            const action: Action = { type: "SET_SELECTED_ENTRY_ID", payload: "entry-123" };
            const result = appReducer(initialState, action);
            expect(result.currentlySelectedEntryId).toBe("entry-123");
        });

        it("can set to null", () => {
            const stateWithEntry = { ...initialState, currentlySelectedEntryId: "entry-123" };
            const action: Action = { type: "SET_SELECTED_ENTRY_ID", payload: null };
            const result = appReducer(stateWithEntry, action);
            expect(result.currentlySelectedEntryId).toBeNull();
        });
    });

    describe("SET_LAST_CAPTURED_REGION", () => {
        it("updates lastCapturedRegion with rectangle", () => {
            const region = { type: "rectangle" as const, coords: { x: 10, y: 20, w: 100, h: 50 } };
            const action: Action = { type: "SET_LAST_CAPTURED_REGION", payload: region };
            const result = appReducer(initialState, action);
            expect(result.lastCapturedRegion).toEqual(region);
        });

        it("can set to null", () => {
            const stateWithRegion = {
                ...initialState,
                lastCapturedRegion: { type: "rectangle" as const, coords: { x: 10, y: 20, w: 100, h: 50 } },
            };
            const action: Action = { type: "SET_LAST_CAPTURED_REGION", payload: null };
            const result = appReducer(stateWithRegion, action);
            expect(result.lastCapturedRegion).toBeNull();
        });
    });

    describe("SET_PAGE_ENTRIES", () => {
        it("sets entries for a page", () => {
            const entries = [
                {
                    id: "1",
                    ocr_text: "OCR",
                    text: "Text",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
            ];
            const action: Action = { type: "SET_PAGE_ENTRIES", payload: { page: "page1.jpg", entries } };
            const result = appReducer(initialState, action);
            expect(result.pageEntriesCache["page1.jpg"]).toEqual(entries);
        });

        it("overwrites existing entries for the same page", () => {
            const stateWithEntries = {
                ...initialState,
                pageEntriesCache: {
                    "page1.jpg": [
                        {
                            id: "old",
                            ocr_text: "Old",
                            text: "Old",
                            fontfile: "",
                            fontname: "",
                            region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                            color: "#000",
                            text_align: "center" as const,
                            visible: true,
                        },
                    ],
                },
            };
            const newEntries = [
                {
                    id: "new",
                    ocr_text: "New",
                    text: "New",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
            ];
            const action: Action = { type: "SET_PAGE_ENTRIES", payload: { page: "page1.jpg", entries: newEntries } };
            const result = appReducer(stateWithEntries, action);
            expect(result.pageEntriesCache["page1.jpg"]).toEqual(newEntries);
        });
    });

    describe("ADD_ENTRIES", () => {
        it("appends entries to a page", () => {
            const existingEntries = [
                {
                    id: "1",
                    ocr_text: "OCR1",
                    text: "Text1",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
            ];
            const stateWithEntries = {
                ...initialState,
                pageEntriesCache: { "page1.jpg": existingEntries },
            };
            const newEntries = [
                {
                    id: "2",
                    ocr_text: "OCR2",
                    text: "Text2",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
            ];
            const action: Action = { type: "ADD_ENTRIES", payload: { page: "page1.jpg", entries: newEntries } };
            const result = appReducer(stateWithEntries, action);
            expect(result.pageEntriesCache["page1.jpg"]).toHaveLength(2);
            expect(result.pageEntriesCache["page1.jpg"][1].id).toBe("2");
        });

        it("creates new page entry if page does not exist", () => {
            const newEntries = [
                {
                    id: "1",
                    ocr_text: "OCR",
                    text: "Text",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
            ];
            const action: Action = { type: "ADD_ENTRIES", payload: { page: "newpage.jpg", entries: newEntries } };
            const result = appReducer(initialState, action);
            expect(result.pageEntriesCache["newpage.jpg"]).toHaveLength(1);
        });
    });

    describe("UPDATE_ENTRY", () => {
        it("updates a specific entry", () => {
            const entries = [
                {
                    id: "1",
                    ocr_text: "OCR",
                    text: "Old Text",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
            ];
            const stateWithEntries = {
                ...initialState,
                pageEntriesCache: { "page1.jpg": entries },
            };
            const action: Action = {
                type: "UPDATE_ENTRY",
                payload: { page: "page1.jpg", entryId: "1", updates: { text: "New Text" } },
            };
            const result = appReducer(stateWithEntries, action);
            expect(result.pageEntriesCache["page1.jpg"][0].text).toBe("New Text");
        });

        it("does not update other entries", () => {
            const entries = [
                {
                    id: "1",
                    ocr_text: "OCR1",
                    text: "Text1",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
                {
                    id: "2",
                    ocr_text: "OCR2",
                    text: "Text2",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
            ];
            const stateWithEntries = {
                ...initialState,
                pageEntriesCache: { "page1.jpg": entries },
            };
            const action: Action = {
                type: "UPDATE_ENTRY",
                payload: { page: "page1.jpg", entryId: "1", updates: { text: "Updated" } },
            };
            const result = appReducer(stateWithEntries, action);
            expect(result.pageEntriesCache["page1.jpg"][1].text).toBe("Text2");
        });
    });

    describe("DELETE_ENTRY", () => {
        it("removes a specific entry", () => {
            const entries = [
                {
                    id: "1",
                    ocr_text: "OCR1",
                    text: "Text1",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
                {
                    id: "2",
                    ocr_text: "OCR2",
                    text: "Text2",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
            ];
            const stateWithEntries = {
                ...initialState,
                pageEntriesCache: { "page1.jpg": entries },
            };
            const action: Action = { type: "DELETE_ENTRY", payload: { page: "page1.jpg", entryId: "1" } };
            const result = appReducer(stateWithEntries, action);
            expect(result.pageEntriesCache["page1.jpg"]).toHaveLength(1);
            expect(result.pageEntriesCache["page1.jpg"][0].id).toBe("2");
        });
    });

    describe("CLEAR_PAGE_ENTRIES", () => {
        it("clears entries for a page and resets selected entry", () => {
            const entries = [
                {
                    id: "1",
                    ocr_text: "OCR",
                    text: "Text",
                    fontfile: "",
                    fontname: "",
                    region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                    color: "#000",
                    text_align: "center" as const,
                    visible: true,
                },
            ];
            const stateWithEntries = {
                ...initialState,
                pageEntriesCache: { "page1.jpg": entries },
                currentlySelectedEntryId: "1",
            };
            const action: Action = { type: "CLEAR_PAGE_ENTRIES", payload: "page1.jpg" };
            const result = appReducer(stateWithEntries, action);
            expect(result.pageEntriesCache["page1.jpg"]).toEqual([]);
            expect(result.currentlySelectedEntryId).toBeNull();
        });
    });

    describe("SET_TRACKING_COLOR_IDX", () => {
        it("updates currentTrackingColorIdx", () => {
            const action: Action = { type: "SET_TRACKING_COLOR_IDX", payload: 5 };
            const result = appReducer(initialState, action);
            expect(result.currentTrackingColorIdx).toBe(5);
        });
    });

    describe("SET_SVG_OVERLAY", () => {
        it("updates svgOverlayElement", () => {
            const mockSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            const action: Action = { type: "SET_SVG_OVERLAY", payload: mockSvg };
            const result = appReducer(initialState, action);
            expect(result.svgOverlayElement).toBe(mockSvg);
        });

        it("can set to null", () => {
            const stateWithSvg = {
                ...initialState,
                svgOverlayElement: document.createElementNS("http://www.w3.org/2000/svg", "svg"),
            };
            const action: Action = { type: "SET_SVG_OVERLAY", payload: null };
            const result = appReducer(stateWithSvg, action);
            expect(result.svgOverlayElement).toBeNull();
        });
    });

    describe("SET_EDITING_TEXT", () => {
        it("updates isEditingText", () => {
            const action: Action = { type: "SET_EDITING_TEXT", payload: true };
            const result = appReducer(initialState, action);
            expect(result.isEditingText).toBe(true);
        });
    });

    describe("unknown action", () => {
        it("returns current state for unknown action", () => {
            const action = { type: "UNKNOWN_ACTION" } as unknown as Action;
            const result = appReducer(initialState, action);
            expect(result).toBe(initialState);
        });
    });

    describe("immutability", () => {
        it("does not mutate original state", () => {
            const action: Action = { type: "SET_ZOOM_LEVEL", payload: 2.0 };
            const result = appReducer(initialState, action);
            expect(result).not.toBe(initialState);
            expect(initialState.zoomLevel).toBe(1.0);
        });

        it("does not mutate pageEntriesCache", () => {
            const stateWithEntries = {
                ...initialState,
                pageEntriesCache: {
                    "page1.jpg": [
                        {
                            id: "1",
                            ocr_text: "OCR",
                            text: "Text",
                            fontfile: "",
                            fontname: "",
                            region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                            color: "#000",
                            text_align: "center" as const,
                            visible: true,
                        },
                    ],
                },
            };
            const action: Action = {
                type: "ADD_ENTRIES",
                payload: {
                    page: "page1.jpg",
                    entries: [
                        {
                            id: "2",
                            ocr_text: "OCR2",
                            text: "Text2",
                            fontfile: "",
                            fontname: "",
                            region: { type: "rectangle" as const, coords: { x: 0, y: 0, w: 10, h: 10 } },
                            color: "#000",
                            text_align: "center" as const,
                            visible: true,
                        },
                    ],
                },
            };
            const result = appReducer(stateWithEntries, action);
            expect(result.pageEntriesCache).not.toBe(stateWithEntries.pageEntriesCache);
            expect(stateWithEntries.pageEntriesCache["page1.jpg"]).toHaveLength(1);
        });
    });
});
