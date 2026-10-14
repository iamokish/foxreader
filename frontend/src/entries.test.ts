import { beforeEach, describe, expect, it, vi } from "vitest";
import {
    addNewEntries,
    confirmCurrentTranslation,
    entryNumber,
    executeInpaintAction,
    moveEntry,
    renderCurrentPageEntries,
    resortPageEntries,
} from "./entries";
import { MAX_FONT_SCALE, MAX_TEXT_ANGLE, MIN_LINE_SPACING, PLAIN_GEOMETRY, UNMEASURED, state } from "./state";
import type { Entry } from "./state";
import type { Region } from "./types";
import * as api from "./api";
import { runInpaint } from "./entries/inpaint";

vi.mock("./api", () => ({ ocrCrop: vi.fn(), ocrFreeform: vi.fn() }));
// The render itself is another module's job; these cases only care about the
// payload it is handed, which is where the geometry has to appear.
vi.mock("./entries/inpaint", () => ({
    runInpaint: vi.fn(async () => {}),
    closeInpaintModal: vi.fn(),
    saveInpaintPreview: vi.fn(),
}));

const REGION: Region = { type: "rectangle", coords: { x: 10, y: 10, w: 60, h: 40 } };

function rectRegion(x: number, y: number, w = 40, h = 40): Region {
    return { type: "rectangle", coords: { x, y, w, h } };
}

/** The editor as Confirm Entry finds it: an OCR box and a translation panel. */
function editor(panelClass: string, panelText: string): void {
    document.body.innerHTML = `
        <textarea id="textArea">  ocr text  </textarea>
        <div id="translatedText" class="${panelClass}">${panelText}</div>
    `;
}

function saved() {
    return state.pageEntriesCache["p.png"] ?? [];
}

describe("confirmCurrentTranslation", () => {
    beforeEach(() => {
        state.pageEntriesCache = {};
        state.currentImageFile = "p.png";
        state.lastCapturedRegion = REGION;
        state.currentTrackingColorIdx = 0;
        document.body.innerHTML = "";
    });

    it("does nothing without a captured region", () => {
        state.lastCapturedRegion = null;
        editor("panel translated", "Hello");
        confirmCurrentTranslation();
        expect(saved()).toHaveLength(0);
    });

    it("ignores the click while a translation is in flight", () => {
        editor("panel translating", "Translating...");
        confirmCurrentTranslation();
        // Neither a half-saved entry nor the placeholder as its text.
        expect(saved()).toHaveLength(0);
        expect(document.body.textContent).toContain("Still translating");
    });

    it("saves the region with an empty translation when the panel is an error", () => {
        editor("panel error", "Select a translator");
        confirmCurrentTranslation();
        expect(saved()).toHaveLength(1);
        expect(saved()[0].text).toBe("");
        // The OCR text is still worth keeping -- the region can be translated after.
        expect(saved()[0].ocr_text).toBe("ocr text");
    });

    it("takes the text out of a translated panel", () => {
        editor("panel translated", "  Hello there\n");
        confirmCurrentTranslation();
        expect(saved()[0].text).toBe("Hello there");
    });

    it("takes the text out of a panel nothing has happened in yet", () => {
        editor("panel", "typed by hand");
        confirmCurrentTranslation();
        expect(saved()[0].text).toBe("typed by hand");
    });

    it("replaces the text when the same region is confirmed again", () => {
        editor("panel translated", "first");
        confirmCurrentTranslation();
        editor("panel translated", "second");
        confirmCurrentTranslation();
        expect(saved()).toHaveLength(1);
        expect(saved()[0].text).toBe("second");
    });

    it("leaves a region's text alone when it is added without confirming", () => {
        editor("panel translated", "not mine");
        addNewEntries([REGION]);
        expect(saved()).toHaveLength(1);
        expect(saved()[0].text).toBe("");
        expect(saved()[0].ocr_text).toBe("");
    });
});

describe("reading order", () => {
    beforeEach(() => {
        state.pageEntriesCache = {};
        state.currentImageFile = "p.png";
        state.lastCapturedRegion = null;
        state.currentTrackingColorIdx = 0;
        localStorage.clear();
        document.body.innerHTML = "";
    });

    it("orders a captured batch geometrically, right to left by default", () => {
        addNewEntries([rectRegion(10, 10), rectRegion(200, 10), rectRegion(110, 10)]);
        const xs = saved().map((e) => (e.region.coords as { x: number }).x);
        expect(xs).toEqual([200, 110, 10]);
        expect(saved().map(entryNumber)).toEqual([1, 2, 3]);
    });

    it("inserts split pieces at the source entry's slot", () => {
        addNewEntries([rectRegion(10, 10), rectRegion(300, 10), rectRegion(600, 10)]);
        // RTL append order above is already right-to-left: 600, 300, 10.
        const source = saved()[1];
        expect((source.region.coords as { x: number }).x).toBe(300);
        addNewEntries([rectRegion(280, 10), rectRegion(340, 10)], false, source);
        // The split flow then hides the source; the pieces hold its slot.
        source.visible = false;
        const xs = saved().map((e) => (e.region.coords as { x: number }).x);
        expect(xs).toEqual([600, 340, 280, 300, 10]);
    });

    it("moves entries up and down, refusing the ends", () => {
        addNewEntries([rectRegion(10, 10), rectRegion(200, 10)]);
        const [first, second] = saved();
        expect(moveEntry(first, -1)).toBe(false);
        expect(moveEntry(second, 1)).toBe(false);
        expect(moveEntry(second, -1)).toBe(true);
        expect(saved().map((e) => e.id)).toEqual([second.id, first.id]);
        expect(saved().map(entryNumber)).toEqual([1, 2]);
        expect(moveEntry(first, 1)).toBe(false);
    });

    it("ignores moves for entries that are not on the page", () => {
        addNewEntries([rectRegion(10, 10)]);
        const ghost = { id: "ghost" } as Entry;
        expect(moveEntry(ghost, 1)).toBe(false);
        expect(entryNumber(ghost)).toBe(0);
    });

    it("resorts the page geometrically", () => {
        addNewEntries([rectRegion(10, 10), rectRegion(200, 10)]);
        const entries = saved();
        entries.reverse();
        resortPageEntries();
        const xs = saved().map((e) => (e.region.coords as { x: number }).x);
        expect(xs).toEqual([200, 10]);
    });

    it("paints the card badge with the number in the entry colour", () => {
        document.body.innerHTML = '<div id="savedEntriesList"></div>';
        addNewEntries([rectRegion(200, 10), rectRegion(10, 10)]);
        renderCurrentPageEntries();
        const badges = Array.from(document.querySelectorAll(".entry-layer-badge"));
        expect(badges.map((b) => b.textContent)).toEqual(["1", "2"]);
        expect(badges[0].getAttribute("title")).toBe("Reading order 1");
        const first = saved()[0];
        // jsdom reports colours back as rgb().
        expect((badges[0] as HTMLElement).style.background).toBe("rgb(0, 255, 102)");
        expect(first.color).toBe("#00FF66");
    });

    it("moves entries from their card buttons and disables the ends", () => {
        document.body.innerHTML = '<div id="savedEntriesList"></div>';
        addNewEntries([rectRegion(10, 10), rectRegion(200, 10)]);
        renderCurrentPageEntries();
        // RTL: 200 is first, 10 is second.
        const secondCard = document.getElementById(`card_${saved()[1].id}`)!;
        const upBtn = secondCard.querySelector<HTMLButtonElement>(".move-entry-up-btn")!;
        const downBtn = secondCard.querySelector<HTMLButtonElement>(".move-entry-down-btn")!;
        expect(upBtn.disabled).toBe(false);
        expect(downBtn.disabled).toBe(true);
        upBtn.click();
        expect(saved().map((e) => (e.region.coords as { x: number }).x)).toEqual([10, 200]);
        const badges = Array.from(document.querySelectorAll(".entry-layer-badge"));
        expect(badges.map((b) => b.textContent)).toEqual(["1", "2"]);
    });
});

describe("entry preview face", () => {
    function previewOf(ocrText: string, text: string): HTMLElement {
        state.pageEntriesCache = {};
        state.currentImageFile = "p.png";
        state.currentTrackingColorIdx = 0;
        document.body.innerHTML = '<div id="savedEntriesList"></div>';
        addNewEntries([rectRegion(10, 10)]);
        const entry = saved()[0];
        entry.ocr_text = ocrText;
        entry.text = text;
        entry.fontname = "'TestFace', sans-serif";
        renderCurrentPageEntries();
        return document.querySelector<HTMLElement>(".entry-text-preview")!;
    }

    // The face dresses the entry's own text; the OCR fallback below it always
    // renders in the default face.
    it.each([
        ["", "", "", ""],
        ["og", "", "og", ""],
        ["og", "hi", "hi", "'TestFace'"],
        ["", "hi", "hi", "'TestFace'"],
    ])("og %p + text %p shows %p dressed in %p", (og, text, shown, face) => {
        const preview = previewOf(og, text);
        expect(preview.textContent).toBe(shown);
        if (face === "") expect(preview.style.fontFamily).toBe("");
        else expect(preview.style.fontFamily).toContain("TestFace");
    });
});

describe("Auto OCR on entry click", () => {
    const ocrCrop = vi.mocked(api.ocrCrop);

    beforeEach(() => {
        document.body.innerHTML = `
            <div id="savedEntriesList"></div>
            <textarea id="textArea"></textarea>
        `;
        // jsdom implements no layout and so no scrolling; the card expand
        // path calls it on every select.
        if (typeof Element.prototype.scrollIntoView !== "function") {
            Element.prototype.scrollIntoView = vi.fn();
        }
        state.pageEntriesCache = {};
        state.currentImageFile = "p.png";
        state.currentTrackingColorIdx = 0;
        state.currentlySelectedEntryId = null;
        state.isAutoOCREnabled = false;
        state.isAutoTranslateEnabled = false;
        vi.clearAllMocks();
        ocrCrop.mockResolvedValue({ text: "read text" });
    });

    function clickFirstHead(): void {
        renderCurrentPageEntries();
        const head = document.querySelector<HTMLElement>("#savedEntriesList .entry-head");
        expect(head).not.toBeNull();
        head!.click();
    }

    function textArea(): HTMLTextAreaElement {
        return document.getElementById("textArea") as HTMLTextAreaElement;
    }

    it("reads the region when Auto OCR is on and the entry has no text", async () => {
        state.isAutoOCREnabled = true;
        addNewEntries([REGION]);
        clickFirstHead();
        await vi.waitFor(() => expect(ocrCrop).toHaveBeenCalledTimes(1));
        expect(ocrCrop).toHaveBeenCalledWith(expect.objectContaining({ filename: "p.png" }));
        await vi.waitFor(() => expect(textArea().value).toBe("read text"));
    });

    it("only selects while Auto OCR is off", async () => {
        addNewEntries([REGION]);
        clickFirstHead();
        await new Promise((resolve) => setTimeout(resolve, 20));
        expect(ocrCrop).not.toHaveBeenCalled();
        expect(state.currentlySelectedEntryId).toBe(saved()[0].id);
        expect(textArea().value).toBe("");
    });

    it("only selects an entry that already holds text", async () => {
        state.isAutoOCREnabled = true;
        addNewEntries([REGION]);
        saved()[0].ocr_text = "already read";
        clickFirstHead();
        await new Promise((resolve) => setTimeout(resolve, 20));
        expect(ocrCrop).not.toHaveBeenCalled();
        expect(textArea().value).toBe("already read");
    });
});

describe("typesetting geometry", () => {
    beforeEach(() => {
        state.pageEntriesCache = {};
        state.currentImageFile = "p.png";
        state.lastCapturedRegion = null;
        state.currentTrackingColorIdx = 0;
        localStorage.clear();
        document.body.innerHTML = "";
    });

    const ADJUSTED = {
        word_spacing: 1.4,
        line_spacing: 0.9,
        font_scale_x: 1.3,
        font_scale_y: 0.8,
        shift_x: -6,
        shift_y: 3,
        angle_x: 10,
        angle_y: -10,
        angle_z: 25,
    };

    const geometryOf = (entry: Entry) => ({
        word_spacing: entry.word_spacing,
        line_spacing: entry.line_spacing,
        font_scale_x: entry.font_scale_x,
        font_scale_y: entry.font_scale_y,
        shift_x: entry.shift_x,
        shift_y: entry.shift_y,
        angle_x: entry.angle_x,
        angle_y: entry.angle_y,
        angle_z: entry.angle_z,
    });

    it("gives a fresh capture neutral geometry", () => {
        addNewEntries([rectRegion(10, 10)]);
        expect(geometryOf(saved()[0])).toEqual({ ...PLAIN_GEOMETRY });
    });

    it("hands a split piece its source's geometry", () => {
        // The halves of one bubble are still set the same way, so a cut that
        // reset them would undo the adjustment on every split.
        addNewEntries([rectRegion(10, 10)]);
        const source = saved()[0];
        Object.assign(source, ADJUSTED);

        addNewEntries([rectRegion(8, 10), rectRegion(30, 10)], false, source);
        const pieces = saved().filter((entry) => entry.id !== source.id);
        expect(pieces).toHaveLength(2);
        for (const piece of pieces) expect(geometryOf(piece)).toEqual(ADJUSTED);
    });

    it("sends the geometry with every item, under the backend's field names", async () => {
        addNewEntries([rectRegion(10, 10), rectRegion(200, 10)]);
        Object.assign(saved()[0], ADJUSTED);

        await executeInpaintAction("preview");
        const deps = vi.mocked(runInpaint).mock.calls[0][1];
        const payload = deps.buildPayload();

        expect(payload?.data).toHaveLength(2);
        expect(geometryOf(payload!.data[0] as unknown as Entry)).toEqual(ADJUSTED);
        // An entry nobody adjusted still carries the nine fields, neutral: the
        // backend reads them off every item rather than guessing at a missing one.
        expect(geometryOf(payload!.data[1] as unknown as Entry)).toEqual({ ...PLAIN_GEOMETRY });
    });

    it("clamps what it sends, so a hand-edited project cannot reach the renderer raw", () => {
        addNewEntries([rectRegion(10, 10)]);
        Object.assign(saved()[0], { font_scale_x: 9, angle_z: 5000, line_spacing: -2 });

        void executeInpaintAction("preview");
        const deps = vi.mocked(runInpaint).mock.calls[0][1];
        const item = deps.buildPayload()!.data[0];
        expect(item.font_scale_x).toBe(MAX_FONT_SCALE);
        expect(item.angle_z).toBe(MAX_TEXT_ANGLE);
        expect(item.line_spacing).toBe(MIN_LINE_SPACING);
    });
});

describe("the colours the OCR step measured", () => {
    const MEASURED = { font_color: "#f2f2f0", stroke_color: "#101010", bg_color: "#1b1b1e" };

    const measurementsOf = (entry: Entry) => ({
        detected_font_color: entry.detected_font_color,
        detected_stroke_color: entry.detected_stroke_color,
        detected_bg: entry.detected_bg,
    });

    beforeEach(() => {
        state.pageEntriesCache = {};
        state.currentImageFile = "p.png";
        state.lastCapturedRegion = REGION;
        state.lastCapturedColors = null;
        state.currentTrackingColorIdx = 0;
        document.body.innerHTML = "";
    });

    it("starts a bubble sweep unmeasured", () => {
        // The detector finds regions; nobody has read them, so there is nothing
        // to know about their colours yet.
        addNewEntries([rectRegion(10, 10)]);
        expect(measurementsOf(saved()[0])).toEqual({ ...UNMEASURED });
    });

    it("attaches the capture's colours when it is confirmed", () => {
        state.lastCapturedColors = MEASURED;
        editor("panel translated", "Hello");
        confirmCurrentTranslation();

        expect(measurementsOf(saved()[0])).toEqual({
            detected_font_color: "#f2f2f0",
            detected_stroke_color: "#101010",
            detected_bg: "#1b1b1e",
        });
    });

    it("does not attach them to regions nobody read", () => {
        // A bubble sweep running while an unrelated capture is still in the
        // editor must not have the sweep's regions claim that capture's colours.
        state.lastCapturedColors = MEASURED;
        addNewEntries([rectRegion(120, 10)]);
        expect(measurementsOf(saved()[0])).toEqual({ ...UNMEASURED });
    });

    it("refreshes them when the same region is confirmed again", () => {
        state.lastCapturedColors = MEASURED;
        editor("panel translated", "Hello");
        confirmCurrentTranslation();

        state.lastCapturedColors = { font_color: "#000000", stroke_color: null, bg_color: "#ffffff" };
        confirmCurrentTranslation();

        expect(measurementsOf(saved()[0])).toEqual({
            detected_font_color: "#000000",
            detected_stroke_color: null,
            detected_bg: "#ffffff",
        });
    });

    it("hands a split piece its source's colours", () => {
        // The pieces are parts of the bubble that was read, so the reading holds.
        state.lastCapturedColors = MEASURED;
        editor("panel translated", "Hello");
        confirmCurrentTranslation();
        const source = saved()[0];

        addNewEntries([rectRegion(8, 10), rectRegion(30, 10)], false, source);
        const pieces = saved().filter((entry) => entry.id !== source.id);
        expect(pieces).toHaveLength(2);
        for (const piece of pieces) expect(measurementsOf(piece)).toEqual(measurementsOf(source));
    });

    it("normalises and rejects what it is handed", () => {
        // They come off the wire, and a project may have been written by a build
        // that stored something else. Short hex is expanded so comparing against
        // a palette swatch stays a string compare; anything unreadable is null,
        // which both renderers read as "work it out the old way".
        state.lastCapturedColors = {
            font_color: "#F0C",
            stroke_color: "not a colour",
            bg_color: null,
        } as unknown as typeof MEASURED;
        editor("panel translated", "Hello");
        confirmCurrentTranslation();

        expect(measurementsOf(saved()[0])).toEqual({
            detected_font_color: "#ff00cc",
            detected_stroke_color: null,
            detected_bg: null,
        });
    });

    it("sends them with every item, under the backend's field names", async () => {
        state.lastCapturedColors = MEASURED;
        editor("panel translated", "Hello");
        confirmCurrentTranslation();
        addNewEntries([rectRegion(200, 10)]);

        await executeInpaintAction("preview");
        const payload = vi.mocked(runInpaint).mock.calls[0][1].buildPayload();

        expect(payload?.data).toHaveLength(2);
        expect(measurementsOf(payload!.data[0] as unknown as Entry)).toEqual({
            detected_font_color: "#f2f2f0",
            detected_stroke_color: "#101010",
            detected_bg: "#1b1b1e",
        });
        // An unmeasured entry still carries the three fields, null: the backend
        // reads them off every item rather than guessing at a missing one.
        expect(measurementsOf(payload!.data[1] as unknown as Entry)).toEqual({ ...UNMEASURED });
    });
});
