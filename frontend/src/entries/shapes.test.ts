/**
 * The overlay's two "tell me what this region is doing" affordances.
 *
 * `.is-empty` -- an entry with neither OCR text nor a translation paints a plate
 * over artwork the user still has to read, so the plate goes translucent until
 * there is text worth hiding it for.
 *
 * `.split-target` -- the split button lives in the sidebar while the cut is drawn
 * on the artwork, so the region about to be divided glows for the duration.
 */

/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Mock } from "vitest";
import { buildEntryShape, collectPlacedBadges, planNumberBadges } from "./shapes";
import { advanceFor, ascentFor, lineHeightFor, spacingFor, resetPerspectiveCache, resetTextMetrics, supportsPerspective, tiltStyle, fontMetricsFromFontData, type FontMetrics } from "./textFit";
import { runSplit } from "./split";
import { PLAIN_GEOMETRY, UNMEASURED, state } from "../state";
import type { Entry, FontData } from "../state";
import { defaultCleanOptions } from "../entryOptions";
import { splitCapture } from "../ocr";

vi.mock("../ocr", () => ({ splitCapture: vi.fn() }));
vi.mock("../api", () => ({ splitBubble: vi.fn() }));
vi.mock("../ui", () => ({
    showNotify: vi.fn(),
    showLoading: vi.fn(),
    hideLoading: vi.fn(),
    updateLoadingText: vi.fn(),
}));

const SVG_NS = "http://www.w3.org/2000/svg";
const stylesheet = readFileSync(resolve(process.cwd(), "static/style.css"), "utf8");

function makeEntry(over: Partial<Entry> = {}): Entry {
    return {
        id: "e1",
        ocr_text: "",
        text: "",
        fontfile: "a.ttf",
        fontname: "'A', sans-serif",
        region: {
            type: "polygon",
            coords: [
                { x: 0, y: 0 },
                { x: 40, y: 0 },
                { x: 40, y: 30 },
                { x: 0, y: 30 },
            ],
        },
        color: "#00FF66",
        text_align: "center",
        visible: true,
        layer: 1,
        font_size: null,
        font_color: null,
        stroke_width: null,
        stroke_color: null,
        bg_mode: "auto",
        bg_color: null,
        clean: defaultCleanOptions(),
        ...PLAIN_GEOMETRY,
        ...UNMEASURED,
        ...over,
    };
}

function makeOverlay(): SVGSVGElement {
    const svg = document.createElementNS(SVG_NS, "svg") as SVGSVGElement;
    svg.id = "regionSvgOverlay";
    svg.setAttribute("viewBox", "0 0 400 300");
    document.body.appendChild(svg);
    return svg;
}

const noopHandlers = { onSelect: vi.fn(), onEditInPlace: vi.fn(), onHover: vi.fn() };

beforeEach(() => {
    document.body.innerHTML = "";
    state.currentImageFile = "page.png";
    state.pageEntriesCache = {};
});

afterEach(() => {
    vi.clearAllMocks();
});

describe("an entry with no text yet", () => {
    it("is marked so its plate can go translucent", () => {
        const group = buildEntryShape(makeEntry(), makeOverlay(), noopHandlers);
        expect(group?.classList.contains("is-empty")).toBe(true);
        // The plate is still painted -- only its opacity changes, in CSS.
        expect(group?.querySelector(".entry-plate")).not.toBeNull();
    });

    it("treats whitespace as empty", () => {
        const group = buildEntryShape(makeEntry({ ocr_text: "  ", text: "\n" }), makeOverlay(), noopHandlers);
        expect(group?.classList.contains("is-empty")).toBe(true);
    });

    it("is not marked once either pass has produced text", () => {
        const withOcr = buildEntryShape(makeEntry({ ocr_text: "こんにちは" }), makeOverlay(), noopHandlers);
        expect(withOcr?.classList.contains("is-empty")).toBe(false);

        document.body.innerHTML = "";
        const withText = buildEntryShape(makeEntry({ id: "e2", text: "Hello" }), makeOverlay(), noopHandlers);
        expect(withText?.classList.contains("is-empty")).toBe(false);
    });

    it("fades the plate but never the outline", () => {
        expect(stylesheet).toMatch(/\.entry-shape\.is-empty\s+\.entry-plate\s*\{[^}]*opacity:\s*0\.\d+/s);
        expect(stylesheet).toMatch(/\.entry-shape\.is-empty\s+\.clean-plate\s*\{[^}]*opacity:\s*0\.\d+/s);
        expect(stylesheet).not.toMatch(/\.entry-shape\.is-empty\s+\.entry-outline/);
    });
});

describe("the region armed for a split", () => {
    it("glows while the cut is being drawn and stops when it is over", async () => {
        const entry = makeEntry();
        const svg = makeOverlay();
        buildEntryShape(entry, svg, noopHandlers);
        const shape = document.getElementById("shape_e1") as unknown as SVGGElement;
        const button = document.createElement("button");
        document.body.appendChild(button);

        let release!: (value: null) => void;
        (splitCapture as Mock).mockReturnValue(
            new Promise<null>((r) => {
                release = r;
            }),
        );

        const pending = runSplit(entry, button, { addEntries: vi.fn(), refresh: vi.fn() });
        expect(shape.classList.contains("split-target")).toBe(true);
        expect(button.classList.contains("split-active")).toBe(true);

        release(null);
        await pending;
        expect(shape.classList.contains("split-target")).toBe(false);
        expect(button.classList.contains("split-active")).toBe(false);
    });

    it("does not arm anything when the entry cannot be split", async () => {
        const entry = makeEntry({ text: "already translated" });
        buildEntryShape(entry, makeOverlay(), noopHandlers);
        const shape = document.getElementById("shape_e1") as unknown as SVGGElement;

        await runSplit(entry, null, { addEntries: vi.fn(), refresh: vi.fn() });
        expect(shape.classList.contains("split-target")).toBe(false);
        expect(splitCapture).not.toHaveBeenCalled();
    });

    it("styles the glow with the same accent the split button uses", () => {
        const rule = /\.entry-shape\.split-target\s+\.entry-outline\s*\{([^}]*)\}/s.exec(stylesheet);
        expect(rule).not.toBeNull();
        const body = rule?.[1] ?? "";
        expect(body).toMatch(/var\(--accent-selection\)/);
        expect(body).toMatch(/drop-shadow/);
        // `.svg-overlay-polygon.active-hover-region` sets these with !important, so
        // the glow has to as well or it never reaches the screen.
        expect(body).toMatch(/stroke-width:[^;]*!important/);
        expect(body).toMatch(/stroke-dasharray:[^;]*!important/);
    });
});

describe("the reading number badge", () => {
    function rectEntry(id: string, x: number, y: number, w: number, h: number, color = "#00FF66") {
        return makeEntry({
            id,
            color,
            region: { type: "rectangle", coords: { x, y, w, h } },
        });
    }

    it("stamps the number in the entry colour with a readable label", () => {
        const entry = rectEntry("n1", 10, 10, 100, 60);
        const group = buildEntryShape(entry, makeOverlay(), noopHandlers, {
            num: 7,
            x: 10,
            y: 10,
            w: 30,
            h: 22,
            fontSize: 13,
        });
        const tag = group?.querySelector("g.entry-number");
        expect(tag).not.toBeNull();
        expect(tag?.querySelector("text")?.textContent).toBe("7");
        expect(tag?.querySelector("rect")?.getAttribute("fill")).toBe("#00FF66");
        // Black on this green, per the backend's contrast weights.
        expect(tag?.querySelector("text")?.getAttribute("fill")).toBe("#000000");
        expect(tag?.getAttribute("pointer-events")).toBe("none");
    });

    it("falls back to a solo badge for lone callers", () => {
        const group = buildEntryShape(rectEntry("solo", 10, 10, 100, 60), makeOverlay(), noopHandlers);
        expect(group?.querySelector("g.entry-number text")?.textContent).toBe("1");
    });

    it("parks neighbours on different spots instead of stacking", () => {
        const a = rectEntry("a", 10, 10, 60, 60);
        const b = rectEntry("b", 12, 12, 60, 60);
        const badges = planNumberBadges([
            { entry: a, num: 1 },
            { entry: b, num: 2 },
        ]);
        const ra = badges.get("a")!;
        const rb = badges.get("b")!;
        expect(ra).toBeDefined();
        expect(rb).toBeDefined();
        const overlap = ra.x < rb.x + rb.w && ra.x + ra.w > rb.x && ra.y < rb.y + rb.h && ra.y + ra.h > rb.y;
        expect(overlap).toBe(false);
    });

    it("keeps badges outside their own region", () => {
        const a = rectEntry("a", 50, 50, 80, 60);
        const b = rectEntry("b", 200, 50, 80, 60);
        const badges = planNumberBadges([
            { entry: a, num: 1 },
            { entry: b, num: 2 },
        ]);
        for (const [id, badge] of badges) {
            const entry = id === "a" ? a : b;
            const r = entry.region.coords as { x: number; y: number; w: number; h: number };
            const inside =
                badge.x < r.x + r.w && badge.x + badge.w > r.x && badge.y < r.y + r.h && badge.y + badge.h > r.y;
            expect(inside).toBe(false);
        }
    });

    it("touches the outline even where bbox corners hang off the ink", () => {
        // A triangle: the bbox's top corners are far from any edge, so a
        // corner-parked badge would float. The apex must carry it instead.
        const triangle = makeEntry({
            id: "tri",
            region: {
                type: "polygon",
                coords: [
                    { x: 10, y: 70 },
                    { x: 60, y: 10 },
                    { x: 110, y: 70 },
                ],
            },
        });
        const badge = planNumberBadges([{ entry: triangle, num: 1 }]).get("tri")!;
        expect(badge).toBeDefined();
        const vertices = triangle.region.coords as { x: number; y: number }[];
        const touches = vertices.some(
            (v) =>
                (Math.abs(badge.y + badge.h - v.y) <= 1 && v.x >= badge.x - 1 && v.x <= badge.x + badge.w + 1) ||
                (Math.abs(badge.y - v.y) <= 1 && v.x >= badge.x - 1 && v.x <= badge.x + badge.w + 1) ||
                (Math.abs(badge.x + badge.w - v.x) <= 1 && v.y >= badge.y - 1 && v.y <= badge.y + badge.h + 1) ||
                (Math.abs(badge.x - v.x) <= 1 && v.y >= badge.y - 1 && v.y <= badge.y + badge.h + 1),
        );
        expect(touches).toBe(true);
    });

    it("still badges a region that fills a tiny image", () => {
        const img = document.createElement("img");
        img.id = "mainImage";
        Object.defineProperty(img, "naturalWidth", { value: 30 });
        Object.defineProperty(img, "naturalHeight", { value: 30 });
        document.body.appendChild(img);
        const entry = rectEntry("tiny", 0, 0, 30, 30);
        const badge = planNumberBadges([{ entry, num: 1 }]).get("tiny")!;
        expect(badge).toBeDefined();
        expect(badge.x).toBeGreaterThanOrEqual(0);
        expect(badge.y).toBeGreaterThanOrEqual(0);
        expect(badge.x + badge.w).toBeLessThanOrEqual(30);
        expect(badge.y + badge.h).toBeLessThanOrEqual(30);
    });

    it("clamps badges inside the image", () => {
        const img = document.createElement("img");
        img.id = "mainImage";
        Object.defineProperty(img, "naturalWidth", { value: 400 });
        Object.defineProperty(img, "naturalHeight", { value: 300 });
        document.body.appendChild(img);
        // Bottom-right region: the badge cannot hang off the page.
        const entry = rectEntry("edge", 350, 260, 40, 30);
        const badge = planNumberBadges([{ entry, num: 1 }]).get("edge")!;
        expect(badge.x + badge.w).toBeLessThanOrEqual(400);
        expect(badge.y + badge.h).toBeLessThanOrEqual(300);
        expect(badge.x).toBeGreaterThanOrEqual(0);
        expect(badge.y).toBeGreaterThanOrEqual(0);
    });

    it("collects on-screen badges so refreshes can dodge them", () => {
        const svg = makeOverlay();
        const entry = rectEntry("c1", 10, 10, 100, 60);
        buildEntryShape(entry, svg, noopHandlers, { num: 3, x: 10, y: 10, w: 30, h: 22, fontSize: 13 });
        const placed = collectPlacedBadges(svg);
        expect(placed).toHaveLength(1);
        expect(placed[0]).toMatchObject({ x: 10, y: 10, w: 30, h: 22 });
        // A newcomer on top of it is pushed to another corner.
        const challenger = rectEntry("c2", 10, 10, 100, 60);
        const moved = planNumberBadges([{ entry: challenger, num: 4 }], placed).get("c2")!;
        expect(moved.x === 10 && moved.y === 10).toBe(false);
    });
});

describe("the typesetting geometry on the preview", () => {
    let metrics: PropertyDescriptor | undefined;

    beforeEach(() => {
        state.isLiveInpaintedEnabled = true;
        // jsdom lays nothing out, so `getComputedTextLength` is missing and every
        // token measures zero -- which collapses the fit onto one line at the
        // maximum size and leaves word spacing no gap to widen. A width
        // proportional to the token is all these cases need, and it makes the
        // numbers below exact: one character is 40 units at the reference size.
        metrics = Object.getOwnPropertyDescriptor(SVGElement.prototype, "getComputedTextLength");
        Object.defineProperty(SVGElement.prototype, "getComputedTextLength", {
            configurable: true,
            value(this: SVGElement): number {
                return (this.textContent || "").length * 40;
            },
        });
        resetTextMetrics();
        // The fallback (`cos` squash) is what these cases assert unless a case
        // opts into CSS 3D explicitly: jsdom may claim `CSS.supports`, which
        // would otherwise take the perspective path and move the transform off
        // the `<text>` node onto an outer `<g>`.
        (globalThis as unknown as { CSS?: unknown }).CSS = undefined;
        resetPerspectiveCache();
    });

    afterEach(() => {
        if (metrics) Object.defineProperty(SVGElement.prototype, "getComputedTextLength", metrics);
        else delete (SVGElement.prototype as unknown as Record<string, unknown>).getComputedTextLength;
        state.currentSymbolFonts = [];
        // Those widths are this block's fiction; nothing after it may be served them.
        resetTextMetrics();
        (globalThis as unknown as { CSS?: unknown }).CSS = undefined;
        resetPerspectiveCache();
    });

    /** A region centred in the 400x300 overlay, with room for the fit to breathe. */
    function textEntry(over: Partial<Entry> = {}): Entry {
        return makeEntry({
            text: "Hello there",
            region: { type: "rectangle", coords: { x: 20, y: 20, w: 360, h: 260 } },
            ...over,
        });
    }

    function shapeOf(entry: Entry): { text: SVGTextElement; plate: SVGPolygonElement } {
        const group = buildEntryShape(entry, makeOverlay(), noopHandlers)!;
        expect(group).not.toBeNull();
        return {
            text: group.querySelector(".entry-text") as SVGTextElement,
            plate: group.querySelector(".entry-plate") as SVGPolygonElement,
        };
    }

    /** The shift, which `textTransform` writes as the outermost translate. */
    function shiftOf(entry: Entry): { dx: number; dy: number } {
        const transform = shapeOf(entry).text.getAttribute("transform") ?? "";
        const parsed = /^translate\((-?[\d.]+) (-?[\d.]+)\)/.exec(transform);
        expect(parsed).not.toBeNull();
        return { dx: Number(parsed![1]), dy: Number(parsed![2]) };
    }

    function withPage(w: number, h: number): void {
        const img = document.createElement("img");
        img.id = "mainImage";
        Object.defineProperty(img, "naturalWidth", { value: w });
        Object.defineProperty(img, "naturalHeight", { value: h });
        document.body.appendChild(img);
    }

    it("leaves an entry nobody adjusted exactly as it was before the options existed", () => {
        const { text, plate } = shapeOf(textEntry());
        expect(text).not.toBeNull();
        expect(text.hasAttribute("transform")).toBe(false);
        expect(text.hasAttribute("word-spacing")).toBe(false);
        expect(plate.hasAttribute("transform")).toBe(false);
    });

    it("transforms the text and leaves the background where it is", () => {
        const adjusted = shapeOf(textEntry({ shift_x: 7, shift_y: -4, angle_z: 30, font_scale_x: 1.5 }));
        expect(adjusted.text.getAttribute("transform")).toBeTruthy();
        // The whole point of the feature: the plate is a sibling node, outside
        // the transform's scope, and it is painted from the region either way.
        expect(adjusted.plate.hasAttribute("transform")).toBe(false);
        const plain = shapeOf(textEntry({ id: "e2" }));
        expect(adjusted.plate.getAttribute("points")).toBe(plain.plate.getAttribute("points"));
    });

    it("stretches, tilts, spins and shifts, outermost last-applied", () => {
        const { text } = shapeOf(
            textEntry({ font_scale_x: 1.5, font_scale_y: 0.5, angle_y: 60, angle_z: 30, shift_x: 6, shift_y: -4 }),
        );
        const transform = text.getAttribute("transform")!;
        expect(transform.indexOf("translate(6 -4)")).toBe(0);
        expect(transform).toMatch(/rotate\(30 -?[\d.]+ -?[\d.]+\)/);
        // Foreshortening only, and turning about Y narrows X alone: cos 60 = 0.5.
        expect(transform).toContain("scale(0.5 1)");
        expect(transform).toContain("scale(1.5 0.5)");
        // Read right to left, the block is stretched, then tilted, then spun.
        expect(transform.indexOf("rotate(")).toBeLessThan(transform.indexOf("scale(0.5 1)"));
        expect(transform.indexOf("scale(0.5 1)")).toBeLessThan(transform.indexOf("scale(1.5 0.5)"));
    });

    it("shifts by the asked-for pixels when no page has loaded", () => {
        // A test, or an image still decoding: the backend clamps for real anyway.
        expect(shiftOf(textEntry({ shift_x: 9999, shift_y: 0 }))).toEqual({ dx: 9999, dy: 0 });
    });

    it("bounds the shift by the page so the text cannot leave it", () => {
        withPage(400, 300);
        const right = shiftOf(textEntry({ id: "r", shift_x: 9999 }));
        const left = shiftOf(textEntry({ id: "l", shift_x: -9999 }));
        const down = shiftOf(textEntry({ id: "d", shift_y: 9999 }));

        expect(right.dx).toBeGreaterThan(0);
        expect(right.dx).toBeLessThan(9999);
        expect(down.dy).toBeGreaterThan(0);
        expect(down.dy).toBeLessThan(9999);
        // The region is centred on the page, so a block pushed as far as it goes
        // one way lands the same distance from centre the other way.
        expect(left.dx).toBe(-right.dx);
    });

    it("charges word spacing per gap, at the size the block rendered at", () => {
        const plain = shapeOf(textEntry({ font_size: 20 })).text;
        expect(plain.hasAttribute("word-spacing")).toBe(false);
        // Untouched entries keep one `<tspan>` per line; the face's own space
        // does the work, exactly as before.
        expect(plain.querySelectorAll("tspan")).toHaveLength(1);
        expect(plain.getAttribute("text-anchor")).toBe("middle");

        // Explicit placement, never `word-spacing`: one `<tspan>` per word at
        // an absolute `x`. With the mocked metrics (40 ref units per char) a
        // size-20 "Hello there" is 40px per word; W=2 doubles the 8px space to
        // a 16px gap, so the words start 56px apart, while W=0 closes the gap
        // entirely and the words abut 40px apart.
        const wide = shapeOf(textEntry({ id: "w", font_size: 20, word_spacing: 2 })).text;
        expect(wide.hasAttribute("word-spacing")).toBe(false);
        const wideSpans = Array.from(wide.querySelectorAll("tspan"));
        expect(wideSpans).toHaveLength(2);
        expect(wideSpans.map((s) => s.textContent)).toEqual(["Hello", "there"]);
        const wideXs = wideSpans.map((s) => Number(s.getAttribute("x")));
        expect(wideXs[1] - wideXs[0]).toBe(56);

        const tight = shapeOf(textEntry({ id: "t", font_size: 20, word_spacing: 0 })).text;
        expect(tight.hasAttribute("word-spacing")).toBe(false);
        const tightSpans = Array.from(tight.querySelectorAll("tspan"));
        expect(tightSpans).toHaveLength(2);
        const tightXs = tightSpans.map((s) => Number(s.getAttribute("x")));
        expect(tightXs[1] - tightXs[0]).toBe(40);
    });

    it("widens the gap even where a lone space measures zero", () => {
        // Some engines collapse a space-only run to zero even with
        // `xml:space="preserve"`. The layout must then fall back to
        // `"a a" - "aa"` rather than turning word spacing into a no-op.
        Object.defineProperty(SVGElement.prototype, "getComputedTextLength", {
            configurable: true,
            value(this: SVGElement): number {
                const text = this.textContent || "";
                if (text === " ") return 0;
                return text.length * 40;
            },
        });
        resetTextMetrics();

        const wide = shapeOf(textEntry({ id: "w", font_size: 20, word_spacing: 2 })).text;
        const spans = Array.from(wide.querySelectorAll("tspan"));
        expect(spans).toHaveLength(2);
        // "a a" (120) minus "aa" (80) recovers the 40-unit space, so the gap
        // is widened exactly as in the test above instead of staying shut.
        expect(Number(spans[1].getAttribute("x")) - Number(spans[0].getAttribute("x"))).toBe(56);
    });

    it("tilts as perspective when CSS 3D is available, with backend-matching signs", () => {
        (globalThis as unknown as { CSS?: unknown }).CSS = {
            supports: (prop: string, value: string) =>
                (prop === "transform" && value === "perspective(1px)") ||
                (prop === "transform-box" && value === "fill-box"),
        };
        resetPerspectiveCache();
        expect(supportsPerspective()).toBe(true);

        const group = buildEntryShape(textEntry({ angle_x: 30, angle_y: 15 }), makeOverlay(), noopHandlers)!;
        const text = group.querySelector(".entry-text") as SVGTextElement;
        expect(text).not.toBeNull();
        // The perspective lives in style (about the block's own centre), the
        // 2D remainder on an outer `<g>` -- never stacked as a `cos` squash.
        const style = text.getAttribute("style") ?? "";
        expect(style).toContain("perspective(");
        expect(style).toContain("transform-box: fill-box");
        expect(style).toContain("transform-origin: center");
        // The backend's +Z points away while CSS's points towards, so both
        // angles are negated to show the same side larger.
        expect(style).toContain("rotateX(-30deg)");
        expect(style).toContain("rotateY(-15deg)");
        expect(text.getAttribute("transform")).toBeNull();
        const outline = text.previousElementSibling as SVGTextElement;
        expect(outline.classList.contains("entry-text-outline")).toBe(true);
        expect(outline.getAttribute("style")).toBe(style);
        expect(outline.innerHTML).toBe(text.innerHTML);
        const wrap = text.parentElement as SVGGElement | null;
        expect(wrap?.tagName.toLowerCase()).toBe("g");
        // No fallback squash alongside the perspective.
        expect(wrap?.getAttribute("transform") ?? "").not.toContain("scale(0.");

        // And the helper degrades to "" without 3D or without angles.
        (globalThis as unknown as { CSS?: unknown }).CSS = undefined;
        resetPerspectiveCache();
        expect(supportsPerspective()).toBe(false);
    });

    it.each([6, 23, 32])("keeps all fills above the complete block outline at width %s", (strokeWidth) => {
        const { text } = shapeOf(textEntry({ text: "HELLO there\nHELLO", font_size: 40,
            stroke_width: strokeWidth, stroke_color: "#ffffff", word_spacing: 2,
            line_spacing: 0.5, font_scale_x: 1.5, angle_z: 15 }));
        const outline = text.previousElementSibling as SVGTextElement;
        expect(outline.classList.contains("entry-text-outline")).toBe(true);
        expect(outline.getAttribute("stroke-width")).toBe(String(strokeWidth * 2));
        expect(outline.getAttribute("stroke")).toBe("#ffffff");
        expect(outline.getAttribute("fill")).toBe("#ffffff");
        expect(text.getAttribute("stroke")).toBe("none");
        expect(text.getAttribute("stroke-width")).toBe("0");
        expect(outline.innerHTML).toBe(text.innerHTML);
        expect(outline.getAttribute("transform")).toBe(text.getAttribute("transform"));
        expect(outline.getAttribute("pointer-events")).toBe("none");
        expect(outline.getAttribute("aria-hidden")).toBe("true");
    });

    it("spaces the baselines by the line factor", () => {
        // An explicit size keeps the fit out of it: a taller block would
        // otherwise be shrunk back into the box and the gaps would not compare.
        const gapOf = (over: Partial<Entry>): number => {
            const text = shapeOf(textEntry({ text: "one\ntwo", font_size: 20, ...over })).text;
            const ys = Array.from(text.querySelectorAll("tspan")).map((s) => Number(s.getAttribute("y")));
            expect(ys).toHaveLength(2);
            return ys[1] - ys[0];
        };
        const natural = gapOf({});
        expect(natural).toBeGreaterThan(0);
        expect(gapOf({ id: "w", line_spacing: 2 })).toBe(Math.round(natural * 2));
        expect(gapOf({ id: "n", line_spacing: 0.5 })).toBe(Math.round(natural * 0.5));
    });

    it("keeps the stretch out of the wrap, so the fit still fills the box", () => {
        // The lines are laid out unscaled and stretched by the transform, so a
        // 2x-wide block has to wrap into half the width to come out the same.
        const plain = shapeOf(textEntry({ text: "one two three four five six" })).text;
        const wide = shapeOf(textEntry({ id: "w", text: "one two three four five six", font_scale_x: 2 })).text;
        expect(wide.querySelectorAll("tspan").length).toBeGreaterThan(plain.querySelectorAll("tspan").length);
    });

    it("matches the backend line-spacing formula at auto (null) and 1x", () => {
        // Backend: _advance_for(line_h, spacing, None) == line_h + spacing,
        // where line_h = ascent + descent (each scaled to size and rounded
        // separately, like Pillow's font.getmetrics()), and factor 1.0 is
        // identical. Auto and 1x must therefore agree.
        const faceMetrics = { ascentPerEm: 0.9, descentPerEm: 0.2 } as FontMetrics;
        const size = 20;
        const lineHeight = Math.round(size * 0.9) + Math.round(size * 0.2);
        const expected = lineHeight + Math.round(size * 0.14);
        expect(lineHeightFor(size, faceMetrics)).toBe(lineHeight);
        expect(spacingFor(size)).toBe(Math.max(1, Math.round(size * 0.14)));
        expect(advanceFor(size, null, faceMetrics)).toBe(expected);
        expect(advanceFor(size, 1, faceMetrics)).toBe(expected);
        // A real factor scales the whole advance, as the backend does.
        expect(advanceFor(size, 1.5, faceMetrics)).toBe(Math.max(1, Math.round(expected * 1.5)));
    });

    it("reads the per-size metrics table when the payload ships one", () => {
        // FreeType's per-size rounding cannot be derived from font units
        // (49 of 67 fit sizes drift a pixel for ComicMono), so the backend
        // ships its `getmetrics()` ints -- one `[ascent, descent]` pair per
        // size from MIN_FONT_SIZE (6). Those ints win over the formulas.
        const tabled = { ascentPerEm: 0.8, descentPerEm: 0.3, lineMetrics: [8, 3, 9, 3, 10, 4, 11, 4] } as FontMetrics;
        // The formula would answer 5+2=7 for size 6; the render says 8+3=11.
        expect(lineHeightFor(6, tabled)).toBe(11);
        expect(lineHeightFor(9, tabled)).toBe(15);
        expect(ascentFor(6, tabled)).toBe(8);
        expect(ascentFor(8, tabled)).toBe(10);
        // Baseline/leading/advance all key off the same pairs, so a block
        // sized by the table is drawn by exactly the same arithmetic.
        expect(advanceFor(7, null, tabled)).toBe(lineHeightFor(7, tabled) + spacingFor(7));
    });

    it("sizes outside the table keep the rounded formula", () => {
        const tabled = { ascentPerEm: 0.8, descentPerEm: 0.3, lineMetrics: [8, 3, 9, 3] } as FontMetrics;
        // Size 5 is below the table's first row (MIN_FONT_SIZE = 6).
        expect(lineHeightFor(5, tabled)).toBe(Math.round(5 * 0.8) + Math.round(5 * 0.3));
        expect(ascentFor(5, tabled)).toBeCloseTo(4, 10);
        // Size 8 is past its last row (7).
        expect(lineHeightFor(8, tabled)).toBe(Math.round(8 * 0.8) + Math.round(8 * 0.3));
        // And a payload with no table at all is unchanged.
        const plain = { ascentPerEm: 0.8, descentPerEm: 0.3 } as FontMetrics;
        expect(lineHeightFor(6, plain)).toBe(Math.round(6 * 0.8) + Math.round(6 * 0.3));
        expect(ascentFor(6, plain)).toBeCloseTo(4.8, 10);
    });

    it("a mangled or legacy payload falls back to the estimate", () => {
        const base = {
            font_filename: "f.ttf",
            font_name: "F",
            font_data_uri: "data:font/ttf;base64,AA==",
            font_format: "truetype",
            unitsPerEm: 1000,
            ascent: 800,
            descent: 300,
        } as FontData;
        // Legacy: no table.
        expect(fontMetricsFromFontData(base)?.lineMetrics).toBeUndefined();
        // Odd length / non-numbers: fail closed to the formulas rather than
        // misread half a pair (or a fraction as a pixel count).
        expect(fontMetricsFromFontData({ ...base, line_metrics: [8, 3, 9] })?.lineMetrics).toBeUndefined();
        expect(
            fontMetricsFromFontData({ ...base, line_metrics: [8, "3"] as unknown as number[] })?.lineMetrics,
        ).toBeUndefined();
        expect(
            fontMetricsFromFontData({ ...base, line_metrics: [8, Number.NaN] })?.lineMetrics,
        ).toBeUndefined();
        // An empty table is a deliberate "back could not measure" answer.
        expect(fontMetricsFromFontData({ ...base, line_metrics: [] })?.lineMetrics).toEqual([]);
        expect(fontMetricsFromFontData({ ...base, line_metrics: [8, 3] })?.lineMetrics).toEqual([8, 3]);
    });

    it("applies word spacing as a multiple of the face's space advance", () => {
        // Backend: gap = spaceAdvance * factor (word_extra = (factor-1)*space).
        // factor 1.0 (and null) -> the face's own space; factor 2.0 -> double.
        const entry = textEntry({ text: "Hello there", font_size: 20 });
        const normal = shapeOf(entry).text;
        const spans = Array.from(normal.querySelectorAll("tspan"));
        expect(spans).toHaveLength(1);
        expect(normal.getAttribute("text-anchor")).toBe("middle");

        const wide = shapeOf(textEntry({ id: "w", text: "Hello there", font_size: 20, word_spacing: 2 })).text;
        const wideSpans = Array.from(wide.querySelectorAll("tspan"));
        expect(wideSpans).toHaveLength(2);
        // "Hello" = 40px at size 20 (5 chars * 40 ref units * 0.2 scale); factor 2
        // doubles the 8px space to 16px, so the second word starts 40 + 16 = 56px
        // after the first.
        const xs = wideSpans.map((s) => Number(s.getAttribute("x")));
        expect(xs[1] - xs[0]).toBe(56);

        const tight = shapeOf(textEntry({ id: "t", text: "Hello there", font_size: 20, word_spacing: 0 })).text;
        const tightSpans = Array.from(tight.querySelectorAll("tspan"));
        const tightXs = tightSpans.map((s) => Number(s.getAttribute("x")));
        // factor 0 collapses the gap to 0 -> words abut at 40px.
        expect(tightXs[1] - tightXs[0]).toBe(40);
    });

    it("wraps a line only the shared wrap margin can push over", () => {
        // The browser and Pillow round advances independently, close enough to
        // eat a one-pixel margin: the traced "just once," (69.0 of 70) wrapped
        // on screen but not in the render. Both engines now wrap with two
        // pixels off the budget, so a boundary call cannot be won by rounding.
        // Stub width is 40 ref units per character. The 40px-wide region gives
        // a 30px text area: the old limit was 250 ref units at 12px, the new
        // one (30-2)*100/12 = 233.33 -- so a six-character line (240) fits the
        // old budget and must now break, while a five-character one (200) still
        // fits.
        const five = shapeOf(
            textEntry({
                text: "Ab Cd",
                font_size: 12,
                region: { type: "rectangle", coords: { x: 20, y: 20, w: 40, h: 30 } },
            }),
        ).text;
        expect(five.querySelectorAll("tspan")).toHaveLength(1);

        const six = shapeOf(
            textEntry({
                id: "w",
                text: "Abb Cc",
                font_size: 12,
                region: { type: "rectangle", coords: { x: 20, y: 20, w: 40, h: 30 } },
            }),
        ).text;
        const spans = Array.from(six.querySelectorAll("tspan"));
        expect(spans.map((s) => s.textContent)).toEqual(["Abb", "Cc"]);
    });
});

describe("the colours on the preview", () => {
    let metrics: PropertyDescriptor | undefined;

    beforeEach(() => {
        state.isLiveInpaintedEnabled = true;
        // Same fiction as the geometry block above: jsdom measures every token
        // at zero, which collapses the fit and leaves no text node to inspect.
        metrics = Object.getOwnPropertyDescriptor(SVGElement.prototype, "getComputedTextLength");
        Object.defineProperty(SVGElement.prototype, "getComputedTextLength", {
            configurable: true,
            value(this: SVGElement): number {
                return (this.textContent || "").length * 40;
            },
        });
        resetTextMetrics();
    });

    afterEach(() => {
        if (metrics) Object.defineProperty(SVGElement.prototype, "getComputedTextLength", metrics);
        else delete (SVGElement.prototype as unknown as Record<string, unknown>).getComputedTextLength;
        resetTextMetrics();
    });

    function drawn(over: Partial<Entry> = {}): { fill: string | null; stroke: string | null; plate: string | null } {
        const group = buildEntryShape(
            makeEntry({
                text: "Hello there",
                region: { type: "rectangle", coords: { x: 20, y: 20, w: 360, h: 260 } },
                ...over,
            }),
            makeOverlay(),
            noopHandlers,
        )!;
        const text = group.querySelector(".entry-text") as SVGTextElement;
        expect(text).not.toBeNull();
        return {
            fill: text.getAttribute("fill"),
            stroke: group.querySelector(".entry-text-outline")?.getAttribute("stroke") ?? text.getAttribute("stroke"),
            plate: group.querySelector(".entry-plate")?.getAttribute("fill") ?? null,
        };
    }

    it("falls back to the stand-ins for a region nobody has read", () => {
        // What the overlay has always drawn, and what the backend works out for
        // the same entry: a white plate, black text, outlined in the plate.
        expect(drawn()).toEqual({ fill: "#000000", stroke: "#ffffff", plate: "#ffffff" });
    });

    it("draws what the OCR step measured instead", () => {
        // The case the stand-ins got wrong: white lettering on a dark panel used
        // to preview as black text on white.
        expect(
            drawn({
                detected_font_color: "#ffffff",
                detected_stroke_color: "#101010",
                detected_bg: "#1b1b1e",
            }),
        ).toEqual({ fill: "#ffffff", stroke: "#101010", plate: "#1b1b1e" });
    });

    it("outlines in the measured plate when the lettering has no outline of its own", () => {
        // `detected_stroke_color` is only sent for real outlines, so a null one
        // has to keep meaning "the plate's colour" -- not "no outline".
        const { stroke, plate } = drawn({ detected_font_color: "#101010", detected_bg: "#f6f1e7" });
        expect(stroke).toBe("#f6f1e7");
        expect(plate).toBe("#f6f1e7");
    });

    it("lets the user's own picks win over the measurements", () => {
        const { fill, stroke } = drawn({
            font_color: "#ff0000",
            stroke_color: "#00ff00",
            detected_font_color: "#ffffff",
            detected_stroke_color: "#101010",
        });
        expect(fill).toBe("#ff0000");
        expect(stroke).toBe("#00ff00");
    });

    it("doubles the outline value in the stroke-width attribute", () => {
        // Pillow's stroke sits wholly outside the glyph while SVG halves the
        // attribute: rendered with value N the plugins' outline is N px of ring
        // per side, so the preview asks 2N for the same ring (measured on
        // `tmp_img/check_stroke.py`). The bounded slider value stays the entry's
        // own number; only the attribute doubles.
        const attrOf = (over: Partial<Entry>): string | null => {
            const group = buildEntryShape(
                makeEntry({
                    text: "Hello",
                    stroke_color: "#101010",
                    region: { type: "rectangle", coords: { x: 20, y: 20, w: 360, h: 260 } },
                    ...over,
                }),
                makeOverlay(),
                noopHandlers,
            )!;
            const text = group.querySelector(".entry-text-outline") as SVGTextElement;
            expect(text).not.toBeNull();
            expect(text.getAttribute("paint-order")).toBe("stroke");
            return text.getAttribute("stroke-width");
        };
        expect(attrOf({ stroke_width: 1 })).toBe("2");
        expect(attrOf({ stroke_width: 3 })).toBe("6");
        expect(attrOf({ stroke_width: 12 })).toBe("24");

        const auto = attrOf({ stroke_width: null });
        // Auto outline at the fit's size: 0.09 of it, at least 1 -> doubled, so
        // whatever size the fit landed on, the attribute is even and >= 2.
        expect(Number(auto)).toBeGreaterThanOrEqual(2);
        expect(Number(auto) % 2).toBe(0);

        // A zero outline stays off: no attribute doubling, no stroke.
        const zeroGroup = buildEntryShape(
            makeEntry({
                text: "Hello",
                stroke_width: 0,
                stroke_color: "#101010",
                region: { type: "rectangle", coords: { x: 20, y: 20, w: 360, h: 260 } },
            }),
            makeOverlay(),
            noopHandlers,
        )!;
        const textEl = zeroGroup.querySelector(".entry-text") as SVGTextElement;
        expect(textEl.getAttribute("stroke")).toBe("none");
        expect(textEl.getAttribute("stroke-width")).toBe("0");
        expect(zeroGroup.querySelector(".entry-text-outline")).toBeNull();
    });

    it("drops a measurement that would vanish into an explicit plate", () => {
        // White lettering measured on a dark panel, then set on a white plate the
        // user chose. Mirrors `typeset._legible_on`, so the preview and the render
        // reach the same answer.
        const { fill, plate } = drawn({
            bg_mode: "color",
            bg_color: "#ffffff",
            detected_font_color: "#fdfdfd",
            detected_bg: "#1b1b1e",
        });
        expect(plate).toBe("#ffffff");
        expect(fill).toBe("#000000");
    });

    it("keeps a measurement that still reads on the new plate", () => {
        const { fill } = drawn({
            bg_mode: "color",
            bg_color: "#fffaf0",
            detected_font_color: "#b21824",
        });
        expect(fill).toBe("#b21824");
    });

    it("paints no plate for a clean entry, measured or not", () => {
        const { plate, fill } = drawn({
            bg_mode: "clean",
            detected_bg: "#1b1b1e",
            detected_font_color: "#ffffff",
        });
        expect(plate).toBeNull();
        expect(fill).toBe("#ffffff");
    });
});

describe("the per-glyph symbol fallback", () => {
    let metrics: PropertyDescriptor | undefined;

    /** Fabricated `/api/fonts` data; these tests do not load font files. */
    function face(over: Partial<FontData> = {}): FontData {
        return {
            font_filename: "face.ttf",
            font_name: "Face",
            font_data_uri: "data:font/ttf;base64,AA==",
            font_format: "truetype",
            unitsPerEm: 2048,
            ascent: 1800,
            descent: 400,
            ...over,
        };
    }

    // Widths are the fiction of the harness below: 40 reference units per
    // character, so "Hi" is 80 and "♡" measures 40 in either face; the
    // composite is that sum with the symbol run multiplied by the face's
    // calibration, exactly the linear walk `measure` performs.
    const comic: FontData = face({
        font_name: "Comic Mono",
        font_filename: "comic.ttf",
        symbol_scale: 0.971,
        coverage: [32, 126],
    });
    const expandedText: FontData = face({
        font_name: "Test Text",
        font_filename: "test-text.ttf",
        symbol_scale: 1.136,
        coverage: [32, 126],
    });
    const hearts: FontData = face({
        font_name: "Noto Sans Symbols 2",
        font_filename: "symbols.ttf",
        symbol_scale: 1,
        coverage: [32, 32, 0x2661, 0x2661, 0x2605, 0x2605],
    });

    beforeEach(() => {
        state.currentFontsData = [comic, expandedText, hearts];
        state.currentSymbolFont = hearts;
        state.currentSymbolFonts = [];
        metrics = Object.getOwnPropertyDescriptor(SVGElement.prototype, "getComputedTextLength");
        Object.defineProperty(SVGElement.prototype, "getComputedTextLength", {
            configurable: true,
            value(this: SVGElement): number {
                return (this.textContent || "").length * 40;
            },
        });
        resetTextMetrics();
    });

    afterEach(() => {
        if (metrics) Object.defineProperty(SVGElement.prototype, "getComputedTextLength", metrics);
        else delete (SVGElement.prototype as unknown as Record<string, unknown>).getComputedTextLength;
        resetTextMetrics();
    });

    /** A text-only entry whose family resolves as the CSS family fonts.ts uses. */
    function symbolEntry(over: Partial<Entry> = {}): Entry {
        return makeEntry({
            text: "Hi♡!",
            fontname: "Test_Text",
            font_size: 40,
            region: { type: "rectangle", coords: { x: 20, y: 20, w: 360, h: 260 } },
            ...over,
        });
    }

    function spansOf(entry: Entry): SVGTSpanElement[] {
        const group = buildEntryShape(entry, makeOverlay(), noopHandlers)!;
        expect(group).not.toBeNull();
        return Array.from(
            (group.querySelector(".entry-text") as SVGTextElement).querySelectorAll("tspan"),
        );
    }

    it.each(["'Test_Text', sans-serif", '"Test_Text", sans-serif'])(
        "keeps calibrated fallback positions for the saved CSS family %s",
        (family) => {
            const expected = spansOf(symbolEntry({ fontname: "Test_Text" }));
            const actual = spansOf(symbolEntry({ fontname: family }));
            expect(actual.map(span => span.textContent)).toEqual(["Hi", "♡", "!"]);
            expect(actual.map(span => [span.getAttribute("x"), span.getAttribute("y"), span.getAttribute("font-size")]))
                .toEqual(expected.map(span => [span.getAttribute("x"), span.getAttribute("y"), span.getAttribute("font-size")]));
        },
    );

    it("splits a mixed line into one run per face, at the calibrated em", () => {
        const spans = spansOf(symbolEntry());
        expect(spans.map((s) => s.textContent)).toEqual(["Hi", "♡", "!"]);
        // The hidden face carries the symbol run, at `round(size * scale)`:
        // round(40 * 1.136) = 45. Primary runs inherit the entry's own face.
        expect(spans[1].getAttribute("font-family")).toBe("Noto_Sans_Symbols_2");
        expect(spans[1].getAttribute("font-size")).toBe("45");
        expect(spans[0].hasAttribute("font-family")).toBe(false);
        expect(spans[2].hasAttribute("font-family")).toBe(false);
        // Explicit placement: cumulative x from the composite widths at
        // REF_SIZE -- "Hi" = 80, scaled "♡" = 40*1.136 = 45.44, "!" = 40, all
        // at scale 0.4 against the anchor x = 200.
        const xs = spans.map((s) => Number(s.getAttribute("x")));
        const total = (80 + 40 * (45 / 40) + 40) * 0.4;
        expect(xs[0]).toBeCloseTo(200 - total / 2, 3);
        expect(xs[1]).toBeCloseTo(200 - total / 2 + 80 * 0.4, 3);
        expect(xs[2]).toBeCloseTo(200 - total / 2 + 80 * 0.4 + 40 * (45 / 40) * 0.4, 3);
    });

    it("a face whose calibration shrinks keeps the match its own face earned", () => {
        // The fixture's shrinking scale rounds to a smaller em instead of
        // jumping to the shared full size.
        const spans = spansOf(symbolEntry({ fontname: "Comic_Mono", id: "c" }));
        expect(spans.map((s) => s.textContent)).toEqual(["Hi", "♡", "!"]);
        expect(spans[1].getAttribute("font-size")).toBe("39");
    });

    it("keeps a line with nothing to swap as one tspan", () => {
        const spans = spansOf(symbolEntry({ text: "Hi!", id: "u" }));
        expect(spans).toHaveLength(1);
        expect(spans[0].textContent).toBe("Hi!");
        expect(spans[0].hasAttribute("font-family")).toBe(false);
        // Whitespace stays primary too, per `_needs_symbol`'s space guard.
        const spaced = spansOf(symbolEntry({ text: "Hi there!", id: "s" }));
        // Word-wrapped only if the region forces it; it fits here.
        expect(spaced).toHaveLength(1);
    });

    it("a character neither face covers stays the primary's instead of swapping tofu for tofu", () => {
        const spans = spansOf(symbolEntry({ text: "Hi\u3042!", id: "t" }));
        expect(spans).toHaveLength(1);
        expect(spans[0].textContent).toBe("Hi\u3042!");
    });

    it("spaces mixed words like plain ones: the token, not its glyphs, owns the gap", () => {
        // Space advance 40 ref units, W=2 doubles it; "♡!" is one token of
        // 40*(23/20) + 40 = 86 ref units at 20px. The runs are placed at
        // cumulative sub-cursors, no extra gaps inserted.
        const spans = spansOf(symbolEntry({ text: "Hi ♡!", font_size: 20, word_spacing: 2, id: "w" }));
        expect(spans.map((s) => s.textContent)).toEqual(["Hi", "♡", "!"]);
        const xs = spans.map((s) => Number(s.getAttribute("x")));
        const scale = 20 / 100;
        const gap = (40 + (2 - 1) * 40) * scale;
        expect(xs[1] - xs[0]).toBeCloseTo(80 * scale + gap, 3); // token 1 starts after "Hi" + its gap
        expect(xs[2] - xs[1]).toBeCloseTo(40 * (23 / 20) * scale, 3); // rounded 23px fallback em
    });

    it("every placed run is anchored at its own x, whatever the line's alignment", () => {
        // A `tspan` with an explicit `x` starts a new text chunk and inherits
        // the parent's `text-anchor`: under `center` every run would centre
        // *at its own x* and the cumulative walk would fall apart (verified in
        // Chromium: runs aimed at 100/120/x landed at 92/114.5/x). The walk --
        // and the backend's `_line_pieces` -- are start-anchored, so each
        // chunk pins it.
        const mixed = spansOf(symbolEntry({ id: "an" }));
        expect(mixed.map((s) => s.getAttribute("text-anchor"))).toEqual(["start", "start", "start"]);
        // The spaced path's sub-runs, and the token slots beside them, too.
        const spaced = spansOf(symbolEntry({ text: "Hi ♡!", font_size: 20, word_spacing: 2, id: "an2" }));
        expect(spaced.map((s) => s.getAttribute("text-anchor"))).toEqual(["start", "start", "start"]);
    });

    it("the calibration clamp is inclusive on both ends, like the render's", () => {
        // `SCALE_LIMITS = (0.5, 3.0)`: the render clamps an out-of-band
        // calibration *to* those values and ships them, so a payload can
        // legitimately carry exactly 0.5 or 3 -- which the old exclusive
        // comparison threw away, desynchronising the two engines.
        state.currentFontsData = [
            face({ font_name: "Large Text", font_filename: "big.ttf", symbol_scale: 3, coverage: [32, 126] }),
            face({ font_name: "Small Text", font_filename: "small.ttf", symbol_scale: 0.5, coverage: [32, 126] }),
            hearts,
        ];
        resetTextMetrics();
        expect(spansOf(symbolEntry({ fontname: "Large_Text", id: "bi" }))[1].getAttribute("font-size")).toBe(
            String(Math.round(40 * 3)),
        );
        expect(spansOf(symbolEntry({ fontname: "Small_Text", id: "sm" }))[1].getAttribute("font-size")).toBe(
            String(Math.round(40 * 0.5)),
        );
    });

    it("a refreshed font list re-reads calibration even when its length is unchanged", () => {
        expect(spansOf(symbolEntry({ id: "a" }))[1].getAttribute("font-size")).toBe("45");
        // Replacing the list invalidates the primary specs even when the
        // refreshed inventory has exactly the same number of faces.
        state.currentFontsData = [
            comic,
            face({
                font_name: "Test Text",
                font_filename: "test-text.ttf",
                symbol_scale: 0.971,
                coverage: [32, 126],
            }),
            hearts,
        ];
        expect(spansOf(symbolEntry({ id: "b", fontname: "Test_Text" }))[1].getAttribute("font-size")).toBe("39");
        // The font event also clears cached layouts and raw measurements.
        resetTextMetrics();
        const spans = spansOf(symbolEntry({ id: "c", fontname: "Test_Text" }));
        expect(spans[1].getAttribute("font-size")).toBe("39");
        // And the widths went with it: the run walk uses the new calibration.
        expect(Number(spans[2].getAttribute("x")) - Number(spans[1].getAttribute("x"))).toBeCloseTo(40 * (39 / 40) * 0.4, 3);
    });

    it("uses three independently calibrated fallbacks in order, then Noto", () => {
        const first = face({ font_filename: "first.ttf", font_name: "First", coverage: [0x2661, 0x2661] });
        const second = face({ font_filename: "second.ttf", font_name: "Second", coverage: [0x2661, 0x2661, 0x2713, 0x2713] });
        const third = face({ font_filename: "third.ttf", font_name: "Third", coverage: [0xe001, 0xe001] });
        state.currentFontsData = [{ ...expandedText, symbol_scales: {
            "first.ttf": 3, "second.ttf": 1, "third.ttf": 0.5, "symbols.ttf": 0.75,
        } }];
        state.currentSymbolFonts = [first, second, third, hearts];
        resetTextMetrics();
        const spans = spansOf(symbolEntry({ text: "Hi♡✓\ue001★!", font_size: 20 }));
        expect(spans.map((span) => span.textContent)).toEqual(["Hi", "♡", "✓", "\ue001", "★", "!"]);
        expect(spans.slice(1, 5).map((span) => span.getAttribute("font-family")))
            .toEqual(["First", "Second", "Third", "Noto_Sans_Symbols_2"]);
        expect(spans.slice(1, 5).map((span) => span.getAttribute("font-size")))
            .toEqual(["60", "20", "10", "15"]);
        expect(spans.map(span => span.getAttribute("y")))
            .toEqual(Array(spans.length).fill(spans[0].getAttribute("y")));
        const x = spans.map((span) => Number(span.getAttribute("x")));
        expect(x[2] - x[1]).toBeCloseTo(24); // 40 reference units at 60px
        expect(x[3] - x[2]).toBeCloseTo(8);
        expect(x[4] - x[3]).toBeCloseTo(4);
        state.currentSymbolFonts = [second, first, third, hearts];
        resetTextMetrics();
        const reordered = spansOf(symbolEntry({ text: "Hi♡!", font_size: 20 }));
        expect(reordered[1].getAttribute("font-family")).toBe("Second");
        expect(reordered[1].getAttribute("font-size")).toBe("20");
    });

    it("keeps primary glyphs when a symbol font also maps them", () => {
        state.currentSymbolFonts = [face({ font_name: "Icons", coverage: [32, 126] }), hearts];
        resetTextMetrics();
        const spans = spansOf(symbolEntry({ text: "Hi!" }));
        expect(spans).toHaveLength(1);
        expect(spans[0].hasAttribute("font-family")).toBe(false);
    });

    it.each(["Test_Text", "'Test_Text', sans-serif", '"Test_Text", sans-serif'])(
        "explicit character ownership keeps calibrated spacing for %s",
        (family) => {
            const first = face({ font_filename: "first.ttf", font_name: "First", coverage: [32, 126, 0x2661, 0x2661] });
            const second = face({ font_filename: "second.ttf", font_name: "Second",
                coverage: [32, 126, 0x2661, 0x2661], forced_codepoints: [0x48, 0x2661] });
            state.currentFontsData = [{ ...expandedText, symbol_scales: { "first.ttf": 2, "second.ttf": 0.5 } }];
            state.currentSymbolFonts = [first, second, hearts];
            resetTextMetrics();
            const spans = spansOf(symbolEntry({ text: "Hi♡!", fontname: family, font_size: 20 }));
            expect(spans.map(span => span.textContent)).toEqual(["H", "i", "♡", "!"]);
            expect(spans[0].getAttribute("font-family")).toBe("Second");
            expect(spans[0].getAttribute("font-size")).toBe("10");
            expect(spans[2].getAttribute("font-family")).toBe("Second");
            expect(spans[1].hasAttribute("font-family")).toBe(false);
            expect(Number(spans[1].getAttribute("x")) - Number(spans[0].getAttribute("x"))).toBeCloseTo(4);
            state.currentSymbolFonts = [first, { ...second, forced_codepoints: [] }, hearts];
            resetTextMetrics();
            const restored = spansOf(symbolEntry({ text: "Hi♡!", fontname: family, font_size: 20 }));
            expect(restored.map(span => span.textContent)).toEqual(["Hi", "♡", "!"]);
            expect(restored[1].getAttribute("font-family")).toBe("First");
        },
    );

    it.each([
        { size: 20, spacing: 1 }, { size: 21, spacing: 2 },
        { size: 40, spacing: 1 }, { size: 72, spacing: 2 },
    ])("matches the original question mark bottom at $size px with spacing $spacing", ({ size, spacing }) => {
        const symbols = face({ font_filename: "test-symbols.ttf", font_name: "Test Symbols",
            coverage: [32, 126], forced_codepoints: [63] });
        state.currentFontsData = [{ ...expandedText, symbol_scales: { "test-symbols.ttf": 2 },
            symbol_glyph_adjustments: { "test-symbols.ttf": { "63": {
                scale: 1.485, primary_bottom: 0.05, fallback_bottom: 0.1,
            } } } }];
        state.currentSymbolFonts = [symbols, hearts];
        resetTextMetrics();
        const spans = spansOf(symbolEntry({ text: "Hi ??!", font_size: size, word_spacing: spacing }));
        expect(spans.map(span => span.textContent)).toEqual([spacing === 1 ? "Hi " : "Hi", "??", "!"]);
        const pixels = Math.round(size * 1.485);
        expect(spans[1].getAttribute("font-size")).toBe(String(pixels));
        const baseline = Number(spans[0].getAttribute("y"));
        expect(Number(spans[1].getAttribute("y"))).toBeCloseTo(baseline + 0.1 * pixels - 0.05 * size);
        expect(Number(spans[2].getAttribute("y"))).toBe(baseline);
        expect(Number(spans[2].getAttribute("x")) - Number(spans[1].getAttribute("x")))
            .toBeCloseTo(80 * pixels / 100);
    });

    it.each([0.5, 1.5])("aligns adjacent replacements and descenders at comma scale %s", (commaScale) => {
        const symbols = face({ font_filename: "test-symbols.ttf", font_name: "Test Symbols",
            coverage: [32, 126, 0x2661, 0x2661], forced_codepoints: [63, 44] });
        state.currentFontsData = [{ ...expandedText, symbol_scales: { "test-symbols.ttf": 2 },
            symbol_glyph_adjustments: { "test-symbols.ttf": {
                "63": { scale: 1.5, primary_bottom: 0.05, fallback_bottom: 0.1 },
                "44": { scale: commaScale, primary_bottom: -0.2, fallback_bottom: 0.1 },
            } } }];
        state.currentSymbolFonts = [symbols, hearts];
        resetTextMetrics();
        const spans = spansOf(symbolEntry({ text: "Hi?,?♡!", font_size: 20 }));
        expect(spans.map(span => span.textContent)).toEqual(["Hi", "?", ",", "?", "♡", "!"]);
        const baseline = Number(spans[0].getAttribute("y"));
        expect(spans.slice(1, 5).map(span => span.getAttribute("font-size")))
            .toEqual(["30", String(20 * commaScale), "30", "40"]);
        expect(spans.map(span => Number(span.getAttribute("y"))))
            .toEqual([baseline, baseline + 2, baseline + 2 * commaScale + 4, baseline + 2, baseline, baseline]);
        const x = spans.map(span => Number(span.getAttribute("x")));
        expect(x[2] - x[1]).toBeCloseTo(12);
        expect(x[3] - x[2]).toBeCloseTo(8 * commaScale);
        expect(x[4] - x[3]).toBeCloseTo(12);

        // Removing ownership ignores old adjustment metadata immediately.
        state.currentSymbolFonts = [{ ...symbols, forced_codepoints: [] }, hearts];
        resetTextMetrics();
        const restored = spansOf(symbolEntry({ text: "Hi?,?♡!", font_size: 20 }));
        expect(restored.map(span => span.textContent)).toEqual(["Hi?,?", "♡", "!"]);
        expect(restored[1].getAttribute("font-size")).toBe("40");
        expect(restored.map(span => span.getAttribute("y"))).toEqual(
            Array(3).fill(restored[0].getAttribute("y")),
        );
    });

    it.each([
        { scale: NaN }, { scale: 0.49 }, { scale: 3.01 },
        { primary_bottom: Infinity }, { primary_bottom: -9 },
        { fallback_bottom: NaN }, { fallback_bottom: 9 },
    ])("uses normal font calibration when replacement metadata is invalid: %j", (invalid) => {
        state.currentFontsData = [{ ...expandedText, symbol_scales: { "test-symbols.ttf": 2 },
            symbol_glyph_adjustments: { "test-symbols.ttf": { "63": {
                scale: 1.5, primary_bottom: 0.05, fallback_bottom: 0.1, ...invalid,
            } } } }];
        state.currentSymbolFonts = [face({ font_filename: "test-symbols.ttf", font_name: "Test Symbols",
            coverage: [32, 126], forced_codepoints: [63] }), hearts];
        resetTextMetrics();
        const spans = spansOf(symbolEntry({ text: "Hi?!", font_size: 20 }));
        expect(spans[1].getAttribute("font-size")).toBe("40");
        expect(spans[1].getAttribute("y")).toBe(spans[0].getAttribute("y"));
    });

    it("keeps normal calibration if the primary font lacks an assigned character", () => {
        state.currentFontsData = [{ ...expandedText, coverage: [33, 33, 72, 72, 105, 105],
            symbol_scales: { "test-symbols.ttf": 2 }, symbol_glyph_adjustments: { "test-symbols.ttf": {
                "63": { scale: 1.5, primary_bottom: 0.05, fallback_bottom: 0.1 },
            } } }];
        state.currentSymbolFonts = [face({ font_filename: "test-symbols.ttf", font_name: "Test Symbols",
            coverage: [32, 126], forced_codepoints: [63] }), hearts];
        resetTextMetrics();
        const spans = spansOf(symbolEntry({ text: "Hi?!", font_size: 20 }));
        expect(spans[1].getAttribute("font-size")).toBe("40");
        expect(spans[1].getAttribute("y")).toBe(spans[0].getAttribute("y"));
    });
});
