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
import { resetPerspectiveCache, resetTextMetrics, supportsPerspective, tiltStyle } from "./textFit";
import { runSplit } from "./split";
import { PLAIN_GEOMETRY, state } from "../state";
import type { Entry } from "../state";
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
        const wrap = text.parentElement as SVGGElement | null;
        expect(wrap?.tagName.toLowerCase()).toBe("g");
        // No fallback squash alongside the perspective.
        expect(wrap?.getAttribute("transform") ?? "").not.toContain("scale(0.");

        // And the helper degrades to "" without 3D or without angles.
        (globalThis as unknown as { CSS?: unknown }).CSS = undefined;
        resetPerspectiveCache();
        expect(supportsPerspective()).toBe(false);
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
});
