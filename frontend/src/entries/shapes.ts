/**
 * The live preview: one `<g>` per visible entry in the page overlay.
 *
 * Each group is painted bottom-up in the same order the backend composites:
 * text-clean effect, background plate, text, then the selection outline on top.
 * Groups themselves are inserted in layer order, so an entry on layer 3 draws
 * over one on layer 1 exactly as `typeset._ordered` will.
 *
 * ## Colours here are deliberately not sampled from the page
 *
 * `auto` shows white plate / black text / white outline. The backend reads the
 * region's real dominant colour, but doing that in the browser means pulling the
 * page into a canvas and running `getImageData` per entry on every repaint --
 * seconds of main-thread work for a hint the user is about to override anyway.
 * The preview's job is to show the *layout*; the plate colour is settled at
 * render time.
 */

import { state, MIN_STROKE_WIDTH, MAX_STROKE_WIDTH } from "../state";
import type { Entry } from "../state";
import type { Point } from "../types";
import { regionPoints, regionBBox, cssEscape } from "./geometry";
import { drawFitLayout, fitText, readGeometry, textTransform, tiltStyle, typeArea } from "./textFit";

const SVG_NS = "http://www.w3.org/2000/svg";
const XHTML_NS = "http://www.w3.org/1999/xhtml";

/** What `auto` stands in for until the backend measures the artwork. */
const AUTO_PLATE = "#ffffff";

export interface ShapeHandlers {
    /** The region was clicked. */
    onSelect: (entry: Entry) => void;
    /** The region was clicked while showing a text preview -- edit in place. */
    onEditInPlace: (entry: Entry, group: SVGGElement) => void;
    /** Pointer entered or left the region. */
    onHover: (entry: Entry, hovering: boolean) => void;
    /**
     * The region was right-clicked. Never accompanied by `onSelect`: `contextmenu`
     * is not a `click`, and no other gesture is derived from it here.
     *
     * Optional so that a caller wanting nothing but the drawing -- a test, a
     * read-only preview -- does not have to supply a handler it will not use.
     */
    onContextMenu?: (entry: Entry) => void;
}

// ------------------------------------------------------------------------ colours

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));

function parseHex(value: string | null): [number, number, number] | null {
    if (!value) return null;
    const text = value.trim();
    const m = /^#([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i.exec(text);
    if (!m) return null;
    let hex = m[1];
    if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
}

/** Black or white, whichever reads on `background`. Same weights as the backend. */
export function contrastColor(background: string): string {
    const rgb = parseHex(background) ?? [255, 255, 255];
    const brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000;
    return brightness > 127 ? "#000000" : "#ffffff";
}

/** The plate an entry's text will sit on, or `null` when nothing is painted. */
function plateColor(entry: Entry): string | null {
    if (entry.bg_mode === "color") return entry.bg_color || AUTO_PLATE;
    if (entry.bg_mode === "transparent" || entry.bg_mode === "clean") return null;
    return AUTO_PLATE;
}

/** Auto outline width, matching `_draw_entry`'s `round(size * 0.09)`. */
function autoStrokeWidth(size: number): number {
    return Math.max(1, clamp(Math.round(size * 0.09), MIN_STROKE_WIDTH, MAX_STROKE_WIDTH));
}

// --------------------------------------------------------------------------- defs

function buildPattern(id: string, size: number, child: SVGElement): SVGPatternElement {
    const pattern = document.createElementNS(SVG_NS, "pattern");
    pattern.id = id;
    pattern.setAttribute("width", String(size));
    pattern.setAttribute("height", String(size));
    pattern.setAttribute("patternUnits", "userSpaceOnUse");
    pattern.appendChild(child);
    return pattern;
}

/**
 * The overlay's shared `<defs>`: the two hatch patterns that tell the clean
 * methods apart even where `backdrop-filter` is unavailable.
 */
export function ensureOverlayDefs(svg: SVGSVGElement): SVGDefsElement {
    let defs = svg.querySelector<SVGDefsElement>("#foxOverlayDefs");
    if (defs) return defs;

    defs = document.createElementNS(SVG_NS, "defs");
    defs.id = "foxOverlayDefs";

    const stripe = document.createElementNS(SVG_NS, "rect");
    stripe.setAttribute("class", "fx-mark");
    stripe.setAttribute("x", "0");
    stripe.setAttribute("y", "0");
    stripe.setAttribute("width", "4");
    stripe.setAttribute("height", "10");
    const lines = buildPattern("fox-fx-lines", 10, stripe);
    lines.setAttribute("patternTransform", "rotate(45)");
    defs.appendChild(lines);

    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("class", "fx-mark");
    dot.setAttribute("cx", "5");
    dot.setAttribute("cy", "5");
    dot.setAttribute("r", "2.5");
    defs.appendChild(buildPattern("fox-fx-dots", 11, dot));

    svg.insertBefore(defs, svg.firstChild);
    return defs;
}

const pointsAttr = (points: Point[]): string => points.map((p) => `${p.x},${p.y}`).join(" ");

// ------------------------------------------------------------- number badges

/** Where one entry's number badge landed, in overlay (image-pixel) units. */
export interface NumberBadge {
    num: number;
    x: number;
    y: number;
    w: number;
    h: number;
    fontSize: number;
}

interface PlacedRect {
    x: number;
    y: number;
    w: number;
    h: number;
}

/** The page size the overlay maps to, or `null` before the image loads. */
export function imageBounds(): { w: number; h: number } | null {
    try {
        const img = document.getElementById("mainImage") as HTMLImageElement | null;
        if (img && img.naturalWidth > 0 && img.naturalHeight > 0) {
            return { w: img.naturalWidth, h: img.naturalHeight };
        }
    } catch {
        /* tests or a torn-down page: corners are used unclamped */
    }
    return null;
}

function badgeSize(digits: number, refH: number): { w: number; h: number; fontSize: number } {
    const h = clamp(Math.round(refH), 16, 48);
    const fontSize = Math.max(10, Math.round(h * 0.58));
    const w = Math.round(fontSize * 0.62 * Math.max(1, digits) + h * 0.56);
    return { w, h, fontSize };
}

function overlaps(a: PlacedRect, b: PlacedRect, margin: number): boolean {
    return a.x - margin < b.x + b.w && a.x + a.w + margin > b.x && a.y - margin < b.y + b.h && a.y + a.h + margin > b.y;
}

/**
 * Badge spots centred on the outline's extreme vertices: above the two top
 * corners, below the two bottom ones, beside the left/right extremes. The
 * badge edge always carries its anchor vertex, so the number touches the
 * outline it belongs to instead of floating off a bbox corner.
 */
function vertexSpots(points: Point[], w: number, h: number): PlacedRect[] {
    const clean = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (!clean.length) return [];

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of clean) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    }

    const topXs = clean.filter((p) => p.y === minY).map((p) => p.x);
    const bottomXs = clean.filter((p) => p.y === maxY).map((p) => p.x);
    const leftYs = clean.filter((p) => p.x === minX).map((p) => p.y);
    const rightYs = clean.filter((p) => p.x === maxX).map((p) => p.y);

    // Each badge edge carries its anchor vertex: the bottom edge sits on a top
    // vertex, the top edge under a bottom one, and so on.
    const above = (vx: number): PlacedRect => ({ x: vx - w / 2, y: minY - h, w, h });
    const below = (vx: number): PlacedRect => ({ x: vx - w / 2, y: maxY, w, h });
    const leftOf = (vy: number): PlacedRect => ({ x: minX - w, y: vy - h / 2, w, h });
    const rightOf = (vy: number): PlacedRect => ({ x: maxX, y: vy - h / 2, w, h });

    const spots: PlacedRect[] = [];
    if (topXs.length) spots.push(above(Math.min(...topXs)), above(Math.max(...topXs)));
    if (bottomXs.length) spots.push(below(Math.min(...bottomXs)), below(Math.max(...bottomXs)));
    if (leftYs.length) spots.push(leftOf(Math.min(...leftYs)), leftOf(Math.max(...leftYs)));
    if (rightYs.length) spots.push(rightOf(Math.min(...rightYs)), rightOf(Math.max(...rightYs)));
    return spots;
}

/**
 * Pick a spot for every entry's number badge, in order.
 *
 * Badges sit *outside* their region -- above, below, or beside it, never
 * covering the artwork or text they number. Candidates must fit fully inside
 * the image (or at least not cross the origin when the image size is unknown),
 * skip corners other badges already claimed, and preferably avoid covering
 * neighbouring regions too. Because callers pass entries in *number* order,
 * badge 1 always wins the contested spot. Pure geometry after the image-size
 * lookup, so a full render plans once and every shape just stamps its badge
 * down. The last resort is a clamped top-left corner, which may overlap the
 * region on a postage-stamp page -- a badge somewhere beats none.
 */
export function planNumberBadges(
    items: { entry: Entry; num: number }[],
    alreadyPlaced: PlacedRect[] = [],
): Map<string, NumberBadge> {
    const bounds = imageBounds();
    const refH = bounds ? Math.min(bounds.w, bounds.h) * 0.04 : 0;
    const placed: PlacedRect[] = alreadyPlaced.map((r) => ({ ...r }));
    const boxes = new Map(items.map(({ entry }) => [entry.id, regionBBox(entry.region)]));
    const outlines = new Map(items.map(({ entry }) => [entry.id, regionPoints(entry.region)]));
    const out = new Map<string, NumberBadge>();

    const fitsImage = (x: number, y: number, w: number, h: number): boolean => {
        if (x < 0 || y < 0) return false;
        return !bounds || (x + w <= bounds.w && y + h <= bounds.h);
    };
    const coversRegion = (rect: PlacedRect, skipId: string): boolean => {
        for (const [id, box] of boxes) {
            if (id === skipId || box.w <= 0 || box.h <= 0) continue;
            if (overlaps(rect, box, 1)) return true;
        }
        return false;
    };

    for (const { entry, num } of items) {
        const box = boxes.get(entry.id) ?? { x: 0, y: 0, w: 0, h: 0 };
        if (![box.x, box.y, box.w, box.h].every((v) => Number.isFinite(v)) || box.w <= 0 || box.h <= 0) continue;

        const digits = String(num).length;
        const size = badgeSize(digits, refH > 0 ? refH : box.h * 0.5);
        // Anchored to outline vertices, not bbox corners: on a concave or
        // freehand shape a bbox corner can hover far from any ink, while a
        // badge edge centred on a real vertex touches the outline by
        // construction. Bbox spots follow as fallbacks.
        const spots = [
            ...vertexSpots(outlines.get(entry.id) ?? [], size.w, size.h),
            { x: box.x, y: box.y - size.h },
            { x: box.x + box.w - size.w, y: box.y - size.h },
            { x: box.x, y: box.y + box.h },
            { x: box.x + box.w - size.w, y: box.y + box.h },
            { x: box.x - size.w, y: box.y },
            { x: box.x + box.w, y: box.y },
            { x: box.x - size.w, y: box.y + box.h - size.h },
            { x: box.x + box.w, y: box.y + box.h - size.h },
        ];

        let chosen: PlacedRect | null = null;
        // First pass dodges neighbouring regions as well as badges; the second
        // accepts covering artwork rather than dropping the number.
        for (let pass = 0; pass < 2 && !chosen; pass++) {
            for (const spot of spots) {
                const rect = { x: spot.x, y: spot.y, w: size.w, h: size.h };
                if (!fitsImage(rect.x, rect.y, rect.w, rect.h)) continue;
                if (placed.some((p) => overlaps(p, rect, 3))) continue;
                if (pass === 0 && coversRegion(rect, entry.id)) continue;
                chosen = rect;
                break;
            }
        }
        if (!chosen && bounds) {
            chosen = {
                x: clamp(box.x, 0, Math.max(0, bounds.w - size.w)),
                y: clamp(box.y, 0, Math.max(0, bounds.h - size.h)),
                w: size.w,
                h: size.h,
            };
        }
        if (!chosen) continue;

        placed.push(chosen);
        out.set(entry.id, { num, ...chosen, fontSize: size.fontSize });
    }
    return out;
}

/** Badge rects already on screen, so a single-shape refresh can dodge them. */
export function collectPlacedBadges(svg: SVGSVGElement): PlacedRect[] {
    const out: PlacedRect[] = [];
    try {
        svg.querySelectorAll("g.entry-number[data-badge]").forEach((node) => {
            const parts = (node.getAttribute("data-badge") ?? "").split(",").map(Number);
            if (parts.length === 4 && parts.every((v) => Number.isFinite(v))) {
                out.push({ x: parts[0], y: parts[1], w: parts[2], h: parts[3] });
            }
        });
    } catch {
        /* a foreign svg: no dodging, corners still apply */
    }
    return out;
}

function buildNumberBadge(group: SVGGElement, entry: Entry, badge: NumberBadge): void {
    const tag = document.createElementNS(SVG_NS, "g") as SVGGElement;
    tag.setAttribute("class", "entry-number");
    tag.setAttribute("pointer-events", "none");
    tag.setAttribute("data-badge", `${badge.x},${badge.y},${badge.w},${badge.h}`);

    const plate = document.createElementNS(SVG_NS, "rect");
    plate.setAttribute("x", String(badge.x));
    plate.setAttribute("y", String(badge.y));
    plate.setAttribute("width", String(badge.w));
    plate.setAttribute("height", String(badge.h));
    plate.setAttribute("rx", String(Math.round(badge.h * 0.35)));
    plate.setAttribute("fill", entry.color);
    plate.setAttribute("stroke", "rgba(0,0,0,0.35)");
    plate.setAttribute("stroke-width", String(Math.max(1, Math.round(badge.h * 0.045))));
    tag.appendChild(plate);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", String(badge.x + badge.w / 2));
    label.setAttribute("y", String(badge.y + badge.h / 2));
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("dominant-baseline", "central");
    label.setAttribute("font-size", String(badge.fontSize));
    label.setAttribute("font-weight", "700");
    label.setAttribute("font-family", "sans-serif");
    label.setAttribute("fill", contrastColor(entry.color));
    label.textContent = String(badge.num);
    tag.appendChild(label);

    group.appendChild(tag);
}

// ----------------------------------------------------------------------- clean FX

/**
 * The text-clean indicator: a blurred, clipped copy of the artwork under the
 * region, plus a hatch that identifies which detector will decide the shape.
 *
 * The user asked not to fetch PaddleOCR or Text Seg masks for the preview, so
 * this deliberately does *not* pretend to know what will be erased -- it marks
 * the region and says which method owns it. Region gets a plain blur, PaddleOCR
 * diagonal stripes, Text Seg dots.
 */
function buildCleanFx(entry: Entry, points: Point[], svg: SVGSVGElement, group: SVGGElement): void {
    const box = regionBBox(entry.region);
    if (box.w <= 0 || box.h <= 0) return;

    const defs = ensureOverlayDefs(svg);
    const clipId = `fxclip_${entry.id}`;
    defs.querySelector(`#${cssEscape(clipId)}`)?.remove();
    const clip = document.createElementNS(SVG_NS, "clipPath");
    clip.id = clipId;
    clip.setAttribute("clipPathUnits", "userSpaceOnUse");
    const clipShape = document.createElementNS(SVG_NS, "polygon");
    clipShape.setAttribute("points", pointsAttr(points));
    clip.appendChild(clipShape);
    defs.appendChild(clip);

    const method = entry.clean?.method || "region";
    const variant = method === "ppocr" ? "is-ppocr" : method === "textseg" ? "is-textseg" : "is-region";

    // `backdrop-filter` inside a foreignObject blurs whatever is painted behind
    // it -- the page image -- which is the cheapest honest "something happens
    // here" without touching pixel data.
    const fo = document.createElementNS(SVG_NS, "foreignObject");
    fo.setAttribute("x", String(box.x));
    fo.setAttribute("y", String(box.y));
    fo.setAttribute("width", String(box.w));
    fo.setAttribute("height", String(box.h));
    fo.setAttribute("clip-path", `url(#${clipId})`);
    fo.setAttribute("pointer-events", "none");
    const blur = document.createElementNS(XHTML_NS, "div") as unknown as HTMLDivElement;
    blur.setAttribute("class", `clean-fx ${variant}`);
    fo.appendChild(blur);
    group.appendChild(fo);

    const plate = document.createElementNS(SVG_NS, "polygon");
    plate.setAttribute("class", "clean-plate");
    plate.setAttribute("points", pointsAttr(points));
    group.appendChild(plate);

    if (method !== "region") {
        const marks = document.createElementNS(SVG_NS, "polygon");
        marks.setAttribute("class", "clean-marks");
        marks.setAttribute("points", pointsAttr(points));
        marks.setAttribute("fill", `url(#${method === "ppocr" ? "fox-fx-lines" : "fox-fx-dots"})`);
        group.appendChild(marks);
    }
}

// -------------------------------------------------------------------- entry shape

/**
 * Build (and insert) the overlay group for one entry.
 *
 * Returns the group, or `null` if the region is degenerate. Every child carries
 * an explicit `fill`, because the overlay's `pointer-events:none` and the
 * `.svg-overlay-polygon` tint both inherit and would otherwise wash the preview.
 */
export function buildEntryShape(
    entry: Entry,
    svg: SVGSVGElement,
    handlers: ShapeHandlers,
    badge?: NumberBadge,
): SVGGElement | null {
    const points = regionPoints(entry.region);
    if (points.length < 3) return null;
    const box = regionBBox(entry.region);
    if (box.w <= 0 || box.h <= 0) return null;

    const group = document.createElementNS(SVG_NS, "g") as SVGGElement;
    group.id = `shape_${entry.id}`;
    // Neither pass has produced text, so whatever plate this entry paints is
    // hiding artwork the user still needs to read -- `.is-empty` fades it in CSS
    // rather than here, which leaves the outline and the hit target untouched.
    const empty = !(entry.ocr_text || "").trim() && !(entry.text || "").trim();
    group.setAttribute("class", `entry-shape interactive-inpaint-group${empty ? " is-empty" : ""}`);
    group.dataset.layer = String(entry.layer ?? 1);
    group.dataset.entry = entry.id;

    if (entry.bg_mode === "clean") buildCleanFx(entry, points, svg, group);

    const plate = plateColor(entry);
    if (plate) {
        const bg = document.createElementNS(SVG_NS, "polygon");
        bg.setAttribute("class", "entry-plate");
        bg.setAttribute("points", pointsAttr(points));
        bg.setAttribute("fill", plate);
        bg.setAttribute("stroke", plate);
        bg.setAttribute("stroke-width", "1");
        group.appendChild(bg);
    }

    let hasText = false;
    const body = (entry.text || "").trim();
    if (body && state.isLiveInpaintedEnabled) {
        const area = typeArea(box);
        const geom = readGeometry(entry);
        const layout = fitText(svg, entry.text, entry.fontname, entry.text_align, area, entry.font_size, geom);
        if (layout) {
            const reference = plate ?? AUTO_PLATE;
            const strokeWidth =
                entry.stroke_width === null
                    ? autoStrokeWidth(layout.size)
                    : clamp(Math.round(entry.stroke_width), MIN_STROKE_WIDTH, MAX_STROKE_WIDTH);
            const text = document.createElementNS(SVG_NS, "text") as SVGTextElement;
            text.setAttribute("class", "entry-text");
            const drawStyle = {
                fill: entry.font_color || contrastColor(reference),
                stroke: entry.stroke_color || reference,
                strokeWidth,
                family: entry.fontname,
            };
            // The transform lives on the text node alone, so the plate polygon
            // above is untouched -- "move the text, not the background".
            //
            // A true tilt is CSS 3D (`tiltStyle`): perspective + rotateX/Y about
            // the block's own centre, so the near edge widens like the backend's
            // pinhole projection. CSS and the `transform` attribute are the same
            // property, so they cannot share one node -- the 2D part
            // (stretch/spin/shift, without the fallback squash) goes on an outer
            // `<g>`, the perspective on the inner `<text>`. Without CSS 3D the
            // style is `""` and the single node keeps the `cos` fallback inside
            // `textTransform`, which is also what jsdom-based tests exercise.
            const tiltCss = tiltStyle(layout, entry.text_align, area, geom, svg);
            if (tiltCss) {
                drawFitLayout(text, layout, entry.text_align, area, drawStyle, "");
                text.setAttribute("style", tiltCss);
                const outer = textTransform(layout, entry.text_align, area, geom, strokeWidth, imageBounds(), {
                    includeTilt: false,
                });
                if (outer) {
                    const wrap = document.createElementNS(SVG_NS, "g");
                    wrap.setAttribute("transform", outer);
                    wrap.setAttribute("pointer-events", "none");
                    wrap.appendChild(text);
                    group.appendChild(wrap);
                } else {
                    group.appendChild(text);
                }
            } else {
                drawFitLayout(
                    text,
                    layout,
                    entry.text_align,
                    area,
                    drawStyle,
                    textTransform(layout, entry.text_align, area, geom, strokeWidth, imageBounds()),
                );
                group.appendChild(text);
            }
            hasText = true;
        }
    }

    // The outline is last so its dashes stay visible, and it is the hit target
    // for the whole region (`pointer-events: all` catches the interior even
    // though it is unfilled -- see `.entry-outline` in style.css).
    //
    // fill/stroke go in an inline *style*, not a presentation attribute:
    // `.svg-overlay-polygon { fill: var(--svg-fill) }` is a CSS rule and would
    // win against an attribute, washing every plate with a grey veil. The hover
    // rules still work, because a stylesheet `!important` beats inline style.
    const covered = hasText || Boolean(plate) || entry.bg_mode === "clean";
    const outline = document.createElementNS(SVG_NS, "polygon");
    outline.setAttribute("class", `entry-outline svg-overlay-polygon${covered ? " no-hover-fill" : ""}`);
    outline.setAttribute("points", pointsAttr(points));
    outline.setAttribute("style", `fill:${covered ? "none" : `${entry.color}25`};stroke:${entry.color};`);
    group.appendChild(outline);

    // The reading number rides on the group, so it hides, hovers and deletes
    // with its region. Planned by the caller in number order; a lone caller
    // gets a solo plan, parked just outside its region.
    const tag = badge ?? planNumberBadges([{ entry, num: 1 }]).get(entry.id);
    if (tag) buildNumberBadge(group, entry, tag);

    group.addEventListener("click", (event) => {
        handlers.onSelect(entry);
        if (hasText) {
            event.stopPropagation();
            handlers.onEditInPlace(entry, group);
        }
    });
    group.addEventListener("mouseenter", () => handlers.onHover(entry, true));
    group.addEventListener("mouseleave", () => handlers.onHover(entry, false));
    group.addEventListener("contextmenu", (event) => {
        const onContextMenu = handlers.onContextMenu;
        if (!onContextMenu) return;
        // Not stopped, only defaulted away: `#nocontextmenu` up in main.tsx is what
        // suppresses the browser menu for the whole app, and it has to keep seeing
        // this event to do that.
        event.preventDefault();
        onContextMenu(entry);
    });

    insertShape(svg, group, entry.layer ?? 1);
    return group;
}

/**
 * Insert `group` among its siblings so document order matches layer order.
 *
 * SVG has no z-index, so "layer" *is* document order. Re-sorting the whole
 * overlay for one changed entry would rebuild every shape; finding the first
 * sibling on a higher layer costs one pass over a handful of nodes.
 */
export function insertShape(svg: SVGSVGElement, group: SVGElement, layer: number): void {
    const siblings = Array.from(svg.children) as SVGElement[];
    for (const sibling of siblings) {
        if (!sibling.classList?.contains("entry-shape")) continue;
        if (Number(sibling.dataset?.layer ?? "1") > layer) {
            svg.insertBefore(group, sibling);
            return;
        }
    }
    svg.appendChild(group);
}

/** Drop an entry's group and the clip path that belonged to it. */
export function removeEntryShape(entryId: string): void {
    document.getElementById(`shape_${entryId}`)?.remove();
    document.getElementById(`fxclip_${entryId}`)?.remove();
}

/** Toggle the hover wash / thick outline on an entry's region. */
export function setShapeHover(entryId: string, hovering: boolean): void {
    const outline = document.querySelector(`#shape_${cssEscape(entryId)} .entry-outline`);
    outline?.classList.toggle("active-hover-region", hovering);
}
