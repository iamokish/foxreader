/**
 * Region geometry shared by the entry list, the overlay and the split flow.
 *
 * Everything here works on the two shapes an entry can have -- a rectangle
 * `{x,y,w,h}` or a polygon point list -- and normalises them to a point list so
 * the rest of the code never has to branch on `region.type` again. That is what
 * makes rectangles first-class in the splitter: the backend already accepts
 * either (`Region.to_polygon`), so the frontend only had to stop refusing them.
 */

import type { Point, RectangleCoords, Region } from "../types";

export interface BBox {
    x: number;
    y: number;
    w: number;
    h: number;
}

/**
 * `id` made safe to drop into a selector.
 *
 * `CSS.escape` is the right tool but not always there: jsdom leaves the `CSS`
 * global undefined, so calling it directly turns a unit test into a TypeError
 * rather than a lookup. The fallback escapes what an entry id can actually
 * contain -- anything outside `[A-Za-z0-9_-]` -- which is a superset of what
 * these generated ids ever hold.
 */
export function cssEscape(id: string): string {
    const text = String(id);

    if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
        return CSS.escape(text);
    }

    return text.replace(/[^\w-]/g, (ch) => `\\${ch}`);
}

/** A region as a closed point list. Rectangles become their four corners. */
export function regionPoints(region: Region | null): Point[] {
    if (!region) return [];
    if (region.type === "rectangle") {
        const r = region.coords as RectangleCoords;
        const x0 = Math.min(r.x, r.x + r.w);
        const x1 = Math.max(r.x, r.x + r.w);
        const y0 = Math.min(r.y, r.y + r.h);
        const y1 = Math.max(r.y, r.y + r.h);
        return [
            { x: x0, y: y0 },
            { x: x1, y: y0 },
            { x: x1, y: y1 },
            { x: x0, y: y1 },
        ];
    }
    const points = region.coords as Point[];
    return Array.isArray(points) ? points : [];
}

/** The same list the backend receives, as `[x, y]` pairs. */
export function regionPairs(region: Region | null): [number, number][] {
    return regionPoints(region).map((p) => [Math.round(p.x), Math.round(p.y)] as [number, number]);
}

export function bboxOf(points: Point[]): BBox {
    if (!points.length) return { x: 0, y: 0, w: 0, h: 0 };
    let minX = points[0].x;
    let maxX = points[0].x;
    let minY = points[0].y;
    let maxY = points[0].y;
    // A hand loop rather than Math.min(...xs): a freehand cut can carry a few
    // thousand points and spreading that into apply() is what blows the stack.
    for (let i = 1; i < points.length; i++) {
        const p = points[i];
        if (p.x < minX) minX = p.x;
        else if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        else if (p.y > maxY) maxY = p.y;
    }
    return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

export function regionBBox(region: Region | null): BBox {
    return bboxOf(regionPoints(region));
}

/** Shoelace area. Sign-agnostic, so winding order does not matter. */
export function polygonArea(points: Point[]): number {
    const n = points.length;
    if (n < 3) return 0;
    let sum = 0;
    for (let i = 0; i < n; i++) {
        const a = points[i];
        const b = points[(i + 1) % n];
        sum += a.x * b.y - b.x * a.y;
    }
    return Math.abs(sum) / 2;
}

/** Exact equality -- used to recognise a region the page already has. */
export function isSameRegion(a: Region | null, b: Region | null): boolean {
    if (a === b) return true;
    if (!a || !b) return false;
    if (a.type === "rectangle" && b.type === "rectangle") {
        const c1 = a.coords as RectangleCoords;
        const c2 = b.coords as RectangleCoords;
        return c1.x === c2.x && c1.y === c2.y && c1.w === c2.w && c1.h === c2.h;
    }
    if (a.type !== b.type) return false;
    const p1 = a.coords as Point[];
    const p2 = b.coords as Point[];
    if (!Array.isArray(p1) || !Array.isArray(p2) || p1.length !== p2.length) return false;
    for (let i = 0; i < p1.length; i++) {
        if (p1[i].x !== p2[i].x || p1[i].y !== p2[i].y) return false;
    }
    return true;
}

/**
 * Whether two regions are the same region for practical purposes.
 *
 * The split guard needs this and not `isSameRegion`: the backend rebuilds each
 * piece from a rasterised mask and simplifies its contour, so a cut that failed
 * to divide anything comes back as a polygon with a different point count and a
 * pixel or two of drift -- exactly equal to nothing, visually identical to the
 * source. Comparing area ratio *and* bounding box catches that without rejecting
 * a genuine piece, which always loses real area.
 */
export function nearlyIdentical(a: Region | null, b: Region | null): boolean {
    const pa = regionPoints(a);
    const pb = regionPoints(b);
    if (pa.length < 3 || pb.length < 3) return false;

    const areaA = polygonArea(pa);
    const areaB = polygonArea(pb);
    if (areaA <= 0 || areaB <= 0) return false;
    if (Math.min(areaA, areaB) / Math.max(areaA, areaB) < 0.97) return false;

    const ba = bboxOf(pa);
    const bb = bboxOf(pb);
    const tol = Math.max(2, 0.02 * Math.max(ba.w, ba.h, bb.w, bb.h));
    return (
        Math.abs(ba.x - bb.x) <= tol &&
        Math.abs(ba.y - bb.y) <= tol &&
        Math.abs(ba.w - bb.w) <= tol &&
        Math.abs(ba.h - bb.h) <= tol
    );
}

/** Below this a "piece" is rasterisation debris, not something to letter. */
export const MIN_PIECE_AREA = 24;

export function isUsablePiece(region: Region | null): boolean {
    const points = regionPoints(region);
    if (points.length < 3) return false;
    const box = bboxOf(points);
    if (box.w < 2 || box.h < 2) return false;
    return polygonArea(points) >= MIN_PIECE_AREA;
}
