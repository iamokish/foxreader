/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * The entry font picker's height contract, as static CSS text: jsdom does no
 * layout, so equality is asserted on the declared boxes instead.
 *
 * The menu must read as the trigger's continuation -- each row exactly the
 * trigger's height -- and the row restyle must stay scoped to the entry
 * popover, so the main font menu keeps its roomier rows.
 */

const css = readFileSync(resolve(process.cwd(), "static/style.css"), "utf8").replace(
    /\/\*[\s\S]*?\*\//g,
    "",
);

/** Declarations of the first rule whose selector list contains `selector`. */
function bodyOfIn(cssText: string, selector: string): string {
    for (const chunk of cssText.split("}")) {
        const brace = chunk.indexOf("{");
        if (brace === -1) continue;
        const selectors = chunk
            .slice(0, brace)
            .split(",")
            .map((part) => part.trim());
        if (selectors.includes(selector)) return chunk.slice(brace + 1);
    }
    throw new Error(`no rule for ${selector}`);
}

function bodyOf(selector: string): string {
    return bodyOfIn(css, selector);
}

/** Value of `property` in a rule body, or null. */
function valueOf(body: string, property: string): string | null {
    const match = body.match(new RegExp(`${property}\\s*:\\s*([^;]+);`));
    return match ? match[1].trim() : null;
}

describe("entry font popover heights", () => {
    it("matches each row to the trigger bar height", () => {
        // The bar's own box (`.eo-select` carries it); compared, not hardcoded,
        // so both sides may evolve as long as they stay equal.
        const triggerHeight = valueOf(bodyOf(".eo-select"), "height");
        const rowHeight = valueOf(bodyOf(".eo-font-pop .font-option-item"), "height");
        expect(triggerHeight).not.toBeNull();
        expect(rowHeight).toBe(triggerHeight);
    });

    it("pins both boxes to border-box, whatever the UA button model is", () => {
        expect(valueOf(bodyOf(".eo-font-trigger"), "box-sizing")).toBe("border-box");
        expect(valueOf(bodyOf(".eo-font-pop .font-option-item"), "box-sizing")).toBe("border-box");
    });
});

describe("action row gap below the wrapper", () => {
    it("matches the wrapper's own vertical rhythm", () => {
        // Compared, not hardcoded, so both may evolve as long as they stay equal.
        const above = valueOf(bodyOf(".translation-wrapper"), "margin-top");
        const below = valueOf(bodyOf(".content-area > #confirmActionRow"), "margin-top");
        expect(above).not.toBeNull();
        expect(below).toBe(above);
    });

    it("cannot reach the Vulpes footer, which nests the same row elsewhere", () => {
        // `.content-area` exists solely in the Basic layout; and no bare
        // `#confirmActionRow` rule may carry a margin of its own.
        for (const chunk of css.split("}")) {
            const brace = chunk.indexOf("{");
            if (brace === -1) continue;
            const selectors = chunk
                .slice(0, brace)
                .split(",")
                .map((part) => part.trim());
            if (selectors.includes("#confirmActionRow")) {
                expect(valueOf(chunk.slice(brace + 1), "margin-top")).toBeNull();
                expect(valueOf(chunk.slice(brace + 1), "margin-bottom")).toBeNull();
            }
        }
    });
});

describe("bottom gap below the action row", () => {
    it("leaves exactly 10px between the row and the screen bottom", () => {
        // The row is the column's last child, so the column's bottom padding
        // is the gap. Locked whole: top and sides must not drift with it.
        expect(valueOf(bodyOf(".content-area"), "padding")).toBe("10px 20px 10px 20px");
    });
});

describe("highlighted entry card", () => {
    it("keeps the golden outline but carries no shadow", () => {
        const body = bodyOf(".saved-entry-card.highlighted-entry");
        expect(body).toContain("2px solid var(--accent-gold)");
        expect(valueOf(body, "box-shadow")).toBe("none");
        // The golden glow this replaced.
        expect(body).not.toContain("rgba(255, 209, 102");
    });

    it("stays flat when hovered, against the Vulpes hover elevation", () => {
        expect(valueOf(bodyOf(".saved-entry-card.highlighted-entry:hover"), "box-shadow")).toBe("none");
    });

    it("exempts hovered selections where the Vulpes elevation lives", () => {
        // The layout's own `body.layout-vulpes .saved-entry-card:hover` outranks
        // the shared rule on specificity (`body` breaks the tie), so the
        // exemption has to live beside it -- assert it is there, flat.
        const vulpes = readFileSync(resolve(process.cwd(), "src/layouts/vulpes.css"), "utf8").replace(
            /\/\*[\s\S]*?\*\//g,
            "",
        );
        const body = bodyOfIn(vulpes, "body.layout-vulpes .saved-entry-card.highlighted-entry:hover");
        expect(valueOf(body, "box-shadow")).toBe("none");
    });
});

describe("entry font popover scoping", () => {
    it("keeps the compact rows scoped to the entry popover", () => {
        // No bare `.font-option-item` height rule: the main menu's rows stay tall.
        for (const chunk of css.split("}")) {
            const brace = chunk.indexOf("{");
            if (brace === -1) continue;
            const selectors = chunk
                .slice(0, brace)
                .split(",")
                .map((part) => part.trim());
            if (selectors.includes(".font-option-item")) {
                expect(valueOf(chunk.slice(brace + 1), "height")).toBeNull();
            }
        }
    });
});
