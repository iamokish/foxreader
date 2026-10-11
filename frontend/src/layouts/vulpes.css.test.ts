/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const layout = readFileSync(resolve(process.cwd(), "src/layouts/vulpes.css"), "utf8");
const palette = readFileSync(resolve(process.cwd(), "src/themes/vulpes.css"), "utf8");
const dark = readFileSync(resolve(process.cwd(), "src/themes/dark.css"), "utf8");
const strip = readFileSync(resolve(process.cwd(), "src/workspace/strip.css"), "utf8");
const reader = readFileSync(resolve(process.cwd(), "src/reader/reader.css"), "utf8");
const actionRow = readFileSync(resolve(process.cwd(), "src/components/Translation/actionRow.css"), "utf8");

/** Comments hold prose about selectors and `!important`; they are not rules. */
function withoutComments(css: string): string {
    return css.replace(/\/\*[\s\S]*?\*\//g, "");
}

interface Rule {
    /** The selector text, or the at-rule prelude for `@media` and friends. */
    selector: string;
    /** The declarations, excluding any nested blocks. */
    body: string;
    /** 0 for a top-level rule, 1 for one inside an at-rule. */
    depth: number;
}

/**
 * The rules of a stylesheet, flat.
 *
 * A brace walker rather than a regex: a regex for "selector, then body" runs past
 * the closing brace and happily matches a declaration several rules later, which
 * is exactly the false pass these tests exist to avoid. Nesting beyond one level
 * (a rule inside `@media`) does not occur in this project's CSS.
 */
function rulesOf(css: string): Rule[] {
    const rules: Rule[] = [];
    const clean = withoutComments(css);
    const stack: { selector: string; depth: number; start: number }[] = [];
    let buffer = "";

    for (let i = 0; i < clean.length; i++) {
        const char = clean[i];
        if (char === "{") {
            stack.push({ selector: buffer.trim(), depth: stack.length, start: i + 1 });
            buffer = "";
        } else if (char === "}") {
            const open = stack.pop();
            if (open) rules.push({ selector: open.selector, body: clean.slice(open.start, i), depth: open.depth });
            buffer = "";
        } else {
            buffer += char;
        }
    }
    return rules;
}

/** Every custom property a stylesheet declares, e.g. `--bg-primary`. */
function declaredTokens(css: string): string[] {
    return [...withoutComments(css).matchAll(/(--[a-z0-9-]+)\s*:/g)].map((match) => match[1]);
}

describe("Vulpes palette", () => {
    it("restates every token the base dark theme declares", () => {
        // A token left out would fall through to the base palette and leave one
        // component half-themed, which is the failure mode this guards.
        const required = new Set(declaredTokens(dark));
        const provided = new Set(declaredTokens(palette));
        const missing = [...required].filter((token) => !provided.has(token));
        expect(missing).toEqual([]);
    });

    it("rides on the body, so it beats :root and html.theme-light without !important", () => {
        const selectors = rulesOf(palette).map((rule) => rule.selector);
        expect(selectors).toContain("body.layout-vulpes");
        expect(selectors).toContain("html.theme-light body.layout-vulpes");
    });

    it("covers the light side with the colours that have to differ", () => {
        const light = rulesOf(palette).find((rule) => rule.selector === "html.theme-light body.layout-vulpes");
        expect(light).toBeDefined();
        for (const token of ["--bg-primary", "--text-primary", "--border-default", "--vx-accent"]) {
            expect(light!.body).toContain(`${token}:`);
        }
    });

    it("needs no !important to win", () => {
        for (const rule of rulesOf(palette)) {
            expect(rule.body).not.toContain("!important");
        }
    });
});

describe("Vulpes layout stylesheet", () => {
    const rules = rulesOf(layout);

    it("scopes every rule to the layout, so the other two are untouched", () => {
        // Anything at the top level would leak into Basic and Default.
        const unscoped = rules
            .filter((rule) => !rule.selector.startsWith("@"))
            .flatMap((rule) => rule.selector.split(","))
            .map((part) => part.trim())
            .filter((part) => part.length > 0 && !part.startsWith("body.layout-vulpes"));
        expect(unscoped).toEqual([]);
    });

    it("overrides the legacy 80vh image-container height with the stage's own", () => {
        // The same trap the Default workspace had to step around: `static/style.css`
        // pins `#imageContainer` to 80vh, which would push the captured-text section
        // off the bottom of the window.
        const stage = rules.find((rule) => rule.selector.includes(".vx-stage > #imageContainer"));
        expect(stage?.body).toMatch(/height:\s*100%/);
    });

    it("stands the gallery up as a column for the pages rail", () => {
        const rail = rules.find((rule) => rule.selector.endsWith(".vx-rail .gallery-bar"));
        expect(rail?.body).toMatch(/flex-direction:\s*column/);
    });

    it("hides the workspace chrome in reader mode", () => {
        const hidden = rules.find(
            (rule) => rule.selector.includes(".workspace-reader") && /display:\s*none/.test(rule.body),
        );
        expect(hidden?.selector).toContain(".vx-topbar");
        expect(hidden?.selector).toContain(".vx-editor");
    });

    it("never hides the editor's viewport or the gallery outright", () => {
        // `#imageContainer` must stay laid out: the strip builds inside it, and the
        // editor's pan/zoom listeners were bound to it once at startup. `#gallery`
        // must stay laid out because `navigateGallery` reads its thumbnails.
        const hiddenCritical = rules.filter(
            (rule) => /#imageContainer|#gallery\b/.test(rule.selector) && /display:\s*none/.test(rule.body),
        );
        expect(hiddenCritical.map((rule) => rule.selector)).toEqual([]);
    });

    it("hides the floating Original/Saved switch, which the footer replaces", () => {
        // Basic keeps it: there it is the only copy. Mounted but never painted
        // here -- and `!important`, to outrank the inline `display` the switch
        // writer sets when a saved copy exists.
        const rule = rules.find((rule) => rule.selector === "body.layout-vulpes #variantSwitch");
        expect(rule?.body).toMatch(/display:\s*none\s*!important/);
    });

    it("honours a reduced-motion preference", () => {
        expect(rules.some((rule) => rule.selector.includes("prefers-reduced-motion"))).toBe(true);
    });

    it("keeps the Translators header at section height, however many buttons", () => {
        // The bar itself stays pinned; only the rail inside it scrolls.
        const prefix = "body.layout-vulpes .vx-editor__translation > .vx-section__head";
        const base = rules.find((rule) => rule.selector === "body.layout-vulpes .vx-section__head");
        expect(base?.body).toMatch(/height:\s*var\(--vx-section-h\)/);
        // The rail never wraps (which would spill past the fixed bar) and
        // scrolls invisibly instead.
        const rail = rules.find((rule) => rule.selector === `${prefix} .translator-buttons`);
        expect(rail?.body).toMatch(/flex-wrap:\s*nowrap/);
        expect(rail?.body).toMatch(/overflow-x:\s*auto/);
        // Header-tall, unsquashable buttons: the base button is 38px and would
        // spill out of the 30px bar.
        const buttons = rules.find((rule) => rule.selector === `${prefix} .translator-buttons button`);
        expect(buttons?.body).toMatch(/height:\s*22px/);
        expect(buttons?.body).toMatch(/flex:\s*0 0 auto/);
    });
});

describe("Vulpes translation region", () => {
    const rules = rulesOf(layout);
    // Exact selectors: the shared block's own selector list *ends* with
    // `.panel.error`, so an `endsWith` match would find that one for both.
    const shared = rules.find(
        (rule) =>
            rule.selector.includes("#translationContainer .panel,") &&
            rule.selector.includes("#translationContainer .panel.error"),
    );
    const placeholder = rules.find((rule) => rule.selector === "body.layout-vulpes #translationContainer .panel.error");

    it("lets the translation panel fill the region instead of hugging its text", () => {
        // `#translationContainer` carries an inline `justify-content: center` from
        // `TranslationRegion`, which a stylesheet cannot override -- so the panel has
        // to *grow*, which leaves centring no free space to distribute. Without this
        // `#translatedText` is a thin strip floating in the middle of the region.
        expect(shared?.body).toMatch(/flex:\s*1/);
        expect(shared?.body).toMatch(/min-height:\s*0/);
    });

    it("keeps the placeholder the same size as a real translation", () => {
        // Otherwise the region changes shape the moment a translation arrives.
        expect(placeholder?.body).toMatch(/justify-content:\s*center/);
        expect(placeholder?.body).not.toMatch(/width:\s*auto/);
    });
});

describe("Vulpes rendering cost", () => {
    const rules = rulesOf(layout);

    it("declares no inherited custom property for a panel size", () => {
        // Sizes are an inline style on the element, written once per animation
        // frame during a drag; a `--vx-size-*` custom property on `<body>` would
        // re-resolve every dependent on every pointer move instead.
        for (const rule of rules) {
            expect(rule.body).not.toContain("--vx-size");
        }
    });

    it("gives each resizable section a default size for the first paint", () => {
        // The inline style arrives from JSX, so this is only the fallback for a
        // frame -- but a section with no size at all would collapse to its content.
        const sized: [string, "width" | "height"][] = [
            [".vx-rail", "width"],
            [".vx-editor", "width"],
            [".vx-capture", "height"],
            ['[data-section="entries"]', "height"],
        ];
        for (const [selector, axis] of sized) {
            // Double backslashes: this is a template literal, so `\s` would
            // collapse to `s` and the expression would never match anything.
            const rule = rules.find(
                (candidate) =>
                    candidate.selector.endsWith(selector) && new RegExp(`${axis}:\\s*\\d+px`).test(candidate.body),
            );
            expect(rule, `${selector} should declare a default ${axis}`).toBeDefined();
        }
    });

    it("does not force a looping animation to restart every frame under reduced motion", () => {
        // The canonical reduced-motion snippet pairs `animation-duration: 0.01ms`
        // with `animation-iteration-count: 1`. Shipping only the first half makes an
        // `infinite` animation restart about a hundred thousand times a second, and
        // each frame samples a different phase -- so every looping animation in the
        // layout vibrates. That is what it did.
        const reduced = rules.filter((rule) => rule.selector.includes("prefers-reduced-motion"));
        expect(reduced.length).toBeGreaterThan(0);
        for (const rule of rules) {
            if (!/animation-duration/.test(rule.body)) continue;
            expect(rule.body).toMatch(/animation-iteration-count:\s*1/);
        }
    });

    it("styles scrollbars on named containers, never with a universal selector", () => {
        // Two reasons. A universal pseudo-element selector is matched against every
        // element in the layout; and styling `::-webkit-scrollbar` at all opts an
        // element out of the platform's overlay scrollbars into classic ones that
        // take layout space. `#imageContainer` must keep the platform default,
        // because `syncContainerOverflow` flips its `overflow` on every zoom step
        // and a space-taking scrollbar turns each flip into a visible jump.
        for (const rule of rules) {
            if (!rule.selector.includes("::-webkit-scrollbar")) continue;
            expect(rule.selector).not.toContain("*::-webkit-scrollbar");
            expect(rule.selector).not.toContain("#imageContainer");
        }
    });

    it("uses no backdrop-filter in the layout chrome", () => {
        // A backdrop blur re-computes from whatever is behind it, every frame. None
        // of this layout's chrome sits over moving artwork, so none of it has an
        // excuse for one. (The reader's edge veil does -- see the reader stylesheet,
        // where it is gated instead of banned.)
        for (const rule of rules) {
            expect(rule.body).not.toContain("backdrop-filter");
        }
    });

    it("skips off-screen thumbnails in the pages rail", () => {
        // A folder can run to hundreds of pages; laying all of them out on every rail
        // resize is work for pages nobody is looking at.
        const thumb = rules.find((rule) => rule.selector.endsWith(".vx-rail .gallery-item-container"));
        expect(thumb?.body).toMatch(/content-visibility:\s*auto/);
        expect(thumb?.body).toMatch(/contain-intrinsic-size:/);
    });
});

describe("strip stylesheet", () => {
    const rules = rulesOf(strip);

    it("is unscoped, so any layout with a reader can use the one renderer", () => {
        expect(rules.map((rule) => rule.selector)).toContain("#strip-scroller");
        for (const rule of rules) {
            expect(rule.selector).not.toContain("body.layout-");
        }
    });

    it("spaces pages with an inherited custom property, defaulting to no gap", () => {
        // Inherited because `buildStrip` rebuilds the scroller on every view change.
        const gap = rules.find((rule) => rule.selector === ".strip-page + .strip-page");
        expect(gap?.body).toMatch(/margin-top:\s*var\(--strip-gap,\s*0px\)/);
    });
});

describe("reader stylesheet", () => {
    const rules = rulesOf(reader);

    it("is unscoped, so Basic and Vulpes get the same reader", () => {
        // The reader is mounted by both layouts. A `body.layout-*` prefix anywhere in
        // here would mean one of them silently loses that rule.
        const scoped = rules
            .filter((rule) => !rule.selector.startsWith("@"))
            .flatMap((rule) => rule.selector.split(","))
            .map((part) => part.trim())
            .filter((part) => part.includes("body.layout-") && !part.startsWith("body.layout-vulpes {"));
        // The one permitted exception is the on-accent pairing, which genuinely
        // differs: Vulpes fills with ember, the base themes with gold.
        expect(scoped.filter((part) => part !== "body.layout-vulpes")).toEqual([]);
    });

    it("gives every --vx-* token it uses a base value, so Basic is not unstyled", () => {
        // Those tokens are declared by `themes/vulpes.css` under `body.layout-vulpes`.
        // Outside that layout they would resolve to nothing, and every rule using one
        // would quietly lose its colour or its spacing.
        const root = rules.find((rule) => rule.selector === ":root");
        expect(root).toBeDefined();
        const used = new Set([...reader.matchAll(/var\((--vx-[a-z0-9-]+)/g)].map((m) => m[1]));
        const declared = new Set([...(root?.body ?? "").matchAll(/(--vx-[a-z0-9-]+)\s*:/g)].map((m) => m[1]));
        expect([...used].filter((token) => !declared.has(token))).toEqual([]);
    });

    it("derives its accent from the palette rather than hardcoding one", () => {
        const root = rules.find((rule) => rule.selector === ":root");
        expect(root?.body).toMatch(/--vx-accent:\s*var\(--accent-gold\)/);
    });

    it("fills an active segment with text the palette picks, not a hardcoded white", () => {
        // White on Basic's gold is unreadable; `--vx-on-accent` is dark there and
        // white under Vulpes' ember.
        const active = rules.find((rule) => rule.selector === ".vx-seg__btn.is-active");
        expect(active?.body).toMatch(/color:\s*var\(--vx-on-accent\)/);
    });

    it("centres the page-turn arrows and hides them until hovered", () => {
        const zone = rules.find((rule) => rule.selector === ".vx-reader__zone");
        const arrow = rules.find((rule) => rule.selector === ".vx-reader__arrow");
        // Centred by the zone's own flex alignment, so it stays centred at any window
        // height -- including fullscreen, and mid-resize.
        expect(zone?.body).toMatch(/align-items:\s*center/);
        // ...which only works if the zone actually spans the height: the shared
        // `button` rule gives every button `height: 38px`, and an absolutely
        // positioned box with `top` + `height` + `bottom` drops the `bottom`
        // and pins itself 38px tall to the top. `height: auto` restores the
        // top-to-bottom stretch the centring depends on.
        expect(zone?.body).toMatch(/top:\s*0/);
        expect(zone?.body).toMatch(/bottom:\s*0/);
        expect(zone?.body).toMatch(/height:\s*auto/);
        expect(arrow?.body).toMatch(/opacity:\s*0/);
        const shown = rules.find((rule) => rule.selector.includes(".vx-reader__zone:hover .vx-reader__arrow"));
        expect(shown?.body).toMatch(/opacity:\s*1/);
    });

    it("keeps the click zones slim and square-cornered, the button round and soft", () => {
        // 30% off every axis of the old 11% / 56px / 150px box; the squareness
        // belongs on the region, never on the disc.
        const zone = rules.find((rule) => rule.selector === ".vx-reader__zone");
        expect(zone?.body).toMatch(/width:\s*7\.7%/);
        expect(zone?.body).toMatch(/min-width:\s*39px/);
        expect(zone?.body).toMatch(/max-width:\s*105px/);
        expect(zone?.body).toMatch(/border-radius:\s*0/);
        // Round badge, softened with a diffuse shadow and a thinner fill. The
        // blur belongs to the veil behind it, not to the disc.
        const arrow = rules.find((rule) => rule.selector === ".vx-reader__arrow");
        expect(arrow?.body).toMatch(/border-radius:\s*50%/);
        expect(arrow?.body).toMatch(/box-shadow:[^;]*28px/);
    });

    it("lets the continuous view show through the reader surface", () => {
        // The strip *is* the editor's stage underneath; an opaque surface would hide it.
        const vertical = rules.find((rule) => rule.selector === ".vx-reader--vertical");
        expect(vertical?.body).toMatch(/background:\s*transparent/);
    });

    it("keeps the reader bar reachable while the rest of the chrome is click-through", () => {
        const chrome = rules.find((rule) => rule.selector === ".vx-reader__chrome");
        const bar = rules.find((rule) => rule.selector === ".vx-reader__bar");
        expect(chrome?.body).toMatch(/pointer-events:\s*none/);
        expect(bar?.body).toMatch(/pointer-events:\s*auto/);
    });

    it("moves the pager with a transform the compositor can take", () => {
        // The pages are laid out at natural size and the row is scaled as a unit, so
        // a frame of panning or zooming is one transform -- not a relayout and
        // re-raster of two full-resolution scans, which is what the first cut did.
        const row = rules.find((rule) => rule.selector === ".vx-pager__row");
        expect(row?.body).toMatch(/transform-origin:\s*50% 50%/);
        expect(row?.body).toMatch(/will-change:\s*transform/);
        // Not shrinkable: once zoomed in the row is wider than the stage, and a
        // shrinkable flex item would be squeezed back to fit instead of overflowing.
        expect(row?.body).toMatch(/flex:\s*0 0 auto/);
    });

    it("centres the pager row by layout, so the transform only carries pan and zoom", () => {
        const pager = rules.find((rule) => rule.selector === ".vx-pager");
        expect(pager?.body).toMatch(/align-items:\s*center/);
        expect(pager?.body).toMatch(/justify-content:\s*center/);
    });

    it("blurs only the edge veil, and nothing else over the artwork", () => {
        // One live blur is affordable: two strips of at most 105px each, at the edges
        // where no artwork is being read. Anywhere else -- the control bar, the pager,
        // the arrow disc -- it would re-blur the page the reader is looking at, on
        // every frame of a pan or a zoom.
        const blurred = rules
            .filter((rule) => rule.body.includes("backdrop-filter"))
            .flatMap((rule) => rule.selector.split(","))
            .map((part) => part.trim());
        const unexpected = blurred.filter((selector) => !selector.includes(".vx-reader__zone"));
        expect(unexpected).toEqual([]);
    });

    it("fades the veil out toward the page instead of ending at a hard edge", () => {
        // The mask is the whole point of putting the blur on a pseudo-element: it
        // fades the blur as well as the tint, so the band dissolves into the artwork
        // rather than drawing a seam down it. One gradient per side, each running
        // from its own screen edge.
        const start = rules.find((rule) => rule.selector === ".vx-reader__zone--start::before");
        const end = rules.find((rule) => rule.selector === ".vx-reader__zone--end::before");
        expect(start?.body).toMatch(/mask-image:\s*linear-gradient\(to right/);
        expect(end?.body).toMatch(/mask-image:\s*linear-gradient\(to left/);
        // Prefixed too: `-webkit-mask-image` is still what ships in Chromium.
        expect(start?.body).toContain("-webkit-mask-image");
        expect(end?.body).toContain("-webkit-mask-image");
    });

    it("shows nothing at all until the zone is hovered or focused", () => {
        // At rest a tinted band over the edges of the page reads as a shadow on the
        // artwork, which is the one thing the reader must never draw. Both halves of
        // the veil are gated: the tint by `opacity`, and the blur by not being
        // declared at rest at all.
        const veil = rules.find((rule) => rule.selector === ".vx-reader__zone::before");
        expect(veil?.body).toMatch(/opacity:\s*0/);
        expect(veil?.body).not.toContain("backdrop-filter");

        const shown = rules.find(
            (rule) => rule.selector.includes(".vx-reader__zone:hover::before") && !rule.selector.includes("is-panning"),
        );
        expect(shown?.body).toMatch(/opacity:\s*1/);
        expect(shown?.body).toMatch(/backdrop-filter:\s*blur/);
        // Keyboard users get it too.
        expect(shown?.selector).toContain(":focus-visible::before");
    });

    it("leaves the arrow invisible at rest as well, shadow included", () => {
        // `opacity` takes the disc's `box-shadow` with it; a `visibility` or a
        // `display` toggle would not fade, and a transparent background would leave
        // the shadow painting over the page.
        const arrow = rules.find((rule) => rule.selector === ".vx-reader__arrow");
        expect(arrow?.body).toMatch(/opacity:\s*0/);
    });

    it("drops the live blur while the page is being dragged", () => {
        // A pan is when the reader can least spare a per-frame re-blur, and a drag
        // that began mid-page can still end up under a zone -- so hover and panning
        // do overlap. The pager sets `is-panning` on itself and the zones are its
        // later siblings, so this costs no JavaScript; it only works while that
        // sibling order holds, which a layout test guards.
        const gated = rules.find((rule) => rule.selector.includes(".vx-pager.is-panning"));
        expect(gated?.selector).toContain(".vx-reader__zone:hover::before");
        expect(gated?.body).toMatch(/backdrop-filter:\s*none/);
    });

    it("keeps the arrow above the veil without an explicit z-index", () => {
        // The veil is an absolutely positioned pseudo-element, so it paints over
        // in-flow content; the arrow has to be positioned too for tree order to put
        // it on top.
        const arrow = rules.find((rule) => rule.selector === ".vx-reader__arrow");
        expect(arrow?.body).toMatch(/position:\s*relative/);
        expect(arrow?.body).not.toMatch(/z-index/);
    });

    it("hides the whole reader, cursor included, once the pointer goes still", () => {
        const surface = rules.find((rule) => rule.selector === ".vx-reader.is-idle");
        expect(surface?.body).toMatch(/cursor:\s*none/);
        const hidden = rules.find((rule) => rule.selector.includes(".vx-reader.is-idle .vx-reader__chrome"));
        expect(hidden?.body).toMatch(/opacity:\s*0/);
        // `visibility` as well: a transparent bar still takes the clicks aimed at the
        // page behind it, and a hover would re-dim it.
        expect(hidden?.body).toMatch(/visibility:\s*hidden/);
        expect(hidden?.body).toMatch(/pointer-events:\s*none/);
    });
});

describe("action row stylesheet", () => {
    const rules = rulesOf(actionRow);

    it("is unscoped: the row is the same markup in every layout", () => {
        for (const rule of rules) {
            expect(rule.selector).not.toContain("body.layout-");
        }
    });

    it("shows the glyphs and hides the words by default", () => {
        // The verbs are icon-only in every layout; the tooltip names each one
        // on hover, and Vulpes restates these same values for its footer.
        const icon = rules.find((rule) => rule.selector.endsWith(".btn-icon"));
        const label = rules.find((rule) => rule.selector.endsWith(".btn-label"));
        expect(icon?.body).toMatch(/display:\s*inline-flex/);
        expect(label?.body).toMatch(/display:\s*none/);
    });
});

describe("Vulpes section toggles", () => {
    it("sizes the toggle to its 30px bar", () => {
        // Without its own height it inherits the shared 38px button height and
        // overflows the bar top and bottom.
        const toggle = rulesOf(layout).find((rule) => rule.selector === "body.layout-vulpes .vx-section__toggle");
        expect(toggle?.body).toMatch(/height:\s*22px/);
    });

    it("keeps hover to the brightening title, with no background to spill", () => {
        // Must outrank the shared `button:hover:not(...)` chain, which would
        // otherwise paint its tertiary pill here.
        const hover = rulesOf(layout).find((rule) => rule.selector === "body.layout-vulpes .vx-section__toggle:hover");
        expect(hover?.body).toMatch(/background-color:\s*transparent\s*!important/);
    });
});

describe("Vulpes footer font picker", () => {
    it("takes the same share as the Basic row", () => {
        // Compared, not hardcoded, so both sides may evolve as long as they stay equal.
        const base = rulesOf(actionRow).find((rule) => rule.selector === ".confirm-action-row .custom-font-dropdown");
        const vulpes = rulesOf(layout).find(
            (rule) => rule.selector === "body.layout-vulpes .vx-action-row .custom-font-dropdown",
        );
        expect(base).toBeDefined();
        expect(vulpes).toBeDefined();
        const flexOf = (body: string | undefined): string | null => body?.match(/flex:\s*([^;]+);/)?.[1].trim() ?? null;
        expect(flexOf(vulpes?.body)).toBe(flexOf(base?.body));
    });

    it("keeps the trigger clear of the statusbar's own border", () => {
        // 32px in a 34px bar all but touches it; the footer verbs are 24px.
        const trigger = rulesOf(layout).find(
            (rule) => rule.selector === "body.layout-vulpes .vx-action-row .custom-font-dropdown .font-select-trigger",
        );
        expect(trigger?.body).toMatch(/height:\s*24px/);
    });
});

describe("Vulpes captured text", () => {
    it("focuses the text area through the border alone, with no glow ring", () => {
        const focus = rulesOf(layout).find(
            (rule) => rule.selector === "body.layout-vulpes .vx-capture__body #textArea:focus-visible",
        );
        expect(focus?.body).toMatch(/border-color:\s*var\(--vx-accent\)/);
        expect(focus?.body).toMatch(/box-shadow:\s*none/);
    });
});
