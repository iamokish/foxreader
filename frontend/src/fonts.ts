import { state } from "./state";
import type { FontData } from "./state";

export async function loadFontsAndBuildMenu(): Promise<void> {
    try {
        const response = await fetch("/api/fonts");
        state.currentFontsData = (await response.json()) as FontData[];

        const styleTag = document.getElementById("dynamicFontStyles");
        const container = document.getElementById("fontOptionsContainer");
        if (!container || !styleTag) return;

        let cssRules = "";
        let htmlOptions = "";

        state.currentFontsData.forEach((font) => {
            const cssFontFamily = font.font_name.replace(/\s+/g, "_");
            cssRules += `@font-face{font-family:'${cssFontFamily}';src:url('${font.font_data_uri}') format('${font.font_format}');font-display:swap;}`;
            const safeName = font.font_name
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;");
            const safeArg = font.font_name.replace(/'/g, "\\'");
            const safeFile = font.font_filename.replace(/'/g, "\\'");
            // `data-filename` is the lookup key for the selected highlight below,
            // so names with quotes still match without relying on the `onclick`
            // text (where those same quotes are backslash-escaped).
            const safeFileAttr = font.font_filename
                .replace(/&/g, "&amp;")
                .replace(/"/g, "&quot;");
            // The name rides in its own face (inherited from the row's inline
            // `font-family`) inside a measuring span. On hover of an overflowing
            // row, `__checkMarquee` arms a transform-only ping-pong on that span --
            // left-to-right and back -- so the box itself never moves.
            htmlOptions += `<div class="font-option-item" title="${safeName}" data-filename="${safeFileAttr}" onclick="window.__selectFontOption('${safeFile}','${safeArg}','${cssFontFamily}')" onmouseenter="window.__checkMarquee(this)" onmouseleave="window.__stopMarquee(this)" style="font-family:'${cssFontFamily}',sans-serif;"><span class="font-option-item__text">${safeName}</span></div>`;
        });

        // Reduced-motion note: the marquee is deliberately *bounded* rather than
        // killed there (`animation-iteration-count: 2`). It only ever runs while
        // the pointer is on that one row, so it is an invoked content reveal --
        // the only way to read a truncated name besides the delayed tooltip --
        // not ambient motion. The Vulpes half of the rule exists because that
        // layout flattens every animation duration under reduced motion; without
        // the restore the traverse would complete in 0.01ms there (a snap,
        // worse than either motion or stillness).
        styleTag.innerHTML =
            cssRules +
            `
            .font-option-item{padding:8px 12px;color:var(--text-primary);cursor:pointer;font-size:14px;transition:background .1s;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;box-sizing:border-box;}
            .font-option-item:hover{background-color:var(--bg-tertiary);}
            .font-option-item.selected{background-color:var(--color-primary)!important;font-weight:bold;}
            .font-option-item__text{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
            .font-option-item.scroll-active:hover .font-option-item__text{overflow:visible;text-overflow:clip;will-change:transform;animation:fontMarquee var(--marquee-dur,5s) ease-in-out infinite;}
            @keyframes fontMarquee{0%,15%{transform:translateX(0);}45%,55%{transform:translateX(calc(var(--marquee-dist,0px)*-1));}85%,100%{transform:translateX(0);}}
            @media (prefers-reduced-motion:reduce){.font-option-item.scroll-active:hover .font-option-item__text{animation-iteration-count:2;}body.layout-vulpes .font-option-item.scroll-active:hover .font-option-item__text{animation-duration:var(--marquee-dur,5s)!important;animation-iteration-count:2;}}
        `;
        container.innerHTML = htmlOptions;

        if (state.currentFontsData.length > 0) {
            const first = state.currentFontsData[0];
            selectFontOption(first.font_filename, first.font_name, first.font_name.replace(/\s+/g, "_"));
        }
    } catch (error) {
        console.error("Error loading font configuration menu:", error);
    }
}

/**
 * Arm the hover marquee for one row, if its name actually overflows.
 *
 * Measured once per hover entry (one sync read, not per frame): the row's box
 * never changes, so `:hover` cannot flap. The distance and a speed-scaled
 * duration land in `--marquee-dist` / `--marquee-dur`, which the transform-only
 * `fontMarquee` keyframes read -- per-frame work is a compositor-layer shift,
 * never a layout, which is what the old `text-indent` version got wrong.
 */
export function checkAndStartMarquee(el: HTMLElement): void {
    // Measure the text span, not the row: the span clips the name
    // (`overflow: hidden`), so the clipped tail no longer shows up in the
    // row's own `scrollWidth` and measuring the row always reads ~0 -- the
    // marquee would never arm. The span's own `scrollWidth` still reports its
    // full content width. Falls back to the row for markup without the span.
    const text = el.querySelector<HTMLElement>(".font-option-item__text") ?? el;
    const overflow = text.scrollWidth - text.clientWidth;
    if (overflow <= 4) return;
    const dist = Math.ceil(overflow);
    el.style.setProperty("--marquee-dist", `${dist}px`);
    // ~50px/s of travel with pauses at both ends; clamped so short names do
    // not whip and very long ones do not crawl.
    const dur = Math.min(8, Math.max(2.5, 1.2 + dist / 50));
    el.style.setProperty("--marquee-dur", `${dur.toFixed(2)}s`);
    el.classList.add("scroll-active");
}

export function stopMarquee(el: HTMLElement): void {
    el.classList.remove("scroll-active");
}

/**
 * Open or close the font menu.
 *
 * The `is-open` class is the single source of truth for whether the menu is showing
 * -- see the note in `FontSelector.tsx`. It used to be an inline `display`, which
 * JSX also declared, so "is it open?" was answered by reading back whatever the last
 * writer left rather than by what the user last asked for.
 */
export function toggleFontDropdown(): void {
    const container = document.getElementById("fontOptionsContainer");
    if (!container) return;
    const isOpening = container.classList.toggle("is-open");
    if (isOpening) {
        const active = container.querySelector(".font-option-item.selected");
        if (active) active.scrollIntoView({ behavior: "auto", block: "nearest" });
    }
}

export function closeFontDropdown(): void {
    document.getElementById("fontOptionsContainer")?.classList.remove("is-open");
}

export function selectFontOption(filename: string, fontName: string, cssFontFamily: string): void {
    const fontSelect = document.getElementById("fontSelect") as HTMLInputElement | null;
    const selectedFontLabel = document.getElementById("selectedFontLabel");
    const fontSelectTrigger = document.getElementById("fontSelectTrigger");

    if (fontSelect) fontSelect.value = filename;
    if (selectedFontLabel) selectedFontLabel.textContent = fontName;
    if (fontSelectTrigger) fontSelectTrigger.style.fontFamily = `'${cssFontFamily}', sans-serif`;

    const container = document.getElementById("fontOptionsContainer");
    if (container) {
        const prev = container.querySelector(".font-option-item.selected");
        if (prev) prev.classList.remove("selected");
        // Exact match on the key first (quote-safe); the substring fallback keeps
        // older built menus working.
        const items = container.querySelectorAll(".font-option-item");
        let cur: Element | null = null;
        for (const item of items) {
            if (item.getAttribute("data-filename") === filename) {
                cur = item;
                break;
            }
        }
        cur ??= container.querySelector(`[onclick*="${filename}"]`);
        if (cur) cur.classList.add("selected");
    }
    // Picking a face closes the menu. `closeFontDropdown`, not a raw class write, so
    // there is exactly one place that knows how "closed" is expressed.
    closeFontDropdown();
}

export function setSelectedFontByFilename(filename: string): void {
    const target = state.currentFontsData.find((f) => f.font_filename === filename);
    if (target) selectFontOption(target.font_filename, target.font_name, target.font_name.replace(/\s+/g, "_"));
}

export function getCurrentFontFilename(): string {
    const el = document.getElementById("fontSelect") as HTMLInputElement | null;
    const value = el ? el.value : "";
    return value || getDefaultFontFilename();
}

export function getCurrentFontName(): string {
    const el = document.getElementById("fontSelectTrigger");
    return el ? el.style.fontFamily || "sans-serif" : "sans-serif";
}

export function getDefaultFontFilename(): string {
    if (state.currentFontsData.length > 0) return state.currentFontsData[0].font_filename;
    return "ComicMono.ttf";
}

export function getDefaultFontName(): string {
    if (state.currentFontsData.length > 0) {
        const f = state.currentFontsData[0];
        return `'${f.font_name.replace(/\s+/g, "_")}', sans-serif`;
    }
    return "sans-serif";
}

export function getCurrentRawFontName(): string {
    const el = document.getElementById("selectedFontLabel");
    return el ? (el.textContent?.trim() ?? "") : "";
}

export function getDefaultRawFontName(): string {
    return state.currentFontsData.length > 0 ? state.currentFontsData[0].font_name : "";
}

export function initFontListeners(): void {
    window.__selectFontOption = selectFontOption;
    window.__checkMarquee = checkAndStartMarquee;
    window.__stopMarquee = stopMarquee;

    window.addEventListener("click", (e) => {
        if (!(e.target as HTMLElement).closest(".custom-font-dropdown")) closeFontDropdown();
    });
}
