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
            htmlOptions += `<div class="font-option-item" onclick="window.__selectFontOption('${font.font_filename}','${font.font_name}','${cssFontFamily}')" onmouseenter="window.__checkMarquee(this)" onmouseleave="window.__stopMarquee(this)" style="font-family:'${cssFontFamily}',sans-serif;">${font.font_name}</div>`;
        });

        styleTag.innerHTML =
            cssRules +
            `
            .font-option-item{padding:8px 12px;color:var(--text-primary);cursor:pointer;font-size:14px;transition:background .1s;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;box-sizing:border-box;}
            .font-option-item:hover{background-color:var(--bg-tertiary);}
            .font-option-item.selected{background-color:var(--color-primary)!important;font-weight:bold;}
            .font-option-item.scroll-active:hover{text-overflow:clip;animation:marqueeTextScroll 5s linear infinite;}
            @keyframes marqueeTextScroll{0%{text-indent:0%}50%{text-indent:-30%}100%{text-indent:0%}}
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

export function checkAndStartMarquee(el: HTMLElement): void {
    if (el.scrollWidth > el.clientWidth) el.classList.add("scroll-active");
}

export function stopMarquee(el: HTMLElement): void {
    el.classList.remove("scroll-active");
}

export function toggleFontDropdown(): void {
    const container = document.getElementById("fontOptionsContainer");
    if (!container) return;
    const isOpening = container.style.display === "none" || container.style.display === "";
    container.style.display = isOpening ? "block" : "none";
    if (isOpening) {
        const active = container.querySelector(".font-option-item.selected");
        if (active) active.scrollIntoView({ behavior: "auto", block: "nearest" });
    }
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
        const cur = container.querySelector(`[onclick*="${filename}"]`);
        if (cur) cur.classList.add("selected");
    }
    const fontOptionsContainer = document.getElementById("fontOptionsContainer");
    if (fontOptionsContainer) fontOptionsContainer.style.display = "none";
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
        if (!(e.target as HTMLElement).closest(".custom-font-dropdown")) {
            const c = document.getElementById("fontOptionsContainer");
            if (c) c.style.display = "none";
        }
    });
}
