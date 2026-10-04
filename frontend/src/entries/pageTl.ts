/**
 * Page TL: OCR then translate every visible region on the page, in one pass.
 *
 * The old version reported nothing but "OCR..." / "Translating...", swallowed
 * every per-region failure into an identical toast, and — when no translator was
 * selected — quietly fell back to a built-in endpoint, so a page could come back
 * translated by something the user never chose. This one counts what happened and
 * says so, and refuses to guess a translator.
 */

import { state } from "../state";
import type { Entry } from "../state";
import type { Point, RectangleCoords, TranslateEndpoint } from "../types";
import * as api from "../api";
import { hideLoading, showLoading, showNotify, updateLoadingText } from "../ui";
import { beginCapture, endCapture, isCapturing } from "../ocr";
import { buildTranslationCharacters, buildTranslationContext } from "../translationContext";

/** Courtesy gap between translation calls, so a public endpoint is not hammered. */
const TRANSLATE_GAP_MS = 400;

interface Tally {
    ocrDone: number;
    ocrEmpty: number;
    ocrFail: number;
    tlDone: number;
    tlFail: number;
}

interface Translator {
    endpoint: TranslateEndpoint;
    lang: string;
    langGroup: string;
}

/** Whoever the user picked. `null` means nobody did — never a default. */
function activeTranslator(): Translator | null {
    const btn = document.querySelector<HTMLElement>(".translator-buttons button.active-translator");
    const endpoint = btn?.dataset.endpoint;
    if (!btn || !endpoint) return null;
    return {
        endpoint: endpoint as TranslateEndpoint,
        lang: btn.dataset.lang ?? "",
        langGroup: btn.dataset.langGroup ?? "",
    };
}

/** True when the chosen translator needs the local model and it is switched off. */
function blockedByML(): boolean {
    const btn = document.querySelector<HTMLElement>(".translator-buttons button.active-translator");
    if (btn?.dataset.requiresMl !== "true") return false;
    return !(document.getElementById("enableML") as HTMLInputElement | null)?.checked;
}

async function ocrEntry(entry: Entry, filename: string, lang: string): Promise<string> {
    if (entry.region.type === "polygon") {
        const data = await api.ocrFreeform({
            filename,
            points: (entry.region.coords as Point[]).map((p) => ({ x: p.x, y: p.y })),
            lang,
            isGrayScale: state.isGrayScaleEnabled,
        });
        return (data.text || "").trim();
    }
    const r = entry.region.coords as RectangleCoords;
    const data = await api.ocrCrop({
        filename,
        x: r.x,
        y: r.y,
        width: r.w,
        height: r.h,
        lang,
        isGrayScale: state.isGrayScaleEnabled,
    });
    return (data.text || "").trim();
}

/** Reads as `OCR 3/7 · 2 read · 1 blank · 1 failed`. */
function progressLine(stage: string, index: number, total: number, tally: Tally, translating: boolean): string {
    const bits: string[] = [`${stage} ${index}/${total}`];
    if (translating) {
        if (tally.tlDone) bits.push(`${tally.tlDone} translated`);
        if (tally.tlFail) bits.push(`${tally.tlFail} failed`);
    } else {
        if (tally.ocrDone) bits.push(`${tally.ocrDone} read`);
        if (tally.ocrEmpty) bits.push(`${tally.ocrEmpty} blank`);
        if (tally.ocrFail) bits.push(`${tally.ocrFail} failed`);
    }
    return bits.join(" · ");
}

export interface PageTlDeps {
    refresh: () => void;
}

let running = false;

/**
 * OCR and translate the whole page.
 *
 * Holds the capture lock for the duration, so the four capture tools cannot be
 * started underneath it, and aborts if the user navigates to another page.
 */
export async function pageCapture(deps: PageTlDeps): Promise<void> {
    if (running) {
        showNotify("⏳ Page TL is already running.");
        return;
    }
    if (!state.currentImageFile) {
        showNotify("⚠️ No page is open.");
        return;
    }
    if (isCapturing()) {
        showNotify("⚠️ Finish the active capture first.");
        return;
    }

    const filename = state.currentImageFile;
    const entries = (state.pageEntriesCache[filename] ||= []);
    const pending = entries.filter((e) => e.visible);
    if (!pending.length) {
        showNotify("⚠️ No regions on this page — capture some bubbles first.");
        return;
    }

    if (blockedByML()) {
        showNotify("⚠️ The selected translator needs the local model — enable MTL first.");
        return;
    }

    running = true;
    beginCapture("PageCapture");
    const lang = (document.getElementById("languageSelect") as HTMLSelectElement | null)?.value ?? "";
    const tally: Tally = { ocrDone: 0, ocrEmpty: 0, ocrFail: 0, tlDone: 0, tlFail: 0 };
    /** True once the page changed under us; everything after is discarded. */
    const stale = () => state.currentImageFile !== filename;

    showLoading("Page TL starting...");
    try {
        // ---- OCR pass -------------------------------------------------------
        const needOcr = pending.filter((e) => !e.ocr_text);
        for (let i = 0; i < needOcr.length; i++) {
            if (stale()) {
                showNotify("⚠️ Page changed — Page TL stopped.");
                return;
            }
            const entry = needOcr[i];
            updateLoadingText(progressLine("OCR", i + 1, needOcr.length, tally, false));
            try {
                const text = await ocrEntry(entry, filename, lang);
                entry.ocr_text = text;
                if (text) tally.ocrDone++;
                else tally.ocrEmpty++;
            } catch {
                // One unreadable bubble is not a reason to abandon the page.
                tally.ocrFail++;
            }
        }

        // Showing the recognised text before translation starts makes the second
        // pass feel like progress rather than a second wait.
        if (needOcr.length && !stale()) deps.refresh();

        // ---- translation pass -----------------------------------------------
        const needTl = pending.filter((e) => !e.text && e.ocr_text);
        if (!needTl.length) {
            summarise(tally, needOcr.length, 0);
            return;
        }

        const translator = activeTranslator();
        if (!translator) {
            // Deliberately no fallback endpoint: translating through something
            // the user did not pick is worse than not translating.
            deps.refresh();
            showNotify(
                `⚠️ No active translator selected — OCR done (${tally.ocrDone} read). Pick a translator to translate.`,
            );
            return;
        }

        for (let i = 0; i < needTl.length; i++) {
            if (stale()) {
                showNotify("⚠️ Page changed — Page TL stopped.");
                return;
            }
            const entry = needTl[i];
            updateLoadingText(progressLine("Translate", i + 1, needTl.length, tally, true));
            if (i > 0) await new Promise((r) => setTimeout(r, TRANSLATE_GAP_MS));
            try {
                // MTL with the Context switch on carries the entries before
                // this one -- `entry.text` is written back as the run goes, so
                // each bubble sees everything translated earlier in the run.
                // Other engines never receive context. Characters ride along
                // on the same MTL path while their switch is on: roster plus
                // per-turn links and this entry's own speaker. Custom
                // endpoints receive both lists too, rendering them only when
                // their request schema carries the matching nodes.
                const sendsLists =
                    translator.endpoint === "/translate/ml" || translator.endpoint === "/translate/custom";
                const context = sendsLists && state.isContextEnabled ? buildTranslationContext(entry.id) : undefined;
                const built =
                    sendsLists && state.isCharactersEnabled ? buildTranslationCharacters(entry.id) : undefined;
                const extras = built && (built.character_info.length || built.meta_id) ? built : undefined;
                const result = extras
                    ? await api.translate(
                          translator.endpoint,
                          entry.ocr_text,
                          translator.lang,
                          translator.langGroup,
                          context,
                          extras,
                      )
                    : await api.translate(
                          translator.endpoint,
                          entry.ocr_text,
                          translator.lang,
                          translator.langGroup,
                          context,
                      );
                if (result.translated) {
                    entry.text = result.translated;
                    tally.tlDone++;
                } else {
                    tally.tlFail++;
                }
            } catch {
                tally.tlFail++;
            }
        }

        if (!stale()) deps.refresh();
        summarise(tally, needOcr.length, needTl.length);
    } finally {
        running = false;
        endCapture();
        hideLoading();
    }
}

function summarise(tally: Tally, ocrAttempted: number, tlAttempted: number): void {
    if (!ocrAttempted && !tlAttempted) {
        showNotify("✅ Nothing to do — every region already has text.");
        return;
    }
    const bits: string[] = [];
    if (ocrAttempted) {
        bits.push(`OCR ${tally.ocrDone}/${ocrAttempted}`);
        if (tally.ocrEmpty) bits.push(`${tally.ocrEmpty} blank`);
        if (tally.ocrFail) bits.push(`${tally.ocrFail} failed`);
    }
    if (tlAttempted) {
        bits.push(`TL ${tally.tlDone}/${tlAttempted}`);
        if (tally.tlFail) bits.push(`${tally.tlFail} failed`);
    }
    const clean = !tally.ocrFail && !tally.tlFail;
    showNotify(`${clean ? "✅" : "⚠️"} Page TL — ${bits.join(" · ")}`);
}
