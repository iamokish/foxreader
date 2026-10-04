import * as api from "./api";
import { showNotify } from "./ui";
import { on } from "./eventbus";
import { state as appState } from "./state";
import { buildTranslationCharacters, buildTranslationContext } from "./translationContext";
import type { CharacterInfo, TranslationResponse, TranslateEndpoint } from "./types";

/**
 * What `#translatedText` is showing.
 *
 * `text` covers both a finished translation and a panel nothing has happened in
 * yet -- either way what it holds is the user's text, so Confirm takes it. The
 * other two are not text: one is a message, the other is a placeholder for a
 * reply that has not arrived.
 */
export type TranslationPanelState = "error" | "translating" | "text";

const STATE_CLASS: Record<TranslationPanelState, string> = {
    error: "error",
    translating: "translating",
    text: "translated",
};

/** The translation panel, if this layout has one. */
export function translationPanel(): HTMLElement | null {
    return document.getElementById("translatedText");
}

/**
 * Read the panel's state.
 *
 * `data-state` is the contract; the class list is the fallback, because the
 * panel's opening markup is server- (or Preact-) rendered and carries only a
 * class until `renderTranslationUI` first runs. A missing panel reads as `text`,
 * which pairs with the empty string {@link translationPanelText} returns for it.
 */
export function translationPanelState(panel: HTMLElement | null = translationPanel()): TranslationPanelState {
    if (!panel) return "text";
    const stamped = panel.dataset.state;
    if (stamped === "error" || stamped === "translating" || stamped === "text") return stamped;
    if (panel.classList.contains("error")) return "error";
    if (panel.classList.contains("translating")) return "translating";
    return "text";
}

/**
 * What the panel offers as an entry's translation.
 *
 * Empty unless it is actually showing text: an error panel holds a message
 * ("Select a translator", a failed request) and a translating one holds
 * "Translating...", and storing either as a translation is worse than storing
 * nothing.
 */
export function translationPanelText(panel: HTMLElement | null = translationPanel()): string {
    if (!panel || translationPanelState(panel) !== "text") return "";
    // `innerText` keeps the line breaks the typesetter honours; jsdom has no
    // layout and so no `innerText`, hence the fallback.
    return (panel.innerText ?? panel.textContent ?? "").trim();
}

export function renderTranslationUI(
    data: Partial<TranslationResponse> & { error?: string } & { translating?: boolean },
): void {
    const translatedDiv = translationPanel();

    if (!translatedDiv) return;

    const state: TranslationPanelState = data.error ? "error" : data.translating ? "translating" : "text";
    translatedDiv.className = `panel ${STATE_CLASS[state]}`;
    translatedDiv.dataset.state = state;
    translatedDiv.contentEditable = state === "text" ? "true" : "false";
    translatedDiv.textContent =
        state === "error" ? (data.error ?? "") : state === "translating" ? "Translating..." : (data.translated ?? "");
}

export async function executeTranslationPipeline(
    engineEndpoint: TranslateEndpoint,
    currentLang: string,
    pressedBtn: HTMLElement,
    errorMessage: string,
    context?: [string, string][],
    extras?: { character_info?: CharacterInfo[]; context_character_links?: (string | null)[]; meta_id?: string | null },
): Promise<void> {
    const rawText = (document.getElementById("textArea") as HTMLTextAreaElement)?.value.trim() ?? "";
    const container = document.querySelector(".translator-buttons");
    if (container) {
        const cur = container.querySelector("button.active-translator");
        if (cur) cur.classList.remove("active-translator");
    }
    pressedBtn.classList.add("active-translator");

    if (!rawText) {
        renderTranslationUI({ original: "", translated: "", alt_translated: "" });
        return;
    }

    const isML = pressedBtn.dataset.requiresMl === "true";
    const mtlBtns = isML
        ? document.querySelectorAll<HTMLElement>(".translator-buttons button[data-requires-ml='true']")
        : [];

    if (isML) mtlBtns.forEach((b) => ((b as HTMLButtonElement).disabled = true));
    try {
        renderTranslationUI({
            translating: true,
        });
        // Six args only when characters ride along: older tests and older
        // backends see the exact 5-arg shape otherwise.
        const result = extras
            ? await api.translate(engineEndpoint, rawText, currentLang, pressedBtn.dataset.langGroup, context, extras)
            : await api.translate(engineEndpoint, rawText, currentLang, pressedBtn.dataset.langGroup, context);
        renderTranslationUI(result);
    } catch {
        showNotify(errorMessage);
        renderTranslationUI({ error: "An unexpected routing pipeline connection error occurred." });
    } finally {
        if (isML) mtlBtns.forEach((b) => ((b as HTMLButtonElement).disabled = false));
    }
}

export function runTranslatorButton(btn: HTMLElement): void {
    const endpoint = btn.dataset.endpoint;
    const btnLang = btn.dataset.lang;
    if (!endpoint || !btnLang || (btn as HTMLButtonElement).disabled) return;
    if (btn.dataset.requiresMl === "true") {
        const isMLEnabled = (document.getElementById("enableML") as HTMLInputElement)?.checked;
        if (!isMLEnabled) {
            showNotify("Enable MTL!");
            return;
        }
    }
    // Context rides on MTL buttons and on custom endpoints while the switch
    // is on; the backend decides whether it lands anywhere. MTL renders it
    // into the prompt of supporting models, and a custom endpoint only sees
    // it when its request schema carries a Context node -- anything else
    // builds byte-identical requests with or without it. The pairs come from
    // the entries before the selected one, in reading order.
    //
    // Characters ride the same two paths while the Characters switch is on:
    // the roster snapshot plus per-turn links aligned to the pairs and the
    // selected entry's own speaker. A custom endpoint only sees them when its
    // request schema carries the Character Info / Context Links nodes.
    // Other engines never receive either.
    const sendsContext = btn.dataset.requiresMl === "true" || btn.dataset.endpoint === "/translate/custom";
    const context =
        sendsContext && appState.isContextEnabled
            ? buildTranslationContext(appState.currentlySelectedEntryId)
            : undefined;
    const characters =
        sendsContext && appState.isCharactersEnabled
            ? buildTranslationCharacters(appState.currentlySelectedEntryId)
            : undefined;
    const extras = characters && (characters.character_info.length || characters.meta_id) ? characters : undefined;
    executeTranslationPipeline(
        endpoint as TranslateEndpoint,
        btnLang,
        btn,
        `❌ ${btn.textContent?.trim() ?? ""} failed`,
        context,
        extras,
    );
}

let selectedLangGroup = "japanese";

/** Show only the buttons belonging to the selected language group.
 *
 * The buttons are queried on every call: custom endpoint buttons arrive
 * after the first render, and they have to obey the same filter.
 */
export function refreshTranslatorButtons(group?: string): void {
    if (group) selectedLangGroup = group;

    const cur = document.querySelector(".translator-buttons button.active-translator");
    if (cur && (cur as HTMLElement).dataset.langGroup !== selectedLangGroup) {
        cur.classList.remove("active-translator");
    }

    document.querySelectorAll<HTMLElement>(".translator-buttons button").forEach((btn) => {
        btn.style.display = btn.dataset.langGroup === selectedLangGroup ? "inline-block" : "none";
    });
}

export function initTranslatorListeners(): void {
    document.querySelector(".translator-buttons")?.addEventListener("click", (e) => {
        const btn = (e.target as HTMLElement).closest("button");
        if (btn) runTranslatorButton(btn);
    });

    const select = document.getElementById("languageSelect") as HTMLSelectElement | null;

    if (select) {
        select.value = selectedLangGroup;
        refreshTranslatorButtons(selectedLangGroup);
        select.addEventListener("change", function () {
            refreshTranslatorButtons(this.value);
        });
    }

    on("translators:changed", () => refreshTranslatorButtons());
}

export function handleAutoTranslation(options?: { force?: boolean }): void {
    // The Auto TL switch gates every *automatic* trigger (post-OCR). An
    // explicit user gesture -- the entry card's TL button -- passes
    // force:true and always translates. Without this gate the checkbox only
    // flipped a flag nobody read, so disabling Auto TL changed nothing.
    if (!options?.force && !appState.isAutoTranslateEnabled) return;
    const activeBtn = document.querySelector<HTMLElement>(".translator-buttons button.active-translator");
    if (activeBtn) runTranslatorButton(activeBtn);
}
