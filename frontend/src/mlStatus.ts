import { mlStatus } from "./api";
import { refreshTranslatorButtons } from "./translate";

/** Normalize the backend's language list: lowercase, trimmed, de-duplicated. */
function asLangList(value: unknown): string[] {
    if (!Array.isArray(value)) return [];
    const out: string[] = [];
    for (const item of value) {
        if (typeof item !== "string") continue;
        const lang = item.trim().toLowerCase();
        if (lang && !out.includes(lang)) out.push(lang);
    }
    return out;
}

/**
 * Reconcile the MTL switch and the language select with the model the
 * backend still has loaded after a page refresh.
 *
 * Rules (all best-effort, in order):
 * - nothing loaded, or anything unreadable: leave the UI exactly as it is;
 * - loaded: re-check `#enableML` so the MTL buttons work again;
 * - loaded with known languages: set `#languageSelect` to the first option,
 *   in dropdown order, that the model supports; when nothing matches, only
 *   the checkbox is restored and the select is left alone.
 *
 * Never throws and never notifies: a backend that is down, slow, or answering
 * oddly must not break page startup, and the user can always flip the switch
 * by hand. A switch the user already touched (or a load in flight) always
 * wins over this sync, and the select is updated before the box is checked so
 * the language-change handler does not fire a redundant model reload.
 */
export async function syncMLStateFromBackend(): Promise<void> {
    let checkbox: HTMLInputElement | null = null;
    let select: HTMLSelectElement | null = null;

    try {
        checkbox = document.getElementById("enableML") as HTMLInputElement | null;
        select = document.getElementById("languageSelect") as HTMLSelectElement | null;
    } catch {
        return;
    }

    if (!checkbox || !select) return;
    if (checkbox.checked || checkbox.disabled) return;

    let data: Awaited<ReturnType<typeof mlStatus>> | null = null;

    try {
        data = await mlStatus();
    } catch {
        return;
    }

    if (!data || typeof data !== "object" || data.loaded !== true) return;

    try {
        const supported = new Set(asLangList(data.languages));
        if (supported.size > 0) {
            const match = Array.from(select.options).find((option) =>
                supported.has((option.value ?? "").trim().toLowerCase()),
            );
            if (match && select.value !== match.value) {
                select.value = match.value;
                try {
                    select.dispatchEvent(new Event("change", { bubbles: true }));
                } catch {
                    /* a listener-free select still holds the value */
                }
                try {
                    refreshTranslatorButtons(select.value);
                } catch {
                    /* the delegated click handler filters on next render */
                }
            }
        }
    } catch {
        /* an unreadable select keeps its value; the checkbox is what matters */
    }

    try {
        checkbox.disabled = false;
        checkbox.checked = true;
    } catch {
        /* a non-input element with this id: nothing to sync */
    }
}
