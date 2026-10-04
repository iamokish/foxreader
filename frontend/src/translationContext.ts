/**
 * Translation context: the already-translated entries before the one being
 * translated, as [[source, english], ...] pairs in list (reading) order.
 *
 * Only previous entries with *both* sides present count -- an untranslated
 * predecessor is skipped, the rest are kept -- so the first entry naturally
 * yields no context. The backend treats an empty list exactly like that and
 * only renders context for models that support it.
 */

import { state } from "./state";
import { getCharacters } from "./characters";
import type { CharacterInfo } from "./types";

/**
 * How many previous pairs travel with a request at most. The single obvious
 * knob: recent context matters most, and the prompt window is finite.
 */
export const MAX_CONTEXT_PAIRS = 10;

/**
 * Pairs for translating the target entry: the `MAX_CONTEXT_PAIRS` newest
 * valid pairs strictly before it. Unknown entry, no page, or nothing
 * translated yet all read as "no context" rather than throwing.
 */
export function buildTranslationContext(targetEntryId: string | null | undefined): [string, string][] {
    if (!targetEntryId || !state.currentImageFile) return [];
    const entries = state.pageEntriesCache[state.currentImageFile];
    if (!Array.isArray(entries)) return [];

    const target = entries.findIndex((e) => e?.id === targetEntryId);
    if (target < 0) return [];

    const pairs: [string, string][] = [];
    for (let i = 0; i < target; i++) {
        const entry = entries[i];
        // Unchecked (hidden) entries are out of the render and out of the
        // story: they contribute no pair, exactly like untranslated ones.
        if (!entry || !entry.visible) continue;
        const source = (entry.ocr_text ?? "").trim();
        const english = (entry.text ?? "").trim();
        if (source && english) pairs.push([source, english]);
    }
    return pairs.length > MAX_CONTEXT_PAIRS ? pairs.slice(-MAX_CONTEXT_PAIRS) : pairs;
}

/**
 * Character payload for translating the target entry.
 *
 * Returns the roster snapshot plus per-turn speaker links aligned to the
 * pairs {@link buildTranslationContext} produced, and the target's own
 * speaker. Unknown/deleted ids read as null (no tag) rather than throwing,
 * and an empty roster or a target with no speaker yields empty links plus a
 * null `meta_id` -- the backend then translates untagged, exactly like
 * context-off. Callers gate on `state.isCharactersEnabled` themselves.
 */
export function buildTranslationCharacters(targetEntryId: string | null | undefined): {
    character_info: CharacterInfo[];
    context_character_links: (string | null)[];
    meta_id: string | null;
} {
    const empty = {
        character_info: [] as CharacterInfo[],
        context_character_links: [] as (string | null)[],
        meta_id: null as string | null,
    };
    if (!targetEntryId || !state.currentImageFile) return empty;
    const entries = state.pageEntriesCache[state.currentImageFile];
    if (!Array.isArray(entries)) return empty;

    const roster = getCharacters();
    if (!roster.length) return empty;
    const known = new Set(roster.map((c) => c.meta_id));

    const target = entries.findIndex((e) => e?.id === targetEntryId);
    if (target < 0) return { character_info: roster, context_character_links: [], meta_id: null };

    // Links align to the surviving pairs (same skip rules as the pairs: only
    // visible entries with both sides, strictly before the target).
    const links: (string | null)[] = [];
    for (let i = 0; i < target; i++) {
        const entry = entries[i];
        if (!entry || !entry.visible) continue;
        const source = (entry.ocr_text ?? "").trim();
        const english = (entry.text ?? "").trim();
        if (!source || !english) continue;
        const id = (entry as { character_id?: string | null }).character_id ?? null;
        links.push(id && known.has(id) ? id : null);
    }
    const trimmed = links.length > MAX_CONTEXT_PAIRS ? links.slice(-MAX_CONTEXT_PAIRS) : links;

    const speakerRaw = (entries[target] as { character_id?: string | null } | undefined)?.character_id ?? null;
    const speaker = speakerRaw && known.has(speakerRaw) ? speakerRaw : null;
    return { character_info: roster, context_character_links: trimmed, meta_id: speaker };
}
