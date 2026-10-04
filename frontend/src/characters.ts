/**
 * Character roster: the in-memory mirror of the backend's roster.
 *
 * The backend (`fox_reader.translate.characters.store`) is the source of
 * truth for ids; this module mirrors it for the entry selectors and the
 * translate payload, and owns the CSV parse/generate. Nothing here touches
 * disk except through a user-triggered download/upload: a restart starts
 * empty and the user re-imports their file.
 */

import { state } from "./state";
import type { CharacterFields, CharacterInfo } from "./types";

export type { CharacterFields, CharacterInfo };

export const CSV_FIELDS = ["name_en", "name_ja", "gender", "alias_en", "alias_ja"] as const;

export const MAX_CHARACTERS = 50;
export const MAX_FIELD_CHARS = 100;

let roster: CharacterInfo[] = [];
const listeners = new Set<() => void>();

function notify(): void {
    for (const fn of Array.from(listeners)) {
        try {
            fn();
        } catch {
            /* a stale subscriber must not break the roster */
        }
    }
}

export function subscribeCharacters(fn: () => void): () => void {
    listeners.add(fn);
    return () => {
        listeners.delete(fn);
    };
}

export function getCharacters(): CharacterInfo[] {
    return roster.map((c) => ({ ...c }));
}

function setRoster(next: CharacterInfo[]): void {
    roster = (Array.isArray(next) ? next : [])
        .filter((c) => c && typeof c.meta_id === "string" && c.meta_id.trim())
        .map((c) => ({
            meta_id: c.meta_id.trim(),
            name_en: c.name_en?.trim() || null,
            name_ja: c.name_ja?.trim() || null,
            gender:
                c.gender?.trim().toLowerCase() === "male" || c.gender?.trim().toLowerCase() === "female"
                    ? c.gender.trim().toLowerCase()
                    : null,
            alias_en: c.alias_en?.trim() || null,
            alias_ja: c.alias_ja?.trim() || null,
        }));
    resetDanglingEntries();
    notify();
}

/** Entries pointing at a deleted/unknown character read as "none". */
function resetDanglingEntries(): void {
    try {
        const known = new Set(roster.map((c) => c.meta_id));
        for (const entries of Object.values(state.pageEntriesCache)) {
            if (!Array.isArray(entries)) continue;
            for (const entry of entries) {
                const id = (entry as { character_id?: string | null }).character_id;
                if (id !== null && id !== undefined && id !== "" && !known.has(id)) {
                    (entry as { character_id?: string | null }).character_id = null;
                }
            }
        }
    } catch {
        /* the cache is best-effort; the selectors fall back to none anyway */
    }
}

/** Human label for selectors: EN name first, then JA, then alias, then id. */
export function characterLabel(char: CharacterInfo): string {
    const name = (char.name_en || "").trim() || (char.name_ja || "").trim();
    if (name) return name;
    const alias = (char.alias_en || "").trim() || (char.alias_ja || "").trim();
    if (alias) return alias;
    return `Character ${String(char.meta_id || "").slice(0, 6) || "?"}`;
}

/** Gender symbols for speaker pickers: visual info only, never sent anywhere. */
export const GENDER_SYMBOLS: Record<string, string> = {
    male: "♂",
    female: "♀",
};

/**
 * Speaker-picker label: the name plus the gender symbol when one is known
 * (`Eren (♂)`), so entries sharing a name stay distinguishable at a glance.
 * Unspecified gender reads as the bare name. The symbol is display-only: the
 * option value stays the `meta_id`, and translation tags are unaffected.
 */
export function characterPickerLabel(char: CharacterInfo): string {
    const base = characterLabel(char);
    const symbol = GENDER_SYMBOLS[(char.gender || "").trim().toLowerCase()];
    return symbol ? `${base} (${symbol})` : base;
}

function asJson(value: unknown): Record<string, CharacterInfo[]> | null {
    if (value && typeof value === "object" && Array.isArray((value as { characters?: unknown }).characters)) {
        return value as Record<string, CharacterInfo[]>;
    }
    return null;
}

// ------------------------------------------------------------------ backend

export async function fetchCharacters(): Promise<CharacterInfo[]> {
    const res = await fetch("/api/characters", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = asJson(await res.json());
    setRoster(data?.characters ?? []);
    return getCharacters();
}

export async function addCharacter(fields: CharacterFields): Promise<CharacterInfo> {
    const res = await fetch("/api/characters", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields ?? {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data as { error?: string }).error || `HTTP ${res.status}`);
    const created = (data as { character?: CharacterInfo }).character;
    if (!created) throw new Error("Backend did not return the character");
    setRoster([...roster, created]);
    return created;
}

export async function updateCharacter(meta_id: string, fields: CharacterFields): Promise<CharacterInfo> {
    const res = await fetch(`/api/characters/${encodeURIComponent(meta_id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields ?? {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data as { error?: string }).error || `HTTP ${res.status}`);
    const updated = (data as { character?: CharacterInfo }).character;
    if (!updated) throw new Error("Backend did not return the character");
    setRoster(roster.map((c) => (c.meta_id === meta_id ? updated : c)));
    return updated;
}

export async function deleteCharacter(meta_id: string): Promise<void> {
    const res = await fetch(`/api/characters/${encodeURIComponent(meta_id)}`, { method: "DELETE" });
    if (!res.ok && res.status !== 404) {
        const data = await res.json().catch(() => ({}));
        throw new Error((data as { error?: string }).error || `HTTP ${res.status}`);
    }
    setRoster(roster.filter((c) => c.meta_id !== meta_id));
}

/** Drop the whole roster (called when the source folder changes). Never throws. */
export async function clearCharacters(): Promise<void> {
    try {
        await fetch("/api/characters", { method: "DELETE" });
    } catch {
        /* backend down: the local mirror still clears */
    }
    setRoster([]);
}

export async function importCharacterRows(rows: CharacterFields[]): Promise<CharacterInfo[]> {
    const res = await fetch("/api/characters/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ characters: rows ?? [] }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data as { error?: string }).error || `HTTP ${res.status}`);
    setRoster((data as { characters?: CharacterInfo[] }).characters ?? []);
    return getCharacters();
}

// ---------------------------------------------------------------------- CSV

function cleanField(value: unknown): string {
    if (value === null || value === undefined) return "";
    if (typeof value !== "string") value = String(value);
    let text = (value as string).trim().replace(/\s+/g, " ");
    if (text.length > MAX_FIELD_CHARS) text = text.slice(0, MAX_FIELD_CHARS).trim();
    return text;
}

function cleanGender(value: unknown): string {
    const text = cleanField(value).toLowerCase();
    return text === "male" || text === "female" ? text : "";
}

/** Parse roster CSV (header `name_en,name_ja,gender,alias_en,alias_ja`). */
export function parseCharactersCsv(text: string): CharacterFields[] {
    const content = (text ?? "").replace(/^\uFEFF/, "");
    if (!content.trim()) return [];
    const lines = content.split(/\r?\n/);
    if (!lines.length) return [];
    const split = (line: string): string[] => {
        // Minimal RFC-4180: quotes, doubled quotes, commas inside quotes.
        const out: string[] = [];
        let cur = "";
        let quoted = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (quoted) {
                if (ch === '"') {
                    if (line[i + 1] === '"') {
                        cur += '"';
                        i++;
                    } else {
                        quoted = false;
                    }
                } else {
                    cur += ch;
                }
            } else if (ch === '"') {
                quoted = true;
            } else if (ch === ",") {
                out.push(cur);
                cur = "";
            } else {
                cur += ch;
            }
        }
        out.push(cur);
        return out;
    };
    const header = split(lines[0]).map((h) => h.trim().toLowerCase());
    const indexOf = (name: string): number => header.indexOf(name);
    const rows: CharacterFields[] = [];
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const cells = split(lines[i]);
        const cell = (name: string): string => {
            const idx = indexOf(name);
            return idx < 0 ? "" : (cells[idx] ?? "").trim();
        };
        const row: CharacterFields = {
            name_en: cleanField(cell("name_en")),
            name_ja: cleanField(cell("name_ja")),
            gender: cleanGender(cell("gender")),
            alias_en: cleanField(cell("alias_en")),
            alias_ja: cleanField(cell("alias_ja")),
        };
        if (!row.name_en && !row.name_ja && !row.gender && !row.alias_en && !row.alias_ja) continue;
        rows.push(row);
        if (rows.length >= MAX_CHARACTERS) break;
    }
    return rows;
}

function csvCell(value: string): string {
    if (/[",\n\r]/.test(value)) return `"${value.replace(/"/g, '""')}"`;
    return value;
}

/** Roster as CSV text (header, no `meta_id` -- ids regenerate on import). */
export function charactersToCsv(chars: CharacterInfo[] | CharacterFields[]): string {
    const lines = [CSV_FIELDS.join(",")];
    for (const c of chars ?? []) {
        const row = c as Record<string, unknown>;
        lines.push(
            [
                csvCell(cleanField(row.name_en)),
                csvCell(cleanField(row.name_ja)),
                csvCell(cleanGender(row.gender)),
                csvCell(cleanField(row.alias_en)),
                csvCell(cleanField(row.alias_ja)),
            ].join(","),
        );
    }
    return `${lines.join("\n")}\n`;
}

/** Download the roster as `characters.csv`. */
export function downloadCharactersCsv(): void {
    const blob = new Blob([charactersToCsv(roster)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    try {
        const link = document.createElement("a");
        link.href = url;
        link.download = "characters.csv";
        document.body.appendChild(link);
        link.click();
        link.remove();
    } finally {
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
}
