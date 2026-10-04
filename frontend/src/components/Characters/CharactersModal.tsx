import { useEffect, useRef, useState } from "preact/hooks";
import {
    addCharacter,
    characterLabel,
    charactersToCsv,
    clearCharacters,
    deleteCharacter,
    downloadCharactersCsv,
    fetchCharacters,
    getCharacters,
    importCharacterRows,
    parseCharactersCsv,
    updateCharacter,
} from "../../characters";
import type { CharacterFields, CharacterInfo } from "../../types";
import { showConfirm } from "../common";
import { showNotify } from "../../ui";
import "./CharactersModal.css";

interface CharactersModalProps {
    open: boolean;
    onClose: () => void;
}

const EMPTY_DRAFT: CharacterFields = { name_en: "", name_ja: "", gender: "", alias_en: "", alias_ja: "" };

function draftOf(char: CharacterInfo): CharacterFields {
    return {
        name_en: char.name_en ?? "",
        name_ja: char.name_ja ?? "",
        gender: char.gender ?? "",
        alias_en: char.alias_en ?? "",
        alias_ja: char.alias_ja ?? "",
    };
}

function draftValid(draft: CharacterFields): boolean {
    return Boolean(
        (draft.name_en || "").trim() ||
        (draft.name_ja || "").trim() ||
        (draft.gender || "").trim() ||
        (draft.alias_en || "").trim() ||
        (draft.alias_ja || "").trim(),
    );
}

export function CharactersModal({ open, onClose }: CharactersModalProps) {
    const [rows, setRows] = useState<CharacterInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [adding, setAdding] = useState(false);
    const [draft, setDraft] = useState<CharacterFields>({ ...EMPTY_DRAFT });
    const [editingId, setEditingId] = useState<string | null>(null);
    const [editDraft, setEditDraft] = useState<CharacterFields>({ ...EMPTY_DRAFT });
    const [busyId, setBusyId] = useState<string | null>(null);
    const fileRef = useRef<HTMLInputElement | null>(null);

    useEffect(() => {
        if (!open) return;
        setError("");
        setLoading(true);
        fetchCharacters()
            .then((list) => {
                setRows(list);
                setError("");
            })
            .catch((err) => {
                // Offline backend: show the local mirror rather than a blank
                // modal, and say so once.
                setRows(getCharacters());
                setError(err instanceof Error ? err.message : "Could not load characters");
            })
            .finally(() => setLoading(false));
    }, [open]);

    useEffect(() => {
        if (!open) return;
        const onKey = (e: KeyboardEvent): void => {
            if (e.key === "Escape") onClose();
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [open, onClose]);

    if (!open) return null;

    const refresh = (list: CharacterInfo[]): void => setRows(list.map((c) => ({ ...c })));

    const set = (patch: Partial<CharacterFields>, which: "add" | "edit"): void => {
        if (which === "add") setDraft((d) => ({ ...d, ...patch }));
        else setEditDraft((d) => ({ ...d, ...patch }));
    };

    const doAdd = async (): Promise<void> => {
        if (!draftValid(draft)) {
            showNotify("⚠️ Give the character at least one name, alias or gender.");
            return;
        }
        setAdding(true);
        try {
            await addCharacter(draft);
            refresh(getCharacters());
            setDraft({ ...EMPTY_DRAFT });
        } catch (err) {
            showNotify(`❌ Could not add character: ${err instanceof Error ? err.message : err}`);
        } finally {
            setAdding(false);
        }
    };

    const startEdit = (char: CharacterInfo): void => {
        setEditingId(char.meta_id);
        setEditDraft(draftOf(char));
    };

    const doSave = async (meta_id: string): Promise<void> => {
        if (!draftValid(editDraft)) {
            showNotify("⚠️ A character needs at least one name, alias or gender.");
            return;
        }
        setBusyId(meta_id);
        try {
            await updateCharacter(meta_id, editDraft);
            refresh(getCharacters());
            setEditingId(null);
        } catch (err) {
            showNotify(`❌ Could not save character: ${err instanceof Error ? err.message : err}`);
        } finally {
            setBusyId(null);
        }
    };

    const doDelete = async (char: CharacterInfo): Promise<void> => {
        if (!(await showConfirm(`Delete "${characterLabel(char)}"? Entries using this speaker fall back to none.`)))
            return;
        setBusyId(char.meta_id);
        try {
            await deleteCharacter(char.meta_id);
            refresh(getCharacters());
            if (editingId === char.meta_id) setEditingId(null);
        } catch (err) {
            showNotify(`❌ Could not delete character: ${err instanceof Error ? err.message : err}`);
        } finally {
            setBusyId(null);
        }
    };

    const doClear = async (): Promise<void> => {
        if (!rows.length) return;
        if (
            !(await showConfirm(
                `Clear all ${rows.length} characters? Entries fall back to no speaker. Export first to keep a copy.`,
            ))
        )
            return;
        try {
            await clearCharacters();
            refresh([]);
            setEditingId(null);
            showNotify("Characters cleared.");
        } catch (err) {
            showNotify(`❌ Could not clear characters: ${err instanceof Error ? err.message : err}`);
        }
    };

    const doExport = (): void => {
        try {
            // Fresh text from the live rows (not the mirror) so an export
            // mid-edit still matches what is on screen.
            const text = charactersToCsv(rows);
            const blob = new Blob([text], { type: "text/csv;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "characters.csv";
            document.body.appendChild(link);
            link.click();
            link.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch {
            // Fall back to the mirror download.
            try {
                downloadCharactersCsv();
            } catch {
                showNotify("❌ Could not export characters.");
            }
        }
    };

    const doImportFile = async (file: File | undefined): Promise<void> => {
        if (!file) return;
        try {
            const text = await file.text();
            const parsed = parseCharactersCsv(text);
            if (!parsed.length) {
                showNotify("⚠️ No characters found in that CSV (header: name_en,name_ja,gender,alias_en,alias_ja).");
                return;
            }
            const list = await importCharacterRows(parsed);
            refresh(list);
            showNotify(`✅ Imported ${list.length} character${list.length === 1 ? "" : "s"} (new ids assigned).`);
        } catch (err) {
            showNotify(`❌ Could not import CSV: ${err instanceof Error ? err.message : err}`);
        } finally {
            if (fileRef.current) fileRef.current.value = "";
        }
    };

    const field = (
        value: string,
        placeholder: string,
        which: "add" | "edit",
        key: keyof CharacterFields,
        opts?: { wide?: boolean },
    ): preact.JSX.Element => (
        <input
            type="text"
            class={`ch-input${opts?.wide ? " is-wide" : ""}`}
            value={value}
            placeholder={placeholder}
            maxLength={100}
            onInput={(e) => set({ [key]: (e.target as HTMLInputElement).value } as Partial<CharacterFields>, which)}
        />
    );

    const genderSelect = (value: string, which: "add" | "edit"): preact.JSX.Element => (
        <select
            class="ch-input ch-gender"
            value={value}
            onChange={(e) => set({ gender: (e.target as HTMLSelectElement).value }, which)}
            title="Gender (steers pronouns; blank means unspecified)"
        >
            <option value="">—</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
        </select>
    );

    return (
        <div class="ch-overlay" onClick={onClose} role="presentation">
            <div
                class="ch-modal"
                role="dialog"
                aria-modal="true"
                aria-label="Characters"
                onClick={(e) => e.stopPropagation()}
            >
                <div class="ch-head">
                    <div class="ch-title">
                        <span class="material-icons" aria-hidden="true">
                            group
                        </span>
                        <span>Characters</span>
                        <span class="ch-count" title="Characters in the roster">
                            {rows.length}
                        </span>
                    </div>
                    <button type="button" class="ch-close" onClick={onClose} title="Close" aria-label="Close">
                        <span class="material-icons" aria-hidden="true">
                            close
                        </span>
                    </button>
                </div>

                <p class="ch-hint">
                    Roster for VNTL speaker tags. Ids assign on add/import; entries pointing at a removed character fall
                    back to none. Nothing is saved to disk -- export a CSV to keep a copy.
                </p>

                {error && <p class="ch-error">{error}</p>}

                <div class="ch-add">
                    {field(draft.name_en ?? "", "Name (EN)", "add", "name_en")}
                    {field(draft.name_ja ?? "", "Name (JA)", "add", "name_ja")}
                    {genderSelect(draft.gender ?? "", "add")}
                    {field(draft.alias_en ?? "", "Alias (EN)", "add", "alias_en")}
                    {field(draft.alias_ja ?? "", "Alias (JA)", "add", "alias_ja")}
                    <button type="button" class="ch-btn is-primary" disabled={adding} onClick={() => void doAdd()}>
                        {adding ? "Adding…" : "Add"}
                    </button>
                </div>

                <div class="ch-list custom-scrollbar">
                    {loading && <p class="ch-empty">Loading…</p>}
                    {!loading && !rows.length && (
                        <p class="ch-empty">No characters yet. Add one above or import a CSV.</p>
                    )}
                    {rows.map((char) => {
                        const editing = editingId === char.meta_id;
                        const busy = busyId === char.meta_id;
                        return (
                            <div class={editing ? "ch-row is-editing" : "ch-row"} key={char.meta_id}>
                                {editing ? (
                                    <>
                                        {field(editDraft.name_en ?? "", "Name (EN)", "edit", "name_en")}
                                        {field(editDraft.name_ja ?? "", "Name (JA)", "edit", "name_ja")}
                                        {genderSelect(editDraft.gender ?? "", "edit")}
                                        {field(editDraft.alias_en ?? "", "Alias (EN)", "edit", "alias_en")}
                                        {field(editDraft.alias_ja ?? "", "Alias (JA)", "edit", "alias_ja")}
                                        <span class="ch-actions">
                                            <button
                                                type="button"
                                                class="ch-btn is-primary"
                                                disabled={busy}
                                                onClick={() => void doSave(char.meta_id)}
                                            >
                                                Save
                                            </button>
                                            <button
                                                type="button"
                                                class="ch-btn"
                                                disabled={busy}
                                                onClick={() => setEditingId(null)}
                                            >
                                                Cancel
                                            </button>
                                        </span>
                                    </>
                                ) : (
                                    <>
                                        <span class="ch-name" title={char.meta_id}>
                                            {characterLabel(char)}
                                        </span>
                                        <span class="ch-sub">
                                            {[
                                                char.name_ja,
                                                char.gender ? char.gender[0].toUpperCase() + char.gender.slice(1) : "",
                                                char.alias_en,
                                                char.alias_ja,
                                            ]
                                                .filter((part) => (part || "").trim())
                                                .join(" · ")}
                                        </span>
                                        <span class="ch-actions">
                                            <button
                                                type="button"
                                                class="ch-btn"
                                                disabled={busy}
                                                onClick={() => startEdit(char)}
                                            >
                                                Edit
                                            </button>
                                            <button
                                                type="button"
                                                class="ch-btn is-danger"
                                                disabled={busy}
                                                onClick={() => void doDelete(char)}
                                            >
                                                Delete
                                            </button>
                                        </span>
                                    </>
                                )}
                            </div>
                        );
                    })}
                </div>

                <div class="ch-foot">
                    <input
                        ref={fileRef}
                        type="file"
                        accept=".csv,text/csv"
                        hidden
                        onChange={(e) => void doImportFile((e.target as HTMLInputElement).files?.[0])}
                    />
                    <button
                        type="button"
                        class="ch-btn"
                        onClick={() => fileRef.current?.click()}
                        title="Import name_en,name_ja,gender,alias_en,alias_ja (new ids assigned)"
                    >
                        <span class="material-icons" aria-hidden="true">
                            upload
                        </span>{" "}
                        Import CSV
                    </button>
                    <button type="button" class="ch-btn" onClick={doExport} title="Download the roster as CSV">
                        <span class="material-icons" aria-hidden="true">
                            download
                        </span>{" "}
                        Export CSV
                    </button>
                    <span class="ch-spacer" />
                    <button
                        type="button"
                        class="ch-btn is-danger"
                        disabled={!rows.length}
                        onClick={() => void doClear()}
                    >
                        Clear
                    </button>
                    <button type="button" class="ch-btn is-primary" onClick={onClose}>
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
}
