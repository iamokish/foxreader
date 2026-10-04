import { useEffect, useRef, useState } from "preact/hooks";
import "./folder.css";
import * as api from "../../api";
import type { DirReport, FolderInspectResponse } from "../../types";

/** What a destination subfolder is called when the user does not name one. */
export const DEFAULT_DEST_NAME = "fox_tled";

/** Other names offered as one-click destinations. */
const DEST_PRESETS = [DEFAULT_DEST_NAME, "typeset", "output"];

/** How long after the last keystroke the paths are checked. */
const INSPECT_DEBOUNCE = 250;

export interface FolderPickerRequest {
    /** Prefill for the source field. */
    source?: string;
    /** Called with the two resolved paths when the user presses Load. */
    onSubmit: (source: string, dest: string) => void | Promise<void>;
}

type Opener = (request: FolderPickerRequest) => void;

let openImpl: Opener | null = null;
let isOpen = false;

/**
 * Open the picker. No-op until `<FolderModal />` has mounted.
 *
 * The submit handler is passed in rather than imported so this component knows
 * nothing about what loading a folder involves.
 */
export function openFolderModal(request: FolderPickerRequest): void {
    openImpl?.(request);
}

/**
 * Whether the picker is on screen.
 *
 * The window-level hotkeys read this: with focus on one of the dialog's buttons
 * rather than a text field, `c` would otherwise start a capture on the page
 * behind the dialog.
 */
export function isFolderModalOpen(): boolean {
    return isOpen;
}

/** The tidy-up the backend does, repeated here so the field reacts immediately. */
function cleanPath(raw: string): string {
    return raw
        .trim()
        .replace(/^['"]+|['"]+$/g, "")
        .trim()
        .replace(/\\/g, "/")
        .replace(/\/+$/, "");
}

/** `<source>/<name>`, the shape of every offered destination. */
export function subfolderOf(source: string, name: string): string {
    const base = cleanPath(source);
    return base ? `${base}/${name}` : "";
}

/**
 * Whether this pair may be loaded.
 *
 * Exported for its own sake: this is the whole gate on the Load button, and it is
 * worth being able to test it without a DOM.
 */
export function canLoadFolder(report: FolderInspectResponse | null): boolean {
    if (!report) return false;
    const images = report.source.image_count ?? 0;
    return Boolean(report.source.ok_as_source) && images > 0 && Boolean(report.dest.ok_as_dest);
}

type Tone = "ok" | "bad" | "warn" | "muted";

const TONE_ICONS: Record<Tone, string> = {
    ok: "check_circle",
    bad: "error",
    warn: "warning",
    muted: "info",
};

function Status({ tone, text }: { tone: Tone; text: string }) {
    return (
        <div class={`folder-modal__status is-${tone}`}>
            <span class="material-icons">{TONE_ICONS[tone]}</span>
            <span>{text}</span>
        </div>
    );
}

function Perms({ report }: { report: DirReport }) {
    const flags: [string, boolean][] = [
        ["Read", report.readable],
        ["Write", report.writable],
        ["Open", report.executable],
    ];
    return (
        <div class="folder-modal__perms">
            {flags.map(([label, on]) => (
                <span key={label} class={`folder-modal__perm ${on ? "is-on" : "is-off"}`}>
                    <span class="material-icons">{on ? "done" : "block"}</span>
                    {label}
                </span>
            ))}
        </div>
    );
}

/** The verdict on the source folder, in the order the user cares about it. */
function sourceVerdict(report: DirReport | undefined, checking: boolean, typed: string): [Tone, string] {
    if (!typed.trim()) return ["muted", "Paste or type the folder holding the pages."];
    if (checking || !report) return ["muted", "Checking…"];
    if (!report.exists) return ["bad", report.reason || "This folder does not exist."];
    if (report.is_file) return ["bad", "This is a file, not a folder."];
    if (!report.readable) return ["bad", report.reason || "This folder cannot be read."];
    if (!report.executable) return ["bad", report.reason || "This folder cannot be opened."];
    const count = report.image_count ?? 0;
    if (count === 0) return ["bad", "No images in this folder."];
    return ["ok", `${count} image${count === 1 ? "" : "s"} found.`];
}

/** The verdict on the destination, including the "will be created" case. */
function destVerdict(report: DirReport | undefined, checking: boolean, sameDir: boolean): [Tone, string] {
    if (checking || !report) return ["muted", "Checking…"];
    if (!report.path) return ["muted", "Enter a folder for the typeset pages."];
    if (sameDir) return ["warn", "This is the source folder — saving replaces the original pages."];
    if (report.is_file) return ["bad", "This is a file, not a folder."];
    if (!report.exists) {
        return report.can_create
            ? ["ok", "Does not exist yet — it will be created when you load."]
            : ["bad", report.reason || "This folder cannot be created."];
    }
    if (!report.writable) return ["bad", report.reason || "This folder cannot be written to."];
    return ["ok", "Writable — typeset pages will be saved here."];
}

export function FolderModal() {
    const [open, setOpen] = useState(false);
    const [source, setSource] = useState("");
    const [dest, setDest] = useState("");
    // Until the user edits the destination it follows the source, so retyping the
    // source does not leave a stale output path behind.
    const [destTouched, setDestTouched] = useState(false);
    const [report, setReport] = useState<FolderInspectResponse | null>(null);
    const [checking, setChecking] = useState(false);
    const [failed, setFailed] = useState("");

    const submitRef = useRef<FolderPickerRequest["onSubmit"] | null>(null);
    const sourceRef = useRef<HTMLInputElement | null>(null);

    useEffect(() => {
        openImpl = (request) => {
            submitRef.current = request.onSubmit;
            setSource(request.source ?? "");
            setDest("");
            setDestTouched(false);
            setReport(null);
            setFailed("");
            setOpen(true);
        };
        return () => {
            openImpl = null;
        };
    }, []);

    useEffect(() => {
        isOpen = open;
        if (!open) return;
        // After the browser has laid the dialog out, or the focus lands nowhere.
        const raf = requestAnimationFrame(() => {
            sourceRef.current?.focus();
            sourceRef.current?.select();
        });
        return () => cancelAnimationFrame(raf);
    }, [open]);

    // Debounced, and re-run when either field changes. The endpoint is read-only,
    // so the cost of an extra call is a listdir and one write probe.
    useEffect(() => {
        if (!open) return;
        const typed = source.trim();
        if (!typed) {
            setReport(null);
            setChecking(false);
            return;
        }

        let alive = true;
        setChecking(true);
        const timer = setTimeout(async () => {
            try {
                const answer = await api.inspectFolder({
                    source: typed,
                    // Empty asks the backend for its own suggestion: the folder
                    // this source was loaded with last time, else <source>/fox_tled.
                    dest: destTouched ? dest.trim() : "",
                });
                if (!alive) return;
                setReport(answer);
                setFailed("");
            } catch (err) {
                if (!alive) return;
                setReport(null);
                setFailed(err instanceof Error && err.message ? err.message : "The server did not answer.");
            } finally {
                if (alive) setChecking(false);
            }
        }, INSPECT_DEBOUNCE);

        return () => {
            alive = false;
            clearTimeout(timer);
        };
    }, [open, source, dest, destTouched]);

    if (!open) return null;

    // What the destination field shows: the user's text once they have touched
    // it, otherwise the backend's answer, with a local guess to fill the gap
    // before the first check comes back.
    const shownDest = destTouched ? dest : report?.dest.path || subfolderOf(source, DEFAULT_DEST_NAME);
    const [sourceTone, sourceText] = sourceVerdict(report?.source, checking, source);
    const [destTone, destText] = destVerdict(report?.dest, checking, Boolean(report?.same_dir));
    const ready = !checking && canLoadFolder(report);

    const setPreset = (name: string) => {
        setDestTouched(true);
        setDest(subfolderOf(source, name));
    };

    const close = () => {
        setOpen(false);
        submitRef.current = null;
    };

    const submit = () => {
        if (!ready || !report) return;
        const handler = submitRef.current;
        const chosen = { source: report.source.path, dest: report.dest.path };
        close();
        void handler?.(chosen.source, chosen.dest);
    };

    return (
        <div
            id="folderModalOverlay"
            class="folder-modal custom-scrollbar"
            style="display: flex;"
            onKeyDown={(e) => {
                if (e.key === "Escape") {
                    e.preventDefault();
                    close();
                } else if (e.key === "Enter") {
                    e.preventDefault();
                    submit();
                }
            }}
        >
            <div class="folder-modal__panel" role="dialog" aria-modal="true" aria-label="Load folder">
                <h2 class="folder-modal__title">
                    <span class="material-icons">folder_open</span>
                    Load folder
                </h2>
                <p class="folder-modal__subtitle">
                    Pages are read from the source folder and never written to. Typeset pages are saved into the
                    destination folder.
                </p>

                <div class="folder-modal__field">
                    <label class="folder-modal__label" for="folderModalSource">
                        Source folder
                        <span class="folder-modal__hint">the pages to translate</span>
                    </label>
                    <input
                        id="folderModalSource"
                        ref={sourceRef}
                        class="folder-modal__input"
                        type="text"
                        spellcheck={false}
                        autocomplete="off"
                        placeholder="X:\My Comic\Chapter 1"
                        value={source}
                        onInput={(e) => setSource((e.target as HTMLInputElement).value)}
                    />
                    <Status tone={sourceTone} text={sourceText} />
                    {report?.source && report.source.exists && report.source.is_dir && (
                        <Perms report={report.source} />
                    )}
                </div>

                <div class="folder-modal__divider" />

                <div class="folder-modal__field">
                    <label class="folder-modal__label" for="folderModalDest">
                        Destination folder
                        <span class="folder-modal__hint">where saves go</span>
                    </label>
                    <input
                        id="folderModalDest"
                        class="folder-modal__input"
                        type="text"
                        spellcheck={false}
                        autocomplete="off"
                        placeholder={subfolderOf(source, DEFAULT_DEST_NAME) || "…"}
                        value={shownDest}
                        onInput={(e) => {
                            setDestTouched(true);
                            setDest((e.target as HTMLInputElement).value);
                        }}
                    />
                    <div class="folder-modal__quick">
                        {DEST_PRESETS.map((name) => (
                            <button
                                key={name}
                                type="button"
                                class="folder-modal__chip"
                                disabled={!source.trim()}
                                onClick={() => setPreset(name)}
                            >
                                {name}
                            </button>
                        ))}
                        <button
                            type="button"
                            class="folder-modal__chip"
                            disabled={!source.trim()}
                            onClick={() => {
                                setDestTouched(true);
                                setDest(report?.source.path || cleanPath(source));
                            }}
                        >
                            Same as source
                        </button>
                        {destTouched && (
                            <button
                                type="button"
                                class="folder-modal__chip"
                                onClick={() => {
                                    setDestTouched(false);
                                    setDest("");
                                }}
                            >
                                Reset
                            </button>
                        )}
                    </div>
                    <Status tone={destTone} text={destText} />
                    {report?.dest && report.dest.exists && report.dest.is_dir && !report.same_dir && (
                        <Perms report={report.dest} />
                    )}
                </div>

                {report?.same_dir && (
                    <div class="folder-modal__note">
                        Saving will overwrite the original pages in this folder. Choose a different destination if you
                        want to keep them.
                    </div>
                )}
                {!report?.same_dir && report?.remembered_dest && !destTouched && (
                    <div class="folder-modal__note">This folder was last saved into {report.remembered_dest}.</div>
                )}
                {failed && <Status tone="bad" text={failed} />}

                <div class="folder-modal__actions">
                    <span class="folder-modal__spacer">Enter to load · Esc to cancel</span>
                    <button id="folderModalCancel" type="button" class="btn-secondary" onClick={close}>
                        Cancel
                    </button>
                    <button
                        id="folderModalLoad"
                        type="button"
                        class="btn-primary"
                        disabled={!ready}
                        onClick={submit}
                    >
                        <span class="material-icons" style="font-size: 16px;">
                            folder_open
                        </span>
                        Load
                    </button>
                </div>
            </div>
        </div>
    );
}
