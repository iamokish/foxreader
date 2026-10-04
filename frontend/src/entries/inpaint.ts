/**
 * Preview and Generate: progress reporting, the preview modal, and saving.
 *
 * The backend publishes both jobs through `/api/progress/{name}` while they run,
 * so the modal shows what stage the page is in rather than an anonymous spinner.
 * The preview it produces is *kept* server-side under a token, which is what
 * makes "Save this image" cheap: it commits the bytes the user is looking at
 * instead of typesetting the page a second time and hoping for the same result.
 *
 * Both saves write into the destination folder chosen when the folder was loaded,
 * so the original scans survive and a page can be re-saved from its entries as
 * many times as it takes. The only wording that still warns about destroying the
 * original is the case where the user pointed the destination at the source.
 */

import { state } from "../state";
import type { ProcessImageRequest, ProgressSnapshot } from "../types";
import * as api from "../api";
import { hideLoading, showLoading, showNotify, updateLoadingText } from "../ui";
import { showConfirm } from "../components/common";
import { markPageSaved, setViewVariant } from "../viewer";

/** How often a running job is polled. Slow enough to be free, fast enough to read. */
const POLL_INTERVAL = 450;

/** How many log lines the modal keeps on screen. */
const LOG_TAIL = 8;

export interface InpaintDeps {
    /** The payload for the current page, or `null` when there is nothing to do. */
    buildPayload: () => ProcessImageRequest | null;
    /** Re-render the entry list and overlay after a save. */
    refresh: () => void;
}

// -------------------------------------------------------------------- progress

/**
 * Poll a backend job until it stops running. Returns a stop function.
 *
 * Each tick is chained with `setTimeout` rather than scheduled on an interval:
 * an interval whose handler outlives its period stacks up overlapping requests,
 * and the health check is the first thing that suffers.
 */
export function pollProgress(
    name: string,
    onSnapshot: (snap: ProgressSnapshot) => void,
    interval = POLL_INTERVAL,
): () => void {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
        if (stopped) return;
        // `api.getProgress` swallows its own failures, so a dropped poll leaves
        // the previous message on screen and the loop keeps going.
        const snap = await api.getProgress(name);
        if (stopped) return;
        if (snap && !snap.idle) onSnapshot(snap);
        timer = setTimeout(tick, interval);
    };

    void tick();

    return () => {
        stopped = true;
        if (timer !== null) clearTimeout(timer);
    };
}

// ----------------------------------------------------------------------- modal

let stopPolling: (() => void) | null = null;
let previewToken = "";
let previewFilename = "";
let previewObjectURL: string | null = null;

const el = <T extends HTMLElement>(id: string): T | null => document.getElementById(id) as T | null;

function setModalStatus(visible: boolean): void {
    const status = el("inpaintModalStatus");
    if (status) status.style.display = visible ? "flex" : "none";
}

function setPreviewActions(visible: boolean): void {
    const actions = el("inpaintPreviewActions");
    if (actions) actions.style.display = visible ? "flex" : "none";
}

/** Mirror one snapshot into the modal's message, bar, counter and log. */
function renderSnapshot(snap: ProgressSnapshot): void {
    const message = el("inpaintProgressMessage");
    if (message) message.textContent = snap.message || "Working...";

    const bar = el("inpaintProgressFill");
    const wrap = bar?.parentElement;
    if (bar) {
        if (snap.total > 0) {
            wrap?.classList.remove("is-indeterminate");
            const pct = Math.max(0, Math.min(100, (snap.done / snap.total) * 100));
            bar.style.width = `${pct.toFixed(1)}%`;
        } else {
            // No total yet: a stripe that moves says "running" without implying
            // a fraction the backend has not committed to.
            wrap?.classList.add("is-indeterminate");
            bar.style.width = "100%";
        }
    }

    const count = el("inpaintProgressCount");
    if (count) {
        const parts: string[] = [];
        if (snap.total > 0) parts.push(`${snap.done}/${snap.total}`);
        if (snap.elapsed) parts.push(`${snap.elapsed.toFixed(1)}s`);
        count.textContent = parts.join(" · ");
    }

    const log = el("inpaintProgressLog");
    if (log && snap.lines?.length) {
        log.textContent = snap.lines.slice(-LOG_TAIL).join("\n");
        log.scrollTop = log.scrollHeight;
    }
}

/** Leave the failure on screen: closing the modal would take the reason with it. */
function renderFailure(reason: string): void {
    setModalStatus(true);
    setPreviewActions(false);
    const loader = el("inpaintModalLoader");
    if (loader) loader.style.display = "none";
    const message = el("inpaintProgressMessage");
    if (message) {
        message.textContent = reason;
        message.classList.add("is-error");
    }
    const wrap = el("inpaintProgressFill")?.parentElement;
    wrap?.classList.remove("is-indeterminate");
}

function openInpaintModal(): void {
    const overlay = el("inpaintModalOverlay");
    const img = el<HTMLImageElement>("inpaintModalImage");
    const loader = el("inpaintModalLoader");

    releaseObjectURL();
    if (img) {
        img.style.display = "none";
        img.removeAttribute("src");
    }
    if (loader) loader.style.display = "block";

    const message = el("inpaintProgressMessage");
    if (message) {
        message.textContent = "Preparing...";
        message.classList.remove("is-error");
    }
    const bar = el("inpaintProgressFill");
    if (bar) bar.style.width = "0%";
    bar?.parentElement?.classList.add("is-indeterminate");
    const count = el("inpaintProgressCount");
    if (count) count.textContent = "";
    const log = el("inpaintProgressLog");
    if (log) log.textContent = "";

    setModalStatus(true);
    setPreviewActions(false);
    if (overlay) overlay.style.display = "flex";
}

function releaseObjectURL(): void {
    if (previewObjectURL) {
        URL.revokeObjectURL(previewObjectURL);
        previewObjectURL = null;
    }
}

export function closeInpaintModal(): void {
    stopPolling?.();
    stopPolling = null;
    releaseObjectURL();
    const overlay = el("inpaintModalOverlay");
    if (overlay) overlay.style.display = "none";
    const img = el<HTMLImageElement>("inpaintModalImage");
    if (img) img.removeAttribute("src");
}

// -------------------------------------------------------------- saving to disk

/** Where saves land, phrased for a dialog. */
function destLabel(): string {
    return state.destDir || state.sourceDir || "the loaded folder";
}

/** What the user agreed to in `confirmSave`, if anything. */
type SaveChoice = "no" | "save" | "overwrite";

/**
 * Ask before writing, in the terms that actually apply.
 *
 * Three different questions hide behind the one button. Saving into a separate
 * destination is harmless and only needs to name where it goes; replacing a page
 * already saved there is worth a red button; and a destination that *is* the
 * source still destroys the original scan, which is the strongest warning of the
 * three and the only case that used to exist.
 *
 * Returns `"overwrite"` rather than a plain yes for the two cases that knowingly
 * replace a file, so the request can carry the permission on its *first* attempt
 * -- sending `overwrite: false` after the user has already agreed just earns a
 * 409 and asks them the same question a second time.
 */
async function confirmSave(filename: string, what: string): Promise<SaveChoice> {
    if (state.sameDir) {
        const go = await showConfirm({
            message:
                `⚠️ Saves are going into the folder the pages were loaded from, so this ` +
                `permanently overwrites the original:\n\n${filename}\n${destLabel()}`,
            confirmLabel: "Overwrite",
            danger: true,
        });
        // In place: the page on disk is the one being replaced, whether or not
        // this session has saved it before.
        return go ? "overwrite" : "no";
    }
    if (state.savedFiles.has(filename)) {
        return (await confirmOverwrite(filename)) ? "overwrite" : "no";
    }
    const go = await showConfirm({
        message: `Save ${what} to:\n\n${destLabel()}\n\nThe original page is left untouched.`,
        confirmLabel: "Save",
    });
    return go ? "save" : "no";
}

/** The question the backend's `exists` conflict is asking on the user's behalf. */
function confirmOverwrite(filename: string): Promise<boolean> {
    return showConfirm({
        message: `${filename} is already saved in:\n\n${destLabel()}\n\nReplace it?`,
        confirmLabel: "Overwrite",
        danger: true,
    });
}

interface SaveHooks {
    /** Drop any progress overlay before the overwrite question is asked. */
    pause?: () => void;
    /** Put it back once the answer is yes and the work resumes. */
    resume?: () => void;
}

/**
 * Run a save, and if the destination turns out to already hold that page, ask
 * once and retry.
 *
 * `granted` is the answer already given in `confirmSave`, and is what the first
 * attempt carries. The retry below is therefore only reached when *we* did not
 * know a copy was there -- one saved in an earlier session, or by another window
 * -- which is exactly the case worth a second question. The server is still the
 * one that decides: it checks at write time and answers 409 with
 * `X-Fox-Code: exists`, so nothing is clobbered silently.
 *
 * Returns false when the user declines; the caller then leaves everything alone.
 */
async function saveWithOverwrite(
    filename: string,
    attempt: (overwrite: boolean) => Promise<void>,
    granted: boolean,
    hooks: SaveHooks = {},
): Promise<boolean> {
    try {
        await attempt(granted);
        return true;
    } catch (err) {
        if (!(err instanceof api.ApiError) || err.code !== "exists") throw err;
        hooks.pause?.();
        const go = await confirmOverwrite(filename);
        if (!go) return false;
        hooks.resume?.();
        await attempt(true);
        return true;
    }
}

/** Reflect a completed save: badge the page, show it, and re-render the entries. */
function afterSave(filename: string, deps: InpaintDeps): void {
    // The entries are deliberately kept. They describe the typesetting that
    // produced the saved page, and every render reads the *original* again, so
    // they are still exactly what this page needs -- to fix a line, edit it and
    // save again.
    deps.refresh();
    markPageSaved(filename);
    // Land on what was just written, so the save is visibly a save. In-place
    // saves have no second copy to switch to.
    // Land on what was just written, so the save is visibly a save. In-place
    // saves have no second copy to switch to; `markPageSaved` has already
    // pointed the page at the rewritten bytes for them.
    //
    // Nothing re-navigates the gallery here any more. That used to be how the
    // new bytes were pulled in, back when every image URL carried a fresh
    // `?t=`; it also rebuilt the whole page -- refit, re-render, and a reload of
    // every thumbnail -- for one file that changed. The two calls above now
    // re-point exactly the images whose URL moved.
    if (!state.sameDir) setViewVariant("saved");
}

// -------------------------------------------------------------------- the jobs

/**
 * Run Preview or Generate for the current page.
 *
 * Preview renders into the modal and keeps the artifact so it can be saved
 * verbatim; Generate writes the page into the destination folder after a
 * confirmation.
 */
export async function runInpaint(mode: "preview" | "generate", deps: InpaintDeps): Promise<void> {
    if (!state.currentImageFile) {
        showNotify("⚠️ No page is open.");
        return;
    }

    const payload = deps.buildPayload();
    if (!payload) {
        showNotify("⚠️ No page is open.");
        return;
    }

    // An empty item list is not an error. A page that needs nothing done to it --
    // a title page, a spread with no dialogue, one the reader has decided to leave
    // alone -- renders as a faithful copy of the original, which is exactly what
    // Preview should show and what Save should write into the destination folder.
    // Refusing it meant those pages could not be put through at all, so a folder
    // could not be finished without hand-copying the gaps.

    if (mode === "preview") {
        await runPreview(payload);
        return;
    }

    const filename = payload.filename;
    const choice = await confirmSave(filename, "the typeset page");
    if (choice === "no") return;

    const render = () => {
        showLoading("Rendering...");
        stopPolling?.();
        stopPolling = pollProgress("generate", (snap) => {
            const suffix = snap.total > 0 ? ` (${snap.done}/${snap.total})` : "";
            updateLoadingText(`${snap.message || "Rendering"}${suffix}`);
        });
    };
    const stopRender = () => {
        stopPolling?.();
        stopPolling = null;
        hideLoading();
    };

    render();
    try {
        const saved = await saveWithOverwrite(
            filename,
            (overwrite) => api.inpaintGenerate({ ...payload, overwrite }),
            choice === "overwrite",
            { pause: stopRender, resume: render },
        );
        if (!saved) {
            showNotify("Left the saved page as it was.");
            return;
        }
        afterSave(filename, deps);
        showNotify(state.sameDir ? "✅ Page written." : `✅ Saved to ${destLabel()}`);
    } catch (err) {
        showNotify(`❌ Render failed: ${errText(err)}`);
    } finally {
        stopRender();
    }
}

async function runPreview(payload: ProcessImageRequest): Promise<void> {
    openInpaintModal();
    previewToken = "";
    previewFilename = payload.filename;

    stopPolling?.();
    stopPolling = pollProgress("preview", renderSnapshot);

    try {
        const { blob, token } = await api.inpaintPreview(payload);
        stopPolling?.();
        stopPolling = null;

        // A page change mid-render would make the saved bytes belong to the
        // wrong file, and the backend's token check would reject it anyway.
        if (state.currentImageFile !== previewFilename) {
            closeInpaintModal();
            showNotify("⚠️ Page changed while rendering — preview discarded.");
            return;
        }

        previewToken = token;
        releaseObjectURL();
        previewObjectURL = URL.createObjectURL(blob);

        const img = el<HTMLImageElement>("inpaintModalImage");
        if (!img) return;
        img.onload = () => {
            setModalStatus(false);
            img.style.display = "block";
            // Without a token the artifact was not cached, so offering "Save
            // this image" would promise something the backend cannot honour.
            setPreviewActions(Boolean(previewToken));
            const hint = el("inpaintPreviewHint");
            if (hint) {
                hint.textContent = previewToken
                    ? state.sameDir
                        ? "Saves exactly this image, over the original — no re-render."
                        : "Saves exactly this image to the destination folder — no re-render."
                    : "This preview cannot be saved directly; use Generate.";
                hint.title = previewToken ? destLabel() : "";
            }
        };
        img.onerror = () => renderFailure("The preview image could not be decoded.");
        img.src = previewObjectURL;
    } catch (err) {
        stopPolling?.();
        stopPolling = null;
        renderFailure(`Preview failed: ${errText(err)}`);
    }
}

/**
 * Commit the preview currently on screen into the destination folder.
 *
 * Exposed on `window` for the modal's button.
 */
export async function saveInpaintPreview(deps: InpaintDeps): Promise<void> {
    if (!previewToken || !previewFilename) {
        showNotify("⚠️ Nothing to save — render a preview first.");
        return;
    }
    const filename = previewFilename;
    const token = previewToken;
    const choice = await confirmSave(filename, "this preview");
    if (choice === "no") return;

    const btn = el<HTMLButtonElement>("savePreviewBtn");
    if (btn) btn.disabled = true;
    try {
        const saved = await saveWithOverwrite(
            filename,
            (overwrite) => api.savePreview({ filename, token, overwrite }),
            choice === "overwrite",
        );
        // Declining leaves the modal up with the preview still in it, so the
        // decision can be taken again without re-rendering.
        if (!saved) {
            showNotify("Left the saved page as it was.");
            return;
        }
        previewToken = "";
        closeInpaintModal();
        afterSave(filename, deps);
        showNotify(state.sameDir ? "✅ Saved." : `✅ Saved to ${destLabel()}`);
    } catch (err) {
        // A 409 that is not `exists` means the artifact expired or belongs to
        // another page; say so rather than silently doing nothing.
        showNotify(`❌ Save failed: ${errText(err)}`);
    } finally {
        if (btn) btn.disabled = false;
    }
}

const errText = (err: unknown): string =>
    err instanceof Error && err.message ? err.message : "the server did not respond";
