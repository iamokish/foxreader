import { render } from "preact";
import "./themes/dark.css";
import "./themes/light.css";
import { App } from "./App";
import { getInitialTheme } from "./themes";
import { initWebSocket, initSessionLock } from "./core/ws";
import { loadFontsAndBuildMenu, initFontListeners, toggleFontDropdown } from "./fonts";
import {
    renderGallery,
    selectImage,
    setupImageErrorHandler,
    initPanning,
    initZoom,
    initResize,
    navigateGallery,
    zoomViewerBy,
    refitViewer,
    updateReaderPageIndicator,
    toggleStripMode,
} from "./viewer";
import { startCapture, startFreeCapture, bubbleCapture } from "./ocr";
import { initTranslatorListeners } from "./translate";
import { showNotify, showLoading, hideLoading, cycleTextAlignment, updateActionRow } from "./ui";
import { loadFolder, folderState, mlLoad, mlUnload } from "./api";
import { initConfirmModal, initBackendStatusOverlay, isBackendStatusVisible } from "./components/common";
import { openFolderModal, isFolderModalOpen } from "./components/Folder";
import { initHealthMonitor } from "./core/session";
import {
    renderCurrentPageEntries,
    confirmCurrentTranslation,
    clearCurrentPageEntries,
    executeInpaintAction,
    pageCapture,
    addNewEntries,
    initEntriesListeners,
    closeInpaintModal,
    saveInpaintPreview,
    clearPageWorkspace,
} from "./entries";
import { state } from "./state";
import { on } from "./eventbus";
import { syncMLStateFromBackend } from "./mlStatus";
import { clearCharacters } from "./characters";

function toggleReaderFullscreen() {
    if (document.fullscreenElement) {
        document.exitFullscreen().catch(() => {});
    } else {
        document.documentElement.requestFullscreen().catch(() => {});
    }
}

/**
 * Mirror the Characters option switch onto the roster button.
 *
 * While the switch is off the button stays hidden; opening the roster via
 * the button must never flip the switch back on, so no handler on
 * `#charactersBtn` touches `state.isCharactersEnabled` or `#enableCharacters`.
 */
function syncCharactersButtonVisibility(): void {
    const btn = document.getElementById("charactersBtn") as HTMLElement | null;
    if (!btn) return;
    btn.style.display = state.isCharactersEnabled ? "" : "none";
    if (state.isCharactersEnabled) btn.removeAttribute("hidden");
    else btn.setAttribute("hidden", "");
}

/**
 * The source folder to offer the picker when the path field is empty.
 *
 * Seeded from `/api/folder/state` at startup, so reopening the app offers the
 * folder the last session was working on.
 */
let lastKnownSource = "";

const root = document.getElementById("nocontextmenu")!;
root.innerHTML = "";

const initialTheme = getInitialTheme();
document.documentElement.classList.remove("theme-dark", "theme-light");
document.documentElement.classList.add(`theme-${initialTheme}`);

render(<App />, root);

const socket = initWebSocket();
initSessionLock(socket);

initHealthMonitor();

initFontListeners();
loadFontsAndBuildMenu();

initPanning();
initZoom();
initResize();
setupImageErrorHandler();
initTranslatorListeners();
initEntriesListeners();

on("image:loaded", () => {
    renderCurrentPageEntries();
    updateReaderPageIndicator();
});
// The captured text, its translation and the last region belong to the page
// they were taken from; carrying them onto the next one is how a translation
// ends up saved against the wrong page.
on("page:changed", () => clearPageWorkspace());
on("entries:render", () => renderCurrentPageEntries());
on("gallery:navigate", (dir: unknown) => navigateGallery(dir as number));

document.addEventListener("DOMContentLoaded", () => {
    initConfirmModal();
    initBackendStatusOverlay();
    void initFolderModal();
    // The backend keeps a loaded MTL model across a page refresh while the
    // switch resets: reconcile the two, silently staying as-is when the
    // backend is unreachable or reports nothing loaded.
    void syncMLStateFromBackend();
    const noCtx = document.getElementById("nocontextmenu");
    if (noCtx) noCtx.addEventListener("contextmenu", (e) => e.preventDefault());

    const folderInput = document.getElementById("folderPathInput");
    if (folderInput)
        folderInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") loadLocalFolder();
        });

    document.getElementById("grayScaleOCR")?.addEventListener("change", (e) => {
        state.isGrayScaleEnabled = (e.target as HTMLInputElement).checked;
        showNotify(state.isGrayScaleEnabled ? "Gray Scale: ON" : "Gray Scale: OFF");
    });

    document.getElementById("horizontalFit")?.addEventListener("change", (e) => {
        state.isHorizontalFitEnabled = (e.target as HTMLInputElement).checked;
        if (state.currentImageFile) {
            const mainImg = document.getElementById("mainImage");
            mainImg?.dispatchEvent(new Event("load"));
        }
        showNotify(state.isHorizontalFitEnabled ? "Horizontal Fit: ON" : "Vertical Fit: ON");
    });

    document.getElementById("autoOCRresponse")?.addEventListener("change", (e) => {
        state.isAutoOCREnabled = (e.target as HTMLInputElement).checked;
        showNotify(state.isAutoOCREnabled ? "Auto OCR : ON" : "Auto OCR : OFF");
    });

    document.getElementById("autoTLresponse")?.addEventListener("change", (e) => {
        state.isAutoTranslateEnabled = (e.target as HTMLInputElement).checked;
        showNotify(state.isAutoTranslateEnabled ? "Auto Translation : ON" : "Auto Translation : OFF");
    });

    document.getElementById("liveInpaint")?.addEventListener("change", (e) => {
        state.isLiveInpaintedEnabled = (e.target as HTMLInputElement).checked;
        renderCurrentPageEntries();
        showNotify(state.isLiveInpaintedEnabled ? "Live Inpainting: ON" : "Live Inpainting: OFF");
    });

    document.getElementById("enableContext")?.addEventListener("change", (e) => {
        state.isContextEnabled = (e.target as HTMLInputElement).checked;
        showNotify(state.isContextEnabled ? "Context: ON" : "Context: OFF");
    });

    document.getElementById("enableCharacters")?.addEventListener("change", (e) => {
        state.isCharactersEnabled = (e.target as HTMLInputElement).checked;
        syncCharactersButtonVisibility();
        showNotify(state.isCharactersEnabled ? "Characters: ON" : "Characters: OFF");
    });
    syncCharactersButtonVisibility();

    document.querySelectorAll(".dropdown").forEach((dropdown) => {
        const trigger = dropdown.querySelector(".dropdown-trigger");
        const content = dropdown.querySelector(".dropdown-content");
        if (!trigger || !content) return;

        trigger.addEventListener("click", (e) => {
            e.stopPropagation();
            const el = content as HTMLElement;
            const isOpen = el.style.display === "block";

            document.querySelectorAll(".dropdown-content").forEach((c) => {
                if (c !== content) {
                    (c as HTMLElement).style.display = "";
                    (c as HTMLElement).style.top = "";
                    (c as HTMLElement).style.left = "";
                }
            });

            if (isOpen) {
                el.style.display = "";
                el.style.top = "";
                el.style.left = "";
            } else {
                const triggerRect = trigger.getBoundingClientRect();
                el.style.visibility = "hidden";
                el.style.display = "block";
                const contentRect = el.getBoundingClientRect();

                let top = triggerRect.bottom + 4;
                let left = triggerRect.left;

                if (top + contentRect.height > window.innerHeight) {
                    top = triggerRect.top - contentRect.height - 4;
                }
                if (left + contentRect.width > window.innerWidth) {
                    left = window.innerWidth - contentRect.width - 8;
                }
                if (left < 0) left = 8;

                el.style.top = `${top}px`;
                el.style.left = `${left}px`;
                el.style.visibility = "";
            }
        });
    });

    document.addEventListener("click", () => {
        document.querySelectorAll(".dropdown-content").forEach((c) => {
            (c as HTMLElement).style.display = "";
            (c as HTMLElement).style.top = "";
            (c as HTMLElement).style.left = "";
        });
    });
});

document.addEventListener("DOMContentLoaded", () => {
    const mlCheckbox = document.getElementById("enableML") as HTMLInputElement | null;

    async function handleMLLoading() {
        if (!mlCheckbox) return;
        const isEnabled = mlCheckbox.checked;
        const lang = (document.getElementById("languageSelect") as HTMLSelectElement)?.value ?? "";
        mlCheckbox.disabled = true;
        try {
            showLoading(isEnabled ? "Loading MTL Model..." : "Unloading MTL Model...");
            const data = isEnabled ? await mlLoad(lang) : await mlUnload();
            if (!data.status) {
                mlCheckbox.checked = !isEnabled;
                showNotify(data.message);
            }
        } catch {
            mlCheckbox.checked = !isEnabled;
            showNotify("ML control request failed");
        }
        mlCheckbox.disabled = false;
        hideLoading();
    }

    if (mlCheckbox) mlCheckbox.onchange = handleMLLoading;
    document.getElementById("languageSelect")?.addEventListener("change", () => {
        if (mlCheckbox?.checked) handleMLLoading();
    });
});

async function loadLocalFolder() {
    const pathInput = document.getElementById("folderPathInput") as HTMLInputElement | null;
    const typed = pathInput?.value.trim() ?? "";

    // The picker is the only way in now: it is where the destination is chosen and
    // where read/write permission is reported before anything is committed to.
    openFolderModal({
        source: typed || lastKnownSource,
        onSubmit: (source, dest) => applyFolder(source, dest),
    });
}

/**
 * Commit to a source/destination pair: load the pages and reset the viewer.
 *
 * This is the body `loadLocalFolder` used to have, plus the bookkeeping the
 * destination introduces. Everything the viewer and the entry panel read is set
 * before `renderGallery` runs, so the first thumbnail is already correct.
 */
async function applyFolder(source: string, dest: string) {
    if (!source) {
        showNotify("Please enter a folder path");
        return;
    }
    showLoading();

    const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
    const container = document.getElementById("imageContainer");
    if (mainImg) {
        mainImg.onerror = null;
        mainImg.src = "";
        // The stamp `viewer.setImageSrc` compares against; leaving it behind
        // would make the first page of the new folder look already-loaded and
        // skip the fetch.
        delete mainImg.dataset.foxSrc;
        mainImg.style.display = "none";
        mainImg.style.pointerEvents = "auto";
        mainImg.style.userSelect = "auto";
        mainImg.draggable = true;
    }
    if (container) container.style.overflow = "auto";
    state.zoomLevel = 1.0;
    if (mainImg) mainImg.style.transform = "scale(1)";
    const gallery = document.getElementById("gallery");
    if (gallery) gallery.innerHTML = "";
    state.currentImageFile = "";

    try {
        const data = await loadFolder(source, { dest });
        if (data.status === "success") {
            state.pageEntriesCache = {};
            // The roster belongs to the folder's cast: a new folder starts
            // with no characters (backend cleared too; restart/empty is fine
            // and never blocks the folder load).
            void clearCharacters();
            // Before renderGallery: it reads `savedFiles` to badge the thumbnails
            // and `viewVariant` to build their URLs.
            state.sourceDir = data.source ?? source;
            state.destDir = data.dest ?? data.source ?? source;
            state.sameDir = Boolean(data.same_dir);
            state.savedFiles = new Set(data.saved ?? []);
            state.viewVariant = "original";
            lastKnownSource = state.sourceDir;

            const pathInput = document.getElementById("folderPathInput") as HTMLInputElement | null;
            if (pathInput) pathInput.value = state.sourceDir;

            renderGallery(data.files, data.dimensions ?? []);
            setupImageErrorHandler();
            if (data.files.length > 0) {
                selectImage(data.files[0]);
                setTimeout(() => {
                    const firstItem = document.querySelector(".gallery-item-container");
                    if (firstItem) firstItem.classList.add("active");
                }, 10);
            }
            showNotify(
                state.sameDir
                    ? "⚠️ Saving will overwrite the originals in this folder."
                    : `Saves go to ${state.destDir}`,
            );
        } else {
            showNotify(data.error ?? "Unknown error");
        }
    } catch (err) {
        showNotify(err instanceof Error && err.message ? err.message : "Server connection failed");
    }
    hideLoading();
}

/**
 * Prefill the picker with the folder this machine used last.
 *
 * The pairing lives in `config/workspaces.yaml`, so this survives a restart. Only
 * the text field is touched -- nothing is loaded without the user asking.
 */
async function initFolderModal() {
    try {
        const info = await folderState();
        lastKnownSource = info.source || info.last_source || "";
        const pathInput = document.getElementById("folderPathInput") as HTMLInputElement | null;
        if (pathInput && !pathInput.value.trim() && lastKnownSource) pathInput.value = lastKnownSource;
    } catch {
        /* An older backend, or one still starting: the picker just opens empty. */
    }
}

let isEditingText = false;
document.addEventListener("focusin", (e) => {
    if ((e.target as HTMLElement).closest("#translatedText") || (e.target as HTMLElement).isContentEditable)
        isEditingText = true;
});
document.addEventListener("focusout", (e) => {
    if ((e.target as HTMLElement).closest("#translatedText") || (e.target as HTMLElement).isContentEditable)
        isEditingText = false;
});

window.addEventListener("keydown", (e) => {
    const activeEl = document.activeElement?.tagName;
    if (activeEl === "INPUT" || activeEl === "TEXTAREA" || isEditingText) return;

    const modalOverlay = document.getElementById("inpaintModalOverlay");
    if (modalOverlay && modalOverlay.style.display === "flex") {
        // Route through the real close: it also stops the progress poller and
        // revokes the preview's object URL.
        if (e.key === "Escape") closeInpaintModal();
        return;
    }
    const loadingOverlay = document.getElementById("loadingModalOverlay");
    if (loadingOverlay && loadingOverlay.style.display !== "none") return;
    if (isBackendStatusVisible()) return;
    // The picker's own overlay handles Escape/Enter. Focus sitting on one of its
    // buttons is not an INPUT, so without this the editor hotkeys would fire
    // behind it.
    if (isFolderModalOpen()) return;

    // Reader mode: navigation/zoom take over; editor hotkeys are suppressed.
    if (document.body.classList.contains("workspace-reader")) {
        if (e.ctrlKey || e.altKey || state.isDrawing || state.isCaptureMode) return;
        switch (e.key.toLowerCase()) {
            case "arrowright":
            case "arrowdown":
            case " ":
                e.preventDefault();
                navigateGallery(1);
                break;
            case "arrowleft":
            case "arrowup":
                e.preventDefault();
                navigateGallery(-1);
                break;
            case "escape":
                e.preventDefault();
                (window as any).setWorkspaceMode?.("work");
                break;
            case "f":
                e.preventDefault();
                toggleReaderFullscreen();
                break;
            case "s":
                e.preventDefault();
                toggleStripMode();
                break;
            case "+":
            case "=":
                e.preventDefault();
                zoomViewerBy(0.1);
                break;
            case "-":
                e.preventDefault();
                zoomViewerBy(-0.1);
                break;
            case "0":
                e.preventDefault();
                refitViewer();
                break;
        }
        return;
    }

    if (e.ctrlKey || e.shiftKey || e.altKey || state.isDrawing || state.isCaptureMode) return;

    switch (e.key.toLowerCase()) {
        case "h":
            e.preventDefault();
            if (document.body.classList.contains("layout-default")) {
                (window as any).togglePanel?.("gallery");
            } else {
                document.getElementById("gallery")?.classList.toggle("hidden");
            }
            navigateGallery(0);
            break;
        case "c":
            e.preventDefault();
            startCapture();
            break;
        case "f":
            e.preventDefault();
            startFreeCapture();
            break;
        case "arrowright":
            e.preventDefault();
            navigateGallery(1);
            break;
        case "arrowleft":
            e.preventDefault();
            navigateGallery(-1);
            break;
    }
});

window.loadLocalFolder = loadLocalFolder;
window.showLoading = showLoading;
window.hideLoading = hideLoading;
window.showNotify = showNotify;
window.cycleTextAlignment = cycleTextAlignment;
window.updateActionRow = updateActionRow;
window.confirmCurrentTranslation = confirmCurrentTranslation;
window.clearCurrentPageEntries = clearCurrentPageEntries;
window.executeInpaintAction = executeInpaintAction;
window.pageCapture = pageCapture;
window.addNewEntries = addNewEntries;
window.renderCurrentPageEntries = renderCurrentPageEntries;
window.navigateGallery = navigateGallery;
window.startCapture = startCapture;
window.startFreeCapture = startFreeCapture;
window.bubbleCapture = bubbleCapture;
window.toggleFontDropdown = toggleFontDropdown;
window.closeInpaintModal = closeInpaintModal;
window.saveInpaintPreview = saveInpaintPreview;
