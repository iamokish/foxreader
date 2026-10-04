import { isViewingSaved, TEXT_ALIGNMENTS, ALIGNMENT_ICONS } from "./state";
import type { TextAlignment } from "./types";

const MAX_TOASTS = 3;
const TOAST_HEIGHT = 54;
const TOAST_GAP = 8;

const activeToasts: HTMLElement[] = [];

function repositionToasts(): void {
    activeToasts.forEach((toast, i) => {
        if (!toast.isConnected) return;
        toast.style.top = `${20 + i * (TOAST_HEIGHT + TOAST_GAP)}px`;
    });
}

function dismissToast(toast: HTMLElement): void {
    const idx = activeToasts.indexOf(toast);
    if (idx !== -1) activeToasts.splice(idx, 1);
    repositionToasts();
    toast.style.opacity = "0";
    setTimeout(() => {
        toast.remove();
    }, 300);
}

export function showNotify(message: string): void {
    if (activeToasts.length >= MAX_TOASTS) {
        dismissToast(activeToasts[0]);
    }

    const toast = document.createElement("div");
    toast.className = "toast-notification";
    toast.style.cssText = `
        position:fixed;left:20px;z-index:10000;
        background:#333;color:#fff;padding:10px 14px 10px 18px;border-radius:8px;
        font-size:14px;box-shadow:0 4px 12px rgba(0,0,0,0.3);
        opacity:0;transition:opacity 0.3s;
        display:flex;align-items:center;gap:12px;max-width:360px;
    `;

    const text = document.createElement("span");
    text.style.cssText = "flex:1;word-break:break-word;";
    text.textContent = message;

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "\u00d7";
    closeBtn.style.cssText = `
        background:none;border:none;color:#999;cursor:pointer;
        font-size:18px;padding:0 2px;line-height:1;flex-shrink:0;
    `;
    closeBtn.onmouseenter = () => (closeBtn.style.color = "#fff");
    closeBtn.onmouseleave = () => (closeBtn.style.color = "#999");
    closeBtn.onclick = () => dismissToast(toast);

    toast.appendChild(text);
    toast.appendChild(closeBtn);
    activeToasts.push(toast);
    document.body.appendChild(toast);
    repositionToasts();
    requestAnimationFrame(() => (toast.style.opacity = "1"));
    setTimeout(() => dismissToast(toast), 3000);
}

export function showLoading(initialText = "Loading..."): void {
    const overlay = document.getElementById("loadingModalOverlay");
    const textContainer = document.getElementById("loadingModalText");
    if (textContainer) textContainer.innerText = initialText;
    if (overlay) {
        overlay.style.display = "flex";
        overlay.style.alignItems = "center";
        overlay.style.justifyContent = "center";
    }
}

export function updateLoadingText(newText: string): void {
    const textContainer = document.getElementById("loadingModalText");
    if (textContainer) textContainer.innerText = newText;
}

export function hideLoading(): void {
    const overlay = document.getElementById("loadingModalOverlay");
    if (overlay) overlay.style.display = "none";
}

let alignIdx = 0;

/** Step the capture-defaults alignment button through every alignment. */
export function cycleTextAlignment(): void {
    // Read the button rather than trusting the counter: selecting an entry also
    // sets this control, so a stored index drifts out of sync on the first click
    // after a selection and the first press appears to do nothing.
    const current = TEXT_ALIGNMENTS.indexOf(getTextAlignmentValue());
    alignIdx = (current === -1 ? alignIdx : current) + 1;
    setTextAlignmentValue(TEXT_ALIGNMENTS[alignIdx % TEXT_ALIGNMENTS.length]);
}

export function setTextAlignmentValue(value: TextAlignment): void {
    const btn = document.getElementById("alignmentCycleBtn");
    if (!btn) return;
    const align = TEXT_ALIGNMENTS.includes(value) ? value : "center";
    btn.dataset.current = align;
    btn.title = `Alignment: ${align}`;
    const icon = btn.querySelector(".material-icons");
    if (icon) icon.textContent = ALIGNMENT_ICONS[align];
}

export function getTextAlignmentValue(): TextAlignment {
    const btn = document.getElementById("alignmentCycleBtn");
    const value = btn?.dataset.current as TextAlignment | undefined;
    return value && TEXT_ALIGNMENTS.includes(value) ? value : "center";
}

export function updateActionRow(): void {
    const btn = document.getElementById("confirmTranslationBtn") as HTMLButtonElement | null;
    // The saved copy is read-only, so the button is locked there. A missing
    // region is *not* a reason to disable it: nothing calls back in when a
    // capture lands, so the button would stay greyed out for good. Confirming
    // without a region notifies instead (see `confirmCurrentTranslation`).
    if (btn) btn.disabled = isViewingSaved();
}

export function setTextAreaLoading(loading: boolean): void {
    const textArea = document.getElementById("textArea") as HTMLTextAreaElement | null;
    if (!textArea) return;
    const loader = document.getElementById("ocrLoader") as HTMLTextAreaElement | null;
    if (!loader) return;

    if (textArea) {
        textArea.classList.toggle("ocr-loading", loading);
    }

    if (loader) {
        loader.classList.toggle("active", loading);
    }
}
