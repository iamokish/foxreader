export function InpaintModal() {
    return (
        <div
            id="inpaintModalOverlay"
            class="custom-scrollbar"
            style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: var(--bg-overlay); z-index: 99999; align-items: flex-start; justify-content: center; overflow-y: auto; padding: 40px 0; box-sizing: border-box;"
        >
            <div
                id="inpaintModalContent"
                style="position: relative; width: 50vw; background: var(--bg-secondary); padding: 16px; border-radius: 8px; border: 1px solid var(--border-default); box-shadow: var(--shadow-lg); display: flex; flex-direction: column; align-items: center; justify-content: flex-start; min-height: 200px; box-sizing: border-box;"
            >
                <button
                    onClick={() => (window as any).closeInpaintModal?.()}
                    style="position: fixed; top: 20px; right: calc(25vw - 50px); background: var(--color-danger); border: none; color: white; width: 36px; height: 36px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(0,0,0,0.4); z-index: 100000; transition: transform 0.1s;"
                >
                    <span class="material-icons" style="font-size: 20px; margin-right: 0px;">
                        close
                    </span>
                </button>
                <div id="inpaintModalStatus" class="inpaint-status">
                    <div id="inpaintModalLoader" class="modal-spinner" />
                    <div class="inpaint-status__body">
                        <div id="inpaintProgressMessage" class="inpaint-status__msg">Preparing…</div>
                        <div class="inpaint-progress">
                            <span id="inpaintProgressFill" class="inpaint-progress__fill" />
                        </div>
                        <div id="inpaintProgressCount" class="inpaint-status__count" />
                        <pre id="inpaintProgressLog" class="inpaint-status__log custom-scrollbar" />
                    </div>
                </div>
                <img
                    id="inpaintModalImage"
                    src=""
                    style="display: none; width: 100%; height: auto; object-fit: contain; border-radius: 4px; box-sizing: border-box;"
                />
                <div id="inpaintPreviewActions" class="inpaint-actions" style="display: none;">
                    <span id="inpaintPreviewHint" class="inpaint-actions__hint" />
                    <button
                        id="savePreviewBtn"
                        class="btn-success"
                        onClick={() => (window as any).saveInpaintPreview?.()}
                    >
                        <span class="material-icons" style="font-size: 16px;">
                            save
                        </span>
                        Save this image
                    </button>
                </div>
            </div>
        </div>
    );
}

export function LoadingModal() {
    return (
        <div
            id="loadingModalOverlay"
            style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: var(--bg-overlay); z-index: 99999; box-sizing: border-box;"
        >
            <div
                id="loadingModalContent"
                style="background: var(--bg-secondary); padding: 30px; border-radius: 8px; border: 1px solid var(--border-default); box-shadow: var(--shadow-lg); display: flex; flex-direction: column; align-items: center; justify-content: center; min-width: 250px; max-width: 50vw; max-height: 50vh; overflow: hidden; box-sizing: border-box;"
            >
                <div class="modal-spinner" style="flex-shrink: 0;" />
                <div
                    id="loadingModalText"
                    class="custom-scrollbar"
                    style="color: var(--text-primary); font-family: sans-serif; font-size: 16px; margin-top: 20px; text-align: center; max-width: 100%; word-wrap: break-word; overflow-y: auto;"
                >
                    Loading...
                </div>
            </div>
        </div>
    );
}

let resolveConfirm: ((value: boolean) => void) | null = null;

export function ConfirmModal() {
    return (
        <div
            id="confirmModalOverlay"
            style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: var(--bg-overlay); z-index: 99999; align-items: center; justify-content: center; box-sizing: border-box;"
        >
            <div
                id="confirmModalContent"
                style="background: var(--bg-secondary); padding: 24px; border-radius: 8px; border: 1px solid var(--border-default); box-shadow: var(--shadow-lg); min-width: 320px; max-width: 480px; box-sizing: border-box;"
            >
                <p
                    id="confirmModalMessage"
                    style="color: var(--text-primary); font-size: 15px; margin: 0 0 20px 0; line-height: 1.5; white-space: pre-wrap;"
                ></p>
                <div style="display: flex; gap: 10px; justify-content: flex-end;">
                    <button id="confirmModalCancel" class="btn-secondary" style="padding: 8px 20px; cursor: pointer;">
                        Cancel
                    </button>
                    <button id="confirmModalOk" class="btn-primary" style="padding: 8px 20px; cursor: pointer;">
                        Confirm
                    </button>
                </div>
            </div>
        </div>
    );
}

/** How a confirmation should read. Everything but the message is optional. */
export interface ConfirmOptions {
    message: string;
    /** The affirmative button's text. Defaults to "Confirm". */
    confirmLabel?: string;
    /** Colour the affirmative button as destructive, e.g. for an overwrite. */
    danger?: boolean;
}

/**
 * Ask the user a yes/no question.
 *
 * Takes either a bare message, as it always has, or an options object for the
 * cases that need a verb on the button -- "Overwrite" reads very differently
 * from "Confirm" when the answer replaces a file.
 */
export function showConfirm(input: string | ConfirmOptions): Promise<boolean> {
    const opts: ConfirmOptions = typeof input === "string" ? { message: input } : input;
    return new Promise((resolve) => {
        resolveConfirm = resolve;
        const overlay = document.getElementById("confirmModalOverlay");
        const msgEl = document.getElementById("confirmModalMessage");
        const okBtn = document.getElementById("confirmModalOk");
        if (msgEl) msgEl.textContent = opts.message;
        if (okBtn) {
            okBtn.textContent = opts.confirmLabel || "Confirm";
            // Reset both ways round: the button is shared, so a dialog that does
            // not ask for danger must not inherit it from the last one that did.
            okBtn.classList.toggle("btn-danger", Boolean(opts.danger));
            okBtn.classList.toggle("btn-primary", !opts.danger);
        }
        if (overlay) overlay.style.display = "flex";
    });
}

export function initConfirmModal(): void {
    document.getElementById("confirmModalOk")?.addEventListener("click", () => {
        const overlay = document.getElementById("confirmModalOverlay");
        if (overlay) overlay.style.display = "none";
        resolveConfirm?.(true);
        resolveConfirm = null;
    });
    document.getElementById("confirmModalCancel")?.addEventListener("click", () => {
        const overlay = document.getElementById("confirmModalOverlay");
        if (overlay) overlay.style.display = "none";
        resolveConfirm?.(false);
        resolveConfirm = null;
    });
}
