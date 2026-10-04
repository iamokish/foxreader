import { FontSelector } from "../FontSelector/FontSelector";

export function TranslationRegion() {
    return (
        <>
            <div
                id="translationContainer"
                style="flex: 1; min-height: 0; display: flex; flex-direction: column; width: 100%; box-sizing: border-box; justify-content: center; position: relative;"
            >
                {/* `data-state` is what Confirm Entry reads to decide whether the
                    panel holds a translation or a message -- see
                    `translationPanelState`. It starts as a message. */}
                <div id="translatedText" class="panel error" data-state="error">
                    <strong>Select a translator</strong>
                </div>
            </div>

            <div
                id="confirmActionRow"
                class="buttons"
                style="flex: 0 0 auto; width:100%; display: flex; justify-content: flex-end; padding-top: 5px; box-sizing: border-box; gap: 5px;"
            >
                <FontSelector />

                <button
                    id="alignmentCycleBtn"
                    title="Cycle Alignment"
                    data-current="center"
                    class="btn-purple"
                    onClick={() => (window as any).cycleTextAlignment?.()}
                    style="height: 32px; font-size: 13px; display: flex; align-items: center; justify-content: center; box-sizing: border-box;"
                >
                    <span class="material-icons" style="font-size: 16px;">
                        format_align_center
                    </span>
                </button>
                <button
                    id="inpaintGenerateBtn"
                    title="Generate InPaint & Overwrite File"
                    class="btn-primary"
                    onClick={() => (window as any).executeInpaintAction?.("generate")}
                    style="height: 32px; font-size: 13px;"
                >
                    Save
                </button>
                <button
                    id="inpaintPreviewBtn"
                    title="Preview InPaint Visual Track"
                    class="btn-secondary"
                    onClick={() => (window as any).executeInpaintAction?.("preview")}
                    style="height: 32px; font-size: 13px;"
                >
                    Preview
                </button>
                <button
                    id="clearPageBtn"
                    title="Clear All Entries For This Page"
                    class="btn-danger"
                    onClick={() => (window as any).clearCurrentPageEntries?.()}
                    style="height: 32px; font-size: 13px;"
                >
                    Clear Page
                </button>
                <button
                    id="confirmTranslationBtn"
                    class="btn-success"
                    onClick={() => (window as any).confirmCurrentTranslation?.()}
                    style="height: 32px; font-size: 13px;"
                >
                    Confirm Entry
                </button>
            </div>
        </>
    );
}
