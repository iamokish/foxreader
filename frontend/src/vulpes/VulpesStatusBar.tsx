/**
 * The Vulpes footer: where you are, which copy you are looking at, the editing
 * verbs, and the zoom.
 *
 * Everything here acts on the *editor's* stage (`#imageContainer` / `#mainImage`),
 * which is what work mode shows -- the reader has its own chrome. So the zoom and
 * fit controls go through `viewer.ts` rather than the pager, and the fit toggle
 * writes the same `state.isHorizontalFitEnabled` and `#horizontalFit` checkbox the
 * Options dropdown does, so the two can never disagree.
 *
 * The action row is `ConfirmActionRow` itself, not a copy of it: it carries the ids
 * `viewer.ts`, `ui.ts`, `entries.ts` and `fonts.ts` look up, so there can only be
 * one of it in the document. Vulpes renders it here instead of in the translation
 * section (`vulpes.css` hides the words and shows the glyphs, at this height).
 * `BasicLayout` gets the same component in the same place, from
 * `TranslationRegion` -- which is why this bar renders it through the shared slots
 * rather than restating the buttons.
 */

import { useCallback, useEffect, useState } from "preact/hooks";
import { state } from "../state";
import { on, off } from "../eventbus";
import { refitViewer, setViewVariant, zoomViewerBy } from "../viewer";
import { ConfirmActionRow } from "../slots";
import { currentPageIndex, gotoPage, pageCount } from "../reader/pageNav";
import { useVulpesPanels } from "./VulpesPanelsContext";

/**
 * Turn horizontal fit on or off, from either direction.
 *
 * The switch in the Options dropdown is the one `main.tsx` listens to, so this
 * dispatches a real `change` on it instead of poking `state` behind its back --
 * that way the toast, the re-fit and the checkbox all happen exactly once, in the
 * handler that already does them. The fallback covers a layout rendered without
 * the dropdown mounted.
 */
export function setHorizontalFit(enabled: boolean): void {
    const box = document.getElementById("horizontalFit") as HTMLInputElement | null;
    if (box) {
        if (box.checked === enabled) return;
        box.checked = enabled;
        box.dispatchEvent(new Event("change", { bubbles: true }));
        return;
    }
    state.isHorizontalFitEnabled = enabled;
    refitViewer();
}

export function VulpesStatusBar() {
    const { getPanel, togglePanel } = useVulpesPanels();
    const [, setTick] = useState(0);
    const refresh = useCallback(() => setTick((n) => n + 1), []);

    // One subscription for everything on this bar: the page number, the Saved
    // badge, the Original/Saved switch and the zoom all change in response to the
    // same handful of events.
    useEffect(() => {
        on("page:changed", refresh);
        on("image:loaded", refresh);
        on("entries:render", refresh);
        on("page:saved", refresh);
        return () => {
            off("page:changed", refresh);
            off("image:loaded", refresh);
            off("entries:render", refresh);
            off("page:saved", refresh);
        };
    }, [refresh]);

    const count = pageCount();
    const index = currentPageIndex();
    const file = state.currentFiles[index] ?? "";
    const hasSaved = !state.sameDir && state.savedFiles.has(file);
    const zoomPercent = Math.round((state.zoomLevel || 0) * 100);

    /**
     * Zoom the editor's stage, then re-read it.
     *
     * `zoomViewerBy` writes `state.zoomLevel` and emits nothing -- it is called
     * from a wheel handler many times a second, where an event per notch would be
     * wasted. So the readout is refreshed here, where the press is known about.
     */
    const zoomStage = useCallback(
        (delta: number) => {
            zoomViewerBy(delta);
            refresh();
        },
        [refresh],
    );

    const resetStageZoom = useCallback(() => {
        refitViewer();
        refresh();
    }, [refresh]);

    const pickVariant = useCallback(
        (variant: "original" | "saved") => {
            setViewVariant(variant);
            refresh();
        },
        [refresh],
    );

    const pages = getPanel("pages");
    const editor = getPanel("editor");
    const ocr = getPanel("ocr");

    return (
        <footer class="vx-statusbar">
            <div class="vx-statusbar__group">
                <button
                    type="button"
                    class="vx-icon-btn vx-icon-btn--sm"
                    title="Previous page (←)"
                    disabled={count === 0}
                    onClick={() => gotoPage(index - 1)}
                >
                    <span class="material-icons">chevron_left</span>
                </button>
                <span class="vx-statusbar__page" title={file || "No page"}>
                    {count > 0 && index >= 0 ? (
                        <>
                            <strong>{String(index + 1).padStart(2, "0")}</strong>
                            <span class="vx-statusbar__of">/ {String(count).padStart(2, "0")}</span>
                        </>
                    ) : (
                        <span class="vx-statusbar__of">— / —</span>
                    )}
                </span>
                <button
                    type="button"
                    class="vx-icon-btn vx-icon-btn--sm"
                    title="Next page (→)"
                    disabled={count === 0}
                    onClick={() => gotoPage(index + 1)}
                >
                    <span class="material-icons">chevron_right</span>
                </button>
            </div>

            <div class="vx-statusbar__group vx-statusbar__group--center">
                <div class="vx-seg vx-seg--sm" role="group" aria-label="Page copy">
                    <button
                        type="button"
                        class={`vx-seg__btn${state.viewVariant === "original" ? " is-active" : ""}`}
                        title="Show the original scan"
                        aria-pressed={state.viewVariant === "original"}
                        onClick={() => pickVariant("original")}
                    >
                        Original
                    </button>
                    <button
                        type="button"
                        class={`vx-seg__btn${state.viewVariant === "saved" ? " is-active" : ""}`}
                        title={
                            hasSaved
                                ? "Show the typeset copy from the destination folder"
                                : "This page has no typeset copy yet"
                        }
                        aria-pressed={state.viewVariant === "saved"}
                        disabled={!hasSaved}
                        onClick={() => pickVariant("saved")}
                    >
                        Saved
                    </button>
                </div>
                <span class={`vx-dot${hasSaved ? " is-on" : ""}`} aria-hidden="true" />
                <span class="vx-statusbar__note">{hasSaved ? "Saved" : "Not saved"}</span>
            </div>

            <div class="vx-statusbar__group vx-statusbar__group--actions">
                <ConfirmActionRow class="vx-action-row" />
            </div>

            <div class="vx-statusbar__group vx-statusbar__group--end">
                <button
                    type="button"
                    class={`vx-chip vx-chip--sm${state.isHorizontalFitEnabled ? " is-active" : ""}`}
                    title="Fit the page to the stage width and scroll down it"
                    aria-pressed={state.isHorizontalFitEnabled}
                    onClick={() => setHorizontalFit(!state.isHorizontalFitEnabled)}
                >
                    Fit width
                </button>
                <button
                    type="button"
                    class="vx-icon-btn vx-icon-btn--sm"
                    title="Zoom out (−)"
                    onClick={() => zoomStage(-0.1)}
                >
                    <span class="material-icons">remove</span>
                </button>
                <button
                    type="button"
                    class="vx-statusbar__zoom"
                    title="Reset zoom to the current fit"
                    onClick={resetStageZoom}
                >
                    {zoomPercent > 0 ? `${zoomPercent}%` : "—"}
                </button>
                <button
                    type="button"
                    class="vx-icon-btn vx-icon-btn--sm"
                    title="Zoom in (+)"
                    onClick={() => zoomStage(0.1)}
                >
                    <span class="material-icons">add</span>
                </button>

                <span class="vx-statusbar__sep" />

                <button
                    type="button"
                    class={`vx-icon-btn vx-icon-btn--sm${pages.visible ? " is-active" : ""}`}
                    title="Pages rail (H)"
                    aria-pressed={pages.visible}
                    onClick={() => togglePanel("pages")}
                >
                    <span class="material-icons">view_sidebar</span>
                </button>
                <button
                    type="button"
                    class={`vx-icon-btn vx-icon-btn--sm${ocr.visible ? " is-active" : ""}`}
                    title="Captured text"
                    aria-pressed={ocr.visible}
                    onClick={() => togglePanel("ocr")}
                >
                    <span class="material-icons">text_fields</span>
                </button>
                <button
                    type="button"
                    class={`vx-icon-btn vx-icon-btn--sm${editor.visible ? " is-active" : ""}`}
                    title="Editor"
                    aria-pressed={editor.visible}
                    onClick={() => togglePanel("editor")}
                >
                    <span class="material-icons">tune</span>
                </button>
            </div>
        </footer>
    );
}
