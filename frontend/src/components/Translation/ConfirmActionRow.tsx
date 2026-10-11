/**
 * The entry actions: font, alignment, and the page's editing verbs.
 *
 * Pulled out of `TranslationRegion` so a layout can put the row where it belongs
 * rather than wherever the translation panel happens to be. Its **ids**, its order
 * and its behaviour are unchanged -- `viewer.ts`, `ui.ts`, `entries.ts` and
 * `fonts.ts` all find these buttons by id, and `applySavedViewLocks` in `viewer.ts`
 * disables exactly this set while the typeset copy is on screen.
 *
 * The ids are the contract, so a layout renders this **once**. Two copies would put
 * two elements with `id="confirmTranslationBtn"` in the document, and every
 * `getElementById` caller would silently get whichever came first.
 *
 * Each button carries both an icon and a label: the glyph shows, the word stays
 * in the markup hidden, and the tooltip (`title`) names the verb on hover.
 * That is a stylesheet decision (`actionRow.css`) rather than a prop, so the
 * markup stays the same everywhere and no layout needs a variant of this
 * component.
 */

import { FontSelector } from "../FontSelector/FontSelector";
import "./actionRow.css";

/** Icon and label widths for one verb, so the two are always declared together. */
interface Action {
    id: string;
    icon: string;
    label: string;
    title: string;
    className: string;
    onClick: () => void;
}

const ACTIONS: Action[] = [
    {
        id: "inpaintGenerateBtn",
        icon: "save",
        label: "Save",
        title: "Generate InPaint & Overwrite File",
        className: "btn-primary",
        onClick: () => (window as any).executeInpaintAction?.("generate"),
    },
    {
        id: "inpaintPreviewBtn",
        icon: "visibility",
        label: "Preview",
        title: "Preview InPaint Visual Track",
        className: "btn-secondary",
        onClick: () => (window as any).executeInpaintAction?.("preview"),
    },
    {
        id: "clearPageBtn",
        icon: "delete_sweep",
        label: "Clear Page",
        title: "Clear All Entries For This Page",
        className: "btn-danger",
        onClick: () => (window as any).clearCurrentPageEntries?.(),
    },
    {
        id: "confirmTranslationBtn",
        icon: "check",
        label: "Confirm Entry",
        title: "Add this translation to the page's entries",
        className: "btn-success",
        onClick: () => (window as any).confirmCurrentTranslation?.(),
    },
];

export function ConfirmActionRow({ class: className = "" }: { class?: string } = {}) {
    return (
        <div id="confirmActionRow" class={`confirm-action-row ${className}`.trim()}>
            <FontSelector />

            <button
                id="alignmentCycleBtn"
                type="button"
                title="Cycle Alignment"
                data-current="center"
                class="btn-purple confirm-action-row__btn"
                onClick={() => (window as any).cycleTextAlignment?.()}
            >
                <span class="material-icons btn-icon" aria-hidden="true">
                    format_align_center
                </span>
                <span class="btn-label">Align</span>
            </button>

            {ACTIONS.map((action) => (
                <button
                    key={action.id}
                    id={action.id}
                    type="button"
                    title={action.title}
                    class={`${action.className} confirm-action-row__btn`}
                    onClick={action.onClick}
                >
                    <span class="material-icons btn-icon" aria-hidden="true">
                        {action.icon}
                    </span>
                    <span class="btn-label">{action.label}</span>
                </button>
            ))}
        </div>
    );
}
