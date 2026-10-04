import "./variant.css";
import { setViewVariant } from "../../viewer";

/**
 * The page viewport.
 *
 * The Original/Saved switch is rendered here (rather than in either layout) so
 * both layouts get it, and starts hidden: `updateVariantSwitch` reveals it once
 * a page has actually been saved to a destination that is not the source.
 *
 * It is a *sibling* of `#imageContainer`, not a child, on purpose. The container
 * is the pan/zoom scroll box -- `initPanning` writes its `scrollLeft`/`scrollTop`
 * and horizontal-fit turns its overflow on -- so an absolutely positioned child
 * would scroll out of view the moment the page is panned. Anchoring to the
 * viewer cell instead (see `variant.css`) keeps the switch pinned. The
 * container's own children are untouched.
 */
export function ImageViewer() {
    return (
        <>
            <div id="imageContainer" class="image-container">
                <canvas
                    id="drawCanvas"
                    style="position: absolute; top: 0; left: 0; z-index: 10; pointer-events: none; display: none;"
                />
                <img id="mainImage" src="" style="display:none;" />
                <div id="cropSelector" />
            </div>
            <div
                id="variantSwitch"
                class="variant-switch"
                title="View only — OCR and typesetting always read the original page."
                onMouseDown={(e) => e.stopPropagation()}
            >
                <span class="variant-switch__label">View</span>
                <button
                    id="variantOriginalBtn"
                    type="button"
                    class="variant-switch__btn is-active"
                    onClick={() => setViewVariant("original")}
                >
                    Original
                </button>
                <button
                    id="variantSavedBtn"
                    type="button"
                    class="variant-switch__btn"
                    onClick={() => setViewVariant("saved")}
                >
                    Saved
                </button>
            </div>
            <div id="variantLoader" class="variant-loader" aria-hidden="true">
                <div class="variant-loader__badge">
                    {/* `.modal-spinner` is the app's existing spinner, so this
                        themes itself and there is only one of them to maintain. */}
                    <span class="modal-spinner" />
                    <span id="variantLoaderText">Loading</span>
                </div>
            </div>
        </>
    );
}
