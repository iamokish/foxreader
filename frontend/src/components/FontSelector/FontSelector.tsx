/**
 * The font picker.
 *
 * The control's *shape* is here and in `fontSelector.css`; its *contents* belong to
 * `fonts.ts`, which builds the option list into `#fontOptionsContainer` by
 * `innerHTML` and writes the chosen face onto the trigger and the label.
 *
 * Nothing here owns the menu's open state. That is one `is-open` class, toggled by
 * `fonts.ts`, because the alternative -- an inline `display` from JSX that script
 * also writes -- is two writers on one property, and the toggle then decides
 * open-vs-closed by reading back whatever the last writer left there.
 *
 * There is deliberately no `memo` wrapper. It would be harmless, but it is not what
 * keeps the menu open: Preact diffs the new vnode against the previous one, so an
 * unchanged `style` string is never written to the DOM and an imperative change
 * survives a re-render regardless.
 */

import "./fontSelector.css";

export function FontSelector() {
    return (
        <div class="custom-font-dropdown">
            <div
                id="fontSelectTrigger"
                class="font-select-trigger"
                onClick={() => (window as any).toggleFontDropdown?.()}
            >
                <span id="selectedFontLabel">Select Font</span>
                <span class="material-icons" aria-hidden="true">
                    arrow_drop_up
                </span>
            </div>
            <input type="hidden" id="fontSelect" value="" />
            {/* Filled by `loadFontsAndBuildMenu`. No inline `display` here: see the
                note above about who owns the menu's open state. */}
            <div id="fontOptionsContainer" class="font-options-container custom-scrollbar" />
        </div>
    );
}
