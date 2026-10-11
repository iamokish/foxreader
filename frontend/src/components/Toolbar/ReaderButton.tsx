/**
 * The reader-mode toggle.
 *
 * One button, shared: the Vulpes top bar draws its own (it has a label and a place
 * to put it), and every other layout gets this one beside the folder button.
 * Both go through `WorkspaceProvider`, which owns the mode and the
 * `body.workspace-reader` class the viewer reads -- so there is one piece of state
 * behind both, and `Esc` works the same in either.
 *
 * Icon-only: the tooltip and `aria-label` carry the name, since there is no
 * visible text. `static/style.css` zeroes the icon's trailing margin for it,
 * so the glyph centres instead of leaning left.
 */

import { useWorkspace } from "../../workspace";

export function ReaderButton() {
    const { mode, setMode } = useWorkspace();
    const isReader = mode === "reader";

    return (
        <button
            id="readerModeBtn"
            type="button"
            class={isReader ? "active-capture" : undefined}
            title={isReader ? "Exit reader mode (Esc)" : "Read these pages (single, spread, continuous or compare)"}
            aria-label="Reader"
            aria-pressed={isReader}
            onClick={() => setMode(isReader ? "work" : "reader")}
        >
            <span class="material-icons" aria-hidden="true">
                {isReader ? "exit_to_app" : "auto_stories"}
            </span>
        </button>
    );
}
