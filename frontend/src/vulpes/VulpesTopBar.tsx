/**
 * The Vulpes top bar.
 *
 * Identity on the left, where the folder is in the middle, and the things that
 * change what you are looking at on the right. The folder field keeps the ids the
 * app already wires (`#folderPathInput`, `#loadFolderBtn`): `main.tsx` binds Enter
 * to the input and `loadLocalFolder` opens the picker, and both read the field by
 * id.
 *
 * The breadcrumb is derived, not stored -- `state.sourceDir` is whatever the
 * backend resolved, so the last two path segments are the closest thing to
 * "series / chapter" that is true without asking the user to tell us twice.
 */

import { useEffect, useState } from "preact/hooks";
import { state } from "../state";
import { on, off } from "../eventbus";
import { useWorkspace } from "../workspace";

/**
 * "series / chapter" from a folder path.
 *
 * The last two segments, because that is how comic folders are nearly always laid
 * out (`.../Series Name/Volume 3`). A one-segment path yields just that segment,
 * and an empty one yields nothing so the caller can show a placeholder. Handles
 * both separators: the backend returns native paths, and this runs on Windows.
 */
export function folderCrumbs(path: string): string[] {
    const parts = (path || "")
        .split(/[\\/]+/)
        .map((part) => part.trim())
        // A trailing separator, a bare drive letter, or a UNC prefix's empty leader.
        .filter((part) => part.length > 0 && !/^[a-z]:$/i.test(part));
    return parts.slice(-2);
}

export function VulpesTopBar({ onResetSections }: { onResetSections?: () => void }) {
    const { mode, setMode } = useWorkspace();
    const [crumbs, setCrumbs] = useState<string[]>(() => folderCrumbs(state.sourceDir));

    // `state.sourceDir` is set by `applyFolder` in `main.tsx` just before the
    // gallery is rendered and the first page selected, so the first `page:changed`
    // of a folder is the moment it is known to be current.
    useEffect(() => {
        const sync = () => setCrumbs(folderCrumbs(state.sourceDir));
        on("page:changed", sync);
        on("image:loaded", sync);
        return () => {
            off("page:changed", sync);
            off("image:loaded", sync);
        };
    }, []);

    const isReader = mode === "reader";

    return (
        <header class="vx-topbar">
            {/* The app's own icon rather than a text lockup: at this size a wordmark
                is the loudest thing in the bar, and the identity is already in the
                window title and the favicon. Served from `static/`, so it swaps with
                the rest of the icon set when the app is re-branded. A `span`, not
                a link: the mark identifies the app, it navigates nowhere, so it
                takes no clicks, no focus and no pointer cursor. */}
            <span class="vx-brand" title="Fox Reader">
                <img
                    class="vx-brand__mark"
                    src="/static/android-chrome-192x192.png"
                    alt="Fox Reader"
                    width="22"
                    height="22"
                    draggable={false}
                />
            </span>

            <div class="vx-crumbs" aria-label="Current folder">
                {crumbs.length === 0 ? (
                    <span class="vx-crumbs__empty">No folder loaded</span>
                ) : (
                    crumbs.map((crumb, i) => (
                        <span class="vx-crumb" key={`${i}:${crumb}`}>
                            {i > 0 && (
                                <span class="vx-crumb__sep" aria-hidden="true">
                                    /
                                </span>
                            )}
                            <span class={`vx-crumb__text${i === crumbs.length - 1 ? " is-current" : ""}`}>{crumb}</span>
                        </span>
                    ))
                )}
            </div>

            <div class="vx-folder folder-nav">
                <input
                    type="text"
                    id="folderPathInput"
                    placeholder="Folder path, e.g. X:\My Comic\XYZ"
                    spellcheck={false}
                />
                <button id="loadFolderBtn" type="button" onClick={() => (window as any).loadLocalFolder?.()}>
                    <span class="material-icons" aria-hidden="true">
                        folder_open
                    </span>
                    <span class="vx-folder__label">Load</span>
                </button>
            </div>

            <div class="vx-topbar__actions">
                <button
                    type="button"
                    class={`vx-icon-btn vx-icon-btn--wide${isReader ? " is-active" : ""}`}
                    title={isReader ? "Exit reader mode (Esc)" : "Enter reader mode"}
                    aria-pressed={isReader}
                    onClick={() => setMode(isReader ? "work" : "reader")}
                >
                    <span class="material-icons" aria-hidden="true">
                        {isReader ? "exit_to_app" : "auto_stories"}
                    </span>
                    <span class="vx-icon-btn__label">Reader</span>
                </button>
                {onResetSections && (
                    <button
                        type="button"
                        class="vx-icon-btn"
                        title="Reset the section sizes to their defaults"
                        onClick={onResetSections}
                        aria-label="Reset section sizes"
                    >
                        <span class="material-icons" aria-hidden="true">
                            restart_alt
                        </span>
                    </button>
                )}
            </div>
        </header>
    );
}
