/**
 * The legacy split view: viewer on the left, tools on the right.
 *
 * Its colours are the base theme tokens from `themes/dark.css` /
 * `themes/light.css`, untouched -- `themes/vulpes.css` is scoped to
 * `body.layout-vulpes` and `layouts/vulpes.css` to the same class, so nothing
 * about that layout reaches this one.
 *
 * The reader is shared rather than forked: `reader/reader.css` resolves every
 * colour through a token, and the `--vx-*` values it needs fall back to this
 * layout's own accent and surfaces (`:root` in that file), so the same overlay
 * comes out in Basic's colours here. `WorkspaceProvider` owns the mode, which is
 * the only thing that decides whether the overlay draws anything.
 */

import { ImageViewer as Viewer, GalleryBar } from "../components/Viewer";
import { Toolbar } from "../components/Toolbar";
import { OcrRegion, TranslatorButtons } from "../slots";
import { TranslationPanel } from "../components/Translation";
import { Reader } from "../reader";
import "./basic.css";

export function BasicLayout() {
    return (
        <>
            <div class="main-layout">
                <div class="viewer-section non-select">
                    <Viewer />
                    <GalleryBar />
                </div>

                <div class="content-area">
                    <Toolbar showFolderNav />
                    <OcrRegion />
                    <TranslatorButtons />
                    <TranslationPanel />
                </div>
            </div>

            <Reader />
        </>
    );
}
