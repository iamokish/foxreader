/**
 * Vulpes -- a three-column workspace with a reader on top.
 *
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │ topbar: icon · folder/chapter · folder field · reader         │
 *   ├────────┬───────────────────────────────────┬─────────────────┤
 *   │ PAGES  │ tools                             │ EDITOR          │
 *   │ (rail) │ ─────────── stage ─────────────── │  Entries        │
 *   │        │ captured text                     │  Translators    │
 *   ├────────┴───────────────────────────────────┴─────────────────┤
 *   │ statusbar: page · Original/Saved · zoom · actions · sections  │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * Every element the imperative modules bind to is mounted here exactly once and
 * never unmounted:
 *
 *   #imageContainer, #mainImage, #drawCanvas, #cropSelector   `ImageViewer`
 *   #gallery                                                  `GalleryBar`
 *   #textArea, #ocrLoader                                     `OcrRegion`
 *   #savedEntriesList                                         `EntriesRegion`
 *   #translationContainer, #translatedText                    `TranslationRegion`
 *   #folderPathInput, #loadFolderBtn                          `VulpesTopBar`
 *   the capture buttons, language select, Options dropdown     `Toolbar`
 *
 * That is not stylistic: `main.tsx` calls `initPanning`, `initZoom`,
 * `setupImageErrorHandler` and the dropdown wiring once, at startup, against those
 * ids. Reader mode therefore *covers* the stage instead of replacing it -- see
 * `reader/Reader.tsx`.
 *
 * The editor column holds the entry cards as they are. Each card already carries the
 * text, speaker, layer, font, outline, spacing, transform, background and clean
 * controls (`entryOptions.ts`), so the Style and Transform sections of the
 * editor are those cards rather than a second set of controls that would have
 * to be kept in step with the first.
 *
 * The action row (`#confirmActionRow`) is rendered by `TranslationRegion` in the
 * translation section, as it is in every layout; the footer adds a second, icon-only
 * set of the same controls that *moves the real buttons* by clicking them. See
 * `VulpesStatusBar`.
 */

import { useCallback, useEffect, useRef } from "preact/hooks";
import type { ComponentChildren } from "preact";
import "../themes/vulpes.css";
import "./vulpes.css";
import { Toolbar } from "../components/Toolbar";
import { ImageViewer, GalleryBar } from "../components/Viewer";
import { EntriesRegion, OcrRegion, TranslationBody, TranslatorButtons } from "../slots";
import { Splitter, useWorkspace } from "../workspace";
import { refitViewer } from "../viewer";
import { Reader, ReaderViewProvider } from "../reader";
import { useVulpesPanels, VulpesPanelsProvider, type VxPanelId } from "../vulpes/VulpesPanelsContext";
import { VulpesTopBar } from "../vulpes/VulpesTopBar";
import { VulpesStatusBar } from "../vulpes/VulpesStatusBar";

/**
 * A splitter below its panel resizes it *against* the pointer.
 *
 * The captured-text box sits under the stage and above its splitter, so dragging
 * down grows it; the entry list in the editor is the same shape. Dragging a
 * splitter that sits to the *left* of its panel (the editor's) is likewise
 * inverted. Pulled out as a function so the sign is stated once and tested.
 */
export function invertDelta(pointerDelta: number): number {
    return -pointerDelta;
}

interface SectionProps {
    id: VxPanelId;
    title: string;
    /** Shown to the right of the title, e.g. a count or a hint. */
    note?: string;
    /** Rendered in the header, right of the note: small actions belonging to the section. */
    actions?: ComponentChildren;
    /**
     * Whether this `<section>` is the element a splitter resizes.
     *
     * Only the entry list is: the captured-text panel's size lives on the
     * `.vx-capture` wrapper around it, because the collapsed state has to keep the
     * header visible at its natural height.
     */
    sized?: boolean;
    /**
     * False for sections that cannot collapse: the header renders statically
     * (title, no toggle) instead of a collapse button. There is then no
     * affordance that can hide them -- and none must be added elsewhere
     * (status bar, keyboard) without revisiting this.
     */
    collapsible?: boolean;
    children: ComponentChildren;
}

function Section({ id, title, note, actions, sized = false, collapsible = true, children }: SectionProps) {
    const { getPanel, togglePanel, registerPanel } = useVulpesPanels();
    const panel = getPanel(id);
    return (
        <section
            class={`vx-section${panel.visible ? "" : " is-collapsed"}`}
            data-section={id}
            ref={sized ? (el) => registerPanel(id, el as HTMLElement | null) : undefined}
            style={sized && panel.visible ? { height: `${panel.size}px` } : undefined}
        >
            <header class={collapsible ? "vx-section__head" : "vx-section__head vx-section__head--static"}>
                {collapsible ? (
                    <button
                        type="button"
                        class="vx-section__toggle"
                        title={panel.visible ? `Hide ${title}` : `Show ${title}`}
                        aria-expanded={panel.visible}
                        onClick={() => togglePanel(id)}
                    >
                        <span class="material-icons" aria-hidden="true">
                            {panel.visible ? "expand_more" : "chevron_right"}
                        </span>
                        <span class="vx-section__title">{title}</span>
                    </button>
                ) : (
                    <span class="vx-section__title">{title}</span>
                )}
                {note && <span class="vx-section__note">{note}</span>}
                {actions && <span class="vx-section__actions">{actions}</span>}
            </header>
            <div class="vx-section__body">{children}</div>
        </section>
    );
}

function VulpesShell() {
    const { mode } = useWorkspace();
    const { getPanel, resizePanel, commitResize, resetPanels, registerPanel } = useVulpesPanels();
    const stageRef = useRef<HTMLDivElement>(null);

    const pages = getPanel("pages");
    const editor = getPanel("editor");
    const ocr = getPanel("ocr");
    const entries = getPanel("entries");

    /**
     * Re-fit the page whenever the stage changes size, once per frame at most.
     *
     * A splitter drag resizes the stage every frame, and the page inside it was laid
     * out for the old size -- so without this the artwork sits at the wrong scale for
     * the whole drag and snaps into place on release, which is most of what made
     * resizing feel laggy rather than live.
     *
     * A `ResizeObserver` rather than the splitter's own callback, because the stage
     * also changes size when a section is collapsed, when the editor's splitter
     * moves, and on a window resize. Coalesced into one `requestAnimationFrame`:
     * `applyImageFit` re-derives the zoom and re-syncs the SVG overlay, which is
     * cheap once a frame and wasteful several times.
     */
    useEffect(() => {
        const el = stageRef.current;
        if (!el || typeof ResizeObserver === "undefined") return;
        let frame = 0;
        const observer = new ResizeObserver(() => {
            if (frame) return;
            frame = requestAnimationFrame(() => {
                frame = 0;
                refitViewer();
            });
        });
        observer.observe(el);
        return () => {
            observer.disconnect();
            if (frame) cancelAnimationFrame(frame);
        };
    }, []);

    /** Persist the size a drag arrived at. The re-fit is the observer's job. */
    const onResizeEnd = useCallback(() => {
        commitResize();
    }, [commitResize]);

    return (
        <>
            <div class={`vx-root${mode === "reader" ? " is-reader" : ""}`}>
                <VulpesTopBar onResetSections={resetPanels} />

                <div class="vx-body">
                    <aside
                        class={`vx-rail${pages.visible ? "" : " is-collapsed"}`}
                        aria-label="Pages"
                        ref={(el) => registerPanel("pages", el as HTMLElement | null)}
                        style={{ width: `${pages.size}px` }}
                    >
                        <header class="vx-rail__head">
                            <span class="vx-rail__title">Pages</span>
                        </header>
                        <div class="vx-rail__body">
                            <GalleryBar />
                        </div>
                    </aside>

                    {pages.visible && (
                        <Splitter
                            axis="vertical"
                            onResize={(delta) => resizePanel("pages", delta)}
                            onResizeEnd={onResizeEnd}
                            title="Drag to resize the pages rail"
                        />
                    )}

                    <main class="vx-canvas">
                        <div class="vx-tools">
                            <Toolbar showFolderNav={false} />
                        </div>

                        {/* The editor's viewport. Mounted once, for the lifetime of
                            the page: `initPanning`/`initZoom` bind to it at startup. */}
                        <div class="vx-stage" ref={stageRef}>
                            <ImageViewer />
                        </div>

                        {ocr.visible && (
                            <Splitter
                                axis="horizontal"
                                onResize={(delta) => resizePanel("ocr", invertDelta(delta))}
                                onResizeEnd={onResizeEnd}
                                title="Drag to resize the captured text"
                            />
                        )}

                        <div
                            class={`vx-capture${ocr.visible ? "" : " is-collapsed"}`}
                            ref={(el) => registerPanel("ocr", el as HTMLElement | null)}
                            style={ocr.visible ? { height: `${ocr.size}px` } : undefined}
                        >
                            <Section id="ocr" title="Captured text" note="OCR">
                                <div class="vx-capture__body">
                                    <OcrRegion />
                                </div>
                            </Section>
                        </div>
                    </main>

                    {editor.visible && (
                        <Splitter
                            axis="vertical"
                            onResize={(delta) => resizePanel("editor", invertDelta(delta))}
                            onResizeEnd={onResizeEnd}
                            title="Drag to resize the editor"
                        />
                    )}

                    {/* No column header here: the rail keeps its "Pages" head,
                        but this column's own head was removed -- the entries
                        and translation sections below name themselves, and the
                        aside keeps `aria-label="Editor"` so nothing
                        accessibility-wise goes with it. */}
                    <aside
                        class={`vx-editor${editor.visible ? "" : " is-collapsed"}`}
                        aria-label="Editor"
                        ref={(el) => registerPanel("editor", el as HTMLElement | null)}
                        style={{ width: `${editor.size}px` }}
                    >
                        <div class="vx-editor__body">
                            <Section
                                id="entries"
                                title="Entries"
                                note="text · style · transform"
                                sized
                                collapsible={false}
                            >
                                <EntriesRegion />
                            </Section>

                            {entries.visible && (
                                <Splitter
                                    axis="horizontal"
                                    onResize={(delta) => resizePanel("entries", delta)}
                                    onResizeEnd={commitResize}
                                    title="Drag to resize the entry list"
                                />
                            )}

                            <div class="vx-editor__translation">
                                {/* The translator picker lives in this header, beside
                                    the title, rather than under the captured text:
                                    the header height is fixed, so the buttons ride
                                    in a horizontal scroller that never grows it. */}
                                <header class="vx-section__head vx-section__head--static">
                                    <span class="vx-section__title">Translators</span>
                                    <TranslatorButtons />
                                </header>
                                <div class="vx-translation">
                                    {/* The action row is NOT here: it is the footer's,
                                        further down. `TranslationRegion` renders it
                                        inline for every other layout, and `#confirmActionRow`
                                        has to exist exactly once, so Vulpes supplies its
                                        own translation body instead of the region. */}
                                    <TranslationBody />
                                </div>
                            </div>
                        </div>
                    </aside>
                </div>

                <VulpesStatusBar />
            </div>

            <Reader />
        </>
    );
}

export function VulpesLayout() {
    return (
        <VulpesPanelsProvider>
            <ReaderViewProvider>
                <VulpesShell />
            </ReaderViewProvider>
        </VulpesPanelsProvider>
    );
}
