import type { ComponentChildren } from "preact";
import { useEffect } from "preact/hooks";
import "./DefaultLayout.css";
import "./../workspace/workspace.css";
import { Toolbar } from "../components/Toolbar";
import { ImageViewer as Viewer, GalleryBar } from "../components/Viewer";
import { OcrRegion, EntriesRegion, TranslationRegion, TranslatorButtons } from "../slots";
import { getInitialFoxConfig } from "../core/initialConfig";
import { useWorkspace, Splitter, WorkspacePanel, ReaderOverlay, ReaderToggle } from "../workspace";
import { applyImageFit, refitViewer, buildStrip, destroyStrip } from "../viewer";
import { state } from "../state";
import { emit } from "../eventbus";

function ToolbarSlot() {
    const config = getInitialFoxConfig();
    return <Toolbar showFolderNav={false} />;
}

function ActionBar() {
    return (
        <div class="default-topbar">
            <div class="folder-nav default-topbar__folder">
                <input type="text" id="folderPathInput" placeholder="Input Folder Path Here (e.g., X:\My Comic\XYZ)" />
                <button id="loadFolderBtn" onClick={() => (window as any).loadLocalFolder?.()}>
                    <span class="material-icons">folder_open</span> Load Folder
                </button>
            </div>

            <div class="default-topbar__actions">
                <ReaderToggle />
            </div>
        </div>
    );
}

function WorkspacePanelBody({ children }: { children: ComponentChildren }) {
    return <>{children}</>;
}

/** The gallery sits below its splitter, so its height changes opposite to pointer Y movement. */
export function getGalleryResizeDelta(pointerDelta: number): number {
    return -pointerDelta;
}

export function DefaultLayout() {
    const workspace = useWorkspace();
    const mode = workspace.mode;
    const config = getInitialFoxConfig();

    useEffect(() => {
        if (mode === "reader") {
            const shouldStrip =
                state.stripOverride === "on" || (state.stripOverride === "auto" && state.stripEligible);
            if (state.stripMode !== shouldStrip) {
                state.stripMode = shouldStrip;
                emit("strip:changed");
            }
            if (state.stripMode) buildStrip();
            else destroyStrip();
        } else {
            destroyStrip();
            if (state.stripMode) {
                state.stripMode = false;
                emit("strip:changed");
            }
        }
        if (!state.currentImageFile) return;
        const t = setTimeout(() => {
            const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
            const container = document.getElementById("imageContainer");
            if (mainImg && container) applyImageFit(container, mainImg);
        }, 0);
        return () => clearTimeout(t);
    }, [mode]);
    const viewer = workspace.getPanel("viewer");
    const gallery = workspace.getPanel("gallery");
    const ocr = workspace.getPanel("ocr");
    const entries = workspace.getPanel("entries");
    const translation = workspace.getPanel("translation");

    const resizeViewer = (delta: number) => workspace.resizePanel("viewer", delta, 220, window.innerWidth - 320);
    const resizeGallery = (delta: number) => workspace.resizePanel("gallery", delta, 40, 320);
    const resizeOcr = (delta: number) => workspace.resizePanel("ocr", delta, 60, 480);
    const resizeEntries = (delta: number) => workspace.resizePanel("entries", delta, 60, 600);

    return (
        <>
            <ActionBar />

            <div class="ws-root">
                <div class="ws-capture-rail">
                    <ToolbarSlot />
                </div>

                <div class="ws-main">
                    {/* Left column: viewer + gallery */}
                    <div class="ws-col--viewer" style={mode === "reader" ? undefined : { width: `${viewer.size}px` }}>
                        <div class="ws-cell--fill">
                            <Viewer />
                        </div>

                        {gallery.visible && (
                            <Splitter
                                axis="horizontal"
                                onResize={(delta) => resizeGallery(getGalleryResizeDelta(delta))}
                                onResizeEnd={refitViewer}
                                title="Drag to resize gallery"
                            />
                        )}

                        <div
                            class="ws-cell ws-cell--gallery"
                            style={{ height: gallery.visible ? `${gallery.size}px` : undefined }}
                        >
                            <WorkspacePanel
                                panelId="gallery"
                                title="Gallery"
                                visible={gallery.visible}
                                onToggle={workspace.togglePanel}
                            >
                                <GalleryBar />
                            </WorkspacePanel>
                        </div>
                    </div>

                    <Splitter
                        axis="vertical"
                        onResize={resizeViewer}
                        onResizeEnd={refitViewer}
                        title="Drag to resize viewer"
                    />

                    {/* Right column: OCR / entries / translation */}
                    <div class="ws-col--side">
                        <div class="ws-cell" style={{ height: ocr.visible ? `${ocr.size}px` : undefined }}>
                            <WorkspacePanel
                                panelId="ocr"
                                title="OCR"
                                visible={ocr.visible}
                                onToggle={workspace.togglePanel}
                            >
                                <WorkspacePanelBody>
                                    <OcrRegion />
                                    <TranslatorButtons />
                                </WorkspacePanelBody>
                            </WorkspacePanel>
                        </div>

                        {ocr.visible && (
                            <Splitter
                                axis="horizontal"
                                onResize={resizeOcr}
                                onResizeEnd={refitViewer}
                                title="Drag to resize OCR"
                            />
                        )}

                        <div class="ws-cell" style={{ height: entries.visible ? `${entries.size}px` : undefined }}>
                            <WorkspacePanel
                                panelId="entries"
                                title="Entries"
                                visible={entries.visible}
                                onToggle={workspace.togglePanel}
                            >
                                <EntriesRegion />
                            </WorkspacePanel>
                        </div>

                        {entries.visible && (
                            <Splitter
                                axis="horizontal"
                                onResize={resizeEntries}
                                onResizeEnd={refitViewer}
                                title="Drag to resize entries"
                            />
                        )}

                        <div class="ws-cell--fill">
                            <WorkspacePanel
                                panelId="translation"
                                title="Translation"
                                visible={translation.visible}
                                onToggle={workspace.togglePanel}
                            >
                                <TranslationRegion />
                            </WorkspacePanel>
                        </div>
                    </div>
                </div>
            </div>

            <ReaderOverlay />
        </>
    );
}
