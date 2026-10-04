import { ImageViewer as Viewer, GalleryBar } from "../components/Viewer";
import { Toolbar } from "../components/Toolbar";
import { OcrRegion, TranslatorButtons } from "../slots";
import { TranslationPanel } from "../components/Translation";
import { getInitialFoxConfig } from "../core/initialConfig";

export function BasicLayout() {
    const config = getInitialFoxConfig();

    return (
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
    );
}
