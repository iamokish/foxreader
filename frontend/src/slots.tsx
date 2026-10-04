export { ImageViewer as Viewer, GalleryBar } from "./components/Viewer";
export { EntriesRegion, TranslationRegion, TranslatorButtons } from "./components/Translation";

export function OcrRegion() {
    return (
        <div class="textarea-wrapper">
            <textarea id="textArea" placeholder="Recognized text will appear here..." />
            <div id="ocrLoader" class="ocr-loader">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    );
}
