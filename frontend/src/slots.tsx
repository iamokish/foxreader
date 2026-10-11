export { ImageViewer as Viewer, GalleryBar } from "./components/Viewer";
export {
    EntriesRegion,
    TranslationRegion,
    TranslationBody,
    ConfirmActionRow,
    TranslatorButtons,
} from "./components/Translation";

export function OcrRegion() {
    return (
        <div class="textarea-wrapper">
            {/* No spellcheck: OCR text in any language would be underlined as misspelled.
                Boolean, not the string: the IDL is a real boolean, so Preact's
                `dom[name] = false` disables checking (and reflects
                `spellcheck="false"`); the string `"false"` would coerce to true
                and force checking ON. */}
            <textarea id="textArea" placeholder="Recognized text will appear here..." spellcheck={false} />
            <div id="ocrLoader" class="ocr-loader">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    );
}
