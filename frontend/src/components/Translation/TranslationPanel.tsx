import { EntriesRegion } from "./EntriesRegion";
import { TranslationRegion } from "./TranslationRegion";

export function TranslationPanel() {
    return (
        <div
            class="translation-wrapper"
            style="display: flex; flex-direction: column; height: 65vh; min-height: 350px;"
        >
            <div
                class="panel"
                style="height: 50%; min-height: 0; display: flex; flex-direction: column; padding: 8px; box-sizing: border-box; margin-bottom: 10px;"
            >
                <EntriesRegion />
            </div>
            <TranslationRegion />
        </div>
    );
}
