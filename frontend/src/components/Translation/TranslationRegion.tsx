import { ConfirmActionRow } from "./ConfirmActionRow";
import { TranslationBody } from "./TranslationBody";

/**
 * The translation result with its verbs directly underneath, in one fragment.
 *
 * Kept as the off-the-shelf composition; Basic instead places the row *below*
 * the tinted wrapper (`TranslationPanel`), and Vulpes renders `TranslationBody`
 * with the row in its footer -- which is why the two stay separate components
 * (see `TranslationBody`).
 */
export function TranslationRegion() {
    return (
        <>
            <TranslationBody />
            <ConfirmActionRow />
        </>
    );
}
