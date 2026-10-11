import { ConfirmActionRow } from "./ConfirmActionRow";
import { EntriesRegion } from "./EntriesRegion";
import { TranslationBody } from "./TranslationBody";

/**
 * Basic's right column: the entry list and translation result in the tinted
 * wrapper, with the verbs in a row *below* it -- not inside it.
 *
 * The row used to be the wrapper's last flex child (via `TranslationRegion`,
 * which is body-plus-row). Hoisting it out is Basic-only: Vulpes never renders
 * this panel, composing `TranslationBody` and `ConfirmActionRow` itself, so no
 * other layout can tell. Every lookup is by id, and no stylesheet scopes the
 * row under the wrapper, so nothing else follows the move.
 */
export function TranslationPanel() {
    return (
        <>
            <div
                class="translation-wrapper"
                style="display: flex; flex-direction: column; height: 65vh; min-height: 350px;"
            >
                {/* Entries take 65%; the translation container below keeps
                    `flex: 1` and takes the ~35% remainder -- deliberately not
                    an explicit 35%, which plus this 65% and the 10px gutter
                    would overflow the wrapper by exactly that gutter. */}
                <div
                    class="panel"
                    style="height: 65%; min-height: 0; display: flex; flex-direction: column; padding: 8px; box-sizing: border-box; margin-bottom: 10px;"
                >
                    <EntriesRegion />
                </div>
                <TranslationBody />
            </div>
            <ConfirmActionRow />
        </>
    );
}
