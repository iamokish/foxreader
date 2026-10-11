/**
 * The translation result area, without the action row.
 *
 * `TranslationRegion` is `TranslationBody` plus `ConfirmActionRow`, which is what
 * every layout except Vulpes wants: the verbs sit right under the translation they
 * act on. Vulpes puts the verbs in its footer instead, so it renders this half.
 *
 * Split rather than parameterised because the split is not a style: the row carries
 * ids (`#confirmTranslationBtn` and friends) that `getElementById` resolves, so a
 * layout has to choose which of the two components to render -- it cannot render the
 * whole region and hide half of it without leaving duplicate ids in the document.
 *
 * `#translatedText`'s `data-state` is what Confirm Entry reads to decide whether the
 * panel holds a translation or a message -- see `translationPanelState`. It starts as
 * a message.
 */

export function TranslationBody() {
    return (
        <div
            id="translationContainer"
            style="flex: 1; min-height: 0; display: flex; flex-direction: column; width: 100%; box-sizing: border-box; justify-content: center; position: relative;"
        >
            <div id="translatedText" class="panel error" data-state="error">
                <strong>Select a translator</strong>
            </div>
        </div>
    );
}
