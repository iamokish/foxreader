export function FontSelector() {
    return (
        <div
            class="custom-font-dropdown"
            style="position: relative; max-width: 220px; width: 100%; box-sizing: border-box;"
        >
            <div
                id="fontSelectTrigger"
                class="font-select-trigger"
                onClick={() => (window as any).toggleFontDropdown?.()}
            >
                <span
                    id="selectedFontLabel"
                    style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 5px; flex-grow: 1;"
                >
                    Select Font
                </span>
                <span class="material-icons" style="font-size: 16px; flex-shrink: 0;">
                    arrow_drop_up
                </span>
            </div>
            <input type="hidden" id="fontSelect" value="" />
            <div
                id="fontOptionsContainer"
                class="font-options-container custom-scrollbar"
                style="display: none; position: absolute; bottom: 34px; left: 0; width: 100%; max-height: 250px; overflow-y: auto; overflow-x: hidden; border-radius: 4px; z-index: 10000; box-sizing: border-box;"
            />
        </div>
    );
}
