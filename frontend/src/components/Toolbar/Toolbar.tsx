import { useState } from "preact/hooks";
import { useLayout } from "../../layouts/LayoutContext";
import { useTheme } from "../../themes/ThemeContext";
import { AppearanceModal } from "../Appearance";
import { CharactersModal } from "../Characters";
import { applyReadingDirection } from "../../entries";
import { getInitialFoxConfig } from "../../core/initialConfig";
import { getReadingDirection, type ReadingDirection } from "../../readingOrder";
import { emit } from "../../eventbus";
import { state } from "../../state";

const LTR_ICON = (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
    >
        <rect width="18" height="18" x="3" y="3" rx="2" />
        <path d="M8 12h8" />
        <path d="m12 16 4-4-4-4" />
    </svg>
);

const RTL_ICON = (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
    >
        <rect width="18" height="18" x="3" y="3" rx="2" />
        <path d="m12 8-4 4 4 4" />
        <path d="M16 12H8" />
    </svg>
);

interface ToolbarProps {
    /** Hide the folder path input/button, e.g. when a layout already renders it in a top bar. */
    showFolderNav?: boolean;
}

/**
 * Whether the backend built a bubble service for this session.
 *
 * Read once at module load, not per render: the flag is baked into the page by
 * the server and cannot change without a reload. `!== false` is deliberate --
 * a page cached from before the flag existed has `undefined` here, and losing
 * a working button to a stale cache is worse than showing one that is already
 * guarded server-side.
 */
const bubbleAvailable = getInitialFoxConfig().bubbleAvailable !== false;

export function Toolbar({ showFolderNav = true }: ToolbarProps) {
    const { layout } = useLayout();
    const { theme } = useTheme();
    const [appearanceOpen, setAppearanceOpen] = useState(false);
    const [charactersOpen, setCharactersOpen] = useState(false);
    const [direction, setDirection] = useState<ReadingDirection>(getReadingDirection);
    // Mirrors the Characters option switch: while it is off the roster button
    // stays hidden. Opening the roster must never flip this back on -- it only
    // opens the modal (the toast/state update lives in `main.tsx`, which owns
    // the switch, so this handler stays notify-free to avoid double toasts).
    const [charactersEnabled, setCharactersEnabled] = useState<boolean>(() => state.isCharactersEnabled);
    const handleCharactersToggle = (e: Event): void => {
        const checked = (e.target as HTMLInputElement).checked;
        state.isCharactersEnabled = checked;
        setCharactersEnabled(checked);
    };
    const layoutLabel = layout === "default" ? "Default" : "Basic";
    const themeLabel = theme === "light" ? "Light" : "Dark";

    const pickDirection = (next: ReadingDirection): void => {
        if (next === direction) return;
        // Resort first, then render: the numbers on the cards and the overlay
        // come from list positions, so the re-render picks them up.
        applyReadingDirection(next);
        setDirection(next);
        emit("entries:render");
    };
    return (
        <>
            {showFolderNav && (
                <div class="folder-nav">
                    <input
                        type="text"
                        id="folderPathInput"
                        placeholder="Input Folder Path Here (e.g., X:\My Comic\XYZ)"
                    />
                    <button id="loadFolderBtn" onClick={() => (window as any).loadLocalFolder?.()}>
                        <span class="material-icons">folder_open</span> Load Folder
                    </button>
                </div>
            )}

            <div class="buttons">
                <button id="rectCapture" title="Capture Region" onClick={() => (window as any).startCapture?.()}>
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2.25"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        class="lucide lucide-maximize-icon lucide-maximize"
                    >
                        <path d="M8 3H5a2 2 0 0 0-2 2v3" />
                        <path d="M21 8V5a2 2 0 0 0-2-2h-3" />
                        <path d="M3 16v3a2 2 0 0 0 2 2h3" />
                        <path d="M16 21h3a2 2 0 0 0 2-2v-3" />
                    </svg>
                </button>

                <button id="freeCapture" title="Free-form Capture" onClick={() => (window as any).startFreeCapture?.()}>
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2.25"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        class="lucide lucide-line-squiggle-icon lucide-line-squiggle"
                    >
                        <path d="M7 3.5c5-2 7 2.5 3 4C1.5 10 2 15 5 16c5 2 9-10 14-7s.5 13.5-4 12c-5-2.5.5-11 6-2" />
                    </svg>
                </button>

                {bubbleAvailable && (
                    <button id="bubCapture" title="Bubble Capture" onClick={() => (window as any).bubbleCapture?.()}>
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="2"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            class="lucide lucide-message-circle-icon lucide-message-circle"
                        >
                            <path d="M2.992 16.342a2 2 0 0 1 .094 1.167l-1.065 3.29a1 1 0 0 0 1.236 1.168l3.413-.998a2 2 0 0 1 1.099.092 10 10 0 1 0-4.777-4.719" />
                        </svg>
                    </button>
                )}

                <button id="PageCapture" title="Page TL" onClick={() => (window as any).pageCapture?.()}>
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        class="lucide lucide-book-open-text-icon lucide-book-open-text"
                    >
                        <path d="M12 7v14" />
                        <path d="M16 12h2" />
                        <path d="M16 8h2" />
                        <path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z" />
                        <path d="M6 12h2" />
                        <path d="M6 8h2" />
                    </svg>
                </button>

                <select id="languageSelect">
                    <option value="japanese">Japanese</option>
                    <option value="chinese">Chinese</option>
                    <option value="korean">Korean</option>
                </select>

                <div class="ml-control-box dropdown">
                    <button id="settingsBtn" class="dropdown-trigger">
                        <span class="material-icons">settings</span>
                        <span class="ml-label">Options</span>
                    </button>

                    <div class="dropdown-content">
                        <div class="menu-item">
                            <span class="ml-label">MTL</span>
                            <label class="switch">
                                <input type="checkbox" id="enableML" />
                                <span class="slider"></span>
                            </label>
                        </div>

                        <div class="menu-item">
                            <span class="ml-label">Gray OCR</span>
                            <label class="switch">
                                <input type="checkbox" id="grayScaleOCR" />
                                <span class="slider"></span>
                            </label>
                        </div>

                        <div class="menu-item">
                            <span class="ml-label">H-Fit</span>
                            <label class="switch">
                                <input type="checkbox" id="horizontalFit" />
                                <span class="slider"></span>
                            </label>
                        </div>

                        <div class="menu-item">
                            <span class="ml-label">Auto OCR</span>
                            <label class="switch">
                                <input type="checkbox" id="autoOCRresponse" />
                                <span class="slider"></span>
                            </label>
                        </div>

                        <div class="menu-item">
                            <span class="ml-label">Auto TL</span>
                            <label class="switch">
                                <input type="checkbox" id="autoTLresponse" checked />
                                <span class="slider"></span>
                            </label>
                        </div>

                        <div class="menu-item">
                            <span class="ml-label">Live Inpainting</span>
                            <label class="switch">
                                <input type="checkbox" id="liveInpaint" checked />
                                <span class="slider"></span>
                            </label>
                        </div>

                        <div class="menu-item">
                            <span class="ml-label">Context</span>
                            <label class="switch">
                                <input type="checkbox" id="enableContext" checked />
                                <span class="slider"></span>
                            </label>
                        </div>

                        <div class="menu-item">
                            <span class="ml-label">Characters</span>
                            <label class="switch">
                                <input
                                    type="checkbox"
                                    id="enableCharacters"
                                    checked={charactersEnabled}
                                    onChange={handleCharactersToggle}
                                />
                                <span class="slider"></span>
                            </label>
                        </div>

                        <div class="menu-item">
                            <span class="ml-label">Reading</span>
                            <div class="reading-dir-group" role="group" aria-label="Reading direction">
                                <button
                                    id="readingDirLtrBtn"
                                    type="button"
                                    class={`reading-dir-btn${direction === "ltr" ? " is-active" : ""}`}
                                    title="Left to right"
                                    aria-pressed={direction === "ltr"}
                                    onClick={() => pickDirection("ltr")}
                                >
                                    {LTR_ICON}
                                </button>
                                <button
                                    id="readingDirRtlBtn"
                                    type="button"
                                    class={`reading-dir-btn${direction === "rtl" ? " is-active" : ""}`}
                                    title="Right to left"
                                    aria-pressed={direction === "rtl"}
                                    onClick={() => pickDirection("rtl")}
                                >
                                    {RTL_ICON}
                                </button>
                            </div>
                        </div>

                        <div class="menu-item menu-item--appearance">
                            <button
                                id="appearanceSettingsBtn"
                                type="button"
                                class="appearance-menu-btn"
                                title="Change layout and theme"
                                onClick={() => setAppearanceOpen(true)}
                            >
                                <span class="material-icons" aria-hidden="true">
                                    palette
                                </span>
                                <span class="appearance-menu-btn__text">
                                    <span class="ml-label">Appearance</span>
                                    <span class="appearance-menu-btn__value">
                                        {layoutLabel} · {themeLabel}
                                    </span>
                                </span>
                                <span class="material-icons" aria-hidden="true">
                                    chevron_right
                                </span>
                            </button>
                        </div>

                        <a class="menu-item menu-link" href="/settings" title="Open settings">
                            <span class="material-icons" aria-hidden="true">
                                settings
                            </span>
                            <span class="ml-label">Settings</span>
                            <span class="material-icons" aria-hidden="true">
                                chevron_right
                            </span>
                        </a>
                    </div>
                </div>

                <button
                    id="charactersBtn"
                    type="button"
                    title="Manage characters (speaker metadata for VNTL models)"
                    hidden={!charactersEnabled}
                    style={{ display: charactersEnabled ? "" : "none" }}
                    // Opens the roster only: never touches `enableCharacters` /
                    // `state.isCharactersEnabled` here.
                    onClick={() => setCharactersOpen(true)}
                >
                    <span class="material-icons" aria-hidden="true">
                        group
                    </span>
                    <span class="ml-label">Characters</span>
                </button>
            </div>

            <AppearanceModal open={appearanceOpen} onClose={() => setAppearanceOpen(false)} />
            <CharactersModal open={charactersOpen} onClose={() => setCharactersOpen(false)} />
        </>
    );
}
