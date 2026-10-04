import { useEffect, useRef } from "preact/hooks";
import { useLayout } from "../../layouts/LayoutContext";
import type { Layout } from "../../layouts/LayoutContext";
import { useTheme } from "../../themes/ThemeContext";
import type { Theme } from "../../themes/ThemeContext";
import "./AppearanceModal.css";

interface AppearanceModalProps {
    open: boolean;
    onClose: () => void;
}

const LAYOUT_OPTIONS: { value: Layout; title: string; description: string }[] = [
    {
        value: "basic",
        title: "Basic",
        description: "Classic split view. Viewer on the left, tools on the right.",
    },
    {
        value: "default",
        title: "Default",
        description: "Workspace with resizable panels, a top bar and reader mode.",
    },
];

const THEME_OPTIONS: { value: Theme; title: string; description: string }[] = [
    {
        value: "light",
        title: "Light",
        description: "Bright surfaces for daylight reading.",
    },
    {
        value: "dark",
        title: "Dark",
        description: "Dim surfaces for night reading.",
    },
];

function LayoutPreview({ value }: { value: Layout }) {
    if (value === "default") {
        return (
            <span class="appearance-preview" aria-hidden="true">
                <span class="ap-mock ap-mock--default">
                    <span class="ap-mock__topbar" />
                    <span class="ap-mock__main">
                        <span class="ap-mock__col-viewer" />
                        <span class="ap-mock__col-side">
                            <span class="ap-mock__cell" />
                            <span class="ap-mock__cell" />
                            <span class="ap-mock__cell" />
                        </span>
                    </span>
                </span>
            </span>
        );
    }
    return (
        <span class="appearance-preview" aria-hidden="true">
            <span class="ap-mock">
                <span class="ap-mock__viewer" />
                <span class="ap-mock__side">
                    <span class="ap-mock__bar" />
                    <span class="ap-mock__bar" />
                    <span class="ap-mock__bar is-wide" />
                </span>
            </span>
        </span>
    );
}

function ThemePreview({ value }: { value: Theme }) {
    return (
        <span
            class={`appearance-preview appearance-preview--theme appearance-preview--theme-${value}`}
            aria-hidden="true"
        >
            <span class="ap-theme-dot">
                <span class="material-icons">{value === "light" ? "light_mode" : "dark_mode"}</span>
            </span>
            <span class="ap-theme-lines">
                <i />
                <i />
                <i />
            </span>
        </span>
    );
}

/**
 * Centered Appearance overlay: pick a layout (basic/default) and a theme
 * (light/dark). Opened from the "Appearance" row in the Options dropdown.
 *
 * Styling is 100% theme tokens so the card follows the live theme, with
 * body.layout-* overrides for per-layout rounding.
 */
export function AppearanceModal({ open, onClose }: AppearanceModalProps) {
    const { layout, setLayout } = useLayout();
    const { theme, setTheme } = useTheme();
    const closeRef = useRef<HTMLButtonElement>(null);

    useEffect(() => {
        if (!open) return;
        closeRef.current?.focus();
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                e.stopPropagation();
                onClose();
            }
        };
        document.addEventListener("keydown", onKey, true);
        const prevOverflow = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        return () => {
            document.removeEventListener("keydown", onKey, true);
            document.body.style.overflow = prevOverflow;
        };
    }, [open, onClose]);

    if (!open) return null;

    const onBackdropClick = (e: MouseEvent) => {
        if (e.target === e.currentTarget) onClose();
    };

    const activeLayoutLabel = layout === "default" ? "Default" : "Basic";
    const activeThemeLabel = theme === "light" ? "Light" : "Dark";

    return (
        <div id="appearanceOverlay" class="appearance-overlay" onClick={onBackdropClick}>
            <div class="appearance-card" role="dialog" aria-modal="true" aria-label="Appearance settings">
                <div class="appearance-header">
                    <div class="appearance-header__titles">
                        <h2 class="appearance-title">
                            <span class="material-icons" aria-hidden="true">
                                palette
                            </span>
                            Appearance
                        </h2>
                        <p class="appearance-subtitle">Layouts and themes follow the app colours.</p>
                    </div>
                    <button
                        id="appearanceCloseBtn"
                        ref={closeRef}
                        type="button"
                        class="appearance-close"
                        aria-label="Close appearance settings"
                        title="Close"
                        onClick={onClose}
                    >
                        <span class="material-icons" aria-hidden="true">
                            close
                        </span>
                    </button>
                </div>

                <div class="appearance-section">
                    <h3 class="appearance-section__label">
                        Layout
                        <span class="appearance-section__current">Current: {activeLayoutLabel}</span>
                    </h3>
                    <div class="appearance-grid" role="radiogroup" aria-label="Layout">
                        {LAYOUT_OPTIONS.map((item) => {
                            const active = layout === item.value;
                            return (
                                <button
                                    key={item.value}
                                    type="button"
                                    role="radio"
                                    aria-checked={active}
                                    data-layout={item.value}
                                    class={`appearance-option${active ? " is-active" : ""}`}
                                    onClick={() => {
                                        if (item.value !== layout) setLayout(item.value);
                                        else onClose();
                                    }}
                                >
                                    <span class="appearance-option__check" aria-hidden="true">
                                        <span class="material-icons">check</span>
                                    </span>
                                    <LayoutPreview value={item.value} />
                                    <span class="appearance-option__body">
                                        <span class="appearance-option__title">{item.title}</span>
                                        <span class="appearance-option__desc">{item.description}</span>
                                    </span>
                                </button>
                            );
                        })}
                    </div>
                    <p class="appearance-note">
                        <span class="material-icons" aria-hidden="true">
                            info
                        </span>
                        Switching layout reloads the app so panels rebuild cleanly.
                    </p>
                </div>

                <div class="appearance-section">
                    <h3 class="appearance-section__label">
                        Theme
                        <span class="appearance-section__current">Current: {activeThemeLabel}</span>
                    </h3>
                    <div class="appearance-grid" role="radiogroup" aria-label="Theme">
                        {THEME_OPTIONS.map((item) => {
                            const active = theme === item.value;
                            return (
                                <button
                                    key={item.value}
                                    type="button"
                                    role="radio"
                                    aria-checked={active}
                                    data-theme={item.value}
                                    class={`appearance-option${active ? " is-active" : ""}`}
                                    onClick={() => setTheme(item.value)}
                                >
                                    <span class="appearance-option__check" aria-hidden="true">
                                        <span class="material-icons">check</span>
                                    </span>
                                    <ThemePreview value={item.value} />
                                    <span class="appearance-option__body">
                                        <span class="appearance-option__title">{item.title}</span>
                                        <span class="appearance-option__desc">{item.description}</span>
                                    </span>
                                </button>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
}
