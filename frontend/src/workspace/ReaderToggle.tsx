import { useWorkspace } from "./WorkspaceContext";

export function ReaderToggle() {
    const { mode, setMode } = useWorkspace();
    const isReader = mode === "reader";

    return (
        <button
            class={`theme-toggle ${isReader ? "theme-toggle--active" : ""}`}
            title={isReader ? "Exit Reader Mode (Esc)" : "Enter Reader Mode"}
            onClick={() => setMode(isReader ? "work" : "reader")}
        >
            <span class="material-icons">{isReader ? "exit_to_app" : "auto_stories"}</span>
        </button>
    );
}
