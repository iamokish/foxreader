import type { ComponentChildren } from "preact";
import type { PanelId } from "./WorkspaceContext";

interface WorkspacePanelProps {
    panelId: PanelId;
    title: string;
    visible: boolean;
    onToggle: (id: PanelId) => void;
    children: ComponentChildren;
}

export function WorkspacePanel({ panelId, title, visible, onToggle, children }: WorkspacePanelProps) {
    return (
        <div class={`ws-panel ${visible ? "" : "ws-panel--collapsed"}`} data-panel={panelId}>
            <div class="ws-panel__header">
                <span class="ws-panel__title">{title}</span>
                <button
                    class="ws-panel__toggle"
                    title={visible ? `Hide ${title}` : `Show ${title}`}
                    onClick={() => onToggle(panelId)}
                >
                    <span class="material-icons" style="font-size:16px;">
                        {visible ? "expand_less" : "expand_more"}
                    </span>
                </button>
            </div>
            <div class="ws-panel__body">{children}</div>
        </div>
    );
}
