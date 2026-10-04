import { useEffect, useState } from "preact/hooks";
import { useWorkspace } from "./WorkspaceContext";
import { state, type ReaderFitMode } from "../state";
import { on, off } from "../eventbus";
import { navigateGallery, refitViewer, zoomViewerBy, applyImageFit, updateReaderPageIndicator, toggleStripMode } from "../viewer";

const FIT_OPTIONS: { mode: ReaderFitMode; label: string; title: string }[] = [
    { mode: "width", label: "Width", title: "Fit width" },
    { mode: "height", label: "Height", title: "Fit height" },
];

/**
 * Floating reader-mode chrome: click zones + minimal controls.
 * Shown only while the workspace is in "reader" mode (mode switch is
 * in-session, no reload — the loaded folder/image survive).
 */
export function ReaderOverlay() {
    const { mode, setMode } = useWorkspace();
    const [, setTick] = useState(0);

    useEffect(() => {
        const onStripChanged = () => setTick((n) => n + 1);
        on("strip:changed", onStripChanged);
        return () => off("strip:changed", onStripChanged);
    }, []);

    useEffect(() => {
        updateReaderPageIndicator();
    });

    if (mode !== "reader") return null;

    const navigate = (dir: number) => {
        navigateGallery(dir);
    };

    const setFit = (fit: ReaderFitMode) => {
        state.readerFit = fit;
        const mainImg = document.getElementById("mainImage") as HTMLImageElement | null;
        const container = document.getElementById("imageContainer");
        if (mainImg && container) applyImageFit(container, mainImg);
        setTick((n) => n + 1);
    };

    return (
        <div class="ws-reader-overlay">
            <div class="ws-reader-controls">
                <button title="Previous page" onClick={() => navigate(-1)}>
                    <span class="material-icons" style="font-size:20px;">
                        chevron_left
                    </span>
                </button>
                <span class="ws-reader-page" title="Page">
                    <span id="readerPageIndicator">-</span>
                </span>
                <button title="Next page" onClick={() => navigate(1)}>
                    <span class="material-icons" style="font-size:20px;">
                        chevron_right
                    </span>
                </button>
                <span class="ws-reader-sep" />
                <button
                    class={`ws-reader-fit ${state.stripMode ? "active" : ""}`}
                    title={state.stripMode ? "Strip mode: ON — click to exit (S)" : "Strip/webtoon mode: continuous vertical reading (S)"}
                    onClick={() => toggleStripMode()}
                >
                    <span class="ws-reader-fit__label">Strip</span>
                </button>
                {!state.stripMode &&
                    FIT_OPTIONS.map((opt) => (
                        <button
                            key={opt.mode}
                            class={`ws-reader-fit ${state.readerFit === opt.mode ? "active" : ""}`}
                            title={opt.title}
                            onClick={() => setFit(opt.mode)}
                        >
                            <span class="ws-reader-fit__label">{opt.label}</span>
                        </button>
                    ))}
                <span class="ws-reader-sep" />
                <button title="Zoom out (-)" onClick={() => zoomViewerBy(-0.1)}>
                    <span class="material-icons" style="font-size:20px;">
                        remove
                    </span>
                </button>
                <button title="Reset zoom (0)" onClick={refitViewer}>
                    <span class="material-icons" style="font-size:20px;">
                        refresh
                    </span>
                </button>
                <button title="Zoom in (+)" onClick={() => zoomViewerBy(0.1)}>
                    <span class="material-icons" style="font-size:20px;">
                        add
                    </span>
                </button>
                <span class="ws-reader-sep" />
                <button
                    title="Toggle fullscreen (F)"
                    onClick={() => {
                        if (document.fullscreenElement) {
                            document.exitFullscreen().catch(() => {});
                        } else {
                            document.documentElement.requestFullscreen().catch(() => {});
                        }
                    }}
                >
                    <span class="material-icons" style="font-size:20px;">
                        fullscreen
                    </span>
                </button>
                <button title="Exit reader (Esc)" class="ws-reader-exit" onClick={() => setMode("work")}>
                    <span class="material-icons" style="font-size:20px;">
                        close
                    </span>
                </button>
            </div>
        </div>
    );
}
