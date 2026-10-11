import { BasicLayout } from "./BasicLayout";
import { VulpesLayout } from "./VulpesLayout";
import { useLayout } from "./LayoutContext";
import { InpaintModal, LoadingModal, ConfirmModal, BackendStatusOverlay } from "../components/common";
import { FolderModal } from "../components/Folder";
import { ReaderViewProvider } from "../reader";

/**
 * Layout wrapper that renders the active layout.
 *
 * Two: "vulpes" (three columns, resizable sections, footer chrome) and "basic"
 * (the legacy split view). Retired layout names are mapped by `normalizeLayout`.
 *
 * Shared fixed-position overlays (folder picker/inpaint/loading/confirm/backend
 * status) are mounted here so both layouts get them, matching the legacy
 * AppContent shell. `ReaderViewProvider` is one of them: the reader itself is
 * mounted by each layout, because it has to sit inside that layout's DOM, but its
 * view state is the same state either way and is cheapest to own in one place.
 */
export function ActiveLayout() {
    const { layout } = useLayout();

    return (
        <ReaderViewProvider>
            {layout === "vulpes" ? <VulpesLayout /> : <BasicLayout />}
            <FolderModal />
            <InpaintModal />
            <LoadingModal />
            <ConfirmModal />
            <BackendStatusOverlay />
        </ReaderViewProvider>
    );
}
