import { BasicLayout } from "./BasicLayout";
import { DefaultLayout } from "./DefaultLayout";
import { useLayout } from "./LayoutContext";
import { InpaintModal, LoadingModal, ConfirmModal, BackendStatusOverlay } from "../components/common";
import { FolderModal } from "../components/Folder";

/**
 * Layout wrapper that renders the active layout.
 * Only "basic" (legacy) and "default" (workspace) are real layouts now;
 * "reader"/"translation" ideas were folded into the Default workspace.
 *
 * Shared fixed-position overlays (folder picker/inpaint/loading/confirm/backend
 * status) are mounted here so both layouts get them, matching the legacy
 * AppContent shell.
 */
export function ActiveLayout() {
    const { layout } = useLayout();

    return (
        <>
            {layout === "default" ? <DefaultLayout /> : <BasicLayout />}
            <FolderModal />
            <InpaintModal />
            <LoadingModal />
            <ConfirmModal />
            <BackendStatusOverlay />
        </>
    );
}
