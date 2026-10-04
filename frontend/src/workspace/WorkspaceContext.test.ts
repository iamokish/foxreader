import { describe, expect, it } from "vitest";
import { resizePanelState } from "./WorkspaceContext";

describe("resizePanelState", () => {
    it("accumulates consecutive drag deltas from the latest panel size", () => {
        const panels = {
            viewer: { visible: true, size: 640 },
        };

        const afterFirstMove = resizePanelState(panels, "viewer", 20, 220, 1200);
        const afterSecondMove = resizePanelState(afterFirstMove, "viewer", 15, 220, 1200);

        expect(afterSecondMove.viewer.size).toBe(675);
    });

    it("clamps the latest size to the supplied bounds", () => {
        const panels = {
            viewer: { visible: true, size: 230 },
        };

        expect(resizePanelState(panels, "viewer", -100, 220, 1200).viewer.size).toBe(220);
    });
});
