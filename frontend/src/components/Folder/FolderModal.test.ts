import { describe, expect, it } from "vitest";
import { canLoadFolder, subfolderOf } from "./FolderModal";
import type { DirReport, FolderInspectResponse } from "../../types";

function report(overrides: Partial<DirReport> = {}): DirReport {
    return {
        path: "/scan",
        exists: true,
        is_dir: true,
        is_file: false,
        readable: true,
        writable: true,
        executable: true,
        image_count: null,
        parent: "/",
        parent_exists: true,
        parent_writable: true,
        reason: "",
        can_create: false,
        ok_as_source: true,
        ok_as_dest: true,
        ...overrides,
    };
}

function answer(source: Partial<DirReport>, dest: Partial<DirReport> = {}): FolderInspectResponse {
    return {
        status: "success",
        source: report({ image_count: 3, ...source }),
        dest: report({ path: "/scan/fox_tled", ...dest }),
        default_dest: "/scan/fox_tled",
        remembered_dest: "",
        same_dir: false,
    };
}

describe("canLoadFolder", () => {
    it("allows a readable source with pages and a writable destination", () => {
        expect(canLoadFolder(answer({}))).toBe(true);
    });

    it("allows a destination that does not exist yet", () => {
        expect(
            canLoadFolder(answer({}, { exists: false, is_dir: false, writable: false, can_create: true })),
        ).toBe(true);
    });

    it("refuses before the first check has come back", () => {
        expect(canLoadFolder(null)).toBe(false);
    });

    it("refuses a source with no images", () => {
        // Loading an empty folder leaves the viewer blank with no explanation.
        expect(canLoadFolder(answer({ image_count: 0 }))).toBe(false);
    });

    it("refuses a source that cannot be read", () => {
        expect(canLoadFolder(answer({ ok_as_source: false, readable: false }))).toBe(false);
    });

    it("refuses a destination that can be neither written to nor created", () => {
        expect(canLoadFolder(answer({}, { ok_as_dest: false, writable: false }))).toBe(false);
    });

    it("refuses an uncounted source rather than guessing", () => {
        expect(canLoadFolder(answer({ image_count: null }))).toBe(false);
    });
});

describe("subfolderOf", () => {
    it("appends the name to a cleaned source path", () => {
        expect(subfolderOf("D:\\Comics\\Ch 1", "fox_tled")).toBe("D:/Comics/Ch 1/fox_tled");
    });

    it("strips quotes and trailing separators", () => {
        expect(subfolderOf("'/home/me/scan/'", "typeset")).toBe("/home/me/scan/typeset");
    });

    it("has nothing to append to without a source", () => {
        expect(subfolderOf("   ", "fox_tled")).toBe("");
    });
});
