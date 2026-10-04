/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const stylesheet = readFileSync(resolve(process.cwd(), "src/workspace/workspace.css"), "utf8");

describe("Default workspace viewer sizing", () => {
    it("overrides the legacy 80vh image-container height with its flex cell height", () => {
        expect(stylesheet).toMatch(/\.ws-cell--fill\s*>\s*#imageContainer\s*\{[^}]*height:\s*100%/s);
    });
});
