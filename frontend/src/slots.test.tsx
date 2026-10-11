/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve as resolvePath } from "node:path";
import { render } from "@testing-library/preact";
import { describe, expect, it } from "vitest";

import { OcrRegion } from "./slots";

describe("OcrRegion", () => {
    it("renders the textarea", () => {
        const { container } = render(<OcrRegion />);
        expect(container.querySelector("#textArea")).not.toBeNull();
    });

    it("disables spellcheck on the OCR textarea", () => {
        // Raw OCR output in any language would otherwise be underlined red.
        // Asserted on the source, not the DOM: jsdom implements no spellcheck
        // IDL, so Preact renders nothing observable there -- while in a real
        // browser the boolean below assigns the IDL and disables checking.
        // The tempting `spellcheck="false"` string must never come back: IDL
        // boolean coercion turns that non-empty string into TRUE. Comments
        // stripped first, so prose about the trap cannot trip the guard.
        const src = readFileSync(resolvePath(process.cwd(), "src/slots.tsx"), "utf8").replace(
            /\/\*[\s\S]*?\*\//g,
            "",
        );
        expect(src).toContain("spellcheck={false}");
        expect(src).not.toContain('spellcheck="false"');
        expect(src).not.toContain("spellcheck={'false'}");
        expect(src).not.toContain('spellcheck={"false"}');
    });

    it("renders the OCR loading indicator", () => {
        const { container } = render(<OcrRegion />);
        expect(container.querySelector("#ocrLoader")).not.toBeNull();
        expect(container.querySelectorAll("#ocrLoader span").length).toBe(3);
    });
});
