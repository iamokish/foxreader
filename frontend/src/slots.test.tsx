import { render } from "@testing-library/preact";
import { describe, expect, it } from "vitest";

import { OcrRegion } from "./slots";

describe("OcrRegion", () => {
    it("renders the textarea", () => {
        const { container } = render(<OcrRegion />);
        expect(container.querySelector("#textArea")).not.toBeNull();
    });

    it("renders the OCR loading indicator", () => {
        const { container } = render(<OcrRegion />);
        expect(container.querySelector("#ocrLoader")).not.toBeNull();
        expect(container.querySelectorAll("#ocrLoader span").length).toBe(3);
    });
});
