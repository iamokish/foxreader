import { render } from "@testing-library/preact";
import { describe, expect, it } from "vitest";
import { TranslationRegion } from "./TranslationRegion";
import { translationPanelState, translationPanelText } from "../../translate";
import { setTextAlignmentValue } from "../../ui";

describe("TranslationRegion", () => {
    it("renders the translated-text target used by translation and confirmation handlers", () => {
        const { container } = render(<TranslationRegion />);

        expect(container.querySelector("#translatedText")).not.toBeNull();
    });

    it("carries the alignment glyph the cycle repaints, beside the word", () => {
        // The word always reads "Align"; the glyph is the state. Basic hides
        // every other glyph but must keep this one visible -- see actionRow.css.
        const { container } = render(<TranslationRegion />);
        const btn = container.querySelector<HTMLElement>("#alignmentCycleBtn")!;
        expect(btn.querySelector(".btn-icon")?.textContent).toBe("format_align_center");
        expect(btn.querySelector(".btn-label")?.textContent).toBe("Align");

        setTextAlignmentValue("left");
        expect(btn.querySelector(".btn-icon")?.textContent).toBe("format_align_left");
        expect(btn.title).toBe("Alignment: left");
        expect(btn.dataset.current).toBe("left");

        setTextAlignmentValue("right");
        expect(btn.querySelector(".btn-icon")?.textContent).toBe("format_align_right");
    });

    it("opens as a message, not as a translation Confirm Entry would store", () => {
        const { container } = render(<TranslationRegion />);
        const panel = container.querySelector<HTMLElement>("#translatedText")!;

        // "Select a translator" is a prompt. It used to reach the entry as its
        // translated text when Confirm Entry was pressed before translating.
        expect(panel.textContent?.trim()).toBe("Select a translator");
        expect(translationPanelState(panel)).toBe("error");
        expect(translationPanelText(panel)).toBe("");
    });
});
