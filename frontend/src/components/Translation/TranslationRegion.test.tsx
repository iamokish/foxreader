import { render } from "@testing-library/preact";
import { describe, expect, it } from "vitest";
import { TranslationRegion } from "./TranslationRegion";
import { translationPanelState, translationPanelText } from "../../translate";

describe("TranslationRegion", () => {
    it("renders the translated-text target used by translation and confirmation handlers", () => {
        const { container } = render(<TranslationRegion />);

        expect(container.querySelector("#translatedText")).not.toBeNull();
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
