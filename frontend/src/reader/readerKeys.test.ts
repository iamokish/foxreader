import { beforeEach, describe, expect, it, vi } from "vitest";
import { handleReaderKey, setReaderKeyHandler } from "./readerKeys";

describe("handleReaderKey", () => {
    beforeEach(() => setReaderKeyHandler(null));

    it("declines everything while no reader is mounted", () => {
        // Which is what lets `main.tsx` keep its own reader bindings as the fallback.
        expect(handleReaderKey(new KeyboardEvent("keydown", { key: "ArrowRight" }))).toBe(false);
    });

    it("hands the event to the registered reader", () => {
        const handler = vi.fn(() => true);
        setReaderKeyHandler(handler);
        const event = new KeyboardEvent("keydown", { key: "2" });
        expect(handleReaderKey(event)).toBe(true);
        expect(handler).toHaveBeenCalledWith(event);
    });

    it("reports not-consumed for a key the reader passes on", () => {
        setReaderKeyHandler(() => false);
        expect(handleReaderKey(new KeyboardEvent("keydown", { key: "Escape" }))).toBe(false);
    });

    it("treats a throwing handler as not-consumed rather than taking the app's only listener down", () => {
        setReaderKeyHandler(() => {
            throw new Error("boom");
        });
        expect(handleReaderKey(new KeyboardEvent("keydown", { key: "f" }))).toBe(false);
    });

    it("stops consuming once the reader unregisters", () => {
        setReaderKeyHandler(() => true);
        setReaderKeyHandler(null);
        expect(handleReaderKey(new KeyboardEvent("keydown", { key: "1" }))).toBe(false);
    });
});
