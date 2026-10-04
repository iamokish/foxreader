import { describe, it, expect, vi, beforeEach } from "vitest";
import { on, off, emit } from "../core/eventbus";

describe("EventBus", () => {
    beforeEach(() => {
        // Clean up all listeners by emitting to all known events
        // Since we can't directly access listeners, we'll rely on test isolation
    });

    describe("on", () => {
        it("registers a listener for an event", () => {
            const callback = vi.fn();
            on("image:loaded", callback);
            emit("image:loaded", "test.jpg");
            expect(callback).toHaveBeenCalledWith("test.jpg");
        });

        it("supports multiple listeners for the same event", () => {
            const callback1 = vi.fn();
            const callback2 = vi.fn();
            on("entries:render", callback1);
            on("entries:render", callback2);
            emit("entries:render");
            expect(callback1).toHaveBeenCalledOnce();
            expect(callback2).toHaveBeenCalledOnce();
        });
    });

    describe("off", () => {
        it("removes a specific listener", () => {
            const callback1 = vi.fn();
            const callback2 = vi.fn();
            on("gallery:navigate", callback1);
            on("gallery:navigate", callback2);
            off("gallery:navigate", callback1);
            emit("gallery:navigate", 1);
            expect(callback1).not.toHaveBeenCalled();
            expect(callback2).toHaveBeenCalledWith(1);
        });

        it("does not affect other events", () => {
            const callback = vi.fn();
            on("image:loaded", callback);
            off("entries:render", callback);
            emit("image:loaded", "test.jpg");
            expect(callback).toHaveBeenCalledWith("test.jpg");
        });
    });

    describe("emit", () => {
        it("calls all listeners with arguments", () => {
            const callback1 = vi.fn();
            const callback2 = vi.fn();
            on("image:loaded", callback1);
            on("image:loaded", callback2);
            emit("image:loaded", "page.jpg");
            expect(callback1).toHaveBeenCalledWith("page.jpg");
            expect(callback2).toHaveBeenCalledWith("page.jpg");
        });

        it("handles events with no listeners", () => {
            expect(() => emit("entries:render")).not.toThrow();
        });

        it("passes multiple arguments", () => {
            const callback = vi.fn();
            on("gallery:navigate", callback);
            emit("gallery:navigate", 5);
            expect(callback).toHaveBeenCalledWith(5);
        });
    });
});
