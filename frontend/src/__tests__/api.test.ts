import { describe, it, expect, vi, beforeEach } from "vitest";
import {
    loadFolder,
    ocrCrop,
    ocrFreeform,
    bubbleDetect,
    splitBubble,
    translate,
    inpaintPreview,
    inpaintGenerate,
    mlLoad,
    mlUnload,
    mlStatus,
    getSession,
    getHealth,
} from "../core/api";
import type { ProcessImageRequest } from "../core/types";
import { PLAIN_GEOMETRY } from "../core/state";

describe("API Client", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    describe("loadFolder", () => {
        it("sends POST request with path", async () => {
            const mockResponse = {
                files: ["page1.jpg", "page2.jpg"],
                dimensions: [
                    [800, 1200],
                    [800, 1200],
                ],
            };
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            } as Response);

            const result = await loadFolder("/path/to/comic");
            expect(result).toEqual(mockResponse);
            expect(fetch).toHaveBeenCalledWith("/api/load_folder", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: "/path/to/comic" }),
            });
        });

        it("throws on HTTP error", async () => {
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: false,
                status: 500,
            } as Response);

            await expect(loadFolder("/bad/path")).rejects.toThrow("HTTP 500");
        });
    });

    describe("ocrCrop", () => {
        it("sends crop request with correct parameters", async () => {
            const mockResponse = { text: "Hello World", detected_lang: "japanese" };
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            } as Response);

            const data = {
                filename: "page1.jpg",
                x: 10,
                y: 20,
                width: 100,
                height: 50,
                lang: "japanese",
                isGrayScale: false,
            };

            const result = await ocrCrop(data);
            expect(result).toEqual(mockResponse);
            expect(fetch).toHaveBeenCalledWith("/api/ocr_crop", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });
        });
    });

    describe("ocrFreeform", () => {
        it("sends freeform request with points", async () => {
            const mockResponse = { text: "自由なテキスト" };
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            } as Response);

            const data = {
                filename: "page1.jpg",
                points: [
                    { x: 10, y: 20 },
                    { x: 30, y: 40 },
                    { x: 20, y: 50 },
                ],
                lang: "japanese",
                isGrayScale: true,
            };

            const result = await ocrFreeform(data);
            expect(result).toEqual(mockResponse);
            expect(fetch).toHaveBeenCalledWith("/api/ocr_freeform", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });
        });
    });

    describe("bubbleDetect", () => {
        it("sends bubble detection request", async () => {
            const mockResponse = [{ type: "rectangle", coords: { x: 10, y: 20, w: 100, h: 50 } }];
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            } as Response);

            const data = { filename: "page1.jpg", isGrayScale: false };
            const result = await bubbleDetect(data);
            expect(result).toEqual(mockResponse);
            expect(fetch).toHaveBeenCalledWith("/api/bubble", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });
        });
    });

    describe("splitBubble", () => {
        it("sends split bubble request", async () => {
            const mockResponse = [
                {
                    type: "polygon",
                    coords: [
                        { x: 10, y: 20 },
                        { x: 30, y: 40 },
                    ],
                },
            ];
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            } as Response);

            const data = {
                region: { type: "rectangle" as const, coords: { x: 10, y: 20, w: 100, h: 50 } },
                split_object: [{ x: 50, y: 20 }],
                isGrayScale: false,
                filename: "page1.jpg",
            };

            const result = await splitBubble(data);
            expect(result).toEqual(mockResponse);
            expect(fetch).toHaveBeenCalledWith("/api/splitbubble", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });
        });
    });

    describe("translate", () => {
        it("sends translation request to specified endpoint", async () => {
            const mockResponse = {
                original: "こんにちは",
                translated: "Hello",
                alt_translated: "Hi",
            };
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            } as Response);

            const result = await translate("/translate/ml", "こんにちは", "japanese");
            expect(result).toEqual(mockResponse);
            expect(fetch).toHaveBeenCalledWith("/translate/ml", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: "こんにちは", source_lang: "japanese" }),
            });
        });
    });

    describe("inpaintPreview", () => {
        it("sends inpaint preview request and returns the blob with its save token", async () => {
            const mockBlob = new Blob(["image data"], { type: "image/png" });
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                blob: async () => mockBlob,
                headers: { get: (name: string) => (name === "X-Preview-Token" ? "tok-123" : null) },
            } as unknown as Response);

            const data: ProcessImageRequest = {
                filename: "page1.jpg",
                data: [
                    {
                        og_text: "OCR",
                        text: "Translated",
                        fontfile: "",
                        text_align: "center",
                        points: [
                            [10, 20],
                            [30, 40],
                        ],
                        layer: 1,
                        font_size: null,
                        font_color: null,
                        stroke_width: null,
                        stroke_color: null,
                        bg_mode: "auto",
                        bg_color: null,
                        clean: null,
                        ...PLAIN_GEOMETRY,
                    },
                ],
            };

            const result = await inpaintPreview(data);
            expect(result.blob).toBe(mockBlob);
            expect(result.token).toBe("tok-123");
            expect(fetch).toHaveBeenCalledWith("/inpaint/preview", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });
        });
    });

    describe("inpaintGenerate", () => {
        it("sends inpaint generate request", async () => {
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
            } as Response);

            const data = { filename: "page1.jpg", data: [] };
            await inpaintGenerate(data);
            expect(fetch).toHaveBeenCalledWith("/inpaint/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });
        });

        it("throws on failure", async () => {
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: false,
                status: 500,
            } as Response);

            await expect(inpaintGenerate({ filename: "page1.jpg", data: [] })).rejects.toThrow("HTTP 500");
        });
    });

    describe("mlLoad", () => {
        it("sends ML load request", async () => {
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ status: true, message: "Model loaded successfully", lang: "japanese" }),
            } as Response);

            await mlLoad("japanese");
            expect(fetch).toHaveBeenCalledWith("/ml/control/load", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lang: "japanese" }),
            });
        });
    });

    describe("mlUnload", () => {
        it("sends ML unload request", async () => {
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ status: true, message: "Model unloaded successfully" }),
            } as Response);

            await mlUnload();
            expect(fetch).toHaveBeenCalledWith("/ml/control/unload", {
                method: "POST",
            });
        });
    });

    describe("mlStatus", () => {
        it("asks for the loaded model without caching", async () => {
            const mockResponse = {
                loaded: true,
                lang: "japanese",
                model_id: "vntl-llama3-8b-v2",
                languages: ["japanese"],
            };
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            } as Response);

            const result = await mlStatus();
            expect(fetch).toHaveBeenCalledWith("/ml/control/status", { cache: "no-store" });
            expect(result).toEqual(mockResponse);
        });

        it("throws on failure", async () => {
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: false,
                status: 500,
            } as Response);

            await expect(mlStatus()).rejects.toThrow("HTTP 500");
        });
    });

    describe("translate", () => {
        it("attaches context for MTL requests that carry pairs", async () => {
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => ({ translated: "hi" }),
            } as Response);

            await translate("/translate/ml", "やあ", "japanese", "japanese", [["a", "A"]]);
            expect(fetch).toHaveBeenCalledWith("/translate/ml", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: "やあ",
                    source_lang: "japanese",
                    lang_group: "japanese",
                    context: [["a", "A"]],
                }),
            });
        });

        it("omits context when there are no pairs", async () => {
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => ({ translated: "hi" }),
            } as Response);

            await translate("/translate/deepl", "hi", "JA", undefined, []);
            expect(fetch).toHaveBeenCalledWith("/translate/deepl", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: "hi", source_lang: "JA" }),
            });
        });
    });

    describe("getSession", () => {
        it("returns the session id", async () => {
            const mockResponse = { session_id: "abc-123" };
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            } as Response);

            const result = await getSession();
            expect(result).toEqual(mockResponse);
            expect(fetch).toHaveBeenCalledWith("/api/session");
        });

        it("throws on HTTP error", async () => {
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: false,
                status: 500,
            } as Response);

            await expect(getSession()).rejects.toThrow("HTTP 500");
        });
    });

    describe("getHealth", () => {
        it("returns the session id from health endpoint", async () => {
            const mockResponse = { session_id: "abc-123" };
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            } as Response);

            const result = await getHealth();
            expect(result).toEqual(mockResponse);
            expect(fetch).toHaveBeenCalledWith("/api/health");
        });

        it("throws on HTTP error", async () => {
            vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
                ok: false,
                status: 500,
            } as Response);

            await expect(getHealth()).rejects.toThrow("HTTP 500");
        });
    });
});
