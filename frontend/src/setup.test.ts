// @vitest-environment node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { JSDOM, VirtualConsole } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

const template = readFileSync(resolve("templates/setup.html"), "utf8")
    .replace(/\{\{\s*mtl_dir\s*\}\}/g, "models")
    .replace(/\{\{\s*mtl_(?:default|resolved)_dir\s*\}\}/g, "/app/models");
const models = [
    { id: "ppocrv6-det", name: "PaddleOCRv6 Detection" },
    { id: "ppocrv6-rec", name: "PaddleOCRv6 Recognition" },
    { id: "ppocrv5-rec-korean", name: "PaddleOCRv5 Recognition [Korean]" },
];
const localModel = { id: "local-mtl", name: "Local translation" };
const optionalModel = { id: "optional-model", name: "Optional model" };
type ModelGroup = "required" | "optional" | "mtl";
interface DownloadProgress {
    phase: "preparing" | "downloading" | "finalizing" | "complete";
    current_file: string;
    current_bytes: number;
    current_total: number;
    bytes_done: number;
    bytes_total: number;
    files_done: number;
    files_total: number;
    transfer_bytes?: number;
    transfer_total?: number;
    done?: boolean;
    downloaded?: boolean;
}
const MiB = 1024 ** 2;
const GiB = 1024 ** 3;
const downloadProgress = (overrides: Partial<DownloadProgress> = {}): DownloadProgress => ({
    phase: "downloading",
    current_file: "model.safetensors",
    current_bytes: 10 * MiB,
    current_total: 100 * MiB,
    bytes_done: 10 * MiB,
    bytes_total: 0,
    files_done: 0,
    files_total: 1,
    ...overrides,
});
interface StorageReport {
    ok: boolean;
    path: string;
    error: string;
    exists: boolean;
    can_create: boolean;
    free_bytes?: number;
}
const pages: JSDOM[] = [];

afterEach(() => {
    for (const page of pages.splice(0)) page.window.close();
    vi.useRealTimers();
});

function setupPage(workerError?: string, selection = false, defaultDownloaded = false) {
    const accepted = new Set<string>();
    const requests: { path: string; method: string; body: unknown }[] = [];
    const errors: Error[] = [];
    const stalled = new Map<string, "request" | "body">();
    const held = new Map<string, Promise<void>>();
    const rejected = new Map<string, string>();
    const lostReplies = new Map<string, "before" | "after">();
    const malformedReplies = new Map<string, unknown>();
    const storageReports = new Map<string, Partial<StorageReport>>();
    const aborted: string[] = [];
    const progressSnapshots = new Map<string, DownloadProgress>();
    const completedModels = new Set<string>();
    let started = false;
    let optionalStarted = false;
    let appliedStoragePath = "/app/models";
    const allModels = [...models, localModel, optionalModel];
    const terms = () => Object.fromEntries(allModels.map(({ id }) => [id, { accepted: accepted.has(id) }]));
    const mtlModels = () => [
        {
            ...localModel,
            downloaded:
                completedModels.has(localModel.id) || (defaultDownloaded && appliedStoragePath === "/app/models"),
            terms_accepted: accepted.has(localModel.id),
        },
    ];
    const inspectStorage = (value: string): StorageReport => ({
        ok: true,
        path: value === "models" ? "/app/models" : value.trim(),
        error: "",
        exists: true,
        can_create: false,
        free_bytes: 50 * 1024 ** 3,
        ...storageReports.get(value),
    });
    const fetchMock = vi.fn(async (url: string, options: RequestInit = {}) => {
        const path = new URL(url, "http://localhost");
        const method = options.method || "GET";
        const body = options.body ? JSON.parse(String(options.body)) : undefined;
        requests.push({ path: path.pathname, method, body });
        const stall = () =>
            new Promise<never>((_, reject) => {
                const abort = () => {
                    aborted.push(path.pathname);
                    reject(new globalThis.DOMException("Aborted", "AbortError"));
                };
                if (options.signal?.aborted) abort();
                else options.signal?.addEventListener("abort", abort, { once: true });
            });
        if (stalled.get(path.pathname) === "request") return stall();
        if (lostReplies.get(path.pathname) === "before") throw new Error("Response lost");
        let payload: unknown;
        let ok = true;
        if (rejected.has(path.pathname)) {
            ok = false;
            payload = { error: rejected.get(path.pathname) };
        } else if (path.pathname === "/api/setup/status") {
            const currentId = selection ? localModel.id : models[0].id;
            const progress = Object.fromEntries([...progressSnapshots].filter(([id]) => id !== optionalModel.id));
            if (started && !workerError && !progress[currentId]) {
                progress[currentId] = {
                    phase: "downloading",
                    current_bytes: 1024,
                    current_total: 4096,
                    bytes_done: 1024,
                    bytes_total: 0,
                    current_file: "model.safetensors",
                    files_done: 0,
                    files_total: 1,
                };
            }
            payload = {
                stage: selection ? (started && !workerError ? "mtl" : "mtl_selection") : "non_mtl",
                active: started && !workerError,
                mtl_dir: appliedStoragePath,
                error: started ? workerError || null : null,
                current_model: started && !workerError ? currentId : null,
                progress,
                required_models: [
                    {
                        id: "paddleocr",
                        name: "Text Detection & Recognition",
                        items: models.map((model) => ({
                            ...model,
                            downloaded: selection || completedModels.has(model.id),
                            terms_accepted: accepted.has(model.id),
                        })),
                    },
                ],
                mtl_models: mtlModels(),
                all_mtl_downloaded: mtlModels().length > 0 && mtlModels().every((model) => model.downloaded),
                terms: terms(),
                optional: {
                    active: optionalStarted,
                    current_model: optionalStarted ? optionalModel.id : null,
                    models: [
                        {
                            ...optionalModel,
                            downloaded: completedModels.has(optionalModel.id),
                            terms_accepted: accepted.has(optionalModel.id),
                        },
                    ],
                    progress: progressSnapshots.has(optionalModel.id)
                        ? { [optionalModel.id]: progressSnapshots.get(optionalModel.id) }
                        : {},
                    error: null,
                },
            };
        } else if (path.pathname === "/api/setup/terms") {
            const id = path.searchParams.get("id");
            payload = id
                ? {
                      document: {
                          ...allModels.find((model) => model.id === id),
                          licence: "Apache 2.0",
                          sections: [{ paragraphs: ["Read and accept these model terms before downloading."] }],
                      },
                      accepted: accepted.has(id),
                  }
                : { terms: terms() };
        } else if (path.pathname === "/api/setup/terms/accept" && method === "POST") {
            for (const id of body.ids) accepted.add(id);
            payload = { accepted: body.ids, terms: terms() };
        } else if (path.pathname === "/api/setup/download" && method === "POST") {
            if (selection) {
                expect(body.selected_mtl).toEqual([localModel.id]);
                expect(accepted.has(localModel.id)).toBe(true);
            } else expect(accepted.size).toBe(models.length);
            started = true;
            payload = { status: "started", stage: selection ? "mtl" : "non_mtl" };
        } else if (path.pathname === "/api/setup/optional/download" && method === "POST") {
            expect(body.ids).toEqual([optionalModel.id]);
            expect(accepted.has(optionalModel.id)).toBe(true);
            optionalStarted = true;
            payload = { ok: true };
        } else if (path.pathname === "/api/setup/storage/inspect" && method === "POST") {
            payload = inspectStorage(body.path);
        } else if (path.pathname === "/api/setup/storage" && method === "POST") {
            const report = inspectStorage(body.path);
            ok = report.ok;
            if (ok) {
                appliedStoragePath = report.path;
                payload = {
                    path: report.path,
                    mtl_dir: body.path === "models" ? "models" : report.path,
                    mtl_models: mtlModels(),
                    all_mtl_downloaded: mtlModels().length > 0 && mtlModels().every((model) => model.downloaded),
                };
            } else payload = report;
        } else if (path.pathname === "/api/setup/close" && method === "POST") {
            payload = { status: "closing" };
        } else {
            throw new Error(`Unexpected setup request: ${method} ${url}`);
        }
        if (lostReplies.get(path.pathname) === "after") throw new Error("Response lost");
        return {
            ok,
            json: async () => {
                await (held.get(`${path.pathname}:${body?.path}`) ?? held.get(path.pathname));
                if (stalled.get(path.pathname) === "body") return stall();
                return malformedReplies.has(path.pathname) ? malformedReplies.get(path.pathname) : payload;
            },
        };
    });
    const virtualConsole = new VirtualConsole();
    virtualConsole.on("jsdomError", (error) => errors.push(error));
    const page = new JSDOM(template, {
        url: "http://localhost/setup",
        runScripts: "dangerously",
        pretendToBeVisual: true,
        virtualConsole,
        beforeParse(window) {
            Object.assign(window, {
                fetch: fetchMock,
                WebSocket: class {
                    close() {}
                },
                BroadcastChannel: class {
                    postMessage() {}
                    close() {}
                },
            });
        },
    });
    pages.push(page);
    const document = page.window.document;
    const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
    const button = (id: string) => element<HTMLButtonElement>(id);
    const downloads = () => requests.filter((request) => request.path === "/api/setup/download");
    const selectTab = (name: "required" | "optional" | "mtl") => button(`${name}Tab`).click();
    const ready = async (tab: "required" | "optional" | "mtl" = selection ? "mtl" : "required") => {
        await vi.waitFor(() => expect(element("requiredModels").textContent).toContain(models[0].name));
        expect(errors).toEqual([]);
        expect(document.documentElement.classList.contains("fox-gate-open")).toBe(true);
        button("copyrightAckBtn").click();
        expect(document.documentElement.classList.contains("fox-gate-open")).toBe(false);
        if (tab !== "required") selectTab(tab);
    };
    const acceptAll = async () => {
        for (const [index, model] of models.entries()) {
            await vi.waitFor(() => expect(element("termsModelName").textContent).toBe(model.name));
            expect(document.documentElement.classList.contains("fox-terms-open")).toBe(true);
            expect(element("termsStep").textContent).toBe(`Licence ${index + 1} of ${models.length}`);
            expect(button("termsAcceptBtn").disabled).toBe(true);
            expect(downloads()).toHaveLength(0);
            element<HTMLInputElement>("termsAcceptChk").click();
            expect(button("termsAcceptBtn").disabled).toBe(false);
            button("termsAcceptBtn").click();
        }
    };
    const holdResponse = (path: string, value?: string) => {
        let release!: () => void;
        const key = value === undefined ? path : `${path}:${value}`;
        held.set(
            key,
            new Promise<void>((resume) => {
                release = resume;
            }),
        );
        return () => {
            held.delete(key);
            release();
        };
    };
    const poll = () => (page.window as unknown as { pollOnce: () => Promise<void> }).pollOnce();
    const editStorage = (value: string) => {
        element<HTMLInputElement>("storageCustom").click();
        element<HTMLInputElement>("mtlDir").value = value;
        element("mtlDir").dispatchEvent(new page.window.Event("input", { bubbles: true }));
    };
    return {
        document,
        element,
        button,
        downloads,
        ready,
        selectTab,
        isShown: (id: string) => !element(id).closest("[hidden], .hidden"),
        acceptAll,
        requests,
        accepted,
        errors,
        stalled,
        aborted,
        rejected,
        lostReplies,
        malformedReplies,
        storageReports,
        holdResponse,
        poll,
        editStorage,
        setProgress: (group: ModelGroup, snapshot: DownloadProgress) => {
            const id = group === "required" ? models[0].id : group === "mtl" ? localModel.id : optionalModel.id;
            progressSnapshots.set(id, snapshot);
            if (group === "optional") optionalStarted = true;
            else started = true;
        },
        completeProgress: (group: ModelGroup) => {
            const id = group === "required" ? models[0].id : group === "mtl" ? localModel.id : optionalModel.id;
            const previous = progressSnapshots.get(id)!;
            progressSnapshots.set(id, {
                ...previous,
                phase: "complete",
                done: true,
                downloaded: true,
                files_done: previous.files_total,
            });
            completedModels.add(id);
            if (group === "optional") optionalStarted = false;
            else started = false;
        },
        completeRequired: () => {
            selection = true;
            started = false;
        },
        pressTabKey: (name: "required" | "optional" | "mtl", key: string) => {
            return button(`${name}Tab`).dispatchEvent(
                new page.window.KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
            );
        },
        commitStorage: (path: string) => {
            appliedStoragePath = path;
        },
        pageTransition: (type: "pagehide" | "pageshow", persisted = true) => {
            page.window.dispatchEvent(new page.window.PageTransitionEvent(type, { persisted }));
        },
    };
}

describe("first-run setup download", () => {
    it("walks every licence from the actual button before starting one download and showing progress", async () => {
        const page = setupPage();
        await page.ready();
        page.button("downloadBtn").click();
        await page.acceptAll();
        await vi.waitFor(() => expect(page.element("requiredModels").textContent).toContain("25%"));
        expect(page.downloads()).toEqual([{ path: "/api/setup/download", method: "POST", body: {} }]);
        expect(
            page.requests
                .filter((request) => request.path === "/api/setup/terms/accept")
                .map((request) => request.body),
        ).toEqual(models.map(({ id }) => ({ ids: [id] })));
        expect(page.button("downloadBtn").disabled).toBe(true);
        expect(page.button("downloadBtn").textContent).toContain("Downloading Required Models");
        expect(page.document.documentElement.classList.contains("fox-terms-open")).toBe(false);
        expect(page.errors).toEqual([]);
    });

    it("cancels without downloading when a licence is declined", async () => {
        const page = setupPage();
        await page.ready();
        page.button("downloadBtn").click();
        await vi.waitFor(() => expect(page.element("termsModelName").textContent).toBe(models[0].name));
        page.button("termsDeclineBtn").click();
        await vi.waitFor(() => expect(page.button("downloadBtn").disabled).toBe(false));
        expect(page.downloads()).toHaveLength(0);
        expect(page.accepted.size).toBe(0);
        expect(page.document.documentElement.classList.contains("fox-terms-open")).toBe(false);
        expect(page.errors).toEqual([]);
    });

    it("shows a background download failure and leaves the button ready to retry", async () => {
        const message = "No module named 'filelock._soft'";
        const page = setupPage(message);
        await page.ready();
        page.button("downloadBtn").click();
        await page.acceptAll();
        await vi.waitFor(() => expect(page.element("errorMsg").textContent).toBe(message));
        expect(page.element("errorMsg").style.display).not.toBe("none");
        expect(page.button("downloadBtn").disabled).toBe(false);
        expect(page.button("downloadBtn").textContent).toBe("Download Required Models");
        expect(page.downloads()).toHaveLength(1);
        expect(page.errors).toEqual([]);
    });

    it("times out a stalled licence request without downloading and allows retry", async () => {
        vi.useFakeTimers();
        const page = setupPage();
        await page.ready();
        page.stalled.set("/api/setup/terms", "request");
        page.button("downloadBtn").click();
        await vi.advanceTimersByTimeAsync(15000);
        expect(page.element("errorMsg").textContent).toContain("took too long to respond");
        expect(page.button("downloadBtn").disabled).toBe(false);
        expect(page.downloads()).toHaveLength(0);
        expect(page.accepted.size).toBe(0);
        expect(page.aborted).toEqual(["/api/setup/terms"]);

        page.stalled.clear();
        page.button("downloadBtn").click();
        await page.acceptAll();
        await vi.waitFor(() => expect(page.element("requiredModels").textContent).toContain("25%"));
        expect(page.element("errorMsg").style.display).toBe("none");
        expect(page.downloads()).toHaveLength(1);
        expect(page.errors).toEqual([]);
    });

    it("checks status after a stalled download response body without submitting another download", async () => {
        vi.useFakeTimers();
        const page = setupPage();
        await page.ready();
        page.stalled.set("/api/setup/download", "body");
        page.button("downloadBtn").click();
        await page.acceptAll();
        await vi.waitFor(() => expect(page.downloads()).toHaveLength(1));
        await vi.advanceTimersByTimeAsync(15000);
        expect(page.aborted).toEqual(["/api/setup/download"]);
        expect(page.element("errorMsg").textContent).toContain("took too long to respond");
        expect(page.element("requiredModels").textContent).toContain("25%");
        expect(page.button("downloadBtn").disabled).toBe(true);
        expect(page.button("downloadBtn").textContent).toContain("Downloading Required Models");
        expect(page.downloads()).toHaveLength(1);
        expect(page.errors).toEqual([]);
    });

    it("shows failed status checks and resumes progress when the backend responds again", async () => {
        vi.useFakeTimers();
        const page = setupPage();
        await page.ready();
        page.button("downloadBtn").click();
        await page.acceptAll();
        await vi.waitFor(() => expect(page.element("requiredModels").textContent).toContain("25%"));
        page.stalled.set("/api/setup/status", "request");
        await vi.advanceTimersByTimeAsync(17000);
        expect(page.element("connectionMsg").textContent).toContain("Retrying automatically");
        expect(page.element("connectionMsg").style.display).toBe("block");
        expect(page.button("downloadBtn").disabled).toBe(true);
        expect(page.element("requiredModels").textContent).toContain("25%");
        expect(page.aborted).toEqual(["/api/setup/status"]);

        page.stalled.clear();
        await vi.advanceTimersByTimeAsync(2000);
        expect(page.element("connectionMsg").style.display).toBe("none");
        expect(page.button("downloadBtn").textContent).toContain("Downloading Required Models");
        expect(page.downloads()).toHaveLength(1);
        expect(page.errors).toEqual([]);
    });

    it("refreshes again when a download starts during an older status request", async () => {
        const page = setupPage();
        await page.ready();
        const release = page.holdResponse("/api/setup/status");
        const pending = page.poll();
        page.button("downloadBtn").click();
        await page.acceptAll();
        await vi.waitFor(() => expect(page.downloads()).toHaveLength(1));
        release();
        await pending;
        await vi.waitFor(() => expect(page.element("requiredModels").textContent).toContain("25%"));
        expect(page.requests.filter((request) => request.path === "/api/setup/status")).toHaveLength(3);
        expect(page.button("downloadBtn").disabled).toBe(true);
        expect(page.downloads()).toHaveLength(1);
        expect(page.errors).toEqual([]);
    });

    it("keeps download controls disabled during connection failure even when selections change", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready();
        page.stalled.set("/api/setup/status", "request");
        const pending = page.poll();
        await vi.advanceTimersByTimeAsync(15000);
        await pending;
        page.element<HTMLInputElement>("mtl-local-mtl").click();
        page.selectTab("optional");
        page.element<HTMLInputElement>("opt-optional-model").click();
        page.element<HTMLInputElement>("opt-optional-model").click();
        page.selectTab("mtl");
        expect(page.button("downloadBtn").disabled).toBe(true);
        expect(page.button("optionalBtn").disabled).toBe(true);
        expect(page.button("skipMtlBtn").disabled).toBe(true);
        expect(page.downloads()).toHaveLength(0);

        page.stalled.clear();
        await vi.advanceTimersByTimeAsync(2000);
        expect(page.button("downloadBtn").disabled).toBe(false);
        expect(page.button("optionalBtn").disabled).toBe(false);
        expect(page.button("skipMtlBtn").disabled).toBe(false);
        expect(page.element("connectionMsg").style.display).toBe("none");
        expect(page.errors).toEqual([]);
    });

    it("restores setup when the close request times out", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready();
        page.stalled.set("/api/setup/close", "request");
        page.button("skipMtlBtn").click();
        await vi.advanceTimersByTimeAsync(15000);
        expect(page.element("errorMsg").textContent).toContain("took too long to respond");
        expect(page.element("doneMsg").style.display).toBe("none");
        expect(page.element("closedMsg").style.display).toBe("none");
        expect(page.button("skipMtlBtn").disabled).toBe(false);
        expect(page.aborted).toEqual(["/api/setup/close"]);
        expect(page.errors).toEqual([]);
    });

    it("does not claim the backend closed when its liveness check times out", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready();
        page.button("skipMtlBtn").click();
        await vi.waitFor(() => expect(page.requests.some((request) => request.path === "/api/setup/close")).toBe(true));
        page.stalled.set("/api/setup/status", "request");
        await vi.advanceTimersByTimeAsync(15250);
        expect(page.aborted).toEqual(["/api/setup/status"]);
        expect(page.element("closedMsg").style.display).toBe("none");
        expect(page.element("doneMsg").style.display).toBe("block");
        expect(page.errors).toEqual([]);
    });
});

describe("download progress", () => {
    it.each(["required", "mtl", "optional"] as const)(
        "updates %s file progress without replacing controls or losing focus",
        async (group) => {
            vi.useFakeTimers();
            const page = setupPage(undefined, group !== "required");
            await page.ready(group);
            page.setProgress(group, downloadProgress());
            await page.poll();
            const container = page.element(`${group}Models`);
            const card = container.querySelector<HTMLElement>(".model")!;
            const checkbox = card.querySelector<HTMLInputElement>("input")!;
            const licence = card.querySelector<HTMLButtonElement>(".terms-peek")!;
            licence.focus();
            expect(card.querySelector('[role="progressbar"]')?.getAttribute("aria-valuenow")).toBe("10");
            expect(card.textContent).toContain("10%");
            expect(card.textContent).toMatch(/10(?:\.0)? MiB/);

            page.setProgress(group, downloadProgress({ current_bytes: 40 * MiB, bytes_done: 40 * MiB }));
            await vi.advanceTimersByTimeAsync(2000);
            expect(container.querySelector(".model")).toBe(card);
            expect(card.querySelector("input")).toBe(checkbox);
            expect(page.document.activeElement).toBe(licence);
            expect(card.querySelector('[role="progressbar"]')?.getAttribute("aria-valuenow")).toBe("40");
            expect(card.textContent).toContain("40%");
            expect(page.button(`${group}Tab`).getAttribute("aria-selected")).toBe("true");

            page.setProgress(
                group,
                downloadProgress({ phase: "finalizing", current_bytes: 100 * MiB, bytes_done: 100 * MiB }),
            );
            await vi.advanceTimersByTimeAsync(2000);
            expect(container.textContent).toContain("Finalizing");
            expect(container.querySelector(".model .badge.downloaded")).toBeNull();
            expect(
                group === "optional" ? page.button("optionalBtn").disabled : page.button("downloadBtn").disabled,
            ).toBe(true);
            page.completeProgress(group);
            await vi.advanceTimersByTimeAsync(2000);
            expect(container.querySelector(".model .badge.downloaded")?.textContent).toBe("Downloaded");
            expect(container.querySelector(".model")?.classList.contains("downloading")).toBe(false);
            expect(page.downloads()).toHaveLength(0);
            expect(page.errors).toEqual([]);
        },
    );

    it("uses one status request every two seconds and waits for a slow response before scheduling another", async () => {
        vi.useFakeTimers();
        const page = setupPage();
        await page.ready();
        page.setProgress("required", downloadProgress());
        await page.poll();
        const statusCalls = () => page.requests.filter((request) => request.path === "/api/setup/status").length;
        const before = statusCalls();
        await vi.advanceTimersByTimeAsync(1999);
        expect(statusCalls()).toBe(before);
        await vi.advanceTimersByTimeAsync(1);
        expect(statusCalls()).toBe(before + 1);

        const release = page.holdResponse("/api/setup/status");
        page.setProgress("required", downloadProgress({ current_bytes: 40 * MiB, bytes_done: 40 * MiB }));
        await vi.advanceTimersByTimeAsync(2000);
        expect(statusCalls()).toBe(before + 2);
        await vi.advanceTimersByTimeAsync(6000);
        expect(statusCalls()).toBe(before + 2);
        expect(page.element("requiredModels").textContent).toContain("10%");
        release();
        await vi.advanceTimersByTimeAsync(0);
        expect(page.element("requiredModels").textContent).toContain("40%");
        await vi.advanceTimersByTimeAsync(1999);
        expect(statusCalls()).toBe(before + 2);
        await vi.advanceTimersByTimeAsync(1);
        expect(statusCalls()).toBe(before + 3);
        expect(page.errors).toEqual([]);
    });

    it("shows received bytes without inventing a percentage when the file total is unknown", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready("optional");
        page.setProgress(
            "optional",
            downloadProgress({
                current_total: 0,
                bytes_total: 0,
                current_bytes: 7 * MiB,
                bytes_done: 7 * MiB,
                files_total: 2,
            }),
        );
        await page.poll();
        const container = page.element("optionalModels");
        const bar = container.querySelector('[role="progressbar"]');
        expect(bar).not.toBeNull();
        expect(bar?.hasAttribute("aria-valuenow")).toBe(false);
        expect(container.textContent).toMatch(/7(?:\.0)? MiB/);
        expect(container.textContent).not.toMatch(/\d+%/);
        expect(container.querySelector(".badge.downloaded")).toBeNull();
        expect(page.errors).toEqual([]);
    });

    it("shows native transfer progress while reconstructed file bytes remain zero", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready("optional");
        const native = downloadProgress({
            current_bytes: 0,
            current_total: 120 * MiB,
            bytes_done: 0,
            transfer_bytes: 10 * MiB,
            transfer_total: 100 * MiB,
        });
        page.setProgress("optional", native);
        await page.poll();
        const container = page.element("optionalModels");
        const card = container.querySelector(".model")!;
        const bar = card.querySelector('[role="progressbar"]')!;
        expect(bar.getAttribute("aria-valuenow")).toBe("10");
        expect(card.textContent).toContain("Transferred");
        expect(card.textContent).toMatch(/10(?:\.0)? MiB/);
        expect(card.textContent).not.toMatch(/120(?:\.0)? MiB/);

        page.setProgress("optional", { ...native, transfer_bytes: 40 * MiB });
        await vi.advanceTimersByTimeAsync(2000);
        expect(container.querySelector(".model")).toBe(card);
        expect(card.querySelector('[role="progressbar"]')).toBe(bar);
        expect(bar.getAttribute("aria-valuenow")).toBe("40");
        expect(card.querySelector(".badge.downloaded")).toBeNull();

        page.setProgress("optional", { ...native, phase: "finalizing", transfer_bytes: 100 * MiB });
        await vi.advanceTimersByTimeAsync(2000);
        expect(bar.getAttribute("aria-valuenow")).toBe("100");
        expect(card.textContent).toContain("Finalizing");
        expect(card.querySelector(".badge.downloaded")).toBeNull();
        expect(page.button("optionalBtn").disabled).toBe(true);
        page.completeProgress("optional");
        await vi.advanceTimersByTimeAsync(2000);
        expect(container.querySelector(".badge.downloaded")?.textContent).toBe("Downloaded");
        expect(page.errors).toEqual([]);
    });

    it("does not substitute the reconstructed file size for an unknown native transfer total", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready("mtl");
        page.setProgress(
            "mtl",
            downloadProgress({
                current_bytes: 25 * MiB,
                current_total: 100 * MiB,
                transfer_bytes: 7 * MiB,
                transfer_total: 0,
            }),
        );
        await page.poll();
        const container = page.element("mtlModels");
        expect(container.querySelector('[role="progressbar"]')?.hasAttribute("aria-valuenow")).toBe(false);
        expect(container.textContent).toContain("Transferred");
        expect(container.textContent).toMatch(/7(?:\.0)? MiB/);
        expect(container.textContent).not.toMatch(/\d+%/);
        expect(container.querySelector(".badge.downloaded")).toBeNull();
        expect(page.errors).toEqual([]);
    });

    it("resets native transfer counters for a new file and falls back when the next file has no native metrics", async () => {
        vi.useFakeTimers();
        const page = setupPage();
        await page.ready();
        page.setProgress(
            "required",
            downloadProgress({
                current_file: "first.bin",
                current_bytes: 0,
                transfer_bytes: 50 * MiB,
                transfer_total: 100 * MiB,
                files_total: 3,
            }),
        );
        await page.poll();
        const container = page.element("requiredModels");
        const bar = container.querySelector('[role="progressbar"]')!;
        expect(bar.getAttribute("aria-valuenow")).toBe("50");

        page.setProgress(
            "required",
            downloadProgress({
                current_file: "second.bin",
                current_bytes: 0,
                transfer_bytes: 0,
                transfer_total: 200 * MiB,
                files_done: 1,
                files_total: 3,
            }),
        );
        await vi.advanceTimersByTimeAsync(2000);
        expect(container.querySelector('[role="progressbar"]')).toBe(bar);
        expect(bar.getAttribute("aria-valuenow")).toBe("0");
        expect(container.textContent).toContain("second.bin");
        expect(container.textContent).not.toContain("first.bin");
        expect(container.textContent).toContain("Transferred");

        page.setProgress(
            "required",
            downloadProgress({
                current_file: "third.bin",
                current_bytes: 4 * MiB,
                current_total: 20 * MiB,
                files_done: 2,
                files_total: 3,
            }),
        );
        await vi.advanceTimersByTimeAsync(2000);
        expect(bar.getAttribute("aria-valuenow")).toBe("20");
        expect(container.textContent).toContain("third.bin");
        expect(container.textContent).not.toContain("Transferred");
        expect(container.textContent).toMatch(/4(?:\.0)? MiB/);
        expect(page.errors).toEqual([]);
    });

    it("does not round a nearly received file up to 100 percent before its bytes arrive", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready("optional");
        page.setProgress("optional", downloadProgress({ current_bytes: 9996, current_total: 10000, bytes_done: 9996 }));
        await page.poll();
        const container = page.element("optionalModels");
        const received = Number(container.querySelector('[role="progressbar"]')?.getAttribute("aria-valuenow"));
        expect(received).toBeGreaterThanOrEqual(99);
        expect(received).toBeLessThan(100);
        expect(container.textContent).not.toContain("100%");
        expect(container.querySelector(".badge.downloaded")).toBeNull();
        expect(page.errors).toEqual([]);
    });

    it("separates preparation and file counts from bytes and resets percentage for a larger next file", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready("mtl");
        page.setProgress(
            "mtl",
            downloadProgress({
                phase: "preparing",
                current_file: "",
                current_bytes: 0,
                current_total: 0,
                bytes_done: 0,
                bytes_total: 0,
                files_total: 2,
            }),
        );
        await page.poll();
        const container = page.element("mtlModels");
        expect(container.textContent).toContain("Preparing");
        expect(container.textContent).not.toMatch(/\d+%/);

        page.setProgress(
            "mtl",
            downloadProgress({
                current_file: "config.json",
                current_bytes: 64 * 1024,
                current_total: 64 * 1024,
                bytes_done: 64 * 1024,
                bytes_total: 64 * 1024,
                files_total: 2,
            }),
        );
        await vi.advanceTimersByTimeAsync(2000);
        expect(container.textContent).toContain("config.json");
        expect(container.querySelector(".badge.downloaded")).toBeNull();

        page.setProgress(
            "mtl",
            downloadProgress({
                current_file: "weights.gguf",
                current_bytes: 0,
                current_total: 2 * GiB,
                bytes_done: 64 * 1024,
                files_done: 1,
                files_total: 2,
            }),
        );
        await vi.advanceTimersByTimeAsync(2000);
        expect(container.textContent).toContain("weights.gguf");
        expect(container.textContent).not.toContain("config.json");
        expect(container.querySelector('[role="progressbar"]')?.getAttribute("aria-valuenow")).toBe("0");
        expect(container.textContent).toMatch(/2(?:\.0+)? GiB/);
        expect(container.textContent).toMatch(/(?:file\s*2\s*(?:\/|of)\s*2|1\s*(?:\/|of)\s*2\s*files)/i);

        page.setProgress(
            "mtl",
            downloadProgress({
                current_file: "weights.gguf",
                current_bytes: GiB / 2,
                current_total: 2 * GiB,
                bytes_done: GiB / 2 + 64 * 1024,
                files_done: 1,
                files_total: 2,
            }),
        );
        await vi.advanceTimersByTimeAsync(2000);
        expect(container.querySelector('[role="progressbar"]')?.getAttribute("aria-valuenow")).toBe("25");
        expect(container.querySelector(".badge.downloaded")).toBeNull();
        expect(page.errors).toEqual([]);
    });

    it("preserves idle selections and storage edits while another tab's download progresses", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready("mtl");
        page.element<HTMLInputElement>("mtl-local-mtl").click();
        page.setProgress("optional", downloadProgress());
        await page.poll();
        const checkbox = page.element<HTMLInputElement>("mtl-local-mtl");
        checkbox.focus();
        page.setProgress("optional", downloadProgress({ current_bytes: 40 * MiB, bytes_done: 40 * MiB }));
        await vi.advanceTimersByTimeAsync(2000);
        expect(page.element("mtl-local-mtl")).toBe(checkbox);
        expect(page.document.activeElement).toBe(checkbox);
        expect(checkbox.checked).toBe(true);
        page.editStorage("/draft during download");
        await vi.advanceTimersByTimeAsync(400);
        const input = page.element<HTMLInputElement>("mtlDir");
        input.focus();
        input.setSelectionRange(2, 7);
        await vi.advanceTimersByTimeAsync(2000);
        expect(page.document.activeElement).toBe(input);
        expect(input.value).toBe("/draft during download");
        expect([input.selectionStart, input.selectionEnd]).toEqual([2, 7]);
        expect(page.button("mtlTab").getAttribute("aria-selected")).toBe("true");
        expect(page.errors).toEqual([]);
    });
});

describe("setup tabs", () => {
    it("starts on Required and lets users inspect the other tabs without bypassing prerequisites", async () => {
        const page = setupPage();
        await page.ready();
        expect(page.element("setupTabs").getAttribute("role")).toBe("tablist");
        for (const name of ["required", "optional", "mtl"]) {
            const tab = page.button(`${name}Tab`);
            const panel = page.element(`${name}Panel`);
            expect(tab.getAttribute("role")).toBe("tab");
            expect(tab.getAttribute("aria-controls")).toBe(panel.id);
            expect(panel.getAttribute("role")).toBe("tabpanel");
            expect(panel.getAttribute("aria-labelledby")).toBe(tab.id);
            expect(tab.getAttribute("aria-selected")).toBe(String(name === "required"));
            expect(tab.tabIndex).toBe(name === "required" ? 0 : -1);
            expect(panel.hidden).toBe(name !== "required");
            expect(tab.disabled).toBe(false);
        }
        expect(page.isShown("primaryAction")).toBe(true);
        expect(page.isShown("mtlStorage")).toBe(false);

        page.selectTab("optional");
        expect(page.isShown("optionalPanel")).toBe(true);
        expect(page.isShown("primaryAction")).toBe(false);
        expect(page.isShown("tabPrerequisite")).toBe(true);
        expect(page.element<HTMLInputElement>("opt-optional-model").disabled).toBe(true);
        expect(page.button("optionalBtn").disabled).toBe(true);

        page.selectTab("mtl");
        expect(page.isShown("mtlStorage")).toBe(true);
        expect(page.element<HTMLInputElement>("mtlDir").disabled).toBe(false);
        expect(page.element<HTMLInputElement>("mtl-local-mtl").disabled).toBe(true);
        expect(page.isShown("primaryAction")).toBe(false);
        expect(page.isShown("tabPrerequisite")).toBe(true);
        expect(page.downloads()).toHaveLength(0);
        expect(page.errors).toEqual([]);
    });

    it("activates and focuses tabs with arrows, Home, and End", async () => {
        const page = setupPage();
        await page.ready();
        const steps = [
            ["required", "ArrowRight", "optional"],
            ["optional", "ArrowRight", "mtl"],
            ["mtl", "ArrowRight", "required"],
            ["required", "ArrowLeft", "mtl"],
            ["mtl", "Home", "required"],
            ["required", "End", "mtl"],
        ] as const;
        for (const [from, key, to] of steps) {
            expect(page.pressTabKey(from, key)).toBe(false);
            expect(page.document.activeElement).toBe(page.button(`${to}Tab`));
            expect(page.button(`${to}Tab`).getAttribute("aria-selected")).toBe("true");
            expect(page.element(`${to}Panel`).hidden).toBe(false);
        }
        expect(page.errors).toEqual([]);
    });

    it("keeps the selected tab when polls unlock later stages and scopes primary actions to their tab", async () => {
        const page = setupPage();
        await page.ready();
        page.selectTab("optional");
        page.completeRequired();
        await page.poll();
        expect(page.button("optionalTab").getAttribute("aria-selected")).toBe("true");
        expect(page.isShown("tabPrerequisite")).toBe(false);
        expect(page.button("optionalBtn").disabled).toBe(false);
        expect(page.isShown("primaryAction")).toBe(false);
        page.button("downloadBtn").click();
        expect(page.downloads()).toHaveLength(0);
        expect(page.isShown("skipMtlBtn")).toBe(true);
        expect(page.button("skipMtlBtn").textContent).toBe("Finish & Close");

        page.selectTab("required");
        expect(page.isShown("primaryAction")).toBe(false);
        page.button("optionalBtn").click();
        expect(page.requests.filter((request) => request.path === "/api/setup/optional/download")).toHaveLength(0);
        expect(page.element("requiredTabState").textContent).toBe("Ready");
        page.selectTab("mtl");
        expect(page.isShown("primaryAction")).toBe(true);
        expect(page.button("skipMtlBtn").textContent).toBe("Skip & Close");
        expect(page.errors).toEqual([]);
    });

    it("preserves model choices and a checked storage draft across tab changes and polls", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready();
        page.element<HTMLInputElement>("mtl-local-mtl").click();
        page.selectTab("required");
        page.selectTab("mtl");
        expect(page.element<HTMLInputElement>("mtl-local-mtl").checked).toBe(true);
        page.editStorage("/new draft");
        await vi.advanceTimersByTimeAsync(400);
        page.selectTab("optional");
        page.element<HTMLInputElement>("opt-optional-model").click();
        await page.poll();
        expect(page.element<HTMLInputElement>("opt-optional-model").checked).toBe(false);
        expect(page.isShown("optionalPanel")).toBe(true);
        page.selectTab("mtl");
        expect(page.element<HTMLInputElement>("mtlDir").value).toBe("/new draft");
        expect(page.element("storageResolvedPath").textContent).toBe("/new draft");
        expect(page.button("storageApplyBtn").disabled).toBe(false);
        expect(page.element<HTMLInputElement>("mtl-local-mtl").checked).toBe(true);
        expect(page.button("downloadBtn").disabled).toBe(true);
        expect(page.requests.filter((request) => request.path === "/api/setup/storage")).toHaveLength(0);
        expect(page.errors).toEqual([]);
    });

    it("keeps a global finish action available outside MTL when its models are already downloaded", async () => {
        const page = setupPage(undefined, true, true);
        await page.ready("required");
        expect(page.isShown("primaryAction")).toBe(false);
        expect(page.isShown("skipMtlBtn")).toBe(true);
        expect(page.button("skipMtlBtn").textContent).toBe("Finish & Close");
        page.selectTab("mtl");
        expect(page.isShown("primaryAction")).toBe(true);
        expect(page.button("downloadBtn").textContent).toBe("Finish & Close");
        expect(page.isShown("skipMtlBtn")).toBe(false);
        page.selectTab("optional");
        expect(page.isShown("skipMtlBtn")).toBe(true);
        expect(page.errors).toEqual([]);
    });

    it("downloads optional models through their licence while keeping the primary action hidden", async () => {
        const page = setupPage(undefined, true);
        await page.ready("optional");
        expect(page.isShown("primaryAction")).toBe(false);
        page.button("optionalBtn").click();
        await vi.waitFor(() => expect(page.element("termsModelName").textContent).toBe(optionalModel.name));
        expect(page.requests.filter((request) => request.path === "/api/setup/optional/download")).toHaveLength(0);
        page.element<HTMLInputElement>("termsAcceptChk").click();
        page.button("termsAcceptBtn").click();
        await vi.waitFor(() => expect(page.button("optionalBtn").textContent).toContain("Downloading"));
        expect(page.requests.filter((request) => request.path === "/api/setup/optional/download")).toEqual([
            { path: "/api/setup/optional/download", method: "POST", body: { ids: [optionalModel.id] } },
        ]);
        expect(page.downloads()).toHaveLength(0);
        page.selectTab("required");
        await page.poll();
        expect(page.button("requiredTab").getAttribute("aria-selected")).toBe("true");
        expect(page.element("optionalTabState").textContent).toBe("Downloading");
        expect(page.isShown("primaryAction")).toBe(false);
        expect(page.errors).toEqual([]);
    });
});

describe("MTL storage location", () => {
    it("requires applying a draft before using model status from the new folder", async () => {
        const page = setupPage(undefined, true, true);
        await page.ready();
        expect(page.element<HTMLInputElement>("storageDefault").checked).toBe(true);
        expect(page.element<HTMLInputElement>("mtlDir").readOnly).toBe(true);
        expect(page.button("downloadBtn").textContent).toBe("Finish & Close");

        const release = page.holdResponse("/api/setup/status");
        const pending = page.poll();
        page.storageReports.set("/large models", { exists: false, can_create: true });
        page.editStorage("/large models");
        expect(page.button("downloadBtn").disabled).toBe(true);
        await vi.waitFor(() => expect(page.button("storageApplyBtn").disabled).toBe(false));
        expect(page.element("storageResolvedPath").textContent).toBe("/large models");
        expect(page.requests.filter((request) => request.path === "/api/setup/storage")).toHaveLength(0);

        page.button("storageApplyBtn").click();
        await vi.waitFor(() =>
            expect(page.requests.filter((request) => request.path === "/api/setup/storage")).toEqual([
                { path: "/api/setup/storage", method: "POST", body: { path: "/large models" } },
            ]),
        );
        release();
        await pending;
        await vi.waitFor(() => expect(page.element<HTMLInputElement>("mtl-local-mtl").disabled).toBe(false));
        expect(page.button("downloadBtn").textContent).not.toBe("Finish & Close");
        expect(page.element<HTMLInputElement>("mtlDir").value).toBe("/large models");
        expect(page.errors).toEqual([]);
    });

    it("shows empty and unusable paths inline without silently choosing the default", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready();
        page.editStorage("");
        await vi.advanceTimersByTimeAsync(400);
        expect(page.element("mtlDir").getAttribute("aria-invalid")).toBe("true");
        expect(page.button("storageApplyBtn").disabled).toBe(true);
        expect(page.button("downloadBtn").disabled).toBe(true);
        expect(
            page.requests.some(
                (request) =>
                    request.path === "/api/setup/storage/inspect" && (request.body as { path: string }).path === "",
            ),
        ).toBe(false);

        page.storageReports.set("/locked", { ok: false, error: "Cannot write to this folder." });
        page.editStorage("/locked");
        await vi.advanceTimersByTimeAsync(400);
        await vi.waitFor(() =>
            expect(page.element("storageFeedback").textContent).toContain("Cannot write to this folder."),
        );
        expect(page.element("mtlDir").getAttribute("aria-invalid")).toBe("true");
        expect(page.button("storageApplyBtn").textContent).toBe("Check folder");
        page.button("storageApplyBtn").click();
        await vi.advanceTimersByTimeAsync(0);
        expect(page.requests.filter((request) => request.path === "/api/setup/storage")).toHaveLength(0);
        expect(page.downloads()).toHaveLength(0);
    });

    it("uses the backend's resolved destination and allows returning to the portable default", async () => {
        const page = setupPage(undefined, true);
        await page.ready();
        page.storageReports.set("custom/models", { path: "/app/custom/models" });
        page.editStorage("custom/models");
        await vi.waitFor(() => expect(page.button("storageApplyBtn").disabled).toBe(false));
        expect(page.element("storageResolvedPath").textContent).toBe("/app/custom/models");
        page.button("storageApplyBtn").click();
        await vi.waitFor(() => expect(page.element<HTMLInputElement>("mtlDir").value).toBe("/app/custom/models"));

        page.element<HTMLInputElement>("storageDefault").click();
        expect(page.element<HTMLInputElement>("mtlDir").readOnly).toBe(true);
        expect(page.element<HTMLInputElement>("mtlDir").value).toBe("models");
        await vi.waitFor(() => expect(page.button("storageApplyBtn").disabled).toBe(false));
        page.button("storageApplyBtn").click();
        await vi.waitFor(() =>
            expect(page.requests.filter((request) => request.path === "/api/setup/storage")).toHaveLength(2),
        );
        expect(page.requests.filter((request) => request.path === "/api/setup/storage")[1]!.body).toEqual({
            path: "models",
        });
        expect(page.element("storageResolvedPath").textContent).toBe("/app/models");
        expect(page.errors).toEqual([]);
    });

    it("ignores a late validation response for a path that has been replaced", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready();
        const release = page.holdResponse("/api/setup/storage/inspect", "/old");
        page.storageReports.set("/old", { ok: false, error: "Old folder is unavailable." });
        page.editStorage("/old");
        await vi.advanceTimersByTimeAsync(400);
        expect(
            page.requests.some(
                (request) =>
                    request.path === "/api/setup/storage/inspect" && (request.body as { path: string }).path === "/old",
            ),
        ).toBe(true);

        page.editStorage("/new");
        await vi.advanceTimersByTimeAsync(400);
        await vi.waitFor(() => expect(page.element("storageResolvedPath").textContent).toBe("/new"));
        release();
        await vi.advanceTimersByTimeAsync(0);
        expect(page.element("storageResolvedPath").textContent).toBe("/new");
        expect(page.element("storageFeedback").textContent).not.toContain("Old folder is unavailable.");
        expect(page.button("storageApplyBtn").disabled).toBe(false);
        expect(page.errors).toEqual([]);
    });

    it("retains a draft after a save failure and permits retry without using stale model status", async () => {
        const page = setupPage(undefined, true);
        await page.ready();
        page.editStorage("/custom");
        await vi.waitFor(() => expect(page.button("storageApplyBtn").disabled).toBe(false));
        page.rejected.set("/api/setup/storage", "The configuration could not be saved.");
        page.button("storageApplyBtn").click();
        await vi.waitFor(() => expect(page.element("storageFeedback").textContent).toContain("could not be saved"));
        expect(page.element<HTMLInputElement>("mtlDir").value).toBe("/custom");
        expect(page.element<HTMLInputElement>("mtlDir").disabled).toBe(false);
        expect(page.button("storageApplyBtn").disabled).toBe(false);
        expect(page.button("downloadBtn").disabled).toBe(true);

        page.rejected.clear();
        page.button("storageApplyBtn").click();
        await vi.waitFor(() => expect(page.button("storageApplyBtn").disabled).toBe(true));
        await vi.waitFor(() => expect(page.element<HTMLInputElement>("mtlDir").disabled).toBe(false));
        await vi.waitFor(() => expect(page.element<HTMLInputElement>("mtl-local-mtl").disabled).toBe(false));
        page.element<HTMLInputElement>("mtl-local-mtl").click();
        expect(page.button("downloadBtn").disabled).toBe(false);
        expect(page.errors).toEqual([]);
    });

    it("blocks conflicting changes after a lost save but lets the same folder be retried", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true, true);
        await page.ready();
        page.editStorage("/custom");
        await vi.advanceTimersByTimeAsync(400);
        page.lostReplies.set("/api/setup/storage", "before");
        page.button("storageApplyBtn").click();
        await vi.advanceTimersByTimeAsync(0);
        expect(page.element("storageFeedback").textContent).toContain("has not been confirmed");
        expect(page.button("storageApplyBtn").textContent).toBe("Retry saving folder");
        expect(page.button("downloadBtn").disabled).toBe(true);

        // A status still showing the old folder cannot rule out a delayed save.
        page.element<HTMLInputElement>("storageDefault").click();
        await vi.advanceTimersByTimeAsync(400);
        expect(page.button("storageApplyBtn").disabled).toBe(true);
        expect(page.button("downloadBtn").disabled).toBe(true);
        expect(page.button("skipMtlBtn").disabled).toBe(true);

        page.rejected.set("/api/setup/status", "Temporarily unavailable");
        await page.poll();
        page.editStorage("/custom");
        await vi.advanceTimersByTimeAsync(400);
        expect(page.button("storageApplyBtn").disabled).toBe(false);
        expect(page.button("downloadBtn").disabled).toBe(true);
        page.lostReplies.clear();
        page.rejected.clear();
        page.button("storageApplyBtn").click();
        await vi.advanceTimersByTimeAsync(0);
        expect(page.element("storageFeedback").textContent).toContain("Storage folder saved");
        expect(page.element<HTMLInputElement>("mtl-local-mtl").disabled).toBe(false);
        expect(page.errors).toEqual([]);
    });

    it("reconciles a delayed committed save without replacing the user's newer draft", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true, true);
        await page.ready();
        page.editStorage("/custom");
        await vi.advanceTimersByTimeAsync(400);
        page.lostReplies.set("/api/setup/storage", "before");
        page.button("storageApplyBtn").click();
        await vi.advanceTimersByTimeAsync(0);
        page.element<HTMLInputElement>("storageDefault").click();
        await vi.advanceTimersByTimeAsync(400);
        expect(page.button("storageApplyBtn").disabled).toBe(true);

        // The original request finishes on the server after its response failed.
        page.commitStorage("/custom");
        await page.poll();
        expect(page.element<HTMLInputElement>("mtlDir").value).toBe("models");
        expect(page.element<HTMLInputElement>("storageDefault").checked).toBe(true);
        expect(page.element("storageResolvedPath").textContent).toBe("/app/models");
        expect(page.element("mtlStorage").dataset.appliedPath).toBe("/custom");
        expect(page.button("storageApplyBtn").disabled).toBe(false);
        expect(page.button("downloadBtn").disabled).toBe(true);

        page.lostReplies.clear();
        page.button("storageApplyBtn").click();
        await vi.advanceTimersByTimeAsync(0);
        expect(page.element<HTMLInputElement>("mtlDir").value).toBe("models");
        expect(page.element("mtlStorage").dataset.appliedPath).toBe("/app/models");
        expect(page.button("downloadBtn").textContent).toBe("Finish & Close");
        expect(page.button("downloadBtn").disabled).toBe(false);
        expect(page.errors).toEqual([]);
    });

    it("does not publish an incomplete save confirmation before authoritative status arrives", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready();
        page.editStorage("/custom");
        await vi.advanceTimersByTimeAsync(400);
        const release = page.holdResponse("/api/setup/status");
        page.malformedReplies.set("/api/setup/storage", { path: "/custom" });
        page.button("storageApplyBtn").click();
        await vi.advanceTimersByTimeAsync(0);
        expect(page.element("storageFeedback").textContent).toContain("incomplete storage confirmation");
        expect(page.element<HTMLInputElement>("mtlDir").value).toBe("/custom");
        expect(page.element("mtlStorage").dataset.appliedPath).toBe("/app/models");
        expect(page.button("downloadBtn").disabled).toBe(true);
        release();
        await vi.advanceTimersByTimeAsync(0);
        expect(page.element("storageFeedback").textContent).toContain("Storage folder confirmed");
        expect(page.element("mtlStorage").dataset.appliedPath).toBe("/custom");
        expect(page.element<HTMLInputElement>("mtl-local-mtl").disabled).toBe(false);
        expect(page.errors).toEqual([]);
    });

    it("preserves a relative folder spelling when status confirms a lost save response", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready();
        page.storageReports.set("custom/models", { path: "/app/custom/models" });
        page.editStorage("custom/models");
        await vi.advanceTimersByTimeAsync(400);
        page.lostReplies.set("/api/setup/storage", "after");
        page.button("storageApplyBtn").click();
        await vi.advanceTimersByTimeAsync(0);
        expect(page.element("storageFeedback").textContent).toContain("Storage folder confirmed");
        expect(page.element<HTMLInputElement>("mtlDir").value).toBe("custom/models");
        expect(page.element("storageResolvedPath").textContent).toBe("/app/custom/models");
        expect(page.element<HTMLInputElement>("mtl-local-mtl").disabled).toBe(false);
        expect(page.button("storageApplyBtn").disabled).toBe(true);
        expect(page.errors).toEqual([]);
    });

    it.each(["debounce", "request"] as const)(
        "restarts a %s interrupted by a persisted page restore",
        async (phase) => {
            vi.useFakeTimers();
            const page = setupPage(undefined, true);
            await page.ready();
            const statusCount = page.requests.filter((request) => request.path === "/api/setup/status").length;
            page.stalled.set("/api/setup/storage/inspect", "request");
            page.editStorage("/custom");
            if (phase === "request") await vi.advanceTimersByTimeAsync(400);
            page.pageTransition("pagehide");
            await vi.advanceTimersByTimeAsync(400);
            page.stalled.clear();
            page.pageTransition("pageshow");
            await vi.advanceTimersByTimeAsync(0);
            expect(page.element("storageResolvedPath").textContent).toBe("/custom");
            expect(page.button("storageApplyBtn").disabled).toBe(false);
            expect(page.requests.filter((request) => request.path === "/api/setup/status")).toHaveLength(
                statusCount + 1,
            );
            expect(page.aborted).toEqual(phase === "request" ? ["/api/setup/storage/inspect"] : []);
            expect(page.errors).toEqual([]);
        },
    );

    it("does not restart storage checks on initial pageshow or while closing", async () => {
        vi.useFakeTimers();
        const page = setupPage(undefined, true);
        await page.ready();
        const initialRequests = page.requests.length;
        page.pageTransition("pageshow", false);
        await vi.advanceTimersByTimeAsync(0);
        expect(page.requests).toHaveLength(initialRequests);
        page.button("skipMtlBtn").click();
        await vi.advanceTimersByTimeAsync(0);
        const closingRequests = page.requests.length;
        page.pageTransition("pagehide");
        page.pageTransition("pageshow");
        await vi.advanceTimersByTimeAsync(0);
        expect(page.requests).toHaveLength(closingRequests);
        expect(page.element<HTMLInputElement>("mtlDir").disabled).toBe(true);
        expect(page.errors).toEqual([]);
    });

    it("keeps storage locked during licence prompts and downloads to the captured applied destination", async () => {
        const page = setupPage(undefined, true);
        await page.ready();
        page.editStorage("/custom");
        await vi.waitFor(() => expect(page.button("storageApplyBtn").disabled).toBe(false));
        page.button("storageApplyBtn").click();
        await vi.waitFor(() => expect(page.element<HTMLInputElement>("mtlDir").disabled).toBe(false));
        await vi.waitFor(() => expect(page.element<HTMLInputElement>("mtl-local-mtl").disabled).toBe(false));
        page.element<HTMLInputElement>("mtl-local-mtl").click();
        page.button("downloadBtn").click();
        await vi.waitFor(() => expect(page.element("termsModelName").textContent).toBe(localModel.name));
        expect(page.element<HTMLInputElement>("mtlDir").disabled).toBe(true);
        expect(page.element<HTMLInputElement>("storageDefault").disabled).toBe(true);
        expect(page.element<HTMLInputElement>("storageCustom").disabled).toBe(true);
        await page.poll();
        expect(page.element<HTMLInputElement>("mtlDir").disabled).toBe(true);

        page.button("termsDeclineBtn").click();
        await vi.waitFor(() => expect(page.element<HTMLInputElement>("mtlDir").disabled).toBe(false));
        expect(page.downloads()).toHaveLength(0);
        page.button("downloadBtn").click();
        await vi.waitFor(() => expect(page.document.documentElement.classList.contains("fox-terms-open")).toBe(true));
        expect(page.element("termsModelName").textContent).toBe(localModel.name);
        // A later DOM update cannot redirect a request whose licence flow already began.
        page.element<HTMLInputElement>("mtlDir").value = "/changed-after-start";
        page.element<HTMLInputElement>("termsAcceptChk").click();
        expect(page.button("termsAcceptBtn").disabled).toBe(false);
        page.button("termsAcceptBtn").click();
        await vi.waitFor(() => expect(page.downloads()).toHaveLength(1));
        expect(page.downloads()[0]!.body).toEqual({ mtl_dir: "/custom", selected_mtl: [localModel.id] });
        expect(page.element<HTMLInputElement>("mtlDir").disabled).toBe(true);
        expect(page.errors).toEqual([]);
    });
});
