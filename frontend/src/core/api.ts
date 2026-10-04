import type {
    FreeformRequest,
    CropRequest,
    BubbleRequest,
    SplitRequest,
    TranslationRequest,
    TranslationResponse,
    ActiveCustomEndpointsResponse,
    OCRResponse,
    FolderResponse,
    FolderInspectRequest,
    FolderInspectResponse,
    FolderStateResponse,
    Region,
    ProcessImageRequest,
    SavePreviewRequest,
    PreviewResult,
    ProgressSnapshot,
    CleanMethodsResponse,
    TranslateEndpoint,
} from "./types";

/**
 * A failed request, with the two things a caller may need to branch on.
 *
 * `code` is the backend's `X-Fox-Code` header: a 409 from a save means either
 * "that file is already there" (answerable -- ask, then retry with `overwrite`)
 * or "this preview is stale" (not answerable), and matching on the prose to tell
 * them apart would break the first time the wording changed.
 *
 * The message stays whatever it has always been -- the backend's `detail`, or
 * `HTTP <status>` when there is nothing better -- so existing call sites that
 * only show `err.message` are unaffected.
 */
export class ApiError extends Error {
    status: number;
    code?: string;

    constructor(message: string, status: number, code?: string) {
        super(message);
        this.name = "ApiError";
        this.status = status;
        this.code = code;
    }
}

/** Pull the most specific message the response offers, without ever throwing. */
async function errorFrom(res: Response): Promise<ApiError> {
    let detail = `HTTP ${res.status}`;
    try {
        const body = await res.json();
        detail = body?.detail || body?.error || detail;
    } catch {
        /* not JSON, or no body to read: the status is all we have */
    }
    let code: string | undefined;
    try {
        code = res.headers?.get?.("X-Fox-Code") ?? undefined;
    } catch {
        /* a stubbed Response without headers */
    }
    return new ApiError(detail, res.status, code);
}

async function post(url: string, body: unknown): Promise<Response> {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw await errorFrom(res);
    return res;
}

/**
 * Load a folder of pages, optionally naming where saves should go.
 *
 * The extra keys are only sent when given, so a bare `loadFolder(path)` posts
 * exactly `{path}` and the backend applies its own destination precedence
 * (explicit, then remembered, then `<source>/fox_tled`).
 */
export async function loadFolder(
    path: string,
    opts?: { dest?: string; createDest?: boolean },
): Promise<FolderResponse & { status: string; error?: string }> {
    const body: Record<string, unknown> = { path };
    if (opts?.dest !== undefined) body.dest = opts.dest;
    if (opts?.createDest !== undefined) body.create_dest = opts.createDest;
    const res = await post("/api/load_folder", body);
    return res.json();
}

/**
 * Report on a source/destination pair without touching either.
 *
 * Safe to call while the user types (the picker debounces): nothing here creates
 * a folder. Leaving `dest` empty asks the backend for its own suggestion.
 */
export async function inspectFolder(data: FolderInspectRequest): Promise<FolderInspectResponse> {
    const res = await post("/api/folder/inspect", data);
    return res.json();
}

/** The folder pair the backend currently has loaded, plus the last one it saw. */
export async function folderState(): Promise<FolderStateResponse> {
    const res = await fetch("/api/folder/state");
    if (!res.ok) throw await errorFrom(res);
    return res.json();
}

export async function ocrCrop(data: CropRequest): Promise<OCRResponse> {
    const res = await post("/api/ocr_crop", data);
    return res.json();
}

export async function ocrFreeform(data: FreeformRequest): Promise<OCRResponse> {
    const res = await post("/api/ocr_freeform", data);
    return res.json();
}

export async function bubbleDetect(data: BubbleRequest): Promise<Region[]> {
    const res = await post("/api/bubble", data);
    return res.json();
}

export async function splitBubble(data: SplitRequest): Promise<Region[]> {
    const res = await post("/api/splitbubble", data);
    return res.json();
}

export interface TranslateExtras {
    character_info?: TranslationRequest["character_info"];
    context_character_links?: TranslationRequest["context_character_links"];
    meta_id?: TranslationRequest["meta_id"];
}

export async function translate(
    endpoint: TranslateEndpoint,
    text: string,
    source_lang: string,
    lang_group?: string,
    context?: [string, string][],
    extras?: TranslateExtras,
): Promise<TranslationResponse> {
    const body: TranslationRequest = { text, source_lang };
    if (lang_group) body.lang_group = lang_group;
    if (context?.length) body.context = context;
    if (extras?.character_info?.length) body.character_info = extras.character_info;
    if (extras?.context_character_links?.length) body.context_character_links = extras.context_character_links;
    if (extras?.meta_id) body.meta_id = extras.meta_id;
    const res = await post(endpoint, body);
    return res.json();
}

export async function getActiveCustomEndpoints(): Promise<ActiveCustomEndpointsResponse> {
    const res = await fetch("/api/user_endpoints/active");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

export async function inpaintPreview(data: ProcessImageRequest): Promise<PreviewResult> {
    const res = await post("/inpaint/preview", data);
    // The token identifies the artifact the backend just cached, so "Save this
    // image" can commit exactly what the user is looking at instead of asking for
    // the page to be typeset a second time.
    return { blob: await res.blob(), token: res.headers.get("X-Preview-Token") ?? "" };
}

export async function savePreview(data: SavePreviewRequest): Promise<void> {
    // The 409s from this route are the interesting ones: `exists` means the page
    // is already in the destination and the user has to answer for it, and
    // `stale_preview` means the artifact is gone. `post` puts both on the
    // ApiError as `code`, and keeps the backend's wording as the message.
    await post("/inpaint/save-preview", data);
}

export async function inpaintGenerate(data: ProcessImageRequest): Promise<void> {
    await post("/inpaint/generate", data);
}

/**
 * What a long backend job is doing right now.
 *
 * Never throws: this is polled on a timer while a job runs, and a single dropped
 * poll should leave the last message on screen rather than surface an error.
 */
export async function getProgress(name: string): Promise<ProgressSnapshot | null> {
    try {
        const res = await fetch(`/api/progress/${encodeURIComponent(name)}`);
        if (!res.ok) return null;
        return (await res.json()) as ProgressSnapshot;
    } catch {
        return null;
    }
}

export async function getCleanMethods(): Promise<CleanMethodsResponse> {
    const res = await fetch("/api/clean/methods");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

export interface MLControlResponse {
    status: boolean;
    message: string;
    lang?: string;
}

export async function mlLoad(lang: string): Promise<MLControlResponse> {
    const res = await post("/ml/control/load", { lang });
    return res.json();
}

export async function mlUnload(): Promise<MLControlResponse> {
    const res = await fetch("/ml/control/unload", { method: "POST" });
    return res.json();
}

export interface MLStatusResponse {
    loaded: boolean;
    lang?: string | null;
    model_id?: string | null;
    languages?: unknown;
}

/**
 * Which local MTL model (if any) the backend still has loaded.
 *
 * Asked once on startup so the MTL switch can be re-checked after a refresh.
 * Throws on transport/HTTP failure -- the caller treats that as "unknown"
 * and leaves the UI exactly as it was.
 */
export async function mlStatus(): Promise<MLStatusResponse> {
    const res = await fetch("/ml/control/status", { cache: "no-store" });
    if (!res.ok) throw await errorFrom(res);
    return res.json();
}

export interface SessionResponse {
    session_id: string;
}

async function getSessionData(url: string): Promise<SessionResponse> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

export async function getSession(): Promise<SessionResponse> {
    return getSessionData("/api/session");
}

export async function getHealth(): Promise<SessionResponse> {
    return getSessionData("/api/health");
}
