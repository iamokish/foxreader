/* Shared by the reader, setup wizard and settings pages. No native dependency. */
const MAX_QUEUE = 20;
const MAX_BATCH = 10;

function redact(value) {
    return value
        .replace(/(https?:\/\/[^\s?#]+)[?#][^\s]*/gi, "$1?[redacted]")
        .replace(/(https?:\/\/)[^\s/@]+:[^\s/@]+@/gi, "$1[redacted]@")
        .replace(/\b(authorization|api[ _-]?key|access[ _-]?token|token|password|secret)([\s"']*[:=][\s"']*)(?:bearer\s+)?[^\s,;"'<>]+/gi, "$1$2[redacted]")
        .replace(/\bbearer\s+[a-z0-9._~+/=-]+/gi, "Bearer [redacted]")
        .slice(0, 2000);
}

export function toggleFullscreen() {
    try {
        if (typeof window.foxReaderNativeFullscreen === "function") {
            window.foxReaderNativeFullscreen();
            return;
        }
        const result = document.fullscreenElement
            ? document.exitFullscreen?.()
            : document.documentElement.requestFullscreen?.();
        result?.catch(() => {});
    } catch (_) {
        // Older WebKit and restricted browser contexts can refuse fullscreen.
    }
}

export function installDesktopIntegration() {
    if (window.__FOX_DESKTOP__) return window.__FOX_DESKTOP__;
    let queue = [];
    let timer = 0;
    let sending = false;
    let stopped = false;
    let abort = null;
    const oldWarn = console.warn;
    const oldError = console.error;

    const schedule = () => {
        if (!timer && !stopped) timer = window.setTimeout(flush, 1000);
    };
    const record = (level, message, source = "") => {
        if (stopped || !message || queue.length >= MAX_QUEUE) return;
        queue.push({ level, message: redact(message), source: redact(source).slice(0, 200) });
        schedule();
    };
    async function flush() {
        timer = 0;
        if (sending || !queue.length || stopped) return;
        sending = true;
        const batch = [];
        // Keep UTF-8 payloads under the server's 24 KiB cap, including CJK text.
        while (queue.length && batch.length < MAX_BATCH) {
            const candidate = [...batch, queue[0]];
            if (new TextEncoder().encode(JSON.stringify(candidate)).length > 24 * 1024) break;
            batch.push(queue.shift());
        }
        abort = new AbortController();
        const timeout = window.setTimeout(() => abort?.abort(), 3000);
        try {
            const response = await fetch("/api/frontend-log", {
                method: "POST",
                headers: { "Content-Type": "application/json", "X-Fox-Reader-Client": "frontend" },
                body: JSON.stringify(batch),
                credentials: "same-origin",
                signal: abort.signal,
            });
            if (response.status === 429) queue = [];
        } catch (_) {
            // Do not retry a diagnostic or report its own transport failure.
        } finally {
            clearTimeout(timeout);
            abort = null;
            sending = false;
            if (queue.length) schedule();
        }
    }
    const onError = (event) => {
        if (event instanceof ErrorEvent) {
            let source = "";
            try {
                source = `${new URL(event.filename, location.href).pathname}:${event.lineno}:${event.colno}`;
            } catch (_) {}
            record("error", event.message || "Unhandled script error", source);
        }
    };
    const onRejection = (event) => {
        // Avoid serialising response objects, OCR text or request bodies.
        const reason = event.reason;
        const message = reason instanceof Error
            ? `${reason.name}: ${reason.message}`
            : typeof reason === "string" ? reason : "Unhandled promise rejection";
        record("error", message);
    };
    const wrapConsole = (original, level) => function (...args) {
        original.apply(console, args);
        const messages = args.map((value) => value instanceof Error
            ? `${value.name}: ${value.message}`
            : typeof value === "string" ? value : "[details omitted]");
        record(level, messages.join(" "));
    };
    const warn = wrapConsole(oldWarn, "warn");
    const error = wrapConsole(oldError, "error");
    const onKey = (event) => {
        const shortcut = event.key === "F11" || (event.altKey && event.key === "Enter");
        if (!shortcut || event.ctrlKey || event.metaKey || event.shiftKey || event.isComposing) return;
        // The native host consumes its accelerators before they reach the page.
        // Capture here also works while a settings input or modal has focus.
        event.preventDefault();
        event.stopPropagation();
        if (!event.repeat) toggleFullscreen();
    };
    console.warn = warn;
    console.error = error;
    window.addEventListener("error", onError);
    window.addEventListener("unhandledrejection", onRejection);
    window.addEventListener("keydown", onKey, true);
    const cleanup = () => {
        stopped = true;
        clearTimeout(timer);
        abort?.abort();
        queue = [];
        window.removeEventListener("error", onError);
        window.removeEventListener("unhandledrejection", onRejection);
        window.removeEventListener("keydown", onKey, true);
        if (console.warn === warn) console.warn = oldWarn;
        if (console.error === error) console.error = oldError;
        delete window.__FOX_DESKTOP__;
    };
    window.__FOX_DESKTOP__ = cleanup;
    record("info", "Frontend loaded", location.pathname);
    return cleanup;
}

