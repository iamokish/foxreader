/* Explicit update checks only. Polling this backend never contacts GitHub. */
function bytes(value) {
    if (!Number.isFinite(value) || value < 0) return "0 B";
    const units = ["B", "KiB", "MiB", "GiB"];
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit++;
    }
    return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

export function installUpdates() {
    const panel = document.getElementById("updatesPanel");
    if (!panel) return () => {};
    const element = (id) => document.getElementById(id);
    const check = element("checkUpdateBtn");
    const download = element("downloadUpdateBtn");
    const cancel = element("cancelUpdateBtn");
    const status = element("updateStatus");
    const version = element("updateVersion");
    const progress = element("updateProgress");
    const ready = element("updateReady");
    let latest = null;
    let downloading = false;
    let busy = true;
    let stopped = false;
    let polling = 0;
    let cancelling = false;
    let pollSerial = 0;
    const controllers = new Set();

    const controls = () => {
        check.disabled = busy || downloading;
        download.disabled = busy || downloading;
        download.hidden = !latest?.available || !latest?.compatible || downloading;
        cancel.hidden = !downloading;
        cancel.disabled = cancelling;
    };
    async function request(path, method = "GET") {
        const controller = new AbortController();
        controllers.add(controller);
        const timeout = window.setTimeout(() => controller.abort(), path === "check" ? 45000 : 10000);
        try {
            const response = await fetch(`/api/updates/${path}`, {
                method,
                cache: "no-store",
                credentials: "same-origin",
                headers: method === "POST" || path === "check" ? { "X-Fox-Reader-Update": "1" } : {},
                signal: controller.signal,
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "The update request failed.");
            return data;
        } finally {
            clearTimeout(timeout);
            controllers.delete(controller);
        }
    }
    const showVersion = (current) => {
        version.textContent = `Installed: ${current || "source checkout"}`;
        if (latest?.version) version.textContent += ` · Latest: ${latest.version}`;
    };
    const showStatus = (data, initial = false) => {
        showVersion(data.current_version);
        downloading = data.state === "downloading";
        if (!downloading) cancelling = false;
        progress.hidden = !downloading;
        if (downloading) {
            const total = Math.max(0, Number(data.total) || 0);
            const received = Math.max(0, Number(data.downloaded) || 0);
            progress.max = total || 1;
            progress.value = Math.min(total || 1, received);
            status.textContent = `${data.message || "Downloading update…"} ${bytes(received)}${total ? ` / ${bytes(total)}` : ""}`;
        } else if (data.state === "complete") {
            status.textContent = "The update archive has been verified.";
        } else if (data.state === "error" || data.state === "cancelled") {
            status.textContent = data.message || (data.state === "cancelled" ? "Download cancelled." : "The download failed.");
        } else if (initial && data.download_supported === false) {
            status.textContent = data.reason || "Download a packaged release to use verified updates.";
        }
        ready.hidden = data.state !== "complete" || !data.archive;
        element("updateArchive").textContent = data.state === "complete" ? data.archive || "" : "";
        controls();
    };
    async function poll(initial = false) {
        clearTimeout(polling);
        const serial = ++pollSerial;
        try {
            const data = await request("status");
            if (stopped || serial !== pollSerial) return;
            showStatus(data, initial);
            if (downloading) polling = window.setTimeout(poll, 1000);
        } catch (_) {
            if (stopped || serial !== pollSerial) return;
            if (downloading) {
                status.textContent = "Waiting for the backend to report download progress…";
                polling = window.setTimeout(poll, 2000);
            } else if (initial) {
                version.textContent = "Installed version unavailable";
                status.textContent = "Could not read update status. Try checking when the backend is ready.";
            }
        } finally {
            if (initial && !stopped) {
                busy = false;
                controls();
            }
        }
    }
    const onCheck = async () => {
        busy = true;
        latest = null;
        controls();
        element("updateDetails").hidden = true;
        status.textContent = "Checking for updates…";
        try {
            const data = await request("check");
            if (stopped) return;
            latest = data;
            showVersion(data.current_version);
            element("updateNotes").textContent = data.notes || "No release notes provided.";
            element("updateDetails").hidden = !data.version;
            const link = element("updateReleaseLink");
            link.hidden = true;
            try {
                const url = new URL(data.release_url);
                if (url.protocol === "https:" && url.hostname === "github.com" && !url.username && !url.password) {
                    link.href = url.href;
                    link.hidden = false;
                }
            } catch (_) {}
            if (data.available && data.compatible) {
                status.textContent = `Version ${data.version} is available${data.variant ? ` (${data.variant})` : ""}. Download: ${bytes(data.download_size)}.`;
            } else {
                status.textContent = data.reason || (data.available
                    ? "This release has no download matching your installation."
                    : "You are up to date.");
            }
        } catch (error) {
            if (!stopped) status.textContent = error instanceof Error ? error.message : "Could not check for updates.";
        } finally {
            busy = false;
            if (!stopped) controls();
        }
    };
    const onDownload = async () => {
        busy = true;
        ready.hidden = true;
        controls();
        status.textContent = "Preparing verified download…";
        try {
            const data = await request("download", "POST");
            if (stopped) return;
            showStatus(data);
            await poll();
        } catch (error) {
            if (!stopped) status.textContent = error instanceof Error ? error.message : "Could not start the download.";
        } finally {
            busy = false;
            if (!stopped) controls();
        }
    };
    const onCancel = async () => {
        cancelling = true;
        ++pollSerial;
        clearTimeout(polling);
        controls();
        try {
            const data = await request("cancel", "POST");
            if (!stopped) {
                showStatus(data);
                await poll();
            }
        } catch (error) {
            cancelling = false;
            if (!stopped) {
                status.textContent = error instanceof Error ? error.message : "Could not cancel the download.";
                controls();
            }
        }
    };
    const cleanup = () => {
        stopped = true;
        clearTimeout(polling);
        for (const controller of controllers) controller.abort();
        check.removeEventListener("click", onCheck);
        download.removeEventListener("click", onDownload);
        cancel.removeEventListener("click", onCancel);
        window.removeEventListener("pagehide", cleanup);
    };
    check.addEventListener("click", onCheck);
    download.addEventListener("click", onDownload);
    cancel.addEventListener("click", onCancel);
    window.addEventListener("pagehide", cleanup);
    controls();
    void poll(true);
    return cleanup;
}
