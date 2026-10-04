import { beforeEach, describe, expect, it, vi, afterEach } from "vitest";
import { syncMLStateFromBackend } from "./mlStatus";

const SELECT_HTML = `
<input type="checkbox" id="enableML" />
<select id="languageSelect">
  <option value="japanese">Japanese</option>
  <option value="chinese">Chinese</option>
  <option value="korean">Korean</option>
</select>`;

function elements() {
    const checkbox = document.getElementById("enableML") as HTMLInputElement;
    const select = document.getElementById("languageSelect") as HTMLSelectElement;
    return { checkbox, select };
}

function mockStatus(payload: unknown) {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => payload,
    } as Response);
}

beforeEach(() => {
    vi.restoreAllMocks();
    document.body.innerHTML = SELECT_HTML;
});

afterEach(() => {
    document.body.innerHTML = "";
});

describe("syncMLStateFromBackend", () => {
    it("checks the switch and selects the single supported language", async () => {
        const { select } = elements();
        select.value = "chinese";
        mockStatus({ loaded: true, lang: "japanese", model_id: "vntl-llama3-8b-v2", languages: ["japanese"] });

        await syncMLStateFromBackend();

        const { checkbox, select: after } = elements();
        expect(checkbox.checked).toBe(true);
        expect(after.value).toBe("japanese");
    });

    it("picks the first dropdown option the multi-language model supports", async () => {
        const { select } = elements();
        select.value = "korean";
        mockStatus({
            loaded: true,
            lang: "korean",
            model_id: "gemma-4-e4b-q8-uncensored",
            languages: ["japanese", "chinese", "korean"],
        });

        await syncMLStateFromBackend();

        expect(elements().select.value).toBe("japanese");
        expect(elements().checkbox.checked).toBe(true);
    });

    it("normalizes messy language entries before matching", async () => {
        mockStatus({ loaded: true, lang: null, model_id: "x", languages: ["  Chinese ", 7, "", "chinese"] });

        await syncMLStateFromBackend();

        expect(elements().select.value).toBe("chinese");
        expect(elements().checkbox.checked).toBe(true);
    });

    it("leaves everything alone when nothing is loaded", async () => {
        const { select } = elements();
        select.value = "korean";
        mockStatus({ loaded: false, lang: null, model_id: null, languages: [] });

        await syncMLStateFromBackend();

        expect(elements().checkbox.checked).toBe(false);
        expect(elements().select.value).toBe("korean");
    });

    it("stays silent and unchecked when the backend is unreachable", async () => {
        vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new Error("down"));

        await expect(syncMLStateFromBackend()).resolves.toBeUndefined();
        expect(elements().checkbox.checked).toBe(false);
    });

    it("ignores malformed payloads without touching the UI", async () => {
        const { select } = elements();
        select.value = "chinese";
        mockStatus({ loaded: "yes", languages: "japanese" });

        await syncMLStateFromBackend();

        expect(elements().checkbox.checked).toBe(false);
        expect(elements().select.value).toBe("chinese");
    });

    it("checks the switch but keeps the select when no option matches", async () => {
        elements().select.value = "korean";
        mockStatus({ loaded: true, lang: "klingon", model_id: "x", languages: ["klingon"] });

        await syncMLStateFromBackend();

        expect(elements().select.value).toBe("korean");
        expect(elements().checkbox.checked).toBe(true);
    });

    it("checks the switch when languages are missing but loaded is true", async () => {
        mockStatus({ loaded: true });

        await syncMLStateFromBackend();

        expect(elements().checkbox.checked).toBe(true);
    });

    it("lets a switch the user already touched win (no fetch, no changes)", async () => {
        const { checkbox, select } = elements();
        checkbox.checked = true;
        select.value = "korean";
        const spy = vi.spyOn(globalThis, "fetch");

        await syncMLStateFromBackend();

        expect(spy).not.toHaveBeenCalled();
        expect(select.value).toBe("korean");
    });

    it("skips while a load is in flight", async () => {
        elements().checkbox.disabled = true;
        const spy = vi.spyOn(globalThis, "fetch");

        await syncMLStateFromBackend();

        expect(spy).not.toHaveBeenCalled();
        expect(elements().checkbox.checked).toBe(false);
    });

    it("resolves quietly when the elements are absent", async () => {
        document.body.innerHTML = "";
        mockStatus({ loaded: true, languages: ["japanese"] });

        await expect(syncMLStateFromBackend()).resolves.toBeUndefined();
    });
});
