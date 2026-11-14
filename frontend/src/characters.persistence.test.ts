import { beforeEach, describe, expect, it, vi } from "vitest";
import {
    addCharacter,
    applyFolderCharacters,
    clearCharacters,
    deleteCharacter,
    fetchCharacters,
    getCharacters,
    importCharacterRows,
    updateCharacter,
} from "./characters";
import { state } from "./state";
import type { Entry } from "./state";
import type { CharacterInfo } from "./types";

function character(meta_id: string, name_en = meta_id): CharacterInfo {
    return { meta_id, name_en, name_ja: null, gender: null, alias_en: null, alias_ja: null };
}

function response(body: unknown, status = 200): Response {
    return { ok: status < 400, status, json: async () => body } as Response;
}

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((done) => {
        resolve = done;
    });
    return { promise, resolve };
}

beforeEach(() => {
    vi.restoreAllMocks();
    state.pageEntriesCache = {};
    applyFolderCharacters([], "workspace");
});

describe("destination character snapshots", () => {
    it("adopts loaded characters without deleting the saved roster or changing speaker ids", () => {
        const fetch = vi.spyOn(globalThis, "fetch");
        const restored = character("saved-speaker", "Aiko");
        const entry = { character_id: restored.meta_id } as Entry;
        state.pageEntriesCache["page.png"] = [entry];

        applyFolderCharacters([restored], "first");
        applyFolderCharacters([restored], "second"); // Reloading keeps ids but rotates the workspace identity.

        expect(fetch).not.toHaveBeenCalled();
        expect(getCharacters()).toEqual([restored]);
        expect(entry.character_id).toBe("saved-speaker");
    });

    it("replaces the old destination's roster, including with an empty validated snapshot", () => {
        applyFolderCharacters([character("old")], "old-workspace");
        applyFolderCharacters([character("new")], "new-workspace");
        expect(getCharacters()).toEqual([character("new")]);
        applyFolderCharacters([], "empty-workspace");
        expect(getCharacters()).toEqual([]);
    });

    it("ignores a startup read that finishes after folder activation", async () => {
        const pending = deferred<Response>();
        vi.spyOn(globalThis, "fetch").mockReturnValueOnce(pending.promise);
        const reading = fetchCharacters();
        const restored = character("saved-speaker");
        applyFolderCharacters([restored], "new-workspace");
        const entry = { character_id: restored.meta_id } as Entry;
        state.pageEntriesCache["page.png"] = [entry];
        pending.resolve(response({ characters: [] }));

        expect(await reading).toEqual([restored]);
        expect(getCharacters()).toEqual([restored]);
        expect(entry.character_id).toBe(restored.meta_id);
    });

    it("ignores a read overtaken by a saved edit", async () => {
        const pending = deferred<Response>();
        vi.spyOn(globalThis, "fetch")
            .mockReturnValueOnce(pending.promise)
            .mockResolvedValueOnce(response({ character: character("new") }, 201));
        const reading = fetchCharacters();
        await addCharacter({ name_en: "new" });
        pending.resolve(response({ characters: [] }));

        expect(await reading).toEqual([character("new")]);
        expect(getCharacters()).toEqual([character("new")]);
    });

    it("keeps the latest refresh when responses arrive out of order", async () => {
        const first = deferred<Response>();
        const second = deferred<Response>();
        vi.spyOn(globalThis, "fetch").mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
        const older = fetchCharacters();
        const newer = fetchCharacters();
        second.resolve(response({ characters: [character("new")], character_workspace: "workspace" }));
        await newer;
        first.resolve(response({ characters: [character("old")], character_workspace: "workspace" }));
        await older;

        expect(getCharacters()).toEqual([character("new")]);
    });

    it.each([
        ["add", () => addCharacter({ name_en: "Added" }), { character: character("added") }],
        ["update", () => updateCharacter("old", { name_en: "Updated" }), { character: character("old", "Updated") }],
        ["delete", () => deleteCharacter("old"), {}],
        ["clear", () => clearCharacters(), {}],
        ["import", () => importCharacterRows([{ name_en: "Imported" }]), { characters: [character("imported")] }],
    ])("does not apply a late %s result to another destination", async (_name, mutate, body) => {
        applyFolderCharacters([character("old")], "old-workspace");
        const pending = deferred<Response>();
        vi.spyOn(globalThis, "fetch").mockReturnValueOnce(pending.promise);
        const updating = (mutate as () => Promise<unknown>)();
        applyFolderCharacters([character("new")], "new-workspace");
        pending.resolve(response(body));
        await updating;

        expect(getCharacters()).toEqual([character("new")]);
    });
});

describe("saving character changes", () => {
    it("reports a failed clear and retains the roster and speaker assignments", async () => {
        const saved = character("saved");
        applyFolderCharacters([saved], "workspace");
        const entry = { character_id: saved.meta_id } as Entry;
        state.pageEntriesCache["page.png"] = [entry];
        vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(response({ error: "Could not save characters" }, 500));

        await expect(clearCharacters()).rejects.toThrow("Could not save characters");
        expect(getCharacters()).toEqual([saved]);
        expect(entry.character_id).toBe(saved.meta_id);
    });

    it("reports a disconnected backend without pretending clear was saved", async () => {
        applyFolderCharacters([character("saved")], "workspace");
        vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new TypeError("Failed to fetch"));

        await expect(clearCharacters()).rejects.toThrow("Failed to fetch");
        expect(getCharacters()).toEqual([character("saved")]);
    });

    it("clears the mirror only after the backend confirms the explicit clear", async () => {
        applyFolderCharacters([character("saved")], "workspace");
        const pending = deferred<Response>();
        const fetch = vi.spyOn(globalThis, "fetch").mockReturnValueOnce(pending.promise);
        const clearing = clearCharacters();
        expect(getCharacters()).toEqual([character("saved")]);
        pending.resolve(response({ removed: 1 }));
        await clearing;

        expect(fetch).toHaveBeenCalledWith("/api/characters", {
            method: "DELETE",
            headers: { "X-Fox-Character-Workspace": "workspace" },
        });
        expect(getCharacters()).toEqual([]);
    });

    it("does not erase a loaded roster when refresh returns an invalid response", async () => {
        applyFolderCharacters([character("saved")], "workspace");
        vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(response({ error: "Unavailable" }));

        await expect(fetchCharacters()).rejects.toThrow("Backend did not return the character roster");
        expect(getCharacters()).toEqual([character("saved")]);
    });
});

describe("character workspace identity", () => {
    const mutations = [
        ["add", () => addCharacter({ name_en: "Added" }), { character: character("added") }],
        [
            "update",
            () => updateCharacter("saved", { name_en: "Updated" }),
            { character: character("saved", "Updated") },
        ],
        ["delete", () => deleteCharacter("saved"), {}],
        ["clear", () => clearCharacters(), {}],
        ["import", () => importCharacterRows([{ name_en: "Imported" }]), { characters: [character("imported")] }],
    ] as const;

    it.each(mutations)("sends the loaded workspace identity with %s", async (_name, mutate, body) => {
        applyFolderCharacters([character("saved")], "loaded-workspace");
        const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(response(body));

        await mutate();

        expect(fetch).toHaveBeenCalledTimes(1);
        expect(fetch.mock.calls[0][1]?.headers).toMatchObject({
            "X-Fox-Character-Workspace": "loaded-workspace",
        });
    });

    it.each(mutations)("retains the roster after a stale %s and never replays it", async (_name, mutate) => {
        const saved = character("saved");
        applyFolderCharacters([saved], "old-workspace");
        const fetch = vi
            .spyOn(globalThis, "fetch")
            .mockResolvedValueOnce(
                response({ error: "The folder changed. Reload it before editing characters." }, 409),
            );

        await expect(mutate()).rejects.toThrow("The folder changed. Reload it before editing characters.");
        expect(fetch).toHaveBeenCalledTimes(1);
        expect(getCharacters()).toEqual([saved]);
    });

    it("fetches a workspace identity before an edit made before the first roster read", async () => {
        applyFolderCharacters([]);
        const fetch = vi
            .spyOn(globalThis, "fetch")
            .mockResolvedValueOnce(response({ characters: [], character_workspace: "initial-workspace" }))
            .mockResolvedValueOnce(response({ character: character("added") }, 201));

        await addCharacter({ name_en: "Added" });

        expect(fetch).toHaveBeenCalledTimes(2);
        expect(fetch.mock.calls[0]).toEqual(["/api/characters", { cache: "no-store" }]);
        expect(fetch.mock.calls[1][1]?.headers).toMatchObject({
            "X-Fox-Character-Workspace": "initial-workspace",
        });
        expect(getCharacters()).toEqual([character("added")]);
    });

    it("cancels the edit if a folder loads while its initial identity read is pending", async () => {
        applyFolderCharacters([]);
        const pending = deferred<Response>();
        const fetch = vi.spyOn(globalThis, "fetch").mockReturnValueOnce(pending.promise);
        const editing = addCharacter({ name_en: "Added" });
        applyFolderCharacters([character("new")], "new-workspace");
        pending.resolve(response({ characters: [], character_workspace: "old-workspace" }));

        await expect(editing).rejects.toThrow("The folder changed. Reload it before editing characters.");
        expect(fetch).toHaveBeenCalledTimes(1);
        expect(getCharacters()).toEqual([character("new")]);
    });

    it("does not replace the loaded identity with a stale GET's identity", async () => {
        const pending = deferred<Response>();
        const fetch = vi
            .spyOn(globalThis, "fetch")
            .mockReturnValueOnce(pending.promise)
            .mockResolvedValueOnce(response({ removed: 1 }));
        const reading = fetchCharacters();
        applyFolderCharacters([character("new")], "new-workspace");
        pending.resolve(response({ characters: [], character_workspace: "old-workspace" }));
        await reading;
        await clearCharacters();

        expect(fetch.mock.calls[1][1]?.headers).toEqual({ "X-Fox-Character-Workspace": "new-workspace" });
    });

    it("refuses an unguarded edit when the backend returns no identity", async () => {
        applyFolderCharacters([]);
        const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(response({ characters: [] }));

        await expect(addCharacter({ name_en: "Added" })).rejects.toThrow(
            "The folder changed. Reload it before editing characters.",
        );
        expect(fetch).toHaveBeenCalledTimes(1);
    });

    it("keeps the displayed folder's roster and identity when another window switches folders", async () => {
        const saved = character("saved");
        applyFolderCharacters([saved], "old-workspace");
        const entry = { character_id: saved.meta_id } as Entry;
        state.pageEntriesCache["page.png"] = [entry];
        const fetch = vi
            .spyOn(globalThis, "fetch")
            .mockResolvedValueOnce(
                response({ characters: [character("another-cast")], character_workspace: "new-workspace" }),
            )
            .mockResolvedValueOnce(
                response({ error: "The folder changed. Reload it before editing characters." }, 409),
            );

        await expect(fetchCharacters()).rejects.toThrow("The folder changed. Reload it before editing characters.");
        expect(getCharacters()).toEqual([saved]);
        expect(entry.character_id).toBe(saved.meta_id);
        await expect(clearCharacters()).rejects.toThrow("The folder changed. Reload it before editing characters.");
        expect(fetch.mock.calls[1][1]?.headers).toEqual({ "X-Fox-Character-Workspace": "old-workspace" });
        expect(fetch).toHaveBeenCalledTimes(2);
        expect(getCharacters()).toEqual([saved]);
    });

    it("accepts refreshes within the same loaded workspace", async () => {
        applyFolderCharacters([character("saved")], "workspace");
        vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
            response({
                characters: [character("saved", "Updated")],
                character_workspace: "workspace",
            }),
        );

        expect(await fetchCharacters()).toEqual([character("saved", "Updated")]);
    });
});
