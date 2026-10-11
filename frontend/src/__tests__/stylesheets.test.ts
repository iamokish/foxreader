/// <reference types="node" />
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * Every stylesheet must be reachable from a module the app actually loads.
 *
 * This exists because it was not, and the failure was invisible in every way that
 * mattered. Extracting `reader.css` out of `vulpes.css` left a complete, valid,
 * *unimported* stylesheet: the reader rendered unstyled in both layouts, and the
 * Vulpes header and footer lost every `.vx-icon-btn` rule. Type checks pass, unit
 * tests pass, the build succeeds and the bundle is smaller -- all of which is why
 * nobody noticed until it was looked at.
 *
 * A CSS file that nothing imports is always a mistake: Vite tree-shakes it out of
 * the bundle, so it is dead weight and a missing feature at the same time.
 */

const SRC = resolve(process.cwd(), "src");

/** Every `.css` file under `src/`, as a repo-relative POSIX path. */
function stylesheets(dir: string): string[] {
    const found: string[] = [];
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) {
            found.push(...stylesheets(full));
        } else if (entry.endsWith(".css")) {
            found.push(full);
        }
    }
    return found;
}

/** Every `.ts`/`.tsx` file under `src/`, excluding tests and the stylesheets. */
function modules(dir: string): string[] {
    const found: string[] = [];
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) {
            found.push(...modules(full));
        } else if (/\.(ts|tsx)$/.test(entry) && !/\.test\.(ts|tsx)$/.test(entry)) {
            found.push(full);
        }
    }
    return found;
}

/** The stylesheet paths a module imports, resolved against its own directory. */
function cssImports(file: string): string[] {
    const source = readFileSync(file, "utf8");
    // Anchored at line start so a path quoted inside a comment or a JSDoc block --
    // "see `reader.css`" -- is not mistaken for an import.
    const matches = [...source.matchAll(/^\s*import\s+"([^"]+\.css)";/gm)];
    return matches.map((match) => resolve(file, "..", match[1]));
}

const allSheets = stylesheets(SRC);
const allModules = modules(SRC);

describe("stylesheet reachability", () => {
    const importedBy = new Map<string, string[]>();
    for (const file of allModules) {
        for (const target of cssImports(file)) {
            importedBy.set(target, [...(importedBy.get(target) ?? []), relative(SRC, file)]);
        }
    }

    it("finds stylesheets to check, so the test cannot pass vacuously", () => {
        // A broken glob would make the real assertion below trivially true.
        expect(allSheets.length).toBeGreaterThan(5);
        expect(allModules.length).toBeGreaterThan(20);
    });

    it.each(allSheets.map((sheet) => [relative(SRC, sheet), sheet]))(
        "%s is imported by some module",
        (_name, sheet) => {
            expect(importedBy.get(sheet) ?? []).not.toEqual([]);
        },
    );

    it("imports every stylesheet from a module that is itself reachable", () => {
        // An import from an unreferenced module would still be tree-shaken away.
        // `main.tsx` is the entry, so everything must descend from it.
        //
        // Both `import ... from` and `export ... from` count: the `index.ts` barrels
        // re-export, so a module can be reachable through one without ever being
        // named in an `import` statement.
        const entry = join(SRC, "main.tsx");
        const reachable = new Set<string>([entry]);
        const specifiers = /^\s*(?:import|export)\s+[^;]*?from\s+"([^"]+)";/gm;
        let grew = true;
        while (grew) {
            grew = false;
            for (const file of allModules) {
                if (!reachable.has(file)) continue;
                const source = readFileSync(file, "utf8");
                for (const match of source.matchAll(specifiers)) {
                    const spec = match[1];
                    if (!spec.startsWith(".")) continue;
                    const base = resolve(file, "..", spec);
                    for (const candidate of [
                        `${base}.ts`,
                        `${base}.tsx`,
                        join(base, "index.ts"),
                        join(base, "index.tsx"),
                    ]) {
                        if (allModules.includes(candidate) && !reachable.has(candidate)) {
                            reachable.add(candidate);
                            grew = true;
                        }
                    }
                }
            }
        }

        const orphans = allSheets
            .filter((sheet) => {
                const owners = importedBy.get(sheet) ?? [];
                return owners.length > 0 && !owners.some((owner) => reachable.has(join(SRC, owner)));
            })
            .map((sheet) => relative(SRC, sheet));

        expect(orphans).toEqual([]);
    });
});
