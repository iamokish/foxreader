import { useEffect, useState } from "preact/hooks";
import { getActiveCustomEndpoints } from "../../api";
import { emit } from "../../eventbus";
import type { ActiveCustomEndpoint } from "../../core/types";

/** Active user endpoints, grouped by the language they were activated for. */
type CustomGroups = Record<string, ActiveCustomEndpoint[]>;

export function TranslatorButtons() {
    const [custom, setCustom] = useState<CustomGroups>({});
    // Token-gated translators. `null` means "status not loaded yet": render
    // the buttons (previous behaviour) until the backend answers, then hide
    // what has no valid token. A failed fetch keeps the old behaviour so a
    // missing endpoint never breaks the reader.
    const [translateStatus, setTranslateStatus] = useState<{ deepl: boolean; jpdb: boolean } | null>(null);

    useEffect(() => {
        let cancelled = false;

        getActiveCustomEndpoints()
            .then((data) => {
                if (!cancelled) setCustom(data.active ?? {});
            })
            .catch(() => {
                /* the page still works with the built-in engines only */
            });

        fetch("/api/translate/status", { cache: "no-store" })
            .then((res) => (res.ok ? res.json() : null))
            .then((data) => {
                if (cancelled || !data) return;
                setTranslateStatus({
                    deepl: !!data.deepl,
                    jpdb: !!data.jpdb,
                });
            })
            .catch(() => {
                /* keep previous behaviour when the status is unreachable */
            });

        // Immediate pass from the server-injected flags so unavailable
        // buttons never flash before the fetch returns.
        try {
            const config = (window as unknown as { __FOX_CONFIG__?: { deeplAvailable?: boolean; jpdbAvailable?: boolean } })
                .__FOX_CONFIG__;
            if (config && (config.deeplAvailable !== undefined || config.jpdbAvailable !== undefined)) {
                setTranslateStatus({
                    deepl: !!config.deeplAvailable,
                    jpdb: !!config.jpdbAvailable,
                });
            }
        } catch {
            /* workers/tests without window */
        }

        return () => {
            cancelled = true;
        };
    }, []);

    // The language filter is applied imperatively in translate.ts, so it has
    // to run again once these buttons exist.
    useEffect(() => {
        emit("translators:changed");
    }, [custom, translateStatus]);

    const showDeepL = translateStatus === null || translateStatus.deepl;
    const showJpdb = translateStatus === null || translateStatus.jpdb;

    return (
        <div class="translator-buttons">
            {showDeepL && (
                <button data-lang-group="japanese" data-endpoint="/translate/deepl" data-lang="JA">
                    DeepL
                </button>
            )}
            {showJpdb && (
                <button data-lang-group="japanese" data-endpoint="/translate/jpdb" data-lang="japanese">
                    JPDB
                </button>
            )}

            {showDeepL && (
                <button data-lang-group="chinese" data-endpoint="/translate/deepl" data-lang="ZH">
                    DeepL
                </button>
            )}
            {showJpdb && (
                <button data-lang-group="chinese" data-endpoint="/translate/jpdb" data-lang="chinese">
                    JPDB
                </button>
            )}

            {showDeepL && (
                <button data-lang-group="korean" data-endpoint="/translate/deepl" data-lang="KO">
                    DeepL
                </button>
            )}
            {showJpdb && (
                <button data-lang-group="korean" data-endpoint="/translate/jpdb" data-lang="korean">
                    JPDB
                </button>
            )}

            <button data-lang-group="japanese" data-endpoint="/translate/ml" data-lang="japanese" data-requires-ml="true">
                MTL
            </button>
            <button data-lang-group="chinese" data-endpoint="/translate/ml" data-lang="chinese" data-requires-ml="true">
                MTL
            </button>
            <button data-lang-group="korean" data-endpoint="/translate/ml" data-lang="korean" data-requires-ml="true">
                MTL
            </button>

            {Object.entries(custom).flatMap(([language, endpoints]) =>
                endpoints.map((endpoint) => (
                    <button
                        key={`${language}:${endpoint.id}`}
                        data-lang-group={language}
                        data-endpoint="/translate/custom"
                        data-lang={endpoint.id}
                    >
                        {endpoint.name}
                    </button>
                )),
            )}
        </div>
    );
}
