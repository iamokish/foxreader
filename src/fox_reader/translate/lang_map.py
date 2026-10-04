from fastapi import HTTPException


# =============================================================================
# DeepL language mappings (official API codes, uppercase)
#
# The official DeepL API is case-insensitive, but we always send the canonical
# uppercase form (e.g. "JA", "EN-US", "ZH-HANS") and accept the lowercase
# legacy values the scraper used, plus Fox Reader's own language names.
# =============================================================================

DEEPL_TARGET_LANGS = {
    "AR": "AR",
    "BG": "BG",
    "CS": "CS",
    "DA": "DA",
    "DE": "DE",
    "EL": "EL",
    "EN-GB": "EN-GB",
    "EN-US": "EN-US",
    "ES": "ES",
    "ES-419": "ES-419",
    "ET": "ET",
    "FI": "FI",
    "FR": "FR",
    "HE": "HE",
    "HU": "HU",
    "ID": "ID",
    "IT": "IT",
    "JA": "JA",
    "KO": "KO",
    "LT": "LT",
    "LV": "LV",
    "NB": "NB",
    "NL": "NL",
    "PL": "PL",
    "PT-BR": "PT-BR",
    "PT-PT": "PT-PT",
    "RO": "RO",
    "RU": "RU",
    "SK": "SK",
    "SL": "SL",
    "SV": "SV",
    "TR": "TR",
    "UK": "UK",
    "VI": "VI",
    "ZH": "ZH",
    "ZH-HANS": "ZH-HANS",
    "ZH-HANT": "ZH-HANT",
    # Convenient defaults: "EN" means US English, "PT" means Brazilian.
    "EN": "EN-US",
    "PT": "PT-BR",
    # Fox Reader language names and common aliases.
    "ENGLISH": "EN-US",
    "JAPANESE": "JA",
    "CHINESE": "ZH",
    "KOREAN": "KO",
    "JP": "JA",
    "CN": "ZH",
    "KR": "KO",
}

DEEPL_SOURCE_LANGS = {
    **DEEPL_TARGET_LANGS,
    # Source codes have no region: "EN", not "EN-US".
    "EN": "EN",
    "EN-GB": "EN",
    "EN-US": "EN",
    "PT": "PT",
    "PT-BR": "PT",
    "PT-PT": "PT",
    "ENGLISH": "EN",
    "ZH-HANS": "ZH",
    "ZH-HANT": "ZH",
}


# =============================================================================
# Generic resolver
# =============================================================================

def resolve_lang(code: str, lang_map: dict, field: str, allow_auto: bool = False) -> str:
    """Resolve a language code against a map. Raises HTTPException on failure."""
    if not code:
        raise HTTPException(status_code=400, detail=f"{field} is required")

    if code.lower() == "auto":
        if allow_auto:
            return lang_map.get("AUTO", "auto")
        supported = ", ".join(sorted(k for k in lang_map if k != "AUTO"))
        raise HTTPException(
            status_code=400,
            detail=f'{field} cannot be "auto"; pick one of: {supported}',
        )

    normalized = code.upper()
    if normalized in lang_map:
        return lang_map[normalized]

    supported = ", ".join(sorted(lang_map.keys()))
    raise HTTPException(
        status_code=400,
        detail=f"unsupported {field} '{code}'; valid codes: {supported}",
    )


def get_supported_langs(lang_map: dict) -> str:
    """Return a comma-separated string of supported language codes."""
    return ", ".join(sorted(lang_map.keys()))


# =============================================================================
# Fox Reader language helpers
#
# The reader UI works with three language groups ("japanese", "chinese",
# "korean") plus the short codes the translator buttons send ("JA", "ZH",
# "KO"). Both spellings arrive at the translate routes, so the official API
# clients accept either and normalise to the DeepL source code.
# =============================================================================

#: Fox language group -> official DeepL source code.
FOX_TO_DEEPL_SOURCE = {
    "japanese": "JA",
    "chinese": "ZH",
    "korean": "KO",
}

#: Every spelling the UI may send for a Fox language group.
FOX_LANGUAGE_ALIASES = {
    "japanese": "japanese",
    "ja": "japanese",
    "jp": "japanese",
    "chinese": "chinese",
    "zh": "chinese",
    "cn": "chinese",
    "korean": "korean",
    "ko": "korean",
    "kr": "korean",
}


def normalize_fox_language(value: str | None) -> str:
    """Normalise a UI language to "japanese" | "chinese" | "korean"."""
    if not value:
        raise ValueError("language is required")
    key = value.strip().lower()
    if key in FOX_LANGUAGE_ALIASES:
        return FOX_LANGUAGE_ALIASES[key]
    # Also accept the DeepL codes directly ("JA", "ZH", "KO").
    upper = value.strip().upper()
    if upper in ("JA", "JAPANESE", "JP"):
        return "japanese"
    if upper in ("ZH", "ZH-HANS", "ZH-HANT", "CHINESE", "CN"):
        return "chinese"
    if upper in ("KO", "KOREAN", "KR"):
        return "korean"
    raise ValueError(
        f"unsupported language '{value}'; expected one of: japanese, chinese, korean"
    )


def deepl_source_for_fox_language(value: str | None) -> str:
    """Map a Fox language group to the official DeepL source code."""
    return FOX_TO_DEEPL_SOURCE[normalize_fox_language(value)]
