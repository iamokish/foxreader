from __future__ import annotations

import asyncio
import time

from fox_reader.config import FoxConfig
from fox_reader.models.responses import TranslationResult
from fox_reader.settings import SettingsManager
from fox_reader.user_endpoints import UserEndpointStore
from fox_reader.translate.local_mtl import LocalMTLManager
from fox_reader.utils import PROJECT_ROOT


class TranslateService:
    """Unified interface to all translation backends."""

    def __init__(
        self,
        config: FoxConfig,
        settings: SettingsManager | None = None,
        user_endpoints: UserEndpointStore | None = None,
    ) -> None:
        self._config = config
        self._settings = settings
        self._user_endpoints = user_endpoints
        self._mtl = LocalMTLManager(
            config.resolve_mtl_dir(PROJECT_ROOT),
            settings=settings,
        )

    def _api_token(self, name: str) -> str:
        """A stored API token, or "" when none is configured.

        Tokens are validated on save, so a non-empty value here means "valid
        when it was saved". Live re-validation on every translation would add
        a ping/usage round-trip to each button press, so availability gating
        uses presence; an actually-revoked key surfaces as a 403 from the
        translate call itself.
        """
        if self._settings is None:
            return ""
        try:
            tokens = self._settings.get_api_tokens()
        except Exception:
            return ""
        return (tokens.get(name, "") or "").strip()

    def deepl_token(self) -> str:
        return self._api_token("deepl_api_token")

    def jpdb_token(self) -> str:
        return self._api_token("jpdb_api_token")

    @property
    def deepl_available(self) -> bool:
        return bool(self.deepl_token())

    @property
    def jpdb_available(self) -> bool:
        return bool(self.jpdb_token())

    def translate_status(self) -> dict[str, bool]:
        """Which token-gated translators may be offered. No tokens leak."""
        return {"deepl": self.deepl_available, "jpdb": self.jpdb_available}

    async def deepl(self, text: str, src_lang: str) -> TranslationResult:
        from fox_reader.translate.deepl_api import translate_by_deepl

        return await translate_by_deepl(
            src_lang, "EN", text, api_token=self.deepl_token()
        )

    async def jpdb(self, text: str, src_lang: str) -> TranslationResult:
        from fox_reader.translate.jpdb_api import translate_by_jpdb

        return await translate_by_jpdb(
            src_lang, "english", text, api_token=self.jpdb_token()
        )

    async def custom(
        self,
        text: str,
        selector: str,
        language: str | None = None,
        context: object = None,
        character_info: object = None,
        context_character_links: object = None,
    ) -> TranslationResult:
        """Translate through a user-defined endpoint.

        `selector` is an endpoint id (what the translator buttons send). It
        also accepts a language name, in which case the first active endpoint
        for that language is used. `context` is the earlier [[source,
        english], ...] pairs; it only reaches the wire when the endpoint's
        request schema carries a Context node. `character_info` (the roster
        snapshot) and `context_character_links` (one meta_id or None per
        pair) behave the same way for the Character Info and Context Links
        nodes.
        """
        selector = (selector or "").strip()
        language = (language or "").strip().lower()

        def failure(code: int, message: str) -> TranslationResult:
            return TranslationResult(
                code=code,
                id=int(time.time() * 1000),
                data="",
                source_lang=(language or selector).upper(),
                target_lang="EN",
                method="Custom",
                message=message,
            )

        if self._user_endpoints is None:
            return failure(503, "User endpoints are not configured")

        endpoint = self._user_endpoints.get(selector)

        if endpoint is None:
            # Fall back to treating the selector as a language name.
            active = self._user_endpoints.active_for_language(
                language or selector
            )

            if not active:
                return failure(
                    404,
                    "No active custom endpoint for "
                    f"'{language or selector}'",
                )

            endpoint = active[0]

        if language not in endpoint.languages:
            language = endpoint.languages[0]

        from fox_reader.translate.custom_endpoint import translate_by_custom_endpoint

        return await translate_by_custom_endpoint(
            endpoint,
            text=text,
            source_lang=language,
            context=context,
            character_info=character_info,
            context_character_links=context_character_links,
        )

    def ml(
        self,
        text: str,
        src_lang: str,
        context: object = None,
        character_info: object = None,
        context_character_links: object = None,
        meta_id: object = None,
    ) -> TranslationResult:
        if self._mtl is None:
            raise RuntimeError("MTL is disabled")

        translated = self._mtl.translate(
            lang=src_lang,
            text=text,
            context=context,
            character_info=character_info,
            context_character_links=context_character_links,
            meta_id=meta_id,
        )
        return TranslationResult(
            code=200,
            id=int(time.time() * 1000),
            data=translated,
            source_lang=src_lang.upper(),
            target_lang="EN",
            method="LocalMTL",
        )

    async def ml_async(
        self,
        text: str,
        src_lang: str,
        context: object = None,
        character_info: object = None,
        context_character_links: object = None,
        meta_id: object = None,
    ) -> TranslationResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.ml(
                text,
                src_lang,
                context,
                character_info,
                context_character_links,
                meta_id,
            ),
        )

    def ml_load(self, lang: str) -> None:
        if self._mtl is None:
            return
        self._mtl.load(lang)

    def ml_unload(self) -> None:
        if self._mtl is None:
            return
        self._mtl.unload()

    def ml_status(self) -> dict:
        """Which local model (if any) is currently loaded.

        Never raises: a missing manager or a half-built model reads as
        "nothing loaded", which is what the reader page falls back to anyway.
        """
        empty: dict[str, object] = {"loaded": False, "lang": None, "model_id": None, "languages": []}

        if self._mtl is None:
            return dict(empty)

        try:
            status = self._mtl.status()
        except Exception:
            return dict(empty)

        if not isinstance(status, dict):
            return dict(empty)

        return status

    def ml_memory(self) -> dict:
        """Whether every configured local model fits on its device.

        Answered without loading anything, so the reader page can warn before
        the user flips the MTL switch and waits.
        """
        if self._mtl is None:
            return {
                "enabled": False,
                "device": "",
                "ok": True,
                "checks": [],
                "issues": [],
            }

        return self._mtl.memory_report()
