from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from PIL import Image

from fox_reader.constants import DEFAULT_OCR_ENGINE, OCR_ENGINE_PADDLEOCR_VL
from fox_reader.ocr import MultiLangOCR, describe_ocr_engines, normalize_ocr_engine
from fox_reader.ocr_vl import PaddleOCRVLEngine

logger = logging.getLogger(__name__)


class OCRService:
    """Thread-safe wrapper around the configured OCR engine.

    The classic pipeline's detector is always built: text clean borrows it
    for ``ppocr`` boxes whatever the recognition engine is. Its recognizers
    load only when classic recognition is selected or actually needed: in
    PaddleOCR-VL mode they stay unloaded (two large models' worth of RAM
    saved) and are loaded on first fallback use.

    When PaddleOCR-VL is selected it loads eagerly here at startup, never on
    first predict: the first OCR request must not pay a gigabyte-scale model
    load. A selected-but-broken VL (missing package or weights) still falls
    back to classic with a warning instead of failing startup, and from then
    on the service behaves exactly like classic mode.
    """

    def __init__(self, settings: Any | None = None, engine: str | None = None) -> None:
        if engine is not None:
            requested = normalize_ocr_engine(engine)
        elif settings is not None:
            try:
                requested = normalize_ocr_engine(settings.get_ocr_engine())
            except Exception as exc:
                logger.warning("Could not read the OCR engine setting: %s -- using classic", exc)
                requested = DEFAULT_OCR_ENGINE
        else:
            requested = DEFAULT_OCR_ENGINE
        self._requested_engine = requested
        self._settings = settings
        want_recognizers = requested != OCR_ENGINE_PADDLEOCR_VL
        self._engine = MultiLangOCR(recognizers=want_recognizers)
        if not want_recognizers:
            logger.info(
                "Classic OCR recognizers deferred (PaddleOCR-VL selected): "
                "detector loaded for text clean, recognition loads on fallback use."
            )
        self._vl: Any | None = None
        self._vl_error: str | None = None
        self._active_engine = "paddleocr"
        self._lock = threading.RLock()
        if requested == OCR_ENGINE_PADDLEOCR_VL:
            self._load_vl_eagerly()

    # ------------------------------------------------------------- selection

    @property
    def requested_engine(self) -> str:
        return self._requested_engine

    @property
    def active_engine(self) -> str:
        with self._lock:
            return self._active_engine

    def engine_info(self) -> dict[str, Any]:
        """What the settings page renders. Never raises."""
        try:
            payload = describe_ocr_engines(self._requested_engine)
        except Exception as exc:
            logger.debug("Could not describe OCR engines: %s", exc)
            payload = describe_ocr_engines(DEFAULT_OCR_ENGINE)
        try:
            with self._lock:
                payload["active"] = self._active_engine
                if self._vl_error:
                    payload["last_error"] = self._vl_error
        except Exception:
            pass
        # Restart semantics: the engine binds at construction, like devices.
        payload["restart_required"] = True
        return payload

    def _load_vl_eagerly(self) -> None:
        """Load the selected VL engine now, falling back to classic on failure.

        Runs once from ``__init__`` so the first predict never pays the model
        load. Never raises: a broken VL degrades to fully-classic operation
        (recognizers included, so later predicts take no lazy-load detour),
        with the cause kept in ``_vl_error`` for the settings page. Startup
        itself must never fail because an optional engine is broken.
        """
        try:
            self._ensure_vl()
        except Exception as exc:  # noqa: BLE001 -- fallback, not failure; see above
            logger.warning("%s -- falling back to classic PaddleOCR", exc)
            try:
                self._engine.ensure_recognizers()
            except Exception as rec_exc:  # noqa: BLE001 -- reported at predict time
                logger.warning("Classic OCR recognizers could not load either: %s", rec_exc)
            return
        with self._lock:
            self._active_engine = OCR_ENGINE_PADDLEOCR_VL

    def _ensure_vl(self) -> Any:
        """The VL engine, loading it once. Raises RuntimeError when unusable."""
        with self._lock:
            if self._vl is not None:
                return self._vl
            if self._vl_error is not None:
                raise RuntimeError(self._vl_error)
            try:
                if self._requested_engine != OCR_ENGINE_PADDLEOCR_VL:
                    raise RuntimeError("VL engine is not selected.")
                vl = PaddleOCRVLEngine()
                vl.ensure_loaded()
            except RuntimeError as exc:
                self._vl_error = str(exc)
                raise
            except Exception as exc:
                self._vl_error = f"PaddleOCR-VL could not start: {exc}"
                raise RuntimeError(self._vl_error) from exc
            self._vl = vl
            self._vl_error = None
            return vl

    def _classic_predict(self, lang: str, image: Image.Image, grayscale: bool) -> str:
        """Classic recognition, loading deferred recognizers first if needed.

        In PaddleOCR-VL mode the recognizers were skipped at startup, so the
        first fallback pays their load here, inside the service lock. A load
        failure is reported plainly: at that point both engines are unusable
        and hiding the cause would only complicate the bug report.
        """
        try:
            self._engine.ensure_recognizers()
        except Exception as exc:
            raise RuntimeError(f"Classic OCR recognizers could not load: {exc}") from exc
        return self._engine.predict(lang=lang, input=image, isGrayScaled=grayscale)

    def predict(self, lang: str, image: Image.Image, grayscale: bool) -> str:
        with self._lock:
            if self._requested_engine == OCR_ENGINE_PADDLEOCR_VL:
                try:
                    vl = self._ensure_vl()
                except RuntimeError as exc:
                    # Fallback, not failure: classic is always available, and
                    # OCR must never 500 because an optional engine is broken.
                    logger.warning("%s -- falling back to classic PaddleOCR", exc)
                    self._active_engine = "paddleocr"
                    return self._classic_predict(lang=lang, image=image, grayscale=grayscale)
                try:
                    text = vl.predict(input=image, isGrayScaled=grayscale, lang=lang)
                except Exception as exc:
                    logger.warning("PaddleOCR-VL failed (%s) -- falling back to classic", exc)
                    self._active_engine = "paddleocr"
                    try:
                        return self._classic_predict(lang=lang, image=image, grayscale=grayscale)
                    except Exception:
                        # If even classic fails, surface the VL error context
                        # only when classic gave nothing to add.
                        raise
                self._active_engine = OCR_ENGINE_PADDLEOCR_VL
                return text
            self._active_engine = "paddleocr"
            return self._engine.predict(lang=lang, input=image, isGrayScaled=grayscale)

    def unload_vl(self) -> None:
        """Release VL weights (e.g. before process shutdown). Never raises."""
        with self._lock:
            vl, self._vl = self._vl, None
            self._active_engine = "paddleocr"
        if vl is not None:
            try:
                vl.unload()
            except Exception as exc:
                logger.debug("VL unload failed: %s", exc)

    # ------------------------------------------------------------------ clean

    @property
    def detector(self) -> Any | None:
        """The PP-OCRv6 text *detector*, borrowed for text cleaning.

        Always the classic detector, whatever the recognition engine is:
        VL transcribes crops end-to-end and exposes no boxes, so clean keeps
        reading them from here. Reached through the service rather than from
        the engine directly so that `fox_reader.ocr` stays untouched and
        every caller goes through the lock below.
        """
        try:
            return self._engine._engines["default"].detector
        except (AttributeError, KeyError, TypeError):
            return None

    def detect_text_boxes(self, bgr: np.ndarray, **kw: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(polys, scores)`` for a BGR page. Serialised with OCR itself.

        One detector instance, one lock: a clean started while an OCR request is
        mid-flight waits rather than reentering the model.
        """
        from fox_reader.clean import ppocr

        with self._lock:
            return ppocr.boxes(self.detector, bgr, **kw)
