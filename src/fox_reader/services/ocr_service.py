from __future__ import annotations

import logging
import threading
from typing import Any, NamedTuple

import numpy as np
from PIL import Image

from fox_reader import ocr_colors
from fox_reader.constants import DEFAULT_OCR_ENGINE, OCR_ENGINE_PADDLEOCR_VL
from fox_reader.ocr import (
    MultiLangOCR,
    Transcription,
    describe_ocr_engines,
    normalize_ocr_engine,
)
from fox_reader.ocr_vl import PaddleOCRVLEngine

logger = logging.getLogger(__name__)


class Reading(NamedTuple):
    """One OCR result: the text, and the colours the region is drawn in.

    ``colors`` is a hint for the typesetter and the live preview, never a
    requirement -- see :mod:`fox_reader.ocr_colors`. It is
    :data:`fox_reader.ocr_colors.NOTHING` when nothing could be measured, which
    both renderers read as "detect it the old way".
    """

    text: str
    colors: ocr_colors.RegionColors = ocr_colors.NOTHING


class OCRService:
    """Thread-safe wrapper around the configured OCR engine.

    The classic pipeline's detector is always built, whatever the recognition
    engine is: text clean borrows it for ``ppocr`` boxes, and the colour read
    borrows it for the quads PaddleOCR-VL cannot provide (see ``_vl_boxes``).
    Its recognizers load only when classic recognition is selected or actually
    needed: in PaddleOCR-VL mode they stay unloaded (two large models' worth of
    RAM saved) and are loaded on first fallback use.

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

    def _classic_read(self, lang: str, image: Image.Image,
                      grayscale: bool) -> Transcription:
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
        return self._engine.read(lang=lang, input=image, isGrayScaled=grayscale)

    def predict(self, lang: str, image: Image.Image, grayscale: bool) -> str:
        """The recognised text, and nothing else."""
        return self.read(lang=lang, image=image, grayscale=grayscale,
                         colors=False).text

    def read(self, lang: str, image: Image.Image, grayscale: bool, *,
             colors: bool = True, mask: Any = None) -> Reading:
        """Transcribe ``image``, and measure its colours while it is in hand.

        ``mask`` restricts the colour read to the part of ``image`` that belongs
        to the region -- a freeform selection arrives as a polygon inside a
        white-filled rectangle, and the filler is not the plate. The read itself
        is pixel work on one crop, outside any model; the text quads it samples
        inside come free from the classic detector's own pass, or from borrowing
        that detector when PaddleOCR-VL did the reading (``_vl_boxes``).
        ``colors=False`` skips both for callers that only want text.

        The colour step can never fail a transcription: everything it does is
        wrapped in :func:`fox_reader.ocr_colors.estimate`, which answers
        "nothing" rather than raising.
        """
        with self._lock:
            transcription = self._transcribe(lang, image, grayscale,
                                             want_boxes=colors)
        if not colors:
            return Reading(transcription.text)
        # Outside the lock on purpose: this is numpy and OpenCV over one crop,
        # it touches no model, and holding the engine lock through it would
        # serialise it against every other OCR request for nothing.
        return Reading(transcription.text,
                       ocr_colors.estimate(image, mask=mask,
                                           polys=transcription.boxes or None))

    def _vl_boxes(self, image: Image.Image) -> tuple[Any, ...]:
        """Text quads for a crop the VL engine read. Call with the lock held.

        PaddleOCR-VL transcribes a crop end to end and exposes no boxes, so the
        classic detector is borrowed for them -- the same instance text clean
        borrows, already loaded at startup even in VL mode (see the class
        docstring), so this costs one forward pass on a crop and no extra RAM.
        Worth it: the quads are what let the colour read sample the plate from
        the paper immediately around the lettering instead of from the whole
        crop, which is the difference between measuring a bubble and measuring
        the panel it sits in.

        Measured on the *unprocessed* crop, not the grayscale-thresholded one
        the classic recognition path builds: binarising at a fixed level erases
        light-on-dark lettering entirely, and all that is wanted here is where
        the text is. The PIL crop goes straight through, which is what the
        detector's own preprocessing wants -- converting it to an array first
        would only have it rebuilt on the other side. Never raises -- no
        detector, or a detector that fails, means an empty tuple and a
        whole-crop colour read, which is exactly what this did before the
        detector was borrowed.
        """
        detector = self.detector
        if detector is None:
            return ()
        try:
            from fox_reader.clean import ppocr

            polys, _ = ppocr.boxes_rgb(detector, image)
        except Exception as exc:  # noqa: BLE001 -- a hint, never worth an error
            logger.debug("Could not borrow the detector for colour sampling: %s", exc)
            return ()
        # A tuple of per-quad arrays, not the (N, 4, 2) array itself: callers
        # test it for emptiness, and an ndarray has no truth value.
        return tuple(polys)

    def _transcribe(self, lang: str, image: Image.Image, grayscale: bool, *,
                    want_boxes: bool = False) -> Transcription:
        """The engine half of :meth:`read`. Call with the lock held."""
        if self._requested_engine == OCR_ENGINE_PADDLEOCR_VL:
            try:
                vl = self._ensure_vl()
            except RuntimeError as exc:
                # Fallback, not failure: classic is always available, and
                # OCR must never 500 because an optional engine is broken.
                logger.warning("%s -- falling back to classic PaddleOCR", exc)
                self._active_engine = "paddleocr"
                return self._classic_read(lang=lang, image=image, grayscale=grayscale)
            try:
                text = vl.predict(input=image, isGrayScaled=grayscale, lang=lang)
            except Exception as exc:
                logger.warning("PaddleOCR-VL failed (%s) -- falling back to classic", exc)
                self._active_engine = "paddleocr"
                try:
                    return self._classic_read(lang=lang, image=image, grayscale=grayscale)
                except Exception:
                    # If even classic fails, surface the VL error context
                    # only when classic gave nothing to add.
                    raise
            self._active_engine = OCR_ENGINE_PADDLEOCR_VL
            # The VL engine exposes no boxes of its own; the classic detector is
            # borrowed for them, and only when someone is going to use them.
            return Transcription(text, self._vl_boxes(image) if want_boxes else ())
        self._active_engine = "paddleocr"
        return self._engine.read(lang=lang, input=image, isGrayScaled=grayscale)

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
