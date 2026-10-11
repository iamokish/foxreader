"""PP-OCR detection and recognition, straight on torch.

Replaces ``paddleocr.TextDetection`` / ``paddleocr.TextRecognition``, which
reached the same weights through ``paddlex.create_predictor`` and
``transformers.AutoModelForObjectDetection``. The model code now lives in
:mod:`fox_reader.hf`; what is left for this module is the settings those three
layers used to resolve, and the little bit of plumbing
:mod:`fox_reader.ocr` and :mod:`fox_reader.clean.ppocr` expect.

Settings, and why they are not simply the JSON's
------------------------------------------------
``preprocessor_config.json`` and the predictor disagree, and the predictor
won. PaddleX keeps a list of detector checkpoints whose resize behaviour its
own config files get wrong (``_TEXT_DET_MAX_LIMIT_MODELS``) and overrides them
to ``960``/``max``; PP-OCRv6_medium_det is on that list, so the 736/``min`` in
its ``preprocessor_config.json`` was never what ran. The thresholds come from
a third file, ``inference.yml``.

All of that is reproduced here rather than tidied up, because tidying it would
change what the detector returns. :data:`_TEXT_DET_MAX_LIMIT_MODELS` and the
default triples below are the paddlex values, read off a live predictor before
it was removed.

One deliberate divergence: ``max_candidates``. PaddleX's transformers engine
never forwarded it to the image processor, so the ``max_candidates: 3000`` in
PP-OCRv6_medium_det's ``inference.yml`` was dead config and the processor's own
default of 1000 applied. It is read here, because the model ships the number
and dropping it looks like an oversight. On any page with fewer than 1000
connected components -- which measured 20-51 on text pages, screentoned or
speckled -- the two are identical; see :data:`_DEFAULT_MAX_CANDIDATES`.

Interfaces
----------
``predict`` answers with a one-element list of dicts, which is the shape
:meth:`fox_reader.ocr.PaddleOCRWrapper._extract_boxes` and
:func:`fox_reader.clean.ppocr.boxes` already read -- the latter defensively,
accepting either a dict or an attribute-bearing object. Dicts are the simpler
half of that contract, so dicts are what it gets.

One image per call. Batching the crops of a page would be faster, but the
recogniser's image processor sizes a batch from its *widest* member
(``get_target_size``), so a short crop batched with a long line is stretched to
the long line's aspect ratio and transcribes differently. That is a change to
output, not just to speed.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

logger = logging.getLogger(__name__)

#: Side of ``inference.yml`` in a model directory.
INFERENCE_CONFIG_NAME = "inference.yml"

#: Detector checkpoints that resize by the *longest* side to 960, overriding
#: whatever their own ``preprocessor_config.json`` says. Carried over verbatim
#: from ``paddlex.inference.models.text_detection.predictor``; PP-OCRv6_medium_det
#: is in it, which is why the live predictor read 960/max and not 736/min.
_TEXT_DET_MAX_LIMIT_MODELS = frozenset({
    "PP-OCRv5_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
    "PP-OCRv3_server_det",
    "PP-OCRv3_mobile_det",
    "PP-OCRv6_medium_det",
    "PP-OCRv6_small_det",
    "PP-OCRv6_tiny_det",
})

#: Resize defaults for a checkpoint on, and off, the list above.
_RESIZE_ON_LIST = (960, "max")
_RESIZE_OFF_LIST = (736, "min")

#: DB post-processing defaults, used only when ``inference.yml`` is silent.
#: PP-OCRv6_medium_det's own file supplies 0.2 / 0.45 / 1.4 / 3000.
_DEFAULT_THRESH = 0.3
_DEFAULT_BOX_THRESH = 0.6
_DEFAULT_UNCLIP_RATIO = 2.0

#: Fallback cap on contours considered per page, when ``inference.yml`` names
#: none. 1000 is the image processor's own default, and the value paddleocr
#: effectively used: its ``postprocess`` never forwarded ``max_candidates``, so
#: the 3000 in ``inference.yml`` was dead config on the transformers engine.
#: Reading it now is a deliberate change -- the model ships the number, and
#: PaddleX dropping it looks like an oversight rather than a decision.
#:
#: The cap counts raw connected components in the DB probability map *before*
#: `min_size` and `box_thresh` filtering, so noise blobs spend the budget too.
#: It is reached at roughly a 26x38 grid of separate marks on a page, and that
#: is resolution-independent: `limit_side_len`/`limit_type: max` caps the map's
#: long side at 960, so a 4000px scan and a 900px one produce the same
#: 672x960 map. Measured on text pages with heavy screentone and 8% speckle:
#: 20-51 contours, 20x or more of headroom. A kanji chart or font specimen
#: sheet is the kind of page that actually reaches it.
_DEFAULT_MAX_CANDIDATES = 1000
_MIN_SIZE = 3
_MAX_SIDE_LIMIT = 4000


class OCRModelUnavailable(RuntimeError):
    """A model cannot be loaded here, in words the settings page can show."""


def torch_device(device: str | None) -> str:
    """A torch device string from whatever the device module handed over.

    ``FOX_DEVICE.paddleocr`` is still spelled PaddleOCR's way (``gpu:0`` /
    ``cpu``) because that slot fed Paddle for years. Translating here rather
    than changing `fox_reader.device` keeps one concern in one place: the
    saved settings, the settings page and the device probe all keep their
    current meaning, and only the consumer moves.

    (``device.py`` maps Apple GPUs to the CPU for this slot, on the grounds
    that Paddle has no Metal backend. Pure torch does. Lifting that is a
    settings-visible change, so it is left alone here.)
    """
    text = str(device or "cpu").strip().lower() or "cpu"
    if text in {"gpu", "cuda"}:
        return "cuda:0"
    if text.startswith("gpu:"):
        tail = text[4:]
        return f"cuda:{int(tail)}" if tail.isdigit() else "cuda:0"
    return text


def _read_inference_config(model_dir: Path) -> dict[str, Any]:
    """``inference.yml`` as a dict; ``{}`` when it is absent or unreadable.

    Tolerant on purpose. The file carries the thresholds, so losing it means
    falling back to the image processor's own defaults -- a worse detector,
    but a running one. A checkpoint that ships without it should not stop the
    app from starting.
    """
    path = model_dir / INFERENCE_CONFIG_NAME
    if not path.is_file():
        logger.debug("%s has no %s; using built-in defaults", model_dir, INFERENCE_CONFIG_NAME)
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        logger.warning("%s could not be read (%s); using built-in defaults", path, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _config_model_name(config: dict[str, Any]) -> str | None:
    name = config.get("Global", {}).get("model_name") if isinstance(config.get("Global"), dict) else None
    return name if isinstance(name, str) and name else None


def _det_resize_defaults(model_name: str, config: dict[str, Any]) -> tuple[int, str]:
    """``(limit_side_len, limit_type)`` the way paddlex resolved them."""
    resize_cfg: dict[str, Any] = {}
    pre = config.get("PreProcess")
    if isinstance(pre, dict):
        for op in pre.get("transform_ops") or []:
            if isinstance(op, dict) and isinstance(op.get("DetResizeForTest"), dict):
                resize_cfg = op["DetResizeForTest"]
                break
    fallback = _RESIZE_ON_LIST if model_name in _TEXT_DET_MAX_LIMIT_MODELS else _RESIZE_OFF_LIST
    return (
        int(resize_cfg.get("resize_long", fallback[0])),
        str(resize_cfg.get("limit_type", fallback[1])),
    )


def _det_postprocess_defaults(config: dict[str, Any]) -> tuple[float, float, float, int]:
    post = config.get("PostProcess")
    post = post if isinstance(post, dict) else {}
    return (
        float(post.get("thresh", _DEFAULT_THRESH)),
        float(post.get("box_thresh", _DEFAULT_BOX_THRESH)),
        float(post.get("unclip_ratio", _DEFAULT_UNCLIP_RATIO)),
        int(post.get("max_candidates", _DEFAULT_MAX_CANDIDATES)),
    )


def _as_pil(image: Image.Image | np.ndarray) -> Image.Image:
    """An RGB PIL image, from either of the two things callers pass.

    `fox_reader.ocr` hands over PIL crops; `fox_reader.clean.ppocr` hands over
    an RGB ndarray it built with OpenCV.
    """
    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[2] == 4:
        return Image.fromarray(array[:, :, :3])
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array)
    raise ValueError(f"Cannot read an image of shape {array.shape}")


class _Predictor:
    """Shared loading for the two predictors."""

    def __init__(self, model_dir: str | Path, device: str | None = None, model_name: str | None = None):
        from fox_reader.hf.models.auto import AutoImageProcessor

        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise OCRModelUnavailable(
                f"the OCR model directory {self.model_dir.name} is missing; "
                "download the models again from the setup page"
            )
        self.device = torch_device(device)
        self.inference_config = _read_inference_config(self.model_dir)
        # The checkpoint names itself; a caller's override wins, which is what
        # keeps `korean_PP-OCRv5_mobile_rec` working through `ocr.py`'s
        # existing `config_model_name` plumbing.
        self.model_name = model_name or _config_model_name(self.inference_config) or ""

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_dir)
            self.model = self._load_model()
        except OSError as exc:
            raise OCRModelUnavailable(
                f"the OCR model in {self.model_dir.name} could not be loaded ({exc})"
            ) from exc

        # One model, one lock: torch modules are not re-entrant across threads
        # and `OCRService` already serialises, but a direct caller should not
        # be able to interleave two forwards through the same weights.
        self._lock = threading.Lock()

    def _load_model(self):
        raise NotImplementedError

    def close(self) -> None:
        """Present because paddleocr's predictors had it. Nothing to release."""


class TextDetection(_Predictor):
    """DB text detection. ``predict`` answers with quadrilaterals."""

    #: Overridable per call, matching what `clean.ppocr` passes through.
    _PREDICT_KEYS = ("limit_side_len", "limit_type", "thresh", "box_thresh", "unclip_ratio")

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        model_name: str | None = None,
        limit_side_len: int | None = None,
        limit_type: str | None = None,
        thresh: float | None = None,
        box_thresh: float | None = None,
        unclip_ratio: float | None = None,
        max_candidates: int | None = None,
        max_side_limit: int | None = None,
    ):
        super().__init__(model_dir, device=device, model_name=model_name)

        default_side, default_type = _det_resize_defaults(self.model_name, self.inference_config)
        default_thresh, default_box, default_unclip, default_candidates = _det_postprocess_defaults(
            self.inference_config
        )

        self.limit_side_len = default_side if limit_side_len is None else int(limit_side_len)
        self.limit_type = default_type if limit_type is None else str(limit_type)
        self.thresh = default_thresh if thresh is None else float(thresh)
        self.box_thresh = default_box if box_thresh is None else float(box_thresh)
        self.unclip_ratio = default_unclip if unclip_ratio is None else float(unclip_ratio)
        self.max_candidates = default_candidates if max_candidates is None else int(max_candidates)
        self.max_side_limit = _MAX_SIDE_LIMIT if max_side_limit is None else int(max_side_limit)

        logger.info(
            "PP-OCR detector %s on %s: limit %s/%s, thresh %s, box_thresh %s, unclip %s, "
            "max_candidates %s",
            self.model_name or self.model_dir.name, self.device, self.limit_side_len,
            self.limit_type, self.thresh, self.box_thresh, self.unclip_ratio,
            self.max_candidates,
        )

    def _load_model(self):
        from fox_reader.hf.models.auto import AutoModelForObjectDetection

        return AutoModelForObjectDetection.from_pretrained(self.model_dir, device=self.device)

    def predict(self, image: Image.Image | np.ndarray, **overrides: Any) -> list[dict[str, Any]]:
        """``[{"dt_polys": (N, 4, 2) int16, "dt_scores": (N,) float32}]``."""
        unknown = set(overrides) - set(self._PREDICT_KEYS)
        if unknown:
            raise TypeError(f"predict() got unexpected keyword argument(s): {sorted(unknown)}")

        pil = _as_pil(image)
        limit_side_len = overrides.get("limit_side_len") or self.limit_side_len
        limit_type = overrides.get("limit_type") or self.limit_type
        thresh = overrides.get("thresh")
        box_thresh = overrides.get("box_thresh")
        unclip_ratio = overrides.get("unclip_ratio")

        with self._lock:
            model_inputs = self.image_processor(
                images=[pil],
                return_tensors="pt",
                limit_side_len=limit_side_len,
                limit_type=limit_type,
                max_side_limit=self.max_side_limit,
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.model(pixel_values=model_inputs["pixel_values"])
            predictions = self.image_processor.post_process_object_detection(
                outputs,
                threshold=self.thresh if thresh is None else float(thresh),
                target_sizes=model_inputs["target_sizes"],
                box_threshold=self.box_thresh if box_thresh is None else float(box_thresh),
                max_candidates=self.max_candidates,
                min_size=_MIN_SIZE,
                unclip_ratio=self.unclip_ratio if unclip_ratio is None else float(unclip_ratio),
            )

        prediction = predictions[0]
        polys = prediction["boxes"].detach().cpu().numpy().astype(np.int16, copy=False)
        scores = prediction["scores"].detach().cpu().numpy().astype(np.float32, copy=False)
        if polys.size == 0:
            polys = np.empty((0, 4, 2), dtype=np.int16)
            scores = np.empty((0,), dtype=np.float32)
        return [{"dt_polys": polys, "dt_scores": scores}]


class TextRecognition(_Predictor):
    """CTC text recognition. ``predict`` answers with text and a score."""

    def _load_model(self):
        from fox_reader.hf.models.auto import AutoModelForTextRecognition

        return AutoModelForTextRecognition.from_pretrained(self.model_dir, device=self.device)

    def predict(self, image: Image.Image | np.ndarray, **_: Any) -> list[dict[str, Any]]:
        """``[{"rec_text": str, "rec_score": float}]``."""
        pil = _as_pil(image)
        with self._lock:
            model_inputs = self.image_processor(images=[pil], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model(pixel_values=model_inputs["pixel_values"])
            results = self.image_processor.post_process_text_recognition(outputs)
        result = results[0]
        return [{"rec_text": result["text"], "rec_score": float(result["score"])}]


__all__ = [
    "OCRModelUnavailable",
    "TextDetection",
    "TextRecognition",
    "torch_device",
]
