# ============================================================================
# LOCAL DIRECTORY ISOLATION
# ============================================================================
import logging
import os
from pathlib import Path

from fox_reader.device import FOX_DEVICE
from fox_reader.utils import MODELS_DIR

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
from PIL import Image

# ============================================================================
# READING-ORDER HELPERS
# ============================================================================
#: A box only votes for an orientation when one side clearly dominates the
#: other. Near-square boxes (single kana/kanji, furigana, punctuation such as
#: "…" or "!!") carry no directional information and abstain instead of
#: flipping a close vote.
_ORIENTATION_RATIO = 1.3


def _median(values):
    """Median of a non-empty sequence, 0.0 for anything unusable."""
    try:
        clean = sorted(float(v) for v in values)
    except Exception:
        return 0.0

    clean = [v for v in clean if np.isfinite(v)]

    if not clean:
        return 0.0

    mid = len(clean) // 2

    if len(clean) % 2:
        return clean[mid]

    return (clean[mid - 1] + clean[mid]) / 2.0


def _is_vertical_text(boxes):
    """Decide whether the detections form vertical (right-to-left) text.

    Three signals, weakest last: confident per-box aspect votes, then the
    aspect of the whole block, then a horizontal default. The old code counted
    every box with ``h > w`` as vertical, so one tall box (a split column, a
    vertical "!!") could outvote the layout, and a 1-1 tie silently fell
    through to horizontal.
    """
    vertical = 0
    horizontal = 0

    for b in boxes:
        w, h = b["w"], b["h"]

        if h >= w * _ORIENTATION_RATIO:
            vertical += 1
        elif w >= h * _ORIENTATION_RATIO:
            horizontal += 1

    if vertical != horizontal:
        return vertical > horizontal

    try:
        block_w = max(b["x"] + b["w"] / 2.0 for b in boxes) - min(b["x"] - b["w"] / 2.0 for b in boxes)
        block_h = max(b["y"] + b["h"] / 2.0 for b in boxes) - min(b["y"] - b["h"] / 2.0 for b in boxes)
    except Exception:
        return False

    if block_h >= block_w * _ORIENTATION_RATIO:
        return True

    if block_w >= block_h * _ORIENTATION_RATIO:
        return False

    return False


def _cluster_1d(items, key, tolerance):
    """Group items whose ``key`` values lie within ``tolerance`` of each other.

    Sorts by the key, then cuts a new group wherever the gap to the previous
    item exceeds the tolerance. Groups come back ordered by their mean key so
    callers can lay lines top-to-bottom or columns right-to-left. A non-positive
    tolerance means every item stands alone instead of collapsing everything
    into one group.
    """
    if tolerance is None or not np.isfinite(tolerance) or tolerance <= 0:
        return [[item] for item in sorted(items, key=key)]

    ordered = sorted(items, key=key)
    groups = [[ordered[0]]]

    for item in ordered[1:]:
        if abs(key(item) - key(groups[-1][-1])) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])

    groups.sort(key=lambda g: sum(key(b) for b in g) / len(g))
    return groups


def _sort_horizontal(boxes):
    """Top-to-bottom lines, left-to-right within each line.

    The old ``(y, x)`` float sort interleaved boxes from neighbouring lines
    whenever their centres differed by a pixel. Clustering by the median line
    height keeps a line together first and only then orders within it.
    """
    tolerance = _median([b["h"] for b in boxes]) * 0.5
    lines = _cluster_1d(boxes, key=lambda b: b["y"], tolerance=tolerance)
    ordered = []

    for line in lines:
        ordered.extend(sorted(line, key=lambda b: (b["x"], b["y"])))

    return ordered


def _sort_vertical(boxes):
    """Right-to-left columns, top-to-bottom within each column.

    Same pixel-jitter problem as the horizontal case, transposed: furigana and
    neighbouring columns whose centres differ by a pixel or two were split into
    phantom columns by the raw ``(-x, y)`` sort. Clustering by the median
    column width keeps a column (ruby or main) together first.
    """
    tolerance = _median([b["w"] for b in boxes]) * 0.5
    columns = _cluster_1d(boxes, key=lambda b: b["x"], tolerance=tolerance)
    ordered = []

    for column in reversed(columns):
        ordered.extend(sorted(column, key=lambda b: (b["y"], b["x"])))

    return ordered


# ============================================================================
# PADDLE OCR WRAPPER
# ============================================================================
from paddleocr import TextDetection, TextRecognition


class PaddleOCRModelManager:
    _detectors = {}
    _recognizers = {}

    @classmethod
    def get_detector(cls, model_name: str, device: str):
        key = (model_name, device)
        paddle_ocr_path = MODELS_DIR / "paddleocr"

        if key not in cls._detectors:
            cls._detectors[key] = TextDetection(
                model_dir=(paddle_ocr_path / model_name),
                engine="transformers",
                device=device,
            )

        return cls._detectors[key]

    @classmethod
    def get_recognizer(cls, model_name: str, device: str, config_model_name: str | None = None):
        key = (model_name, device)
        paddle_ocr_path = MODELS_DIR / "paddleocr"

        if key not in cls._recognizers:
            kwargs = dict(
                model_dir=(paddle_ocr_path / model_name),
                engine="transformers",
                device=device,
            )
            if config_model_name:
                kwargs["model_name"] = config_model_name
            cls._recognizers[key] = TextRecognition(**kwargs)

        return cls._recognizers[key]

    @classmethod
    def clear(cls):
        cls._detectors.clear()
        cls._recognizers.clear()


class PaddleOCRWrapper:
    def __init__(
        self,
        rec_model_name: str,
        det_model_name: str,
        langs: list[str],
        device: str = "cpu",
        config_model_name: str | None = None,
        load_recognizer: bool = True,
    ):
        self.detector = PaddleOCRModelManager.get_detector(
            model_name=det_model_name,
            device=device,
        )

        # The recognizers are the heavy half of the pipeline. `load_recognizer`
        # exists so the PaddleOCR-VL mode can keep the detector (text clean
        # needs its boxes) without paying for recognition it never calls;
        # `ensure_recognizer` loads it on first actual use.
        self._rec_model_name = rec_model_name
        self._rec_device = device
        self._config_model_name = config_model_name
        self.recognizer = (
            PaddleOCRModelManager.get_recognizer(
                model_name=rec_model_name,
                device=device,
                config_model_name=config_model_name,
            )
            if load_recognizer
            else None
        )

        self.languages = []
        if isinstance(langs, list):
            self.languages = [lng for lng in langs if isinstance(lng, str)]

    def ensure_recognizer(self):
        """Load the recognizer unless it is already there; answer with it.

        Idempotent and safe to call on every predict: after the first load it
        is a single None check. Callers sharing one wrapper across threads
        must serialize, the same as for predict itself (OCRService does).
        """
        if self.recognizer is None:
            self.recognizer = PaddleOCRModelManager.get_recognizer(
                model_name=self._rec_model_name,
                device=self._rec_device,
                config_model_name=self._config_model_name,
            )
        return self.recognizer


    def _preprocess(self, input: Image.Image, isGrayScaled: bool) -> np.ndarray:
        img = input.convert("RGB")
        if isGrayScaled:
            img = img.convert("L")
            threshold = 160
            img = img.point(lambda p: 255 if p > threshold else 0, mode="1")
            img = img.convert("RGB")

        return np.array(img)


    def _postprocess(self, items):
        boxes = []

        if isinstance(items, dict):
            items = [items]

        if not isinstance(items, (list, tuple)):
            return ""

        for item in items:
            if not isinstance(item, dict):
                continue

            text = item.get("text", "")

            if not isinstance(text, str):
                try:
                    text = str(text)
                except Exception:
                    continue

            if not text:
                continue

            try:
                box = np.array(item["box"], dtype=np.float64).reshape(-1, 2)
            except Exception:
                continue

            if box.shape[0] < 3:
                continue

            xs = box[:, 0]
            ys = box[:, 1]

            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())

            if not all(np.isfinite([x0, x1, y0, y1])):
                continue

            boxes.append({
                "text": text,
                "x": (x0 + x1) / 2.0,
                "y": (y0 + y1) / 2.0,
                "w": max(x1 - x0, 0.0),
                "h": max(y1 - y0, 0.0),
            })

        if not boxes:
            return ""

        if _is_vertical_text(boxes):
            # vertical Japanese: columns run right -> left,
            # characters within a column run top -> bottom
            ordered = _sort_vertical(boxes)
        else:
            # horizontal text: lines run top -> bottom,
            # characters within a line run left -> right
            ordered = _sort_horizontal(boxes)

        ocr_text = "".join(b["text"] for b in ordered)
        ocr_text = ocr_text.replace("…", "...")
        ocr_text = ocr_text.replace("！", "!")
        return ocr_text


    def predict(self, input: Image.Image, isGrayScaled: bool, **kwargs) -> str:
        # Defensive: a detector-only wrapper must still transcribe if anyone
        # calls it directly instead of going through MultiLangOCR.
        self.ensure_recognizer()
        img_np = self._preprocess(input, isGrayScaled)
        det_result = self.detector.predict(img_np)
        boxes = self._extract_boxes(det_result)

        results = []
        for box in boxes:
            crop = self._crop_image(input, box)
            crop_np = np.array(crop)
            rec_result = self.recognizer.predict(crop_np)
            text, score = self._extract_text(rec_result)

            if text:
                results.append({
                    "text": text,
                    "score": score,
                    "box": box,
                })

        return self._postprocess(results)


    def _extract_boxes(self, result):
        result = result[0]
        if isinstance(result, dict):
            return result["dt_polys"]
        return result.dt_polys


    def _extract_text(self, result):
        rec_text = ""
        rec_score = 0

        if isinstance(result, list):
            for r in result:
                rec_text = r.get("rec_text", "")
                rec_score = r.get("rec_score", 0)

        elif isinstance(result, dict):
            rec_text = result.get("rec_text", "")
            rec_score = result.get("rec_score", 0)

        return rec_text, rec_score


    def _crop_image(self, image, box):
        def __find_coeffs(pa, pb):
            matrix = []

            for p1, p2 in zip(pa, pb):
                matrix.append(
                    [p1[0], p1[1], 1, 0, 0, 0,
                    -p2[0] * p1[0], -p2[1] * p1[0]]
                )
                matrix.append(
                    [0, 0, 0, p1[0], p1[1], 1,
                    -p2[0] * p1[1], -p2[1] * p1[1]]
                )

            A = np.array(matrix, dtype=np.float64)
            B = np.array(pb).reshape(8)

            coeffs = np.linalg.solve(A, B)
            return coeffs

        def __get_rotate_crop_image(img, points):

            points = np.asarray(points, dtype=np.float32)

            width = int(
                max(
                    np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3]),
                )
            )

            height = int(
                max(
                    np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2]),
                )
            )

            dst = [
                (0, 0),
                (width - 1, 0),
                (width - 1, height - 1),
                (0, height - 1),
            ]

            src = [tuple(p) for p in points]

            coeffs = __find_coeffs(dst, src)

            crop = img.transform(
                (width, height),
                Image.Transform.PERSPECTIVE,
                coeffs,
                Image.Resampling.BICUBIC,
            )

            if crop.height >= crop.width * 1.5:
                crop = crop.rotate(90, expand=True)

            return crop


        return __get_rotate_crop_image(image, box)


# ============================================================================
# MULTI OCR MODEL
# ============================================================================
class MultiLangOCR:
    def __init__(self, recognizers: bool = True):
        """Build the classic pipeline.

        ``recognizers=False`` builds a detector-only instance: the detector
        (which text clean borrows) loads, the two recognition models do not.
        That is the mode OCRService uses when PaddleOCR-VL is selected, and
        `ensure_recognizers` below loads them later if the VL engine ever
        fails and the classic fallback is actually needed.
        """
        self.device: str = FOX_DEVICE.paddleocr

        main_engine = PaddleOCRWrapper(
            device=self.device,
            det_model_name="PP-OCRv6_medium_det_safetensors",
            rec_model_name="PP-OCRv6_medium_rec_safetensors",
            langs=["japanese", "chinese", "english"],
            load_recognizer=recognizers,
        )
        korean_engine = PaddleOCRWrapper(
            device=self.device,
            det_model_name="PP-OCRv6_medium_det_safetensors",
            rec_model_name="korean_PP-OCRv5_mobile_rec_safetensors",
            config_model_name="korean_PP-OCRv5_mobile_rec",
            langs=["korean"],
            load_recognizer=recognizers,
        )

        self._engines: dict[str, PaddleOCRWrapper] = {
            "default": main_engine,
            "japanese": main_engine,
            "chinese": main_engine,
            "english": main_engine,
            "korean": korean_engine,
        }
        self._recognizers_loaded = recognizers

    def ensure_recognizers(self) -> None:
        """Load any missing recognizers; no-op when all are present.

        Several language keys share one wrapper instance, so instances are
        deduplicated before loading. See `PaddleOCRWrapper.ensure_recognizer`
        for the threading contract.
        """
        if self._recognizers_loaded:
            return
        seen: set[int] = set()
        for engine in self._engines.values():
            if id(engine) in seen:
                continue
            seen.add(id(engine))
            engine.ensure_recognizer()
        self._recognizers_loaded = True

    def predict(self, lang: str, input: Image.Image, isGrayScaled: bool, **kwargs):
        # A detector-only instance still transcribes on demand: the first
        # classic predict pays the recognizer load instead of failing.
        self.ensure_recognizers()
        lang_clean = lang.strip().lower()
        engine = self._engines.get(lang_clean, self._engines["default"])
        return engine.predict(input=input, isGrayScaled=isGrayScaled, **kwargs)


# ============================================================================
# OCR ENGINE SELECTION (classic PaddleOCR vs PaddleOCR-VL GGUF)
# ============================================================================
# The settings page offers ``paddleocr`` (this file's MultiLangOCR, always
# available once the required weights exist) and ``paddleocr-vl`` (the GGUF in
# fox_reader.ocr_vl, only when llama-cpp-python and its weights are present).
# Helpers here never raise and never import llama_cpp: ocr_vl keeps that
# import lazy so probing from the settings page cannot fail startup.


def normalize_ocr_engine(value) -> str:
    """A saved engine id as a supported one, defaulting to classic."""
    from fox_reader.constants import DEFAULT_OCR_ENGINE, SUPPORTED_OCR_ENGINES

    text = str(value or "").strip().lower() or DEFAULT_OCR_ENGINE
    if text in SUPPORTED_OCR_ENGINES:
        return text
    logger.warning("Unknown OCR engine %r; using %r", value, DEFAULT_OCR_ENGINE)
    return DEFAULT_OCR_ENGINE


def vl_engine_available() -> tuple[bool, str]:
    """Whether the VL engine can run: (ok, reason). Never raises."""
    try:
        from fox_reader.ocr_vl import is_llama_cpp_available, vl_model_downloaded
    except Exception as exc:
        return False, f"VL engine could not be probed: {exc}"
    try:
        if not is_llama_cpp_available():
            return False, (
                "llama-cpp-python is not installed (the 'gguf' extra). "
                "Install it to enable PaddleOCR-VL."
            )
        if not vl_model_downloaded():
            return False, "PaddleOCR-VL weights are not downloaded (see Setup)."
        return True, ""
    except Exception as exc:
        return False, f"VL engine could not be probed: {exc}"


def describe_ocr_engines(selected: str | None = None) -> dict:
    """Payload the settings page renders. Never raises."""
    from fox_reader.constants import (
        DEFAULT_OCR_ENGINE,
        OCR_ENGINE_PADDLEOCR,
        OCR_ENGINE_PADDLEOCR_VL,
    )

    sel = normalize_ocr_engine(selected)
    vl_ok, vl_reason = vl_engine_available()
    try:
        from fox_reader.ocr_vl import is_llama_cpp_available, llama_cpp_version, vl_model_downloaded

        has_llama = bool(is_llama_cpp_available())
        has_weights = bool(vl_model_downloaded())
        llama_ver = llama_cpp_version()
    except Exception:
        has_llama, has_weights, llama_ver = False, False, None
    return {
        "selected": sel,
        "default": DEFAULT_OCR_ENGINE,
        "restart_required": True,
        "llama_cpp": has_llama,
        "llama_cpp_version": llama_ver,
        "vl_downloaded": has_weights,
        "engines": [
            {
                "id": OCR_ENGINE_PADDLEOCR,
                "label": "PaddleOCR (classic)",
                "available": True,
                "reason": "",
                "selected": sel == OCR_ENGINE_PADDLEOCR,
            },
            {
                "id": OCR_ENGINE_PADDLEOCR_VL,
                "label": "PaddleOCR-VL-1.6 (GGUF)",
                "available": vl_ok,
                "reason": vl_reason,
                "selected": sel == OCR_ENGINE_PADDLEOCR_VL,
            },
        ],
    }
