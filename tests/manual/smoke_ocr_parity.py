"""Prove the vendored OCR path matches paddleocr exactly, before it is removed.

Run this while `paddleocr` is still installed. It builds both stacks against
the same weights and compares what they return, tensor for tensor:

    .venv/Scripts/python.exe tests/manual/smoke_ocr_parity.py

Detection is compared on whole synthetic pages (``dt_polys`` must be
*identical* int16 arrays, ``dt_scores`` within 1e-5), recognition on crops the
old detector found (``rec_text`` must be equal strings, ``rec_score`` within
1e-5). Both recognisers are covered, including the Korean one, because it
takes a different model class and image processor than the main pair.

Exits non-zero on any mismatch, so it can gate the dependency removal.

This is a manual smoke test rather than a pytest case on purpose: it needs the
real ~190 MB of weights *and* the dependency being deleted, so it can only run
on a developer machine mid-migration. `tests/test_hf_vendored.py` is the part
that keeps working afterwards.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models" / "paddleocr"
FONT = ROOT / "fonts" / "ComicMono-Bold.ttf"

DET_DIR = MODELS / "PP-OCRv6_medium_det_safetensors"
REC_DIR = MODELS / "PP-OCRv6_medium_rec_safetensors"
KOR_DIR = MODELS / "korean_PP-OCRv5_mobile_rec_safetensors"
KOR_NAME = "korean_PP-OCRv5_mobile_rec"

#: Pages to detect on. Varied in size and line count so the resize path
#: (960/max, rounded to a multiple of 32) is exercised at more than one shape.
PAGES = (
    ("two-lines", (640, 240), (("HELLO WORLD", 30, 30), ("Fox Reader 123", 30, 110))),
    ("one-line", (420, 120), (("Tiny", 20, 30),)),
    ("tall", (300, 700), (("AB", 20, 40), ("CD", 20, 200), ("EF", 20, 380), ("GH", 20, 560))),
    ("wide", (1600, 200), (("a very wide line of text indeed", 40, 70),)),
)

_failures: list[str] = []


def fail(message: str) -> None:
    _failures.append(message)
    print(f"  FAIL  {message}")


def render(size, lines) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), 40)
    for text, x, y in lines:
        draw.text((x, y), text, font=font, fill="black")
    return image


def crops_from(polys: np.ndarray, image: Image.Image) -> list[Image.Image]:
    """Perspective-crop each quad, the way `fox_reader.ocr` does.

    Imported from the real thing rather than reimplemented, so the parity
    check cannot pass because both sides share a private bug here.
    """
    from fox_reader.ocr import PaddleOCRWrapper

    cropper = PaddleOCRWrapper.__new__(PaddleOCRWrapper)
    return [cropper._crop_image(image, poly) for poly in polys]


def compare_det(name: str, old, new, image: Image.Image) -> np.ndarray | None:
    old_result, new_result = old.predict(np.array(image)), new.predict(np.array(image))
    old_polys = np.asarray(old_result[0]["dt_polys"])
    new_polys = np.asarray(new_result[0]["dt_polys"])
    old_scores = np.asarray(old_result[0]["dt_scores"], dtype=np.float32)
    new_scores = np.asarray(new_result[0]["dt_scores"], dtype=np.float32)

    if old_polys.shape != new_polys.shape:
        fail(f"det[{name}] poly shape {old_polys.shape} -> {new_polys.shape}")
        return None
    if not np.array_equal(old_polys, new_polys):
        worst = int(np.abs(old_polys.astype(int) - new_polys.astype(int)).max())
        fail(f"det[{name}] polys differ, worst by {worst}px")
        return None
    if old_scores.shape != new_scores.shape or not np.allclose(old_scores, new_scores, atol=1e-5):
        fail(f"det[{name}] scores differ by up to {np.abs(old_scores - new_scores).max():.2e}")
        return None

    print(f"  ok    det[{name}] {len(old_polys)} quads, polys identical, scores within 1e-5")
    return old_polys


def compare_rec(label: str, old, new, crops: list[Image.Image]) -> None:
    mismatched = 0
    for index, crop in enumerate(crops):
        old_result = old.predict(np.array(crop))[0]
        new_result = new.predict(np.array(crop))[0]
        if old_result["rec_text"] != new_result["rec_text"]:
            fail(f"rec[{label}] crop {index}: {old_result['rec_text']!r} -> {new_result['rec_text']!r}")
            mismatched += 1
        elif abs(float(old_result["rec_score"]) - float(new_result["rec_score"])) > 1e-5:
            fail(
                f"rec[{label}] crop {index} score "
                f"{old_result['rec_score']:.6f} -> {new_result['rec_score']:.6f}"
            )
            mismatched += 1
    if not mismatched:
        texts = [old.predict(np.array(c))[0]["rec_text"] for c in crops]
        print(f"  ok    rec[{label}] {len(crops)} crops identical: {texts}")


def main() -> int:
    for path in (DET_DIR, REC_DIR, KOR_DIR, FONT):
        if not path.exists():
            print(f"missing {path}; download the models from the setup page first")
            return 2
    try:
        from paddleocr import TextDetection as OldDet
        from paddleocr import TextRecognition as OldRec
    except ImportError:
        print("paddleocr is not installed, so there is nothing to compare against.")
        print("Run this before removing the dependency; afterwards the reference is gone.")
        return 2

    from fox_reader.ocr_predict import TextDetection as NewDet
    from fox_reader.ocr_predict import TextRecognition as NewRec

    print("building both stacks")
    started = time.time()
    old_det = OldDet(model_dir=DET_DIR, engine="transformers", device="cpu")
    old_rec = OldRec(model_dir=REC_DIR, engine="transformers", device="cpu")
    old_kor = OldRec(model_dir=KOR_DIR, engine="transformers", device="cpu", model_name=KOR_NAME)
    old_elapsed = time.time() - started

    started = time.time()
    new_det = NewDet(DET_DIR, device="cpu")
    new_rec = NewRec(REC_DIR, device="cpu")
    new_kor = NewRec(KOR_DIR, device="cpu", model_name=KOR_NAME)
    new_elapsed = time.time() - started
    print(f"  paddleocr {old_elapsed:.1f}s   vendored {new_elapsed:.1f}s")

    # The settings paddlex resolved, which the new predictor has to reproduce.
    px = old_det.paddlex_predictor
    for field in ("limit_side_len", "limit_type", "thresh", "box_thresh", "unclip_ratio", "max_side_limit"):
        old_value, new_value = getattr(px, field), getattr(new_det, field)
        if old_value != new_value:
            fail(f"det setting {field}: {old_value!r} -> {new_value!r}")
    if not _failures:
        print(
            f"  ok    det settings: limit {new_det.limit_side_len}/{new_det.limit_type}, "
            f"thresh {new_det.thresh}, box_thresh {new_det.box_thresh}, "
            f"unclip {new_det.unclip_ratio}"
        )

    print("comparing")
    for name, size, lines in PAGES:
        page = render(size, lines)
        polys = compare_det(name, old_det, new_det, page)
        if polys is None or not len(polys):
            continue
        crops = crops_from(polys, page)
        compare_rec(f"{name}/main", old_rec, new_rec, crops)
        compare_rec(f"{name}/korean", old_kor, new_kor, crops)

    print()
    if _failures:
        print(f"{len(_failures)} mismatch(es) -- do NOT remove paddleocr yet")
        return 1
    print("identical on every comparison")
    return 0


if __name__ == "__main__":
    sys.exit(main())
