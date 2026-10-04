"""Glyph-level text segmentation (Manga-Text-Segmentation-2025).

Ported from the standalone ``seg.py`` research script. Differences that matter:

* the model lives under ``models/misc/Manga-Text-Segmentation-2025`` and is loaded
  once into a process-wide singleton, so a second clean on the same page does not
  pay for the load again;
* the device follows ``FOX_DEVICE.textseg`` -- its own slot on the settings page,
  separate from the OCR detector, so cleaning and recognition can sit on
  different devices;
* the architecture is :mod:`fox_reader.clean.unetplusplus` -- a written-out
  replica of the ``segmentation_models_pytorch`` / ``timm`` model the checkpoint
  was trained against, so that loading one model does not mean installing two
  model zoos. It is a replica down to the parameter names, which is what lets
  the load below stay ``strict=True``;
* ``torch`` is imported lazily, and so is that module, which imports torch at
  its own scope. torch is an optional extra, and neither the app nor the health
  endpoint may fail to start because a clean method the user has not selected is
  unavailable;
* the input is an in-memory BGR array, not a path, so a single region can be
  segmented from a crop instead of the whole page.

Nothing is written to disk and nothing is cached: a clean costs one forward pass
per (scale, polarity, flip), and how many of those there are is the speed knob.
Two cheaper measures stand in for the map cache this used to keep. The model is
held in channels_last layout, which is about a quarter of the CPU runtime, and a
crop with no dark ground on it skips the tone-inverted pass altogether. What made
the cache worth its disk -- 20-40 s for a full page -- is also not what is spent
here: the network sees the padded union of the regions that asked for it, and a
speech bubble is a twentieth of a page.

The speed and tile presets live in :mod:`fox_reader.clean.tuning`, which imports
nothing, so a caller can name one without importing this module and its torch.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from fox_reader.clean.tuning import (
    DEFAULT_SPEED,
    DEFAULT_TILE,
    DEFAULT_TTA,
    SPEEDS,
    TILE_OVERLAPS,
    normalise_speed,
    normalise_tile,
    overlap_for,
)
from fox_reader.device import FOX_DEVICE
from fox_reader.utils import MODELS_DIR
from fox_reader.constants import TEXT_SEG_MODEL_DIR

log = logging.getLogger(__name__)

#: Re-exported so that a caller already importing ``seg`` need not also reach
#: into ``tuning``; ``tuning`` stays the import-cheap door for everyone else.
__all__ = [
    "DEFAULT_SCALES", "DEFAULT_SPEED", "DEFAULT_TILE", "DEFAULT_TTA", "SPEEDS",
    "TILE_OVERLAPS", "SegUnavailable", "detect", "detect_at", "is_available",
    "load_model", "normalise_speed", "normalise_tile", "overlap_for",
    "scale_maps", "torch_device", "unload",
]

MODEL_DIR = MODELS_DIR / TEXT_SEG_MODEL_DIR
MODEL_PATH = MODEL_DIR / "model.safetensors"
LEGACY_PATH = MODEL_DIR / "model.pth"
#: Still spelled the way smp spelled it -- ``tu-`` for a timm-universal encoder.
#: Duplicated from :data:`.unetplusplus.ENCODER_NAME` rather than imported,
#: because importing it here would pull torch in at module scope; the constructor
#: rejects a mismatch, so the two cannot drift apart unnoticed.
ENCODER = "tu-efficientnetv2_rw_m"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_SCALES: tuple[float, ...] = (1.0, 0.6, 0.4)

_lock = threading.Lock()
_model: Any | None = None
_model_device: str | None = None
#: The device that was *asked* for, as opposed to the one the model ended up on.
#: Keyed separately because the two differ whenever `_build_model` falls back to
#: the CPU: caching only the real device would never match the request again, and
#: the 216 MB model would be rebuilt on every single clean.
_model_want: str | None = None

Progress = Callable[[str], None]


class SegUnavailable(RuntimeError):
    """Text Seg cannot run: weights missing, or the optional extra is absent.

    Carries a message meant for the user, because this is the one failure the
    frontend has to explain rather than log -- the other two clean methods keep
    working, so the request is not simply broken.
    """


# --------------------------------------------------------------------- device

def torch_device() -> str:
    """A torch device string derived from the Text Seg slot.

    Text Seg has its own device picker (``FOX_DEVICE.textseg``) so that cleaning
    and OCR can sit on different devices — a user who moved OCR to the CPU to
    free VRAM may still want segmentation on the GPU, or vice versa. The slot
    is stored torch-spelled already (``cuda:0`` / ``mps`` / ``cpu``), but the
    legacy PaddleOCR spelling (``gpu:0``) is still accepted, and an install
    that predates the slot falls back to ``FOX_DEVICE.paddleocr`` — which is
    what Text Seg used to run on — rather than jumping devices after an
    upgrade.
    """
    raw: Any = getattr(FOX_DEVICE, "textseg", None)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        # Pre-separation install, or a FOX_DEVICE built before the slot
        # existed (tests, hot-reload): keep the old coupling instead of
        # silently moving the model.
        raw = getattr(FOX_DEVICE, "paddleocr", None) or "cpu"
    text = str(raw).strip().lower()
    if text.startswith("gpu"):
        _, _, tail = text.partition(":")
        return f"cuda:{int(tail)}" if tail.isdigit() else "cuda:0"
    if text.startswith("cuda") or text == "mps":
        return text
    return "cpu"


def is_available() -> tuple[bool, str]:
    """``(usable, reason)`` -- cheap enough to call before offering the method.

    Either weight file will do. ``model.safetensors`` is the one to have, but a
    fresh clone of the model repository carries ``model.pth`` and nothing else,
    and refusing to segment until it has been converted would report a method as
    broken when it is merely slower to load.
    """
    safetensors_weights = MODEL_PATH.is_file()
    if not safetensors_weights and not LEGACY_PATH.is_file():
        return False, f"the Text Seg model is missing from {MODEL_DIR}"
    from importlib.util import find_spec

    # ``(module, what to do about it)``. There is no 'segmentation' extra to
    # point at any more: the architecture is :mod:`.unetplusplus`, part of this
    # package, and safetensors arrives with transformers, which is a base
    # dependency. So neither of these being absent is an uninstalled option --
    # it is a broken install, and the remedy differs accordingly.
    needed = [("torch", "install one of the torch extras (cpu, cu126, ...)")]
    if safetensors_weights:
        # Only that file needs the reader; model.pth is torch's own format.
        needed.append(("safetensors", "reinstall the app's dependencies"))
    for name, remedy in needed:
        try:
            found = find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            return False, (f"the Text Seg backend needs {name}; {remedy} to "
                           f"use this method")
    return True, ""


# ---------------------------------------------------------------------- model

def _convert_batchnorm_to_groupnorm(module: Any, nn: Any) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            groups = 8
            if channels % groups != 0:
                for g in range(min(channels, 8), 0, -1):
                    if channels % g == 0:
                        groups = g
                        break
            setattr(module, name, nn.GroupNorm(groups, channels))
        else:
            _convert_batchnorm_to_groupnorm(child, nn)


def _channels_last(device: str) -> bool:
    """Whether this device gets the channels_last layout.

    A convnet in channels_last is worth about a quarter of the CPU runtime --
    oneDNN can use its blocked convolution kernels instead of transposing at every
    layer -- and it is not an approximation: on a full-resolution page the largest
    probability difference against contiguous layout is 6e-4, which moves no pixel
    across any threshold the mask chain uses.

    CPU and CUDA only, because those are the two the gain was measured on. MPS is
    left contiguous rather than handed an unmeasured layout change: this is a speed
    measure, and one that made masks differ there would not be worth it.
    """
    return device.split(":")[0] in ("cpu", "cuda")


def _load_state() -> Any:
    """The checkpoint's tensors, from whichever weight file is present.

    ``model.safetensors`` is preferred: nothing is unpickled, so no code from the
    checkpoint runs, and the tensors are mapped rather than deserialised. The
    original ``model.pth`` is a fallback, so that a fresh clone of the model
    repository segments before ``to_safetensors.py`` has been run over it.

    Any failure here is reported as :class:`SegUnavailable` rather than as whatever
    the reader raised. A half-finished download is the likely cause and the user is
    the only one who can fix it, so this has to reach the panel as a sentence.
    """
    src = MODEL_PATH if MODEL_PATH.is_file() else LEGACY_PATH
    try:
        if src is MODEL_PATH:
            from safetensors.torch import load_file

            return load_file(str(src), device="cpu")
        import torch

        log.warning("Text Seg is reading %s; running to_safetensors.py over it "
                    "once makes every later load cheaper", src.name)
        # weights_only=True is torch's own default from 2.6 on, and is named here
        # so that an older torch cannot quietly unpickle instead. The file holds a
        # state dict of tensors and nothing else, so the permissive mode -- which
        # would run code from the checkpoint -- buys nothing.
        return torch.load(str(src), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - one answer for any unreadable file
        raise SegUnavailable(
            f"the Text Seg weights in {src.name} could not be read "
            f"({type(exc).__name__}: {exc})") from exc


def _build_model(device: str) -> Any:
    ok, why = is_available()
    if not ok:
        raise SegUnavailable(why)
    try:
        from fox_reader.clean.unetplusplus import UnetPlusPlus
    except Exception as exc:  # noqa: BLE001 - any import failure is the same answer
        raise SegUnavailable(
            "the Text Seg backend could not be loaded "
            f"({type(exc).__name__}: {exc}); install one of the torch extras"
        ) from exc

    import torch
    import torch.nn as nn

    # The keyword-for-keyword spelling smp took, kept so that this stays
    # checkable against the package it replaces: `unetplusplus` rejects any
    # combination it does not reproduce rather than quietly building something
    # else, so a changed argument here is a ValueError, not a silent mismatch.
    model = UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    )
    _convert_batchnorm_to_groupnorm(model.decoder, nn)
    # strict=True still has to hold: the decoder's BatchNorms were replaced above
    # and the checkpoint carries GroupNorm parameters for them, so a mismatch here
    # means the wrong file.
    model.load_state_dict(_load_state())
    try:
        model.to(device)
    except Exception as exc:
        if device == "cpu":
            raise
        log.warning("Text Seg could not use %s (%s), using the CPU", device, exc)
        model.to("cpu")
        device = "cpu"
    # Outside the try/except on purpose. Returning early from the fallback would
    # leave the encoder in training mode -- only the decoder's BatchNorms were
    # replaced with GroupNorm above, so the encoder's still update their running
    # statistics from whatever page happens to be cleaned first, and the masks
    # drift page to page.
    model.eval()
    if _channels_last(device):
        # In place, and only the 4-D convolution weights are touched.
        model.to(memory_format=torch.channels_last)
    return model


def load_model(device: str | None = None) -> tuple[Any, str]:
    """The segmentation model, loading it on first use. Thread-safe."""
    global _model, _model_device, _model_want
    want = device or torch_device()
    with _lock:
        if _model is not None and _model_want == want:
            return _model, _model_device or want
        if _model is not None:
            unload(_locked=True)
        t0 = time.time()
        log.info("Loading the Text Seg model on %s", want)
        model = _build_model(want)
        _model = model
        _model_want = want
        _model_device = next(
            (str(p.device) for p in model.parameters()), want)
        log.info("Text Seg model ready on %s in %.1fs", _model_device,
                 time.time() - t0)
        return _model, _model_device


def unload(_locked: bool = False) -> None:
    """Drop the model and its VRAM. Safe to call when nothing is loaded."""
    global _model, _model_device, _model_want

    def _drop() -> None:
        global _model, _model_device, _model_want
        _model = None
        _model_device = None
        _model_want = None
        from fox_reader.device import torch_module

        torch = torch_module()
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - freeing memory is best-effort
                pass

    if _locked:
        _drop()
    else:
        with _lock:
            _drop()


# ------------------------------------------------------------------ inference

def _to_tensor(rgb: np.ndarray, channels_last: bool = True) -> Any:
    """HWC uint8 RGB -> normalised 1CHW float tensor.

    The CHW transpose is a numpy *view* of HWC memory, so what ``from_numpy``
    wraps already has channels_last strides. The explicit conversion is what says
    so, and what would still hold if the input ever arrived in another layout.
    """
    import torch

    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    out = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    if channels_last:
        out = out.contiguous(memory_format=torch.channels_last)
    return out


def _forward(model: Any, tensor: Any, channels_last: bool = True) -> Any:
    """Pad to /32, forward, crop back. Returns 1x1xHxW probabilities."""
    import torch
    import torch.nn.functional as F

    h, w = tensor.shape[-2:]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    if channels_last:
        # After the pad, not once at the top: `F.pad` hands back an NCHW tensor
        # whatever it was given, and feeding that to channels_last weights costs
        # the transpose at every convolution that the layout was chosen to avoid.
        # (The flip TTA does carry the layout through, and the HWC array the
        # tensor was made from is channels_last-strided to begin with.)
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    # `inference_mode` rather than `no_grad`: it also drops version counting and
    # view tracking, which nothing here needs and a full-page forward pass pays
    # for. The result is an inference tensor, which is why the flip TTA below only
    # ever reads it.
    with torch.inference_mode():
        prob = model(tensor).sigmoid()
    return prob[..., :h, :w]


def _predict_plain(model: Any, device: str, rgb: np.ndarray,
                   tta: bool) -> np.ndarray:
    """Whole-image prediction, optionally averaged over 4 flip variants."""
    import torch

    layout = _channels_last(device)
    tensor = _to_tensor(rgb, layout).to(device)
    variants = [()] if not tta else [(), (3,), (2,), (2, 3)]
    acc = None
    for i, dims in enumerate(variants):
        t0 = time.time()
        inp = torch.flip(tensor, dims) if dims else tensor
        out = _forward(model, inp, layout)
        if dims:
            out = torch.flip(out, dims)
        acc = out if acc is None else acc + out
        log.debug("  pass %d/%d flip=%s %.1fs", i + 1, len(variants),
                  dims or "none", time.time() - t0)
    return (acc / len(variants))[0, 0].float().cpu().numpy()


def _predict_tiled(model: Any, device: str, rgb: np.ndarray, tta: bool,
                   tile: int, overlap: int,
                   progress: Progress | None = None) -> np.ndarray:
    """Tiled prediction with a cosine window to avoid visible seams."""
    h, w = rgb.shape[:2]
    step = max(1, tile - overlap)
    ys = list(range(0, max(h - overlap, 1), step))
    xs = list(range(0, max(w - overlap, 1), step))
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    total = len(ys) * len(xs)
    n = 0
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            y0c, x0c = max(0, y1 - tile), max(0, x1 - tile)
            patch = rgb[y0c:y1, x0c:x1]
            n += 1
            t0 = time.time()
            prob = _predict_plain(model, device, patch, tta)
            ph, pw = prob.shape
            wy = np.hanning(ph + 2)[1:-1].astype(np.float32)
            wx = np.hanning(pw + 2)[1:-1].astype(np.float32)
            win = np.maximum(np.outer(wy, wx), 1e-3)
            acc[y0c:y1, x0c:x1] += prob * win
            wsum[y0c:y1, x0c:x1] += win
            log.debug(" tile %d/%d @(%d,%d) %.1fs", n, total, x0c, y0c,
                      time.time() - t0)
            if progress is not None:
                progress(f"tile {n}/{total}")
    return acc / np.maximum(wsum, 1e-6)


# ------------------------------------------------------------------- polarity

def _light_on_dark(bgr: np.ndarray, dark_max: float = 120.0,
                   margin: float = 20.0) -> np.ndarray:
    """Where lettering would be lighter than the page around it.

    The tone-inverted pass is worth running because the model cannot see light
    lettering on dark ground, but taken at face value it is far too generous: on
    an ordinary dark-on-light page, inverting turns every speech-bubble border
    into what looks like white text on black, which more than doubled the detected
    area on the reference pages. The inverted evidence is therefore only accepted
    where the normal pass is structurally blind -- a locally dark page with a
    lighter mark on it. The local page tone is a median (robust to the mark
    itself), and the result is dilated a little so that stroke interiors and
    anti-aliased edges are inside the gate too.
    """
    h, w = bgr.shape[:2]
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = max(15, (min(h, w) // 48) | 1)
    page = cv2.medianBlur(lum, k).astype(np.float32)
    gate = ((page < dark_max) &
            (lum.astype(np.float32) > page + margin)).astype(np.uint8)
    d = max(1, k // 4)
    return cv2.dilate(gate, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))) > 0


# --------------------------------------------------------------------- public

def _scaled_size(shape: tuple[int, int], s: float) -> tuple[int, int]:
    """``(w, h)`` that scale ``s`` predicts at, for ``shape`` given as ``(h, w)``.

    Never smaller than 32 px a side: that is the network's stride, so anything
    under it has nothing left to convolve, and 0.4 of a small crop lands there.
    """
    h, w = shape
    if s == 1.0:
        return w, h
    return max(32, int(round(w * s))), max(32, int(round(h * s)))


def _predict_one(model: Any, device: str, rgb: np.ndarray, s: float, inv: bool,
                 tta: bool, tile: int, overlap: int, shape: tuple[int, int],
                 progress: Progress | None) -> np.ndarray:
    """One (scale, polarity) prediction, resampled back to ``shape``."""
    h, w = shape
    if s == 1.0:
        src = rgb
    else:
        src = cv2.resize(rgb, _scaled_size(shape, s),
                         interpolation=cv2.INTER_AREA)
    if inv:
        src = 255 - src
    log.debug("predicting scale %g%s -> %dx%d tta=%s tile=%s", s,
              " inverted" if inv else "", src.shape[1], src.shape[0], tta,
              tile or "off")
    t0 = time.time()
    if tile and max(src.shape[:2]) > tile:
        prob = _predict_tiled(model, device, src, tta, tile, overlap, progress)
    else:
        prob = _predict_plain(model, device, src, tta)
    if prob.shape != (h, w):
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    log.debug("  done in %.1fs (>0.5: %d px)", time.time() - t0,
              int((prob > 0.5).sum()))
    return prob.astype(np.float32)


def scale_maps(bgr: np.ndarray, tta: bool = DEFAULT_TTA,
               scales: tuple[float, ...] = DEFAULT_SCALES,
               polarity: bool = True, tile: int = DEFAULT_TILE,
               overlap: int | None = None,
               progress: Progress | None = None) -> dict[float, np.ndarray]:
    """Per-scale probability maps. Every call predicts; nothing is cached.

    Agreement across scales is a strong text/art discriminator: a glyph is still
    a glyph at 60% size, so the network finds it at several scales, whereas the
    art it mistakes for text usually only fires at one. Polarity is *not* such a
    discriminator -- text has only one polarity -- so the two polarities of a
    scale are merged into that scale's map before voting, with the inverted pass
    restricted to where light-on-dark lettering is possible at all
    (:func:`_light_on_dark`).

    A crop with nothing dark on it skips the inverted pass outright rather than
    predicting a map the all-zero gate would then throw away. That halves the
    passes where it fires and costs nothing where it does not, since the gate has
    to be computed either way -- but it does not fire often: manga has ink
    everywhere.

    ``tile`` is one of :data:`.tuning.TILE_OVERLAPS`; anything else falls back to
    whole-image prediction rather than raising, since it can arrive from a saved
    project. ``overlap`` defaults to the one tabulated for that tile -- passing
    them separately is only for the tests.
    """
    if bgr is None or bgr.size == 0:
        raise SegUnavailable("nothing to segment")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

    h, w = bgr.shape[:2]
    scales = tuple(s for s in scales if s > 0) or (1.0,)
    gate = _light_on_dark(bgr) if polarity else None
    pols = (False, True) if gate is not None and gate.any() else (False,)
    if polarity and len(pols) == 1:
        log.debug("no dark ground here: skipping the tone-inverted pass")
    tile = normalise_tile(tile)
    if overlap is None:
        overlap = overlap_for(tile)
    # A step of zero would loop forever; a negative one is the same mistake.
    overlap = int(min(max(0, overlap), max(0, tile - 1))) if tile else overlap

    rgb: np.ndarray | None = None
    model: Any | None = None
    device = torch_device()

    t_all = time.time()
    out: dict[float, np.ndarray] = {}
    steps = len(scales) * len(pols)
    step = 0
    for s in scales:
        merged = None
        for inv in pols:
            step += 1
            if model is None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                model, device = load_model(device)
                if device.startswith("cpu"):
                    _limit_threads()
            if progress is not None:
                progress(f"Text Seg {step}/{steps}"
                         f" (scale {s:g}{', inverted' if inv else ''})")
            prob = _predict_one(model, device, rgb, s, inv, tta, tile,
                                overlap, (h, w), progress)
            if inv and gate is not None:
                prob = np.where(gate, prob, 0.0).astype(np.float32)
            merged = prob if merged is None else np.maximum(merged, prob)
        out[s] = merged
    passes = steps * (4 if tta else 1)
    log.info("Text Seg inference done in %.1fs (%d pass%s)",
             time.time() - t_all, passes, "es" if passes != 1 else "")
    return out


def _limit_threads() -> None:
    """Leave the event loop a core.

    Torch on the CPU will happily take every core for 40 s, and the health
    endpoint has to answer during that. One core held back is a few percent of
    throughput against a request that would otherwise time out.
    """
    from fox_reader.device import torch_module

    torch = torch_module()
    if torch is None:
        return
    try:
        cores = os.cpu_count() or 4
        torch.set_num_threads(max(1, cores - 1))
    except Exception:  # noqa: BLE001 - not worth failing a clean over
        pass


def detect(bgr: np.ndarray, tta: bool = DEFAULT_TTA,
           scales: tuple[float, ...] = DEFAULT_SCALES, vote_thr: float = 0.5,
           votes: int = 2, polarity: bool = True,
           progress: Progress | None = None,
           **kw: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(coverage, agreement)`` for a BGR page or crop.

    ``coverage`` is the pixel-wise maximum over scales -- generous, so that the
    interior of oversized strokes is included. ``agreement`` marks pixels that at
    least ``votes`` scales call text; it is used to decide *whether* a detected
    region is text at all, which the maximum on its own cannot do.

    With a single scale there is nothing to agree with, so the agreement is
    ``None`` -- meaning "no vote was taken" -- rather than a map that would call
    every detection unsupported and empty the mask. :func:`.mask.refine_mask`
    reads ``None`` as "keep everything you found, false positives included",
    which is the honest answer for the ``single`` preset. Returning a one-scale
    vote map instead would have looked like agreement and quietly changed what
    ``--speed single`` means.

    The network responds to text *ink edges*; inside very thick strokes its
    response collapses, because such strokes are far outside the widths it was
    trained on. Predicting on downscaled copies brings them back into range.
    """
    sm = scale_maps(bgr, tta=tta, scales=scales, polarity=polarity,
                    progress=progress, **kw)
    maps = list(sm.values())
    coverage = maps[0]
    for m in maps[1:]:
        coverage = np.maximum(coverage, m)
    if len(maps) < 2:
        return coverage, None
    votes_map = np.zeros(coverage.shape, np.uint8)
    for m in maps:
        votes_map += (m > vote_thr).astype(np.uint8)
    return coverage, votes_map >= min(votes, len(maps))


def detect_at(bgr: np.ndarray, speed: str = DEFAULT_SPEED,
              tta: bool = DEFAULT_TTA, tile: int = DEFAULT_TILE,
              progress: Progress | None = None,
              **kw: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """:func:`detect` driven by a named preset, the way the CLI's ``main`` does.

    ``tta`` here is the user's switch, not the preset's: it can only turn flip TTA
    *off*, since a preset that does not ask for it has none to enable. Same for
    the tile, which the presets say nothing about.
    """
    preset_tta, scales, polarity = SPEEDS[normalise_speed(speed)]
    tile = normalise_tile(tile)
    return detect(bgr, tta=preset_tta and bool(tta), scales=scales,
                  polarity=polarity, tile=tile, overlap=overlap_for(tile),
                  progress=progress, **kw)
