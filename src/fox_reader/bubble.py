"""Speech-bubble segmentation behind `BubbleService`.

Pure-torch inference over ``model.safetensors`` — no detection framework at
runtime. Only torch, safetensors, numpy, OpenCV and Pillow.

Layout
------
* :class:`BubbleModel` — the public loader. ``weights`` is a
  ``model.safetensors`` file (or the directory holding it); ``device`` is a
  torch-style string (``cpu`` / ``cuda:N`` / ``mps``) defaulting to the
  resolved ``bubble`` slot. Construction loads, fuses and evals once.
* :func:`getBubbleModel` — thread-safe cached construction for the service.
* Grayscale lives here, not in the service: ``detect(..., grayscale=True)``
  binarises at 160 exactly the way the reader always has, so every caller
  gets the same pixels.
* A GPU that is not usable is a fallback, not a refusal: this net is 2.86M
  parameters, so the CPU is seconds rather than death (unlike a GGUF).

Nothing here raises on import: torch is an optional extra and safetensors is
only needed when weights actually load. Every failure after that — missing
torch, missing file, unreadable weights — arrives as :class:`BubbleUnavailable`
with a sentence the routes can show.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

logger = logging.getLogger(__name__)

try:
    from safetensors.torch import load_file as _load_safe
except Exception:  # pragma: no cover - reported cleanly at load time
    _load_safe = None  # type: ignore[assignment]

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

WEIGHTS_FILE = "model.safetensors"

_IMG = 1600
_CONF = 0.25
_IOU = 0.7
_MAX_DET = 300
_NC = 1
_NM = 32
_REG_MAX = 16
_PAD_VALUE = 114

#: Binarisation cut matching the reader's long-standing grayscale path.
_GRAY_THRESHOLD = 160


class BubbleUnavailable(RuntimeError):
    """The bubble model cannot run here, in words for the UI."""


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------


def _normalize_device(device: str | Any | None) -> str:
    """A torch device string for the bubble net.

    Accepts torch spellings (``cuda:N`` / ``mps`` / ``cpu``) and, defensively,
    the old Paddle-style ``gpu:N``. Bare ``cuda`` means card 0. Anything
    unknown — including ``None`` handled by the caller — lands on the CPU
    rather than raising.
    """
    if device is None:
        return "cpu"
    if not isinstance(device, str):
        try:
            device = str(device)
        except Exception:
            return "cpu"
    text = device.strip().lower() or "cpu"
    if text in {"cuda", "gpu"}:
        return "cuda:0"
    for prefix, canonical in (("cuda:", "cuda:"), ("gpu:", "cuda:")):
        if text.startswith(prefix):
            tail = text[len(prefix):]
            return f"{canonical}{int(tail)}" if tail.isdigit() else "cuda:0"
    if text == "mps" or text.startswith("mps"):
        return "mps"
    if text == "cpu":
        return "cpu"
    logger.warning("Unknown bubble device %r; using the CPU", device)
    return "cpu"


def _default_device() -> str:
    """Best torch device when the caller did not name one."""
    try:
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    try:
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        if mps is not None and bool(mps.is_available()):
            return "mps"
    except Exception:
        pass
    return "cpu"


def _effective_device(device: str | Any | None) -> str:
    """Requested device if usable, else the CPU with a warning.

    Unlike a multi-GB translator, this net runs fine on the CPU, so a GPU
    that is not there (or a torch build without its backend) falls back
    instead of refusing the page.
    """
    wanted = _normalize_device(device) if device is not None else _default_device()
    if wanted.startswith("cuda"):
        try:
            if torch.cuda.is_available():
                return wanted
        except Exception as exc:
            logger.debug("Could not ask torch about CUDA: %s", exc)
        logger.warning("Bubble model: %s is not usable here; using the CPU", wanted)
        return "cpu"
    if wanted == "mps":
        try:
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None) if backends is not None else None
            if mps is not None and bool(mps.is_available()):
                return wanted
        except Exception as exc:
            logger.debug("Could not ask torch about MPS: %s", exc)
        logger.warning("Bubble model: mps is not usable here; using the CPU")
        return "cpu"
    return "cpu"


# --------------------------------------------------------------------------
# Grayscale (model owns it)
# --------------------------------------------------------------------------


def _to_grayscale(img: Image.Image) -> Image.Image:
    """Binarised grayscale exactly as the reader has always done it.

    Luminance, hard threshold at 160, back to RGB so the net always sees
    three channels. Kept here so the service and any future caller cannot
    drift apart pixel by pixel.
    """
    img = img.convert("L")
    img = img.point(lambda p: 255 if p > _GRAY_THRESHOLD else 0, mode="1")
    return img.convert("RGB")


# --------------------------------------------------------------------------
# Basic blocks
# --------------------------------------------------------------------------


def _autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class _ConvBlock(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        if act is True:
            self.act = nn.SiLU(inplace=True)
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def _fuse_block(m):
    """Fold batch-norm into conv, matching reference inference exactly."""
    if not isinstance(m, _ConvBlock):
        return
    if isinstance(m.bn, nn.Identity):
        return
    conv, bn = m.conv, m.bn
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device)
    # Under no_grad on purpose: `copy_` from a source that requires grad would
    # otherwise taint the destination into a non-leaf grad-tracking tensor, and
    # every later `.to()` then warns while reading its `.grad` (torch#30531).
    # Fusion is pure weight plumbing, so no history is ever wanted here.
    with torch.no_grad():
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    m.conv = fused
    m.bn = nn.Identity()


class _DWConvBlock(_ConvBlock):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        import math
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class _Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = _ConvBlock(c1, c_, k[0], 1)
        self.cv2 = _ConvBlock(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class _C3kInner(nn.Module):
    """CSP block with 3 convs and inner bottlenecks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = _ConvBlock(c1, c_, 1, 1)
        self.cv2 = _ConvBlock(c1, c_, 1, 1)
        self.cv3 = _ConvBlock(2 * c_, c2, 1)
        self.m = nn.Sequential(*(_Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class _CspStage(nn.Module):
    """CSP stage used throughout backbone and neck."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = _ConvBlock(c1, 2 * self.c, 1, 1)
        self.cv2 = _ConvBlock((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            _C3kInner(self.c, self.c, 2, shortcut, g) if c3k
            else _Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class _SpatialPool(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = _ConvBlock(c1, c_, 1, 1)
        self.cv2 = _ConvBlock(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class _Attn(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = _ConvBlock(dim, h, 1, act=False)
        self.proj = _ConvBlock(dim, dim, 1, act=False)
        self.pe = _ConvBlock(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class _PsaBlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = _Attn(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(
            _ConvBlock(c, c * 2, 1),
            _ConvBlock(c * 2, c, 1, act=False),
        )
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class _AttnRefine(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = _ConvBlock(c1, 2 * self.c, 1, 1)
        self.cv2 = _ConvBlock(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(_PsaBlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class _MaskProto(nn.Module):
    def __init__(self, c1, c_=64, c2=32):
        super().__init__()
        self.cv1 = _ConvBlock(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = _ConvBlock(c_, c_, k=3)
        self.cv3 = _ConvBlock(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class _DistIntegral(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class _Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.d = dim

    def forward(self, x):
        return torch.cat(x, self.d)


# --------------------------------------------------------------------------
# Head
# --------------------------------------------------------------------------


class _SegHead(nn.Module):
    def __init__(self, nc=1, nm=32, npr=64, reg_max=16, ch=(64, 128, 256)):
        super().__init__()
        self.nc = nc
        self.nm = nm
        self.npr = npr
        self.reg_max = reg_max
        self.nl = len(ch)
        self.stride = torch.zeros(self.nl)
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(_ConvBlock(x, c2, 3), _ConvBlock(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(_DWConvBlock(x, x, 3), _ConvBlock(x, c3, 1)),
                nn.Sequential(_DWConvBlock(c3, c3, 3), _ConvBlock(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        self.proto = _MaskProto(ch[0], npr, nm)
        c4 = max(ch[0] // 4, nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(_ConvBlock(x, c4, 3), _ConvBlock(c4, c4, 3), nn.Conv2d(c4, nm, 1))
            for x in ch
        )
        self.dfl = _DistIntegral(reg_max)
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)
        self.shape = None

    def _decode(self, boxes, scores, feats):
        anchors, strides = _make_anchors(feats, self.stride, 0.5)
        anchors = anchors.transpose(0, 1)
        strides = strides.transpose(0, 1)
        dbox = _dist2box(self.dfl(boxes), anchors.unsqueeze(0)) * strides
        return torch.cat((dbox, scores.sigmoid()), 1)

    def forward(self, x):
        bs = x[0].shape[0]
        boxes = torch.cat([self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        masks = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], dim=2)
        dbox = self._decode(boxes, scores, x)
        pred = torch.cat([dbox, masks], dim=1)
        proto = self.proto(x[0])
        return pred, proto


# --------------------------------------------------------------------------
# Full net
# --------------------------------------------------------------------------


class _BubbleNet(nn.Module):
    """Backbone + neck + head. Index layout mirrors the training checkpoint."""

    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([
            _ConvBlock(3, 16, 3, 2),                          # 0
            _ConvBlock(16, 32, 3, 2),                         # 1
            _CspStage(32, 64, n=1, c3k=False, e=0.25),        # 2
            _ConvBlock(64, 64, 3, 2),                         # 3
            _CspStage(64, 128, n=1, c3k=False, e=0.25),       # 4
            _ConvBlock(128, 128, 3, 2),                       # 5
            _CspStage(128, 128, n=1, c3k=True, e=0.5),        # 6
            _ConvBlock(128, 256, 3, 2),                       # 7
            _CspStage(256, 256, n=1, c3k=True, e=0.5),        # 8
            _SpatialPool(256, 256, k=5),                      # 9
            _AttnRefine(256, 256, n=1, e=0.5),                # 10
            nn.Upsample(scale_factor=2.0, mode="nearest"),    # 11
            _Concat(1),                                       # 12
            _CspStage(384, 128, n=1, c3k=False, e=0.5),       # 13
            nn.Upsample(scale_factor=2.0, mode="nearest"),    # 14
            _Concat(1),                                       # 15
            _CspStage(256, 64, n=1, c3k=False, e=0.5),        # 16
            _ConvBlock(64, 64, 3, 2),                         # 17
            _Concat(1),                                       # 18
            _CspStage(192, 128, n=1, c3k=False, e=0.5),        # 19
            _ConvBlock(128, 128, 3, 2),                       # 20
            _Concat(1),                                       # 21
            _CspStage(384, 256, n=1, c3k=True, e=0.5),        # 22
            _SegHead(nc=_NC, nm=_NM, npr=64, reg_max=_REG_MAX, ch=(64, 128, 256)),  # 23
        ])
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        # route table: (from,)  -1 means previous
        self._routes = {
            12: (11, 6),
            15: (14, 4),
            18: (17, 13),
            21: (20, 10),
        }

    def forward(self, x):
        feats: dict[int, Any] = {}
        out = x
        for i, m in enumerate(self.stages):
            if i in self._routes:
                a, b = self._routes[i]
                out = m([feats[a], feats[b]])
            elif isinstance(m, _Concat):
                raise RuntimeError("unrouted concat")
            elif isinstance(m, _SegHead):
                f = [feats[16], feats[19], feats[22]]
                m.stride = self.stride.to(f[0].device).to(f[0].dtype)
                pred, proto = m(f)
                return pred, proto, f
            else:
                out = m(out)
            feats[i] = out
        raise RuntimeError("head missing")


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _make_anchors(feats, strides, offset=0.5):
    anchor_points, stride_tensor = [], []
    dtype = feats[0].dtype
    for i in range(len(feats)):
        stride = strides[i]
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, dtype=dtype, device=feats[0].device) + offset
        sy = torch.arange(h, dtype=dtype, device=feats[0].device) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(feats[0].new_full((h * w, 1), stride, dtype=dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def _dist2box(distance, anchor_points, dim=1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return torch.cat([c_xy, wh], dim)


def _xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _clip_boxes(boxes, shape):
    h, w = shape[:2]
    boxes[..., 0].clamp_(0, w)
    boxes[..., 1].clamp_(0, h)
    boxes[..., 2].clamp_(0, w)
    boxes[..., 3].clamp_(0, h)
    return boxes


def _scale_boxes(img1_shape, boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
    pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
    boxes[..., 0] -= pad_x
    boxes[..., 1] -= pad_y
    boxes[..., 2] -= pad_x
    boxes[..., 3] -= pad_y
    boxes[..., 0] /= gain
    boxes[..., 1] /= gain
    boxes[..., 2] /= gain
    boxes[..., 3] /= gain
    return _clip_boxes(boxes, img0_shape)


def _clip_coords(coords, shape):
    h, w = shape[:2]
    coords[..., 0] = coords[..., 0].clip(0, w)
    coords[..., 1] = coords[..., 1].clip(0, h)
    return coords


def _scale_coords(img1_shape, coords, img0_shape):
    img0_h, img0_w = img0_shape[:2]
    img1_h, img1_w = img1_shape[:2]
    gain = min(img1_h / img0_h, img1_w / img0_w)
    pad = (
        round((img1_w - round(img0_w * gain)) / 2 - 0.1),
        round((img1_h - round(img0_h * gain)) / 2 - 0.1),
    )
    coords[..., 0] -= pad[0]
    coords[..., 1] -= pad[1]
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    return _clip_coords(coords, img0_shape)


def _crop_mask(masks, boxes):
    if boxes.device != masks.device:
        boxes = boxes.to(masks.device)
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    masks *= (r >= x1) * (r < x2)
    masks *= (c >= y1) * (c < y2)
    return masks


def _process_mask(protos, masks_in, bboxes, shape):
    c, mh, mw = protos.shape
    if masks_in.shape[0] == 0:
        return torch.zeros((0, shape[0], shape[1]), dtype=torch.uint8, device=masks_in.device)
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode="bilinear")[0]
    return _crop_mask(masks.gt_(0.0).byte(), bboxes)


def _min_index(arr1, arr2):
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def _merge_multi(segments):
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]
    for i in range(1, len(segments)):
        idx1, idx2 = _min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)
    for k in range(2):
        if k == 0:
            for i, idx in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]
                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]: idx[1] + 1])
        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def _masks2segments(masks):
    masks = masks.byte().cpu().numpy()
    segments = []
    for x in np.ascontiguousarray(masks):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if len(c) > 1:
                c = np.concatenate(_merge_multi([xi.reshape(-1, 2) for xi in c])) if len(c) > 1 else c[0].reshape(-1, 2)
            else:
                c = c[0].reshape(-1, 2)
        else:
            c = np.zeros((0, 2))
        segments.append(c.astype("float32"))
    return segments


def _torch_nms(boxes, scores, thresh):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(0, descending=True)
    keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
    idx = 0
    while order.numel() > 0:
        i = order[0]
        keep[idx] = i
        idx += 1
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        w = (xx2 - xx1).clamp_(min=0)
        h = (yy2 - yy1).clamp_(min=0)
        inter = w * h
        if inter.sum() == 0:
            order = rest
            continue
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= thresh]
    return keep[:idx]


def _nms_boxes(boxes, scores, thresh):
    try:
        import torchvision.ops as _ops
        return _ops.nms(boxes, scores, thresh)
    except Exception:
        return _torch_nms(boxes, scores, thresh)


def _non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.7, max_det=300):
    bs = prediction.shape[0]
    nc = 1
    extra = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres
    prediction = prediction.transpose(-1, -2)
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        filt = xc[xi]
        x = x[filt]
        if not x.shape[0]:
            continue
        x[:, :4] = _xywh2xyxy(x[:, :4])
        box, cls, mask = x.split((4, nc, extra), 1)
        conf, j = cls.max(1, keepdim=True)
        filt2 = conf.view(-1) > conf_thres
        x = torch.cat((box, conf, j.float(), mask), 1)[filt2]
        if not x.shape[0]:
            continue
        n = x.shape[0]
        if n > 30000:
            x = x[x[:, 4].argsort(descending=True)[:30000]]
        c = x[:, 5:6] * 7680
        boxes, scores = x[:, :4] + c, x[:, 4]
        keep = _nms_boxes(boxes, scores, iou_thres)
        keep = keep[:max_det]
        output[xi] = x[keep]
    return output


# --------------------------------------------------------------------------
# Pre / post
# --------------------------------------------------------------------------


def _letterbox(img, new_shape=(_IMG, _IMG), stride=32):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (round(shape[1] * r), round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw % stride, dh % stride
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(_PAD_VALUE,) * 3)
    return img


def _preprocess_pil(image: Image.Image, device: torch.device | str):
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image)[:, :, ::-1]  # RGB -> BGR
    orig_shape = arr.shape[:2]  # h, w
    boxed = _letterbox(arr, new_shape=(_IMG, _IMG), stride=32)
    t = torch.from_numpy(boxed).to(device)
    t = t.permute(2, 0, 1).contiguous().float().div_(255.0)
    t = t.flip(0)  # BGR -> RGB
    return t.unsqueeze(0), orig_shape, boxed.shape[:2]


# --------------------------------------------------------------------------
# Public model
# --------------------------------------------------------------------------


class BubbleModel(nn.Module):
    """Single-class speech-bubble segmentation net (torch + safetensors)."""

    def __init__(self, weights: str | Path | None = None, device: str | Any | None = None):
        super().__init__()
        self.net = _BubbleNet()
        self.device = torch.device(_effective_device(device) if device is not None else _default_device())
        self.names = {0: "bubble"}
        if weights is not None:
            self.load_weights(weights)
            self.fuse()
        try:
            self.to(self.device)
        except Exception as exc:
            if str(self.device) != "cpu":
                logger.warning("Bubble model: could not use %s (%s); using the CPU", self.device, exc)
                self.device = torch.device("cpu")
                self.to(self.device)
            else:
                raise
        self.eval()

    def fuse(self):
        for m in self.modules():
            if isinstance(m, _ConvBlock):
                _fuse_block(m)
        return self

    def load_weights(self, path: str | Path):
        if _load_safe is None:
            raise BubbleUnavailable(
                "the bubble model needs the safetensors package; reinstall the app's dependencies"
            )
        candidate = Path(path)
        if candidate.is_dir():
            candidate = candidate / WEIGHTS_FILE
        if not candidate.is_file():
            raise BubbleUnavailable(
                f"the bubble weights are missing ({candidate.name}); "
                "download them again from the setup page"
            )
        try:
            tensors = _load_safe(str(candidate), device="cpu")
        except Exception as exc:
            raise BubbleUnavailable(
                f"the bubble weights in {candidate.name} could not be read "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        state = self.state_dict()
        mapped = {}
        try:
            for k, v in tensors.items():
                if k not in state:
                    raise BubbleUnavailable(f"unexpected tensor {k} in {candidate.name}")
                if tuple(state[k].shape) != tuple(v.shape):
                    raise BubbleUnavailable(f"shape mismatch for {k} in {candidate.name}")
                mapped[k] = v
        except BubbleUnavailable:
            raise
        except Exception as exc:
            raise BubbleUnavailable(
                f"the bubble weights in {candidate.name} could not be checked "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        # Buffers unused in eval (running-stats counter) and the fixed
        # integral kernel are recomputed locally, so they may be absent.
        skip = ("num_batches_tracked", "stages.23.dfl.conv.weight")
        missing = [k for k in state.keys() if k not in mapped and not any(s in k for s in skip)]
        if missing:
            raise BubbleUnavailable(f"the bubble weights are incomplete (missing {missing[0]})")
        try:
            self.load_state_dict(mapped, strict=False)
        except Exception as exc:
            raise BubbleUnavailable(
                f"the bubble weights in {candidate.name} could not be loaded "
                f"({type(exc).__name__}: {exc})"
            ) from exc

    @torch.no_grad()
    def raw(self, image: Image.Image | np.ndarray, conf: float = _CONF, iou: float = _IOU):
        if isinstance(image, np.ndarray):
            # Assume an RGB array.
            image = Image.fromarray(image)
        tensor, orig_shape, boxed_shape = _preprocess_pil(image, self.device)
        pred, proto, _ = self.net(tensor)
        dets = _non_max_suppression(pred.float().cpu(), conf_thres=conf, iou_thres=iou, max_det=_MAX_DET)
        det = dets[0]
        if det.shape[0] == 0:
            return {
                "boxes": torch.zeros((0, 6)),
                "masks": torch.zeros((0, boxed_shape[0], boxed_shape[1]), dtype=torch.uint8),
                "segments": [],
                "orig_shape": orig_shape,
                "boxed_shape": boxed_shape,
            }
        masks = _process_mask(proto[0].float().cpu(), det[:, 6:].float().cpu(), det[:, :4].float().cpu(), boxed_shape)
        keep = masks.amax((-2, -1)) > 0
        det = det[keep]
        masks = masks[keep]
        # det boxes are still in boxed coords; map them back to original size.
        scaled = _scale_boxes(boxed_shape, det[:, :4].clone(), orig_shape)
        boxes = torch.cat([scaled, det[:, 4:6]], dim=1)
        segments = [_scale_coords(masks.shape[1:], s.copy(), orig_shape) for s in _masks2segments(masks)]
        return {
            "boxes": boxes,
            "masks": masks,
            "segments": segments,
            "orig_shape": orig_shape,
            "boxed_shape": boxed_shape,
        }

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image | np.ndarray,
        grayscale: bool = False,
        conf: float = _CONF,
        iou: float = _IOU,
    ):
        """Polygons in original-image pixels, one dict per bubble."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if grayscale:
            image = _to_grayscale(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        out = self.raw(image, conf=conf, iou=iou)
        orig_h, orig_w = out["orig_shape"]
        response = []
        for poly in out["segments"]:
            if poly.shape[0] == 0:
                continue
            canvas = np.zeros((orig_h, orig_w), dtype=np.uint8)
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [pts], 255)
            dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
            _, padded = cv2.threshold(dist, 5, 255, cv2.THRESH_BINARY)
            padded = padded.astype(np.uint8)
            contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            for contour in contours:
                coords = [{"x": int(p[0][0]), "y": int(p[0][1])} for p in contour]
                if len(coords) < 3:
                    continue
                response.append({"type": "polygon", "coords": coords})
        return response


# --------------------------------------------------------------------------
# Cached construction for the service
# --------------------------------------------------------------------------

_lock = threading.Lock()
_cached: dict[tuple[str, str], BubbleModel] = {}


def _resolve_weights(model_path: Path | str) -> Path:
    candidate = Path(model_path)
    if candidate.is_dir():
        candidate = candidate / WEIGHTS_FILE
    return candidate


def getBubbleModel(model_path: Path | str, device: str | Any | None = None) -> BubbleModel:
    """Load (once per path+device) the bubble model.

    Late binding on purpose: :mod:`fox_reader.device` may not have applied the
    saved slot yet when this module is imported, so the device is resolved
    here, from the argument or from ``FOX_DEVICE.bubble``.
    """
    if device is None:
        try:
            from fox_reader.device import FOX_DEVICE
            device = FOX_DEVICE.bubble or "cpu"
        except Exception:
            device = "cpu"
    wanted = _normalize_device(device)
    # Availability can change between calls (driver unloaded); the effective
    # device, not the requested spelling, is the cache key.
    effective = _effective_device(wanted)
    key = (str(_resolve_weights(model_path)), effective)
    with _lock:
        cached = _cached.get(key)
        if cached is not None:
            return cached
        model = BubbleModel(weights=model_path, device=effective)
        # Bound the cache: a device change restarts the app anyway, so one
        # entry per path is the steady state; anything more is a leak.
        _cached.clear()
        _cached[key] = model
        return model
