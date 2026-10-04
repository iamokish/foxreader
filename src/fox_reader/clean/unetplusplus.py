"""UNet++ over EfficientNetV2-RW-M, with no ``segmentation_models_pytorch``.

The Text Seg checkpoint (Manga-Text-Segmentation-2025) was trained against
``smp.UnetPlusPlus(encoder_name="tu-efficientnetv2_rw_m", ...)``, and loading it
used to mean installing ``segmentation-models-pytorch`` and ``timm`` -- two model
zoos, a few hundred megabytes of wheels and a second copy of every architecture
in the world, to instantiate exactly one of them. This module is that one
architecture, written out, so the extra can go away.

It is a *replica*, not a reimplementation: every parameter tensor has the same
name, shape and position it has in the packages, so the checkpoint loads with
``strict=True`` and the forward pass is arithmetically the same graph. Three
things had to hold for that, and all three are asserted in
``tests/test_clean_unetplusplus.py``:

* **names.** ``load_state_dict(strict=True)`` matches on strings. The encoder
  therefore keeps timm's layout -- an ``encoder.model.`` prefix, ``conv_stem`` /
  ``bn1`` / ``blocks.<stage>.<block>.<part>`` -- and the decoder keeps smp's,
  including its ``nn.Sequential`` index keys (``conv1.0`` is the convolution,
  ``conv1.1`` the norm) and the ``attention.attention`` double hop that smp's
  ``Attention`` wrapper produces. None of that is how one would spell it from
  scratch. All of it is load-bearing.
* **shapes.** The channel widths are written out as a table
  (:data:`_STAGES`) rather than derived from timm's architecture-definition
  strings and rounding helpers. The table was read back off a built
  ``smp.UnetPlusPlus``, which is the only definition that matters here; the
  generator that produced it is not, and reproducing it would be a second
  chance to get the rounding wrong.
* **arithmetic.** Block bodies follow timm's ``EdgeResidual`` and
  ``InvertedResidual`` and smp's ``DecoderBlock`` operation for operation,
  including the parts that look like waste. See ``_DecoderBlock`` for the one
  that is (an attention module that is built and never called).

What is deliberately *not* replicated is everything that only matters for
training or for the other 200 encoders: pretrained-weight download, timm's
feature-hook machinery, drop-path, anti-aliased downsampling, the classification
head, output strides other than 32, and the ``activation=`` string-to-module
lookup. Text Seg does one forward pass over a crop with weights it already has,
and the parts left out contribute no parameters -- which is why leaving them out
does not move a single key.

``torch`` is imported at module scope, not lazily as in :mod:`fox_reader.clean.seg`.
This module *is* the lazy boundary: nothing imports it until
``seg._build_model`` does, so the app and the health endpoint still start on a
machine with no torch installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ENCODER_NAME", "UnetPlusPlus"]

#: The one encoder this module is. Named so that a caller can pass it the way it
#: passed smp's encoder name and get a loud failure rather than a silent
#: substitution if the checkpoint ever changes backbone.
ENCODER_NAME = "tu-efficientnetv2_rw_m"

# --------------------------------------------------------------- encoder parts


class _BatchNormAct2d(nn.BatchNorm2d):
    """timm's ``BatchNormAct2d``: a BatchNorm2d with the activation folded in.

    A subclass rather than ``Sequential(bn, act)`` because the parameter names
    have to stay ``bn1.weight`` and not ``bn1.0.weight``. ``nn.BatchNorm2d``
    already holds every tensor timm's version holds, and timm's ``forward`` is a
    copy of ``nn.BatchNorm2d``'s with the activation appended, so inheriting is
    both the shortest and the exact answer.

    Being a ``BatchNorm2d`` subclass does mean
    :func:`fox_reader.clean.seg._convert_batchnorm_to_groupnorm` would replace
    these -- and drop their activation with them -- if it were ever pointed at
    the encoder. It is pointed at the decoder, which is where the checkpoint has
    GroupNorm parameters; a converted encoder would fail ``strict=True`` on the
    running statistics it no longer has, so the mistake cannot pass quietly.
    """

    def __init__(self, num_features: int, apply_act: bool = True) -> None:
        super().__init__(num_features)
        # Registered as a child, unparameterised, exactly as timm has it: it
        # contributes no state-dict key either way, and a `print(model)` that
        # matches the package's is worth having when comparing the two.
        self.act: nn.Module = nn.SiLU(inplace=True) if apply_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The named local is for the type checker, here and in the forwards
        # below: torch types `Module.__call__` as returning `Any`, so handing
        # one straight back is a return of `Any` from a function that promises a
        # tensor. Naming it is the annotation.
        out: torch.Tensor = self.act(super().forward(x))
        return out


class _SqueezeExcite(nn.Module):
    """timm's EfficientNet-family squeeze-excite.

    Note which channel count sets the bottleneck: ``rd_channels`` comes from the
    *block's* input width, not from the expanded width the module actually sees
    (timm's ``se_from_exp=False``, which is what every EfficientNetV2 uses). At
    stage 5 that is the difference between 82 channels and 492.
    """

    def __init__(self, channels: int, rd_channels: int) -> None:
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, rd_channels, 1, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `mean`, not an AdaptiveAvgPool2d module -- same arithmetic, and it is
        # what timm does, so the module tree matches too.
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        gate: torch.Tensor = self.gate(x_se)
        return x * gate


class _EdgeResidual(nn.Module):
    """timm's ``EdgeResidual`` -- FusedMBConv, the V2 paper's name for it.

    One k3 convolution does the expansion *and* the spatial mixing that MBConv
    splits into a 1x1 and a depthwise; at the resolutions of the first three
    stages that is faster on real hardware despite the extra multiplies.
    """

    def __init__(self, in_chs: int, out_chs: int, mid_chs: int,
                 kernel_size: int, stride: int) -> None:
        super().__init__()
        self.has_skip = in_chs == out_chs and stride == 1
        self.conv_exp = nn.Conv2d(in_chs, mid_chs, kernel_size, stride=stride,
                                  padding=kernel_size // 2, bias=False)
        self.bn1 = _BatchNormAct2d(mid_chs)
        self.conv_pwl = nn.Conv2d(mid_chs, out_chs, 1, bias=False)
        self.bn2 = _BatchNormAct2d(out_chs, apply_act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.bn1(self.conv_exp(x))
        x = self.bn2(self.conv_pwl(x))
        if self.has_skip:
            x = x + shortcut
        return x


class _InvertedResidual(nn.Module):
    """timm's ``InvertedResidual`` -- MBConv, with squeeze-excite."""

    def __init__(self, in_chs: int, out_chs: int, mid_chs: int,
                 kernel_size: int, stride: int, se_channels: int) -> None:
        super().__init__()
        self.has_skip = in_chs == out_chs and stride == 1
        self.conv_pw = nn.Conv2d(in_chs, mid_chs, 1, bias=False)
        self.bn1 = _BatchNormAct2d(mid_chs)
        self.conv_dw = nn.Conv2d(mid_chs, mid_chs, kernel_size, stride=stride,
                                 padding=kernel_size // 2, groups=mid_chs,
                                 bias=False)
        self.bn2 = _BatchNormAct2d(mid_chs)
        self.se: nn.Module = (_SqueezeExcite(mid_chs, se_channels)
                              if se_channels else nn.Identity())
        self.conv_pwl = nn.Conv2d(mid_chs, out_chs, 1, bias=False)
        self.bn3 = _BatchNormAct2d(out_chs, apply_act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.bn1(self.conv_pw(x))
        x = self.bn2(self.conv_dw(x))
        x = self.se(x)
        x = self.bn3(self.conv_pwl(x))
        if self.has_skip:
            x = x + shortcut
        return x


#: ``efficientnetv2_rw_m``'s stem width.
_STEM_CHANNELS = 32

#: The backbone, stage by stage: ``(block, repeats, out_chs, kernel, stride,
#: exp_ratio, se_ratio)``. Read off a built ``smp.UnetPlusPlus``; see the module
#: docstring for why it is a table and not a generator.
#:
#: Only the first block of a stage takes the stride and sees the previous
#: stage's width -- the repeats after it are ``out_chs -> out_chs`` at stride 1,
#: which is also what makes them residual.
_STAGES: tuple[tuple[str, int, int, int, int, float, float], ...] = (
    ("fused",   3,  32, 3, 1, 1.0, 0.0),
    ("fused",   5,  56, 3, 2, 4.0, 0.0),
    ("fused",   5,  80, 3, 2, 4.0, 0.0),
    ("mbconv",  8, 152, 3, 2, 4.0, 0.25),
    ("mbconv", 15, 192, 3, 1, 6.0, 0.25),
    ("mbconv", 24, 328, 3, 2, 6.0, 0.25),
)

#: Which stage outputs are handed to the decoder. Stage 4 is absent because it
#: has stride 1: stages 3 and 4 are both at 1/16 of the input, and a U-net wants
#: one feature map per resolution, so timm's ``feature_info`` keeps the deeper
#: of the two. Skipping it is the whole reason this cannot be
#: ``enumerate(self.blocks)``.
_FEATURE_STAGES = (0, 1, 2, 4, 5)


class _EfficientNetV2Features(nn.Module):
    """timm's ``EfficientNetFeatures``: the backbone with no classifier.

    ``features_only=True`` drops timm's ``conv_head``/``bn2``/``classifier``
    outright rather than leaving them unused, which is why the checkpoint has no
    parameters for them and why this class stops at ``blocks``.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv_stem = nn.Conv2d(in_channels, _STEM_CHANNELS, 3, stride=2,
                                   padding=1, bias=False)
        self.bn1 = _BatchNormAct2d(_STEM_CHANNELS)

        stages: list[nn.Module] = []
        chs = _STEM_CHANNELS
        for kind, repeats, out_chs, kernel, stride, exp, se in _STAGES:
            blocks: list[nn.Module] = []
            for i in range(repeats):
                in_chs = chs if i == 0 else out_chs
                # Every width in this backbone is already a multiple of 8, so
                # timm's `make_divisible` rounding is the identity here and
                # plain multiplication reproduces it. The test pins the shapes,
                # so a future checkpoint where that stops being true fails
                # rather than loads something subtly different.
                mid_chs = int(in_chs * exp)
                se_chs = round(in_chs * se) if se else 0
                if kind == "fused":
                    blocks.append(_EdgeResidual(
                        in_chs, out_chs, mid_chs, kernel,
                        stride if i == 0 else 1))
                else:
                    blocks.append(_InvertedResidual(
                        in_chs, out_chs, mid_chs, kernel,
                        stride if i == 0 else 1, se_chs))
            stages.append(nn.Sequential(*blocks))
            chs = out_chs
        self.blocks = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.bn1(self.conv_stem(x))
        features = []
        for i, stage in enumerate(self.blocks):
            x = stage(x)
            if i in _FEATURE_STAGES:
                features.append(x)
        return features


class _TimmUniversalEncoder(nn.Module):
    """smp's ``TimmUniversalEncoder``, narrowed to this one backbone.

    Two details of the wrapper are visible from outside and both matter. The
    backbone hangs off ``self.model``, which is where the ``encoder.model.``
    prefix in every encoder key comes from. And ``forward`` returns the *input*
    as the first feature: the decoder's channel bookkeeping counts it (hence the
    leading ``3`` in :attr:`out_channels`) and then discards it, because at
    full resolution there is no decoder stage left to consume it.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.model = _EfficientNetV2Features(in_channels)
        self.out_channels: list[int] = (
            [in_channels] + [_STAGES[i][2] for i in _FEATURE_STAGES])
        self.output_stride = 32

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = self.model(x)
        return [x] + features


# --------------------------------------------------------------- decoder parts


class _Conv2dReLU(nn.Sequential):
    """smp's ``Conv2dReLU``.

    Unnamed on purpose: smp builds it as a bare ``Sequential``, so the
    checkpoint's keys are ``conv1.0.weight`` for the convolution and
    ``conv1.1.*`` for the norm. Giving these children names would read better
    and would not load.

    ``bias=False`` because a norm follows and would cancel it.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _SCSEModule(nn.Module):
    """smp's concurrent spatial and channel squeeze-excite.

    The two gates are summed, not composed: ``x * cSE + x * sSE``. It is worth
    saying because the obvious reading -- gate the channels, then gate the
    pixels -- is a different function.
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel: torch.Tensor = self.cSE(x)
        spatial: torch.Tensor = self.sSE(x)
        return x * channel + x * spatial


class _Attention(nn.Module):
    """smp's ``Attention``: a one-of-several dispatcher around the real module.

    Kept even though there is only one choice here, because it is what puts the
    second ``attention`` in ``attention1.attention.cSE.1.weight``. Flattening it
    would rename twelve tensors per decoder block.
    """

    def __init__(self, name: str | None, in_channels: int) -> None:
        super().__init__()
        if name is None:
            self.attention: nn.Module = nn.Identity()
        elif name == "scse":
            self.attention = _SCSEModule(in_channels=in_channels)
        else:
            raise ValueError(f"Attention {name} is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.attention(x)
        return out


class _DecoderBlock(nn.Module):
    """smp's UNet++ ``DecoderBlock``: upsample, concatenate, gate, convolve.

    ``attention1`` is built whether or not a skip will ever arrive, and the
    final block of the ladder is called with ``skip=None``. So that block
    carries an attention module it can never use -- twelve tensors of dead
    weight in the checkpoint. They are replicated because they are *in* the
    checkpoint: ``strict=True`` wants them, and inference does not care that
    nothing reads them.
    """

    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int, attention_type: str | None,
                 interpolation_mode: str = "nearest") -> None:
        super().__init__()
        self.conv1 = _Conv2dReLU(in_channels + skip_channels, out_channels)
        self.attention1 = _Attention(attention_type,
                                     in_channels=in_channels + skip_channels)
        self.conv2 = _Conv2dReLU(out_channels, out_channels)
        self.attention2 = _Attention(attention_type, in_channels=out_channels)
        self.interpolation_mode = interpolation_mode

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode=self.interpolation_mode)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class _UnetPlusPlusDecoder(nn.Module):
    """smp's ``UnetPlusPlusDecoder``: the dense grid of nested skip paths.

    UNet++'s difference from a U-net is that a skip does not go straight across.
    Encoder level *j* feeds a row of blocks ``x_0_j, x_1_j, ... x_j_j``, each
    taking the *concatenation* of every earlier output on its own row plus the
    encoder feature one level down -- so the deepest features reach the top of
    the ladder through several partly-decoded paths rather than one raw one.
    That is where the ``skip_channels[j] * (j + 1 - depth)`` widths come from:
    a block on row *j* is fed however many of its own row's siblings precede it.

    The construction is smp's loop rather than a table of the eleven resulting
    blocks. It is short, it is the definition the widths follow from, and
    transcribing eleven ``(in, skip, out)`` triples by hand is how one of them
    ends up off by 8 channels.
    """

    def __init__(self, encoder_channels: list[int],
                 decoder_channels: tuple[int, ...],
                 n_blocks: int = 5, attention_type: str | None = None,
                 interpolation_mode: str = "nearest") -> None:
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provide "
                f"`decoder_channels` for {len(decoder_channels)} blocks.")

        # Drop the full-resolution feature (the input image) and count down from
        # the deepest, which is the order the ladder is built in.
        channels = list(encoder_channels[1:])[::-1]
        head_channels = channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(channels[1:]) + [0]
        self.out_channels = list(decoder_channels)

        blocks: dict[str, nn.Module] = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (
                        layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = _DecoderBlock(
                    in_ch, skip_ch, out_ch, attention_type, interpolation_mode)
        # The last rung has no encoder feature left to concatenate: it is the
        # one that brings the map back to full resolution.
        last = len(self.in_channels) - 1
        blocks[f"x_0_{last}"] = _DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], attention_type,
            interpolation_mode)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = last

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        features = features[1:][::-1]

        dense_x: dict[str, torch.Tensor] = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = (
                        self.blocks[f"x_{depth_idx}_{depth_idx}"](
                            features[depth_idx], features[depth_idx + 1]))
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"]
                                    for idx in range(depth_idx + 1,
                                                     dense_l_i + 1)]
                    # smp reuses the `cat_features` name for the tensor; a
                    # second name says what it is, which is the skip argument.
                    skip = torch.cat(
                        cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
                        f"x_{depth_idx}_{dense_l_i}"](
                            dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], skip)
        key = f"x_0_{self.depth}"
        dense_x[key] = self.blocks[key](dense_x[f"x_0_{self.depth - 1}"])
        return dense_x[key]


class _SegmentationHead(nn.Sequential):
    """smp's ``SegmentationHead``, minus the parts this configuration disables.

    The two ``Identity`` children are smp's optional upsampling and activation.
    They are here so the module tree matches; neither holds a tensor, so the
    checkpoint's ``segmentation_head.0.weight`` lands the same either way.
    Logits come out -- :func:`fox_reader.clean.seg._forward` applies the sigmoid.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.Identity(),
            nn.Identity(),
        )


# --------------------------------------------------------------------- the net


def _initialize_decoder(module: nn.Module) -> None:
    """smp's decoder initialisation.

    Dead weight for Text Seg, which overwrites all of it from the checkpoint a
    moment later. Kept because a randomly-initialised net that is *not* the one
    the package builds is a trap for anyone who ever fine-tunes from here, and
    because it costs a millisecond once per process.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in",
                                     nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm,
                            nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def _initialize_head(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class UnetPlusPlus(nn.Module):
    """Drop-in for ``smp.UnetPlusPlus`` at the one configuration Text Seg uses.

    The signature is smp's, keyword for keyword, so the call site reads the same
    as it did against the package -- and so that anything smp supported but this
    does not is a ``ValueError`` at construction rather than a wrong answer. The
    encoder is fixed, weights always start random (the caller loads a
    checkpoint), and ``activation`` stays ``None`` because the sigmoid belongs in
    the inference path where it can be applied under ``inference_mode``.

    ``encoder``, ``decoder`` and ``segmentation_head`` are named as smp names
    them: the checkpoint keys begin with those words, and
    :func:`fox_reader.clean.seg._build_model` reaches for ``.decoder`` to swap
    its BatchNorms for GroupNorms before loading.
    """

    def __init__(self, encoder_name: str = ENCODER_NAME,
                 encoder_weights: str | None = None,
                 decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
                 decoder_attention_type: str | None = None,
                 decoder_interpolation: str = "nearest",
                 in_channels: int = 3, classes: int = 1,
                 activation: str | None = None) -> None:
        super().__init__()
        if encoder_name != ENCODER_NAME:
            raise ValueError(
                f"{type(self).__name__} is only {ENCODER_NAME}; "
                f"got {encoder_name!r}")
        if encoder_weights is not None:
            raise ValueError(
                "encoder_weights must be None: this module has no weight "
                "download, and Text Seg loads a full checkpoint over the "
                "encoder anyway")
        if activation is not None:
            raise ValueError(
                "activation must be None: the forward pass returns logits")

        self.encoder = _TimmUniversalEncoder(in_channels=in_channels)
        self.decoder = _UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation,
        )
        self.segmentation_head = _SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=classes,
            kernel_size=3)

        self.name = f"unetplusplus-{encoder_name}"
        _initialize_decoder(self.decoder)
        _initialize_head(self.segmentation_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Every decoder rung doubles the resolution, so a side that is not a
        # multiple of 32 comes back a pixel or two short and the concatenations
        # fail somewhere deep with a shape error about numbers the caller never
        # chose. `seg._forward` pads before calling, so this only ever fires for
        # a new caller -- which is exactly who needs the sentence.
        h, w = x.shape[-2:]
        stride = self.encoder.output_stride
        if h % stride or w % stride:
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image "
                f"height and width divisible by {stride}. Consider pad your "
                f"images to shape "
                f"({-(-h // stride) * stride}, {-(-w // stride) * stride}).")
        logits: torch.Tensor = self.segmentation_head(
            self.decoder(self.encoder(x)))
        return logits
