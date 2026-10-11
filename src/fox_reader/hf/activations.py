"""Activation lookup, trimmed from ``transformers.activations``.

The vendored PP-OCR models name four activations between them -- ``relu``,
``silu``, ``hardswish`` and ``gelu`` -- but the table is kept wide enough that
a model folder copied in later is unlikely to miss one. Everything here is
either a torch built-in or a handful of lines, so breadth is nearly free.

Dropped from upstream: the kernel-from-hub decorators (which download fused
CUDA kernels at import), ``XIELUActivation`` (an optional CUDA extension), and
the deprecated aliases kept for archived third-party packages.

:data:`ACT2FN` maps a name to a fresh *instance*, matching upstream: the
vendored modules write ``self.act_fn = ACT2FN[activation]`` and then register
it as a submodule, so each lookup must return its own object.
"""

from __future__ import annotations

import functools
import math
from collections import OrderedDict

import torch
from torch import Tensor, nn


class GELUActivation(nn.Module):
    """BERT's original GELU. ``use_gelu_python`` forces the erf formulation."""

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        self.act = self._gelu_python if use_gelu_python else nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class GELUTanh(nn.Module):
    """The tanh approximation of GELU. Not a bit-exact match for plain GELU."""

    def __init__(self, use_gelu_tanh_python: bool = False):
        super().__init__()
        if use_gelu_tanh_python:
            self.act = self._gelu_tanh_python
        else:
            self.act = functools.partial(nn.functional.gelu, approximate="tanh")

    def _gelu_tanh_python(self, input: Tensor) -> Tensor:
        return (
            input
            * 0.5
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class NewGELUActivation(nn.Module):
    """GELU as used in Google BERT and OpenAI GPT, in Python."""

    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        )


class FastGELUActivation(nn.Module):
    """A faster, slightly less accurate GELU approximation."""

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    """``x * sigmoid(1.702 x)`` -- CLIP's GELU approximation."""

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """GELU clipped to ``[min, max]``, for models quantised against a range."""

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(_gelu_python(x), self.min, self.max)


class AccurateGELUActivation(nn.Module):
    """A more numerically careful GELU approximation."""

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (1 + torch.tanh(self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))))
        )


class MishActivation(nn.Module):
    """``x * tanh(softplus(x))``."""

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.mish(input)


class LinearActivation(nn.Module):
    """Identity, so ``hidden_act: "linear"`` is a valid config value."""

    def forward(self, input: Tensor) -> Tensor:
        return input


class ReLUSquaredActivation(nn.Module):
    """``relu(x) ** 2``."""

    def forward(self, input: Tensor) -> Tensor:
        return torch.square(nn.functional.relu(input))


class LaplaceActivation(nn.Module):
    """The Laplace CDF, from "Primer: Searching for Efficient Transformers"."""

    def forward(self, input: Tensor, mu: float = 0.707107, sigma: float = 0.282095) -> Tensor:
        return 0.5 * (1.0 + torch.erf((input - mu).div(sigma * math.sqrt(2.0))))


def _gelu_python(input: Tensor) -> Tensor:
    return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class ClassInstantier(OrderedDict):
    """A dict whose lookup returns a new instance, optionally with kwargs."""

    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_accurate": AccurateGELUActivation,
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_python_tanh": (GELUTanh, {"use_gelu_tanh_python": True}),
    "gelu_pytorch_tanh": GELUTanh,
    "hardswish": nn.Hardswish,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "prelu": nn.PReLU,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}

ACT2FN = ClassInstantier(ACT2CLS)

__all__ = ["ACT2CLS", "ACT2FN", "ClassInstantier"]
