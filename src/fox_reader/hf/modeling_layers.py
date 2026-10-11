"""Layer bases, trimmed from ``transformers.modeling_layers``.

:class:`GradientCheckpointingLayer` is upstream's hook for trading compute
against activation memory during *training*. Fox Reader only ever runs these
models in ``eval()`` under ``torch.inference_mode``, so the checkpointing
branch is unreachable: ``gradient_checkpointing`` stays ``False`` and
``self.training`` stays ``False``.

It is kept as a named base class rather than removed, because the vendored
model files inherit from it by name and must stay byte-identical to upstream.
The attribute is kept too, so ``set_gradient_checkpointing()``-style code
copied in later fails loudly on the missing ``_gradient_checkpointing_func``
rather than silently training without checkpointing.
"""

from __future__ import annotations

from torch import nn


class GradientCheckpointingLayer(nn.Module):
    """A plain ``nn.Module``; gradient checkpointing is not supported here."""

    gradient_checkpointing = False


__all__ = ["GradientCheckpointingLayer"]
