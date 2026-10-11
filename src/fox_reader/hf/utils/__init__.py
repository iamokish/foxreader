"""The ``...utils`` surface the vendored model files import.

Five names, re-exported or defined here. Two are documentation machinery that
Fox Reader turns into no-ops (:func:`auto_docstring`, and ``requires`` in
:mod:`.import_utils`) -- upstream uses them to assemble the published API docs
and to produce tidy "install this extra" errors. Neither job exists here: the
docs are not built and the backends are hard dependencies.

:func:`requires_backends` stays a real check. It is cheap, it runs once per
``post_process_*`` call, and its failure mode -- a missing OpenCV in a Nuitka
build that dropped it -- is exactly the kind of thing worth an explicit
sentence rather than an ``AttributeError`` inside a contour loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from .constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from .generic import ExplicitEnum, TensorType, can_return_tuple, merge_with_config_defaults
from .import_utils import (
    is_cv2_available,
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
    requires,
)

if TYPE_CHECKING:
    import torch


class TransformersKwargs(TypedDict, total=False):
    """Keyword arguments threaded through a model's forward pass.

    Only ``output_hidden_states`` is load-bearing for the vendored PP-OCR
    models -- the backbones force it on to build their feature maps. The rest
    are kept so a model copied in later annotates against the same shape.
    """

    num_items_in_batch: "torch.Tensor | None"
    output_hidden_states: bool | None
    output_attentions: bool | None
    output_router_logits: bool | None
    cu_seq_lens_q: "torch.LongTensor | None"
    cu_seq_lens_k: "torch.LongTensor | None"
    max_length_q: int | None
    max_length_k: int | None
    position_ids: "torch.LongTensor | None"
    is_causal: bool | None
    seq_idx: "torch.IntTensor | None"


#: Backends every vendored file can assume. Anything outside this set is a real
#: question, so `requires_backends` looks it up rather than waving it through.
_BACKEND_CHECKS = {
    "torch": is_torch_available,
    "torchvision": is_torchvision_available,
    "vision": is_vision_available,
    "cv2": is_cv2_available,
}


def requires_backends(obj: Any, backends: str | list[str] | tuple[str, ...]) -> None:
    """Raise a readable ImportError when a needed backend is absent."""
    if isinstance(backends, str):
        backends = [backends]
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    missing = [b for b in backends if b in _BACKEND_CHECKS and not _BACKEND_CHECKS[b]()]
    if missing:
        raise ImportError(
            f"{name} needs {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed. "
            "Reinstall the app's dependencies."
        )


def auto_docstring(obj=None, **kwargs: Any):
    """No-op stand-in for transformers' docstring generator.

    Upstream rewrites ``__doc__`` from a template so huggingface.co renders a
    uniform API reference. Nothing here reads those docstrings, and the
    generator pulls in a large chunk of transformers' introspection machinery.

    Usable bare (``@auto_docstring``) or called
    (``@auto_docstring(checkpoint="...")``), because the vendored files use
    both spellings.
    """
    if obj is None:
        return lambda inner: inner
    return obj


__all__ = [
    "IMAGENET_DEFAULT_MEAN",
    "IMAGENET_DEFAULT_STD",
    "IMAGENET_STANDARD_MEAN",
    "IMAGENET_STANDARD_STD",
    "OPENAI_CLIP_MEAN",
    "OPENAI_CLIP_STD",
    "ExplicitEnum",
    "TensorType",
    "TransformersKwargs",
    "auto_docstring",
    "can_return_tuple",
    "is_cv2_available",
    "is_torch_available",
    "is_torchvision_available",
    "is_vision_available",
    "merge_with_config_defaults",
    "requires",
    "requires_backends",
]
