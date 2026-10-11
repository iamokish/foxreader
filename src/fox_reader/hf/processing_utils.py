"""Processor kwarg typing, trimmed from ``transformers.processing_utils``.

:class:`ImagesKwargs` is the ``TypedDict`` every vendored image processor
extends to declare its own knobs::

    class PPOCRV5ServerDetImageProcessorKwargs(ImagesKwargs, total=False):
        limit_side_len: int
        limit_type: str
        max_side_limit: int

``BaseImageProcessor._set_attributes`` walks ``valid_kwargs.__annotations__``
to decide which attributes to read off the JSON and off the class defaults, so
the *names* here are load-bearing even though the types are not enforced.

``Unpack`` is re-exported from ``typing`` so the vendored
``**kwargs: Unpack[ImagesKwargs]`` annotations resolve.

Not ported: upstream wraps several of these in ``Annotated[..., validator()]``
and type-checks every call. The validators pull in a chunk of transformers'
introspection machinery to catch mistakes in hand-written call sites; here the
only callers are `fox_reader.ocr_predict` and the vendored processors
themselves, both covered by tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, Unpack

from .image_utils import ChannelDimension, PILImageResampling
from .utils.generic import TensorType

if TYPE_CHECKING:
    import torch


class ImagesKwargs(TypedDict, total=False):
    """Knobs common to every image processor."""

    do_convert_rgb: bool | None
    do_resize: bool | None
    size: int | list[int] | tuple[int, ...] | dict[str, int] | None
    default_to_square: bool | None
    crop_size: int | list[int] | tuple[int, ...] | dict[str, int] | None
    resample: "PILImageResampling | int | None"
    do_rescale: bool | None
    rescale_factor: float | None
    do_normalize: bool | None
    image_mean: float | list[float] | tuple[float, ...] | None
    image_std: float | list[float] | tuple[float, ...] | None
    do_pad: bool | None
    pad_size: int | list[int] | tuple[int, ...] | dict[str, int] | None
    do_center_crop: bool | None
    data_format: "str | ChannelDimension | None"
    input_data_format: "str | ChannelDimension | None"
    device: "str | torch.device | None"
    return_tensors: "str | TensorType | None"
    disable_grouping: bool | None


__all__ = ["ImagesKwargs", "Unpack"]
