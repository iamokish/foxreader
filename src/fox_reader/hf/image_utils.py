"""Image types and size helpers, trimmed from ``transformers.image_utils``.

The vendored image processors import :class:`SizeDict` and
``PILImageResampling`` by name; the rest of what is here is what
:class:`~fox_reader.hf.image_processing_backends.TorchvisionBackend` needs to
turn whatever the caller handed in (PIL image, HWC numpy array, CHW tensor)
into a batch of channels-first tensors.

``PILImageResampling`` is Pillow's own ``Image.Resampling`` enum, re-exported
rather than redefined, so a ``"resample": 2`` in a ``preprocessor_config.json``
keeps meaning ``BILINEAR``.

Not ported: annotation formats (COCO detection/panoptic), video frame
handling, URL and base64 loading, and the ``is_valid_annotation_*`` family.
Fox Reader hands these processors in-memory crops it produced itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import PIL.Image
import torch
from torchvision.transforms import InterpolationMode

from .utils.generic import ExplicitEnum

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

PILImageResampling = PIL.Image.Resampling

#: Pillow resample id -> torchvision interpolation mode.
#:
#: ``NEAREST`` maps to ``NEAREST_EXACT``, not ``NEAREST``: torchvision's plain
#: ``NEAREST`` reproduces a long-standing off-by-half-pixel bug for backwards
#: compatibility, and upstream deliberately picks the corrected one. Getting
#: this wrong shifts every resized image by half a pixel.
pil_torch_interpolation_mapping = {
    PILImageResampling.NEAREST: InterpolationMode.NEAREST_EXACT,
    PILImageResampling.BOX: InterpolationMode.BOX,
    PILImageResampling.BILINEAR: InterpolationMode.BILINEAR,
    PILImageResampling.HAMMING: InterpolationMode.HAMMING,
    PILImageResampling.BICUBIC: InterpolationMode.BICUBIC,
    PILImageResampling.LANCZOS: InterpolationMode.LANCZOS,
}
torch_pil_interpolation_mapping = {v: k for k, v in pil_torch_interpolation_mapping.items()}

ImageInput = Union[
    "PIL.Image.Image",
    np.ndarray,
    "torch.Tensor",
    list["PIL.Image.Image"],
    list[np.ndarray],
    list["torch.Tensor"],
]


class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"
    LAST = "channels_last"


class ImageType(ExplicitEnum):
    PIL = "pillow"
    TORCH = "torch"
    NUMPY = "numpy"


@dataclass(frozen=True)
class SizeDict:
    """Hashable image-size record.

    Hashable because the processors use it as a dict key when grouping images
    by target shape, and because ``_fuse_mean_std_and_rescale_factor`` is
    ``lru_cache``d on its arguments.
    """

    height: int | None = None
    width: int | None = None
    longest_edge: int | None = None
    shortest_edge: int | None = None
    max_height: int | None = None
    max_width: int | None = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in SizeDict.")

    def get(self, key, default=None):
        if hasattr(self, key) and getattr(self, key) is not None:
            return getattr(self, key)
        return default

    def __iter__(self):
        # Only the set fields, so dict(size_dict) drops the Nones.
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                yield f.name, value

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            raise KeyError(f"Key {key} is not a valid field of SizeDict.")
        object.__setattr__(self, key, value)


def get_image_type(image) -> ImageType:
    if isinstance(image, PIL.Image.Image):
        return ImageType.PIL
    if isinstance(image, torch.Tensor):
        return ImageType.TORCH
    if isinstance(image, np.ndarray):
        return ImageType.NUMPY
    raise ValueError(f"Unsupported image type {type(image)}")


def is_valid_image(image) -> bool:
    return isinstance(image, (PIL.Image.Image, np.ndarray, torch.Tensor))


def infer_channel_dimension_format(
    image, num_channels: int | tuple[int, ...] | None = None
) -> ChannelDimension:
    """Guess whether channels lead or trail, from the channel count."""
    num_channels = (1, 3) if num_channels is None else num_channels
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    elif image.ndim == 5:
        first_dim, last_dim = 2, 4
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in num_channels and image.shape[last_dim] in num_channels:
        # A 3x3 or 1x1 image. Upstream assumes channels-first; so do we.
        logger.warning(
            "Ambiguous channel dimension for image shape %s; assuming channels first.",
            tuple(image.shape),
        )
        return ChannelDimension.FIRST
    if image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    if image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError(f"Unable to infer channel dimension format for shape {tuple(image.shape)}")


def get_image_size(image, channel_dim: ChannelDimension | None = None) -> tuple[int, int]:
    """``(height, width)`` of a single image."""
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)
    if channel_dim == ChannelDimension.FIRST:
        return int(image.shape[-2]), int(image.shape[-1])
    return int(image.shape[-3]), int(image.shape[-2])


def get_max_height_width(images: list) -> tuple[int, int]:
    """The bounding ``(height, width)`` over a list of CHW tensors."""
    max_height = max(int(image.shape[-2]) for image in images)
    max_width = max(int(image.shape[-1]) for image in images)
    return max_height, max_width


def make_flat_list_of_images(images: Any, expected_ndims: int = 3) -> list:
    """Normalise whatever was passed into a flat list of single images."""
    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            return []
        if all(isinstance(image, (list, tuple)) for image in images):
            return [image for nested in images for image in nested]
        return list(images)
    if isinstance(images, PIL.Image.Image):
        return [images]
    if isinstance(images, (np.ndarray, torch.Tensor)):
        # A batch arrives one dimension deeper than a single image.
        if images.ndim == expected_ndims + 1:
            return list(images)
        return [images]
    raise ValueError(f"Unsupported image input type {type(images)}")


def convert_to_rgb(image):
    """Drop alpha / expand grayscale, for whichever representation this is."""
    if isinstance(image, PIL.Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    # numpy and torch arrive channel-last or channel-first; the backend has
    # not normalised them yet, so only the unambiguous cases are handled and
    # anything else is left for the caller's own conversion.
    return image


__all__ = [
    "ChannelDimension",
    "ImageInput",
    "ImageType",
    "PILImageResampling",
    "SizeDict",
    "convert_to_rgb",
    "get_image_size",
    "get_image_type",
    "get_max_height_width",
    "infer_channel_dimension_format",
    "is_valid_image",
    "make_flat_list_of_images",
    "pil_torch_interpolation_mapping",
    "torch_pil_interpolation_mapping",
]
