"""``TorchvisionBackend``, trimmed from ``transformers.image_processing_backends``
and ``transformers.image_processing_utils``.

Every vendored PP-OCR image processor subclasses this and overrides one
method, ``_preprocess``. Everything else -- reading
``preprocessor_config.json``, resolving each knob from kwarg → instance →
class default, turning raw input into channels-first tensors, and the
resize / pad / crop / normalise primitives -- lives here.

The call chain, which the vendored ``_preprocess`` methods assume:

    processor(images, **overrides)
      -> preprocess()              fill kwargs from attributes, standardise sizes
         -> _prepare_image_like_inputs()
            -> process_image()     PIL/numpy/tensor -> CHW torch.Tensor
         -> _preprocess()          the model-specific part (vendored)

Two collapses relative to upstream. ``BaseImageProcessor`` /
``ImageProcessingMixin`` / ``TorchvisionBackend`` are folded into one class,
because the three-level split only exists to share with ``PilBackend``, which
is not vendored. And ``from_pretrained`` reads a local directory rather than
resolving against the Hub, for the same reason as in
:mod:`~fox_reader.hf.modeling_utils`: the weights are already on disk by the
time anything here runs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms.v2 import functional as tvF

from .feature_extraction_utils import BatchFeature
from .image_transforms import group_images_by_shape, reorder_images
from .image_utils import (
    ChannelDimension,
    ImageInput,
    ImageType,
    PILImageResampling,
    SizeDict,
    convert_to_rgb,
    get_image_type,
    get_max_height_width,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    pil_torch_interpolation_mapping,
)
from .processing_utils import ImagesKwargs, Unpack

IMAGE_PROCESSOR_NAME = "preprocessor_config.json"

#: Key sets a `size` dict is allowed to use.
VALID_SIZE_DICT_KEYS = (
    {"height", "width"},
    {"shortest_edge"},
    {"shortest_edge", "longest_edge"},
    {"longest_edge"},
    {"max_height", "max_width"},
)


def is_valid_size_dict(size_dict) -> bool:
    if not isinstance(size_dict, dict):
        return False
    return set(size_dict.keys()) in VALID_SIZE_DICT_KEYS


def convert_to_size_dict(
    size: int | Iterable[int] | None = None,
    max_size: int | None = None,
    default_to_square: bool = True,
    height_width_order: bool = True,
) -> dict[str, int]:
    """Normalise the several shapes a ``size`` may arrive in."""
    if isinstance(size, int) and default_to_square:
        if max_size is not None:
            raise ValueError("Cannot give size as an int with default_to_square=True and a max_size")
        return {"height": size, "width": size}
    if isinstance(size, int) and not default_to_square:
        size_dict = {"shortest_edge": size}
        if max_size is not None:
            size_dict["longest_edge"] = max_size
        return size_dict
    if isinstance(size, (tuple, list)) and height_width_order:
        return {"height": size[0], "width": size[1]}
    if isinstance(size, (tuple, list)) and not height_width_order:
        return {"height": size[1], "width": size[0]}
    if size is None and max_size is not None:
        if default_to_square:
            raise ValueError("Cannot give default_to_square=True and a max_size with no size")
        return {"longest_edge": max_size}
    raise ValueError(f"Could not convert to a size dict: {size!r}")


def get_size_dict(
    size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
    max_size: int | None = None,
    height_width_order: bool = True,
    default_to_square: bool = True,
    param_name: str = "size",
) -> dict:
    """A validated ``{"height": ..., "width": ...}``-style dict."""
    if isinstance(size, SizeDict):
        size_dict = dict(size)
    elif isinstance(size, dict):
        size_dict = size
    else:
        size_dict = convert_to_size_dict(size, max_size, default_to_square, height_width_order)
    if not is_valid_size_dict(size_dict):
        raise ValueError(
            f"{param_name} must use one of these key sets: {VALID_SIZE_DICT_KEYS}, "
            f"got {sorted(size_dict.keys())}"
        )
    return size_dict


class TorchvisionBackend:
    """Base class for the vendored image processors."""

    #: The TypedDict naming every knob this processor accepts. Subclasses
    #: override it; `_set_attributes` walks its annotations.
    valid_kwargs: type = ImagesKwargs

    # Defaults, overridden by subclasses and then by preprocessor_config.json.
    do_convert_rgb: bool | None = None
    do_resize: bool | None = None
    size: Any = None
    default_to_square: bool | None = None
    crop_size: Any = None
    resample: Any = None
    do_rescale: bool | None = None
    rescale_factor: float = 1 / 255
    do_normalize: bool | None = None
    image_mean: Any = None
    image_std: Any = None
    do_pad: bool | None = None
    pad_size: Any = None
    do_center_crop: bool | None = None
    data_format: Any = ChannelDimension.FIRST
    input_data_format: Any = None
    device: Any = None
    return_tensors: Any = None
    disable_grouping: bool | None = None

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        self._set_attributes(**kwargs)

    @property
    def backend(self) -> str:
        return "torchvision"

    # ------------------------------------------------------------- kwargs

    def _set_attributes(self, **kwargs) -> None:
        """Resolve every declared knob from kwargs, then class defaults."""
        attributes = {}
        for key in self.valid_kwargs.__annotations__:
            value = kwargs.pop(key, None)
            attributes[key] = value if value is not None else getattr(self, key, None)
        attributes = self._standardize_kwargs(**attributes)
        for key, value in attributes.items():
            setattr(self, key, value)
        # Anything the JSON carried that is not a declared knob is still worth
        # keeping: `character_list` reaches the recognisers' CTC decode this way.
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._valid_kwargs_names = list(self.valid_kwargs.__annotations__.keys())

    def _standardize_kwargs(
        self,
        size=None,
        crop_size=None,
        pad_size=None,
        default_to_square=None,
        image_mean=None,
        image_std=None,
        **kwargs,
    ) -> dict:
        """Coerce the size-shaped and sequence-shaped knobs to canonical form."""
        if size is not None and not isinstance(size, SizeDict):
            size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square))
        if crop_size is not None and not isinstance(crop_size, SizeDict):
            crop_size = SizeDict(**get_size_dict(crop_size, param_name="crop_size"))
        if pad_size is not None and not isinstance(pad_size, SizeDict):
            pad_size = SizeDict(**get_size_dict(size=pad_size, param_name="pad_size"))
        # Tuples, not lists: `_fuse_mean_std_and_rescale_factor` is lru_cached
        # on these and a list is unhashable.
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)

        kwargs["size"] = size
        kwargs["crop_size"] = crop_size
        kwargs["pad_size"] = pad_size
        kwargs["default_to_square"] = default_to_square
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std
        return kwargs

    # ------------------------------------------------------------- loading

    @classmethod
    def from_pretrained(cls, model_dir: str | os.PathLike, **kwargs) -> TorchvisionBackend:
        """Build from ``<model_dir>/preprocessor_config.json``."""
        path = Path(model_dir)
        config_file = path / IMAGE_PROCESSOR_NAME if path.is_dir() else path
        if not config_file.is_file():
            raise OSError(
                f"{config_file} does not exist. Download the model again from the setup page."
            )
        try:
            raw = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OSError(f"{config_file} could not be read ({type(exc).__name__}: {exc})") from exc
        return cls.from_dict(raw, **kwargs)

    @classmethod
    def from_dict(cls, processor_dict: dict[str, Any], **kwargs) -> TorchvisionBackend:
        values = {**processor_dict, **kwargs}
        # Bookkeeping that names the class rather than configuring it.
        for key in ("image_processor_type", "processor_class", "_valid_processor_keys", "model_type"):
            values.pop(key, None)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"image_processor_type": type(self).__name__}
        for key in getattr(self, "_valid_kwargs_names", []):
            value = getattr(self, key, None)
            if isinstance(value, SizeDict):
                value = dict(value)
            out[key] = value
        return out

    # ---------------------------------------------------------- preprocess

    def __call__(self, images: ImageInput, *args, **kwargs: Unpack[ImagesKwargs]) -> BatchFeature:
        return self.preprocess(images, *args, **kwargs)

    def preprocess(self, images: ImageInput, *args, **kwargs: Unpack[ImagesKwargs]) -> BatchFeature:
        """Fill in defaults, standardise, then hand off to ``_preprocess``."""
        for name in self._valid_kwargs_names:
            kwargs.setdefault(name, getattr(self, name, None))
        kwargs = self._standardize_kwargs(**kwargs)
        return self._preprocess_image_like_inputs(images, *args, **kwargs)

    def _preprocess_image_like_inputs(
        self, images: ImageInput, *args, **kwargs: Unpack[ImagesKwargs]
    ) -> BatchFeature:
        prepared = self._prepare_image_like_inputs(images, **kwargs)
        return self._preprocess(prepared, *args, **kwargs)

    def _prepare_image_like_inputs(
        self, images: ImageInput, *args, expected_ndims: int = 3, **kwargs
    ) -> list[torch.Tensor]:
        images = make_flat_list_of_images(images, expected_ndims=expected_ndims)
        process = partial(self.process_image, *args, **kwargs)
        return [process(image) for image in images]

    def _preprocess(self, images: list[torch.Tensor], **kwargs) -> BatchFeature:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _preprocess; the vendored model file provides it."
        )

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: torch.device | None = None,
        **kwargs: Unpack[ImagesKwargs],
    ) -> torch.Tensor:
        """One raw image -> one channels-first ``torch.Tensor``."""
        image_type = get_image_type(image)
        if image_type not in (ImageType.PIL, ImageType.TORCH, ImageType.NUMPY):
            raise ValueError(f"Unsupported input image type {image_type}")

        if do_convert_rgb:
            image = convert_to_rgb(image)

        if image_type == ImageType.PIL:
            image = tvF.pil_to_tensor(image)
        elif image_type == ImageType.NUMPY:
            image = torch.from_numpy(image).contiguous()

        if image.ndim == 2:
            image = image.unsqueeze(0)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        if input_data_format == ChannelDimension.LAST:
            image = image.permute(2, 0, 1).contiguous()

        if device is not None:
            image = image.to(device)
        return image

    def convert_to_rgb(self, image: ImageInput) -> ImageInput:
        return convert_to_rgb(image)

    # -------------------------------------------------------- primitives

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        resample: Any = None,
        antialias: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Resize to ``size.height`` x ``size.width``.

        Only the explicit height/width form is supported, which is all the
        vendored PP-OCR processors use -- they compute the target themselves
        (``get_image_size`` for detection, ``get_target_size`` for
        recognition) rather than describing it as a shortest/longest edge.
        """
        if resample is None:
            interpolation = tvF.InterpolationMode.BILINEAR
        elif isinstance(resample, (PILImageResampling, int)):
            interpolation = pil_torch_interpolation_mapping[PILImageResampling(resample)]
        else:
            interpolation = resample

        if size.height is None or size.width is None:
            raise ValueError(f"resize needs size.height and size.width; got {size}")
        return tvF.resize(image, (size.height, size.width), interpolation=interpolation, antialias=antialias)

    def pad(
        self,
        images: list[torch.Tensor],
        pad_size: SizeDict | None = None,
        fill_value: int | None = 0,
        padding_mode: str | None = "constant",
        return_mask: bool = False,
        disable_grouping: bool | None = False,
        **kwargs,
    ):
        """Pad right and bottom up to ``pad_size`` (or the batch maximum)."""
        if pad_size is not None:
            if not (pad_size.height and pad_size.width):
                raise ValueError(f"pad_size needs height and width; got {pad_size}")
            target = (pad_size.height, pad_size.width)
        else:
            target = get_max_height_width(images)

        grouped, index = group_images_by_shape(images, disable_grouping=disable_grouping)
        padded: dict[Any, torch.Tensor] = {}
        masks: dict[Any, torch.Tensor] = {}
        for shape, stacked in grouped.items():
            height, width = stacked.shape[-2:]
            pad_height, pad_width = target[0] - height, target[1] - width
            if pad_height < 0 or pad_width < 0:
                raise ValueError(
                    f"pad_size {target} is smaller than the image size {(height, width)}."
                )
            if (height, width) != target:
                # torchvision padding is (left, top, right, bottom).
                stacked = tvF.pad(stacked, (0, 0, pad_width, pad_height), fill=fill_value, padding_mode=padding_mode)
            padded[shape] = stacked
            if return_mask:
                mask = torch.zeros_like(stacked, dtype=torch.int64)[..., 0, :, :]
                mask[..., :height, :width] = 1
                masks[shape] = mask

        out = reorder_images(padded, index)
        if return_mask:
            return out, reorder_images(masks, index)
        return out

    def center_crop(self, image: torch.Tensor, size: SizeDict, **kwargs) -> torch.Tensor:
        if size.height is None or size.width is None:
            raise ValueError(f"center_crop needs size.height and size.width; got {size}")
        image_height, image_width = image.shape[-2:]
        crop_height, crop_width = size.height, size.width

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            image = tvF.pad(image, padding_ltrb, fill=0)
            image_height, image_width = image.shape[-2:]
            if crop_width == image_width and crop_height == image_height:
                return image

        crop_top = int((image_height - crop_height) / 2.0)
        crop_left = int((image_width - crop_width) / 2.0)
        return tvF.crop(image, crop_top, crop_left, crop_height, crop_width)

    def rescale(self, image: torch.Tensor, scale: float, **kwargs) -> torch.Tensor:
        return image * scale

    def normalize(self, image: torch.Tensor, mean, std, **kwargs) -> torch.Tensor:
        return tvF.normalize(image, mean, std)

    @lru_cache(maxsize=10)
    def _fuse_mean_std_and_rescale_factor(
        self,
        do_normalize: bool | None = None,
        image_mean=None,
        image_std=None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        device: torch.device | None = None,
    ) -> tuple:
        """Fold the 1/255 rescale into the normalisation constants.

        ``(x/255 - m) / s`` is ``(x - 255m) / 255s``, so scaling mean and std
        up by ``1/rescale_factor`` lets one pass do both -- and keeps the uint8
        tensor out of a separate float multiply.
        """
        if do_rescale and do_normalize:
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
            do_rescale = False
        return image_mean, image_std, do_rescale

    def rescale_and_normalize(
        self,
        images: torch.Tensor,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
    ) -> torch.Tensor:
        image_mean, image_std, do_rescale = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            device=images.device,
        )
        if do_normalize:
            return self.normalize(images.to(dtype=torch.float32), image_mean, image_std)
        if do_rescale:
            return self.rescale(images, rescale_factor)
        return images

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


__all__ = [
    "IMAGE_PROCESSOR_NAME",
    "TorchvisionBackend",
    "convert_to_size_dict",
    "get_size_dict",
    "is_valid_size_dict",
]
