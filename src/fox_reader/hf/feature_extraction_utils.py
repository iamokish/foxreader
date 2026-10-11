"""``BatchFeature``, trimmed from ``transformers.feature_extraction_utils``.

The container an image processor returns: a dict of model inputs that also
exposes them as attributes, converts lists of tensors into batched tensors,
and can move the whole lot to a device in one call.

Three consumers here, all in the OCR path:

* each vendored ``_preprocess`` ends with
  ``BatchFeature(data={...}, tensor_type=return_tensors)``;
* ``fox_reader.ocr_predict`` reads ``model_inputs["target_sizes"]`` and
  forwards ``**model_inputs`` into the model;
* the device move calls ``.to(device)``.

Not ported: the ``np``/``mlx`` tensor types (the vendored processors always
ask for ``"pt"``), the ``skip_tensor_conversion`` key list, and the
save/load-from-JSON half of upstream's ``FeatureExtractionMixin``.
"""

from __future__ import annotations

from collections import UserDict
from typing import Any

import numpy as np
import torch

from .utils.generic import TensorType


def _as_torch(value: Any) -> Any:
    """Best-effort conversion of one value to a torch tensor.

    ``torch.tensor()`` cannot take a list of tensors, which is exactly what a
    per-image processing loop produces, so that case is stacked explicitly.
    """
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) > 0:
        if torch.is_tensor(value[0]):
            return torch.stack(list(value))
        if isinstance(value[0], np.ndarray):
            value = np.asarray(value)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.tensor(value)


def _is_convertible(value: Any) -> bool:
    """Whether converting this value to a tensor is even meaningful."""
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return True
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and not isinstance(value[0], str)
    return isinstance(value, (int, float))


class BatchFeature(UserDict):
    """Model inputs, reachable as both keys and attributes."""

    def __init__(self, data: dict[str, Any] | None = None, tensor_type: str | TensorType | None = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def __getitem__(self, item: str) -> Any:
        if isinstance(item, str):
            return self.data[item]
        raise KeyError("Indexing with integers is not available through BatchFeature.")

    def __getattr__(self, item: str):
        # Only reached for names that are not real attributes.
        try:
            return self.data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __getstate__(self):
        return {"data": self.data}

    def __setstate__(self, state):
        self.data = state["data"]

    def convert_to_tensors(self, tensor_type: str | TensorType | None = None) -> BatchFeature:
        """Turn array-like values into tensors, in place."""
        if tensor_type is None:
            return self
        if TensorType(tensor_type) is not TensorType.PYTORCH:
            raise ValueError(
                f"BatchFeature here only produces torch tensors; got return_tensors={tensor_type!r}."
            )
        for key, value in list(self.items()):
            if not _is_convertible(value):
                continue
            if torch.is_tensor(value):
                continue
            try:
                self[key] = _as_torch(value)
            except Exception as exc:
                raise ValueError(
                    f"Could not convert {key!r} ({type(value).__name__}) to a tensor: {exc}"
                ) from exc
        return self

    def to(self, *args, **kwargs) -> BatchFeature:
        """Move (and optionally cast) every tensor value.

        Only floating-point tensors are cast, matching upstream: casting an
        integer index tensor to float would silently corrupt it.
        """
        device = kwargs.get("device")
        non_blocking = kwargs.get("non_blocking", False)
        if device is None and args:
            arg = args[0]
            if isinstance(arg, torch.dtype):
                pass  # a dtype-only call
            elif isinstance(arg, (str, int, torch.device)):
                device = arg
            else:
                raise ValueError(f"Cannot move a BatchFeature to {arg!r}.")

        def maybe_to(value):
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                return value.to(*args, **kwargs)
            if isinstance(value, torch.Tensor) and device is not None:
                return value.to(device=device, non_blocking=non_blocking)
            if isinstance(value, (list, tuple)):
                return type(value)(maybe_to(item) for item in value)
            return value

        self.data = {k: maybe_to(v) for k, v in self.items()}
        return self


__all__ = ["BatchFeature"]
