"""Model output containers, trimmed from ``transformers.modeling_outputs``.

:class:`ModelOutput` is an ``OrderedDict`` that is also a dataclass: attribute
access always works, but only the fields that are not ``None`` appear as dict
keys. Two consumers in this package depend on exactly that:

* :func:`~fox_reader.hf.utils.output_capturing.capture_outputs` does
  ``outputs["hidden_states"] = (...)``, which must update the attribute too;
* :func:`~fox_reader.hf.backbone_utils.filter_output_hidden_states` rebuilds an
  output with ``type(output)(**{k: v for k, v in output.items() if ...})``,
  which needs ``.items()`` to yield only the set keys.

Not ported: upstream registers each subclass as a pytree node for
``torch.export``/``torch.compile``, and coerces a dict or ``(key, value)``
iterator passed as the *first positional* field. Fox Reader neither compiles
nor exports these models, and every vendored forward constructs its output
with keywords. Passing a bare dict positionally raises here rather than being
silently unpacked.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch


class ModelOutput(OrderedDict):
    """Dataclass that also behaves as an ordered dict of its set fields."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        class_fields = fields(self)
        if not class_fields:
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )
        for field in class_fields:
            value = self.__dict__.get(field.name)
            if value is not None:
                self[field.name] = value

    def __getitem__(self, k):
        if isinstance(k, str):
            return dict(self.items())[k]
        return self.to_tuple()[k]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {field.name for field in fields(self)} and value is not None:
            # Keep the dict view in step, without recursing through __setitem__.
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        # Not self.__setattr__: that would come straight back here.
        super().__setattr__(key, value)

    def __delitem__(self, *args, **kwargs):
        raise TypeError(f"You cannot use __delitem__ on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise TypeError(f"You cannot use setdefault on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise TypeError(f"You cannot use pop on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise TypeError(f"You cannot use update on a {self.__class__.__name__} instance.")

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable_, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return (callable_, args, *remaining)

    def to_tuple(self) -> tuple:
        """Every set (non-``None``) field, in declaration order."""
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutputWithNoAttention(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        Output of the model's last layer.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        One entry per layer, plus the initial embedding output. Returned when
        `output_hidden_states=True`.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BackboneOutput(ModelOutput):
    r"""
    feature_maps (`tuple(torch.FloatTensor)` of shape `(batch_size, num_channels, height, width)`):
        Feature maps of the stages named by `out_features`.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        One entry per stage, plus the initial embedding output.
    attentions (`tuple(torch.FloatTensor)`, *optional*):
        Only applicable if the backbone uses attention. The PP-LCNet backbones
        do not (`has_attentions = False`).
    """

    feature_maps: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


__all__ = ["BackboneOutput", "BaseModelOutputWithNoAttention", "ModelOutput"]
