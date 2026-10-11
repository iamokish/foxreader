"""``PreTrainedConfig``, trimmed from ``transformers.configuration_utils``.

A config is a strict dataclass: fields are declared as class annotations with
defaults, unknown keys from ``config.json`` are absorbed as plain attributes
(``accept_kwargs=True``), and ``__post_init__`` runs the model's own
normalisation. The vendored config files are byte-identical to upstream, so
this base has to support that shape exactly:

    @auto_docstring(checkpoint="PaddlePaddle/PP-OCRv6_medium_det_safetensors")
    @strict
    class PPOCRV6MediumDetConfig(PreTrainedConfig):
        sub_configs = {"backbone_config": AutoConfig}
        model_type = "pp_ocrv6_medium_det"
        neck_out_channels: int = 256
        ...
        def __post_init__(self, **kwargs):
            self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(...)
            super().__post_init__(**kwargs)

``strict`` comes from ``huggingface_hub``, which stays a dependency for model
downloads, so the validation semantics are upstream's rather than a
re-implementation.

One trap worth naming: ``model_type`` and ``sub_configs`` must be ``ClassVar``
here. Annotate them as ordinary fields and ``@strict`` turns them into
validated instance fields, at which point a subclass writing
``sub_configs = {...}`` with no annotation sets a *default* for a required
field and construction fails with a type error about ``NoneType``.

Dropped from upstream: ``PushToHubMixin`` (no uploads), rope-scaling and
heterogeneous-layer mixins (no attention-cache models vendored here), the
quantization and generation-parameter plumbing, and hub resolution --
``from_pretrained`` takes a local directory, because `fox_reader.assets`
downloads weights long before a config is read.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import MISSING, dataclass, fields
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar, dataclass_transform

from huggingface_hub.dataclasses import strict

CONFIG_NAME = "config.json"


def _wrap_init_to_accept_kwargs(cls):
    """Reinstall an ``__init__`` that knows about *this* class's fields.

    ``huggingface_hub``'s ``strict(accept_kwargs=True)`` builds an ``__init__``
    that closes over the class it was applied to -- here
    :class:`PreTrainedConfig`. A subclass inherits that ``__init__`` verbatim,
    so ``fields(cls)`` inside it still lists only the *base* fields and every
    field the subclass declared is mistaken for an unknown key and forwarded to
    ``__post_init__``. The vendored detector config then fails with
    "got multiple values for keyword argument 'backbone_config'".

    Rebinding the closure to the subclass is what upstream does too
    (``transformers.configuration_utils.wrap_init_to_accept_kwargs``).
    """
    original_init = cls.__init__

    @wraps(original_init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args:
            raise ValueError(
                f"{cls.__name__} takes keyword arguments only, got {len(args)} positional."
            )
        declared = {f.name for f in fields(cls)}
        for f in fields(cls):
            if f.name in kwargs:
                setattr(self, f.name, kwargs[f.name])
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())
            else:
                raise TypeError(f"{cls.__name__} is missing required field {f.name!r}")

        # Everything else is the config.json's business, not the dataclass's.
        extra = {k: v for k, v in kwargs.items() if k not in declared}
        self.__post_init__(**extra)

    cls.__init__ = __init__
    return cls



@dataclass_transform(kw_only_default=True)
@strict(accept_kwargs=True)
@dataclass(repr=False)
class PreTrainedConfig:
    """Base class for every vendored model configuration."""

    #: Set by each subclass; the key `AutoConfig` resolves a folder by.
    model_type: ClassVar[str] = ""

    #: ``{attribute_name: config_class_or_AutoConfig}``. Any entry here that
    #: arrives as a plain dict in ``config.json`` is rebuilt into a real config
    #: object by :meth:`from_dict`.
    sub_configs: ClassVar[dict[str, Any]] = {}

    #: Read by `can_return_tuple` and `capture_outputs`. Models may override.
    return_dict: bool = True
    output_hidden_states: bool = False
    output_attentions: bool = False

    #: Which attention kernel `ALL_ATTENTION_FUNCTIONS` hands out. ``None``
    #: means "decide at load time" -- see `PreTrainedModel.from_pretrained`.
    _attn_implementation: str | None = None

    #: Where this config was loaded from, for error messages.
    name_or_path: str = ""

    def __post_init__(self, **kwargs: Any) -> None:
        """Absorb keys the dataclass does not declare.

        Released ``config.json`` files carry fields no model class reads --
        ``PP-OCRv6_medium_det`` ships ``mode``, ``use_lab``, ``class_num`` and
        ``class_expand``. Upstream keeps them as attributes rather than
        rejecting the file, and so must this: a future weights release that
        adds a field must not stop the app from starting.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        """Make every subclass a keyword-only dataclass.

        The vendored config files declare their fields as bare class
        annotations and are decorated with ``@strict`` alone; nothing in them
        applies ``@dataclass``. Upstream does it here, and so must this, or the
        annotations never become fields at all.

        ``kw_only=True`` is what lets a subclass declare a field without a
        default after the base's defaulted ones -- the ordering rule that would
        otherwise make every config a ``TypeError`` at class-creation time.
        """
        super().__init_subclass__(*args, **kwargs)
        has_custom_init = "__init__" in cls.__dict__
        cls = dataclass(cls, repr=False, kw_only=True)
        if not has_custom_init:
            _wrap_init_to_accept_kwargs(cls)

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_pretrained(cls, model_dir: str | os.PathLike, **kwargs: Any) -> PreTrainedConfig:
        """Build a config from ``<model_dir>/config.json``."""
        path = Path(model_dir)
        config_file = path / CONFIG_NAME if path.is_dir() else path
        if not config_file.is_file():
            raise OSError(
                f"{config_file} does not exist. Download the model again from the setup page."
            )
        try:
            raw = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OSError(
                f"{config_file} could not be read ({type(exc).__name__}: {exc})"
            ) from exc
        raw.setdefault("name_or_path", str(path))
        return cls.from_dict({**raw, **kwargs})

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> PreTrainedConfig:
        """Build a config from a plain dict, recursing into ``sub_configs``."""
        values = {**config_dict, **kwargs}
        # `model_type` identifies the class; it is not a field on it.
        values.pop("model_type", None)

        for name, sub_cls in (cls.sub_configs or {}).items():
            sub = values.get(name)
            if not isinstance(sub, dict):
                continue
            resolved = sub_cls
            if getattr(sub_cls, "_is_auto_class", False):
                # AutoConfig: the nested dict names its own class.
                resolved = sub_cls.for_model_type(sub.get("model_type"))
            values[name] = resolved.from_dict(sub)

        return cls(**values)

    # ----------------------------------------------------------- serialising

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict, with sub-configs expanded in place."""
        out: dict[str, Any] = {"model_type": type(self).model_type}
        declared = {f.name for f in fields(self)}
        for key, value in vars(self).items():
            if key.startswith("__"):
                continue
            if isinstance(value, PreTrainedConfig):
                out[key] = value.to_dict()
            elif key in declared or not key.startswith("_"):
                out[key] = copy.deepcopy(value)
        return out

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_json_string()})"


__all__ = ["CONFIG_NAME", "PreTrainedConfig"]
