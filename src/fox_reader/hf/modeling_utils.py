"""``PreTrainedModel`` and the attention registry, trimmed from
``transformers.modeling_utils`` (5,260 lines upstream).

What a vendored PP-OCR model actually asks of its base class is small: hold a
config, expose ``post_init()``, resolve ``config_class`` from the annotated
``config`` attribute, and load a ``model.safetensors`` off disk. None of
upstream's tensor-parallel / pipeline / FSDP plans, device maps, quantization,
PEFT adapters, weight tying, hub resolution, sharded checkpoints or
``safetensors``-to-``.bin`` fallbacks are reachable from here.

Loading is done on the ``meta`` device and completed with
``load_state_dict(assign=True)``. That is not a micro-optimisation: every
module's ``__init__`` would otherwise fill its weights with random numbers
that the checkpoint immediately overwrites, and `load_backbone` makes the
detector build a whole second randomly-initialised PP-LCNet backbone inside
its own ``__init__``. On meta, no storage is allocated and no RNG runs; the
tensors arrive straight from the file.

The one wrinkle that buys is buffers the checkpoint does not carry --
BatchNorm's ``num_batches_tracked``, which is why the vendored detector
declares ``_keys_to_ignore_on_load_missing = ["num_batches_tracked"]``. Those
would stay as meta tensors and explode on first use, so
:func:`_materialize_leftover_meta_tensors` finishes them off, and anything
*else* still on meta is a genuinely missing weight and raises.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable, get_type_hints

import torch
from torch import nn

from .configuration_utils import PreTrainedConfig
from .utils.output_capturing import _CAN_RECORD_REGISTRY

logger = logging.getLogger(__name__)

SAFE_WEIGHTS_NAME = "model.safetensors"


# ---------------------------------------------------------------- attention


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Plain matmul attention. The fallback every vendored model passes in."""
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """`torch.nn.functional.scaled_dot_product_attention`, same contract.

    ``is_causal`` follows upstream's precedence -- explicit kwarg, then the
    module's own attribute, then ``True``. Both vendored recognisers set
    ``self.is_causal = False`` in ``__init__``: the SVTR encoder is a
    bidirectional vision transformer, and a causal mask here would quietly
    corrupt every transcription rather than fail.
    """
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)
    # SDPA's flash kernel takes the flag rather than a mask, but only when
    # there is no explicit mask and more than one query position.
    is_causal = bool(query.shape[2] > 1 and attention_mask is None and is_causal)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    # SDPA fuses the softmax, so the weights are never materialised.
    return attn_output, None


class AttentionInterface(dict):
    """Name-to-implementation registry for attention kernels.

    Upstream also carries flash-attention 2/3/4, flex attention and the paged
    variants. Those need optional CUDA packages that Fox Reader does not ship,
    and the only attention in the vendored set is a short bidirectional SVTR
    encoder, where SDPA is already the fast path.
    """

    _global_mapping: dict[str, Callable] = {
        "sdpa": sdpa_attention_forward,
        "eager": eager_attention_forward,
    }

    def __init__(self) -> None:
        super().__init__(self._global_mapping)

    def get_interface(self, attn_implementation: str | None, default: Callable) -> Callable:
        """Resolve a name, raising on one that was never registered."""
        if attn_implementation is None:
            # A module used standalone, outside a `from_pretrained` model.
            return default
        if attn_implementation != "eager" and attn_implementation not in self:
            raise KeyError(
                f"{attn_implementation!r} is not a registered attention implementation; "
                f"expected one of {sorted(self)}"
            )
        return self.get(attn_implementation, default)


ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()


# -------------------------------------------------------------------- model


def _materialize_leftover_meta_tensors(model: nn.Module, device: torch.device) -> list[str]:
    """Give real storage to anything the checkpoint did not supply.

    Returns the names it had to invent, so the caller can decide whether each
    was an expected omission (``num_batches_tracked``) or a missing weight.
    """
    invented: list[str] = []
    for name, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        parent.register_buffer(
            name.rsplit(".", 1)[-1], torch.zeros(buf.shape, dtype=buf.dtype, device=device)
        )
        invented.append(name)
    for name, param in list(model.named_parameters()):
        if param.device.type != "meta":
            continue
        invented.append(name)
    return invented


class PreTrainedModel(nn.Module):
    """Base class for the vendored models.

    Subclasses declare their config with an annotation
    (``config: PPOCRV6MediumDetConfig``); :meth:`__init_subclass__` turns that
    into ``config_class`` so :meth:`from_pretrained` can build one.
    """

    config_class: type[PreTrainedConfig] | None = None
    base_model_prefix: str = ""
    main_input_name: str = "input_ids"
    input_modalities: tuple[str, ...] = ("text",)
    supports_gradient_checkpointing: bool = False

    _can_compile_fullgraph: bool = False
    _supports_sdpa: bool = True
    _supports_flash_attn: bool = False
    _supports_flex_attn: bool = False
    _supports_attention_backend: bool = False
    _no_split_modules: list[str] | None = None
    _skip_keys_device_placement: list[str] | None = None
    _keys_to_ignore_on_load_missing: list[str] | None = None
    _keys_to_ignore_on_load_unexpected: list[str] | None = None
    _can_record_outputs: dict[str, Any] | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Priority matches upstream: a `config_class` on this class, then a
        # `config:` annotation on it, then either inherited from a parent.
        child_attribute = cls.__dict__.get("config_class", None)
        child_annotation = inspect.get_annotations(cls).get("config", None)
        if child_attribute is not None:
            cls.config_class = child_attribute
            return
        if isinstance(child_annotation, type) and issubclass(child_annotation, PreTrainedConfig):
            cls.config_class = child_annotation
            return
        if cls.config_class is not None:
            return
        try:
            full_annotation = get_type_hints(cls).get("config", None)
        except Exception:
            # Forward references into modules that are still importing.
            full_annotation = None
        if isinstance(full_annotation, type) and issubclass(full_annotation, PreTrainedConfig):
            cls.config_class = full_annotation

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise TypeError(
                f"config passed to {self.__class__.__name__}(config) must be a PreTrainedConfig; "
                f"got {type(config).__name__}. Build the model with "
                f"{self.__class__.__name__}.from_pretrained(<model dir>) instead."
            )
        self.config = config
        self.name_or_path = getattr(config, "name_or_path", "")
        # Keyed by class repr; read back by `capture_outputs` on every forward.
        _CAN_RECORD_REGISTRY[str(self.__class__)] = self._can_record_outputs

    @property
    def base_model(self) -> nn.Module:
        return getattr(self, self.base_model_prefix, self)

    def post_init(self) -> None:
        """Gather load-time key filters from the children.

        Upstream also assembles parallelism plans, tied-weight keys and fp32
        module sets here. None of those apply: these models are single-device,
        untied and fp32.
        """
        self._keys_to_ignore_on_load_missing = set(self._keys_to_ignore_on_load_missing or [])
        self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected or [])
        for _, module in self.named_children():
            for attr in ("_keys_to_ignore_on_load_missing", "_keys_to_ignore_on_load_unexpected"):
                inherited = getattr(module, attr, None)
                if inherited:
                    getattr(self, attr).update(inherited)

    def _init_weights(self, module: nn.Module) -> None:
        """Hook kept for signature compatibility.

        Every model here is built from a complete checkpoint on the meta
        device, so there is nothing to initialise. A model copied in later that
        relies on this must override it *and* be constructed off the meta path.
        """

    # --------------------------------------------------------------- loading

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | os.PathLike,
        *,
        config: PreTrainedConfig | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        """Build the model and load ``model.safetensors`` from a local folder.

        There is no hub fallback on purpose: `fox_reader.assets` has already
        downloaded the weights by the time anything calls this, so a missing
        file is a setup problem to report, not a download to start behind the
        user's back.
        """
        from safetensors.torch import load_file

        path = Path(model_dir)
        weights = path / SAFE_WEIGHTS_NAME
        if not weights.is_file():
            raise OSError(
                f"{weights} is missing. Download the model again from the setup page."
            )

        if config is None:
            if cls.config_class is None:
                raise TypeError(
                    f"{cls.__name__} does not resolve a config class; pass config= explicitly."
                )
            config = cls.config_class.from_pretrained(path, **kwargs)

        # Default to SDPA where the model supports it: for the SVTR recogniser
        # encoder it is a straight win over the eager matmul path, and it is
        # what transformers itself selects for these checkpoints.
        if attn_implementation is None:
            attn_implementation = getattr(config, "_attn_implementation", None)
        if attn_implementation is None:
            attn_implementation = "sdpa" if cls._supports_sdpa else "eager"
        config._attn_implementation = attn_implementation

        target = torch.device(device) if device is not None else torch.device("cpu")

        # Build with no storage and no RNG; the checkpoint supplies everything.
        with torch.device("meta"):
            model = cls(config)

        state = load_file(str(weights), device="cpu")
        if dtype is not None:
            state = {
                k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state.items()
            }

        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        model._report_load(weights, missing, unexpected)

        leftover = _materialize_leftover_meta_tensors(model, torch.device("cpu"))
        real_gaps = [
            name for name in leftover if not model._is_ignorable_missing_key(name)
        ]
        if real_gaps:
            raise OSError(
                f"{weights} is incomplete: {len(real_gaps)} tensor(s) had no value, "
                f"first is {real_gaps[0]!r}. Download the model again from the setup page."
            )

        model.eval()
        if target.type != "cpu":
            model.to(target)
        return model

    def _is_ignorable_missing_key(self, key: str) -> bool:
        patterns = self._keys_to_ignore_on_load_missing or ()
        return any(pattern in key for pattern in patterns)

    def _is_ignorable_unexpected_key(self, key: str) -> bool:
        patterns = self._keys_to_ignore_on_load_unexpected or ()
        return any(pattern in key for pattern in patterns)

    def _report_load(self, weights: Path, missing: list[str], unexpected: list[str]) -> None:
        """Log what the checkpoint and the model disagreed about.

        Missing keys are only a warning here -- the meta-tensor sweep in
        :meth:`from_pretrained` is what actually decides whether a gap is
        fatal, and it has the stricter view.
        """
        notable_missing = [k for k in missing if not self._is_ignorable_missing_key(k)]
        notable_unexpected = [k for k in unexpected if not self._is_ignorable_unexpected_key(k)]
        if notable_missing:
            logger.warning(
                "%s: %d tensor(s) in %s had no value (first: %s)",
                type(self).__name__, len(notable_missing), weights.name, notable_missing[0],
            )
        if notable_unexpected:
            logger.warning(
                "%s: %s holds %d tensor(s) this model has no place for (first: %s)",
                type(self).__name__, weights.name, len(notable_unexpected), notable_unexpected[0],
            )

    @classmethod
    def from_config(cls, config: PreTrainedConfig, **kwargs: Any) -> PreTrainedModel:
        """Build an uninitialised model from a config alone.

        This is the path `load_backbone` takes while a parent model is being
        constructed. Under ``from_pretrained`` that happens inside the meta
        device context, so it allocates nothing.
        """
        return cls(config, **kwargs)


__all__ = [
    "ALL_ATTENTION_FUNCTIONS",
    "AttentionInterface",
    "PreTrainedModel",
    "eager_attention_forward",
    "sdpa_attention_forward",
]
