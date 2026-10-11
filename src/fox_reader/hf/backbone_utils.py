"""Backbone plumbing, trimmed from ``transformers.backbone_utils``.

A backbone exposes a chosen subset of its stages as ``feature_maps``. Two
mixins make that work, and the vendored PP-LCNet files inherit both by name:

* :class:`BackboneConfigMixin` on the *config*, reconciling ``out_features``
  (``["stage1", ...]``) and ``out_indices`` (``[1, 2, 3, 4]``) against
  ``stage_names``. Either may be given; each implies the other.
* :class:`BackboneMixin` on the *model*, reading those back and installing the
  output-capture hooks.

:func:`filter_output_hidden_states` is the reason the hook machinery in
:mod:`.utils.output_capturing` is on the hot path rather than a debug feature:
a backbone builds its feature maps *out of* the per-stage hidden states, so it
forces ``output_hidden_states=True`` on every call and then strips the field
back out unless the caller actually asked for it.

Not ported: the timm backbone branch (``BackboneType.TIMM``, ``feature_info``
introspection) and the hub lookups in
:func:`consolidate_backbone_kwargs_to_config` -- upstream will resolve a
``backbone="resnet50"`` string by calling ``repo_exists`` against
huggingface.co. Every vendored PP-OCR config ships a literal
``backbone_config`` dict in its ``config.json``, so the only branch reachable
here is the dict one, and a network call during model construction would be a
bug rather than a feature.
"""

from __future__ import annotations

import enum
import functools

from .utils.output_capturing import maybe_install_capturing_hooks


class BackboneType(enum.Enum):
    TIMM = "timm"
    TRANSFORMERS = "transformers"


class BackboneConfigMixin:
    """Keeps ``out_features`` and ``out_indices`` aligned on a config."""

    def set_output_features_output_indices(
        self,
        out_features: list | None,
        out_indices: list | None,
    ) -> None:
        """Set both, deriving whichever one was not given."""
        self._out_features = out_features
        self._out_indices = list(out_indices) if isinstance(out_indices, tuple) else out_indices

        self.verify_out_features_out_indices()

        out_features, out_indices = self._out_features, self._out_indices
        if out_indices is None and out_features is None:
            out_indices = [len(self.stage_names) - 1]
            out_features = [self.stage_names[-1]]
        elif out_indices is None and out_features is not None:
            out_indices = [self.stage_names.index(layer) for layer in out_features]
        elif out_features is None and out_indices is not None:
            out_features = [self.stage_names[idx] for idx in out_indices]

        self._out_features, self._out_indices = out_features, out_indices
        self.verify_out_features_out_indices()

    def verify_out_features_out_indices(self) -> None:
        """Reject a selection that does not describe these stages.

        Strict on purpose, and upstream is too: a mismatch between the
        checkpoint's stage list and the config's silently produces feature maps
        from the wrong depths, which shows up as a detector that finds nothing
        rather than as an error.
        """
        if getattr(self, "stage_names", None) is None:
            raise ValueError("stage_names must be set for transformers backbones")

        if self._out_features is not None:
            if not isinstance(self._out_features, list):
                raise ValueError(f"out_features must be a list, got {type(self._out_features)}")
            if any(feat not in self.stage_names for feat in self._out_features):
                raise ValueError(
                    f"out_features must be a subset of stage_names: {self.stage_names} "
                    f"got {self._out_features}"
                )
            if len(self._out_features) != len(set(self._out_features)):
                raise ValueError(f"out_features must not contain duplicates, got {self._out_features}")
            sorted_feats = [feat for feat in self.stage_names if feat in self._out_features]
            if self._out_features != sorted_feats:
                raise ValueError(
                    f"out_features must be in stage_names order, expected {sorted_feats} "
                    f"got {self._out_features}"
                )

        if self._out_indices is not None:
            if not isinstance(self._out_indices, list):
                raise ValueError(f"out_indices must be a list, got {type(self._out_indices)}")
            positive = tuple(
                idx % len(self.stage_names) if idx < 0 else idx for idx in self._out_indices
            )
            if any(idx not in range(len(self.stage_names)) for idx in positive):
                raise ValueError(
                    f"out_indices must index stage_names {self.stage_names}, got {self._out_indices}"
                )
            if len(positive) != len(set(positive)):
                raise ValueError(f"out_indices must not contain duplicates, got {self._out_indices}")
            if positive != tuple(sorted(positive)):
                raise ValueError(
                    f"out_indices must be in stage_names order, got {self._out_indices}"
                )

        if self._out_features is not None and self._out_indices is not None:
            if len(self._out_features) != len(self._out_indices):
                raise ValueError("out_features and out_indices must have the same length if both are set")
            if self._out_features != [self.stage_names[idx] for idx in self._out_indices]:
                raise ValueError("out_features and out_indices must name the same stages if both are set")

    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: list[str]):
        self.set_output_features_output_indices(out_features=out_features, out_indices=None)

    @property
    def out_indices(self):
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices):
        out_indices = list(out_indices) if out_indices is not None else None
        self.set_output_features_output_indices(out_features=None, out_indices=out_indices)

    def to_dict(self) -> dict:
        output = super().to_dict()
        output["out_features"] = output.pop("_out_features", None)
        output["out_indices"] = output.pop("_out_indices", None)
        return output


def filter_output_hidden_states(forward_function):
    """Force hidden-state capture on, then hide it again unless asked for.

    A backbone cannot build ``feature_maps`` without the per-stage hidden
    states, so they are always computed. Returning them as well would change
    what every caller sees, so they are stripped back out.

    Assumes ``can_return_tuple`` is applied *outside* this decorator, so the
    value here is still a dict-like ``ModelOutput``.
    """

    @functools.wraps(forward_function)
    def wrapper(self, *args, **kwargs):
        output_hidden_states = kwargs.get(
            "output_hidden_states", getattr(self.config, "output_hidden_states", False)
        )
        kwargs["output_hidden_states"] = True
        output = forward_function(self, *args, **kwargs)
        if not output_hidden_states:
            kept = {k: v for k, v in output.items() if k != "hidden_states"}
            output = type(output)(**kept)
        return output

    return wrapper


class BackboneMixin:
    """Model-side half: stage names, channel counts and capture hooks."""

    backbone_type: BackboneType | None = None

    #: Convolutional backbones have no attention to report. The PP-LCNets set
    #: this to False, which is what lets `forward_with_filtered_kwargs` drop
    #: `output_attentions` rather than pass an argument the forward rejects.
    has_attentions: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone_type = BackboneType.TRANSFORMERS
        self._init_transformers_backbone()

    def post_init(self) -> None:
        """Install the capture hooks as soon as the submodules exist.

        A backbone *always* captures, so there is no point deferring to the
        first forward. Doing it here also keeps hook installation out of the
        forward path entirely.
        """
        super().post_init()
        maybe_install_capturing_hooks(self)

    def _init_transformers_backbone(self) -> None:
        self.stage_names = self.config.stage_names
        self.config.verify_out_features_out_indices()
        # Filled in by the concrete backbone's __init__.
        self.num_features = None

    @property
    def out_features(self):
        return self.config._out_features

    @out_features.setter
    def out_features(self, out_features: list[str]):
        self.config.out_features = out_features

    @property
    def out_indices(self):
        return self.config._out_indices

    @out_indices.setter
    def out_indices(self, out_indices):
        self.config.out_indices = out_indices

    @property
    def out_feature_channels(self) -> dict[str, int]:
        # Every stage, including ones not selected -- upstream's contract.
        return {stage: self.num_features[i] for i, stage in enumerate(self.stage_names)}

    @property
    def channels(self) -> list[int]:
        return [self.out_feature_channels[name] for name in self.out_features]

    def forward_with_filtered_kwargs(self, *args, **kwargs):
        if not self.has_attentions:
            kwargs.pop("output_attentions", None)
        return self(*args, **kwargs)


def consolidate_backbone_kwargs_to_config(
    backbone_config,
    default_backbone: str | None = None,
    default_config_type: str | None = None,
    default_config_kwargs: dict | None = None,
    timm_default_kwargs: dict | None = None,
    **kwargs,
):
    """Turn a ``backbone_config`` dict from ``config.json`` into a config object.

    Returns ``(backbone_config, remaining_kwargs)`` -- the vendored configs
    thread the leftovers straight into ``super().__post_init__(**kwargs)``.
    """
    from .models.auto import CONFIG_MAPPING

    # Accepted and dropped: upstream's timm-backbone switches. Nothing vendored
    # here uses timm, and silently ignoring them beats rejecting a config.json
    # that happens to carry one.
    kwargs.pop("use_timm_backbone", None)
    kwargs.pop("use_pretrained_backbone", None)
    backbone_kwargs = kwargs.pop("backbone_kwargs", {})
    backbone = kwargs.pop("backbone", None) or default_backbone

    if backbone is not None and backbone_config is None:
        raise ValueError(
            f"backbone={backbone!r} names a backbone to fetch from the Hugging Face Hub, "
            "which this build does not do. Ship a literal backbone_config in config.json."
        )

    if isinstance(backbone_config, dict):
        model_type = backbone_config.get("model_type")
        config_class = CONFIG_MAPPING[model_type]
        merged = {**backbone_config, **backbone_kwargs}
        backbone_config = config_class.from_dict(merged)
    elif backbone_config is None and default_config_type is not None:
        backbone_config = CONFIG_MAPPING[default_config_type](**(default_config_kwargs or {}))

    return backbone_config, kwargs


def load_backbone(config):
    """Build the backbone a parent model's config points at.

    Called from inside the parent's ``__init__``, which under
    ``PreTrainedModel.from_pretrained`` runs in a meta-device context -- so
    this allocates nothing and the parent's checkpoint fills it in.
    """
    from .models.auto import AutoBackbone

    backbone_config = getattr(config, "backbone_config", None)
    return AutoBackbone.from_config(config=backbone_config if backbone_config is not None else config)


__all__ = [
    "BackboneConfigMixin",
    "BackboneMixin",
    "BackboneType",
    "consolidate_backbone_kwargs_to_config",
    "filter_output_hidden_states",
    "load_backbone",
]
