"""Generic helpers the vendored model files import from ``...utils.generic``.

Three symbols, two of them real:

* :class:`TensorType` -- the ``return_tensors`` enum, verbatim.
* :func:`can_return_tuple` -- honours ``return_dict=False``, verbatim behaviour.
* :func:`merge_with_config_defaults` -- upstream fills ``use_cache``,
  ``vision_feature_layer``, ``vision_feature_select_strategy``,
  ``vision_aspect_ratio`` and ``is_causal`` from the config when the caller did
  not pass them. None of those exist on a PP-OCR backbone or recogniser config,
  so for this model set it is a pass-through. It is still implemented rather
  than aliased to the identity, because the day a copied-in model does carry
  ``use_cache`` the silent wrong answer would be very hard to find.
"""

from __future__ import annotations

from enum import Enum
from functools import wraps

#: Config fields upstream back-fills into the wrapped forward's arguments.
#: Kept as data so the list can be compared against a newer transformers.
_ARGS_WITH_CONFIG_DEFAULTS = (
    "use_cache",
    "vision_feature_layer",
    "vision_feature_select_strategy",
    "vision_aspect_ratio",
)


class ExplicitEnum(str, Enum):
    """Enum whose error message names the values it would have accepted."""

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of "
            f"{list(cls._value2member_map_.keys())}"
        )


class TensorType(ExplicitEnum):
    """Possible values for a ``return_tensors`` argument."""

    PYTORCH = "pt"
    NUMPY = "np"
    MLX = "mlx"


def can_return_tuple(func):
    """Call ``output.to_tuple()`` when ``return_dict=False`` was asked for.

    ``return_dict`` is read from the kwarg first and the config second, exactly
    as upstream does, so a caller can override a config that says otherwise.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = self.config.return_dict if hasattr(self, "config") else True
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output

    return wrapper


def merge_with_config_defaults(func):
    """Use config fields as defaults for args the caller left unset.

    Explicitly passed args and kwargs always win. Upstream additionally forces
    ``use_cache=False`` under gradient checkpointing while training; Fox Reader
    only ever runs these models in ``eval()`` under ``inference_mode``, so that
    branch is left out rather than carried as unreachable code.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        config = getattr(self, "config", None)
        if config is None:
            return func(self, *args, **kwargs)

        varnames = func.__code__.co_varnames
        for name in _ARGS_WITH_CONFIG_DEFAULTS:
            # -1 drops `self`, which is not part of *args.
            index = varnames.index(name) - 1 if name in varnames else None
            if index is not None and len(args) > index and args[index] is not None:
                continue  # passed positionally
            if kwargs.get(name) is not None:
                continue  # passed by keyword
            value = getattr(config, name, None)
            if value is None:
                continue
            if index is not None and len(args) > index:
                args = args[:index] + (value,) + args[index + 1 :]
            else:
                kwargs[name] = value

        return func(self, *args, **kwargs)

    return wrapper


__all__ = ["ExplicitEnum", "TensorType", "can_return_tuple", "merge_with_config_defaults"]
