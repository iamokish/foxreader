"""Forward-hook output capture, ported from ``transformers.utils.output_capturing``.

A model declares what it is willing to record as a class attribute::

    _can_record_outputs = {"hidden_states": PPLCNetV4Block}

and wraps its forward in :func:`capture_outputs`. When the caller asks for
``output_hidden_states=True``, a forward hook on every matching submodule
appends that submodule's output to a context-local list, and the decorator
folds the list into the returned ``ModelOutput``.

This is on the hot path, not an optional debug feature: every PP-OCR backbone's
forward is wrapped in ``filter_output_hidden_states``, which *forces*
``output_hidden_states=True`` because the backbone builds its feature maps out
of the per-stage hidden states. Detection and recognition both run through it
on every call.

Differences from upstream, all of them subtractive:

* no ``torch.compile``/dynamo branch -- upstream swaps the ContextVar for a
  plain global when dynamo cannot trace it. Fox Reader never compiles these
  models, and a thread-unsafe global behind a web server would be a bug.
* no ``cross_attentions`` / ``mask_decoder_attentions`` backwards-compatibility
  aliases, and no ``vision_hidden_states`` special case. Those exist for model
  families this package does not vendor.
* hooks are installed from ``post_init`` for backbones (see
  :meth:`BackboneMixin.post_init`), so the common path never reaches the lazy
  installer at all.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

#: ``str(cls) -> cls._can_record_outputs``. Keyed by the repr rather than the
#: class itself so nothing here keeps a model class alive.
_CAN_RECORD_REGISTRY: dict[str, dict] = {}

#: Context-local, so two OCR requests on different threads cannot collect into
#: each other's lists. ``OCRService`` serialises predicts today; this does not
#: rely on that.
_active_collector: ContextVar[dict[str, list] | None] = ContextVar("output_collector", default=None)

#: Guards hook installation only. Two threads reaching a model's first forward
#: together must not install the same hook twice -- that would duplicate every
#: hidden state and silently corrupt the backbone's feature maps.
_hook_installation_lock = threading.Lock()


@dataclass
class OutputRecorder:
    """Which submodule output to record, and where to find it.

    ``target_class`` matches by ``isinstance``; ``class_name`` matches against
    the dotted module path's suffix, which is how composite models name a
    backbone layer they do not import. ``index`` picks an element when the
    submodule returns a tuple.
    """

    target_class: type[nn.Module] | None
    index: int = 0
    layer_name: str | None = None
    class_name: str | None = None
    capture_initial_hidden_state: bool = True


def install_output_capturing_hook(
    module: nn.Module, key: str, index: int, capture_initial_hidden_state: bool = True
) -> None:
    """Attach the forward hook that records ``key`` from ``module``."""

    def output_capturing_hook(module, args, output):
        collected = _active_collector.get()
        # Inactive unless the current forward asked for this key.
        if collected is None or key not in collected:
            return
        if capture_initial_hidden_state and key == "hidden_states" and len(collected[key]) == 0:
            collected[key].append(args[0])
        if not isinstance(output, tuple):
            collected[key].append(output)
        elif output[index] is not None:
            collected[key].append(output[index])

    module.register_forward_hook(output_capturing_hook)


def _recursively_install_hooks(
    parent_module: nn.Module, module_name: str, capture_tasks: list[tuple[str, OutputRecorder]]
) -> None:
    """Walk the module tree, honouring each nested model's own capture tasks.

    Recursive rather than a flat ``named_modules()`` sweep because a nested
    ``PreTrainedModel`` -- the backbone inside a detector, say -- declares its
    own ``_can_record_outputs``, and its children must use that rather than the
    parent's.
    """
    from ..modeling_utils import PreTrainedModel

    for name, module in parent_module.named_children():
        if isinstance(module, PreTrainedModel):
            install_all_output_capturing_hooks(module, prefix=f"{module_name}.{name}")
        else:
            _recursively_install_hooks(module, f"{module_name}.{name}", capture_tasks)

    for key, specs in capture_tasks:
        matches_class = specs.target_class is not None and isinstance(parent_module, specs.target_class)
        matches_name = specs.class_name is not None and module_name.endswith(specs.class_name)
        if not (matches_class or matches_name):
            continue
        if specs.layer_name is not None:
            target = "." + specs.layer_name.strip(".") + "."
            if target not in module_name + ".":
                continue
        install_output_capturing_hook(parent_module, key, specs.index, specs.capture_initial_hidden_state)


def install_all_output_capturing_hooks(model: Any, prefix: str | None = None) -> None:
    """Install every hook ``model._can_record_outputs`` asks for."""
    capture_flags = _CAN_RECORD_REGISTRY.get(str(model.__class__)) or {}

    capture_tasks: list[tuple[str, OutputRecorder]] = []
    for key, layer_specs in capture_flags.items():
        if not isinstance(layer_specs, list):
            layer_specs = [layer_specs]
        for specs in layer_specs:
            if not isinstance(specs, OutputRecorder):
                # Bare class or class-name string. Attentions live at tuple
                # index 1, hidden states at 0 -- upstream's convention.
                index = 0 if "hidden_states" in key else 1
                class_name = specs if isinstance(specs, str) else None
                target_class = None if isinstance(specs, str) else specs
                specs = OutputRecorder(target_class=target_class, index=index, class_name=class_name)
            capture_tasks.append((key, specs))

    _recursively_install_hooks(model, prefix if prefix is not None else "", capture_tasks)
    model._output_capturing_hooks_installed = True


def maybe_install_capturing_hooks(model: Any) -> None:
    """Install the hooks once, from whichever thread gets there first."""
    if getattr(model, "_output_capturing_hooks_installed", False):
        return
    with _hook_installation_lock:
        # Re-check: another thread may have installed them while we waited.
        if getattr(model, "_output_capturing_hooks_installed", False):
            return
        install_all_output_capturing_hooks(model)


def capture_outputs(func=None, *, tie_last_hidden_states: bool = True):
    """Collect hook-recorded outputs into the wrapped forward's return value.

    Args:
        tie_last_hidden_states: overwrite ``hidden_states[-1]`` with
            ``last_hidden_state``. True for everything vendored here; CLIP-style
            vision towers turn it off so the last entry stays pre-layernorm.
    """

    def wrapped_fn(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Popped, not forwarded: inner modules must keep returning a dict
            # even when the outermost caller asked for a tuple.
            return_dict = kwargs.pop("return_dict", getattr(self.config, "return_dict", True))

            capturable = _CAN_RECORD_REGISTRY.get(str(self.__class__)) or {}
            wanted = {
                key: kwargs.get(f"output_{key}", getattr(self.config, f"output_{key}", False))
                for key in capturable
            }
            collected: dict[str, list] = {key: [] for key, on in wanted.items() if on}

            if not collected:
                # Nothing to record: skip the hooks entirely, but still honour
                # the `return_dict` this decorator just swallowed.
                outputs = func(self, *args, **kwargs)
                return outputs.to_tuple() if return_dict is False else outputs

            maybe_install_capturing_hooks(self)
            token = _active_collector.set(collected)
            try:
                outputs = func(self, *args, **kwargs)
            finally:
                _active_collector.reset(token)

            for key in collected:
                if key == "hidden_states" and tie_last_hidden_states and hasattr(outputs, "last_hidden_state"):
                    # The hook saw the last block's raw output; the forward may
                    # have normalised or reshaped it afterwards. The caller
                    # wants what the forward returned.
                    collected[key] = collected[key][:-1] + [outputs.last_hidden_state]
                outputs[key] = tuple(collected[key])

            return outputs.to_tuple() if return_dict is False else outputs

        return wrapper

    return wrapped_fn(func) if func is not None else wrapped_fn


__all__ = [
    "OutputRecorder",
    "capture_outputs",
    "install_all_output_capturing_hooks",
    "maybe_install_capturing_hooks",
]
