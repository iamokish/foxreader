"""Model-type registry -- the stand-in for ``transformers.models.auto``.

Upstream's auto classes build four large name-to-class maps over every model
transformers ships and resolve them lazily through an import hook. Measured on
this venv, ``from transformers import AutoModelForObjectDetection`` alone cost
**5.3 seconds** -- more than torch's own import -- which is most of why
``import fox_reader.ocr`` used to take 15 seconds.

Here the maps are four literal dicts over seven folders, so resolution is a
dict lookup plus one module import.

Adding a model
--------------
1. Copy ``transformers/models/<model_type>/`` into ``../<model_type>/``
   (configuration, modeling and image_processing files, unchanged).
2. Write its ``__init__.py`` -- plain imports, as the siblings do.
3. Add one line to :data:`MODEL_TYPE_TO_MODULE` and one to whichever of the
   task maps below applies.

``tests/test_hf_vendored.py`` then checks that every relative import in the
new file resolves against the shim, so a model needing a symbol this package
does not have fails at test time with that symbol named.

Why explicit tables rather than deriving class names
----------------------------------------------------
Folder name, ``model_type`` and class prefix do line up for every PP-OCR model
(``pp_ocrv6_medium_det`` -> ``PPOCRV6MediumDet*``), so this could be computed.
It is not, for two reasons: a derived name fails at load time with an
``AttributeError`` rather than at review time, and a static table is something
Nuitka can follow -- which is what lets ``packaging/build.py`` drop the
per-model ``--include-package`` flags it used to need.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

#: ``model_type`` -> the subpackage holding its code.
#:
#: Equal to the folder name in every case so far. Kept as data because the
#: *checkpoint* name need not match: PP-OCRv6_medium_rec declares
#: ``"model_type": "pp_ocrv6_small_rec"`` and loads from that folder.
MODEL_TYPE_TO_MODULE = {
    "pp_lcnet_v3": "pp_lcnet_v3",
    "pp_lcnet_v4": "pp_lcnet_v4",
    "pp_ocrv5_mobile_rec": "pp_ocrv5_mobile_rec",
    "pp_ocrv6_medium_det": "pp_ocrv6_medium_det",
    "pp_ocrv6_small_rec": "pp_ocrv6_small_rec",
}

#: ``model_type`` -> config class name.
CONFIG_MAPPING_NAMES = {
    "pp_lcnet_v3": "PPLCNetV3Config",
    "pp_lcnet_v4": "PPLCNetV4Config",
    "pp_ocrv5_mobile_rec": "PPOCRV5MobileRecConfig",
    "pp_ocrv6_medium_det": "PPOCRV6MediumDetConfig",
    "pp_ocrv6_small_rec": "PPOCRV6SmallRecConfig",
}

#: ``model_type`` -> backbone class name. Read by
#: :func:`~fox_reader.hf.backbone_utils.load_backbone`.
MODEL_FOR_BACKBONE_MAPPING_NAMES = {
    "pp_lcnet_v3": "PPLCNetV3Backbone",
    "pp_lcnet_v4": "PPLCNetV4Backbone",
}

MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = {
    "pp_ocrv6_medium_det": "PPOCRV6MediumDetForObjectDetection",
}

MODEL_FOR_TEXT_RECOGNITION_MAPPING_NAMES = {
    "pp_ocrv5_mobile_rec": "PPOCRV5MobileRecForTextRecognition",
    "pp_ocrv6_small_rec": "PPOCRV6SmallRecForTextRecognition",
}

#: ``model_type`` -> ``(module, image processor class)``.
#:
#: Not derivable, and this is the one place upstream's own mapping is the only
#: source of truth: three of these checkpoints reuse a *different* model's
#: processor. PP-OCRv6_medium_det says so in its own
#: ``preprocessor_config.json`` (``"image_processor_type":
#: "PPOCRV5ServerDetImageProcessor"``), but the Korean mobile recogniser's
#: config names no processor at all, so only the mapping resolves it.
IMAGE_PROCESSOR_MAPPING_NAMES = {
    "pp_ocrv5_mobile_rec": ("pp_ocrv5_server_rec", "PPOCRV5ServerRecImageProcessor"),
    "pp_ocrv6_medium_det": ("pp_ocrv5_server_det", "PPOCRV5ServerDetImageProcessor"),
    "pp_ocrv6_small_rec": ("pp_ocrv6_small_rec", "PPOCRV6SmallRecImageProcessor"),
}

#: Class name -> module, for resolving an explicit ``image_processor_type``
#: without having to know which model_type asked for it.
_IMAGE_PROCESSOR_MODULES = {
    name: module for module, name in IMAGE_PROCESSOR_MAPPING_NAMES.values()
}

CONFIG_NAME = "config.json"
IMAGE_PROCESSOR_NAME = "preprocessor_config.json"


def _import_class(module: str, class_name: str) -> type:
    """Import ``class_name`` from ``fox_reader.hf.models.<module>``."""
    package = importlib.import_module(f".{module}", __name__.rsplit(".", 1)[0])
    try:
        return getattr(package, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"{class_name} is not exported by fox_reader.hf.models.{module}. "
            f"Add it to that package's __init__.py."
        ) from exc


def _read_json(path: Path, filename: str) -> dict[str, Any]:
    target = path / filename if path.is_dir() else path
    if not target.is_file():
        raise OSError(f"{target} does not exist. Download the model again from the setup page.")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OSError(f"{target} could not be read ({type(exc).__name__}: {exc})") from exc


def _resolve(mapping: dict[str, Any], model_type: str | None, what: str) -> type:
    if not model_type:
        raise ValueError(f"The config names no model_type, so no {what} can be chosen for it.")
    if model_type not in mapping:
        raise ValueError(
            f"{model_type!r} has no {what} registered in fox_reader.hf.models.auto. "
            f"Known: {sorted(mapping)}. See that module's docstring for how to add one."
        )
    entry = mapping[model_type]
    if isinstance(entry, tuple):
        return _import_class(*entry)
    return _import_class(MODEL_TYPE_TO_MODULE[model_type], entry)


class _LazyConfigMapping:
    """``CONFIG_MAPPING[model_type]`` -> config class, imported on demand.

    A mapping rather than a dict of classes so that importing this module does
    not import all five model packages -- which is the whole point of the
    exercise. `consolidate_backbone_kwargs_to_config` indexes it directly.
    """

    def __getitem__(self, model_type: str) -> type:
        return _resolve(CONFIG_MAPPING_NAMES, model_type, "config class")

    def __contains__(self, model_type: object) -> bool:
        return model_type in CONFIG_MAPPING_NAMES

    def __iter__(self):
        return iter(CONFIG_MAPPING_NAMES)

    def keys(self):
        return CONFIG_MAPPING_NAMES.keys()


CONFIG_MAPPING = _LazyConfigMapping()


class AutoConfig:
    """Resolves a config class from a ``model_type``."""

    #: Read by `PreTrainedConfig.from_dict`, which uses it to tell "recurse
    #: into this sub-config by its own model_type" from "this sub_configs entry
    #: already names a concrete class".
    _is_auto_class = True

    def __init__(self):
        raise TypeError("AutoConfig is a resolver; use AutoConfig.from_pretrained(...).")

    @classmethod
    def for_model_type(cls, model_type: str | None) -> type:
        return _resolve(CONFIG_MAPPING_NAMES, model_type, "config class")

    @classmethod
    def from_pretrained(cls, model_dir: str | os.PathLike, **kwargs):
        raw = _read_json(Path(model_dir), CONFIG_NAME)
        config_class = cls.for_model_type(raw.get("model_type"))
        raw.setdefault("name_or_path", str(model_dir))
        return config_class.from_dict({**raw, **kwargs})

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        config_class = cls.for_model_type(config_dict.get("model_type"))
        return config_class.from_dict(config_dict, **kwargs)


class _AutoModel:
    """Shared resolution for the task-specific auto-model classes."""

    _mapping: dict[str, str] = {}
    _task = "model"

    def __init__(self):
        raise TypeError(
            f"{type(self).__name__} is a resolver; use .from_pretrained(...) or .from_config(...)."
        )

    @classmethod
    def for_model_type(cls, model_type: str | None) -> type:
        return _resolve(cls._mapping, model_type, cls._task)

    @classmethod
    def from_config(cls, config, **kwargs):
        model_class = cls.for_model_type(type(config).model_type)
        return model_class(config, **kwargs)

    @classmethod
    def from_pretrained(cls, model_dir: str | os.PathLike, **kwargs):
        raw = _read_json(Path(model_dir), CONFIG_NAME)
        model_class = cls.for_model_type(raw.get("model_type"))
        return model_class.from_pretrained(model_dir, **kwargs)


class AutoBackbone(_AutoModel):
    _mapping = MODEL_FOR_BACKBONE_MAPPING_NAMES
    _task = "backbone class"


class AutoModelForObjectDetection(_AutoModel):
    _mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES
    _task = "object-detection model class"


class AutoModelForTextRecognition(_AutoModel):
    _mapping = MODEL_FOR_TEXT_RECOGNITION_MAPPING_NAMES
    _task = "text-recognition model class"


class AutoImageProcessor:
    """Resolves an image processor from a model folder.

    Two sources, in upstream's order of precedence: an explicit
    ``image_processor_type`` in ``preprocessor_config.json`` (which the
    detector checkpoint carries), then :data:`IMAGE_PROCESSOR_MAPPING_NAMES`
    keyed by the ``model_type`` from ``config.json`` (which is the only thing
    that resolves the recogniser checkpoints, as theirs name no processor).
    """

    def __init__(self):
        raise TypeError("AutoImageProcessor is a resolver; use .from_pretrained(...).")

    @classmethod
    def from_pretrained(cls, model_dir: str | os.PathLike, **kwargs):
        path = Path(model_dir)
        raw = _read_json(path, IMAGE_PROCESSOR_NAME)

        explicit = raw.get("image_processor_type")
        if explicit:
            module = _IMAGE_PROCESSOR_MODULES.get(explicit)
            if module is None:
                raise ValueError(
                    f"{IMAGE_PROCESSOR_NAME} in {path} asks for {explicit!r}, which is not "
                    f"vendored. Known: {sorted(_IMAGE_PROCESSOR_MODULES)}. See "
                    "fox_reader.hf.models.auto for how to add it."
                )
            processor_class = _import_class(module, explicit)
        else:
            # `preprocessor_config.json` may carry its own `model_type`; fall
            # back to the model's `config.json`, which always does.
            model_type = raw.get("model_type") or _read_json(path, CONFIG_NAME).get("model_type")
            processor_class = _resolve(
                IMAGE_PROCESSOR_MAPPING_NAMES, model_type, "image processor class"
            )

        return processor_class.from_dict(raw, **kwargs)


__all__ = [
    "CONFIG_MAPPING",
    "CONFIG_MAPPING_NAMES",
    "IMAGE_PROCESSOR_MAPPING_NAMES",
    "MODEL_FOR_BACKBONE_MAPPING_NAMES",
    "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES",
    "MODEL_FOR_TEXT_RECOGNITION_MAPPING_NAMES",
    "MODEL_TYPE_TO_MODULE",
    "AutoBackbone",
    "AutoConfig",
    "AutoImageProcessor",
    "AutoModelForObjectDetection",
    "AutoModelForTextRecognition",
]
