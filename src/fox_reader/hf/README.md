# `fox_reader.hf` — a vendored subset of Hugging Face transformers

This package exists so Fox Reader can run PaddleOCR's PP-OCR models without
depending on `paddleocr`, `paddlex`, `transformers` or `accelerate`. The weights
are plain `model.safetensors` and the models are plain `torch.nn`; those four
packages were between us and them only to turn a `model_type` string into a
class name.

Measured on a clean CPU venv when this was done:

| | before | after |
|---|---|---|
| `import fox_reader.ocr` | 15.4 s | 2.0 s |
| └ of which | `paddlex` 3.2 s · `transformers` 1.6 s · building `AutoModel*` maps 5.3 s | — |
| `MultiLangOCR()` (3 models) | 8.5 s | 5.8 s (4.4 s of it `import torchvision`) |
| warm `predict` | 0.67 s | 0.46 s |
| reachable packages | 92 | 54 |

Output is bit-identical to the old stack. That was checked before the
dependencies were removed — see `tests/manual/smoke_ocr_parity.py`.

## Layout

The directory mirrors `transformers/` **exactly**, and that is the whole trick:

```
fox_reader/hf/
  activations.py            backbone_utils.py       configuration_utils.py
  feature_extraction_utils.py                       image_processing_backends.py
  image_transforms.py       image_utils.py          modeling_layers.py
  modeling_outputs.py       modeling_utils.py       processing_utils.py
  utils/{__init__,constants,generic,import_utils,output_capturing}.py
  models/
    auto/__init__.py          <- the model_type registry
    pp_ocrv6_medium_det/      <- copied verbatim from transformers
    pp_lcnet_v4/              <- copied verbatim from transformers
    ...
```

A model file copied from upstream keeps working **unedited**, because its own
imports resolve here at the same relative depth:

```python
from ...modeling_utils import PreTrainedModel   # -> fox_reader.hf.modeling_utils
from ..auto import AutoConfig                   # -> fox_reader.hf.models.auto
```

## Adding or updating a model

1. Copy `transformers/models/<model_type>/` to `models/<model_type>/`. Take the
   `configuration_*.py`, `modeling_*.py` and `image_processing_*.py` files.
   **Do not** take `modular_*.py` (a code-generation input, not runtime code)
   or upstream's `__init__.py`.
2. Write `models/<model_type>/__init__.py` — plain imports, copy a sibling. This
   is the only hand-written file per model. It replaces upstream's
   `_LazyModule` + `define_import_structure`, which resolves imports through an
   AST scan at runtime; plain imports keep the graph static, which is what lets
   `packaging/build.py` get away with no per-model Nuitka flag.
3. Add one line to `MODEL_TYPE_TO_MODULE` and `CONFIG_MAPPING_NAMES` in
   `models/auto/__init__.py`, plus one to whichever task map applies
   (`MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES`, `..._TEXT_RECOGNITION_...`,
   `..._BACKBONE_...`, `IMAGE_PROCESSOR_MAPPING_NAMES`).
4. `pytest tests/test_hf_vendored.py`

Step 4 is the point. `TestImportSurface` reads every vendored file's relative
imports and asserts each one resolves, so a model that needs more of the shim
fails there with the missing symbol named — rather than at a user's first OCR
run, eleven frames into a model constructor.

If a copied model does need something new, add it to the matching shim module
and keep the signature upstream's. The rule throughout: **the vendored files are
never edited.** If one has to be, the shim is wrong.

## What the shim provides

The complete surface the vendored files import — 24 symbols. Anything not on
this list is not available:

| Module | Symbols |
|---|---|
| `activations` | `ACT2FN` |
| `backbone_utils` | `BackboneConfigMixin`, `BackboneMixin`, `filter_output_hidden_states`, `consolidate_backbone_kwargs_to_config`, `load_backbone` |
| `configuration_utils` | `PreTrainedConfig` |
| `feature_extraction_utils` | `BatchFeature` |
| `image_processing_backends` | `TorchvisionBackend` |
| `image_transforms` | `group_images_by_shape`, `reorder_images` |
| `image_utils` | `PILImageResampling`, `SizeDict` |
| `modeling_layers` | `GradientCheckpointingLayer` |
| `modeling_outputs` | `BackboneOutput`, `BaseModelOutputWithNoAttention` |
| `modeling_utils` | `PreTrainedModel`, `ALL_ATTENTION_FUNCTIONS` |
| `processing_utils` | `ImagesKwargs`, `Unpack` |
| `utils` | `TransformersKwargs`, `auto_docstring`, `can_return_tuple`, `requires_backends`, `is_cv2_available` |
| `utils.constants` | `IMAGENET_STANDARD_MEAN`, `IMAGENET_STANDARD_STD` |
| `utils.generic` | `TensorType`, `merge_with_config_defaults` |
| `utils.import_utils` | `requires` |
| `utils.output_capturing` | `capture_outputs` |
| `models.auto` | `AutoConfig` (+ `AutoBackbone`, `AutoModelFor*`, `AutoImageProcessor`) |

## What is deliberately missing

Nothing generative. `GenerationMixin`, `cache_utils`, `masking_utils`,
`modeling_rope_utils`, `integrations` and `vision_utils` are not here, so
`paddleocr_vl`, `pp_formulanet`, `pp_doclayout_v2`/`v3` and `pp_chart2table`
cannot be vendored without adding roughly three times the current shim. None is
reachable from Fox Reader: PaddleOCR-VL runs through llama.cpp GGUF in
`fox_reader/ocr_vl.py`, and there is no layout, formula or chart feature.

Also absent, from modules that *are* here: hub downloads (`from_pretrained`
takes a local directory — `fox_reader.assets` has already fetched the weights),
`save_pretrained`/`push_to_hub`, quantization, PEFT, device maps, tensor and
pipeline parallel plans, weight tying, `PilBackend`, timm backbones, and
transformers' docstring generator (`auto_docstring` is a no-op).

## Things that look like bugs and are not

Both are load-bearing; each has a longer comment at the site.

- **`group_images_by_shape` always groups.** Upstream intends to *disable*
  grouping on CPU, but checks `torch.device("cpu") == "cpu"`, which is `False`.
  The branch never fires. Reproducing the quirk is required: with grouping
  disabled the dict keys are indices, and
  `PPOCRV6SmallRecImageProcessor.get_target_size` unpacks them as
  `(height, width)`.
- **`PreTrainedConfig.__init_subclass__` applies `@dataclass`.** The vendored
  config files declare fields as bare annotations and are decorated with
  `@strict` alone. Without this, `huggingface_hub`'s strict wrapper sees only
  the base class's fields and forwards every subclass field to `__post_init__`.

## Licensing

Everything in this directory is Apache-2.0, copyright The HuggingFace Inc. team
and The PaddlePaddle Team, copied from or derived from transformers 5.14.1. It
keeps that licence; Fox Reader's own AGPL-3.0-or-later covers the combined work.
See `NOTICE` § 1C.
