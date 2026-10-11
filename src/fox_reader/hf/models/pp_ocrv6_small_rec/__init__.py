"""PP-OCRv6 small recogniser. Vendored from transformers; see ../../README.md.

Note the name: PP-OCRv6_medium_rec's `config.json` declares
`"model_type": "pp_ocrv6_small_rec"`, so the medium checkpoint loads through
this folder. The folder is named after the model type, not the checkpoint.
"""

from .configuration_pp_ocrv6_small_rec import PPOCRV6SmallRecConfig
from .image_processing_pp_ocrv6_small_rec import PPOCRV6SmallRecImageProcessor
from .modeling_pp_ocrv6_small_rec import (
    PPOCRV6SmallRecEncoderWithSVTR,
    PPOCRV6SmallRecForTextRecognition,
    PPOCRV6SmallRecModel,
    PPOCRV6SmallRecPreTrainedModel,
)

__all__ = [
    "PPOCRV6SmallRecConfig",
    "PPOCRV6SmallRecEncoderWithSVTR",
    "PPOCRV6SmallRecForTextRecognition",
    "PPOCRV6SmallRecImageProcessor",
    "PPOCRV6SmallRecModel",
    "PPOCRV6SmallRecPreTrainedModel",
]
