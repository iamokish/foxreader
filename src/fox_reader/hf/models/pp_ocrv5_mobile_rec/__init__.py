"""PP-OCRv5 mobile recogniser. Vendored from transformers; see ../../README.md."""

from .configuration_pp_ocrv5_mobile_rec import PPOCRV5MobileRecConfig
from .modeling_pp_ocrv5_mobile_rec import (
    PPOCRV5MobileRecEncoderWithSVTR,
    PPOCRV5MobileRecForTextRecognition,
    PPOCRV5MobileRecModel,
    PPOCRV5MobileRecPreTrainedModel,
)

__all__ = [
    "PPOCRV5MobileRecConfig",
    "PPOCRV5MobileRecEncoderWithSVTR",
    "PPOCRV5MobileRecForTextRecognition",
    "PPOCRV5MobileRecModel",
    "PPOCRV5MobileRecPreTrainedModel",
]
