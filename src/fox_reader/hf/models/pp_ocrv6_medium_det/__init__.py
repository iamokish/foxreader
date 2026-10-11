"""PP-OCRv6 medium detector. Vendored from transformers; see ../../README.md."""

from .configuration_pp_ocrv6_medium_det import PPOCRV6MediumDetConfig
from .modeling_pp_ocrv6_medium_det import (
    PPOCRV6MediumDetForObjectDetection,
    PPOCRV6MediumDetModel,
    PPOCRV6MediumDetPreTrainedModel,
)

__all__ = [
    "PPOCRV6MediumDetConfig",
    "PPOCRV6MediumDetForObjectDetection",
    "PPOCRV6MediumDetModel",
    "PPOCRV6MediumDetPreTrainedModel",
]
