"""PP-OCRv5 server detector -- image processor only.

The modeling files are not vendored: Fox Reader runs PP-OCRv6_medium_det, which
reuses this processor (its `preprocessor_config.json` names
`PPOCRV5ServerDetImageProcessor`) but has its own weights and model code. Copy
`modeling_pp_ocrv5_server_det.py` and `configuration_pp_ocrv5_server_det.py` in
here if the server detector itself is ever wanted.
"""

from .image_processing_pp_ocrv5_server_det import PPOCRV5ServerDetImageProcessor

__all__ = ["PPOCRV5ServerDetImageProcessor"]
