"""PP-OCRv5 server recogniser -- image processor only.

Vendored for the same reason as pp_ocrv5_server_det: the Korean
PP-OCRv5_mobile_rec checkpoint resolves its image processor to
`PPOCRV5ServerRecImageProcessor` while using the mobile model code.
"""

from .image_processing_pp_ocrv5_server_rec import PPOCRV5ServerRecImageProcessor

__all__ = ["PPOCRV5ServerRecImageProcessor"]
