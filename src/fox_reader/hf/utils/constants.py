"""Normalisation constants, verbatim from ``transformers.utils.constants``.

Only the two the vendored PP-OCR image processors name are re-exported by the
models; the rest are here because a model folder copied in later is as likely
to want ``IMAGENET_DEFAULT_*`` or the CLIP pair, and six floats cost nothing.
"""

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
