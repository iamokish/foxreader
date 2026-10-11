"""A minimal, vendored subset of Hugging Face transformers for PP-OCR.

Laid out to mirror ``transformers/`` so that a model folder copied in from
upstream keeps working unedited -- its own ``from ...modeling_utils import
PreTrainedModel`` and ``from ..auto import AutoConfig`` resolve here, at the
same relative depth. See README.md for the supported symbol surface and for
what to do when a newly copied model needs something this package lacks.

Nothing is imported eagerly: pulling in all five model packages to serve one
would cost exactly what dropping transformers was meant to save.
"""
