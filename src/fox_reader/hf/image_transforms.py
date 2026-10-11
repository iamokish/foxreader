"""Batch grouping helpers, trimmed from ``transformers.image_transforms``.

The vendored image processors resize and normalise a *list* of images. Doing
that one at a time wastes the GPU, so same-shaped images are stacked into one
tensor, transformed once, and then scattered back into the caller's order.
That is the whole job of :func:`group_images_by_shape` and
:func:`reorder_images`.

Worth knowing: the vendored ``_preprocess`` methods pass
``disable_grouping=None``, and that resolves to *grouping enabled* -- not to
the CPU-sensitive choice upstream's comment describes. The reason is a
type-mismatch quirk in upstream, documented at the branch in
:func:`group_images_by_shape`; it matters because the recogniser processor can
only consume the grouped layout.

Not ported: the nested (list-of-lists) structure used by models that split an
image into patches, and the paired-input grouping used to carry per-image
metadata alongside. No vendored PP-OCR processor uses either.
"""

from __future__ import annotations

from collections import defaultdict

import torch


def group_images_by_shape(
    images: list[torch.Tensor] | torch.Tensor,
    *,
    disable_grouping: bool | None,
) -> tuple[dict, dict]:
    """Stack same-shaped images together.

    Returns ``(grouped, index)`` where ``grouped`` maps a shape key to a
    stacked ``(N, C, H, W)`` tensor and ``index`` maps each original position
    to ``(shape_key, offset_within_group)``.

    Args:
        disable_grouping: ``None`` means "group", matching what upstream
            actually does -- see below. An explicit bool is honoured.
    """
    if disable_grouping is None:
        # Upstream intends to disable grouping on CPU, where stacking costs a
        # copy that the batched kernel does not repay. The check it uses is
        #
        #     disable_grouping = _get_device_from_images(...) == "cpu"
        #
        # and `torch.device("cpu") == "cpu"` is False (verified on torch
        # 2.13), so the branch never fires and grouping is always on.
        #
        # That quirk is reproduced rather than corrected, because it is the
        # path the vendored processors were written against: with grouping
        # disabled the keys below are indices, and
        # `PPOCRV6SmallRecImageProcessor.get_target_size` unpacks them as
        # `(height, width)` -- so "fixing" this raises TypeError on the first
        # crop. Change it only together with that method.
        disable_grouping = False

    if disable_grouping:
        # Keyed by index, not shape, so two same-shaped images stay distinct.
        # This is upstream's layout; the recogniser processor cannot consume it
        # (see above), which is why nothing asks for it.
        return (
            {i: image.unsqueeze(0) for i, image in enumerate(images)},
            {i: (i, 0) for i in range(len(images))},
        )

    grouped: dict[tuple[int, ...], list[torch.Tensor]] = defaultdict(list)
    index: dict[int, tuple] = {}
    for i, image in enumerate(images):
        shape = tuple(image.shape[1:])
        grouped[shape].append(image)
        index[i] = (shape, len(grouped[shape]) - 1)

    return {shape: torch.stack(group, dim=0) for shape, group in grouped.items()}, index


def reorder_images(
    processed_images: dict,
    grouped_images_index: dict,
) -> list[torch.Tensor]:
    """Undo :func:`group_images_by_shape`, restoring the caller's order."""
    return [
        processed_images[grouped_images_index[i][0]][grouped_images_index[i][1]]
        for i in range(len(grouped_images_index))
    ]


__all__ = ["group_images_by_shape", "reorder_images"]
