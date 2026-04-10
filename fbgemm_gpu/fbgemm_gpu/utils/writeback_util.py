# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

# ---- Primitives: compute indices + apply mask ----


def writeback_apply_mask(
    grad: tuple[torch.Tensor],
    keep_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Apply precomputed dedup mask to gradient. No GPU→CPU sync.

    Args:
        grad: Gradient tensor tuple from backward hook
        keep_indices: Precomputed indices of rows to keep

    Returns:
        Masked gradient tensor
    """
    grad_tensor = grad[0]
    if keep_indices.numel() == 0:
        return grad_tensor
    mask = torch.zeros_like(grad_tensor)
    mask[keep_indices] = grad_tensor[keep_indices]
    return mask


def compute_writeback_indices(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    feature_table_map: list[int],
) -> torch.Tensor:
    """
    Compute the dedup index for writeback (bag mode) during forward pass.

    Returns ``original_index`` — the gradient row indices to keep.

    Args:
        indices: Embedding indices tensor
        offsets: Offsets tensor for batched embeddings
        feature_table_map: Mapping from feature to table

    Returns:
        original_index tensor (GPU). Empty tensor if indices is empty.
    """
    if indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=indices.device)

    num_of_tables = len(feature_table_map)
    batch_size = offsets.shape[0] // num_of_tables
    max_indices = indices.max()
    torch._assert_async(
        num_of_tables * max_indices < torch.iinfo(indices.dtype).max,
        "num_of_tables * max_indices exceeds dtype max",
    )
    non_empty_index = (offsets[1:] - offsets[:-1]).nonzero().flatten()
    # disable dedup across different table
    indices = ((offsets[non_empty_index]) // batch_size) * (1 + max_indices) + indices
    _, idx, counts = torch.unique(
        indices, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(indices.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    original_index = non_empty_index[first_indices]

    return original_index


def compute_writeback_indices_first_feature_only(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    feature_table_map: list[int],
) -> torch.Tensor:
    """
    Compute dedup index for the first-feature-only writeback case.

    Args:
        indices: Embedding indices tensor
        offsets: Offsets tensor for batched embeddings
        feature_table_map: Mapping from feature to table

    Returns:
        first_indices tensor (GPU). Empty tensor if indices is empty.
    """
    num_of_tables = len(feature_table_map)
    batch_size = (offsets.shape[0] - 1) // num_of_tables
    shrink_indices = indices[: offsets[batch_size]]
    if shrink_indices.numel() == 0 or indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=indices.device)
    torch._assert_async(
        num_of_tables * indices.max() < torch.iinfo(indices.dtype).max,
        "num_of_tables * max_indices exceeds dtype max",
    )

    _, idx, counts = torch.unique(
        shrink_indices, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(shrink_indices.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]

    return first_indices


def compute_writeback_indices_nobag(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    feature_table_map: list[int],
) -> torch.Tensor:
    """
    Compute the dedup index for writeback (nobag/sequence mode) during forward pass.

    Unlike bag mode, nobag mode uses first_indices directly as gradient row indices
    (no mapping through non_empty_index) because each index maps 1:1 to a gradient row.

    Args:
        indices: Embedding indices tensor
        offsets: Offsets tensor for batched embeddings
        feature_table_map: Mapping from feature to table

    Returns:
        first_indices tensor (GPU). Empty tensor if indices is empty.
    """
    if indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=indices.device)

    num_of_tables: int = len(feature_table_map)
    batch_size = offsets.shape[0] // num_of_tables
    max_indices = indices.max()
    torch._assert_async(
        num_of_tables * max_indices < torch.iinfo(indices.dtype).max,
        "num_of_tables * max_indices exceeds dtype max",
    )
    non_empty_index = (offsets[1:] - offsets[:-1]).nonzero().flatten()
    # disable dedup across different table
    indices = ((offsets[non_empty_index]) // batch_size) * (1 + max_indices) + indices
    # TODO: revisit if dedup is needed when EC dedup is enabled.
    _, idx, counts = torch.unique(
        indices, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(indices.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]

    return first_indices


def compute_writeback_indices_dispatch(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    feature_table_map: list[int],
    writeback_first_feature_only: bool = False,
    nobag: bool = False,
) -> torch.Tensor:
    """
    Dispatch to the appropriate compute_writeback_indices variant.

    Args:
        indices: Embedding indices tensor
        offsets: Offsets tensor for batched embeddings
        feature_table_map: Mapping from feature to table
        writeback_first_feature_only: If True, use first-feature-only variant
        nobag: If True, use nobag/sequence variant

    Returns:
        Precomputed dedup indices tensor
    """
    if writeback_first_feature_only:
        return compute_writeback_indices_first_feature_only(
            indices, offsets, feature_table_map
        )
    elif nobag:
        return compute_writeback_indices_nobag(indices, offsets, feature_table_map)
    else:
        return compute_writeback_indices(indices, offsets, feature_table_map)


# ---- Entry points: compute + apply in one call ----


def writeback_update_gradient(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    grad: tuple[torch.Tensor],
    feature_table_map: list[int],
    original_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Update gradient tensor by deduplicating indices across all features/tables.
    For duplicate indices, only the first occurrence receives the gradient to achieve the assign purpose via gradient update

    NOTE: This function is not supporting VBE yet

    Args:
        indices (torch.Tensor): Embedding indices tensor
        offsets (torch.Tensor): Offsets tensor for batched embeddings
        grad (tuple[torch.Tensor]): Gradient tensor to be updated
        feature_table_map (list[int]): Mapping from feature to table
        original_index (Optional[torch.Tensor]): Precomputed indices. If None, computed here.

    Returns:
        torch.Tensor: Updated gradient tensor with duplicates masked out
    """
    if original_index is None:
        original_index = compute_writeback_indices(indices, offsets, feature_table_map)

    return writeback_apply_mask(grad, original_index)


def writeback_update_gradient_first_feature_only(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    grad: tuple[torch.Tensor],
    feature_table_map: list[int],
    first_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Special case of writeback_update_gradient where gradient only needs to be updated for the first feature. Other features will be forward-only

    NOTE: This function is not supporting VBE yet

    Args:
        indices (torch.Tensor): Embedding indices tensor
        offsets (torch.Tensor): Offsets tensor for batched embeddings
        grad (tuple[torch.Tensor]): Gradient tensor to be updated
        feature_table_map (list[int]): Mapping from feature to table
        first_indices (Optional[torch.Tensor]): Precomputed indices. If None, computed here.

    Returns:
        torch.Tensor: Updated gradient tensor with duplicates masked out
    """
    if first_indices is None:
        first_indices = compute_writeback_indices_first_feature_only(
            indices, offsets, feature_table_map
        )

    return writeback_apply_mask(grad, first_indices)


def writeback_update_gradient_nobag(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    grad: tuple[torch.Tensor],
    feature_table_map: list[int],
    first_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Hook for using TBE no bag to update gradient tensor by deduplicating indices
    (in one feature or across multiple features) across all features/tables.
    For duplicate indices, only the first occurrence receives the gradient to achieve the assign purpose via gradient update

    Args:
        indices (torch.Tensor): Embedding indices tensor
        offsets (torch.Tensor): Offsets tensor for batched embeddings
        grad (tuple[torch.Tensor]): Gradient tensor to be updated
        feature_table_map (list[int]): Mapping from feature to table
        first_indices (Optional[torch.Tensor]): Precomputed indices. If None, computed here.

    Returns:
        torch.Tensor: Updated gradient tensor with duplicates masked out
    """
    if first_indices is None:
        first_indices = compute_writeback_indices_nobag(
            indices, offsets, feature_table_map
        )

    return writeback_apply_mask(grad, first_indices)


# ---- Entry point ----


def writeback_gradient(
    grad: tuple[torch.Tensor],
    indices: torch.Tensor,
    offsets: torch.Tensor,
    feature_table_map: list[int],
    writeback_first_feature_only: bool = False,
    nobag: bool = False,
    precomputed_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor]:
    """
    Compute deduplicated gradient for writeback operation.

    Args:
        grad (tuple[torch.Tensor]): Gradient tensor to be updated
        indices (torch.Tensor): Embedding indices tensor
        offsets (torch.Tensor): Offsets tensor for batched embeddings
        feature_table_map (list[int]): Mapping from feature to table
        writeback_first_feature_only (bool): If True, only first feature will apply gradient update, other features will be read-only
        nobag (bool): If True, we use TBE with sequence embeddings, otherwise we use TBE with pooled embeddings.
        precomputed_indices (Optional[torch.Tensor]): Precomputed dedup indices from forward pass. If None, computed here.

    Returns:
        tuple[torch.Tensor]: Tuple containing the updated gradient tensor
    """
    if writeback_first_feature_only:
        return (
            writeback_update_gradient_first_feature_only(
                indices, offsets, grad, feature_table_map, precomputed_indices
            ),
        )
    elif nobag:
        return (
            writeback_update_gradient_nobag(
                indices, offsets, grad, feature_table_map, precomputed_indices
            ),
        )
    else:
        return (
            writeback_update_gradient(
                indices, offsets, grad, feature_table_map, precomputed_indices
            ),
        )
