# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def writeback_update_gradient(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    grad: torch.Tensor,
    feature_table_map: list[int],
) -> torch.Tensor:
    """
    Update gradient tensor by deduplicating indices across all features/tables.
    For duplicate indices, only the first occurrence receives the gradient to achieve the assign purpose via gradient update

    NOTE: This function is not supporting VBE yet

    Args:
        indices (torch.Tensor): Embedding indices tensor
        offsets (torch.Tensor): Offsets tensor for batched embeddings
        grad (torch.Tensor): Gradient tensor to be updated
        feature_table_map (list[int]): Mapping from feature to table

    Returns:
        torch.Tensor: Updated gradient tensor with duplicates masked out
    """
    if indices.numel() == 0:
        return grad[0]
    # get num of feature to estimate batch size
    num_of_tables = len(feature_table_map)
    assert num_of_tables * indices.max() < torch.iinfo(indices.dtype).max
    batch_size = offsets.shape[0] // num_of_tables
    max_indices = indices.max()
    non_empty_index = (offsets[1:] - offsets[:-1]).nonzero().flatten()
    # disable dedup across different table
    indices = ((offsets[non_empty_index]) // batch_size) * (1 + max_indices) + indices
    grad = grad[0]
    _, idx, counts = torch.unique(
        indices, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(indices.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    mask = torch.zeros_like(grad, device=grad.device)
    original_index = non_empty_index[first_indicies]

    mask[original_index] = grad[original_index]
    return mask


def writeback_update_gradient_first_feature_only(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    grad: torch.Tensor,
    feature_table_map: list[int],
) -> torch.Tensor:
    """
    Special case of writeback_update_gradient where gradient only needs to be updated for the first feature. Other features will be forward-only

    NOTE: This function is not supporting VBE yet

    Args:
        indices (torch.Tensor): Embedding indices tensor
        offsets (torch.Tensor): Offsets tensor for batched embeddings
        grad (torch.Tensor): Gradient tensor to be updated
        feature_table_map (list[int]): Mapping from feature to table

    Returns:
        torch.Tensor: Updated gradient tensor with duplicates masked out
    """
    num_of_tables = len(feature_table_map)
    batch_size = (offsets.shape[0] - 1) // num_of_tables
    shrink_indices = indices[: offsets[batch_size]]
    if shrink_indices.numel() == 0 or indices.numel() == 0:
        return grad[0]
    assert num_of_tables * indices.max() < torch.iinfo(indices.dtype).max

    grad = grad[0]
    _, idx, counts = torch.unique(
        shrink_indices, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(shrink_indices.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    mask = torch.zeros_like(grad, device=grad.device)

    mask[first_indicies] = grad[first_indicies]
    return mask


def writeback_gradient(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    feature_table_map: list[int],
    writeback_first_feature_only: bool = False,
) -> tuple[torch.Tensor]:
    """
    Compute deduplicated gradient for writeback operation.

    Args:
        grad (torch.Tensor): Gradient tensor to be updated
        indices (torch.Tensor): Embedding indices tensor
        offsets (torch.Tensor): Offsets tensor for batched embeddings
        feature_table_map (list[int]): Mapping from feature to table
        writeback_first_feature_only (bool): If True, only first feature will apply gradient update, other features will be read-only

    Returns:
        tuple[torch.Tensor]: Tuple containing the updated gradient tensor
    """
    if writeback_first_feature_only:
        return (
            writeback_update_gradient_first_feature_only(
                indices, offsets, grad, feature_table_map
            ),
        )
    else:
        return (writeback_update_gradient(indices, offsets, grad, feature_table_map),)
