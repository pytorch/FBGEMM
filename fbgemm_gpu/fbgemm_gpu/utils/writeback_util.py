# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


def writeback_update_gradient(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    grad: tuple[torch.Tensor],
    feature_table_map: list[int],
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

    Returns:
        torch.Tensor: Updated gradient tensor with duplicates masked out
    """
    if indices.numel() == 0:
        return grad[0]
    # grad[0] has the same size as offsets for EBC.
    # get num of feature to estimate batch size
    num_of_tables = len(feature_table_map)
    assert num_of_tables * indices.max() < torch.iinfo(indices.dtype).max
    batch_size = offsets.shape[0] // num_of_tables
    max_indices = indices.max()
    non_empty_index = (offsets[1:] - offsets[:-1]).nonzero().flatten()
    # disable dedup across different table
    indices = ((offsets[non_empty_index]) // batch_size) * (1 + max_indices) + indices
    grad_tensor = grad[0]
    _, idx, counts = torch.unique(
        indices, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(indices.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    mask = torch.zeros_like(grad_tensor, device=grad_tensor.device)
    original_index = non_empty_index[first_indicies]

    mask[original_index] = grad_tensor[original_index]
    return mask


def writeback_update_gradient_first_feature_only(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    grad: tuple[torch.Tensor],
    feature_table_map: list[int],
) -> torch.Tensor:
    """
    Special case of writeback_update_gradient where gradient only needs to be updated for the first feature. Other features will be forward-only

    NOTE: This function is not supporting VBE yet

    Args:
        indices (torch.Tensor): Embedding indices tensor
        offsets (torch.Tensor): Offsets tensor for batched embeddings
        grad (tuple[torch.Tensor]): Gradient tensor to be updated
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

    grad_tensor = grad[0]
    _, idx, counts = torch.unique(
        shrink_indices, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(shrink_indices.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    mask = torch.zeros_like(grad_tensor, device=grad_tensor.device)

    mask[first_indices] = grad_tensor[first_indices]
    return mask


def writeback_update_gradient_ec(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    grad: tuple[torch.Tensor],
    feature_table_map: list[int],
) -> torch.Tensor:
    """
    Hook for using TBE EC to update gradient tensor by deduplicating indices across all features/tables.
    For duplicate indices, only the first occurrence receives the gradient to achieve the assign purpose via gradient update

    Args:
        indices (torch.Tensor): Embedding indices tensor
        offsets (torch.Tensor): Offsets tensor for batched embeddings
        grad (tuple[torch.Tensor]): Gradient tensor to be updated
        feature_table_map (list[int]): Mapping from feature to table

    Returns:
        torch.Tensor: Updated gradient tensor with duplicates masked out
    """
    if indices.numel() == 0:
        return grad[0]
    # grad[0] has the same size as indices for EC.
    num_of_tables: int = len(feature_table_map)
    assert num_of_tables * indices.max() < torch.iinfo(indices.dtype).max
    batch_size = offsets.shape[0] // num_of_tables
    max_indices = indices.max()
    non_empty_index = (offsets[1:] - offsets[:-1]).nonzero().flatten()
    # disable dedup across different table
    indices = ((offsets[non_empty_index]) // batch_size) * (1 + max_indices) + indices
    grad_tensor = grad[0]
    # TODO: revisit if dedup is needed when EC dedup is enabled.
    _, idx, counts = torch.unique(
        indices, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    _idx_sorted, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(indices.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    mask = torch.zeros_like(grad_tensor, device=grad_tensor.device)
    mask[first_indices] = grad_tensor[first_indices]
    return mask


def writeback_gradient(
    grad: tuple[torch.Tensor],
    indices: torch.Tensor,
    offsets: torch.Tensor,
    feature_table_map: list[int],
    writeback_first_feature_only: bool = False,
    no_bag: bool = False,
) -> tuple[torch.Tensor]:
    """
    Compute deduplicated gradient for writeback operation.

    Args:
        grad (tuple[torch.Tensor]): Gradient tensor to be updated
        indices (torch.Tensor): Embedding indices tensor
        offsets (torch.Tensor): Offsets tensor for batched embeddings
        feature_table_map (list[int]): Mapping from feature to table
        writeback_first_feature_only (bool): If True, only first feature will apply gradient update, other features will be read-only
        no_bag (bool): If True, we use TBE EC, otherwise we use TBE EBC.

    Returns:
        tuple[torch.Tensor]: Tuple containing the updated gradient tensor
    """
    if writeback_first_feature_only:
        return (
            writeback_update_gradient_first_feature_only(
                indices, offsets, grad, feature_table_map
            ),
        )
    elif no_bag:
        return (
            writeback_update_gradient_ec(indices, offsets, grad, feature_table_map),
        )
    else:
        return (writeback_update_gradient(indices, offsets, grad, feature_table_map),)
