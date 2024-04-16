# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, List, Optional, Tuple

import torch

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode

try:
    # pyre-ignore
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    if torch.version.hip:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_hip")
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_hip"
        )
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_hip"
        )
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine_hip")
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu/codegen:index_select_ops_hip"
        )
    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings"
        )
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops")
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine")

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_cpu"
    )
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:index_select_ops")

import torch.utils._pytree as pytree
from torch import SymInt, Tensor


if hasattr(torch.library, "impl_abstract"):
    impl_abstract = torch.library.impl_abstract
else:
    # pyre-ignore
    def impl_abstract(schema: str) -> Callable[[Callable], Callable]:
        # no-op
        # pyre-ignore
        def wrapper(f: Callable) -> Callable:
            return f

        return wrapper


@impl_abstract("fbgemm::permute_2D_sparse_data")
def permute_2D_sparse_data_meta(
    permute: Tensor,
    lengths: Tensor,
    values: Tensor,
    weights: Optional[Tensor] = None,
    permuted_lengths_sum: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    torch._check(
        lengths.dim() == 2, lambda: f"expected lengths.dim() == 2, got {lengths.dim()}"
    )
    T = permute.numel()
    B = lengths.size(1)
    indices = values
    permuted_lengths = lengths.new_empty([T, B])
    permuted_indices_size = 0
    if permuted_lengths_sum is not None:
        permuted_indices_size = permuted_lengths_sum
    else:
        ctx = torch._custom_op.impl.get_ctx()
        permuted_indices_size = ctx.new_dynamic_size()
    # pyre-fixme
    permuted_indices = indices.new_empty(permuted_indices_size)
    permuted_weights = None
    if weights is not None:
        # pyre-fixme
        permuted_weights = weights.new_empty(permuted_indices_size)
    return permuted_lengths, permuted_indices, permuted_weights


@impl_abstract("fbgemm::permute_1D_sparse_data")
def permute_1D_sparse_data_meta(
    permute: Tensor,
    lengths: Tensor,
    values: Tensor,
    weights: Optional[Tensor] = None,
    permuted_lengths_sum: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    indices = values
    permuted_lengths_size = permute.numel()
    permuted_lengths = lengths.new_empty([permuted_lengths_size])
    permuted_indices_size = 0
    if permuted_lengths_sum is not None:
        permuted_indices_size = permuted_lengths_sum
    else:
        ctx = torch._custom_op.impl.get_ctx()
        permuted_indices_size = ctx.new_dynamic_size()
    # pyre-fixme
    permuted_indices = indices.new_empty(permuted_indices_size)
    permuted_weights = None
    if weights is not None:
        # pyre-fixme
        permuted_weights = weights.new_empty(permuted_indices_size)
    return permuted_lengths, permuted_indices, permuted_weights


@impl_abstract("fbgemm::masked_select_jagged_1d")
def masked_select_jagged_1d(
    values: Tensor, lengths: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    torch._check(values.dim() == 1)
    torch._check(lengths.dim() == 1)
    torch._check(values.device == lengths.device)
    torch._check(values.device == mask.device)

    s0 = torch.library.get_ctx().new_dynamic_size()
    masked_values = values.new_empty([s0])
    masked_lengths = torch.empty_like(lengths)
    return masked_values, masked_lengths


@impl_abstract("fbgemm::tbe_input_combine")
def tbe_input_combine_abstract(
    indices_list: List[Tensor],
    offsets_list: List[Tensor],
    per_sample_weights: List[Tensor],
    include_last_offsets: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    torch._check(len(indices_list) > 0)
    torch._check(len(indices_list) == len(offsets_list))
    torch._check(len(indices_list) == len(per_sample_weights))
    torch._check(len(indices_list) == include_last_offsets.numel())
    total_indices = 0
    need_weight = False
    for index, offset, weight in zip(indices_list, offsets_list, per_sample_weights):
        torch._check(index.dtype == torch.int or index.dtype == torch.long)
        torch._check(offset.dtype == torch.int or offset.dtype == torch.long)
        torch._check(index.dim() == 1)
        torch._check(offset.dim() == 1)
        torch._check(index.is_contiguous())
        torch._check(offset.is_contiguous())
        total_indices = total_indices + index.numel()
        if weight.numel() > 0:
            torch._check(weight.dim() == 1)
            torch._check(weight.numel() == index.numel())
            torch._check(weight.is_contiguous())
            need_weight = True
    total_offsets = torch.library.get_ctx().new_dynamic_size()
    combined_indices = indices_list[0].new_empty([total_indices], dtype=torch.int)
    combined_offsets = offsets_list[0].new_empty([total_offsets], dtype=torch.int)
    if need_weight:
        combined_weights = per_sample_weights[0].new_empty(
            [total_indices], dtype=torch.float
        )
    else:
        combined_weights = torch.empty(0)
    return combined_indices, combined_offsets, combined_weights


@impl_abstract("fbgemm::tbe_input_combine_with_length")
def tbe_input_combine_with_length_abstract(
    indices_list: List[Tensor],
    offsets_list: List[Tensor],
    per_sample_weights: List[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    torch._check(len(indices_list) > 0)
    torch._check(len(indices_list) == len(offsets_list))
    torch._check(len(indices_list) == len(per_sample_weights))
    total_indices = 0
    total_offsets = 0
    need_weight = False
    for index, offset, weight in zip(indices_list, offsets_list, per_sample_weights):
        torch._check(index.dtype == torch.int or index.dtype == torch.long)
        torch._check(offset.dtype == torch.int or offset.dtype == torch.long)
        torch._check(index.dim() == 1)
        torch._check(offset.dim() == 1)
        torch._check(index.is_contiguous())
        torch._check(offset.is_contiguous())
        total_indices = total_indices + index.numel()
        total_offsets = total_offsets + offset.numel()
        if weight.numel() > 0:
            torch._check(weight.dim() == 1)
            torch._check(weight.numel() == index.numel())
            torch._check(weight.is_contiguous())
            need_weight = True
    combined_indices = indices_list[0].new_empty([total_indices], dtype=torch.int)
    combined_offsets = offsets_list[0].new_empty([total_offsets], dtype=torch.int)
    if need_weight:
        combined_weights = per_sample_weights[0].new_empty(
            [total_indices], dtype=torch.float
        )
    else:
        combined_weights = torch.empty(0, device=indices_list[0].device)
    return combined_indices, combined_offsets, combined_weights


@impl_abstract("fbgemm::jagged_index_select_2d_forward_v2")
def jagged_index_select_2d_forward_v2_abstract(
    values: Tensor,
    indices: Tensor,
    input_offsets: Tensor,
    output_offsets: Tensor,
    num_dense_output_rows: Optional[int] = None,
) -> Tensor:
    torch._check(values.device == indices.device)
    torch._check(values.device == input_offsets.device)
    torch._check(values.device == output_offsets.device)
    torch._check(values.dim() == 2)
    dynamic_num_dense_output_rows = torch.library.get_ctx().new_dynamic_size()
    num_cols = values.size(1)
    return values.new_empty([dynamic_num_dense_output_rows, num_cols])


@impl_abstract("fbgemm::jagged_index_add_2d_forward_v2")
def jagged_index_add_2d_forward_v2_abstract(
    values: Tensor,
    indices: Tensor,
    input_offsets: Tensor,
    output_offsets: Tensor,
    num_output_rows: int,
    num_dense_input_rows: Optional[int] = None,
) -> Tensor:
    torch._check(values.device == indices.device)
    torch._check(values.device == input_offsets.device)
    torch._check(values.device == output_offsets.device)
    torch._check(values.dim() == 2)
    num_cols = values.size(1)
    return values.new_empty([num_output_rows, num_cols])


@impl_abstract("fbgemm::expand_into_jagged_permute")
def expand_into_jagged_permute_meta(
    permute: Tensor,
    input_offsets: Tensor,
    output_offsets: Tensor,
    output_size: Tuple[int, ...],
) -> Tensor:
    torch._check(permute.numel() > 0, lambda: "expected {permute.numel} > 0")
    torch._check(
        permute.numel() == input_offsets.numel() - 1,
        lambda: f"expected {permute.numel()} == {input_offsets.numel()} - 1",
    )
    torch._check(
        permute.numel() == output_offsets.numel() - 1,
        lambda: f"expected {permute.numel()} == {output_offsets.numel()} - 1",
    )
    output_permute = input_offsets.new_empty(output_size)
    return output_permute


def check_all_same_device(*tensors: Optional[Tensor]) -> None:
    # pyre-ignore[9]
    tensors, _ = pytree.tree_flatten(tensors)
    if len(tensors) == 0:
        return
    first_tensor: Optional[Tensor] = None
    for tensor in tensors:
        if tensor is None:
            continue
        if first_tensor is None:
            first_tensor = tensor
        if first_tensor.device.type == "cpu" and tensor.device.type == "cpu":
            return
        torch._check(tensor.device == first_tensor.device)


@impl_abstract("fbgemm::pruned_array_lookup")
def pruned_array_lookup_meta(
    indices: Tensor,
    offsets: Tensor,
    index_remappings: Tensor,
    index_remappings_offsets: Tensor,
) -> Tensor:
    check_all_same_device(indices, offsets, index_remappings, index_remappings_offsets)
    return indices.new_empty(indices.shape)


@impl_abstract("fbgemm::int_nbit_split_embedding_codegen_lookup_function")
def int_nbit_split_embedding_codegen_lookup_function_meta(
    dev_weights: torch.Tensor,
    uvm_weights: torch.Tensor,
    weights_placements: torch.Tensor,
    weights_offsets: torch.Tensor,
    weights_tys: torch.Tensor,
    D_offsets: torch.Tensor,
    total_D: int,
    max_int2_D: int,
    max_int4_D: int,
    max_int8_D: int,
    max_float16_D: int,
    max_float32_D: int,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: int,
    indice_weights: Optional[torch.Tensor] = None,
    output_dtype_int: int = 1,
    lxu_cache_weights: Optional[torch.Tensor] = None,
    lxu_cache_locations: Optional[torch.Tensor] = None,
    row_alignment: Optional[int] = None,
    max_float8_D: Optional[int] = None,
    fp8_exponent_bits: Optional[int] = None,
    fp8_exponent_bias: Optional[int] = None,
) -> Tensor:
    check_all_same_device(
        dev_weights,
        uvm_weights,
        weights_placements,
        weights_offsets,
        weights_tys,
        D_offsets,
        indices,
        offsets,
        indice_weights,
    )
    output_dtype = SparseType.from_int(output_dtype_int).as_dtype()
    kINT8QparamsBytes = 8

    if pooling_mode == PoolingMode.NONE:
        D = max(
            [
                max_int2_D,
                max_int4_D,
                max_int8_D,
                max_float16_D,
                max_float32_D,
                max_float8_D if max_float8_D is not None else 0,
            ]
        )
        total_L = indices.numel()
        T = weights_offsets.numel()
        torch._check(D > 0)
        adjusted_D = D
        if SparseType.from_int(output_dtype_int) == SparseType.INT8:
            adjusted_D += T * kINT8QparamsBytes
        output = dev_weights.new_empty([total_L, adjusted_D], dtype=output_dtype)
        return output

    T = D_offsets.numel() - 1
    torch._check(T > 0)
    torch._check(total_D > 0)
    B = (offsets.size(0) - 1) // T
    total_adjusted_D = total_D
    if SparseType.from_int(output_dtype_int) == SparseType.INT8:
        total_adjusted_D += T * kINT8QparamsBytes
    output = dev_weights.new_empty([B, total_adjusted_D], dtype=output_dtype)
    return output


@impl_abstract("fbgemm::block_bucketize_sparse_features")
def block_bucketize_sparse_features_meta(
    lengths: torch.Tensor,
    indices: torch.Tensor,
    bucketize_pos: bool,
    sequence: bool,
    block_sizes: torch.Tensor,
    my_size: int,
    weights: Optional[torch.Tensor] = None,
    batch_size_per_feature: Optional[torch.Tensor] = None,
    max_B: int = -1,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    # Output: lengths, indices, weights", pos?, unbucketize_permute?
    num_buckets = my_size
    num_features = lengths.size(0)
    num_values = indices.size(0)
    return (
        lengths.new_empty([num_buckets * num_features]),
        indices.new_empty([num_values]),
        weights.new_empty(weights.shape) if weights is not None else None,
        indices.new_empty([num_values]) if bucketize_pos else None,
        indices.new_empty([num_values]),
    )


@impl_abstract("fbgemm::merge_pooled_embeddings")
def merge_pooled_embeddings(
    pooled_embeddings: List[torch.Tensor],
    uncat_dim_size: int,
    target_device: torch.device,
    cat_dim: int = 1,
) -> torch.Tensor:
    if len(pooled_embeddings) == 0:
        return torch.empty([], device=target_device)
    torch._check_is_size(cat_dim)
    torch._check(cat_dim >= 0)
    torch._check(cat_dim <= 1)
    total_cat_dim_size = 0
    for e in pooled_embeddings:
        torch._check(e.dim() == 2)
        torch._check(e.size(1 - cat_dim) == uncat_dim_size)
        total_cat_dim_size += e.size(cat_dim)
    torch._check_is_size(total_cat_dim_size)
    e = pooled_embeddings[0]
    if cat_dim == 0:
        return e.new_empty(
            [total_cat_dim_size, e.size(1)],
            device=target_device,
        )

    return e.new_empty(
        [e.size(0), total_cat_dim_size],
        device=target_device,
    )


@impl_abstract("fbgemm::permute_sparse_features")
def permute_sparse_features_abstract(
    permute: Tensor, lengths: Tensor, indices: Tensor, weights: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    torch._check(lengths.dtype == indices.dtype)
    torch._check(permute.device == lengths.device)
    torch._check(permute.device == indices.device)
    if weights is not None:
        torch._check(permute.device == weights.device)
    num_output_features = permute.numel()
    B = lengths.size(1)
    permuted_lengths = lengths.new_empty(num_output_features, B)
    output_size = torch.library.get_ctx().new_dynamic_size()
    # pyre-fixme[6]: In call `torch._C.TensorBase.new_empty`, for 1st positional argument,
    # expected `Sequence[Union[int, types.SymInt]]` but got `Union[int, torch.SymInt]`
    permuted_indices = indices.new_empty(output_size)
    permuted_weights = None
    if weights is not None:
        # pyre-fixme[6]: In call `torch._C.TensorBase.new_empty`, for 1st positional argument,
        # expected `Sequence[Union[int, types.SymInt]]` but got `Union[int, torch.SymInt]`
        permuted_weights = weights.new_empty(output_size)
    return (permuted_lengths, permuted_indices, permuted_weights)


@impl_abstract("fbgemm::segment_sum_csr")
def segment_sum_csr_abstract(
    batch_size: int, csr_seg: Tensor, values: Tensor
) -> Tensor:
    output_size = csr_seg.numel() - 1
    output = values.new_empty(output_size)
    return output


@impl_abstract("fbgemm::dense_to_jagged_forward")
def dense_to_jagged_forward(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[torch.SymInt] = None,
) -> torch.Tensor:
    if not total_L:
        total_L = torch.library.get_ctx().new_dynamic_size()
    return dense.new_zeros(
        [total_L, dense.size()[-1]],
        dtype=dense.dtype,
        device=dense.device,
        layout=dense.layout,
    )


@impl_abstract("fbgemm::dense_to_jagged")
def dense_to_jagged(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[torch.SymInt] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if not total_L:
        total_L = torch.library.get_ctx().new_dynamic_size()
    return (dense_to_jagged_forward(dense, offsets, total_L), offsets)


@impl_abstract("fbgemm::batch_index_select_dim0")
def batch_index_select_dim0_abstract(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    input_num_indices: List[int],
    input_rows: List[int],
    input_columns: List[int],
    permute_output_dim_0_1: bool,
) -> torch.Tensor:
    """
    This meta function is used to calculate the shape of output tensor
    from the original function `fbgemm::batch_index_select_dim0` without the actual data.
    """
    # input lists must have the same length
    torch._check(len(input_num_indices) == len(input_rows))
    torch._check(len(input_num_indices) == len(input_columns))

    if permute_output_dim_0_1 and len(input_num_indices) > 0:
        # All num_indices must be the same if permute_output_dim_0_1 is True
        for x in input_num_indices:
            torch._check(x == input_num_indices[0])

    size = sum([row * col for row, col in zip(input_rows, input_columns)])
    torch._check(inputs.size(0) == size)

    output_numel = 0
    for i, cols in enumerate(input_columns):
        output_numel += input_num_indices[i] * cols
    return inputs.new_empty([output_numel])


@impl_abstract("fbgemm::keyed_jagged_index_select_dim1")
def keyed_jagged_index_select_dim1_abstract(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
    batch_size: torch.SymInt,
    weights: Optional[torch.Tensor] = None,
    selected_lengths_sum: Optional[torch.SymInt] = None,
) -> List[torch.Tensor]:
    """
    This meta function is used to calculate the shape of output tensors
    from the original function `fbgemm::keyed_jagged_index_select_dim1` without the actual data.
    """
    # pyre-ignore
    num_batches = len(lengths) // batch_size
    # offsets = [0] + lengths.cumsum(0)
    torch._check(len(lengths) + 1 == len(offsets))
    # len(lengths) == batch_size * num_batches
    # pyre-ignore
    torch._check(len(lengths) % batch_size == 0)
    if weights is not None:
        # weights must have the same shape as values
        torch._check(values.shape == weights.shape)

    if selected_lengths_sum is None:
        length_indices = torch.cat(
            # pyre-ignore
            [indices + i * batch_size for i in range(num_batches)]
        )
        selected_lengths_sum = (
            torch.index_select(lengths, 0, length_indices).sum().item()
        )

    ret: List[torch.Tensor] = [
        # pyre-ignore
        values.new_empty([selected_lengths_sum]),
        lengths.new_empty([indices.shape[0] * num_batches]),
    ]

    if weights is not None:
        # pyre-ignore
        ret.append(weights.new_empty([selected_lengths_sum]))

    return ret


@impl_abstract("fbgemm::bounds_check_indices")
def bounds_check_indices_abstract(
    rows_per_table: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    bounds_check_mode_int: int,
    bounds_check_warning: torch.Tensor,
    per_sample_weights: Optional[torch.Tensor] = None,
    B_offsets: Optional[torch.Tensor] = None,
    max_B: Optional[SymInt] = None,
) -> None:
    """
    This meta function is used to fake the bounds checking
    from the original function `fbgemm::bounds_check_indices`
    """
    return
