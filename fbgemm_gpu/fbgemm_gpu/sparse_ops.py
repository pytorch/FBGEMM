# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple

import torch

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode

try:
    # pyre-ignore
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings")
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_cpu"
    )
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine_cpu")

from torch import Tensor


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


@impl_abstract("fbgemm::jagged_index_select_2d_forward_v2")
def jagged_index_select_2d_forward_v2_abstract(
    values: Tensor,
    indices: Tensor,
    input_offsets: Tensor,
    output_offsets: Tensor,
) -> Tensor:
    torch._check(values.device == indices.device)
    torch._check(values.device == input_offsets.device)
    torch._check(values.device == output_offsets.device)
    torch._check(values.dim() == 2)
    num_dense_output_rows = torch.library.get_ctx().new_dynamic_size()
    num_cols = values.size(1)
    return values.new_empty([num_dense_output_rows, num_cols])


@impl_abstract("fbgemm::jagged_index_add_2d_forward_v2")
def jagged_index_add_2d_forward_v2_abstract(
    values: Tensor,
    indices: Tensor,
    input_offsets: Tensor,
    output_offsets: Tensor,
    num_output_rows: int,
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
    output_dtype_int: Optional[int] = None,
    lxu_cache_weights: Optional[torch.Tensor] = None,
    lxu_cache_locations: Optional[torch.Tensor] = None,
    row_alignment: Optional[int] = None,
    max_float8_D: Optional[int] = None,
    fp8_exponent_bits: Optional[int] = None,
    fp8_exponent_bias: Optional[int] = None,
) -> torch.Tensor:
    T = D_offsets.numel() - 1
    B = (offsets.size(0) - 1) // T
    output_dtype = torch.float32
    torch._check(
        output_dtype_int in (0, 1, 5),
        lambda: f"expected output_dtype to be fp32, fp16 or bf16, got {indices.dtype}",
    )
    if output_dtype_int == SparseType.FP32.value:
        output_dtype = torch.float32
    elif output_dtype_int == SparseType.FP16.value:
        output_dtype = torch.float16
    elif output_dtype_int == SparseType.BF16.value:
        output_dtype = torch.bfloat16

    if pooling_mode == PoolingMode.NONE:
        # pyre-ignore
        offsets_last: int = offsets[-1].item()
        total_D_T: int = total_D // T
        torch._check_is_size(offsets[-1].item())
        torch._check_is_size(total_D_T)
        torch._check_is_size(B)
        return dev_weights.new_empty(
            [offsets_last, total_D_T],
            dtype=output_dtype,
            device=torch.device("meta"),
        )
    torch._check_is_size(B)
    torch._check_is_size(total_D)
    return dev_weights.new_empty(
        (B, total_D),
        dtype=output_dtype,
        device=torch.device("meta"),
    )


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


@impl_abstract("fbgemm::bounds_check_indices")
def bounds_check_indices(
    rows_per_table: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    bounds_check_mode: int,
    warning: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    B_offsets: Optional[torch.Tensor] = None,
    max_B: int = -1,
) -> None:
    pass


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
