# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from fbgemm_gpu.utils.loader import load_torch_module

try:
    # pyre-ignore
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    load_torch_module("//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings")
    load_torch_module("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")

    if torch.version.hip:
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_hip"
        )

    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops")

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine")

    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_cpu"
    )
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:index_select_ops")
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_split_cpu"
    )


import torch.utils._pytree as pytree
from torch import SymInt, Tensor
from torch.fx.experimental.symbolic_shapes import guard_size_oblivious


if hasattr(torch.library, "register_fake"):
    # pyre-ignore[9]
    impl_abstract = torch.library.register_fake
elif hasattr(torch.library, "impl_abstract"):
    impl_abstract = torch.library.impl_abstract
else:
    # pyre-ignore
    def impl_abstract(schema: str) -> Callable[[Callable], Callable]:
        # no-op
        # pyre-ignore
        def wrapper(f: Callable) -> Callable:
            return f

        return wrapper


def permute_2D_sparse_data_input1D_meta(
    permute: Tensor,
    lengths: Tensor,
    values: Tensor,
    stride: int,
    weights: Optional[Tensor] = None,
    permuted_lengths_sum: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    torch._check(
        lengths.dim() == 1, lambda: f"expected lengths.dim() == 1, got {lengths.dim()}"
    )
    T = permute.numel()
    B = stride
    indices = values
    permuted_lengths = lengths.new_empty([T * B])
    permuted_indices_size = 0
    if permuted_lengths_sum is not None:
        permuted_indices_size = permuted_lengths_sum
    else:
        ctx = torch.library.get_ctx()
        permuted_indices_size = ctx.new_dynamic_size()
    # pyre-fixme
    permuted_indices = indices.new_empty(permuted_indices_size)
    permuted_weights = None
    if weights is not None:
        # pyre-fixme
        permuted_weights = weights.new_empty(permuted_indices_size)
    return permuted_lengths, permuted_indices, permuted_weights


# pyre-ignore
def permute_2D_sparse_data_input1D_setup_context(ctx, inputs, output):
    permute, lengths, values, stride, weights, permuted_lengths_sum = inputs
    permuted_lengths, permuted_values, permuted_weights = output
    ctx.permute = permute
    ctx.permuted_lengths = permuted_lengths
    ctx.stride = stride


def permute_2D_sparse_data_input1D_backward(
    ctx,  # pyre-ignore
    grad_lengths: torch.Tensor,
    grad_values: torch.Tensor,
    grad_weights: torch.Tensor,
) -> Tuple[None, Tensor, Tensor, None, Tensor, None]:
    inv_permute = torch.ops.fbgemm.invert_permute(ctx.permute)
    permuted_grad_lengths, permuted_grad_values, permuted_grad_weights = (
        torch.ops.fbgemm.permute_2D_sparse_data_input1D(
            inv_permute,
            ctx.permuted_lengths,
            grad_values,
            ctx.stride,
            grad_weights,
            None,
        )
    )
    return (
        None,
        permuted_grad_lengths,
        permuted_grad_values,
        None,
        permuted_grad_weights,
        None,
    )


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
        ctx = torch.library.get_ctx()
        permuted_indices_size = ctx.new_dynamic_size()
    # pyre-fixme
    permuted_indices = indices.new_empty(permuted_indices_size)
    permuted_weights = None
    if weights is not None:
        # pyre-fixme
        permuted_weights = weights.new_empty(permuted_indices_size)
    return permuted_lengths, permuted_indices, permuted_weights


def invert_permute_abstract(permute: Tensor) -> Tensor:
    return torch.empty_like(permute)


# pyre-ignore
def permute_2D_sparse_data_setup_context(ctx, inputs, output):
    permute, lengths, values, weights, permuted_lengths_sum = inputs
    permuted_lengths, permuted_values, permuted_weights = output
    ctx.permute = permute
    ctx.permuted_lengths = permuted_lengths


# pyre-ignore
def permute_2D_sparse_data_backward(ctx, grad_lengths, grad_values, grad_weights):
    inv_permute = torch.ops.fbgemm.invert_permute(ctx.permute)
    permuted_grad_lengths, permuted_grad_values, permuted_grad_weights = (
        torch.ops.fbgemm.permute_2D_sparse_data(
            inv_permute, ctx.permuted_lengths, grad_values, grad_weights
        )
    )
    return (
        None,
        permuted_grad_lengths,
        permuted_grad_values,
        permuted_grad_weights,
        None,
    )


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
        ctx = torch.library.get_ctx()
        permuted_indices_size = ctx.new_dynamic_size()
    # pyre-fixme
    permuted_indices = indices.new_empty(permuted_indices_size)
    permuted_weights = None
    if weights is not None:
        # pyre-fixme
        permuted_weights = weights.new_empty(permuted_indices_size)
    return permuted_lengths, permuted_indices, permuted_weights


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
        if guard_size_oblivious(weight.numel() > 0):
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
        if guard_size_oblivious(weight.numel() > 0):
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
    if all(t.device.type in ["cpu", "meta"] for t in tensors if t is not None):
        return
    first_tensor: Optional[Tensor] = None
    for tensor in tensors:
        if tensor is None:
            continue
        if first_tensor is None:
            first_tensor = tensor
        torch._check(tensor.device == first_tensor.device)


def pruned_array_lookup_meta(
    indices: Tensor,
    offsets: Tensor,
    index_remappings: Tensor,
    index_remappings_offsets: Tensor,
) -> Tensor:
    check_all_same_device(indices, offsets, index_remappings, index_remappings_offsets)
    return indices.new_empty(indices.shape)


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
    block_bucketize_pos: Optional[torch.Tensor] = None,
    keep_orig_idx: bool = False,
    total_num_blocks: Optional[torch.Tensor] = None,
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


def segment_sum_csr_abstract(
    batch_size: int, csr_seg: Tensor, values: Tensor
) -> Tensor:
    output_size = csr_seg.numel() - 1
    output = values.new_empty(output_size)
    return output


def dense_to_jagged_forward(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[torch.SymInt] = None,
) -> torch.Tensor:
    if total_L is None:
        total_L = torch.library.get_ctx().new_dynamic_size()
    return dense.new_zeros(
        [total_L, dense.size()[-1]],
        dtype=dense.dtype,
        device=dense.device,
        layout=dense.layout,
    )


def dense_to_jagged(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[torch.SymInt] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if total_L is None:
        total_L = torch.library.get_ctx().new_dynamic_size()
    return (dense_to_jagged_forward(dense, offsets, total_L), offsets)


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


def batch_index_select_dim0_tensor_abstract(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    input_num_indices: torch.Tensor,
    input_rows: torch.Tensor,
    input_columns: torch.Tensor,
    permute_output_dim_0_1: bool,
) -> torch.Tensor:
    torch._check(input_num_indices.size(0) == input_rows.size(0))
    torch._check(input_num_indices.size(0) == input_columns.size(0))
    output_numel = torch.library.get_ctx().new_dynamic_size()
    return inputs.new_empty([output_numel])


def batch_index_select_dim0_forward_cuda_impl_abstract(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    input_num_indices: List[int],
    input_rows: List[int],
    input_columns: List[int],
    permute_output_dim_0_1: bool,
) -> List[torch.Tensor]:
    num_inputs = len(input_rows)
    torch._check(len(input_num_indices) == len(input_rows))
    torch._check(len(input_num_indices) == len(input_columns))

    output_numel = 0
    for i, cols in enumerate(input_columns):
        output_numel += input_num_indices[i] * cols

    output_offsets = (
        inputs.new_empty([0], dtype=torch.int64)
        if permute_output_dim_0_1
        else inputs.new_empty([num_inputs + 1], dtype=torch.int64)
    )

    if permute_output_dim_0_1:
        for i in range(num_inputs):
            torch._check(input_num_indices[0] == input_num_indices[i])

    return [
        inputs.new_empty([output_numel]),
        inputs.new_empty([num_inputs], dtype=torch.int64),
        inputs.new_empty([num_inputs + 1], dtype=torch.int64),
        inputs.new_empty([num_inputs + 1], dtype=torch.int32),  # D_offsets
        output_offsets,
        inputs.new_empty([num_inputs + 1], dtype=torch.int64),
        inputs.new_empty([4], dtype=torch.int64, device="cpu"),
    ]


def batch_index_select_dim0_tensor_forward_cuda_impl_abstract(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    input_num_indices: torch.Tensor,
    input_rows: torch.Tensor,
    input_columns: torch.Tensor,
    permute_output_dim_0_1: bool,
) -> List[torch.Tensor]:
    num_inputs: int = input_rows.size(0)
    torch._check(input_num_indices.size(0) == input_rows.size(0))
    torch._check(input_num_indices.size(0) == input_columns.size(0))
    output_numel = torch.library.get_ctx().new_dynamic_size()
    if permute_output_dim_0_1:
        output_offsets = inputs.new_empty([0], dtype=torch.int64)
    else:
        output_offsets = inputs.new_empty([num_inputs + 1], dtype=torch.int64)

    return [
        inputs.new_empty([output_numel]),
        inputs.new_empty([num_inputs], dtype=torch.int64),
        inputs.new_empty([num_inputs + 1], dtype=torch.int64),
        inputs.new_empty([num_inputs + 1], dtype=torch.int32),  # D_offsets
        output_offsets,
        inputs.new_empty([num_inputs + 1], dtype=torch.int64),  # total_L_offsets
        inputs.new_empty([4], dtype=torch.int64, device="cpu"),
    ]


def batch_index_select_dim0_tensor_backward_cuda_impl_abstract(
    grad_output: torch.Tensor,
    dev_weights: torch.Tensor,
    weights_offsets: torch.Tensor,
    D_offsets: torch.Tensor,
    hash_size_cumsum: torch.Tensor,
    indices: torch.Tensor,
    max_segment_length_per_warp: int,
    grad_offsets: torch.Tensor,
    total_L_offsets: torch.Tensor,
    permute_output_dim_0_1: bool,
    saved_tensor: torch.Tensor,
) -> torch.Tensor:
    return grad_output.new_empty(dev_weights.shape)


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


def batch_index_select_dim0_backward_cuda_impl_abstract(
    grad_output: torch.Tensor,
    dev_weights: torch.Tensor,
    weights_offsets: torch.Tensor,
    D_offsets: torch.Tensor,
    hash_size_cumsum: torch.Tensor,
    indices: torch.Tensor,
    max_segment_length_per_warp: int,
    grad_offsets: torch.Tensor,
    total_L_offsets: torch.Tensor,
    permute_output_dim_0_1: bool,
    saved_tensor: torch.Tensor,
) -> torch.Tensor:
    return grad_output.new_empty(dev_weights.shape)


def batch_index_select_dim0_forward_cpu_impl_abstract(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    input_num_indices: List[int],
    input_rows: List[int],
    input_columns: List[int],
    permute_output_dim_0_1: bool,
) -> List[torch.Tensor]:
    # input lists must have the same length
    num_inputs = len(input_num_indices)
    torch._check(num_inputs == len(input_rows))
    torch._check(num_inputs == len(input_columns))

    if permute_output_dim_0_1 and guard_size_oblivious(len(input_num_indices) > 0):
        # All num_indices must be the same if permute_output_dim_0_1 is True
        for x in input_num_indices:
            torch._check(x == input_num_indices[0])

    output_numel: int = sum([i * c for i, c in zip(input_num_indices, input_columns)])

    return [
        inputs.new_empty([output_numel]),
        inputs.new_empty([len(input_num_indices)], dtype=torch.int64),
        inputs.new_empty([len(input_rows)], dtype=torch.int64),
        inputs.new_empty([len(input_columns)], dtype=torch.int64),
        inputs.new_empty([num_inputs], dtype=torch.int64),  # indices_numels
        inputs.new_empty([1], dtype=torch.int64),  # saved_tensor
    ]


def batch_index_select_dim0_tensor_forward_cpu_impl_abstract(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    input_num_indices: torch.Tensor,
    input_rows: torch.Tensor,
    input_columns: torch.Tensor,
    permute_output_dim_0_1: bool,
) -> List[torch.Tensor]:
    # input lists must have the same length
    num_inputs = len(input_num_indices)
    torch._check(num_inputs == len(input_rows))
    torch._check(num_inputs == len(input_columns))

    output_numel = torch.library.get_ctx().new_dynamic_size()

    return [
        inputs.new_empty([output_numel]),
        inputs.new_empty([1], dtype=torch.int64),
    ]


def batch_index_select_dim0_backward_cpu_impl_abstract(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    indices_numels: torch.Tensor,
    input_num_indices: torch.Tensor,
    input_rows: torch.Tensor,
    input_columns: torch.Tensor,
    permute_output_dim_0_1: bool,
    saved_tensor: torch.Tensor,
) -> torch.Tensor:
    return grad_output.new_empty([torch.library.get_ctx().new_dynamic_size()])


def bounds_check_indices_abstract(
    rows_per_table: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    bounds_check_mode_int: int,
    bounds_check_warning: torch.Tensor,
    per_sample_weights: Optional[torch.Tensor] = None,
    B_offsets: Optional[torch.Tensor] = None,
    max_B: Optional[SymInt] = None,
    b_t_map: Optional[torch.Tensor] = None,
    info_B_num_bits: int = -1,
    info_B_mask: int = -1,
    bounds_check_version: int = 1,
) -> None:
    """
    This meta function is used to fake the bounds checking
    from the original function `fbgemm::bounds_check_indices`
    """
    return


def group_index_select_dim0_gpu_impl_abstract(
    inputs: List[torch.Tensor], group_size: int
) -> List[torch.Tensor]:
    """
    Calculate output shapes for group_index_select_dim0_gpu_impl
    without the actual data.
    """
    indices_group = inputs[:group_size]
    input_group = inputs[group_size:]
    torch._check(len(input_group) == group_size)

    ret = []
    for i in range(group_size):
        size = list(input_group[i].size())
        ret.append(input_group[i].new_empty([indices_group[i].size(0)] + size[1:]))

    # divide by 2 since sizeof(int64_t) / sizeof(int32_t) = 2
    args_tensor_numel = 4 * group_size + 1 + int(math.ceil(group_size / 2))

    ret.append(
        # sizeof(int64_t) = 8, torch.uint8 = at::kByte
        input_group[0].new_empty(
            args_tensor_numel * 8, dtype=torch.uint8, pin_memory=True
        )
    )

    ret.append(torch.zeros(5, dtype=torch.int64, device="cpu"))

    return ret


def group_index_select_dim0_gpu_backward_abstract(
    all_inputs: List[torch.Tensor], output_shape_group_ref: List[torch.SymInt]
) -> List[torch.Tensor]:
    """
    Calculate output shapes for group_index_select_dim0_gpu_backward
    without the actual data.
    """
    torch._check(len(all_inputs) > 3)
    group_size = (len(all_inputs) - 3) // 2
    ret = []

    # indices
    for _ in range(group_size):
        ret.append(all_inputs[0].new_empty(0))

    # inputs
    output_dim = len(output_shape_group_ref) // group_size
    for i in range(group_size):
        ret.append(
            all_inputs[0].new_empty(
                output_shape_group_ref[i * output_dim : (i + 1) * output_dim]
            )
        )

    return ret


def keyed_jagged_index_select_dim1_forward_cuda_impl_abstract(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
    batch_size: torch.SymInt,
    weights: Optional[torch.Tensor] = None,
    selected_lengths_sum: Optional[torch.SymInt] = None,
) -> List[torch.Tensor]:
    num_batches = lengths.size(0) // batch_size
    torch._check(lengths.size(0) + 1 == offsets.size(0))
    # pyre-ignore
    torch._check(lengths.size(0) % batch_size == 0)

    if weights is not None:
        # weights must have the same shape as values
        torch._check(values.shape == weights.shape)

    if selected_lengths_sum is None:
        selected_lengths_sum = torch.library.get_ctx().new_dynamic_size()

    torch._check_is_size(selected_lengths_sum)
    vlw: List[torch.Tensor] = [
        values.new_empty([selected_lengths_sum]),  # output
        lengths.new_empty([indices.shape[0] * num_batches]),  # output_lengths
    ]
    if weights is not None:
        vlw.append(weights.new_empty([selected_lengths_sum]))  # output_weights

    return [
        *vlw,
        offsets.new_empty([indices.shape[0] * num_batches]),  # output_offsets
        torch.empty([4], dtype=torch.int64, device="cpu"),  # saved_data_tensor
    ]


def keyed_jagged_index_select_dim1_backward_cuda_impl_abstract(
    grad: torch.Tensor,
    indices: torch.Tensor,
    grad_offsets: torch.Tensor,
    output_offsets: torch.Tensor,
    saved_tensor: torch.Tensor,
) -> torch.Tensor:
    return grad.new_empty([torch.library.get_ctx().new_dynamic_size()])


def permute_pooled_embs_split_abstract(
    pooled_embs: Tensor,
    offset_dim_list: Tensor,
    permute_list: Tensor,
    inv_offset_dim_list: Tensor,
    inv_permute_list: Tensor,
) -> Tensor:
    return torch.empty_like(pooled_embs)


def histogram_binning_calibration_abstract(
    logit: Tensor,
    bin_num_examples: Tensor,
    bin_num_positives: Tensor,
    positive_weight: float,
    lower_bound: float,
    upper_bound: float,
    bin_ctr_in_use_after: int,
    bin_ctr_weight_value: float,
) -> Tuple[Tensor, Tensor]:
    return torch.empty_like(logit), torch.empty([logit.numel()], dtype=torch.int64)


def float_to_hfp8_quantized(
    input: Tensor, ebits: int, exponent_bias: int, max_pos: float
) -> Tensor:
    return torch.empty_like(input, dtype=torch.uint8)


def hfp8_quantized_to_float(input: Tensor, ebits: int, exponent_bias: int) -> Tensor:
    return torch.empty_like(input, dtype=torch.float32)


def float_or_half_to_fused_nbit_rowwise_quantized_sbhalf(
    input_t: Tensor,
    bit_rate: int,
) -> Tensor:
    input_sizes = input_t.size()
    torch._check(len(input_sizes) == 2)
    nrows = input_sizes[0]
    ncols = input_sizes[1]
    num_elem_per_byte = 8 // bit_rate

    torch._check(ncols % (2 * num_elem_per_byte) == 0)
    output_columns = (ncols + num_elem_per_byte - 1) // num_elem_per_byte + 2 * 2
    output = torch.empty(
        (nrows, output_columns), device=input_t.device, dtype=torch.uint8
    )
    return output


def fused_nbit_rowwise_quantized_sb_half_to_float_or_half(
    input_t: Tensor,
    bit_rate: int,
    output_dtype: int = 0,
) -> Tensor:
    torch._check(output_dtype in [SparseType.FP32.as_int(), SparseType.FP16.as_int()])
    nrows = input_t.size(0)
    ncols = input_t.size(1)
    if input_t.dtype == torch.quint2x4:
        ncols = (ncols + 3) // 4
    elif input_t.dtype == torch.quint4x2:
        ncols = (ncols + 1) // 2
    num_elem_per_byte = 8 // bit_rate
    output_columns = (ncols - 2 * 2) * num_elem_per_byte
    if output_dtype == SparseType.FP32.as_int():
        return torch.empty(
            (nrows, output_columns), dtype=torch.float32, device=input_t.device
        )
    else:  # output_dtype is SparseType.FP16
        return torch.empty(
            (nrows, output_columns), dtype=torch.float16, device=input_t.device
        )


def fused_8_bit_rowwise_quantized_to_float_or_half(
    input_t: Tensor,
    output_dtype: int = 0,
    scale_bias_last: bool = True,
    quant_padding_float_type: bool = True,
) -> Tensor:
    torch._check(
        output_dtype
        in [
            SparseType.FP32.as_int(),
            SparseType.FP16.as_int(),
            SparseType.BF16.as_int(),
        ]
    )
    torch._check(quant_padding_float_type or not scale_bias_last)
    torch._check(input_t.dim() >= 2)
    last_dim = input_t.dim() - 1
    output_shape = list(input_t.shape)
    ncols = input_t.size(last_dim)
    quant_padding_size = 4 if quant_padding_float_type else 2
    ncols_aligned = (
        (ncols + quant_padding_size - 1) // quant_padding_size * quant_padding_size
    )
    output_columns = ncols_aligned - 2 * quant_padding_size
    output_shape[last_dim] = output_columns
    if output_dtype == SparseType.FP32.as_int():
        return torch.empty(output_shape, dtype=torch.float32, device=input_t.device)
    elif output_dtype == SparseType.FP16.as_int():
        return torch.empty(output_shape, dtype=torch.float16, device=input_t.device)
    else:  # output_dtype is SparseType.BF16
        return torch.empty(output_shape, dtype=torch.bfloat16, device=input_t.device)


def float_or_half_to_fused_8_bit_rowwise(
    input_t: Tensor,
) -> Tensor:
    torch._check(input_t.dim() >= 2)
    last_dim = input_t.dim() - 1
    output_shape = list(input_t.shape)
    ncols = input_t.size(last_dim)
    ncols_aligned = (ncols + 4 - 1) // 4 * 4
    output_columns = ncols_aligned + 2 * 4
    output_shape[last_dim] = output_columns
    return torch.empty(output_shape, dtype=torch.uint8, device=input_t.device)


def fused_8_bit_rowwise_quantized_to_float(
    input_t: Tensor,
    scale_bias_last: bool = True,
    quant_padding_float_type: bool = True,
) -> Tensor:
    torch._check(quant_padding_float_type or not scale_bias_last)
    torch._check(input_t.dim() >= 2)
    last_dim = input_t.dim() - 1
    output_shape = list(input_t.shape)
    ncols = input_t.size(last_dim)
    quant_padding_size = 4 if quant_padding_float_type else 2
    ncols_aligned = (
        (ncols + quant_padding_size - 1) // quant_padding_size * quant_padding_size
    )
    output_columns = ncols_aligned - 2 * quant_padding_size
    output_shape[last_dim] = output_columns
    return torch.empty(output_shape, dtype=torch.float32, device=input_t.device)


def fused_8_bit_rowwise_quantized_to_half(
    input_t: Tensor,
    scale_bias_last: bool = True,
    quant_padding_float_type: bool = True,
) -> Tensor:
    torch._check(quant_padding_float_type or not scale_bias_last)
    torch._check(input_t.dim() >= 2)
    last_dim = input_t.dim() - 1
    output_shape = list(input_t.shape)
    ncols = input_t.size(last_dim)
    quant_padding_size = 4 if quant_padding_float_type else 2
    ncols_aligned = (
        (ncols + quant_padding_size - 1) // quant_padding_size * quant_padding_size
    )
    output_columns = ncols_aligned - 2 * quant_padding_size
    output_shape[last_dim] = output_columns
    return torch.empty(output_shape, dtype=torch.float16, device=input_t.device)


def generic_histogram_binning_calibration_by_feature(
    logit: Tensor,
    segment_value: Tensor,
    segment_lengths: Tensor,
    num_segments: int,
    bin_num_examples: Tensor,
    bin_num_positives: Tensor,
    bin_boundaries: Tensor,
    positive_weight: float,
    bin_ctr_in_use_after: int,
    bin_ctr_weight_value: float,
) -> Tuple[Tensor, Tensor]:
    torch._check(bin_num_examples.numel() == bin_num_positives.numel())
    torch._check(
        bin_num_examples.numel() == (num_segments + 1) * (bin_boundaries.numel() + 1)
    )
    return torch.empty_like(logit), torch.empty(
        [logit.numel()], dtype=torch.int64, device=logit.device
    )


def _setup() -> None:
    # pyre-ignore[16]
    _setup.done = getattr(_setup, "done", False)

    # pyre-ignore[2]
    def impl_abstract(op_name, fn) -> None:
        # NOTE: Failures have occasionally been observed with register_fake,
        # where the error signatures can be found in:
        # https://github.com/pytorch/pytorch/blob/main/torch/_library/fake_impl.py
        #
        # To work around this, we first check if the kernel is already registered
        # for the following dispatch keys, and if so, we skip the registration.
        for dkey in ["CompositeImplicitAutograd", "Meta"]:
            if torch._C._dispatch_has_kernel_for_dispatch_key(op_name, dkey):
                return
        torch.library.register_fake(op_name, fn)

    # pyre-ignore[2,24]
    def impl_autograd(op_name, fn, setup_context: Optional[Callable] = None) -> None:
        name_split = op_name.split("::")
        key = f"{name_split[0]}/{name_split[-1]}/Autograd"
        if key not in torch.library._impls:
            torch.library.register_autograd(op_name, fn, setup_context=setup_context)

    if not _setup.done:
        impl_autograd(
            "fbgemm::permute_2D_sparse_data",
            permute_2D_sparse_data_backward,
            setup_context=permute_2D_sparse_data_setup_context,
        )

        impl_abstract("fbgemm::permute_2D_sparse_data", permute_2D_sparse_data_meta)
        impl_abstract(
            "fbgemm::permute_2D_sparse_data_input1D",
            permute_2D_sparse_data_input1D_meta,
        )
        impl_abstract("fbgemm::invert_permute", invert_permute_abstract)
        impl_abstract("fbgemm::permute_1D_sparse_data", permute_1D_sparse_data_meta)
        impl_abstract("fbgemm::masked_select_jagged_1d", masked_select_jagged_1d)
        impl_abstract("fbgemm::tbe_input_combine", tbe_input_combine_abstract)
        impl_abstract(
            "fbgemm::tbe_input_combine_with_length",
            tbe_input_combine_with_length_abstract,
        )
        impl_abstract(
            "fbgemm::jagged_index_select_2d_forward_v2",
            jagged_index_select_2d_forward_v2_abstract,
        )
        impl_abstract(
            "fbgemm::jagged_index_add_2d_forward_v2",
            jagged_index_add_2d_forward_v2_abstract,
        )
        impl_abstract(
            "fbgemm::expand_into_jagged_permute", expand_into_jagged_permute_meta
        )
        impl_abstract("fbgemm::pruned_array_lookup", pruned_array_lookup_meta)
        impl_abstract(
            "fbgemm::int_nbit_split_embedding_codegen_lookup_function",
            int_nbit_split_embedding_codegen_lookup_function_meta,
        )
        impl_abstract(
            "fbgemm::block_bucketize_sparse_features",
            block_bucketize_sparse_features_meta,
        )
        impl_abstract("fbgemm::merge_pooled_embeddings", merge_pooled_embeddings)
        impl_abstract(
            "fbgemm::permute_sparse_features", permute_sparse_features_abstract
        )
        impl_abstract("fbgemm::segment_sum_csr", segment_sum_csr_abstract)
        impl_abstract("fbgemm::dense_to_jagged_forward", dense_to_jagged_forward)
        impl_abstract(
            "fbgemm::batch_index_select_dim0", batch_index_select_dim0_abstract
        )
        impl_abstract(
            "fbgemm::batch_index_select_dim0_tensor",
            batch_index_select_dim0_tensor_abstract,
        )
        impl_abstract(
            "fbgemm::batch_index_select_dim0_forward_cuda_impl",
            batch_index_select_dim0_forward_cuda_impl_abstract,
        )
        impl_abstract(
            "fbgemm::batch_index_select_dim0_tensor_forward_cuda_impl",
            batch_index_select_dim0_tensor_forward_cuda_impl_abstract,
        )
        impl_abstract(
            "fbgemm::batch_index_select_dim0_tensor_backward_cuda_impl",
            batch_index_select_dim0_tensor_backward_cuda_impl_abstract,
        )
        impl_abstract(
            "fbgemm::batch_index_select_dim0_backward_cuda_impl",
            batch_index_select_dim0_backward_cuda_impl_abstract,
        )
        impl_abstract(
            "fbgemm::keyed_jagged_index_select_dim1",
            keyed_jagged_index_select_dim1_abstract,
        )
        impl_abstract(
            "fbgemm::batch_index_select_dim0_forward_cpu_impl",
            batch_index_select_dim0_forward_cpu_impl_abstract,
        )
        impl_abstract(
            "fbgemm::batch_index_select_dim0_tensor_forward_cpu_impl",
            batch_index_select_dim0_tensor_forward_cpu_impl_abstract,
        )
        impl_abstract(
            "fbgemm::batch_index_select_dim0_backward_cpu_impl",
            batch_index_select_dim0_backward_cpu_impl_abstract,
        )
        impl_abstract("fbgemm::bounds_check_indices", bounds_check_indices_abstract)
        impl_abstract(
            "fbgemm::group_index_select_dim0_gpu_impl",
            group_index_select_dim0_gpu_impl_abstract,
        )
        impl_abstract(
            "fbgemm::group_index_select_dim0_gpu_backward",
            group_index_select_dim0_gpu_backward_abstract,
        )
        impl_abstract(
            "fbgemm::keyed_jagged_index_select_dim1_forward",
            keyed_jagged_index_select_dim1_forward_cuda_impl_abstract,
        )
        impl_abstract(
            "fbgemm::keyed_jagged_index_select_dim1_backward",
            keyed_jagged_index_select_dim1_backward_cuda_impl_abstract,
        )
        impl_abstract(
            "fbgemm::permute_pooled_embs_split", permute_pooled_embs_split_abstract
        )
        impl_abstract(
            "fbgemm::histogram_binning_calibration",
            histogram_binning_calibration_abstract,
        )
        impl_abstract(
            "fbgemm::generic_histogram_binning_calibration_by_feature",
            generic_histogram_binning_calibration_by_feature,
        )
        impl_abstract(
            "fbgemm::FloatToHFP8Quantized",
            float_to_hfp8_quantized,
        )
        impl_abstract(
            "fbgemm::HFP8QuantizedToFloat",
            hfp8_quantized_to_float,
        )
        impl_abstract(
            "fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf",
            float_or_half_to_fused_nbit_rowwise_quantized_sbhalf,
        )
        impl_abstract(
            "fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf",
            fused_nbit_rowwise_quantized_sb_half_to_float_or_half,
        )
        impl_abstract(
            "fbgemm::Fused8BitRowwiseQuantizedToFloatOrHalf",
            fused_8_bit_rowwise_quantized_to_float_or_half,
        )
        impl_abstract(
            "fbgemm::FloatToFused8BitRowwiseQuantized",
            float_or_half_to_fused_8_bit_rowwise,
        )
        impl_abstract(
            "fbgemm::FloatOrHalfToFused8BitRowwiseQuantized",
            float_or_half_to_fused_8_bit_rowwise,
        )
        impl_abstract(
            "fbgemm::HalfToFused8BitRowwiseQuantized",
            float_or_half_to_fused_8_bit_rowwise,
        )
        impl_abstract(
            "fbgemm::Fused8BitRowwiseQuantizedToFloat",
            fused_8_bit_rowwise_quantized_to_float,
        )
        impl_abstract(
            "fbgemm::Fused8BitRowwiseQuantizedToHalf",
            fused_8_bit_rowwise_quantized_to_half,
        )
        _setup.done = True


_setup()


@torch.library.register_fake("fbgemm::lengths_range")
def lengths_range_abstract(
    lengths: Tensor,
    output_shape: Optional[Sequence[int]] = None,
) -> Tensor:
    torch._check(lengths.dim() == 1, lambda: "lengths must be a 1D tensor")
    output_size = 0
    if output_shape is not None:
        output_size = math.prod(output_shape)
    else:
        ctx = torch.library.get_ctx()
        output_size = ctx.new_dynamic_size()
    return lengths.new_empty([output_size], dtype=lengths.dtype)


@torch.library.register_fake("fbgemm::all_to_one_device")
def all_to_one_device(
    input_tensors: List[Tensor],
    target_device: torch.device,
) -> List[Tensor]:
    return [
        torch.empty_like(input_tensor, device=torch.device("meta"))
        for input_tensor in input_tensors
    ]
