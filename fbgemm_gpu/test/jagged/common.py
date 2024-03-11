#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import itertools
import sys
import unittest
from typing import Callable, Dict, List, Tuple

import fbgemm_gpu
import fbgemm_gpu.sparse_ops
import numpy as np
import torch
from hypothesis import HealthCheck, settings

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if not open_source:
    if torch.version.hip:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_hip")
    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


suppressed_list: List[HealthCheck] = (
    [HealthCheck.differing_executors]
    if getattr(HealthCheck, "differing_executors", False)
    else []
)

# This health check seems incorrect
settings.register_profile(
    "suppress_differing_executors_check", suppress_health_check=suppressed_list
)
settings.load_profile("suppress_differing_executors_check")


# e.g. "test_faketensor__test_cumsum": [unittest.expectedFailure]
# Please avoid putting tests here, you should put operator-specific
# skips and failures in deeplearning/fbgemm/fbgemm_gpu/test/failures_dict.json
# pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
additional_decorators: Dict[str, List[Callable]] = {
    "test_pt2_compliant_tag_fbgemm_jagged_dense_elementwise_add": [
        # This operator has been grandfathered in. We need to fix this test failure.
        unittest.expectedFailure,
    ],
}


def lengths_to_segment_ids(lengths: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch._dim_arange(lengths, 0).long(),
        lengths.long(),
    )


# Converts lengths + values format to COO format
# [B], [N] -> [B, N'].
# pyre-ignore Missing return annotation [3]
def var_list_to_coo_1d(
    lengths: torch.Tensor,
    values: torch.Tensor,
    N: int,
):
    rows = lengths_to_segment_ids(lengths)
    num_rows = lengths.size()[0]
    # This does D&H sync
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    output_size = lengths.sum()
    # This does D&H sync
    cols = torch.ops.fbgemm.offsets_range(offsets, output_size)
    indices = torch.stack([rows, cols])
    dims = [num_rows, N]
    # torch.sparse_coo_tensor is not supported by torch.fx, wrap it.
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=dims,
    )


# Converts lengths + values format to COO format
# [B], [N, D] -> [B, N', D].
# pyre-ignore Missing return annotation [3]
def var_list_to_coo(lengths: torch.Tensor, values: torch.Tensor, N: int, D: int):
    rows = lengths_to_segment_ids(lengths)
    num_rows = lengths.size()[0]
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    output_size = lengths.sum()
    # This does D&H sync
    cols = torch.ops.fbgemm.offsets_range(offsets, output_size)
    indices = torch.stack([rows, cols])
    dims = [num_rows, N, D]
    # torch.sparse_coo_tensor is not supported by torch.fx, wrap it.
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=dims,
    )


# TODO: Reuse this code in test_(stacked)_jagged_1/2d
def generate_jagged_tensor(
    num_jagged_dim: int,
    outer_dense_size: int,
    inner_dense_size: int,
    dtype: torch.dtype,
    device: torch.device,
    fold_inner_dense: bool = False,
    # dynamo to mark the input as dynamic shape to make sure symbolic
    # shape is generated
    mark_dynamic: bool = False,
) -> Tuple[torch.Tensor, List[torch.LongTensor], np.ndarray]:
    max_lengths = np.random.randint(low=1, high=10, size=(num_jagged_dim,))
    x_offsets: List[torch.LongTensor] = []
    num_lengths = outer_dense_size
    for d in range(num_jagged_dim):
        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(
            # PT2 specialize 0/1 dims as non-symbolic shape. So we need
            # to make it non 0/1 for testing. In real cases it'll likelyl
            # not 0/1 anyway (if so, they'll be recompiled)
            low=0 if not mark_dynamic else 1,
            high=max_lengths[d] * 2,
            # pyre-fixme[6]: For 3rd param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Union[bool, float, int]]`.
            size=(num_lengths,),
            device=device,
        )
        offset = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        if mark_dynamic:
            torch._dynamo.mark_dynamic(offset, 0)
        x_offsets.append(offset)
        num_lengths = x_offsets[-1][-1].item()

    x_values = torch.rand(
        # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
        #  typing.Tuple[int, ...]]` but got `Tensor`.
        x_offsets[-1][-1] * inner_dense_size,
        dtype=dtype,
        device=device,
    )
    if inner_dense_size != 1 or not fold_inner_dense:
        # pyre-fixme[6]: For 1st param expected `int` but got `Union[bool, float, int]`.
        x_values = x_values.reshape(x_offsets[-1][-1].item(), inner_dense_size)

    if mark_dynamic:
        for i in range(inner_dense_size):
            torch._dynamo.mark_dynamic(x_values, i)

    return x_values, x_offsets, max_lengths


def to_padded_dense(
    values: torch.Tensor,
    offsets: List[torch.LongTensor],
    max_lengths: np.ndarray,
    padding_value: float = 0,
) -> torch.Tensor:
    outer_dense_size = len(offsets[0]) - 1
    # canonicalize by unsqueeze the last dim if the inner dense dimension
    # is 1 and folded.
    inner_dense_size = 1 if values.ndim == 1 else values.size(-1)
    dense = torch.empty(
        (outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,),
        dtype=values.dtype,
        device=values.device,
    )
    for i in range(outer_dense_size):
        for jagged_coord in itertools.product(
            *(list(range(max_l)) for max_l in max_lengths)
        ):
            cur_offset = i
            is_zero = False
            for d in range(len(max_lengths)):
                # pyre-fixme[6]: For 1st argument expected `Union[None, _NestedSe...
                begin = offsets[d][cur_offset].item()
                # pyre-fixme[6]: For 1st argument expected `Union[None, _NestedSe...
                end = offsets[d][cur_offset + 1].item()
                # pyre-fixme[6]: For 1st param expected `int` but got
                #  `Union[bool, float, int]`.
                if jagged_coord[d] >= end - begin:
                    is_zero = True
                    break
                cur_offset = begin + jagged_coord[d]
            dense[(i,) + jagged_coord] = (
                padding_value
                if is_zero
                # pyre-fixme[6]: For 1st argument expected `Union[None, _NestedSe...
                else values[cur_offset]
            )
    return dense.squeeze(-1) if values.ndim == 1 else dense


# pyre-fixme[2]
# pyre-fixme[24]
def torch_compiled(model: Callable, **kwargs) -> Callable:
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    else:
        return model
