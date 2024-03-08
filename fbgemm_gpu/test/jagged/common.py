#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import sys
import unittest
from typing import Callable, Dict, List

import fbgemm_gpu
import fbgemm_gpu.sparse_ops
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


def hash_size_cumsum_to_offsets(hash_size_cum_sum_list: List[int]) -> List[int]:
    hash_size_offsets_list = [0]
    count = 0
    for f in range(1, len(hash_size_cum_sum_list)):
        count = count + 1
        if hash_size_cum_sum_list[f] == hash_size_cum_sum_list[f - 1]:
            curr_offsets = hash_size_offsets_list[-1]
            hash_size_offsets_list.append(curr_offsets)
        else:
            hash_size_offsets_list.append(count)
    hash_size_offsets_list[-1] = count
    return hash_size_offsets_list


# pyre-fixme[2]
# pyre-fixme[24]
def torch_compiled(model: Callable, **kwargs) -> Callable:
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    else:
        return model
