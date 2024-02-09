#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import os
import unittest
from itertools import accumulate
from typing import Callable, Dict, List, Optional, Tuple, Type

import fbgemm_gpu
import torch
from hypothesis import HealthCheck, settings
from torch._utils_internal import get_file_path_2
from torch.testing._internal.optests import generate_opcheck_tests

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if not open_source:
    if torch.version.hip:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_hip")
    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:index_select_ops")

suppressed_list: List[HealthCheck] = (
    # pyre-fixme[16]: Module `HealthCheck` has no attribute `differing_executors`.
    [HealthCheck.differing_executors]
    if getattr(HealthCheck, "differing_executors", False)
    else []
)


@settings(suppress_health_check=suppressed_list)
def permute_indices_ref_(
    lengths: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor],
    permute: torch.LongTensor,
    is_1D: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    T = lengths.size(0)
    B = lengths.size(1)
    if T == 0 or B == 0:
        if is_1D:
            lengths = lengths.view(-1)
        return lengths, indices, weights

    if is_1D:
        permuted_lengths = torch.index_select(lengths.view(-1), 0, permute).view(-1)
        original_segment_lengths = lengths.view(-1)
        original_segment_start = [0] + list(accumulate(lengths.view(-1)))

        permuted_indices = []
        permuted_weights = []
        for i in range(permute.numel()):
            start = original_segment_start[permute[i]]
            end = start + original_segment_lengths[permute[i]]
            permuted_indices.append(indices[start:end])
            if weights is not None:
                permuted_weights.append(weights[start:end])

        permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

        if weights is None:
            permuted_weights = None
        else:
            permuted_weights = torch.cat(permuted_weights, dim=0).flatten()
    else:
        permuted_lengths = torch.index_select(lengths.view(T, -1), 0, permute)
        original_segment_lengths = lengths.view(T, -1).sum(dim=1, dtype=torch.int32)
        original_segment_start = [0] + list(
            accumulate(original_segment_lengths.view(-1))
        )

        permuted_indices = []
        permuted_weights = []
        for i in range(permute.size(0)):
            start = original_segment_start[permute[i]]
            end = start + original_segment_lengths[permute[i]]
            permuted_indices.append(indices[start:end])
            if weights is not None:
                permuted_weights.append(weights[start:end])

        permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

        if weights is None:
            permuted_weights = None
        else:
            permuted_weights = torch.cat(permuted_weights, dim=0).flatten()

    return permuted_lengths, permuted_indices, permuted_weights


@torch.jit.script
def permute_scripted(
    permute: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    (
        permuted_lengths_cpu,
        permuted_indices_cpu,
        permuted_weights_cpu,
    ) = torch.ops.fbgemm.permute_2D_sparse_data(permute, lengths, indices, None, None)
    return (
        permuted_lengths_cpu,
        permuted_indices_cpu,
        permuted_weights_cpu,
    )


def extend_test_class(
    klass: Type[unittest.TestCase],
    # e.g. "test_faketensor__test_cumsum": [unittest.expectedFailure]
    # Please avoid putting tests here, you should put operator-specific
    # skips and failures in deeplearning/fbgemm/fbgemm_gpu/test/failures_dict.json
    # pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
    additional_decorators: Optional[Dict[str, List[Callable]]] = None,
) -> None:
    failures_dict_path: str = get_file_path_2(
        "", os.path.dirname(__file__), "failures_dict.json"
    )

    # pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
    base_decorators: Dict[str, List[Callable]] = {
        "test_pt2_compliant_tag_fbgemm_jagged_dense_elementwise_add": [
            # This operator has been grandfathered in. We need to fix this test failure.
            unittest.expectedFailure,
        ],
        "test_pt2_compliant_tag_fbgemm_jagged_dense_elementwise_add_jagged_output": [
            # This operator has been grandfathered in. We need to fix this test failure.
            unittest.expectedFailure,
        ],
    }

    additional_decorators = additional_decorators or {}

    # Only generate tests for PyTorch 2.2+
    if (
        torch.__version__ >= "2.2.*"
        and hasattr(torch.library, "impl_abstract")
        and not hasattr(fbgemm_gpu, "open_source")
    ):
        generate_opcheck_tests(
            klass,
            ["fb", "fbgemm"],
            failures_dict_path,
            {**base_decorators, **additional_decorators},
            [
                "test_schema",
                "test_autograd_registration",
                "test_faketensor",
                "test_aot_dispatch_dynamic",
            ],
        )
