#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import fbgemm_gpu
import numpy as np
import torch
from hypothesis import settings, Verbosity

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if not open_source:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:cumem_utils")

torch.ops.import_module("fbgemm_gpu.sparse_ops")
settings.register_profile("derandomize", derandomize=True)
settings.load_profile("derandomize")


MAX_EXAMPLES = 40

# For long running tests reduce the number of iterations to reduce timeout errors.
MAX_EXAMPLES_LONG_RUNNING = 15

VERBOSITY: Verbosity = Verbosity.verbose


def gen_mixed_B_batch_sizes(B: int, T: int) -> Tuple[List[List[int]], List[int]]:
    num_ranks = np.random.randint(low=1, high=4)
    low = max(int(0.25 * B), 1)
    high = int(B)
    if low == high:
        Bs_rank_feature = [[B] * num_ranks for _ in range(T)]
    else:
        Bs_rank_feature = [
            np.random.randint(low=low, high=high, size=num_ranks).tolist()
            for _ in range(T)
        ]
    Bs = [sum(Bs_feature) for Bs_feature in Bs_rank_feature]
    return Bs_rank_feature, Bs


def format_ref_tensors_in_mixed_B_layout(
    ref_tensors: List[torch.Tensor], Bs_rank_feature: List[List[int]]
) -> torch.Tensor:
    # Relayout the reference tensor
    # Jagged dimension: (rank, table, local batch)
    num_ranks = len(Bs_rank_feature[0])
    split_tensors = [[] for _ in range(num_ranks)]  # shape (rank, table)
    for t, ref_tensor in enumerate(ref_tensors):
        assert ref_tensor.shape[0] == sum(Bs_rank_feature[t])
        tensors = ref_tensor.split(Bs_rank_feature[t])
        for r, tensor in enumerate(tensors):
            split_tensors[r].append(tensor.flatten())
    concat_list = []
    for r in range(num_ranks):
        concat_list += split_tensors[r]
    return torch.cat(concat_list, dim=0)
