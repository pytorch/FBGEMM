#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import unittest
from itertools import accumulate
from typing import List, Tuple

import hypothesis.strategies as st
import torch
from fbgemm_gpu import sparse_ops  # noqa: F401
from hypothesis import given, HealthCheck, settings

try:
    from test_utils import gpu_unavailable  # pyre-ignore[21]
except Exception:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


def gen_inputs(
    hash_sizes: List[int],
    batch_size: int,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    the lengths of bags are chosen from
    a uniform distribution from [0, max_len]
    """

    T = len(hash_sizes)

    offsets = [0]
    indices_per_table = []
    for t in range(T):
        len_sum = 0
        for _ in range(batch_size):
            length = random.randint(0, max_len)
            len_sum += length
            offsets.append(offsets[-1] + length)

        n_rows = hash_sizes[t]
        indices_per_table.append(torch.randint(n_rows, [len_sum], dtype=torch.int64))

    indices = torch.cat(indices_per_table, dim=0)
    offsets = torch.tensor(offsets, dtype=torch.int64)

    return indices, offsets


def transpose_embedding_input_ref(
    hash_size_cumsum: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    info_B_num_bits: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    reference implementation of torch.ops.fbgemm.transpose_embedding_input
    """

    T = hash_size_cumsum.numel() - 1
    B = (offsets.numel() - 1) // T
    linear_indices = torch.zeros_like(indices)
    infos = torch.zeros_like(indices)
    for b_t in range(B * T):
        t = b_t // B
        b = b_t % B
        start = int(offsets[b_t].item())
        end = int(offsets[b_t + 1].item())
        for i in range(start, end):
            linear_indices[i] = indices[i] + hash_size_cumsum[t]
            infos[i] = (t << info_B_num_bits) | b

    linear_indices_sorted, sorted_idx = torch.sort(linear_indices, stable=True)
    infos_sorted = infos[sorted_idx]

    (
        sorted_linear_indices_run,
        sorted_linear_indices_run_lengths,
    ) = torch.unique_consecutive(linear_indices_sorted, return_counts=True)

    sorted_linear_indices_num_runs = torch.tensor(
        sorted_linear_indices_run.numel(), dtype=torch.int64
    )
    sorted_linear_indices_cumulative_run_lengths = torch.tensor(
        [0] + list(accumulate(sorted_linear_indices_run_lengths.tolist())),
        dtype=torch.int64,
    )

    return (
        linear_indices,
        linear_indices_sorted,
        infos_sorted,
        sorted_linear_indices_run,
        sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths,
    )


class SplitEmbeddingsUtilsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=10, max_value=25),
        T=st.integers(min_value=5, max_value=20),
        E=st.integers(min_value=10, max_value=50),
    )
    @settings(deadline=30000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_transpose(self, B: int, T: int, E: int) -> None:
        hash_sizes = [random.randint(E, 2 * E) for _ in range(T)]
        batch_size = B
        max_len = 3 * E

        total_hash_size_bits: int = int(math.log2(sum(hash_sizes)) + 1)
        hash_size_cumsum = torch.tensor(
            [0] + list(accumulate(hash_sizes)), dtype=torch.int64
        )

        indices, offsets = gen_inputs(hash_sizes, batch_size, max_len)
        hash_size_cumsum_cuda = hash_size_cumsum.cuda()

        info_B_num_bits, _ = torch.ops.fbgemm.get_infos_metadata(
            hash_size_cumsum_cuda, B, T
        )

        (
            linear_indices,
            linear_indices_sorted,
            infos_sorted,
            sorted_linear_indices_run,
            sorted_linear_indices_run_lengths,
            sorted_linear_indices_num_runs,
            sorted_linear_indices_cumulative_run_lengths,
        ) = torch.ops.fbgemm.transpose_embedding_input(
            hash_size_cumsum_cuda,
            total_hash_size_bits,
            indices.cuda(),
            offsets.cuda(),
            info_B_num_bits=info_B_num_bits,
        )

        (
            linear_indices_ref,
            linear_indices_sorted_ref,
            infos_sorted_ref,
            sorted_linear_indices_run_ref,
            sorted_linear_indices_run_lengths_ref,
            sorted_linear_indices_num_runs_ref,
            sorted_linear_indices_cumulative_run_lengths_ref,
        ) = transpose_embedding_input_ref(
            hash_size_cumsum, indices, offsets, info_B_num_bits
        )

        self.assertTrue(torch.equal(linear_indices.cpu(), linear_indices_ref))
        self.assertTrue(
            torch.equal(linear_indices_sorted.cpu(), linear_indices_sorted_ref)
        )
        self.assertTrue(torch.equal(infos_sorted.cpu(), infos_sorted_ref))

        # fbgemm impl has padding so we need slice
        num = sorted_linear_indices_run_ref.numel()
        self.assertTrue(
            torch.equal(
                sorted_linear_indices_run.cpu()[:num], sorted_linear_indices_run_ref
            )
        )
        self.assertTrue(
            torch.equal(
                sorted_linear_indices_run_lengths.cpu()[:num],
                sorted_linear_indices_run_lengths_ref,
            )
        )
        self.assertEqual(
            sorted_linear_indices_num_runs.item(),
            sorted_linear_indices_num_runs_ref.item(),
        )
        self.assertTrue(
            torch.equal(
                sorted_linear_indices_cumulative_run_lengths.cpu()[: num + 1],
                sorted_linear_indices_cumulative_run_lengths_ref,
            )
        )
