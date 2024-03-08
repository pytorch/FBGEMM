#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import itertools
import random
import unittest
from typing import List

import hypothesis.strategies as st
import numpy as np
import torch
import torch._dynamo
from hypothesis import given, settings, Verbosity

from .common import additional_decorators, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests, symint_vector_unsupported
else:
    from fbgemm_gpu.test.test_utils import (
        gpu_unavailable,
        optests,
        symint_vector_unsupported,
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


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class UniqueIndicesTest(unittest.TestCase):
    def setUp(self) -> None:
        if symint_vector_unsupported()[0]:
            return

        assert hasattr(
            torch._dynamo.config, "assume_static_by_default"
        ), "Need to update the config as the dynamic/auto-dynamic setting has changed"
        # Turn off static assumption for auto-dynamic
        torch._dynamo.config.assume_static_by_default = False

    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=100, max_value=200),
        F=st.integers(min_value=50, max_value=100),
        max_length=st.integers(min_value=5, max_value=10),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_jagged_unique_indices(
        self,
        B: int,  # Batch size
        F: int,  # The number of features
        max_length: int,  # The maximum value of pooling factor
    ) -> None:
        hash_size_list = []
        lengths_list = []
        indices_list = []
        linearized_indices_list = []
        hash_size_offsets_list = [0]
        for _ in range(F):
            # We generate a small hash size to increase index duplication
            hash_size = random.randint(3, 5)
            hash_size_list.append(hash_size)
            hash_size_offset = hash_size_offsets_list[-1] + 1
            hash_size_offsets_list.append(hash_size_offset)
            for _ in range(B):
                length = random.randint(0, max_length)
                lengths_list.append(length)
                if length > 0:
                    indices = np.random.randint(0, hash_size, size=length)
                    linearized_indices = indices + sum(hash_size_list[:-1])
                    indices_list.extend(indices)
                    linearized_indices_list.extend(linearized_indices)

        device = torch.device("cuda")
        dtype = torch.int64
        hash_size = torch.as_tensor(hash_size_list, dtype=dtype, device=device)
        hash_size_offsets = torch.as_tensor(
            hash_size_offsets_list, dtype=dtype, device=device
        )
        lengths = torch.as_tensor(lengths_list, dtype=dtype, device=device)
        indices = torch.as_tensor(indices_list, dtype=dtype, device=device)
        linearized_indices = torch.as_tensor(
            linearized_indices_list, dtype=dtype, device=device
        )

        hash_size_cum_sum = torch.zeros(F + 1, dtype=dtype, device=device)
        hash_size_cum_sum[1:] = torch.cumsum(hash_size, dim=0)
        offsets = torch.zeros(F * B + 1, dtype=dtype, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0)

        (
            output_lengths,
            output_offsets,
            unique_indices,
            reverse_index,
        ) = torch.ops.fbgemm.jagged_unique_indices(
            hash_size_cum_sum, hash_size_offsets, offsets, indices
        )

        # Check hash size cumsum to offsets function
        output_hash_size_offsets_list = hash_size_cumsum_to_offsets(
            hash_size_cum_sum.tolist()
        )
        self.assertEqual(output_hash_size_offsets_list, hash_size_offsets_list)

        # Compute hash size cumsum and offsets based on KJT offsets and indices
        (
            inferred_hash_size_cum_sum,
            inferred_hash_size_offsets,
        ) = torch.ops.fbgemm.jagged_hash_size_cumsum(offsets, indices, B)
        (
            output_lengths_inf,
            output_offsets_inf,
            unique_indices_inf,
            reverse_index_inf,
        ) = torch.ops.fbgemm.jagged_unique_indices(
            inferred_hash_size_cum_sum, inferred_hash_size_offsets, offsets, indices
        )

        self.assertTrue(torch.equal(output_lengths, output_lengths_inf))
        self.assertTrue(torch.equal(output_offsets, output_offsets_inf))
        self.assertTrue(torch.equal(unique_indices, unique_indices_inf))
        self.assertTrue(torch.equal(reverse_index, reverse_index_inf))

        unique_linearized_indices = torch.unique(linearized_indices, sorted=True)
        self.assertTrue(unique_linearized_indices.numel() == unique_indices.numel())

        unique_indices_list = unique_indices.tolist()
        reverse_index_list = reverse_index.tolist()
        for i in range(len(reverse_index_list)):
            pos = reverse_index_list[i]
            self.assertTrue(unique_indices_list[pos] == indices_list[i])

        input_offsets_list = offsets.tolist()
        output_offsets_list = output_offsets.tolist()
        for i in range(F):
            input_start = input_offsets_list[i * B]
            input_end = input_offsets_list[(i + 1) * B]
            output_start = output_offsets_list[i * B]
            output_end = output_offsets_list[(i + 1) * B]
            for each_offset in range(input_start, input_end):
                pos = reverse_index_list[each_offset]
                self.assertTrue((output_start <= pos) and (pos < output_end))

    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=100, max_value=200),
        F=st.integers(min_value=50, max_value=100),
        max_length=st.integers(min_value=5, max_value=10),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_jagged_unique_indices_multi_keys(
        self,
        B: int,  # Batch size
        F: int,  # The number of features
        max_length: int,  # The maximum value of pooling factor
    ) -> None:
        hash_size_list = []
        lengths_list = []
        indices_list = []
        linearized_indices_list = []
        MAX_HASH_SIZE = 10
        for _ in range(F):
            # We generate a small hash size to increase index duplication
            hash_size = random.randint(3, 6)
            self.assertTrue(hash_size <= MAX_HASH_SIZE)
            masked_hash_size = MAX_HASH_SIZE if random.randint(1, 3) == 3 else 0
            hash_size_list.append(masked_hash_size)
            for _ in range(B):
                length = random.randint(0, max_length)
                lengths_list.append(length)
                if length > 0:
                    indices = np.random.randint(0, hash_size, size=length)
                    linearized_indices = indices + sum(hash_size_list[:-1])
                    indices_list.extend(indices)
                    linearized_indices_list.extend(linearized_indices)

        device = torch.device("cuda")
        dtype = torch.int64
        hash_size = torch.as_tensor(hash_size_list, dtype=dtype, device=device)
        lengths = torch.as_tensor(lengths_list, dtype=dtype, device=device)
        indices = torch.as_tensor(indices_list, dtype=dtype, device=device)
        linearized_indices = torch.as_tensor(
            linearized_indices_list, dtype=dtype, device=device
        )

        hash_size_cum_sum = torch.zeros(F + 1, dtype=dtype, device=device)
        hash_size_cum_sum[1:] = torch.cumsum(hash_size, dim=0)
        offsets = torch.zeros(F * B + 1, dtype=dtype, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0)

        # Compute hash size offsets based on hash size cumsum to dedup
        # indices from multiple keys
        hash_size_cum_sum_list = hash_size_cum_sum.tolist()
        hash_size_offsets_list = hash_size_cumsum_to_offsets(hash_size_cum_sum_list)
        hash_size_offsets = torch.as_tensor(
            hash_size_offsets_list, dtype=dtype, device=device
        )

        (
            _,  # output lengths
            _,  # output offsets
            unique_indices,
            reverse_index,
        ) = torch.ops.fbgemm.jagged_unique_indices(
            hash_size_cum_sum, hash_size_offsets, offsets, indices
        )

        unique_linearized_indices = torch.unique(linearized_indices, sorted=True)
        self.assertTrue(unique_linearized_indices.numel() == unique_indices.numel())

        unique_indices_list = unique_indices.tolist()
        reverse_index_list = reverse_index.tolist()
        for i in range(len(reverse_index_list)):
            pos = reverse_index_list[i]
            self.assertTrue(unique_indices_list[pos] == indices_list[i])

    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=100, max_value=200),
        F=st.integers(min_value=50, max_value=100),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_jagged_unique_indices_empty(
        self,
        B: int,  # Batch size
        F: int,  # The number of features
    ) -> None:
        hash_size_cumsum_list = [0] + list(itertools.accumulate([10] * F))
        hash_size_offsets_list = [0] + list(itertools.accumulate([1] * F))
        offsets_list = [0] * (B * F + 1)
        indices_list = []

        device = torch.device("cuda")
        dtype = torch.int64
        hash_size_cumsum = torch.as_tensor(
            hash_size_cumsum_list, device=device, dtype=dtype
        )
        hash_size_offsets = torch.as_tensor(
            hash_size_offsets_list, device=device, dtype=dtype
        )
        offsets = torch.as_tensor(offsets_list, device=device, dtype=dtype)
        indices = torch.as_tensor(indices_list, device=device, dtype=dtype)

        (
            output_lengths,
            output_offsets,
            unique_indices,
            reverse_index,
        ) = torch.ops.fbgemm.jagged_unique_indices(
            hash_size_cumsum, hash_size_offsets, offsets, indices
        )

        # The output should be empty since there are no input indices
        self.assertEqual(unique_indices.numel(), 0)
        self.assertEqual(reverse_index.numel(), 0)
        self.assertEqual(torch.sum(output_lengths).item(), 0)
        self.assertEqual(torch.sum(output_offsets).item(), 0)


if __name__ == "__main__":
    unittest.main()
