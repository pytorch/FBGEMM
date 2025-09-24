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

    @given(
        num_elements=st.integers(min_value=100, max_value=10000),
        num_unique_indices=st.integers(min_value=5, max_value=100),
        weight_dtype=st.sampled_from([torch.float32, torch.float16]),
        use_cpu=st.booleans() if not gpu_unavailable[0] else st.just(True),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_acc_weights_and_counts_1d(
        self,
        num_elements: int,
        num_unique_indices: int,
        weight_dtype: torch.dtype,
        use_cpu: bool,
    ) -> None:
        """Test 1D weight accumulation kernel against torch native implementation.

        Tests both CPU and GPU implementations.
        """
        device = torch.device("cpu" if use_cpu else "cuda")

        # Generate test data
        weights = torch.randn(num_elements, dtype=weight_dtype, device=device)
        reverse_indices = torch.randint(
            0, num_unique_indices, (num_elements,), dtype=torch.int64, device=device
        )

        # Test our optimized kernel
        result_optimized = torch.ops.fbgemm.jagged_acc_weights_and_counts(
            weights, reverse_indices, num_unique_indices
        )

        # Reference implementation using torch native operations
        result_reference = torch.zeros(
            (num_unique_indices, 2), dtype=torch.float32, device=device
        )

        # Accumulate weights and counts using scatter_add (torch native)
        weights_float = weights.float()
        counts = torch.ones_like(weights_float)

        result_reference[:, 0].scatter_add_(0, reverse_indices, weights_float)
        result_reference[:, 1].scatter_add_(0, reverse_indices, counts)

        # Compare results
        torch.testing.assert_close(
            result_optimized, result_reference, rtol=1e-4, atol=1e-5
        )

        # Verify output shape and types
        self.assertEqual(result_optimized.shape, (num_unique_indices, 2))
        self.assertEqual(result_optimized.dtype, torch.float32)
        self.assertEqual(result_optimized.device.type, device.type)

    @given(
        use_cpu=st.booleans() if not gpu_unavailable[0] else st.just(True),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_jagged_acc_weights_and_counts_edge_cases(self, use_cpu: bool) -> None:
        """Test edge cases for both 1D and 2D accumulation kernels.

        Tests both CPU and GPU implementations.
        """
        device = torch.device("cpu" if use_cpu else "cuda")

        # Test case 1: Single element
        weights_1d = torch.tensor([5.0], device=device)
        reverse_indices = torch.tensor([0], dtype=torch.int64, device=device)
        result = torch.ops.fbgemm.jagged_acc_weights_and_counts(
            weights_1d, reverse_indices, 1
        )
        expected = torch.tensor([[5.0, 1.0]], device=device)
        torch.testing.assert_close(result, expected)

        # Test case 2: All elements map to same unique index
        weights_1d = torch.tensor([1.0, 2.0, 3.0], device=device)
        reverse_indices = torch.tensor([0, 0, 0], dtype=torch.int64, device=device)
        result = torch.ops.fbgemm.jagged_acc_weights_and_counts(
            weights_1d, reverse_indices, 1
        )
        expected = torch.tensor([[6.0, 3.0]], device=device)
        torch.testing.assert_close(result, expected)

    @given(use_cpu=st.booleans() if not gpu_unavailable[0] else st.just(True))
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_jagged_acc_weights_and_counts_different_sizes(self, use_cpu: bool) -> None:
        """Test that the kernel works correctly with different dataset sizes.

        Tests both small and large datasets to ensure the implementation works
        correctly across different scales. For CPU, just tests basic functionality.
        """
        device = torch.device("cpu" if use_cpu else "cuda")

        # Test small dataset
        small_weights = torch.randn(500, device=device)
        small_reverse_indices = torch.randint(
            0, 10, (500,), dtype=torch.int64, device=device
        )
        result_small = torch.ops.fbgemm.jagged_acc_weights_and_counts(
            small_weights, small_reverse_indices, 10
        )

        # Test large dataset
        large_weights = torch.randn(5000, device=device)
        large_reverse_indices = torch.randint(
            0, 50, (5000,), dtype=torch.int64, device=device
        )
        result_large = torch.ops.fbgemm.jagged_acc_weights_and_counts(
            large_weights, large_reverse_indices, 50
        )

        # Both should produce valid results
        self.assertEqual(result_small.shape, (10, 2))
        self.assertEqual(result_large.shape, (50, 2))

        # Verify results are reasonable (non-negative counts, finite weights)
        self.assertTrue(
            torch.all(result_small[:, 1] >= 0)
        )  # Counts should be non-negative
        self.assertTrue(
            torch.all(result_large[:, 1] >= 0)
        )  # Counts should be non-negative
        self.assertTrue(torch.all(torch.isfinite(result_small)))
        self.assertTrue(torch.all(torch.isfinite(result_large)))


if __name__ == "__main__":
    unittest.main()
