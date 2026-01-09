#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import random
import unittest
from typing import cast, Optional

import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from .common import extend_test_class, open_source, permute_indices_ref_

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable, on_oss_clang, running_in_oss
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import (
        gpu_available,
        gpu_unavailable,
        on_oss_clang,
        running_in_oss,
    )


class PermuteSparseFeaturesTest(unittest.TestCase):
    def permute_sparse_features_ref_(
        self,
        lengths: torch.Tensor,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        permute: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        T = lengths.size(0)
        B = lengths.size(1)
        permuted_lengths = torch.index_select(lengths.view(T, B), 0, permute)

        original_segment_lengths = lengths.view(T, B).sum(dim=1, dtype=torch.int32)
        original_segment_start = torch.ops.fbgemm.asynchronous_exclusive_cumsum(
            original_segment_lengths.view(-1)
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

    @unittest.skipIf(*on_oss_clang)
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_permute_sparse_features(
        self,
        B: int,
        T: int,
        L: int,
        long_index: bool,
        has_weight: bool,
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        weights = torch.rand(int(lengths.sum().item())).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            size=cast(tuple[int, ...], (lengths.sum().item(),)),
        ).type(index_dtype)
        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_sparse_features(permute, lengths, indices, weights)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
            # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
        ) = permute_indices_ref_(lengths, indices, weights, permute.long())
        torch.testing.assert_close(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_close(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_close(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if gpu_available:
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_sparse_features(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                weights.cuda() if has_weight and weights is not None else None,
            )
            torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
            torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_gpu is None

    @unittest.skipIf(*on_oss_clang)
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_permute_sparse_features_with_repeats(
        self, B: int, T: int, L: int, long_index: bool, has_weight: bool
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        weights = torch.rand(int(lengths.sum().item())).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            size=cast(tuple[int, ...], (lengths.sum().item(),)),
        ).type(index_dtype)
        permute_list = list(range(T))

        num_repeats = random.randint(0, T)
        for _ in range(num_repeats):
            permute_list.append(random.randint(0, T - 1))

        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_sparse_features(permute, lengths, indices, weights)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
            # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
        ) = permute_indices_ref_(lengths, indices, weights, permute.long())
        torch.testing.assert_close(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_close(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_close(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if gpu_available:
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_sparse_features(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                weights.cuda() if has_weight and weights is not None else None,
            )
            torch.testing.assert_close(permuted_indices_gpu.cpu(), permuted_indices_cpu)
            torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_cpu is None


class Permute1DSparseFeaturesTest(unittest.TestCase):
    @unittest.skipIf(*running_in_oss)
    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
        weight_columns=st.integers(min_value=1, max_value=20),
    )
    @settings(
        max_examples=20,
        deadline=None,
    )
    def test_permute_1D_sparse_data(
        self,
        T: int,
        L: int,
        long_index: bool,
        has_weight: bool,
        weight_columns: int,
    ) -> None:
        # Setup: Choose index data type based on test parameter
        index_dtype = torch.int64 if long_index else torch.int32

        # Create 1D lengths tensor representing sparse feature counts for T features
        lengths = torch.randint(
            low=1,
            high=L,
            size=(T,),  # 1D tensor with T elements
            device=torch.accelerator.current_accelerator(),
        ).type(index_dtype)

        # Create optional 2D weights tensor with dimensions [total_indices, weight_columns]
        weights = (
            torch.rand(
                int(lengths.sum().item()),
                weight_columns,
                device=torch.accelerator.current_accelerator(),
            ).float()
            if has_weight
            else None
        )

        # Create indices tensor containing sparse feature indices
        indices = torch.randint(
            low=1,
            high=int(1e5),
            size=cast(tuple[int, ...], (lengths.sum().item(),)),
            device=torch.accelerator.current_accelerator(),
        ).type(index_dtype)

        # Create random permutation for shuffling features
        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list).to(
            device=torch.accelerator.current_accelerator()
        )
        # Execute: Call the permute_1D_sparse_data operation
        (
            lengths_actual,
            values_actual,
            weights_actual,
        ) = torch.ops.fbgemm.permute_1D_sparse_data(
            permute, lengths, indices, weights, indices.numel()
        )

        # Assert: Verify that the lengths were correctly permuted
        # The permuted lengths should match the original lengths indexed by the permutation
        self.assertTrue(
            torch.equal(
                lengths_actual, torch.index_select(lengths, dim=0, index=permute)
            )
        )

        # Track the current position in the permuted output for validation
        permuted_cumulated_index = 0

        # Compute cumulative offsets to locate each feature's data in the original arrays
        # Prepend a zero to get offsets: [0, lengths[0], lengths[0]+lengths[1], ...]
        cumulative_indices = torch.cumsum(
            torch.cat(
                (
                    torch.zeros((1,), dtype=index_dtype, device=lengths.device),
                    lengths,
                )
            ),
            dim=0,
        )

        # Verify each feature's data was correctly permuted
        for i in range(T):
            # Get the original feature index that should appear at position i in the permuted output
            permuted_index = permute[i]

            # Assert: Verify that the indices for this feature were correctly copied
            # Compare the segment in the permuted output against the original segment
            self.assertTrue(
                torch.equal(
                    values_actual[
                        permuted_cumulated_index : permuted_cumulated_index
                        + lengths[permuted_index]
                    ],
                    indices[
                        cumulative_indices[permuted_index] : lengths[permuted_index]
                        + cumulative_indices[permuted_index]
                    ],
                )
            )

            # Assert: If weights are present, verify they were also correctly permuted
            if has_weight and weights is not None:
                self.assertTrue(
                    torch.equal(
                        weights_actual[
                            permuted_cumulated_index : permuted_cumulated_index
                            + lengths[permuted_index]
                        ],
                        weights[
                            cumulative_indices[permuted_index] : lengths[permuted_index]
                            + cumulative_indices[permuted_index]
                        ],
                    )
                )
            else:
                # Assert: If no weights were provided, ensure the output also has no weights
                assert weights_actual is None

            # Move to the next segment in the permuted output
            permuted_cumulated_index += lengths[permuted_index]


class Permute2DSparseFeaturesTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_permute_2D_sparse_data(self) -> None:
        lengths = torch.tensor(
            [[0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]],
            dtype=torch.int32,
            device=torch.accelerator.current_accelerator(),
        )
        indices = torch.tensor(
            [500, 1000, 1999],
            dtype=torch.int32,
            device=torch.accelerator.current_accelerator(),
        )
        permute = torch.tensor(
            [0, 3, 1, 4, 2, 5],
            dtype=torch.int32,
            device=torch.accelerator.current_accelerator(),
        )
        weights = torch.rand((3, 64), device=torch.accelerator.current_accelerator())
        (
            lengths_actual,
            values_actual,
            weights_actual,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, weights, indices.numel()
        )
        self.assertTrue(
            torch.equal(
                lengths_actual,
                torch.tensor(
                    [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]],
                    dtype=torch.int32,
                    device=torch.accelerator.current_accelerator(),
                ),
            )
        )
        self.assertTrue(torch.equal(values_actual, indices))
        self.assertTrue(torch.equal(weights_actual, weights))


extend_test_class(PermuteSparseFeaturesTest)

if __name__ == "__main__":
    unittest.main()
