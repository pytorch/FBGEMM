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
from typing import Optional

import torch
from hypothesis import Verbosity

from .. import common  # noqa E402
from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests


VERBOSITY: Verbosity = Verbosity.verbose


@optests.generate_opcheck_tests(fast=True)
class LinearizeCacheIndicesTest(unittest.TestCase):
    def execute_linearize_cache_indices_ref(
        self,
        hash_size_cumsum: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        B_offsets: Optional[torch.Tensor] = None,
        max_B: int = -1,
    ) -> torch.Tensor:
        T = hash_size_cumsum.numel() - 1
        B = 0
        B_offsets_ = None
        if B_offsets is not None:
            assert max_B > 0, "Invalid max_B"
            B_offsets_ = B_offsets.cpu().tolist()
            use_vbe = True
        else:
            B = (offsets.numel() - 1) // T
            use_vbe = False
        # Move offsets to CPU
        offsets_ = offsets.cpu().tolist()
        # Sentinel value
        max_offset = hash_size_cumsum[-1].to(indices.dtype)
        # Output
        linear_cache_indices = indices.detach().clone()
        for t in range(T):
            hash_size_offset = hash_size_cumsum[t]
            # Get slicing indices
            if use_vbe:
                assert B_offsets_ is not None, "B_offsets cannot be None"
                indices_start = offsets_[B_offsets_[t]]
                indices_end = offsets_[B_offsets_[t + 1]]
            else:
                indices_start = offsets_[t * B]
                indices_end = offsets_[(t + 1) * B]
            if hash_size_offset >= 0:
                # Add hash size offset if the table is on cache
                linear_cache_indices[indices_start:indices_end] += hash_size_offset
            else:
                # Set indices of the table that is not on cache to max_offset
                linear_cache_indices[indices_start:indices_end] = max_offset
        # Overwrite pruned indices with max_offset
        pruned_pos = (indices < 0).nonzero(as_tuple=True)
        if len(pruned_pos) > 0:
            linear_cache_indices[pruned_pos] = max_offset
        return linear_cache_indices

    @unittest.skipIf(*gpu_unavailable)
    def test_linearize_cache_indices(self) -> None:
        indices = torch.tensor(
            [10, 2, 3, 7, 1, 4, 5, 9, 2, 7, 6, 8, 5, 1, 0, 4],
            dtype=torch.int,
            device="cuda",
        )
        pruned_indices = torch.tensor(
            [10, -1, 3, 7, 1, 4, -1, 9, 2, -1, 6, 8, 5, 1, -1, 4],
            dtype=torch.int,
            device="cuda",
        )
        equal_offsets = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int, device="cuda")
        varying_offsets = torch.tensor(
            [0, 1, 3, 6, 8, 10, 14, 15, 16], dtype=torch.int, device="cuda"
        )

        test_args = [
            # Testing equal sized tables
            ([0, 12, 24, 36, 48], indices, equal_offsets),
            # Testing partially cached tables
            ([0, 12, -1, 24, 36], indices, equal_offsets),
            # Testing batched with varying pooling factor
            ([0, 12, -1, 24, 36], indices, varying_offsets),
            # Testing when multiple features share the same table
            ([0, 0, 12, 12, 24], indices, varying_offsets),
            # Testing equal sized tables + pruned indices
            ([0, 12, 24, 36, 48], pruned_indices, equal_offsets),
            # Testing batched with varying pooling factor + pruned indices
            ([0, 12, -1, 24, 36], pruned_indices, varying_offsets),
        ]

        for hash_size_cumsum_list, indices, offsets in test_args:
            for test_vbe in [False, True]:
                B_offsets = None
                max_B = -1
                if test_vbe:
                    T = len(hash_size_cumsum_list) - 1
                    assert T >= 2, "Require at least two features for testing VBE"
                    B = (offsets.numel() - 1) // T
                    if B <= 1:
                        continue
                    Bs = [B] * T
                    # Randomize two features to alter the batch sizes
                    tables = torch.randperm(T)[0:2].tolist()
                    # Difference
                    diff = 0
                    while diff == 0:
                        diff = random.randint(0, B - 1)
                    # Adjust batch sizes
                    Bs[tables[0]] -= diff
                    Bs[tables[1]] += diff
                    # Compute VBE metadata
                    max_B = max(Bs)
                    B_offsets = torch.tensor(
                        [0] + Bs, dtype=torch.int, device="cuda"
                    ).cumsum(0)

                hash_size_cumsum = torch.tensor(hash_size_cumsum_list).cuda()
                args = (hash_size_cumsum, indices, offsets, B_offsets, max_B)
                output_test = torch.ops.fbgemm.linearize_cache_indices(*args)
                output_ref = self.execute_linearize_cache_indices_ref(*args)
                self.assertTrue(torch.equal(output_test, output_ref))

    @unittest.skipIf(*gpu_unavailable)
    def test_linearize_cache_indices_from_row_idx(self) -> None:
        update_row_indices = torch.tensor(
            [10, 2, 3, 7, 1, 4, 5, 9, 2, 7, 6, 8, 5, 1, 0, 4],
            dtype=torch.int,
            device="cuda",
        )
        update_table_indices = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            dtype=torch.int,
            device="cuda",
        )
        varying_update_table_indices = torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3],
            dtype=torch.int,
            device="cuda",
        )

        # Testing equal sized tables.
        cache_hash_size_cumsum_0 = torch.tensor([0, 12, 24, 36, 48]).cuda()
        linear_cache_indices_0 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_0,
            update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_0.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 26, 31, 30, 32, 41, 37, 36, 40],
                    dtype=torch.int,
                ),
            )
        )

        # Testing partially cached tables.
        cache_hash_size_cumsum_1 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_1 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_1,
            update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_1.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 36, 36, 36, 36, 29, 25, 24, 28],
                    dtype=torch.int,
                ),
            )
        )

        # Testing batched with varying pooling factor.
        cache_hash_size_cumsum_2 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_2 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_2,
            varying_update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_2.cpu(),
                torch.tensor(
                    [10, 2, 3, 19, 13, 16, 17, 21, 36, 36, 36, 36, 36, 36, 24, 28],
                    dtype=torch.int,
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
