#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import unittest

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

        # Testing equal sized tables.
        cache_hash_size_cumsum_0 = torch.tensor([0, 12, 24, 36, 48]).cuda()
        linear_cache_indices_0 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_0, indices, equal_offsets
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
        linear_cache_indices_1 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_1, indices, equal_offsets
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
        linear_cache_indices_2 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_2, indices, varying_offsets
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

        # Testing when multiple features share the same table.
        cache_hash_size_cumsum_3 = torch.tensor([0, 0, 12, 12, 24]).cuda()
        linear_cache_indices_3 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_3, indices, varying_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_3.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 1, 4, 5, 9, 14, 19, 18, 20, 17, 13, 12, 16],
                    dtype=torch.int,
                ),
            )
        )

        # Testing equal sized tables + pruned indices
        cache_hash_size_cumsum_4 = torch.tensor([0, 12, 24, 36, 48]).cuda()
        linear_cache_indices_4 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_4, pruned_indices, equal_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_4.cpu(),
                torch.tensor(
                    [10, 48, 3, 7, 13, 16, 48, 21, 26, 48, 30, 32, 41, 37, 48, 40],
                    dtype=torch.int,
                ),
            )
        )

        # Testing batched with varying pooling factor + pruned indices
        cache_hash_size_cumsum_5 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_5 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_5, pruned_indices, varying_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_5.cpu(),
                torch.tensor(
                    [10, 36, 3, 19, 13, 16, 36, 21, 36, 36, 36, 36, 36, 36, 36, 28],
                    dtype=torch.int,
                ),
            )
        )

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
