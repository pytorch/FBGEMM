# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import fbgemm_gpu  # noqa: F401
import torch


class BlockBucketizeCapAmdTest(unittest.TestCase):
    def test_default_path_covers_tail_after_grid_cap(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")

        device = torch.accelerator.current_accelerator()
        lengths_size = 1 << 27
        sentinel_rows = torch.tensor(
            [0, lengths_size // 2, lengths_size - 1], device=device
        )
        lengths = torch.zeros(lengths_size, dtype=torch.int, device=device)
        lengths[sentinel_rows] = 1
        indices = torch.tensor([0, 1, 2], dtype=torch.int, device=device)
        block_sizes = torch.tensor([4], dtype=torch.int, device=device)

        new_lengths, *_ = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths,
            indices,
            False,
            False,
            block_sizes,
            1,
        )

        torch.testing.assert_close(
            new_lengths[sentinel_rows],
            torch.ones(3, dtype=torch.int, device=device),
            rtol=0,
            atol=0,
        )

    def test_2d_weights_default_path_covers_tail_after_grid_cap(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")

        device = torch.accelerator.current_accelerator()
        lengths_size = 1 << 27
        sentinel_rows = torch.tensor(
            [0, lengths_size // 2, lengths_size - 1], device=device
        )
        lengths = torch.zeros(lengths_size, dtype=torch.int, device=device)
        lengths[sentinel_rows] = 1
        indices = torch.tensor([0, 1, 2], dtype=torch.int, device=device)
        block_sizes = torch.tensor([4], dtype=torch.int, device=device)
        weights = torch.tensor([[1.0], [2.0], [3.0]], device=device)

        new_lengths, *_ = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
            lengths,
            indices,
            False,
            False,
            block_sizes,
            1,
            weights,
            1,
        )

        torch.testing.assert_close(
            new_lengths[sentinel_rows],
            torch.ones(3, dtype=torch.int, device=device),
            rtol=0,
            atol=0,
        )
