#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch

from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)

from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


class StorePrefetchedTensorsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_get_prefetched_info(self) -> None:
        hash_zch_identities = torch.tensor(
            [
                [3350213393928437575],  # for index 54
                [6548733451892409412],  # for index 27
                [4126118985661274454],  # for index 43
                [2565973416302224539],  # for index 90
            ],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        total_cache_hash_size = 100
        linear_cache_indices_merged = torch.tensor(
            [54, 27, 43, 90],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

        prefetched_info = SplitTableBatchedEmbeddingBagsCodegen._get_prefetched_info(
            linear_cache_indices_merged,
            total_cache_hash_size,
            hash_zch_identities,
        )

        self.assertEqual(
            [27, 43, 54, 90],
            prefetched_info.linear_unique_indices.tolist(),
        )
        self.assertEqual(
            prefetched_info.linear_unique_indices_length[0].item(),
            4,
        )
        assert prefetched_info.hash_zch_identities is not None
        self.assertEqual(
            prefetched_info.hash_zch_identities.shape[0],
            4,
        )
        self.assertEqual(
            [
                [6548733451892409412],
                [4126118985661274454],
                [3350213393928437575],
                [2565973416302224539],
            ],
            prefetched_info.hash_zch_identities.tolist(),
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_get_prefetched_info_with_duplicate_hash_zch_identities(self) -> None:
        """
        Test that duplicate cache indices are correctly deduplicated.
        When the same cache index appears multiple times with the same identity,
        only the first occurrence should be kept in the output.
        """
        hash_zch_identities = torch.tensor(
            [
                [3350213393928437575],  # for index 54 (first occurrence)
                [6548733451892409412],  # for index 27
                [3350213393928437575],  # for index 54 (duplicate - same identity)
                [4126118985661274454],  # for index 43
                [6548733451892409412],  # for index 27 (duplicate - same identity)
                [3350213393928437575],  # for index 54 (duplicate - same identity)
                [2565973416302224539],  # for index 90
            ],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        total_cache_hash_size = 100
        linear_cache_indices_merged = torch.tensor(
            [54, 27, 54, 43, 27, 54, 90],  # Duplicates: 54 appears 3x, 27 appears 2x
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

        prefetched_info = SplitTableBatchedEmbeddingBagsCodegen._get_prefetched_info(
            linear_cache_indices_merged,
            total_cache_hash_size,
            hash_zch_identities,
        )

        self.assertEqual(
            [27, 43, 54, 90],
            prefetched_info.linear_unique_indices.tolist(),
        )
        self.assertEqual(
            prefetched_info.linear_unique_indices_length[0].item(),
            4,
        )
        assert prefetched_info.hash_zch_identities is not None
        self.assertEqual(
            prefetched_info.hash_zch_identities.shape[0],
            4,
        )
        self.assertEqual(
            [
                [6548733451892409412],  # for index 27
                [4126118985661274454],  # for index 43
                [3350213393928437575],  # for index 54
                [2565973416302224539],  # for index 90
            ],
            prefetched_info.hash_zch_identities.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
