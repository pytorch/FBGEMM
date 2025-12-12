#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch

from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    ComputeDevice,
    EmbeddingLocation,
)
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
        # All indices same as cache indices (all tables cached)
        linear_indices = torch.tensor(
            [54, 27, 43, 90],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

        prefetched_info = SplitTableBatchedEmbeddingBagsCodegen._get_prefetched_info(
            linear_indices,
            linear_cache_indices_merged,
            total_cache_hash_size,
            hash_zch_identities,
            max_indices_length=200,  # Arbitrary large enough value
        )

        self.assertEqual(
            [27, 43, 54, 90],
            prefetched_info.linear_unique_indices.tolist(),
        )
        self.assertEqual(
            [27, 43, 54, 90],
            prefetched_info.linear_unique_cache_indices.tolist(),
        )
        self.assertEqual(
            prefetched_info.linear_unique_indices_length[0].item(),
            4,
        )
        self.assertIsNotNone(prefetched_info.hash_zch_identities)
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
        # All indices same as cache indices in this test (all tables cached)
        linear_indices = torch.tensor(
            [54, 27, 54, 43, 27, 54, 90],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

        prefetched_info = SplitTableBatchedEmbeddingBagsCodegen._get_prefetched_info(
            linear_indices,
            linear_cache_indices_merged,
            total_cache_hash_size,
            hash_zch_identities,
            max_indices_length=200,  # Arbitrary large enough value
        )

        linear_unique_indices_length_scalar = (
            prefetched_info.linear_unique_indices_length[0].item()
        )
        # Verify count matches deduplicated cache indices
        self.assertEqual(
            linear_unique_indices_length_scalar,
            4,
        )
        self.assertEqual(
            prefetched_info.linear_unique_indices.shape[0],
            prefetched_info.linear_unique_cache_indices.shape[0],
        )
        # linear_unique_cache_indices should be deduplicated and sorted
        self.assertEqual(
            [27, 43, 54, 90],
            prefetched_info.linear_unique_cache_indices.tolist()[
                :linear_unique_indices_length_scalar
            ],
        )
        self.assertIsNotNone(prefetched_info.hash_zch_identities)
        self.assertEqual(
            prefetched_info.hash_zch_identities.shape[0],
            prefetched_info.linear_unique_cache_indices.shape[0],
        )
        self.assertEqual(
            [
                [6548733451892409412],  # for index 27
                [4126118985661274454],  # for index 43
                [3350213393928437575],  # for index 54
                [2565973416302224539],  # for index 90
            ],
            prefetched_info.hash_zch_identities.tolist()[
                :linear_unique_indices_length_scalar
            ],
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_get_prefetched_info_mixed_cache_non_cache(self) -> None:
        """
        Test the scenario where some tables have UVM cache enabled and some don't.
        This is the main scenario the commit is designed to support.

        In this test:
        - total_cache_hash_size = 100 means indices < 100 are cached
        - Indices >= 100 are non-cached
        - _store_prefetched_tensors masks non-cached indices:
          * linear_cache_indices_merged: non-cached -> total_cache_hash_size (100)
          * linearize_indices: non-cached -> -1
        - This test simulates the inputs AFTER that masking step
        """
        total_cache_hash_size = 100  # Indices < 100 are cached

        # Original state BEFORE masking:
        # - TBE indices: [54, 27, 150, 43, 200, 90, 175]
        # - Cache indices (pre-normalized): [54, 27, 100, 43, 100, 90, 100]
        #   where non-cached (150, 200, 175) are already normalized to 100
        # - final_lxu_cache_locations: [1, 1, -1, 1, -1, 1, -1]
        #   where -1 indicates non-cached

        # After masking (as done in _store_prefetched_tensors):
        linear_cache_indices_merged = torch.tensor(
            [
                54,
                27,
                100,
                43,
                100,
                90,
                100,
            ],  # Non-cached slots set to total_cache_hash_size
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

        linear_indices = torch.tensor(
            [54, 27, -1, 43, -1, 90, -1],  # Non-cached slots set to -1
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

        # Identities for ALL indices (including non-cached)
        hash_zch_identities = torch.tensor(
            [
                [1111111111111111111],  # for cached index 54
                [2222222222222222222],  # for cached index 27
                [3333333333333333333],  # for non-cached index 150
                [4444444444444444444],  # for cached index 43
                [5555555555555555555],  # for non-cached index 200
                [6666666666666666666],  # for cached index 90
                [7777777777777777777],  # for non-cached index 175
            ],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

        prefetched_info = SplitTableBatchedEmbeddingBagsCodegen._get_prefetched_info(
            linear_indices=linear_indices,
            linear_cache_indices_merged=linear_cache_indices_merged,
            total_cache_hash_size=total_cache_hash_size,
            hash_zch_identities=hash_zch_identities,
            max_indices_length=200,  # Arbitrary large enough value
        )

        # linear_unique_indices should contain sorted unique cached indices
        # The function processes ALL indices including -1 placeholders
        # Only indices < total_cache_hash_size (100) are considered valid cached indices
        linear_unique_indices_length_scalar = (
            prefetched_info.linear_unique_indices_length[0].item()
        )
        self.assertEqual(linear_unique_indices_length_scalar, 5)

        self.assertEqual(
            [27, 43, 54, 90, -1],
            prefetched_info.linear_unique_indices.tolist()[
                :linear_unique_indices_length_scalar
            ],
        )

        self.assertEqual(
            [27, 43, 54, 90, total_cache_hash_size],
            prefetched_info.linear_unique_cache_indices.tolist()[
                :linear_unique_indices_length_scalar
            ],
        )

        # Identities should match the cached indices (excluding -1 placeholders)
        self.assertIsNotNone(prefetched_info.hash_zch_identities)
        self.assertEqual(
            [
                [2222222222222222222],  # for index 27
                [4444444444444444444],  # for index 43
                [1111111111111111111],  # for index 54
                [6666666666666666666],  # for index 90
            ],
            prefetched_info.hash_zch_identities.tolist()[
                : linear_unique_indices_length_scalar - 1
            ],
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_get_prefetched_info_without_identities(self) -> None:
        """
        Test the case where hash_zch_identities is None.
        This can happen when identity tracking is disabled.
        """
        total_cache_hash_size = 100
        linear_cache_indices_merged = torch.tensor(
            [54, 27, 43, 90],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        linear_indices = torch.tensor(
            [54, 27, 43, 90],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

        prefetched_info = SplitTableBatchedEmbeddingBagsCodegen._get_prefetched_info(
            linear_indices=linear_indices,
            linear_cache_indices_merged=linear_cache_indices_merged,
            total_cache_hash_size=total_cache_hash_size,
            hash_zch_identities=None,
            max_indices_length=200,  # Arbitrary large enough value
        )

        self.assertEqual(
            [27, 43, 54, 90],
            prefetched_info.linear_unique_indices.tolist(),
        )
        self.assertEqual(
            [27, 43, 54, 90],
            prefetched_info.linear_unique_cache_indices.tolist(),
        )
        self.assertEqual(
            prefetched_info.linear_unique_indices_length[0].item(),
            4,
        )
        # Identities should be None when not provided
        self.assertIsNone(prefetched_info.hash_zch_identities)

    @unittest.skipIf(*gpu_unavailable)
    def test_store_prefetched_tensors_disabled(self) -> None:
        """
        Test that _store_prefetched_tensors returns early when
        enable_raw_embedding_streaming is False.
        """
        tbe = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (100, 16, EmbeddingLocation.MANAGED_CACHING, ComputeDevice.CUDA),
            ],
            enable_raw_embedding_streaming=False,
        )

        indices = torch.tensor(
            [1, 2, 3], device=torch.cuda.current_device(), dtype=torch.int64
        )
        offsets = torch.tensor(
            [0, 3], device=torch.cuda.current_device(), dtype=torch.int64
        )
        linear_cache_indices_merged = torch.tensor(
            [10, 20, 30], device=torch.cuda.current_device(), dtype=torch.int64
        )

        tbe._store_prefetched_tensors(
            indices=indices,
            offsets=offsets,
            vbe_metadata=None,
            linear_cache_indices_merged=linear_cache_indices_merged,
            final_lxu_cache_locations=torch.ones_like(linear_cache_indices_merged) * -1,
            hash_zch_identities=None,
        )

        self.assertEqual(len(tbe.prefetched_info_list), 0)

    @unittest.skipIf(*gpu_unavailable)
    def test_store_prefetched_tensors_filters_non_cached_indices(self) -> None:
        """
        Test that _store_prefetched_tensors correctly filters out indices
        that are >= total_cache_hash_size.

        This test verifies the masking behavior where only indices less than
        total_cache_hash_size are processed and stored.
        """
        # Setup: create a TBE with raw embedding streaming enabled
        # Using a small cache size to test filtering
        tbe = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (100, 16, EmbeddingLocation.MANAGED_CACHING, ComputeDevice.CUDA),
                (100, 16, EmbeddingLocation.DEVICE, ComputeDevice.CUDA),
            ],
            enable_raw_embedding_streaming=True,
        )

        # Execute: provide indices from both tables
        # Table 0 (MANAGED_CACHING): indices 10, 20, 60, 70
        # Table 1 (DEVICE): indices 5, 15 (both non-cached since DEVICE location)
        indices = torch.tensor(
            [10, 20, 60, 70, 5, 15],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        # Offsets indicate: table 0 has 4 indices [0:4], table 1 has 2 indices [4:6]
        offsets = torch.tensor(
            [0, 4, 6], device=torch.cuda.current_device(), dtype=torch.int64
        )
        # linear_cache_indices contains all indices (before filtering)
        # For DEVICE table (table 1), indices are normalized to total_cache_hash_size
        linear_cache_indices_merged = torch.tensor(
            [10, 20, 60, 70, tbe.total_cache_hash_size, tbe.total_cache_hash_size],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        hash_zch_identities = torch.tensor(
            [[1111], [2222], [6666], [7777], [8888], [9999]],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

        tbe._store_prefetched_tensors(
            indices=indices,
            offsets=offsets,
            vbe_metadata=None,
            linear_cache_indices_merged=linear_cache_indices_merged,
            final_lxu_cache_locations=torch.where(
                linear_cache_indices_merged < tbe.total_cache_hash_size,
                torch.ones_like(linear_cache_indices_merged),
                torch.ones_like(linear_cache_indices_merged) * -1,
            ),
            hash_zch_identities=hash_zch_identities,
        )

        # Assert: prefetched_info_list should have one entry
        self.assertEqual(len(tbe.prefetched_info_list), 1)
        prefetched_info = tbe.prefetched_info_list[0]

        linear_unique_indices_length_scalar = (
            prefetched_info.linear_unique_indices_length[0].item()
        )
        self.assertEqual(linear_unique_indices_length_scalar, 5)

        self.assertEqual(
            [10, 20, 60, 70, -1],
            prefetched_info.linear_unique_indices.tolist()[
                :linear_unique_indices_length_scalar
            ],
        )

        self.assertEqual(
            [10, 20, 60, 70, tbe.total_cache_hash_size],
            prefetched_info.linear_unique_cache_indices.tolist()[
                :linear_unique_indices_length_scalar
            ],
        )

        # Identities should match the cached indices (excluding -1 placeholders)
        self.assertIsNotNone(prefetched_info.hash_zch_identities)
        identities = prefetched_info.hash_zch_identities
        self.assertEqual(
            [
                [1111],
                [2222],
                [6666],
                [7777],
            ],
            identities.tolist()[: linear_unique_indices_length_scalar - 1],
        )
        self.assertEqual(identities.shape[0], 6)
        self.assertEqual(prefetched_info.linear_unique_cache_indices.shape[0], 6)
        self.assertEqual(prefetched_info.linear_unique_indices.shape[0], 6)


if __name__ == "__main__":
    unittest.main()
