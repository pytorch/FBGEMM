#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unit tests for fbgemm_gpu.tbe.cache.cache_config

Tests cover:
- CacheState.construct() parity with legacy construct_cache_state()
- JIT integration (TBE with MANAGED_CACHING must still JIT-script)
"""

import unittest

import torch


class CacheStateConstructTest(unittest.TestCase):
    """Test CacheState.construct() classmethod — must match legacy function exactly."""

    def test_matches_legacy_construct_cache_state(self) -> None:
        """CRITICAL: New classmethod must produce identical results to legacy function."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            construct_cache_state,
            EmbeddingLocation as LegacyEmbeddingLocation,
        )
        from fbgemm_gpu.tbe.cache import CacheState
        from fbgemm_gpu.tbe.config import EmbeddingLocation

        row_list = [500, 1000, 750, 250, 2000]
        location_list: list[EmbeddingLocation] = [
            EmbeddingLocation.MANAGED_CACHING,
            EmbeddingLocation.MANAGED_CACHING,
            EmbeddingLocation.DEVICE,
            EmbeddingLocation.MANAGED_CACHING,
            EmbeddingLocation.HOST,
        ]
        legacy_location_list: list[LegacyEmbeddingLocation] = [
            LegacyEmbeddingLocation(loc.value) for loc in location_list
        ]
        feature_table_map = [0, 1, 2, 3, 4]

        legacy = construct_cache_state(
            row_list, legacy_location_list, feature_table_map
        )
        new = CacheState.construct(row_list, location_list, feature_table_map)

        self.assertEqual(legacy.total_cache_hash_size, new.total_cache_hash_size)
        self.assertEqual(legacy.cache_hash_size_cumsum, new.cache_hash_size_cumsum)
        self.assertEqual(legacy.cache_index_table_map, new.cache_index_table_map)

    def test_matches_legacy_with_feature_sharing(self) -> None:
        """Test with feature_table_map that has multiple features per table."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            construct_cache_state,
            EmbeddingLocation as LegacyEmbeddingLocation,
        )
        from fbgemm_gpu.tbe.cache import CacheState
        from fbgemm_gpu.tbe.config import EmbeddingLocation

        row_list = [100, 200, 300]
        location_list: list[EmbeddingLocation] = [
            EmbeddingLocation.MANAGED_CACHING,
            EmbeddingLocation.DEVICE,
            EmbeddingLocation.MANAGED_CACHING,
        ]
        legacy_location_list: list[LegacyEmbeddingLocation] = [
            LegacyEmbeddingLocation(loc.value) for loc in location_list
        ]
        feature_table_map = [0, 0, 1, 1, 2]

        legacy = construct_cache_state(
            row_list, legacy_location_list, feature_table_map
        )
        new = CacheState.construct(row_list, location_list, feature_table_map)

        self.assertEqual(legacy.total_cache_hash_size, new.total_cache_hash_size)
        self.assertEqual(legacy.cache_hash_size_cumsum, new.cache_hash_size_cumsum)
        self.assertEqual(legacy.cache_index_table_map, new.cache_index_table_map)


class CacheConfigJITTest(unittest.TestCase):
    """Verify cache types don't break JIT scripting of TBE with caching."""

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `not torch.cuda.is_available()` to decorator factory `unittest.skipIf`.
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_tbe_managed_caching_jit_script(self) -> None:
        """TBE with MANAGED_CACHING must still JIT-script correctly."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            CacheAlgorithm,
            ComputeDevice,
            EmbeddingLocation,
            SplitTableBatchedEmbeddingBagsCodegen,
        )

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (1000, 64, EmbeddingLocation.MANAGED_CACHING, ComputeDevice.CUDA),
            ],
            cache_algorithm=CacheAlgorithm.LRU,
            cache_load_factor=0.5,
        )
        cc_scripted = torch.jit.script(cc)

        indices = torch.randint(
            0, 1000, (20,), device=torch.accelerator.current_accelerator()
        )
        offsets = torch.tensor(
            [0, 10, 20],
            device=torch.accelerator.current_accelerator(),
            dtype=torch.long,
        )
        output = cc_scripted(indices, offsets)
        self.assertEqual(output.shape, (2, 64))
