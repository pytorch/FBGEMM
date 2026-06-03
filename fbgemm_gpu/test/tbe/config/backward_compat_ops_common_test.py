#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Test that the re-export shell in split_table_batched_embeddings_ops_common.py
correctly re-exports all symbols from their canonical locations.

This ensures backward compatibility for code that imports from the old location
while allowing new code to use the canonical locations.
"""

import unittest


class BackwardCompatOpsCommonTest(unittest.TestCase):
    """Test backward-compatible re-exports from ops_common.py."""

    def test_all_symbols_importable_from_old_path(self) -> None:
        """Verify all currently-exported symbols remain importable from old path."""
        # This should not raise ImportError
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            # SSD types
            BackendType,
            # Config types
            BoundsCheckMode,
            # Cache types
            CacheAlgorithm,
            CacheState,
            ComputeDevice,
            # Constants
            DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
            EmbeddingLocation,
            EmbeddingSpecInfo,
            EnrichmentPolicy,
            EnrichmentResponseFormat,
            EnrichmentType,
            EvictionPolicy,
            # Utility functions
            get_bounds_check_version_for_platform,
            get_new_embedding_location,
            KVZCHParams,
            KVZCHTBEConfig,
            MAX_PREFETCH_DEPTH,
            MultiPassPrefetchConfig,
            PoolingMode,
            # Other types
            RecordCacheMetrics,
            round_up,
            SplitState,
            tensor_to_device,
        )

        # Verify symbols are not None
        self.assertIsNotNone(BoundsCheckMode)
        self.assertIsNotNone(ComputeDevice)
        self.assertIsNotNone(EmbeddingLocation)
        self.assertIsNotNone(EmbeddingSpecInfo)
        self.assertIsNotNone(PoolingMode)
        self.assertIsNotNone(SplitState)
        self.assertIsNotNone(CacheAlgorithm)
        self.assertIsNotNone(CacheState)
        self.assertIsNotNone(MultiPassPrefetchConfig)
        self.assertIsNotNone(BackendType)
        self.assertIsNotNone(EnrichmentPolicy)
        self.assertIsNotNone(EnrichmentResponseFormat)
        self.assertIsNotNone(EnrichmentType)
        self.assertIsNotNone(EvictionPolicy)
        self.assertIsNotNone(KVZCHParams)
        self.assertIsNotNone(KVZCHTBEConfig)
        self.assertIsNotNone(DEFAULT_SCALE_BIAS_SIZE_IN_BYTES)
        self.assertIsNotNone(MAX_PREFETCH_DEPTH)
        self.assertIsNotNone(get_bounds_check_version_for_platform)
        self.assertIsNotNone(get_new_embedding_location)
        self.assertIsNotNone(round_up)
        self.assertIsNotNone(tensor_to_device)
        self.assertIsNotNone(RecordCacheMetrics)

    def test_type_identity_config_types(self) -> None:
        """Verify config types have identical identity from old and new paths."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            BoundsCheckMode as OldBoundsCheckMode,
            ComputeDevice as OldComputeDevice,
            EmbeddingLocation as OldEmbeddingLocation,
            EmbeddingSpecInfo as OldEmbeddingSpecInfo,
            PoolingMode as OldPoolingMode,
            SplitState as OldSplitState,
        )
        from fbgemm_gpu.tbe.config import (
            BoundsCheckMode,
            ComputeDevice,
            EmbeddingLocation,
            EmbeddingSpecInfo,
            PoolingMode,
            SplitState,
        )

        self.assertIs(OldBoundsCheckMode, BoundsCheckMode)
        self.assertIs(OldComputeDevice, ComputeDevice)
        self.assertIs(OldEmbeddingLocation, EmbeddingLocation)
        self.assertIs(OldEmbeddingSpecInfo, EmbeddingSpecInfo)
        self.assertIs(OldPoolingMode, PoolingMode)
        self.assertIs(OldSplitState, SplitState)

    def test_type_identity_cache_types(self) -> None:
        """Verify cache types have identical identity from old and new paths."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            CacheAlgorithm as OldCacheAlgorithm,
            CacheState as OldCacheState,
            MultiPassPrefetchConfig as OldMultiPassPrefetchConfig,
        )
        from fbgemm_gpu.tbe.cache import (
            CacheAlgorithm,
            CacheState,
            MultiPassPrefetchConfig,
        )

        self.assertIs(OldCacheAlgorithm, CacheAlgorithm)
        self.assertIs(OldCacheState, CacheState)
        self.assertIs(OldMultiPassPrefetchConfig, MultiPassPrefetchConfig)

    def test_type_identity_ssd_types(self) -> None:
        """Verify SSD types have identical identity from old and new paths."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            BackendType as OldBackendType,
            EnrichmentPolicy as OldEnrichmentPolicy,
            EnrichmentResponseFormat as OldEnrichmentResponseFormat,
            EnrichmentType as OldEnrichmentType,
            EvictionPolicy as OldEvictionPolicy,
            KVZCHParams as OldKVZCHParams,
            KVZCHTBEConfig as OldKVZCHTBEConfig,
        )
        from fbgemm_gpu.tbe.ssd import (
            BackendType,
            EnrichmentPolicy,
            EnrichmentResponseFormat,
            EnrichmentType,
            EvictionPolicy,
            KVZCHParams,
            KVZCHTBEConfig,
        )

        self.assertIs(OldBackendType, BackendType)
        self.assertIs(OldEnrichmentPolicy, EnrichmentPolicy)
        self.assertIs(OldEnrichmentResponseFormat, EnrichmentResponseFormat)
        self.assertIs(OldEnrichmentType, EnrichmentType)
        self.assertIs(OldEvictionPolicy, EvictionPolicy)
        self.assertIs(OldKVZCHParams, KVZCHParams)
        self.assertIs(OldKVZCHTBEConfig, KVZCHTBEConfig)

    def test_get_new_embedding_location_works(self) -> None:
        """Verify utility function works from old path."""
        import torch
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            EmbeddingLocation,
            get_new_embedding_location,
        )

        # Test basic functionality
        result = get_new_embedding_location(torch.device("cpu"), 0.0)
        self.assertIsInstance(result, EmbeddingLocation)

    def test_constants_accessible(self) -> None:
        """Verify constants are accessible from old path."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
            MAX_PREFETCH_DEPTH,
        )

        self.assertIsInstance(DEFAULT_SCALE_BIAS_SIZE_IN_BYTES, int)
        self.assertIsInstance(MAX_PREFETCH_DEPTH, int)
        self.assertGreater(DEFAULT_SCALE_BIAS_SIZE_IN_BYTES, 0)
        self.assertGreater(MAX_PREFETCH_DEPTH, 0)

    def test_utility_functions(self) -> None:
        """Verify utility functions work from old path."""
        import torch
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            round_up,
            tensor_to_device,
        )

        # Test round_up
        self.assertEqual(round_up(10, 8), 16)
        self.assertEqual(round_up(16, 8), 16)
        self.assertEqual(round_up(17, 8), 24)

        # Test tensor_to_device (basic functionality)
        tensor = torch.tensor([1, 2, 3])
        result = tensor_to_device(tensor, torch.device("cpu"))
        self.assertIsInstance(result, torch.Tensor)
