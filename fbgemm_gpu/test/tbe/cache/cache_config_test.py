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
- JIT integration (TBE with MANAGED_CACHING must still JIT-script)
"""

import unittest

import torch


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
