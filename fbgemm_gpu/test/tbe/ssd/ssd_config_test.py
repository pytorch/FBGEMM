#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unit tests for fbgemm_gpu.tbe.ssd.ssd_config

Tests cover:
- Import path correctness
- EvictionPolicy.validate() behavior
- KVZCHParams.validate() behavior
- BackendType.from_str() behavior
"""

import unittest


class SsdConfigImportTest(unittest.TestCase):
    """Verify all SSD/KVZCH types importable from tbe.ssd."""

    def test_import_all_ssd_types(self) -> None:
        from fbgemm_gpu.tbe.ssd import (
            BackendType,
            EnrichmentPolicy,
            EnrichmentResponseFormat,
            EnrichmentType,
            EvictionPolicy,
            KVZCHParams,
            KVZCHTBEConfig,
        )

        self.assertIsNotNone(BackendType)
        self.assertIsNotNone(EnrichmentPolicy)
        self.assertIsNotNone(EnrichmentResponseFormat)
        self.assertIsNotNone(EnrichmentType)
        self.assertIsNotNone(EvictionPolicy)
        self.assertIsNotNone(KVZCHParams)
        self.assertIsNotNone(KVZCHTBEConfig)


class EvictionPolicyValidateTest(unittest.TestCase):
    """Test EvictionPolicy.validate() logic."""

    def test_disabled_validates(self) -> None:
        from fbgemm_gpu.tbe.ssd import EvictionPolicy

        ep = EvictionPolicy()  # trigger_mode=0 (disabled)
        ep.validate()  # Should not raise

    def test_iteration_mode_requires_step_intervals(self) -> None:
        from fbgemm_gpu.tbe.ssd import EvictionPolicy

        ep = EvictionPolicy(
            eviction_trigger_mode=1,
            eviction_strategy=0,
            eviction_step_intervals=None,
        )
        with self.assertRaises(AssertionError):
            ep.validate()

    def test_iteration_mode_valid(self) -> None:
        from fbgemm_gpu.tbe.ssd import EvictionPolicy

        ep = EvictionPolicy(
            eviction_trigger_mode=1,
            eviction_strategy=0,
            eviction_step_intervals=100,
            ttls_in_mins=[60],
        )
        ep.validate()  # Should not raise


class KVZCHParamsValidateTest(unittest.TestCase):
    """Test KVZCHParams.validate() logic."""

    def test_valid_params(self) -> None:
        from fbgemm_gpu.tbe.ssd import KVZCHParams

        params = KVZCHParams(
            bucket_offsets=[(0, 128)],
            bucket_sizes=[128],
        )
        params.validate()  # Should not raise

    def test_mismatched_lengths_fails(self) -> None:
        from fbgemm_gpu.tbe.ssd import KVZCHParams

        params = KVZCHParams(
            bucket_offsets=[(0, 128), (128, 256)],
            bucket_sizes=[128],  # Length mismatch
        )
        with self.assertRaises(AssertionError):
            params.validate()

    def test_backend_return_whole_row_requires_optimizer_offloading(self) -> None:
        from fbgemm_gpu.tbe.ssd import KVZCHParams

        params = KVZCHParams(
            bucket_offsets=[],
            bucket_sizes=[],
            backend_return_whole_row=True,
            enable_optimizer_offloading=False,
        )
        with self.assertRaises(AssertionError):
            params.validate()


class BackendTypeFromStrTest(unittest.TestCase):
    """Test BackendType.from_str() behavior."""

    def test_ssd(self) -> None:
        from fbgemm_gpu.tbe.ssd import BackendType

        self.assertEqual(BackendType.from_str("ssd"), BackendType.SSD)

    def test_dram(self) -> None:
        from fbgemm_gpu.tbe.ssd import BackendType

        self.assertEqual(BackendType.from_str("dram"), BackendType.DRAM)

    def test_ps(self) -> None:
        from fbgemm_gpu.tbe.ssd import BackendType

        self.assertEqual(BackendType.from_str("ps"), BackendType.PS)

    def test_dram_ssd(self) -> None:
        from fbgemm_gpu.tbe.ssd import BackendType

        self.assertEqual(BackendType.from_str("dram_ssd"), BackendType.DRAM_SSD)
        self.assertEqual(BackendType.DRAM_SSD.value, 3)

    def test_invalid_raises(self) -> None:
        from fbgemm_gpu.tbe.ssd import BackendType

        with self.assertRaises(ValueError):
            BackendType.from_str("invalid")


class SsdImportCycleRegressionTest(unittest.TestCase):
    """Regression test for the D107684315 import cycle: importing ops_training ->
    ops_common used to eagerly trigger tbe.ssd.__init__ -> .training -> ops_training
    (partial), leaving SSDTableBatchedEmbeddingBags unbound. buck build does not
    execute imports, so only an import-time test catches this.
    """

    def test_heavy_symbols_bound_after_ops_training_import(self) -> None:
        import importlib

        # Importing ops_training first is the exact trigger of the cycle.
        importlib.import_module(
            "fbgemm_gpu.split_table_batched_embeddings_ops_training"
        )
        ssd = importlib.import_module("fbgemm_gpu.tbe.ssd")
        for sym in ("ASSOC", "SSDTableBatchedEmbeddingBags", "DramKvPerfStat"):
            self.assertTrue(
                hasattr(ssd, sym),
                f"fbgemm_gpu.tbe.ssd is missing {sym!r}: the tbe.ssd import cycle "
                "regressed (see D107684315).",
            )

    def test_from_import_heavy_symbols(self) -> None:
        from fbgemm_gpu.tbe.ssd import ASSOC, SSDTableBatchedEmbeddingBags  # noqa: F401

        self.assertIsNotNone(ASSOC)
        self.assertIsNotNone(SSDTableBatchedEmbeddingBags)

    def test_ops_common_lazy_ssd_reexports_resolve(self) -> None:
        # The lazy __getattr__ in ops_common must still resolve the SSD config
        # types (legacy import path + old-pickle BC).
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            BackendType,
            EvictionPolicy,
            KVZCHParams,
            KVZCHTBEConfig,
        )

        self.assertIsNotNone(BackendType)
        self.assertIsNotNone(EvictionPolicy)
        self.assertIsNotNone(KVZCHParams)
        self.assertIsNotNone(KVZCHTBEConfig)
