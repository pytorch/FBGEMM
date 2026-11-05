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
- EnrichmentPolicy.validate() behavior
- KVZCHParams.validate() behavior
- BackendType.from_str() behavior
- Field order of the config NamedTuples
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

    def test_opentab_id_list_type_matches_native_api(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentType

        self.assertEqual(EnrichmentType.ONEFLOW_OPENTAB_ID_LIST.value, 6)


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


class EnrichmentPolicyValidateTest(unittest.TestCase):
    """Test EnrichmentPolicy.validate() logic for the Laser key suffixes."""

    def test_laser_key_suffix_validates(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy

        policy = EnrichmentPolicy(enrichment_dim=128, laser_key_suffix=",v1")
        policy.validate()  # Should not raise

    def test_both_suffix_fields_fail(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy

        policy = EnrichmentPolicy(
            enrichment_dim=256,
            laser_key_suffix=",v1",
            laser_key_suffixes=(",v1", ",v2"),
        )
        with self.assertRaises(AssertionError):
            policy.validate()

    def test_single_suffix_validates(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy, EnrichmentType

        policy = EnrichmentPolicy(
            enrichment_type=EnrichmentType.IGR_LASER_SID,
            enrichment_dim=127,
            laser_key_suffixes=(",v1",),
        )
        policy.validate()  # Should not raise

    def test_multiple_suffixes_split_dim_validates(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy

        policy = EnrichmentPolicy(
            enrichment_dim=256,
            laser_key_suffixes=(",v1", ",v2"),
        )
        policy.validate()  # Should not raise

    def test_multiple_suffixes_uneven_dim_fails(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy

        policy = EnrichmentPolicy(
            enrichment_dim=128,
            laser_key_suffixes=(",v1", ",v2", ",v3"),
        )
        with self.assertRaises(AssertionError):
            policy.validate()

    def test_multiple_suffixes_require_laser_embedding(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy, EnrichmentType

        policy = EnrichmentPolicy(
            enrichment_type=EnrichmentType.IGR_LASER_SID,
            enrichment_dim=256,
            laser_key_suffixes=(",v1", ",v2"),
        )
        with self.assertRaises(AssertionError):
            policy.validate()

    def test_repr_leaves_out_unset_serving_qdq_fields(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy

        policy = EnrichmentPolicy(provider_name="p", enrichment_dim=8)

        self.assertNotIn("serving_qdq_", repr(policy))
        self.assertTrue(repr(policy).startswith("EnrichmentPolicy(enrichment_type="))
        self.assertTrue(repr(policy).endswith(", owner_cache_gather_reserved_sms=0)"))
        self.assertEqual(str(policy), repr(policy))

    def test_repr_keeps_set_serving_qdq_fields(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy

        policy = EnrichmentPolicy(serving_qdq_bit_rate=4)

        self.assertTrue(repr(policy).endswith(", serving_qdq_bit_rate=4)"))


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

    def test_direct_enrichment_owner_cache_config_validates(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy, KVZCHParams

        params = KVZCHParams(
            enrichment_policy=EnrichmentPolicy(
                direct_backend_gather=True,
                owner_cache_size_mb=1024,
                owner_cache_read_populate=True,
                owner_cache_read_populate_max_lookups=8,
            )
        )
        params.validate()

    def test_owner_cache_requires_direct_backend_gather(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy, KVZCHParams

        params = KVZCHParams(
            enrichment_policy=EnrichmentPolicy(owner_cache_size_mb=1024)
        )
        with self.assertRaises(AssertionError):
            params.validate()

    def test_bounded_population_requires_read_population(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy, KVZCHParams

        params = KVZCHParams(
            enrichment_policy=EnrichmentPolicy(
                direct_backend_gather=True,
                owner_cache_size_mb=1024,
                owner_cache_read_populate_max_lookups=8,
            )
        )
        with self.assertRaises(AssertionError):
            params.validate()

    def test_owner_cache_int8_requires_owner_cache(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy, KVZCHParams

        KVZCHParams(
            enrichment_policy=EnrichmentPolicy(
                direct_backend_gather=True,
                owner_cache_size_mb=1024,
                owner_cache_int8=True,
                owner_cache_int8_fused_float=True,
            )
        ).validate()
        for policy in (
            EnrichmentPolicy(direct_backend_gather=True, owner_cache_int8=True),
            EnrichmentPolicy(
                direct_backend_gather=True,
                owner_cache_size_mb=1024,
                owner_cache_int8_fused_float=True,
            ),
        ):
            with self.assertRaises(AssertionError):
                KVZCHParams(enrichment_policy=policy).validate()

    def test_owner_cache_gather_reserved_sms_requires_owner_cache(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy, KVZCHParams

        KVZCHParams(
            enrichment_policy=EnrichmentPolicy(
                direct_backend_gather=True,
                owner_cache_size_mb=1024,
                owner_cache_gather_reserved_sms=32,
            )
        ).validate()
        for policy in (
            EnrichmentPolicy(
                direct_backend_gather=True, owner_cache_gather_reserved_sms=32
            ),
            EnrichmentPolicy(
                direct_backend_gather=True,
                owner_cache_size_mb=1024,
                owner_cache_gather_reserved_sms=-1,
            ),
        ):
            with self.assertRaises(AssertionError):
                KVZCHParams(enrichment_policy=policy).validate()

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

    def test_validates_enrichment_policy(self) -> None:
        from fbgemm_gpu.tbe.ssd import EnrichmentPolicy, KVZCHParams

        params = KVZCHParams(
            bucket_offsets=[],
            bucket_sizes=[],
            enrichment_policy=EnrichmentPolicy(
                enrichment_dim=128,
                laser_key_suffixes=(",v1", ",v2", ",v3"),
            ),
        )
        with self.assertRaises(AssertionError):
            params.validate()


# Field order before rank-coordinated eviction. NamedTuples are constructed and
# pickled by position, so these must stay a prefix of the current fields.
_EVICTION_POLICY_FIELDS_V1: tuple[str, ...] = (
    "eviction_trigger_mode",
    "eviction_strategy",
    "eviction_step_intervals",
    "eviction_mem_threshold_gb",
    "counter_thresholds",
    "ttls_in_mins",
    "counter_decay_rates",
    "feature_score_counter_decay_rates",
    "training_id_eviction_trigger_count",
    "training_id_keep_count",
    "l2_weight_thresholds",
    "threshold_calculation_bucket_stride",
    "threshold_calculation_bucket_num",
    "interval_for_insufficient_eviction_s",
    "interval_for_sufficient_eviction_s",
    "interval_for_feature_statistics_decay_s",
    "meta_header_lens",
    "eviction_free_mem_threshold_gb",
    "eviction_free_mem_check_interval_batch",
    "enable_eviction_for_feature_score_eviction_policy",
    "enable_ssd_writeback",
)
_KVZCH_TBE_CONFIG_FIELDS_V1: tuple[str, ...] = (
    "kvzch_eviction_trigger_mode",
    "eviction_free_mem_threshold_gb",
    "eviction_free_mem_check_interval_batch",
    "threshold_calculation_bucket_stride",
    "threshold_calculation_bucket_num",
    "load_ckpt_without_opt",
    "optimizer_type_for_st",
    "optimizer_state_dtypes_for_st",
    "enrichment_policy",
)


class ConfigFieldOrderTest(unittest.TestCase):
    """New fields are appended, so values stored by position keep their field."""

    def test_earlier_fields_keep_their_positions(self) -> None:
        from fbgemm_gpu.tbe.ssd import EvictionPolicy, KVZCHTBEConfig

        for cls, v1 in (
            (EvictionPolicy, _EVICTION_POLICY_FIELDS_V1),
            (KVZCHTBEConfig, _KVZCH_TBE_CONFIG_FIELDS_V1),
        ):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls._fields[: len(v1)], v1)

    def test_a_v1_positional_value_tuple_loads_with_new_fields_defaulted(
        self,
    ) -> None:
        from fbgemm_gpu.tbe.ssd import KVZCHTBEConfig

        v1_defaults = tuple(
            KVZCHTBEConfig._field_defaults[f] for f in _KVZCH_TBE_CONFIG_FIELDS_V1
        )
        # Unpickling a v1 instance constructs it from its values by position.
        loaded = KVZCHTBEConfig(*v1_defaults)
        self.assertEqual(loaded.threshold_calculation_bucket_stride, 0.2)
        self.assertEqual(loaded.threshold_calculation_bucket_num, 1000000)
        self.assertIsNone(loaded.eviction_insufficient_backoff_batch)
        self.assertIsNone(loaded.eviction_free_mem_floor_gb)


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
