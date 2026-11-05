#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Trigger-domain isolation against *real* DRAM TBEs and the real C++ backend.

`eviction_coordinator_test.py` covers the grouping logic with fakes, and
`eviction_coordinator_distributed_test.py` covers the cross-rank vote. Neither
exercises the real `should_trigger_evict` path, so neither would catch a mode
whose predicate or reading is wired up wrongly in C++. These tests build two
DRAM TBEs on one process group and assert that each domain's demand stays inside
that domain. A model runs a single trigger mode, so the case that matters is two
same-mode TBEs with different per-table budgets.
"""

import tempfile
import unittest
from unittest.mock import patch

import fbgemm_gpu
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.ssd.eviction_coordinator import EvictionCoordinator
from fbgemm_gpu.tbe.ssd.ssd_config import EvictionPolicy
from fbgemm_gpu.tbe.ssd.training import BackendType, KVZCHParams

# This target does not ship test/tbe/common.py, so guard on cuda directly
# rather than importing gpu_unavailable from there.
_NO_GPU: bool = not torch.cuda.is_available()
# OSS builds do not register the DRAM KV torch classes these TBEs construct.
_OPEN_SOURCE: bool = bool(getattr(fbgemm_gpu, "open_source", False))

_NUM_TABLES = 2
_DIM = 16
_ROWS = 1000
# BY_FEATURE_SCORE, which is the strategy every KVZCH production table uses.
_FEATURE_SCORE_STRATEGY = 5


def _eviction_policy(
    trigger_mode: int,
    *,
    trigger_count: int = 0,
    free_mem_threshold_gb: int = 1,
    interval_for_sufficient_eviction_s: int = 60,
    free_mem_floor_gb: int = 0,
) -> EvictionPolicy:
    return EvictionPolicy(
        eviction_trigger_mode=trigger_mode,
        eviction_strategy=_FEATURE_SCORE_STRATEGY,
        eviction_free_mem_threshold_gb=free_mem_threshold_gb,
        eviction_check_interval_batch=1,
        training_id_eviction_trigger_count=[trigger_count] * _NUM_TABLES,
        training_id_keep_count=[max(trigger_count // 2, 1)] * _NUM_TABLES,
        feature_score_counter_decay_rates=[0.99] * _NUM_TABLES,
        ttls_in_mins=[0] * _NUM_TABLES,
        enable_eviction_for_feature_score_eviction_policy=[True] * _NUM_TABLES,
        interval_for_sufficient_eviction_s=interval_for_sufficient_eviction_s,
        # Off by default so that the isolation tests do not depend on this
        # host's free memory.
        eviction_free_mem_floor_gb=free_mem_floor_gb,
    )


@unittest.skipIf(_NO_GPU, "requires a GPU")
@unittest.skipIf(_OPEN_SOURCE, "DRAM KV backend is not built in OSS")
class EvictionDomainTBETest(unittest.TestCase):
    def _make_tbe(
        self, trigger_mode: int, **kwargs: int
    ) -> SSDTableBatchedEmbeddingBags:
        return SSDTableBatchedEmbeddingBags(
            embedding_specs=[(_ROWS, _DIM)] * _NUM_TABLES,
            feature_table_map=list(range(_NUM_TABLES)),
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=1,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            weights_precision=SparseType.FP32,
            l2_cache_size=0,
            ssd_rocksdb_shards=1,
            backend_type=BackendType.DRAM,
            kv_zch_params=KVZCHParams(
                enable_optimizer_offloading=True,
                # pyre-ignore[6]: EvictionPolicy is the expected type here
                eviction_policy=_eviction_policy(trigger_mode, **kwargs),
            ),
        ).train()

    def _populate(self, tbe: SSDTableBatchedEmbeddingBags, count: int) -> None:
        indices = torch.arange(count, dtype=torch.long)
        # Optimizer offloading packs momentum into the same row, so the backend
        # expects cache_row_dim wide rows rather than _DIM wide ones.
        weights = torch.ones(count, tbe.cache_row_dim, dtype=torch.float)
        tbe.ssd_db.set(indices, weights, torch.tensor([count], dtype=torch.int))

    def test_backend_reports_the_reading_its_mode_judged(self) -> None:
        id_count = self._make_tbe(4, trigger_count=10)
        self._populate(id_count, 64)

        wants, gate_open, reading = id_count.ssd_db.should_trigger_evict(1, 0)

        # ID_COUNT judges on row count, so that is what it must report -- this
        # is the number the e2e drift analysis reads back to measure overshoot.
        self.assertEqual(reading, 64)
        self.assertTrue(wants)
        self.assertTrue(gate_open)

    def test_gate_reopens_when_a_round_finishes(self) -> None:
        # The wall-clock interval only gates uncoordinated rounds; a
        # coordinated round may follow as soon as the previous one is done.
        tbe = self._make_tbe(
            4, trigger_count=10, interval_for_sufficient_eviction_s=3600
        )
        self._populate(tbe, 200)

        tbe.ssd_db.trigger_feature_evict(force=True)
        tbe.ssd_db.wait_until_eviction_done()
        self.assertTrue(tbe.ssd_db.last_eviction_sufficient())
        self._populate(tbe, 200)

        wants, gate_open, _ = tbe.ssd_db.should_trigger_evict(2, 0)
        self.assertTrue(wants)
        self.assertTrue(gate_open)

    def test_free_mem_backend_reports_the_free_memory_it_was_given(self) -> None:
        free_mem = self._make_tbe(5, free_mem_threshold_gb=1)
        self._populate(free_mem, 64)

        one_gb = 1024**3
        below = free_mem.ssd_db.should_trigger_evict(1, one_gb // 2)
        above = free_mem.ssd_db.should_trigger_evict(1, 4 * one_gb)

        self.assertEqual(below[0], True)
        self.assertEqual(below[2], one_gb // 2)
        self.assertEqual(above[0], False)

    def test_over_budget_tbe_does_not_evict_its_under_budget_peer(self) -> None:
        # A model runs a single trigger mode, but its TBEs carry separate
        # per-table budgets. Only the one actually over budget may evict:
        # compute_thresholds_from_buckets() cuts down to keep_count, so pulling
        # the peer in would drop rows it was entitled to keep.
        over_budget = self._make_tbe(4, trigger_count=10)
        under_budget = self._make_tbe(4, trigger_count=1_000_000)
        self._populate(over_budget, 64)
        self._populate(under_budget, 64)

        coordinator = EvictionCoordinator()
        coordinator.register(over_budget, None)
        coordinator.register(under_budget, None)
        coordinator.maybe_trigger(
            step=1,
            pg=None,
            device=torch.device("cpu"),
            free_mem_bytes=over_budget.get_free_cpu_memory_bytes,
        )

        over_budget.ssd_db.wait_until_eviction_done()
        under_budget.ssd_db.wait_until_eviction_done()

        self.assertLess(over_budget.ssd_db.get_keys_in_range(0, _ROWS).numel(), 64)
        self.assertEqual(under_budget.ssd_db.get_keys_in_range(0, _ROWS).numel(), 64)

    def test_free_mem_tbes_evict_together(self) -> None:
        # Free host memory is host-wide and cannot be attributed to a table, so
        # FREE_MEM is the one mode where every TBE shares a decision.
        hungry = self._make_tbe(5, free_mem_threshold_gb=1024 * 1024)
        quiet = self._make_tbe(5, free_mem_threshold_gb=1024 * 1024)
        self._populate(hungry, 64)
        self._populate(quiet, 64)

        coordinator = EvictionCoordinator()
        coordinator.register(hungry, None)
        coordinator.register(quiet, None)
        coordinator.maybe_trigger(
            step=1,
            pg=None,
            device=torch.device("cpu"),
            free_mem_bytes=hungry.get_free_cpu_memory_bytes,
        )

        hungry.ssd_db.wait_until_eviction_done()
        quiet.ssd_db.wait_until_eviction_done()

        self.assertLess(hungry.ssd_db.get_keys_in_range(0, _ROWS).numel(), 64)
        self.assertLess(quiet.ssd_db.get_keys_in_range(0, _ROWS).numel(), 64)

    def test_free_mem_floor_starts_a_round_in_id_count_mode(self) -> None:
        # The table is under its trigger, so ID_COUNT alone would not evict; a
        # floor above this host's free memory starts the round anyway.
        tbe = self._make_tbe(4, trigger_count=100, free_mem_floor_gb=1024 * 1024)
        self._populate(tbe, 64)

        coordinator = EvictionCoordinator()
        coordinator.register(tbe, None)
        coordinator.maybe_trigger(
            step=1,
            pg=None,
            device=torch.device("cpu"),
            free_mem_bytes=tbe.get_free_cpu_memory_bytes,
        )
        tbe.ssd_db.wait_until_eviction_done()

        self.assertLess(tbe.ssd_db.get_keys_in_range(0, _ROWS).numel(), 64)

    def test_only_registered_tbes_in_training_call_the_coordinator(self) -> None:
        # Otherwise a group with no eviction would run a collective every
        # batch, and eval would move the group's agreed check schedule.
        disabled = self._make_tbe(0)
        manual = self._make_tbe(3, trigger_count=10)
        id_count = self._make_tbe(4, trigger_count=10)
        with patch.object(
            SSDTableBatchedEmbeddingBags._eviction_coordinator, "maybe_trigger"
        ) as maybe_trigger:
            disabled.may_trigger_eviction()
            manual.may_trigger_eviction()
            id_count.eval()
            id_count.may_trigger_eviction()
            self.assertEqual(maybe_trigger.call_count, 0)
            id_count.train()
            id_count.may_trigger_eviction()
        self.assertEqual(maybe_trigger.call_count, 1)

    def test_manual_mode_tbe_is_never_registered(self) -> None:
        # MANUAL still builds a real evictor (only DISABLED skips it), so if it
        # were registered it would both veto the gate and be force-evicted.
        manual = self._make_tbe(3, trigger_count=10)
        self.assertFalse(manual.participates_in_coordinated_eviction())
        self.assertFalse(manual.should_check_eviction(1))
