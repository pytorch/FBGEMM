#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    ComputeDevice,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    RESParams,
    SplitTableBatchedEmbeddingBagsCodegen,
)

from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable

# Table height the fixtures build and the tests do their linear-index
# arithmetic against. One name because the two have to agree.
ROWS = 64


class ResEnabledTablesTest(unittest.TestCase):
    """
    Tests for the ``res_enabled_tables`` allowlist -> per-feature
    ``res_enabled_feature_mask`` that scopes RES streaming to specific tables.

    Requires CUDA: the mask buffer is built on ``current_device`` and
    ``_get_enabled_feature_mask_and_indices`` runs ``searchsorted`` on GPU.
    """

    def _build_tbe(
        self,
        table_names: list[str],
        res_enabled_tables: list[str],
        location: EmbeddingLocation = EmbeddingLocation.DEVICE,
        uvm_host_mapped: bool = False,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        """One table per name (one feature per table), RES enabled."""
        n = len(table_names)
        res_params = RESParams(
            res_store_shards=1,
            table_names=list(table_names),
            table_offsets=[i * ROWS for i in range(n)],
            table_sizes=[ROWS] * n,
            res_enabled_tables=list(res_enabled_tables),
        )
        return SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (ROWS, 16, location, ComputeDevice.CUDA) for _ in range(n)
            ],
            enable_raw_embedding_streaming=True,
            res_params=res_params,
            uvm_host_mapped=uvm_host_mapped,
        )

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_res_count_and_copy_done_start_zero(self) -> None:
        # new_unified_tensor hands back an unzeroed allocation. res_count is
        # read as a row count by a std::copy that does not bound check, and any
        # nonzero copy_done reads as "the GPU is done writing" -- so an unzeroed
        # pair makes the first drain over-read and ship mid-write.
        # Both arguments are load-bearing, and the test is vacuous without
        # either. MANAGED_CACHING, not DEVICE: a DEVICE table has no UVM cache,
        # so cache_size is 0 and _register_res_buffers takes the empty branch,
        # which allocates with plain torch.zeros. uvm_host_mapped, because only
        # that branch of new_unified_tensor allocates with malloc; the default
        # cudaMallocManaged branch hands back fresh zeroed pages, so the
        # unzeroed value is never observed there.
        tbe = self._build_tbe(
            ["t0"],
            ["t0"],
            location=EmbeddingLocation.MANAGED_CACHING,
            uvm_host_mapped=True,
        )
        self.assertGreater(tbe.get_buffer("lxu_cache_weights").size(0), 0)
        self.assertEqual(tbe.get_buffer("res_count").tolist(), [0])
        self.assertEqual(tbe.get_buffer("res_copy_done").tolist(), [0])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_empty_allowlist_enables_all(self) -> None:
        # Empty allowlist => no scoping; preserves the pre-allowlist behavior.
        tbe = self._build_tbe(["t0", "t1", "t2"], res_enabled_tables=[])
        self.assertTrue(tbe._res_all_features_enabled)
        mask = tbe.get_buffer("res_enabled_feature_mask")
        self.assertEqual(mask.tolist(), [True, True, True])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_all_tables_listed_short_circuits(self) -> None:
        # Listing every table is equivalent to the empty/all-enabled fast path.
        tbe = self._build_tbe(["t0", "t1", "t2"], res_enabled_tables=["t0", "t1", "t2"])
        self.assertTrue(tbe._res_all_features_enabled)
        mask = tbe.get_buffer("res_enabled_feature_mask")
        self.assertEqual(mask.tolist(), [True, True, True])

    def _build_mixed_tbe(
        self,
        table_names: list[str],
        locations: list[EmbeddingLocation],
        res_enabled_tables: list[str],
        enable_hbm_streaming: bool = True,
        heights: list[int] | None = None,
        feature_table_map: list[int] | None = None,
        dims: list[int] | None = None,
        res_hbm_drain_interval: int = 1,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        """One table per name, each with its own placement, row count and dim."""
        n = len(table_names)
        rows = heights if heights is not None else [ROWS] * n
        widths = dims if dims is not None else [16] * n
        offsets = []
        running = 0
        for h in rows:
            offsets.append(running)
            running += h
        res_params = RESParams(
            res_store_shards=1,
            table_names=list(table_names),
            table_offsets=offsets,
            table_sizes=list(rows),
            res_enabled_tables=list(res_enabled_tables),
            enable_hbm_streaming=enable_hbm_streaming,
            # Drain every prefetch: these tests assert on what a drain ships,
            # not on when one fires. The cadence tests set their own.
            res_hbm_drain_interval=res_hbm_drain_interval,
        )
        tbe = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (h, w, loc, ComputeDevice.CUDA)
                for h, w, loc in zip(rows, widths, locations)
            ],
            enable_raw_embedding_streaming=True,
            res_params=res_params,
            feature_table_map=feature_table_map,
        )
        return tbe

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_linear_mask_covers_disjoint_device_tables(self) -> None:
        # DEVICE tables need not be contiguous, so the mask cannot be a range.
        # t1 is MANAGED and must be excluded even though it is allowlisted.
        rows = ROWS
        tbe = self._build_mixed_tbe(
            ["t0", "t1", "t2"],
            [
                EmbeddingLocation.DEVICE,
                EmbeddingLocation.MANAGED,
                EmbeddingLocation.DEVICE,
            ],
            res_enabled_tables=["t0", "t1", "t2"],
        )
        mask = tbe.get_buffer("_res_hbm_linear_mask").tolist()
        self.assertEqual(len(mask), 3 * rows + 1)
        self.assertTrue(all(mask[0:rows]))
        self.assertFalse(any(mask[rows : 2 * rows]))
        self.assertTrue(all(mask[2 * rows : 3 * rows]))
        # The pruning sentinel slot is never in the enabled set.
        self.assertFalse(mask[3 * rows])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_linear_mask_uses_table_not_feature_offsets(self) -> None:
        # Non-uniform heights and a non-identity feature_table_map: a mask built
        # from the per-feature cumsum, or from a fixed stride, lands on the
        # wrong rows here even though it would pass a uniform-height test.
        tbe = self._build_mixed_tbe(
            ["t0", "t1", "t2"],
            [
                EmbeddingLocation.DEVICE,
                EmbeddingLocation.MANAGED,
                EmbeddingLocation.DEVICE,
            ],
            res_enabled_tables=["t2"],
            heights=[10, 20, 30],
            feature_table_map=[0, 0, 1, 2],
        )
        mask = tbe.get_buffer("_res_hbm_linear_mask").tolist()
        self.assertEqual(len(mask), 61)
        # t2 occupies rows [30, 60) -- offsets are 0, 10, 30.
        self.assertFalse(any(mask[0:30]))
        self.assertTrue(all(mask[30:60]))
        self.assertFalse(mask[60])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_empty_allowlist_enables_all_tables(self) -> None:
        # Empty means all tables, the same as it does for the UVM-cached lane.
        # Mixed placement on purpose -- with both tables DEVICE an all-True
        # mask would also pass if the placement filter were gone.
        rows = ROWS
        tbe = self._build_mixed_tbe(
            ["t0", "t1"],
            [EmbeddingLocation.DEVICE, EmbeddingLocation.MANAGED],
            res_enabled_tables=[],
        )
        mask = tbe.get_buffer("_res_hbm_linear_mask").tolist()
        self.assertTrue(all(mask[0:rows]))
        self.assertFalse(any(mask[rows:]))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_table_names_must_cover_every_table(self) -> None:
        # Surplus names, not a deficit: with fewer names than tables and a
        # non-empty allowlist `_register_res_enabled_feature_mask` hits
        # IndexError first and the validator never runs.
        with self.assertRaisesRegex(ValueError, "must name every table"):
            self._build_mixed_tbe(
                ["t0", "t1", "t2"],
                [EmbeddingLocation.DEVICE] * 2,
                res_enabled_tables=["t0"],
                heights=[64, 64],
            )

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_linear_mask_excludes_allowlisted_non_device_table(self) -> None:
        # Allowlisting a table the lane cannot serve enables nothing
        # rather than streaming it. This is the state a config produces when it
        # names a UVM table for the HBM lane. t2 is the positive control: the
        # zero over t0 and t1 says nothing without a one in the same mask.
        rows = ROWS
        tbe = self._build_mixed_tbe(
            ["t0", "t1", "t2"],
            [
                EmbeddingLocation.MANAGED,
                EmbeddingLocation.MANAGED_CACHING,
                EmbeddingLocation.DEVICE,
            ],
            res_enabled_tables=["t0", "t1", "t2"],
        )
        mask = tbe.get_buffer("_res_hbm_linear_mask").tolist()
        self.assertFalse(any(mask[: 2 * rows]))
        self.assertTrue(all(mask[2 * rows : 3 * rows]))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_linear_mask_intersects_allowlist(self) -> None:
        # DEVICE-placed is not sufficient: t2 is DEVICE but not allowlisted.
        rows = ROWS
        tbe = self._build_mixed_tbe(
            ["t0", "t1", "t2"],
            [
                EmbeddingLocation.DEVICE,
                EmbeddingLocation.MANAGED,
                EmbeddingLocation.DEVICE,
            ],
            res_enabled_tables=["t0"],
        )
        mask = tbe.get_buffer("_res_hbm_linear_mask").tolist()
        self.assertTrue(all(mask[0:rows]))
        self.assertFalse(any(mask[rows:]))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_buffers_not_allocated_when_lane_disabled(self) -> None:
        # The buffers span the whole linear index space, so a RES model that leaves
        # the lane off must not pay for them.
        tbe = self._build_mixed_tbe(
            ["t0"],
            [EmbeddingLocation.DEVICE],
            res_enabled_tables=[],
            enable_hbm_streaming=False,
        )
        # hasattr, not named_buffers: registering one as a plain attribute
        # instead would keep a named_buffers assertion green while still paying
        # the whole allocation this test exists to prevent.
        self.assertFalse(hasattr(tbe, "_res_rows_seen"))
        self.assertFalse(hasattr(tbe, "_res_hbm_linear_mask"))
        self.assertFalse(hasattr(tbe, "_res_drain_count_cpu"))
        self.assertFalse(hasattr(tbe, "_res_hbm_copy_done"))
        # The staging buffers are the largest thing the lane allocates, and
        # they ratchet to the biggest drain the run has seen.
        self.assertFalse(hasattr(tbe, "_res_hbm_indices_buf"))
        self.assertFalse(hasattr(tbe, "_res_hbm_weights_buf"))
        self.assertFalse(hasattr(tbe, "_res_hbm_count_buf"))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_mark_records_every_touched_row(self) -> None:
        # The mark is unconditional on cache state: t1 is DEVICE-placed and so
        # never hits the cache, and its row must still be recorded. At
        # res_hbm_drain_interval=1 the drain consumes in-mask marks in the same
        # call, so the t1 row is observed in what the next iteration ships while
        # the out-of-mask t0 row is observed in the residue the drain never
        # clears.
        rows = ROWS
        tbe = self._build_mixed_tbe(
            ["t0", "t1"],
            [EmbeddingLocation.MANAGED_CACHING, EmbeddingLocation.DEVICE],
            res_enabled_tables=["t1"],
            heights=[rows, rows],
        )
        device = torch.cuda.current_device()
        tbe._prefetch(
            torch.tensor([3, 5], device=device, dtype=torch.int64),
            torch.tensor([0, 1, 2], device=device, dtype=torch.int64),
        )
        self.assertEqual(self._idle_iteration(tbe), [rows + 5])
        present = tbe.get_buffer("_res_rows_seen").tolist()
        self.assertEqual(len(present), 2 * rows + 1)
        self.assertEqual([i for i, p in enumerate(present) if p], [3])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_prefetch_with_lane_off(self) -> None:
        # The buffer does not exist when the lane is off, so the mark must be
        # gated rather than guarded -- an ungated index_fill_ is AttributeError
        # on every RES model that leaves the lane off. Drives _prefetch, which
        # is where the mark lives; calling _store_prefetched_tensors here would
        # exercise a method the mark is no longer in.
        rows = ROWS
        tbe = self._build_mixed_tbe(
            ["t0", "t1"],
            [EmbeddingLocation.MANAGED_CACHING, EmbeddingLocation.DEVICE],
            res_enabled_tables=["t1"],
            enable_hbm_streaming=False,
            heights=[rows, rows],
        )
        device = torch.cuda.current_device()
        tbe._prefetch(
            torch.tensor([3, 5], device=device, dtype=torch.int64),
            torch.tensor([0, 1, 2], device=device, dtype=torch.int64),
        )
        self.assertFalse(hasattr(tbe, "_res_rows_seen"))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_mark_fires_through_prefetch(self) -> None:
        # Drives _prefetch rather than calling _store_prefetched_tensors by
        # hand: the mark has to survive the cacheless early-return, and a test
        # that invokes the inner method directly cannot see that guard at all.
        # The DEVICE-only TBE is the shape TorchRec actually builds for a
        # DEVICE table, and the only shape this lane exists to serve.
        rows = ROWS
        device = torch.cuda.current_device()
        indices = torch.tensor([3], device=device, dtype=torch.int64)
        offsets = torch.tensor([0, 1], device=device, dtype=torch.int64)

        for location in (
            EmbeddingLocation.MANAGED_CACHING,  # positive control: has a cache
            EmbeddingLocation.DEVICE,  # no cache, so no _prefetch tail
        ):
            with self.subTest(location=location):
                tbe = self._build_mixed_tbe(
                    ["t0"], [location], res_enabled_tables=["t0"], heights=[rows]
                )
                tbe._prefetch(indices, offsets)
                self.assertEqual(
                    [
                        i
                        for i, p in enumerate(tbe.get_buffer("_res_rows_seen").tolist())
                        if p
                    ],
                    [3],
                )

    def _drain_tbe(
        self,
        drain_interval: int = 1,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        """t0 MANAGED_CACHING (out of the lane), t1 DEVICE and allowlisted.

        The cached table is what gives the cache lane a non-empty
        ``total_cache_hash_size``; only t1's rows are in the HBM lane's mask.
        """
        return self._build_mixed_tbe(
            ["t0", "t1"],
            [EmbeddingLocation.MANAGED_CACHING, EmbeddingLocation.DEVICE],
            res_enabled_tables=["t1"],
            heights=[ROWS, ROWS],
            res_hbm_drain_interval=drain_interval,
        )

    def _mark_and_store_one_row(
        self,
        tbe: SplitTableBatchedEmbeddingBagsCodegen,
        t1_row: int,
        rows: int = 64,
    ) -> list[int] | None:
        """One iteration's mark + store for a single t1 row, bypassing
        `_prefetch`; returns drained ids or None.
        """
        device = torch.cuda.current_device()
        tbe._mark_res_hbm_rows(
            torch.tensor([t1_row], device=device, dtype=torch.int64),
            torch.tensor([0, 0, 1], device=device, dtype=torch.int64),
            None,
        )
        tbe._store_prefetched_tensors(
            # Empty bag for feature 0, one index for feature 1.
            indices=torch.tensor([t1_row], device=device, dtype=torch.int64),
            offsets=torch.tensor([0, 0, 1], device=device, dtype=torch.int64),
            vbe_metadata=None,
            linear_cache_indices_merged=torch.tensor(
                [rows], device=device, dtype=torch.int64
            ),
            final_lxu_cache_locations=torch.tensor(
                [-1], device=device, dtype=torch.int32
            ),
            hash_zch_identities=None,
            hash_zch_runtime_meta=None,
        )
        drained = tbe.prefetched_info_list[-1].res_hbm_indices
        return None if drained is None else drained.tolist()

    def _idle_iteration(
        self,
        tbe: SplitTableBatchedEmbeddingBagsCodegen,
    ) -> list[int] | None:
        """Advance one iteration without a lookup, to collect the pending drain.

        The drain queues its row count to the host asynchronously, so what it
        compacted ships on the iteration after it ran, not its own.

        Both this and `_mark_and_store_one_row` mark before storing, mirroring
        `_prefetch`: the mark sits above the cache guard, and
        `_store_prefetched_tensors` does not mark.
        """
        device = torch.cuda.current_device()
        empty = torch.zeros(0, device=device, dtype=torch.int64)
        # Every fixture this serves has two features, and offsets is one longer.
        offsets = torch.zeros(3, device=device, dtype=torch.int64)
        tbe._mark_res_hbm_rows(empty, offsets, None)
        tbe._store_prefetched_tensors(
            indices=empty,
            offsets=offsets,
            vbe_metadata=None,
            linear_cache_indices_merged=empty,
            final_lxu_cache_locations=torch.zeros(0, device=device, dtype=torch.int32),
            hash_zch_identities=None,
            hash_zch_runtime_meta=None,
        )
        drained = tbe.prefetched_info_list[-1].res_hbm_indices
        return None if drained is None else drained.tolist()

    def _prefetch_until_pending(
        self,
        tbe: SplitTableBatchedEmbeddingBagsCodegen,
        t1_row: int,
        rows: int = 64,
    ) -> None:
        """Leave one prefetched_info, carrying `t1_row` for the ship to send.

        The prefetch that marks a row and the one that ships it are different
        iterations, and ``raw_embedding_stream`` pops the oldest entry -- so the
        marking prefetch has to be dropped or the ship would pop an empty one.
        """
        self._mark_and_store_one_row(tbe, t1_row, rows)
        tbe.prefetched_info_list.clear()
        self._idle_iteration(tbe)

    def _device_tbe(
        self, heights: list[int], dims: list[int]
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        """All-DEVICE TBE whose flat weights are filled with arange."""
        names = [f"t{i}" for i in range(len(heights))]
        tbe = self._build_mixed_tbe(
            names,
            [EmbeddingLocation.DEVICE] * len(heights),
            res_enabled_tables=names,
            heights=heights,
            dims=dims,
        )
        flat = tbe.get_buffer("weights_dev")
        flat.copy_(torch.arange(flat.numel(), device=flat.device, dtype=flat.dtype))
        return tbe

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_drain_ships_only_rows_marked_since_the_last_drain(self) -> None:
        # Four drains over disjoint row sets, at interval 1 so each prefetch
        # drains. A single drain cannot see the bug: a drain that fails to clear
        # still returns the right rows the first time and only ratchets on the
        # second, so the shape of this test is load-bearing.
        rows = ROWS
        tbe = self._drain_tbe()
        drains = [self._mark_and_store_one_row(tbe, i, rows) for i in range(4)]
        drains.append(self._idle_iteration(tbe))

        # 1. Every iteration drains, and each one ships only its own row, one
        #    iteration after the drain that compacted it. Pinned as one literal
        #    because that is also where "no row ships twice and none is lost"
        #    is enforced: a ratchet lands here as a duplicate, a failure to
        #    clear as a short list.
        self.assertEqual(drains, [None, [rows + 0], [rows + 1], [rows + 2], [rows + 3]])
        # 2. Nothing in the lane's mask is still set after the last drain.
        present = tbe.get_buffer("_res_rows_seen")
        mask = tbe.get_buffer("_res_hbm_linear_mask")
        self.assertFalse(bool((present & mask).any()))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_flag_is_not_the_cache_lane_flag(self) -> None:
        # The protocol is chosen per stream() call, so the two lanes cannot
        # share a flag: the poller clears it once it has consumed a signal, so
        # one lane's poll would swallow the other's.
        tbe = self._drain_tbe()
        hbm_flag = tbe.get_buffer("_res_hbm_copy_done")
        cache_flag = tbe.get_buffer("res_copy_done")
        self.assertIsNot(hbm_flag, cache_flag)
        self.assertNotEqual(hbm_flag.data_ptr(), cache_flag.data_ptr())
        # Allocated at construction rather than on first ship, and zeroed. This
        # flag is allocated is_host_mapped=True unconditionally, which is the
        # malloc branch of new_unified_tensor -- the one that comes back
        # unzeroed -- and 1 is the value the first ship waits for. The cache
        # lane's flag follows uvm_host_mapped, so asserting its zero here would
        # be reading a fresh cudaMallocManaged page;
        # test_res_count_and_copy_done_start_zero covers it where it is malloc.
        self.assertEqual(hbm_flag.tolist(), [0])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_res_host_buf_grows_and_keeps(self) -> None:
        tbe = self._build_mixed_tbe(
            ["t0"], [EmbeddingLocation.DEVICE], res_enabled_tables=["t0"]
        )
        first = tbe._grown_res_host_buf_if_needed(None, 4, (), torch.int64)
        self.assertEqual(list(first.shape), [4])
        # A smaller ask reuses the buffer rather than shrinking it.
        self.assertIs(
            tbe._grown_res_host_buf_if_needed(first, 2, (), torch.int64), first
        )
        grown = tbe._grown_res_host_buf_if_needed(first, 9, (), torch.int64)
        self.assertIsNot(grown, first)
        self.assertEqual(list(grown.shape), [9])
        weights_buf = tbe._grown_res_host_buf_if_needed(None, 3, (8,), torch.float32)
        self.assertEqual(list(weights_buf.shape), [3, 8])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_ship_joins_previous_copy_before_reusing_buffers(self) -> None:
        # Structural, not behavioural. The hazard is a write-after-read race:
        # this drain overwrites the grow-and-keep staging buffers while the
        # previous drain's copy pool may still be reading them. It cannot be
        # reproduced here -- the harness is synchronous, so drain i finishes
        # before i+1 starts, and a MagicMock streamer never has an in-flight
        # reader. So assert the call exists instead, which is what makes
        # deleting it fail rather than pass silently.
        rows = ROWS
        tbe = self._drain_tbe()
        # One parent, so the join and the buffer writes land on a single
        # timeline. Pinning the join against `stream()` is too weak: the writes
        # sit between the two, so the join can be moved down to just above
        # `stream()` -- fully restoring the race -- with that assert still green.
        parent = MagicMock()
        tbe._raw_embedding_streamer = parent.streamer
        parent.attach_mock(
            MagicMock(side_effect=tbe._grown_res_host_buf_if_needed), "alloc"
        )
        tbe._grown_res_host_buf_if_needed = parent.alloc
        self._prefetch_until_pending(tbe, 3, rows)
        tbe.raw_embedding_stream()

        parent.streamer.join_hbm_dispatch_and_workers.assert_called_once()
        names = [c[0] for c in parent.mock_calls]
        join_at = names.index("streamer.join_hbm_dispatch_and_workers")
        first_write_at = names.index("alloc")
        self.assertLess(join_at, first_write_at)

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_ship_leaves_cache_lane_flag_undisturbed(self) -> None:
        # A drain must not touch res_copy_done. Mocking only the C++ streamer
        # keeps the real gather and the real flag writes in the path.
        rows = ROWS
        tbe = self._drain_tbe()
        streamer = MagicMock()
        tbe._raw_embedding_streamer = streamer
        self._prefetch_until_pending(tbe, 3, rows)
        tbe.raw_embedding_stream()

        calls = {
            c.kwargs.get("use_hbm", False): c.kwargs
            for c in streamer.stream.call_args_list
        }
        # Keyed by use_hbm, so the dict keeps only the last call per lane and
        # would look identical with twenty calls. Count them too.
        self.assertEqual(streamer.stream.call_count, 2)
        self.assertEqual(sorted(calls), [False, True])
        hbm = calls[True]
        # Each lane polls its own flag.
        self.assertIs(hbm["copy_done_flag"], tbe.get_buffer("_res_hbm_copy_done"))
        self.assertIsNot(hbm["copy_done_flag"], tbe.get_buffer("res_copy_done"))
        self.assertEqual(tbe.get_buffer("_res_hbm_copy_done").tolist(), [1])
        self.assertIsNone(hbm["identities"])
        self.assertFalse(hbm["blocking_tensor_copy"])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_ship_gathers_the_drained_rows(self) -> None:
        # End to end: the weights handed to stream() are the rows the drain
        # named, read out of weights_dev.
        rows, dim = 8, 16
        tbe = self._build_mixed_tbe(
            ["t0", "t1"],
            [EmbeddingLocation.MANAGED_CACHING, EmbeddingLocation.DEVICE],
            res_enabled_tables=["t1"],
            heights=[rows, rows],
            dims=[dim, dim],
        )
        flat = tbe.get_buffer("weights_dev")
        flat.copy_(torch.arange(flat.numel(), device=flat.device, dtype=flat.dtype))
        streamer = MagicMock()
        tbe._raw_embedding_streamer = streamer
        self._prefetch_until_pending(tbe, 2, rows)
        tbe.raw_embedding_stream()

        hbm = next(
            c.kwargs
            for c in streamer.stream.call_args_list
            if c.kwargs.get("use_hbm", False)
        )
        self.assertEqual(hbm["indices"].tolist(), [rows + 2])
        self.assertEqual(hbm["count"].tolist(), [1])
        # t1 is DEVICE-placed, so weights_dev holds only t1: row 2 is elements
        # 2*dim .. 3*dim - 1.
        self.assertEqual(
            hbm["weights"][0].tolist(),
            list(range(2 * dim, 3 * dim)),
        )

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_drain_ships_only_rows_the_mask_allows(self) -> None:
        # The mark is over-broad on placement, allowlist and the pruning
        # sentinel; the intersect is the only thing that removes them. t0 is
        # cached and not allowlisted, so its row must not ship even though the
        # bitmap records it.
        rows = ROWS
        tbe = self._drain_tbe()
        device = torch.cuda.current_device()
        tbe._mark_res_hbm_rows(
            torch.tensor([5, 7, -1], device=device, dtype=torch.int64),
            torch.tensor([0, 1, 3], device=device, dtype=torch.int64),
            None,
        )
        tbe._store_prefetched_tensors(
            # t0 row 5 (cached, out of the lane), t1 row 7 (in the lane),
            # and a pruned -1 that linearizes to the sentinel.
            indices=torch.tensor([5, 7, -1], device=device, dtype=torch.int64),
            offsets=torch.tensor([0, 1, 3], device=device, dtype=torch.int64),
            vbe_metadata=None,
            linear_cache_indices_merged=torch.tensor(
                [5, rows, rows], device=device, dtype=torch.int64
            ),
            final_lxu_cache_locations=torch.tensor(
                [0, -1, -1], device=device, dtype=torch.int32
            ),
            hash_zch_identities=None,
            hash_zch_runtime_meta=None,
        )
        present = tbe.get_buffer("_res_rows_seen")
        # All three really were marked, including the sentinel -- otherwise this
        # test would pass for the wrong reason.
        self.assertTrue(bool(present[2 * rows]))
        self.assertEqual(self._idle_iteration(tbe), [rows + 7])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_drain_ships_no_more_rows_than_res_streams(self) -> None:
        # A drain carries at most one row per row RES streams, however many
        # iterations of marks it covers: the window here is nearly two full
        # passes over the 64 streamed rows. The t0 and sentinel marks are what
        # make the bound binding -- without the mask intersect this drain would
        # carry 129 rows.
        rows = ROWS
        tbe = self._drain_tbe(drain_interval=rows * 2)
        device = torch.cuda.current_device()
        last = rows * 2 - 1
        drained: list[int] = []
        for i in range(rows * 2):
            tbe._mark_res_hbm_rows(
                torch.tensor(
                    [i % rows, i % rows, -1], device=device, dtype=torch.int64
                ),
                torch.tensor([0, 1, 3], device=device, dtype=torch.int64),
                None,
            )
            shipped = self._idle_iteration(tbe)
            if i < last:
                # Nothing ships until the interval is up.
                self.assertIsNone(shipped)
            else:
                assert shipped is not None
                drained = shipped
        self.assertEqual(sorted(drained), [rows + i for i in range(rows)])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_drain_fires_only_on_the_interval(self) -> None:
        # The interval must stay > 1: at 1 every drain fires on every call
        # whatever the counter does, so this is where the off-by-one is pinned.
        rows = ROWS
        tbe = self._drain_tbe(drain_interval=3)
        drains = [self._mark_and_store_one_row(tbe, i, rows) for i in range(6)]
        drains.append(self._idle_iteration(tbe))
        # Marks keep landing every prefetch. The compact fires one call BEFORE
        # the interval -- calls 2 and 5 -- so the rows go out on calls 3 and 6,
        # which is where the interval lands.
        self.assertEqual([drains[i] for i in (0, 1, 3, 4, 6)], [None] * 5)
        # A drain carries its whole window, not just one iteration. The first
        # window is a call short, since the phase puts the first compact on
        # call K-1.
        self.assertEqual(drains[2], [rows + 0, rows + 1])
        self.assertEqual(drains[5], [rows + 2, rows + 3, rows + 4])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hot_row_ships_once_per_interval(self) -> None:
        # The reason to queue in the trainer at all. Setting a bit is
        # idempotent, so a row touched every iteration costs one shipped row per
        # interval, not one per iteration -- that is what bounds how much delta
        # the downstream absorbs. A drain that shipped per-mark would return
        # four copies here. The trailing idle call is what shows the row does
        # not ship again once the interval has passed.
        rows = ROWS
        tbe = self._drain_tbe(drain_interval=4)
        drains = [self._mark_and_store_one_row(tbe, 7, rows) for _ in range(4)]
        drains.append(self._idle_iteration(tbe))
        self.assertEqual(drains, [None, None, None, [rows + 7], None])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hot_row_ships_every_drain_when_remarked(self) -> None:
        # The complement of the interval bound above: a row that keeps changing
        # keeps shipping, once per drain. Clearing at pickup instead of at
        # compact would wipe the re-mark with the previous drain's mask, so the
        # second update would be dropped and iteration 3 would return None.
        rows = ROWS
        tbe = self._drain_tbe()
        drains = [self._mark_and_store_one_row(tbe, 7, rows) for _ in range(4)]
        self.assertEqual(drains, [None, [rows + 7], [rows + 7], [rows + 7]])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_drain_leaves_its_count_pending(self) -> None:
        # The drain must not read the count. Asserting on the pending buffer is
        # the only way to see that from the outside: a drain that resolved the
        # count itself would ship on its own iteration and leave nothing here,
        # which is the host block the deferral exists to remove.
        rows = ROWS
        tbe = self._drain_tbe()
        self._mark_and_store_one_row(tbe, 3, rows)

        pending = tbe._res_compacted_rows
        assert pending is not None
        # Sized by the whole linear index space, because the count that would
        # narrow it has not landed.
        linear_size = tbe.get_buffer("_res_hbm_linear_mask").numel()
        self.assertEqual(pending.numel(), linear_size + 1)
        # Pinned, or the count copy could not be async.
        self.assertTrue(tbe._res_drain_count_cpu.is_pinned())

        self.assertEqual(self._idle_iteration(tbe), [rows + 3])
        # What ships is a clone, not a view: a view would hold the whole
        # compaction buffer alive for as long as the slice sits in
        # prefetched_info_list, which is one int64 per row of every table.
        drained = tbe.prefetched_info_list[-1].res_hbm_indices
        assert drained is not None
        self.assertEqual(drained.untyped_storage().nbytes(), drained.numel() * 8)

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_drain_interval_must_be_positive(self) -> None:
        # interval 0 raises ZeroDivisionError deep in prefetch rather than at
        # construction; a negative interval silently behaves as its absolute
        # value, so -3 would drain every third iteration while looking rejected.
        # Anchored past the knob name rather than matching it loosely: any
        # future knob prefixed `res_hbm_drain_interval` would otherwise satisfy
        # this regex from its own raise.
        with self.assertRaisesRegex(ValueError, r"res_hbm_drain_interval must be"):
            self._drain_tbe(drain_interval=0)
        with self.assertRaisesRegex(ValueError, r"res_hbm_drain_interval must be"):
            self._drain_tbe(drain_interval=-3)

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_dim_uniformity_records_per_feature_layout(self) -> None:
        # feature_table_map duplicates t0, so the cached layout tensors are 3
        # long (per feature), not 2 (per table) -- rows_per_table included.
        tbe = self._build_mixed_tbe(
            ["t0", "t1"],
            [EmbeddingLocation.DEVICE, EmbeddingLocation.DEVICE],
            res_enabled_tables=["t0", "t1"],
            heights=[4, 3],
            dims=[8, 4],
            feature_table_map=[0, 0, 1],
        )
        self.assertFalse(tbe._res_hbm_dims_equal)
        self.assertEqual(tbe._res_hbm_dim, 0)
        self.assertEqual(tbe._res_rows_per_table_cpu.tolist(), [4, 4, 3])
        self.assertEqual(tbe._res_D_offsets_cpu.tolist(), [0, 8, 16, 20])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_copy_weights_uniform_dim(self) -> None:
        tbe = self._device_tbe(heights=[4, 3], dims=[8, 8])
        self.assertTrue(tbe._res_hbm_dims_equal)
        device = torch.cuda.current_device()
        res_indices = torch.tensor([1, 5], device=device, dtype=torch.int64)
        out = torch.full((2, 8), -1.0, device=device)
        tbe._gather_res_rows(res_indices, out, tbe.get_buffer("weights_dev"))
        self.assertEqual(out[0].tolist(), list(range(8, 16)))
        self.assertEqual(out[1].tolist(), list(range(40, 48)))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_copy_weights_mixed_dim_zero_pads_narrow_table(self) -> None:
        # Without the zero pad the receiver would ship whatever `out` held.
        tbe = self._device_tbe(heights=[4, 3], dims=[8, 4])
        self.assertFalse(tbe._res_hbm_dims_equal)
        device = torch.cuda.current_device()
        # t0 row 1, then t1 row 1 (linear 5) -- ascending, as the gather requires.
        res_indices = torch.tensor([1, 5], device=device, dtype=torch.int64)
        out = torch.full((2, 8), -1.0, device=device)
        tbe._gather_res_rows(res_indices, out, tbe.get_buffer("weights_dev"))
        self.assertEqual(out[0].tolist(), list(range(8, 16)))
        # t1's region starts at element 4*8=32; its row 1 is elements 36..39.
        self.assertEqual(out[1].tolist(), [36.0, 37.0, 38.0, 39.0, 0.0, 0.0, 0.0, 0.0])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_gather_uniformity_counts_non_allowlisted_pool_tables(self) -> None:
        # `pad0`/`pad1` are DEVICE and share `weights_dev` with the allowlisted
        # table, but are not themselves allowlisted. Their widths still decide
        # whether `view(-1, dim)` is a legal reinterpretation of the pool, and
        # their elements still push `t1` off a `dim` boundary.
        #
        # Sized so the pool DIVIDES: 4 + 64 + 4 = 72 elements and 72 % 8 == 0,
        # so a uniform path here would not raise -- it would return wrong rows.
        # The non-dividing case is the safe one, since `view` catches it.
        tbe = self._build_mixed_tbe(
            ["pad0", "t1", "pad1"],
            [EmbeddingLocation.DEVICE] * 3,
            res_enabled_tables=["t1"],
            heights=[1, 8, 1],
            dims=[4, 8, 4],
        )
        flat = tbe.get_buffer("weights_dev")
        self.assertEqual(flat.numel(), 72)
        flat.copy_(torch.arange(flat.numel(), device=flat.device, dtype=flat.dtype))

        device = torch.cuda.current_device()
        # t1 row 0 is linear 1, since pad0 holds linear 0. Its elements start at
        # 1*4 = 4, but a uniform path would read row 1 of an 8-wide grid --
        # elements 8..15 -- because pad0's 4 elements shifted the grid.
        # Asserted before the mechanism below so that a regression fails on the
        # WRONG VALUES, which is the bug, rather than on the flag that causes it.
        res_indices = torch.tensor([1], device=device, dtype=torch.int64)
        out = torch.full((1, 8), -1.0, device=device)
        tbe._gather_res_rows(res_indices, out, flat)
        self.assertEqual(out[0].tolist(), [float(v) for v in range(4, 12)])

        # Uniformity is a property of the pool, not of the allowlist.
        self.assertFalse(tbe._res_hbm_dims_equal)

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_gather_logs_cached_feature_ids(self) -> None:
        # weights_offsets restarts at zero per placement pool, so a cached id
        # resolves to an offset INSIDE weights_dev and returns a DEVICE row
        # under a cached row's name -- in range, no error. t0 and t1 are the
        # same shape so the aliasing is exact.
        #
        # The drain cannot produce this: _res_hbm_linear_mask marks only
        # allowlisted DEVICE rows. So it is logged rather than raised, and the
        # aliased row is asserted here because that is what the log is warning
        # about -- a silent wrong row, not a crash.
        rows = 4
        tbe = self._build_mixed_tbe(
            ["t0", "t1"],
            [EmbeddingLocation.DEVICE, EmbeddingLocation.MANAGED_CACHING],
            res_enabled_tables=["t0", "t1"],
            heights=[rows, rows],
            dims=[8, 8],
        )
        flat = tbe.get_buffer("weights_dev")
        flat.copy_(torch.arange(flat.numel(), device=flat.device, dtype=flat.dtype))
        device = torch.cuda.current_device()
        out = torch.full((1, 8), -1.0, device=device)
        # Linear id `rows` is t1 row 0 -- a cached feature.
        cached_id = torch.tensor([rows], device=device, dtype=torch.int64)
        with self.assertLogs(level="ERROR") as logs:
            tbe._gather_res_rows(cached_id, out, flat)
        self.assertIn("not DEVICE-placed", "\n".join(logs.output))
        # t1 row 0 aliased onto t0 row 0, which is elements 0..7.
        self.assertEqual(out[0].tolist(), list(range(8)))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_gather_empty_indices_is_a_noop(self) -> None:
        # An empty drain is the normal condition, not an edge case: on a real
        # model most TBEs mark nothing most iterations.
        tbe = self._device_tbe(heights=[4], dims=[8])
        device = torch.cuda.current_device()
        out = torch.full((2, 8), -1.0, device=device)
        tbe._gather_res_rows(
            torch.empty(0, device=device, dtype=torch.int64),
            out,
            tbe.get_buffer("weights_dev"),
        )
        # Untouched, not zeroed: callers slice to the count.
        self.assertEqual(out.unique().tolist(), [-1.0])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_mark_clamps_out_of_range_into_the_sentinel(self) -> None:
        # _prefetch sits below prepare_inputs, so bounds_check_indices has not
        # run and index 999 into 64 rows reaches the clamp raw. Without the
        # clamp this is a device-side assert, not a wrong row.
        rows = ROWS
        device = torch.cuda.current_device()
        tbe = self._build_mixed_tbe(
            ["t0"],
            [EmbeddingLocation.DEVICE],
            res_enabled_tables=["t0"],
            heights=[rows],
        )
        tbe._prefetch(
            torch.tensor([999], device=device, dtype=torch.int64),
            torch.tensor([0, 1], device=device, dtype=torch.int64),
        )
        present = [
            i for i, p in enumerate(tbe.get_buffer("_res_rows_seen").tolist()) if p
        ]
        self.assertEqual(present, [rows])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_mark_negative_index_never_reaches_the_clamp(self) -> None:
        # Why the clamp is upper-bound only: linearize_cache_indices.cu guards
        # both its write sites on `indices[index] >= 0` and routes anything
        # negative to the sentinel, so a negative input arrives here already in
        # range.
        rows = ROWS
        device = torch.cuda.current_device()
        tbe = self._build_mixed_tbe(
            ["t0"],
            [EmbeddingLocation.DEVICE],
            res_enabled_tables=["t0"],
            heights=[rows],
        )
        tbe._prefetch(
            torch.tensor([-1], device=device, dtype=torch.int64),
            torch.tensor([0, 1], device=device, dtype=torch.int64),
        )
        present = [
            i for i, p in enumerate(tbe.get_buffer("_res_rows_seen").tolist()) if p
        ]
        self.assertEqual(present, [rows])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_subset_allowlist_builds_feature_mask(self) -> None:
        tbe = self._build_tbe(["t0", "t1", "t2"], res_enabled_tables=["t1"])
        self.assertFalse(tbe._res_all_features_enabled)
        mask = tbe.get_buffer("res_enabled_feature_mask")
        # one feature per table => mask lines up with table order
        self.assertEqual(mask.tolist(), [False, True, False])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_unknown_table_name_ignored(self) -> None:
        # A name not in this TBE contributes nothing (no error, not enabled) --
        # it may legitimately belong to another TBE in a multi-TBE model.
        tbe = self._build_tbe(["t0", "t1"], res_enabled_tables=["t1", "nonexistent"])
        self.assertFalse(tbe._res_all_features_enabled)
        mask = tbe.get_buffer("res_enabled_feature_mask")
        self.assertEqual(mask.tolist(), [False, True])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_get_enabled_feature_mask_and_indices(self) -> None:
        rows = ROWS
        tbe = self._build_tbe(["t0", "t1", "t2"], res_enabled_tables=["t1"])
        device = torch.cuda.current_device()
        # linear indices: t0 row 5, t1 row 0 (boundary), t2 row 10, t1 row 63
        linear = torch.tensor(
            [5, rows, 2 * rows + 10, rows + 63], device=device, dtype=torch.int64
        )
        enabled_mask, feature_indices = tbe._get_enabled_feature_mask_and_indices(
            linear
        )
        # right=True puts the exact boundary (linear==rows) into t1, not t0.
        self.assertEqual(feature_indices.tolist(), [0, 1, 2, 1])
        # only t1 is enabled
        self.assertEqual(enabled_mask.tolist(), [False, True, False, True])
