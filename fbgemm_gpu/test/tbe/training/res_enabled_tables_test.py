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
    RESParams,
    SplitTableBatchedEmbeddingBagsCodegen,
)

from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


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
        rows: int = 64,
        dim: int = 16,
        location: EmbeddingLocation = EmbeddingLocation.DEVICE,
        uvm_host_mapped: bool = False,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        """One table per name (one feature per table), RES enabled."""
        n = len(table_names)
        res_params = RESParams(
            res_store_shards=1,
            table_names=list(table_names),
            table_offsets=[i * rows for i in range(n)],
            table_sizes=[rows] * n,
            res_enabled_tables=list(res_enabled_tables),
        )
        return SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (rows, dim, location, ComputeDevice.CUDA) for _ in range(n)
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
        dim: int = 16,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        """One table per name, each with its own placement and row count."""
        n = len(table_names)
        rows = heights if heights is not None else [64] * n
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
        )
        return SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (h, dim, loc, ComputeDevice.CUDA) for h, loc in zip(rows, locations)
            ],
            enable_raw_embedding_streaming=True,
            res_params=res_params,
            feature_table_map=feature_table_map,
        )

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
        # hasattr, not just named_buffers: registering it as a plain attribute
        # would keep the buffer assertion green while still paying the whole
        # allocation this test exists to prevent.
        self.assertNotIn("_res_rows_seen", dict(tbe.named_buffers()))
        self.assertFalse(hasattr(tbe, "_res_rows_seen"))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_mark_records_every_touched_row(self) -> None:
        # The mark is unconditional on cache state: t1 is DEVICE-placed and so
        # never hits the cache, and its row must still be recorded.
        rows = 64
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
        present = tbe.get_buffer("_res_rows_seen").tolist()
        self.assertEqual(len(present), 2 * rows + 1)
        self.assertEqual([i for i, p in enumerate(present) if p], [3, rows + 5])

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_prefetch_with_lane_off(self) -> None:
        # The buffer does not exist when the lane is off, so the mark must be
        # gated rather than guarded -- an ungated index_fill_ is AttributeError
        # on every RES model that leaves the lane off. Drives _prefetch, which
        # is where the mark lives; calling _store_prefetched_tensors here would
        # exercise a method the mark is no longer in.
        rows = 64
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
        self.assertNotIn("_res_rows_seen", dict(tbe.named_buffers()))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_mark_fires_through_prefetch(self) -> None:
        # Drives _prefetch rather than calling _store_prefetched_tensors by
        # hand: the mark has to survive the cacheless early-return, and a test
        # that invokes the inner method directly cannot see that guard at all.
        # The DEVICE-only TBE is the shape TorchRec actually builds for a
        # DEVICE table, and the only shape this lane exists to serve.
        rows = 64
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

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_hbm_mark_clamps_out_of_range_into_the_sentinel(self) -> None:
        # _prefetch sits below prepare_inputs, so bounds_check_indices has not
        # run and index 999 into 64 rows reaches the clamp raw. Without the
        # clamp this is a device-side assert, not a wrong row.
        rows = 64
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
        # Why the clamp is upper-bound only: linearize_cache_indices guards both
        # its write sites on `indices[index] >= 0`
        # (linearize_cache_indices.cu:57 and :165) and routes anything negative
        # to the sentinel, so a negative input arrives here already in range.
        rows = 64
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
        rows = 64
        tbe = self._build_tbe(["t0", "t1", "t2"], res_enabled_tables=["t1"], rows=rows)
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
