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
        dims: list[int] | None = None,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        """One table per name, each with its own placement, row count and dim."""
        n = len(table_names)
        rows = heights if heights is not None else [64] * n
        widths = dims if dims is not None else [dim] * n
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
                (h, w, loc, ComputeDevice.CUDA)
                for h, w, loc in zip(rows, widths, locations)
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
