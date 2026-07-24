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
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        """One DEVICE table per name (one feature per table), RES enabled."""
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
                (rows, dim, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
                for _ in range(n)
            ],
            enable_raw_embedding_streaming=True,
            res_params=res_params,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_empty_allowlist_enables_all(self) -> None:
        # Empty allowlist => no scoping; preserves the pre-allowlist behavior.
        tbe = self._build_tbe(["t0", "t1", "t2"], res_enabled_tables=[])
        self.assertTrue(tbe._res_all_features_enabled)
        mask = tbe.get_buffer("res_enabled_feature_mask")
        self.assertEqual(mask.tolist(), [True, True, True])

    @unittest.skipIf(*gpu_unavailable)
    def test_all_tables_listed_short_circuits(self) -> None:
        # Listing every table is equivalent to the empty/all-enabled fast path.
        tbe = self._build_tbe(["t0", "t1", "t2"], res_enabled_tables=["t0", "t1", "t2"])
        self.assertTrue(tbe._res_all_features_enabled)
        mask = tbe.get_buffer("res_enabled_feature_mask")
        self.assertEqual(mask.tolist(), [True, True, True])

    @unittest.skipIf(*gpu_unavailable)
    def test_subset_allowlist_builds_feature_mask(self) -> None:
        tbe = self._build_tbe(["t0", "t1", "t2"], res_enabled_tables=["t1"])
        self.assertFalse(tbe._res_all_features_enabled)
        mask = tbe.get_buffer("res_enabled_feature_mask")
        # one feature per table => mask lines up with table order
        self.assertEqual(mask.tolist(), [False, True, False])

    @unittest.skipIf(*gpu_unavailable)
    def test_unknown_table_name_ignored(self) -> None:
        # A name not in this TBE contributes nothing (no error, not enabled) --
        # it may legitimately belong to another TBE in a multi-TBE model.
        tbe = self._build_tbe(["t0", "t1"], res_enabled_tables=["t1", "nonexistent"])
        self.assertFalse(tbe._res_all_features_enabled)
        mask = tbe.get_buffer("res_enabled_feature_mask")
        self.assertEqual(mask.tolist(), [False, True])

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
