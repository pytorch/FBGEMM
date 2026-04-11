#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

"""
TurboSSD v2 inference serving module.

Wraps SSDIntNBitTableBatchedEmbeddingBags with a serving-friendly API for
Video Retrieval HSTU and similar models that currently use EmbeddingDB.

Key differences from raw SSD TBE:
    - Automatic prefetch before forward (single-call API)
    - streaming_update() and load_snapshot() for delta model updates
    - HBM cache budget validation and sizing helpers
    - Factory method from_embedding_specs() with HSTU-style defaults
"""

import logging
from typing import Optional

import torch  # usort:skip
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    rounded_row_size_in_bytes,
)
from torch import nn, Tensor  # usort:skip

from .common import ASSOC
from .inference import SSDIntNBitTableBatchedEmbeddingBags


class TurboSSDInferenceModule(nn.Module):
    """
    Drop-in serving module backed by FBGEMM SSD TBE with TurboSSD v2
    streaming support.

    Designed to replace EmbeddingDB for models that need:
        - HBM-cached SSD embedding lookups (vs. CPU-only DRAM cache)
        - Streaming delta updates during inference
        - Zero-downtime snapshot transitions

    Usage::

        module = TurboSSDInferenceModule.from_embedding_specs(
            specs=[("post_id", 1_600_000_000, 128, SparseType.INT8)],
            hbm_budget_gb=32.0,
            ssd_directory="/mnt/ssd/embeddings",
        )
        output = module(indices, offsets)
        module.streaming_update(delta_indices, delta_weights)
    """

    def __init__(
        self,
        tbe: SSDIntNBitTableBatchedEmbeddingBags,
    ) -> None:
        super().__init__()
        self.tbe = tbe

    @classmethod
    def from_embedding_specs(
        cls,
        specs: list[tuple[str, int, int, SparseType]],
        ssd_directory: str = "/tmp",
        hbm_budget_gb: float = 0.0,
        cache_hit_rate: float = 0.90,
        pooling_mode: PoolingMode = PoolingMode.SUM,
        ssd_shards: int = 8,
        ssd_write_buffer_size: int = 2 * 1024 * 1024 * 1024,
        ssd_max_write_buffer_num: int = 16,
        ssd_rate_limit_mbps: int = 0,
        enable_cache_locking: bool = False,
    ) -> "TurboSSDInferenceModule":
        """
        Create a TurboSSD inference module from embedding specifications.

        Automatically sizes the HBM cache based on the target hit rate and
        available HBM budget.

        Args:
            specs: List of (name, num_rows, embedding_dim, sparse_type).
            ssd_directory: Base directory for RocksDB storage.
            hbm_budget_gb: Maximum HBM to use for the cache (0 = auto-size).
            cache_hit_rate: Target cache hit rate for sizing (0.0 to 1.0).
            pooling_mode: Pooling mode (SUM, MEAN, NONE).
            ssd_shards: Number of RocksDB shards.
            ssd_write_buffer_size: RocksDB write buffer size in bytes.
            ssd_max_write_buffer_num: Max number of RocksDB write buffers.
            ssd_rate_limit_mbps: RocksDB rate limit in MB/s (0 = unlimited).
            enable_cache_locking: Enable cache line locking for concurrent
                prefetch/forward safety.

        Returns:
            A configured TurboSSDInferenceModule ready for serving.
        """
        if not specs:
            raise ValueError("specs must contain at least one embedding table")
        if not (0.0 < cache_hit_rate <= 1.0):
            raise ValueError(f"cache_hit_rate must be in (0, 1], got {cache_hit_rate}")

        cache_sets = cls._compute_cache_sets(
            specs,
            hbm_budget_gb,
            cache_hit_rate,
        )

        tbe = SSDIntNBitTableBatchedEmbeddingBags(
            embedding_specs=specs,
            feature_table_map=list(range(len(specs))),
            pooling_mode=pooling_mode,
            ssd_storage_directory=ssd_directory,
            ssd_shards=ssd_shards,
            cache_sets=cache_sets,
            ssd_write_buffer_size=ssd_write_buffer_size,
            ssd_max_write_buffer_num=ssd_max_write_buffer_num,
            ssd_rate_limit_mbps=ssd_rate_limit_mbps,
            enable_cache_locking=enable_cache_locking,
        ).cuda()

        module = cls(tbe)

        total_rows = sum(r for _, r, _, _ in specs)
        cache_capacity = cache_sets * ASSOC
        logging.info(
            f"TurboSSD inference module: {len(specs)} tables, "
            f"{total_rows:,} total rows, "
            f"{cache_sets:,} cache sets ({cache_capacity:,} slots), "
            f"target hit rate {cache_hit_rate:.0%}"
        )

        return module

    @staticmethod
    def _compute_cache_sets(
        specs: list[tuple[str, int, int, SparseType]],
        hbm_budget_gb: float,
        cache_hit_rate: float,
    ) -> int:
        """
        Compute the number of cache sets for the target hit rate.

        For a set-associative cache with ASSOC ways, we need enough sets
        so that the total number of cache slots >= target fraction of the
        working set.

        If an HBM budget is specified, the cache is capped to fit within it.
        """

        total_rows = sum(rows for _, rows, _, _ in specs)
        target_cached_rows = int(total_rows * cache_hit_rate)
        cache_sets_from_hit_rate = max((target_cached_rows + ASSOC - 1) // ASSOC, 1)

        if hbm_budget_gb > 0:
            max_d_cache = max(
                rounded_row_size_in_bytes(dim, ty, 16) for _, _, dim, ty in specs
            )
            budget_bytes = int(hbm_budget_gb * 1024 * 1024 * 1024)
            cache_sets_from_budget = budget_bytes // (ASSOC * max_d_cache)
            cache_sets = min(cache_sets_from_hit_rate, max(cache_sets_from_budget, 1))
        else:
            cache_sets = cache_sets_from_hit_rate

        return cache_sets

    @staticmethod
    def estimate_hbm_gb(
        specs: list[tuple[str, int, int, SparseType]],
        cache_hit_rate: float = 0.90,
    ) -> float:
        """
        Estimate HBM usage in GB for the given specs and target hit rate.

        Returns the projected HBM cache size. Useful for capacity planning
        on H100 (96 GB) and MI350X (288 GB).
        """
        total_rows = sum(rows for _, rows, _, _ in specs)
        target_rows = int(total_rows * cache_hit_rate)
        cache_sets = max((target_rows + ASSOC - 1) // ASSOC, 1)

        max_d_cache = max(
            rounded_row_size_in_bytes(dim, ty, 16) for _, _, dim, ty in specs
        )
        cache_bytes = cache_sets * ASSOC * max_d_cache
        return cache_bytes / (1024 * 1024 * 1024)

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Single-call forward: automatically prefetches then looks up.

        Args:
            indices: 1D int32/int64 tensor of embedding indices.
            offsets: 1D int32/int64 tensor of bag offsets.
            per_sample_weights: Optional per-sample weights for weighted bags.

        Returns:
            Output tensor of shape [B, total_D].
        """
        self.tbe.prefetch(indices.long(), offsets.long())
        return self.tbe(indices.int(), offsets.int(), per_sample_weights)

    @torch.jit.export
    def streaming_update(
        self,
        indices: Tensor,
        weights: Tensor,
    ) -> None:
        """
        Apply streaming embedding updates (delta model updates).

        Writes new values to RocksDB and invalidates HBM cache entries
        so the next forward() reloads fresh values.

        Args:
            indices: 1D int64 tensor of linear embedding indices.
            weights: 2D uint8 tensor of shape [N, max_D_cache].
        """
        self.tbe.streaming_update(indices, weights)

    @torch.jit.export
    def load_snapshot(
        self,
        ssd_storage_directory: str,
        ssd_shards: int = 1,
    ) -> None:
        """
        Swap to a new RocksDB snapshot without downtime.

        Opens a fresh RocksDB at the given path and fully invalidates
        the HBM cache. After calling this, populate the new DB via
        streaming_update() in batches.

        Args:
            ssd_storage_directory: Path to store the new RocksDB instance.
            ssd_shards: Number of RocksDB shards for the new instance.
        """
        self.tbe.load_snapshot(
            ssd_storage_directory=ssd_storage_directory,
            ssd_shards=ssd_shards,
        )

    @property
    def max_D_cache(self) -> int:
        return self.tbe.max_D_cache

    @property
    def embedding_specs(self) -> list[tuple[str, int, int, SparseType]]:
        return self.tbe.embedding_specs
