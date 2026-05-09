#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Cache-related type definitions for TBE.

Contains CacheAlgorithm, CacheState, MultiPassPrefetchConfig, UVMCacheStatsIndex,
and the DEFAULT_ASSOC constant.

These types are publicly accessible via ``fbgemm_gpu.tbe.cache.<Name>`` (eager
re-export from ``fbgemm_gpu/tbe/cache/__init__.py``). Any new types added here
that will be re-exported from the package ``__init__.py`` MUST avoid
``@dataclass`` decoration to remain torch.package-safe under Python 3.12+:
``dataclasses._is_type`` does ``sys.modules.get(cls.__module__).__dict__``
while resolving annotations, and the ``<torch_package_0>`` sandbox does not
register the mangled module name in ``sys.modules`` at decoration time, so the
``__dict__`` access raises ``AttributeError``. ``NamedTuple`` (or a plain
class) sidesteps this entirely. Re-evaluate this constraint when the
upstream Python / torch.package interaction is fixed.
"""

from __future__ import annotations

import enum
from typing import NamedTuple

import torch
from fbgemm_gpu.tbe.config import EmbeddingLocation


# Default associativity for the UVM cache (32 for CUDA, 64 for ROCm)
DEFAULT_ASSOC: int = 32 if torch.version.hip is None else 64


class CacheAlgorithm(enum.Enum):
    LRU = 0
    LFU = 1


class MultiPassPrefetchConfig(NamedTuple):
    # Number of passes to split indices tensor into. Actual number of passes may
    # be less if indices tensor is too small to split.
    num_passes: int = 12

    # The minimal number of element in indices tensor to be able to split into
    # two passes. This is useful to prevent too many prefetch kernels spamming
    # the CUDA launch queue.
    # The default 6M indices means 6M * 8 * 6 = approx. 300MB of memory overhead
    # per pass.
    min_splitable_pass_size: int = 6 * 1024 * 1024


# Keep in sync with fbgemm_gpu/include/fbgemm_gpu/split_embeddings_cache_cuda.cuh
class UVMCacheStatsIndex(enum.IntEnum):
    num_calls = 0
    num_requested_indices = 1
    num_unique_indices = 2
    num_unique_misses = 3
    num_conflict_unique_misses = 4
    num_conflict_misses = 5


class CacheState(NamedTuple):
    # T + 1 elements and cache_hash_size_cumsum[-1] == total_cache_hash_size
    cache_hash_size_cumsum: list[int]
    cache_index_table_map: list[int]
    total_cache_hash_size: int

    @classmethod
    def construct(
        cls,
        row_list: list[int],
        location_list: list[EmbeddingLocation],
        feature_table_map: list[int],
    ) -> CacheState:
        """Build CacheState from row/location lists and feature table map.

        Formerly the free function construct_cache_state() in
        split_table_batched_embeddings_ops_common.py.
        """
        _cache_hash_size_cumsum = [0]
        total_cache_hash_size = 0
        for num_embeddings, location in zip(row_list, location_list):
            if location == EmbeddingLocation.MANAGED_CACHING:
                total_cache_hash_size += num_embeddings
            _cache_hash_size_cumsum.append(total_cache_hash_size)
        # [T], -1: non-cached table
        cache_hash_size_cumsum = []
        # [total_cache_hash_size], linear cache index -> table index
        cache_index_table_map = [-1] * total_cache_hash_size
        unique_feature_table_map = {}
        for t, t_ in enumerate(feature_table_map):
            unique_feature_table_map[t_] = t
        for t_, t in unique_feature_table_map.items():
            start, end = _cache_hash_size_cumsum[t_], _cache_hash_size_cumsum[t_ + 1]
            cache_index_table_map[start:end] = [t] * (end - start)
        cache_hash_size_cumsum = [
            (
                _cache_hash_size_cumsum[t_]
                if location_list[t_] == EmbeddingLocation.MANAGED_CACHING
                else -1
            )
            for t_ in feature_table_map
        ]
        cache_hash_size_cumsum.append(total_cache_hash_size)
        return cls(
            cache_hash_size_cumsum=cache_hash_size_cumsum,
            cache_index_table_map=cache_index_table_map,
            total_cache_hash_size=total_cache_hash_size,
        )
