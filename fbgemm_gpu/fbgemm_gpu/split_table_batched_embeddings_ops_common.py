#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import enum
from dataclasses import dataclass
from typing import List, NamedTuple


# Maximum number of times prefetch() can be called without
# a corresponding forward() call
MAX_PREFETCH_DEPTH = 100

# GPU and CPU use 16-bit scale and bias for quantized embedding bags in TBE
# The total size is 2 + 2 = 4 bytes
DEFAULT_SCALE_BIAS_SIZE_IN_BYTES = 4


class EmbeddingLocation(enum.IntEnum):
    DEVICE = 0
    MANAGED = 1
    MANAGED_CACHING = 2
    HOST = 3
    MTIA = 4


class CacheAlgorithm(enum.Enum):
    LRU = 0
    LFU = 1


class PoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1
    NONE = 2


class BoundsCheckMode(enum.IntEnum):
    # Raise an exception (CPU) or device-side assert (CUDA)
    FATAL = 0
    # Log the first out-of-bounds instance per kernel, and set to zero.
    WARNING = 1
    # Set to zero.
    IGNORE = 2
    # No bounds checks.
    NONE = 3


RecordCacheMetrics: NamedTuple = NamedTuple(
    "RecordCacheMetrics",
    [("record_cache_miss_counter", bool), ("record_tablewise_cache_miss", bool)],
)

SplitState: NamedTuple = NamedTuple(
    "SplitState",
    [
        ("dev_size", int),
        ("host_size", int),
        ("uvm_size", int),
        ("placements", List[EmbeddingLocation]),
        ("offsets", List[int]),
    ],
)


@dataclass
class CacheState:
    # T + 1 elements and cache_hash_size_cumsum[-1] == total_cache_hash_size
    cache_hash_size_cumsum: List[int]
    cache_index_table_map: List[int]
    total_cache_hash_size: int


def construct_cache_state(
    row_list: List[int],
    location_list: List[EmbeddingLocation],
    feature_table_map: List[int],
) -> CacheState:
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
    s = CacheState(
        cache_hash_size_cumsum=cache_hash_size_cumsum,
        cache_index_table_map=cache_index_table_map,
        total_cache_hash_size=total_cache_hash_size,
    )
    return s


# NOTE: This is also defined in fbgemm_gpu.split_embedding_utils, but declaring
# target dependency on :split_embedding_utils will result in compatibility
# breakage with Caffe2 module_factory because it will pull in numpy
def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b
