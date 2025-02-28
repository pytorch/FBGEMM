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

import torch
from torch import Tensor


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


def str_to_embedding_location(key: str) -> EmbeddingLocation:
    lookup = {
        "device": EmbeddingLocation.DEVICE,
        "managed": EmbeddingLocation.MANAGED,
        "managed_caching": EmbeddingLocation.MANAGED_CACHING,
        "host": EmbeddingLocation.HOST,
        "mtia": EmbeddingLocation.MTIA,
    }
    if key in lookup:
        return lookup[key]
    else:
        raise ValueError(f"Cannot parse value into EmbeddingLocation: {key}")


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


class PoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1
    NONE = 2

    def do_pooling(self) -> bool:
        return self is not PoolingMode.NONE


def str_to_pooling_mode(key: str) -> PoolingMode:
    lookup = {
        "sum": PoolingMode.SUM,
        "mean": PoolingMode.MEAN,
        "none": PoolingMode.NONE,
    }
    if key in lookup:
        return lookup[key]
    else:
        raise ValueError(f"Cannot parse value into PoolingMode: {key}")


class BoundsCheckMode(enum.IntEnum):
    # Raise an exception (CPU) or device-side assert (CUDA)
    FATAL = 0
    # Log the first out-of-bounds instance per kernel, and set to zero.
    WARNING = 1
    # Set to zero.
    IGNORE = 2
    # No bounds checks.
    NONE = 3
    # IGNORE with V2 enabled
    V2_IGNORE = 4
    # WARNING with V2 enabled
    V2_WARNING = 5
    # FATAL with V2 enabled
    V2_FATAL = 6


class EmbeddingSpecInfo(enum.IntEnum):
    feature_names = 0
    rows = 1
    dims = 2
    sparse_type = 3
    embedding_location = 4


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


# NOTE: This is also defined in fbgemm_gpu.tbe.utils, but declaring
# target dependency on :split_embedding_utils will result in compatibility
# breakage with Caffe2 module_factory because it will pull in numpy
def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def tensor_to_device(tensor: torch.Tensor, device: torch.device) -> Tensor:
    if tensor.device == torch.device("meta"):
        return torch.empty_like(tensor, device=device)
    return tensor.to(device)


def get_new_embedding_location(
    device: torch.device, cache_load_factor: float
) -> EmbeddingLocation:
    """
    Based on the cache_load_factor and device, return the embedding location intended
    for the TBE weights.
    """
    # Only support CPU and GPU device
    assert device.type == "cpu" or device.type == "cuda"
    if cache_load_factor < 0 or cache_load_factor > 1:
        raise ValueError(
            f"cache_load_factor must be between 0.0 and 1.0, got {cache_load_factor}"
        )

    if device.type == "cpu":
        return EmbeddingLocation.HOST
    # UVM only
    elif cache_load_factor == 0:
        return EmbeddingLocation.MANAGED
    # HBM only
    elif cache_load_factor == 1.0:
        return EmbeddingLocation.DEVICE
    # UVM caching
    else:
        return EmbeddingLocation.MANAGED_CACHING
