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
from typing import FrozenSet, NamedTuple, Optional, Tuple

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

    @classmethod
    # pyre-ignore[3]
    def str_values(cls):
        return [
            "device",
            "managed",
            "managed_caching",
            "host",
            "mtia",
        ]

    @classmethod
    # pyre-ignore[3]
    def from_str(cls, key: str):
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


class EvictionPolicy(NamedTuple):
    eviction_trigger_mode: int = (
        0  # disabled, 0: disabled, 1: iteration, 2: mem_util, 3: manual 4: id count
    )
    eviction_strategy: int = (
        0  # 0: timestamp, 1: counter , 2: counter + timestamp, 3: feature l2 norm 4: timestamp threshold 5: feature score
    )
    eviction_step_intervals: Optional[int] = (
        None  # trigger_step_interval if trigger mode is iteration
    )
    eviction_mem_threshold_gb: Optional[int] = (
        None  # eviction trigger condition if trigger mode is mem_util
    )
    counter_thresholds: Optional[list[int]] = (
        None  # count_thresholds for each table if eviction strategy is counter
    )
    ttls_in_mins: Optional[list[int]] = (
        None  # ttls_in_mins for each table if eviction strategy is timestamp
    )
    counter_decay_rates: Optional[list[float]] = (
        None  # count_decay_rates for each table if eviction strategy is counter
    )
    feature_score_counter_decay_rates: Optional[list[float]] = (
        None  # feature_score_counter_decay_rates for each table if eviction strategy is feature score
    )
    training_id_eviction_trigger_count: Optional[list[int]] = (
        None  # Number of training IDs that, when exceeded, will trigger eviction for each table.
    )
    training_id_keep_count: Optional[list[int]] = (
        None  # Target number of training IDs to retain in each table after eviction.
    )
    l2_weight_thresholds: Optional[list[float]] = (
        None  # l2_weight_thresholds for each table if eviction strategy is feature l2 norm
    )
    threshold_calculation_bucket_stride: Optional[float] = (
        0.2  # The width of each feature score bucket used for threshold calculation in feature score-based eviction.
    )
    threshold_calculation_bucket_num: Optional[int] = (
        1000000  # 1M, Total number of feature score buckets used for threshold calculation in feature score-based eviction.
    )
    interval_for_insufficient_eviction_s: int = (
        # wait at least # seconds before trigger next round of eviction, if last finished eviction is insufficient
        # insufficient means we didn't evict enough rows, so we want to wait longer time to
        # avoid another insufficient eviction
        600
    )
    interval_for_sufficient_eviction_s: int = (
        # wait at least # seconds before trigger next round of eviction, if last finished eviction is sufficient
        60
    )
    interval_for_feature_statistics_decay_s: int = (
        24 * 3600  # 1 day, interval for feature statistics decay
    )
    meta_header_lens: Optional[list[int]] = None  # metaheader length for each table
    eviction_free_mem_threshold_gb: Optional[int] = (
        None  # Minimum free memory (in GB) required before triggering eviction when using free_mem trigger mode.
    )
    eviction_free_mem_check_interval_batch: Optional[int] = (
        None  # Number of batches between checks for free memory threshold when using free_mem trigger mode.
    )
    enable_eviction_for_feature_score_eviction_policy: Optional[list[bool]] = (
        None  # enable eviction if eviction policy is feature score, false means no eviction
    )

    def validate(self) -> None:
        assert self.eviction_trigger_mode in [0, 1, 2, 3, 4, 5], (
            "eviction_trigger_mode must be 0, 1, 2, 3, 4, 5"
            f"actual {self.eviction_trigger_mode}"
        )
        if self.eviction_trigger_mode == 0:
            return

        assert self.eviction_strategy in [0, 1, 2, 3, 4, 5], (
            "eviction_strategy must be 0, 1, 2, 3, 4 or 5, "
            f"actual {self.eviction_strategy}"
        )
        if self.eviction_trigger_mode == 1:
            assert (
                self.eviction_step_intervals is not None
                and self.eviction_step_intervals > 0
            ), (
                "eviction_step_intervals must be positive if eviction_trigger_mode is 1, "
                f"actual {self.eviction_step_intervals}"
            )
        elif self.eviction_trigger_mode == 2:
            assert (
                self.eviction_mem_threshold_gb is not None
            ), "eviction_mem_threshold_gb must be set if eviction_trigger_mode is 2"
        elif self.eviction_trigger_mode == 4:
            assert (
                self.training_id_eviction_trigger_count is not None
            ), "training_id_eviction_trigger_count must be set if eviction_trigger_mode is 4"
        elif self.eviction_trigger_mode == 5:
            assert (
                self.eviction_free_mem_threshold_gb is not None
            ), "eviction_free_mem_threshold_gb must be set if eviction_trigger_mode is 5"
            assert (
                self.eviction_free_mem_check_interval_batch is not None
            ), "eviction_free_mem_check_interval_batch must be set if eviction_trigger_mode is 5"

        if self.eviction_strategy == 0:
            assert self.ttls_in_mins is not None, (
                "ttls_in_mins must be set if eviction_strategy is 0, "
                f"actual {self.ttls_in_mins}"
            )
        elif self.eviction_strategy == 1:
            assert self.counter_thresholds is not None, (
                "counter_thresholds must be set if eviction_strategy is 1, "
                f"actual {self.counter_thresholds}"
            )
            assert self.counter_decay_rates is not None, (
                "counter_decay_rates must be set if eviction_strategy is 1, "
                f"actual {self.counter_decay_rates}"
            )
            assert len(self.counter_thresholds) == len(self.counter_decay_rates), (
                "counter_thresholds and counter_decay_rates must have the same length, "
                f"actual {self.counter_thresholds} vs {self.counter_decay_rates}"
            )
        elif self.eviction_strategy == 2:
            assert self.counter_thresholds is not None, (
                "counter_thresholds must be set if eviction_strategy is 2, "
                f"actual {self.counter_thresholds}"
            )
            assert self.counter_decay_rates is not None, (
                "counter_decay_rates must be set if eviction_strategy is 2, "
                f"actual {self.counter_decay_rates}"
            )
            assert self.ttls_in_mins is not None, (
                "ttls_in_mins must be set if eviction_strategy is 2, "
                f"actual {self.ttls_in_mins}"
            )
            assert len(self.counter_thresholds) == len(self.counter_decay_rates), (
                "counter_thresholds and counter_decay_rates must have the same length, "
                f"actual {self.counter_thresholds} vs {self.counter_decay_rates}"
            )
            assert len(self.counter_thresholds) == len(self.ttls_in_mins), (
                "counter_thresholds and ttls_in_mins must have the same length, "
                f"actual {self.counter_thresholds} vs {self.ttls_in_mins}"
            )
        elif self.eviction_strategy == 5:
            assert self.feature_score_counter_decay_rates is not None, (
                "feature_score_counter_decay_rates must be set if eviction_strategy is 5, "
                f"actual {self.feature_score_counter_decay_rates}"
            )
            assert self.training_id_eviction_trigger_count is not None, (
                "training_id_eviction_trigger_count must be set if eviction_strategy is 5,"
                f"actual {self.training_id_eviction_trigger_count}"
            )
            assert self.training_id_keep_count is not None, (
                "training_id_keep_count must be set if eviction_strategy is 5,"
                f"actual {self.training_id_keep_count}"
            )
            assert self.threshold_calculation_bucket_stride is not None, (
                "threshold_calculation_bucket_stride must be set if eviction_strategy is 5,"
                f"actual {self.threshold_calculation_bucket_stride}"
            )
            assert self.threshold_calculation_bucket_num is not None, (
                "threshold_calculation_bucket_num must be set if eviction_strategy is 5,"
                f"actual {self.threshold_calculation_bucket_num}"
            )
            assert self.enable_eviction_for_feature_score_eviction_policy is not None, (
                "enable_eviction_for_feature_score_eviction_policy must be set if eviction_strategy is 5,"
                f"actual {self.enable_eviction_for_feature_score_eviction_policy}"
            )
            assert (
                len(self.enable_eviction_for_feature_score_eviction_policy)
                == len(self.training_id_keep_count)
                == len(self.feature_score_counter_decay_rates)
            ), (
                "feature_score_thresholds, enable_eviction_for_feature_score_eviction_policy, and training_id_keep_count must have the same length, "
                f"actual {self.training_id_keep_count} vs {self.feature_score_counter_decay_rates} vs {self.enable_eviction_for_feature_score_eviction_policy}"
            )


class KVZCHParams(NamedTuple):
    # global bucket id start and global bucket id end offsets for each logical table,
    # where start offset is inclusive and end offset is exclusive
    bucket_offsets: list[tuple[int, int]] = []
    # bucket size for each logical table
    # the value indicates corresponding input space for each bucket id, e.g. 2^50 / total_num_buckets
    bucket_sizes: list[int] = []
    # enable optimizer offloading or not
    enable_optimizer_offloading: bool = False
    # when enabled, backend will return whole row(metaheader + weight + optimizer) instead of weight only
    # can only be enabled when enable_optimizer_offloading is enabled
    backend_return_whole_row: bool = False
    eviction_policy: EvictionPolicy = EvictionPolicy()
    embedding_cache_mode: bool = False
    load_ckpt_without_opt: bool = False
    optimizer_type_for_st: Optional[str] = None
    optimizer_state_dtypes_for_st: Optional[FrozenSet[Tuple[str, int]]] = None

    def validate(self) -> None:
        assert len(self.bucket_offsets) == len(self.bucket_sizes), (
            "bucket_offsets and bucket_sizes must have the same length, "
            f"actual {self.bucket_offsets} vs {self.bucket_sizes}"
        )
        self.eviction_policy.validate()
        assert (
            not self.backend_return_whole_row or self.enable_optimizer_offloading
        ), "backend_return_whole_row can only be enabled when enable_optimizer_offloading is enabled"


class KVZCHTBEConfig(NamedTuple):
    # Eviction trigger model for kvzch table: 0: disabled, 1: iteration, 2: mem_util, 3: manual, 4: id count, 5: free_mem
    kvzch_eviction_trigger_mode: int = 2  # mem_util
    # Minimum free memory (in GB) required before triggering eviction when using free_mem trigger mode.
    eviction_free_mem_threshold_gb: int = 200  # 200GB
    # Number of batches between checks for free memory threshold when using free_mem trigger mode.
    eviction_free_mem_check_interval_batch: int = 1000
    # The width of each feature score bucket used for threshold calculation in feature score-based eviction.
    threshold_calculation_bucket_stride: float = 0.2
    # Total number of feature score buckets used for threshold calculation in feature score-based eviction.
    threshold_calculation_bucket_num: Optional[int] = 1000000  # 1M
    # When true, we only save weight to kvzch backend and not optimizer state.
    load_ckpt_without_opt: bool = False
    # [DO NOT USE] This is for st publish only, do not set it in your config
    optimizer_type_for_st: Optional[str] = None
    # [DO NOT USE] This is for st publish only, do not set it in your config
    optimizer_state_dtypes_for_st: Optional[FrozenSet[Tuple[str, int]]] = None


class BackendType(enum.IntEnum):
    SSD = 0
    DRAM = 1
    PS = 2

    @classmethod
    # pyre-ignore[3]
    def from_str(cls, key: str):
        lookup = {
            "ssd": BackendType.SSD,
            "dram": BackendType.DRAM,
        }
        if key in lookup:
            return lookup[key]
        else:
            raise ValueError(f"Cannot parse value into BackendType: {key}")


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

    @classmethod
    # pyre-ignore[3]
    def from_str(cls, key: str):
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


class ComputeDevice(enum.IntEnum):
    CPU = 0
    CUDA = 1
    MTIA = 2


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
        ("placements", list[EmbeddingLocation]),
        ("offsets", list[int]),
    ],
)


@dataclass
class CacheState:
    # T + 1 elements and cache_hash_size_cumsum[-1] == total_cache_hash_size
    cache_hash_size_cumsum: list[int]
    cache_index_table_map: list[int]
    total_cache_hash_size: int


def construct_cache_state(
    row_list: list[int],
    location_list: list[EmbeddingLocation],
    feature_table_map: list[int],
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


def get_bounds_check_version_for_platform() -> int:
    # NOTE: Use bounds_check_indices v2 on ROCm because ROCm has a
    # constraint that the gridDim * blockDim has to be smaller than
    # 2^32. The v1 kernel can be launched with gridDim * blockDim >
    # 2^32 while the v2 kernel limits the gridDim size to 64 * # of
    # SMs.  Thus, its gridDim * blockDim is guaranteed to be smaller
    # than 2^32
    return 2 if (torch.cuda.is_available() and torch.version.hip) else 1
