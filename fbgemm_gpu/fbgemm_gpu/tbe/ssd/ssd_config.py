#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

"""
SSD/KVZCH-specific type definitions for TBE.

Contains EvictionPolicy, EnrichmentType, EnrichmentPolicy, KVZCHParams,
KVZCHTBEConfig, and BackendType.
"""

import enum
from typing import NamedTuple


class EvictionPolicy(NamedTuple):
    eviction_trigger_mode: int = (
        0  # disabled, 0: disabled, 1: iteration, 2: mem_util, 3: manual, 4: id count, 5: free_mem
    )
    eviction_strategy: int = (
        0  # 0: timestamp, 1: counter , 2: counter + timestamp, 3: feature l2 norm 4: timestamp threshold 5: feature score
    )
    eviction_step_intervals: int | None = (
        None  # trigger_step_interval if trigger mode is iteration
    )
    eviction_mem_threshold_gb: int | None = (
        None  # eviction trigger condition if trigger mode is mem_util
    )
    counter_thresholds: list[int] | None = (
        None  # count_thresholds for each table if eviction strategy is counter
    )
    ttls_in_mins: list[int] | None = (
        None  # ttls_in_mins for each table if eviction strategy is timestamp
    )
    counter_decay_rates: list[float] | None = (
        None  # count_decay_rates for each table if eviction strategy is counter
    )
    feature_score_counter_decay_rates: list[float] | None = (
        None  # feature_score_counter_decay_rates for each table if eviction strategy is feature score
    )
    training_id_eviction_trigger_count: list[int] | None = (
        None  # Number of training IDs that, when exceeded, will trigger eviction for each table.
    )
    training_id_keep_count: list[int] | None = (
        None  # Target number of training IDs to retain in each table after eviction.
    )
    l2_weight_thresholds: list[float] | None = (
        None  # l2_weight_thresholds for each table if eviction strategy is feature l2 norm
    )
    threshold_calculation_bucket_stride: float | None = (
        0.2  # The width of each feature score bucket used for threshold calculation in feature score-based eviction.
    )
    threshold_calculation_bucket_num: int | None = (
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
    meta_header_lens: list[int] | None = None  # metaheader length for each table
    eviction_free_mem_threshold_gb: int | None = (
        None  # Minimum free memory (in GB) required before triggering eviction when using free_mem trigger mode.
    )
    eviction_free_mem_check_interval_batch: int | None = (
        None  # Number of batches between checks for free memory threshold when using free_mem trigger mode.
    )
    enable_eviction_for_feature_score_eviction_policy: list[bool] | None = (
        None  # enable eviction if eviction policy is feature score, false means no eviction
    )

    def validate(self) -> None:
        assert self.eviction_trigger_mode in [0, 1, 2, 3, 4, 5], (
            "eviction_trigger_mode must be 0, 1, 2, 3, 4, 5, "
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
        elif self.eviction_strategy == 3:
            assert self.l2_weight_thresholds is not None, (
                "l2_weight_thresholds must be set if eviction_strategy is 3, "
                f"actual {self.l2_weight_thresholds}"
            )
        elif self.eviction_strategy == 4:
            assert self.ttls_in_mins is not None, (
                "ttls_in_mins must be set if eviction_strategy is 4, "
                f"actual {self.ttls_in_mins}"
            )
        elif self.eviction_strategy == 5:
            assert self.feature_score_counter_decay_rates is not None, (
                "feature_score_counter_decay_rates must be set if eviction_strategy is 5, "
                f"actual {self.feature_score_counter_decay_rates}"
            )
            assert self.training_id_eviction_trigger_count is not None, (
                "training_id_eviction_trigger_count must be set if eviction_strategy is 5, "
                f"actual {self.training_id_eviction_trigger_count}"
            )
            assert self.training_id_keep_count is not None, (
                "training_id_keep_count must be set if eviction_strategy is 5, "
                f"actual {self.training_id_keep_count}"
            )
            assert self.threshold_calculation_bucket_stride is not None, (
                "threshold_calculation_bucket_stride must be set if eviction_strategy is 5, "
                f"actual {self.threshold_calculation_bucket_stride}"
            )
            assert self.threshold_calculation_bucket_num is not None, (
                "threshold_calculation_bucket_num must be set if eviction_strategy is 5, "
                f"actual {self.threshold_calculation_bucket_num}"
            )
            assert self.enable_eviction_for_feature_score_eviction_policy is not None, (
                "enable_eviction_for_feature_score_eviction_policy must be set if eviction_strategy is 5, "
                f"actual {self.enable_eviction_for_feature_score_eviction_policy}"
            )
            assert (
                len(self.enable_eviction_for_feature_score_eviction_policy)
                == len(self.training_id_keep_count)
                == len(self.feature_score_counter_decay_rates)
                == len(self.training_id_eviction_trigger_count)
            ), (
                "feature_score_counter_decay_rates, enable_eviction_for_feature_score_eviction_policy, training_id_keep_count, and training_id_eviction_trigger_count must have the same length, "
                f"actual {self.training_id_keep_count} vs {self.feature_score_counter_decay_rates} vs {self.enable_eviction_for_feature_score_eviction_policy} vs {self.training_id_eviction_trigger_count}"
            )


class EnrichmentType(enum.IntEnum):
    IGR_LASER_EMBEDDING = 0
    IGR_LASER_SID = 1
    ONEFLOW_OPENTAB_SID = 2
    ONEFLOW_FEATURE_STORE_SID = 3


class EnrichmentResponseFormat(enum.IntEnum):
    JSON = 0
    THRIFT_FLOAT = 1
    THRIFT_INT64 = 2


class EnrichmentPolicy(NamedTuple):
    # Model and method identifier
    enrichment_type: EnrichmentType = EnrichmentType.IGR_LASER_EMBEDDING
    # External provider name (e.g. Laser provider name)
    provider_name: str = ""
    # Client identifier for the external service
    client_id: str = ""
    # Dimension of data returned by the source
    enrichment_dim: int = 0
    # Deserialization format
    response_format: EnrichmentResponseFormat = EnrichmentResponseFormat.JSON
    # OpenTab/Maple configuration (used for ONEFLOW_OPENTAB_SID)
    opentab_tier_name: str = ""
    opentab_payload_ids: str = ""  # comma-separated, e.g. "31739"
    opentab_payload_types: str = ""  # comma-separated, e.g. "2"
    opentab_column_group_ids: str = ""  # comma-separated, e.g. "12"
    opentab_vec_payload_indexes: str = ""  # comma-separated, e.g. "0"
    opentab_timeout_ms: int = 5000
    opentab_batch_size: int = 100
    # Feature Store configuration (used for ONEFLOW_FEATURE_STORE_SID)
    fs_tier: str = ""
    fs_caller_id: str = ""
    fs_timeout_ms: int = 5000
    fs_batch_size: int = 500
    fs_feature_group_id: int = 0
    fs_feature_group_name: str = ""
    fs_feature_name: str = ""
    # Laser IGR batch size (0 = no batching, send all IDs in one RPC)
    laser_batch_size: int = 0


class KVZCHParams(NamedTuple):
    # global bucket id start and global bucket id end offsets for each logical table,
    # where start offset is inclusive and end offset is exclusive
    bucket_offsets: list[tuple[int, int]] | None = None
    # bucket size for each logical table
    # the value indicates corresponding input space for each bucket id, e.g. 2^50 / total_num_buckets
    bucket_sizes: list[int] | None = None
    # enable optimizer offloading or not
    enable_optimizer_offloading: bool = False
    # when enabled, backend will return whole row(metaheader + weight + optimizer) instead of weight only
    # can only be enabled when enable_optimizer_offloading is enabled
    backend_return_whole_row: bool = False
    eviction_policy: EvictionPolicy = EvictionPolicy()
    embedding_cache_mode: bool = False
    load_ckpt_without_opt: bool = False
    optimizer_type_for_st: str | None = None
    optimizer_state_dtypes_for_st: frozenset[tuple[str, int]] | None = None
    # Enrichment config for embedding cache enrichment from external sources
    enrichment_policy: EnrichmentPolicy | None = None
    feature_score_collection_enabled: bool = False

    def validate(self) -> None:
        offsets = self.bucket_offsets if self.bucket_offsets is not None else []
        sizes = self.bucket_sizes if self.bucket_sizes is not None else []
        assert len(offsets) == len(sizes), (
            "bucket_offsets and bucket_sizes must have the same length, "
            f"actual {offsets} vs {sizes}"
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
    threshold_calculation_bucket_num: int | None = 1000000  # 1M
    # When true, we only save weight to kvzch backend and not optimizer state.
    load_ckpt_without_opt: bool = False
    # [DO NOT USE] This is for st publish only, do not set it in your config
    optimizer_type_for_st: str | None = None
    # [DO NOT USE] This is for st publish only, do not set it in your config
    optimizer_state_dtypes_for_st: frozenset[tuple[str, int]] | None = None
    # Enrichment policy for embedding cache enrichment from external sources (e.g. Laser)
    enrichment_policy: EnrichmentPolicy | None = None


class BackendType(enum.IntEnum):
    SSD = 0
    DRAM = 1
    PS = 2

    @classmethod
    def from_str(cls, key: str) -> "BackendType":
        lookup = {
            "ssd": BackendType.SSD,
            "dram": BackendType.DRAM,
            "ps": BackendType.PS,
        }
        if key in lookup:
            return lookup[key]
        else:
            raise ValueError(f"Cannot parse value into BackendType: {key}")
