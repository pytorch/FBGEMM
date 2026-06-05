#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

"""
Backward-compatible re-export shell for TBE common types.

All type definitions have been moved to their canonical locations under
``fbgemm_gpu.tbe.config``, ``fbgemm_gpu.tbe.cache.cache_config``, and
``fbgemm_gpu.tbe.ssd.ssd_config``. This module re-exports them so that
existing import paths (``from fbgemm_gpu.split_table_batched_embeddings_ops_common
import ...``) continue to work without modification.
"""

from fbgemm_gpu.tbe.cache import (  # @manual  # noqa: F401
    CacheAlgorithm,
    CacheState,
    MultiPassPrefetchConfig,
)
from fbgemm_gpu.tbe.config import (  # @manual  # noqa: F401
    BoundsCheckMode,
    ComputeDevice,
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
    EmbeddingSpecInfo,
    get_bounds_check_version_for_platform,
    get_new_embedding_location,
    MAX_PREFETCH_DEPTH,
    PoolingMode,
    RecordCacheMetrics,
    round_up,
    SplitState,
    tensor_to_device,
)
from fbgemm_gpu.tbe.ssd import (  # @manual  # noqa: F401
    BackendType,
    EnrichmentPolicy,
    EnrichmentResponseFormat,
    EnrichmentType,
    EvictionPolicy,
    KVZCHParams,
    KVZCHTBEConfig,
)


def construct_cache_state(
    row_list: list[int],
    location_list: list[EmbeddingLocation],
    feature_table_map: list[int],
) -> CacheState:
    """DEPRECATED: Use CacheState.construct() directly."""
    return CacheState.construct(row_list, location_list, feature_table_map)
