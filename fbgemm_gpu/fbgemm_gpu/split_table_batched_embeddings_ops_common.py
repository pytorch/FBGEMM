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

Canonical definitions live in the ``fbgemm_gpu.tbe.*`` leaf modules
(``tbe.config.embedding_config``, ``tbe.cache.cache_config``,
``tbe.ssd.ssd_config``); this module re-exports them so the legacy
``from fbgemm_gpu.split_table_batched_embeddings_ops_common import ...`` path keeps
working. ``__module__`` is intentionally not pinned (pinning breaks
``torch.jit.script``, which resolves a type's source via ``cls.__module__``).
"""

from typing import TYPE_CHECKING

from fbgemm_gpu.tbe.cache.cache_config import (  # noqa: F401
    CacheAlgorithm,
    CacheState,
    MultiPassPrefetchConfig,
)
from fbgemm_gpu.tbe.config.embedding_config import (  # noqa: F401
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

# SSD/KVZCH types are re-exported lazily (see ``__getattr__`` below). Importing
# ``tbe.ssd.ssd_config`` eagerly here triggers a tbe.ssd -> .training -> ops_training
# import cycle (D107684315); ``TYPE_CHECKING`` keeps static types without that import.
if TYPE_CHECKING:
    from fbgemm_gpu.tbe.ssd.ssd_config import (  # noqa: F401
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
    """Backward-compatible free-function form of CacheState.construct()."""
    return CacheState.construct(row_list, location_list, feature_table_map)


# Lazy re-export of the SSD/KVZCH config types (canonical in tbe.ssd.ssd_config);
# deferring to first access avoids the import cycle noted above.
_SSD_CONFIG_REEXPORTS: tuple[str, ...] = (
    "BackendType",
    "EnrichmentPolicy",
    "EnrichmentResponseFormat",
    "EnrichmentType",
    "EvictionPolicy",
    "KVZCHParams",
    "KVZCHTBEConfig",
)


def __getattr__(name: str) -> object:
    if name in _SSD_CONFIG_REEXPORTS:
        from fbgemm_gpu.tbe.ssd import ssd_config

        value = getattr(ssd_config, name)
        globals()[name] = value  # cache; subsequent access bypasses __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_SSD_CONFIG_REEXPORTS})
