#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from .cache_config import (  # noqa: F401
    CacheAlgorithm,
    CacheState,
    DEFAULT_ASSOC,
    MultiPassPrefetchConfig,
    UVMCacheStatsIndex,
)

# `.split_embeddings_cache_ops` ships only with the heavy `:tbe_cache_ops` BUCK
# target. The lightweight `:tbe_cache_config` target legitimately omits it; guard
# the import so the lightweight target loads cleanly. Eager (not PEP 562 lazy) so
# the symbol resolves under torch.package, whose importer does not honor
# importlib.import_module calls inside a module-level __getattr__.
try:
    # pyre-ignore[21]: `.split_embeddings_cache_ops` is only present in heavy target.
    from .split_embeddings_cache_ops import get_unique_indices_v2  # noqa: F401
except ImportError:
    pass
