#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Re-export SSD/KVZCH config types from the canonical sub-module.
from .ssd_config import (  # noqa: F401
    BackendType,
    EnrichmentPolicy,
    EnrichmentResponseFormat,
    EnrichmentType,
    EvictionPolicy,
    KVZCHParams,
    KVZCHTBEConfig,
)

# `.common`, `.inference`, `.inference_serving`, `.training` ship only with
# the heavy `:ssd_split_table_batched_embeddings_ops` BUCK target. The
# lightweight `:tbe_ssd_config` target legitimately omits them; guard each
# import so the lightweight target loads cleanly.
#
# Eager (not PEP 562 lazy `__getattr__`) because:
#  - PEP 562 module-level `__getattr__` calls `importlib.import_module(...)`,
#    which does not honor torch.package's hermetic namespace and breaks
#    serialized models that re-import these submodules from inside an
#    archive.
#  - The previous circular-import problem (this package -> .training ->
#    fbgemm_gpu.split_embedding_configs.SparseType -> back to the shell)
#    is now broken at the source by having `split_embedding_configs.py`
#    import directly from `fbgemm_gpu.tbe.config` instead of from the
#    backward-compat shell.
#  - Loading `.common` registers the fbgemm SSD C++ ops (e.g.
#    `torch.ops.fbgemm.compact_indices`) via its top-level
#    `load_torch_module(...)` side effect.
try:
    # pyre-ignore[21]: `.common` is only present in the heavy target.
    from .common import ASSOC  # noqa: F401
except ImportError:
    pass

try:
    # pyre-ignore[21]: fbgemm_gpu C extensions are not analyzed by Pyre.
    from .inference import SSDIntNBitTableBatchedEmbeddingBags  # noqa: F401
except ImportError:
    pass

try:
    # pyre-ignore[21]: fbgemm_gpu C extensions are not analyzed by Pyre.
    from .inference_serving import TurboSSDInferenceModule  # noqa: F401
except ImportError:
    pass

try:
    # pyre-ignore[21]: fbgemm_gpu C extensions are not analyzed by Pyre.
    from .training import DramKvPerfStat, SSDTableBatchedEmbeddingBags  # noqa: F401
except ImportError:
    pass
