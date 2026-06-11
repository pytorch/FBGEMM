#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Re-export SSD/KVZCH config types from the canonical sub-module (lightweight,
# always available via the :tbe_ssd_config target).
from .ssd_config import (  # noqa: F401
    BackendType,
    EnrichmentPolicy,
    EnrichmentResponseFormat,
    EnrichmentType,
    EvictionPolicy,
    KVZCHParams,
    KVZCHTBEConfig,
)

# `.common`, `.inference`, `.inference_serving`, `.training` ship only with the heavy
# `:ssd_split_table_batched_embeddings_ops` target; the lightweight `:tbe_ssd_config`
# omits them. Guard each import and re-raise anything other than the specific absent
# submodule, so a real circular / "cannot import name" error isn't masked (D107684315).
# Eager (not a lazy `__getattr__`) so the symbols resolve under torch.package.
try:
    # pyre-ignore[21]: `.common` is only present in the heavy target.
    from .common import ASSOC  # noqa: F401
except ModuleNotFoundError as e:
    if e.name != "fbgemm_gpu.tbe.ssd.common":
        raise

try:
    # pyre-ignore[21]: fbgemm_gpu C extensions are not analyzed by Pyre.
    from .inference import SSDIntNBitTableBatchedEmbeddingBags  # noqa: F401
except ModuleNotFoundError as e:
    if e.name != "fbgemm_gpu.tbe.ssd.inference":
        raise

try:
    # pyre-ignore[21]: fbgemm_gpu C extensions are not analyzed by Pyre.
    from .inference_serving import TurboSSDInferenceModule  # noqa: F401
except ModuleNotFoundError as e:
    if e.name != "fbgemm_gpu.tbe.ssd.inference_serving":
        raise

try:
    # pyre-ignore[21]: fbgemm_gpu C extensions are not analyzed by Pyre.
    from .training import DramKvPerfStat, SSDTableBatchedEmbeddingBags  # noqa: F401
except ModuleNotFoundError as e:
    if e.name != "fbgemm_gpu.tbe.ssd.training":
        raise
