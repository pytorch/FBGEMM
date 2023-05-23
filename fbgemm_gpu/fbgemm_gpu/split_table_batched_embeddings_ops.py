#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]
# flake8: noqa F401

import torch  # usort:skip
import warnings

# This module is a compatibility wrapper that re-exports the symbols from:
#   fbgemm_gpu.split_table_batched_embeddings_ops_common
#   fbgemm_gpu.split_table_batched_embeddings_ops_inference
#   fbgemm_gpu.split_table_batched_embeddings_ops_training

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheState,
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
    PoolingMode,
    RecordCacheMetrics,
    SplitState,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    align_to_cacheline,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    round_up,
    rounded_row_size_in_bytes,
    unpadded_row_size_in_bytes,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    CounterBasedRegularizationDefinition,
    CounterWeightDecayMode,
    DEFAULT_ASSOC,
    DenseTableBatchedEmbeddingBagsCodegen,
    GradSumDecay,
    INT8_EMB_ROW_DIM_OFFSET,
    LearningRateMode,
    SplitTableBatchedEmbeddingBagsCodegen,
    TailIdThreshold,
    WeightDecayMode,
)

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cpu")
except Exception:
    pass

warnings.warn(
    f"""\033[93m
    The Python module {__name__} is now DEPRECATED and will be removed in the
    future.  Users should instead declare dependencies on
    //deeplearning/fbgemm/fbgemm_gpu/split_table_batched_embeddings_ops_{{training, inference}}
    in their TARGETS file and import the
    fbgemm_gpu.split_table_batched_embeddings_ops_{{training, inference}}
    modules as needed in their scripts.
    \033[0m""",
    DeprecationWarning,
)
