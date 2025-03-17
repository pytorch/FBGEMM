# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import torch

torch.ops.load_library(
    "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:index_shuffling_ops_gpu"
)

from fbgemm_gpu.experimental.gen_ai.moe import index_shuffling  # noqa F401
