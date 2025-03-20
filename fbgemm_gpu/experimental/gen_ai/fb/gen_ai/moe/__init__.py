# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import torch

torch.ops.load_library(
    "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:index_shuffling_ops_gpu"
)

index_shuffling = torch.ops.fbgemm.index_shuffling  # noqa F401
from .gather_scatter import (  # noqa F401
    gather_scale_dense_tokens,
    scatter_add_padded_tokens,
)
from .shuffling import combine_shuffling, split_shuffling  # noqa F401
