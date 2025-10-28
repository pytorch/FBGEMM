# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os

import torch

try:
    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu import open_source
except Exception:
    open_source: bool = False

# pyre-ignore[16]
if open_source:
    torch.ops.load_library(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "fbgemm_gpu_experimental_gen_ai.so",
        )
    )
    torch.classes.load_library(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "fbgemm_gpu_experimental_gen_ai.so",
        )
    )
else:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:index_shuffling_ops"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:gather_scatter_ops"
    )

index_shuffling = None
gather_along_first_dim = None
scatter_add_along_first_dim = None

if torch.cuda.is_available():
    index_shuffling = torch.ops.fbgemm.index_shuffling  # noqa F401
    if not torch.version.hip:
        # SM90 support
        gather_along_first_dim = torch.ops.fbgemm.gather_along_first_dim  # noqa F401
        scatter_add_along_first_dim = torch.ops.fbgemm.scatter_add_along_first_dim  # noqa F401

from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import (  # noqa F401
    grouped_gemm,
    grouped_gemm_fp8_rowwise,
)

from .activation import silu_mul, silu_mul_quant  # noqa F401

from .gather_scatter import (  # noqa F401
    gather_scale_dense_tokens,
    gather_scale_quant_dense_tokens,
    scatter_add_dense_tokens,
    scatter_add_padded_tokens,
)
from .shuffling import combine_shuffling, split_shuffling  # noqa F401
