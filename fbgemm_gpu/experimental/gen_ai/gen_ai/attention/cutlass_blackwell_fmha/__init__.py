# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

try:
    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu import open_source
except Exception:
    open_source: bool = False

if open_source:
    import os

    torch.ops.load_library(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "..",
            "fbgemm_gpu_experimental_gen_ai.so",
        )
    )
else:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:blackwell_attention_ops_gpu"
    )

from . import cutlass_blackwell_fmha_custom_op  # noqa: F401
from .cutlass_blackwell_fmha_interface import (  # noqa: F401
    _cutlass_blackwell_fmha_forward,
    cutlass_blackwell_fmha_decode_forward,
    cutlass_blackwell_fmha_func,
)

# Note: _cutlass_blackwell_fmha_forward is an internal function (indicated by leading underscore)
# that is exported here specifically for testing purposes. It allows tests to access the LSE
# (log-sum-exp) values returned by the forward pass without modifying the public API.
# Production code should use cutlass_blackwell_fmha_func instead.
__all__ = [
    "_cutlass_blackwell_fmha_forward",
    "cutlass_blackwell_fmha_decode_forward",
    "cutlass_blackwell_fmha_func",
]
