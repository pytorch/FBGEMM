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
from .cutlass_blackwell_fmha_interface import cutlass_blackwell_fmha_func  # noqa: F401
