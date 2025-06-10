#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

# pyre-strict

import logging
import os

import torch

try:
    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu import open_source

    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu.docs.version import __version__  # noqa: F401
except Exception:
    open_source: bool = False

if (
    torch.cuda.is_available()
    and torch.version.cuda is not None
    and torch.version.cuda > "12.4"
):
    if open_source:
        torch.ops.load_library(
            os.path.join(os.path.dirname(__file__), "fbgemm_gpu_experimental_hstu.so")
        )
        torch.classes.load_library(
            os.path.join(os.path.dirname(__file__), "fbgemm_gpu_experimental_hstu.so")
        )
    else:
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu/experimental/hstu/src:hstu_ops_gpu_sm80"
        )

        if torch.cuda.get_device_capability() >= (9, 0):
            torch.ops.load_library(
                "//deeplearning/fbgemm/fbgemm_gpu/experimental/hstu/src:hstu_ops_gpu_sm90"
            )

else:
    logging.warning("CUDA is not available for FBGEMM HSTU")


from .cuda_hstu_attention import hstu_attn_varlen_func, HstuAttnVarlenFunc  # noqa: F401
