#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
        os.path.join(os.path.dirname(__file__), "fbgemm_gpu_experimental_gen_ai.so")
    )
    torch.classes.load_library(
        os.path.join(os.path.dirname(__file__), "fbgemm_gpu_experimental_gen_ai.so")
    )
else:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:attention_ops"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:comm_ops"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:quantize_ops"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:kv_cache_ops"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:gather_scatter_ops"
    )

    gemm_ops = [
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai/src/quantize/cutlass_extensions:cutlass_bf16bf16bf16_grouped_grad",
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai/src/quantize/cutlass_extensions:cutlass_bf16bf16bf16_grouped_wgrad",
    ]
    for op in gemm_ops:
        try:
            torch.ops.load_library(
                op,
            )
        except OSError:
            pass
