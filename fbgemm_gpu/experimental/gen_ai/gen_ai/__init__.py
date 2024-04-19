#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

# pyre-ignore[16]
if open_source:
    torch.ops.load_library(
        os.path.join(os.path.dirname(__file__), "fbgemm_gpu_experimental_gen_ai_py.so")
    )
else:
    if torch.version.hip:
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:attention_ops_hip"
        )
    else:
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:attention_ops_cuda"
        )
