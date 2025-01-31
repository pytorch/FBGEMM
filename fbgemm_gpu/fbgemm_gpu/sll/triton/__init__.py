#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from fbgemm_gpu.sll.triton.multi_head_jagged_flash_attention import (  # noqa F401
    multi_head_jagged_flash_attention,
    MultiHeadJaggedFlashAttention,
)

op_registrations = {
    "sll_multi_head_jagged_flash_attention": {
        "CUDA": multi_head_jagged_flash_attention,
        "AutogradCUDA": multi_head_jagged_flash_attention,
    },
}
