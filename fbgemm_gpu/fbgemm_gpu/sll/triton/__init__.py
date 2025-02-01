#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from fbgemm_gpu.sll.triton.triton_jagged_dense_elementwise_add import (  # noqa F401
    jagged_dense_elementwise_add,
    JaggedDenseAdd,  # noqa F401
)
from fbgemm_gpu.sll.triton.triton_jagged_dense_flash_attention import (  # noqa F401
    jagged_dense_flash_attention,
    JaggedDenseFlashAttention,  # noqa F401
)

from fbgemm_gpu.sll.triton.triton_multi_head_jagged_flash_attention import (  # noqa F401
    multi_head_jagged_flash_attention,
    MultiHeadJaggedFlashAttention,  # noqa F401
)

# pyre-ignore[5]
op_registrations = {
    "sll_jagged_dense_elementwise_add": {
        "CUDA": jagged_dense_elementwise_add,
        "AutogradCUDA": jagged_dense_elementwise_add,
    },
    "sll_jagged_dense_flash_attention": {
        "CUDA": jagged_dense_flash_attention,
        "AutogradCUDA": jagged_dense_flash_attention,
    },
    "sll_multi_head_jagged_flash_attention": {
        "CUDA": multi_head_jagged_flash_attention,
        "AutogradCUDA": multi_head_jagged_flash_attention,
    },
}
