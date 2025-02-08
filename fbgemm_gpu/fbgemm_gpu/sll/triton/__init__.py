#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from fbgemm_gpu.sll.triton.triton_dense_jagged_cat_jagged_out import (
    dense_jagged_cat_jagged_out,
)

from fbgemm_gpu.sll.triton.triton_jagged2_to_padded_dense import (  # noqa F401
    jagged2_to_padded_dense,
    Jagged2ToPaddedDense,  # noqa F401
)

from fbgemm_gpu.sll.triton.triton_jagged_bmm import (  # noqa F401
    jagged_dense_bmm,
    jagged_jagged_bmm,
    JaggedDenseBmm,  # noqa F401
    JaggedJaggedBmm,  # noqa F401
)

from fbgemm_gpu.sll.triton.triton_jagged_bmm_jagged_out import (  # noqa F401
    array_jagged_bmm_jagged_out,
    ArrayJaggedBmmNopadding,  # noqa F401
    jagged_jagged_bmm_jagged_out,
    JaggedJaggedBmmNoPadding,  # noqa F401
    triton_array_jagged_bmm_jagged_out,  # noqa F401
    triton_jagged_jagged_bmm_jagged_out,  # noqa F401
)

from fbgemm_gpu.sll.triton.triton_jagged_dense_elementwise_add import (  # noqa F401
    jagged_dense_elementwise_add,
    JaggedDenseAdd,  # noqa F401
)

from fbgemm_gpu.sll.triton.triton_jagged_dense_elementwise_mul_jagged_out import (  # noqa F401
    jagged_dense_elementwise_mul_jagged_out,
    JaggedDenseElementwiseMul,  # noqa F401
)

from fbgemm_gpu.sll.triton.triton_jagged_dense_flash_attention import (  # noqa F401
    jagged_dense_flash_attention,
    JaggedDenseFlashAttention,  # noqa F401
)

from fbgemm_gpu.sll.triton.triton_jagged_flash_attention_basic import (  # noqa F401
    jagged_flash_attention_basic,
    JaggedFlashAttentionBasic,  # noqa F401
)

from fbgemm_gpu.sll.triton.triton_jagged_self_substraction_jagged_out import (
    triton_jagged_self_substraction_jagged_out,
)

from fbgemm_gpu.sll.triton.triton_jagged_softmax import (  # noqa F401
    jagged2_softmax,
    Jagged2Softmax,  # noqa F401
    jagged_softmax,
    JaggedSoftmax,  # noqa F401
)

from fbgemm_gpu.sll.triton.triton_multi_head_jagged_flash_attention import (  # noqa F401
    multi_head_jagged_flash_attention,
    MultiHeadJaggedFlashAttention,  # noqa F401
)

# pyre-ignore[5]
op_registrations = {
    "sll_dense_jagged_cat_jagged_out": {
        "CUDA": dense_jagged_cat_jagged_out,
    },
    "sll_jagged_dense_bmm": {
        "CUDA": jagged_dense_bmm,
        "AutogradCUDA": jagged_dense_bmm,
    },
    "sll_jagged_jagged_bmm": {
        "CUDA": jagged_jagged_bmm,
        "AutogradCUDA": jagged_jagged_bmm,
    },
    "sll_jagged2_to_padded_dense": {
        "CUDA": jagged2_to_padded_dense,
        "AutogradCUDA": jagged2_to_padded_dense,
    },
    "sll_array_jagged_bmm_jagged_out": {
        "CUDA": array_jagged_bmm_jagged_out,
        "AutogradCUDA": array_jagged_bmm_jagged_out,
    },
    "sll_jagged_jagged_bmm_jagged_out": {
        "CUDA": jagged_jagged_bmm_jagged_out,
        "AutogradCUDA": jagged_jagged_bmm_jagged_out,
    },
    "sll_jagged_softmax": {
        "CUDA": jagged_softmax,
        "AutogradCUDA": jagged_softmax,
    },
    "sll_jagged2_softmax": {
        "CUDA": jagged2_softmax,
        "AutogradCUDA": jagged2_softmax,
    },
    "sll_jagged_dense_elementwise_add": {
        "CUDA": jagged_dense_elementwise_add,
        "AutogradCUDA": jagged_dense_elementwise_add,
    },
    "sll_jagged_dense_flash_attention": {
        "CUDA": jagged_dense_flash_attention,
        "AutogradCUDA": jagged_dense_flash_attention,
    },
    "sll_jagged_flash_attention_basic": {
        "CUDA": jagged_flash_attention_basic,
        "AutogradCUDA": jagged_flash_attention_basic,
    },
    "sll_multi_head_jagged_flash_attention": {
        "CUDA": multi_head_jagged_flash_attention,
        "AutogradCUDA": multi_head_jagged_flash_attention,
    },
    "sll_jagged_self_substraction_jagged_out": {
        "CUDA": triton_jagged_self_substraction_jagged_out,
    },
    "sll_jagged_dense_elementwise_mul_jagged_out": {
        "CUDA": jagged_dense_elementwise_mul_jagged_out,
        "AutogradCUDA": jagged_dense_elementwise_mul_jagged_out,
    },
}
