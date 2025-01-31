#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from fbgemm_gpu.sll.cpu_sll import (  # noqa F401
    cpu_array_jagged_bmm_jagged_out,
    cpu_dense_jagged_cat_jagged_out,
    cpu_jagged2_softmax,
    cpu_jagged2_to_padded_dense,
    cpu_jagged_dense_bmm,
    cpu_jagged_dense_elementwise_add,
    cpu_jagged_dense_elementwise_mul_jagged_out,
    cpu_jagged_dense_flash_attention,
    cpu_jagged_flash_attention_basic,
    cpu_jagged_jagged_bmm,
    cpu_jagged_jagged_bmm_jagged_out,
    cpu_jagged_self_substraction_jagged_out,
    cpu_jagged_softmax,
)

from fbgemm_gpu.sll.meta_sll import (  # noqa F401
    meta_array_jagged_bmm_jagged_out,
    meta_jagged2_softmax,
    meta_jagged_dense_elementwise_mul_jagged_out,
    meta_jagged_jagged_bmm_jagged_out,
    meta_jagged_self_substraction_jagged_out,
)

from fbgemm_gpu.sll.triton_sll import (  # noqa F401
    array_jagged_bmm_jagged_out,
    dense_jagged_cat_jagged_out,
    jagged2_softmax,
    jagged2_to_padded_dense,
    jagged_dense_bmm,
    jagged_dense_elementwise_add,
    jagged_dense_elementwise_mul_jagged_out,
    jagged_dense_flash_attention,
    jagged_flash_attention_basic,
    jagged_jagged_bmm,
    jagged_jagged_bmm_jagged_out,
    jagged_softmax,
    multi_head_jagged_flash_attention,
    triton_jagged_self_substraction_jagged_out,
)

from fbgemm_gpu.utils import TorchLibraryFragment

lib = TorchLibraryFragment("fbgemm")

lib.define(
    """sll_jagged_dense_bmm(
        Tensor x,
        Tensor y,
        Tensor x_offsets,
        int N,
        bool allow_tf32,
        bool use_fbgemm_kernel=True
    ) -> Tensor
    """
)

lib.define(
    """sll_jagged_jagged_bmm(
        Tensor x,
        Tensor y,
        Tensor x_offsets,
        int N,
        bool allow_tf32,
        bool use_fbgemm_kernel=True
    ) -> Tensor
    """
)

lib.define(
    """sll_dense_jagged_cat_jagged_out(
        Tensor a,
        Tensor b,
        Tensor a_offsets,
        int max_seq_len
    ) -> (Tensor, Tensor)
    """
)

lib.define(
    """sll_jagged_self_substraction_jagged_out(
        Tensor a,
        Tensor offsets_a,
        Tensor offsets_b,
        int max_seq_len
    ) -> Tensor
    """
)

lib.define(
    """sll_jagged2_to_padded_dense(
        Tensor values,
        Tensor offsets,
        int max_length,
        float padding_value
    ) -> Tensor
    """
)

lib.define(
    """sll_jagged_dense_elementwise_mul_jagged_out(
        Tensor x,
        Tensor y,
        Tensor x_seq_lengths,
        Tensor x_offsets,
        int max_seq_len
    ) -> Tensor
    """
)

lib.define(
    """sll_jagged_softmax(Tensor x, Tensor x_offsets, int max_seq_len, bool use_fbgemm_kernel=True) -> Tensor
    """
)

lib.define(
    """sll_jagged2_softmax(Tensor x, Tensor offsets, Tensor offsets_total, int max_seq_len, bool transpose) -> Tensor
    """
)

lib.define(
    """sll_array_jagged_bmm_jagged_out(
        Tensor x,
        Tensor y,
        Tensor x_lengths,
        Tensor x_offsets,
        Tensor y_lengths,
        Tensor y_offsets,
        Tensor z_lengths,
        Tensor z_offsets,
        int max_seq_len,
        bool allow_tf32
    ) -> Tensor
    """
)

lib.define(
    """sll_jagged_jagged_bmm_jagged_out(
        Tensor x,
        Tensor y,
        Tensor x_lengths,
        Tensor x_offsets,
        Tensor y_lengths,
        Tensor y_offsets,
        Tensor z_lengths,
        Tensor z_offsets,
        int max_seq_len,
        bool allow_tf32
    ) -> Tensor
    """
)

lib.define(
    """sll_jagged_flash_attention_basic(
        Tensor q_weights,
        Tensor k_weights,
        Tensor v_weights,
        Tensor offsets,
        int max_seq_len,
        bool use_mask=False,
        bool allow_tf32=True
    ) -> Tensor
    """
)

lib.define(
    """sll_jagged_dense_elementwise_add(
        Tensor x,
        Tensor x_offsets,
        Tensor y,
        int max_seq_len,
        bool use_fbgemm_kernel=True
    ) -> Tensor
    """
)

lib.define(
    """sll_jagged_dense_flash_attention(
        Tensor q_weights,
        Tensor k_weights,
        Tensor v_weights,
        Tensor attn_bias,
        Tensor offsets,
        int max_seq_len,
        bool allow_tf32=True
    ) -> Tensor
    """
)

lib.define(
    """sll_multi_head_jagged_flash_attention(
        Tensor q_weights,
        Tensor k_weights,
        Tensor v_weights,
        Tensor offsets,
        int max_seq_len,
        bool allow_tf32=True
    ) -> Tensor
    """
)

# NOTE: here we register the op for AutogradCUDA/CPU and CUDA/CPU with the same
# function however, this is not ideal because in the inference case, we don't
# need the autograd forward to save the context because we don't need to do
# backward.

# pyre-ignore[5]
sll_cpu_registrations = {
    "sll_jagged_dense_bmm": {
        "CPU": cpu_jagged_dense_bmm,
        "AutogradCPU": cpu_jagged_dense_bmm,
    },
    "sll_jagged_jagged_bmm": {
        "CPU": cpu_jagged_jagged_bmm,
        "AutogradCPU": cpu_jagged_jagged_bmm,
    },
    "sll_dense_jagged_cat_jagged_out": {
        "CPU": cpu_dense_jagged_cat_jagged_out,
    },
    "sll_jagged_self_substraction_jagged_out": {
        "CPU": cpu_jagged_self_substraction_jagged_out,
        "Meta": meta_jagged_self_substraction_jagged_out,
    },
    "sll_jagged2_to_padded_dense": {
        "CPU": cpu_jagged2_to_padded_dense,
        "AutogradCPU": cpu_jagged2_to_padded_dense,
    },
    "sll_jagged_dense_elementwise_mul_jagged_out": {
        "CPU": cpu_jagged_dense_elementwise_mul_jagged_out,
        "AutogradCPU": cpu_jagged_dense_elementwise_mul_jagged_out,
        "Meta": meta_jagged_dense_elementwise_mul_jagged_out,
    },
    "sll_jagged_softmax": {
        "CPU": cpu_jagged_softmax,
        "AutogradCPU": cpu_jagged_softmax,
    },
    "sll_jagged2_softmax": {
        "CPU": cpu_jagged2_softmax,
        "AutogradCPU": cpu_jagged2_softmax,
        "AutogradMeta": meta_jagged2_softmax,
    },
    "sll_array_jagged_bmm_jagged_out": {
        "CPU": cpu_array_jagged_bmm_jagged_out,
        "AutogradCPU": cpu_array_jagged_bmm_jagged_out,
        "AutogradMeta": meta_array_jagged_bmm_jagged_out,
    },
    "sll_jagged_jagged_bmm_jagged_out": {
        "CPU": cpu_jagged_jagged_bmm_jagged_out,
        "AutogradCPU": cpu_jagged_jagged_bmm_jagged_out,
        "AutogradMeta": meta_jagged_jagged_bmm_jagged_out,
    },
    "sll_jagged_flash_attention_basic": {
        "CPU": cpu_jagged_flash_attention_basic,
        "AutogradCPU": cpu_jagged_flash_attention_basic,
    },
    "sll_jagged_dense_elementwise_add": {
        "CPU": cpu_jagged_dense_elementwise_add,
        "AutogradCPU": cpu_jagged_dense_elementwise_add,
    },
    "sll_jagged_dense_flash_attention": {
        "CPU": cpu_jagged_dense_flash_attention,
        "AutogradCPU": cpu_jagged_dense_flash_attention,
    },
}

# pyre-ignore[5]
sll_gpu_registrations = {
    "sll_jagged_dense_bmm": {
        "CUDA": jagged_dense_bmm,
        "AutogradCUDA": jagged_dense_bmm,
    },
    "sll_jagged_jagged_bmm": {
        "CUDA": jagged_jagged_bmm,
        "AutogradCUDA": jagged_jagged_bmm,
    },
    "sll_dense_jagged_cat_jagged_out": {
        "CUDA": dense_jagged_cat_jagged_out,
    },
    "sll_jagged_self_substraction_jagged_out": {
        "CUDA": triton_jagged_self_substraction_jagged_out,
    },
    "sll_jagged2_to_padded_dense": {
        "CUDA": jagged2_to_padded_dense,
        "AutogradCUDA": jagged2_to_padded_dense,
    },
    "sll_jagged_dense_elementwise_mul_jagged_out": {
        "CUDA": jagged_dense_elementwise_mul_jagged_out,
        "AutogradCUDA": jagged_dense_elementwise_mul_jagged_out,
    },
    "sll_jagged_softmax": {
        "CUDA": jagged_softmax,
        "AutogradCUDA": jagged_softmax,
    },
    "sll_jagged2_softmax": {
        "CUDA": jagged2_softmax,
        "AutogradCUDA": jagged2_softmax,
    },
    "sll_array_jagged_bmm_jagged_out": {
        "CUDA": array_jagged_bmm_jagged_out,
        "AutogradCUDA": array_jagged_bmm_jagged_out,
    },
    "sll_jagged_jagged_bmm_jagged_out": {
        "CUDA": jagged_jagged_bmm_jagged_out,
        "AutogradCUDA": jagged_jagged_bmm_jagged_out,
    },
    "sll_jagged_flash_attention_basic": {
        "CUDA": jagged_flash_attention_basic,
        "AutogradCUDA": jagged_flash_attention_basic,
    },
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

for op_name, dispatches in sll_cpu_registrations.items():
    lib.register(op_name, dispatches)

if torch.cuda.is_available():
    for op_name, dispatches in sll_gpu_registrations.items():
        lib.register(op_name, dispatches)
