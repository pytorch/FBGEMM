#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from fbgemm_gpu.sll.cpu import op_registrations as sll_cpu_registrations
from fbgemm_gpu.sll.meta import op_registrations as sll_meta_registrations
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

for op_name, dispatches in sll_cpu_registrations.items():
    lib.register(op_name, dispatches)

for op_name, dispatches in sll_meta_registrations.items():
    lib.register(op_name, dispatches)

if torch.cuda.is_available():
    from fbgemm_gpu.sll.triton import op_registrations as sll_gpu_registrations

    for op_name, dispatches in sll_gpu_registrations.items():
        lib.register(op_name, dispatches)
