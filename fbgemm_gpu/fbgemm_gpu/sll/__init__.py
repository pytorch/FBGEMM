#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Dict

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
    meta_jagged_dense_elementwise_mul_jagged_out,
    meta_jagged_self_substraction_jagged_out,
)

from fbgemm_gpu.sll.meta_sll import (  # noqa F401
    meta_array_jagged_bmm_jagged_out,
    meta_jagged2_softmax,
    meta_jagged_jagged_bmm_jagged_out,
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


def op_registeration(
    lib,  # pyre-ignore[2]
    op_name,  # pyre-ignore[2]
    fn,  # pyre-ignore[2]
    dispatch_key,  # pyre-ignore[2]
) -> None:
    """
    Registers an op with the given name and dispatch key only once.

    Args:
        lib: torch.library  (e.g., torch.library.Library("fbgemm", "FRAGMENT"))
        op_name: operator name
        fn: function that's the operator implementation for the input dispatch key
        dispatch_key: dispatch key that the function should be registered for (e.g., "CUDA")

    Returns:
        None

    Example:
        lib = torch.library.Library("fbgemm", "FRAGMENT")
        lib.define(...)
        op_registeration(lib, "jagged_dense_bmm", jagged_dense_bmm, "CUDA")
    """
    full_op_name = "fbgemm::" + op_name
    if not torch._C._dispatch_has_kernel_for_dispatch_key(full_op_name, dispatch_key):
        if dispatch_key == "Meta":
            lib._register_fake(op_name, fn)
        else:
            lib.impl(op_name, fn, dispatch_key)


lib = torch.library.Library("fbgemm", "FRAGMENT")


# pyre-ignore[24]
def register_sll_op(op_name: str, functors: Dict[str, Callable]) -> None:
    valid_backends = [
        "CUDA",
        "AutogradCUDA",
        "CPU",
        "AutogradCPU",
        "AutogradMeta",
        "Meta",
    ]
    for backend, func in functors.items():
        assert backend in valid_backends
        op_registeration(
            lib,
            op_name,
            func,
            backend,
        )


if "fbgemm::sll_jagged_dense_bmm" not in torch.library._defs:
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

if "fbgemm::sll_jagged_jagged_bmm" not in torch.library._defs:
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

if "fbgemm::sll_dense_jagged_cat_jagged_out" not in torch.library._defs:
    lib.define(
        """sll_dense_jagged_cat_jagged_out(
            Tensor a,
            Tensor b,
            Tensor a_offsets,
            int max_seq_len
        ) -> (Tensor, Tensor)
        """
    )

if "fbgemm::sll_jagged_self_substraction_jagged_out" not in torch.library._defs:
    lib.define(
        """sll_jagged_self_substraction_jagged_out(
            Tensor a,
            Tensor offsets_a,
            Tensor offsets_b,
            int max_seq_len
        ) -> Tensor
        """
    )

if "fbgemm::sll_jagged2_to_padded_dense" not in torch.library._defs:
    lib.define(
        """sll_jagged2_to_padded_dense(
            Tensor values,
            Tensor offsets,
            int max_length,
            float padding_value
        ) -> Tensor
        """
    )

if "fbgemm::sll_jagged_dense_elementwise_mul_jagged_out" not in torch.library._defs:
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

if "fbgemm::sll_jagged_softmax" not in torch.library._defs:
    lib.define(
        """sll_jagged_softmax(Tensor x, Tensor x_offsets, int max_seq_len, bool use_fbgemm_kernel=True) -> Tensor
        """
    )

if "fbgemm::sll_jagged2_softmax" not in torch.library._defs:
    lib.define(
        """sll_jagged2_softmax(Tensor x, Tensor offsets, Tensor offsets_total, int max_seq_len, bool transpose) -> Tensor
        """
    )

if "fbgemm::sll_array_jagged_bmm_jagged_out" not in torch.library._defs:
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

if "fbgemm::sll_jagged_jagged_bmm_jagged_out" not in torch.library._defs:
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

if "fbgemm::sll_jagged_flash_attention_basic" not in torch.library._defs:
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

if "fbgemm::sll_jagged_dense_elementwise_add" not in torch.library._defs:
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

if "fbgemm::sll_jagged_dense_flash_attention" not in torch.library._defs:
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

if "fbgemm::sll_multi_head_jagged_flash_attention" not in torch.library._defs:
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

# NOTE: here we register the op for AutogradCUDA/CPU and CUDA/CPU with the same function
# however, this is not ideal because in the inference case, we don't need the autograd forward
# to save the context because we don't need to do backward.
register_sll_op(
    "sll_jagged_dense_bmm",
    {
        "CUDA": jagged_dense_bmm,
        "AutogradCUDA": jagged_dense_bmm,
        "CPU": cpu_jagged_dense_bmm,
        "AutogradCPU": cpu_jagged_dense_bmm,
    },
)

register_sll_op(
    "sll_jagged_jagged_bmm",
    {
        "CUDA": jagged_jagged_bmm,
        "AutogradCUDA": jagged_jagged_bmm,
        "CPU": cpu_jagged_jagged_bmm,
        "AutogradCPU": cpu_jagged_jagged_bmm,
    },
)

register_sll_op(
    "sll_dense_jagged_cat_jagged_out",
    {
        "CUDA": dense_jagged_cat_jagged_out,
        "CPU": cpu_dense_jagged_cat_jagged_out,
    },
)

register_sll_op(
    "sll_jagged_self_substraction_jagged_out",
    {
        "CUDA": triton_jagged_self_substraction_jagged_out,
        "CPU": cpu_jagged_self_substraction_jagged_out,
        "Meta": meta_jagged_self_substraction_jagged_out,
    },
)

register_sll_op(
    "sll_jagged2_to_padded_dense",
    {
        "CUDA": jagged2_to_padded_dense,
        "AutogradCUDA": jagged2_to_padded_dense,
        "CPU": cpu_jagged2_to_padded_dense,
        "AutogradCPU": cpu_jagged2_to_padded_dense,
    },
)

register_sll_op(
    "sll_jagged_dense_elementwise_mul_jagged_out",
    {
        "CUDA": jagged_dense_elementwise_mul_jagged_out,
        "AutogradCUDA": jagged_dense_elementwise_mul_jagged_out,
        "CPU": cpu_jagged_dense_elementwise_mul_jagged_out,
        "AutogradCPU": cpu_jagged_dense_elementwise_mul_jagged_out,
        "Meta": meta_jagged_dense_elementwise_mul_jagged_out,
    },
)

register_sll_op(
    "sll_jagged_softmax",
    {
        "CUDA": jagged_softmax,
        "AutogradCUDA": jagged_softmax,
        "CPU": cpu_jagged_softmax,
        "AutogradCPU": cpu_jagged_softmax,
    },
)

register_sll_op(
    "sll_jagged2_softmax",
    {
        "CUDA": jagged2_softmax,
        "AutogradCUDA": jagged2_softmax,
        "CPU": cpu_jagged2_softmax,
        "AutogradCPU": cpu_jagged2_softmax,
        "AutogradMeta": meta_jagged2_softmax,
    },
)

register_sll_op(
    "sll_array_jagged_bmm_jagged_out",
    {
        "CUDA": array_jagged_bmm_jagged_out,
        "AutogradCUDA": array_jagged_bmm_jagged_out,
        "CPU": cpu_array_jagged_bmm_jagged_out,
        "AutogradCPU": cpu_array_jagged_bmm_jagged_out,
        "AutogradMeta": meta_array_jagged_bmm_jagged_out,
    },
)

register_sll_op(
    "sll_jagged_jagged_bmm_jagged_out",
    {
        "CUDA": jagged_jagged_bmm_jagged_out,
        "AutogradCUDA": jagged_jagged_bmm_jagged_out,
        "CPU": cpu_jagged_jagged_bmm_jagged_out,
        "AutogradCPU": cpu_jagged_jagged_bmm_jagged_out,
        "AutogradMeta": meta_jagged_jagged_bmm_jagged_out,
    },
)

register_sll_op(
    "sll_jagged_flash_attention_basic",
    {
        "CUDA": jagged_flash_attention_basic,
        "AutogradCUDA": jagged_flash_attention_basic,
        "CPU": cpu_jagged_flash_attention_basic,
        "AutogradCPU": cpu_jagged_flash_attention_basic,
    },
)

register_sll_op(
    "sll_jagged_dense_elementwise_add",
    {
        "CUDA": jagged_dense_elementwise_add,
        "AutogradCUDA": jagged_dense_elementwise_add,
        "CPU": cpu_jagged_dense_elementwise_add,
        "AutogradCPU": cpu_jagged_dense_elementwise_add,
    },
)

register_sll_op(
    "sll_jagged_dense_flash_attention",
    {
        "CUDA": jagged_dense_flash_attention,
        "AutogradCUDA": jagged_dense_flash_attention,
        "CPU": cpu_jagged_dense_flash_attention,
        "AutogradCPU": cpu_jagged_dense_flash_attention,
    },
)

register_sll_op(
    "sll_multi_head_jagged_flash_attention",
    {
        "CUDA": multi_head_jagged_flash_attention,
        "AutogradCUDA": multi_head_jagged_flash_attention,
    },
)
