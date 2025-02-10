#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from fbgemm_gpu.sll.cpu.cpu_sll import (  # noqa F401
    cpu_array_jagged_bmm_jagged_out,
    cpu_array_jagged_bmm_jagged_out_kernel,  # noqa F401
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
    cpu_jagged_jagged_bmm_jagged_out_kernel,  # noqa F401
    cpu_jagged_self_substraction_jagged_out,
    cpu_jagged_softmax,
)

# pyre-ignore[5]
op_registrations = {
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
    },
    "sll_jagged2_to_padded_dense": {
        "CPU": cpu_jagged2_to_padded_dense,
        "AutogradCPU": cpu_jagged2_to_padded_dense,
    },
    "sll_jagged_dense_elementwise_mul_jagged_out": {
        "CPU": cpu_jagged_dense_elementwise_mul_jagged_out,
        "AutogradCPU": cpu_jagged_dense_elementwise_mul_jagged_out,
    },
    "sll_jagged_softmax": {
        "CPU": cpu_jagged_softmax,
        "AutogradCPU": cpu_jagged_softmax,
    },
    "sll_jagged2_softmax": {
        "CPU": cpu_jagged2_softmax,
        "AutogradCPU": cpu_jagged2_softmax,
    },
    "sll_array_jagged_bmm_jagged_out": {
        "CPU": cpu_array_jagged_bmm_jagged_out,
        "AutogradCPU": cpu_array_jagged_bmm_jagged_out,
    },
    "sll_jagged_jagged_bmm_jagged_out": {
        "CPU": cpu_jagged_jagged_bmm_jagged_out,
        "AutogradCPU": cpu_jagged_jagged_bmm_jagged_out,
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
