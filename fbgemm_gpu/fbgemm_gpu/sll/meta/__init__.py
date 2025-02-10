#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from fbgemm_gpu.sll.meta.meta_sll import (  # noqa F401
    meta_array_jagged_bmm_jagged_out,
    meta_jagged2_softmax,
    meta_jagged_dense_elementwise_mul_jagged_out,
    meta_jagged_jagged_bmm_jagged_out,
    meta_jagged_self_substraction_jagged_out,
)

# pyre-ignore[5]
op_registrations = {
    "sll_jagged_self_substraction_jagged_out": {
        "Meta": meta_jagged_self_substraction_jagged_out,
    },
    "sll_jagged_dense_elementwise_mul_jagged_out": {
        "Meta": meta_jagged_dense_elementwise_mul_jagged_out,
    },
    "sll_jagged2_softmax": {
        "AutogradMeta": meta_jagged2_softmax,
    },
    "sll_array_jagged_bmm_jagged_out": {
        "AutogradMeta": meta_array_jagged_bmm_jagged_out,
    },
    "sll_jagged_jagged_bmm_jagged_out": {
        "AutogradMeta": meta_jagged_jagged_bmm_jagged_out,
    },
}
