#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from fbgemm_gpu.sll.cpu_sll import (  # noqa F401
    cpu_dense_jagged_cat_jagged_out,
    cpu_jagged2_to_padded_dense,
    cpu_jagged_dense_bmm,
    cpu_jagged_jagged_bmm,
    cpu_jagged_self_substraction_jagged_out,
    meta_jagged_self_substraction_jagged_out,
)

from fbgemm_gpu.sll.triton_sll import (  # noqa F401
    dense_jagged_cat_jagged_out,
    jagged2_to_padded_dense,
    jagged_dense_bmm,
    jagged_jagged_bmm,
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


op_registeration(lib, "sll_jagged_dense_bmm", jagged_dense_bmm, "CUDA")
op_registeration(lib, "sll_jagged_dense_bmm", jagged_dense_bmm, "AutogradCUDA")
op_registeration(lib, "sll_jagged_dense_bmm", cpu_jagged_dense_bmm, "CPU")
op_registeration(lib, "sll_jagged_dense_bmm", cpu_jagged_dense_bmm, "AutogradCPU")
op_registeration(lib, "sll_jagged_jagged_bmm", jagged_jagged_bmm, "CUDA")
op_registeration(lib, "sll_jagged_jagged_bmm", jagged_jagged_bmm, "AutogradCUDA")
op_registeration(lib, "sll_jagged_jagged_bmm", cpu_jagged_jagged_bmm, "CPU")
op_registeration(lib, "sll_jagged_jagged_bmm", cpu_jagged_jagged_bmm, "AutogradCPU")
op_registeration(
    lib, "sll_dense_jagged_cat_jagged_out", dense_jagged_cat_jagged_out, "CUDA"
)
op_registeration(
    lib, "sll_dense_jagged_cat_jagged_out", cpu_dense_jagged_cat_jagged_out, "CPU"
)
op_registeration(
    lib,
    "sll_jagged_self_substraction_jagged_out",
    triton_jagged_self_substraction_jagged_out,
    "CUDA",
)
op_registeration(
    lib,
    "sll_jagged_self_substraction_jagged_out",
    cpu_jagged_self_substraction_jagged_out,
    "CPU",
)
op_registeration(
    lib,
    "sll_jagged_self_substraction_jagged_out",
    meta_jagged_self_substraction_jagged_out,
    "Meta",
)
op_registeration(lib, "sll_jagged2_to_padded_dense", jagged2_to_padded_dense, "CUDA")
op_registeration(
    lib, "sll_jagged2_to_padded_dense", jagged2_to_padded_dense, "AutogradCUDA"
)
op_registeration(lib, "sll_jagged2_to_padded_dense", cpu_jagged2_to_padded_dense, "CPU")
op_registeration(
    lib, "sll_jagged2_to_padded_dense", cpu_jagged2_to_padded_dense, "AutogradCPU"
)
