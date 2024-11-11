#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from fbgemm_gpu.sll.cpu_sll import (  # noqa F401
    cpu_jagged_dense_bmm,
    cpu_jagged_jagged_bmm,
)

from fbgemm_gpu.sll.triton_sll import jagged_dense_bmm, jagged_jagged_bmm  # noqa F401


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

op_registeration(lib, "sll_jagged_dense_bmm", jagged_dense_bmm, "CUDA")
op_registeration(lib, "sll_jagged_dense_bmm", jagged_dense_bmm, "AutogradCUDA")
op_registeration(lib, "sll_jagged_dense_bmm", cpu_jagged_dense_bmm, "CPU")
op_registeration(lib, "sll_jagged_dense_bmm", cpu_jagged_dense_bmm, "AutogradCPU")
op_registeration(lib, "sll_jagged_jagged_bmm", jagged_jagged_bmm, "CUDA")
op_registeration(lib, "sll_jagged_jagged_bmm", jagged_jagged_bmm, "AutogradCUDA")
op_registeration(lib, "sll_jagged_jagged_bmm", cpu_jagged_jagged_bmm, "CPU")
op_registeration(lib, "sll_jagged_jagged_bmm", cpu_jagged_jagged_bmm, "AutogradCPU")
