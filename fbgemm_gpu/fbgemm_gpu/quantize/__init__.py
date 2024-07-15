# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from fbgemm_gpu.quantize.quantize_ops import dequantize_mx, quantize_mx  # noqa F401


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
        op_registeration(lib, "quantize_mx", quantize_mx, "CUDA")
    """
    full_op_name = "fbgemm::" + op_name
    if not torch._C._dispatch_has_kernel_for_dispatch_key(full_op_name, dispatch_key):
        lib.impl(op_name, fn, dispatch_key)


lib = torch.library.Library("fbgemm", "FRAGMENT")

if "fbgemm::quantize_mx" not in torch.library._defs:
    lib.define(
        """quantize_mx(
            Tensor input,
            int scale_bits,
            int elem_ebits,
            int elem_mbits,
            float elem_max_norm,
            int mx_group_size,
            int? rounding_mode = None
        ) -> Tensor
        """
    )

if "fbgemm::dequantize_mx" not in torch.library._defs:
    lib.define(
        """dequantize_mx(
            Tensor input,
            int mx_group_size
        ) -> Tensor
        """
    )

op_registeration(lib, "quantize_mx", quantize_mx, "CUDA")
op_registeration(lib, "quantize_mx", quantize_mx, "CPU")
op_registeration(lib, "dequantize_mx", dequantize_mx, "CUDA")
op_registeration(lib, "dequantize_mx", dequantize_mx, "CPU")
