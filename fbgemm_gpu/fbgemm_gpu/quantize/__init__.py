# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from fbgemm_gpu.quantize.quantize_ops import dequantize_mx, quantize_mx  # noqa F401
from fbgemm_gpu.utils import TorchLibraryFragment

lib = TorchLibraryFragment("fbgemm")

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

lib.define(
    """dequantize_mx(
        Tensor input,
        int mx_group_size
    ) -> Tensor
    """
)

lib.register(
    "quantize_mx",
    {"CUDA": quantize_mx, "CPU": quantize_mx},
)

lib.register(
    "dequantize_mx",
    {"CUDA": dequantize_mx, "CPU": dequantize_mx},
)
