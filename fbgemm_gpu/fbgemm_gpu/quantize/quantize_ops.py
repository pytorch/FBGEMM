# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Union

import torch

from fbgemm_gpu.quantize_utils import fp32_to_mx4, mx4_to_fp32, RoundingMode


def quantize_mx(
    input: torch.Tensor,
    scale_bits: int = 8,
    elem_ebits: int = 2,
    elem_mbits: int = 3,
    elem_max_norm: float = 6.0,
    mx_group_size: int = 32,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
) -> torch.Tensor:
    """
    Registered quantize_mx ops for E2E comm.
    (registration is done in __init__.py)
    We use Triton implementation for quantization
    Args:
        input: FP32 tensor of size total_elems to be quantized
        scale_bits: num bits of the shared exponent (i.e., 8 for MX4 e2m1)
        elem_ebits: num bits of the exponent (i.e., 2 for MX4 e2m1)
        elem_mbits: num bits of the mantissa incl. sign and implicit bits (
                    i.e., 3 for MX4 e2m1)
        elem_max_norm: max value of the float (i.e., 6.0 for MX4 e2m1)
        mx_group_size: num elements that share the max shared_exponent
        rounding_mode: Which type of rounding to use when calculating shared exponent.

    Return:
        output: MX4 tensor packed into int8 values with size
                (total_elems / 2 + total_elems / groupsize)
                the shared exponent of each group is stored at the last byte
                of output of each group
    """
    return fp32_to_mx4(
        input, mx_group_size, rounding_mode=rounding_mode, use_triton=True
    )


def dequantize_mx(
    input: torch.Tensor,
    mx_group_size: int = 32,
) -> torch.Tensor:
    """
    Registered dequantize_mx ops for E2E comm
    (registration is done in __init__.py to prevent multiple loading)
    We use triton implementation for quantization
    Args:
        input: FP8 tensor (MX4 packed in FP8)
        mx_group_size: number of elements that shares the same max shared_exponent

    Return:
        output: FP32 tensor with total elements (total_elems)
    """
    return mx4_to_fp32(input, mx_group_size, use_triton=True)
