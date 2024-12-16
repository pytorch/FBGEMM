# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .common import add_docs

add_docs(
    torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf,
    """
FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(input, bit_rate) -> Tensor

Convert FP32/16 to INT8/4/2 using rowwise quantization.

Args:
    input (Tensor): An input tensor. Must be either FP32 (`torch.float`)
        or FP16 (`torch.half`) and must be 2 dimensions.

    bit_rate (int): Quantized bit rate (2 for INT2, 4 for INT4, or 8 for
        INT8)

Returns:
    Quantized output (Tensor). Data type is `torch.uint8` (byte type)

**Example:**

    >>> # Randomize input
    >>> input = torch.randn(2, 4, dtype=torch.float32, device="cuda")
    >>> print(input)
    tensor([[ 0.8247,  0.0031, -1.0068, -1.2081],
            [ 0.5427,  1.5772,  1.0291, -0.7626]], device='cuda:0')
    >>> # Quantize
    >>> output = torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(input, bit_rate=4)
    >>> print(output)
    tensor([[159,   1,  86,  48, 213, 188],
            [248,  11, 254,  48,  26, 186]], device='cuda:0', dtype=torch.uint8)
    """,
)
