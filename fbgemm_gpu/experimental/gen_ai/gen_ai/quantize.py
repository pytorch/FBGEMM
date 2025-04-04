# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Helper functions for using FBGEMM quantized operators.

from typing import Tuple

import torch

from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import quantize_fp8_row


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    # Given int8 x, pack adjacent int4 values into a single int8.
    low_x = x[:, ::2]
    high_x = x[:, 1::2]

    # High bits need to left shift, this also masks off extra bits.
    high_x = torch.bitwise_left_shift(high_x, 4)
    # Low bits need to have sign bits removed.
    low_x = torch.bitwise_and(low_x, 0xF)

    # Recombine into a single value with bitwise or.
    return torch.bitwise_or(low_x, high_x).contiguous()


def int4_row_quantize_zp(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_bit = 4  # Number of target bits.
    to_quant = x.reshape(-1, group_size).to(torch.float)

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int

    zeros = min_val + scales * (2 ** (n_bit - 1))

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)

    # Recenter output and move to int8.
    out = (out - 2 ** (n_bit - 1)).to(dtype=torch.int8).reshape(x.shape)

    # Cutlass expects column major layout for scale and zero point,
    # so we transpose here and make them contiguous.
    scales = scales.view(x.shape[0], -1).t().contiguous()
    zeros = zeros.view(x.shape[0], -1).t().contiguous()

    return out, scales, zeros


def int4_row_quantize(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to quantize a tensor to int4 with groupwise scales.

    Args:
        x (Tensor): [N, K] Higher precision weight tensor to quantize.
        group_size (int): Number of elements to calculate group scale for.
    Returns:
        wq (Tensor): [N, K // 2] Quantized int4 tensor stored in int8 elements.
        group_scale (Tensor): [K / group_size, N] FP32 Scale per group.
    """
    n_bit = 4  # Number of target bits.
    to_quant = x.reshape(-1, group_size).to(torch.float)

    max_val = torch.abs(to_quant).amax(dim=1, keepdim=True)
    max_int = 2 ** (n_bit - 1)
    min_int = -(2 ** (n_bit - 1))
    scales = max_val.clamp(min=1e-6) / max_int

    out = to_quant.div(scales).round().clamp_(min_int, max_int - 1)

    # Cast to int8 and restore shape.
    out = out.to(dtype=torch.int8).reshape(x.shape)

    # Scales should be in [num_groups, N] layout.
    scales = scales.view(x.shape[0], -1).t().contiguous()

    return out, scales


def quantize_int4_preshuffle(
    w: torch.Tensor, group_size: int = 128, dtype: str = "fp8", use_zp: bool = True
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Quantizes an input weight tensor to int4 using preshuffling and scale packing.
    This function is intended to be used with fbgemms mixed dtype kernels and is expected
    to be applied to weights ahead of time. As such, it is not perfectly optimized.

    Args:
        w (Tensor): [N, K] Higher precision weight tensor to quantize. May optionally have a batch dimension.
        group_size (int): Number of elements to calculate group scale for, must be at least 128.
        dtype (torch.dtype): Type of corresponding activations. Must be fp8 or bf16.
        use_zp (bool): If true, uses zero points during weight quantization. Only relevant for bf16 currently.
    Returns:
        wq (Tensor): [N, K // 2] Quantized int4 weight tensor packed into int8 elements.
        scales (Tuple[Tensor]): Scale tensors for the specified activation type. When FP8 is used,
        scales is a tuple of row_scale ([N]) and group_scale ([K / group_size, 8, N]). When BF16 is
        used, scales is a tuple of group_scale([K / group_size, N]) and group_zero ([K / group_size, N])
    """

    def _quantize(
        w: torch.Tensor, dtype: str = "fp8"
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if dtype == "fp8":
            # Start by lowering weights to FP8 and producing row scales.
            wq, row_scale = quantize_fp8_row(w)

            # Now reduce to INT4.
            wq, group_scale = int4_row_quantize(wq, group_size)
            # Reduce group scale to FP8.
            group_scale = group_scale.to(torch.float8_e4m3fn)
            # Take quantized weights and pack them efficiently.
            wq = pack_int4(wq)
            # Finally pack weights and scales into efficient preshuffled format.
            wq, group_scale = torch.ops.fbgemm.preshuffle_i4(wq, group_scale)
            return wq, (group_scale, row_scale)

        elif dtype == "bf16":
            if use_zp:
                wq, group_scale, group_zero = int4_row_quantize_zp(w, group_size)
            else:
                wq, group_scale = int4_row_quantize(w, group_size)
                group_zero = torch.zeros_like(group_scale)
            # Set scales to activation type.
            group_scale = group_scale.to(torch.bfloat16)
            group_zero = group_zero.to(torch.bfloat16)
            # Take quantized weights and pack them efficiently.
            wq = pack_int4(wq)
            # Finally pack weights and scales into efficient preshuffled format.
            wq, group_scale = torch.ops.fbgemm.preshuffle_i4(wq, group_scale)
            return wq, (group_scale, group_zero)
        else:
            raise NotImplementedError("Only fp8 and bf16 activations supported.")

    if w.ndim >= 3:
        orig_shape = w.shape
        # Flatten to 3 dimensions then iterate over batches.
        wq, scales = zip(*[_quantize(i, dtype=dtype) for i in w])
        wq = torch.stack(wq).view(*orig_shape[:-2], *wq[0].shape)
        # Decompose then stack scales back into a tuple.
        a_scales, b_scales = zip(*scales)
        scales = (
            torch.stack(a_scales).view(*orig_shape[:-2], *a_scales[0].shape),
            torch.stack(b_scales).view(*orig_shape[:-2], *b_scales[0].shape),
        )
    else:
        wq, scales = _quantize(w, dtype=dtype)

    return wq, scales
