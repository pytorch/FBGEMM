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
    w: torch.Tensor, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes an input weight tensor to int4 using preshuffling and scale packing.
    This function is intended to be used with fbgemms mixed dtype kernels and is expected
    to be applied to weights ahead of time. As such, it is not perfectly optimized.

    Args:
        w (Tensor): [N, K] Higher precision weight tensor to quantize. May optionally have a batch dimension.
        group_size (int): Number of elements to calculate group scale for, must be at least 128.
    Returns:
        wq (Tensor): [N, K // 2] Quantized int4 weight tensor packed into int8 elements.
        row_scale (Tensor): [N] FP32 Scale per row of the weight tensor.
        group_scale (Tensor): [K / group_size, 8, N] FP8 Scale per group of the weight tensor.
    """

    def _quantize(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        return wq, row_scale, group_scale

    if w.ndim >= 3:
        orig_shape = w.shape
        # Flatten to 3 dimensions then iterate over batches.
        w = w.view(-1, *w.shape[1:])
        w.unbind(dim=0)
        wq = []
        row_scale = []
        group_scale = []
        for batch in w:
            wq_, row_scale_, group_scale_ = _quantize(batch)
            wq.append(wq_)
            row_scale.append(row_scale_)
            group_scale.append(group_scale_)
        wq = torch.stack(wq).view(*orig_shape[:-2], *wq[0].shape)
        row_scale = torch.stack(row_scale).view(*orig_shape[:-2], *row_scale[0].shape)
        group_scale = torch.stack(group_scale).view(
            *orig_shape[:-2], *group_scale[0].shape
        )
    else:
        wq, row_scale, group_scale = _quantize(w)
    return wq, row_scale, group_scale
