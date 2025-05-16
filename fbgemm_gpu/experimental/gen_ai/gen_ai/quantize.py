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
    # Check that K is divisible by group size.
    assert w.shape[-1] % group_size == 0, "K must be divisible by group size."

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


def scale_nvfp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.
    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).
    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Due to the
    # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
    # So, we first pad the scales to multiples of 128 and 4. Then, the scales
    # (in float8_e4m3fn) are packed into an int32 for every 4 values. More:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
    def round_up(x: int, y: int) -> int:
        return (x + y - 1) // y * y

    rounded_m = round_up(m, 128)
    scale_n = n // block_size
    rounded_n = round_up(scale_n, 4)
    output_scale = torch.empty(
        (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
    )

    torch.ops.fbgemm.scaled_fp4_quant(output, input, output_scale, input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


def _fp32_to_fp4_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Converts a float32 tensor to a unpacked float4 tensor.
    Args:
        x (torch.Tensor): The input float32 tensor.
        ebits (int): The number of bits in the exponent.
        mbits (int): The number of bits in the mantissa.
    Returns:
        torch.Tensor: The resulting unpacked float4 tensor.
    """

    def _n_ones(n: int) -> int:
        return (1 << n) - 1

    EBITS_F32, MBITS_F32 = 8, 23
    F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)

    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _to_blocked(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor to the blocked layout.
    Args:
        x (torch.Tensor): The input tensor in non-blocked layout.
    Returns:
        torch.Tensor: The output tensor in the blocked layout.
    """

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    rows, cols = x.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = x
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=x.device,
            dtype=x.dtype,
        )
        padded[:rows, :cols] = x

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


# This PyTorch version refers to https://github.com/pytorch/ao/blob/v0.10.0/torchao/prototype/mx_formats/mx_tensor.py#L146
def scale_mxfp4_quant(
    x: torch.Tensor, block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.
    Args:
        x (torch.Tensor): The input tensor to be quantized to FP4
        block_size (int): The block size to use for quantization. Default is 32.
    Returns:
        xq (torch.Tensor): Quantized FP4 output tensor
        scale (torch.Tensor): Scale E8M0 tensor
    """

    F4_E2M1_MAX = 6.0
    E8M0_EXPONENT_BIAS = 127
    EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1

    # calculate the scale in e8m0 format
    orig_shape = x.shape
    x = x.reshape(-1, block_size)

    # find max value of the data
    # Note: this only implements the `minimally supported` version of
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3.
    max_abs = torch.amax(torch.abs(x), 1)
    max_pos = F4_E2M1_MAX

    descale = max_abs / max_pos
    scale = torch.where(
        torch.isnan(descale),
        0xFF,  # Handle biased exponent for nan
        # NOTE: descale < (torch.finfo(torch.float32).smallest_normal / 2) is handled through clamping
        (
            torch.clamp(
                torch.ceil(torch.log2(descale)),
                min=-E8M0_EXPONENT_BIAS,
                max=E8M0_EXPONENT_BIAS,
            )
            + E8M0_EXPONENT_BIAS
        ).to(torch.uint8),
    )

    descale_fp = torch.where(
        scale == 0,
        1.0,
        torch.exp2(E8M0_EXPONENT_BIAS - scale.to(torch.float32)),
    )

    # scale and saturated cast the data elements to max of target dtype
    xq = torch.clamp(x * descale_fp.unsqueeze(1), min=-1 * max_pos, max=max_pos)

    xq = xq.reshape(orig_shape)
    xq = _fp32_to_fp4_unpacked(xq, EBITS_F4_E2M1, MBITS_F4_E2M1)
    orig_shape = [*orig_shape[:-1], orig_shape[-1] // 2]

    shape = xq.shape
    assert shape[-1] % 2 == 0
    xq = xq.contiguous().view(-1)
    xq = (xq[::2] << 4 | xq[1::2]).view((*shape[:-1], shape[-1] // 2))

    target_numel = scale.numel() * block_size / 2
    assert target_numel == xq.numel(), f"{target_numel} != {xq.numel()}"

    scale = scale.view(torch.float8_e8m0fnu)
    scale = scale.view(orig_shape[0], -1)
    scale = _to_blocked(scale)

    return xq, scale
