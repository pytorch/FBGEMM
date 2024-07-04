#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import math

import torch
import triton  # @manual

import triton.language as tl  # @manual
from triton import Config  # @manual


@triton.autotune(
    configs=[
        Config({"BLOCK_SIZE": 512}),
        Config({"BLOCK_SIZE": 1024}),
        Config({"BLOCK_SIZE": 2048}),
        Config({"BLOCK_SIZE": 4096}),
        Config({"BLOCK_SIZE": 8192}),
    ],
    key=["K"],
)
@triton.jit
def _kernel_quantize_mx4(
    A,
    shared_exp,
    out,
    M,
    K,
    stride_am,
    stride_ak,
    stride_exp_m,
    stride_exp_k,
    stride_out_m,
    stride_out_k,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Quantize a float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M, K] float tensor to be quantized.
        shared_exp (Tensor): [M / group_size] output containing shared exponent.
        out (Tensor): [M, K / 2] output containing packed mx4 values.
        M (int): Number of rows.
        K (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_exp_m (int): Stride of shared exponent in m dimension.
        stride_exp_k (int): Stride of shared exponent in k dimension.
        stride_out_m (int): Stride of output in m dimension.
        stride_out_k (int): Stride of output in k dimension.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        BLOCK_SIZE (int): Size of each block.
    """
    pid = tl.program_id(0)
    # Initiate offset ranges used in kernel.
    k_offset = tl.arange(0, BLOCK_SIZE)
    packed_offset = tl.arange(0, BLOCK_SIZE // 2)
    group_offset = tl.arange(0, BLOCK_SIZE // GROUP_SIZE)

    # Define Constant Expressions.
    FP32_EXP_MASK: tl.constexpr = 0x7F800000  # type: ignore[Incompatible variable type]
    FP32_EXP_OFFSET: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
    FP32_SIGN_OFFSET: tl.constexpr = 31  # type: ignore[Incompatible variable type]
    SIGN_MASK: tl.constexpr = 0x1  # type: ignore[Incompatible variable type]
    FP32_MANTISSA_MASK: tl.constexpr = 0x007FFFFF  # type: ignore[Incompatible variable type]
    # FP4 has 2 mantissa bits, one explicit one implicit.
    MBITS: tl.constexpr = 2  # type: ignore[Incompatible variable type]
    FP4_EXP_BIAS: tl.constexpr = 1  # type: ignore[Incompatible variable type]
    MAX_FP32_MANTISSA_BITS: tl.constexpr = 24  # type: ignore[Incompatible variable type]
    IMPLIED_1_BIT: tl.constexpr = 1 << 23  # type: ignore[Incompatible variable type]
    OVERFLOW_THRESHOLD: tl.constexpr = 4  # type: ignore[Incompatible variable type]

    # First we need to compute shared exponent.
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE)):
        # Load a block of values.
        a = tl.load(
            A + pid * stride_am + k_offset * stride_ak,
            mask=k_offset < K,
            other=-float("inf"),
        )

        # Scaling step
        ##############

        # View the block in terms of groups.
        a_groups = tl.reshape(a, [BLOCK_SIZE // GROUP_SIZE, GROUP_SIZE])
        # Compute the shared exponent of each group.
        group_max = tl.max(tl.abs(a_groups), axis=1)
        # Convert max to exponent via bit operations.
        group_exp = group_max.to(tl.int32, bitcast=True) & FP32_EXP_MASK
        group_exp = group_exp >> FP32_EXP_OFFSET
        # Subtract largest exponent in target datatype and remove bias.
        group_exp = group_exp - 2 - FP32_EXP_BIAS
        # Clamp to valid int8 range.
        group_exp = tl.maximum(group_exp, -127)

        # Next we scale A in preparation for quantization.
        scale = tl.exp2(group_exp.to(tl.float64)).to(tl.float32)
        # Apply scale to input. We do this by broadcasting scale.
        scaled_a = tl.reshape(a, [BLOCK_SIZE // GROUP_SIZE, GROUP_SIZE]) / tl.reshape(
            scale, [BLOCK_SIZE // GROUP_SIZE, 1]
        )
        # Reshape back to a flat array.
        scaled_a = tl.reshape(scaled_a, [BLOCK_SIZE])

        # We're done with group_exp now so we can write it out.
        # We readd fp32_exp_bias for compatibility with cuda dequant.
        group_exp = group_exp.to(tl.int8)
        tl.store(
            shared_exp + pid * stride_exp_m + stride_exp_k * group_offset,
            group_exp + FP32_EXP_BIAS,
            mask=group_offset < K // GROUP_SIZE,
        )

        # Quantization step
        ###################

        # During quantization, we're going to be doing a lot of bitwise operations.
        # This is easier to work with in int32.
        scaled_a = scaled_a.to(tl.int32, bitcast=True)

        # Extract sign bit of value.
        sign_bit = (scaled_a >> FP32_SIGN_OFFSET) & SIGN_MASK

        # Extract exponent.
        biased_exp = (scaled_a & FP32_EXP_MASK) >> FP32_EXP_OFFSET

        # Extract mantissa.
        trailing_mantissa = scaled_a & FP32_MANTISSA_MASK

        # Adjust exponent bias for FP4.
        new_biased_exp = biased_exp - FP32_EXP_BIAS + FP4_EXP_BIAS

        # Compute difference between ideal exponent and what fp4 can represent.
        exp_diff = tl.where(new_biased_exp <= 0, 1 - new_biased_exp, 0)

        # Clip this difference to maximum number of fp32 mantissa bits.
        exp_diff = tl.minimum(exp_diff, MAX_FP32_MANTISSA_BITS)

        # Now we round our fp32 mantissa down to fp4.
        is_subnorm = biased_exp == 0
        # Add implied 1 bit to normal values.
        mantissa = tl.where(
            is_subnorm, trailing_mantissa, trailing_mantissa + IMPLIED_1_BIT
        )
        # Compute base number of bits corresponding to the mantissa, smaller for subnorms
        # since implied one is included in exp_diff.
        fp32_sig_bits = tl.where(is_subnorm, 23, 24).to(tl.int32)
        # Now we're ready to shift down to target bitwidth (with an extra bit for rounding).
        mantissa = mantissa >> (fp32_sig_bits + exp_diff - MBITS - 1)
        # Perform rounding by adding 1 and shifting down.
        mantissa = (mantissa + 1) >> 1

        # Check for overflow and adjust exponent accordingly.
        overflow = mantissa >= OVERFLOW_THRESHOLD
        # Allow subnorms to overflow into normals, otherwise shift away overflow.
        mantissa = tl.where(overflow and (not is_subnorm), mantissa >> 1, mantissa)
        # Special case where a value is subnormal and has a large mantissa, overflow it.
        new_biased_exp = tl.where(
            (new_biased_exp <= 0) and (mantissa == 2), 1, new_biased_exp
        )
        # Remove implicit 1.
        mantissa = mantissa & 0x1
        # Add overflow to exponent.
        new_biased_exp = tl.where(overflow, new_biased_exp + 1, new_biased_exp)
        # If exp overflows, set mantissa to maximum value (equivalent to clamping).
        mantissa = tl.where(new_biased_exp >= OVERFLOW_THRESHOLD, 1, mantissa)

        # Construct FP4 value from components.
        new_biased_exp = tl.maximum(tl.minimum(new_biased_exp, 3), 0)
        mx4_value = (new_biased_exp << 1) | mantissa
        mx4_value = (sign_bit << 3) | mx4_value

        # Extract low and high bits from values.
        low_mx4, high_mx4 = tl.split(tl.reshape(mx4_value, [BLOCK_SIZE // 2, 2]))
        # Shift mx4 values together so they are packed into int8.
        packed_mx4 = ((high_mx4 << 4) | (low_mx4)).to(tl.int8)

        # Next step is packing, lets write this out to check how it looks.
        tl.store(
            out + pid * stride_out_m + packed_offset * stride_out_k,
            packed_mx4,
            mask=packed_offset < K // 2,
        )

        # Update offsets so we work on the next block.
        k_offset += BLOCK_SIZE
        group_offset += BLOCK_SIZE // GROUP_SIZE
        packed_offset += BLOCK_SIZE // 2


def triton_quantize_mx4(a: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Quantize a tensor to mx4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.

    Returns:
        torch.Tensor: [M / 2 + M / group_size] mx4 scaled tensor packed into in8
        with group exponents attached to each row.

        eg.
        Input with shape [1, 8192] will be quantized to [1, 4096 + 256] as
        each value contain two elements packed into an int8 and
        there are 32 groups in each row.
    """
    # If given an empty shape, return an empty tensor.
    if a.numel() == 0:
        return torch.empty(a.shape, device=a.device, dtype=torch.uint8)
    # For now, only tensors with total elements that are a multiple of 32
    # are supported. This can be improved in the future.
    if a.numel() % group_size != 0:
        raise RuntimeError(
            f"Input must have total elements that are a multiple of group_size={group_size}, but got {a.numel()} elements."
        )
    orig_shape = a.shape
    # Find a shape that distributes work evenly over threads.
    # We do this by finding the power of two that is closest to
    # the sqrt of the number of elements.
    num_threads = int(2 ** round(math.log2(math.sqrt(a.numel()))))
    # Make sure that num_threads is a multiple of group_size.
    num_threads = (num_threads // group_size) * group_size
    if num_threads == 0:
        num_threads = a.numel() // group_size
    a = a.view(num_threads, -1)
    M, K = a.shape
    # If K is less than group_size, we compute a single group per row.
    if K < group_size:
        a = a.view(-1, group_size)
        M, K = a.shape
    # Create output tensors.
    shared_exp = torch.empty([M, K // group_size], device=a.device, dtype=torch.uint8)
    out = torch.empty([M, K // 2], device=a.device, dtype=torch.uint8)

    # Invoke triton quantization kernel over rows.
    grid = (M,)
    _kernel_quantize_mx4[grid](
        a,
        shared_exp,
        out,
        M,
        K,
        a.stride(0),
        a.stride(1),
        shared_exp.stride(0),
        shared_exp.stride(1),
        out.stride(0),
        out.stride(1),
        GROUP_SIZE=group_size,
    )
    # Ravel together output and shared exponent.
    packed_mx4 = torch.concat(
        [out.view(-1, group_size // 2), shared_exp.view(-1, 1)], dim=1
    )
    # Inputs are now fully quantized and ready to return.
    # Try to return in the original shape if possible.
    if orig_shape[-1] % group_size == 0:
        output_shape = list(orig_shape[:-1]) + [-1]
        return packed_mx4.view(output_shape)
    # If we cant, return as a flat array.
    else:
        return packed_mx4.view(-1)


@triton.autotune(
    configs=[
        Config({"BLOCK_SIZE": 512}),
        Config({"BLOCK_SIZE": 1024}),
        Config({"BLOCK_SIZE": 2048}),
        Config({"BLOCK_SIZE": 4096}),
        Config({"BLOCK_SIZE": 8192}),
    ],
    key=["K"],
)
@triton.jit
def _kernel_dequantize_mx4(
    A,
    shared_exp,
    mx4_lookup_table,
    out,
    M,
    K,
    stride_am,
    stride_ak,
    stride_exp_m,
    stride_exp_k,
    stride_out_m,
    stride_out_k,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Dequantize a packed MX4 tensor and apply scaling.

    Args:
        A (Tensor): [M, K] MX4 tensor packed into int8.
        shared_exp (Tensor): Int8 tensor representing group exponent.
        mx4_lookup_table (Tensor): Map from mx4 integer value to floating point.
        M (int): Number of rows.
        K (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_exp_m (int): Stride of m dimension of shared exponent tensor.
        stride_exp_k (int): Stride of k dimension of shared exponent tensor.
        stride_out_m (int): Stride of m dimension of output.
        stride_out_k (int): Stride of k dimension of output.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        BLOCK_SIZE (int): Size of each block.
    """
    pid = tl.program_id(0)
    k_offset = tl.arange(0, BLOCK_SIZE)
    output_offset = tl.arange(0, 2 * BLOCK_SIZE)

    # Define constants.
    MX4_BIT_MASK: tl.constexpr = 0xF  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]

    # Iterate over input tensor and unpack mx4 values.
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE)):
        a = tl.load(
            A + pid * stride_am + k_offset * stride_ak, mask=k_offset < K, other=0.0
        )
        # Extract high and low values from loaded mx4 tile.
        low_mx4 = a & MX4_BIT_MASK
        high_mx4 = (a >> 4) & MX4_BIT_MASK

        # Get equivalent fp32 values.
        low_fp32 = tl.load(mx4_lookup_table + low_mx4)
        high_fp32 = tl.load(mx4_lookup_table + high_mx4)

        # Get proper shared exponent and convert it to a float scale.
        group_offset = (2 * k_offset) // GROUP_SIZE
        exp = tl.load(shared_exp + pid * stride_exp_m + group_offset * stride_exp_k)
        # Remove fp32 exponent bias.
        exp = exp.to(tl.uint8, bitcast=True) - FP32_EXP_BIAS

        # Convert exponent to scale and apply to input.
        # Requires higher precision to avoid rounding out small values.
        # This might be slow so we should consider just letting them round away.
        scale = tl.exp2(exp.to(tl.float64)).to(tl.float32)
        scaled_low_fp32 = scale * low_fp32
        scaled_high_fp32 = scale * high_fp32

        # Combine the two components into a single tensor, interweave them.
        scaled_fp32 = tl.interleave(scaled_low_fp32, scaled_high_fp32)

        # Write final outputs.
        tl.store(
            out + pid * stride_out_m + output_offset * stride_out_k,
            scaled_fp32,
            mask=output_offset < 2 * K,
        )

        k_offset += BLOCK_SIZE
        output_offset += 2 * BLOCK_SIZE


def triton_dequantize_mx4(a: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Dequantize a tensor from mx4 format to fp32.

    Args:
        a (Tensor): [M / 2 + M / group_size] MX4 tensor packed into int8 values
        with group exponents attached to end of each row.
        group_size (int): Size of chunks that use the same shared exponent.

    Returns:
        torch.Tensor: [M, K] dequantized fp32 tensor.
    """
    # If given an empty shape, return an empty tensor.
    if a.numel() == 0:
        return torch.empty(a.shape, device=a.device, dtype=torch.float32)
    # View a as 2D for simplicity.
    orig_shape = a.shape
    # Unravel packed inputs from shared exponents.
    a = a.view(-1, (group_size // 2) + 1)
    packed_input = a[:, :-1]
    shared_exp = a[:, -1:]
    # Find a shape that distributes work evenly over threads.
    # We do this by finding the power of two that is closest to
    # the sqrt of the number of elements.
    num_threads = int(2 ** round(math.log2(math.sqrt(packed_input.numel()))))
    # Make sure that num_threads is a multiple of group_size.
    num_threads = (num_threads // group_size) * group_size
    if num_threads == 0:
        num_threads = packed_input.numel() // group_size
    packed_input = packed_input.reshape(num_threads, -1)
    shared_exp = shared_exp.reshape(num_threads, -1)
    M, K_2 = packed_input.shape

    # Use a lookup table to convert
    mx4_to_fp_values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float,
    )

    # Create output tensor.
    out = torch.empty([M, 2 * K_2], device=a.device, dtype=torch.float)

    # Invoke triton dequantization kernel over rows.
    grid = (M,)
    _kernel_dequantize_mx4[grid](
        packed_input,
        shared_exp,
        mx4_to_fp_values,
        out,
        M,
        K_2,
        packed_input.stride(0),
        packed_input.stride(1),
        shared_exp.stride(0),
        shared_exp.stride(1),
        out.stride(0),
        out.stride(1),
        GROUP_SIZE=group_size,
    )

    out_shape = list(orig_shape[:-1]) + [-1]
    return out.view(out_shape)
