#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import math
from typing import Union

import torch
import triton  # @manual

import triton.language as tl  # @manual
from triton import Config  # @manual

from .common import get_mx4_exp_bias, get_mx4_lookup_table, RoundingMode


@triton.jit
def _floor_log2(x):
    """Helper function to efficiently compute floor(log2(x))

    Args:
        x (Tensor): FP32 Input tensor to operate on.

    Returns:
        Tensor: Floor of log2(x).
    """
    # Helpful bit constants.
    FP32_EXP_MASK: tl.constexpr = 0x7F800000  # type: ignore[Incompatible variable type]
    FP32_EXP_OFFSET: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]

    # View x as an integer and extract its exponent.
    x = x.to(tl.int32, bitcast=True) & FP32_EXP_MASK
    # Shift exponent down to bottom bits.
    x = x >> FP32_EXP_OFFSET
    # Remove FP32 exponent bias and return.
    return (x - FP32_EXP_BIAS).to(tl.float32)


@triton.jit
def _compute_exp(
    group_max,
    rounding_mode,
    rand_bits,
    MBITS: tl.constexpr,
):
    """Compute shared exponent of group using specified rounding mode.

    Args:
        group_max (Tensor): Group of values to compute exponent of.
        rounding_mode (int or RoundingMode): Which rounding mode to use.
        rand_bits (int): Random integer values used for stochastic rounding.
        mbits (int): Number of mantissa bits in target mx4 format.

    Returns:
        Tensor: Shared exponent of group.
    """
    # Define some helpful constants.
    MBITS_FP32: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    M_ROUND: tl.constexpr = (1 << (MBITS_FP32 - MBITS - 1)) - 1  # type: ignore[Incompatible variable type]
    RAND_MASK: tl.constexpr = (1 << (MBITS_FP32 - MBITS)) - 1  # type: ignore[Incompatible variable type]

    # Nearest rounding mode.
    if rounding_mode == 0:
        return tl.floor(tl.log2(group_max) + 0.5)
    # Floor rounding mode. This can be done with fast bit ops.
    if rounding_mode == 1:
        return _floor_log2(group_max)
    # Even pre-rounding mode.
    elif rounding_mode == 2:
        # Add fixed rounding to the mantissa bits of the input to round during truncation.
        group_max = group_max.to(tl.int32, bitcast=True) + M_ROUND
        # Then perform floor rounding of log.
        return _floor_log2(group_max)
    # Stochastic rounding mode.
    elif rounding_mode == 3:
        # Use random bits to add noise to mantissa that would otherwise
        # be rounded away.
        group_max = group_max.to(tl.int32, bitcast=True) + (RAND_MASK & rand_bits)
        # Now compute log and truncate.
        return _floor_log2(group_max)
    else:
        return tl.ceil(tl.log2(group_max))


def get_autotune_config():
    # Enumerate possible kernel configurations for triton autotuning.
    # These values represent the number of rows each thread processes.
    # Typically, inputs will either have a single row or many.
    BLOCK_SIZE_M_VALS = [1, 8]

    # BLOCK_GROUPS_K represents how many groups each thread is responsible for computing,
    # where one group contains GROUP_SIZE elements.
    # GROUP_LOAD indicates how many groups should be loaded from memory, processed, and
    # written to the output at once.
    # In general BLOCK_GROUPS_K and GROUP_LOAD should be similar in magnitude since
    # if rows are large, we want to distribute them over more threads and vectorize
    # heavily. If rows are small, we dont distribute as much and dont have as much
    # to load.
    BLOCK_GROUPS_AND_LOAD_VALS = [[4, 4], [8, 8], [32, 32], [128, 64], [512, 64]]
    # Get all combinations of these two configurations.
    configs = []
    for m in BLOCK_SIZE_M_VALS:
        for k, g in BLOCK_GROUPS_AND_LOAD_VALS:
            configs.append(
                Config({"BLOCK_SIZE_M": m, "BLOCK_GROUPS_K": k, "GROUP_LOAD": g})
            )
    return configs


@triton.autotune(
    configs=get_autotune_config(),
    key=[
        "M",
        "K",
    ],
)
@triton.jit
def _kernel_quantize_mx4(
    A,
    out,
    M,
    K,
    K_OUT,
    rand_bits,
    GROUP_SIZE: tl.constexpr,
    EBITS: tl.constexpr,
    MBITS: tl.constexpr,
    ROUNDING_MODE: tl.constexpr,
    STOCHASTIC_CASTING: tl.constexpr,
    FP4_EXP_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_GROUPS_K: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
) -> None:
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        shared_exp (Tensor): [M / group_size] output containing shared exponent.
        out (Tensor): [M / 2 + M / GROUP_SIZE] output containing packed mx4 values.
        M (int): Number of rows in input and output.
        K (int): Number of elements in each row of the input.
        K_OUT (int): Number of elements in each row after padding.
        rand_bits (Optional Tensor): [M, K_out / 2] random integers used for stochastic rounding.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias for target datatype.
        BLOCK_SIZE_M (int): Number of rows one thread is responsible for processing.
        BLOCK_GROUPS_K (int): Number of groups one thread is responsible for processing.
        GROUP_LOAD (int): Number of groups that should be read and processed at once.
    """
    # Define Constant Expressions.
    FP32_EXP_MASK: tl.constexpr = 0x7F800000  # type: ignore[Incompatible variable type]
    FP32_EXP_OFFSET: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
    FP32_SIGN_OFFSET: tl.constexpr = 31  # type: ignore[Incompatible variable type]
    SIGN_MASK: tl.constexpr = 0x1  # type: ignore[Incompatible variable type]
    FP32_MANTISSA_MASK: tl.constexpr = 0x007FFFFF  # type: ignore[Incompatible variable type]
    # FP4 has 2 mantissa bits, one explicit one implicit.
    MBITS_IMPLICIT: tl.constexpr = MBITS + 1  # type: ignore[Incompatible variable type]
    MAX_FP32_MANTISSA_BITS: tl.constexpr = 24  # type: ignore[Incompatible variable type]
    IMPLIED_1_BIT: tl.constexpr = 1 << 23  # type: ignore[Incompatible variable type]
    FP32_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]
    MANTISSA_OVERFLOW_THRESHOLD: tl.constexpr = (1 << MBITS_IMPLICIT) - 1  # type: ignore[Incompatible variable type]
    EXPONENT_OVERFLOW_THRESHOLD: tl.constexpr = (1 << EBITS) - 1  # type: ignore[Incompatible variable type]
    IMPLICIT_1_MASK = (1 << (MBITS_IMPLICIT - 1)) - 1
    RAND_MASK: tl.constexpr = (1 << (FP32_EXP_OFFSET - MBITS)) - 1  # type: ignore[Incompatible variable type]
    # Boundaries for writing to output tensor.
    OUTPUT_ROW_SIZE: tl.constexpr = K_OUT // 2 + K_OUT // GROUP_SIZE  # type: ignore[Incompatible variable type]
    PACKED_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2 + 1  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
    # Calcluate which position in the grid we're in.
    num_pid_k = tl.cdiv(K_OUT, BLOCK_GROUPS_K * GROUP_SIZE)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k
    # Compute starting offsets for this block of the problem.
    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_GROUPS_K * GROUP_SIZE
    out_k_start = pid_k * ((BLOCK_GROUPS_K * GROUP_SIZE) // 2 + BLOCK_GROUPS_K)
    # Create offsets for a set of rows.
    # Prevent out of bounds by modulo with M.
    m_offsets = (tl.arange(0, BLOCK_SIZE_M) + m_start) % M
    # Create offsets for groups of values in the row.
    k_offsets = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + k_start
    # Create equivalent output offsets for this set of groups.
    out_k_offsets = tl.arange(0, GROUP_LOAD * GROUP_SIZE // 2)
    # Stochastic rounding loads chunks of random values for each group.
    if ROUNDING_MODE == 3:
        rand_bits_k_offsets = tl.arange(0, GROUP_LOAD) + (pid_k * K_OUT // GROUP_SIZE)
    # Ceil rounding uses single values as a seed.
    else:
        rand_bits_k_offsets = tl.arange(0, 1) + (pid_k * K_OUT // GROUP_SIZE)
    # We also need to write out group exponents. Calculate their positions.
    # Find first position of shared exponent.
    exp_k_start = out_k_start + GROUP_SIZE // 2
    exp_k_offsets = (tl.arange(0, GROUP_LOAD) * PACKED_GROUP_SIZE) + exp_k_start
    # Finally, we need to make space in the output offsets for these exponents.
    out_k_offsets += out_k_offsets // (GROUP_SIZE // 2) + out_k_start

    # Now that we have all our offsets set up, we're ready to compute full offsets.
    input_offsets = m_offsets[:, None] * K + k_offsets[None, :]
    output_offsets = m_offsets[:, None] * OUTPUT_ROW_SIZE + out_k_offsets[None, :]
    exp_offsets = m_offsets[:, None] * OUTPUT_ROW_SIZE + exp_k_offsets[None, :]
    rand_bits_offsets = (
        m_offsets[:, None] * (K_OUT // GROUP_SIZE) + rand_bits_k_offsets[None, :]
    )

    # Load and process blocks of values for this chunk.
    for _k in range(0, tl.cdiv(BLOCK_GROUPS_K, GROUP_LOAD)):
        # Load a block of values.
        a = tl.load(
            A + input_offsets,
            # Mask values that extend out of this row.
            mask=(k_offsets[None, :] < K - _k * GROUP_LOAD * GROUP_SIZE),
            other=0,
        )

        # Scaling step
        ##############

        # View the block in terms of groups.
        a_groups = tl.reshape(a, [BLOCK_SIZE_M, GROUP_LOAD, GROUP_SIZE])
        # Compute the shared exponent of each group.
        group_max = tl.max(tl.abs(a_groups), axis=2)
        # Prevent infinite values in log.
        group_max = tl.where(group_max == 0, FP32_MIN_NORMAL, group_max)
        # Load relevant random values if doing stochastic rounding
        # or stochastic casting.
        group_rand_bits = None
        if (ROUNDING_MODE) == 3 or STOCHASTIC_CASTING:
            group_rand_bits = tl.load(
                rand_bits + rand_bits_offsets,
                mask=rand_bits_k_offsets[None, :]
                < ((K // GROUP_SIZE) - _k * GROUP_LOAD),
                other=0,
            )
            rand_bits_offsets += GROUP_LOAD
        # Compute shared exponent using specified rounding mode.
        group_exp = _compute_exp(group_max, ROUNDING_MODE, group_rand_bits, MBITS)
        # Subtract largest exponent in target datatype and remove bias.
        group_exp = group_exp - EBITS
        # Make sure exponent is in valid range.
        group_exp = tl.clamp(group_exp, -127, 125)

        # Next we scale A in preparation for quantization.
        scale = tl.exp2(group_exp.to(tl.float64)).to(tl.float32)
        # Apply scale to input. We do this by broadcasting scale.
        scaled_a = tl.reshape(a, [BLOCK_SIZE_M, GROUP_LOAD, GROUP_SIZE]) / tl.reshape(
            scale, [BLOCK_SIZE_M, GROUP_LOAD, 1]
        )
        # Reshape back to a flat array.
        scaled_a = tl.reshape(scaled_a, [BLOCK_SIZE_M, GROUP_LOAD * GROUP_SIZE])

        # We're done with group_exp now so we can write it out.
        # We readd fp32_exp_bias for compatibility with cuda dequant.
        tl.store(
            out + exp_offsets,
            (group_exp + FP32_EXP_BIAS).to(tl.int8),
            # Prevent writing outside this chunk or the main array.
            mask=(
                (m_offsets[:, None] < M)
                & (
                    exp_k_offsets[None, :]
                    < (OUTPUT_ROW_SIZE - _k * GROUP_LOAD * PACKED_GROUP_SIZE)
                )
            ),
        )

        # Quantization step
        ###################

        # During quantization, we're going to be doing a lot of bitwise operations.
        # This is easier to work with in int32.
        scaled_a = scaled_a.to(tl.int32, bitcast=True)

        # When doing stochastic downcasting, generate random values for this block
        # and apply it to the mantissa.
        if STOCHASTIC_CASTING:
            # We're going to generate 4 blocks at once so we only need
            # one fourth of the input offsets.
            # Start by splitting down to half of offsets.
            philox_4x_offset = tl.split(
                tl.reshape(
                    input_offsets,
                    [BLOCK_SIZE_M, GROUP_LOAD * GROUP_SIZE // 2, 2],
                    can_reorder=True,
                )
            )
            # Split down to fourth.
            philox_4x_offset = tl.split(
                tl.reshape(
                    philox_4x_offset,
                    [BLOCK_SIZE_M, GROUP_LOAD * GROUP_SIZE // 4, 2],
                    can_reorder=True,
                )
            )
            # Generate 4 blocks of random bits for this block.
            a_4x, b_4x, c_4x, d_4x = tl.randint4x(
                group_rand_bits, philox_4x_offset, n_rounds=7
            )
            # Combine the 4 blocks into a single chunk of random values.
            # This needs to be done incrementally.
            stochastic_round_bits = tl.join(tl.join(a_4x, b_4x), tl.join(c_4x, d_4x))
            # Flatten back to simple array.
            stochastic_round_bits = tl.reshape(
                stochastic_round_bits, [BLOCK_SIZE_M, GROUP_LOAD * GROUP_SIZE]
            ).to(tl.int32, bitcast=True)

            # Mask off mantissa bits of random value and add to mantissa.
            scaled_a = scaled_a + (stochastic_round_bits & RAND_MASK)

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
        mantissa = mantissa >> (fp32_sig_bits + exp_diff - MBITS_IMPLICIT - 1)
        # Perform rounding by adding 1 and shifting down.
        mantissa = (mantissa + 1) >> 1

        # Check for overflow and adjust exponent accordingly.
        overflow = mantissa > MANTISSA_OVERFLOW_THRESHOLD
        # Allow subnorms to overflow into normals, otherwise shift away overflow.
        mantissa = tl.where(overflow and (not is_subnorm), mantissa >> 1, mantissa)
        # Special case where a value is subnormal and has a large mantissa, overflow it.
        new_biased_exp = tl.where(
            (new_biased_exp <= 0) and (mantissa == 2), 1, new_biased_exp
        )
        # Remove implicit 1.
        mantissa = mantissa & IMPLICIT_1_MASK
        # Add overflow to exponent.
        new_biased_exp = tl.where(overflow, new_biased_exp + 1, new_biased_exp)
        # If exp overflows, set mantissa to maximum value (equivalent to clamping).
        mantissa = tl.where(new_biased_exp > EXPONENT_OVERFLOW_THRESHOLD, 1, mantissa)

        # Construct FP4 value from components.
        new_biased_exp = tl.maximum(
            tl.minimum(new_biased_exp, EXPONENT_OVERFLOW_THRESHOLD), 0
        )
        mx4_value = (new_biased_exp << (MBITS_IMPLICIT - 1)) | mantissa
        mx4_value = (sign_bit << (EBITS + MBITS)) | mx4_value

        # Extract low and high bits from values.
        low_mx4, high_mx4 = tl.split(
            tl.reshape(mx4_value, [BLOCK_SIZE_M, (GROUP_LOAD * GROUP_SIZE) // 2, 2])
        )
        # Shift mx4 values together so they are packed into int8.
        packed_mx4 = ((high_mx4 << 4) | (low_mx4)).to(tl.int8)

        # Write out packed values to output tensor.
        tl.store(
            out + output_offsets,
            packed_mx4,
            # Prevent writing outside this chunk or the main array.
            mask=(
                (m_offsets[:, None] < M)
                & (
                    out_k_offsets[None, :]
                    < (OUTPUT_ROW_SIZE - _k * GROUP_LOAD * PACKED_GROUP_SIZE)
                )
            ),
        )

        # Update offsets so we work on the next block.
        input_offsets += GROUP_LOAD * GROUP_SIZE
        exp_offsets += GROUP_LOAD * PACKED_GROUP_SIZE
        output_offsets += GROUP_LOAD * PACKED_GROUP_SIZE


def triton_quantize_mx4(
    a: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    stochastic_casting: bool = False,
) -> torch.Tensor:
    """
    Quantize a tensor to mx4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target mx4 format.
        mbits (int): Number of bits to use for mantissa in target mx4 format.
        rounding_mode (Union[RoundingMode, int]): Which type of rounding to use
        when calculating shared exponent. Defaults to pre-rounding to nearest even int.
        stochastic_casting (bool): Whether to use stochastic casting.

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
    # Make sure input is continuous in memory.
    assert a.is_contiguous(), "Inputs to mx4 quantize must be contiguous in memory."

    orig_shape = a.shape

    # For simplicity, view input as a 2D tensor.
    a = a.view(-1, a.shape[-1])
    # Get rows and columns of now 2D input.
    M, K = a.shape

    # If K isnt divisible by group size, we'll pad it with zeros,
    # this makes the output a bit bigger.
    if K % group_size != 0:
        K_out = K + (group_size - (K % group_size))
    else:
        K_out = K
    # Outputs have two values packed into a single int element, and shared exponents.
    out = torch.empty(
        [M, K_out // 2 + K_out // group_size], device=a.device, dtype=torch.uint8
    )

    # If using stochastic rounding, create random noise for each group.
    # We use the same random bits as seeds when doing stochastic downcasting.
    if rounding_mode == RoundingMode.stochastic or stochastic_casting:
        # Each group will need a seed.
        rand_bits = torch.randint(
            low=0,
            high=2**31 - 1,
            size=(M, K_out // group_size),
            dtype=torch.int32,
            device=a.device,
        )
    else:
        rand_bits = None

    # Compute grid of PIDs to distribute work across threads.
    def grid(META):
        return (
            (
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(K_out, META["BLOCK_GROUPS_K"] * group_size)
            ),
        )

    _kernel_quantize_mx4[grid](
        a,
        out,
        M,
        K,
        K_out,
        rand_bits=rand_bits,
        GROUP_SIZE=group_size,
        EBITS=ebits,
        MBITS=mbits,
        ROUNDING_MODE=rounding_mode,
        STOCHASTIC_CASTING=stochastic_casting,
        FP4_EXP_BIAS=get_mx4_exp_bias(ebits),
    )
    # Inputs are now fully quantized and ready to return.
    # Try to return in the original shape if possible.
    try:
        output_shape = list(orig_shape[:-1]) + [-1]
        return out.view(output_shape)
    # If we cant, return as a flat array.
    except RuntimeError:
        return out.view(-1)


@triton.autotune(
    configs=[
        Config({"GROUP_LOAD": 1}),
        Config({"GROUP_LOAD": 4}),
        Config({"GROUP_LOAD": 8}),
        Config({"GROUP_LOAD": 16}),
        Config({"GROUP_LOAD": 32}),
    ],
    key=["K"],
)
@triton.jit
def _kernel_dequantize_mx4(
    A,
    mx4_lookup_table,
    out,
    M,
    K,
    GROUP_SIZE: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
) -> None:
    """Dequantize a packed MX4 tensor and apply scaling.

    Args:
        A (Tensor): [M] MX4 tensor packed into int8.
        shared_exp (Tensor): Int8 tensor representing group exponent.
        mx4_lookup_table (Tensor): Map from mx4 integer value to floating point.
        M (int): Total number of elements in input.
        K (int): Number of elements each thread should operate on.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        GROUP_LOAD (int): Number of groups to process simultaneously.
    """
    # Define constants.
    MX4_BIT_MASK: tl.constexpr = 0xF  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
    PACKED_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2 + 1  # type: ignore[Incompatible variable type]
    # Boundaries for writing to output tensor.
    OUTPUT_LIMIT: tl.constexpr = (K // PACKED_GROUP_SIZE) * GROUP_SIZE  # type: ignore[Incompatible variable type]
    OUTPUT_SIZE: tl.constexpr = (M // PACKED_GROUP_SIZE) * GROUP_SIZE  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
    # Find the starting offsets for this thread.
    input_start = pid * K
    exp_start = input_start + GROUP_SIZE // 2
    # Remove shared exponents from output offset.
    output_start = pid * GROUP_SIZE * (K // PACKED_GROUP_SIZE)
    # Initiate offset ranges used in this thread.
    # This is a little complicated because we need to skip one value (the shared exponent)
    # every group_size elements.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE // 2)
    # Add 1 every GROUP_SIZE / 2 steps so we skip shared exponent.
    exp_indices = input_offset // (GROUP_SIZE // 2)
    input_offset = input_offset + exp_indices + input_start
    # We need to space out each group of the input by 1 since thats the shared exp.
    output_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + output_start
    # Stride exponent access across packed groups.
    exp_offset = exp_indices * PACKED_GROUP_SIZE + exp_start

    # Iterate over input tensor and unpack mx4 values.
    for _k in range(0, tl.cdiv(K, GROUP_LOAD * PACKED_GROUP_SIZE)):
        a = tl.load(
            A + input_offset,
            # Mask values that are out of this chunk or the main array.
            mask=(input_offset < M) & (input_offset < (K * (pid + 1))),
            other=0.0,
        )
        # Extract high and low values from loaded mx4 tile.
        low_mx4 = a & MX4_BIT_MASK
        high_mx4 = (a >> 4) & MX4_BIT_MASK

        # Get equivalent fp32 values.
        low_fp32 = tl.load(mx4_lookup_table + low_mx4)
        high_fp32 = tl.load(mx4_lookup_table + high_mx4)

        # Get proper shared exponent and convert it to a float scale.
        exp = tl.load(
            A + exp_offset,
            mask=(exp_offset < M) & (exp_offset < (K * (pid + 1))),
            other=0.0,
        )
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
            out + output_offset,
            scaled_fp32,
            # Mask values that are out of this chunk or the main array.
            mask=(output_offset < OUTPUT_SIZE)
            & (output_offset < OUTPUT_LIMIT * (pid + 1)),
        )

        # Update indices for next group.
        input_offset += GROUP_LOAD * PACKED_GROUP_SIZE
        exp_offset += GROUP_LOAD * PACKED_GROUP_SIZE
        output_offset += GROUP_LOAD * GROUP_SIZE


def triton_dequantize_mx4(
    a: torch.Tensor, group_size: int = 32, ebits: int = 2, mbits: int = 1
) -> torch.Tensor:
    """
    Dequantize a tensor from mx4 format to fp32.

    Args:
        a (Tensor): [M / 2 + M / group_size] MX4 tensor packed into int8 values
        with group exponents attached to end of each row.
        group_size (int): Size of chunks that use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target mx4 format.
        mbits (int): Number of bits to use for mantissa in target mx4 format.

    Returns:
        torch.Tensor: [M, K] dequantized fp32 tensor.
    """
    # If given an empty shape, return an empty tensor.
    if a.numel() == 0:
        return torch.empty(a.shape, device=a.device, dtype=torch.float32)
    # View a as 2D for simplicity.
    orig_shape = a.shape
    a = a.flatten()
    # Find number of groups.
    packed_group_size = group_size // 2 + 1
    # Find a shape that distributes work evenly over threads.
    # We do this by finding the power of two that is closest to
    # the sqrt of the number of elements.
    num_threads = int(2 ** round(math.log2(math.sqrt(a.numel()))))
    # Make sure that the number of elements per row is a multiple of packed group_size.
    K = a.numel() // num_threads
    K = (K // packed_group_size) * packed_group_size
    if K == 0:
        K = packed_group_size
    # Try to evenly divide input into chunks of size K, allow last chunk to be smaller.
    M = int(math.ceil(a.numel() / K))

    # Use a lookup table to convert
    mx4_to_fp_values = torch.tensor(
        get_mx4_lookup_table(ebits, mbits),
        device="cuda",
        dtype=torch.float,
    )

    # Create output tensor.
    num_groups = a.numel() // packed_group_size
    output_elems = num_groups * group_size
    out = torch.empty([output_elems], device=a.device, dtype=torch.float)
    # Invoke triton dequantization kernel over rows.
    grid = (M,)
    _kernel_dequantize_mx4[grid](
        a,
        mx4_to_fp_values,
        out,
        a.numel(),
        K,
        GROUP_SIZE=group_size,
    )

    out_shape = list(orig_shape[:-1]) + [-1]
    return out.view(out_shape)
