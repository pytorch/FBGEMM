#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import functools
import math

import torch
import triton  # @manual
import triton.language as tl  # @manual
from fbgemm_gpu.config import FeatureGate, FeatureGateName

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


@triton.jit
def _quantize_e2m1(x):
    """Round scaled FP32 values (as int32 bits) to MX4 e2m1 codes, sign in bit 3.

    Same results as the general bit-manipulation path for EBITS=2, MBITS=1: magnitudes
    0, 0.5, 1, 1.5, 2, 3, 4, 6 with ties rounded away from zero, and anything from 5 up
    (including inf and nan) saturated to 6.
    """
    # Clamp nan to inf so the rounding add below cannot overflow into the sign bit.
    mag = tl.minimum(x & 0x7FFFFFFF, 0x7F800000)
    # From 1.0 up, adding half a mantissa step and keeping one mantissa bit rounds ties away
    # from zero; exponent and mantissa bit then give the code (1.0 -> 2 ... 6.0 -> 7, saturating).
    hi = tl.minimum(((mag + 0x200000) >> 22) - 252, 7)
    # Below 1.0 the step is a fixed 0.5: from 0.25 -> 0.5 (code 1), from 0.75 -> 1.0 (code 2).
    lo = (mag >= 0x3E800000).to(tl.int32) + (mag >= 0x3F400000).to(tl.int32)
    code = tl.where(mag >= 0x3F800000, hi, lo)
    return code | ((x >> 28) & 0x8)


@triton.jit
def _scale_groups(
    a_groups,
    group_rand_bits,
    ROUNDING_MODE: tl.constexpr,
    EBITS: tl.constexpr,
    MBITS: tl.constexpr,
):
    """Shared exponent of each row (one group) of a_groups, and the rows scaled by it.

    Returns (scaled_groups, group_exp), with group_exp an integer in [-127, 125].
    """
    FP32_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]
    # Compute the shared exponent of each group.
    group_max = tl.max(tl.abs(a_groups), axis=1)
    # Prevent infinite values in log.
    group_max = tl.where(group_max == 0, FP32_MIN_NORMAL, group_max)
    # Compute shared exponent using specified rounding mode.
    group_exp = _compute_exp(group_max, ROUNDING_MODE, group_rand_bits, MBITS)
    # Subtract largest exponent in target datatype and remove bias.
    group_exp = group_exp - EBITS
    # Make sure exponent is in valid range.
    group_exp = tl.clamp(group_exp, -127, 125)
    # Multiply by 2^-group_exp, built exactly from FP32 bits, instead of dividing by
    # 2^group_exp. group_exp is an integer in [-127, 125], so 2^-group_exp is always a
    # normal fp32 and the product is the exact quotient.
    inv_scale = ((127 - group_exp.to(tl.int32)) << 23).to(tl.float32, bitcast=True)
    return a_groups * tl.reshape(inv_scale, [inv_scale.shape[0], 1]), group_exp


@triton.jit
def _kernel_quantize_mx4(
    A,
    out,
    rand_bits,
    M,
    K,
    GROUPS_PER_ROW,
    GROUPS_PER_THREAD,
    ROW_PADDING,
    GROUP_SIZE: tl.constexpr,
    EBITS: tl.constexpr,
    MBITS: tl.constexpr,
    ROUNDING_MODE: tl.constexpr,
    STOCHASTIC_CASTING: tl.constexpr,
    FP4_EXP_BIAS: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
    USE_INT64: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    E2M1_FAST_PATH: tl.constexpr,
) -> None:
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        out (Tensor): [M / 2 + M / GROUP_SIZE] output containing packed mx4 values.
        rand_bits (Optional Tensor): [M, K / 2] random integers used for stochastic rounding.
        M (int): Number of input rows.
        K (int): Number of input columns.
        GROUPS_PER_ROW (int): Number of groups in each row of the input.
        GROUPS_PER_THREAD (int): Number of groups to process per thread.
        ROW_PADDING (int): Number of elements of padding to insert into each row.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias of target mx4 format.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing. This is needed for large tensors.
        HAS_PADDING (bool): Whether ROW_PADDING is nonzero.
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
    MANTISSA_OVERFLOW_THRESHOLD: tl.constexpr = (1 << MBITS_IMPLICIT) - 1  # type: ignore[Incompatible variable type]
    EXPONENT_OVERFLOW_THRESHOLD: tl.constexpr = (1 << EBITS) - 1  # type: ignore[Incompatible variable type]
    IMPLICIT_1_MASK = (1 << (MBITS_IMPLICIT - 1)) - 1
    RAND_MASK: tl.constexpr = (1 << (FP32_EXP_OFFSET - MBITS)) - 1  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
    # For very large inputs, we need to use int64 indexes. This is slower but necessary.
    if USE_INT64:
        pid = pid.to(tl.int64)
        M = tl.cast(M, tl.int64)
        K = tl.cast(K, tl.int64)
        GROUPS_PER_THREAD = tl.cast(GROUPS_PER_THREAD, tl.int64)

    # Boundaries for writing to output tensor.
    PACKED_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2 + 1  # type: ignore[Incompatible variable type]
    NUM_GROUPS = M * GROUPS_PER_ROW
    OUTPUT_CHUNK_SIZE = (GROUPS_PER_THREAD * GROUP_SIZE) // 2 + GROUPS_PER_THREAD
    OUTPUT_SIZE = (GROUP_SIZE * NUM_GROUPS) // 2 + NUM_GROUPS

    # Find starting offsets for this thread. These are calculated before adjusting for padding.
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    # Per-element offsets are int32 and relative to this thread's chunk; only the scalar chunk
    # starts and bounds need int64. Bounds are clamped to [0, chunk size] so they fit in int32.
    input_bound = tl.minimum(  # noqa: TR012 (tl.clamp is float-only; these are int)
        tl.maximum(M * K - input_start, 0),  # noqa: TR012
        GROUPS_PER_THREAD * GROUP_SIZE,
    ).to(tl.int32)
    output_bound = tl.minimum(  # noqa: TR012 (tl.clamp is float-only; these are int)
        tl.maximum(OUTPUT_SIZE - output_start, 0),  # noqa: TR012
        OUTPUT_CHUNK_SIZE,
    ).to(tl.int32)
    A_chunk = A + input_start
    out_chunk = out + output_start
    # Initiate offset ranges used in kernel.
    local_input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE)
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 2))
    # Stochastic rounding loads chunks of random values.
    if ROUNDING_MODE == 3:
        rand_bits_offset = tl.arange(0, GROUP_LOAD) + pid * GROUPS_PER_THREAD
    # Ceil rounding uses single values as a seed.
    else:
        rand_bits_offset = pid * GROUPS_PER_THREAD
    # We need to shift output offsets to make space for shared exponent storage.
    output_offset += output_offset // (GROUP_SIZE // 2)
    # Now create offsets for writing the shared exponent.
    exp_offset = tl.arange(0, GROUP_LOAD) * PACKED_GROUP_SIZE + GROUP_SIZE // 2

    # Load and process blocks of values for this chunk.
    for _k in range(0, tl.cdiv(GROUPS_PER_THREAD, GROUP_LOAD)):
        if HAS_PADDING:
            # Padded rows need the global offset to locate row boundaries.
            input_offset = local_input_offset + input_start
            pad_mask = (input_offset % (GROUPS_PER_ROW * GROUP_SIZE)) < K
            # Shift the input to account for padding.
            padded_input_offset = (
                input_offset
                - (input_offset // (GROUPS_PER_ROW * GROUP_SIZE)) * ROW_PADDING
            )
            a = tl.load(
                A + padded_input_offset,
                # Mask values out of range for both the main array and this chunk. Also pad.
                mask=(padded_input_offset < (M * K))
                & (padded_input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE))
                & pad_mask,
                other=0,
            )
        else:
            # Without padding every offset is in its row, so only the range check remains.
            a = tl.load(
                A_chunk + local_input_offset,
                mask=local_input_offset < input_bound,
                other=0,
            )

        # Scaling step
        ##############

        # Load relevant random values if doing stochastic rounding
        # or stochastic casting.
        group_rand_bits = None
        if (ROUNDING_MODE) == 3 or STOCHASTIC_CASTING:
            group_rand_bits = tl.load(
                rand_bits + rand_bits_offset,
                mask=rand_bits_offset < K // GROUP_SIZE,
                other=0,
            )
            rand_bits_offset += GROUP_LOAD
        # View the block in terms of groups and scale each by its shared exponent.
        scaled_a, group_exp = _scale_groups(
            tl.reshape(a, [GROUP_LOAD, GROUP_SIZE]),
            group_rand_bits,
            ROUNDING_MODE,
            EBITS,
            MBITS,
        )
        # Reshape back to a flat array.
        scaled_a = tl.reshape(scaled_a, [GROUP_LOAD * GROUP_SIZE])

        # We're done with group_exp now so we can write it out.
        # We readd fp32_exp_bias for compatibility with cuda dequant.
        tl.store(
            out_chunk + exp_offset,
            (group_exp + FP32_EXP_BIAS).to(tl.uint8),
            # Prevent writing outside this chunk or the main array.
            mask=exp_offset < output_bound,
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
                    local_input_offset + input_start,
                    [GROUP_LOAD * GROUP_SIZE // 2, 2],
                    can_reorder=True,
                )
            )
            # Split down to fourth.
            philox_4x_offset = tl.split(
                tl.reshape(
                    philox_4x_offset,
                    [GROUP_LOAD * GROUP_SIZE // 4, 2],
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
                stochastic_round_bits, [GROUP_LOAD * GROUP_SIZE]
            ).to(tl.int32, bitcast=True)

            # Mask off mantissa bits of random value and add to mantissa.
            scaled_a = scaled_a + (stochastic_round_bits & RAND_MASK)

        if (
            E2M1_FAST_PATH
            and EBITS == 2
            and MBITS == 1
            and FP4_EXP_BIAS == 1
            and not STOCHASTIC_CASTING
        ):
            mx4_value = _quantize_e2m1(scaled_a)
        else:
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
            # The shift can reach 45 for tiny values. Shifting an int32 by >= 32 is undefined:
            # NVIDIA saturates to 0 but AMD GPUs only use the low 5 bits of the shift amount.
            # Mantissa fits in 24 bits, so clamping to 31 gives 0 on all backends.
            mantissa_shift = tl.minimum(
                fp32_sig_bits + exp_diff - MBITS_IMPLICIT - 1, 31
            )
            mantissa = mantissa >> mantissa_shift
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
            mantissa = tl.where(
                new_biased_exp > EXPONENT_OVERFLOW_THRESHOLD, 1, mantissa
            )

            # Construct FP4 value from components.
            new_biased_exp = tl.maximum(  # noqa: TR012 (tl.clamp is float-only; these are int32)
                tl.minimum(new_biased_exp, EXPONENT_OVERFLOW_THRESHOLD),  # noqa: TR012
                0,
            )
            mx4_value = (new_biased_exp << (MBITS_IMPLICIT - 1)) | mantissa
            mx4_value = (sign_bit << (EBITS + MBITS)) | mx4_value

        # Extract low and high bits from values.
        low_mx4, high_mx4 = tl.split(
            tl.reshape(mx4_value, [(GROUP_LOAD * GROUP_SIZE) // 2, 2])
        )
        # Shift mx4 values together so they are packed into int8.
        packed_mx4 = ((high_mx4 << 4) | (low_mx4)).to(tl.int8)

        # Write out packed values to output tensor.
        tl.store(
            out_chunk + output_offset,
            packed_mx4,
            # Prevent writing outside this chunk or the main array.
            mask=output_offset < output_bound,
        )

        # Update offsets so we work on the next block.
        local_input_offset += GROUP_LOAD * GROUP_SIZE
        exp_offset += GROUP_LOAD * PACKED_GROUP_SIZE
        output_offset += GROUP_LOAD * PACKED_GROUP_SIZE


def triton_quantize_mx4(
    a: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: RoundingMode | int = RoundingMode.ceil,
    stochastic_casting: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Quantize a tensor to mx4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target mx4 format.
        mbits (int): Number of bits to use for mantissa in target mx4 format.
        rounding_mode (RoundingMode | int): Which type of rounding to use
        when calculating shared exponent. Defaults to pre-rounding to nearest even int.
        stochastic_casting (bool): Whether to use stochastic casting.
        out (Tensor, optional): Contiguous uint8 tensor with M / 2 + M / group_size
        elements to write the result into, e.g. a communication buffer.

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
    # For simplicity, view input as a 2D array.
    a = a.view(-1, a.shape[-1])
    # Extract rows and columns.
    M, K = a.shape
    # In this kernel, we want each row to be divisible by group_size.
    # If the rows are not, then we will pad them. Find the number of
    # groups per row after padding.
    groups_per_row = math.ceil(K / group_size)
    num_groups = M * groups_per_row
    # Find how many groups each thread should process. We do this
    # by assuming that it is good to distribute work evenly over threads.
    num_threads = math.ceil(math.sqrt(a.numel()))
    # Data is loaded in chunks of GROUP_LOAD elements, so theres no reason
    # to ever fewer groups per thread than it.
    GROUP_LOAD = 64
    groups_per_thread = max(math.ceil(num_groups / num_threads), GROUP_LOAD)
    # Determine how much padding, if any is needed for each row.
    if K % group_size != 0:
        padding = group_size - (K % group_size)
    else:
        padding = 0

    # Create output tensor.
    out_elems = (num_groups * group_size) // 2 + num_groups
    if out is None:
        out = torch.empty([out_elems], device=a.device, dtype=torch.uint8)
    else:
        assert (
            out.dtype == torch.uint8
            and out.is_contiguous()
            and out.numel() == out_elems
        ), f"out must be a contiguous uint8 tensor with {out_elems} elements"
        out = out.view(-1)

    # If using stochastic rounding, create random noise for each group.
    # We use the same random bits as seeds when doing stochastic downcasting.
    if rounding_mode == RoundingMode.stochastic or stochastic_casting:
        # Each group will need a seed.
        rand_bits = torch.randint(
            low=0,
            high=2**31 - 1,
            size=(num_groups,),
            dtype=torch.int32,
            device=a.device,
        )
    else:
        rand_bits = None

    # Check if we need to use int64 for indexing.
    use_int64 = num_threads * groups_per_thread * group_size > 2**31 - 1
    fast_path = FeatureGate.is_enabled(FeatureGateName.TRITON_MX4_QUANTIZE_FAST_PATH)

    # Invoke triton quantization kernel over rows.
    grid = (num_threads,)
    _kernel_quantize_mx4[grid](
        a,
        out,
        rand_bits=rand_bits,
        M=M,
        K=K,
        GROUPS_PER_ROW=groups_per_row,
        GROUPS_PER_THREAD=groups_per_thread,
        ROW_PADDING=padding,
        # pyre-ignore[6]
        GROUP_SIZE=group_size,
        # pyre-ignore[6]
        EBITS=ebits,
        # pyre-ignore[6]
        MBITS=mbits,
        # pyre-ignore[6]
        ROUNDING_MODE=rounding_mode,
        # pyre-ignore[6]
        STOCHASTIC_CASTING=stochastic_casting,
        FP4_EXP_BIAS=get_mx4_exp_bias(ebits),
        # pyre-ignore[6]
        GROUP_LOAD=GROUP_LOAD,
        # pyre-ignore[6]
        USE_INT64=use_int64,
        # pyre-ignore[6]
        HAS_PADDING=padding != 0 or not fast_path,
        # pyre-ignore[6]
        E2M1_FAST_PATH=fast_path,
    )
    # Inputs are now fully quantized and ready to return.
    # Try to return in the original shape if possible.
    try:
        output_shape = list(orig_shape[:-1]) + [-1]
        return out.view(output_shape)
    # If we cant, return as a flat array.
    except RuntimeError:
        return out.view(-1)


@triton.jit
def _dequantize_mx4_pairs(a, exp, mx4_lookup_table):
    """FP32 values of the low and high nibbles of packed MX4 bytes a.

    exp is the stored (biased) shared exponent, broadcastable to a.
    """
    MX4_BIT_MASK: tl.constexpr = 0xF  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
    # Get equivalent fp32 values.
    low_fp32 = tl.load(mx4_lookup_table + (a & MX4_BIT_MASK))
    high_fp32 = tl.load(mx4_lookup_table + ((a >> 4) & MX4_BIT_MASK))
    # Remove fp32 exponent bias.
    exp = exp.to(tl.int32) - FP32_EXP_BIAS
    # Compute scale = 2^exp exactly via FP32 bit construction instead of a slow
    # FP64 exp2. exp is an integer in [-127, 125]; for e >= -126, 2^e is a normal
    # fp32 (zero mantissa) so we set the exponent field directly. e == -127 ->
    # 2^-127 is subnormal and cannot be built that way, so special-case it.
    scale = tl.where(
        exp >= -126,
        ((exp + 127) << 23).to(tl.float32, bitcast=True),
        2.0**-127,
    )
    return scale * low_fp32, scale * high_fp32


@triton.jit
def _kernel_dequantize_mx4(
    A,
    mx4_lookup_table,
    out,
    M,
    GROUPS_PER_THREAD,
    INV_DIVISOR,
    GROUP_SIZE: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
    USE_INT64: tl.constexpr,
    HAS_DIVISOR: tl.constexpr,
) -> None:
    """Dequantize a packed MX4 tensor and apply scaling.

    Args:
        A (Tensor): [M] MX4 tensor packed into int8.
        shared_exp (Tensor): Int8 tensor representing group exponent.
        mx4_lookup_table (Tensor): Map from mx4 integer value to floating point.
        M (int): Total number of elements in input.
        GROUPS_PER_THREAD (int): Number of groups each thread is responsible for.
        INV_DIVISOR (float): FP32 reciprocal of the divisor, used when HAS_DIVISOR.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing.
        HAS_DIVISOR (bool): Whether to divide outputs by the divisor.
    """
    # Define constants.
    PACKED_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2 + 1  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
    # For very large tensors, use int64 for indexing. This is slower but necessary.
    if USE_INT64:
        pid = pid.to(tl.int64)
        M = tl.cast(M, tl.int64)
        GROUPS_PER_THREAD = tl.cast(GROUPS_PER_THREAD, tl.int64)

    # Boundaries for reading input and writing to output tensor.
    INPUT_CHUNK_SIZE = GROUPS_PER_THREAD * PACKED_GROUP_SIZE
    OUTPUT_CHUNK_SIZE = GROUPS_PER_THREAD * GROUP_SIZE
    OUTPUT_SIZE = (M // PACKED_GROUP_SIZE) * GROUP_SIZE

    # Find the starting offsets for this thread.
    input_start = pid * (GROUPS_PER_THREAD * PACKED_GROUP_SIZE)
    exp_start = input_start + GROUP_SIZE // 2
    # Remove shared exponents from output offset.
    output_start = pid * OUTPUT_CHUNK_SIZE
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
    for _k in range(0, tl.cdiv(GROUPS_PER_THREAD, GROUP_LOAD)):
        a = tl.load(
            A + input_offset,
            # Mask values that are out of this chunk or the main array.
            mask=(input_offset < M) & (input_offset < (INPUT_CHUNK_SIZE * (pid + 1))),
            other=0.0,
        )
        # Get proper shared exponent of each byte.
        exp = tl.load(
            A + exp_offset,
            mask=(exp_offset < M) & (exp_offset < (INPUT_CHUNK_SIZE * (pid + 1))),
            other=0.0,
        )
        scaled_low_fp32, scaled_high_fp32 = _dequantize_mx4_pairs(
            a, exp, mx4_lookup_table
        )

        # Combine the two components into a single tensor, interweave them.
        scaled_fp32 = tl.interleave(scaled_low_fp32, scaled_high_fp32)
        scaled_out = scaled_fp32.to(out.dtype.element_ty)
        if HAS_DIVISOR:
            # Same rounding as out.div_(divisor), which multiplies by the FP32 reciprocal.
            scaled_out = (scaled_out.to(tl.float32) * INV_DIVISOR).to(
                out.dtype.element_ty
            )

        # Write final outputs.
        tl.store(
            out + output_offset,
            scaled_out,
            # Mask values that are out of this chunk or the main array.
            mask=(output_offset < OUTPUT_SIZE)
            & (output_offset < OUTPUT_CHUNK_SIZE * (pid + 1)),
        )

        # Update indices for next group.
        input_offset += GROUP_LOAD * PACKED_GROUP_SIZE
        exp_offset += GROUP_LOAD * PACKED_GROUP_SIZE
        output_offset += GROUP_LOAD * GROUP_SIZE


def triton_dequantize_mx4(
    a: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
    output_dtype: torch.dtype = torch.float32,
    divisor: int | None = None,
) -> torch.Tensor:
    """
    Dequantize a tensor from mx4 format to fp32 or bf16.

    Args:
        a (Tensor): [M / 2 + M / group_size] MX4 tensor packed into int8 values
        with group exponents attached to end of each row.
        group_size (int): Size of chunks that use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target mx4 format.
        mbits (int): Number of bits to use for mantissa in target mx4 format.
        output_dtype (torch.dtype): Output dtype (FP32 or BF16).
            Defaults to torch.float32 for backward compatibility.
        divisor (int, optional): If set, outputs are divided by it, bitwise identical
            to calling out.div_(divisor) afterwards.

    Returns:
        torch.Tensor: [M, K] dequantized tensor in the specified dtype.
    """
    # If given an empty shape, return an empty tensor.
    if a.numel() == 0:
        return torch.empty(a.shape, device=a.device, dtype=output_dtype)
    # View a as 2D for simplicity.
    orig_shape = a.shape
    a = a.flatten()
    # Find number of groups.
    packed_group_size = group_size // 2 + 1
    num_groups = a.numel() // packed_group_size
    # Find a workload that distributes work evenly over threads.
    num_threads = math.ceil(math.sqrt(a.numel()))
    # There is no need to ever have fewer groups per thread than the amount
    # loaded at once.
    GROUP_LOAD = 64
    groups_per_thread = max(math.ceil(num_groups / num_threads), GROUP_LOAD)

    # Use a lookup table to convert
    mx4_to_fp_values = get_mx4_lookup_table(ebits, mbits, a.device)

    # Create output tensor in target dtype.
    output_elems = num_groups * group_size
    out = torch.empty([output_elems], device=a.device, dtype=output_dtype)
    # Check if we need to use int64 for indexing.
    use_int64 = num_threads * groups_per_thread * group_size > 2**31 - 1
    # out.div_(divisor) multiplies by the FP32 reciprocal. For integer divisors below 2**24,
    # rounding the float64 reciprocal to FP32 (done when launching the kernel) gives exactly it.
    assert divisor is None or 0 < divisor < 2**24, f"unsupported divisor {divisor}"
    inv_divisor = 1.0 if divisor is None else 1.0 / divisor
    # Invoke triton dequantization kernel over rows.
    grid = (num_threads,)
    _kernel_dequantize_mx4[grid](
        a,
        mx4_to_fp_values,
        out,
        a.numel(),
        GROUPS_PER_THREAD=groups_per_thread,
        INV_DIVISOR=inv_divisor,
        # pyre-ignore[6]
        GROUP_SIZE=group_size,
        # pyre-ignore[6]
        GROUP_LOAD=GROUP_LOAD,
        # pyre-ignore[6]
        USE_INT64=use_int64,
        # pyre-ignore[6]
        HAS_DIVISOR=divisor is not None,
    )

    out_shape = list(orig_shape[:-1]) + [-1]
    return out.view(out_shape)


@functools.lru_cache(maxsize=64)
def _column_block_info(
    num_rows: int,
    group_size: int,
    padded_cols: tuple[int, ...],
    cols: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Per column block, int64 [first MX4 group in the packed stream, padded width, width,
    first column]. Blocks are num_rows x padded_cols[i], packed one after another."""
    info, first_group, first_col = [], 0, 0
    for padded, width in zip(padded_cols, cols):
        info.append([first_group, padded, width, first_col])
        first_group += num_rows * padded // group_size
        first_col += width
    return torch.tensor(info, dtype=torch.int64, device=device)


@triton.jit
def _kernel_dequantize_mx4_to_columns(
    A,
    mx4_lookup_table,
    out,
    block_info,
    num_rows,
    total_cols,
    GROUP_SIZE: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    """Dequantize packed MX4 blocks into the column blocks of a [num_rows, total_cols] output.

    Block b of A is num_rows x padded_cols[b], row-major; its first cols[b] columns go to
    output columns [first_col[b], first_col[b] + cols[b]) and the rest (padding) is dropped.
    padded_cols[b] is a multiple of GROUP_SIZE, so every group lies within one row.
    Program (i, b) handles groups [i * GROUPS_PER_PROGRAM, (i + 1) * GROUPS_PER_PROGRAM) of
    block b.

    Args:
        A (Tensor): Packed MX4 input, blocks one after another.
        mx4_lookup_table (Tensor): Map from mx4 integer value to floating point.
        out (Tensor): [num_rows, total_cols] output.
        block_info (Tensor): Per block, int64 [first group in A, padded_cols, cols, first_col].
        num_rows (int): Rows of every block and of the output.
        total_cols (int): Output columns.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        GROUPS_PER_PROGRAM (int): Groups per program, a multiple of GROUP_LOAD.
        USE_INT64 (bool): Whether offsets into A or out need int64.
    """
    PACKED_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2 + 1  # type: ignore[Incompatible variable type]
    block = tl.program_id(1)
    first_group = tl.load(block_info + 4 * block)
    groups_per_row = tl.load(block_info + 4 * block + 1).to(tl.int32) // GROUP_SIZE
    cols = tl.load(block_info + 4 * block + 2).to(tl.int32)
    first_col = tl.load(block_info + 4 * block + 3)
    if not USE_INT64:
        first_group = first_group.to(tl.int32)
        first_col = first_col.to(tl.int32)
    num_groups = num_rows * groups_per_row
    start = tl.program_id(0) * GROUPS_PER_PROGRAM
    byte_offset = tl.arange(0, GROUP_SIZE // 2)
    elem_offset = tl.arange(0, GROUP_SIZE)

    for _k in range(
        0, tl.cdiv(tl.minimum(GROUPS_PER_PROGRAM, num_groups - start), GROUP_LOAD)
    ):
        group = start + _k * GROUP_LOAD + tl.arange(0, GROUP_LOAD)
        valid = group < num_groups
        row = group // groups_per_row
        col = (group % groups_per_row) * GROUP_SIZE
        if USE_INT64:
            row = row.to(tl.int64)
        packed = (first_group + group) * PACKED_GROUP_SIZE
        a = tl.load(
            A + packed[:, None] + byte_offset[None, :], mask=valid[:, None], other=0
        )
        exp = tl.load(A + packed + GROUP_SIZE // 2, mask=valid, other=0)
        low_fp32, high_fp32 = _dequantize_mx4_pairs(a, exp[:, None], mx4_lookup_table)
        out_col = col[:, None] + elem_offset[None, :]
        tl.store(
            out + (row * total_cols + first_col)[:, None] + out_col,
            tl.interleave(low_fp32, high_fp32).to(out.dtype.element_ty),
            # Skip padding columns.
            mask=valid[:, None] & (out_col < cols),
        )


def triton_dequantize_mx4_to_columns(
    a: torch.Tensor,
    group_size: int,
    num_rows: int,
    padded_cols: list[int],
    cols: list[int],
    ebits: int = 2,
    mbits: int = 1,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize MX4 holding one [num_rows, padded_cols[i]] block per i, and lay the blocks
    side by side, dropping each block's padding columns. Same result as

        blocks = triton_dequantize_mx4(a, group_size, ebits, mbits, output_dtype).split(
            [num_rows * p for p in padded_cols])
        torch.cat([b.view(num_rows, p)[:, :c] for b, p, c in zip(blocks, padded_cols, cols)], 1)

    but written straight into that layout.

    Args:
        a (Tensor): Packed MX4 tensor, blocks one after another.
        group_size (int): Size of chunks that use the same shared exponent.
        num_rows (int): Rows of every block.
        padded_cols (list[int]): Width of each block, a multiple of group_size.
        cols (list[int]): Columns kept from each block, at most its padded width.
        ebits (int): Number of bits to use for exponent in target mx4 format.
        mbits (int): Number of bits to use for mantissa in target mx4 format.
        output_dtype (torch.dtype): Output dtype (FP32 or BF16).

    Returns:
        torch.Tensor: [num_rows, sum(cols)] dequantized tensor.
    """
    assert all(
        p % group_size == 0 and 0 <= c <= p for p, c in zip(padded_cols, cols)
    ), f"bad column blocks {padded_cols=} {cols=} for {group_size=}"
    packed_group_size = group_size // 2 + 1
    assert (
        a.numel()
        == sum(num_rows * p for p in padded_cols) // group_size * packed_group_size
    ), "input size does not match the column blocks"
    out = torch.empty([num_rows, sum(cols)], device=a.device, dtype=output_dtype)
    max_groups = max((num_rows * p // group_size for p in padded_cols), default=0)
    if out.numel() == 0 or max_groups == 0:
        return out
    assert max_groups * group_size < 2**31, "column block too large"
    GROUP_LOAD = 64
    GROUPS_PER_PROGRAM = 16 * GROUP_LOAD
    grid = (triton.cdiv(max_groups, GROUPS_PER_PROGRAM), len(cols))
    _kernel_dequantize_mx4_to_columns[grid](
        a,
        get_mx4_lookup_table(ebits, mbits, a.device),
        out,
        _column_block_info(
            num_rows, group_size, tuple(padded_cols), tuple(cols), a.device
        ),
        num_rows,
        out.shape[1],
        # pyre-ignore[6]
        GROUP_SIZE=group_size,
        # pyre-ignore[6]
        GROUP_LOAD=GROUP_LOAD,
        # pyre-ignore[6]
        GROUPS_PER_PROGRAM=GROUPS_PER_PROGRAM,
        # pyre-ignore[6]
        USE_INT64=max(a.numel(), out.numel()) > 2**31 - 1,
    )
    return out


@triton.jit
def _kernel_quantize_mx4_from_columns(
    A,
    out,
    block_info,
    num_rows,
    total_cols,
    GROUP_SIZE: tl.constexpr,
    ROUNDING_MODE: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    """Quantize the column blocks of a [num_rows, total_cols] tensor to e2m1 MX4, one block
    after another.

    Block b is A[:, first_col[b] : first_col[b] + cols[b]] taken row-major, so a group can
    continue onto the next row; cols[b] >= GROUP_SIZE (or 0), so it does so at most once.
    Writes the same bytes as _kernel_quantize_mx4 on the concatenated blocks. Program (i, b)
    handles groups [i * GROUPS_PER_PROGRAM, (i + 1) * GROUPS_PER_PROGRAM) of block b.

    Args:
        A (Tensor): [num_rows, total_cols] input.
        out (Tensor): Packed MX4 output.
        block_info (Tensor): Per block, int64 [first group in out, cols, cols, first_col].
        num_rows (int): Rows of A.
        total_cols (int): Columns of A.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent;
            not stochastic.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        GROUPS_PER_PROGRAM (int): Groups per program, a multiple of GROUP_LOAD.
        USE_INT64 (bool): Whether offsets into A or out need int64.
    """
    PACKED_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2 + 1  # type: ignore[Incompatible variable type]
    block = tl.program_id(1)
    first_group = tl.load(block_info + 4 * block)
    cols = tl.load(block_info + 4 * block + 2).to(tl.int32)
    first_col = tl.load(block_info + 4 * block + 3)
    if not USE_INT64:
        first_group = first_group.to(tl.int32)
        first_col = first_col.to(tl.int32)
    num_groups = num_rows * cols // GROUP_SIZE
    start = tl.program_id(0) * GROUPS_PER_PROGRAM
    byte_offset = tl.arange(0, GROUP_SIZE // 2)
    elem_offset = tl.arange(0, GROUP_SIZE)

    for _k in range(
        0, tl.cdiv(tl.minimum(GROUPS_PER_PROGRAM, num_groups - start), GROUP_LOAD)
    ):
        group = start + _k * GROUP_LOAD + tl.arange(0, GROUP_LOAD)
        valid = group < num_groups
        first_elem = group * GROUP_SIZE
        row = first_elem // cols
        if USE_INT64:
            row = row.to(tl.int64)
        col = (first_elem % cols)[:, None] + elem_offset[None, :]
        # Elements past the end of the row continue at the start of the next row.
        in_offset = (
            (row * total_cols + first_col)[:, None]
            + col
            + tl.where(col >= cols, total_cols - cols, 0)
        )
        a = tl.load(A + in_offset, mask=valid[:, None], other=0)
        scaled_a, group_exp = _scale_groups(a, None, ROUNDING_MODE, 2, 1)
        mx4_value = _quantize_e2m1(scaled_a.to(tl.int32, bitcast=True))
        low_mx4, high_mx4 = tl.split(
            tl.reshape(mx4_value, [GROUP_LOAD, GROUP_SIZE // 2, 2])
        )
        packed = (first_group + group) * PACKED_GROUP_SIZE
        tl.store(
            out + packed[:, None] + byte_offset[None, :],
            ((high_mx4 << 4) | low_mx4).to(tl.int8),
            mask=valid[:, None],
        )
        tl.store(
            out + packed + GROUP_SIZE // 2,
            (group_exp + 127).to(tl.uint8),
            mask=valid,
        )


def triton_quantize_mx4_from_columns(
    a: torch.Tensor,
    group_size: int,
    cols: list[int],
    rounding_mode: RoundingMode | int = RoundingMode.ceil,
) -> torch.Tensor:
    """
    Quantize the column blocks of a to e2m1 MX4, one block after another. Same result as

        triton_quantize_mx4(
            torch.cat([c.contiguous().view(-1) for c in a.split(cols, dim=1)]),
            group_size, rounding_mode=rounding_mode).view(-1)

    but read straight from a.

    Args:
        a (Tensor): [num_rows, sum(cols)] contiguous input.
        group_size (int): Size of chunks that use the same shared exponent.
        cols (list[int]): Width of each block; num_rows * cols[i] must be a multiple of
            group_size.
        rounding_mode (RoundingMode | int): Which type of rounding to use when calculating
            the shared exponent; not stochastic.

    Returns:
        torch.Tensor: [M / 2 + M / group_size] packed MX4, M = a.numel().
    """
    num_rows, total_cols = a.shape
    assert a.is_contiguous() and total_cols == sum(cols), f"bad input for {cols=}"
    assert all(
        num_rows * c % group_size == 0 for c in cols
    ), f"{num_rows=} x each of {cols=} must be a multiple of {group_size=}"
    assert rounding_mode != RoundingMode.stochastic, "stochastic rounding unsupported"
    max_elems = max((num_rows * c for c in cols), default=0)
    if (
        not FeatureGate.is_enabled(FeatureGateName.TRITON_MX4_QUANTIZE_FAST_PATH)
        or any(0 < c < group_size for c in cols)
        or max_elems >= 2**31
    ):
        # Fast path disabled, a group could span more than two rows, or a block would
        # overflow int32 offsets: regroup first.
        regrouped = torch.cat([c.contiguous().view(-1) for c in a.split(cols, dim=1)])
        return triton_quantize_mx4(
            regrouped, group_size, rounding_mode=rounding_mode
        ).view(-1)
    out = torch.empty(
        [a.numel() // 2 + a.numel() // group_size], device=a.device, dtype=torch.uint8
    )
    if max_elems == 0:
        return out
    GROUP_LOAD = 64
    GROUPS_PER_PROGRAM = 16 * GROUP_LOAD
    grid = (triton.cdiv(max_elems // group_size, GROUPS_PER_PROGRAM), len(cols))
    _kernel_quantize_mx4_from_columns[grid](
        a,
        out,
        _column_block_info(num_rows, group_size, tuple(cols), tuple(cols), a.device),
        num_rows,
        total_cols,
        # pyre-ignore[6]
        GROUP_SIZE=group_size,
        # pyre-ignore[6]
        ROUNDING_MODE=rounding_mode,
        # pyre-ignore[6]
        GROUP_LOAD=GROUP_LOAD,
        # pyre-ignore[6]
        GROUPS_PER_PROGRAM=GROUPS_PER_PROGRAM,
        # pyre-ignore[6]
        USE_INT64=a.numel() > 2**31 - 1,
    )
    return out
