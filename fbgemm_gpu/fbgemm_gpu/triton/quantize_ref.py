#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import torch


def py_quantize_mx4(a: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Quantize a tensor to mx4 format.

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
    # Make sure input has a supported shape.
    if a.numel() % 32 != 0:
        raise RuntimeError(
            f"Input must have total elements that are a multiple of 32, but got {a.numel()} elements."
        )
    # Keep track of original shape.
    orig_shape = a.shape
    # Prepare for grouping by subdiving the last axis.
    a = a.view(a.numel() // group_size, group_size)
    # Now we can easily compute the shared exponents for each group.
    shared_exp, _ = torch.max(torch.abs(a), dim=1, keepdim=True)
    # Replace zero values with the minimum expressible normal value.
    FP32_MIN_NORMAL = 2 ** (-126)
    shared_exp = torch.where(shared_exp == 0, FP32_MIN_NORMAL, shared_exp)
    # Convert max into an intger exponent.
    # Note this can be more efficient by just shifting and masking exp bits.
    # We can even use those directly.
    shared_exp = torch.floor(torch.log2(shared_exp))
    # Offset exponent by largest exponent in target datatype.
    shared_exp = shared_exp - 2
    # Restrict to range expressible as int8.
    shared_exp = torch.clamp(shared_exp, min=-127, max=127)
    # Convert exponent to scale and apply to input.
    # Need to do this calculation on cpu for accuracy.
    _shared_exp = shared_exp.cpu()
    scale = (2**_shared_exp).to(device=a.device)
    a = a / scale
    # View as integer for bitwise ops.
    a = a.view(torch.int32)

    # Quantization step: convert fp32 values to fp4.
    # Start by extracting float components.
    FP32_SIGN_OFFSET = 31
    sign_bit = torch.bitwise_right_shift(a, FP32_SIGN_OFFSET).to(torch.int8)
    # Torch does arithmetic shifts so we need to isolate sign bit.
    SIGN_MASK = 0x1
    sign_bit = torch.bitwise_and(sign_bit, SIGN_MASK)

    # Next extract exponent.
    FP32_EXP_MASK = 0x7F800000
    biased_exp = torch.bitwise_and(a, FP32_EXP_MASK)
    # Shift exponent over to least significant bits.
    FP32_EXP_OFFSET = 23
    biased_exp = torch.bitwise_right_shift(biased_exp, FP32_EXP_OFFSET).to(torch.int8)

    # Finally extract the mantissa.
    FP32_MANTISSA_MASK = 0x007FFFFF
    trailing_mantissa = torch.bitwise_and(a, FP32_MANTISSA_MASK)

    # Fp4 has 2 mantissa bits, one explicit, one implicit.
    MBITS = 2
    # FP32 and and FP4 have very different exponent biases, adjust to fp4.
    FP32_EXP_BIAS = 127
    FP4_EXP_BIAS = 1
    new_biased_exp = biased_exp - FP32_EXP_BIAS + FP4_EXP_BIAS

    # Compute difference between ideal exponent and what can be represented.
    exp_diff = torch.where(new_biased_exp <= 0, 1 - new_biased_exp, 0)
    # Clip this difference to the maximum number of fp32 mantissa bits (23 + implicit).
    MAX_FP32_MANTISSA_BITS = 24
    exp_diff = torch.clamp(exp_diff, max=MAX_FP32_MANTISSA_BITS)

    # Now perform mantissa rounding down to fp4.
    is_subnorm = biased_exp == 0
    # Add implied 1 to normal values.
    mantissa = torch.where(is_subnorm, trailing_mantissa, trailing_mantissa + (1 << 23))
    # Compute base number of bits corresponding to the mantissa. We use a smaller value
    # for subnorms since implicit one is included in exp_diff above.
    fp32_sig_bits = torch.where(is_subnorm, 23, 24).to(torch.int32)
    # Shift down to target bitwidth - 1 and efficiently represent.
    mantissa = torch.bitwise_right_shift(
        mantissa, fp32_sig_bits + exp_diff - MBITS - 1
    ).to(torch.int8)
    # Perform rounding by adding 1 then shifting down.
    mantissa = mantissa + 1
    mantissa = torch.bitwise_right_shift(mantissa, 1)

    # Check for overflow and adjust exponent accordingly.
    OVERFLOW_THRESHOLD = 4
    overflow = mantissa >= OVERFLOW_THRESHOLD
    # Allow subnorms to overflow into normals, otherwise shift off overflow.
    mantissa = torch.where(
        torch.bitwise_and(overflow, torch.bitwise_not(is_subnorm)),
        torch.bitwise_right_shift(mantissa, 1),
        mantissa,
    )
    # Special case where a value is subnorm and has a large mantissa, overflow it.
    new_biased_exp = torch.where(
        torch.bitwise_and(new_biased_exp <= 0, mantissa == 2), 1, new_biased_exp
    )
    # Remove implicit 1.
    IMPLICIT_1_MASK = 0x1
    mantissa = torch.bitwise_and(mantissa, IMPLICIT_1_MASK)
    # Add overflow to exponent.
    new_biased_exp = torch.where(overflow, new_biased_exp + 1, new_biased_exp)
    # If exp overflows, set mantissa so we're at max representable value.
    mantissa = torch.where(new_biased_exp >= OVERFLOW_THRESHOLD, 1, mantissa)

    # Construct fp4 value from components.
    new_biased_exp = torch.clamp(new_biased_exp, min=0, max=3)
    mx4_value = torch.bitwise_or(torch.bitwise_left_shift(new_biased_exp, 1), mantissa)
    mx4_value = torch.bitwise_or(torch.bitwise_left_shift(sign_bit, 3), mx4_value)

    # Pack int4 values into single int8 outputs.
    low_mx4 = mx4_value[:, ::2]
    high_mx4 = mx4_value[:, 1::2]
    high_mx4 = torch.bitwise_left_shift(high_mx4, 4)
    packed_mx4 = torch.bitwise_or(low_mx4, high_mx4)

    # Ravel packed values together with shared exponent.
    packed_mx4 = torch.concat(
        [
            packed_mx4.view(-1, group_size // 2),
            (shared_exp + FP32_EXP_BIAS).to(torch.int8).view(-1, 1),
        ],
        dim=1,
    )

    # Inputs are now fully quantized and ready to return.
    # Try to return in the original shape if possible.
    if orig_shape[-1] % group_size == 0:
        output_shape = list(orig_shape[:-1]) + [-1]
        return packed_mx4.view(output_shape).view(torch.uint8)
    # If we cant, return as a flat array.
    else:
        return packed_mx4.view(-1).view(torch.uint8)


def py_dequantize_mx4(a: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Dequantize a tensor from mx4 format to fp32.

    Args:
        a (Tensor): [M / 2 + M / group_size] MX4 tensor packed into int8 values
        with group exponents attached to end of each row.
        group_size (int): Size of chunks that use the same shared exponent.

    Returns:
        torch.Tensor: [M] dequantized fp32 tensor.
    """
    # If given an empty shape, return an empty tensor.
    if a.numel() == 0:
        return torch.empty(a.shape, device=a.device, dtype=torch.float32)
    # Keep track of starting shape.
    orig_shape = a.shape
    device = a.device
    # Unravel packed inputs from shared exponents.
    a = a.view(-1, (group_size // 2) + 1).view(torch.int8)
    num_groups = a.numel() // ((group_size // 2) + 1)
    packed_input = a[:, :-1]
    shared_exp = a[:, -1:]
    # Remove fp32 exponent bias
    FP32_EXP_BIAS = 127
    shared_exp = shared_exp - FP32_EXP_BIAS
    # First pull shared exponent off the end of each row.
    M, K_2 = packed_input.shape

    # Pull out high and low mx4 values.
    FP4_BIT_MASK = 0xF
    low_mx4 = torch.bitwise_and(packed_input, FP4_BIT_MASK)
    high_mx4 = torch.bitwise_right_shift(packed_input, 4)
    # Remove sign bit from high values since shift was arithmetic.
    high_mx4 = torch.bitwise_and(high_mx4, FP4_BIT_MASK)
    # Recombine into a single tensor.
    a = torch.stack([low_mx4, high_mx4], dim=0).view(2, -1).t().contiguous()

    # Use a lookup table to convert
    mx4_to_fp_values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=device,
        dtype=torch.float,
    )
    # Convert values into float32 equivalent via lookup.
    out = torch.index_select(mx4_to_fp_values, 0, a.to(torch.int32).view(-1))

    # Exponent needs to be computed on cpu for perfect precision.
    _shared_exp = shared_exp.cpu().to(torch.float)
    scale = (2**_shared_exp).to(device)

    # Finally, apply shared exponent to restore full value.
    out = out.view(-1, num_groups, group_size) * scale.view(1, num_groups, 1)
    # Restore original shape and return.
    out_shape = list(orig_shape[:-1]) + [-1]
    return out.view(out_shape)
