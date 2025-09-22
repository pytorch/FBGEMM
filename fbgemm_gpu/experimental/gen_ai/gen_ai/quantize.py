# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Helper functions for using FBGEMM quantized operators.


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_bit = 4  # Number of target bits.
    # Split input into chunks of group_size. This approach allows K that isnt divisible by group_size.
    to_quant = torch.split(x.to(torch.float), group_size, dim=-1)

    max_val = [chunk.amax(dim=1, keepdim=True) for chunk in to_quant]
    min_val = [chunk.amin(dim=1, keepdim=True) for chunk in to_quant]
    max_int = 2**n_bit - 1
    min_int = 0
    scales = [
        (max_chunk - min_chunk).clamp(min=1e-6) / max_int
        for max_chunk, min_chunk in zip(max_val, min_val)
    ]

    zeros = [
        min_chunk + scale_chunk * (2 ** (n_bit - 1))
        for min_chunk, scale_chunk in zip(min_val, scales)
    ]

    out = [
        chunk.sub(min_chunk).div(scale_chunk).round().clamp_(min_int, max_int)
        for chunk, min_chunk, scale_chunk in zip(to_quant, min_val, scales)
    ]

    # Recenter output and move to int8.
    out = [(chunk - 2 ** (n_bit - 1)).to(dtype=torch.int8) for chunk in out]

    # Recombine chunks.
    out = torch.cat(out, dim=-1)

    # Cutlass expects column major layout for scale and zero point,
    # so we transpose here and make them contiguous.
    scales = torch.cat(scales, dim=-1).t().contiguous()
    zeros = torch.cat(zeros, dim=-1).t().contiguous()

    return out, scales, zeros


def int4_row_quantize(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to quantize a tensor to int4 with groupwise scales.

    Args:
        x (Tensor): [N, K] Higher precision weight tensor to quantize.
        group_size (int): Number of elements to calculate group scale for.
    Returns:
        wq (Tensor): [N, K] Quantized int4 tensor stored in int8 elements.
        group_scale (Tensor): [K / group_size, N] FP32 Scale per group.
    """
    n_bit = 4  # Number of target bits.
    # Split input into chunks of group_size. This approach allows K that isnt divisible by group_size.
    to_quant = torch.split(x.to(torch.float), group_size, dim=-1)

    max_val = [torch.abs(chunk).amax(dim=-1, keepdim=True) for chunk in to_quant]
    max_int = 2 ** (n_bit - 1)
    min_int = -(2 ** (n_bit - 1))
    scales = [chunk.clamp(min=1e-6) / max_int for chunk in max_val]

    out = [
        chunk.div(chunk_scale).round().clamp_(min_int, max_int - 1)
        for chunk, chunk_scale in zip(to_quant, scales)
    ]
    # Recombine chunks.
    out = torch.cat(out, dim=-1)

    # Cast to int8 and restore shape.
    out = out.to(dtype=torch.int8)

    # Scales should be in [num_groups, N] layout.
    scales = torch.cat(scales, dim=-1).t().contiguous()

    return out, scales


def quantize_int4_preshuffle(
    w: torch.Tensor, group_size: int = 128, dtype: str = "fp8", use_zp: bool = True
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

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


def shuffle_slice(
    x: torch.Tensor, dim: int, start: int, length: int, dtype: str = "fp8"
) -> torch.Tensor:
    """
    Helper function to slice a preshuffled int4 tensor. This is needed since the shuffling
    reorders rows based on the size of the input. Slicing a tensor shuffled for a larger input
    is no longer valid. We must reorder the tensor to the appropriate size then slice.
    Args:
        x (Tensor): [N, K // 2] Preshuffled int4 tensor.
        dim (int): Dimension to slice.
        start (int): Start of slice.
        length (int): Number of elements to slice in the original [N, K] dimension.
        dtype (str): Type of corresponding activations. Must be fp8 or bf16.
    Returns:
        sliced (Tensor): [stop-start, K // 2] Sliced tensor.
    """
    # Get the size of the input tensor.
    assert dim in [x.ndim - 2, x.ndim - 1], "Only slicing along N or K is supported."
    assert length % 16 == 0, "Slicing must be a multiple of 16."
    orig_shape = x.shape
    N = x.shape[-2]
    K = x.shape[-1]
    # Tile shape is based on the activation dtype.
    assert dtype in ("fp8", "bf16"), "Only fp8 and bf16 activations supported."
    # Handle slice along M
    if dim == x.ndim - 2:
        tile_shape = 8 if dtype == "fp8" else 16
        block_size = N // length
        # View the shape in terms of shuffled tiles then permute to allow slicing.
        x_s = x.view(-1, tile_shape, block_size, length // tile_shape, K)
        x_s = x_s.permute(0, 2, 1, 3, 4).contiguous().view(-1, N, K)
        out_slice = x_s.narrow(1, start, length)
        # Reshape back to original shape.
        return out_slice.view(*orig_shape[:-2], length, K)
    # Handle slice along K
    else:
        outer_dim = x.view(-1, N, K).shape[0]
        x_s = x.view(outer_dim, -1, length // 2)
        row_factor = x_s.shape[1] * (length // 2) // K
        # Take slices of rows corresponding to column slice.
        return x_s.narrow(1, start * 2 * K // length, row_factor).view(
            *orig_shape[:-2], N, length // 2
        )


def scale_nvfp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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


def ck_preshuffle(src: torch.Tensor, NXdl: int = 16) -> torch.Tensor:
    """
    Applies shuffling to make weights more efficient for use with CK kernels.
    Args:
        src (torch.Tensor): Input tensor with dtype float8_e4m3fnuz.
        NXdl (int): Wave tile size along N.
    Returns:
        torch.Tensor: The shuffled tensor.
    """
    # Check input datatype
    if src.dtype != torch.float8_e4m3fnuz:
        raise TypeError("Input must be type float8_e4m3fnuz.")
    N, K = src.shape
    KPack = 16
    NLane = NXdl
    KLane = 64 // NLane
    K0 = K // (KLane * KPack)
    # Reshape src to enable the required permutation
    # Original shape: (N, K)
    # Desired intermediate shape for permutation: (N0, NLane, K0, KLane, KPack)
    src = src.reshape(N // NLane, NLane, K0, KLane, KPack)
    # Apply permutation: (N0, NLane, K0, KLane, KPack) -> (N0, K0, KLane, NLane, KPack)
    dst = src.permute(0, 2, 3, 1, 4).contiguous()
    # Reshape to original input shape.
    dst = dst.reshape(N, K)
    return dst
