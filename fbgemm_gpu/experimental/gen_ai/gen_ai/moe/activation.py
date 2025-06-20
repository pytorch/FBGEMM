# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import get_fp8_constants


# Function APIs
def silu_mul(
    x0: torch.Tensor,
    x1: torch.Tensor,
    valid_token_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused silu and mul operations.

    y = x0 * sigmoid(x0) * x1

    Args:
        x0: input tensor of shape (T, D)
        x1: input tensor of shape (T, D)
        valid_token_count: tensor of shape (1,) to indicate the number of valid tokens.

    Returns:
        output tensor of shape (T, D)
    """

    assert x0.ndim == 2 and x0.stride(1) == 1
    assert x1.ndim == 2 and x1.stride(1) == 1
    assert x0.shape == x1.shape
    assert x0.dtype == x1.dtype

    T, D = x0.shape
    stride_0 = x0.stride(0)
    stride_1 = x1.stride(0)

    out = torch.empty((T, D), device="cuda", dtype=x0.dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    if T >= NUM_SMS:
        BLOCK_D_OUTER = D
        BLOCK_D_INNER = 1024
        assert D % BLOCK_D_INNER == 0
    else:
        BLOCK_D_OUTER = 512
        BLOCK_D_INNER = 256
        assert D % BLOCK_D_OUTER == 0
    grid = (T, D // BLOCK_D_OUTER)
    _fbgemm_silu_mul[grid](
        out,
        x0,
        x1,
        stride_0,
        stride_1,
        valid_token_count,
        D,  # pyre-ignore
        BLOCK_D_OUTER,  # pyre-ignore
        BLOCK_D_INNER,  # pyre-ignore
    )
    return out


def silu_mul_quant(
    x0: torch.Tensor,
    x1: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    valid_token_count: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused silu, mul, and FP8 rowwise quantization operations.

    y, y_scale = quantize(x0 * sigmoid(x0) * x1)

    Args:
        x0: input tensor of shape (T, D)
        x1: input tensor of shape (T, D)
        scale_ub: tensor of shape (1,) to indicate the upper bound of the scale.
        valid_token_count: tensor of shape (1,) to indicate the number of valid tokens.

    Returns:
        output quantized tensor of shape (T, D) and its inverse scale of shape (T,)
    """

    assert x0.ndim == 2 and x0.stride(1) == 1
    assert x1.ndim == 2 and x1.stride(1) == 1
    assert x0.shape == x1.shape
    assert x0.dtype == x1.dtype

    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()

    T, D = x0.shape
    stride_0 = x0.stride(0)
    stride_1 = x1.stride(0)

    out = torch.empty((T, D), device="cuda", dtype=pt_dtype)
    out_inv_scale = torch.empty((T,), device="cuda", dtype=torch.float32)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    BLOCK_T = triton.cdiv(T, NUM_SMS)
    NUM_CTAS = triton.cdiv(T, BLOCK_T)

    grid = (NUM_CTAS,)
    _fbgemm_silu_mul_quant[grid](
        out,
        out_inv_scale,
        x0,
        x1,
        scale_ub,
        stride_0,
        stride_1,
        valid_token_count,
        T,
        D,  # pyre-ignore
        BLOCK_T,
        TL_FP8_DTYPE=tl_dtype,  # pyre-ignore
        MAX_FP8=max_fp8,  # pyre-ignore
        EPS=eps,  # pyre-ignore
        CLAMP_MAX=scale_ub is not None,  # pyre-ignore
    )
    return out, out_inv_scale


# Torch Custom Op Registrations
_SILU_MUL_OP_NAME = "fbgemm::silu_mul"

torch.library.define(
    "fbgemm::silu_mul",
    "(Tensor x0, Tensor x1, Tensor? valid_token_count=None) -> Tensor",
)


@torch.library.impl(_SILU_MUL_OP_NAME, "Meta")
def silu_mul_meta(x0, x1, valid_token_count):
    return x0.new_empty(x0.shape)


@torch.library.impl(_SILU_MUL_OP_NAME, "CUDA")
def silu_mul_cuda(x0, x1, valid_token_count):
    return silu_mul(x0, x1, valid_token_count)


_SILU_MUL_OP_QUANT_NAME = "fbgemm::silu_mul_quant"

torch.library.define(
    "fbgemm::silu_mul_quant",
    "(Tensor x0, Tensor x1, Tensor? scale_ub=None, Tensor? valid_token_count=None) -> (Tensor, Tensor)",
)


@torch.library.impl(_SILU_MUL_OP_QUANT_NAME, "Meta")
def silu_mul_quant_meta(x0, x1, scale_ub, valid_token_count):
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    return torch.empty(x0.shape, device=x0.device, dtype=pt_dtype)


@torch.library.impl(_SILU_MUL_OP_QUANT_NAME, "CUDA")
def silu_mul_quant_cuda(x0, x1, scale_ub=None, valid_token_count=None):
    return silu_mul_quant(x0, x1, scale_ub, valid_token_count)


# Kernel Implementations
@triton.jit
def _fbgemm_silu_mul(
    y_ptr,
    x0_ptr,
    x1_ptr,
    stride_0,
    stride_1,
    valid_token_count,
    D: tl.constexpr,
    BLOCK_D_OUTER: tl.constexpr,
    BLOCK_D_INNER: tl.constexpr,
) -> None:
    token_index = tl.program_id(0)
    feature_offset = tl.program_id(1) * BLOCK_D_OUTER + tl.arange(0, BLOCK_D_INNER)[:]

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if token_index >= valid_token_count:
            return

    for _ in tl.range(0, BLOCK_D_OUTER // BLOCK_D_INNER, num_stages=3):
        x0 = tl.load(
            x0_ptr + token_index * stride_0 + feature_offset,
            None,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x1 = tl.load(
            x1_ptr + token_index * stride_1 + feature_offset,
            None,
            eviction_policy="evict_first",
        ).to(tl.float32)

        y = x0 * tl.sigmoid(x0) * x1

        tl.store(
            y_ptr + token_index * D + feature_offset,
            y,
            None,
        )
        feature_offset += BLOCK_D_INNER


@triton.jit
def _fbgemm_silu_mul_quant(
    y_ptr,
    y_inv_scale_ptr,
    x0_ptr,
    x1_ptr,
    scale_ub_ptr,
    stride_0,
    stride_1,
    valid_token_count,
    T,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
) -> None:
    PADDED_D: tl.constexpr = triton.next_power_of_2(D)  # pyre-ignore

    tidx = tl.program_id(0)
    start_idx = tidx * BLOCK_T
    end_idx = tl.minimum(start_idx + BLOCK_T, T)

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if start_idx >= valid_token_count:
            return

    offsets = tl.arange(0, PADDED_D)[:]
    mask = offsets < D

    if CLAMP_MAX:
        ub = tl.load(scale_ub_ptr, eviction_policy="evict_last")
    else:
        ub = float("inf")

    for token_index in tl.range(start_idx, end_idx, 1, num_stages=2):
        x0 = tl.load(
            x0_ptr + token_index * stride_0 + offsets,
            mask,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x1 = tl.load(
            x1_ptr + token_index * stride_1 + offsets,
            mask,
            eviction_policy="evict_first",
        ).to(tl.float32)

        y = x0 * tl.sigmoid(x0) * x1

        # Masked values are set to 0.0.
        row_max = tl.max(tl.where(mask, tl.abs(y), 0.0))
        if CLAMP_MAX:
            row_max = tl.clamp(row_max, EPS, ub)
        else:
            row_max = tl.maximum(row_max, EPS)

        y_scale = MAX_FP8 / row_max
        tl.store(y_inv_scale_ptr + token_index, 1.0 / y_scale)

        y = y * y_scale
        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        y_fp8 = tl.clamp(y, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)

        tl.store(
            y_ptr + token_index * D + offsets,
            y_fp8,
            mask,
        )
