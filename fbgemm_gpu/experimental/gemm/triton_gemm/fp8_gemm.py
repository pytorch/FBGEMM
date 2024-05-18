# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import logging
from typing import List, Optional, Tuple

import torch
import triton  # @manual

import triton.language as tl  # @manual
from torch._tensor import Tensor

from triton import Config  # @manual
from triton.ops.matmul_perf_model import (  # @manual
    early_config_prune,
    estimate_matmul_time,
)
from triton.runtime.jit import reinterpret as tl_reinterpret, TensorWrapper  # @manual

MAX_FP8 = 448.0

logger: logging.Logger = logging.getLogger(__name__)


def convert_fp8_type(tensor) -> triton.TensorWrapper:
    """
    Converts tensor to triton fp8 type.

    Args:
        tensor (torch.Tensor): input tensor.

    Returns:
        triton.TensorWrapper: fp8 tensor.
    """
    return tl_reinterpret(tensor, dtype=tl.float8e4nv)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound() -> List[Config]:
    """
    Returns a list of configs for matmul that are IO bound.

    Returns:
        List[Config]: list of configs.
    """
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "SPLIT_K": 1,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k
                    for split_k in []:  # Disabled [2, 4, 8, 16]:
                        configs.append(
                            Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "SPLIT_K": split_k,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                                pre_hook=init_to_zero("C"),
                            )
                        )
    return configs


MATMUL_CONFIGS: List[Config] = [
    # basic configs for compute-bound matmuls
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=5,
        num_warps=2,
    ),
    # good for int8
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
        num_stages=5,
        num_warps=2,
    ),
] + get_configs_io_bound()


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row(
    A,
    B,
    C,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        m_key (int): Autotuning key for M dimension of input tensor.
        n_key (int): Autotuning key for N dimension of input tensor.
        k_key (int): Autotuning key for K dimension of input tensor.
        A_scale (TensorWrapper): [M] scale tensor per row. A / A_scale = original A
        B_scale (TensorWrapper): [N] scale tensor per row. B / B_scale = original B
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Wether to cast A and B to C.dtype before tensor core.
    """
    # Matrix multiplication.
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # Re-order program ID for better L2 performance (swizzle).
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # Do matrix multiplication.
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # Pointers.
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Invert scaling.
    a_scale = tl.load(A_scale + rm, mask=rm < M)
    b_scale = tl.load(B_scale + rn, mask=rn < N)
    # Invert vector, then multiply on matrix for speed.
    inv_a_scale = 1.0 / a_scale
    inv_b_scale = 1.0 / b_scale
    # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
    scale = inv_a_scale[:, None] * inv_b_scale[None, :]
    acc *= scale

    acc = acc.to(C.dtype.element_ty)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # Handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def matmul_fp8_row(
    a: TensorWrapper,
    b: TensorWrapper,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
) -> torch.Tensor:
    """
    Performs matmul on [M, K] and [N, K] fp8 matrices with row-wise scalings [M], [N].

    Args:
        a (TensorWrapper): [M, K] input tensor.
        b (TensorWrapper): [N, K] input tensor.
        a_scale (torch.Tensor): [M] scale tensor per row. A / a_scale = original A
        b_scale (torch.Tensor): [N] scale tensor per row. B / b_scale = original B
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.

    Returns:
        torch.Tensor: [M, N] Output tensor a @ b / (a_scale[:, None] * b_scale[None, :])
    """
    M, N, K, m_key, n_key, k_key, c, dot_out_dtype_triton, device = prep_matmul(
        a, b, dot_out_dtype
    )
    # launch kernel
    if a.device == torch.device("cpu"):
        logger.info(
            "FP8 Row-wise Triton kernel not supported on cpu, fallback to torch"
        )
        return (
            torch.matmul(a.base.to(torch.bfloat16), b.base.to(torch.bfloat16).T)
            / (a_scale[:, None] * b_scale[None, :])
        ).to(dtype=c.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )

    _kernel_matmul_fp8_row[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        m_key,
        n_key,
        k_key,
        a_scale,
        b_scale,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        dot_out_dtype=dot_out_dtype_triton,
        allow_tf32=allow_tf32,
        fp8_fast_accum=fp8_fast_accum,
        GROUP_M=8,
        AB_DTYPE=False,
    )
    return c


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],  # TODO caller side bin keys so similar shapes can use same triton.autotune.
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_block(
    A,
    B,
    C,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    scale_block_m: tl.constexpr,
    scale_block_n: tl.constexpr,
    scale_block_k: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_scale_am,
    stride_scale_ak,
    stride_scale_bn,
    stride_scale_bk,
    dot_out_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with block-wise scales

    Performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles and
    A and B scaled by a scaling factor per [scale_block_m, scale_block_k] and
    [scale_block_n, scale_block_k] tiles
    respectively.

    Todo:
        * Support scale_block_{mnk} < BLOCK{MNK} for each dim.
    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        m_key (int): Autotuning key for M dimension of input tensor.
        n_key (int): Autotuning key for N dimension of input tensor.
        k_key (int): Autotuning key for K dimension of input tensor.
        A_scale (TensorWrapper): [cdiv(M, scale_block_m), cdiv(K, scale_block_k)] scale tensor per block. A / A_scale = original A
        B_scale (TensorWrapper): [cdiv(N, scale_block_n), cdiv(K, scale_block_k)] scale tensor per block. B / B_scale = original B
        scale_block_m (int): Block size for M dimension of A_scale.
        scale_block_n (int): Block size for N dimension of B_scale.
        scale_block_k (int): Block size for K dimension of A_scale and B_scale.
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        stride_scale_am (int): Stride of M dimension of A_scale.
        stride_scale_ak (int): Stride of K dimension of A_scale.
        stride_scale_bn (int): Stride of N dimension of B_scale.
        stride_scale_bk (int): Stride of K dimension of B_scale.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Wether to cast A and B to C.dtype before tensor core.
    """
    assert BLOCK_M < scale_block_m
    assert BLOCK_N < scale_block_n
    assert BLOCK_K < scale_block_k
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    scale_m = pid_m * BLOCK_M // scale_block_m
    scale_n = pid_n * BLOCK_N // scale_block_n
    _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
    scale_next = 0.0
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)

            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        # Note: Due to split_k access "pid_k" = k * SPLIT_K + pid_z
        # Access a_scale[pid_m, k * SPLIT_K + pid_z]
        # and b_scale[k * SPLIT_K + pid_z, pid_n]
        pid_k = k * SPLIT_K + pid_z
        pid_k_next = (k + 1) * SPLIT_K + pid_z

        # Some math to precompute on scalars, and apply once on matrix.
        # a + c/s = (as + c) / s
        # (((a_i-1 * s_i-1 + c_i-1) / s_i-1) * s_i + c_i) / s_i ... ) * s_k + c_k) * 1.0 / s_k
        # Simplifies to (a_i-1 + c) * (s_i+1/s_i)
        # And have s_k+1 be 1.
        # Scale_i = pid_i * BLOCK_I / scale_block_i
        scale_k = pid_k * BLOCK_K // scale_block_k

        a_scale = tl.load(A_scale + scale_m * stride_scale_am + scale_k)
        b_scale = tl.load(B_scale + scale_n * stride_scale_bn + scale_k)
        scale = a_scale * b_scale

        # Normalize last scale with 1.
        if k + 1 == tl.cdiv(K, BLOCK_K * SPLIT_K):
            scale_next = 1.0
        else:
            scale_k_next = pid_k_next * BLOCK_K // scale_block_k
            a_scale_next = tl.load(
                A_scale + scale_m * stride_scale_am + scale_k_next * stride_scale_bk
            )
            b_scale_next = tl.load(
                B_scale + scale_n * stride_scale_bn + scale_k_next * stride_scale_bk
            )
            scale_next = a_scale_next * b_scale_next
        inv_scale = 1.0 / scale
        scale_next_inv_scale = scale_next / scale

        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

            acc *= scale_next_inv_scale
        else:
            acc += (
                tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=allow_tf32) * inv_scale
            )
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = acc.to(C.dtype.element_ty)
    c = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(c, acc, mask=mask)
    else:
        tl.atomic_add(c, acc, mask=mask)


def matmul_fp8_block(
    a: TensorWrapper,
    b: TensorWrapper,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_block_m: int = 256,
    scale_block_n: int = 256,
    scale_block_k: int = 256,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
) -> Tensor:
    """Performs matmul on [M, K] and [N, K] fp8 matrices with block-wise scalings.

    Args:
        a (TensorWrapper): [M, K] input tensor.
        b (TensorWrapper): [N, K] input tensor.
        a_scale (torch.Tensor): [cdiv(M, scale_block_m), cdiv(K, scale_block_k)] scale tensor per scale block. A / A_scale = original A
        b_scale (torch.Tensor): [cdiv(N, scale_block_n), cdiv(K, scale_block_k)] scale tensor per scale block. B / B_scale = original B
        scale_block_m (int): Block size for M dimension of A_scale.
        scale_block_n (int): Block size for N dimension of B_scale.
        scale_block_k (int): Block size for K dimension of A_scale and B_scale.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.

    Returns:
        Tensor: [M, N] output tensor, (a / a_scale) @ (b / b_scale)
    """
    M, N, K, m_key, n_key, k_key, c, dot_out_dtype_triton, device = prep_matmul(
        a, b, dot_out_dtype
    )

    # launch kernel
    assert device != torch.device(
        "cpu"
    ), "Blockwise matmul not supported on cpu, please use row-wise instead."

    # noqa: E731:
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )

    _kernel_matmul_fp8_block[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        m_key,
        n_key,
        k_key,
        a_scale,
        b_scale,
        scale_block_m,
        scale_block_n,
        scale_block_k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        dot_out_dtype=dot_out_dtype_triton,
        allow_tf32=allow_tf32,
        fp8_fast_accum=fp8_fast_accum,
        GROUP_M=8,
        AB_DTYPE=False,
    )
    return c


def get_matmul_tune(M: int, N: int, K: int) -> Tuple[int, int, int]:
    """
    Generate a simplified matmul tune key for A @ B.T
    with [M, K] A and [N, K] B to reduce excessive autotuning.

    Args:
        M (int): Number of rows in A.
        N (int): Number of rows in B.
        K (int): Number of cols in A and cols in B.

    Returns:
        m_key (int): Autotuning key for M dim.
        n_key (int): Autotuning key for N dim.
        k_key (int): Autotuning key for K dim.

    TODO: Refine this. For now it's useful for LLM inference where N, K dims are fixed
          and M dim varies due to seq_len.
    """
    if M < 256:
        m_key = M
    else:
        m_key = 256 + M // 1024
    return m_key, N, K


def prep_matmul(
    a: TensorWrapper, b: TensorWrapper, dot_out_dtype: Optional[torch.dtype]
) -> Tuple[int, int, int, int, int, int, torch.Tensor, str, torch.device]:
    """
    Shared bookkeeping for a @ b.T matmul.

    Args:
        a (TensorWrapper): [M, K] input tensor.
        b (TensorWrapper): [N, K] input tensor.
        dot_out_dtype (tl.dtype): Output type of tensor core.

    Returns:
        M (int): Number of rows in A.
        N (int): Number of rows in B.
        K (int): Number of cols in A and cols in B.
        m_key (int): Autotuning key for M dim.
        n_key (int): Autotuning key for N dim.
        k_key (int): Autotuning key for K dim.
        c (Tensor): [M, N] output tensor.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        device (torch.device): Device of output tensor.
    """
    device = a.device

    # checks constraints
    assert (
        a.shape[1] == b.shape[1]
    ), f"incompatible dimensions, a: {a.shape}, b: {b.shape}"
    M, K = a.shape
    N, _ = b.shape
    m_key, n_key, k_key = get_matmul_tune(M, N, K)

    # allocates output
    assert a.dtype in [tl.float8e4nv, tl.float8e4b15, tl.float8e5] and b.dtype in [
        tl.float8e4nv,
        tl.float8e4b15,
        tl.float8e5,
    ]
    c_dtype = torch.bfloat16

    c = torch.empty((M, N), device=device, dtype=c_dtype)
    if dot_out_dtype is None:
        dot_out_dtype_triton = tl.float32
    else:
        assert isinstance(
            dot_out_dtype, torch.dtype
        ), f"dot_out_dtype type {type(dot_out_dtype)} must be a torch.dtype"
        if dot_out_dtype == torch.bfloat16:
            dot_out_dtype_triton = tl.bfloat16
        elif dot_out_dtype == torch.float32:
            dot_out_dtype_triton = tl.float32
        else:
            dot_out_dtype_triton = tl.int32

    return M, N, K, m_key, n_key, k_key, c, dot_out_dtype_triton, device


@triton.autotune(
    configs=[
        Config({"BLOCK_SIZE": 512}),
        Config({"BLOCK_SIZE": 1024}),
        Config({"BLOCK_SIZE": 2048}),
        Config({"BLOCK_SIZE": 4096}),
        Config({"BLOCK_SIZE": 8192}),
    ],
    key=["N"],
)
@triton.jit
def _kernel_quantize_fp8_row(
    A,
    A_scale,
    A_fp8,
    M,
    N,
    stride_am,
    stride_an,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Quantize and scale each row.

    Scale per row i is computed as MAX_FP8 / max(abs(A[i, :]))

    Kernel naively iterates through  matrix with [1, BLOCK_SIZE] tiles
    in a max pass then scale/quantize pass.

    Todo:
        * Better tiling schemes.

    Args:
        A (Tensor): [m, n] higher precision input tensor.
        A_scale (Tensor): [m] scale tensor per row.
        A_fp8 (Tensor): [m, n] fp8 scaled tensor. A_fp8 = A * a_scale
        M (int): Number of rows.
        N (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        BLOCK_SIZE (int): Block size for reduction.
    """
    pid = tl.program_id(0)
    n_offset = tl.arange(0, BLOCK_SIZE)

    # Calculate max.
    cur_max = 0.0
    for _k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        a = tl.load(
            A + pid * stride_am + n_offset * stride_an, mask=n_offset < N, other=0.0
        )
        tile_max = tl.max(tl.abs(a))
        cur_max = tl.maximum(tile_max, cur_max)

        n_offset += BLOCK_SIZE

    # Scale and quantize.
    a_scale = MAX_FP8 / cur_max
    tl.store(A_scale + pid, a_scale)
    n_offset = tl.arange(0, BLOCK_SIZE)
    for _k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        a = tl.load(
            A + pid * stride_am + n_offset * stride_an, mask=n_offset < N, other=0.0
        )
        a_fp8 = a * a_scale
        a_fp8.to(tl.float8e4nv)
        tl.store(
            A_fp8 + pid * stride_am + n_offset * stride_an, a_fp8, mask=n_offset < N
        )
        n_offset += BLOCK_SIZE


def triton_quantize_fp8_row(a: Tensor) -> Tuple[TensorWrapper, torch.Tensor]:
    """
    Call the triton quantize fp8 row kernel to quantize a tensor to fp8 with row-wise scalings.

    Args:
        a (Tensor): [m, n] higher precision input tensor.

    Returns:
        TensorWrapper: fp8 scaled tensor.
        torch.Tensor: scale tensor per row.
    """
    num_rows = a.shape[0]
    a_scale = torch.empty((num_rows), dtype=torch.float32, device=a.device)
    a_fp8 = torch.empty(
        (a.shape[0], a.shape[1]), device=a.device, dtype=torch.float8_e4m3fn
    )

    a_fp8 = convert_fp8_type(a_fp8)
    grid = (num_rows,)
    _kernel_quantize_fp8_row[grid](
        a, a_scale, a_fp8, a.shape[0], a.shape[1], a.stride(0), a.stride(1)
    )

    return a_fp8, a_scale


def quantize_fp8_row(
    a: Tensor,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> Tuple[TensorWrapper, torch.Tensor]:
    """
    Quantize a to fp8 with row-wise scalings and optionally move to output device.

    Args:
        a (Tensor): Input high precision tensor.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.

    Returns:
        TensorWrapper: fp8 scaled tensor.
        torch.Tensor: scale tensor per row.
    """
    if a.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        return triton_quantize_fp8_row(a)
    # else use pytorch implementation.
    if not output_device:
        output_device = a.device

    row_max: torch.Tensor = torch.max(torch.abs(a), dim=1)[0]
    a_scale = torch.empty((a.shape[0]), dtype=torch.float32, device=output_device)
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    a_scale = max_fp8 / row_max.to(torch.float32)  # pyre-ignore
    a_scale[a_scale == float("inf")] = 1.0  # pyre-ignore
    a_fp8 = a * a_scale[:, None]  # pyre-ignore
    # Cast and move data to output device (for cpu weight loading).
    a_fp8 = convert_fp8_type(a_fp8.to(device=output_device, dtype=torch.float8_e4m3fn))
    a_scale = a_scale.to(output_device)  # pyre-ignore
    del a
    return a_fp8, a_scale


@triton.jit
def _kernel_quantize_fp8_block(
    A,
    A_scale,
    A_fp8,
    M,
    K,
    stride_am,
    stride_ak,
    stride_a_scale_m,
    stride_a_scale_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Quantize and scale each [BLOCK_M, BLOCK_K] block.

    Scale per block i, j is computed as MAX_FP8 / max(abs(A[i:i+BLOCK_M, j:j+BLOCK_K]))

    Kernel naively iterates through  matrix with [BLOCK_M, BLOCK_K] tiles.

    Todo:
        * Better tiling and ordering schemes.

    Args:
        A (Tensor): [M, K] higher precision input tensor.
        A_scale (Tensor): [cdiv(M, BLOCK_M), cdiv(K, BLOCK_K)] scale tensor per block.
        A_fp8 (Tensor): [M, K] fp8 scaled tensor. A_fp8 = A * a_scale
        M (int): Number of rows.
        K (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_a_scale_m (int): Stride of m dimension of A_scale.
        stride_a_scale_k (int): Stride of k dimension of A_scale.
        BLOCK_M (int): Block size for M dimension of A_scale and kernel.
        BLOCK_K (int): Block size for K dimension of A_scale and kernel.
    """
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BLOCK_K)
    block_m = pid // grid_k
    block_k = pid % grid_k
    rm = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_offset = rm[:, None] * stride_am + rk[None, :] * stride_ak
    a_mask = (rm < M)[:, None] & (rk < K)[None, :]
    a_block = tl.load(A + a_offset, mask=a_mask, other=0.0)

    scale = MAX_FP8 / tl.max(tl.abs(a_block))

    tl.store(A_scale + block_m * stride_a_scale_m + block_k * stride_a_scale_k, scale)
    a_fp8 = a_block * scale
    a_fp8.to(tl.float8e4nv)
    tl.store(A_fp8 + a_offset, a_fp8, mask=a_mask)


def quantize_fp8_block(
    x: torch.Tensor, block_m: int = 256, block_k: int = 256
) -> Tuple[TensorWrapper, torch.Tensor]:
    """
    Quantize a tensor to fp8 with block-wise scalings.

    Scale per block i, j is computed as MAX_FP8 / max(abs(x[i:i+block_m, j:j+block_k]))

    Args:
        x (Tensor): [M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        block_k (int): Block size for K dimension of scale.

    Returns:
        TensorWrapper: [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] scale tensor per block.
    """
    assert x.device != torch.device(
        "cpu"
    ), "Blockwise quantization not support on cpu, please use row-wise quantization instead."
    M, K = x.shape
    grid_m = triton.cdiv(M, block_m)
    grid_k = triton.cdiv(K, block_k)
    x_scale = torch.ones((grid_m, grid_k), device=x.device, dtype=torch.float32)
    x_fp8 = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
    x_fp8 = convert_fp8_type(x_fp8)

    _kernel_quantize_fp8_block[(grid_m * grid_k,)](
        x,
        x_scale,
        x_fp8,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        BLOCK_M=triton.cdiv(block_m, 1),  # Hack to appease pyre.
        BLOCK_K=triton.cdiv(block_k, 1),
    )

    return x_fp8, x_scale
