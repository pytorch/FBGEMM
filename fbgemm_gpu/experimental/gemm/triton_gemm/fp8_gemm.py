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

logger: logging.Logger = logging.getLogger(__name__)


def get_fp8_constants() -> Tuple[torch.dtype, tl.dtype, float, float]:
    """
    Helper function to get constant values for the current platform.

    Returns:
        pt_dtype (torch.dtype): The correct torch fp8 datatype.
        tl_dtype (tl.dtype): The correct triton fp8 datatype.
        max_fp8 (float): The maximum reprsentable value for the fp8 datatype.
        eps (float): Minimum clip value to prevent divide by zero.
    """
    if torch.version.hip is not None:
        pt_fp8_dtype = torch.float8_e4m3fnuz
        tl_fp8_dtype = tl.float8e4b8
    else:
        pt_fp8_dtype = torch.float8_e4m3fn
        tl_fp8_dtype = tl.float8e4nv
    return pt_fp8_dtype, tl_fp8_dtype, torch.finfo(pt_fp8_dtype).max, 1e-12


def convert_fp8_type(tensor, dtype) -> triton.TensorWrapper:
    """
    Converts tensor to triton fp8 type.

    Args:
        tensor (torch.Tensor): input tensor.
        dtype (tl.dtype): target triton dtype.

    Returns:
        triton.TensorWrapper: fp8 tensor.
    """
    return tl_reinterpret(tensor, dtype=dtype)


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
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
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
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    Bias,
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
    USE_BIAS: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    NUM_SMS: tl.constexpr,
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
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A.
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B.
        Bias (tensorWrapper): [N] Optional bias tensor.
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
        USE_BIAS (bool): Whether to use bias.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Wether to cast A and B to C.dtype before tensor core.
    """
    # Matrix multiplication.
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_K)

    num_pid_in_group = GROUP_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_M)
    offs_bn = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_M
            start_n = pid_n * BLOCK_N
            offs_am = start_m + tl.arange(0, BLOCK_M)
            offs_bn = start_n + tl.arange(0, BLOCK_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        A = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(A, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_K, other=0.0)
        b = tl.load(B, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Invert scaling.
            a_scale = tl.load(A_scale + rm, mask=rm < M)
            b_scale = tl.load(B_scale + rn, mask=rn < N)
            # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
            scale = a_scale[:, None] * b_scale[None, :]
            acc *= scale

            # Load and add bias if specified.
            if USE_BIAS:
                bias = tl.load(Bias + rn, mask=rn < N)
                acc += bias[None, :]

            acc = acc.to(C_ptr.dtype.element_ty)
            C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            # Handles write-back with reduction-splitting
            tl.store(C, acc, mask=mask)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)


@triton.autotune(
    configs=MATMUL_CONFIGS
    + [
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row_no_fast_acc(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    Bias,
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
    USE_BIAS: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    NUM_SMS: tl.constexpr,
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
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
        Bias (TensorWrapper): [N] Optional bias tensor.
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
        USE_BIAS(bool): Whether to use bias.
        AB_DTYPE (bool): Wether to cast A and B to C.dtype before tensor core.
    """
    # Matrix multiplication.

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_K)

    num_pid_in_group = GROUP_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_M)
    offs_bn = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_M
            start_n = pid_n * BLOCK_N
            offs_am = start_m + tl.arange(0, BLOCK_M)
            offs_bn = start_n + tl.arange(0, BLOCK_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        A = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(A, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_K, other=0.0)
        b = tl.load(B, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_K, other=0.0)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Invert scaling.
            a_scale = tl.load(A_scale + rm, mask=rm < M)
            b_scale = tl.load(B_scale + rn, mask=rn < N)
            # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
            scale = a_scale[:, None] * b_scale[None, :]
            acc *= scale

            # Load and add bias if specified.
            if USE_BIAS:
                bias = tl.load(Bias + rn, mask=rn < N)
                acc += bias[None, :]

            acc = acc.to(C_ptr.dtype.element_ty)
            C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            # Handles write-back with reduction-splitting
            tl.store(C, acc, mask=mask)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row_imprecise_acc(
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
    Bias,
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
    USE_BIAS: tl.constexpr,
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
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
        Bias (TensorWrapper): [N] Optional bias tensor.
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
        USE_BIAS (bool): Whether to use bias.
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
            acc = tl.dot(
                a,
                b,
                acc,
                max_num_imprecise_acc=32,
                out_dtype=dot_out_dtype,
                allow_tf32=allow_tf32,
            )
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
    # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
    scale = a_scale[:, None] * b_scale[None, :]
    acc *= scale

    # Apply bias.
    if USE_BIAS:
        bias = tl.load(Bias + rn, mask=rn < N)
        acc += bias[None, :]

    acc = acc.to(C.dtype.element_ty)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # Handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


@triton.autotune(
    configs=[
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
    ],
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row_tma_persistent(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    Bias,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    c_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_BIAS: tl.constexpr,
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
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
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
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_M * num_pid_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    dtype_fp8 = tl.float8e4nv
    scale_dtype = tl.float32
    bias_dtype = tl.float32

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N
            offs_am = tl.multiple_of(offs_am, BLOCK_M)
            offs_bn = tl.multiple_of(offs_bn, BLOCK_N)

        offs_k = ki * BLOCK_K

        a = tl._experimental_descriptor_load(
            A_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], dtype_fp8
        )
        b = tl._experimental_descriptor_load(
            B_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K], dtype_fp8
        )

        if fp8_fast_accum:
            acc = tl.dot(a, b.T, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(a, b.T, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers

            # # Invert scaling.
            a_scale = tl._experimental_descriptor_load(
                A_scale, [offs_am], [BLOCK_M], scale_dtype
            )
            b_scale = tl._experimental_descriptor_load(
                B_scale, [offs_bn], [BLOCK_N], scale_dtype
            )
            # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
            scale = a_scale[:, None] * b_scale[None, :]
            acc *= scale

            # Load and add bias if specified.
            if USE_BIAS:
                bias = tl._experimental_descriptor_load(
                    Bias, [offs_bn], [BLOCK_N], bias_dtype
                )
                acc += bias[None, :]

            acc = acc.to(c_dtype)
            tl._experimental_descriptor_store(C_ptr, acc, [offs_am, offs_bn])
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)


# check if we have the TMA version in Triton PR #4498 (https://github.com/triton-lang/triton/pull/4498).
HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

if HAS_TMA_DESC:
    print(
        "TMA benchmarks will be running with experimental grid constant TMA descriptor."
    )
else:
    print("TMA benchmarks will be running without grid constant TMA descriptor.")


class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        )
        self.fill_2d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        )
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
            )
        else:
            self.cuda_descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8
            )

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


@torch.library.custom_op("triton::matmul_fp8_row", mutates_args=())
def matmul_fp8_row(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
    imprecise_acc: bool = False,
    tma_persistent: bool = True,
) -> torch.Tensor:
    """
    Performs matmul on [M, K] and [N, K] fp8 matrices with row-wise scalings [M], [N].

    Args:
        a (torch.Tensor): [M, K] input tensor.
        b (torch.Tensor): [N, K] input tensor.
        a_scale (torch.Tensor): [M] reciprocal scale tensor per row. A * a_scale = original A
        b_scale (torch.Tensor): [N] reciprocal scale tensor per row. B * b_scale = original B
        bias (torch.Tensor): [N] optional bias tensor to add to output if provided.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        tma_persistent (bool): Whether to use TMA persistent kernel impl.

    Returns:
        torch.Tensor: [M, N] Output tensor a @ b / (a_scale[:, None] * b_scale[None, :])
    """
    # Get datatypes and constants to use.
    _, tl_dtype, _, _ = get_fp8_constants()
    # Handle 3D+ a shape
    a_shape = a.shape
    a = a.view(-1, a.size(-1))
    # Reinterpret inputs into proper triton fp8 dtype.
    a_tl = convert_fp8_type(a, tl_dtype)
    b_tl = convert_fp8_type(b, tl_dtype)
    M, N, K, m_key, n_key, k_key, c, c_dtype_triton, dot_out_dtype_triton, device = (
        prep_matmul(a_tl, b_tl, dot_out_dtype)
    )

    output_shape = a_shape[:-1] + (N,)
    # launch kernel
    if a.device == torch.device("cpu"):
        logger.info(
            "FP8 Row-wise Triton kernel not supported on cpu, fallback to torch"
        )
        output = torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16).T) * (
            a_scale[:, None] * b_scale[None, :]
        )
        if bias is not None:
            output += bias[None, :]
        return output.to(c.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def persistent_grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            ),
        )

    if tma_persistent:
        # used by TMA persistent kernel
        desc_helper = TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("a")
        desc_helper.init_tma_descriptor("b")
        desc_helper.init_tma_descriptor("c")
        desc_helper.init_tma_descriptor("a_scale")
        desc_helper.init_tma_descriptor("b_scale")
        desc_helper.init_tma_descriptor("bias")

        def persistent_grid_tma(META):
            nonlocal desc_helper
            desc_helper.fill_2d_tma_descriptor(
                "a",
                a_tl.data_ptr(),
                M,
                K,
                META["BLOCK_M"],
                META["BLOCK_K"],
                a_tl.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "b",
                b_tl.data_ptr(),
                N,
                K,
                META["BLOCK_N"],
                META["BLOCK_K"],
                b_tl.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "c",
                c.data_ptr(),
                M,
                N,
                META["BLOCK_M"],
                META["BLOCK_N"],
                c.element_size(),
            )
            desc_helper.fill_1d_tma_descriptor(
                "a_scale",
                a_scale.data_ptr(),
                M,
                META["BLOCK_M"],
                a_scale.element_size(),
            )
            desc_helper.fill_1d_tma_descriptor(
                "b_scale",
                b_scale.data_ptr(),
                N,
                META["BLOCK_N"],
                b_scale.element_size(),
            )
            if bias is not None:
                desc_helper.fill_1d_tma_descriptor(
                    "bias",
                    bias.data_ptr(),
                    N,
                    META["BLOCK_N"],
                    bias.element_size(),
                )
            return (
                min(
                    NUM_SMS,
                    triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
                ),
            )

        desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
        desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
        desc_c = desc_helper.get_tma_descriptor_kernel_param("c")
        desc_a_scale = desc_helper.get_tma_descriptor_kernel_param("a_scale")
        desc_b_scale = desc_helper.get_tma_descriptor_kernel_param("b_scale")
        desc_bias = desc_helper.get_tma_descriptor_kernel_param("bias")

        # pyre-ignore[28]:
        _kernel_matmul_fp8_row_tma_persistent[persistent_grid_tma](
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            m_key,
            n_key,
            k_key,
            desc_a_scale,
            desc_b_scale,
            desc_bias,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype_triton,
            c_dtype=c_dtype_triton,
            allow_tf32=allow_tf32,
            fp8_fast_accum=fp8_fast_accum,
            GROUP_M=8,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
            USE_BIAS=bias is not None,
        )
    elif imprecise_acc:
        _kernel_matmul_fp8_row_imprecise_acc[grid](
            a_tl,
            b_tl,
            c,
            M,
            N,
            K,
            m_key,
            n_key,
            k_key,
            a_scale,
            b_scale,
            bias,
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
            USE_BIAS=bias is not None,
            AB_DTYPE=False,
        )
    elif fp8_fast_accum:
        _kernel_matmul_fp8_row[persistent_grid](
            a_tl,
            b_tl,
            c,
            M,
            N,
            K,
            m_key,
            n_key,
            k_key,
            a_scale,
            b_scale,
            bias,
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
            USE_BIAS=bias is not None,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
        )
    else:
        _kernel_matmul_fp8_row_no_fast_acc[persistent_grid](
            a_tl,
            b_tl,
            c,
            M,
            N,
            K,
            m_key,
            n_key,
            k_key,
            a_scale,
            b_scale,
            bias,
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
            USE_BIAS=bias is not None,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
        )
    return c.view(output_shape)


@matmul_fp8_row.register_fake
def matmul_fp8_row_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
    imprecise_acc: bool = False,
    tma_persistent: bool = False,
) -> torch.Tensor:
    """Shape function for torch compile."""
    M, K = a.shape
    N, K = b.shape
    return torch.empty((M, N), device=a.device, dtype=torch.bfloat16)


# pruned some unreasonable config
def prune_configs_block(configs, named_args, **kwargs):
    configs = early_config_prune(configs, named_args, **kwargs)
    scale_block_k = named_args["scale_block_k"]
    pruned_configs = []
    # Further rule out configs with scale_block_k is not a multiple of BLOCK_K
    for config in configs:
        kw = config.kwargs
        BLOCK_K = kw["BLOCK_K"]
        if scale_block_k % BLOCK_K != 0:
            continue
        pruned_configs.append(config)
    return pruned_configs


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],  # TODO caller side bin keys so similar shapes can use same triton.autotune.
    prune_configs_by={
        "early_config_prune": prune_configs_block,
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
def _kernel_matmul_fp8_block_fastacc(
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
        A_scale (TensorWrapper): [cdiv(M, scale_block_m), cdiv(K, scale_block_k)] reciprocal scale tensor per block. A * A_scale = original A
        B_scale (TensorWrapper): [cdiv(N, scale_block_n), cdiv(K, scale_block_k)] reciprocal scale tensor per block. B * B_scale = original B
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
    _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
    scale_m = pid_m * BLOCK_M // scale_block_m
    scale_n = pid_n * BLOCK_N // scale_block_n
    k_multiple = scale_block_k // BLOCK_K

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):

        k_remaining = K - k * (BLOCK_K * SPLIT_K)

        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)

        acc = tl.dot(a, b, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

        # Some math to precompute on scalars, and apply once on matrix.
        # a + c/s = (as + c) / s
        # (((a_i-1 * s_i-1 + c_i-1) / s_i-1) * s_i + c_i) / s_i ... ) * s_k + c_k) * 1.0 / s_k
        # Simplifies to (a_i-1 + c) * (s_i+1/s_i)
        # And have s_k+1 be 1.
        # Scale_i = pid_i * BLOCK_I / scale_block_i
        pid_k = k * SPLIT_K + pid_z
        if ((pid_k + 1) % k_multiple == 0) or (k_remaining < BLOCK_K * SPLIT_K):
            # Note: Due to split_k access "pid_k" = k * SPLIT_K + pid_z
            # Access a_scale[pid_m, k * SPLIT_K + pid_z]
            # and b_scale[k * SPLIT_K + pid_z, pid_n]

            scale_k = pid_k // k_multiple
            scale_k_next = scale_k + 1
            a_scale = tl.load(
                A_scale + scale_m * stride_scale_am + scale_k * stride_scale_ak
            )
            b_scale = tl.load(
                B_scale + scale_n * stride_scale_bn + scale_k * stride_scale_bk
            )
            scale = a_scale * b_scale
            if k + 1 == tl.cdiv(K, BLOCK_K * SPLIT_K):
                scale_next_inv_scale = scale
            else:
                a_scale_next = tl.load(
                    A_scale + scale_m * stride_scale_am + scale_k_next * stride_scale_ak
                )
                b_scale_next = tl.load(
                    B_scale + scale_n * stride_scale_bn + scale_k_next * stride_scale_bk
                )
                scale_next = a_scale_next * b_scale_next
                scale_next_inv_scale = scale / scale_next
            acc *= scale_next_inv_scale

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
def _kernel_matmul_fp8_block_slowacc(
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
        A_scale (TensorWrapper): [cdiv(M, scale_block_m), cdiv(K, scale_block_k)] reciprocal scale tensor per block. A * A_scale = original A
        B_scale (TensorWrapper): [cdiv(N, scale_block_n), cdiv(K, scale_block_k)] reciprocal scale tensor per block. B * B_scale = original B
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

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        # Note: Due to split_k access "pid_k" = k * SPLIT_K + pid_z
        # Access a_scale[pid_m, k * SPLIT_K + pid_z]
        # and b_scale[k * SPLIT_K + pid_z, pid_n]
        pid_k = k * SPLIT_K + pid_z
        scale_k = pid_k * BLOCK_K // scale_block_k
        a_scale = tl.load(
            A_scale + scale_m * stride_scale_am + scale_k * stride_scale_ak
        )
        b_scale = tl.load(
            B_scale + scale_n * stride_scale_bn + scale_k * stride_scale_bk
        )
        scale = a_scale * b_scale

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

        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=allow_tf32) * scale
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


@torch.library.custom_op("triton::matmul_fp8_block", mutates_args=())
def matmul_fp8_block(
    a: torch.Tensor,
    b: torch.Tensor,
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
        a (torch.Tensor): [M, K] input tensor.
        b (torch.Tensor): [N, K] input tensor.
        a_scale (torch.Tensor): [cdiv(M, scale_block_m), cdiv(K, scale_block_k)] reciprocal scale tensor per scale block. A * A_scale = original A
        b_scale (torch.Tensor): [cdiv(N, scale_block_n), cdiv(K, scale_block_k)] reciprocal scale tensor per scale block. B * B_scale = original B
        scale_block_m (int): Block size for M dimension of A_scale.
        scale_block_n (int): Block size for N dimension of B_scale.
        scale_block_k (int): Block size for K dimension of A_scale and B_scale.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.

    Returns:
        Tensor: [M, N] output tensor, (a / a_scale) @ (b / b_scale)
    """
    # Get datatypes and constants to use.
    _, tl_dtype, _, _ = get_fp8_constants()
    # Handle 3D+ a shape
    a_shape = a.shape
    a = a.view(-1, a.size(-1))
    # Reinterpret inputs into proper triton fp8 dtype.
    a_tl = convert_fp8_type(a, tl_dtype)
    b_tl = convert_fp8_type(b, tl_dtype)

    M, N, K, m_key, n_key, k_key, c, _, dot_out_dtype_triton, device = prep_matmul(
        a_tl, b_tl, dot_out_dtype
    )

    output_shape = a_shape[:-1] + (N,)
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

    if fp8_fast_accum:
        _kernel_matmul_fp8_block_fastacc[grid](
            a_tl,
            b_tl,
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
            GROUP_M=8,
            AB_DTYPE=False,
        )
    else:
        _kernel_matmul_fp8_block_slowacc[grid](
            a_tl,
            b_tl,
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
            GROUP_M=8,
            AB_DTYPE=False,
        )
    return c.view(output_shape)


@matmul_fp8_block.register_fake
def matmul_fp8_block_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_block_m: int = 256,
    scale_block_n: int = 256,
    scale_block_k: int = 256,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
) -> torch.Tensor:
    """Shape function for torch compile."""
    M, K = a.shape
    N, K = b.shape
    return torch.empty((M, N), device=a.device, dtype=torch.bfloat16)


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
) -> Tuple[int, int, int, int, int, int, torch.Tensor, str, str, torch.device]:
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
    assert a.dtype in [
        tl.float8e4nv,
        tl.float8e4b15,
        tl.float8e5,
        tl.float8e4b8,
    ] and b.dtype in [
        tl.float8e4nv,
        tl.float8e4b15,
        tl.float8e5,
        tl.float8e4b8,
    ]
    c_dtype = torch.bfloat16
    c_dtype_triton = tl.bfloat16

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

    return M, N, K, m_key, n_key, k_key, c, c_dtype_triton, dot_out_dtype_triton, device


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
    scale_ub,
    M,
    N,
    stride_am,
    stride_an,
    stride_om,
    stride_on,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
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
        A_scale (Tensor): [m] reciprocal scale tensor per row.
        A_fp8 (Tensor): [m, n] fp8 scaled tensor. A_fp8 = A / a_scale
        scale_ub (Tensor): [1] Maximum value allowed for scale.
        M (int): Number of rows.
        N (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whethar to apply scale_ub.
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

    # Clamp max value appropriately.
    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        cur_max = tl.clamp(cur_max, EPS, ub)
    else:
        cur_max = tl.maximum(cur_max, EPS)
    # Scale and quantize.
    a_scale = MAX_FP8 / cur_max
    tl.store(A_scale + pid, 1.0 / a_scale)
    n_offset = tl.arange(0, BLOCK_SIZE)
    for _k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        a = tl.load(
            A + pid * stride_am + n_offset * stride_an, mask=n_offset < N, other=0.0
        )
        a_fp8 = a * a_scale
        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8)
        a_fp8.to(TL_FP8_DTYPE)
        tl.store(
            A_fp8 + pid * stride_om + n_offset * stride_on, a_fp8, mask=n_offset < N
        )
        n_offset += BLOCK_SIZE


def triton_quantize_fp8_row(
    a: Tensor, scale_ub: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Call the triton quantize fp8 row kernel to quantize a tensor to fp8 with row-wise scalings.

    Args:
        a (Tensor): [m, n] higher precision input tensor.
        scale_ub (Tensor): Maximum allowed value for scale.

    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: reciprocal scale tensor per row.
    """
    a_shape = a.shape
    a = a.view(-1, a.size(-1))
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    num_rows = a.shape[0]
    a_scale = torch.empty((num_rows), dtype=torch.float32, device=a.device)
    a_fp8 = torch.empty((a.shape[0], a.shape[1]), device=a.device, dtype=pt_dtype)

    grid = (num_rows,)
    _kernel_quantize_fp8_row[grid](
        a,
        a_scale,
        a_fp8,
        scale_ub,
        a.shape[0],
        a.shape[1],
        a.stride(0),
        a.stride(1),
        a_fp8.stride(0),
        a_fp8.stride(1),
        TL_FP8_DTYPE=tl_dtype,
        MAX_FP8=max_fp8,
        EPS=eps,
        CLAMP_MAX=scale_ub is not None,
    )

    return a_fp8.view(a_shape), a_scale


@torch.library.custom_op("triton::quantize_fp8_row", mutates_args=())
def quantize_fp8_row(
    a: Tensor,
    scale_ub: Optional[Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a to fp8 with row-wise scalings and optionally move to output device.

    Args:
        a (Tensor): Input high precision tensor.
        scale_ub (Tensor): Maximum allowed value for scale.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.

    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: The reciprocal scale tensor per row.
    """
    a_shape = a.shape
    a = a.view(-1, a.size(-1))
    if a.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        aq, a_scale = triton_quantize_fp8_row(a, scale_ub)
        return aq.view(a_shape), a_scale
    # else use pytorch implementation.
    if not output_device:
        output_device = a.device

    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()
    row_max: torch.Tensor = torch.max(torch.abs(a), dim=1)[0]
    # Apply clamping.
    if scale_ub is not None:
        row_max = torch.clamp(row_max, min=eps, max=scale_ub.item())
    else:
        # pyre-ignore[6]: Incompatible parameter type [6]
        row_max = torch.clamp(row_max, min=eps)
    a_scale = torch.empty((a.shape[0]), dtype=torch.float32, device=output_device)
    a_scale = max_fp8 / row_max.to(torch.float32)  # pyre-ignore
    a_scale[a_scale == float("inf")] = 1.0  # pyre-ignore
    a_fp8 = a * a_scale[:, None]  # pyre-ignore
    # Cast and move data to output device (for cpu weight loading).
    a_fp8 = a_fp8.to(device=output_device, dtype=pt_dtype)
    a_scale = a_scale.to(output_device)  # pyre-ignore
    del a
    return a_fp8.view(a_shape), 1 / a_scale  # pyre-ignore


@quantize_fp8_row.register_fake
def quantize_fp8_row_meta(
    a: Tensor,
    scale_ub: Optional[Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shape function for torch compile."""
    if output_device is None:
        output_device = a.device
    M, K = a.shape
    dtype = get_fp8_constants()[0]
    fake_out = torch.empty((M, K), device=output_device, dtype=dtype)
    fake_scale = torch.empty((M), device=output_device, dtype=torch.float32)
    return fake_out, fake_scale


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
def _kernel_scale_fp8_row(
    A,
    x_scale,
    w_scale,
    scaled_out,
    M,
    N,
    stride_am,
    stride_an,
    stride_om,
    stride_on,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    Scale each row of A by x_scale and each column of A by w_scale.

    Args:
        A (Tensor): [m, n] Input tensor to scale.
        x_scale (Tensor): [m] Row-wise scale tensor.
        w_scale (Tensor): [n] Col-wise scale tensor.
        scaled_out (Tensor): [m, n] Output tensor.
        M (int): Number of rows.
        N (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        BLOCK_SIZE (int): Block size for data loads.
    """
    pid = tl.program_id(0)
    n_offset = tl.arange(0, BLOCK_SIZE)
    # Load activation scale for this row.
    row_scale = tl.load(x_scale + pid)

    # Iterate over chunks of the row and apply scales.
    for _k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        a = tl.load(A + pid * stride_am + n_offset * stride_an)
        col_scale = tl.load(w_scale + n_offset)
        scaled_a = a * row_scale * col_scale
        tl.store(
            scaled_out + pid * stride_om + n_offset * stride_on,
            scaled_a,
            mask=n_offset < N,
        )
        n_offset += BLOCK_SIZE


def scale_fp8_row(
    a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
) -> torch.Tensor:
    """
    Apply only rowwise scaling to a tensor. Useful when combining with kernels
    that do not support fused rowwise scaling.

    Args:
        a (Tensor): Input floating point tensor to be scaled.
        x_scale (Tensor): Row-wise activation scale tensor.
        w_scale (Tensor): Col-wise weight scale tensor.
    """
    if a.device == torch.device("cpu"):
        # On CPU we'll just use native pytorch to scale.
        return a * x_scale[:, None] * w_scale[None, :]

    # Otherwise, use a fast triton kernel to implement.
    # We'll parallelize over rows.
    num_rows = a.shape[0]
    scaled_out = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    grid = (num_rows,)
    _kernel_scale_fp8_row[grid](
        a,
        x_scale,
        w_scale,
        scaled_out,
        a.shape[0],
        a.shape[1],
        a.stride(0),
        a.stride(1),
        scaled_out.stride(0),
        scaled_out.stride(1),
    )

    return scaled_out


@triton.jit
def _kernel_quantize_fp8_block(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    M,
    K,
    stride_am,
    stride_ak,
    stride_om,
    stride_ok,
    stride_a_scale_m,
    stride_a_scale_k,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Quantize and scale each [BLOCK_M, BLOCK_K] block.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(A[i:i+BLOCK_M, j:j+BLOCK_K])))

    Kernel naively iterates through  matrix with [BLOCK_M, BLOCK_K] tiles.

    Todo:
        * Better tiling and ordering schemes.

    Args:
        A (Tensor): [M, K] higher precision input tensor.
        A_scale (Tensor): [cdiv(M, BLOCK_M), cdiv(K, BLOCK_K)] reciprocal scale tensor per block.
        A_fp8 (Tensor): [M, K] fp8 scaled tensor. A_fp8 = A * a_scale
        scale_ub (Tensor): [1] Maximum allowed value for scale.
        M (int): Number of rows.
        K (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_ok (int): Stride of k dimension of output.
        stride_a_scale_m (int): Stride of m dimension of A_scale.
        stride_a_scale_k (int): Stride of k dimension of A_scale.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whether to apply scale_ub.
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
    out_offset = rm[:, None] * stride_om + rk[None, :] * stride_ok
    a_mask = (rm < M)[:, None] & (rk < K)[None, :]
    a_block = tl.load(A + a_offset, mask=a_mask, other=0.0)

    block_max = tl.max(tl.abs(a_block))
    # Apply appropriate clamping.
    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        block_max = tl.clamp(block_max, EPS, ub)
    else:
        block_max = tl.maximum(block_max, EPS)
    scale = MAX_FP8 / block_max

    tl.store(
        A_scale + block_m * stride_a_scale_m + block_k * stride_a_scale_k, 1.0 / scale
    )
    a_fp8 = a_block * scale
    # Clamp A to fp8 range to make sure there's no overflow.
    # This is required for AMD. Nvidia's default saturation
    # handles it, but it's nice to have anyway.
    a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8)
    a_fp8.to(TL_FP8_DTYPE)
    tl.store(A_fp8 + out_offset, a_fp8, mask=a_mask)


def triton_quantize_fp8_block(
    x: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
    scale_ub: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with block-wise scalings.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(x[i:i+block_m, j:j+block_k])))

    Args:
        x (torch.Tensor): [M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        block_k (int): Block size for K dimension of scale.
        scale_ub: Maximum allowed value for scale.

    Returns:
        torch.Tensor : [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] reciprocal scale tensor per block.
    """
    assert x.device != torch.device(
        "cpu"
    ), "Blockwise quantization not support on cpu, please use row-wise quantization instead."
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    M, K = x.shape
    grid_m = triton.cdiv(M, block_m)
    grid_k = triton.cdiv(K, block_k)
    x_scale = torch.ones((grid_m, grid_k), device=x.device, dtype=torch.float32)
    x_fp8 = torch.empty((M, K), device=x.device, dtype=pt_dtype)

    _kernel_quantize_fp8_block[(grid_m * grid_k,)](
        x,
        x_scale,
        x_fp8,
        scale_ub,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        # pyre-ignore[6]: Incompatible parameter type [6]
        TL_FP8_DTYPE=tl_dtype,
        # pyre-ignore[6]: Incompatible parameter type [6]
        MAX_FP8=max_fp8,
        # pyre-ignore[6]: Incompatible parameter type [6]
        EPS=eps,
        # pyre-ignore[6]: Incompatible parameter type [6]
        CLAMP_MAX=scale_ub is not None,
        # pyre-ignore[6]: Incompatible parameter type [6]
        BLOCK_M=block_m,
        # pyre-ignore[6]: Incompatible parameter type [6]
        BLOCK_K=block_k,
    )

    return x_fp8.view(x_shape), x_scale


def quantize_fp8_block(
    x: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
    scale_ub: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with block-wise scalings and optionally move to output device.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(x[i:i+block_m, j:j+block_k])))

    Args:
        x (Tensor): [M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        block_k (int): Block size for K dimension of scale.
        scale_ub: Maximum allowed value for scale.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.

    Returns:
        torch.Tensor: [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] reciprocal scale tensor per block.
    """
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    if x.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        xq, x_scale = triton_quantize_fp8_block(x, block_m, block_k, scale_ub)
        return xq.view(x_shape), x_scale
    # else use pytorch implementation.
    if not output_device:
        output_device = x.device

    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()

    M, K = x.shape
    grid_m = triton.cdiv(M, block_m)
    grid_k = triton.cdiv(K, block_k)

    # Pad x to multiple of block size.
    padded_m = grid_m * block_m
    padded_k = grid_k * block_k
    x_padded = torch.zeros(padded_m, padded_k, dtype=x.dtype, device=x.device)
    x_padded[:M, :K] = x

    # Blockwise max.
    block_max = (
        x_padded.abs().reshape(grid_m, block_m, grid_k, block_k).amax(dim=(1, 3))
    )

    # Apply clamping.
    if scale_ub is not None:
        block_max = torch.clamp(block_max, min=eps, max=scale_ub.item())
    else:
        # pyre-ignore[6]: Incompatible parameter type [6]
        block_max = torch.clamp(block_max, min=eps)
    x_scale = torch.empty((grid_m, grid_k), dtype=torch.float32, device=output_device)
    x_scale = max_fp8 / block_max.to(torch.float32)  # pyre-ignore
    # pyre-ignore[16]: Undefined attribute [16]
    x_scale[x_scale == float("inf")] = 1.0
    x_fp8 = (
        x_padded
        # pyre-ignore[16]: Undefined attribute [16]
        * x_scale.repeat_interleave(block_m, dim=0).repeat_interleave(block_k, dim=1)
    )[:M, :K]

    # Cast and move data to output device (for cpu weight loading).
    x_fp8 = x_fp8.to(device=output_device, dtype=pt_dtype)
    x_scale = x_scale.to(output_device)  # pyre-ignore
    del x, x_padded
    return x_fp8.view(x_shape), 1 / x_scale  # pyre-ignore
