# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import functools
import logging
import os
from typing import List, Optional, Tuple, Union

import torch
import triton  # @manual

import triton.language as tl  # @manual

from fbgemm_gpu.experimental.gemm.triton_gemm.matmul_perf_model import (
    early_config_prune,
    estimate_matmul_time,
)
from fbgemm_gpu.experimental.gemm.triton_gemm.utils import (
    map_dtype_to_triton,
    TmaAutoTuneHelper,
)

from packaging import version
from torch._tensor import Tensor

from triton import Config  # @manual
from triton.runtime.jit import reinterpret as tl_reinterpret, TensorWrapper  # @manual

logger: logging.Logger = logging.getLogger(__name__)

running_on_github: bool = os.getenv("GITHUB_ENV") is not None

try:
    # pyre-ignore[21]
    from triton.fb.compat import disable_bufferops  # @manual
except ModuleNotFoundError:
    # Ensure we can call disable_bufferops if compat is not included (e.g. opensource)
    # TODO(njriasan): Remove when we integrate triton.fb.compat into every Triton
    # version.
    from contextlib import contextmanager

    @contextmanager
    def disable_bufferops(_unused: bool):
        yield None


@functools.lru_cache
def supports_float8_fnuz(throw_on_hip_incompatibility: bool = True) -> bool:
    if torch.version.hip:
        device_capability = torch.cuda.get_device_capability()

        if device_capability < (9, 4):
            gpu_arch = torch.cuda.get_device_properties("cuda").gcnArchName
            msg = f"Unsupported GPU arch: {gpu_arch} for FP8"
            if throw_on_hip_incompatibility:
                raise RuntimeError(msg)
            else:
                logging.error(msg)
                return False

        elif device_capability == (9, 4):
            return True

    return False


def get_fp8_constants() -> Tuple[torch.dtype, tl.dtype, float, float]:
    """
    Helper function to get constant values for the current platform.

    Returns:
        pt_dtype (torch.dtype): The correct torch fp8 datatype.
        tl_dtype (tl.dtype): The correct triton fp8 datatype.
        max_fp8 (float): The maximum reprsentable value for the fp8 datatype.
        eps (float): Minimum clip value to prevent divide by zero.
    """
    if supports_float8_fnuz(throw_on_hip_incompatibility=(not running_on_github)):
        pt_fp8_dtype = torch.float8_e4m3fnuz
        tl_fp8_dtype = tl.float8e4b8
    else:
        pt_fp8_dtype = torch.float8_e4m3fn
        tl_fp8_dtype = tl.float8e4nv

    return pt_fp8_dtype, tl_fp8_dtype, torch.finfo(pt_fp8_dtype).max, 1e-12


def reinterpret_fp8_type(tensor: torch.Tensor, dtype: tl.dtype) -> TensorWrapper:
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


def dummy_prune_configs(configs, named_args, **kwargs):

    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]

    logger.info(f"{len(configs)=} {len(configs)=} for {M=} {N=} {K=}")
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
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256, "SPLIT_K": 1},
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
    prune_configs_by={
        "early_config_prune": dummy_prune_configs,
    },
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
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
    skip_scaling_a: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
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
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
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
    acc_dtype = tl.float32 if allow_tf32 else dot_out_dtype
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

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
        acc = tl.dot(a, b, acc, out_dtype=acc_dtype, allow_tf32=allow_tf32)

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Invert scaling.
            b_scale = tl.load(B_scale + rn, mask=rn < N)
            if skip_scaling_a:
                acc *= b_scale[None, :]
            else:
                a_scale = tl.load(A_scale + rm, mask=rm < M)
                # pyre-ignore[16]: Undefined attribute [16]: `float`
                # has no attribute `__getitem__`.
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
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)


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
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
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
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
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
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 512, "SPLIT_K": 1},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
    use_cuda_graph=True,
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
    bias_dtype: tl.constexpr,
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
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
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


has_warp_specialization = hasattr(tl, "async_task")


def make_autotuner_config(dictargs, **kwargs):
    # NOTE: Triton 3.4.x removed some keyword arguments from Config constructor;
    # however, fbcode uses 3.3.1, and so this shim is provided to support both
    # versions.
    #
    # https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/runtime/autotuner.py#L275
    # https://github.com/triton-lang/triton/blame/release/3.4.x/python/triton/runtime/autotuner.py#L319
    if version.parse(triton.__version__) > version.parse("3.3.1"):
        for key in ["num_buffers_warp_spec", "num_consumer_groups"]:
            kwargs.pop(key, None)
    return Config(dictargs, **kwargs)


def get_ws_configs() -> List[Config]:
    if not has_warp_specialization:
        return []
    return [
        make_autotuner_config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 128,
                "SPLIT_K": 1,
                "NUM_CONSUMER_GROUPS": 2,
            },
            num_stages=3,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=3,
        ),
        make_autotuner_config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 128,
                "SPLIT_K": 1,
                "NUM_CONSUMER_GROUPS": 2,
            },
            num_stages=4,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=4,
        ),
        make_autotuner_config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 128,
                "SPLIT_K": 1,
                "NUM_CONSUMER_GROUPS": 1,
            },
            num_stages=3,
            num_warps=8,
            num_consumer_groups=0,
            num_buffers_warp_spec=3,
        ),
        make_autotuner_config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 512,
                "SPLIT_K": 1,
                "NUM_CONSUMER_GROUPS": 1,
            },
            num_stages=3,
            num_warps=4,
            num_consumer_groups=0,
            num_buffers_warp_spec=3,
        ),
    ]


@triton.autotune(
    configs=[
        Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 128,
                "SPLIT_K": 1,
                "NUM_CONSUMER_GROUPS": 1,
            },
            num_stages=3,
            num_warps=8,
        ),
    ]
    + get_ws_configs(),
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
    use_cuda_graph=True,
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row_tma_persistent_ws_cooperative(
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
    bias_dtype: tl.constexpr,
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
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M   , K] input tensor.
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
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    dtype_fp8 = tl.float8e4nv
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        # pyre-ignore
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_M, BLOCK_K] pointers
        # `b_ptrs` is a block of [BLOCK_K, BLOCK_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        offs_k = 0
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
        # pyre-ignore
        tl.assume(tl.cdiv(K, BLOCK_K) > 0)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            # pyre-ignore
            with tl.async_task([0]):
                a = tl._experimental_descriptor_load(
                    A_ptr,
                    [offs_am, offs_k],
                    [BLOCK_M, BLOCK_K],
                    dtype_fp8,
                )
                b = tl._experimental_descriptor_load(
                    B_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K], dtype_fp8
                )

            if fp8_fast_accum:
                acc = tl.dot(
                    a, b.T, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32
                )
            else:
                acc += tl.dot(a, b.T, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

            offs_k += BLOCK_K

        # pyre-ignore
        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
            # Invert scaling.
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            a_scale = tl.load(A_scale + rm, mask=rm < M)
            b_scale = tl.load(B_scale + rn, mask=rn < N)
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


def _is_eligible_for_skip_scaling(
    is_rowwise: bool,
    fp8_fast_accum: bool,
    imprecise_acc: bool,
    tma_persistent: bool,
    no_use_persistent: Optional[bool],
    use_warp_specialization: bool,
) -> bool:
    if not is_rowwise:
        return False

    return (
        fp8_fast_accum
        and not imprecise_acc
        and not tma_persistent
        and not no_use_persistent
        and not use_warp_specialization
    )


@torch._library.triton_op("triton::matmul_fp8_row", mutates_args=())
def matmul_fp8_row(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor],
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
    imprecise_acc: bool = False,
    tma_persistent: bool = True,
    no_use_persistent: Optional[bool] = None,
    use_warp_specialization: bool = False,
) -> torch.Tensor:
    """
    Performs matmul on [M, K] and [N, K] fp8 matrices with row-wise scalings [M], [N].

    Args:
        a (torch.Tensor): [M, K] input tensor.
        b (torch.Tensor): [N, K] input tensor.
        a_scale (Optiona;[torch.Tensor]): [M] reciprocal scale tensor per row.
            A * a_scale = original A. Scaling will be skiped if a_scale is None.
        b_scale (torch.Tensor): [N] reciprocal scale tensor per row. B * b_scale = original B
        bias (torch.Tensor): [N] optional bias tensor to add to output if provided.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        tma_persistent (bool): Whether to use TMA persistent kernel impl.

    Returns:
        torch.Tensor: [M, N] Output tensor a @ b / (a_scale[:, None] * b_scale[None, :])
    """
    if no_use_persistent is None:
        # Default True for AMD and False for Nvidia.
        if torch.version.hip is not None:
            no_use_persistent = True
        else:
            no_use_persistent = False
    # Get datatypes and constants to use.
    pt_fp8_dtype, _, _, _ = get_fp8_constants()
    # Handle 3D+ a shape
    a_shape = a.shape
    a = a.view(-1, a.size(-1))
    # View inputs into proper torch fp8 dtype.
    if torch.version.cuda:
        assert a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    else:
        assert a.dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)
    assert b.dtype == pt_fp8_dtype
    M, N, K, m_key, n_key, k_key, c, c_dtype_triton, dot_out_dtype_triton, device = (
        prep_matmul(a, b, dot_out_dtype)
    )

    # Skip scaling (a_scale is None) can only be applied in certain cases.
    assert a_scale is not None or _is_eligible_for_skip_scaling(
        is_rowwise=True,
        fp8_fast_accum=fp8_fast_accum,
        imprecise_acc=imprecise_acc,
        tma_persistent=tma_persistent,
        no_use_persistent=no_use_persistent,
        use_warp_specialization=use_warp_specialization,
    )

    output_shape = a_shape[:-1] + (N,)
    # Handle tensor with empty inputs.
    if (M == 0) or (N == 0) or (K == 0):
        return torch.zeros(output_shape, device=device, dtype=torch.bfloat16)
    # launch kernel
    if a.device == torch.device("cpu"):
        logger.info(
            "FP8 Row-wise Triton kernel not supported on cpu, fallback to torch"
        )
        if a_scale is None:
            scale = b_scale[None, :]
        else:
            scale = a_scale[:, None] * b_scale[None, :]
        output = torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16).T) * scale
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

    if no_use_persistent:
        logger.debug("Using non-persistent kernel")
        with torch.cuda.device(a.device.index):
            torch._library.capture_triton(_kernel_matmul_fp8_row_non_persistent)[grid](
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
                # GROUP_M=8,
                USE_BIAS=bias is not None,
                AB_DTYPE=False,
            )
    elif use_warp_specialization:
        assert has_warp_specialization
        # used by TMA warp specialization kernel
        desc_helper = TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("a")
        desc_helper.init_tma_descriptor("b")
        desc_helper.init_tma_descriptor("c")
        desc_helper.init_tma_descriptor("a_scale")
        desc_helper.init_tma_descriptor("b_scale")
        desc_helper.init_tma_descriptor("bias")

        def persistent_grid_tma_ws(META):
            nonlocal desc_helper  # noqa: F824
            desc_helper.fill_2d_tma_descriptor(
                "a",
                a.data_ptr(),
                M,
                K,
                META["BLOCK_M"] // META["NUM_CONSUMER_GROUPS"],
                META["BLOCK_K"],
                a.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "b",
                b.data_ptr(),
                N,
                K,
                META["BLOCK_N"],
                META["BLOCK_K"],
                b.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "c",
                c.data_ptr(),
                M,
                N,
                META["BLOCK_M"] // META["NUM_CONSUMER_GROUPS"],
                META["BLOCK_N"],
                c.element_size(),
            )
            desc_helper.fill_1d_tma_descriptor(
                "a_scale",
                a_scale.data_ptr(),
                M,
                META["BLOCK_M"] // META["NUM_CONSUMER_GROUPS"],
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

        bias_dtype_triton = None
        if bias is not None:
            bias_dtype_triton = map_dtype_to_triton(bias.dtype)

        # pyre-ignore
        torch._library.capture_triton(
            _kernel_matmul_fp8_row_tma_persistent_ws_cooperative
        )[persistent_grid_tma_ws](
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            m_key,
            n_key,
            k_key,
            a_scale,
            b_scale,
            desc_bias,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype_triton,
            c_dtype=c_dtype_triton,
            bias_dtype=bias_dtype_triton,
            allow_tf32=allow_tf32,
            fp8_fast_accum=fp8_fast_accum,
            GROUP_M=8,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
            USE_BIAS=bias is not None,
        )
    elif tma_persistent:
        # used by TMA persistent kernel
        desc_helper = TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("a")
        desc_helper.init_tma_descriptor("b")
        desc_helper.init_tma_descriptor("c")
        desc_helper.init_tma_descriptor("a_scale")
        desc_helper.init_tma_descriptor("b_scale")
        desc_helper.init_tma_descriptor("bias")

        def persistent_grid_tma(META):
            nonlocal desc_helper  # noqa: F824
            desc_helper.fill_2d_tma_descriptor(
                "a",
                a.data_ptr(),
                M,
                K,
                META["BLOCK_M"],
                META["BLOCK_K"],
                a.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "b",
                b.data_ptr(),
                N,
                K,
                META["BLOCK_N"],
                META["BLOCK_K"],
                b.element_size(),
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

        bias_dtype_triton = None
        if bias is not None:
            bias_dtype_triton = map_dtype_to_triton(bias.dtype)

        # pyre-ignore
        torch._library.capture_triton(_kernel_matmul_fp8_row_tma_persistent)[
            persistent_grid_tma
        ](
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
            bias_dtype=bias_dtype_triton,
            allow_tf32=allow_tf32,
            fp8_fast_accum=fp8_fast_accum,
            GROUP_M=8,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
            USE_BIAS=bias is not None,
        )
    elif imprecise_acc:
        with torch.cuda.device(a.device.index):
            torch._library.capture_triton(_kernel_matmul_fp8_row_imprecise_acc)[grid](
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
        skip_scaling_a = a_scale is None
        torch._library.capture_triton(_kernel_matmul_fp8_row)[persistent_grid](
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
            skip_scaling_a=skip_scaling_a,
            GROUP_M=8,
            USE_BIAS=bias is not None,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
        )
    else:
        torch._library.capture_triton(_kernel_matmul_fp8_row_no_fast_acc)[
            persistent_grid
        ](
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
    a_scale: Optional[torch.Tensor],
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
    imprecise_acc: bool = False,
    tma_persistent: bool = True,
    no_use_persistent: Optional[bool] = None,
    use_warp_specialization: bool = False,
) -> torch.Tensor:
    """Shape function for torch compile."""
    M, K = a.shape
    N, K = b.shape
    return torch.empty(
        (M, N),
        device=a.device,
        dtype=torch.bfloat16 if dot_out_dtype is None else dot_out_dtype,
    )


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
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
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
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
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
    _, tl_fp8_dtype, _, _ = get_fp8_constants()
    # Handle 3D+ a shape
    a_shape = a.shape
    a = a.view(-1, a.size(-1))
    # View inputs into proper triton fp8 dtype.
    a_tl = reinterpret_fp8_type(a, tl_fp8_dtype)
    b_tl = reinterpret_fp8_type(b, tl_fp8_dtype)

    M, N, K, m_key, n_key, k_key, c, _, dot_out_dtype_triton, device = prep_matmul(
        a_tl, b_tl, dot_out_dtype
    )

    output_shape = a_shape[:-1] + (N,)
    # Handle case where inputs are empty.
    if (M == 0) or (N == 0) or (K == 0):
        return torch.zeros(output_shape, device=device, dtype=torch.bfloat16)

    # launch kernel
    assert device != torch.device(
        "cpu"
    ), "Blockwise matmul not supported on cpu, please use row-wise instead."

    if b.device != a.device:
        raise Exception("'b' must be on the same device as 'a'")
    if a_scale.device != a.device:
        raise Exception("'a_scale' must be on the same device as 'a'")
    if b_scale.device != a.device:
        raise Exception("'b_scale' must be on the same device as 'a'")

    # noqa: E731:
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )

    if fp8_fast_accum:
        with torch.cuda.device(a_tl.device.index):
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
        with torch.cuda.device(a_tl.device.index):
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
    a: Union[TensorWrapper, torch.Tensor],
    b: Union[TensorWrapper, torch.Tensor],
    dot_out_dtype: Optional[torch.dtype],
) -> Tuple[
    int, int, int, int, int, int, torch.Tensor, tl.dtype, tl.dtype, torch.device
]:
    """
    Shared bookkeeping for a @ b.T matmul.

    Args:
        a (torch.Tensor): [M, K] input tensor.
        b (torch.Tensor): [N, K] input tensor.
        dot_out_dtype (tl.dtype): Output type of tensor core.

    Returns:
        M (int): Number of rows in A.
        N (int): Number of rows in B.
        K (int): Number of cols in A and cols in B.
        m_key (int): Autotuning key for M dim.
        n_key (int): Autotuning key for N dim.
        k_key (int): Autotuning key for K dim.
        c (Tensor): [M, N] output tensor.
        c_dtype_triton (tl.dtype): Type of output tensor.
        dot_out_dtype (tl.dtype): Output type of tensor core.
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
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        tl.float8e4nv,
        tl.float8e4b15,
        tl.float8e5,
        tl.float8e4b8,
    ]
    assert b.dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        tl.float8e4nv,
        tl.float8e4b15,
        tl.float8e5,
        tl.float8e4b8,
    ]

    c_dtype, c_dtype_triton = (
        (torch.bfloat16, tl.bfloat16)
        if dot_out_dtype is None
        else (dot_out_dtype, map_dtype_to_triton(dot_out_dtype))
    )

    c = torch.empty((M, N), device=device, dtype=c_dtype)
    if dot_out_dtype is None:
        dot_out_dtype_triton = tl.float32
    else:
        assert isinstance(
            dot_out_dtype, torch.dtype
        ), f"dot_out_dtype type {type(dot_out_dtype)} must be a torch.dtype"
        dot_out_dtype_triton = map_dtype_to_triton(dot_out_dtype)

    return M, N, K, m_key, n_key, k_key, c, c_dtype_triton, dot_out_dtype_triton, device


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
def _kernel_quantize_fp8_row(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    zero_start_index_M,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_an,
    stride_ak,
    stride_ob,
    stride_om,
    stride_on,
    stride_ok,
    stride_zb,
    stride_zm,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    JAGGED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    """Quantize and scale each row.

    Scale per row i is computed as MAX_FP8 / max(abs(A[i, :]))

    Kernel naively iterates through  matrix with [1, BLOCK_SIZE] tiles
    in a max pass then scale/quantize pass.

    Todo:
        * Better tiling schemes.

    Args:
        A (Tensor): higher precision input tensor of 4 dimension.
        A_scale (Tensor): [B * M * N] reciprocal scale tensor per row.
        A_fp8 (Tensor): fp8 scaled tensor. A_fp8 = A / a_scale
        scale_ub (Tensor): [1] Maximum value allowed for scale.
        B (int): Size of dimenion 0
        M (int): Size of dimenion 1
        N (int): Size of dimenion 2
        K (int): Size of dimenion 3
        stride_ab (int): Stride of b dimension of A.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_ob (int): Stride of b dimension of output.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        stride_ok (int): Stride of k dimension of output.
        stride_zb (int): Stride of b dimension of jagged index.
        stride_zm (int): Stride of m dimension of jagged index.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whethar to apply scale_ub.
        JAGGED (bool): Whether to use jagged indexing.
        BLOCK_SIZE (int): Block size for reduction.
        USE_INT64 (bool): Whether to use int64 indexing for large inputs.
    """
    pid = tl.program_id(0)
    # Use int64 indexing for large inputs. This is slower, but
    # needed to avoid index overflows.
    if USE_INT64:
        pid = pid.to(tl.int64)
    n_offset = tl.arange(0, BLOCK_SIZE)
    a_offset_base = (
        pid // (M * N) * stride_ab
        + (pid % (M * N)) // N * stride_am
        + (pid % (M * N)) % N * stride_an
    )
    a_fp8_offset_base = (
        pid // (M * N) * stride_ob
        + (pid % (M * N)) // N * stride_om
        + (pid % (M * N)) % N * stride_on
    )

    K_in = K

    if JAGGED:
        z_offset_base = pid // (M * N) * stride_zb + (pid % (M * N)) // N * stride_zm
        group_rows = tl.load(zero_start_index_M + z_offset_base)
        current_row = pid % N
        # If this row is empty, dont process any of it.
        if current_row >= group_rows:
            K_in = 0

    # Calculate max.
    cur_max = 0.0
    for _k in range(0, tl.cdiv(K_in, BLOCK_SIZE)):
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
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

    for _k in range(0, tl.cdiv(K, BLOCK_SIZE)):
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
        )
        a_fp8 = a * a_scale
        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)
        tl.store(
            A_fp8 + a_fp8_offset_base + n_offset * stride_ok,
            a_fp8,
            mask=n_offset < K,
        )
        n_offset += BLOCK_SIZE


def triton_quantize_fp8_row(
    a: Tensor,
    scale_ub: Optional[Tensor] = None,
    zero_start_index_M: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Call the triton quantize fp8 row kernel to quantize a tensor to fp8 with row-wise scalings.

    Args:
        a (Tensor): higher precision input tensor of 4 dimension.
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.

    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: reciprocal scale tensor per row.
    """
    if scale_ub is not None and scale_ub.device != a.device:
        raise Exception("'scale_ub' must be on the same device as 'a'")
    if zero_start_index_M is not None and zero_start_index_M.device != a.device:
        raise Exception("'zero_start_index_M' must be on the same device as 'a'")

    assert a.dim() <= 4, "Triton only supports up to 4 dimension input tensor."
    a_shape = a.shape
    while a.dim() < 4:
        a = a.unsqueeze(0)
    if zero_start_index_M is not None:
        # There should be one value of zero_start_index_M per NxK matrix.
        zero_start_index_M = zero_start_index_M.view(a.shape[0], a.shape[1])
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    num_rows = a.numel() // a.shape[-1]
    a_scale = torch.empty((num_rows), dtype=torch.float32, device=a.device)
    a_fp8 = torch.empty(a.shape, device=a.device, dtype=pt_dtype)

    # If input tensor is sufficiently large, we need to use int64 indexing.
    use_int64 = a.numel() > (2**31 - 1)
    grid = (num_rows,)
    # Pick a conservative value for inference shapes for disabling BufferOps.
    should_disable_bufferops = torch.version.hip is not None and a_shape[0] < 32
    with disable_bufferops(should_disable_bufferops):
        with torch.cuda.device(a.device.index):
            _kernel_quantize_fp8_row[grid](
                a,
                a_scale,
                a_fp8,
                scale_ub,
                zero_start_index_M,
                a.shape[0],
                a.shape[1],
                a.shape[2],
                a.shape[3],
                a.stride(0),
                a.stride(1),
                a.stride(2),
                a.stride(3),
                a_fp8.stride(0),
                a_fp8.stride(1),
                a_fp8.stride(2),
                a_fp8.stride(3),
                (
                    zero_start_index_M.stride(0)
                    if zero_start_index_M is not None
                    else None
                ),
                (
                    zero_start_index_M.stride(1)
                    if zero_start_index_M is not None
                    else None
                ),
                TL_FP8_DTYPE=tl_dtype,
                MAX_FP8=max_fp8,
                EPS=eps,
                CLAMP_MAX=scale_ub is not None,
                JAGGED=zero_start_index_M is not None,
                USE_INT64=use_int64,
            )

    return a_fp8.view(a_shape), a_scale.view(a_shape[:-1])


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
def _kernel_quantize_fp8_packed_row(
    A,
    A_fp8,
    packed_scale,
    scale_ub,
    zero_start_index_M,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_an,
    stride_ak,
    stride_ob,
    stride_om,
    stride_on,
    stride_ok,
    packed_scale_stride,
    stride_zb,
    stride_zm,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    JAGGED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    """Quantize and scale each row.

    Scale per row i is computed as MAX_FP8 / max(abs(A[i, :]))

    Kernel naively iterates through  matrix with [1, BLOCK_SIZE] tiles
    in a max pass then scale/quantize pass.

    Todo:
        * Better tiling schemes.

    Args:
        A (Tensor): higher precision input tensor of 4 dimension.
        packed_scale (Tensor): [B * M * N] reciprocal scale tensor per row.
        A_fp8 (Tensor): fp8 scaled tensor. A_fp8 = A / a_scale
        scale_ub (Tensor): [1] Maximum value allowed for scale.
        B (int): Size of dimenion 0
        M (int): Size of dimenion 1
        N (int): Size of dimenion 2
        K (int): Size of dimenion 3
        stride_ab (int): Stride of b dimension of A.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_ob (int): Stride of b dimension of output.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        stride_ok (int): Stride of k dimension of output.
        packed_scale_stride (int): Stride of the packed scale, indexing into a_fp8.
        stride_zb (int): Stride of b dimension of jagged index.
        stride_zm (int): Stride of m dimension of jagged index.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whethar to apply scale_ub.
        JAGGED (bool): Whether to use jagged indexing.
        BLOCK_SIZE (int): Block size for reduction.
        USE_INT64 (bool): Whether to use int64 indexing for large inputs.
    """
    pid = tl.program_id(0)
    # Use int64 indexing for large inputs. This is slower, but
    # needed to avoid index overflows.
    if USE_INT64:
        pid = pid.to(tl.int64)
    n_offset = tl.arange(0, BLOCK_SIZE)
    a_offset_base = (
        pid // (M * N) * stride_ab
        + (pid % (M * N)) // N * stride_am
        + (pid % (M * N)) % N * stride_an
    )
    a_fp8_offset_base = (
        pid // (M * N) * stride_ob
        + (pid % (M * N)) // N * stride_om
        + (pid % (M * N)) % N * stride_on
    )

    K_in = K

    if JAGGED:
        z_offset_base = pid // (M * N) * stride_zb + (pid % (M * N)) // N * stride_zm
        group_rows = tl.load(zero_start_index_M + z_offset_base)
        current_row = pid % N
        # If this row is empty, dont process any of it.
        if current_row >= group_rows:
            K_in = 0

    # Calculate max.
    cur_max = 0.0
    for _k in range(0, tl.cdiv(K_in, BLOCK_SIZE)):
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
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

    (fp8_0, fp8_1, fp8_2, fp8_3) = tl.inline_asm_elementwise(
        asm="""
        {
            // $4 is the input register
            .reg .b32 input;
            mov.b32 input, $4;
            mov.b32 $0, $4;
            shr.b32 $1, $4, 8;
            shr.b32 $2, $4, 16;
            shr.b32 $3, $4, 24;
        }
            """,
        constraints=("=r,=r,=r,=r," "r"),
        # Let's pass in 1 uint32 value per iteration, containing 8 packed int4 values
        args=[1.0 / a_scale],
        dtype=(
            tl.uint8,
            tl.uint8,
            tl.uint8,
            tl.uint8,
        ),
        is_pure=True,
        pack=1,
    )

    # There are some compiler issues with FP8 pointers
    packed_scale_ptr = packed_scale.to(tl.pointer_type(tl.uint8))
    tl.store(packed_scale_ptr + pid * packed_scale_stride, fp8_0)
    tl.store(packed_scale_ptr + pid * packed_scale_stride + 1, fp8_1)
    tl.store(packed_scale_ptr + pid * packed_scale_stride + 2, fp8_2)
    tl.store(packed_scale_ptr + pid * packed_scale_stride + 3, fp8_3)

    n_offset = tl.arange(0, BLOCK_SIZE)

    for _k in range(0, tl.cdiv(K, BLOCK_SIZE)):
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
        )
        a_fp8 = a * a_scale
        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)
        tl.store(
            A_fp8 + a_fp8_offset_base + n_offset * stride_ok,
            a_fp8,
            mask=n_offset < K,
        )

        n_offset += BLOCK_SIZE


def triton_quantize_fp8_packed_row(
    a: Tensor,
    scale_ub: Optional[Tensor] = None,
    zero_start_index_M: Optional[Tensor] = None,
    return_only_packed: Optional[bool] = False,
) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor]:
    """
    Call the triton quantize fp8 row kernel to quantize a tensor to fp8 with row-wise scalings.

    This packs the FP32 scale at the end of each row, so the fp8 scaled tensor and the reciprocal scale tensor per row are contiguous in memory.

    Args:
        a (Tensor): higher precision input tensor of 4 dimension.
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.
        return_only_packed (bool): Only return the packed tensor, do not unpack results if True
    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: reciprocal scale tensor per row.
        torch.Tensor: The packed FP8 scaled tensor, with the scale at the end of each row.
    """
    if scale_ub is not None and scale_ub.device != a.device:
        raise Exception("'scale_ub' must be on the same device as 'a'")
    if zero_start_index_M is not None and zero_start_index_M.device != a.device:
        raise Exception("'zero_start_index_M' must be on the same device as 'a'")

    assert a.dim() <= 4, "Triton only supports up to 4 dimension input tensor."
    a_shape = a.shape
    while a.dim() < 4:
        a = a.unsqueeze(0)
    if zero_start_index_M is not None:
        # There should be one value of zero_start_index_M per NxK matrix.
        zero_start_index_M = zero_start_index_M.view(a.shape[0], a.shape[1])
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    num_rows = a.numel() // a.shape[-1]

    # Allocate an extra 4-bytes at the end of each row for the scale.
    a_fp8 = torch.empty(
        (*a.shape[:-1], a.shape[-1] + 4), device=a.device, dtype=pt_dtype
    )

    # create a view of the packed scale
    packed_scale = a_fp8[..., -4:]

    # If input tensor is sufficiently large, we need to use int64 indexing.
    use_int64 = a.numel() > (2**31 - 1)
    grid = (num_rows,)

    with torch.cuda.device(a.device.index):
        _kernel_quantize_fp8_packed_row[grid](
            a,
            a_fp8,
            packed_scale,
            scale_ub,
            zero_start_index_M,
            a.shape[0],
            a.shape[1],
            a.shape[2],
            a.shape[3],
            a.stride(0),
            a.stride(1),
            a.stride(2),
            a.stride(3),
            a_fp8.stride(0),
            a_fp8.stride(1),
            a_fp8.stride(2),
            a_fp8.stride(3),
            packed_scale.stride(2),  # this is the stride that matters
            zero_start_index_M.stride(0) if zero_start_index_M is not None else None,
            zero_start_index_M.stride(1) if zero_start_index_M is not None else None,
            TL_FP8_DTYPE=tl_dtype,
            MAX_FP8=max_fp8,
            EPS=eps,
            CLAMP_MAX=scale_ub is not None,
            JAGGED=zero_start_index_M is not None,
            USE_INT64=use_int64,
        )
    if return_only_packed:
        return None, None, a_fp8.view((*a_shape[:-1], a_shape[-1] + 4))

    # Extract the original shape data without the extra 4 bytes per row
    # The data is still contiguous in memory, so we have to unpack it.
    final_fp8_view = a_fp8[..., :-4].view(a_shape)
    scale_view = a_fp8[..., -4:].reshape((num_rows * 4)).view(torch.float32)

    # the difference with the packed API is that it also
    # returns the full packed tensor as a third return value
    return final_fp8_view, scale_view.view(a_shape[:-1]), a_fp8


@torch.library.custom_op("triton::quantize_fp8_packed_row", mutates_args=())
def quantize_fp8_packed_row(
    a: Tensor,
    scale_ub: Optional[Tensor] = None,
    zero_start_index_M: Optional[Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a to fp8 with row-wise scalings and optionally move to output device.

    Args:
        a (Tensor): Input high precision tensor. Required to have no more than 4 dimension
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.
    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: The reciprocal scale tensor per row.
    """

    if a.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        # ignore the packed tensor here, we aren't testing it
        a_fp8, scale, _ = triton_quantize_fp8_packed_row(
            a, scale_ub, zero_start_index_M, return_only_packed=False
        )
        assert a_fp8 is not None
        assert scale is not None
        return a_fp8, scale
    # else use pytorch implementation.
    if not output_device:
        output_device = a.device

    a_shape = a.shape
    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()
    row_max: torch.Tensor = torch.max(torch.abs(a), dim=-1)[0]
    # Apply clamping.
    if scale_ub is not None:
        row_max = torch.clamp(row_max, min=eps, max=scale_ub.item())
    else:
        # pyre-ignore[6]: Incompatible parameter type [6]
        row_max = torch.clamp(row_max, min=eps)
    a_scale = torch.empty((a.shape[:-1]), dtype=torch.float32, device=output_device)
    a_scale = max_fp8 / row_max.to(torch.float32)  # pyre-ignore
    a_scale[a_scale == float("inf")] = 1.0  # pyre-ignore
    a_fp8 = a * a_scale[..., None]  # pyre-ignore
    # Cast and move data to output device (for cpu weight loading).
    a_fp8 = a_fp8.to(device=output_device, dtype=pt_dtype)
    a_scale = a_scale.to(output_device)  # pyre-ignore
    del a
    return a_fp8, (1 / a_scale).view(a_shape[:-1])  # pyre-ignore


@torch.library.custom_op("triton::quantize_fp8_packed_row_raw", mutates_args=())
def quantize_fp8_packed_row_raw(
    a: Tensor,
    scale_ub: Optional[Tensor] = None,
    zero_start_index_M: Optional[Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Quantize a to fp8 with row-wise scalings and optionally move to output device.

    Identical to quantize_fp8_packed_row, except it only returns the raw packed tensor.

    Args:
        a (Tensor): Input high precision tensor. Required to have no more than 4 dimension
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.
    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: The reciprocal scale tensor per row.
    """

    if a.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        # ignore the packed tensor here, we aren't testing it
        _, _, packed_tensor = triton_quantize_fp8_packed_row(
            a, scale_ub, zero_start_index_M, return_only_packed=True
        )
        return packed_tensor
    else:
        raise Exception(
            "No PyTorch implementation provided for triton::quantize_fp8_packed_row_raw"
        )


@torch.library.custom_op("triton::quantize_fp8_row", mutates_args=())
def quantize_fp8_row(
    a: Tensor,
    scale_ub: Optional[Tensor] = None,
    zero_start_index_M: Optional[Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a to fp8 with row-wise scalings and optionally move to output device.

    Args:
        a (Tensor): Input high precision tensor. Required to have no more than 4 dimension
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.

    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: The reciprocal scale tensor per row.
    """

    if a.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        return triton_quantize_fp8_row(a, scale_ub, zero_start_index_M)
    # else use pytorch implementation.
    if not output_device:
        output_device = a.device

    a_shape = a.shape
    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()
    row_max: torch.Tensor = torch.max(torch.abs(a), dim=-1)[0]
    # Apply clamping.
    if scale_ub is not None:
        row_max = torch.clamp(row_max, min=eps, max=scale_ub.item())
    else:
        # pyre-ignore[6]: Incompatible parameter type [6]
        row_max = torch.clamp(row_max, min=eps)
    a_scale = torch.empty((a.shape[:-1]), dtype=torch.float32, device=output_device)
    a_scale = max_fp8 / row_max.to(torch.float32)  # pyre-ignore
    a_scale[a_scale == float("inf")] = 1.0  # pyre-ignore
    a_fp8 = a * a_scale[..., None]  # pyre-ignore
    # Cast and move data to output device (for cpu weight loading).
    a_fp8 = a_fp8.to(device=output_device, dtype=pt_dtype)
    a_scale = a_scale.to(output_device)  # pyre-ignore
    del a
    return a_fp8, (1 / a_scale).view(a_shape[:-1])  # pyre-ignore


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
    a_shape = a.shape
    # Flatten to 2D since each row of each potential batch gets a scale.
    dtype = get_fp8_constants()[0]
    fake_out = torch.empty(a.shape, device=output_device, dtype=dtype)
    fake_scale = torch.empty(a_shape[:-1], device=output_device, dtype=torch.float32)
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
        a = tl.load(
            A + pid * stride_am + n_offset * stride_an, mask=n_offset < N, other=0.0
        )
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

    if x_scale.device != a.device:
        raise Exception("'x_scale' must be on the same device as 'a'")
    if w_scale.device != a.device:
        raise Exception("'w_scale' must be on the same device as 'a'")

    # Otherwise, use a fast triton kernel to implement.
    # We'll parallelize over rows.
    num_rows = a.shape[0]
    scaled_out = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    grid = (num_rows,)
    with torch.cuda.device(a.device.index):
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
    K_MAJOR: tl.constexpr,
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
        K_MAJOR (bool): Whether output scales should be K major (True) or MN major (False).
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

    # Write in transposed order if specified.
    if K_MAJOR:
        scale_offset = block_m * stride_a_scale_m + block_k * stride_a_scale_k
    else:
        scale_offset = block_k * stride_a_scale_m + block_m * stride_a_scale_k
    tl.store(A_scale + scale_offset, 1.0 / scale)
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
    k_major: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with block-wise scalings.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(x[i:i+block_m, j:j+block_k])))

    Args:
        x (torch.Tensor): [M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        block_k (int): Block size for K dimension of scale.
        scale_ub: Maximum allowed value for scale.
        k_major (bool): Whether output scales should be K major (True) or MN major (False).

    Returns:
        torch.Tensor : [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] reciprocal scale tensor per block
        if k_major is True, otherwise [cdiv(K, block_k), cdiv(M, block_M)].
    """
    assert x.device != torch.device(
        "cpu"
    ), "Blockwise quantization not support on cpu, please use row-wise quantization instead."

    if scale_ub is not None and scale_ub.device != x.device:
        raise Exception("'scale_ub' must be on the same device as 'a'")

    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    M, K = x.shape
    grid_m = triton.cdiv(M, block_m)
    grid_k = triton.cdiv(K, block_k)
    if k_major:
        x_scale = torch.empty((grid_m, grid_k), device=x.device, dtype=torch.float32)
    else:
        x_scale = torch.empty((grid_k, grid_m), device=x.device, dtype=torch.float32)
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
        # pyre-ignore[6]: Incompatible parameter type [6]
        K_MAJOR=k_major,
    )

    return x_fp8.view(x_shape), x_scale


def quantize_fp8_block(
    x: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
    scale_ub: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
    k_major: bool = True,
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
        k_major (bool): Whether output scales should be K major (True) or MN major (False).

    Returns:
        torch.Tensor: [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] reciprocal scale tensor per block
        if k_major is True, otherwise [cdiv(K, block_k), cdiv(M, block_M)].
    """
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    if x.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        xq, x_scale = triton_quantize_fp8_block(x, block_m, block_k, scale_ub, k_major)
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
    if not k_major:
        x_scale = x_scale.t().contiguous()
    return x_fp8.view(x_shape), 1 / x_scale  # pyre-ignore


@triton.autotune(
    configs=[
        Config({"GROUP_LOAD": 2}),
        Config({"GROUP_LOAD": 4}),
        Config({"GROUP_LOAD": 8}),
        Config({"GROUP_LOAD": 16}),
        Config({"GROUP_LOAD": 32}),
    ],
    key=["K"],
)
@triton.jit
def _kernel_quantize_fp8_group(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    m_sizes,
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
    USE_INT64: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    USE_M_MAJOR: tl.constexpr,
    G: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
):
    """Quantize and scale each GROUP_SIZE chunk of each row.

    Scale per group i is computed as 1 / (MAX_FP8 / max(abs(A[i:i+GROUP_SIZE])))

    Each kernel thread is responsible for one row and loads and processes a tunable
    number of groups at once.

    Args:
        A (Tensor): [M, K] higher precision input tensor.
        A_scale (Tensor): [M, cdiv(K, GROUP_SIZE)] reciprocal scale tensor per group.
        A_fp8 (Tensor): [M, K] fp8 scaled tensor. A_fp8 = A * a
        scale_ub (Tensor): [1] Maximum allowed value for scale.
        m_sizes (Optional[Tensor]): [G] Number of rows in each group.
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
        USE_INT64 (bool): Whether to index using int64, which may be needed for large tensors.
        GROUP_SIZE (int): Group size for K dimension of A_scale and kernel.
        USE_M_MAJOR (bool): Whether to use grouped M-major layout for A_scale.
        G (int): Number of groups in A_scale, only relevant when m_sizes is provided.
        GROUP_LOAD (int): Number of groups to load and process simultaneously.
    """
    pid = tl.program_id(0)
    if USE_INT64:
        pid = pid.to(tl.int64)
    # We load group_size * group_load chunks at a time.
    row_offset = pid * stride_am
    out_offset = pid * stride_om
    scale_row_offset = pid * stride_a_scale_m
    k_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE)
    scale_k_offset = tl.arange(0, GROUP_LOAD)
    NUM_GROUPS: tl.constexpr = K // GROUP_SIZE

    # When dealing with an M-major grouped gemm, we need to figure out
    # which group this thread corresponds to and figure out the corresponding
    # scale offset.
    group_offset = 0
    group_cumsum = 0
    group_M = 0
    stop = False
    if USE_M_MAJOR and G > 0:
        # Iterate over groups to both compute the cumulative sum and find which group we are in.
        for i in range(G):
            if not stop:
                group_M = tl.cast(tl.load(m_sizes + i), pid.dtype)
                if (group_cumsum + group_M) <= pid:
                    group_cumsum += group_M
                else:
                    # Indicate we are finished computing cumsum.
                    stop = True

        group_offset = group_cumsum * NUM_GROUPS

    for k in range(0, tl.cdiv(K, (GROUP_LOAD * GROUP_SIZE))):
        # Load groups of the input.
        chunk_offset = k_offset + k * GROUP_LOAD * GROUP_SIZE
        a = tl.load(
            A + row_offset + chunk_offset * stride_ak, mask=chunk_offset < K, other=0.0
        )
        # View loaded chunk as a set of groups.
        a_grouped = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
        # Reduce over groups.
        group_max = tl.max(tl.abs(a_grouped), axis=1)
        # Apply clamping if specified.
        if CLAMP_MAX:
            ub = tl.load(scale_ub)
            group_max = tl.clamp(group_max, EPS, ub)
        else:
            group_max = tl.maximum(group_max, EPS)
        # Scale and quantize.
        a_scale = MAX_FP8 / group_max
        scale_chunk_offset = scale_k_offset + k * GROUP_LOAD

        if USE_M_MAJOR and G > 0:
            tl.store(
                A_scale
                + group_offset
                + (pid - group_cumsum) * stride_a_scale_k
                + (scale_chunk_offset * group_M),
                1.0 / a_scale,
                mask=scale_chunk_offset < NUM_GROUPS,
            )
        else:
            if USE_M_MAJOR:
                tl.store(
                    A_scale
                    + pid * stride_a_scale_k
                    + scale_chunk_offset * stride_a_scale_m,
                    1.0 / a_scale,
                    mask=scale_chunk_offset < NUM_GROUPS,
                )
            else:
                tl.store(
                    A_scale + scale_row_offset + scale_chunk_offset * stride_a_scale_k,
                    1.0 / a_scale,
                    mask=scale_chunk_offset < NUM_GROUPS,
                )
        # Apply scale to input.
        a_fp8 = a_grouped * a_scale[:, None]
        # Clamp to FP8 range to avoid overflow
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)
        # Write to output.
        tl.store(
            A_fp8 + out_offset + chunk_offset * stride_ok,
            tl.ravel(a_fp8),
            mask=chunk_offset < K,
        )


def triton_quantize_fp8_group(
    x: torch.Tensor,
    group_size: int = 128,
    scale_ub: Optional[torch.Tensor] = None,
    m_sizes: Optional[torch.Tensor] = None,
    k_major: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with group-wise scalings.

    Scale per group i is computed as 1 / (MAX_FP8 / max(abs(x[i:i+group_size])))

    Args:
        x (torch.Tensor): [M, K] higher precision input tensor.
        group_size (int): Group size for M dimension of scale.
        scale_ub: Maximum allowed value for scale.
        m_sizes: Optional input for grouped gemm to specify the number of rows in each group.
        k_major (bool): Whether output scales should be K major (True) or MN major (False).

    Returns:
        torch.Tensor: [M, K] fp8 scaled tensor.
        torch.Tensor: [M, cdiv(K, group_size)] reciprocal scale tensor per group.
    """
    assert x.device != torch.device(
        "cpu"
    ), "Triton groupwise quantization not supported on cpu."

    if scale_ub is not None and scale_ub.device != x.device:
        raise Exception("'scale_ub' must be on the same device as 'a'")
    if m_sizes is not None and m_sizes.device != x.device:
        raise Exception("'m_sizes' must be on the same device as 'a'")

    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    M, K = x.shape
    k_groups = triton.cdiv(K, group_size)
    if k_major:
        x_scale = torch.empty((M, k_groups), device=x.device, dtype=torch.float32)
    else:
        x_scale = torch.empty((k_groups, M), device=x.device, dtype=torch.float32)
    x_fp8 = torch.empty((M, K), device=x.device, dtype=pt_dtype)
    _kernel_quantize_fp8_group[(M,)](
        x,
        x_scale,
        x_fp8,
        scale_ub,
        m_sizes,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        TL_FP8_DTYPE=tl_dtype,
        MAX_FP8=max_fp8,
        EPS=eps,
        CLAMP_MAX=scale_ub is not None,
        USE_INT64=x.numel() > (2**32 - 1),
        GROUP_SIZE=group_size,
        USE_M_MAJOR=m_sizes is not None or k_major is False,
        G=m_sizes.numel() if m_sizes is not None else 0,
    )
    return x_fp8.view(x_shape), x_scale


def quantize_fp8_group(
    x: torch.Tensor,
    group_size: int = 128,
    scale_ub: Optional[torch.Tensor] = None,
    m_sizes: Optional[torch.Tensor] = None,
    k_major: bool = True,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with group-wise scalings and optionally move to output device.

    Scale per group i is computed as 1 / (MAX_FP8 / max(abs(x[i:i+group_size])))

    Args:
        x (Tensor): [M, K] higher precision input tensor.
        group_size (int): Group size for M dimension of scale.
        scale_ub: Maximum allowed value for scale.
        m_sizes: Optional input for grouped gemm to specify the number of rows in each group.
        k_major (bool): Whether output scales should be K major (True) or MN major (False).
        This is needed because some kernels like cutlass require a special layout for scales.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.

    Returns:
        torch.Tensor: [M, K] fp8 scaled tensor.
        torch.Tensor: [M, cdiv(K, group_size)] reciprocal scale tensor per group.
    """
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    if x.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        xq, x_scale = triton_quantize_fp8_group(
            x, group_size, scale_ub, m_sizes, k_major
        )
        return xq.view(x_shape), x_scale
    # else use pytorch implementation.
    if not output_device:
        output_device = x.device

    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()

    M, K = x.shape
    assert (
        K % group_size == 0
    ), "K must be divisible by group_size for cpu implementation."
    assert m_sizes is None, "m_sizes is not supported for cpu implementation."
    k_groups = triton.cdiv(K, group_size)
    # View input as colleciton of groups for reduction.
    x_grouped = x.view(M, k_groups, group_size).to(torch.float32)
    # Reduce over groups.
    group_max = x_grouped.abs().amax(dim=2)
    # Apply clamping.
    group_max = (
        torch.clamp(group_max, min=eps, max=scale_ub.item())
        if scale_ub
        else torch.clamp(group_max, min=eps)
    )
    x_scale = torch.empty((M, k_groups), dtype=torch.float32, device=output_device)
    x_scale = max_fp8 / group_max  # pyre-ignore
    # pyre-ignore[16]: Undefined attribute [16]
    x_scale[x_scale == float("inf")] = 1.0
    # pyre-ignore[16]: Undefined attribute [16]
    x_fp8 = x.view(-1, k_groups, group_size) * x_scale.unsqueeze(2)
    # Cast and move data to output device (for cpu weight loading).
    x_fp8 = x_fp8.to(device=output_device, dtype=pt_dtype)
    x_scale = x_scale.to(output_device)  # pyre-ignore
    if not k_major:
        x_scale = x_scale.t().contiguous()
    return x_fp8.view(x_shape), 1 / x_scale  # pyre-ignore


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


# Force a failure instead of a warning when all configs are pruned.
# TODO: Determine a better approach for model level testing. We need
# to standardize our approach around prune_configs in general.
FORCE_FAILURE_ON_EMPTY_CONFIGS = False


def is_invalid_config(config, N, M, K, mfma, use_bias):
    """
    Contains all of the configuration checks for prune_configs
    that will result in an invalid result if select as the config.

    This is done to ensure that if no config is "optimal" for a given
    shape we don't accidentally select
    """
    BLOCK_SIZE_M = config.kwargs.get("BLOCK_M")
    BLOCK_SIZE_N = config.kwargs.get("BLOCK_N")
    BLOCK_SIZE_K = config.kwargs.get("BLOCK_K")
    SPLIT_K = config.kwargs.get("SPLIT_K")
    matrix_instr_nonkdim = config.kwargs.get("matrix_instr_nonkdim")
    if matrix_instr_nonkdim > mfma:
        return True
    if mfma == 4 and BLOCK_SIZE_K < 64:
        return True
    # some layouts could not work properly in case
    # number elements per thread is less 1
    if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
        return True
    if BLOCK_SIZE_M < matrix_instr_nonkdim or BLOCK_SIZE_N < matrix_instr_nonkdim:
        return True
    if M <= matrix_instr_nonkdim and BLOCK_SIZE_M != matrix_instr_nonkdim:
        return True
    if N <= matrix_instr_nonkdim and BLOCK_SIZE_N != matrix_instr_nonkdim:
        return True
    # split_k cannot be used if there is a bias
    if use_bias and SPLIT_K != 1:
        return True
    return False


# Configs adapted from https://github.com/ROCm/triton/blob/main_perf/python/perf-kernels/tools/tune_gemm/tune_gemm.py
def prune_configs(configs, named_args, **kwargs):

    pruned_configs = []
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    elemBytes_a = named_args["A"].element_size()
    elemBytes_b = named_args["B"].element_size()
    use_bias = kwargs["USE_BIAS"]

    if M < 32 or N < 32:
        mfma = 16
    else:
        mfma = 32

    for config in configs:
        BLOCK_SIZE_M = config.kwargs.get("BLOCK_M")
        BLOCK_SIZE_N = config.kwargs.get("BLOCK_N")
        BLOCK_SIZE_K = config.kwargs.get("BLOCK_K")
        SPLIT_K = config.kwargs.get("SPLIT_K")
        GROUP_M = config.kwargs.get("GROUP_M")
        if is_invalid_config(config, N, M, K, mfma, use_bias):
            continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if BLOCK_SIZE_M > M * 2 and BLOCK_SIZE_M != 16:
            continue
        if BLOCK_SIZE_N > N * 2 and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M >= M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = (
            BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a
            + BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b
        )
        if LDS > 65536:
            continue
        pruned_configs.append(config)

    print(f"{len(configs)=} {len(pruned_configs)=} for {M=} {N=} {K=}")
    if len(pruned_configs) == 0:
        if not FORCE_FAILURE_ON_EMPTY_CONFIGS:
            # Prune configs that can lead to incorrect results even if all configs are sub-optimal.
            candidate_configs = [
                c for c in configs if not is_invalid_config(c, N, M, K, mfma, use_bias)
            ]
            print(f"No configs left after pruning! {M=} {N=} {K=}")
            pruned_configs = candidate_configs[:10]
        if len(pruned_configs) == 0:
            raise RuntimeError(
                "No valid configs left after pruning! Consider autotuning further with TritonBench"
            )
    return pruned_configs


def get_full_non_persistent_tuning_space():
    configs = []

    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    split_k_range = [1]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 2, 4, 8, 16, 32]
    num_stage_range = [2]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16, 32]
    kpack_range = [1, 2]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for split_k in split_k_range:
                            for num_stages in num_stage_range:
                                for waves_per_eu in waves_per_eu_range:
                                    for (
                                        matrix_instr_nonkdim
                                    ) in matrix_instr_nonkdim_range:
                                        for kpack in kpack_range:
                                            configs.append(
                                                triton.Config(
                                                    {
                                                        "BLOCK_M": block_m,
                                                        "BLOCK_N": block_n,
                                                        "BLOCK_K": block_k,
                                                        "GROUP_M": group_m,
                                                        "SPLIT_K": split_k,
                                                        "waves_per_eu": waves_per_eu,
                                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                                        "kpack": kpack,
                                                    },
                                                    num_warps=num_warps,
                                                    num_stages=num_stages,
                                                )
                                            )
    logger.info(f"all configs #: {len(configs)}")
    return configs


MATMUL_CONFIGS_NON_PERSISTENT: List[Config] = get_full_non_persistent_tuning_space()
MATMUL_CONFIGS_NON_PERSISTENT_PINGPONG_4K_8K_16K = [
    triton.Config(
        {
            "BLOCK_M": 16,
            "BLOCK_N": 16,
            "BLOCK_K": 256,
            "GROUP_M": 1,
            "SPLIT_K": 1,
            "waves_per_eu": 8,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 16,
            "BLOCK_N": 16,
            "BLOCK_K": 256,
            "GROUP_M": 1,
            "SPLIT_K": 1,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 32,
            "BLOCK_N": 64,
            "BLOCK_K": 512,
            "GROUP_M": 1,
            "SPLIT_K": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 256,
            "GROUP_M": 1,
            "SPLIT_K": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 32,
            "SPLIT_K": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        },
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 2,
            "SPLIT_K": 1,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 32,
            "kpack": 2,
        },
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 1,
            "SPLIT_K": 1,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 32,
            "kpack": 2,
        },
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 4,
            "SPLIT_K": 1,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        },
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 1,
            "SPLIT_K": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 256,
            "GROUP_M": 1,
            "SPLIT_K": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 4,
            "SPLIT_K": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 1,
            "SPLIT_K": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_M": 4,
            "SPLIT_K": 1,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_M": 1,
            "SPLIT_K": 1,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 1,
            "SPLIT_K": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        },
        num_warps=8,
        num_stages=2,
    ),
]

# Set this to enable full autotuning for proper benchmarking.
# This should only be used when invoking the kernel through
# Triton directly (e.g. TritonBench)
#
# NOTE: This will SIGNIFICANTLY increase autotuning time, often
# taking hours. You should combine this with TRITON_PRINT_AUTOTUNING=1
# to extract and add the optimal autotuning configs to
# MATMUL_CONFIGS_NON_PERSISTENT_PINGPONG_4K_8K_16K.

FULL_NON_PERSISTENT_AUTOTUNING = False
USED_MATMUL_NON_PERSISTENT_CONFIGS = (
    MATMUL_CONFIGS_NON_PERSISTENT
    if FULL_NON_PERSISTENT_AUTOTUNING
    else MATMUL_CONFIGS_NON_PERSISTENT_PINGPONG_4K_8K_16K
)


@triton.autotune(
    configs=USED_MATMUL_NON_PERSISTENT_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": prune_configs,
        "perf_model": None,
        "top_k": None,
    },
    use_cuda_graph=FULL_NON_PERSISTENT_AUTOTUNING,
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row_non_persistent(
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
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        USE_BIAS (bool): Whether to use bias.
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    tl.assume(M >= 0)
    tl.assume(N >= 0)
    tl.assume(K >= 0)
    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    # Matrix multiplication.
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # Re-order program ID for better L2 performance (swizzle).
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + ((pid % width) % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    # Do matrix multiplication.
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # Pointers.
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc_dtype = tl.float32 if allow_tf32 else dot_out_dtype
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

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
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, allow_tf32=allow_tf32)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Invert scaling.
    a_scale = tl.load(A_scale + rm, mask=rm < M)
    b_scale = tl.load(B_scale + rn, mask=rn < N)
    # Invert vector, then multiply on matrix for speed.
    # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
    scale = a_scale[:, None] * b_scale[None, :]
    acc *= scale

    # Load and add bias if specified.
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
    configs=[Config({"BLOCK_M": 16, "BLOCK_K": 512, "NUM_STAGES": 2})],
    key=["M", "K"],
)
@triton.jit
def _kernel_dequantize_fp8_row(
    xq_ptr,
    x_scale_ptr,
    x_dequant_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_xdqm,
    stride_xdqk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    """
    Kernel to dequantize FP8 tensor to BF16 tensor.
    Args:
        xq_ptr (tl.constexpr): Pointer to FP8 tensor.
        x_scale_ptr (tl.constexpr): Pointer to FP8 scale tensor.
        x_dequant_ptr (tl.constexpr): Pointer to BF16 tensor.
        M (tl.constexpr): M dimension of input tensor.
        K (tl.constexpr): K dimension of input tensor (along which scales are applied)
        BLOCK_SIZE (tl.constexpr): Block size for the K dimension.
    """
    pid = tl.program_id(axis=0)
    if USE_INT64:
        pid = pid.to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    scales = tl.load(x_scale_ptr + offs_m)

    for _k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        xq = tl.load(
            xq_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask,
        )
        x_dq = xq * scales[:, None]
        tl.store(
            x_dequant_ptr
            + offs_m[:, None] * stride_xdqm
            + offs_k[None, :] * stride_xdqk,
            x_dq,
            mask=mask,
        )
        offs_k += BLOCK_K


def dequantize_fp8_row(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Rowwise Dequantize FP8 tensor to BF16 tensor along last axis.

    Args:
        xq (torch.Tensor): FP8 tensor to be dequantized.
        x_scale (torch.Tensor): FP8 scale tensor.

    Returns:
        torch.Tensor: Dequantized BF16 tensor.
    """

    assert (
        xq.is_contiguous() and x_scale.is_contiguous()
    ), "Input tensors must be contiguous"
    x_dequant = torch.empty_like(xq, dtype=torch.bfloat16)

    # Reshape to 2-d array keeping last dim only.
    K = xq.shape[-1]
    xq = xq.reshape(-1, K)
    M = xq.shape[0]
    use_int64 = xq.numel() > 2**31

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    with torch.cuda.device(xq.device.index):
        _kernel_dequantize_fp8_row[grid](
            xq,
            x_scale,
            x_dequant,
            M,
            K,
            xq.stride(0),
            xq.stride(1),
            xq.stride(0),  # Use squashed stride.
            xq.stride(1),
            USE_INT64=use_int64,
        )
    return x_dequant


@triton.autotune(
    configs=[Config({"BLOCK_M": 16, "BLOCK_K": 512, "NUM_STAGES": 2})],
    key=["M", "K"],
)
@triton.jit
def _kernel_dequantize_fp8_packed_row(
    xq_ptr,
    x_scale_ptr,
    x_dequant_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_xdqm,
    stride_xdqk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    """
    Kernel to dequantize FP8 tensor to BF16 tensor.
    Args:
        xq_ptr (tl.constexpr): Pointer to FP8 tensor.
        x_scale_ptr (tl.constexpr): Pointer to FP8 scale tensor.
        x_dequant_ptr (tl.constexpr): Pointer to BF16 tensor.
        M (tl.constexpr): M dimension of input tensor.
        K (tl.constexpr): K dimension of input tensor (along which scales are applied)
        BLOCK_SIZE (tl.constexpr): Block size for the K dimension.
    """
    pid = tl.program_id(axis=0)
    if USE_INT64:
        pid = pid.to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    scales = tl.load(x_scale_ptr + offs_m)

    for _k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

        xq = tl.load(
            xq_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask,
            other=0.0,
        )
        x_dq = xq * scales[:, None]

        tl.store(
            x_dequant_ptr
            + offs_m[:, None] * stride_xdqm
            + offs_k[None, :] * stride_xdqk,
            x_dq,
            mask=mask,
        )
        offs_k += BLOCK_K


def dequantize_fp8_packed_row(
    xq: torch.Tensor,
) -> torch.Tensor:
    """
    Rowwise Dequantize FP8 tensor to BF16 tensor along last axis.

    Args:
        xq (torch.Tensor): Packed FP8 tensor to be dequantized. The last 4 bytes of each row is the FP32 scale for that row.

    Returns:
        torch.Tensor: Dequantized BF16 tensor.
    """

    # Create a view of the packed tensors, get the scale and actual xq tensor
    # This makes it much easier to write the kernel
    orig_shape = (*xq.shape[:-1], xq.shape[-1] - 4)
    actual_xq = xq[..., :-4].view(orig_shape)

    assert xq.is_contiguous(), "Input tensors must be contiguous"
    x_dequant = torch.empty(orig_shape, dtype=torch.bfloat16, device=xq.device)

    # Calculate number of rows when flattened
    num_rows = actual_xq.numel() // actual_xq.shape[-1]

    # TODO: we take a perf hit from these reshapes, can we do better?
    # It's hard to skip this reshape, we can't create a int32/float32 view because of alignment issues
    scale_view = xq[..., -4:].reshape((num_rows * 4)).view(torch.float32)
    scale_view = scale_view.view(orig_shape[:-1])

    # Reshape to 2-d array keeping last dim only.
    K = actual_xq.shape[-1]
    actual_xq = actual_xq.reshape(-1, K)
    M = actual_xq.shape[0]
    use_int64 = actual_xq.numel() > 2**31

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    with torch.cuda.device(actual_xq.device.index):
        _kernel_dequantize_fp8_packed_row[grid](
            actual_xq,
            scale_view,
            x_dequant,
            M,
            K,
            actual_xq.stride(0),
            actual_xq.stride(1),
            x_dequant.stride(-2),  # Use squashed stride.
            x_dequant.stride(-1),
            USE_INT64=use_int64,
        )

    return x_dequant


@triton.jit
def _kernel_dequantize_fp8_block(
    xq_ptr,
    x_scale_ptr,
    x_dequant_ptr,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Kernel to dequantize FP8 tensor to BF16 tensor.
    Args:
        xq_ptr (tl.constexpr): Pointer to FP8 tensor.
        x_scale_ptr (tl.constexpr): Pointer to FP8 scale tensor.
        x_dequant_ptr (tl.constexpr): Pointer to BF16 tensor.
        M (tl.constexpr): M dimension of input tensor.
        K (tl.constexpr): K dimension of input tensor.
        BLOCK_M (tl.constexpr): Block size for the M dimension.
        BLOCK_K (tl.constexpr): Block size for the K dimension.
    """
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_K)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs = offs_m[:, None] * K + offs_k[None, :]
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    xq = tl.load(xq_ptr + offs, mask=mask).to(tl.bfloat16)
    x_scale = tl.load(x_scale_ptr + pid_m * k + pid_k)
    x_dequant = xq * x_scale
    tl.store(x_dequant_ptr + offs, x_dequant, mask=mask)


def dequantize_fp8_block(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
) -> torch.Tensor:
    """
    Dequantize FP8 tensor to BF16 tensor.

    Args:
        xq (torch.Tensor): FP8 tensor to be dequantized.
        x_scale (torch.Tensor): FP8 scale tensor.
        block_m (int): Block size for the M dimension.
        block_k (int): Block size for the K dimension.

    Returns:
        torch.Tensor: Dequantized BF16 tensor.
    """

    assert (
        xq.is_contiguous() and x_scale.is_contiguous()
    ), "Input tensors must be contiguous"
    assert xq.dim() == 2 and x_scale.dim() == 2, "Input tensors must have 2 dimensions"
    M, K = xq.size()
    x_dequant = torch.empty_like(xq, dtype=torch.bfloat16)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(K, meta["BLOCK_K"]),
        )

    with torch.cuda.device(xq.device.index):
        _kernel_dequantize_fp8_block[grid](
            xq, x_scale, x_dequant, M, K, BLOCK_M=block_m, BLOCK_K=block_k  # pyre-ignore[6]
        )
    return x_dequant


# This function is extracted from https://github.com/pytorch/ao/blob/v0.12.0/torchao/prototype/mx_formats/mx_tensor.py#L142
def to_mxfp8(
    data_hp: torch.Tensor,
    block_size: int = 32,
):
    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    assert (
        data_hp.shape[-1] % block_size == 0
    ), f"the last dimension of shape {data_hp.shape} must be divisible by block_size {block_size}"
    assert data_hp.is_contiguous(), "unsupported"

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(
        *orig_shape[:-1], orig_shape[-1] // block_size, block_size
    )

    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)

    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    max_pos = F8E4M3_MAX

    # RCEIL
    def _to_mx_rceil(
        data_hp: torch.Tensor,
        max_abs: torch.Tensor,
        max_pos: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E8M0_EXPONENT_BIAS = 127
        descale = max_abs / max_pos
        exponent = torch.where(
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
            exponent == 0,
            1.0,
            torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32)),
        )

        # scale and saturated cast the data elements to max of target dtype
        data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_pos, max=max_pos)
        return exponent, data_lp

    scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)

    # cast to target dtype
    data_lp = data_lp.to(torch.float8_e4m3fn)
    # need to reshape at the end to help inductor fuse things
    data_lp = data_lp.reshape(orig_shape)

    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    scale_e8m0_biased = scale_e8m0_biased.squeeze(-1)
    return scale_e8m0_biased, data_lp
