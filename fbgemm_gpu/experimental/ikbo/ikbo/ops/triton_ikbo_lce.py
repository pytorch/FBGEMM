# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton fused IKBO (In-Kernel Broadcast Optimization) LCE kernel.

import os

import torch
import triton
import triton.language as tl

from .tlx_ikbo_lce import swizzle_tile


def _triton_autotune_configs():
    if os.environ.get("FAST_TUNE"):
        return [
            triton.Config(
                {"BM": 128, "BN": 128, "BK": 64, "GROUP_SIZE_M": 8},
                num_stages=3,
                num_warps=4,
            )
        ]
    return [
        triton.Config(
            {"BM": bm, "BN": bn, "BK": bk, "GROUP_SIZE_M": 8},
            num_stages=ns,
            num_warps=nw,
        )
        for bm in [64, 128]
        for bn in [64, 128]
        for bk in [64, 128]
        for ns in [3, 4, 5, 6]
        for nw in [4, 8]
    ]


@triton.autotune(
    configs=_triton_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def triton_ikbo_lce_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [B, K, N]
    cand_to_user_index_ptr,  # [B]
    user_res_ptr,  # [num_users, M, N]
    out_ptr,  # [B, M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    a_stride_a: tl.constexpr,
    a_stride_b: tl.constexpr,
    b_stride_a: tl.constexpr,
    b_stride_b: tl.constexpr,
    b_stride_c: tl.constexpr,
    out_stride_a: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_c: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused IKBO LCE kernel.

    Computes the candidate GEMM and adds pre-computed user results:
        out[b] = W_cand @ E_cand[b] + user_res[u(b)]

    Grid: one program per (batch, M-tile, N-tile) with grouped ordering
    for L2 cache locality.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pids_per_batch = num_pid_m * num_pid_n

    pid_batch, pid_m, pid_n = swizzle_tile(
        pid, num_pids_per_batch, num_pid_m, num_pid_n, GROUP_SIZE_M
    )

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    ram = offs_m % M
    rbn = offs_n % N
    rk = tl.arange(0, BK)

    a_ptrs = a_ptr + (ram[:, None] * a_stride_a + rk[None, :] * a_stride_b)
    b_ptrs = (
        b_ptr
        + pid_batch * b_stride_a
        + rk[:, None] * b_stride_b
        + rbn[None, :] * b_stride_c
    )

    # Candidate GEMM: W_cand[BM, BK] @ E_cand[BK, BN] -> acc[BM, BN]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(K, 0, -BK):
        a = tl.load(a_ptrs, mask=rk[None, :] < k, other=0.0)  # [BM, BK]
        b = tl.load(b_ptrs, mask=rk[:, None] < k, other=0.0)  # [BK, BN]

        acc += tl.dot(a, b)
        a_ptrs += BK * a_stride_b
        b_ptrs += BK * b_stride_b

    # In-batch broadcast: look up pre-computed user_res[u(b)] and add
    user_idx = tl.load(cand_to_user_index_ptr + pid_batch, eviction_policy="evict_last")
    idx_m = offs_m[:, None]
    idx_n = offs_n[None, :]
    mask = (idx_m < M) & (idx_n < N)
    in_batch_offset = out_stride_c * idx_n + out_stride_b * idx_m
    user_res_val = tl.load(
        user_res_ptr + (in_batch_offset + out_stride_a * user_idx),
        mask,
        eviction_policy="evict_last",
    )  # [BM, BN]

    # Store fused result
    output_val = acc.to(a_ptr.dtype.element_ty) + user_res_val
    out_ptrs = out_ptr + in_batch_offset + out_stride_a * pid_batch
    tl.store(out_ptrs, output_val, mask)


def triton_ikbo_lce(
    compression_w_cand: torch.Tensor,
    compression_w_user: torch.Tensor,
    embeddings_cand: torch.Tensor,
    embeddings_user: torch.Tensor,
    cand_to_user_index: torch.Tensor,
) -> torch.Tensor:
    """Fused IKBO LCE: candidate GEMM + in-batch user broadcast in one kernel.

    Two-phase computation:
      1. User: user_res = W_user @ E_user           [num_users, M, N]
      2. Final: out[b] = W_cand @ E_cand[b] + user_res[u(b)]   [B, M, N]

    The kernel reads user_res via cand_to_user_index for in-batch broadcast,
    avoiding the memory cost of materializing the full broadcast tensor.

    Args:
        compression_w_cand: Candidate compression weights [M, K_cand].
        compression_w_user: User compression weights [M, K_user].
        embeddings_cand: Candidate embeddings [B, K_cand, N].
        embeddings_user: User embeddings [num_users, K_user, N].
        cand_to_user_index: Maps candidate index to user index [B], int32.
    """
    B, K, N = embeddings_cand.shape
    M = compression_w_cand.size(0)
    user_res = compression_w_user @ embeddings_user  # [num_users, M, N]
    out = torch.empty(
        (B, M, N), device=embeddings_cand.device, dtype=embeddings_cand.dtype
    )

    grid = lambda META: (B * triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),)  # noqa: E731

    triton_ikbo_lce_kernel[grid](
        compression_w_cand,
        embeddings_cand,
        cand_to_user_index,
        user_res,
        out,
        M,
        N,
        K,
        compression_w_cand.stride(0),
        compression_w_cand.stride(1),
        embeddings_cand.stride(0),
        embeddings_cand.stride(1),
        embeddings_cand.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )

    return out
