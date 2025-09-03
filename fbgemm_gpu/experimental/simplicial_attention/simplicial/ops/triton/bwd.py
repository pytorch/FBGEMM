# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os

import torch

import triton  # @manual  # @manual

import triton.language as tl  # @manual  # @manual

from triton import Config

SIMPLICIAL_AUTOTUNE = os.getenv("SIMPLICIAL_AUTOTUNE", "0") == "1"


def get_configs():
    if SIMPLICIAL_AUTOTUNE:
        return [
            Config(
                {
                    "BLOCK_SIZE_Q": BLOCK_SIZE_Q,
                    "BLOCK_SIZE_KV": BLOCK_SIZE_KV,
                    "num_stages": num_stages,
                },
                num_warps=num_warps,
            )
            for BLOCK_SIZE_Q in [32, 64, 128, 256]
            for BLOCK_SIZE_KV in [32, 64, 128, 256]
            for num_stages in [1, 2, 3, 4]
            for num_warps in [4, 8]
        ]
    return [
        Config(
            {
                "BLOCK_SIZE_Q": BLOCK_SIZE_Q,
                "BLOCK_SIZE_KV": BLOCK_SIZE_KV,
                "num_stages": num_stages,
            },
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [32]
        for BLOCK_SIZE_KV in [64]
        for num_stages in [3]
        for num_warps in [4]
    ]


# kv1 bwd kernel with tiles [kv1, q, 1]
@triton.autotune(
    configs=get_configs(),
    key=[
        "HEAD_DIM",
        "seq_len",
        "w1",
        "w2",
        "is_flipped",
    ],
)
@triton.jit
def simplicial_bwd_kv1_kernel(
    Q_ptr,  # [b, s, k, h]
    K1_ptr,  # [b, s, k, h]
    K2_ptr,  # [b, s, k, h]
    V1_ptr,  # [b, s, k, h]
    V2_ptr,  # [b, s, k, h]
    dO_ptr,  # [b, s, k, h]
    M_ptr,  # [b, k, s]
    D_ptr,  # [b, k, s]
    dQ_ptr,  # [b, s, k, h]
    dK1_ptr,  # [b, s, k, h]
    dV1_ptr,  # [b, s, k, h]
    bs,
    seq_len,
    num_heads,
    w1,  # Q[i]: KV1(i-w1,i]
    w2,  # Q[i]: KV2(i-w2,i]
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    dO_stride_b,
    dO_stride_s,
    dO_stride_k,
    dO_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    d_stride_b,
    d_stride_k,
    d_stride_s,
    dq_stride_b,
    dq_stride_s,
    dq_stride_k,
    dq_stride_h,
    dk1_stride_b,
    dk1_stride_s,
    dk1_stride_k,
    dk1_stride_h,
    dv1_stride_b,
    dv1_stride_s,
    dv1_stride_k,
    dv1_stride_h,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    data_dtype = tl.bfloat16
    compute_dtype = tl.float32
    gemm_dtype = tl.bfloat16

    kv1_start = tl.program_id(0) * BLOCK_SIZE_KV
    kv1_end = kv1_start + BLOCK_SIZE_KV
    bk = tl.program_id(1)
    offs_b = bk // num_heads
    offs_k = bk % num_heads

    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk

    dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
    D_ptr += offs_b * d_stride_b + offs_k * d_stride_k
    dK1_ptr += offs_b * dk1_stride_b + offs_k * dk1_stride_k
    dV1_ptr += offs_b * dv1_stride_b + offs_k * dv1_stride_k

    softmax_scale = tl.cast(SM_SCALE, gemm_dtype)
    qkv_offs_h = tl.arange(0, HEAD_DIM)

    kv1_offs_s = kv1_start + tl.arange(0, BLOCK_SIZE_KV)

    k1_offs = kv1_offs_s[:, None] * k1_stride_s + qkv_offs_h[None, :] * k1_stride_h
    kv1_mask = kv1_offs_s[:, None] < seq_len
    k1_tile = tl.load(K1_ptr + k1_offs, mask=kv1_mask).to(
        gemm_dtype
    )  # [BLOCK_SIZE_KV, HEAD_DIM]
    v1_offs = kv1_offs_s[:, None] * v1_stride_s + qkv_offs_h[None, :] * v1_stride_h
    v1_tile = tl.load(V1_ptr + v1_offs, mask=kv1_mask).to(
        gemm_dtype
    )  # [BLOCK_SIZE_KV, HEAD_DIM]
    dv1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)
    dk1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)
    # for kv2_idx in tl.range(0, seq_len):
    # kv1 - w2 < kv2 <= kv1 + w1
    for kv2_idx in tl.range(
        tl.maximum(0, kv1_start - w2), tl.minimum(seq_len, kv1_end + w1)
    ):
        k2_offs = kv2_idx * k2_stride_s + qkv_offs_h * k2_stride_h
        k2_tile = (tl.load(K2_ptr + k2_offs).to(gemm_dtype))[None, :]  # [1, HEAD_DIM]
        v2_offs = kv2_idx * v2_stride_s + qkv_offs_h * v2_stride_h
        v2_tile = (tl.load(V2_ptr + v2_offs).to(gemm_dtype))[None, :]  # [1, HEAD_DIM]
        k1k2 = k1_tile * k2_tile  # [BLOCK_SIZE_KV, HEAD_DIM]
        v1v2 = v1_tile * v2_tile  # [BLOCK_SIZE_KV, HEAD_DIM]
        k1k2_scaled = k1k2 * softmax_scale
        # kv1 <= q < kv1 + w1
        # kv2 <= q < kv2 + w2
        q_start = tl.maximum(kv1_start, kv2_idx)
        q_end = tl.minimum(seq_len, tl.minimum(kv1_end + w1, kv2_idx + w2))
        # FIXME: Triton kernel compilation fails when pipelining is enabled in this specific case: P1828952934.
        # So the pipelining is disabled for now.
        for q_idx in tl.range(q_start, q_end, BLOCK_SIZE_Q, num_stages=1):
            # Load qt, m, d, dO
            q_offs_s = q_idx + tl.arange(0, BLOCK_SIZE_Q)
            q_offs = q_offs_s[None, :] * q_stride_s + qkv_offs_h[:, None] * q_stride_h
            q_mask_s = q_offs_s < seq_len
            qt_tile = tl.load(
                Q_ptr + q_offs, mask=q_mask_s[None, :]
            )  # [HEAD_DIM, BLOCK_SIZE_Q]
            m_offs = q_offs_s * m_stride_s
            m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s)[
                None, :
            ]  # [1, BLOCK_SIZE_Q]
            d_offs = q_offs_s * d_stride_s
            d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s)[
                None, :
            ]  # [1, BLOCK_SIZE_Q]
            dO_offs = (
                q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h
            )
            dO_tile = tl.load(
                dO_ptr + dO_offs, mask=q_mask_s[:, None]
            )  # [BLOCK_SIZE_Q, HEAD_DIM]

            # Compute dv1.
            # [KV, D] @ [D, Q] => [KV, Q]
            qkkT = tl.dot(
                k1k2_scaled, qt_tile, out_dtype=tl.float32
            )  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

            # Mask qkkT to -inf.
            kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_offs_s[:, None]) & (
                kv1_offs_s[:, None] <= q_offs_s[None, :]
            )
            kv2_local_mask = ((q_offs_s - w2) < kv2_idx) & (kv2_idx <= q_offs_s)
            local_mask = (
                kv1_local_mask & kv2_local_mask[None, :]
            )  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

            valid = tl.sum(local_mask) > 0
            if valid:
                pT = tl.exp(qkkT - m_tile)  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
                pT = tl.where(local_mask, pT, 0.0)
                # dv1[kv1, d] = p[kv1, q] @ dO[q, d] * v2[kv2, d]
                dOv2 = dO_tile * v2_tile  # [BLOCK_SIZE_Q, HEAD_DIM]
                dv1 += tl.dot(
                    pT.to(gemm_dtype), dOv2.to(gemm_dtype), out_dtype=tl.float32
                )  # [BLOCK_SIZE_KV, HEAD_DIM]

                # dpT[kv1, q] = v1v2[kv1, d] @ dO.T[d, q]
                dpT = tl.dot(
                    v1v2, tl.trans(dO_tile.to(gemm_dtype)), out_dtype=tl.float32
                )  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
                dsT = tl.fma(pT, dpT, -pT * d_tile)  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
                # dsT = tl.where(local_mask, dsT, 0.0)
                dsT_scaled = dsT.to(gemm_dtype) * softmax_scale
                # qk2[q, d] = qt.T[q, d] * k2[1, d]

                qt_tile_T = tl.trans(qt_tile) * k2_tile

                dk1 += tl.dot(dsT_scaled, qt_tile_T, out_dtype=tl.float32)
    dv1_offs = kv1_offs_s[:, None] * dv1_stride_s + qkv_offs_h[None, :] * dv1_stride_h
    dk1_offs = kv1_offs_s[:, None] * dk1_stride_s + qkv_offs_h[None, :] * dk1_stride_h
    tl.store(dV1_ptr + dv1_offs, dv1.to(data_dtype), mask=kv1_mask)
    tl.store(dK1_ptr + dk1_offs, dk1.to(data_dtype), mask=kv1_mask)


def simplicial_bwd_kv1_triton(
    q,
    k1,
    k2,
    v1,
    v2,
    dO,
    m,
    d,
    w1,
    w2,
    dk1,
    dv1,
    dq,
    sm_scale,
):
    """Helper function to get bwd dk1 and dv1."""
    bs, seq_len, num_heads, head_dim = q.shape
    sm_scale *= 1.0

    # if w2 > w1:
    #     # NOTE: The total number of inner loop iterations is:
    #     # Total iterations = (w1 + BLOCK_SIZE_KV) * (w2 / BLOCK_SIZE_Q)
    #     #                  = (w1 * w2) / BLOCK_SIZE_Q + (BLOCK_SIZE_KV * w2) / BLOCK_SIZE_Q
    #     # When w1 != w2, assigning the smaller window size to w2 minimizes total iterations.
    #     # This is because w2 appears in both terms, while w1 only affects the first term.
    #     w1, w2 = w2, w1
    #     k1, k2 = k2, k1
    #     v1, v2 = v2, v1

    grid = lambda args: (triton.cdiv(seq_len, args["BLOCK_SIZE_KV"]), bs * num_heads)  # noqa: E731
    simplicial_bwd_kv1_kernel[grid](
        q,
        k1,
        k2,
        v1,
        v2,
        dO,
        m,
        d,
        dq,
        dk1,
        dv1,
        bs,
        seq_len,
        num_heads,
        w1,
        w2,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k1.stride(0),
        k1.stride(1),
        k1.stride(2),
        k1.stride(3),
        k2.stride(0),
        k2.stride(1),
        k2.stride(2),
        k2.stride(3),
        v1.stride(0),
        v1.stride(1),
        v1.stride(2),
        v1.stride(3),
        v2.stride(0),
        v2.stride(1),
        v2.stride(2),
        v2.stride(3),
        dO.stride(0),
        dO.stride(1),
        dO.stride(2),
        dO.stride(3),
        m.stride(0),
        m.stride(1),
        m.stride(2),
        d.stride(0),
        d.stride(1),
        d.stride(2),
        dq.stride(0) if dq is not None else 0,
        dq.stride(1) if dq is not None else 0,
        dq.stride(2) if dq is not None else 0,
        dq.stride(3) if dq is not None else 0,
        dk1.stride(0),
        dk1.stride(1),
        dk1.stride(2),
        dk1.stride(3),
        dv1.stride(0),
        dv1.stride(1),
        dv1.stride(2),
        dv1.stride(3),
        HEAD_DIM=head_dim,
        SM_SCALE=sm_scale,
    )


# "Single" pass kv2q kernel without atomics.
# Only works for tile_KV2 == 2 * tile_q == w2
# Outer loop over q % 2 == 0. And go over KV2[q_start - w2, q_end]
# Second outer looper q % 2 == 1. Go over KV2[q_start - w2, q_end] and inplace add.
@triton.autotune(
    configs=[
        Config(
            {
                "BLOCK_SIZE_Q": 32,
                "BLOCK_SIZE_KV2": 64,
                "num_stages": 1,
            },
            num_warps=4,
        )
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def simplicial_bwd_kv2q_kernel(
    Q_ptr,  # [b, s, k, h]
    K1_ptr,  # [b, s, k, h]
    K2_ptr,  # [b, s, k, h]
    V1_ptr,  # [b, s, k, h]
    V2_ptr,  # [b, s, k, h]
    dO_ptr,  # [b, s, k, h]
    M_ptr,  # [b, k, s]
    D_ptr,  # [b, k, s]
    dQ_ptr,  # [b, s, k, h]
    dK2_ptr,  # [b, s, k, h]
    dV2_ptr,  # [b, s, k, h]
    bs,
    seq_len,
    num_heads,
    head_dim,
    w1,  # Q[i]: KV1(i-w1,i]
    w2,  # Q[i]: KV2(i-w2,i]
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    dO_stride_b,
    dO_stride_s,
    dO_stride_k,
    dO_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    d_stride_b,
    d_stride_k,
    d_stride_s,
    dq_stride_b,
    dq_stride_s,
    dq_stride_k,
    dq_stride_h,
    dk2_stride_b,
    dk2_stride_s,
    dk2_stride_k,
    dk2_stride_h,
    dv2_stride_b,
    dv2_stride_s,
    dv2_stride_k,
    dv2_stride_h,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    num_stages: tl.constexpr,
    IS_SECOND_PASS: tl.constexpr,
):
    assert BLOCK_SIZE_KV2 == BLOCK_SIZE_Q + w2
    compute_dtype = tl.float32
    gemm_dtype = tl.bfloat16

    # First pass does even tiles, second pass does odd tiles.
    q_start = tl.program_id(0) * BLOCK_SIZE_KV2
    if IS_SECOND_PASS:
        q_start += BLOCK_SIZE_Q
    q_end = q_start + BLOCK_SIZE_Q
    kv2_start = q_start - w2

    bk = tl.program_id(1)
    offs_b = bk // num_heads
    offs_k = bk % num_heads

    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk

    dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
    D_ptr += offs_b * d_stride_b + offs_k * d_stride_k
    dQ_ptr += offs_b * dq_stride_b + offs_k * dq_stride_k
    dK2_ptr += offs_b * dk2_stride_b + offs_k * dk2_stride_k
    dV2_ptr += offs_b * dv2_stride_b + offs_k * dv2_stride_k

    softmax_scale = tl.cast(SM_SCALE, gemm_dtype)
    qkv_offs_h = tl.arange(0, HEAD_DIM)
    qkv_mask_h = qkv_offs_h < head_dim

    q_offs_s = q_start + tl.arange(0, BLOCK_SIZE_Q)
    kv2_offs_s = kv2_start + tl.arange(0, BLOCK_SIZE_KV2)
    q_offs = q_offs_s[:, None] * q_stride_s + qkv_offs_h[None, :] * q_stride_h
    kv2_offs = kv2_offs_s[:, None] * k2_stride_s + qkv_offs_h[None, :] * k2_stride_h
    m_offs = q_offs_s * m_stride_s
    d_offs = q_offs_s * d_stride_s
    dO_offs = q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h
    q_mask_s = q_offs_s < seq_len
    q_mask = q_mask_s[:, None] & qkv_mask_h[None, :]
    kv2_mask_s = 0 <= kv2_offs_s and kv2_offs_s < seq_len
    kv2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]

    q_tile = tl.load(Q_ptr + q_offs, mask=q_mask).to(
        compute_dtype
    )  # [BLOCK_SIZE_Q, HEAD_DIM]
    k2_tile = tl.load(K2_ptr + kv2_offs, mask=kv2_mask).to(
        gemm_dtype
    )  # [KV2, HEAD_DIM]
    v2_tile = tl.load(V2_ptr + kv2_offs, mask=kv2_mask).to(
        gemm_dtype
    )  # [KV2, HEAD_DIM]
    m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to(gemm_dtype)  # [BLOCK_SIZE_Q]
    d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to(gemm_dtype)  # [BLOCK_SIZE_Q]
    dO_tile = tl.load(dO_ptr + dO_offs, mask=q_mask).to(
        gemm_dtype
    )  # [BLOCK_SIZE_Q, HEAD_DIM]

    dq = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), tl.float32)
    # dqT = tl.zeros((HEAD_DIM, BLOCK_SIZE_Q), tl.float32)
    dk2 = tl.zeros((BLOCK_SIZE_KV2, HEAD_DIM), tl.float32)
    dv2 = tl.zeros((BLOCK_SIZE_KV2, HEAD_DIM), tl.float32)

    kv1_start = tl.maximum(0, q_start - w1)
    kv1_end = tl.minimum(seq_len, q_end)
    for kv1_idx in tl.range(kv1_start, kv1_end, num_stages=num_stages):
        k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h
        v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h
        k1_tile = tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to(
            compute_dtype
        )  # [HEAD_DIM]

        v1_tile = tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to(
            compute_dtype
        )  # [HEAD_DIM]

        qk1_s = q_tile * (k1_tile[None, :] * softmax_scale)  # [Q, D]
        qk1_s = qk1_s.to(gemm_dtype)
        # k2[KV, Q] @ qk1_s.T[Q, D] => [KV2, Q]
        qkkT = tl.dot(k2_tile, qk1_s.T, out_dtype=tl.float32)  # [KV2, Q]

        qkT_mask = kv2_mask_s[:, None] & q_mask_s[None, :]
        kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_idx) & (
            kv1_idx <= q_offs_s[None, :]
        )  # [KV2, Q]
        kv2_local_mask = ((q_offs_s[None, :] - w2) < kv2_offs_s[:, None]) & (
            kv2_offs_s[:, None] <= q_offs_s[None, :]
        )  # [KV2, Q]
        local_mask = kv1_local_mask & kv2_local_mask  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
        qkT_mask &= kv1_local_mask & kv2_local_mask

        pT = tl.exp(qkkT - m_tile[None, :])  # [KV2, Q]
        pT = tl.where(qkT_mask, pT, 0.0)

        qkkT = tl.where(local_mask, qkkT, -1.0e38)

        dOv1 = dO_tile * v1_tile[None, :]  # [Q, D]
        dOv1 = dOv1.to(gemm_dtype)
        # pT[KV2, Q] @ dOv1[Q, D] => [KV2, D]
        dv2 += tl.dot(pT.to(gemm_dtype), dOv1, out_dtype=tl.float32)

        # v2[KV2, D] @ dOv1.T[D, Q] => dpT[KV2, Q]
        dpT = tl.dot(v2_tile, dOv1.T, out_dtype=tl.float32)
        dsT = pT * (dpT - d_tile[None, :])  # [KV2, Q]
        dsT = tl.where(qkT_mask, dsT, 0.0)
        dsT = dsT.to(gemm_dtype)  # [KV2, Q]

        # dsT[KV2, Q] @ qk1[Q, D] => dk2[KV2, D]
        dk2 += tl.dot(dsT, qk1_s, out_dtype=tl.float32)

        k1k2 = k1_tile[None, :] * k2_tile  # [KV2, D]
        k1k2 = k1k2.to(gemm_dtype)
        # k1k2T = k1_tile[:, None] * k2_tile.T
        # k1k2T = k1k2T.to(gemm_dtype)

        # Normal.
        # dsT.T[Q, KV2] @ [KV2, D] => dq[Q, D]
        dq += tl.dot(dsT.T, k1k2)  # * softmax scale at the end.

    # End. update gradients.
    if IS_SECOND_PASS:
        # load, add.
        prev_dk2 = tl.load(dK2_ptr + kv2_offs, kv2_mask)
        prev_dv2 = tl.load(dV2_ptr + kv2_offs, kv2_mask)
        dk2 += prev_dk2
        dv2 += prev_dv2

    dq *= softmax_scale
    tl.store(dK2_ptr + kv2_offs, dk2, kv2_mask)
    tl.store(dV2_ptr + kv2_offs, dv2, kv2_mask)
    tl.store(dQ_ptr + q_offs, dq, q_mask)


def simplicial_bwd_kv2q_triton(
    q,
    k1,
    k2,
    v1,
    v2,
    dO,
    m,
    d,
    w1,
    w2,
    dk2,
    dv2,
    dq,
    sm_scale,
):
    bs, seq_len, num_heads, head_dim = q.shape
    sm_scale *= 1.0  # TODO math.log2(math.exp(1)) is faster.

    # assert w2 == 32
    # TODO replace with grid assert, w2 + BLOCK_SIZE_Q == BLOCK_SIZE_KV2
    def grid(args):
        return (triton.cdiv(seq_len, args["BLOCK_SIZE_KV2"]), bs * num_heads)

    for is_second_pass in [False, True]:
        simplicial_bwd_kv2q_kernel[grid](
            q,
            k1,
            k2,
            v1,
            v2,
            dO,
            m,
            d,
            dq,
            dk2,
            dv2,
            bs,
            seq_len,
            num_heads,
            head_dim,
            w1,
            w2,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k1.stride(0),
            k1.stride(1),
            k1.stride(2),
            k1.stride(3),
            k2.stride(0),
            k2.stride(1),
            k2.stride(2),
            k2.stride(3),
            v1.stride(0),
            v1.stride(1),
            v1.stride(2),
            v1.stride(3),
            v2.stride(0),
            v2.stride(1),
            v2.stride(2),
            v2.stride(3),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dO.stride(3),
            m.stride(0),
            m.stride(1),
            m.stride(2),
            d.stride(0),
            d.stride(1),
            d.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk2.stride(0),
            dk2.stride(1),
            dk2.stride(2),
            dk2.stride(3),
            dv2.stride(0),
            dv2.stride(1),
            dv2.stride(2),
            dv2.stride(3),
            # BLOCK_SIZE_Q=block_size_q,
            # BLOCK_SIZE_KV=block_size_kv,
            HEAD_DIM=triton.next_power_of_2(head_dim),
            SM_SCALE=sm_scale,
            IS_SECOND_PASS=is_second_pass,
        )


@triton.jit
def simplicial_bwd_pre_kernel(
    O_ptr,  # [b, s, k, h]
    dO_ptr,  # [b, s, k, h]
    d_ptr,  # [b, k, s]
    bs,
    seq_len,
    num_heads,
    head_dim,
    o_stride_b,
    o_stride_s,
    o_stride_k,
    o_stride_h,
    dO_stride_b,
    dO_stride_s,
    dO_stride_k,
    dO_stride_h,
    d_stride_b,
    d_stride_k,
    d_stride_s,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,  # Expect BLOCK_SIZE_H = h
):
    offs_s = tl.program_id(0) * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)

    bk = tl.program_id(1)
    offs_b = bk // num_heads
    offs_k = bk % num_heads

    O_ptr += offs_b * o_stride_b + offs_k * o_stride_k
    dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
    d_ptr += offs_b * dO_stride_b + offs_k * d_stride_k

    offs_h = tl.arange(0, BLOCK_SIZE_H)
    mask_h = offs_h < head_dim

    o_offs = offs_s[:, None] * o_stride_s + offs_h[None, :] * o_stride_h
    dO_offs = offs_s[:, None] * dO_stride_s + offs_h[None, :] * o_stride_h
    d_offs = offs_s * d_stride_s
    mask_s = offs_s < seq_len
    mask = mask_s[:, None] & mask_h[None, :]

    o = tl.load(O_ptr + o_offs, mask=mask)  # [BLOCK_SIZE_S, BLOCK_SIZE_H]
    dO = tl.load(dO_ptr + dO_offs, mask=mask)  # [BLOCK_SIZE_S, BLOCK_SIZE_H]

    delta = tl.sum(o * dO, -1)

    tl.store(d_ptr + d_offs, delta, mask=mask_s)


def bwd_pre_triton(o, dO):
    bs, seq_len, num_heads, head_dim = o.shape
    d = torch.zeros((bs, num_heads, seq_len), dtype=o.dtype, device=o.device)

    def grid(args):
        return (triton.cdiv(seq_len, args["BLOCK_SIZE_S"]), bs * num_heads)

    # d = sum(o * dO, dim=-1)
    simplicial_bwd_pre_kernel[grid](
        o,
        dO,
        d,
        bs,
        seq_len,
        num_heads,
        head_dim,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        dO.stride(0),
        dO.stride(1),
        dO.stride(2),
        dO.stride(3),
        d.stride(0),
        d.stride(1),
        d.stride(2),
        BLOCK_SIZE_S=32,
        BLOCK_SIZE_H=triton.next_power_of_2(head_dim),
    )
    return d


def triton_bwd(q, k1, k2, v1, v2, o, dO, m, w1, w2, k2_bias=None, v2_bias=None):
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk1 = torch.zeros_like(k1)
    dk2 = torch.zeros_like(k2)
    dv1 = torch.zeros_like(v1)
    dv2 = torch.zeros_like(v2)

    bs, seq_len, num_heads, head_dim = q.shape
    sm_scale = head_dim**-0.5
    if not k2_bias:
        k2_bias = 1.0 / head_dim
    if not v2_bias:
        v2_bias = 1.0

    d = bwd_pre_triton(o, dO)

    simplicial_bwd_kv1_triton(
        q, k1, k2, v1, v2, dO, m, d, w1, w2, dk1, dv1, None, sm_scale=sm_scale
    )

    assert w2 == 32
    simplicial_bwd_kv2q_triton(
        q, k1, k2, v1, v2, dO, m, d, w1, w2, dk2, dv2, dq, sm_scale=sm_scale
    )

    return dq, dk1, dk2, dv1, dv2
