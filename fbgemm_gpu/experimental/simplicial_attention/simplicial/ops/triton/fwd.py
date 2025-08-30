# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import triton
import triton.language as tl
from triton import Config

SIMPLICIAL_AUTOTUNE = os.getenv("SIMPLICIAL_AUTOTUNE", "0") == "1"


def get_configs():
    if SIMPLICIAL_AUTOTUNE:
        return [
            Config(
                {
                    "BLOCK_SIZE_KV": BLOCK_SIZE_KV,
                    "num_stages_0": num_stages_0,
                    "num_stages_1": num_stages_1,
                },
                num_warps=num_warps,
            )
            for BLOCK_SIZE_KV in [32, 64, 128, 256]
            for num_stages_0 in [8, 16, 32]
            for num_stages_1 in [1, 2, 3, 4]
            for num_warps in [4, 8]
        ]
    return [
        Config(
            {
                "BLOCK_SIZE_KV": BLOCK_SIZE_KV,
                "num_stages_0": num_stages_0,
                "num_stages_1": num_stages_1,
            },
            num_warps=num_warps,
        )
        for BLOCK_SIZE_KV in [64]
        for num_stages_0 in [16]
        for num_stages_1 in [1]
        for num_warps in [4]
    ]


@triton.jit
def _gqa_pack_fwd_inner(
    K2_ptr,
    V2_ptr,
    k2_stride_s,
    k2_stride_h,
    v2_stride_s,
    v2_stride_h,
    qkv_offs_h,
    q_idx,
    kv2_idx,
    qk1,
    v1_tile,
    acc,
    m_i,
    l_i,
    gemm_dtype: tl.constexpr,
    w2: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    IS_MASK_LOAD: tl.constexpr,
):
    # [BLOCK_SIZE_KV, HEAD_DIM]
    kv2_offs_s = kv2_idx + tl.arange(0, BLOCK_SIZE_KV)
    k2_offs = kv2_offs_s[None, :] * k2_stride_s + qkv_offs_h[:, None] * k2_stride_h
    v2_offs = kv2_offs_s[:, None] * v2_stride_s + qkv_offs_h[None, :] * v2_stride_h

    if IS_MASK_LOAD:
        kv2_mask_s = (q_idx - w2 < kv2_offs_s) and (kv2_offs_s <= q_idx)
        k2t_tile = tl.load(
            K2_ptr + k2_offs, mask=kv2_mask_s[None, :]
        )  # [HEAD_DIM, BLOCK_SIZE_KV]
        v2_tile = tl.load(
            V2_ptr + v2_offs, mask=kv2_mask_s[:, None]
        )  # [BLOCK_SIZE_KV, HEAD_DIM]
    else:
        kv2_mask_s = None
        k2t_tile = tl.load(K2_ptr + k2_offs)  # [HEAD_DIM, BLOCK_SIZE_KV]
        v2_tile = tl.load(V2_ptr + v2_offs)  # [BLOCK_SIZE_KV, HEAD_DIM]

    # k2 @ qk1.T: [kv2, d] @ [d, q] -> [kv2, q]
    # qkkT [kv2, q]
    qk = tl.dot(
        qk1,  # * softmax_scale,
        k2t_tile,
        input_precision="tf32",  # INPUT_PRECISION,
        out_dtype=tl.float32,
    )  # [BLOCK_SIZE_Q, BLOCK_SIZE_KV]

    # Mask for q_idx - w1 < kv1_idx <= q_idx
    # and q_idx - w2 < kv2_offs_s <= q_idx

    if IS_MASK_LOAD:
        qk_mask = kv2_mask_s[None, :]
        # TODO Triton nan's with -inf, but float max is probably sufficient.
        qk += tl.where(qk_mask, 0, -1.0e6)

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # v12T[d, kv2] @ pT[kv2, q]: accT[d, q]
    pv2 = tl.dot(
        p.to(gemm_dtype),
        v2_tile,
        input_precision="ieee",  # INPUT_PRECISION,
        out_dtype=tl.float32,
    )

    pv12 = pv2 * v1_tile  # [BLOCK_SIZE_KV, HEAD_DIM]

    acc += pv12

    m_i = m_ij

    return acc, m_i, l_i


# Without TMA.
@triton.autotune(
    configs=get_configs(),
    key=["HEAD_DIM", "w1", "w2", "seq_len"],
)
@triton.jit
def _gqa_pack_fwd_kernel(
    Q_ptr,  # [b, s, k, h]
    K1_ptr,  # [b, s, 1, h]
    K2_ptr,  # [b, s, 1, h]
    V1_ptr,  # [b, s, 1, h]
    V2_ptr,  # [b, s, 1, h]
    O_ptr,  # [b, s, k, h]
    M_ptr,  # [b, k, s]
    bs,
    seq_len,
    num_heads,
    w1: tl.constexpr,
    w2: tl.constexpr,
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
    out_stride_b,
    out_stride_s,
    out_stride_k,
    out_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    SM_SCALE: tl.constexpr,
    K2_BIAS: tl.constexpr,
    V2_BIAS: tl.constexpr,
    num_stages_0: tl.constexpr,
    num_stages_1: tl.constexpr,
):
    """GQA two simplicial attention fwd kernel. Assume TP=num_kv so kernel is called per kv_group."""
    data_dtype = tl.bfloat16
    compute_dtype = tl.float32
    gemm_dtype = tl.bfloat16

    q_idx = tl.program_id(0)
    offs_b = tl.program_id(1)

    q_offs_b = offs_b * q_stride_b
    kv_offs_b = offs_b * k1_stride_b

    Q_ptr += q_offs_b + q_idx * q_stride_s
    K1_ptr += kv_offs_b
    K2_ptr += kv_offs_b
    V1_ptr += kv_offs_b
    V2_ptr += kv_offs_b
    O_ptr += offs_b * out_stride_b + q_idx * out_stride_s
    M_ptr += offs_b * m_stride_b + q_idx * m_stride_s

    m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=compute_dtype) - float("inf")
    # TODO why does triton impl initialize this as 1, but paper uses 0.
    l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=compute_dtype)
    acc = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=compute_dtype)

    qkv_offs_h = tl.arange(0, HEAD_DIM)
    q_offs_k = tl.arange(0, BLOCK_SIZE_Q)
    q_offs = q_offs_k[:, None] * q_stride_k + qkv_offs_h[None, :] * q_stride_h

    q_tile = tl.load(Q_ptr + q_offs)  # [BLOCK_SIZE_Q, HEAD_DIM]
    softmax_scale = tl.cast(SM_SCALE, gemm_dtype)

    q_tile = q_tile * softmax_scale

    kv2_start = tl.maximum(0, q_idx - w2 + 1)
    kv2_end = tl.minimum(seq_len, q_idx + 1)
    has_mask_load = (kv2_end - kv2_start) % BLOCK_SIZE_KV > 0
    num_n_trips = tl.cdiv(kv2_end - kv2_start, BLOCK_SIZE_KV)
    num_n_trips_inner = num_n_trips - 1 if has_mask_load else num_n_trips

    kv1_start = tl.maximum(0, q_idx - w1 + 1)
    kv1_end = tl.minimum(seq_len, q_idx + 1)
    for kv1_idx in tl.range(kv1_start, kv1_end, num_stages=num_stages_0):
        k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h
        k1_tile = tl.load(K1_ptr + k1_offs)[None, :]  # [1, HEAD_DIM]
        qk1_tile = q_tile * k1_tile  # [BLOCK_SIZE_Q, HEAD_DIM]

        v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h
        v1_tile = tl.load(V1_ptr + v1_offs)[None,]  # [1, HEAD_DIM]

        for _inner_idx in tl.range(
            num_n_trips_inner,
            num_stages=num_stages_1,
        ):
            kv2_idx = kv2_start + _inner_idx * BLOCK_SIZE_KV
            acc, m_i, l_i = _gqa_pack_fwd_inner(
                K2_ptr,
                V2_ptr,
                k2_stride_s,
                k2_stride_h,
                v2_stride_s,
                v2_stride_h,
                qkv_offs_h,
                q_idx,
                kv2_idx,
                qk1_tile,
                v1_tile,
                acc,
                m_i,
                l_i,
                gemm_dtype,
                w2,
                BLOCK_SIZE_KV,
                IS_MASK_LOAD=False,
            )
        if has_mask_load:
            kv2_idx = kv2_start + num_n_trips_inner * BLOCK_SIZE_KV
            acc, m_i, l_i = _gqa_pack_fwd_inner(
                K2_ptr,
                V2_ptr,
                k2_stride_s,
                k2_stride_h,
                v2_stride_s,
                v2_stride_h,
                qkv_offs_h,
                q_idx,
                kv2_idx,
                qk1_tile,
                v1_tile,
                acc,
                m_i,
                l_i,
                gemm_dtype,
                w2,
                BLOCK_SIZE_KV,
                IS_MASK_LOAD=True,
            )

    acc = acc / l_i[:, None]
    acc = acc.to(data_dtype)
    out_offs = q_offs_k[:, None] * out_stride_k + qkv_offs_h[None, :] * out_stride_h
    tl.store(O_ptr + out_offs, acc)

    m = m_i + tl.log(l_i)

    m_offs = q_offs_k * m_stride_k
    tl.store(M_ptr + m_offs, m)


def triton_fwd(q, k1, k2, v1, v2, w1, w2, k2_bias=None, v2_bias=None):
    """2 Simplicial attention kernel with GQA packing.
    L = q @ k1 X k2
    P = softmax(L, axis=[-1, -2])
    O = P @ v1 X v2
    """
    bs, seq_len, num_heads, head_dim = q.shape
    _, seq_len1, _, _ = k1.shape
    _, seq_len2, _, _ = k2.shape
    assert (
        seq_len == seq_len1 and seq_len1 == seq_len2
    ), "input seq lens must match, sliding window is done within kernel"
    assert w1 > 0 and w2 > 0, "block local windows must be positive"
    output = torch.zeros_like(q, memory_format=torch.contiguous_format).to(
        torch.bfloat16
    )
    m = torch.zeros((bs, num_heads, seq_len), dtype=torch.float32, device=q.device)
    # INPUT_PRECISION = "ieee"
    INPUT_PRECISION = "tf32"
    # e^x = 2^(x * log2(e)), so we multiply x by log2(e) to use faster exp2 in kernel.
    sm_scale = 1.44269504  # math.log2(math.exp(1))
    sm_scale *= head_dim**-0.5
    if not k2_bias:
        k2_bias = 1.0 / head_dim
    if not v2_bias:
        v2_bias = 1.0

    # NOTE: to optimize performance, we always make sure w1 is the smaller window size, when
    # w1 and w2 are not equal.
    if w1 > w2:
        k1, k2 = k2, k1
        v1, v2 = v2, v1
        w1, w2 = w2, w1

    grid = lambda args: (seq_len, bs)  # noqa: E731

    _gqa_pack_fwd_kernel[grid](
        q,
        k1,
        k2,
        v1,
        v2,
        output,
        m,
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
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        m.stride(0),
        m.stride(1),
        m.stride(2),
        HEAD_DIM=head_dim,
        INPUT_PRECISION=INPUT_PRECISION,
        SM_SCALE=sm_scale,
        K2_BIAS=k2_bias,
        V2_BIAS=v2_bias,
        BLOCK_SIZE_Q=num_heads,
    )
    return output, m
