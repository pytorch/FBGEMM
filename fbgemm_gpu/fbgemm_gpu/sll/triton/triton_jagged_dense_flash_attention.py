# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
import triton
import triton.language as tl

from .common import expect_contiguous


@triton.jit
def jagged_dense_flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ab_ptr,  # attn bias ptr
    o_ptr,
    lse_ptr,
    jagged_offsets_ptr,
    max_seq_len,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kd,
    stride_kt,
    stride_vn,
    stride_vd,
    stride_ab_b,  # attn bias stride batch
    stride_ab_n,
    stride_ab_t,
    stride_ob,
    stride_ot,
    stride_od,
    D: tl.constexpr,
    T: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_batch = tl.program_id(1)

    # begin offset of the current sample
    begin = tl.load(jagged_offsets_ptr + pid_batch)
    # end offset of the current sample
    end = tl.load(jagged_offsets_ptr + pid_batch + 1)

    # The seq length of the current sample
    length = end - begin
    length = tl.minimum(length, max_seq_len)

    if length == 0:
        return

    q_start_ptr = q_ptr + begin * stride_ql
    k_start_ptr = k_ptr + pid_batch * stride_kb
    ab_start_ptr = ab_ptr + pid_batch * stride_ab_b
    v_start_ptr = v_ptr + begin * stride_vn

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Load a block of Q into [BLOCK_D, BLOCK_T]
    ki_ptrs = k_start_ptr + offs_d[:, None] * stride_kd + offs_t[None, :] * stride_kt

    ki = tl.load(
        ki_ptrs,
        mask=((offs_d[:, None] < D) & (offs_t[None, :] < T)),
        other=0.0,
    )

    mi = tl.zeros([BLOCK_T], dtype=tl.float32) - float("inf")
    li = tl.zeros([BLOCK_T], dtype=tl.float32)
    oi = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    # Loop through the seq length dimension
    for start_l in range(0, length, BLOCK_L):
        offs_l = start_l + tl.arange(0, BLOCK_L)

        # Load a block of K into [BLOCK_L, BLOCK_D]
        qj_ptrs = (
            q_start_ptr
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            + offs_l[:, None] * stride_ql
            + offs_d[None, :] * stride_qd
        )

        qj = tl.load(
            qj_ptrs,
            mask=((offs_l[:, None] < length) & (offs_d[None, :] < D)),
            other=0.0,
        )

        # gemm [BLOCK_L, BLOCK_D] x [BLOCK_D, BLOCK_T] = [BLOCK_L, BLOCK_T]
        qk = tl.dot(qj, ki, allow_tf32=allow_tf32)

        # Load a block of attn bias into [BLOCK_L, BLOCK_T]
        ab_ptrs = (
            ab_start_ptr + offs_l[:, None] * stride_ab_n + offs_t[None, :] * stride_ab_t
        )

        abij = tl.load(
            ab_ptrs,
            mask=((offs_l[:, None] < length) & (offs_t[None, :] < T)),
            other=0.0,
        )

        # q*k output + attn bias
        qk = qk + abij

        # Note: softmax on axis 0
        mij_hat = tl.max(qk, axis=0)
        mi_new = tl.maximum(mi, mij_hat)
        pij_hat = tl.exp(qk - mi_new[None, :])
        pij_hat = tl.where(
            (offs_l[:, None] < length) & (offs_t[None, :] < T), pij_hat, 0.0
        )
        lij_hat = tl.sum(pij_hat, axis=0)
        alpha = tl.exp(mi - mi_new)
        li_new = alpha * li + lij_hat
        oi = alpha[:, None] * oi

        # Load a block of V into [BLOCK_L, BLOCK_D]
        vj_ptrs = (
            v_start_ptr + offs_l[:, None] * stride_vn + offs_d[None, :] * stride_vd
        )

        vj = tl.load(
            vj_ptrs,
            mask=((offs_l[:, None] < length) & (offs_d[None, :] < D)),
            other=0.0,
        )

        pij_hat = pij_hat.to(v_ptr.dtype.element_ty)
        # gemm [BLOCK_T, BLOCK_L] x [BLOCK_L, BLOCK_D] = [BLOCK_T, BLOCK_D]
        oi = oi + tl.dot(tl.trans(pij_hat), vj, allow_tf32=allow_tf32)

        mi = mi_new
        li = li_new

    oi = oi / li[:, None]

    lse_ptrs = lse_ptr + pid_batch * T + offs_t
    # Save both mi and li to avoid recomputation in backward
    lse_i = mi + tl.log(li)
    tl.store(lse_ptrs, lse_i, mask=(offs_t < T))

    # Write the output [BLOCK_T, BLOCK_D]
    attn_out_ptrs = (
        o_ptr
        + pid_batch * stride_ob
        + offs_t[:, None] * stride_ot
        + offs_d[None, :] * stride_od
    )
    tl.store(attn_out_ptrs, oi, mask=((offs_t[:, None] < T) & (offs_d[None, :] < D)))


def jagged_dense_flash_attention_fwd(
    Q,
    K,
    V,
    attn_bias,
    jagged_offsets,
    max_seq_len,
    allow_tf32=False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q: jagged tensor, [sum_B, D]
    K: dense tensor, [B, D, T]
    V: jagged tensor [sum_B, D]
    attn_bias: dense tensor [B, N, T]
    out: dense tenros: [B, T, D]

    Attention steps:
    1. Q * K: [sum_B, D] * [B, D, T] = [sum_B, T]
    2. softmax_input = Q * K + attn_bias
    [sum_B, T] + [B, N, T] = [sum_B, T]
    3. softmax_out = softmax(softmax_input):
    softmax([sum_B, T]) = [sum_B, T]
    4. softmax_out * V:
    [sum_B, T] * [sum_B, D] = [B, T, D]
    """
    assert Q.size(1) == K.size(1), "incompatible dimensions for Q and K"
    assert Q.size() == V.size(), "incompatible dimensions for Q and V"
    assert jagged_offsets.is_contiguous(), "jagged_offsets must be contiguous"

    (B, D, T) = K.size()
    assert D > 0 and (D & (D - 1)) == 0, "D needs to be a power of two"

    attn_out = torch.zeros(B, T, D, dtype=Q.dtype, device=Q.device)
    lse = torch.empty((B, T), dtype=K.dtype, device=K.device)

    BLOCK_T = 32
    BLOCK_L = 32
    BLOCK_D = D

    num_blocks_t = triton.cdiv(T, BLOCK_T)
    grid = (num_blocks_t, B)

    jagged_dense_flash_attention_fwd_kernel[grid](
        Q,
        K,
        V,
        attn_bias,
        attn_out,
        lse,
        jagged_offsets,
        max_seq_len,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        attn_bias.stride(0),
        attn_bias.stride(1),
        attn_bias.stride(2),
        attn_out.stride(0),
        attn_out.stride(1),
        attn_out.stride(2),
        D,
        T,
        allow_tf32,
        BLOCK_T,  # pyre-ignore
        BLOCK_L,  # pyre-ignore
        BLOCK_D,
    )

    return attn_out, lse


@triton.jit
def _bwd_preprocess_do_o_dot(
    o_ptr,
    do_ptr,
    delta_ptr,
    T,
    stride_ob,
    stride_ot,
    stride_od,
    stride_do_b,
    stride_do_t,
    stride_do_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_t = tl.program_id(0)
    offs_t = start_t * BLOCK_T + tl.arange(0, BLOCK_T)
    pid_b = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)

    o_ptrs = (
        o_ptr
        + pid_b * stride_ob
        + offs_t[:, None] * stride_ot
        + offs_d[None, :] * stride_od
    )
    do_ptrs = (
        do_ptr
        + pid_b * stride_do_b
        + offs_t[:, None] * stride_do_t
        + offs_d[None, :] * stride_do_d
    )
    o = tl.load(o_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    delta = tl.sum(o * do, axis=1)

    delta_ptrs = delta_ptr + pid_b * T + offs_t
    tl.store(delta_ptrs, delta, mask=(offs_t < T))


@triton.jit
def _jagged_dense_flash_attention_bwd_dv_db_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ab_ptr,  # attn bias
    jagged_offsets_ptr,
    out_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dbias_ptr,
    max_seq_len,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kd,
    stride_kt,
    stride_vl,
    stride_vd,
    stride_ab_b,  # attn bias stride batch
    stride_ab_l,
    stride_ab_t,
    stride_ob,
    stride_ot,
    stride_od,
    stride_dq_l,
    stride_dq_d,
    stride_dv_l,
    stride_dv_d,
    stride_db_b,
    stride_db_l,
    stride_db_t,
    stride_do_b,
    stride_do_t,
    stride_do_d,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)
    # begin offset of the current sample
    begin = tl.load(jagged_offsets_ptr + pid_b)
    # end offset of the current sample
    end = tl.load(jagged_offsets_ptr + pid_b + 1)

    # The seq length of the current sample
    seqlen = end - begin
    seqlen = tl.minimum(seqlen, max_seq_len)

    if seqlen == 0:
        return

    q_start_ptr = q_ptr + begin * stride_ql
    k_start_ptr = k_ptr + pid_b * stride_kb
    ab_start_ptr = ab_ptr + pid_b * stride_ab_b
    v_start_ptr = v_ptr + begin * stride_vl
    do_start_ptr = do_ptr + pid_b * stride_do_b
    dq_start_ptr = dq_ptr + begin * stride_dq_l
    dv_start_ptr = dv_ptr + begin * stride_dv_l
    dbias_start_ptr = dbias_ptr + pid_b * stride_db_b
    delta_ptrs = delta_ptr + pid_b * T
    lse_ptrs = lse_ptr + pid_b * T

    start_l = pid_l * BLOCK_L
    offs_l_curr = start_l + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_T)

    q_ptrs = (
        q_start_ptr + offs_l_curr[:, None] * stride_ql + offs_d[None, :] * stride_qd
    )
    k_ptrs = k_start_ptr + offs_d[:, None] * stride_kd + offs_t[None, :] * stride_kt
    v_ptrs = (
        v_start_ptr + offs_l_curr[:, None] * stride_vl + offs_d[None, :] * stride_vd
    )

    do_ptrs = (
        do_start_ptr + offs_t[:, None] * stride_do_t + offs_d[None, :] * stride_do_d
    )

    dq = tl.zeros([BLOCK_L, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_L, BLOCK_D], dtype=tl.float32)

    # Load a block of q into [BLOCK_L, BLOCK_D]
    q = tl.load(
        q_ptrs,
        mask=((offs_l_curr[:, None] < seqlen) & (offs_d[None, :] < BLOCK_D)),
        other=0.0,
    )
    v = tl.load(v_ptrs, mask=(offs_l_curr[:, None] < seqlen), other=0.0)

    # for start_t in range(0, T, BLOCK_T):
    start_t = 0
    while start_t < T:
        offs_t_curr = start_t + tl.arange(0, BLOCK_T)

        # Load a block of k into [BLOCK_D, BLOCK_T]
        k = tl.load(
            k_ptrs,
            mask=((offs_t_curr[None, :] < T) & (offs_d[:, None] < BLOCK_D)),
            other=0.0,
        )
        qk = tl.zeros([BLOCK_L, BLOCK_T], dtype=tl.float32)

        # gemm [BLOCK_L, BLOCK_D] x [BLOCK_D, BLOCK_T] -> [BLOCK_L, BLOCK_T]
        qk += tl.dot(q, k, allow_tf32=allow_tf32)

        ab_ptrs = (
            ab_start_ptr
            + offs_l_curr[:, None] * stride_ab_l
            + offs_t_curr[None, :] * stride_ab_t
        )

        ab = tl.load(
            ab_ptrs,
            mask=((offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)),
            other=0.0,
        )

        # q*k output + attn bias
        qk = qk + ab

        # Mask out invalid positions for softmax
        qk_mask = (offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)
        qk = tl.where(qk_mask, qk, float("-inf"))

        lse_t = tl.load(
            lse_ptrs + offs_t_curr, mask=(offs_t_curr < T), other=float("inf")
        )
        # Perform softmax
        p = tl.exp(qk - lse_t[None, :])
        p = tl.where(qk_mask, p, 0.0)

        # Compute dv
        # Load a block of do into [BLOCK_T, BLOCK_D]
        do = tl.load(do_ptrs, mask=(offs_t_curr[:, None] < T), other=0.0)

        # gemm [BLOCK_L, BLOCK_T] x [BLOCK_T, BLOCK_D] -> [BLOCK_L, BLOCK_D]
        dv += tl.dot(p, do, allow_tf32=allow_tf32)

        # Compute dp
        delta = tl.load(delta_ptrs + offs_t_curr, mask=(offs_t_curr < T))
        dp = tl.zeros([BLOCK_L, BLOCK_T], dtype=tl.float32)

        # gemm [BLOCK_T, BLOCK_D] x [BLOCK_D, BLOCK_L] = [BLOCK_T, BLOCK_L]
        # [BLOCK_T, BLOCK_L]^T -> [BLOCK_L, BLOCK_T]
        dp += tl.trans(tl.dot(do, tl.trans(v), allow_tf32=allow_tf32))

        # Compute ds = p * (dp - delta)
        ds = p * (dp - delta[None, :])

        # Save dbias = ds
        dbias_ptrs = (
            dbias_start_ptr
            + offs_l_curr[:, None] * stride_db_l
            + offs_t_curr[None, :] * stride_db_t
        )
        tl.store(
            dbias_ptrs,
            ds,
            mask=((offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)),
        )

        # Compute dq
        # gemm [BLOCK_L, BLOCK_T] x [BLOCK_T, BLOCK_D] -> [BLOCK_L, BLOCK_D]
        dq += tl.dot(ds, tl.trans(k), allow_tf32=allow_tf32)

        k_ptrs += BLOCK_T * stride_kt
        do_ptrs += BLOCK_T * stride_do_t
        start_t += BLOCK_T

    dq_ptrs = (
        dq_start_ptr
        + offs_l_curr[:, None] * stride_dq_l
        + offs_d[None, :] * stride_dq_d
    )
    dv_ptrs = (
        dv_start_ptr
        + offs_l_curr[:, None] * stride_dv_l
        + offs_d[None, :] * stride_dv_d
    )
    tl.store(dq_ptrs, dq, mask=(offs_l_curr[:, None] < seqlen))
    tl.store(dv_ptrs, dv, mask=(offs_l_curr[:, None] < seqlen))


@triton.jit
def _jagged_dense_flash_attention_bwd_dk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ab_ptr,  # attn bias
    jagged_offsets_ptr,
    out_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dbias_ptr,
    max_seq_len,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kd,
    stride_kt,
    stride_vl,
    stride_vd,
    stride_ab_b,  # attn bias stride batch
    stride_ab_l,
    stride_ab_t,
    stride_ob,
    stride_ot,
    stride_od,
    stride_dk_b,
    stride_dk_d,
    stride_dk_t,
    stride_do_b,
    stride_do_t,
    stride_do_d,
    D,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    # begin offset of the current sample
    begin = tl.load(jagged_offsets_ptr + pid_b)
    # end offset of the current sample
    end = tl.load(jagged_offsets_ptr + pid_b + 1)

    # The seq length of the current sample
    seqlen = end - begin
    seqlen = tl.minimum(seqlen, max_seq_len)

    if seqlen == 0:
        return

    q_start_ptr = q_ptr + begin * stride_ql
    k_start_ptr = k_ptr + pid_b * stride_kb
    ab_start_ptr = ab_ptr + pid_b * stride_ab_b
    v_start_ptr = v_ptr + begin * stride_vl
    do_start_ptr = do_ptr + pid_b * stride_do_b
    dk_start_ptr = dk_ptr + pid_b * stride_dk_b
    delta_ptrs = delta_ptr + pid_b * T
    lse_ptrs = lse_ptr + pid_b * T

    offs_t_curr = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    k_ptrs = (
        k_start_ptr + offs_d[:, None] * stride_kd + offs_t_curr[None, :] * stride_kt
    )

    do_ptrs = (
        do_start_ptr
        + offs_t_curr[:, None] * stride_do_t
        + offs_d[None, :] * stride_do_d
    )

    dk_ptrs = (
        dk_start_ptr
        + offs_d[:, None] * stride_dk_d
        + offs_t_curr[None, :] * stride_dk_t
    )

    dk = tl.zeros([BLOCK_D, BLOCK_T], dtype=tl.float32)

    # Load a block of k into [BLOCK_D, BLOCK_T]
    k = tl.load(
        k_ptrs,
        mask=((offs_t_curr[None, :] < T) & (offs_d[:, None] < BLOCK_D)),
        other=0.0,
    )

    start_l = 0
    while start_l < seqlen:
        offs_l_curr = start_l + tl.arange(0, BLOCK_L)

        # Load a block of q into [BLOCK_L, BLOCK_D]
        q_ptrs = (
            q_start_ptr + offs_l_curr[:, None] * stride_ql + offs_d[None, :] * stride_qd
        )

        q = tl.load(
            q_ptrs,
            mask=(offs_l_curr[:, None] < seqlen),
            other=0.0,
        )

        v_ptrs = (
            v_start_ptr + offs_l_curr[:, None] * stride_vl + offs_d[None, :] * stride_vd
        )

        v = tl.load(v_ptrs, mask=(offs_l_curr[:, None] < seqlen), other=0.0)

        qk = tl.zeros([BLOCK_L, BLOCK_T], dtype=tl.float32)
        # gemm [BLOCK_L, BLOCK_D] x [BLOCK_D, BLOCK_T] -> [BLOCK_L, BLOCK_T]

        qk = tl.dot(q, k, allow_tf32=allow_tf32)
        qk = tl.where(
            (offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T), qk, 0.0
        )

        ab_ptrs = (
            ab_start_ptr
            + offs_l_curr[:, None] * stride_ab_l
            + offs_t_curr[None, :] * stride_ab_t
        )

        ab = tl.load(
            ab_ptrs,
            mask=((offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)),
            other=0.0,
        )

        # q*k output + attn bias
        qk = qk + ab

        # Mask out invalid positions for softmax
        qk_mask = (offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)
        qk = tl.where(qk_mask, qk, float("-inf"))

        lse_t = tl.load(lse_ptrs + offs_t_curr, mask=(offs_t_curr < T))
        # Perform softmax
        p = tl.exp(qk - lse_t[None, :])
        p = tl.where(qk_mask, p, 0.0)

        # Compute dv
        # Load a block of do into [BLOCK_T, BLOCK_D]
        do = tl.load(do_ptrs, mask=(offs_t_curr[:, None] < T), other=0.0)

        # Compute dp
        delta = tl.load(delta_ptrs + offs_t_curr, mask=(offs_t_curr < T))

        # gemm [BLOCK_T, BLOCK_D] x [BLOCK_D, BLOCK_L] = [BLOCK_T, BLOCK_L]
        # [BLOCK_T, BLOCK_L]^T -> [BLOCK_L, BLOCK_T]
        dp = tl.trans(tl.dot(do, tl.trans(v), allow_tf32=allow_tf32))

        # Compute ds = p * (dp - delta)
        ds = p * (dp - delta[None, :])

        # Compute dk
        # gemm [BLOCK_D, BLOCK_L] x [BLOCK_L, BLOCK_T] = [BLOCK_D, BLOCK_T]
        dk += tl.dot(tl.trans(q), ds, allow_tf32=allow_tf32)

        start_l += BLOCK_L

    tl.store(dk_ptrs, dk, mask=(offs_t_curr[None, :] < T))


def jagged_dense_flash_attention_bwd(
    Q,
    K,
    V,
    Out,
    lse,
    do,  # derivative of attn_out
    attn_bias,
    jagged_offsets,
    max_seq_len,
    allow_tf32=False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Q: jagged tensor, [sum_B, D]
    K: dense tensor, [B, D, T]
    V: jagged tensor [sum_B, D]
    Out: dense tensor: [B, T, D]
    lse: dense tensor [B, T]
    do: dense tensor [B, T, D]
    attn_bias: dense tensor [B, N, T]
    jagged_offsets: tensor [B + 1]
    """
    assert Q.size(1) == K.size(1), "incompatible dimensions for Q and K"
    assert Q.size() == V.size(), "incompatible dimensions for Q and V"
    assert lse.size(1) == K.size(2), "incompatible dimensions for LSE and K"

    if not do.is_contiguous():
        do = do.contiguous()

    (B, D, T) = K.size()
    BLOCK_T = 32
    BLOCK_L = 32
    BLOCK_D = D
    num_blocks_k = triton.cdiv(T, BLOCK_T)

    dk = torch.zeros_like(K)
    dq = torch.zeros_like(Q)
    dv = torch.zeros_like(V)
    dbias = torch.zeros_like(attn_bias)

    delta = torch.empty_like(lse)
    _bwd_preprocess_do_o_dot[(num_blocks_k, B)](
        Out,
        do,
        delta,
        T,
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        BLOCK_T,  # pyre-ignore
        BLOCK_D,
    )

    num_blocks_l = triton.cdiv(max_seq_len, BLOCK_L)
    _jagged_dense_flash_attention_bwd_dv_db_dq_kernel[(num_blocks_l, B)](
        Q,
        K,
        V,
        attn_bias,
        jagged_offsets,
        Out,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        dbias,
        max_seq_len,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        attn_bias.stride(0),
        attn_bias.stride(1),
        attn_bias.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        dq.stride(0),
        dq.stride(1),
        dv.stride(0),
        dv.stride(1),
        dbias.stride(0),
        dbias.stride(1),
        dbias.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        T,
        BLOCK_T,  # pyre-ignore
        BLOCK_L,  # pyre-ignore
        BLOCK_D,
        allow_tf32,
    )

    num_blocks_t = triton.cdiv(T, BLOCK_T)
    _jagged_dense_flash_attention_bwd_dk_kernel[(num_blocks_t, B)](
        Q,
        K,
        V,
        attn_bias,
        jagged_offsets,
        Out,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        dbias,
        max_seq_len,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        attn_bias.stride(0),
        attn_bias.stride(1),
        attn_bias.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        D,
        T,
        BLOCK_T,  # pyre-ignore
        BLOCK_L,  # pyre-ignore
        BLOCK_D,
        allow_tf32,
    )

    return dq, dk, dv, dbias


class JaggedDenseFlashAttention(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_bias: torch.Tensor,
        jagged_offsets: torch.Tensor,
        max_seq_len: int,
        allow_tf32: bool = False,
    ) -> torch.Tensor:
        attn_out, lse = jagged_dense_flash_attention_fwd(
            Q, K, V, attn_bias, jagged_offsets, max_seq_len, allow_tf32
        )
        ctx.save_for_backward(Q, K, V, attn_bias, jagged_offsets, lse, attn_out)
        ctx.max_seq_len = max_seq_len
        ctx.allow_tf32 = allow_tf32
        return attn_out

    @staticmethod
    # pyre-fixme
    def backward(
        ctx, do: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None
    ]:
        Q, K, V, attn_bias, jagged_offsets, lse, attn_out = ctx.saved_tensors
        max_seq_len = ctx.max_seq_len
        allow_tf32 = ctx.allow_tf32

        dq, dk, dv, dbias = jagged_dense_flash_attention_bwd(
            Q,
            K,
            V,
            attn_out,
            lse,
            do,
            attn_bias,
            jagged_offsets,
            max_seq_len,
            allow_tf32,
        )
        return dq, dk, dv, dbias, None, None, None


def jagged_dense_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor,
    offsets: torch.Tensor,
    max_seq_len: int,
    allow_tf32: bool = True,
):
    """
    q: jagged tensor, [sum_B, D]
    k: dense tensor, [B, D, T]
    v: jagged tensor [sum_B, D]
    attn_bias: dense tensor [B, N, T]
    offsets: offsets for jagged tensor [B + 1]
    """

    q = expect_contiguous(q)
    k = expect_contiguous(k)
    v = expect_contiguous(v)
    attn_bias = expect_contiguous(attn_bias)
    offsets = expect_contiguous(offsets)

    return JaggedDenseFlashAttention.apply(
        q, k, v, attn_bias, offsets, max_seq_len, allow_tf32
    )
