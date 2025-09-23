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
def _multi_head_jagged_flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    offset_ptr,
    o_ptr,
    lse_i_ptr,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oh,
    stride_om,
    stride_od,
    stride_lse_h,
    num_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    D: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    pid_batch = pid_bh // num_heads
    pid_head = pid_bh % num_heads

    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)

    seqlen = end - begin
    seqlen = tl.minimum(seqlen, max_seq_len)

    if pid_m * BLOCK_M >= seqlen:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    mi = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    li = tl.zeros([BLOCK_M], dtype=tl.float32)
    for j in range(0, seqlen, BLOCK_N):
        offs_n = tl.arange(0, BLOCK_N) + j
        q_ptrs = (
            q_ptr
            + pid_head * stride_qh
            + begin * stride_qm
            + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        )

        k_ptrs = (
            k_ptr
            + pid_head * stride_kh
            + begin * stride_kn
            + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        )

        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for d in range(0, D, BLOCK_D):
            curr_d = d + offs_d

            # Load a block of q into [BLOCK_M, BLOCK_D]
            q = tl.load(
                q_ptrs,
                # pyre-fixme[16]: `int` has no attribute `__getitem__`.
                mask=((curr_d[None, :] < D) & (offs_m[:, None] < seqlen)),
                other=0.0,
            )

            # Load a block of k into [BLOCK_D, BLOCK_N]
            k = tl.load(
                k_ptrs,
                mask=((curr_d[:, None] < D) & (offs_n[None, :] < seqlen)),
                other=0.0,
            )

            # gemm [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            qk += tl.dot(q, k, allow_tf32=allow_tf32)

            q_ptrs += BLOCK_D * stride_qd
            k_ptrs += BLOCK_D * stride_kd

        mi_new = tl.maximum(tl.max(qk, axis=1), mi)
        # Add the correct mask here
        mn_mask = (offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen)

        p = tl.exp(qk - mi_new[:, None])
        p = tl.where(mn_mask, p, 0.0)

        lij_hat = tl.sum(p, axis=1)
        alpha = tl.exp(mi - mi_new)

        li = alpha * li + lij_hat
        acc = alpha[:, None] * acc

        # Load V into block [BLOCK_N, BLOCK_D]
        v_ptrs = (
            v_ptr
            + pid_head * stride_vh
            + begin * stride_vn
            + (offs_d[None, :] * stride_vd + offs_n[:, None] * stride_vn)
        )
        v = tl.load(
            v_ptrs,
            mask=((offs_d[None, :] < D) & (offs_n[:, None] < seqlen)),
            other=0.0,
        )

        p /= max_seq_len

        p = p.to(v_ptr.dtype.element_ty)
        # gemm [BLOCK_M, BLOCK_N] x [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
        acc += tl.dot(p, v, allow_tf32=allow_tf32)
        mi = mi_new

    lse_i = mi + tl.math.log(li)
    lse_i_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_i_ptrs = lse_i_ptr + pid_head * stride_lse_h + begin + lse_i_offsets

    tl.store(lse_i_ptrs, lse_i, mask=lse_i_offsets < seqlen)

    acc = acc / li[:, None]

    # Store O
    o_ptrs = o_ptr + (
        pid_head * stride_oh
        + begin * stride_om
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    o_mask = (offs_m[:, None] < seqlen) & (offs_d[None, :] < D)
    tl.store(o_ptrs, acc, mask=o_mask)


def multi_head_jagged_flash_attention_fwd(
    jagged_Q,
    jagged_K,
    jagged_V,
    offsets,
    max_seq_len,
    allow_tf32=False,
):
    assert jagged_Q.size(2) == jagged_K.size(2), "incompatible dimensions"

    B = offsets.size(0) - 1
    D = jagged_Q.size(2)
    num_heads = jagged_Q.size(0)

    jagged_O = torch.zeros_like(jagged_Q)
    lse = torch.zeros(
        (num_heads, jagged_Q.size(1)), device=jagged_Q.device, dtype=jagged_Q.dtype
    )

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = max(triton.next_power_of_2(D), 16)

    grid = (triton.cdiv(max_seq_len, BLOCK_M), B * num_heads)

    _multi_head_jagged_flash_attention_fwd_kernel[grid](
        jagged_Q,
        jagged_K,
        jagged_V,
        offsets,
        jagged_O,
        lse,
        jagged_Q.stride(0),
        jagged_Q.stride(1),
        jagged_Q.stride(2),
        jagged_K.stride(0),
        jagged_K.stride(1),
        jagged_K.stride(2),
        jagged_V.stride(0),
        jagged_V.stride(1),
        jagged_V.stride(2),
        jagged_O.stride(0),
        jagged_O.stride(1),
        jagged_O.stride(2),
        lse.stride(0),
        num_heads,
        max_seq_len,
        D,
        allow_tf32,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return jagged_O, lse


@triton.jit
def _multi_head_jagged_flash_attention_bwd_preprocess_kernel(
    o_ptr,
    o_offset_ptr,
    do_ptr,
    delta_ptr,
    stride_oh,
    stride_om,
    stride_od,
    stride_delta_h,
    num_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    pid_batch = pid_bh // num_heads
    pid_head = pid_bh % num_heads

    begin_o = tl.load(o_offset_ptr + pid_batch)
    end_o = tl.load(o_offset_ptr + pid_batch + 1)

    M = end_o - begin_o
    M = tl.minimum(M, max_seq_len)

    if M == 0:
        return

    offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_od = tl.arange(0, BLOCK_D)

    o_offsets = (
        offs_om[:, None] * stride_om
        + offs_od[None, :] * stride_od
        + pid_head * stride_oh
        + begin_o * stride_om
    )
    o_ptrs = o_ptr + o_offsets
    do_ptrs = do_ptr + o_offsets
    o_mask = (offs_om[:, None] < M) & (offs_od[None, :] < D)

    # Load o and do
    o = tl.load(o_ptrs, mask=o_mask)
    do = tl.load(do_ptrs, mask=o_mask)

    delta = tl.sum(o * do, axis=1)

    tl.store(
        delta_ptr + pid_head * stride_delta_h + begin_o + offs_om,
        delta,
        mask=offs_om < M,
    )


@triton.jit
def _multi_head_jagged_flash_attention_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    offset_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    do_ptr,
    delta_ptr,
    lse_ptr,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oh,
    stride_om,
    stride_od,
    stride_lse_h,
    stride_delta_h,
    stride_dq_h,
    stride_dq_m,
    stride_dq_d,
    stride_dk_h,
    stride_dk_n,
    stride_dk_d,
    stride_dv_h,
    stride_dv_n,
    stride_dv_d,
    stride_do_h,
    stride_do_m,
    stride_do_d,
    num_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    D: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(axis=1)
    pid_batch = pid_bh // num_heads
    pid_head = pid_bh % num_heads

    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)

    seqlen = tl.minimum(end - begin, max_seq_len)

    if seqlen == 0:
        return

    pid_n = tl.program_id(axis=0)
    offs_d = tl.arange(0, BLOCK_D)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    q_ptrs = (
        q_ptr
        + pid_head * stride_qh
        + begin * stride_qm
        + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )

    k_ptrs = (
        k_ptr
        + pid_head * stride_kh
        + begin * stride_kn
        + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    )

    v_ptrs = (
        v_ptr
        + pid_head * stride_vh
        + begin * stride_vn
        + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    )

    do_ptrs = (
        do_ptr
        + pid_head * stride_do_h
        + begin * stride_do_m
        + (offs_m[:, None] * stride_do_m + offs_d[None, :] * stride_do_d)
    )

    # Load a block of K into [BLOCK_N, BLOCK_D]
    k = tl.load(
        k_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < seqlen)), other=0.0
    )
    # Load a block of V into [BLOCK_N, BLOCK_D]
    v = tl.load(
        v_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < seqlen)), other=0.0
    )

    # Initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    for begin_m in range(0, seqlen, BLOCK_M):
        offs_m_curr = begin_m + offs_m

        # Load a block of Q into [BLOCK_M, BLOCK_D]
        q = tl.load(
            q_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < seqlen)),
            other=0.0,
        )
        # gemm [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k), allow_tf32=allow_tf32)

        mn_mask = (offs_m_curr[:, None] < seqlen) & (offs_n[None, :] < seqlen)

        # Load a block of lse_i into [BLOCK_M]
        lse_i = tl.load(
            lse_ptr + pid_head * stride_lse_h + begin + offs_m_curr,
            mask=offs_m_curr < seqlen,
            other=float("inf"),
        )

        p = tl.exp(qk - lse_i[:, None])
        p = tl.where(mn_mask, p, 0.0)
        p /= max_seq_len

        p = p.to(do_ptr.dtype.element_ty)
        do = tl.load(
            do_ptrs,
            mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < seqlen)),
            other=0.0,
        )

        # gemm [BLOCK_N, BLOCK_M] x [BLOCK_M, BLOCK_D] -> [BLOCK_N, BLOCK_D]
        dv += tl.dot(tl.trans(p), do, allow_tf32=allow_tf32)
        # gemm [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v), allow_tf32=allow_tf32)

        # compute ds = p * (dp - delta[:, None])
        Di = tl.load(
            delta_ptr + pid_head * stride_delta_h + begin + offs_m_curr,
            mask=offs_m_curr < seqlen,
        )
        ds = p * (dp - Di[:, None] * max_seq_len)

        # compute dk = dot(ds.T, q)
        ds = ds.to(q_ptr.dtype.element_ty)
        # gemm [BLOCK_N, BLOCK_M] x [BLOCK_M, BLOCK_D] -> [BLOCK_N, BLOCK_D]
        dk += tl.dot(tl.trans(ds), q, allow_tf32=allow_tf32)

        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_do_m

    # store back dk and dv
    dk_ptrs = (
        dk_ptr
        + pid_head * stride_dk_h
        + begin * stride_dk_n
        + (offs_n[:, None] * stride_dk_n + offs_d[None, :] * stride_dk_d)
    )

    dv_ptrs = (
        dv_ptr
        + pid_head * stride_dv_h
        + begin * stride_dv_n
        + (offs_n[:, None] * stride_dv_n + offs_d[None, :] * stride_dv_d)
    )

    tl.store(dk_ptrs, dk, mask=((offs_d[None, :] < D) & (offs_n[:, None] < seqlen)))
    tl.store(dv_ptrs, dv, mask=((offs_d[None, :] < D) & (offs_n[:, None] < seqlen)))

    # Start to compute dq

    start_m = tl.program_id(axis=0) * BLOCK_M
    offs_m_curr = start_m + tl.arange(0, BLOCK_M)

    dq_ptrs_curr = (
        dq_ptr
        + pid_head * stride_dq_h
        + begin * stride_dq_m
        + (offs_m_curr[:, None] * stride_dq_m + offs_d[None, :] * stride_dq_d)
    )

    dq_curr = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    q_ptrs_curr = (
        q_ptr
        + pid_head * stride_qh
        + begin * stride_qm
        + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )

    q_curr = tl.load(
        q_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < seqlen))
    )

    # Load a block of lse_i into [BLOCK_M]
    lse_i_curr = tl.load(
        lse_ptr + pid_head * stride_lse_h + begin + offs_m_curr,
        mask=offs_m_curr < seqlen,
    )

    do_ptrs_curr = (
        do_ptr
        + pid_head * stride_do_h
        + begin * stride_do_m
        + (offs_m_curr[:, None] * stride_do_m + offs_d[None, :] * stride_do_d)
    )

    # Load do
    do_curr = tl.load(
        do_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < seqlen))
    )
    Di_curr = tl.load(
        delta_ptr + pid_head * stride_delta_h + begin + offs_m_curr,
        mask=offs_m_curr < seqlen,
    )

    block_start = 0
    while block_start < seqlen:
        offs_n_curr = block_start + tl.arange(0, BLOCK_N)

        k_ptrs_curr = (
            k_ptr
            + pid_head * stride_kh
            + begin * stride_kn
            + (offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        )
        v_ptrs_curr = (
            v_ptr
            + pid_head * stride_vh
            + begin * stride_vn
            + (offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        )

        k_curr = tl.load(
            k_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < seqlen))
        )
        v_curr = tl.load(
            v_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < seqlen))
        )

        # gemm [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        qk_curr = tl.dot(q_curr, tl.trans(k_curr), allow_tf32=allow_tf32)
        mn_mask_curr = (offs_m_curr[:, None] < seqlen) & (offs_n_curr[None, :] < seqlen)

        # Perform softmax
        p_curr = tl.exp(qk_curr - lse_i_curr[:, None])
        p_curr = tl.where(mn_mask_curr, p_curr, 0.0)
        p_curr /= max_seq_len

        # compute dp = dot(do, v.T)
        # gemm [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        dp_curr = tl.dot(do_curr, tl.trans(v_curr), allow_tf32=allow_tf32)

        # compute ds = p * (dp - delta[:, None])
        ds_curr = p_curr * (dp_curr - Di_curr[:, None] * max_seq_len)

        ds_curr = ds_curr.to(k_ptr.dtype.element_ty)
        # compute dq = dot(ds, k)
        # gemm [BLOCK_M, BLOCK_N] x [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
        dq_curr += tl.dot(ds_curr, k_curr, allow_tf32=allow_tf32)
        block_start += BLOCK_N

    tl.store(
        dq_ptrs_curr,
        dq_curr,
        mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < seqlen)),
    )


def multi_head_jagged_flash_attention_bwd(
    jagged_Q,
    jagged_K,
    jagged_V,
    jagged_O,
    offsets,
    dO,
    lse,
    max_seq_len,
    allow_tf32=False,
):
    BLOCK_M = 32
    BLOCK_N = 32

    B = offsets.size(0) - 1
    num_heads = jagged_Q.size(0)
    D = jagged_Q.size(2)

    num_blocks_m = triton.cdiv(max_seq_len, BLOCK_M)
    pre_grid = (num_blocks_m, B * num_heads)

    # Triton requires the block size to be at least 16
    BLOCK_D = max(triton.next_power_of_2(D), 16)

    delta = torch.empty_like(lse)
    if not dO.is_contiguous():
        dO = dO.contiguous()

    _multi_head_jagged_flash_attention_bwd_preprocess_kernel[pre_grid](
        jagged_O,
        offsets,
        dO,
        delta,
        jagged_O.stride(0),
        jagged_O.stride(1),
        jagged_O.stride(2),
        delta.stride(0),
        num_heads,
        max_seq_len,
        D,
        BLOCK_M,
        BLOCK_D,
    )

    grid = (triton.cdiv(max_seq_len, BLOCK_N), B * num_heads)

    dq = torch.zeros_like(jagged_Q)
    dk = torch.zeros_like(jagged_K)
    dv = torch.zeros_like(jagged_V)

    _multi_head_jagged_flash_attention_bwd_kernel[grid](
        jagged_Q,
        jagged_K,
        jagged_V,
        jagged_O,
        offsets,
        dq,
        dk,
        dv,
        dO,
        delta,
        lse,
        jagged_Q.stride(0),
        jagged_Q.stride(1),
        jagged_Q.stride(2),
        jagged_K.stride(0),
        jagged_K.stride(1),
        jagged_K.stride(2),
        jagged_V.stride(0),
        jagged_V.stride(1),
        jagged_V.stride(2),
        jagged_O.stride(0),
        jagged_O.stride(1),
        jagged_O.stride(2),
        lse.stride(0),
        delta.stride(0),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dO.stride(0),
        dO.stride(1),
        dO.stride(2),
        num_heads,
        max_seq_len,
        D,
        allow_tf32=allow_tf32,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return dq, dk, dv


class MultiHeadJaggedFlashAttention(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        jagged_Q: torch.Tensor,
        jagged_K: torch.Tensor,
        jagged_V: torch.Tensor,
        offsets: torch.Tensor,
        max_seq_len: int,
        allow_tf32: bool = True,
    ) -> torch.Tensor:
        ctx.allow_tf32 = allow_tf32
        ctx.max_seq_len = max_seq_len

        jagged_O, lse = multi_head_jagged_flash_attention_fwd(
            jagged_Q,
            jagged_K,
            jagged_V,
            offsets,
            max_seq_len,
            allow_tf32,
        )

        ctx.save_for_backward(
            jagged_Q,
            jagged_K,
            jagged_V,
            offsets,
            jagged_O,
            lse,
        )

        return jagged_O

    @staticmethod
    # pyre-fixme
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        (
            jagged_Q,
            jagged_K,
            jagged_V,
            offsets,
            jagged_O,
            lse,
        ) = ctx.saved_tensors

        dq, dk, dv = multi_head_jagged_flash_attention_bwd(
            jagged_Q=jagged_Q,
            jagged_K=jagged_K,
            jagged_V=jagged_V,
            jagged_O=jagged_O,
            offsets=offsets,
            dO=grad_output,
            lse=lse,
            max_seq_len=ctx.max_seq_len,
            allow_tf32=ctx.allow_tf32,
        )

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
        )


def multi_head_jagged_flash_attention(
    q_weights: torch.Tensor,
    k_weights: torch.Tensor,
    v_weights: torch.Tensor,
    offsets: torch.Tensor,
    max_seq_len: int,
    allow_tf32: bool = True,
) -> torch.Tensor:
    """
    q_weights: jagged tensor with size [H, sum_B, D]
    k_weights: jagged tensor with size [H, sum_B, D]
    v_weights: jagged tensor with size [H, sum_B, D]
    offsets: offsets for jagged tensor, with size [B + 1]
    max_seq_len: max sequence length
    """
    q_weights = expect_contiguous(q_weights)
    k_weights = expect_contiguous(k_weights)
    v_weights = expect_contiguous(v_weights)
    offsets = expect_contiguous(offsets)

    jagged_O = MultiHeadJaggedFlashAttention.apply(
        q_weights,
        k_weights,
        v_weights,
        offsets,
        max_seq_len,
        allow_tf32,
    )

    return jagged_O
