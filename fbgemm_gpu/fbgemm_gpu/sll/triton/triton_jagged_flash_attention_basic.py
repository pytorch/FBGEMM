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
def jagged_flash_attention_basic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    offset_ptr,
    o_ptr,
    lse_i_ptr,
    stride_qm,
    stride_qd,
    stride_kd,
    stride_kn,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    max_seq_len,
    D: tl.constexpr,
    NEXT_D: tl.constexpr,
    use_mask: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)

    seqlen = end - begin
    seqlen = tl.minimum(seqlen, max_seq_len)

    if pid_m * BLOCK_SIZE_M >= seqlen:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    # Offset till next power of 2 for D
    offs_nextd = tl.arange(0, NEXT_D)

    acc = tl.zeros([BLOCK_SIZE_M, NEXT_D], dtype=tl.float32)

    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for j in range(0, seqlen, BLOCK_SIZE_N):
        offs_n = tl.arange(0, BLOCK_SIZE_N) + j
        q_ptrs = (
            q_ptr
            + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            + begin * stride_qm
        )

        k_ptrs = (
            k_ptr
            + (offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn)
            + begin * stride_kn
        )

        qk = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for d in range(0, D, BLOCK_SIZE_D):
            updated_offset = d + offs_d
            q = tl.load(
                q_ptrs,
                # pyre-fixme[16]: `int` has no attribute `__getitem__`.
                mask=((updated_offset[None, :] < D) & (offs_m[:, None] < seqlen)),
                other=0.0,
            )
            k = tl.load(
                k_ptrs,
                mask=((updated_offset[:, None] < D) & (offs_n[None, :] < seqlen)),
                other=0.0,
            )
            qk += tl.dot(q, k, allow_tf32=allow_tf32)

            q_ptrs += BLOCK_SIZE_D * stride_qd
            k_ptrs += BLOCK_SIZE_D * stride_kd

        m_ij = tl.maximum(tl.max(qk, axis=1), m_i)
        # Add the correct mask here
        mn_mask = (offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen)

        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(mn_mask, p, 0.0)

        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V
        v_ptrs = (
            v_ptr
            + (offs_nextd[None, :] * stride_vd + offs_n[:, None] * stride_vn)
            + begin * stride_vn
        )
        v = tl.load(
            v_ptrs,
            mask=((offs_nextd[None, :] < D) & (offs_n[:, None] < seqlen)),
            other=0.0,
        )

        p /= max_seq_len

        if use_mask:
            attn_mask = offs_m[:, None] - offs_n[None, :]
            attn_mask = tl.where(mn_mask, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            p = tl.where(attn_mask > 0, p, 0.0)

        p = p.to(v_ptr.dtype.element_ty)
        acc_j = tl.dot(p, v, allow_tf32=allow_tf32)
        acc += acc_j
        m_i = m_ij

    lse_i = m_i + tl.math.log(l_i)
    lse_i_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    lse_i_ptrs = lse_i_ptr + lse_i_offsets + begin

    tl.store(lse_i_ptrs, lse_i, mask=lse_i_offsets < seqlen)

    acc = acc / l_i[:, None]

    # Store O
    o_ptrs = o_ptr + (
        offs_m[:, None] * stride_om
        + offs_nextd[None, :] * stride_od
        + begin * stride_om
    )
    o_mask = (offs_m[:, None] < seqlen) & (offs_nextd[None, :] < D)
    tl.store(o_ptrs, acc, mask=o_mask)


def jagged_flash_attention_basic_fwd(
    jagged_Q,
    jagged_K,
    jagged_V,
    offsets,
    max_seq_len,
    use_mask,
    allow_tf32=False,
):
    assert jagged_Q.size(1) == jagged_K.size(0), "incompatible dimensions"

    B = offsets.size(0) - 1
    D = jagged_Q.size(1)

    jagged_O = torch.zeros_like(jagged_Q)
    lse = torch.empty((jagged_Q.size(0)), device=jagged_Q.device, dtype=jagged_Q.dtype)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = 32

    grid = (triton.cdiv(max_seq_len, BLOCK_SIZE_M), B)

    jagged_flash_attention_basic_kernel[grid](
        jagged_Q,
        jagged_K,
        jagged_V,
        offsets,
        jagged_O,
        lse,
        jagged_Q.stride(0),
        jagged_Q.stride(1),
        jagged_K.stride(0),
        jagged_K.stride(1),
        jagged_V.stride(0),
        jagged_V.stride(1),
        jagged_O.stride(0),
        jagged_O.stride(1),
        max_seq_len,
        D,
        triton.next_power_of_2(D),
        use_mask,
        allow_tf32,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return jagged_O, lse


# Similar to fwd kernel, this one is using a grid of
# (num_blocks_m, B) where num_blocks_m is seq_len / BLOCK_SIZE_M
@triton.jit
def _jagged_flash_attention_bwd_preprocess_basic_kernel(
    o_ptr,
    o_offset_ptr,
    do_ptr,
    delta_ptr,
    stride_om,
    stride_od,
    max_seq_len,
    D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    begin_o = tl.load(o_offset_ptr + pid_batch)
    end_o = tl.load(o_offset_ptr + pid_batch + 1)

    M = end_o - begin_o
    M = tl.minimum(M, max_seq_len)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_od = tl.arange(0, BLOCK_SIZE_D)

    o_offsets = (
        offs_om[:, None] * stride_om
        + offs_od[None, :] * stride_od
        + begin_o * stride_om
    )
    o_ptrs = o_ptr + o_offsets
    do_ptrs = do_ptr + o_offsets
    o_mask = (offs_om[:, None] < M) & (offs_od[None, :] < D)

    # Load O
    o = tl.load(o_ptrs, mask=o_mask)
    do = tl.load(do_ptrs, mask=o_mask)

    delta = tl.sum(o * do, axis=1)

    tl.store(delta_ptr + begin_o + offs_om, delta, mask=offs_om < M)


@triton.jit
def _jagged_flash_attention_bwd_basic_kernel(
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
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    stride_dqm,
    stride_dqd,
    stride_dkn,
    stride_dkd,
    stride_dvn,
    stride_dvd,
    stride_dom,
    stride_dod,
    max_seq_len,
    D: tl.constexpr,
    use_mask: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_batch = tl.program_id(axis=1)

    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)

    M = tl.minimum(end - begin, max_seq_len)

    pid_n = tl.program_id(axis=0)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    q_ptrs = (
        q_ptr
        + begin * stride_qm
        + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )

    k_ptrs = (
        k_ptr
        + begin * stride_kn
        + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    )

    v_ptrs = (
        v_ptr
        + begin * stride_vn
        + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    )

    do_ptrs = (
        do_ptr
        + begin * stride_dom
        + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
    )

    # Load K and V
    k = tl.load(k_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))
    v = tl.load(v_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))

    # Initialize dv and dk
    dv = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)

    for begin_m in range(0, M, BLOCK_SIZE_M):
        offs_m_temp = begin_m + offs_m

        # Load Q
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        q = tl.load(q_ptrs, mask=((offs_d[None, :] < D) & (offs_m_temp[:, None] < M)))
        qk = tl.dot(q, tl.trans(k), allow_tf32=allow_tf32)

        mn_mask = (offs_m_temp[:, None] < M) & (offs_n[None, :] < M)

        # Load lse_i
        lse_i = tl.load(lse_ptr + offs_m_temp + begin, mask=offs_m_temp < M)

        p = tl.exp(qk - lse_i[:, None])
        p = tl.where(mn_mask, p, 0.0)
        p /= max_seq_len
        p_masked = p

        attn_mask = None
        if use_mask:
            attn_mask = offs_m_temp[:, None] - offs_n[None, :]
            attn_mask = tl.where(mn_mask, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            p_masked = tl.where(attn_mask > 0, p, 0.0)

        p_masked = p_masked.to(do_ptr.dtype.element_ty)
        do = tl.load(do_ptrs, mask=((offs_d[None, :] < D) & (offs_m_temp[:, None] < M)))
        dv += tl.dot(tl.trans(p_masked), do, allow_tf32=allow_tf32)
        dp = tl.dot(do, tl.trans(v), allow_tf32=allow_tf32)

        # compute ds = p * (dp - delta[:, None])
        Di = tl.load(delta_ptr + offs_m_temp + begin, mask=offs_m_temp < M)
        dp_masked = dp
        if use_mask:
            dp_masked = tl.where(attn_mask > 0, dp, 0.0)

        ds = p * (dp_masked - Di[:, None] * max_seq_len)

        # compute dk = dot(ds.T, q)
        ds = ds.to(q_ptr.dtype.element_ty)
        dk += tl.dot(tl.trans(ds), q, allow_tf32=allow_tf32)

        q_ptrs += BLOCK_SIZE_M * stride_qm
        do_ptrs += BLOCK_SIZE_M * stride_dom

    # store back dk and dv
    dk_ptrs = (
        dk_ptr
        + begin * stride_dkn
        + (offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd)
    )

    dv_ptrs = (
        dv_ptr
        + begin * stride_dvn
        + (offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd)
    )

    tl.store(dk_ptrs, dk, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))
    tl.store(dv_ptrs, dv, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))

    start_m = tl.program_id(axis=0) * BLOCK_SIZE_N
    offs_m_curr = start_m + tl.arange(0, BLOCK_SIZE_N)

    dq_ptrs_curr = (
        dq_ptr
        + begin * stride_dqm
        + (offs_m_curr[:, None] * stride_dqm + offs_d[None, :] * stride_dqd)
    )

    dq_curr = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)

    q_ptrs_curr = (
        q_ptr
        + begin * stride_qm
        + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )

    q_curr = tl.load(
        q_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M))
    )

    # Load lse_i
    lse_i_curr = tl.load(lse_ptr + offs_m_curr + begin, mask=offs_m_curr < M)

    do_ptrs_curr = (
        do_ptr
        + begin * stride_dom
        + (offs_m_curr[:, None] * stride_dom + offs_d[None, :] * stride_dod)
    )

    # Load do
    do_curr = tl.load(
        do_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M))
    )
    Di_curr = tl.load(delta_ptr + offs_m_curr + begin, mask=offs_m_curr < M)

    # When computing dV, we want to compute [BLOCK_SIZE_N] rows of dV.
    # Therefore, the loop's block size is BLOCK_SIZE_M instead of BLOCK_SIZE_N.
    block_start = 0
    while block_start < M:
        offs_n_curr = block_start + tl.arange(0, BLOCK_SIZE_M)

        k_ptrs_curr = (
            k_ptr
            + begin * stride_kn
            + (offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        )
        v_ptrs_curr = (
            v_ptr
            + begin * stride_vn
            + (offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        )

        k_curr = tl.load(
            k_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < M))
        )
        v_curr = tl.load(
            v_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < M))
        )

        qk_curr = tl.dot(q_curr, tl.trans(k_curr), allow_tf32=allow_tf32)
        mn_mask_curr = (offs_m_curr[:, None] < M) & (offs_n_curr[None, :] < M)

        p_curr = tl.exp(qk_curr - lse_i_curr[:, None])
        p_curr = tl.where(mn_mask_curr, p_curr, 0.0)
        p_curr /= max_seq_len

        # compute dp = dot(v, do)
        dp_curr = tl.dot(do_curr, tl.trans(v_curr), allow_tf32=allow_tf32)
        dp_curr_masked = dp_curr

        # compute ds = p * (dp - delta[:, None])
        if use_mask:
            attn_mask = offs_m_curr[:, None] - offs_n_curr[None, :]
            attn_mask = tl.where(mn_mask_curr, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            dp_curr_masked = tl.where(attn_mask > 0, dp_curr, 0.0)

        ds_curr = p_curr * (dp_curr_masked - Di_curr[:, None] * max_seq_len)

        ds_curr = ds_curr.to(k_ptr.dtype.element_ty)
        dq_curr += tl.dot(ds_curr, k_curr, allow_tf32=allow_tf32)
        block_start += BLOCK_SIZE_M

    tl.store(
        dq_ptrs_curr, dq_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M))
    )


def jagged_flash_attention_basic_backward(
    jagged_Q,
    # K is non-transposed
    jagged_K,
    jagged_V,
    jagged_O,
    offsets,
    dO,
    lse,
    max_seq_len,
    use_mask,
    allow_tf32=False,
):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32

    B = offsets.size(0) - 1
    num_blocks_m = triton.cdiv(max_seq_len, BLOCK_SIZE_M)
    pre_grid = (num_blocks_m, B)

    BLOCK_SIZE_D = max(triton.next_power_of_2(jagged_Q.size(1)), 16)

    delta = torch.empty_like(lse)
    if not dO.is_contiguous():
        dO = dO.contiguous()

    _jagged_flash_attention_bwd_preprocess_basic_kernel[pre_grid](
        jagged_O,
        offsets,
        dO,
        delta,
        jagged_O.stride(0),
        jagged_O.stride(1),
        max_seq_len,
        jagged_O.size(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_D,
    )

    grid = (triton.cdiv(max_seq_len, BLOCK_SIZE_N), B)

    dq = torch.zeros_like(jagged_Q)
    dk = torch.zeros_like(jagged_K)
    dv = torch.zeros_like(jagged_V)

    D = jagged_Q.size(1)
    _jagged_flash_attention_bwd_basic_kernel[grid](
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
        jagged_K.stride(0),
        jagged_K.stride(1),
        jagged_V.stride(0),
        jagged_V.stride(1),
        jagged_O.stride(0),
        jagged_O.stride(1),
        dq.stride(0),
        dq.stride(1),
        dk.stride(0),
        dk.stride(1),
        dv.stride(0),
        dv.stride(1),
        dO.stride(0),
        dO.stride(1),
        max_seq_len,
        D,
        use_mask=use_mask,
        allow_tf32=allow_tf32,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return dq, dk, dv


class JaggedFlashAttentionBasic(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        jagged_Q: torch.Tensor,
        jagged_K: torch.Tensor,
        jagged_V: torch.Tensor,
        offsets: torch.Tensor,
        max_seq_len: int,
        use_mask: bool = True,
        allow_tf32: bool = False,
    ) -> torch.Tensor:
        ctx.allow_tf32 = allow_tf32
        ctx.max_seq_len = max_seq_len
        ctx.use_mask = use_mask

        jagged_O, lse = jagged_flash_attention_basic_fwd(
            jagged_Q,
            jagged_K.T,
            jagged_V,
            offsets,
            max_seq_len,
            use_mask,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        (
            jagged_Q,
            jagged_K,
            jagged_V,
            offsets,
            jagged_O,
            lse,
        ) = ctx.saved_tensors

        dq, dk, dv = jagged_flash_attention_basic_backward(
            jagged_Q=jagged_Q,
            jagged_K=jagged_K,
            jagged_V=jagged_V,
            jagged_O=jagged_O,
            offsets=offsets,
            dO=grad_output,
            lse=lse,
            max_seq_len=ctx.max_seq_len,
            use_mask=ctx.use_mask,
            allow_tf32=ctx.allow_tf32,
        )

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
        )


def jagged_flash_attention_basic(
    q_weights: torch.Tensor,
    k_weights: torch.Tensor,
    v_weights: torch.Tensor,
    offsets: torch.Tensor,
    max_seq_len: int,
    use_mask: bool = False,
    allow_tf32: bool = True,
) -> torch.Tensor:
    q_weights = expect_contiguous(q_weights)
    k_weights = expect_contiguous(k_weights)
    v_weights = expect_contiguous(v_weights)
    jagged_offsets = expect_contiguous(offsets)

    jagged_O = JaggedFlashAttentionBasic.apply(
        q_weights,
        k_weights,
        v_weights,
        jagged_offsets,
        max_seq_len,
        use_mask,
        allow_tf32,
    )

    return jagged_O
