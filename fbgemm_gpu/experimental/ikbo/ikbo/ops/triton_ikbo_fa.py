# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    if nargs.get("desc_q", None) is None or not isinstance(
        nargs["desc_q"], TensorDescriptor
    ):
        return
    BLOCK_D = nargs["BLOCK_D"]
    nargs["desc_q"].block_shape = [BLOCK_M, BLOCK_D]
    nargs["desc_v"].block_shape = [BLOCK_N, BLOCK_D]
    nargs["desc_k"].block_shape = [BLOCK_N, BLOCK_D]


def _get_fw_configs():
    configs = [
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
            },
            num_stages=ns,
            num_warps=nw,
            pre_hook=_host_descriptor_pre_hook,
        )
        for bm in [16, 32, 64, 128]
        for bn in [16, 32, 64, 128]
        for nw in [4, 8, 16, 32]
        for ns in [1, 2, 3, 4, 5]
    ]
    return configs


@triton.jit  # pragma: no cover
def _attn_fwd_inner_tma(
    output,
    acc,
    l_i,
    m_i,
    q,
    desc_k,
    desc_v,
    pid_head,
    k_stride1,
    v_stride1,
    seq_start_kv,
    seq_end_kv,
    qk_scale,
    allow_tf32,
    BLOCK_N: tl.constexpr,
):
    offset_seq_n = tl.arange(0, BLOCK_N)
    for start_n in tl.range(0, seq_end_kv - seq_start_kv, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([(start_n + seq_start_kv).to(tl.int32), pid_head * k_stride1])
        # for jaggedness and S is not a multiplier of BLOCK_N which affect softmax
        # mask_seq = offset_seq_n[None, :] + start_n - seq_start_kv < max_seq_len
        mask_seq = offset_seq_n[None, :] < seq_end_kv - start_n - seq_start_kv
        qk = tl.dot(q, tl.trans(k), allow_tf32=allow_tf32)
        qk = tl.where(mask_seq, qk, -1.0e10)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # p = tl.math.exp(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        # alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = desc_v.load([(start_n + seq_start_kv).to(tl.int32), pid_head * v_stride1])
        p = p.to(output.dtype.element_ty)
        acc = tl.dot(p, v, acc, allow_tf32=allow_tf32)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return acc, l_i, m_i


@triton.autotune(
    configs=_get_fw_configs(),
    key=["d_model", "q_seq_len"],
)
@triton.jit  # pragma: no cover
def _attn_fwd_tma(
    desc_q,
    desc_k,
    desc_v,
    output,
    cand_to_user_mapping,
    q_stride0,
    q_stride1,
    q_stride2,
    k_stride0,
    k_stride1,
    k_stride2,
    v_stride0,
    v_stride1,
    v_stride2,
    o_stride0,
    o_stride1,
    o_stride2,
    q_seq_len,
    max_seq_len,
    sm_scale,
    d_head,
    allow_tf32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Kernel for computing the attention: output = softmax(Q * K.T * sm_scale) * V
    """
    # map the slow index of query to the fast index of key
    # this is the index of the first element of the row
    # in the jagged tensor
    pid_q_seq = tl.program_id(0)  # Sequence of Q/BLOCK_M
    pid_cand_batch = tl.program_id(
        1
    )  # Batch ads, launch prio higher than head due to Bu can be shared
    pid_head = tl.program_id(2)  # head
    pid_user_batch = tl.load(cand_to_user_mapping + pid_cand_batch)

    seq_start_kv = pid_user_batch * max_seq_len

    O_block_ptr = tl.make_block_ptr(
        base=output + pid_head * o_stride1 + pid_cand_batch * q_seq_len * o_stride0,
        shape=(q_seq_len, d_head),
        strides=(o_stride0, o_stride2),
        offsets=(pid_q_seq * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    # maximum value of the qkT to avoid float overflow
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # sum of the exp(qkT - m_i) to calculate the softmax demoniator
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    qk_scale = sm_scale
    qo_seq_offset = pid_cand_batch * q_seq_len + pid_q_seq * BLOCK_M
    q = desc_q.load([qo_seq_offset.to(tl.int32), pid_head * q_stride1])

    acc, l_i, m_i = _attn_fwd_inner_tma(
        output,
        acc,
        l_i,
        m_i,
        q,
        desc_k,
        desc_v,
        pid_head,
        k_stride1,
        v_stride1,
        seq_start_kv,
        seq_start_kv + max_seq_len,
        qk_scale,
        allow_tf32,
        BLOCK_N,
    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(output.dtype.element_ty), boundary_check=[0, 1])


def triton_flash_attn_ikbo_tma(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cand_to_user_mapping: torch.Tensor,
    q_seq_len: int,
    max_seq_len: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Ba: candidate batch size, Bu: user batch, H: num heads, D: head dim
    query: [Ba * n_seeds, H, D] Dense tensor
    key: [Bu * max_seq_len, H, D] Dense tensor (similar to jagged tensor expression, jagged tensor is for variable seq length)
    value: [Bu * max_seq_len, H, D] Dense tensor
    max_seq_len: int
    cand_to_user_mapping: [Ba] tensor [0, 0, ..., 1, 1, ..., 2, 2, ...] index: cand batch id, value: user batch id
    scale: float
    output: [Ba * n_seeds, H, D] Dense tensor
    """

    sm_scale = scale
    # d_model = query.shape[-1]
    B_seed, H, d_head = query.shape
    Bu_max_seq_len, _, _ = key.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d_head)

    sm_scale = sm_scale / math.log(2.0)
    BLOCK_D = triton.next_power_of_2(d_head)
    output = torch.empty_like(
        query,
    )
    dummy_block = [1, 1]
    desc_q = TensorDescriptor(
        query,
        shape=[B_seed, H * d_head],
        strides=[H * d_head, 1],
        block_shape=dummy_block,
    )
    desc_v = TensorDescriptor(
        value,
        shape=[Bu_max_seq_len, H * d_head],
        strides=[H * d_head, 1],
        block_shape=dummy_block,
    )
    desc_k = TensorDescriptor(
        key,
        shape=[Bu_max_seq_len, H * d_head],
        strides=[H * d_head, 1],
        block_shape=dummy_block,
    )

    def grid(META):
        return (
            triton.cdiv(q_seq_len, META["BLOCK_M"]),
            B_seed // q_seq_len,
            H,
        )

    _attn_fwd_tma[grid](
        desc_q,
        desc_k,
        desc_v,
        output,
        cand_to_user_mapping,
        query.stride(0),
        query.stride(1),
        query.stride(2),  #
        key.stride(0),
        key.stride(1),
        key.stride(2),  #
        value.stride(0),
        value.stride(1),
        value.stride(2),  #
        output.stride(0),
        output.stride(1),
        output.stride(2),  #
        q_seq_len,
        max_seq_len,
        sm_scale,
        # li,
        # mi,
        d_head=d_head,
        allow_tf32=True if query.dtype == torch.float32 else False,
        BLOCK_D=BLOCK_D,
    )

    return output
