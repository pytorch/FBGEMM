# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from simplicial.utils import get_simplicial_tensor_core_tflops
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

SIMPLICIAL_AUTOTUNE = os.getenv("SIMPLICIAL_AUTOTUNE", "0") == "1"


def _get_consumer_config(num_heads):
    # NOTE: dispatch tiling configs of Q based on num_heads
    # Case #1: num_heads = 64, we use one consumer warp group
    # Case #2: num_heads = 128, we use two consumer warp groups
    if num_heads in [64]:
        return 1, 4
    elif num_heads == 128:
        return 2, 8
    else:
        raise ValueError(f"Unsupported num_heads: {num_heads}")


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    # setup block_shapes of TMA descriptors
    BLOCK_SIZE_KV = nargs["BLOCK_SIZE_KV"]
    HEAD_DIM = nargs["HEAD_DIM"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    W1 = nargs["w1"]
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    nargs["desc_k1"].block_shape = [1, W1 * HEAD_DIM]
    nargs["desc_k2"].block_shape = [BLOCK_SIZE_KV, HEAD_DIM]
    nargs["desc_v1"].block_shape = [1, W1 * HEAD_DIM]
    nargs["desc_v2"].block_shape = [BLOCK_SIZE_KV, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]


def get_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_KV": BLOCK_SIZE_KV,
                "NUM_BUFFERS": num_buffers,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        )
        for BLOCK_SIZE_KV in [128]
        for num_buffers in [2]
    ]


@triton.jit
def qk_gemm_online_softmax(
    qk1_tile_rmem,
    k2_tile,
    k2_full,
    k2_empty,
    kv2_phase,
    m_i,
    l_i,
    acc,
    gemm_dtype,
    q_idx,
    kv2_idx,
    BLOCK_SIZE_KV: tl.constexpr,
    HAS_QK_MASK: tl.constexpr,
):
    """Performs GEMM1 for one tile and updates (m, l, acc) using online softmax."""
    tlx.barrier_wait(k2_full, kv2_phase)
    # Transpose K2 tile for GEMM
    k2_tile_t = tlx.local_trans(k2_tile)  # [HEAD_DIM, BLOCK_SIZE_KV]

    # Compute QK scores
    qk = tlx.async_dot(qk1_tile_rmem, k2_tile_t, out_dtype=tl.float32)
    qk = tlx.async_dot_wait(0, qk)

    # Apply mask if needed
    if HAS_QK_MASK:
        kv2_offs = kv2_idx + tl.arange(0, BLOCK_SIZE_KV)
        kv2_mask = kv2_offs <= q_idx
        qk_mask = kv2_mask[None, :]
        qk = tl.where(qk_mask, qk, -1.0e6)

    tlx.barrier_arrive(k2_empty, 1)

    # Online softmax update
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # Convert probabilities to gemm dtype
    p = p.to(gemm_dtype)

    return p, m_ij, l_i, acc


@triton.jit
def pv_gemm(
    cur_probs,
    v1_tile_rmem,
    acc,
    v2_phase,
    v2_full_pre,
    v2_tile_pre,
    v2_empty_pre,
):
    tlx.barrier_wait(v2_full_pre, v2_phase)
    pv2 = tlx.async_dot(cur_probs, v2_tile_pre)
    tlx.async_dot_wait(0, pv2)
    tlx.barrier_arrive(v2_empty_pre, 1)
    pv12 = pv2 * v1_tile_rmem  # [BLOCK_M_SPLIT, HEAD_DIM]
    acc += pv12
    return acc


@triton.jit
def qk_gemm_online_softmax_pv_gemm(
    qk1_tile_rmem,
    v1_tile_rmem,
    acc,
    m_i,
    l_i,
    probs,
    k2_phase,
    v2_phase,
    k2_full,
    k2_tile,
    k2_empty,
    v2_full_pre,
    v2_tile_pre,
    v2_empty_pre,
    gemm_dtype,
    q_idx,
    kv2_idx,
    BLOCK_SIZE_KV: tl.constexpr,
    HAS_QK_MASK: tl.constexpr,
):
    # assume num_buffers = 4
    # k2_buf_id = 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    # k2_phase =  0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0
    # NOTE about the barrier:
    # Barrier will pass when the value of barrier and phase is different
    # Barrier will wait when the value of barrier and phase is the same
    # Barrier is set to 0 initially

    # wait for the K buffer to be populated by the producer
    tlx.barrier_wait(k2_full, k2_phase)
    k2_tile = tlx.local_trans(k2_tile)  # [HEAD_DIM, BLOCK_SIZE_KV]

    qk = tlx.async_dot(qk1_tile_rmem, k2_tile, out_dtype=tl.float32)

    # compute PV GEMM for the preivous P
    tlx.barrier_wait(v2_full_pre, v2_phase)
    pv2 = tlx.async_dot(probs, v2_tile_pre)

    # wait for the current QK GEMM to finish
    qk = tlx.async_dot_wait(1, qk)
    tlx.barrier_arrive(k2_empty, 1)

    # NOTE: mask is only needed for the last iteration of kv2 loop
    if HAS_QK_MASK:
        kv2_offs = kv2_idx + tl.arange(0, BLOCK_SIZE_KV)
        kv2_mask = kv2_offs <= q_idx
        qk_mask = kv2_mask[None, :]
        qk = tl.where(qk_mask, qk, -1.0e6)

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    cur_p = tl.math.exp2(qk - m_ij[:, None])
    l_ij = tl.sum(cur_p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    m_i = m_ij

    # wait for the previous PV GEMM to finish
    tlx.async_dot_wait(0, pv2)
    tlx.barrier_arrive(v2_empty_pre, 1)
    pv12 = pv2 * v1_tile_rmem  # [BLOCK_M_SPLIT, HEAD_DIM]
    acc += pv12

    # update acc with the current QK GEMM
    acc = acc * alpha[:, None]

    cur_p = cur_p.to(gemm_dtype)  # [BLOCK_M_SPLIT, BLOCK_SIZE_KV]
    return cur_p, m_i, l_i, acc


@triton.autotune(
    configs=get_configs(),
    key=["HEAD_DIM", "w1", "w2", "seq_len", "num_heads"],
)
@triton.jit
def _tlx_fwd_kernel(
    desc_q,  # [b, s, k, h]
    desc_k1,  # [b, s, 1, h]
    desc_k2,  # [b, s, 1, h]
    desc_v1,  # [b, s, 1, h]
    desc_v2,  # [b, s, 1, h]
    desc_o,  # [b, s, k, h]
    M_ptr,  # [b, k, s]
    m_stride_b,
    m_stride_k,
    m_stride_s,
    seq_len,
    num_heads,
    w1: tl.constexpr,
    w2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
):
    # allocate SMEM buffers
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    q_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS
    )
    k2_tiles = tlx.local_alloc(
        (BLOCK_SIZE_KV, HEAD_DIM), tlx.dtype_of(desc_k2), NUM_BUFFERS
    )
    v2_tiles = tlx.local_alloc(
        (BLOCK_SIZE_KV, HEAD_DIM), tlx.dtype_of(desc_v2), NUM_BUFFERS
    )

    k1_tiles = tlx.local_alloc((1, HEAD_DIM), tlx.dtype_of(desc_k1), w1)
    v1_tiles = tlx.local_alloc((1, HEAD_DIM), tlx.dtype_of(desc_v1), w1)

    # allocate barriers
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)

    # k1_tiles, v1_tiles will be loaded from GMEM to SMEM once
    k1_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    v1_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    k2_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS
    )
    k2_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v2_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS
    )
    v2_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    num_kv_heads = 1

    with tlx.async_tasks():
        # producer group
        with tlx.async_task("default"):
            # initialize offsets
            q_idx = tl.program_id(0)
            batch_idx = tl.program_id(1)

            kv2_start = tl.maximum(0, q_idx - w2 + 1)
            kv2_end = tl.minimum(seq_len, q_idx + 1)

            kv1_idx_start = tl.maximum(0, q_idx - w1 + 1)
            kv1_idx_end = tl.minimum(seq_len, q_idx + 1)
            num_of_kv1_trips = kv1_idx_end - kv1_idx_start

            # NOTE: kv1_idx_start doesn't need to mul stride
            k1o_offset = (
                batch_idx * (seq_len * num_kv_heads) + kv1_idx_start * num_kv_heads
            )

            # load q: it will stay in SRAM throughout
            for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                q_full = tlx.local_view(q_fulls, cid)
                tlx.barrier_expect_bytes(
                    q_full, 2 * BLOCK_M_SPLIT * HEAD_DIM
                )  # bfloat16
                q_tile = tlx.local_view(q_tiles, cid)
                qo_offset_ysplit = (
                    batch_idx * (seq_len * num_heads)
                    + q_idx * num_heads
                    + cid * BLOCK_M_SPLIT
                )
                tlx.async_descriptor_load(desc_q, q_tile, [qo_offset_ysplit, 0], q_full)

            k1_full = tlx.local_view(k1_fulls, 0)
            tlx.barrier_expect_bytes(k1_full, 2 * w1 * HEAD_DIM)
            k1_tile = tlx.local_view(k1_tiles, 0)
            reinterpreted_k1 = tlx.local_reinterpret(
                k1_tile, tlx.dtype_of(desc_k1), [1, w1 * HEAD_DIM]
            )
            tlx.async_descriptor_load(
                desc_k1, reinterpreted_k1, [0, k1o_offset * HEAD_DIM], k1_full
            )

            v1_full = tlx.local_view(v1_fulls, 0)
            tlx.barrier_expect_bytes(v1_full, 2 * w1 * HEAD_DIM)
            v1_tile = tlx.local_view(v1_tiles, 0)
            reinterpreted_v1 = tlx.local_reinterpret(
                v1_tile, tlx.dtype_of(desc_v1), [1, w1 * HEAD_DIM]
            )
            tlx.async_descriptor_load(
                desc_v1, reinterpreted_v1, [0, k1o_offset * HEAD_DIM], v1_full
            )

            # loop over loading k, v
            kv_phase = 0
            acc_cnt = 0
            kv2_offset_y_start = (
                batch_idx * (seq_len * num_kv_heads) + kv2_start * num_kv_heads
            )

            for _ in tl.range(num_of_kv1_trips):
                kv2_offset_y = kv2_offset_y_start
                for _ in tl.range(kv2_start, kv2_end, BLOCK_SIZE_KV):
                    buf_id = acc_cnt % NUM_BUFFERS
                    # buffers in a row share the same phase
                    kv_phase = kv_phase ^ (buf_id == 0)
                    # 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1
                    # here the phase is for empty

                    # wait for the K buffer to be released by the consumer
                    k_empty = tlx.local_view(k2_empties, buf_id)
                    # pass when the value of barrier and phase is different
                    # wait when the value of barrier and phase is the same
                    tlx.barrier_wait(k_empty, kv_phase)
                    # load K
                    k2_full = tlx.local_view(k2_fulls, buf_id)
                    k2_tile = tlx.local_view(k2_tiles, buf_id)
                    tlx.barrier_expect_bytes(
                        k2_full, 2 * BLOCK_SIZE_KV * HEAD_DIM
                    )  # bfloat16
                    tlx.async_descriptor_load(
                        desc_k2, k2_tile, [kv2_offset_y, 0], k2_full
                    )

                    # wait for the V buffer to be released by the consumer
                    v_empty = tlx.local_view(v2_empties, buf_id)
                    tlx.barrier_wait(v_empty, kv_phase)
                    # load V
                    v2_full = tlx.local_view(v2_fulls, buf_id)
                    v2_tile = tlx.local_view(v2_tiles, buf_id)
                    tlx.barrier_expect_bytes(
                        v2_full, 2 * BLOCK_SIZE_KV * HEAD_DIM
                    )  # bfloat16
                    tlx.async_descriptor_load(
                        desc_v2, v2_tile, [kv2_offset_y, 0], v2_full
                    )

                    kv2_offset_y += BLOCK_SIZE_KV
                    acc_cnt += 1

        # consumer group - pipelined version
        with tlx.async_task(
            num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
            registers=232,
            replicate=NUM_MMA_GROUPS,
        ):
            # prepare offsets
            q_idx = tl.program_id(0)
            batch_idx = tl.program_id(1)

            kv2_start = tl.maximum(0, q_idx - w2 + 1)
            kv2_end = tl.minimum(seq_len, q_idx + 1)

            kv1_idx_start = tl.maximum(0, q_idx - w1 + 1)
            kv1_idx_end = tl.minimum(seq_len, q_idx + 1)
            num_of_kv1_trips = kv1_idx_end - kv1_idx_start

            gemm_dtype = tlx.dtype_of(desc_q)

            # initialize pointer to m and l
            acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)
            l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
            m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")

            # load scales
            softmax_scale = tl.cast(SM_SCALE, tlx.dtype_of(desc_q))

            cid = tlx.async_task_replica_id()

            # wait for Q tile
            q_full = tlx.local_view(q_fulls, cid)
            tlx.barrier_wait(q_full, 0)
            q_tile = tlx.local_view(q_tiles, cid)

            # wait for K1 tile
            k1_full = tlx.local_view(k1_fulls, 0)
            tlx.barrier_wait(k1_full, 0)
            k1_tile = tlx.local_view(k1_tiles, 0)

            # consumer group 0 and consumer group 1 use the same k1 tile
            q_tile_rmem = tlx.local_load(q_tile)
            q_tile_rmem_scaled = q_tile_rmem * softmax_scale

            # NOTE: make sure all v1 tiles are loaded before entering the loop
            v1_full = tlx.local_view(v1_fulls, 0)
            tlx.barrier_wait(v1_full, 0)

            has_qk_mask = (kv2_end - kv2_start) % BLOCK_SIZE_KV > 0
            num_kv2_trips = tl.cdiv(kv2_end - kv2_start, BLOCK_SIZE_KV)
            masked_kv2_idx = num_kv2_trips - 1

            # Section 1: compute Q_tile * K1_tile and [current] QK_GEMM + online softmax
            k2_phase = 1
            v2_phase = 1
            acc_cnt = 0
            kv2_idx = kv2_start

            # Q_tile * K1_tile
            kv1_idx = 0
            k1_tile = tlx.local_view(k1_tiles, kv1_idx)

            k1_tile_rmem = tlx.local_load(k1_tile)
            qk1_tile_rmem = q_tile_rmem_scaled * k1_tile_rmem

            # [current] QK_GEMM + online softmax
            k2_buf_id = acc_cnt % NUM_BUFFERS
            # kv2_phase_cur should be 0
            k2_phase = k2_phase ^ (k2_buf_id == 0)
            k2_full = tlx.local_view(k2_fulls, k2_buf_id)
            k2_tile = tlx.local_view(k2_tiles, k2_buf_id)
            k2_empty = tlx.local_view(k2_empties, k2_buf_id)

            cur_probs, m_i, l_i, acc = qk_gemm_online_softmax(
                qk1_tile_rmem,
                k2_tile,
                k2_full,
                k2_empty,
                k2_phase,
                m_i,
                l_i,
                acc,
                gemm_dtype,
                q_idx,
                kv2_idx,
                BLOCK_SIZE_KV,
                HAS_QK_MASK=has_qk_mask and num_kv2_trips == 1,
            )

            acc_cnt += 1
            kv2_idx += BLOCK_SIZE_KV

            # Section 1: END

            # Pipelined loop over KV1 trips using modularized functions
            for kv1_idx in tl.range(0, num_of_kv1_trips):
                k1_tile = tlx.local_view(k1_tiles, kv1_idx)
                v1_tile = tlx.local_view(v1_tiles, kv1_idx)

                k1_tile_rmem = tlx.local_load(k1_tile)
                v1_tile_rmem = tlx.local_load(v1_tile).to(tl.float32)

                qk1_tile_rmem = q_tile_rmem_scaled * k1_tile_rmem

                if kv1_idx == 0:
                    # for the first KV1 trip, we already pre-compute QK_GEMM + online softmax
                    kv2_idx = kv2_start + BLOCK_SIZE_KV
                    kv2_trip_start = 1
                else:
                    kv2_idx = kv2_start
                    kv2_trip_start = 0

                # Pipelined steady state: use fused he lper for overlapping computation
                for idx in tl.range(kv2_trip_start, num_kv2_trips):
                    # Section 2: compute QK_GEMM + online softmax [current] and PV_GEMM [previous]
                    k2_buf_id = acc_cnt % NUM_BUFFERS
                    k2_phase = k2_phase ^ (k2_buf_id == 0)

                    k2_full = tlx.local_view(k2_fulls, k2_buf_id)
                    k2_tile = tlx.local_view(k2_tiles, k2_buf_id)
                    k2_empty = tlx.local_view(k2_empties, k2_buf_id)

                    v2_buf_id = (acc_cnt - 1) % NUM_BUFFERS
                    v2_phase = v2_phase ^ (v2_buf_id == 0)

                    v2_full_pre = tlx.local_view(v2_fulls, v2_buf_id)
                    v2_tile_pre = tlx.local_view(v2_tiles, v2_buf_id)
                    v2_empty_pre = tlx.local_view(v2_empties, v2_buf_id)

                    # NOTE: for the beginning for-loop of w2, the v1_tile should be from the previous iteration of w1,
                    # instead of the current iteration of w1
                    if idx == 0:
                        _inner_v1_tile = tlx.local_view(v1_tiles, kv1_idx - 1)
                        _inner_v1_tile_rmem = tlx.local_load(_inner_v1_tile).to(
                            tl.float32
                        )
                    else:
                        _inner_v1_tile_rmem = v1_tile_rmem

                    probs, m_i, l_i, acc = qk_gemm_online_softmax_pv_gemm(
                        qk1_tile_rmem,
                        _inner_v1_tile_rmem,
                        acc,
                        m_i,
                        l_i,
                        cur_probs,
                        k2_phase,
                        v2_phase,
                        k2_full,
                        k2_tile,
                        k2_empty,
                        v2_full_pre,
                        v2_tile_pre,
                        v2_empty_pre,
                        gemm_dtype,
                        q_idx,
                        kv2_idx,
                        BLOCK_SIZE_KV,
                        HAS_QK_MASK=has_qk_mask and idx == masked_kv2_idx,
                    )

                    # Update for next iteration
                    cur_probs = probs
                    acc_cnt += 1
                    kv2_idx += BLOCK_SIZE_KV

                    # Section 2: END

            # Section 3: compute PV GEMM for the last iteration

            v2_buf_id = (acc_cnt - 1) % NUM_BUFFERS
            v2_phase = v2_phase ^ (v2_buf_id == 0)

            v2_full_pre = tlx.local_view(v2_fulls, v2_buf_id)
            v2_tile_pre = tlx.local_view(v2_tiles, v2_buf_id)
            v2_empty_pre = tlx.local_view(v2_empties, v2_buf_id)

            # NOTE: for the last iteration, we need to use the last kv1_idx
            kv1_idx = num_of_kv1_trips - 1
            v1_tile = tlx.local_view(v1_tiles, kv1_idx)
            v1_tile_rmem = tlx.local_load(v1_tile).to(tl.float32)

            acc = pv_gemm(
                cur_probs,
                v1_tile_rmem,
                acc,
                v2_phase,
                v2_full_pre,
                v2_tile_pre,
                v2_empty_pre,
            )
            # Section 3: END

            # epilogue
            # store O
            acc = acc / l_i[:, None]
            qo_offset_ysplit = (
                batch_idx * (seq_len * num_heads)
                + q_idx * num_heads
                + cid * BLOCK_M_SPLIT
            )
            desc_o.store([qo_offset_ysplit, 0], acc.to(tlx.dtype_of(desc_o)))

            # store M
            m = m_i + tl.log(l_i)
            m_offs_k = cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
            M_ptr_start = (
                M_ptr
                + batch_idx * m_stride_b
                + m_offs_k * m_stride_k
                + q_idx * m_stride_s
            )
            tl.store(M_ptr_start, m)


def get_tensor_descriptor(tensor):
    bs, seq_len, num_heads, head_dim = tensor.shape

    y_dim = bs * seq_len * num_heads
    dummy_block = [1, 1]

    return TensorDescriptor(
        tensor, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=dummy_block
    )


def tlx_fwd_ws_pipelined(
    q,
    k1,
    k2,
    v1,
    v2,
    w1,
    w2,
):
    bs, seq_len, num_heads, head_dim = q.shape
    _, seq_len1, _, _ = k1.shape
    _, seq_len2, _, _ = k2.shape
    assert (
        seq_len == seq_len1 and seq_len1 == seq_len2
    ), "input seq lens must match, sliding window is done within kernel"
    assert w1 > 0 and w2 > 0, "block local windows must be positive"
    output = torch.zeros_like(q, memory_format=torch.contiguous_format).to(q.dtype)
    m = torch.zeros((bs, num_heads, seq_len), dtype=torch.float32, device=q.device)

    assert w1 <= w2, "w1 must be no larger than w2"

    desc_q = get_tensor_descriptor(q)
    desc_k1 = get_tensor_descriptor(k1)
    desc_k2 = get_tensor_descriptor(k2)
    desc_v1 = get_tensor_descriptor(v1)
    desc_v2 = get_tensor_descriptor(v2)
    desc_o = get_tensor_descriptor(output)

    y_dim = bs * seq_len1 * num_heads
    dummy_block = [1, 1]
    desc_k1 = TensorDescriptor(
        k1,
        shape=[1, y_dim * head_dim],
        strides=[y_dim * head_dim, 1],
        block_shape=dummy_block,
    )
    desc_v1 = TensorDescriptor(
        v1,
        shape=[1, y_dim * head_dim],
        strides=[y_dim * head_dim, 1],
        block_shape=dummy_block,
    )

    def alloc_fn(size: int, align: int, _):
        return torch.empty(
            size, dtype=torch.int8, device=torch.accelerator.current_accelerator()
        )

    triton.set_allocator(alloc_fn)

    # e^x = 2^(x * log2(e)), so we multiply x by log2(e) to use faster exp2 in kernel.
    sm_scale = 1.44269504  # math.log2(math.exp(1))
    sm_scale *= head_dim**-0.5

    def grid(args):
        return (seq_len, bs)

    NUM_MMA_GROUPS, NUM_MMA_WARPS = _get_consumer_config(num_heads)

    _tlx_fwd_kernel[grid](
        desc_q,
        desc_k1,
        desc_k2,
        desc_v1,
        desc_v2,
        desc_o,
        m,
        m.stride(0),
        m.stride(1),
        m.stride(2),
        seq_len,
        num_heads,
        w1,
        w2,
        HEAD_DIM=head_dim,
        SM_SCALE=sm_scale,
        BLOCK_M=num_heads,
        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
        NUM_MMA_WARPS=NUM_MMA_WARPS,
    )
    return output, m


def _run_bench(
    bs,
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    w1,
    w2,
):
    total_tensor_core_tflops = get_simplicial_tensor_core_tflops(
        bs, seq_len, num_heads, num_kv_heads, head_dim, w1, w2
    )
    q = torch.randn(
        (bs, seq_len, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    k1 = torch.randn(
        (bs, seq_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    k2 = torch.randn(
        (bs, seq_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    v1 = torch.randn(
        (bs, seq_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    v2 = torch.randn(
        (bs, seq_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )

    def fn():
        return tlx_fwd_ws_pipelined(q, k1, k2, v1, v2, w1, w2)  # noqa: F821

    ms = triton.testing.do_bench_cudagraph(fn)

    secs = ms * 1e-3

    tensor_core_tflops = total_tensor_core_tflops / secs

    del q, k1, k2, v1, v2

    return ms, tensor_core_tflops


def _bench():
    bs, num_heads, head_dim = 4, 128, 128
    num_kv_heads = 1

    w1 = 32

    w2s = [512]
    seq_lens = [8192]

    for w2 in w2s:
        for seq_len in seq_lens:
            ms, tensor_core_tflops = _run_bench(
                bs, seq_len, num_heads, num_kv_heads, head_dim, w1, w2
            )
            print(
                f"{bs=} {seq_len=} {num_heads=} {head_dim=} {w1=} {w2=} | latency ms={ms:.3f} tensor_core_tflops={tensor_core_tflops:.3f}"
            )


if __name__ == "__main__":
    _bench()
