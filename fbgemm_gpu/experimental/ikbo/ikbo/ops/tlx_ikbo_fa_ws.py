# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import triton

# proton related
import triton.language as tl

try:
    import triton.language.extra.tlx as tlx
except ImportError:
    print("TLX not found!")

from triton.tools.tensor_descriptor import TensorDescriptor


def _host_descriptor_pre_hook_tlx_persistent(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    if nargs.get("desc_q", None) is None or not isinstance(
        nargs["desc_q"], TensorDescriptor
    ):
        return
    BLOCK_D = nargs["BLOCK_D"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]

    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, BLOCK_D]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, BLOCK_D]
    nargs["desc_v"].block_shape = [BLOCK_N, BLOCK_D]
    nargs["desc_k"].block_shape = [BLOCK_N, BLOCK_D]


configs_tlx_persistent = [
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "NUM_BUFFERS": 2,
            "NUM_MMA_WARPS": 8,
            "NUM_MMA_GROUPS": 2,
        },
        num_warps=4,
        num_stages=1,
        pre_hook=_host_descriptor_pre_hook_tlx_persistent,
    ),
]


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


@triton.autotune(
    configs=configs_tlx_persistent,
    key=["d_model", "q_seq_len", "H"],
)
@triton.jit  # pragma: no cover
def _attn_fwd_tlx_tma_pipeline_persistent_general(
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    cand_to_user_mapping,
    cand_grid,
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
    H,
    cand_batch_launch_kernel_instance,
    NUM_SMS,
    num_cand,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """
    Kernel for computing the attention: output = softmax(Q * K.T * sm_scale) * V
    """
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    start_pid = tl.program_id(0)

    num_tiles = (
        (q_seq_len + BLOCK_M_SPLIT - 1)
        // BLOCK_M_SPLIT
        * cand_batch_launch_kernel_instance
        * H
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    num_pid_B_seed = (
        (q_seq_len + BLOCK_M_SPLIT - 1)
        // BLOCK_M_SPLIT
        * cand_batch_launch_kernel_instance
    )

    num_seq = (q_seq_len + BLOCK_M_SPLIT - 1) // BLOCK_M_SPLIT
    # allocate buffers
    q_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_D), tlx.dtype_of(desc_q), NUM_MMA_GROUPS
    )
    k_tiles = tlx.local_alloc((BLOCK_N, BLOCK_D), tlx.dtype_of(desc_k), NUM_BUFFERS)
    v_tiles = tlx.local_alloc((BLOCK_N, BLOCK_D), tlx.dtype_of(desc_v), NUM_BUFFERS)
    o_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_D), tlx.dtype_of(desc_o), NUM_MMA_GROUPS
    )
    # allocate mbarriers
    q_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
    k_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS
    )
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS
    )
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
        # == producer group == #
        with tlx.async_task("default"):
            # initialize offsets
            q0_cnt = 0
            q1_cnt = 1
            kv_cnt = 0
            for i in tl.range(tiles_per_SM):
                # pid needs special taken care of, B=0, B=1 form 2MMA, q_seq grid.x, batch size grid.y, head grid.z

                pid = start_pid + i * NUM_SMS
                pid_q_cand_batch = pid % num_pid_B_seed

                pid_q_seq = pid_q_cand_batch % (num_seq)
                # dummy refers to the dummy index given even number of candidates ranked by users
                pid_cand_batch_dummy = pid_q_cand_batch // num_seq
                pid_head = pid // num_pid_B_seed
                odd_pid_cand = False
                pid_cand_batch = tl.load(cand_grid + pid_cand_batch_dummy)
                pid_user_batch = tl.load(cand_to_user_mapping + pid_cand_batch)

                if pid_cand_batch + 1 >= num_cand:
                    odd_pid_cand = True
                else:
                    pid_user_batch2 = tl.load(cand_to_user_mapping + pid_cand_batch + 1)

                    # odd number batch ads per user
                    if pid_user_batch2 != pid_user_batch:
                        odd_pid_cand = True

                seq_start_kv = pid_user_batch * max_seq_len
                seq_end_kv = seq_start_kv + max_seq_len
                qo_seq_offset = pid_cand_batch * q_seq_len + pid_q_seq * BLOCK_M_SPLIT

                # load q0, k0, then q1, v0, k1, v1, and etc to help loading pipelining
                # q0
                q0_buf_id, q0_phase = _get_bufidx_phase(q0_cnt, NUM_MMA_GROUPS)
                tlx.barrier_wait(q_empties[q0_buf_id], q0_phase ^ 1)
                tlx.barrier_expect_bytes(
                    q_fulls[q0_buf_id], 2 * BLOCK_M_SPLIT * BLOCK_D
                )  # float16
                tlx.async_descriptor_load(
                    desc_q,
                    q_tiles[q0_buf_id],
                    [qo_seq_offset.to(tl.int32), pid_head * q_stride1],
                    q_fulls[q0_buf_id],
                )
                q0_cnt += NUM_MMA_GROUPS

                # k0
                kv_buf_id, kv_phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS)
                tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
                # load K
                tlx.barrier_expect_bytes(
                    k_fulls[kv_buf_id], 2 * BLOCK_N * BLOCK_D
                )  # float16
                tlx.async_descriptor_load(
                    desc_k,
                    k_tiles[kv_buf_id],
                    [seq_start_kv.to(tl.int32), pid_head * k_stride1],
                    k_fulls[kv_buf_id],
                )

                # q1
                if not odd_pid_cand:
                    q1_buf_id, q1_phase = _get_bufidx_phase(q1_cnt, NUM_MMA_GROUPS)
                    tlx.barrier_wait(q_empties[q1_buf_id], q1_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        q_fulls[q1_buf_id], 2 * BLOCK_M_SPLIT * BLOCK_D
                    )  # float16
                    qo_offset_split = (
                        qo_seq_offset + q1_buf_id * BLOCK_M_SPLIT
                    )  # get another batch_ads
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q1_buf_id],
                        [qo_offset_split.to(tl.int32), pid_head * q_stride1],
                        q_fulls[q1_buf_id],
                    )

                    q1_cnt += NUM_MMA_GROUPS

                kv_cnt_start = kv_cnt

                for start_n in tl.range(seq_start_kv + BLOCK_N, seq_end_kv, BLOCK_N):
                    # k1
                    kv_buf_id, kv_phase = _get_bufidx_phase(kv_cnt + 1, NUM_BUFFERS)
                    # wait for the K buffer to be released by the consumer
                    tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        k_fulls[kv_buf_id], 2 * BLOCK_N * BLOCK_D
                    )  # float16
                    tlx.async_descriptor_load(
                        desc_k,
                        k_tiles[kv_buf_id],
                        [start_n.to(tl.int32), pid_head * k_stride1],
                        k_fulls[kv_buf_id],
                    )

                    # v0
                    kv_buf_id, kv_phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS)
                    # wait for the V buffer to be released by the consumer
                    tlx.barrier_wait(v_empties[kv_buf_id], kv_phase ^ 1)
                    # load V
                    tlx.barrier_expect_bytes(
                        v_fulls[kv_buf_id], 2 * BLOCK_N * BLOCK_D
                    )  # float16
                    tlx.async_descriptor_load(
                        desc_v,
                        v_tiles[kv_buf_id],
                        [(start_n - BLOCK_N).to(tl.int32), pid_head * v_stride1],
                        v_fulls[kv_buf_id],
                    )
                    kv_cnt += 1

                start_n = (kv_cnt - kv_cnt_start) * BLOCK_N + seq_start_kv
                kv_buf_id, kv_phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS)
                # wait for the V buffer to be released by the consumer
                tlx.barrier_wait(v_empties[kv_buf_id], kv_phase ^ 1)
                # load V
                tlx.barrier_expect_bytes(
                    v_fulls[kv_buf_id], 2 * BLOCK_N * BLOCK_D
                )  # float16
                tlx.async_descriptor_load(
                    desc_v,
                    v_tiles[kv_buf_id],
                    [start_n.to(tl.int32), pid_head * v_stride1],
                    v_fulls[kv_buf_id],
                )
                kv_cnt += 1

        # == consumer group == #
        with tlx.async_task(
            num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
            registers=232,
            replicate=NUM_MMA_GROUPS,
        ):
            q_cnt = 0
            kv_cnt = 0

            cid = tlx.async_task_replica_id()

            if cid == 1:
                tlx.named_barrier_arrive(9, 256)
            for i in tl.range(tiles_per_SM):
                kv_cnt_start = kv_cnt
                # pid needs special taken care of, B=0, B=1 form 2MMA, q_seq grid.x, batch size grid.y, head grid.z
                pid = start_pid + i * NUM_SMS
                pid_q_cand_batch = pid % num_pid_B_seed

                pid_q_seq = pid_q_cand_batch % (num_seq)
                # dummy refers to the dummy index given even number of candidates ranked by users
                pid_cand_batch_dummy = pid_q_cand_batch // num_seq
                pid_head = pid // num_pid_B_seed
                odd_pid_cand = False
                pid_cand_batch = tl.load(cand_grid + pid_cand_batch_dummy)
                pid_user_batch = tl.load(cand_to_user_mapping + pid_cand_batch)

                if pid_cand_batch + 1 >= num_cand:
                    odd_pid_cand = True
                else:
                    pid_user_batch2 = tl.load(cand_to_user_mapping + pid_cand_batch + 1)

                    # odd number batch ads per user
                    if pid_user_batch2 != pid_user_batch:
                        # Skip this 2nd consumer warpgroup when odd number of ads
                        odd_pid_cand = True

                seq_start_kv = pid_user_batch * max_seq_len
                seq_end_kv = seq_start_kv + max_seq_len

                # Skip 2nd consumer warpgroup when odd number of ads
                if not (odd_pid_cand and cid == 1):
                    m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                    l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
                    acc = tl.zeros([BLOCK_M_SPLIT, BLOCK_D], dtype=tl.float32)

                    # QKT prefetch
                    q_buf_id, q_phase = _get_bufidx_phase(q_cnt, NUM_MMA_GROUPS)
                    tlx.barrier_wait(q_fulls[cid], q_phase)
                    q_cnt += NUM_MMA_GROUPS
                    offset_seq_n = tl.arange(0, BLOCK_N)

                    k_buf_id, k_phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS)
                    tlx.barrier_wait(k_fulls[k_buf_id], k_phase)
                    k_tile = tlx.local_trans(k_tiles[k_buf_id])
                    if not odd_pid_cand:
                        if cid == 0:
                            # Consumer 0 waits for Consumer 1 to be ready (prevents both issuing simultaneously)
                            tlx.named_barrier_wait(9, 256)
                        else:
                            # Consumer 1 waits for Consumer 0 to finish its async_dot
                            tlx.named_barrier_wait(10, 256)

                    qk = tlx.async_dot(q_tiles[cid], k_tile)
                    if not odd_pid_cand:
                        if cid == 0:
                            # Consumer 0 done, signal Consumer 1 to proceed
                            tlx.named_barrier_arrive(10, 256)
                        else:
                            # Consumer 1 done, signal Consumer 0 for next iteration
                            tlx.named_barrier_arrive(9, 256)

                    # wait for the MMA using to complete
                    qk = tlx.async_dot_wait(0, qk)
                    tlx.barrier_arrive(k_empties[k_buf_id], 1)
                    if seq_end_kv - seq_start_kv < BLOCK_N:
                        mask_seq = offset_seq_n[None, :] < seq_end_kv - seq_start_kv
                        qk = tl.where(mask_seq, qk, -1.0e10)
                    if seq_end_kv - seq_start_kv <= BLOCK_N:
                        tlx.barrier_arrive(q_empties[cid], 1)

                    # -- compute m_i and l_i for prefetch with optimization ----
                    m_i = tl.max(qk, 1) * sm_scale
                    qk = qk * sm_scale - m_i[:, None]
                    p = tl.math.exp2(qk)
                    l_i = tl.sum(p, 1)
                    kv_cnt += 1

                    # == K loop ==
                    for start_n in tl.range(
                        seq_start_kv + BLOCK_N, seq_end_kv, BLOCK_N
                    ):
                        k_buf_id, k_phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS)
                        # wait for the K buffer to be populated by the producer
                        tlx.barrier_wait(k_fulls[k_buf_id], k_phase)

                        k_tile = tlx.local_trans(k_tiles[k_buf_id])
                        if not odd_pid_cand:
                            if cid == 0:
                                # Consumer 0 waits for Consumer 1 to be ready (prevents both issuing simultaneously)
                                tlx.named_barrier_wait(9, 256)
                            else:
                                # Consumer 1 waits for Consumer 0 to finish its async_dot
                                tlx.named_barrier_wait(10, 256)
                        qk = tlx.async_dot(q_tiles[cid], k_tile)
                        if not odd_pid_cand:
                            if cid == 0:
                                # Consumer 0 done, signal Consumer 1 to proceed
                                tlx.named_barrier_arrive(10, 256)
                            else:
                                # Consumer 1 done, signal Consumer 0 for next iteration
                                tlx.named_barrier_arrive(9, 256)

                        # compute pv from the previous iteration
                        # wait for the previous V buffer to be populated by the producer
                        v_buf_id, v_phase = _get_bufidx_phase((kv_cnt - 1), NUM_BUFFERS)
                        tlx.barrier_wait(v_fulls[v_buf_id], v_phase)

                        # prepare p and v for the dot
                        p = p.to(tlx.dtype_of(desc_k))
                        acc = tlx.async_dot(p, v_tiles[v_buf_id], acc)

                        # wait for the current qk MMA to complete
                        qk = tlx.async_dot_wait(1, qk)
                        if start_n + BLOCK_N >= seq_end_kv:
                            # release the Q buffer when the last QK is finished
                            tlx.barrier_arrive(q_empties[cid], 1)
                        if start_n + BLOCK_N > seq_end_kv:
                            # masking logic
                            mask_seq = offset_seq_n[None, :] < seq_end_kv - start_n
                            qk = tl.where(mask_seq, qk, -1.0e10)

                        # release the K buffer
                        tlx.barrier_arrive(k_empties[k_buf_id], 1)

                        # -- compute m_i and l_i ----
                        m_ij = tl.maximum(m_i, tl.max(qk, 1) * sm_scale)
                        qk = qk * sm_scale - m_ij[:, None]
                        p = tl.math.exp2(qk)
                        # -- compute correction factor
                        alpha = tl.math.exp2(m_i - m_ij)
                        l_ij = tl.sum(p, 1)
                        # update m_i and l_i
                        l_i = l_i * alpha + l_ij
                        m_i = m_ij

                        # -- update output accumulator --
                        # wait for the previous pv MMA to complete
                        acc = tlx.async_dot_wait(0, acc)
                        # release the V buffer
                        tlx.barrier_arrive(v_empties[v_buf_id], 1)
                        acc = acc * alpha[:, None]
                        kv_cnt += 1

                    # == epilogue ==
                    # compute pv from the last iteration
                    # wait for the V buffer to be populated by the producer
                    v_buf_id, v_phase = _get_bufidx_phase((kv_cnt - 1), NUM_BUFFERS)
                    tlx.barrier_wait(v_fulls[v_buf_id], v_phase)
                    # prepare p and v for the dot
                    p = p.to(tlx.dtype_of(desc_k))
                    acc = tlx.async_dot(p, v_tiles[v_buf_id], acc)

                    # Overlap reciprocal operation (CUDA core) of li with the epilogue MMA
                    rcp_l_i = 1.0 / l_i

                    # wait for the MMA using to complete
                    acc = tlx.async_dot_wait(0, acc)
                    # release the V buffer
                    tlx.barrier_arrive(v_empties[v_buf_id], 1)

                    qo_seq_offset = (
                        pid_cand_batch * q_seq_len + pid_q_seq * BLOCK_M_SPLIT
                    )
                    qo_offset_split = (
                        qo_seq_offset + cid * BLOCK_M_SPLIT
                    )  # get another batch_ads

                    # replace acc/li by decouple it with reciprocal and multiply, multiply is faster than divide
                    acc = acc * rcp_l_i[:, None]

                    # == store output (async) ==
                    output = acc.to(tlx.dtype_of(desc_o))
                    tlx.async_descriptor_store_wait(0)
                    tlx.local_store(o_tiles[cid], output)
                    tlx.fence_async_shared()
                    tlx.async_descriptor_store(
                        desc_o,
                        o_tiles[cid],
                        [qo_offset_split.to(tl.int32), pid_head * o_stride1],
                    )

                # Always advance counters to stay in sync with producer,
                # even when skipping the computation for consumer 2.
                # kv buffers are shared between both warp groups so phase
                # must be tracked even when one group is idle.
                if odd_pid_cand and cid == 1:
                    # drain first K tile (producer loads k0 before the inner loop)
                    k_buf_id, k_phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS)
                    tlx.barrier_wait(k_fulls[k_buf_id], k_phase)
                    tlx.barrier_arrive(k_empties[k_buf_id], 1)
                    kv_cnt += 1
                    # drain inner-loop K/V tiles
                    for _ in tl.range(seq_start_kv + BLOCK_N, seq_end_kv, BLOCK_N):
                        # K tile (next buffer)
                        k_buf_id, k_phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS)
                        tlx.barrier_wait(k_fulls[k_buf_id], k_phase)
                        tlx.barrier_arrive(k_empties[k_buf_id], 1)
                        # V tile (current buffer)
                        v_buf_id, v_phase = _get_bufidx_phase(kv_cnt - 1, NUM_BUFFERS)
                        tlx.barrier_wait(v_fulls[v_buf_id], v_phase)
                        tlx.barrier_arrive(v_empties[v_buf_id], 1)
                        kv_cnt += 1

                    # drain last V tile
                    v_buf_id, v_phase = _get_bufidx_phase(kv_cnt - 1, NUM_BUFFERS)
                    tlx.barrier_wait(v_fulls[v_buf_id], v_phase)
                    tlx.barrier_arrive(v_empties[v_buf_id], 1)


def tlx_flash_attn_ikbo_tma_persistent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cand_to_user_mapping: torch.Tensor,
    q_seq_len: int,
    max_seq_len: int,
    cand_grid: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Ba: candidate batch size, Bu: user batch, H: num heads, D: head dim
    query: [Ba * n_seeds, H, D] Dense tensor
    key: [Bu * max_seq_len, H, D] Dense tensor (similar to jagged tensor expression, jagged tensor is for variable seq length)
    value: [Bu * max_seq_len, H, D] Dense tensor
    max_seq_len: int
    cand_to_user_mapping: [Ba] tensor [0, 0, ..., 1, 1, ..., 2, 2, ...] index: cand batch id, value: user batch id
    cand_grid: a tensor to tell how many q_iterations need to be launched considering odd number of candidates ranked by users
    scale: float
    output: [Ba * n_seeds, H, D] Dense tensor
    """

    sm_scale = scale
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
    desc_o = TensorDescriptor(
        output,
        shape=[B_seed, H * d_head],
        strides=[H * d_head, 1],
        block_shape=dummy_block,
    )
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    cand_batch_launch_kernel_instance = cand_grid.shape[0]

    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(q_seq_len, META["BLOCK_M"] // META["NUM_MMA_GROUPS"])
                * cand_batch_launch_kernel_instance
                * H,
            ),
        )

    _attn_fwd_tlx_tma_pipeline_persistent_general[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        cand_to_user_mapping,
        cand_grid,
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
        H,
        cand_batch_launch_kernel_instance,
        NUM_SMS=NUM_SMS,
        num_cand=B_seed // q_seq_len,
        BLOCK_D=BLOCK_D,
    )

    return output
