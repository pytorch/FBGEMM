# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
# TLX's type hints do not model constexpr arguments (plain ints / bools passed where `constexpr` is annotated) or
# indexing into barrier / buffer arrays; those are the only errors pyrefly reports for this kernel file.

# TLX IKBO flash attention for Blackwell (sm_100 / sm_103), 2-CTA kernel: the default for users with more than ~10
# candidates. Entry point and kernel choice: tlx_ikbo_fa_blackwell.py; problem statement and tile table:
# tlx_ikbo_fa_blackwell_1cta.py.
#
# Based on the 2-CTA path of tlx.ops flash_attn/sm100.py (the FA4 pipeline). A work tile is a CTA pair: up to 512 Q
# rows of one (user, head), all attending to the same K/V. Rows [256 g + 128 r, + 128) of a tile belong to MMA group g
# (0 or 1) on CTA rank r. The pair's collective (cta_group::2) MMAs halve each SM's K/V SMEM traffic, which is what
# limits the 1-CTA pipeline (GB300 dense fp16: 1755 TF 2-CTA vs 1513 TF 1-CTA; cuDNN 1789 TF).
#
# Tile kinds, from the tile's valid row count (_tile_kind):
#   full  (> 256 rows)   both groups run over all KV blocks.
#   half  (<= 256 rows)  only group 0 has rows. With an even number of KV blocks the tile is "split": both groups take
#                        group 0's rows, group 0 over the first half of the KV blocks and group 1 over the second, and
#                        the correction warpgroup merges the two partial results. Otherwise group 1 is skipped.
#
# Warp roles per CTA (persistent; tiles are claimed with cluster launch control, CLC):
#   load (1 warp)          TMA-loads Q per group and K / V blocks into a NUM_BUFFERS_KV-deep ring; L2-prefetches the
#                          Q of a tile Q_PREFETCH_DIST tiles ahead.
#   MMA (1 warp, leader)   S = Q K^T into TMEM, then O += P V once the softmax has written P.
#   softmax (2 x 4 warps)  one warpgroup per MMA group: online softmax of S in registers, P back to TMEM over S.
#   correction (4 warps)   rescales O when a row maximum grows, normalizes O at the end of a tile (or merges a split
#                          tile) into SMEM for the epilogue. Also the CLC producer.
#   epilogue (1 warp)      TMA-stores O: only the tile's valid rows, in whole 64-row candidates.
# CLC_LEAD > 0 (used with tables that put half tiles last) claims the next tile CLC_LEAD KV blocks before the end of
# the current one.

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.cuda.inline_ptx_lib import _mul_f32x2
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

from .tlx_ikbo_fa_blackwell_1cta import (
    _ikbo_softmax_loop,
    _num_sms,
    _rescale_acc_slices,
)


@triton.jit
def _tile_coords(tile_info, tile):
    """(head, q_row0, kv_row0) of a pair tile; tile_info is int32 [T, 4] = (q_row0, kv_row0, valid_rows, head)."""
    base = tile_info + tile * 4
    return tl.load(base + 3), tl.load(base), tl.load(base + 1)


@triton.jit
def _tile_kind(
    tile_info, tile, seq_len, BLOCK_N: tl.constexpr, HALF_ROWS: tl.constexpr
):
    """(group 1 has rows, tile is split). Every warp role derives the tile kind here (the MMA warp's next_rows test is
    the same predicate): roles that disagree deadlock on the barrier protocol."""
    valid_rows = tl.load(tile_info + tile * 4 + 2)
    split = (valid_rows <= HALF_ROWS) and ((seq_len + BLOCK_N - 1) // BLOCK_N % 2 == 0)
    return valid_rows > HALF_ROWS, split


@triton.jit
def _pv_2cta(
    p_tiles,
    v_tile,
    acc_tiles,
    acc_fulls,
    p_fulls,
    GROUP: tl.constexpr,
    qk_phase,
    use_acc,
    last_bars,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_KV: tl.constexpr,
):
    """O[GROUP] += P[GROUP] V for one KV block, in the softmax's three P chunks (1/2, 1/4, 1/4 of BLOCK_N), each
    issued as soon as its chunk is ready. The first chunk's barrier (acc_fulls) also means "O rescaled". last_bars
    are signalled when all three MMAs complete."""
    tlx.barrier_wait(acc_fulls[GROUP], qk_phase)
    tlx.async_dot(
        p_tiles[2 * GROUP],
        tlx.local_slice(v_tile, [0, 0], [BLOCK_N // 2, HEAD_DIM_KV]),
        acc_tiles[GROUP],
        use_acc=use_acc,
        force_async=True,
        two_ctas=True,
    )
    tlx.barrier_wait(p_fulls[3 * GROUP + 1], qk_phase)
    tlx.async_dot(
        tlx.subslice(p_tiles[2 * GROUP + 1], 0, BLOCK_N // 4),
        tlx.local_slice(v_tile, [BLOCK_N // 2, 0], [BLOCK_N // 4, HEAD_DIM_KV]),
        acc_tiles[GROUP],
        use_acc=True,
        force_async=True,
        two_ctas=True,
    )
    tlx.barrier_wait(p_fulls[3 * GROUP + 2], qk_phase)
    tlx.async_dot(
        tlx.subslice(p_tiles[2 * GROUP + 1], BLOCK_N // 4, BLOCK_N // 4),
        tlx.local_slice(v_tile, [BLOCK_N * 3 // 4, 0], [BLOCK_N // 4, HEAD_DIM_KV]),
        acc_tiles[GROUP],
        use_acc=True,
        mBarriers=last_bars,
        force_async=True,
        two_ctas=True,
    )


@triton.jit
def _claim_next_tile(clc_context, clc_phase, tile_info, NUM_CTAS: tl.constexpr):
    """CLC: the next tile id (-1 when done) and its valid row count."""
    next_tile_id = tlx.clc_consumer(clc_context, clc_phase, multi_ctas=True)
    clc_phase ^= 1
    next_rows = tl.load(tile_info + (tl.maximum(next_tile_id, 0) // NUM_CTAS) * 4 + 2)
    return next_tile_id, clc_phase, next_rows


@triton.jit
def _tma_load(
    desc, tiles, fulls, empties, buf, phase, coords, is_leader, BYTES: tl.constexpr
):
    """Wait for ring slot buf to be free, then load it. The pair's consumers wait on the leader CTA's barrier, which
    expects the bytes of both CTAs."""
    tlx.barrier_wait(empties[buf], phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(fulls[buf], BYTES)
    tlx.async_descriptor_load(
        desc, tiles[buf], coords, fulls[buf], two_ctas=tl.constexpr(True)
    )


@triton.jit
def _ikbo_fa_2cta_kernel(
    sm_scale,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_CTAS: tl.constexpr = 1,
    tile_info=None,
    desc_o64=None,
    HAS_TAIL: tl.constexpr = False,
    CLC_LEAD: tl.constexpr = 0,
    num_tiles=0,
    Q_PREFETCH_DIST=0,
):
    tl.static_assert(NUM_CTAS == 2)
    rank = tlx.cluster_cta_rank()
    is_leader = rank % 2 == 0

    GROUP_ROWS: tl.constexpr = BLOCK_M // 2  # Q rows per MMA group per CTA
    HALF_ROWS: tl.constexpr = NUM_CTAS * GROUP_ROWS  # rows of group 0 across the pair
    NUM_GROUPS: tl.constexpr = 2
    # The collective MMA's B operand is split across the pair: each CTA holds half of the K rows and half of the V
    # columns of a KV block.
    BLOCK_N_KV: tl.constexpr = BLOCK_N // NUM_CTAS
    HEAD_DIM_KV: tl.constexpr = HEAD_DIM // NUM_CTAS
    Q_BYTES: tl.constexpr = (
        tlx.size_of(tlx.dtype_of(desc_q)) * GROUP_ROWS * HEAD_DIM * NUM_CTAS
    )
    K_BYTES: tl.constexpr = (
        tlx.size_of(tlx.dtype_of(desc_k)) * BLOCK_N_KV * HEAD_DIM * NUM_CTAS
    )
    V_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v)) * BLOCK_N * HEAD_DIM

    start_pid = tl.program_id(0)

    # SMEM: Q per group, O per group, K / V ring. Q, O and a 3-deep ring fill the 2-CTA SMEM budget.
    tl.static_assert(NUM_BUFFERS_KV <= 3)
    q_tiles = tlx.local_alloc((GROUP_ROWS, HEAD_DIM), tlx.dtype_of(desc_q), NUM_GROUPS)
    o_tiles = tlx.local_alloc((GROUP_ROWS, HEAD_DIM), tlx.dtype_of(desc_o), NUM_GROUPS)
    k_tiles = tlx.local_alloc(
        (BLOCK_N_KV, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV
    )
    v_tiles = tlx.local_alloc(
        (BLOCK_N, HEAD_DIM_KV), tlx.dtype_of(desc_v), NUM_BUFFERS_KV
    )
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_GROUPS)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_GROUPS)
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    o_empties = tlx.alloc_barriers(num_barriers=NUM_GROUPS)

    # TMEM: S (fp32) per group, overwritten in place by P (fp16 / bf16) in two halves; O (fp32) per group.
    qk_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    qk_tiles = tlx.local_alloc(
        (GROUP_ROWS, BLOCK_N),
        tl.float32,
        NUM_GROUPS,
        tlx.storage_kind.tmem,
        reuse=qk_storage_alias,
    )
    p_tiles = tlx.local_alloc(
        (GROUP_ROWS, BLOCK_N // 2),
        tlx.dtype_of(desc_v),
        NUM_GROUPS * 2,
        tlx.storage_kind.tmem,
        reuse=qk_storage_alias,
    )
    qk_storage_alias.set_buffer_overlap(
        tlx.reuse_group(
            qk_tiles,
            tlx.reuse_group(p_tiles, group_size=2),
            group_type=tlx.reuse_group_type.shared,
        )
    )
    acc_tiles = tlx.local_alloc(
        (GROUP_ROWS, HEAD_DIM), tl.float32, NUM_GROUPS, tlx.storage_kind.tmem
    )
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_GROUPS)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_GROUPS)

    # Barriers the MMA waits on are cluster-wide (both CTAs arrive on the leader's). The softmax publishes P in chunks
    # of 1/2, 1/4 and 1/4: the first chunk shares acc_fulls with the correction's "O rescaled", p_fulls[3 g + 1] and
    # p_fulls[3 g + 2] signal the other two.
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_GROUPS * 3, arrive_count=NUM_CTAS)
    acc_fulls = tlx.alloc_barriers(num_barriers=NUM_GROUPS, arrive_count=2 * NUM_CTAS)
    # Softmax -> correction, within a CTA: each KV block the softmax stores alpha (the O rescale factor) in alpha_smem
    # and arrives on named barrier ALPHA_BAR_ID + g; the correction arrives on alpha_empties after reading it. At the
    # end of a tile the same slot carries the row sum l (and m_smem the row max m, for split tiles).
    ALPHA_BAR_ID: tl.constexpr = 12
    alpha_smem = tlx.local_alloc((GROUP_ROWS, 1), tl.float32, NUM_GROUPS)
    m_smem = tlx.local_alloc((GROUP_ROWS, 1), tl.float32, NUM_GROUPS)
    alpha_empties = tlx.alloc_warp_barrier(num_barriers=NUM_GROUPS, num_warps=4)
    o_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_GROUPS, num_warps=4)

    # CLC consumers per CTA: correction, 2 softmax groups, MMA, load, epilogue.
    clc_context = tlx.clc_create_context(
        num_consumers=(4 + NUM_GROUPS) * NUM_CTAS, num_stages=1
    )
    # Remote arrives need every mbarrier init visible cluster-wide (after all allocations, incl. CLC's).
    tlx.fence_mbarrier_init_cluster()

    with tlx.async_tasks(exclusive=True):
        # ---------------- correction (default, 4 warps) ----------------
        with tlx.async_task("default", registers=120):
            acc_phase0 = 0
            acc_phase1 = 0
            o_phase0 = start_pid * 0
            o_phase1 = 0
            tile_id = start_pid
            clc_phase_producer = 1
            clc_phase_consumer = 0
            while tile_id != -1:
                if CLC_LEAD == 0:
                    tlx.clc_producer(clc_context, clc_phase_producer, multi_ctas=True)
                    clc_phase_producer ^= 1
                g1_rows, split = _tile_kind(
                    tile_info, tile_id // NUM_CTAS, seq_len, BLOCK_N, HALF_ROWS
                )
                g1_active = g1_rows or split
                # a split tile's groups each cover half of the KV blocks
                kv_end = tl.where(
                    split, (seq_len + BLOCK_N - 1) // BLOCK_N // 2 * BLOCK_N, seq_len
                )
                for start_n in tl.range(0, kv_end, BLOCK_N):
                    if CLC_LEAD > 0:
                        if start_n // BLOCK_N == tl.maximum(
                            (kv_end + BLOCK_N - 1) // BLOCK_N - CLC_LEAD, 0
                        ):
                            tlx.clc_producer(
                                clc_context, clc_phase_producer, multi_ctas=True
                            )
                            clc_phase_producer ^= 1
                    for g in tl.static_range(0, NUM_GROUPS):
                        if g == 0 or g1_active:
                            tlx.named_barrier_wait(ALPHA_BAR_ID + g, 256)
                            alpha = tlx.local_load(alpha_smem[g])
                            tlx.barrier_arrive(alpha_empties[g])
                            # The first PV of a tile overwrites O, so there is nothing to rescale. Each warp decides
                            # for its own 32 rows (as in FA4): rescale only if some row's alpha dropped below 1.
                            if start_n != 0:
                                should_rescale = tlx.warp_any(alpha < 1.0)
                                tlx.warp_predicate(
                                    should_rescale,
                                    (),
                                    _rescale_acc_slices,
                                    args=(acc_tiles[g], alpha, HEAD_DIM, 2),
                                    wave_uniform=True,
                                )
                            tlx.barrier_arrive(acc_fulls[g], 1, remote_cta_rank=0)

                if split:
                    # Merge the two KV halves of the same rows: O = (O0 w0 + O1 w1) / (l0 w0 + l1 w1),
                    # w_g = 2^((m_g - max(m0, m1)) * scale).
                    qk_scale = (
                        sm_scale * 1.44269504
                    )  # log2(e): the kernel works in exp2
                    tlx.named_barrier_wait(ALPHA_BAR_ID + 0, 256)
                    l0 = tlx.local_load(alpha_smem[0])
                    m0 = tlx.local_load(m_smem[0])
                    tlx.barrier_arrive(alpha_empties[0])
                    tlx.named_barrier_wait(ALPHA_BAR_ID + 1, 256)
                    l1 = tlx.local_load(alpha_smem[1])
                    m1 = tlx.local_load(m_smem[1])
                    tlx.barrier_arrive(alpha_empties[1])
                    m = tl.maximum(m0, m1)
                    w0 = tl.math.exp2((m0 - m) * qk_scale)
                    w1 = tl.math.exp2((m1 - m) * qk_scale)
                    inv = 1 / (l0 * w0 + l1 * w1)
                    w0 = w0 * inv
                    w1 = w1 * inv
                    tlx.barrier_wait(acc_empties[0], acc_phase0)
                    tlx.barrier_wait(acc_empties[1], acc_phase1)
                    tlx.barrier_wait(o_empties[0], o_phase0 ^ 1)
                    for half in tl.static_range(0, 2):
                        o0 = tlx.local_load(
                            tlx.subslice(
                                acc_tiles[0], HEAD_DIM * half // 2, HEAD_DIM // 2
                            )
                        )
                        o1 = tlx.local_load(
                            tlx.subslice(
                                acc_tiles[1], HEAD_DIM * half // 2, HEAD_DIM // 2
                            )
                        )
                        merged = _mul_f32x2(o0, w0) + _mul_f32x2(o1, w1)
                        tlx.local_store(
                            tlx.local_slice(
                                o_tiles[0],
                                [0, HEAD_DIM * half // 2],
                                [GROUP_ROWS, HEAD_DIM // 2],
                            ),
                            merged.to(tlx.dtype_of(desc_o)),
                        )
                    tlx.barrier_arrive(o_fulls[0])
                else:
                    for g in tl.static_range(0, NUM_GROUPS):
                        if g == 0 or g1_rows:
                            # O / l into SMEM for the epilogue
                            tlx.named_barrier_wait(ALPHA_BAR_ID + g, 256)
                            row_sum = tlx.local_load(alpha_smem[g])
                            tlx.barrier_arrive(alpha_empties[g])
                            acc_phase = acc_phase0 if g == 0 else acc_phase1
                            o_phase = o_phase0 if g == 0 else o_phase1
                            tlx.barrier_wait(acc_empties[g], acc_phase)
                            tlx.barrier_wait(o_empties[g], o_phase ^ 1)
                            scale = 1 / row_sum
                            for half in tl.static_range(0, 2):
                                o = tlx.local_load(
                                    tlx.subslice(
                                        acc_tiles[g],
                                        HEAD_DIM * half // 2,
                                        HEAD_DIM // 2,
                                    )
                                )
                                o = _mul_f32x2(o, scale)
                                o = o.to(tlx.dtype_of(desc_o))
                                tlx.local_store(
                                    tlx.local_slice(
                                        o_tiles[g],
                                        [0, HEAD_DIM * half // 2],
                                        [GROUP_ROWS, HEAD_DIM // 2],
                                    ),
                                    o,
                                )
                            tlx.barrier_arrive(o_fulls[g])

                acc_phase0 ^= 1
                if g1_active:
                    acc_phase1 ^= 1
                if g1_rows:
                    o_phase1 ^= 1
                o_phase0 ^= 1
                tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=True
                )
                clc_phase_consumer ^= 1

        # ---------------- softmax (2 x 4 warps, one warpgroup per MMA group) ----------------
        with tlx.async_task(num_warps=4, registers=184, replicate=NUM_GROUPS):
            accum_cnt_qk = 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            g = tlx.async_task_replica_id()
            while tile_id != -1:
                g1_rows, split = _tile_kind(
                    tile_info, tile_id // NUM_CTAS, seq_len, BLOCK_N, HALF_ROWS
                )
                if g == 0 or g1_rows or split:
                    # Running row max and row sum. l starts at 1 but is multiplied by alpha = 0 at the first block.
                    m_i = tl.zeros([GROUP_ROWS], dtype=tl.float32) - float("inf")
                    l_i = tl.zeros([GROUP_ROWS], dtype=tl.float32) + 1.0
                    qk_scale = sm_scale
                    qk_scale *= 1.44269504
                    p_dtype = tlx.dtype_of(desc_v)
                    # The first alpha store of this tile reuses the slot that held the previous tile's l.
                    tlx.barrier_wait(
                        alpha_empties[g], (accum_cnt_qk + tile_count + 1) & 1
                    )
                    half_blocks = (seq_len + BLOCK_N - 1) // BLOCK_N // 2
                    kv_lo = tl.where(split, g * half_blocks * BLOCK_N, 0)
                    kv_hi = tl.minimum(
                        tl.where(split, (g + 1) * half_blocks * BLOCK_N, seq_len),
                        seq_len,
                    )
                    m_i, l_i, accum_cnt_qk = _ikbo_softmax_loop(
                        qk_fulls,
                        qk_tiles,
                        p_fulls,
                        acc_fulls,
                        p_tiles,
                        alpha_empties,
                        alpha_smem,
                        g,
                        accum_cnt_qk,
                        tile_count,
                        qk_scale,
                        m_i,
                        l_i,
                        seq_len,
                        p_dtype,
                        BLOCK_N,
                        2,
                        ALPHA_BAR_ID=ALPHA_BAR_ID,
                        USE_2CTA=True,
                        HAS_TAIL=HAS_TAIL,
                        SPLIT_KV=True,
                        kv_lo=kv_lo,
                        kv_hi=kv_hi,
                    )
                    # The last iteration waited for alpha_empties, so the alpha slot is free for l.
                    tlx.local_store(alpha_smem[g], l_i[:, None])
                    tlx.local_store(m_smem[g], m_i[:, None])
                    tlx.named_barrier_arrive(ALPHA_BAR_ID + g, 256)
                    tile_count += 1
                tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=True
                )
                clc_phase_consumer ^= 1

        # ---------------- MMA (1 warp; the leader CTA issues the pair's MMAs) ----------------
        with tlx.async_task(num_warps=1, registers=24):
            tlx.fence("async_shared_cluster")
            accum_cnt_kv = 0
            accum_cnt_qk = 0
            accum_cnt_qk1 = 0
            tile_count = 0
            tile_count1 = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            next_rows = tl.load(tile_info + (start_pid // NUM_CTAS) * 4 + 2)
            half_blocks = (seq_len + BLOCK_N - 1) // BLOCK_N // 2
            while tile_id != -1:
                if is_leader:
                    # Unsplit tiles (full, or half with an odd KV block count): S(0), then per block PV(j - 1) and S(j)
                    # for each group, then the last PV. next_rows is the valid row count of tile_id.
                    while (tile_id != -1) & (
                        (next_rows > HALF_ROWS)
                        | ((seq_len + BLOCK_N - 1) // BLOCK_N % 2 == 1)
                    ):
                        g1_rows = next_rows > HALF_ROWS
                        q_buf, q_phase = get_bufidx_phase(tile_count, 1)
                        q_buf1, q_phase1 = get_bufidx_phase(tile_count1, 1)
                        q_buf1 += 1
                        k_buf, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)

                        tlx.barrier_wait(k_fulls[k_buf], k_phase)
                        tlx.barrier_wait(q_fulls[q_buf], q_phase)
                        k_tile = tlx.local_trans(k_tiles[k_buf])
                        tlx.async_dot(
                            q_tiles[q_buf],
                            k_tile,
                            qk_tiles[0],
                            use_acc=False,
                            mBarriers=[qk_fulls[0]],
                            two_ctas=True,
                        )
                        if BLOCK_N >= seq_len:
                            tlx.tcgen05_commit(q_empties[q_buf], two_ctas=True)
                        if g1_rows:
                            tlx.barrier_wait(q_fulls[q_buf1], q_phase1)
                            tlx.async_dot(
                                q_tiles[q_buf1],
                                k_tile,
                                qk_tiles[1],
                                use_acc=False,
                                mBarriers=[qk_fulls[1], k_empties[k_buf]],
                                two_ctas=True,
                            )
                            if BLOCK_N >= seq_len:
                                tlx.tcgen05_commit(q_empties[q_buf1], two_ctas=True)
                        else:
                            tlx.tcgen05_commit(k_empties[k_buf], two_ctas=True)

                        _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
                        _, qk_phase1 = get_bufidx_phase(accum_cnt_qk1, 1)
                        # Claim the next tile mid-tile so the tile boundary does not wait on CLC.
                        if CLC_LEAD == 0:
                            next_tile_id, clc_phase_consumer, next_rows = (
                                _claim_next_tile(
                                    clc_context, clc_phase_consumer, tile_info, NUM_CTAS
                                )
                            )

                        use_acc = False
                        for i in tl.range(BLOCK_N, seq_len, BLOCK_N):
                            v_buf = k_buf
                            v_phase = k_phase
                            kv_wrap = k_buf == NUM_BUFFERS_KV - 1
                            k_buf = 0 if kv_wrap else k_buf + 1
                            k_phase = k_phase ^ 1 if kv_wrap else k_phase
                            tlx.barrier_wait(k_fulls[k_buf], k_phase)
                            k_tile = tlx.local_trans(k_tiles[k_buf])

                            tlx.barrier_wait(v_fulls[v_buf], v_phase)
                            _pv_2cta(
                                p_tiles,
                                v_tiles[v_buf],
                                acc_tiles,
                                acc_fulls,
                                p_fulls,
                                0,
                                qk_phase,
                                use_acc,
                                [],
                                BLOCK_N,
                                HEAD_DIM_KV,
                            )
                            tlx.async_dot(
                                q_tiles[q_buf],
                                k_tile,
                                qk_tiles[0],
                                use_acc=False,
                                mBarriers=[qk_fulls[0]],
                                two_ctas=True,
                            )
                            if i + BLOCK_N >= seq_len:
                                tlx.tcgen05_commit(q_empties[q_buf], two_ctas=True)

                            if g1_rows:
                                _pv_2cta(
                                    p_tiles,
                                    v_tiles[v_buf],
                                    acc_tiles,
                                    acc_fulls,
                                    p_fulls,
                                    1,
                                    qk_phase1,
                                    use_acc,
                                    [v_empties[v_buf]],
                                    BLOCK_N,
                                    HEAD_DIM_KV,
                                )
                                tlx.async_dot(
                                    q_tiles[q_buf1],
                                    k_tile,
                                    qk_tiles[1],
                                    use_acc=False,
                                    mBarriers=[qk_fulls[1], k_empties[k_buf]],
                                    two_ctas=True,
                                )
                                if i + BLOCK_N >= seq_len:
                                    tlx.tcgen05_commit(q_empties[q_buf1], two_ctas=True)
                                qk_phase1 = qk_phase1 ^ 1
                                accum_cnt_qk1 += 1
                            else:
                                tlx.tcgen05_commit(v_empties[v_buf], two_ctas=True)
                                tlx.tcgen05_commit(k_empties[k_buf], two_ctas=True)
                            use_acc = True
                            qk_phase = qk_phase ^ 1
                            accum_cnt_qk += 1
                            accum_cnt_kv += 1

                        # last KV block: PV only, then hand O to the correction
                        v_buf = k_buf
                        v_phase = k_phase
                        tlx.barrier_wait(v_fulls[v_buf], v_phase)
                        _pv_2cta(
                            p_tiles,
                            v_tiles[v_buf],
                            acc_tiles,
                            acc_fulls,
                            p_fulls,
                            0,
                            qk_phase,
                            use_acc,
                            [],
                            BLOCK_N,
                            HEAD_DIM_KV,
                        )
                        tlx.tcgen05_commit(acc_empties[0], two_ctas=True)
                        if g1_rows:
                            _pv_2cta(
                                p_tiles,
                                v_tiles[v_buf],
                                acc_tiles,
                                acc_fulls,
                                p_fulls,
                                1,
                                qk_phase1,
                                use_acc,
                                [acc_empties[1], v_empties[v_buf]],
                                BLOCK_N,
                                HEAD_DIM_KV,
                            )
                            accum_cnt_qk1 += 1
                            tile_count1 += 1
                        else:
                            tlx.tcgen05_commit(v_empties[v_buf], two_ctas=True)

                        accum_cnt_qk += 1
                        accum_cnt_kv += 1
                        if CLC_LEAD > 0:
                            next_tile_id, clc_phase_consumer, next_rows = (
                                _claim_next_tile(
                                    clc_context, clc_phase_consumer, tile_info, NUM_CTAS
                                )
                            )
                        tile_count += 1
                        tile_id = next_tile_id

                    # One split half tile: both groups read Q0; group 0 runs KV blocks [0, half_blocks) and group 1
                    # [half_blocks, 2 half_blocks), so the loader alternates group 0 / group 1 blocks in the ring.
                    if tile_id != -1:
                        kv_cnt = accum_cnt_kv
                        q_buf, q_phase = get_bufidx_phase(tile_count, 1)
                        k0_buf, k0_phase = get_bufidx_phase(kv_cnt, NUM_BUFFERS_KV)
                        k1_buf, k1_phase = get_bufidx_phase(kv_cnt + 1, NUM_BUFFERS_KV)
                        tlx.barrier_wait(k_fulls[k0_buf], k0_phase)
                        tlx.barrier_wait(q_fulls[q_buf], q_phase)
                        tlx.async_dot(
                            q_tiles[q_buf],
                            tlx.local_trans(k_tiles[k0_buf]),
                            qk_tiles[0],
                            use_acc=False,
                            mBarriers=[qk_fulls[0], k_empties[k0_buf]],
                            two_ctas=True,
                        )
                        tlx.barrier_wait(k_fulls[k1_buf], k1_phase)
                        tlx.async_dot(
                            q_tiles[q_buf],
                            tlx.local_trans(k_tiles[k1_buf]),
                            qk_tiles[1],
                            use_acc=False,
                            mBarriers=[qk_fulls[1], k_empties[k1_buf]],
                            two_ctas=True,
                        )
                        if half_blocks == 1:
                            tlx.tcgen05_commit(q_empties[q_buf], two_ctas=True)
                        _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
                        _, qk_phase1 = get_bufidx_phase(accum_cnt_qk1, 1)
                        if CLC_LEAD == 0:
                            next_tile_id, clc_phase_consumer, next_rows = (
                                _claim_next_tile(
                                    clc_context, clc_phase_consumer, tile_info, NUM_CTAS
                                )
                            )
                        use_acc = False
                        for j in tl.range(1, half_blocks):
                            v0_buf = k0_buf
                            v0_phase = k0_phase
                            v1_buf = k1_buf
                            v1_phase = k1_phase
                            kv_cnt += 2
                            k0_buf, k0_phase = get_bufidx_phase(kv_cnt, NUM_BUFFERS_KV)
                            k1_buf, k1_phase = get_bufidx_phase(
                                kv_cnt + 1, NUM_BUFFERS_KV
                            )
                            tlx.barrier_wait(k_fulls[k0_buf], k0_phase)
                            tlx.barrier_wait(v_fulls[v0_buf], v0_phase)
                            _pv_2cta(
                                p_tiles,
                                v_tiles[v0_buf],
                                acc_tiles,
                                acc_fulls,
                                p_fulls,
                                0,
                                qk_phase,
                                use_acc,
                                [v_empties[v0_buf]],
                                BLOCK_N,
                                HEAD_DIM_KV,
                            )
                            tlx.async_dot(
                                q_tiles[q_buf],
                                tlx.local_trans(k_tiles[k0_buf]),
                                qk_tiles[0],
                                use_acc=False,
                                mBarriers=[qk_fulls[0], k_empties[k0_buf]],
                                two_ctas=True,
                            )

                            tlx.barrier_wait(k_fulls[k1_buf], k1_phase)
                            tlx.barrier_wait(v_fulls[v1_buf], v1_phase)
                            _pv_2cta(
                                p_tiles,
                                v_tiles[v1_buf],
                                acc_tiles,
                                acc_fulls,
                                p_fulls,
                                1,
                                qk_phase1,
                                use_acc,
                                [v_empties[v1_buf]],
                                BLOCK_N,
                                HEAD_DIM_KV,
                            )
                            tlx.async_dot(
                                q_tiles[q_buf],
                                tlx.local_trans(k_tiles[k1_buf]),
                                qk_tiles[1],
                                use_acc=False,
                                mBarriers=[qk_fulls[1], k_empties[k1_buf]],
                                two_ctas=True,
                            )
                            if j == half_blocks - 1:
                                tlx.tcgen05_commit(q_empties[q_buf], two_ctas=True)
                            use_acc = True
                            qk_phase = qk_phase ^ 1
                            qk_phase1 = qk_phase1 ^ 1
                            accum_cnt_qk += 1
                            accum_cnt_qk1 += 1

                        tlx.barrier_wait(v_fulls[k0_buf], k0_phase)
                        _pv_2cta(
                            p_tiles,
                            v_tiles[k0_buf],
                            acc_tiles,
                            acc_fulls,
                            p_fulls,
                            0,
                            qk_phase,
                            use_acc,
                            [acc_empties[0], v_empties[k0_buf]],
                            BLOCK_N,
                            HEAD_DIM_KV,
                        )
                        tlx.barrier_wait(v_fulls[k1_buf], k1_phase)
                        _pv_2cta(
                            p_tiles,
                            v_tiles[k1_buf],
                            acc_tiles,
                            acc_fulls,
                            p_fulls,
                            1,
                            qk_phase1,
                            use_acc,
                            [acc_empties[1], v_empties[k1_buf]],
                            BLOCK_N,
                            HEAD_DIM_KV,
                        )
                        accum_cnt_qk += 1
                        accum_cnt_qk1 += 1
                        kv_cnt += 2

                        if CLC_LEAD > 0:
                            next_tile_id, clc_phase_consumer, next_rows = (
                                _claim_next_tile(
                                    clc_context, clc_phase_consumer, tile_info, NUM_CTAS
                                )
                            )
                        tile_count += 1
                        tile_id = next_tile_id
                    # the split tile's KV blocks (an even count; unused once tile_id == -1)
                    accum_cnt_kv += (seq_len + BLOCK_N - 1) // BLOCK_N // 2 * 2
                else:
                    tile_count += 1
                    tile_id = tlx.clc_consumer(
                        clc_context, clc_phase_consumer, multi_ctas=True
                    )
                    clc_phase_consumer ^= 1

        # ---------------- load (1 warp) ----------------
        with tlx.async_task(num_warps=1, registers=24):
            tlx.fence("async_shared")
            accum_cnt_kv = 0
            tile_count = 0
            tile_count1 = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                head, q_row0, kv_row0 = _tile_coords(tile_info, tile_id // NUM_CTAS)
                col = head * HEAD_DIM
                g1_rows, split = _tile_kind(
                    tile_info, tile_id // NUM_CTAS, seq_len, BLOCK_N, HALF_ROWS
                )
                half_blocks = (seq_len + BLOCK_N - 1) // BLOCK_N // 2
                prefetch_tile = tile_id // NUM_CTAS + Q_PREFETCH_DIST
                if prefetch_tile < num_tiles:
                    prefetch_row0 = tl.load(tile_info + prefetch_tile * 4)
                    prefetch_col = tl.load(tile_info + prefetch_tile * 4 + 3) * HEAD_DIM
                    tlx.async_descriptor_prefetch_tensor(
                        desc_q, [prefetch_row0 + rank * GROUP_ROWS, prefetch_col]
                    )
                    tlx.async_descriptor_prefetch_tensor(
                        desc_q,
                        [prefetch_row0 + (NUM_CTAS + rank) * GROUP_ROWS, prefetch_col],
                    )
                q_buf, q_phase = get_bufidx_phase(tile_count, 1)
                _tma_load(
                    desc_q,
                    q_tiles,
                    q_fulls,
                    q_empties,
                    q_buf,
                    q_phase,
                    [q_row0 + rank * GROUP_ROWS, col],
                    is_leader,
                    Q_BYTES,
                )
                if split:
                    # group 0 block j and group 1 block half_blocks + j, interleaved in the ring
                    for j in tl.range(0, half_blocks):
                        for half in tl.static_range(0, 2):
                            buf, phase = get_bufidx_phase(
                                accum_cnt_kv + half, NUM_BUFFERS_KV
                            )
                            _tma_load(
                                desc_k,
                                k_tiles,
                                k_fulls,
                                k_empties,
                                buf,
                                phase,
                                [
                                    kv_row0
                                    + (j + half * half_blocks) * BLOCK_N
                                    + rank * BLOCK_N_KV,
                                    col,
                                ],
                                is_leader,
                                K_BYTES,
                            )
                        for half in tl.static_range(0, 2):
                            buf, phase = get_bufidx_phase(
                                accum_cnt_kv + half, NUM_BUFFERS_KV
                            )
                            _tma_load(
                                desc_v,
                                v_tiles,
                                v_fulls,
                                v_empties,
                                buf,
                                phase,
                                [
                                    kv_row0 + (j + half * half_blocks) * BLOCK_N,
                                    col + rank * HEAD_DIM_KV,
                                ],
                                is_leader,
                                V_BYTES,
                            )
                        accum_cnt_kv += 2
                else:
                    kv_row = kv_row0
                    # K0 before Q1: the first S needs only Q0 and K0.
                    buf, phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    _tma_load(
                        desc_k,
                        k_tiles,
                        k_fulls,
                        k_empties,
                        buf,
                        phase,
                        [kv_row + rank * BLOCK_N_KV, col],
                        is_leader,
                        K_BYTES,
                    )
                    if g1_rows:
                        q_buf1, q_phase1 = get_bufidx_phase(tile_count1, 1)
                        q_buf1 += 1
                        _tma_load(
                            desc_q,
                            q_tiles,
                            q_fulls,
                            q_empties,
                            q_buf1,
                            q_phase1,
                            [q_row0 + (NUM_CTAS + rank) * GROUP_ROWS, col],
                            is_leader,
                            Q_BYTES,
                        )
                        tile_count1 += 1
                    buf, phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    _tma_load(
                        desc_v,
                        v_tiles,
                        v_fulls,
                        v_empties,
                        buf,
                        phase,
                        [kv_row, col + rank * HEAD_DIM_KV],
                        is_leader,
                        V_BYTES,
                    )
                    kv_row += BLOCK_N
                    accum_cnt_kv += 1
                    for _ in tl.range(BLOCK_N, seq_len, BLOCK_N):
                        buf, phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                        _tma_load(
                            desc_k,
                            k_tiles,
                            k_fulls,
                            k_empties,
                            buf,
                            phase,
                            [kv_row + rank * BLOCK_N_KV, col],
                            is_leader,
                            K_BYTES,
                        )
                        buf, phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                        _tma_load(
                            desc_v,
                            v_tiles,
                            v_fulls,
                            v_empties,
                            buf,
                            phase,
                            [kv_row, col + rank * HEAD_DIM_KV],
                            is_leader,
                            V_BYTES,
                        )
                        kv_row += BLOCK_N
                        accum_cnt_kv += 1

                tile_count += 1
                tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=True
                )
                clc_phase_consumer ^= 1

        # ---------------- epilogue: TMA store of O (1 warp) ----------------
        with tlx.async_task(num_warps=1, registers=24):
            tile_count = 0
            tile_count1 = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                head, q_row0, _ = _tile_coords(tile_info, tile_id // NUM_CTAS)
                # Claim the next tile before storing O: the CLC producer waits for every consumer, so claiming after
                # the store would hold the next tile's correction behind it.
                next_tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=True
                )
                clc_phase_consumer ^= 1
                g1_rows, _ = _tile_kind(
                    tile_info, tile_id // NUM_CTAS, seq_len, BLOCK_N, HALF_ROWS
                )
                for g in tl.static_range(0, NUM_GROUPS):
                    if g == 0 or g1_rows:
                        row_block = g * NUM_CTAS + rank
                        _, phase = get_bufidx_phase(
                            tile_count if g == 0 else tile_count1, 1
                        )
                        tlx.barrier_wait(o_fulls[g], phase)
                        # Only this user's rows are stored: a group past the tile's last valid row would overwrite the
                        # next user's rows, and a partial group stores whole 64-row candidates (n_seed % 64 == 0).
                        rows = (
                            tl.load(tile_info + (tile_id // NUM_CTAS) * 4 + 2)
                            - row_block * GROUP_ROWS
                        )
                        o_row0 = q_row0 + row_block * GROUP_ROWS
                        if rows >= GROUP_ROWS:
                            tlx.async_descriptor_store(
                                desc_o, o_tiles[g], [o_row0, head * HEAD_DIM]
                            )
                            tlx.async_descriptor_store_wait(0)
                        elif rows > 0:
                            tlx.async_descriptor_store(
                                desc_o64,
                                tlx.local_slice(
                                    o_tiles[g], [0, 0], [GROUP_ROWS // 2, HEAD_DIM]
                                ),
                                [o_row0, head * HEAD_DIM],
                            )
                            tlx.async_descriptor_store_wait(0)
                        tlx.barrier_arrive(o_empties[g])

                tile_count += 1
                if g1_rows:
                    tile_count1 += 1
                tile_id = next_tile_id


def ikbo_fa_2cta(
    query, key, value, max_seq_len, tile_table, clc_lead=0, scale=None, out=None
):
    """query [B * n_seed, H, D], key / value [U * max_seq_len, H, D], tile_table from make_tile_table(block_m=512)."""
    Bq, H, D = query.shape
    assert D == 128, "the 2-CTA kernel is tuned for head_dim 128"
    if scale is None:
        scale = D**-0.5
    if out is None:
        out = torch.empty_like(query)
    group_rows, block_n, num_ctas = 128, 128, 2

    def desc(t, block_shape):
        return TensorDescriptor(
            t, shape=[t.shape[0], H * D], strides=[H * D, 1], block_shape=block_shape
        )

    num_tiles = tile_table.shape[0]
    _ikbo_fa_2cta_kernel[(num_tiles * num_ctas,)](
        scale,
        desc(query, [group_rows, D]),
        desc(key, [block_n // num_ctas, D]),
        desc(value, [block_n, D // num_ctas]),
        desc(out, [group_rows, D]),
        max_seq_len,
        HEAD_DIM=D,
        BLOCK_M=2 * group_rows,
        BLOCK_N=block_n,
        NUM_BUFFERS_KV=3,
        NUM_CTAS=num_ctas,
        tile_info=tile_table,
        desc_o64=desc(out, [group_rows // 2, D]),
        HAS_TAIL=max_seq_len % block_n != 0,
        CLC_LEAD=clc_lead,
        num_tiles=num_tiles,
        Q_PREFETCH_DIST=_num_sms(query.device.index or 0) // num_ctas,
        num_warps=4,
        num_stages=1,
        ctas_per_cga=(num_ctas, 1, 1),
    )
    return out
