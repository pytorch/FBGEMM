# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
# TLX's type hints do not model constexpr arguments (plain ints / bools passed where `constexpr` is annotated) or
# indexing into barrier / buffer arrays; those are the only errors pyrefly reports for this kernel file.

# TLX IKBO flash attention for Blackwell (sm_100 / sm_103), 1-CTA kernel: used for small users (few candidates per
# user), where the 2-CTA kernel's 512-row tiles would be mostly padding. Entry point: tlx_ikbo_fa_blackwell.py.
#
#   out[c * n_seed:(c + 1) * n_seed, h] = softmax(Q[c, h] K[u(c), h]^T * scale) V[u(c), h]
#
# IKBO FA is jagged-Q / shared-KV cross attention: candidates are sorted by user, so the n_seed query rows of all
# candidates of user u form one contiguous Q segment that attends to the same K/V rows [u * seq_len, (u + 1) * seq_len).
#
# Tile table (built once per candidate layout, like the Hopper kernel's cand_grid): int32 [T, 4] =
# (q_row0, kv_row0, valid_rows, head). Each (user, head) Q segment is cut into 256-row work tiles: two MMA groups of
# 128 rows (4 candidates at n_seed = 64). A user's last tile may be partial; only its valid 64-row candidates are
# stored, and group 1 is skipped when it has no valid rows.
#
# The pipeline is the FA4-style one of tlx.ops flash_attn/sm100.py, with persistent tiles claimed through cluster
# launch control (CLC). Warp roles:
#   load (1 warp)          TMA-loads Q per group and K / V blocks into a NUM_BUFFERS_KV-deep ring; L2-prefetches the
#                          Q of a tile Q_PREFETCH_DIST tiles ahead.
#   MMA (1 warp)           S = Q K^T into TMEM, then O += P V once the softmax has written P.
#   softmax (2 x 4 warps)  one warpgroup per MMA group: online softmax of S in registers, P back to TMEM over S.
#   correction (4 warps)   rescales O when a row maximum grows, then normalizes O into SMEM. Also the CLC producer.
#   epilogue (1 warp)      TMA-stores O.
# Compared with the Hopper kernel (tlx_ikbo_fa_ws.py: 2 candidates per tile, WGMMA) this doubles K/V reuse per SMEM
# load and keeps the accumulators in TMEM.

import functools
import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.cuda.inline_ptx_lib import _fma_f32x2, _mul_f32x2
from triton.language.extra.subtile_ops import _split_n_2D
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor


# From tlx.ops flash_attn/sm100.py (facebookexperimental/triton).
@triton.jit
def _rescale_acc_slices(
    acc_tile, alpha, HEAD_DIM: tl.constexpr, NUM_SLICES: tl.constexpr
):
    """Scale each row of the O accumulator in TMEM by its alpha, one column slice at a time."""
    for slice_id in tl.static_range(0, NUM_SLICES):
        subslice = tlx.subslice(
            acc_tile, HEAD_DIM * slice_id // NUM_SLICES, HEAD_DIM // NUM_SLICES
        )
        acc = tlx.local_load(subslice)
        tlx.local_store(subslice, _mul_f32x2(acc, alpha))


@triton.jit
def _add_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            add.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _rowsum_into_pairs(acc, x):
    # Accumulate adjacent column pairs in order so each packed add can issue as soon as its two exp2 results exist.
    pairs = _split_n_2D(x, x.shape[1] // 2)
    for k in tl.static_range(0, x.shape[1] // 2):
        acc = _add_f32x2(acc, pairs[k])
    return acc


@triton.jit
def _ikbo_softmax_loop(
    qk_fulls,
    qk_tiles,
    p_fulls,
    acc_fulls,
    p_tiles,
    alpha_empties,
    alpha_smem,
    group,
    accum_cnt_qk,
    tile_count,
    qk_scale,
    m_i,
    l_i,
    seq_len,
    out_dtype,
    BLOCK_N: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    ALPHA_BAR_ID: tl.constexpr,
    USE_2CTA: tl.constexpr = False,
    HAS_TAIL: tl.constexpr = False,
    SPLIT_KV: tl.constexpr = False,
    kv_lo=0,
    kv_hi=0,
):
    """Online softmax of one MMA group over the KV blocks of a tile (shared by the 1-CTA and the 2-CTA kernel).

    Per KV block: wait for S, update the running row max m_i, hand the O rescale factor alpha = 2^(m_old - m_new) to
    the correction warpgroup, write P = 2^(S * scale - m) over S in TMEM in three chunks (1/2, 1/4, 1/4 of BLOCK_N) so
    the MMA warp can start P V early, and add P's row sums to l_i. With SPLIT_KV the group covers KV blocks
    [kv_lo, kv_hi) only (2-CTA split half tiles). Returns (m_i, l_i, accum_cnt_qk)."""
    lo, hi = 0, seq_len
    if SPLIT_KV:
        lo, hi = kv_lo, kv_hi

    for start_n in tl.range(lo, hi, BLOCK_N):
        qk_buf = group
        qk_buf_phase = accum_cnt_qk & 1
        # alpha_empties also completes once per tile for l, so its phase counts the tiles too.
        alpha_phase = (accum_cnt_qk + tile_count) & 1
        tlx.barrier_wait(tlx.local_view(qk_fulls, qk_buf), qk_buf_phase)
        qk = tlx.local_load(tlx.local_view(qk_tiles, qk_buf))

        if HAS_TAIL:
            if start_n + BLOCK_N > seq_len:
                # keys past this user's sequence belong to the next user (or are TMA zero-fill): exclude them
                cols = tl.arange(0, BLOCK_N)[None, :]
                qk = tl.where(cols < seq_len - start_n, qk, float("-inf"))

        # Row max in 2 independent chains (as FA4's fmax_reduce does with 4): a single chain is 64 dependent maxes that
        # the exp2s, alpha and the first P chunk all wait for. The max is exact, so this does not change the result.
        row_max = tl.max(tl.max(tl.reshape(qk, [qk.shape[0], BLOCK_N // 2, 2]), 1), 1)
        m_ij = tl.maximum(m_i, row_max)

        # As in FA4, the max (and with it O and l) is only updated when it grows by more than 8 in log2 units: with the
        # stale max P = 2^(S * scale - m) stays below 2^8, and skipping saves the correction warpgroup's TMEM traffic.
        alpha_ = (m_i - m_ij) * qk_scale
        alpha = tl.math.exp2(alpha_)
        rescale_mask = alpha_ >= -8.0
        alpha = tl.where(rescale_mask, 1.0, alpha)
        m_ij = tl.where(rescale_mask, m_i, m_ij)
        # alpha goes to the correction warps through SMEM and a named barrier shared by this group's 128 threads and
        # the correction's 128. The previous iteration waited for alpha_empties, so the slot is free.
        tlx.local_store(tlx.local_view(alpha_smem, group), alpha[:, None])
        tlx.named_barrier_arrive(ALPHA_BAR_ID + group, 256)
        l_i *= alpha  # use alpha right away (short register lifetime); this block's row sum is added below

        # P = 2^(S * scale - m * scale), published in chunks of 1/2, 1/4 and 1/4 so the MMA warp starts P V on the
        # first half and only the last quarter's MMAs trail the last exp2.
        m_scaled = m_ij * qk_scale
        qk = _fma_f32x2(qk, qk_scale, -m_scaled[:, None])
        tl.static_assert(NUM_MMA_SLICES == 2)
        qk_lo, qk_hi = _split_n_2D(qk, 2)
        qk_q2, qk_q3 = _split_n_2D(qk_hi, 2)
        qks = (qk_lo, qk_q2, qk_q3)
        p_slices = ()
        for k in tl.static_range(0, 3):
            p_i = tl.math.exp2(qks[k])
            if k == 0:
                tlx.local_store(
                    tlx.local_view(p_tiles, qk_buf * NUM_MMA_SLICES), p_i.to(out_dtype)
                )
                p_bar = tlx.local_view(acc_fulls, qk_buf)
            else:
                tlx.local_store(
                    tlx.subslice(
                        tlx.local_view(p_tiles, qk_buf * NUM_MMA_SLICES + 1),
                        BLOCK_N // 4 * (k - 1),
                        BLOCK_N // 4,
                    ),
                    p_i.to(out_dtype),
                )
                p_bar = tlx.local_view(p_fulls, qk_buf * 3 + k)
            if USE_2CTA:
                # The TMEM store completes before this readiness token; it does
                # not publish ordinary SMEM to the peer CTA.
                tlx.barrier_arrive(p_bar, 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(p_bar)
            p_slices = p_slices + (p_i,)

        # As in FA4, acquire the alpha slot for the next iteration here, off the S -> P path, and do the row sum
        # behind the wait.
        tlx.barrier_wait(tlx.local_view(alpha_empties, group), alpha_phase)
        l_pairs = tl.zeros([l_i.shape[0], 2], tl.float32)
        for k in tl.static_range(0, 3):
            l_pairs = _rowsum_into_pairs(l_pairs, p_slices[k])
        l_i += tl.sum(l_pairs, 1)
        m_i = m_ij
        accum_cnt_qk += 1

    return m_i, l_i, accum_cnt_qk


@triton.jit
def _tile_entry(tile_info, tile):
    """(q_row0, kv_row0, valid_rows, head) of a tile."""
    base = tile_info + tile * 4
    return tl.load(base), tl.load(base + 1), tl.load(base + 2), tl.load(base + 3)


@triton.jit
def _pv_group(
    p_tiles,
    kv_tiles,
    acc_tiles,
    acc_fulls,
    p_fulls,
    v_buf,
    group: tl.constexpr,
    qk_phase,
    use_acc,
    last_bars,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_KV: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
):
    """O[group] += P[group] V for one KV block, in the softmax's three P chunks, each issued as soon as its chunk is
    ready (the first chunk's barrier acc_fulls also means "O rescaled"). last_bars are signalled when all three MMAs
    complete."""
    tlx.barrier_wait(acc_fulls[group], qk_phase)
    tlx.async_dot(
        p_tiles[group * NUM_MMA_SLICES],
        tlx.local_slice(kv_tiles[v_buf], [0, 0], [BLOCK_N // 2, HEAD_DIM_KV]),
        acc_tiles[group],
        use_acc=use_acc,
        force_async=True,
    )
    tlx.barrier_wait(p_fulls[group * 3 + 1], qk_phase)
    tlx.async_dot(
        tlx.subslice(p_tiles[group * NUM_MMA_SLICES + 1], 0, BLOCK_N // 4),
        tlx.local_slice(
            kv_tiles[v_buf], [BLOCK_N // 2, 0], [BLOCK_N // 4, HEAD_DIM_KV]
        ),
        acc_tiles[group],
        use_acc=True,
        force_async=True,
    )
    tlx.barrier_wait(p_fulls[group * 3 + 2], qk_phase)
    tlx.async_dot(
        tlx.subslice(p_tiles[group * NUM_MMA_SLICES + 1], BLOCK_N // 4, BLOCK_N // 4),
        tlx.local_slice(
            kv_tiles[v_buf], [BLOCK_N * 3 // 4, 0], [BLOCK_N // 4, HEAD_DIM_KV]
        ),
        acc_tiles[group],
        use_acc=True,
        mBarriers=last_bars,
        force_async=True,
    )


@triton.jit
def _mma_tile(
    g1_rows,
    tile_count,
    tile_count1,
    accum_cnt_kv,
    accum_cnt_qk,
    accum_cnt_qk1,
    seq_len,
    q_tiles,
    kv_tiles,
    qk_tiles,
    p_tiles,
    acc_tiles,
    q_fulls,
    q_empties,
    kv_fulls,
    kv_empties,
    qk_fulls,
    p_fulls,
    acc_fulls,
    acc_empties,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_KV: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
):
    """All MMAs of one tile: S(0) for each group, then per KV block j PV(j - 1) and S(j), then the last PV. K and V
    blocks alternate in the ring. Group 1 runs only when it has valid rows (g1_rows); its barrier phases have their own
    counters, and without it group 0's MMAs release the K / V slots."""
    _, q_phase = get_bufidx_phase(tile_count, 1)
    _, q_phase1 = get_bufidx_phase(tile_count1, 1)
    k_buf, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
    tlx.barrier_wait(kv_fulls[k_buf], k_phase)
    tlx.barrier_wait(q_fulls[0], q_phase)
    k_tile = tlx.local_trans(kv_tiles[k_buf])
    if g1_rows:
        tlx.async_dot(
            q_tiles[0], k_tile, qk_tiles[0], use_acc=False, mBarriers=[qk_fulls[0]]
        )
        tlx.barrier_wait(q_fulls[1], q_phase1)
        tlx.async_dot(
            q_tiles[1],
            k_tile,
            qk_tiles[1],
            use_acc=False,
            mBarriers=[qk_fulls[1], kv_empties[k_buf]],
        )
    else:
        tlx.async_dot(
            q_tiles[0],
            k_tile,
            qk_tiles[0],
            use_acc=False,
            mBarriers=[qk_fulls[0], kv_empties[k_buf]],
        )

    _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
    _, qk_phase1 = get_bufidx_phase(accum_cnt_qk1, 1)
    use_acc = False
    for _ in tl.range(BLOCK_N, seq_len, BLOCK_N):
        v_buf, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
        k_buf, k_phase = get_bufidx_phase(accum_cnt_kv + 2, NUM_BUFFERS_KV)
        tlx.barrier_wait(kv_fulls[k_buf], k_phase)
        k_tile = tlx.local_trans(kv_tiles[k_buf])
        tlx.barrier_wait(kv_fulls[v_buf], v_phase)
        if g1_rows:
            _pv_group(
                p_tiles,
                kv_tiles,
                acc_tiles,
                acc_fulls,
                p_fulls,
                v_buf,
                0,
                qk_phase,
                use_acc,
                [],
                BLOCK_N,
                HEAD_DIM_KV,
                NUM_MMA_SLICES,
            )
            tlx.async_dot(
                q_tiles[0], k_tile, qk_tiles[0], use_acc=False, mBarriers=[qk_fulls[0]]
            )
            _pv_group(
                p_tiles,
                kv_tiles,
                acc_tiles,
                acc_fulls,
                p_fulls,
                v_buf,
                1,
                qk_phase1,
                use_acc,
                [kv_empties[v_buf]],
                BLOCK_N,
                HEAD_DIM_KV,
                NUM_MMA_SLICES,
            )
            tlx.async_dot(
                q_tiles[1],
                k_tile,
                qk_tiles[1],
                use_acc=False,
                mBarriers=[qk_fulls[1], kv_empties[k_buf]],
            )
            qk_phase1 = qk_phase1 ^ 1
            accum_cnt_qk1 += 1
        else:
            _pv_group(
                p_tiles,
                kv_tiles,
                acc_tiles,
                acc_fulls,
                p_fulls,
                v_buf,
                0,
                qk_phase,
                use_acc,
                [kv_empties[v_buf]],
                BLOCK_N,
                HEAD_DIM_KV,
                NUM_MMA_SLICES,
            )
            tlx.async_dot(
                q_tiles[0],
                k_tile,
                qk_tiles[0],
                use_acc=False,
                mBarriers=[qk_fulls[0], kv_empties[k_buf]],
            )
        use_acc = True
        qk_phase = qk_phase ^ 1
        accum_cnt_qk += 1
        accum_cnt_kv += 2

    # last KV block: PV only, then release Q and hand O to the correction
    tlx.tcgen05_commit(q_empties[0])
    v_buf, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
    tlx.barrier_wait(kv_fulls[v_buf], v_phase)
    if g1_rows:
        tlx.tcgen05_commit(q_empties[1])
        _pv_group(
            p_tiles,
            kv_tiles,
            acc_tiles,
            acc_fulls,
            p_fulls,
            v_buf,
            0,
            qk_phase,
            use_acc,
            [],
            BLOCK_N,
            HEAD_DIM_KV,
            NUM_MMA_SLICES,
        )
        tlx.tcgen05_commit(acc_empties[0])
        _pv_group(
            p_tiles,
            kv_tiles,
            acc_tiles,
            acc_fulls,
            p_fulls,
            v_buf,
            1,
            qk_phase1,
            use_acc,
            [acc_empties[1], kv_empties[v_buf]],
            BLOCK_N,
            HEAD_DIM_KV,
            NUM_MMA_SLICES,
        )
        accum_cnt_qk1 += 1
    else:
        _pv_group(
            p_tiles,
            kv_tiles,
            acc_tiles,
            acc_fulls,
            p_fulls,
            v_buf,
            0,
            qk_phase,
            use_acc,
            [acc_empties[0], kv_empties[v_buf]],
            BLOCK_N,
            HEAD_DIM_KV,
            NUM_MMA_SLICES,
        )
    accum_cnt_qk += 1
    accum_cnt_kv += 2
    return accum_cnt_kv, accum_cnt_qk, accum_cnt_qk1


@triton.jit
def _tma_load(desc, tiles, fulls, empties, buf, phase, coords, BYTES: tl.constexpr):
    """Wait for ring slot buf to be free, then TMA-load it."""
    tlx.barrier_wait(empties[buf], phase ^ 1)
    tlx.barrier_expect_bytes(fulls[buf], BYTES)
    tlx.async_descriptor_load(desc, tiles[buf], coords, fulls[buf])


@triton.jit
def _load_tile(
    g1_rows,
    tile_count1,
    tile_count,
    accum_cnt_kv,
    seq_len,
    q_row0,
    kv_row0,
    col,
    desc_q,
    desc_k,
    desc_v,
    q_tiles,
    kv_tiles,
    q_fulls,
    q_empties,
    kv_fulls,
    kv_empties,
    ELEM_BYTES: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
):
    """All loads of one tile, in the order the MMA consumes them: Q0, K0, Q1 (if group 1 has rows), V0, then K / V
    for the remaining blocks."""
    Q_BYTES: tl.constexpr = ELEM_BYTES * GROUP_ROWS * HEAD_DIM
    KV_BYTES: tl.constexpr = ELEM_BYTES * BLOCK_N * HEAD_DIM
    q_buf, q_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_Q)
    _tma_load(
        desc_q, q_tiles, q_fulls, q_empties, q_buf, q_phase, [q_row0, col], Q_BYTES
    )
    k_buf, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
    _tma_load(
        desc_k, kv_tiles, kv_fulls, kv_empties, k_buf, k_phase, [kv_row0, col], KV_BYTES
    )
    if g1_rows:
        q_buf, q_phase1 = get_bufidx_phase(tile_count1, NUM_BUFFERS_Q)
        q_buf += NUM_BUFFERS_Q
        _tma_load(
            desc_q,
            q_tiles,
            q_fulls,
            q_empties,
            q_buf,
            q_phase1,
            [q_row0 + GROUP_ROWS, col],
            Q_BYTES,
        )
    v_buf, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
    _tma_load(
        desc_v, kv_tiles, kv_fulls, kv_empties, v_buf, v_phase, [kv_row0, col], KV_BYTES
    )
    kv_row = kv_row0 + BLOCK_N
    accum_cnt_kv += 2
    for _ in tl.range(BLOCK_N, seq_len, BLOCK_N):
        k_buf, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
        _tma_load(
            desc_k,
            kv_tiles,
            kv_fulls,
            kv_empties,
            k_buf,
            k_phase,
            [kv_row, col],
            KV_BYTES,
        )
        v_buf, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
        _tma_load(
            desc_v,
            kv_tiles,
            kv_fulls,
            kv_empties,
            v_buf,
            v_phase,
            [kv_row, col],
            KV_BYTES,
        )
        kv_row += BLOCK_N
        accum_cnt_kv += 2
    return accum_cnt_kv


@triton.jit
def _ikbo_fa_1cta_kernel(
    sm_scale,
    tile_info,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    desc_o64,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    SOFTMAX_REGS: tl.constexpr,
    HAS_TAIL: tl.constexpr = False,
    num_tiles=0,
    Q_PREFETCH_DIST=0,
):
    NUM_GROUPS: tl.constexpr = 2
    NUM_BUFFERS_Q: tl.constexpr = 1
    GROUP_ROWS: tl.constexpr = BLOCK_M // 2
    ELEM_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    start_pid = tl.program_id(0)

    # SMEM: Q and O per group, one K / V ring (K and V blocks alternate).
    q_tiles = tlx.local_alloc(
        (GROUP_ROWS, HEAD_DIM), tlx.dtype_of(desc_q), NUM_GROUPS * NUM_BUFFERS_Q
    )
    o_tiles = tlx.local_alloc((GROUP_ROWS, HEAD_DIM), tlx.dtype_of(desc_o), NUM_GROUPS)
    kv_tiles = tlx.local_alloc(
        (BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV
    )
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_GROUPS * NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_GROUPS * NUM_BUFFERS_Q)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
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
        (GROUP_ROWS, BLOCK_N // NUM_MMA_SLICES),
        tlx.dtype_of(desc_v),
        NUM_GROUPS * NUM_MMA_SLICES,
        tlx.storage_kind.tmem,
        reuse=qk_storage_alias,
    )
    qk_storage_alias.set_buffer_overlap(
        tlx.reuse_group(
            qk_tiles,
            tlx.reuse_group(p_tiles, group_size=NUM_MMA_SLICES),
            group_type=tlx.reuse_group_type.shared,
        )
    )
    acc_tiles = tlx.local_alloc(
        (GROUP_ROWS, HEAD_DIM), tl.float32, NUM_GROUPS, tlx.storage_kind.tmem
    )
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_GROUPS)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_GROUPS)
    # P chunks 2 and 3 have their own barriers; chunk 1 shares acc_fulls with the correction's "O rescaled".
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_GROUPS * 3, arrive_count=1)
    acc_fulls = tlx.alloc_barriers(num_barriers=NUM_GROUPS, arrive_count=2)
    # Softmax -> correction: alpha per KV block, then l at the end of the tile, through alpha_smem and named barrier
    # ALPHA_BAR_ID + g; the correction arrives on alpha_empties after reading.
    ALPHA_BAR_ID: tl.constexpr = 12
    alpha_smem = tlx.local_alloc((GROUP_ROWS, 1), tl.float32, NUM_GROUPS)
    alpha_empties = tlx.alloc_warp_barrier(num_barriers=NUM_GROUPS, num_warps=4)
    o_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_GROUPS, num_warps=4)
    # CLC consumers: correction, 2 softmax groups, MMA, load, epilogue.
    clc_context = tlx.clc_create_context(num_consumers=6)

    with tlx.async_tasks(exclusive=True):
        # ---------------- correction (default, 4 warps) ----------------
        with tlx.async_task("default", registers=120):
            acc_phase0 = 0
            acc_phase1 = 0
            tile_id = start_pid
            clc_phase_producer = 1
            clc_phase_consumer = 0
            while tile_id != -1:
                tlx.clc_producer(clc_context, clc_phase_producer)
                clc_phase_producer ^= 1
                g1_rows = tl.load(tile_info + tile_id * 4 + 2) > GROUP_ROWS
                for start_n in tl.range(0, seq_len, BLOCK_N):
                    for g in tl.static_range(0, NUM_GROUPS):
                        if g == 0 or g1_rows:
                            tlx.named_barrier_wait(ALPHA_BAR_ID + g, 256)
                            alpha = tlx.local_load(alpha_smem[g])
                            tlx.barrier_arrive(alpha_empties[g])
                            # the first PV of a tile overwrites O; later, each warp rescales its 32 rows if needed
                            if start_n != 0:
                                should_rescale = tlx.warp_any(alpha < 1.0)
                                tlx.warp_predicate(
                                    should_rescale,
                                    (),
                                    _rescale_acc_slices,
                                    args=(
                                        acc_tiles[g],
                                        alpha,
                                        HEAD_DIM,
                                        NUM_MMA_SLICES,
                                    ),
                                    wave_uniform=True,
                                )
                            tlx.barrier_arrive(acc_fulls[g])
                for g in tl.static_range(0, NUM_GROUPS):
                    if g == 0 or g1_rows:
                        # O / l into SMEM for the epilogue
                        acc_phase = acc_phase0 if g == 0 else acc_phase1
                        tlx.named_barrier_wait(ALPHA_BAR_ID + g, 256)
                        row_sum = tlx.local_load(alpha_smem[g])
                        tlx.barrier_arrive(alpha_empties[g])
                        tlx.barrier_wait(acc_empties[g], acc_phase)
                        tlx.barrier_wait(o_empties[g], acc_phase ^ 1)
                        scale = 1 / row_sum
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            subslice = tlx.subslice(
                                acc_tiles[g],
                                HEAD_DIM * slice_id // NUM_MMA_SLICES,
                                HEAD_DIM // NUM_MMA_SLICES,
                            )
                            o = tlx.local_load(subslice)
                            o = _mul_f32x2(o, scale).to(tlx.dtype_of(desc_o))
                            tlx.local_store(
                                tlx.local_slice(
                                    o_tiles[g],
                                    [0, HEAD_DIM * slice_id // NUM_MMA_SLICES],
                                    [GROUP_ROWS, HEAD_DIM // NUM_MMA_SLICES],
                                ),
                                o,
                            )
                        tlx.barrier_arrive(o_fulls[g])
                acc_phase0 ^= 1
                if g1_rows:
                    acc_phase1 ^= 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # ---------------- softmax (2 x 4 warps, one warpgroup per MMA group) ----------------
        with tlx.async_task(num_warps=4, registers=SOFTMAX_REGS, replicate=NUM_GROUPS):
            accum_cnt_qk = 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            g = tlx.async_task_replica_id()
            qk_scale = sm_scale * 1.44269504  # log2(e): the kernel works in exp2
            p_dtype = tlx.dtype_of(desc_v)
            while tile_id != -1:
                if g == 0 or tl.load(tile_info + tile_id * 4 + 2) > GROUP_ROWS:
                    # Running row max and row sum. l starts at 1 but is multiplied by alpha = 0 at the first block.
                    m_i = tl.zeros([GROUP_ROWS], dtype=tl.float32) - float("inf")
                    l_i = tl.zeros([GROUP_ROWS], dtype=tl.float32) + 1.0
                    tlx.barrier_wait(
                        alpha_empties[g], (accum_cnt_qk + tile_count + 1) & 1
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
                        NUM_MMA_SLICES,
                        ALPHA_BAR_ID=ALPHA_BAR_ID,
                        USE_2CTA=False,
                        HAS_TAIL=HAS_TAIL,
                    )
                    tlx.local_store(alpha_smem[g], l_i[:, None])
                    tlx.named_barrier_arrive(ALPHA_BAR_ID + g, 256)
                    tile_count += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # ---------------- MMA (1 warp) ----------------
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            accum_cnt_qk = 0
            accum_cnt_qk1 = 0
            tile_count = 0
            tile_count1 = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                g1_rows = tl.load(tile_info + tile_id * 4 + 2) > GROUP_ROWS
                accum_cnt_kv, accum_cnt_qk, accum_cnt_qk1 = _mma_tile(
                    g1_rows,
                    tile_count,
                    tile_count1,
                    accum_cnt_kv,
                    accum_cnt_qk,
                    accum_cnt_qk1,
                    seq_len,
                    q_tiles,
                    kv_tiles,
                    qk_tiles,
                    p_tiles,
                    acc_tiles,
                    q_fulls,
                    q_empties,
                    kv_fulls,
                    kv_empties,
                    qk_fulls,
                    p_fulls,
                    acc_fulls,
                    acc_empties,
                    BLOCK_N,
                    HEAD_DIM,
                    NUM_BUFFERS_KV,
                    NUM_MMA_SLICES,
                )
                tile_count += 1
                if g1_rows:
                    tile_count1 += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # ---------------- load (1 warp) ----------------
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            tile_count = 0
            tile_count1 = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                q_row0, kv_row0, valid_rows, head = _tile_entry(tile_info, tile_id)
                g1_rows = valid_rows > GROUP_ROWS
                prefetch_tile = tile_id + Q_PREFETCH_DIST
                if prefetch_tile < num_tiles:
                    prefetch_row0, _, _, prefetch_head = _tile_entry(
                        tile_info, prefetch_tile
                    )
                    tlx.async_descriptor_prefetch_tensor(
                        desc_q, [prefetch_row0, prefetch_head * HEAD_DIM]
                    )
                    tlx.async_descriptor_prefetch_tensor(
                        desc_q, [prefetch_row0 + GROUP_ROWS, prefetch_head * HEAD_DIM]
                    )
                accum_cnt_kv = _load_tile(
                    g1_rows,
                    tile_count1,
                    tile_count,
                    accum_cnt_kv,
                    seq_len,
                    q_row0,
                    kv_row0,
                    head * HEAD_DIM,
                    desc_q,
                    desc_k,
                    desc_v,
                    q_tiles,
                    kv_tiles,
                    q_fulls,
                    q_empties,
                    kv_fulls,
                    kv_empties,
                    ELEM_BYTES,
                    GROUP_ROWS,
                    BLOCK_N,
                    HEAD_DIM,
                    NUM_BUFFERS_Q,
                    NUM_BUFFERS_KV,
                )
                tile_count += 1
                if g1_rows:
                    tile_count1 += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # ---------------- epilogue: TMA store of O (1 warp) ----------------
        with tlx.async_task(num_warps=1, registers=24):
            tile_count = 0
            tile_count1 = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                q_row0, kv_row0, valid_rows, head = _tile_entry(tile_info, tile_id)
                next_tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1
                g1_rows = valid_rows > GROUP_ROWS
                for g in tl.static_range(0, NUM_GROUPS):
                    if g == 0 or g1_rows:
                        _, phase = get_bufidx_phase(
                            tile_count if g == 0 else tile_count1, 1
                        )
                        tlx.barrier_wait(o_fulls[g], phase)
                        rows = valid_rows - g * GROUP_ROWS
                        if rows >= GROUP_ROWS:
                            tlx.async_descriptor_store(
                                desc_o,
                                o_tiles[g],
                                [q_row0 + g * GROUP_ROWS, head * HEAD_DIM],
                            )
                            tlx.async_descriptor_store_wait(0)
                        else:
                            # a partial group holds one 64-row candidate (n_seed = 64): store only it
                            tlx.async_descriptor_store(
                                desc_o64,
                                tlx.local_slice(
                                    o_tiles[g], [0, 0], [GROUP_ROWS // 2, HEAD_DIM]
                                ),
                                [q_row0 + g * GROUP_ROWS, head * HEAD_DIM],
                            )
                            tlx.async_descriptor_store_wait(0)
                        tlx.barrier_arrive(o_empties[g])
                tile_count += 1
                if g1_rows:
                    tile_count1 += 1
                tile_id = next_tile_id


@functools.lru_cache(maxsize=None)
def _num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def tile_rows(num_cands_per_user, n_seed, num_heads, max_seq_len, block_m):
    """Host-side tile list [(q_row0, kv_row0, valid_rows, head)], ordered user-major, then head, then row tile, so
    concurrently running tiles share K/V in L2."""
    rows = []
    q_row = 0
    for u, c in enumerate(num_cands_per_user):
        nq = c * n_seed
        for h in range(num_heads):
            for m0 in range(0, nq, block_m):
                rows.append((q_row + m0, u * max_seq_len, min(block_m, nq - m0), h))
        q_row += nq
    return rows


def make_tile_table(
    cand_to_user_index,
    n_seed,
    num_heads,
    max_seq_len,
    num_cands_per_user=None,
    block_m=256,
):
    """Tile table int32 [T, 4] = (q_row0, kv_row0, valid_rows, head) for candidates grouped by user in ascending
    user order. Host-side (syncs); build it once per candidate layout."""
    if num_cands_per_user is None:
        num_cands_per_user = torch.bincount(cand_to_user_index.long())
    rows = tile_rows(
        num_cands_per_user.tolist(), n_seed, num_heads, max_seq_len, block_m
    )
    return torch.tensor(rows, dtype=torch.int32, device=cand_to_user_index.device)


def ikbo_fa_1cta(query, key, value, max_seq_len, tile_table, scale=None, out=None):
    """query [B * n_seed, H, D], key / value [U * max_seq_len, H, D], tile_table from make_tile_table(block_m=256)."""
    Bq, H, D = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(query)
    block_m, block_n = 256, 128

    def desc(t, block_shape):
        return TensorDescriptor(
            t, shape=[t.shape[0], H * D], strides=[H * D, 1], block_shape=block_shape
        )

    num_tiles = tile_table.shape[0]
    _ikbo_fa_1cta_kernel[(num_tiles,)](
        scale,
        tile_table,
        desc(query, [block_m // 2, D]),
        desc(key, [block_n, D]),
        desc(value, [block_n, D]),
        desc(out, [block_m // 2, D]),
        desc(out, [block_m // 4, D]),
        max_seq_len,
        HEAD_DIM=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_BUFFERS_KV=3,
        NUM_MMA_SLICES=2,
        SOFTMAX_REGS=184,
        HAS_TAIL=max_seq_len % block_n != 0,
        num_tiles=num_tiles,
        Q_PREFETCH_DIST=_num_sms(query.device.index or 0),
        num_warps=4,
        num_stages=1,
    )
    return out
