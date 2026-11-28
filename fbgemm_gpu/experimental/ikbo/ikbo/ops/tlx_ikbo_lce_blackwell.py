# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
# TLX's type hints do not model constexpr arguments (plain ints / bools passed where `constexpr` is annotated) or
# indexing into barrier / buffer arrays; those are the only errors pyrefly reports for this kernel file.

# TLX IKBO LCE kernel for Blackwell (sm_100 / sm_103).
#
#   out[b] = W_cand @ E_cand[b] + W_user @ E_user[u(b)]       u(b) = cand_to_user_index[b], fp32 accumulation
#
# How the work is split
#   Each candidate's output is [M, N]: N = 256 (embedding dim), M = compressed rows (e.g. 433, padded to a multiple
#   of 64). A "unit" is one user or one candidate times one m-tile of up to 256 output columns. User units come first
#   (ids [0, U * NUM_M_TILES)), then candidate units; cluster c takes units c, c + num_clusters, ...
#
#   The MMA is swap-AB: N is the MMA M dimension, split over the two CTAs of a cluster (128 rows each, tcgen05
#   cta_group::2), and the m-tile is the MMA N dimension. A last m-tile that is not a power of two (e.g. 192) is
#   issued as up to three power-of-two MMAs (128 + 64).
#
# Warp roles (persistent, one CTA per SM)
#   producer (1 warp)   TMA-loads E and W for each K step into a NUM_STAGES-deep SMEM ring.
#   MMA (1 warp)        accumulates a unit into one of NUM_ACC TMEM accumulators (rank 0 issues the pair's MMA).
#   epilogue (4 warps)  drains the accumulator. User unit: store fp32 to the user_res workspace, then publish a
#                       ready flag. Candidate unit: wait for its user's flag, add user_res, store out.
#   Handoffs: SMEM stage  full_bars -> MMA, empty_bars -> producer
#             accumulator acc_full_bars -> epilogue, acc_empty_bars -> MMA
#
# Ready flags
#   Only candidate epilogues wait. The flags persist across calls: a flag holds the stamp (device-side call counter)
#   of the last launch that wrote it, and the last CTA of each launch advances the counter, so there is no memset
#   per call and CUDA-graph replays stay correct. A candidate may wait on a unit owned by another cluster, so every
#   CTA must be resident at once: the grid is at most one CTA per SM and the launch is cooperative (it fails rather
#   than hangs if the GPU cannot host the whole grid).
#
# GB300 (B=1024, M=433, ratio 70, K_user / K_cand = 1184 / 872): ~185-192 us vs 310 us for cuBLAS
# (torch_decomposed_lce) and 400 us for triton_ikbo_lce. The kernel runs at the board power limit, so its time is set
# by energy per call (~230 mJ vs 328 mJ for the cuBLAS decomposition).

import functools
import threading

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

# Fixed tile configuration (tuned on GB300). Passed to the kernel as constexprs; shape-dependent values are computed per
# call in tlx_ikbo_lce_blackwell.
NUM_CTAS = 2  # CTAs per cluster (cta_group::2 MMA)
N_PER_CTA = 128  # rows of N per CTA: the pair covers N = 256
M_TILE = 256  # m-tile width (MMA N)
K_STEP = 64
NUM_STAGES = 4  # SMEM ring depth (one E tile + one W tile per stage)
NUM_ACC = 2  # TMEM accumulators: the epilogue drains one while the MMA fills the other
EPI_SLICE = 64  # epilogue column slice; M is padded to a multiple of it


def _last_tile_pieces(width):
    """Split the last m-tile's width (a multiple of 64, at most 256) into power-of-two MMA widths, zero-padded to
    three entries: 192 -> [128, 64, 0]."""
    pieces = []
    while width:
        p = 1 << (width.bit_length() - 1)
        pieces.append(p)
        width -= p
    assert len(pieces) <= 3, pieces
    return pieces + [0] * (3 - len(pieces))


@triton.jit
def _decode_unit(unit, num_clusters, num_user_units, NUM_M_TILES: tl.constexpr):
    """Unit id -> (is_user, row, m_tile), where row is the user or candidate index.

    With two m-tiles and an even cluster count, a cluster would always get the same m-tile (unit % 2 is fixed per
    cluster); swapping each pair on odd rounds alternates the wide and the narrow m-tile instead. Both the user and
    the candidate block have an even number of units then, so a swap never crosses from users to candidates.
    """
    if NUM_M_TILES == 2:
        flip = ((unit // num_clusters) & 1) & ((num_clusters & 1) == 0)
        unit = unit ^ flip
    is_user = unit < num_user_units
    local = tl.where(is_user, unit, unit - num_user_units)
    return is_user, local // NUM_M_TILES, local % NUM_M_TILES


@triton.jit
def _next_stage(stage, phase, NUM_STAGES: tl.constexpr):
    stage += 1
    if stage == NUM_STAGES:
        stage = 0
        phase ^= 1
    return stage, phase


# ------------------------------------------------------------------------------------------------- MMA warp
@triton.jit
def _mma_piece(
    e_op,
    w_bufs,
    stage,
    acc,
    ROW0: tl.constexpr,
    WIDTH: tl.constexpr,
    use_acc: tl.constexpr,
    bars,
    NUM_CTAS: tl.constexpr,
    K_STEP: tl.constexpr,
):
    """One power-of-two MMA of the last m-tile: this CTA's W rows [ROW0, ROW0 + WIDTH / 2) give accumulator columns
    [2 * ROW0, 2 * ROW0 + WIDTH) (the two CTAs supply WIDTH W rows together)."""
    w_op = tlx.local_trans(
        tlx.local_slice(w_bufs[stage], [ROW0, 0], [WIDTH // NUM_CTAS, K_STEP])
    )
    acc_cols = tlx.local_slice(acc, [0, ROW0 * NUM_CTAS], [acc.shape[0], WIDTH])
    tlx.async_dot(
        e_op,
        w_op,
        acc_cols,
        use_acc=use_acc,
        mBarriers=bars,
        two_ctas=True,
        out_dtype=tl.float32,
    )


@triton.jit
def _mma_k_step(
    e_bufs,
    w_bufs,
    stage,
    acc,
    stage_empty,
    use_acc: tl.constexpr,
    IS_LAST_TILE: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    K_STEP: tl.constexpr,
):
    """One K step of a unit: one MMA for a full m-tile, one per power-of-two piece for the last m-tile. The step's
    last MMA releases the SMEM stage back to the producer."""
    e_op = tlx.local_trans(e_bufs[stage])
    if not IS_LAST_TILE:
        tlx.async_dot(
            e_op,
            tlx.local_trans(w_bufs[stage]),
            acc,
            use_acc=use_acc,
            mBarriers=[stage_empty],
            two_ctas=True,
            out_dtype=tl.float32,
        )
    elif P1 == 0:
        _mma_piece(
            e_op, w_bufs, stage, acc, 0, P0, use_acc, [stage_empty], NUM_CTAS, K_STEP
        )
    elif P2 == 0:
        _mma_piece(e_op, w_bufs, stage, acc, 0, P0, use_acc, [], NUM_CTAS, K_STEP)
        _mma_piece(
            e_op,
            w_bufs,
            stage,
            acc,
            P0 // NUM_CTAS,
            P1,
            use_acc,
            [stage_empty],
            NUM_CTAS,
            K_STEP,
        )
    else:
        _mma_piece(e_op, w_bufs, stage, acc, 0, P0, use_acc, [], NUM_CTAS, K_STEP)
        _mma_piece(
            e_op, w_bufs, stage, acc, P0 // NUM_CTAS, P1, use_acc, [], NUM_CTAS, K_STEP
        )
        _mma_piece(
            e_op,
            w_bufs,
            stage,
            acc,
            (P0 + P1) // NUM_CTAS,
            P2,
            use_acc,
            [stage_empty],
            NUM_CTAS,
            K_STEP,
        )


@triton.jit
def _mma_unit(
    e_bufs,
    w_bufs,
    acc,
    full_bars,
    empty_bars,
    stage,
    phase,
    num_k_steps,
    IS_LAST_TILE: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    K_STEP: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Accumulate all K steps of one unit into acc. Returns the ring position after the unit."""
    tlx.barrier_wait(full_bars[stage], phase)
    _mma_k_step(
        e_bufs,
        w_bufs,
        stage,
        acc,
        empty_bars[stage],
        False,
        IS_LAST_TILE,
        P0,
        P1,
        P2,
        NUM_CTAS,
        K_STEP,
    )
    for _ in range(1, num_k_steps):
        stage, phase = _next_stage(stage, phase, NUM_STAGES)
        tlx.barrier_wait(full_bars[stage], phase)
        _mma_k_step(
            e_bufs,
            w_bufs,
            stage,
            acc,
            empty_bars[stage],
            True,
            IS_LAST_TILE,
            P0,
            P1,
            P2,
            NUM_CTAS,
            K_STEP,
        )
    return _next_stage(stage, phase, NUM_STAGES)


# ------------------------------------------------------------------------------------------------- producer warp
@triton.jit
def _load_w_rows(
    w_desc,
    w_bufs,
    stage,
    bar,
    m0,
    k,
    ROW0: tl.constexpr,
    WIDTH: tl.constexpr,
    rank,
    CHUNK: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    K_STEP: tl.constexpr,
):
    """This CTA's half of W rows [m0, m0 + WIDTH) at K offset k into stage rows [ROW0, ...), CHUNK rows per TMA."""
    ROWS: tl.constexpr = WIDTH // NUM_CTAS
    for c in tl.static_range(ROWS // CHUNK):
        tlx.async_descriptor_load(
            w_desc,
            tlx.local_slice(w_bufs[stage], [ROW0 + c * CHUNK, 0], [CHUNK, K_STEP]),
            [m0 + rank * ROWS + c * CHUNK, k],
            bar,
            two_ctas=True,
        )


@triton.jit
def _produce_unit(
    e_desc,
    w_last_desc,
    w_full_desc,
    e_bufs,
    w_bufs,
    full_bars,
    empty_bars,
    stage,
    phase,
    e_row0,
    K,
    n0,
    m0,
    rank,
    IS_LAST_TILE: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    W_CHUNK: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    N_PER_CTA: tl.constexpr,
    M_TILE: tl.constexpr,
    K_STEP: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Stream one unit into the SMEM ring. Per K step: this CTA's 128 rows of N of E rows [e_row0 + k, + K_STEP), and
    its half of the m-tile's W rows. The pair's MMA waits on rank 0's barrier, which counts both CTAs' bytes.
    """
    ELEM_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(e_desc))
    E_BYTES: tl.constexpr = K_STEP * N_PER_CTA * ELEM_BYTES
    if IS_LAST_TILE:
        W_BYTES: tl.constexpr = ((P0 + P1 + P2) // NUM_CTAS) * K_STEP * ELEM_BYTES
    else:
        W_BYTES: tl.constexpr = (M_TILE // NUM_CTAS) * K_STEP * ELEM_BYTES
    for k in range(0, K, K_STEP):
        tlx.barrier_wait(empty_bars[stage], phase ^ 1)
        tlx.barrier_expect_bytes(
            full_bars[stage], (E_BYTES + W_BYTES) * NUM_CTAS, pred=rank == 0
        )
        tlx.async_descriptor_load(
            e_desc, e_bufs[stage], [e_row0 + k, n0], full_bars[stage], two_ctas=True
        )
        if IS_LAST_TILE:
            _load_w_rows(
                w_last_desc,
                w_bufs,
                stage,
                full_bars[stage],
                m0,
                k,
                0,
                P0,
                rank,
                W_CHUNK,
                NUM_CTAS,
                K_STEP,
            )
            if P1 > 0:
                _load_w_rows(
                    w_last_desc,
                    w_bufs,
                    stage,
                    full_bars[stage],
                    m0 + P0,
                    k,
                    P0 // NUM_CTAS,
                    P1,
                    rank,
                    W_CHUNK,
                    NUM_CTAS,
                    K_STEP,
                )
            if P2 > 0:
                _load_w_rows(
                    w_last_desc,
                    w_bufs,
                    stage,
                    full_bars[stage],
                    m0 + P0 + P1,
                    k,
                    (P0 + P1) // NUM_CTAS,
                    P2,
                    rank,
                    W_CHUNK,
                    NUM_CTAS,
                    K_STEP,
                )
        else:
            _load_w_rows(
                w_full_desc,
                w_bufs,
                stage,
                full_bars[stage],
                m0,
                k,
                0,
                M_TILE,
                rank,
                M_TILE // NUM_CTAS,
                NUM_CTAS,
                K_STEP,
            )
        stage, phase = _next_stage(stage, phase, NUM_STAGES)
    return stage, phase


@triton.jit
def _produce(
    e_desc,
    w_last_desc,
    w_full_desc,
    e_bufs,
    w_bufs,
    full_bars,
    empty_bars,
    stage,
    phase,
    row,
    K,
    n0,
    m0,
    m_tile,
    rank,
    NUM_M_TILES: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    W_CHUNK: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    N_PER_CTA: tl.constexpr,
    M_TILE: tl.constexpr,
    K_STEP: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """_produce_unit for unit (row, m_tile) starting at output column m0, specialized for the last m-tile."""
    if m_tile == NUM_M_TILES - 1:
        stage, phase = _produce_unit(
            e_desc,
            w_last_desc,
            w_full_desc,
            e_bufs,
            w_bufs,
            full_bars,
            empty_bars,
            stage,
            phase,
            row * K,
            K,
            n0,
            m0,
            rank,
            True,
            P0,
            P1,
            P2,
            W_CHUNK,
            NUM_CTAS,
            N_PER_CTA,
            M_TILE,
            K_STEP,
            NUM_STAGES,
        )
    else:
        stage, phase = _produce_unit(
            e_desc,
            w_last_desc,
            w_full_desc,
            e_bufs,
            w_bufs,
            full_bars,
            empty_bars,
            stage,
            phase,
            row * K,
            K,
            n0,
            m0,
            rank,
            False,
            P0,
            P1,
            P2,
            W_CHUNK,
            NUM_CTAS,
            N_PER_CTA,
            M_TILE,
            K_STEP,
            NUM_STAGES,
        )
    return stage, phase


# ------------------------------------------------------------------------------------------------- epilogue warpgroup
@triton.jit
def _ld_acquire_gpu(ptr):
    # Every waiting thread acquires the flag itself, so each thread's user_res loads are ordered after it.
    return tl.inline_asm_elementwise(
        "ld.acquire.gpu.global.b32 $0, [$1];",
        "=r,l",
        [ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _drain_unit(
    acc,
    acc_empty,
    out_ptr,
    user_res_ptr,
    row,
    user,
    n0,
    m_col0,
    M,
    M_PAD: tl.constexpr,
    IS_USER: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    N_PER_CTA: tl.constexpr,
    EPI_SLICE: tl.constexpr,
    user_valid=True,
):
    """Read one accumulator ([128 rows of N, m-tile columns], fp32) out of TMEM, EPI_SLICE columns at a time.
    User unit: store to user_res[row]. Candidate unit: add user_res[user] and store out[row]. The accumulator goes
    back to the MMA warp as soon as its last slice is loaded."""
    N: tl.constexpr = N_PER_CTA * NUM_CTAS
    offs_n = n0 + tl.arange(0, N_PER_CTA)
    for s in tl.static_range(NUM_SLICES):
        v = tlx.local_load(
            tlx.local_slice(acc, [0, s * EPI_SLICE], [N_PER_CTA, EPI_SLICE])
        )  # [n, m]
        if s == NUM_SLICES - 1:
            tlx.barrier_arrive(acc_empty, 1, remote_cta_rank=0)
        offs_m = m_col0 + s * EPI_SLICE + tl.arange(0, EPI_SLICE)
        user_res_offs = offs_m[None, :] * N + offs_n[:, None]
        if IS_USER:
            tl.store(user_res_ptr + row.to(tl.int64) * (N * M_PAD) + user_res_offs, v)
        else:
            # an out-of-range user index gives NaN rows instead of an out-of-bounds read
            v = v + tl.load(
                user_res_ptr + user.to(tl.int64) * (N * M_PAD) + user_res_offs,
                mask=user_valid,
                other=float("nan"),
            )
            ptrs = (
                out_ptr
                + row.to(tl.int64) * M * N
                + offs_m[None, :] * N
                + offs_n[:, None]
            )
            tl.store(ptrs, v.to(out_ptr.dtype.element_ty), mask=offs_m[None, :] < M)


@triton.jit
def _drain(
    acc,
    acc_empty,
    out_ptr,
    user_res_ptr,
    row,
    user,
    n0,
    m_tile,
    M,
    M_PAD: tl.constexpr,
    IS_USER: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    LAST_TILE_WIDTH: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    N_PER_CTA: tl.constexpr,
    M_TILE: tl.constexpr,
    EPI_SLICE: tl.constexpr,
    user_valid=True,
):
    """_drain_unit for m_tile, specialized for the (narrower) last m-tile."""
    if m_tile == NUM_M_TILES - 1:
        _drain_unit(
            acc,
            acc_empty,
            out_ptr,
            user_res_ptr,
            row,
            user,
            n0,
            m_tile * M_TILE,
            M,
            M_PAD,
            IS_USER,
            LAST_TILE_WIDTH // EPI_SLICE,
            NUM_CTAS,
            N_PER_CTA,
            EPI_SLICE,
            user_valid,
        )
    else:
        _drain_unit(
            acc,
            acc_empty,
            out_ptr,
            user_res_ptr,
            row,
            user,
            n0,
            m_tile * M_TILE,
            M,
            M_PAD,
            IS_USER,
            M_TILE // EPI_SLICE,
            NUM_CTAS,
            N_PER_CTA,
            EPI_SLICE,
            user_valid,
        )


@triton.jit
def _epilogue(
    acc_bufs,
    acc_full_bars,
    acc_empty_bars,
    out_ptr,
    user_res_ptr,
    idx_ptr,
    flag_ptr,
    state_ptr,
    U,
    num_user_units,
    num_units,
    M,
    rank,
    M_PAD: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    LAST_TILE_WIDTH: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    N_PER_CTA: tl.constexpr,
    M_TILE: tl.constexpr,
    NUM_ACC: tl.constexpr,
    EPI_SLICE: tl.constexpr,
):
    EPILOGUE_BARRIER: tl.constexpr = 8  # named barrier of this warpgroup (128 threads)
    pid = tl.program_id(0)
    cluster_id = pid // NUM_CTAS
    num_clusters = tl.num_programs(0) // NUM_CTAS
    n0 = rank * N_PER_CTA
    # This launch's stamp. Flags only ever hold stamps of earlier launches or this one, so equality is exact
    # (no reset between calls, and int32 wraparound is harmless).
    stamp = tl.load(state_ptr, volatile=True) + 1
    units_done = 0
    for unit in range(cluster_id, num_units, num_clusters):
        is_user, row, m_tile = _decode_unit(
            unit, num_clusters, num_user_units, NUM_M_TILES
        )
        acc_buf, acc_phase = get_bufidx_phase(units_done, NUM_ACC)
        if is_user:
            tlx.barrier_wait(acc_full_bars[acc_buf], acc_phase)
            _drain(
                acc_bufs[acc_buf],
                acc_empty_bars[acc_buf],
                out_ptr,
                user_res_ptr,
                row,
                0,
                n0,
                m_tile,
                M,
                M_PAD,
                True,
                NUM_M_TILES,
                LAST_TILE_WIDTH,
                NUM_CTAS,
                N_PER_CTA,
                M_TILE,
                EPI_SLICE,
            )
            # Publish: every thread's user_res stores happen before the single gpu-scope release. Candidates on CTA
            # rank r only read the rows that rank r wrote, so each rank has its own flag.
            tlx.named_barrier_wait(EPILOGUE_BARRIER, 128)
            tl.atomic_xchg(
                flag_ptr + (row * NUM_M_TILES + m_tile) * NUM_CTAS + rank,
                stamp,
                sem="release",
                scope="gpu",
            )
        else:
            user = tl.load(idx_ptr + row)
            user_valid = (user >= 0) & (user < U)
            user = tl.where(user_valid, user, 0)
            if user_valid:
                flag = flag_ptr + (user * NUM_M_TILES + m_tile) * NUM_CTAS + rank
                while _ld_acquire_gpu(flag) != stamp:
                    pass
            tlx.barrier_wait(acc_full_bars[acc_buf], acc_phase)
            _drain(
                acc_bufs[acc_buf],
                acc_empty_bars[acc_buf],
                out_ptr,
                user_res_ptr,
                row,
                user,
                n0,
                m_tile,
                M,
                M_PAD,
                False,
                NUM_M_TILES,
                LAST_TILE_WIDTH,
                NUM_CTAS,
                N_PER_CTA,
                M_TILE,
                EPI_SLICE,
                user_valid,
            )
        units_done += 1
    # The last CTA of this launch to get here resets the arrival counter and publishes this launch's stamp for the next
    # one (stream order makes both visible to it), so the grid may change between calls.
    done = tl.atomic_add(state_ptr + 1, 1, sem="acq_rel", scope="gpu")
    if done == tl.num_programs(0) - 1:
        tl.atomic_xchg(state_ptr + 1, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(state_ptr, stamp, sem="release", scope="gpu")


# ------------------------------------------------------------------------------------------------- kernel
@triton.jit
def _ikbo_lce_bw_kernel(
    e_cand_desc,  # E_cand as [B * K_cand, N], box [K_STEP, N_PER_CTA]
    e_user_desc,  # E_user as [U * K_user, N]
    w_cand_last_desc,  # W_cand [M, K_cand], box [W_CHUNK, K_STEP] (last m-tile pieces)
    w_cand_full_desc,  # W_cand, box [M_TILE / NUM_CTAS, K_STEP] (full m-tiles)
    w_user_last_desc,
    w_user_full_desc,
    out_ptr,  # [B, M, N]
    user_res_ptr,  # workspace: fp32 [U, M_PAD, N]
    idx_ptr,  # cand_to_user_index [B] int32
    flag_ptr,  # workspace: int32 [U, NUM_M_TILES, NUM_CTAS] ready flags (stamps)
    state_ptr,  # workspace: int32 [2] = (stamp of the last completed launch, CTAs done in this launch)
    B,
    U,
    M,
    K_cand,
    K_user,
    M_PAD: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    LAST_TILE_WIDTH: tl.constexpr,
    P0: tl.constexpr,  # LAST_TILE_WIDTH as power-of-two pieces (P0 >= P1 >= P2, unused pieces 0)
    P1: tl.constexpr,
    P2: tl.constexpr,
    W_CHUNK: tl.constexpr,  # W rows per TMA box for the last m-tile (half of the smallest piece)
    NUM_CTAS: tl.constexpr,
    N_PER_CTA: tl.constexpr,
    M_TILE: tl.constexpr,
    K_STEP: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_ACC: tl.constexpr,
    EPI_SLICE: tl.constexpr,
):
    tl.static_assert(NUM_CTAS == 2)
    e_bufs = tlx.local_alloc((K_STEP, N_PER_CTA), tlx.dtype_of(e_cand_desc), NUM_STAGES)
    w_bufs = tlx.local_alloc(
        (M_TILE // NUM_CTAS, K_STEP), tlx.dtype_of(w_cand_last_desc), NUM_STAGES
    )
    acc_bufs = tlx.local_alloc(
        (N_PER_CTA, M_TILE), tl.float32, NUM_ACC, tlx.storage_kind.tmem
    )
    full_bars = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    empty_bars = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    acc_full_bars = tlx.alloc_barriers(num_barriers=NUM_ACC, arrive_count=1)
    acc_empty_bars = tlx.alloc_barriers(num_barriers=NUM_ACC, arrive_count=NUM_CTAS)

    rank = tlx.cluster_cta_rank()
    tlx.fence_mbarrier_init_cluster()
    n0 = rank * N_PER_CTA
    num_user_units = U * NUM_M_TILES
    num_units = num_user_units + B * NUM_M_TILES

    with tlx.async_tasks():
        with tlx.async_task("default", num_regs=256):  # epilogue, 4 warps
            _epilogue(
                acc_bufs,
                acc_full_bars,
                acc_empty_bars,
                out_ptr,
                user_res_ptr,
                idx_ptr,
                flag_ptr,
                state_ptr,
                U,
                num_user_units,
                num_units,
                M,
                rank,
                M_PAD,
                NUM_M_TILES,
                LAST_TILE_WIDTH,
                NUM_CTAS,
                N_PER_CTA,
                M_TILE,
                NUM_ACC,
                EPI_SLICE,
            )

        with tlx.async_task(num_warps=1, num_regs=48):  # MMA
            pid = tl.program_id(0)
            cluster_id = pid // NUM_CTAS
            num_clusters = tl.num_programs(0) // NUM_CTAS
            k_steps_cand = tl.cdiv(K_cand, K_STEP)
            k_steps_user = tl.cdiv(K_user, K_STEP)
            stage = 0
            phase = 0
            units_done = 0
            for unit in range(cluster_id, num_units, num_clusters):
                is_user, row, m_tile = _decode_unit(
                    unit, num_clusters, num_user_units, NUM_M_TILES
                )
                num_k_steps = tl.where(is_user, k_steps_user, k_steps_cand)
                if rank == 0:
                    acc_buf, acc_phase = get_bufidx_phase(units_done, NUM_ACC)
                    tlx.barrier_wait(acc_empty_bars[acc_buf], acc_phase ^ 1)
                    if m_tile == NUM_M_TILES - 1:
                        stage, phase = _mma_unit(
                            e_bufs,
                            w_bufs,
                            acc_bufs[acc_buf],
                            full_bars,
                            empty_bars,
                            stage,
                            phase,
                            num_k_steps,
                            True,
                            P0,
                            P1,
                            P2,
                            NUM_CTAS,
                            K_STEP,
                            NUM_STAGES,
                        )
                    else:
                        stage, phase = _mma_unit(
                            e_bufs,
                            w_bufs,
                            acc_bufs[acc_buf],
                            full_bars,
                            empty_bars,
                            stage,
                            phase,
                            num_k_steps,
                            False,
                            P0,
                            P1,
                            P2,
                            NUM_CTAS,
                            K_STEP,
                            NUM_STAGES,
                        )
                    tlx.tcgen05_commit(acc_full_bars[acc_buf], two_ctas=True)
                units_done += 1

        with tlx.async_task(num_warps=1, num_regs=48):  # TMA producer
            pid = tl.program_id(0)
            cluster_id = pid // NUM_CTAS
            num_clusters = tl.num_programs(0) // NUM_CTAS
            stage = 0
            phase = 0
            for unit in range(cluster_id, num_units, num_clusters):
                is_user, row, m_tile = _decode_unit(
                    unit, num_clusters, num_user_units, NUM_M_TILES
                )
                m0 = m_tile * M_TILE
                if is_user:
                    stage, phase = _produce(
                        e_user_desc,
                        w_user_last_desc,
                        w_user_full_desc,
                        e_bufs,
                        w_bufs,
                        full_bars,
                        empty_bars,
                        stage,
                        phase,
                        row,
                        K_user,
                        n0,
                        m0,
                        m_tile,
                        rank,
                        NUM_M_TILES,
                        P0,
                        P1,
                        P2,
                        W_CHUNK,
                        NUM_CTAS,
                        N_PER_CTA,
                        M_TILE,
                        K_STEP,
                        NUM_STAGES,
                    )
                else:
                    stage, phase = _produce(
                        e_cand_desc,
                        w_cand_last_desc,
                        w_cand_full_desc,
                        e_bufs,
                        w_bufs,
                        full_bars,
                        empty_bars,
                        stage,
                        phase,
                        row,
                        K_cand,
                        n0,
                        m0,
                        m_tile,
                        rank,
                        NUM_M_TILES,
                        P0,
                        P1,
                        P2,
                        W_CHUNK,
                        NUM_CTAS,
                        N_PER_CTA,
                        M_TILE,
                        K_STEP,
                        NUM_STAGES,
                    )


@functools.lru_cache(maxsize=None)
def _num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


class _Workspace:
    """Per-device user_res buffer, ready flags and launch state. Grow-only and shared by all LCE shapes (each launch
    is self-contained given the device-side stamp). Launches from different streams are serialized with `event`,
    and launches from different host threads by the device's lock in `_workspace_locks`.
    """

    def __init__(self, device, n_user_res, n_flags):
        self.user_res = torch.empty((n_user_res,), device=device, dtype=torch.float32)
        self.flags = torch.zeros((n_flags,), device=device, dtype=torch.int32)
        self.state = torch.zeros((2,), device=device, dtype=torch.int32)
        self.event = torch.cuda.Event()
        self.last_stream = None
        self.captured = False


_workspaces = {}
# Generations replaced by a larger workspace that a captured CUDA graph may still use: kept alive.
_retired_workspaces = []
# One lock per device, held from the workspace lookup to the event record: otherwise two host threads on different
# streams can both pass the stream check before either records its launch, and their kernels run unordered.
_workspace_locks = {}


def _workspace(device, n_user_res, n_flags, capturing):
    ws = _workspaces.get(device)
    if ws is None or ws.user_res.numel() < n_user_res or ws.flags.numel() < n_flags:
        if capturing:
            raise RuntimeError(
                "tlx_ikbo_lce_blackwell: the workspace cannot be allocated during CUDA graph capture; run the "
                "largest shape (number of users x M) eagerly once before capturing"
            )
        old = ws
        if old is not None:
            if old.captured:
                _retired_workspaces.append(old)
            elif old.last_stream is not None:
                # the last launch may still be running on another stream: delay reuse of the freed blocks
                for t in (old.user_res, old.flags, old.state):
                    t.record_stream(old.last_stream)
            n_user_res = max(n_user_res, old.user_res.numel())
            n_flags = max(n_flags, old.flags.numel())
        ws = _Workspace(device, n_user_res, n_flags)
        if old is not None:  # keep the cross-stream ordering with the last launch
            ws.event, ws.last_stream = old.event, old.last_stream
        _workspaces[device] = ws
    ws.captured |= capturing
    return ws


def pack_weight(w, align=64):
    """Optional one-time weight packing: pad the row stride of W [M, K] to a multiple of `align` elements so every
    row starts on a 128-byte boundary (an unaligned 128 B TMA row touches 5 instead of 4 L2 sectors). Returns a
    [M, K] view to pass as W_cand / W_user (about -2% LCE time for K_cand = 872)."""
    M, K = w.shape
    if w.stride(1) == 1 and w.stride(0) % align == 0:
        return w
    wp = torch.zeros((M, triton.cdiv(K, align) * align), device=w.device, dtype=w.dtype)
    wp[:, :K] = w
    return wp[:, :K]


def tlx_ikbo_lce_blackwell(W_cand, W_user, E_cand, E_user, cand_to_user_index):
    """IKBO LCE on Blackwell: out[b] = W_cand @ E_cand[b] + W_user @ E_user[cand_to_user_index[b]].

    W_cand [M, K_cand], W_user [M, K_user], E_cand [B, K_cand, N], E_user [U, K_user, N] (all fp16 or all bf16, K
    padded to a multiple of 8, N == 256), cand_to_user_index [B] int32 with values in [0, U), in any order (an
    out-of-range index gives NaN rows for that candidate). Returns [B, M, N].

    The kernel uses a per-device workspace (fp32 [U, M, N] user results + ready flags) that persists across calls.
    Calls from different streams or host threads are serialized on it. CUDA graphs: run the largest shape eagerly
    once before capturing, and do not replay a graph concurrently with eager calls on another stream.

    As in the Hopper kernels, E is read as one [B * K, N] matrix: the last K step of a candidate reads the next
    candidate's leading rows times zero weights, so an Inf / NaN there propagates.
    """
    assert (
        E_cand.dim() == 3
        and E_user.dim() == 3
        and W_cand.dim() == 2
        and W_user.dim() == 2
    )
    B, Kc, N = E_cand.shape
    U, Ku, _ = E_user.shape
    M = W_cand.shape[0]
    dtype = E_cand.dtype
    assert dtype in (torch.float16, torch.bfloat16), "fp16 / bf16 only"
    assert (
        W_cand.dtype == dtype and W_user.dtype == dtype and E_user.dtype == dtype
    ), "mixed dtypes"
    assert (
        tuple(W_cand.shape) == (M, Kc)
        and tuple(W_user.shape) == (M, Ku)
        and E_user.shape[2] == N
    )
    assert Kc > 0 and Ku > 0 and M > 0
    assert (
        cand_to_user_index.dtype == torch.int32
        and tuple(cand_to_user_index.shape) == (B,)
        and cand_to_user_index.is_contiguous()
    ), "cand_to_user_index must be a contiguous int32 [B] tensor"
    device = E_cand.device
    assert all(t.device == device for t in (W_cand, W_user, E_user, cand_to_user_index))
    assert N == N_PER_CTA * NUM_CTAS, "N must be 256 (the CTA pair covers all of N)"
    if B == 0:
        return torch.empty((0, M, N), device=device, dtype=dtype)
    assert U > 0
    assert W_cand.stride(1) == 1 and W_user.stride(1) == 1
    assert (
        W_cand.stride(0) % 8 == 0 and W_user.stride(0) % 8 == 0
    ), "pad K to a multiple of 8 (16-byte TMA rows)"
    assert E_cand.is_contiguous() and E_user.is_contiguous()

    M_PAD = triton.cdiv(M, EPI_SLICE) * EPI_SLICE
    num_m_tiles = triton.cdiv(M_PAD, M_TILE)
    last_tile_width = M_PAD - (num_m_tiles - 1) * M_TILE
    p0, p1, p2 = _last_tile_pieces(last_tile_width)
    w_chunk = min(p for p in (p0, p1, p2) if p) // NUM_CTAS

    def e_desc(E, K):  # E [rows, K, N] as [rows * K, N]
        rows = E.shape[0]
        return TensorDescriptor(
            E.view(rows * K, N),
            shape=[rows * K, N],
            strides=[N, 1],
            block_shape=[K_STEP, N_PER_CTA],
        )

    def w_desc(W, K, box_rows):
        return TensorDescriptor(
            W, shape=[M, K], strides=[W.stride(0), 1], block_shape=[box_rows, K_STEP]
        )

    out = torch.empty((B, M, N), device=device, dtype=dtype)
    num_units = (U + B) * num_m_tiles
    grid = (min(num_units, _num_sms(device.index) // NUM_CTAS) * NUM_CTAS,)
    lock = _workspace_locks.setdefault(device, threading.Lock())
    # Triton launches on the current device's current stream, so make the inputs' device current.
    with torch.cuda.device(device.index), lock:
        capturing = torch.cuda.is_current_stream_capturing()
        stream = torch.cuda.current_stream(device)
        ws = _workspace(device, U * M_PAD * N, U * num_m_tiles * NUM_CTAS, capturing)
        if not capturing and ws.last_stream is not None and ws.last_stream != stream:
            stream.wait_event(ws.event)
        _ikbo_lce_bw_kernel[grid](
            e_desc(E_cand, Kc),
            e_desc(E_user, Ku),
            w_desc(W_cand, Kc, w_chunk),
            w_desc(W_cand, Kc, M_TILE // NUM_CTAS),
            w_desc(W_user, Ku, w_chunk),
            w_desc(W_user, Ku, M_TILE // NUM_CTAS),
            out,
            ws.user_res,
            cand_to_user_index,
            ws.flags,
            ws.state,
            B,
            U,
            M,
            Kc,
            Ku,
            M_PAD=M_PAD,
            NUM_M_TILES=num_m_tiles,
            LAST_TILE_WIDTH=last_tile_width,
            P0=p0,
            P1=p1,
            P2=p2,
            W_CHUNK=w_chunk,
            NUM_CTAS=NUM_CTAS,
            N_PER_CTA=N_PER_CTA,
            M_TILE=M_TILE,
            K_STEP=K_STEP,
            NUM_STAGES=NUM_STAGES,
            NUM_ACC=NUM_ACC,
            EPI_SLICE=EPI_SLICE,
            num_warps=4,
            num_stages=1,
            ctas_per_cga=(NUM_CTAS, 1, 1),
            launch_cooperative_grid=True,
        )
        if not capturing:
            ws.event.record(stream)
            ws.last_stream = stream
    return out
