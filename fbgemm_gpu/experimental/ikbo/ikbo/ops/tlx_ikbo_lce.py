# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TLX warp-specialized persistent IKBO LCE kernel (Hopper H100).
#
# Architecture: 1 producer + 2 consumer warp groups per CTA.
#   - Producer:  issues TMA loads into a circular SMEM buffer.
#   - Consumers: perform GEMM accumulation via WGMMA, ping-ponging
#                tiles so one consumer's epilogue overlaps the other's compute.
#
# The kernel fuses both matmul stages into a single persistent launch:
#   Stage 1 (user):      user_res[u] = W_user @ E_user[u]
#   Stage 2 (candidate): out[b]      = W_cand @ E_cand[b] + user_res[u(b)]
#
# Three synchronization mechanisms:
#   1. Producer-consumer mbarriers gate the circular SMEM buffer.
#   2. Named barriers alternate two consumer replicas (ping-pong).
#   3. Per-tile atomic flags signal user-result readiness across CTAs.
#
# Tile dispatch: ascending for user phase, descending for candidate phase,
# balancing per-SM workload across partial waves.

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

# Ping-pong barrier pair: each consumer replica waits on one and signals the other.
PINGPONG_BARRIER_0 = tl.constexpr(9)
PINGPONG_BARRIER_1 = tl.constexpr(10)
# Syncs warp group after single-thread flag polling completes.
POLL_SYNC_BARRIER = tl.constexpr(12)

# Thread counts for named barrier participation.
NUM_CONSUMER_THREADS = tl.constexpr(256)  # 2 warp groups × 4 warps × 32 threads/warp
NUM_WARPGROUP_THREADS = tl.constexpr(128)  # 1 warp group × 4 warps × 32 threads/warp


@triton.jit
def swizzle_tile(
    tile_id,
    num_pids_per_batch: tl.constexpr,
    num_pid_m_per_batch: tl.constexpr,
    num_pid_n: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Map a linear tile_id to (pid_batch, pid_m, pid_n) with grouped ordering for L2 cache reuse."""
    pid_batch = tile_id // num_pids_per_batch
    local_tile = tile_id % num_pids_per_batch
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = local_tile // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m_per_batch - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
    pid_n = (local_tile % num_pid_in_group) // group_size_m
    return pid_batch, pid_m, pid_n


def _block_size_hook(nargs):
    """Set TMA descriptor block shapes from autotuned tile sizes."""
    BM = nargs["BM"]
    BN = nargs["BN"]
    BK = nargs["BK"]
    nargs["user_a_desc"].block_shape = [BM, BK]
    nargs["cand_a_desc"].block_shape = [BM, BK]
    nargs["user_b_desc"].block_shape = [BK, BN]
    nargs["cand_b_desc"].block_shape = [BK, BN]
    nargs["out_desc"].block_shape = [BM, BN // 2]


def _tlx_autotune_configs():
    if os.environ.get("FAST_TUNE"):
        return [
            triton.Config(
                {"BM": 64, "BN": 128, "BK": 64, "NUM_STAGES": 6, "GROUP_SIZE_M": 8},
                num_stages=1,
                num_warps=4,
                pre_hook=_block_size_hook,
            )
        ]
    return [
        triton.Config(
            {"BM": bm, "BN": bn, "BK": bk, "NUM_STAGES": ns, "GROUP_SIZE_M": 8},
            num_stages=1,
            num_warps=4,
            pre_hook=_block_size_hook,
        )
        for bm in [64, 128]
        for bn in [64, 128]
        for bk in [64, 128]
        for ns in [3, 4, 5, 6]
    ]


@triton.autotune(
    configs=_tlx_autotune_configs(),
    key=["M", "N", "K_USER", "K_CAND"],
)
@triton.jit
def tlx_ikbo_lce_kernel(
    user_a_desc,  # W_user [M, K_user]
    user_b_desc,  # E_user [num_users * K_user, N]
    cand_a_desc,  # W_cand [M, K_cand]
    cand_b_desc,  # E_cand [B * K_cand, N]
    cand_to_user_ptr,  # [B], int32
    user_flag_ptr,  # [num_users, M_tiles, N_tiles], int32
    user_res_ptr,  # [num_users, M, N]
    out_desc,  # [B * M_padded, N]
    user_flag_stride_a: tl.constexpr,
    user_flag_stride_b: tl.constexpr,
    user_flag_stride_c: tl.constexpr,
    user_res_stride_a: tl.constexpr,
    user_res_stride_b: tl.constexpr,
    user_res_stride_c: tl.constexpr,
    NUM_USERS,
    B,
    NUM_SMS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K_USER: tl.constexpr,
    K_CAND: tl.constexpr,
    M_PADDED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Warp-specialized persistent IKBO LCE kernel.

    Fuses user GEMM (Stage 1) and candidate GEMM + broadcast (Stage 2) into
    a single persistent launch with 1 producer + 2 consumer warp groups.
    """
    # Allocate SMEM buffers: circular buffer for A/B tiles
    a = tlx.local_alloc((BM, BK), tlx.dtype_of(cand_a_desc), NUM_STAGES)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(cand_b_desc), NUM_STAGES)

    # Producer-consumer mbarriers for the circular buffer
    mainloop_empty_bar = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    mainloop_full_bar = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    with tlx.async_tasks():

        ###########################
        # Producer
        ###########################
        with tlx.async_task("default"):
            pid = tl.program_id(axis=0)
            num_pid_m_per_batch = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pids_per_batch = num_pid_m_per_batch * num_pid_n
            num_user_pids = NUM_USERS * num_pids_per_batch
            num_cand_pids = B * num_pids_per_batch

            p = 1
            buf = 0

            ###########################
            # Producer: User Phase
            ###########################
            # Ascending tile dispatch (start at pid)
            user_start_pid = pid
            for tile_id in range(user_start_pid, num_user_pids, NUM_SMS):
                pid_batch, pid_m, pid_n = swizzle_tile(
                    tile_id,
                    num_pids_per_batch,
                    num_pid_m_per_batch,
                    num_pid_n,
                    GROUP_SIZE_M,
                )

                offset_am = pid_m * BM
                offset_bn = pid_n * BN
                offset_k_base = pid_batch * K_USER
                for offset_k in range(0, K_USER, BK):
                    empty = tlx.local_view(mainloop_empty_bar, buf)
                    full = tlx.local_view(mainloop_full_bar, buf)
                    tlx.barrier_wait(bar=empty, phase=p)
                    tlx.barrier_expect_bytes(full, BM * BK * 2 + BK * BN * 2)
                    data_a = tlx.local_view(a, buf)
                    tlx.async_descriptor_load(
                        user_a_desc, data_a, [offset_am, offset_k], full
                    )
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(
                        user_b_desc,
                        data_b,
                        [offset_k_base + offset_k, offset_bn],
                        full,
                    )
                    p = p ^ (buf == (NUM_STAGES - 1))
                    buf = (buf + 1) % NUM_STAGES

            ###########################
            # Producer: Candidate Phase
            ###########################
            # Descending tile dispatch (start at NUM_SMS-1-pid)
            # to balance per-SM workload across partial waves
            for tile_id in range(NUM_SMS - 1 - pid, num_cand_pids, NUM_SMS):
                pid_batch, pid_m, pid_n = swizzle_tile(
                    tile_id,
                    num_pids_per_batch,
                    num_pid_m_per_batch,
                    num_pid_n,
                    GROUP_SIZE_M,
                )

                offset_am = pid_m * BM
                offset_bn = pid_n * BN
                offset_k_base = pid_batch * K_CAND
                for offset_k in range(0, K_CAND, BK):
                    empty = tlx.local_view(mainloop_empty_bar, buf)
                    full = tlx.local_view(mainloop_full_bar, buf)
                    tlx.barrier_wait(bar=empty, phase=p)
                    tlx.barrier_expect_bytes(full, BM * BK * 2 + BK * BN * 2)
                    data_a = tlx.local_view(a, buf)
                    tlx.async_descriptor_load(
                        cand_a_desc, data_a, [offset_am, offset_k], full
                    )
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(
                        cand_b_desc,
                        data_b,
                        [offset_k_base + offset_k, offset_bn],
                        full,
                    )

                    p = p ^ (buf == (NUM_STAGES - 1))
                    buf = (buf + 1) % NUM_STAGES

        ###########################
        # Consumer
        ###########################
        # Two replicas ping-pong tiles: one's epilogue overlaps the other's GEMM.
        with tlx.async_task(num_warps=4, replicate=2, registers=232):
            pid = tl.program_id(axis=0)
            cid: tl.constexpr = tlx.async_task_replica_id()
            num_pid_m_per_batch = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pids_per_batch = num_pid_m_per_batch * num_pid_n
            num_user_pids = NUM_USERS * num_pids_per_batch
            num_cand_pids = B * num_pids_per_batch
            k_user_tiles = tl.cdiv(K_USER, BK)
            k_cand_tiles = tl.cdiv(K_CAND, BK)

            ###########################
            # Consumer: User Phase
            ###########################
            # Ping-pong via named barriers.
            # Consumer 0 enters first (consumer 1 pre-seeds its barrier).
            if cid == 1:
                tlx.named_barrier_arrive(PINGPONG_BARRIER_0, NUM_CONSUMER_THREADS)
            total_k_offset = cid * k_user_tiles
            user_start_pid = pid + cid * NUM_SMS
            for tile_id in range(user_start_pid, num_user_pids, NUM_SMS * 2):
                buf = total_k_offset % NUM_STAGES
                p = (total_k_offset // NUM_STAGES) % 2
                total_k_offset += 2 * k_user_tiles

                last_buf = buf

                pid_batch, pid_m, pid_n = swizzle_tile(
                    tile_id,
                    num_pids_per_batch,
                    num_pid_m_per_batch,
                    num_pid_n,
                    GROUP_SIZE_M,
                )
                acc = tl.zeros([BM, BN], dtype=tl.float32)

                # Wait for the other replica's GEMM to finish
                if cid == 0:
                    tlx.named_barrier_wait(PINGPONG_BARRIER_0, NUM_CONSUMER_THREADS)
                else:
                    tlx.named_barrier_wait(PINGPONG_BARRIER_1, NUM_CONSUMER_THREADS)

                # K-loop round 0
                full = tlx.local_view(mainloop_full_bar, buf)
                tlx.barrier_wait(bar=full, phase=p)
                data_a = tlx.local_view(a, buf)
                data_b = tlx.local_view(b, buf)
                acc = tlx.async_dot(data_a, data_b, acc)
                p = p ^ (buf == (NUM_STAGES - 1))
                buf = (buf + 1) % NUM_STAGES
                # K-loop body
                for _ in range(1, k_user_tiles):
                    full = tlx.local_view(mainloop_full_bar, buf)
                    tlx.barrier_wait(bar=full, phase=p)
                    data_a = tlx.local_view(a, buf)
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(data_a, data_b, acc)
                    acc = tlx.async_dot_wait(1, acc)
                    empty = tlx.local_view(mainloop_empty_bar, last_buf)
                    tlx.barrier_arrive(empty)

                    last_buf = buf
                    p = p ^ (buf == (NUM_STAGES - 1))
                    buf = (buf + 1) % NUM_STAGES

                # Signal the other replica to start its next tile
                if cid == 0:
                    tlx.named_barrier_arrive(PINGPONG_BARRIER_1, NUM_CONSUMER_THREADS)
                else:
                    tlx.named_barrier_arrive(PINGPONG_BARRIER_0, NUM_CONSUMER_THREADS)

                acc = tlx.async_dot_wait(0, acc)
                empty = tlx.local_view(mainloop_empty_bar, last_buf)
                tlx.barrier_arrive(empty)
                acc = acc.to(tl.float16)

                # Store user result to global memory
                offs_m = pid_m * BM + tl.arange(0, BM)
                offs_n = pid_n * BN + tl.arange(0, BN)
                idx_m = offs_m[:, None]
                idx_n = offs_n[None, :]
                mask = (idx_m < M) & (idx_n < N)
                user_res_ptrs = (
                    user_res_ptr
                    + pid_batch * user_res_stride_a
                    + idx_m * user_res_stride_b
                    + idx_n * user_res_stride_c
                )
                tl.store(user_res_ptrs, acc, mask)

                # Signal tile readiness for cross-CTA candidate consumers
                uf_off = (
                    pid_batch * user_flag_stride_a
                    + pid_m * user_flag_stride_b
                    + pid_n * user_flag_stride_c
                )
                tl.atomic_add(user_flag_ptr + uf_off, 1, sem="release", scope="gpu")

            ###########################
            # Consumer: Candidate Phase
            ###########################
            # Ping-pong flows naturally from user phase. Whichever replica
            # finished last in the user phase arrives second, so the OTHER
            # replica enters the candidate phase first.  Swap tile/buffer
            # assignment so the first-to-enter replica gets producer tile 0.
            user_tiles_for_sm = (num_user_pids + NUM_SMS - 1 - pid) // NUM_SMS
            swap = user_tiles_for_sm % 2

            if cid == 0:
                cand_cid_offset = swap * k_cand_tiles
                cand_start_pid = NUM_SMS - 1 - pid + swap * NUM_SMS
            else:
                cand_cid_offset = (1 - swap) * k_cand_tiles
                cand_start_pid = NUM_SMS - 1 - pid + (1 - swap) * NUM_SMS

            total_k_offset = user_tiles_for_sm * k_user_tiles + cand_cid_offset
            k_cand_stride = 2 * k_cand_tiles
            for tile_id in range(cand_start_pid, num_cand_pids, NUM_SMS * 2):
                buf = total_k_offset % NUM_STAGES
                p = (total_k_offset // NUM_STAGES) % 2
                total_k_offset += k_cand_stride

                last_buf = buf

                pid_batch, pid_m, pid_n = swizzle_tile(
                    tile_id,
                    num_pids_per_batch,
                    num_pid_m_per_batch,
                    num_pid_n,
                    GROUP_SIZE_M,
                )

                acc = tl.zeros([BM, BN], dtype=tl.float32)

                # Wait for the other replica's GEMM to finish
                if cid == 0:
                    tlx.named_barrier_wait(PINGPONG_BARRIER_0, NUM_CONSUMER_THREADS)
                else:
                    tlx.named_barrier_wait(PINGPONG_BARRIER_1, NUM_CONSUMER_THREADS)
                # K-loop round 0
                full = tlx.local_view(mainloop_full_bar, buf)
                tlx.barrier_wait(bar=full, phase=p)
                data_a = tlx.local_view(a, buf)
                data_b = tlx.local_view(b, buf)
                acc = tlx.async_dot(data_a, data_b, acc)
                p = p ^ (buf == (NUM_STAGES - 1))
                buf = (buf + 1) % NUM_STAGES
                # K-loop body
                for _ in range(1, k_cand_tiles):
                    full = tlx.local_view(mainloop_full_bar, buf)
                    tlx.barrier_wait(bar=full, phase=p)
                    data_a = tlx.local_view(a, buf)
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(data_a, data_b, acc)
                    acc = tlx.async_dot_wait(1, acc)
                    empty = tlx.local_view(mainloop_empty_bar, last_buf)
                    tlx.barrier_arrive(empty)

                    last_buf = buf
                    p = p ^ (buf == (NUM_STAGES - 1))
                    buf = (buf + 1) % NUM_STAGES

                # Signal the other replica to start its next tile
                if cid == 0:
                    tlx.named_barrier_arrive(PINGPONG_BARRIER_1, NUM_CONSUMER_THREADS)
                else:
                    tlx.named_barrier_arrive(PINGPONG_BARRIER_0, NUM_CONSUMER_THREADS)

                acc = tlx.async_dot_wait(0, acc)
                empty = tlx.local_view(mainloop_empty_bar, last_buf)
                tlx.barrier_arrive(empty)

                # Cross-CTA sync: wait for user result readiness.
                # One thread per warp group polls (relaxed spin -> acquire),
                # then all threads rendezvous at POLL_SYNC_BARRIER.
                user_pid_batch = tl.load(
                    cand_to_user_ptr + pid_batch, eviction_policy="evict_last"
                )
                if tlx.thread_id(axis=0) % NUM_WARPGROUP_THREADS == 0:
                    user_flag_pid_ptr = (
                        user_flag_ptr
                        + user_pid_batch * user_flag_stride_a
                        + pid_m * user_flag_stride_b
                        + pid_n * user_flag_stride_c
                    )
                    # Initial relaxed load (no ordering guarantees)
                    ready = tl.inline_asm_elementwise(
                        "ld.relaxed.gpu.global.b32 $0, [$1];",
                        "=r,l",
                        [user_flag_pid_ptr],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
                    # Spin with nanosleep until flag is set
                    while ready == 0:
                        ready = tl.inline_asm_elementwise(
                            "nanosleep.u32 50; ld.relaxed.gpu.global.b32 $0, [$1];",
                            "=r,l",
                            [user_flag_pid_ptr],
                            dtype=tl.int32,
                            is_pure=False,
                            pack=1,
                        )
                    # Acquire load to establish happens-before
                    tl.inline_asm_elementwise(
                        "ld.acquire.gpu.global.b32 $0, [$1];",
                        "=r,l",
                        [user_flag_pid_ptr],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
                tlx.named_barrier_wait(POLL_SYNC_BARRIER, NUM_WARPGROUP_THREADS)

                # Epilogue: load user result and combine with candidate GEMM output
                offs_m = (pid_m * BM + tl.arange(0, BM)) % M
                offs_n = (pid_n * BN + tl.arange(0, BN)) % N
                user_res_ptrs = (
                    user_res_ptr
                    + user_pid_batch * user_res_stride_a
                    + offs_m[:, None] * user_res_stride_b
                    + offs_n[None, :] * user_res_stride_c
                )
                # Boundary is handled by TMA store; skip mask for speed
                user_res_rmem = tl.load(user_res_ptrs, eviction_policy="evict_last")
                acc = acc.to(tl.float16) + user_res_rmem

                # TMA store: split into two BN/2 subtiles to spare SMEM to allow more circular buffers
                offset_cm = pid_m * BM + pid_batch * M_PADDED
                offset_cn = pid_n * BN
                acc = tl.reshape(acc, (BM, 2, BN // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                out_desc.store(
                    [offset_cm, offset_cn],
                    acc0.to(tlx.dtype_of(out_desc)),
                )
                out_desc.store(
                    [offset_cm, offset_cn + BN // 2],
                    acc1.to(tlx.dtype_of(out_desc)),
                )


def create_user_flag(compression_w_user, embeddings_user):
    """Allocate per-tile readiness flags for cross-CTA synchronization.

    Returns a zero-initialized [num_users, M_tiles, N_tiles] int32 tensor.
    User-phase consumers atomically increment flags after storing each tile;
    candidate-phase consumers spin on them before the epilogue add.
    """
    configs = tlx_ikbo_lce_kernel.configs
    MIN_BM = min(c.kwargs["BM"] for c in configs)
    MIN_BN = min(c.kwargs["BN"] for c in configs)
    num_users = embeddings_user.shape[0]
    m_tiles = triton.cdiv(compression_w_user.shape[0], MIN_BM)
    n_tiles = triton.cdiv(embeddings_user.shape[2], MIN_BN)
    return torch.zeros(
        (num_users, m_tiles, n_tiles),
        device=compression_w_user.device,
        dtype=torch.int32,
    ).requires_grad_(False)


def tlx_ikbo_lce(
    compression_w_cand: torch.Tensor,
    compression_w_user: torch.Tensor,
    embeddings_cand: torch.Tensor,
    embeddings_user: torch.Tensor,
    cand_to_user_index: torch.Tensor,
    user_flag: torch.Tensor,
) -> torch.Tensor:
    """Warp-specialized persistent IKBO LCE.

    Fuses both matmul stages into a single persistent kernel:
      Stage 1: user_res[u] = W_user @ E_user[u]
      Stage 2: out[b]      = W_cand @ E_cand[b] + user_res[u(b)]

    Uses TMA for global-to-SMEM transfers, WGMMA for tensor core compute,
    and cross-CTA atomic flags for user-result readiness signaling.

    Args:
        compression_w_cand: Candidate compression weights [M, K_cand].
        compression_w_user: User compression weights [M, K_user].
        embeddings_cand: Candidate embeddings [B, K_cand, N].
        embeddings_user: User embeddings [num_users, K_user, N].
        cand_to_user_index: Maps candidate index to user index [B], int32.
        user_flag: Per-tile readiness flags from create_user_flag().
    """
    B = embeddings_cand.shape[0]
    M = compression_w_cand.shape[0]
    K_USER = compression_w_user.shape[1]
    K_CAND = compression_w_cand.shape[1]
    N = embeddings_cand.shape[2]
    NUM_USERS = embeddings_user.shape[0]

    user_res = torch.empty(
        (NUM_USERS, M, N), device=embeddings_cand.device, dtype=embeddings_cand.dtype
    )
    user_flag.zero_()

    MAX_BM = max(c.kwargs["BM"] for c in tlx_ikbo_lce_kernel.configs)
    M_padded = triton.cdiv(M, MAX_BM) * MAX_BM
    out = torch.empty(
        (B, M_padded, N), dtype=torch.float16, device=embeddings_cand.device
    )
    NUM_SMS = torch.cuda.get_device_properties(
        embeddings_cand.device
    ).multi_processor_count

    # TMA descriptors for user-side tensors
    compression_w_user_desc = TensorDescriptor(
        compression_w_user,
        shape=[M, K_USER],
        strides=[K_USER, 1],
        block_shape=[1, 1],
    )
    embeddings_user_desc = TensorDescriptor(
        embeddings_user,
        shape=[NUM_USERS * K_USER, N],
        strides=[N, 1],
        block_shape=[1, 1],
    )

    # TMA descriptors for candidate-side tensors
    compression_w_cand_desc = TensorDescriptor(
        compression_w_cand,
        shape=[M, K_CAND],
        strides=[K_CAND, 1],
        block_shape=[1, 1],
    )
    embeddings_cand_desc = TensorDescriptor(
        embeddings_cand,
        shape=[B * K_CAND, N],
        strides=[N, 1],
        block_shape=[1, 1],
    )

    # TMA descriptor for output
    out_desc = TensorDescriptor(
        out,
        shape=[B * M_padded, N],
        strides=[N, 1],
        block_shape=[1, 1],
    )

    def grid(META):
        num_m_blocks = triton.cdiv(M, META["BM"])
        num_n_blocks = triton.cdiv(N, META["BN"])
        total_blocks = B * num_m_blocks * num_n_blocks
        return (min(NUM_SMS, total_blocks),)

    tlx_ikbo_lce_kernel[grid](
        compression_w_user_desc,
        embeddings_user_desc,
        compression_w_cand_desc,
        embeddings_cand_desc,
        cand_to_user_index,
        user_flag,
        user_res,
        out_desc,
        user_flag.stride(0),
        user_flag.stride(1),
        user_flag.stride(2),
        user_res.stride(0),
        user_res.stride(1),
        user_res.stride(2),
        NUM_USERS,
        B,
        NUM_SMS,
        M,
        N,
        K_USER,
        K_CAND,
        M_padded,
    )
    return out[:, :M, :]
