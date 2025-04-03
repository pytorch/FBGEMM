# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import functools
import logging

from typing import Optional

import torch

import triton
import triton.language as tl

from fbgemm_gpu.experimental.gemm.triton_gemm import utils
from triton.runtime import driver  # @manual

logger: logging.Logger = logging.getLogger(__name__)

_NV_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    for block_size_m in [64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
    for num_ctas in [1]
]

_NV_WS_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "NUM_CONSUMER_GROUPS": max(1, num_consumer_groups),
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=1,
        num_consumer_groups=num_consumer_groups,
        num_buffers_warp_spec=num_stages,
    )
    for block_size_m in [64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [2, 3, 4]
    for num_warps in [4, 8]
    for num_consumer_groups in [0, 2]
]

_AMD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "waves_per_eu": waves_per_cu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [32, 64, 128]
    for block_size_n in [32, 64, 128, 256]
    for block_size_k in [128, 256]
    for num_stages in [1, 2]
    for num_warps, waves_per_cu in [(4, 1), (8, 2), (16, 4)]
    for matrix_instr_nonkdim in [16]
]


def early_config_prune(configs, named_args, dtsize=None, dtype=None, **kwargs):
    device = torch.cuda.current_device()
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    if dtsize is None:
        dtsize = named_args["c_ptr"].element_size()
    if dtype is None:
        dtype = named_args["c_ptr"].dtype

    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, num_consumer_groups = (
            kw["BLOCK_SIZE_M"],
            kw["BLOCK_SIZE_N"],
            kw["BLOCK_SIZE_K"],
            config.num_stages,
            config.num_warps,
            config.num_consumer_groups,
        )
        G, M, N, K = (
            named_args["G"],
            named_args["M_BUCKET"],
            named_args["N"],
            named_args["K"],
        )

        # 1. make sure we have enough smem
        max_shared_memory = driver.active.utils.get_device_properties(device)[
            "max_shared_mem"
        ]
        if torch.version.hip:
            required_shared_memory = BLOCK_N * BLOCK_K * num_stages * dtsize
        else:
            required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        use_warp_specialization = num_consumer_groups >= 1

        M_PER_GROUP = M // G
        MIN_M_TILES = 32 if torch.version.hip else 64
        # 2. make sure we don't load M tiles that are too big
        if (
            not use_warp_specialization
            and BLOCK_M > MIN_M_TILES
            and BLOCK_M > (M_PER_GROUP * 2)
        ):
            continue
        # 3. make sure we don't load N tiles that are too small
        if BLOCK_M < 128 and BLOCK_M < (M_PER_GROUP // 2):
            continue

        num_sm = driver.active.utils.get_device_properties(device)[
            "multiprocessor_count"
        ]
        N_TILES = N // BLOCK_N
        MIN_N_TILES = 32 if torch.version.hip else 64
        # 4. make sure we don't load N tiles that are too big
        if (
            not use_warp_specialization
            and BLOCK_N > MIN_N_TILES
            and M * N_TILES < num_sm
        ):
            continue
        # 5. make sure we don't load N tiles that are too small
        if BLOCK_N < 128 and M * N_TILES > 2 * num_sm:
            continue

        # 6. make sure K can be evenly divided
        if K % BLOCK_K != 0:
            continue

        # 7. make sure we can partition for ws
        if use_warp_specialization:
            if num_warps != 4:
                continue

            # "tritongpu-warp-spec-data-partition"
            m_slice = BLOCK_M // num_consumer_groups
            n_slice = BLOCK_N // num_consumer_groups
            if m_slice < 64 and n_slice < 256:
                continue

        pruned_configs.append(config)

    return pruned_configs


@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _fbgemm_grouped_gemm(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    tidx = tl.program_id(0)

    dtype: tl.dtype = c_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        if m_size > 0:
            N_start_offset = g.to(tl.int64) * N
            n_size = N
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                # pyre-ignore
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=c_ptr + M_start_offset * N,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)
                if USE_TMA_LOAD:
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl._experimental_descriptor_load(
                            a_desc_ptr,
                            [m_offset, k_offset],
                            [BLOCK_SIZE_M, BLOCK_SIZE_K],
                            dtype,
                        )
                        b = tl._experimental_descriptor_load(
                            b_desc_ptr,
                            [n_offset, k_offset],
                            [BLOCK_SIZE_N, BLOCK_SIZE_K],
                            dtype,
                        )
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    a_ptrs = (
                        a_desc_ptr
                        + (M_start_offset + offs_am[:, None]) * K
                        + offs_k[None, :]
                    )
                    b_ptrs = (
                        b_desc_ptr
                        + (N_start_offset + offs_bn[:, None]) * K
                        + offs_k[None, :]
                    )
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                        b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)
                        accumulator += tl.dot(a, b.T)
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                if USE_TMA_STORE:
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        accumulator.to(c_ptr.dtype.element_ty),
                        [m_offset, n_offset],
                    )
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c = accumulator.to(c_ptr.dtype.element_ty)
                    tl.store(
                        c_ptr
                        + (M_start_offset + offs_am[:, None]) * N
                        + offs_bn[None, :],
                        c,
                        mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                    )
                tidx += NUM_SMS

            iterated_tiles += num_tiles


# TODO(shikaili): Too much code duplication. Need to refactor.
@triton.autotune(
    configs=_NV_WS_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _fbgemm_grouped_gemm_ws(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    tl.static_assert(USE_TMA_LOAD, "Always use TMA load with warp specialziation!")

    tidx = tl.program_id(0)

    dtype: tl.dtype = c_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        if m_size > 0:
            N_start_offset = g * N
            n_size = N
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                # pyre-ignore
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=c_ptr + M_start_offset * N,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            next_iterated_tiles = iterated_tiles + num_tiles
            if (tidx >= iterated_tiles) and (tidx < next_iterated_tiles):
                for i in range(tidx, next_iterated_tiles, NUM_SMS):
                    gidx = i - iterated_tiles
                    # Split M first and N second.
                    tile_m_idx = gidx % num_m_tiles
                    tile_n_idx = gidx // num_m_tiles

                    accumulator = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
                    tl.static_assert(K % BLOCK_SIZE_K == 0)
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        with tl.async_task([0]):
                            a = tl._experimental_descriptor_load(
                                a_desc_ptr,
                                [m_offset, k_offset],
                                [BLOCK_SIZE_M, BLOCK_SIZE_K],
                                dtype,
                            )
                            b = tl._experimental_descriptor_load(
                                b_desc_ptr,
                                [n_offset, k_offset],
                                [BLOCK_SIZE_N, BLOCK_SIZE_K],
                                dtype,
                            )
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            if USE_FAST_ACCUM:
                                accumulator = tl.dot(a, b.T, accumulator)
                            else:
                                accumulator += tl.dot(a, b.T)

                    if USE_TMA_STORE:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                            n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                            tl._experimental_descriptor_store(
                                c_desc_ptr,
                                accumulator.to(c_ptr.dtype.element_ty),
                                [m_offset, n_offset],
                            )
                    else:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(
                                0, BLOCK_SIZE_M
                            )
                            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(
                                0, BLOCK_SIZE_N
                            )
                            c = accumulator.to(c_ptr.dtype.element_ty)
                            tl.store(
                                c_ptr
                                + (M_start_offset + offs_am[:, None]) * N
                                + offs_bn[None, :],
                                c,
                                mask=offs_am[:, None] < m_size
                                and offs_bn[None, :] < n_size,
                            )
                    tidx += NUM_SMS

            iterated_tiles += num_tiles


TT_FP8_DTYPE = tl.float8e4b8 if torch.version.hip else tl.float8e4nv


@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={
        "early_config_prune": functools.partial(
            early_config_prune, dtype=TT_FP8_DTYPE, dtsize=1
        )
    },
)
@triton.jit
def _fbgemm_grouped_gemm_fp8_rowwise(
    a_desc_ptr,
    a_scale_ptr,
    b_desc_ptr,
    b_scale_ptr,
    c_ptr,
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    tidx = tl.program_id(0)

    dtype = TT_FP8_DTYPE
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        if m_size > 0:
            N_start_offset = g.to(tl.int64) * N
            n_size = N
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                # pyre-ignore
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=c_ptr + M_start_offset * N,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)
                if USE_TMA_LOAD:
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl._experimental_descriptor_load(
                            a_desc_ptr,
                            [m_offset, k_offset],
                            [BLOCK_SIZE_M, BLOCK_SIZE_K],
                            dtype,
                        )
                        b = tl._experimental_descriptor_load(
                            b_desc_ptr,
                            [n_offset, k_offset],
                            [BLOCK_SIZE_N, BLOCK_SIZE_K],
                            dtype,
                        )
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    a_ptrs = (
                        a_desc_ptr
                        + (M_start_offset + offs_am[:, None]) * K
                        + offs_k[None, :]
                    )
                    b_ptrs = (
                        b_desc_ptr
                        + (N_start_offset + offs_bn[:, None]) * K
                        + offs_k[None, :]
                    )
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                        b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)
                        accumulator += tl.dot(a, b.T)
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                a_scale = tl.load(
                    a_scale_ptr + M_start_offset + offs_am[:, None],
                    mask=offs_am[:, None] < m_size,
                )
                b_scale = tl.load(
                    b_scale_ptr + N_start_offset + offs_bn[None, :],
                    mask=offs_bn[None, :] < n_size,
                )
                c = accumulator.to(tl.float32) * a_scale * b_scale

                if USE_TMA_STORE:
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        c.to(c_ptr.dtype.element_ty),
                        [m_offset, n_offset],
                    )
                else:
                    tl.store(
                        c_ptr
                        + (M_start_offset + offs_am[:, None]) * N
                        + offs_bn[None, :],
                        c,
                        mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                    )
                tidx += NUM_SMS

            iterated_tiles += num_tiles


# TODO(shikaili): Too much code duplication. Need to refactor.
@triton.autotune(
    configs=_NV_WS_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={
        "early_config_prune": functools.partial(
            early_config_prune, dtype=TT_FP8_DTYPE, dtsize=1
        )
    },
)
@triton.jit
def _fbgemm_grouped_gemm_fp8_rowwise_ws(
    a_desc_ptr,
    a_scale_ptr,
    b_desc_ptr,
    b_scale_ptr,
    c_ptr,
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    tl.static_assert(USE_TMA_LOAD, "Always use TMA load with warp specialziation!")

    tidx = tl.program_id(0)

    dtype = TT_FP8_DTYPE
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        if m_size > 0:
            N_start_offset = g * N
            n_size = N
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                # pyre-ignore
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=c_ptr + M_start_offset * N,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            next_iterated_tiles = iterated_tiles + num_tiles
            if (tidx >= iterated_tiles) and (tidx < next_iterated_tiles):
                for i in range(tidx, next_iterated_tiles, NUM_SMS):
                    gidx = i - iterated_tiles
                    # Split M first and N second.
                    tile_m_idx = gidx % num_m_tiles
                    tile_n_idx = gidx // num_m_tiles

                    accumulator = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
                    tl.static_assert(K % BLOCK_SIZE_K == 0)

                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        with tl.async_task([0]):
                            a = tl._experimental_descriptor_load(
                                a_desc_ptr,
                                [m_offset, k_offset],
                                [BLOCK_SIZE_M, BLOCK_SIZE_K],
                                dtype,
                            )
                            b = tl._experimental_descriptor_load(
                                b_desc_ptr,
                                [n_offset, k_offset],
                                [BLOCK_SIZE_N, BLOCK_SIZE_K],
                                dtype,
                            )
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            if USE_FAST_ACCUM:
                                accumulator = tl.dot(a, b.T, accumulator)
                            else:
                                accumulator += tl.dot(a, b.T)

                    with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                        a_scale = tl.load(
                            a_scale_ptr + M_start_offset + offs_am[:, None],
                            mask=offs_am[:, None] < m_size,
                        )
                        b_scale = tl.load(
                            b_scale_ptr + N_start_offset + offs_bn[None, :],
                            mask=offs_bn[None, :] < n_size,
                        )
                        c = accumulator.to(tl.float32) * a_scale * b_scale

                    if USE_TMA_STORE:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                            n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                            tl._experimental_descriptor_store(
                                c_desc_ptr,
                                c.to(c_ptr.dtype.element_ty),
                                [m_offset, n_offset],
                            )
                    else:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            tl.store(
                                c_ptr
                                + (M_start_offset + offs_am[:, None]) * N
                                + offs_bn[None, :],
                                c,
                                mask=offs_am[:, None] < m_size
                                and offs_bn[None, :] < n_size,
                            )
                    tidx += NUM_SMS

            iterated_tiles += num_tiles


def _grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    use_fast_accum: bool = False,
    use_warp_specialization: bool = False,
) -> torch.Tensor:

    USE_TMA_LOAD = not torch.version.hip
    USE_TMA_STORE = False

    if USE_TMA_LOAD and not utils.HAS_TMA_DESC:
        USE_TMA_LOAD = False
        logging.warning("TMA load is disabled as there is no TMA descriptor support!")

    if USE_TMA_STORE and not utils.HAS_TMA_DESC:
        USE_TMA_STORE = False
        logging.warning("TMA store is disabled as there is no TMA descriptor support!")

    if use_warp_specialization and torch.version.hip:
        logging.warning(
            "Warp specialization is disabled as it is not supported on ROCm."
        )
        use_warp_specialization = False

    G = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    M, K = x.shape
    N = w.shape[0] // G
    assert K == w.shape[1]

    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    if M == 0 or N == 0:
        return y

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    desc_helper = None
    desc_x = x
    desc_w = w
    workspace = None

    if USE_TMA_LOAD:
        desc_helper = utils.TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("x")
        desc_helper.init_tma_descriptor("w")
        desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
        desc_w = desc_helper.get_tma_descriptor_kernel_param("w")

    if USE_TMA_STORE:
        workspace = torch.empty(
            NUM_SMS * utils.TmaAutoTuneHelper.TMA_SIZE,
            device=x.device,
            dtype=torch.uint8,
        )

    def grid(META):
        if USE_TMA_LOAD:
            nonlocal desc_helper  # noqa: F824
            desc_helper.fill_2d_tma_descriptor(
                "x",
                x.data_ptr(),
                M,
                K,
                META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
                META["BLOCK_SIZE_K"],
                x.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "w",
                w.data_ptr(),
                N * G,
                K,
                META["BLOCK_SIZE_N"],
                META["BLOCK_SIZE_K"],
                w.element_size(),
            )

        return (NUM_SMS,)

    M_BUCKET_CAP = 16384
    M_BUCKET = min(triton.next_power_of_2(M), M_BUCKET_CAP)
    if x_scale is not None and w_scale is not None:
        assert x_scale.is_contiguous()
        assert w_scale.is_contiguous()
        fn = (
            _fbgemm_grouped_gemm_fp8_rowwise_ws
            if use_warp_specialization
            else _fbgemm_grouped_gemm_fp8_rowwise
        )
        fn[grid](
            desc_x,
            x_scale,
            desc_w,
            w_scale,
            y,
            workspace,
            m_sizes,
            G,
            M_BUCKET,
            N,
            K,
            NUM_SMS,
            USE_TMA_LOAD,
            USE_TMA_STORE,
            use_fast_accum,
        )
    else:
        assert x_scale is None
        assert w_scale is None
        fn = (
            _fbgemm_grouped_gemm_ws if use_warp_specialization else _fbgemm_grouped_gemm
        )
        fn[grid](
            desc_x,
            desc_w,
            y,
            workspace,
            m_sizes,
            G,
            M_BUCKET,
            N,
            K,
            NUM_SMS,
            USE_TMA_LOAD,
            USE_TMA_STORE,
            use_fast_accum,
        )

    return y


def grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_fast_accum: bool = True,
    *,
    _use_warp_specialization: bool = False,
) -> torch.Tensor:
    return _grouped_gemm(
        x,
        w,
        m_sizes,
        use_fast_accum=use_fast_accum,
        use_warp_specialization=_use_warp_specialization,
    )


def grouped_gemm_fp8_rowwise(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    use_fast_accum: bool = True,
    *,
    _use_warp_specialization: bool = False,
) -> torch.Tensor:
    return _grouped_gemm(
        x,
        w,
        m_sizes,
        x_scale,
        w_scale,
        use_fast_accum=use_fast_accum,
        use_warp_specialization=_use_warp_specialization,
    )
