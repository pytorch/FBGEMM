# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl


# Function APIs
def combine_shuffling(
    tokens: torch.Tensor,
    token_counts: torch.Tensor,
    expert_start: Optional[int] = None,
    expert_end: Optional[int] = None,
    is_balanced: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # pyre-ignore
    return _combine_or_split_shuffling(
        tokens=tokens,
        token_counts=token_counts,
        expert_start=expert_start,
        expert_end=expert_end,
        is_balanced=is_balanced,
        is_combine=True,
    )


def split_shuffling(
    tokens: torch.Tensor,
    token_counts: torch.Tensor,
    expert_start: Optional[int] = None,
    expert_end: Optional[int] = None,
    is_balanced: bool = False,
) -> torch.Tensor:
    # pyre-ignore
    return _combine_or_split_shuffling(
        tokens=tokens,
        token_counts=token_counts,
        expert_start=expert_start,
        expert_end=expert_end,
        is_balanced=is_balanced,
        is_combine=False,
    )


def _combine_or_split_shuffling(
    tokens: torch.Tensor,
    token_counts: torch.Tensor,
    expert_start: Optional[int],
    expert_end: Optional[int],
    is_balanced: bool,
    is_combine: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # T is intentionally ignored in kernel interface to avoid recompilation
    assert tokens.is_contiguous()
    assert token_counts.is_contiguous()

    T, D = tokens.shape
    EP, E = token_counts.shape

    if expert_start is None:
        expert_start = 0
    if expert_end is None:
        expert_end = E

    EG: int = expert_end - expert_start

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    SPLIT_D = max(NUM_SMS // (EP * EG), 1)
    SPLIT_D = triton.next_power_of_2(SPLIT_D + 1)
    if T <= 1024:
        SPLIT_D //= 2

    if is_combine:
        grid = (EP * EG * SPLIT_D + 1,)
    else:
        grid = (EP * EG * SPLIT_D,)

    output_tokens = torch.empty_like(tokens)
    if is_combine:
        output_token_counts = torch.empty(
            EG + 1, dtype=token_counts.dtype, device=token_counts.device
        )
    else:
        output_token_counts = None
    T_BUCKET_CAP = 16384
    T_BUCKET = min(triton.next_power_of_2(T), T_BUCKET_CAP)

    _fbgemm_combine_or_split_shuffling[grid](
        tokens,
        token_counts,
        output_tokens,
        output_token_counts,
        is_combine,
        is_balanced,
        T_BUCKET,
        EG,
        EP,
        E,
        D,
        expert_start,
        SPLIT_D,
    )

    if is_combine:
        assert output_token_counts is not None
        return output_tokens, output_token_counts
    else:
        return output_tokens


# Torch Custom Op Registrations
_COMBINE_SHUFFLING_OP_NAME = "fbgemm::combine_shuffling"

torch.library.define(
    "fbgemm::combine_shuffling",
    "(Tensor tokens, Tensor token_counts, int? expert_start = None, int? expert_end = None, bool? is_balanced = False) -> (Tensor, Tensor)",
)


@torch.library.impl(_COMBINE_SHUFFLING_OP_NAME, "Meta")
def combine_shuffling_meta(
    tokens,
    token_counts,
    expert_start,
    expert_end,
    is_balanced,
):
    _, E = token_counts.shape
    if expert_start is None:
        expert_start = 0
    if expert_end is None:
        expert_end = E

    EG: int = expert_end - expert_start
    output_tokens = torch.empty_like(tokens)
    output_token_counts = torch.empty(
        EG + 1, dtype=token_counts.dtype, device=token_counts.device
    )
    return output_tokens, output_token_counts


@torch.library.impl(_COMBINE_SHUFFLING_OP_NAME, "CUDA")
def combine_shuffling_cuda(
    tokens,
    token_counts,
    expert_start=None,
    expert_end=None,
    is_balanced=False,
):
    return combine_shuffling(
        tokens,
        token_counts,
        expert_start,
        expert_end,
        is_balanced,
    )


_SPLIT_SHUFFLING_OP_NAME = "fbgemm::split_shuffling"

torch.library.define(
    "fbgemm::split_shuffling",
    "(Tensor tokens, Tensor token_counts, int? expert_start = None, int? expert_end = None, bool? is_balanced = False) -> Tensor",
)


@torch.library.impl(_SPLIT_SHUFFLING_OP_NAME, "Meta")
def split_shuffling_meta(
    tokens,
    token_counts,
    expert_start,
    expert_end,
    is_balanced,
):
    output_tokens = torch.empty_like(tokens)
    return output_tokens


@torch.library.impl(_SPLIT_SHUFFLING_OP_NAME, "CUDA")
def split_shuffling_cuda(
    tokens,
    token_counts,
    expert_start=None,
    expert_end=None,
    is_balanced=False,
):
    return split_shuffling(
        tokens,
        token_counts,
        expert_start,
        expert_end,
        is_balanced,
    )


# Kernel Implementations
_NV_CONFIGS = [
    triton.Config(
        {
            "BLOCK_T": block_t,
            "BLOCK_D": block_d,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    for block_t in [32, 64]
    for block_d in [256, 512, 1024]
    for num_stages in [1, 3]
    for num_warps in [8, 16]
    for num_ctas in [1]
]

_AMD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_T": block_t,
            "BLOCK_D": block_d,
            "waves_per_eu": waves_per_cu,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_t in [32, 64]
    for block_d in [256, 512, 1024]
    for num_stages in [1, 3]
    for num_warps, waves_per_cu in [(8, 2), (16, 4)]
]


@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=[
        "COMBINE",
        "BALANCED",
        "T_BUCKET",
        "EG",
        "EP",
        "E",
        "D",
    ],
)
@triton.jit
def _fbgemm_combine_or_split_shuffling(
    input_tokens_ptr,
    input_token_counts_ptr,
    output_tokens_ptr,
    output_token_counts_ptr,
    COMBINE: tl.constexpr,
    BALANCED,
    T_BUCKET,
    EG: tl.constexpr,
    EP: tl.constexpr,
    E: tl.constexpr,
    D: tl.constexpr,
    EG_START,
    SPLIT_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """
    tokens: [T, D]
    input_token_counts: [EP, E]
    output_tokens: [T, D]
    output_token_counts: [E]
    """
    tidx = tl.program_id(0)

    NUM_D_BLOCKS: tl.constexpr = (D + SPLIT_D * BLOCK_D - 1) // (SPLIT_D * BLOCK_D)

    rank = tidx // (EG * SPLIT_D)
    local_expert = (tidx % (EG * SPLIT_D)) // SPLIT_D
    didx = tidx % SPLIT_D

    global_expert = local_expert + EG_START

    input_token_counts = tl.load(
        input_token_counts_ptr
        + tl.arange(0, EP)[:, None] * E
        + tl.arange(0, E)[None, :],
        eviction_policy="evict_last",
    )  # [EP, E]

    input_token_counts_eg = tl.load(
        input_token_counts_ptr
        + tl.arange(0, EP)[:, None] * E
        + EG_START
        + tl.arange(0, EG)[None, :],
        eviction_policy="evict_last",
    )  # [EP, EG]

    if COMBINE:
        LAST_TILE: tl.constexpr = EP * EG * SPLIT_D

        if tidx == LAST_TILE:
            if EG == E:
                output_token_counts = tl.sum(input_token_counts, axis=0)
                tl.store(
                    output_token_counts_ptr + tl.arange(0, E)[:], output_token_counts
                )
                output_token_counts = tl.sum(output_token_counts)
                tl.store(output_token_counts_ptr + E, output_token_counts)
            else:
                output_token_counts_eg = tl.sum(input_token_counts_eg, axis=0)
                tl.store(
                    output_token_counts_ptr + tl.arange(0, EG)[:],
                    output_token_counts_eg,
                )
                output_token_counts_eg = tl.sum(output_token_counts_eg)
                tl.store(output_token_counts_ptr + EG, output_token_counts_eg)
            return

    cond0 = tl.arange(0, EP)[:, None] < rank
    cond1 = tl.arange(0, EP)[:, None] == rank

    cond2 = tl.arange(0, E)[None, :] < global_expert
    cond3 = tl.arange(0, E)[None, :] == global_expert

    # r < rank || (r == rank && e < expert)
    ep_first_order = tl.sum(tl.where(cond0 or (cond1 and cond2), input_token_counts, 0))
    if EG == E:
        # e < expert || (e == expert && r < rank)
        expert_first_order = tl.sum(
            tl.where(cond2 or (cond3 and cond0), input_token_counts, 0)
        )
    else:
        # e < expert || (e == expert && r < rank)
        cond4 = tl.arange(0, EG)[None, :] < local_expert
        cond5 = tl.arange(0, EG)[None, :] == local_expert

        expert_first_order = tl.sum(
            tl.where(cond4 or (cond5 and cond0), input_token_counts_eg, 0)
        )

    if COMBINE:
        input_offset = ep_first_order
        output_offset = expert_first_order
    else:
        input_offset = expert_first_order
        output_offset = ep_first_order

    input_offset = input_offset.to(tl.int64)
    output_offset = output_offset.to(tl.int64)

    num_copy_tokens = tl.load(input_token_counts_ptr + rank * E + global_expert)
    if num_copy_tokens == 0:
        return

    STEP_D: tl.constexpr = SPLIT_D * BLOCK_D
    MASK_D: tl.constexpr = D % STEP_D != 0

    num_t_blocks = tl.cdiv(num_copy_tokens, BLOCK_T)

    t_1d_ptr = tl.arange(0, BLOCK_T)[:, None]
    ti_1d_ptr = input_offset + t_1d_ptr
    to_1d_ptr = output_offset + t_1d_ptr

    d_1d_ptr = didx * NUM_D_BLOCKS * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]

    i_2d_ptr = input_tokens_ptr + ti_1d_ptr * D + d_1d_ptr
    o_2d_ptr = output_tokens_ptr + to_1d_ptr * D + d_1d_ptr

    for i in range(num_t_blocks * NUM_D_BLOCKS):
        mask = t_1d_ptr < num_copy_tokens
        if MASK_D:
            mask &= d_1d_ptr < D

        block = tl.load(
            i_2d_ptr,
            mask=mask,
        )
        tl.store(
            o_2d_ptr,
            value=block,
            mask=mask,
        )

        if i % NUM_D_BLOCKS == (NUM_D_BLOCKS - 1):  # pyre-ignore
            # just to make sure constant folding happens
            D_1D_SHIFT: tl.constexpr = -(NUM_D_BLOCKS - 1) * BLOCK_D
            TD_2D_SHIFT: tl.constexpr = BLOCK_T * D + D_1D_SHIFT
            # increment T, D
            t_1d_ptr += BLOCK_T
            i_2d_ptr += TD_2D_SHIFT
            o_2d_ptr += TD_2D_SHIFT
            if MASK_D:
                d_1d_ptr += D_1D_SHIFT
        else:
            # increment D
            i_2d_ptr += BLOCK_D
            o_2d_ptr += BLOCK_D
            if MASK_D:
                d_1d_ptr += BLOCK_D
