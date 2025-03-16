# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

from typing import Tuple

import torch
import triton
import triton.language as tl

torch.ops.load_library(
    "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:index_shuffling_gpu"
)

index_shuffling = torch.ops.fbgemm.index_shuffling


def combine_shuffling(
    tokens: torch.Tensor, token_counts: torch.Tensor, balanced: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    # T is intentionally ignored in kernel interface to avoid recompilation
    assert tokens.is_contiguous()
    assert token_counts.is_contiguous()

    T, D = tokens.shape
    EP, E = token_counts.shape

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    SPLIT_D = max(NUM_SMS // (EP * E), 1)
    SPLIT_D = triton.next_power_of_2(SPLIT_D + 1) // 2
    grid = (EP * E * SPLIT_D + 1,)

    output_tokens = torch.empty_like(tokens)
    output_token_counts = torch.empty(
        E, dtype=token_counts.dtype, device=token_counts.device
    )
    T_BUCKET = triton.next_power_of_2(T)

    _fbgemm_combine_shuffling[grid](
        tokens,
        token_counts,
        output_tokens,
        output_token_counts,
        balanced,
        T_BUCKET,
        EP,
        E,
        D,
        SPLIT_D,
    )

    return output_tokens, output_token_counts


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
    key=["BALANCED", "T_BUCKET", "EP", "E", "D"],
)
@triton.heuristics(
    values={
        "NUM_D_BLOCKS": lambda args: triton.cdiv(
            args["D"], args["SPLIT_D"] * args["BLOCK_D"]
        )
    }
)
@triton.jit
def _fbgemm_combine_shuffling(
    input_tokens_ptr,
    input_token_counts_ptr,
    output_tokens_ptr,
    output_token_counts_ptr,
    BALANCED: tl.constexpr,
    T_BUCKET: tl.constexpr,
    EP: tl.constexpr,
    E: tl.constexpr,
    D: tl.constexpr,
    SPLIT_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
) -> None:
    """
    tokens: [T, D]
    input_token_counts: [EP, E]
    output_tokens: [T, D]
    output_token_counts: [E]
    """
    tidx = tl.program_id(0)

    rank = tidx // (E * SPLIT_D)
    expert = (tidx % (E * SPLIT_D)) // SPLIT_D
    didx = tidx % SPLIT_D

    input_token_counts = tl.load(
        input_token_counts_ptr
        + tl.arange(0, EP)[:, None] * E
        + tl.arange(0, E)[None, :],
        eviction_policy="evict_last",
    )  # [EP, E]

    LAST_TILE: tl.constexpr = EP * E * SPLIT_D
    if tidx == LAST_TILE:
        output_token_counts = tl.sum(input_token_counts, axis=0)
        tl.store(output_token_counts_ptr + tl.arange(0, E)[:], output_token_counts)
        return

    cond0 = tl.arange(0, EP)[:, None] < rank
    cond1 = tl.arange(0, EP)[:, None] == rank
    cond2 = tl.arange(0, E)[None, :] < expert
    cond3 = tl.arange(0, E)[None, :] == expert

    # r < rank || (r == rank && e < expert)
    input_offset = tl.sum(tl.where(cond0 or (cond1 and cond2), input_token_counts, 0))
    # e < expert || (e == expert && r < rank)
    output_offset = tl.sum(tl.where(cond2 or (cond3 and cond0), input_token_counts, 0))

    num_copy_tokens = tl.load(input_token_counts_ptr + rank * E + expert)
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

        if i % NUM_D_BLOCKS == NUM_D_BLOCKS - 1:
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
