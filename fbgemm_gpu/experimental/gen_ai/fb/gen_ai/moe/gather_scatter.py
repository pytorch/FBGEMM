# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import torch
import triton
import triton.language as tl


def scatter_add_padded_tokens(
    in_tokens: torch.Tensor,  # [EP, T, D]
    token_counts: torch.Tensor,  # [E]
    token_indices: torch.Tensor,  # [T]
    out_tokens: torch.Tensor,  # [T, D]
) -> None:
    assert torch.version.cuda >= "12.4", "Requires CUDA version 12.4 or later!"

    assert in_tokens.is_contiguous()
    assert token_counts.is_contiguous()
    assert token_indices.is_contiguous()
    assert out_tokens.is_contiguous()

    EP, T, D = in_tokens.shape
    E = token_counts.shape[0]
    assert tuple(token_indices.shape) == (T,)
    assert tuple(out_tokens.shape) == (T, D)

    def grid(META):
        return (
            E,
            META["SPLIT_T"],
        )

    _fbgemm_scatter_add_padded_tokens[grid](
        in_tokens,
        token_counts,
        token_indices,
        out_tokens,
        EP,
        E,
        T,
        D,
    )


_NV_CONFIGS = [
    triton.Config(
        {
            "SPLIT_T": split_t,
            "BLOCK_D": block_d,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    for split_t in [1, 4, 8, 16]
    for block_d in [512, 1024]
    for num_stages in [1, 3]
    for num_warps in [8, 16]
    for num_ctas in [1]
]

_AMD_CONFIGS = [
    triton.Config(
        {
            "SPLIT_T": split_t,
            "BLOCK_D": block_d,
            "waves_per_eu": waves_per_eu,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for split_t in [2, 8, 16, 32]
    for block_d in [512, 1024]
    for num_stages in [1, 3]
    for num_warps, waves_per_eu in [(8, 2), (16, 4)]
]


@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    restore_value=("out_tokens_ptr",),
    key=["EP", "E", "T", "D", "SPLIT_T"],
)
@triton.jit
def _fbgemm_scatter_add_padded_tokens(
    in_tokens_ptr,
    token_counts_ptr,
    token_indices_ptr,
    out_tokens_ptr,
    EP: tl.constexpr,
    E: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    SPLIT_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    in_tokens: [EP, T, D]
    token_counts: [E]
    out_tokens: [T, D]
    """
    expert = tl.program_id(0)
    t_tile = tl.program_id(1)

    tl.static_assert(D % BLOCK_D == 0)
    NUM_D_BLOCKS: tl.constexpr = D // BLOCK_D

    num_tokens = tl.load(token_counts_ptr + expert)
    if num_tokens == 0:
        return

    num_tokens_per_cta = tl.cdiv(num_tokens, SPLIT_T)
    start_token = t_tile * num_tokens_per_cta
    end_token = min(start_token + num_tokens_per_cta, num_tokens)

    tl.static_assert(E % EP == 0)
    EXPERT_PER_RANK: tl.constexpr = E // EP
    rank = expert // EXPERT_PER_RANK

    token_counts = tl.load(token_counts_ptr + tl.arange(0, E))
    input_local_offset = (
        tl.sum(tl.where(tl.arange(0, E) < expert, token_counts, 0)) + start_token
    )

    for _t in range(start_token, end_token):
        output_local_offset = tl.load(token_indices_ptr + input_local_offset)
        output_global_offset = output_local_offset * D

        d_ptr = tl.arange(0, BLOCK_D)
        input_global_ptr = in_tokens_ptr + rank * T * D + input_local_offset * D + d_ptr
        output_global_ptr = out_tokens_ptr + output_global_offset + d_ptr

        for _d in range(NUM_D_BLOCKS):
            vec = tl.load(input_global_ptr)
            tl.atomic_add(output_global_ptr, vec, sem="relaxed")
            input_global_ptr += BLOCK_D
            output_global_ptr += BLOCK_D

        input_local_offset += 1
