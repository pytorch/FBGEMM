# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import torch
import triton
import triton.language as tl


@triton.jit
def _fbgemm_gather_scale_dense_tokens(
    out,
    x,
    token_indices,
    expert_indices,
    scores,
    stride_t,
    stride_e,
    D: tl.constexpr,
    BLOCK_D_OUTER: tl.constexpr,
    BLOCK_D_INNER: tl.constexpr,
):
    output_token_index = tl.program_id(0)
    feature_offset = tl.program_id(1) * BLOCK_D_OUTER

    input_token_index = tl.load(
        token_indices + output_token_index, None, eviction_policy="evict_last"
    )
    input_expert_index = tl.load(
        expert_indices + output_token_index, None, eviction_policy="evict_last"
    )

    input_score = tl.load(
        scores + input_token_index * stride_t + input_expert_index * stride_e,
        None,
        eviction_policy="evict_last",
    ).to(tl.float32)

    for _ in range(0, BLOCK_D_OUTER // BLOCK_D_INNER):
        input_token_value = tl.load(
            x + input_token_index * D + feature_offset + tl.arange(0, BLOCK_D_INNER)[:],
            None,
        ).to(tl.float32)
        output_token_value = input_token_value * input_score

        tl.store(
            out
            + output_token_index * D
            + feature_offset
            + tl.arange(0, BLOCK_D_INNER)[:],
            output_token_value,
            None,
        )
        feature_offset += BLOCK_D_INNER


def gather_scale_dense_tokens(
    x: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    scores: torch.Tensor,
):
    T, D = x.shape
    E = scores.shape[1]
    # a = K * T
    a = token_indices.shape[0]

    assert x.is_contiguous()
    assert token_indices.is_contiguous()
    assert expert_indices.is_contiguous()

    assert tuple(token_indices.shape) == (a,)
    assert tuple(expert_indices.shape) == (a,)
    assert tuple(scores.shape) == (T, E)

    stride_t = scores.stride(0)
    stride_e = scores.stride(1)

    out = torch.empty((a, D), device="cuda", dtype=torch.bfloat16)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    if a >= NUM_SMS:
        BLOCK_D_OUTER = D
        BLOCK_D_INNER = 1024
        assert D % BLOCK_D_INNER == 0
    else:
        BLOCK_D_OUTER = 512
        BLOCK_D_INNER = 256
        assert D % BLOCK_D_OUTER == 0
    grid = (a, D // BLOCK_D_OUTER)
    _fbgemm_gather_scale_dense_tokens[grid](
        out,
        x,
        token_indices,
        expert_indices,
        scores,
        stride_t,
        stride_e,
        D,  # pyre-ignore
        BLOCK_D_OUTER,  # pyre-ignore
        BLOCK_D_INNER,  # pyre-ignore
    )
    return out


GATHER_SCALE_DENSE_TOKENS = "fbgemm::gather_scale_dense_tokens"

torch.library.define(
    "fbgemm::gather_scale_dense_tokens",
    "(Tensor x, Tensor token_indices, Tensor expert_indices, Tensor scores) -> Tensor",
)


@torch.library.impl(GATHER_SCALE_DENSE_TOKENS, "Meta")
def gather_scale_dense_tokens_meta(
    x,
    token_indices,
    expert_indices,
    scores,
):
    D = x.shape[1]
    a = token_indices.shape[0]
    return x.new_empty((a, D))


@torch.library.impl(GATHER_SCALE_DENSE_TOKENS, "CUDA")
def gather_scale_dense_tokens_cuda(
    x,
    token_indices,
    expert_indices,
    scores,
):
    return gather_scale_dense_tokens(
        x,
        token_indices,
        expert_indices,
        scores,
    )


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
