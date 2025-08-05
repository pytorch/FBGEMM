# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import get_fp8_constants


# Function APIs
def gather_scale_dense_tokens(
    x: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    scores: torch.Tensor,
    valid_token_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gather and scale dense tokens along 1D indices.

    For each input token, token_indices[i] is the index of the token in the input sequence.
    expert_indices[i] is the index of the expert that the token is assigned to.
    scores[i] is the score of the token.

    For each expert, the tokens assigned to this expert are gathered from the input sequence,
    and then their scores are multiplied element-wise.

    valid_token_count is an optional tensor that can be used to filter out some tokens.
    If it is provided, the function will only consider the first valid_token_count tokens in the input sequence.

    The function returns a tensor of shape (a, D), where a is the number of tokens and D is the input dimension.

    Args:
        x (torch.Tensor): input tensor of shape (T, D)
        token_indices (torch.Tensor): token indices of shape (a,)
        expert_indices (torch.Tensor): expert indices of shape (a,)
        scores (torch.Tensor): scores of shape (T, E)
        valid_token_count (torch.Tensor, optional): valid token count of shape (,)

    Returns:
        torch.Tensor: output tensor of shape (a, D)
    """
    T, D = x.shape
    E = scores.shape[1]
    # a = K * T
    a = token_indices.shape[0]

    out = torch.empty((a, D), device="cuda", dtype=torch.bfloat16)
    if a == 0 or D == 0:
        return out

    assert x.is_contiguous()
    assert token_indices.is_contiguous()
    assert expert_indices.is_contiguous()

    assert tuple(token_indices.shape) == (a,)
    assert tuple(expert_indices.shape) == (a,)
    assert tuple(scores.shape) == (T, E)

    stride_t = scores.stride(0)
    stride_e = scores.stride(1)

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
        valid_token_count,
        D,  # pyre-ignore
        BLOCK_D_OUTER,  # pyre-ignore
        BLOCK_D_INNER,  # pyre-ignore
    )
    return out


def gather_scale_quant_dense_tokens(
    x: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    scores: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    valid_token_count: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather, scale, and quantize dense tokens along 1D indices.

    For each input token, token_indices[i] is the index of the token in the input sequence.
    expert_indices[i] is the index of the expert that the token is assigned to.
    scores[i] is the score of the token.

    For each expert, the tokens assigned to this expert are gathered from the input sequence,
    and then their scores are multiplied element-wise, and then quantized to FP8.

    valid_token_count is an optional tensor that can be used to filter out some tokens.
    If it is provided, the function will only consider the first valid_token_count tokens in the input sequence.

    The function returns a tensor of shape (a, D), where a is the number of tokens and D is the input dimension.

    Args:
        x (torch.Tensor): input tensor of shape (T, D)
        token_indices (torch.Tensor): token indices of shape (a,)
        expert_indices (torch.Tensor): expert indices of shape (a,)
        scores (torch.Tensor): scores of shape (T, E)
        scale_ub (torch.Tensor, optional): scale upper bound of shape (1,)
        valid_token_count (torch.Tensor, optional): valid token count of shape (1,)

    Returns:
        torch.Tensor: output tensor of shape (a, D)
    """
    T, D = x.shape
    E = scores.shape[1]
    # a = K * T
    a = token_indices.shape[0]

    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()

    assert x.is_contiguous()
    assert token_indices.is_contiguous()
    assert expert_indices.is_contiguous()

    assert tuple(token_indices.shape) == (a,)
    assert tuple(expert_indices.shape) == (a,)
    assert tuple(scores.shape) == (T, E)

    stride_t = scores.stride(0)
    stride_e = scores.stride(1)

    out = torch.empty((a, D), device="cuda", dtype=pt_dtype)
    out_scale = torch.empty((a,), device="cuda", dtype=torch.float32)

    grid = (a,)
    _fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens[grid](
        out,
        out_scale,
        x,
        token_indices,
        expert_indices,
        scores,
        scale_ub,
        stride_t,
        stride_e,
        valid_token_count,
        D,
        TL_FP8_DTYPE=tl_dtype,
        MAX_FP8=max_fp8,
        EPS=eps,
        CLAMP_MAX=scale_ub is not None,
    )
    return out, out_scale


def scatter_add_dense_tokens(
    out_tokens: torch.Tensor,  # [T, D]
    in_tokens: torch.Tensor,  # [a, D]
    token_indices: torch.Tensor,  # [a]
    valid_token_count: Optional[torch.Tensor] = None,
) -> None:
    """
    Scatter add dense tokens along 1D indices.

    Args:
        out_tokens (torch.Tensor): output tensor of shape (T, D)
        in_tokens (torch.Tensor): input tensor of shape (a, D)
        token_indices (torch.Tensor): token indices of shape (a,)
        valid_token_count (torch.Tensor, optional): valid token count of shape (1,)

    Returns:
        None
    """

    assert torch.version.hip is not None or (
        torch.version.cuda is not None and torch.version.cuda >= "12.4"
    ), "Requires CUDA version 12.4 or later on Nvidia GPUs!"

    assert in_tokens.is_contiguous()
    assert token_indices.is_contiguous()
    assert out_tokens.is_contiguous()

    a, D = in_tokens.shape
    if a == 0:
        return
    assert token_indices.shape == (a,)
    assert out_tokens.ndim == 2 and out_tokens.shape[1] == D

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    if a >= NUM_SMS:
        BLOCK_D_OUTER = D
        BLOCK_D_INNER = 1024
    else:
        BLOCK_D_OUTER = 512
        BLOCK_D_INNER = 256
    while D % BLOCK_D_OUTER != 0:
        BLOCK_D_OUTER //= 2
    while D % BLOCK_D_INNER != 0:
        BLOCK_D_INNER //= 2

    grid = (a, D // BLOCK_D_OUTER)
    _fbgemm_scatter_add_dense_tokens[grid](
        out_tokens,
        in_tokens,
        token_indices,
        valid_token_count,
        D,  # pyre-ignore
        BLOCK_D_OUTER,  # pyre-ignore
        BLOCK_D_INNER,  # pyre-ignore
    )


def scatter_add_padded_tokens(
    in_tokens: torch.Tensor,  # [EP, T_K, D]
    token_counts: torch.Tensor,  # [E]
    token_indices: torch.Tensor,  # [T_K]
    out_tokens: torch.Tensor,  # [T, D]
) -> None:
    """
    Scatter add valid tokens based on token counts metadata.

    Args:
        in_tokens (torch.Tensor): input tensor of shape (EP, T_K, D)
        token_counts (torch.Tensor): token counts of shape (E,)
        token_indices (torch.Tensor): token indices of shape (T_K,)
        out_tokens (torch.Tensor): output tensor of shape (T, D)

    Returns:
        None
    """
    assert torch.version.hip is not None or (
        torch.version.cuda is not None and torch.version.cuda >= "12.4"
    ), "Requires CUDA version 12.4 or later on Nvidia GPUs!"

    assert in_tokens.is_contiguous()
    assert token_counts.is_contiguous()
    assert token_indices.is_contiguous()
    assert out_tokens.is_contiguous()

    EP, T_K, D = in_tokens.shape
    E = token_counts.shape[0]
    assert tuple(token_indices.shape) == (T_K,)
    assert T_K % out_tokens.shape[0] == 0 and out_tokens.shape[1] == D

    def grid(META):
        return (
            E,
            META["SPLIT_T"],
        )

    T_BUCKET_CAP = 16384
    T_BUCKET = min(triton.next_power_of_2(T_K), T_BUCKET_CAP)
    BLOCK_E = max(triton.next_power_of_2(E), 8)
    _fbgemm_scatter_add_padded_tokens[grid](
        in_tokens,
        token_counts,
        token_indices,
        out_tokens,
        EP,
        E,
        T_BUCKET,
        T_K,
        D,
        BLOCK_E,
    )


# Torch Custom Op Registrations
_GATHER_SCALE_DENSE_TOKENS_OP_NAME = "fbgemm::gather_scale_dense_tokens"

torch.library.define(
    "fbgemm::gather_scale_dense_tokens",
    "(Tensor x, Tensor token_indices, Tensor expert_indices, Tensor scores, Tensor? valid_token_count=None) -> Tensor",
)


@torch.library.impl(_GATHER_SCALE_DENSE_TOKENS_OP_NAME, "Meta")
def gather_scale_dense_tokens_meta(
    x,
    token_indices,
    expert_indices,
    scores,
    valid_token_count=None,
):
    D = x.shape[1]
    a = token_indices.shape[0]
    return x.new_empty((a, D))


@torch.library.impl(_GATHER_SCALE_DENSE_TOKENS_OP_NAME, "CUDA")
def gather_scale_dense_tokens_cuda(
    x,
    token_indices,
    expert_indices,
    scores,
    valid_token_count=None,
):
    return gather_scale_dense_tokens(
        x,
        token_indices,
        expert_indices,
        scores,
        valid_token_count,
    )


_GATHER_SCALE_QUANT_DENSE_TOKENS_OP_NAME = "fbgemm::gather_scale_quant_dense_tokens"

torch.library.define(
    "fbgemm::gather_scale_quant_dense_tokens",
    "(Tensor x, Tensor token_indices, Tensor expert_indices, Tensor scores, Tensor? scale_ub=None, Tensor? valid_token_count=None) -> Tensor",
)


@torch.library.impl(_GATHER_SCALE_QUANT_DENSE_TOKENS_OP_NAME, "Meta")
def gather_scale_quant_dense_tokens_meta(
    x,
    token_indices,
    expert_indices,
    scores,
    scale_ub=None,
    valid_token_count=None,
):
    D = x.shape[1]
    a = token_indices.shape[0]
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    return torch.empty((a, D), device=x.device, dtype=pt_dtype), torch.empty(
        (a,), device=x.device, dtype=torch.float32
    )


@torch.library.impl(_GATHER_SCALE_QUANT_DENSE_TOKENS_OP_NAME, "CUDA")
def gather_scale_quant_dense_tokens_cuda(
    x,
    token_indices,
    expert_indices,
    scores,
    scale_ub=None,
    valid_token_count=None,
):
    return gather_scale_quant_dense_tokens(
        x,
        token_indices,
        expert_indices,
        scores,
        scale_ub,
        valid_token_count,
    )


_SCATTER_ADD_DENSE_TOKENS_OP_NAME = "fbgemm::scatter_add_dense_tokens"

torch.library.define(
    "fbgemm::scatter_add_dense_tokens",
    "(Tensor out_tokens, Tensor in_tokens, Tensor token_indices, Tensor? valid_token_count=None) -> None",
)


@torch.library.impl(_SCATTER_ADD_DENSE_TOKENS_OP_NAME, "Meta")
def scatter_add_dense_tokens_meta(
    out_tokens,
    in_tokens,
    token_indices,
    valid_token_count=None,
):
    return None


@torch.library.impl(_SCATTER_ADD_DENSE_TOKENS_OP_NAME, "CUDA")
def scatter_add_dense_tokens_cuda(
    out_tokens,
    in_tokens,
    token_indices,
    valid_token_count=None,
):
    return scatter_add_dense_tokens(
        out_tokens, in_tokens, token_indices, valid_token_count
    )


_SCATTER_ADD_PADDED_TOKENS_OP_NAME = "fbgemm::scatter_add_padded_tokens"

torch.library.define(
    "fbgemm::scatter_add_padded_tokens",
    "(Tensor in_tokens, Tensor token_counts, Tensor token_indices, Tensor out_tokens) -> None",
)


@torch.library.impl(_SCATTER_ADD_PADDED_TOKENS_OP_NAME, "Meta")
def scatter_add_padded_tokens_meta(
    in_tokens,
    token_counts,
    token_indices,
    out_tokens,
):
    return None


@torch.library.impl(_SCATTER_ADD_PADDED_TOKENS_OP_NAME, "CUDA")
def scatter_add_padded_tokens_cuda(
    in_tokens,
    token_counts,
    token_indices,
    out_tokens,
):
    return scatter_add_padded_tokens(
        in_tokens,
        token_counts,
        token_indices,
        out_tokens,
    )


# Kernel Implementations
@triton.jit
def _fbgemm_gather_scale_dense_tokens(
    out,
    x,
    token_indices,
    expert_indices,
    scores,
    stride_t,
    stride_e,
    valid_token_count,
    D: tl.constexpr,
    BLOCK_D_OUTER: tl.constexpr,
    BLOCK_D_INNER: tl.constexpr,
):
    output_token_index = tl.program_id(0)
    feature_offset = tl.program_id(1) * BLOCK_D_OUTER

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if output_token_index >= valid_token_count:
            return

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
            x
            + input_token_index.to(tl.int64) * D
            + feature_offset
            + tl.arange(0, BLOCK_D_INNER)[:],
            None,
        ).to(tl.float32)
        output_token_value = input_token_value * input_score

        tl.store(
            out
            + output_token_index.to(tl.int64) * D
            + feature_offset
            + tl.arange(0, BLOCK_D_INNER)[:],
            output_token_value,
            None,
        )
        feature_offset += BLOCK_D_INNER


@triton.jit
def _fbgemm_scatter_add_dense_tokens(
    out_tokens,
    in_tokens,
    token_indices,
    valid_token_count,
    D: tl.constexpr,
    BLOCK_D_OUTER: tl.constexpr,
    BLOCK_D_INNER: tl.constexpr,
):
    input_token_index = tl.program_id(0).to(tl.int64)
    feature_offset = tl.program_id(1) * BLOCK_D_OUTER + tl.arange(0, BLOCK_D_INNER)[:]

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if input_token_index >= valid_token_count:
            return

    output_token_index = tl.load(
        token_indices + input_token_index, None, eviction_policy="evict_last"
    ).to(tl.int64)

    for _ in range(0, BLOCK_D_OUTER // BLOCK_D_INNER):
        input_token_value = tl.load(
            in_tokens + input_token_index * D + feature_offset,
            None,
            eviction_policy="evict_first",
        )

        tl.atomic_add(
            out_tokens + output_token_index * D + feature_offset,
            input_token_value,
            None,
            sem="relaxed",
        )
        feature_offset += BLOCK_D_INNER


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}),
        triton.Config({"BLOCK_D": 512}),
        triton.Config({"BLOCK_D": 1024}),
    ],
    key=["D"],
)
@triton.jit
def _fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens(
    output_ptr,
    output_scale_ptr,
    input_ptr,
    token_indices_ptr,
    expert_indices_ptr,
    scores_ptr,
    scale_ub_ptr,
    stride_t,
    stride_e,
    valid_token_count,
    D: tl.constexpr,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(D % BLOCK_D == 0, "D must be a multiple of BLOCK_D")

    output_token_index = tl.program_id(0)

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if output_token_index >= valid_token_count:
            return

    input_token_index = tl.load(
        token_indices_ptr + output_token_index, None, eviction_policy="evict_first"
    )
    input_expert_index = tl.load(
        expert_indices_ptr + output_token_index, None, eviction_policy="evict_first"
    )
    input_score = tl.load(
        scores_ptr + input_token_index * stride_t + input_expert_index * stride_e,
        None,
        eviction_policy="evict_first",
    ).to(tl.float32)

    row_max = 0.0
    in_2d_ptr = (
        input_ptr + input_token_index.to(tl.int64) * D + tl.arange(0, BLOCK_D)[:]
    )
    for _ in range(0, D, BLOCK_D):
        input_token_value = tl.load(
            in_2d_ptr,
            None,
            eviction_policy="evict_last",
        ).to(tl.float32)
        output_token_value = input_token_value * input_score

        tile_max = tl.max(tl.abs(output_token_value))
        row_max = tl.maximum(tile_max, row_max)
        in_2d_ptr += BLOCK_D

    # Clamp max value appropriately.
    if CLAMP_MAX:
        ub = tl.load(scale_ub_ptr, eviction_policy="evict_last")
        row_max = tl.clamp(row_max, EPS, ub)
    else:
        row_max = tl.maximum(row_max, EPS)

    # Scale and quantize.
    output_scale = MAX_FP8 / row_max
    tl.store(output_scale_ptr + output_token_index, 1.0 / output_scale)

    in_2d_ptr = (
        input_ptr + input_token_index.to(tl.int64) * D + tl.arange(0, BLOCK_D)[:]
    )
    out_2d_ptr = (
        output_ptr + output_token_index.to(tl.int64) * D + tl.arange(0, BLOCK_D)[:]
    )
    for _ in range(0, D, BLOCK_D):
        # Load from L2
        input_token_value = tl.load(
            in_2d_ptr,
            None,
            eviction_policy="evict_first",
        ).to(tl.float32)
        # Rematerilize
        output_token_value_fp8 = (input_token_value * input_score) * output_scale

        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        output_token_value_fp8 = tl.clamp(output_token_value_fp8, -MAX_FP8, MAX_FP8).to(
            TL_FP8_DTYPE
        )
        tl.store(
            out_2d_ptr,
            output_token_value_fp8,
            None,
            cache_modifier=".cg",
        )
        in_2d_ptr += BLOCK_D
        out_2d_ptr += BLOCK_D


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
    key=["EP", "E", "T_BUCKET", "D"],
)
@triton.jit
def _fbgemm_scatter_add_padded_tokens(
    in_tokens_ptr,
    token_counts_ptr,
    token_indices_ptr,
    out_tokens_ptr,
    EP: tl.constexpr,
    E: tl.constexpr,
    T_BUCKET,
    T_K,
    D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SPLIT_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    in_tokens: [EP, T_K, D]
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

    offs_e = tl.arange(0, BLOCK_E)
    token_counts = tl.load(token_counts_ptr + offs_e, mask=(offs_e < E), other=0)
    input_local_offset = (
        tl.sum(tl.where(offs_e < expert, token_counts, 0)) + start_token
    ).to(tl.int64)

    for _t in range(start_token, end_token):
        output_local_offset = tl.load(token_indices_ptr + input_local_offset).to(
            tl.int64
        )
        output_global_offset = output_local_offset * D

        d_ptr = tl.arange(0, BLOCK_D)
        input_global_ptr = (
            in_tokens_ptr + rank * T_K * D + input_local_offset * D + d_ptr
        )
        output_global_ptr = out_tokens_ptr + output_global_offset + d_ptr

        for _d in range(NUM_D_BLOCKS):
            vec = tl.load(input_global_ptr)
            tl.atomic_add(output_global_ptr, vec, sem="relaxed")
            input_global_ptr += BLOCK_D
            output_global_ptr += BLOCK_D

        input_local_offset += 1
