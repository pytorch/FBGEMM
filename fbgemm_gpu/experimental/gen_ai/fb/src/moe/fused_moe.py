# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import functools
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

torch.ops.load_library(
    "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai/fb/src/moe:fused_moe_ops"
)


@triton.jit
# pyre-fixme[3]: Return type must be annotated.
def fused_moe_kernel(
    # Pointers to matrices
    # pyre-fixme[2]: Parameter must be annotated.
    a_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    b_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    c_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    a_scale_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    b_scale_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    topk_weights_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    sorted_token_ids_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    expert_ids_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    # pyre-fixme[2]: Parameter must be annotated.
    N,
    # pyre-fixme[2]: Parameter must be annotated.
    K,
    # pyre-fixme[2]: Parameter must be annotated.
    EM,
    # pyre-fixme[2]: Parameter must be annotated.
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    # pyre-fixme[2]: Parameter must be annotated.
    stride_am,
    # pyre-fixme[2]: Parameter must be annotated.
    stride_ak,
    # pyre-fixme[2]: Parameter must be annotated.
    stride_be,
    # pyre-fixme[2]: Parameter must be annotated.
    stride_bk,
    # pyre-fixme[2]: Parameter must be annotated.
    stride_bn,
    # pyre-fixme[2]: Parameter must be annotated.
    stride_cm,
    # pyre-fixme[2]: Parameter must be annotated.
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    a_scale = 0
    b_scale = 0
    if use_fp8:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        if use_fp8:
            accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    if use_fp8:
        accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    torch.ops.fbgemm.moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8: bool,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if not use_fp8:
        assert A_scale is None
        assert B_scale is None
    else:
        A, A_scale = scaled_fp8_quant(A, A_scale)
        assert B_scale is not None

    grid = lambda META: (  # noqa: E731
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    fused_moe_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # pyre-ignore
        top_k=top_k,  # pyre-ignore
        compute_type=compute_type,  # pyre-ignore
        use_fp8=use_fp8,  # pyre-ignore
        **config,
    )


def get_config_file_name(E: int, N: int, dtype: Optional[str]) -> str:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}.json"


@functools.lru_cache
# pyre-fixme[3]: Return type must be annotated.
def get_moe_configs(E: int, N: int, dtype: Optional[str]) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(E, N, dtype)

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info("Using configuration from %s for MoE layer.", config_file_path)
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
) -> Dict[str, int]:
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


# pyre-fixme[3]: Return type must be annotated.
def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    override_config: Optional[Dict[str, Any]] = None,
):
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        configs = get_moe_configs(E, N, dtype)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype)
    return config


# pyre-fixme[3]: Return type must be annotated.
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )
    torch.ops.fbgemm.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),
    )
    del token_expert_indicies

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


# This is used by the Deepseek-V2 model
# pyre-fixme[3]: Return type must be annotated.
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
):

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


# pyre-fixme[3]: Return type must be annotated.
def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
):
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    CHUNK_SIZE = 65536
    M = min(num_tokens, CHUNK_SIZE)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        "float8" if use_fp8 else None,
        override_config=override_config,
    )

    config = get_config_func(M)

    intermediate_cache1 = torch.empty(
        (M, topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = torch.empty(
        (M, topk_ids.shape[1], w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], E
        )

        invoke_fused_moe_kernel(
            curr_hidden_states,
            w1,
            intermediate_cache1,
            a1_scale,
            w1_scale,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk_ids.shape[1],
            config,
            compute_type=compute_type,
            use_fp8=use_fp8,
        )

        torch.ops.fbgemm.silu_and_mul(
            intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            a2_scale,
            w2_scale,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            compute_type=compute_type,
            use_fp8=use_fp8,
        )

        torch.sum(
            intermediate_cache3.view(*intermediate_cache3.shape),
            dim=1,
            out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )
    return out_hidden_states


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
        )
    else:
        topk_weights, topk_ids = fused_topk(
            hidden_states, gating_output, topk, renormalize
        )

    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=inplace,
        override_config=override_config,
        use_fp8=use_fp8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )


def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    shape: Union[Tuple[int, int], torch.Size] = input.shape
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    output = torch.empty(shape, device=input.device, dtype=torch.float8_e4m3fn)

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            torch.ops.fbgemm.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub
            )
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            torch.ops.fbgemm.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        # num_token_padding not implemented for this case
        assert scale.numel() == 1 or num_token_padding is None
        torch.ops.fbgemm.static_scaled_fp8_quant(output, input, scale)

    return output, scale
