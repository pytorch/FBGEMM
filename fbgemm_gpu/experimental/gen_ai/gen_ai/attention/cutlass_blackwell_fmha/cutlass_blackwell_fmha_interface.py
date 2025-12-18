# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch

try:
    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu import open_source
except Exception:
    open_source: bool = False

if open_source:
    import os

    torch.ops.load_library(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "..",
            "fbgemm_gpu_experimental_gen_ai.so",
        )
    )
else:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:blackwell_attention_ops_gpu"
    )


from enum import IntEnum


class GenKernelType(IntEnum):
    UMMA_I = 0
    UMMA_P = 1


def get_splitk_heuristic(
    batch: int,
    seqlen_kv: int,
    kv_heads: int = 1,
    tile_n: int = 256,
    sm_count: int | None = None,
) -> int:
    """
    Compute optimal split-K size for Shape<64, 256, 128> tile configuration.

    Targets full GPU utilization by distributing work across all SMs.
    First calculates SMs per batch, then per kv_head, then divides seqlen_kv by that number.
    Ensures split size evenly divides seqlen_kv so all CTAs process same number of tiles.
    Returns 0 (no split) when split would equal seqlen_kv (only 1 split).

    Args:
        batch: Batch size
        seqlen_kv: Maximum sequence length for K/V
        kv_heads: Number of KV heads (default 1 for MQA)
        tile_n: TileN dimension (default 256 for Shape<64, 256, 128>)
        sm_count: Number of SMs on the GPU. If None, queries the current device.

    Returns:
        Optimal split size along the K/V sequence dimension, or 0 to disable split-K
    """
    # Get SM count from current device if not provided
    if sm_count is None:
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count

    # Calculate number of SMs available per batch element
    sms_per_batch = max(1, sm_count // batch)
    # Further divide by kv_heads for multi-head KV
    sms_per_head_batch = max(1, sms_per_batch // kv_heads)

    # Each (batch, kv_head) element should have sms_per_head_batch splits
    # So split size = seqlen_kv / sms_per_head_batch
    ideal_split = seqlen_kv // sms_per_head_batch

    # Round up to multiple of tile_n
    split = ((ideal_split + tile_n - 1) // tile_n) * tile_n

    # Clamp to valid range: [tile_n, seqlen_kv]
    split = max(split, tile_n)
    split = min(split, seqlen_kv)

    # If split equals seqlen_kv, there's only 1 split - disable split-K
    if split == seqlen_kv:
        split = 0

    return split


def maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    """
    We only require the head dim to be contiguous
    """
    return (
        x.contiguous()
        if x is not None and (x.stride(-1) != 1 or x.stride(-2) % 8 != 0)
        else x
    )


def _cutlass_blackwell_fmha_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seq_len_q: int | None = None,
    max_seq_len_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    seqlen_kv: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    seqlen_k: int | None = None,
    window_left: int = -1,
    window_right: int = -1,
    bottom_right: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = maybe_contiguous(q)
    k = maybe_contiguous(k)
    v = maybe_contiguous(v)
    return torch.ops.fbgemm.fmha_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        softmax_scale=softmax_scale,
        causal=causal,
        seqlen_kv=seqlen_kv,
        page_table=page_table,
        seqlen_k=seqlen_k,
        window_size_left=window_left,
        window_size_right=window_right,
        bottom_right=bottom_right,
    )


def _cutlass_blackwell_fmha_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seq_len_q: int | None = None,
    max_seq_len_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_left: int = -1,
    window_right: int = -1,
    bottom_right: bool = True,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    deterministic = deterministic or torch.are_deterministic_algorithms_enabled()
    dout = maybe_contiguous(dout)
    q = maybe_contiguous(q)
    k = maybe_contiguous(k)
    v = maybe_contiguous(v)
    out = maybe_contiguous(out)
    return torch.ops.fbgemm.fmha_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_left,
        window_size_right=window_right,
        bottom_right=bottom_right,
        deterministic=deterministic,
    )


def _validate_and_adjust_split_k_size(split_k_size: int) -> int:
    """
    Validate and adjust split_k_size parameter for optimal performance.

    Args:
        split_k_size: The requested split size along the K/V sequence dimension.

    Returns:
        Adjusted split_k_size that is valid for the kernel.

    Valid values:
        - split_k_size <= 0: Disable split-K (no splitting)
        - split_k_size > 0: Enable split-K with specified split size
    """
    if not isinstance(split_k_size, int):
        raise TypeError(
            f"split_k_size must be an integer, got {type(split_k_size).__name__}"
        )

    # If split-K is disabled, return as-is
    if split_k_size <= 0:
        return split_k_size

    # Constants
    MIN_RECOMMENDED_SPLIT_SIZE = 256
    TILE_SIZE = 128

    # Adjust if split_k_size is too small
    if split_k_size < MIN_RECOMMENDED_SPLIT_SIZE:
        split_k_size = MIN_RECOMMENDED_SPLIT_SIZE

    # Check if split_k_size is a power of 2
    is_power_of_2 = (split_k_size & (split_k_size - 1)) == 0

    # If not a power of 2, round to nearest multiple of tile size (128)
    if not is_power_of_2:
        split_k_size = ((split_k_size + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE

    return split_k_size


def _validate_decode_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlen_kv: torch.Tensor | None,
) -> None:
    assert seqlen_kv is not None, "seqlen_kv must be provided for decode"
    tensors = {"q": q, "k": k, "v": v, "seqlen_kv": seqlen_kv}

    for name, tensor in tensors.items():
        # assert tensor.is_contiguous(), f"{name} is not contiguous"
        assert tensor.is_cuda, f"{name} must be on GPU"


def _prepare_decode_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool, tuple[int, ...]]:
    """
    Prepare inputs for decode kernel by handling both varlen and batch formats.

    Returns:
        - Reshaped q, k, v tensors in batch format [B, 1, H, D]
        - batch_size
        - needs_reshape_output flag
        - original_shape of q
    """
    original_shape = tuple(q.shape)
    needs_reshape_output = False
    batch_size = q.shape[0]

    if q.dim() == 3:
        # Varlen format: [total_queries, num_heads, head_dim]
        q = q.view(batch_size, 1, q.shape[1], q.shape[2])
        needs_reshape_output = True

    if q.dim() != 4:
        raise ValueError(
            f"Invalid query shape: {q.shape}. Expected [B, 1, H, D] or [total_queries, H, D]"
        )
    assert q.shape[1] == 1, "Kernel  have sq=1"

    k = k.view(batch_size, -1, k.shape[1], k.shape[2]) if k.dim() == 3 else k
    v = v.view(batch_size, -1, v.shape[1], v.shape[2]) if v.dim() == 3 else v

    return q, k, v, batch_size, needs_reshape_output, original_shape


def cutlass_blackwell_fmha_decode_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlen_kv: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seq_len_q: int | None = None,
    max_seq_len_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_left: int = -1,
    window_right: int = -1,
    bottom_right: bool = True,
    split_k_size: int = 0,
    use_heuristic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode-optimized forward pass using the gen kernel.
    This is a wrapper to use the gen kernel which is optimized
    for decode (query length = 1).

    This function is called externally by xformers ops.

    Accepts inputs in two formats:
    - Varlen format: [total_queries, num_heads, head_dim] (3D)
    - Batch format: [batch_size, 1, num_heads, head_dim] (4D)

    Args:
        q: Query tensor in varlen [B, H, D] or batch [B, 1, H, D] format
        k: Key tensor [B, Sk, H_kv, D]
        v: Value tensor [B, Sk, H_kv, D]
        seqlen_kv: Per-batch sequence lengths [B] (required)
        split_k_size: Size of each split along the K/V sequence dimension.
                     - split_k_size <= 0 with use_heuristic=True: auto-compute using heuristic
                     - split_k_size <= 0 with use_heuristic=False: disable split-K
                     - split_k_size > 0: use the provided split size directly
                     Values below 256 are adjusted to 256. Non-power-of-2 values
                     are rounded to the nearest multiple of 128.
        use_heuristic: If True and split_k_size <= 0, automatically compute optimal
                      split size using the heuristic. Default is True.

    Returns:
        Kernel output with Q dimension added:
        - out: [B, 1, H, num_splits, D] (num_splits=1 when split-K disabled)
        - lse: [B, num_splits, H, 1]
    """
    _validate_decode_inputs(q, k, v, seqlen_kv)

    # Prepare inputs and handle format conversion
    q, k, v, batch_size, _, original_shape = _prepare_decode_inputs(q, k, v)

    # Determine effective split_k_size
    if split_k_size <= 0 and use_heuristic:
        # Auto-compute using heuristic
        max_seqlen_kv = k.shape[1]
        kv_heads = k.shape[2]  # K shape is [B, Sk, H_kv, D]
        split_k_size = get_splitk_heuristic(batch_size, max_seqlen_kv, kv_heads)

    # Validate and adjust split_k_size
    split_k_size = _validate_and_adjust_split_k_size(split_k_size)

    # Validate window_right: decode kernel only supports causal attention (window_right <= 0)
    if window_right > 0:
        raise ValueError(
            f"window_right={window_right} is not supported for decode attention. "
            "The decode kernel only supports causal attention with window_right <= 0. "
            "Use window_right=0 (causal, current position only)."
        )

    # Call the gen kernel (optimized for decode)
    # Note: window_left specifies how many tokens to look back (exclusive)
    # The kernel will attend to positions [seqlen_kv - window_left, seqlen_kv)
    out, lse = torch.ops.fbgemm.fmha_gen_fwd(
        q,
        k,
        v,
        seqlen_kv,
        None,
        kernel_type=GenKernelType.UMMA_I,
        window_left=window_left,
        window_right=0,
        split_k_size=split_k_size,
    )

    # Kernel returns: out [B, H, num_splits, D], lse [B, num_splits, H]
    # Reshape to consistent format with Q dimension:
    # out: [B, H, num_splits, D] -> [B, 1, H, num_splits, D]
    # lse: [B, num_splits, H] -> [B, num_splits, H, 1]
    out = out.unsqueeze(1)  # [B, 1, H, num_splits, D]
    lse = lse.unsqueeze(-1)  # [B, num_splits, H, 1]
    return out, lse


class CutlassBlackwellFmhaFunc(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seq_len_q: Optional[int] = None,
        max_seq_len_k: Optional[int] = None,
        seqlen_kv: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        seqlen_k: Optional[int] = None,
        window_size: tuple[int, int] = (-1, -1),
        bottom_right: bool = True,
        deterministic: bool = False,
    ) -> torch.Tensor:
        window_left, window_right = window_size
        # Check if this is generation phase (sq = 1)
        sq = q.shape[1]
        if q.dim() == 4 and sq == 1:
            # For gen case, we don't need to save tensors for backward
            ctx.is_gen = True
            out, _ = cutlass_blackwell_fmha_decode_forward(
                q,
                k,
                v,
                seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seq_len_q,
                max_seq_len_k,
                softmax_scale,
                causal,
                window_left,
                window_right,
                bottom_right,
            )
            return out

        ctx.is_gen = False
        # Only check dtype if cu_seqlens_q and cu_seqlens_k are provided
        if cu_seqlens_q is not None and cu_seqlens_k is not None:
            assert (
                cu_seqlens_q.dtype == torch.int32
                and cu_seqlens_q.dtype == cu_seqlens_k.dtype
            ), "cu_seqlens_q and cu_seqlens_k must be int32"

        # handle window_size
        if causal and window_left >= 0:
            window_right = 0
        # Use regular FMHA for non-generation case
        out, softmax_lse = _cutlass_blackwell_fmha_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seq_len_q,
            max_seq_len_k,
            softmax_scale,
            causal,
            seqlen_kv,
            page_table,
            seqlen_k,
            window_left,
            window_right,
            bottom_right,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.max_seq_len_q = max_seq_len_q
        ctx.max_seq_len_k = max_seq_len_k
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.bottom_right = bottom_right
        ctx.deterministic = deterministic
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor, *args: Any) -> tuple[  # type: ignore
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        if ctx.is_gen:
            # For gen case, no backward pass is needed (generation is inference only)
            raise RuntimeError(
                "Backward pass is not supported for generation phase (sq=1)"
            )

        q, k, v, out, softmax_lse = ctx.saved_tensors
        window_left, window_right = ctx.window_size
        dq, dk, dv = _cutlass_blackwell_fmha_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seq_len_q,
            ctx.max_seq_len_k,
            ctx.softmax_scale,
            ctx.causal,
            window_left,
            window_right,
            bottom_right=ctx.bottom_right,
            deterministic=ctx.deterministic,
        )
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def cutlass_blackwell_fmha_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seq_len_q: int | None = None,
    max_seq_len_k: int | None = None,
    seqlen_kv: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    seqlen_k: int | None = None,
    window_size: tuple[int, int] | None = (-1, -1),
    bottom_right: bool = True,
    deterministic: bool = False,
):
    return CutlassBlackwellFmhaFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seq_len_q,
        max_seq_len_k,
        seqlen_kv,
        page_table,
        seqlen_k,
        window_size,
        bottom_right,
        deterministic,
    )
