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


def _create_decode_lse(
    out: torch.Tensor,
    batch_size: int,
    needs_reshape_output: bool,
    q_shape: tuple[int, ...],
) -> torch.Tensor:
    """
    Create dummy LSE tensor for decode output compatibility.
    Gen kernel doesn't return LSE, so we create a zero tensor.
    """
    if needs_reshape_output:
        # For varlen output format
        lse_shape = [batch_size, q_shape[-1]]  # [B, H]
    else:
        # For batch output format
        lse_shape = [batch_size, q_shape[-2], q_shape[1]]  # [B, H, 1]

    return torch.zeros(*lse_shape, dtype=torch.float32, device=out.device)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode-optimized forward pass using the gen kernel.
    This is a wrapper to use the gen kernel which is optimized
    for decode (query length = 1).

    This function is called externally by xformers ops.

    Accepts inputs in two formats:
    - Varlen format: [total_queries, num_heads, head_dim] (3D)
    - Batch format: [batch_size, 1, num_heads, head_dim] (4D)
    """
    _validate_decode_inputs(q, k, v, seqlen_kv)

    # Prepare inputs and handle format conversion
    q, k, v, batch_size, needs_reshape_output, original_shape = _prepare_decode_inputs(
        q, k, v
    )
    # Call the gen kernel (optimized for decode)
    out = torch.ops.fbgemm.fmha_gen_fwd(
        q,
        k,
        v,
        seqlen_kv,
        None,
        kernel_type=GenKernelType.UMMA_I,
        # window_left=window_left,
        # window_right=window_right,
    )

    # Reshape output back to original format if needed
    if needs_reshape_output:
        out = out.view(*original_shape)

    # Create dummy LSE for compatibility
    lse = _create_decode_lse(out, batch_size, needs_reshape_output, original_shape)

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
