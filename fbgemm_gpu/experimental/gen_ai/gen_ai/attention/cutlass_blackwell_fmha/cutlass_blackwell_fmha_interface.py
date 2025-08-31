# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple

import torch

try:
    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu import open_source

    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu.docs.version import __version__  # noqa: F401
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
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


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
    causal: bool = False,
    window_left: int = -1,
    window_right: int = -1,
    bottom_right: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = maybe_contiguous(q)
    k = maybe_contiguous(k)
    v = maybe_contiguous(v)
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
        causal=causal,
        window_size_left=window_left,
        window_size_right=window_right,
        bottom_right=bottom_right,
    )


def _cutlass_blackwell_fmha_gen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlen_kv: torch.Tensor,
    batch_idx: torch.Tensor,
    kernel_type: GenKernelType = GenKernelType.UMMA_I,
) -> torch.Tensor:
    assert q.is_contiguous(), "q is not contiguous"
    assert k.is_contiguous(), "k is not contiguous"
    assert v.is_contiguous(), "v is not contiguous"
    assert seqlen_kv.is_contiguous(), "seqlen_kv is not contiguous"
    assert batch_idx.is_contiguous(), "batch_idx is not contiguous"
    assert q.is_cuda, "q must be on GPU"
    assert k.is_cuda, "k must be on GPU"
    assert v.is_cuda, "v must be on GPU"
    assert seqlen_kv.is_cuda, "seqlen_kv must be on GPU"
    assert batch_idx.is_cuda, "batch_idx must be on GPU"
    return torch.ops.fbgemm.fmha_gen_fwd(
        q,
        k,
        v,
        seqlen_kv,
        batch_idx,
        kernel_type,
    )


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
        window_size: Tuple[int, int] = (-1, -1),
        bottom_right: bool = True,
    ) -> torch.Tensor:
        # Check if this is generation phase (sq = 1)
        sq = q.shape[1]
        # Only check dtype if cu_seqlens_q and cu_seqlens_k are provided
        if cu_seqlens_q is not None and cu_seqlens_k is not None:
            assert (
                cu_seqlens_q.dtype == torch.int32
                and cu_seqlens_q.dtype == cu_seqlens_k.dtype
            ), "cu_seqlens_q and cu_seqlens_k must be int32"

        # handle window_size
        window_left, window_right = window_size
        if causal and window_left >= 0:
            window_right = 0

        if q.dim() == 4 and sq == 1:
            batch_size = q.shape[0]

            # Use provided seqlen_kv
            assert (
                seqlen_kv is not None
            ), "seqlen_kv must be provided for generation phase"

            # Create batch_idx tensor
            batch_idx = torch.arange(batch_size, dtype=torch.int32, device=q.device)

            # Use gen forward (no backward needed for generation)
            out = _cutlass_blackwell_fmha_gen(
                q, k, v, seqlen_kv, batch_idx, kernel_type=GenKernelType.UMMA_I
            )
            # For gen case, we don't need to save tensors for backward
            ctx.is_gen = True
            return out
        else:
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
            ctx.is_gen = False
            ctx.bottom_right = bottom_right
            return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor, *args: Any) -> Tuple[  # type: ignore
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
            ctx.causal,
            window_left,
            window_right,
            bottom_right=ctx.bottom_right,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


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
    window_size: tuple[int, int] | None = (-1, -1),
    bottom_right: bool = True,
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
        window_size,
        bottom_right,
    )
