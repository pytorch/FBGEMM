# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch.library import register_fake


torch.library.define(
    "blackwell_fmha::fmha_fwd",
    "(Tensor q, Tensor k, Tensor v, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, int? max_seq_len_q, int? max_seq_len_k, float? softmax_scale, bool? causal, Tensor? seqlen_kv, Tensor? page_table, int seqlen_k=-1, int window_size_left=-1, int window_size_right=-1, bool bottom_right=True) -> (Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)

torch.library.define(
    "blackwell_fmha::fmha_bwd",
    "(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor softmax_lse, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, int? max_seq_len_q, int? max_seq_len_k, float? softmax_scale, bool? causal, int window_size_left=-1, int window_size_right=-1, bool bottom_right=True, bool deterministic=False) -> (Tensor, Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.impl("blackwell_fmha::fmha_fwd", "cuda")
def custom_op_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    max_seq_len_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    seqlen_kv: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    seqlen_k: Optional[int] = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
    bottom_right: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_contiguous(), "q is not contiguous"
    assert k.is_contiguous(), "k is not contiguous"
    assert v.is_contiguous(), "v is not contiguous"
    assert q.is_cuda, "q must be on GPU"
    assert k.is_cuda, "k must be on GPU"
    assert v.is_cuda, "v must be on GPU"

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
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        bottom_right=bottom_right,
    )


@register_fake("blackwell_fmha::fmha_fwd")
def fmha_fwd_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    max_seq_len_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    seqlen_kv: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    seqlen_k: Optional[int] = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
    bottom_right: bool = True,
):
    if q.dtype == torch.float16:
        out_dtype = torch.float16
    elif q.dtype == torch.bfloat16:
        out_dtype = torch.bfloat16
    elif q.dtype == torch.float8_e4m3fn:
        # Output is BF16 when input is FP8
        out_dtype = torch.bfloat16
    else:
        raise RuntimeError(f"Unsupported dtype for q: {q.dtype}")

    kIsVarlen = max_seq_len_q is not None
    if kIsVarlen:
        assert cu_seqlens_q is not None
        SQ = q.shape[0]
        H_Q = q.shape[1]
        B = cu_seqlens_q.shape[0] - 1
    else:
        SQ = q.shape[1]
        H_Q = q.shape[2]
        B = q.shape[0]
    device = q.device
    options2 = {"dtype": torch.float32, "device": device}
    if kIsVarlen:
        assert max_seq_len_q is not None
        out = torch.empty_like(q, dtype=out_dtype)
        size = out.size()
        stride = out.stride()
        storage_offset = q.shape[-1] * max_seq_len_q * H_Q  # example scalar offset
        out1 = torch.as_strided(
            out, size=size, stride=stride, storage_offset=storage_offset
        )
    else:
        out1 = torch.empty_like(q, dtype=out_dtype)

    if kIsVarlen:
        out2 = torch.empty((1, H_Q, SQ), **options2)  # type: ignore
    else:
        out2 = torch.empty((B, H_Q, SQ), **options2)  # type: ignore
    return out1, out2


@torch.library.impl("blackwell_fmha::fmha_bwd", "cuda")
def custom_op_fmha_bwd(
    dOutput: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    max_seq_len_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    bottom_right: bool = True,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    return torch.ops.fbgemm.fmha_bwd(
        dOutput,
        query,
        key,
        value,
        output,
        softmax_lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        bottom_right=bottom_right,
        deterministic=deterministic,
    )


@register_fake("blackwell_fmha::fmha_bwd")
def fmha_bwd_meta(
    dOutput: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    max_seq_len_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    bottom_right: bool = True,
    deterministic: bool = False,
):
    return (
        torch.empty_like(query),
        torch.empty_like(key),
        torch.empty_like(value),
    )


def _backward(ctx, *grad):
    if ctx.is_gen:
        # For gen case, no backward pass is needed (generation is inference only)
        raise RuntimeError("Backward pass is not supported for generation phase (sq=1)")
    q, k, v, out, softmax_lse = ctx.saved_tensors
    if not grad[0].is_contiguous():
        grad0 = grad[0].contiguous()
    else:
        grad0 = grad[0]
    if not softmax_lse.is_contiguous:
        softmax_lse = softmax_lse.contiguous()
    if not out.is_contiguous:
        out = out.contiguous()
    if not q.is_contiguous:
        q = q.contiguous()
    if not k.is_contiguous:
        k = k.contiguous()

    if not softmax_lse.is_contiguous:
        softmax_lse = softmax_lse.contiguous()
    if not out.is_contiguous:
        out = out.contiguous()
    if not q.is_contiguous:
        q = q.contiguous()
    if not k.is_contiguous:
        k = k.contiguous()

    dq, dk, dv = torch.ops.blackwell_fmha.fmha_bwd(
        grad0,
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
        ctx.window_size_left,
        ctx.window_size_right,
        ctx.bottom_right,
        ctx.deterministic,
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


def _setup_context(ctx, inputs, output):
    (
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
        window_size_left,
        window_size_right,
        bottom_right,
    ) = inputs
    (out, softmax_lse) = output
    ctx.save_for_backward(q, k, v, out, softmax_lse)
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.max_seq_len_q = max_seq_len_q
    ctx.max_seq_len_k = max_seq_len_k
    ctx.cu_seqlens_q = cu_seqlens_q
    ctx.cu_seqlens_k = cu_seqlens_k
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right
    ctx.bottom_right = bottom_right
    ctx.deterministic = False  # Set default value
    ctx.is_gen = False


# This code adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "blackwell_fmha::fmha_fwd", _backward, setup_context=_setup_context
)


def cutlass_blackwell_fmha_custom_op(
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
    seqlen_k: int | None = -1,
    window_size_left: int | None = -1,
    window_size_right: int | None = -1,
    bottom_right: bool | None = True,
):
    return torch.ops.blackwell_fmha.fmha_fwd(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        softmax_scale=softmax_scale,
        causal=causal,
        seqlen_kv=seqlen_kv,
        page_table=page_table,
        seqlen_k=seqlen_k,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        bottom_right=bottom_right,
    )[0]
