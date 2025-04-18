# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Dict, Optional

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

try:
    from xformers.ops.fmha.attn_bias import (  # type: ignore
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
    )

    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False
    from typing import Any

    BlockDiagonalCausalWithOffsetPaddedKeysMask = Any

try:
    # @manual=//triton:triton
    # pyre-fixme[21]: Could not find module `triton.language.libdevice`.
    from triton.language.libdevice import pow
except ImportError:
    try:
        # @manual=//triton:triton
        # pyre-fixme[21]: Could not find name `pow` in `triton.language.math`.
        from triton.language.math import pow
    except ImportError:
        try:
            # @manual=//triton:triton
            from triton.language.extra.libdevice import pow
        except ImportError:
            # @manual=//triton:triton
            from triton.language.extra.cuda.libdevice import pow


_INTERNAL_DTYPE_MAP: Dict[str, int] = {"": 0, "f32": 1, "f64": 2}


@triton.jit
def _rope_padded_kernel(
    xq,
    xk,
    xv,
    out_q,
    cache_k,
    cache_v,
    seqstartq,
    seqstartk,
    seqlenk,
    theta,
    k_start: tl.constexpr,
    v_start: tl.constexpr,
    dim: tl.constexpr,  # dimension of each head
    stride_xqM,
    stride_xqH,
    stride_xkM,
    stride_xkH,
    stride_xvM,
    stride_xvH,
    stride_cachekM,
    stride_cachekH,
    stride_cachevM,
    stride_cachevH,
    stride_seqstartq,
    stride_seqstartk,
    stride_seqlenk,
    stride_outqM,
    stride_outqH,
    internal_dtype: tl.constexpr,
    # If True, seqstartq and seqstartk are not used but rather we
    # assume that every batch element has the same number of
    # queries (i.e. num_queries := tl.num_programs(1) )
    # and the same cache space cache_padding_length.
    # Always False when called below.
    const_batch_strides: tl.constexpr,
    # If const_batch_strides==True, the common cache length for each batch element.
    # (Only the first seqlenk[i] elements are actually in use, and only the last
    #  num_queries of those are actually written to.)
    cache_padding_length,
    # offset added to all values in seqlenk before using them.
    # Always 0 when called below.
    seqlenk_shift: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    adjacents: tl.constexpr,
):
    """
    Each letter in this diagram is a whole row of length dim.

     INPUT      xq        xk       xv

        head_dim ─►

      batch   qqqqqq      kk       vv
        │     qqqqqq      kk       vv
        ▼     qqqqqq      kk       vv

    head_idx:  (goes across all heads of all 3 inputs)
              ▲     ▲     ▲ ▲      ▲ ▲
              │     │     │ │      │ │
                          │        │
              0  k_start  │v_start │n_total_heads
                          │        │
                          │        │
                      k_start    v_start

    Output is to out_q (same shape as xq), an xk-shaped part
    of cache_k and an xv-shaped part of cache_v
    """
    batch_elt = tl.program_id(0)
    query_pos_in_batch_elt = tl.program_id(1)
    head_idx = tl.program_id(2)

    if internal_dtype == 1:
        theta = theta.to(tl.float32)
    elif internal_dtype == 2:
        theta = theta.to(tl.float64)

    if const_batch_strides:
        query_pos = query_pos_in_batch_elt + tl.num_programs(1) * batch_elt
        end_query_pos = tl.num_programs(1) * (batch_elt + 1)
    else:
        query_pos = query_pos_in_batch_elt + tl.load(
            seqstartq + batch_elt * stride_seqstartq
        )
        end_query_pos = tl.load(seqstartq + (batch_elt + 1) * stride_seqstartq)
        if query_pos >= end_query_pos:
            return

    is_q = head_idx < k_start
    is_v = head_idx >= v_start

    xq += query_pos * stride_xqM + head_idx * stride_xqH
    out_q += query_pos * stride_outqM + head_idx * stride_outqH

    if const_batch_strides:
        cache_start = cache_padding_length * batch_elt
    else:
        cache_start = tl.load(seqstartk + batch_elt * stride_seqstartk)
    end_of_batch_elt_cache = (
        cache_start + tl.load(seqlenk + batch_elt * stride_seqlenk) + seqlenk_shift
    )

    cache_pos = end_of_batch_elt_cache - (end_query_pos - query_pos)
    seq_pos = cache_pos - cache_start
    cache_k += (head_idx - k_start) * stride_cachekH + cache_pos * stride_cachekM
    xk += query_pos * stride_xkM + (head_idx - k_start) * stride_xkH
    in_qk = tl.where(is_q, xq, xk)
    out_qk = tl.where(is_q, out_q, cache_k)

    cache_v += (head_idx - v_start) * stride_cachevH + cache_pos * stride_cachevM
    xv += query_pos * stride_xvM + (head_idx - v_start) * stride_xvH

    out = tl.where(is_v, cache_v, out_qk)
    x_in = tl.where(is_v, xv, in_qk)

    for offset in range(0, dim // 2, BLOCK_SIZE // 2):
        c = tl.arange(0, BLOCK_SIZE // 2)
        powers = (offset + c) * 2.0
        if adjacents:
            cols_re = (offset + c) * 2
            cols_im = cols_re + 1
        else:
            cols_re = offset + c
            cols_im = cols_re + dim // 2

        mask = cols_im < dim

        re_x = tl.load(x_in + cols_re, mask=mask)
        im_x = tl.load(x_in + cols_im, mask=mask)
        # freqs = seq_pos / (theta ** (powers / dim))
        # pyre-fixme[16]: Module `language` has no attribute `libdevice`.
        freqs = seq_pos * pow(theta, powers / (-dim))
        sines = tl.sin(freqs)
        cosines = tl.cos(freqs)
        re_out = re_x * cosines - im_x * sines
        im_out = im_x * cosines + re_x * sines

        re_out_ = tl.where(is_v, re_x, re_out)
        im_out_ = tl.where(is_v, im_x, im_out)
        if internal_dtype == 2:
            if re_x.dtype == tl.bfloat16:
                # triton 2.0.0 crashes if you try to convert
                # float64 directly to bfloat16, so make an intermediate step.
                re_out_ = re_out_.to(tl.float32)
                im_out_ = im_out_.to(tl.float32)
        tl.store(out + cols_re, re_out_, mask=mask)
        tl.store(out + cols_im, im_out_, mask=mask)


def rope_padded_ref(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    attn_bias: BlockDiagonalCausalWithOffsetPaddedKeysMask,
    *,
    theta: float = 10000.0,
    out_q: Optional[torch.Tensor] = None,
    adjacents: bool = True,
    internal_dtype: str = "",
):
    # TODO: Support non-adjacent ROPE
    assert adjacents

    def rotary_embedding(tensor: torch.Tensor, varseq_seqpos: torch.Tensor):
        head_dim = tensor.shape[-1]
        powers = torch.arange(0, head_dim, 2, device=tensor.device, dtype=torch.float64)
        re_x, im_x = tensor.unflatten(-1, [-1, 2]).unbind(-1)
        freqs = (
            varseq_seqpos.outer(theta ** (powers / -head_dim)).unsqueeze(1).unsqueeze(0)
        )
        sines = freqs.sin()
        cosines = freqs.cos()
        re_out = re_x * cosines - im_x * sines
        im_out = im_x * cosines + re_x * sines
        tensor_out = torch.stack([re_out, im_out], -1).flatten(-2)
        return tensor_out.to(tensor.dtype)

    seqstartk: torch.Tensor = attn_bias.k_seqinfo.seqstart
    seqlenk: torch.Tensor = attn_bias.k_seqinfo.seqlen

    kv_cache_starts = seqstartk.tolist()
    q_lengths = [end - start for start, end in list(attn_bias.q_seqinfo.intervals())]
    kv_lengths = seqlenk.tolist()

    varseq_seqpos = torch.cat(
        [
            torch.arange(
                kv_length - q_length,
                kv_length,
                dtype=torch.int64,
                device=xq.device,
            )
            for kv_length, q_length in zip(kv_lengths, q_lengths)
        ]
    )

    xq_out = rotary_embedding(xq, varseq_seqpos)
    xk_out = rotary_embedding(xk, varseq_seqpos)

    cache_indices = torch.cat(
        [
            torch.arange(
                kv_start + kv_length - q_length,
                kv_start + kv_length,
                dtype=torch.int64,
                device=xq.device,
            )
            for kv_start, kv_length, q_length in zip(
                kv_cache_starts, kv_lengths, q_lengths
            )
        ]
    ).reshape([1, -1, 1, 1])

    cache_k.scatter_(1, cache_indices, xk_out)
    cache_v.scatter_(1, cache_indices, xv)

    if out_q is not None:
        out_q.copy_(xq_out)

    return xq_out


def rope_padded(
    xq,
    xk,
    xv,
    cache_k,
    cache_v,
    attn_bias: BlockDiagonalCausalWithOffsetPaddedKeysMask,
    *,
    theta: float = 10000.0,
    out_q: Optional[torch.Tensor] = None,
    adjacents: bool = True,
    internal_dtype: str = "",
):
    """
    Applies rope to a heterogeneous batch in the style given
    by xformers' BlockDiagonalCausalWithOffsetPaddedKeysMask.
    The batch is concatted along the sequence dimension, so the
    actual xformers batch size needs to be 1.

    xq, xk and xv should be (1, slen, n_heads, dim), where xq's n_heads can differ from xk and xv

    This function places the roped xk in the right place in cache_k and
    xv (unmodified) in the right place in cache_v, and returns out_q
    such that things are ready to call
    xformers.ops.memory_efficient_attention(out_q, cache_k, cache_v, attn_bias=attn_bias)

    WARNING: This function relies on private details of xformers.

    Arguments:
        xq: tensor of queries to apply rope to
        xk: tensor of keys to apply rope to
        xv: tensor of values to copy into cache_v
        cache_k: cache of keys, modified in place
        cache_v: cache of values, modified in place
        attn_bias: details the layout of caches.
                Used to determine frequencies for the
                RoPE calculation as well as the locations in cache_k and cache_v
                to write to. Must be on the device.
        adjacents: If True, the inputs are in adjacent pairs along the final dim axis.
                  This is like the released LLaMA model and xlformers.
                  If False, the dim axis is split in two equal pieces.
                   I.e. the features are ordered with all the real parts before all
                   the imaginary parts. This matches HuggingFace right now.
                   https://github.com/huggingface/transformers/blob/
                   f143037789288ba532dada934a118e648e715738/
                   src/transformers/models/llama/modeling_llama.py#L126-L130
        internal_dtype: set to "f32" or "f64" to enforce dtype in the calculation
    """

    # Slower path for devices that don't completely support the triton kernel
    if xq.device.type != "cuda":
        return rope_padded_ref(
            xq,
            xk,
            xv,
            cache_k,
            cache_v,
            attn_bias,
            theta=theta,
            out_q=out_q,
            adjacents=adjacents,
            internal_dtype=internal_dtype,
        )

    n_total_queries = attn_bias.q_seqinfo.seqstart_py[-1]
    cache_length = attn_bias.k_seqinfo.seqstart_py[-1]
    assert xq.shape[1] == n_total_queries
    bsz, _, n_q_heads, dim = xq.shape
    assert bsz == 1
    n_kv_heads = xk.shape[2]
    assert xk.shape == (1, n_total_queries, n_kv_heads, dim)
    assert xv.shape == (1, n_total_queries, n_kv_heads, dim)
    assert cache_k.shape == (1, cache_length, n_kv_heads, dim)
    assert cache_v.shape == (1, cache_length, n_kv_heads, dim)
    assert xq.stride(3) == 1
    assert xk.stride(3) == 1
    assert xv.stride(3) == 1
    assert cache_k.stride(3) == 1
    assert cache_v.stride(3) == 1
    n_total_heads = n_q_heads + 2 * n_kv_heads
    v_start = n_total_heads - n_kv_heads
    k_start = n_q_heads
    if out_q is None:
        out_q = xq.new_empty(1, n_total_queries, n_q_heads, dim)
    else:
        assert out_q.shape == xq.shape
        assert out_q.stride(3) == 1
    assert out_q is not None

    logical_bsz = len(attn_bias.q_seqinfo.seqstart_py) - 1

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // xq.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(dim))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    device = xq.device
    # Move these to the right device, like fmha does.
    attn_bias.k_seqinfo.to(device)
    attn_bias.q_seqinfo.to(device)
    seqstartq = attn_bias.q_seqinfo.seqstart
    seqstartk = attn_bias.k_seqinfo.seqstart
    seqlenk = attn_bias.k_seqinfo.seqlen
    assert internal_dtype in ["", "f32", "f64"]
    # experiment with the order of dims here.
    with torch.cuda.device(xq.device.index):
        # pyre-fixme[28]: Unexpected keyword argument `num_warps`.
        _rope_padded_kernel[
            (logical_bsz, attn_bias.q_seqinfo.max_seqlen, n_total_heads)
        ](
            xq,
            xk,
            xv,
            out_q,
            cache_k,
            cache_v,
            seqstartq,
            seqstartk,
            seqlenk,
            theta,
            k_start,
            v_start,
            dim,
            xq.stride(1),
            xq.stride(2),
            xk.stride(1),
            xk.stride(2),
            xv.stride(1),
            xv.stride(2),
            cache_k.stride(1),
            cache_k.stride(2),
            cache_v.stride(1),
            cache_v.stride(2),
            seqstartq.stride(0),
            seqstartk.stride(0),
            seqlenk.stride(0),
            out_q.stride(1),
            out_q.stride(2),
            _INTERNAL_DTYPE_MAP[internal_dtype],
            const_batch_strides=False,
            cache_padding_length=0,
            seqlenk_shift=0,
            BLOCK_SIZE=BLOCK_SIZE,
            adjacents=adjacents,
            num_warps=num_warps,
        )
    return out_q
