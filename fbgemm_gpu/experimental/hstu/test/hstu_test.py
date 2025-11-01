#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

import logging
import math
import os
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from fbgemm_gpu.experimental.hstu import hstu_attn_varlen_func, hstu_attn_qkvpacked_func, quantize_for_two_directions, quantize_for_block_scale, get_bm_and_bn_block_size_fwd, get_bm_and_bn_block_size_bwd, quantize_for_head_batch_tensor

from hypothesis import given, settings, strategies as st, Verbosity

running_on_github: bool = os.getenv("GITHUB_ENV") is not None

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

_MAX_SAMPLES: int = 200
debug = False
example = False
e4m3_max = 448.0

def pad_input(unpadded_input, cu_seqlen, batch, seqlen):
    indices = []
    for i in range(batch):
        indices.append(
            torch.arange(seqlen * i, seqlen * i + cu_seqlen[i + 1] - cu_seqlen[i])
        )
    indices = torch.cat(indices)
    output = torch.zeros(
        (batch * seqlen),
        *unpadded_input.shape[1:],
        device=unpadded_input.device,
        dtype=unpadded_input.dtype
    )
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)

def pad_input_delta_q(unpadded_input, cu_seqlen_q, cu_seqlen_k, batch, seqlen):
    indices = []
    for i in range(batch):
        act_seqlen_q = (cu_seqlen_q[i + 1] - cu_seqlen_q[i]).item()
        act_seqlen_k = (cu_seqlen_k[i + 1] - cu_seqlen_k[i]).item()
        indices.append(
            torch.arange(
                seqlen * i + act_seqlen_k - act_seqlen_q, seqlen * i + act_seqlen_k
            )
        )
    indices = torch.cat(indices)
    output = torch.zeros(
        (batch * seqlen),
        *unpadded_input.shape[1:],
        device=unpadded_input.device,
        dtype=unpadded_input.dtype
    )
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)

def unpad_input(padded_input, cu_seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(len(cu_seqlen) - 1):
        output.append(padded_input[i, : (cu_seqlen[i + 1] - cu_seqlen[i]), :])
    return torch.cat(output, dim=0)

def unpad_input_delta_q(padded_input, cu_seqlen_q, cu_seqlen_k, batch, seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(batch):
        act_seqlen_q = (cu_seqlen_q[i + 1] - cu_seqlen_q[i]).item()
        act_seqlen_k = (cu_seqlen_k[i + 1] - cu_seqlen_k[i]).item()
        output.append(padded_input[i, act_seqlen_k - act_seqlen_q : act_seqlen_k, :])
    return torch.cat(output, dim=0)

def construct_mask(
    batch_func,
    seqlen_c,
    seqlen,
    seqlen_t=0,
    target_group_size=1,
    window_size=(-1, -1),  # -1 means infinite window size
    func=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    num_contexts=None,
    device=torch.device("cuda"),
):
    seqlen = seqlen_c + seqlen + seqlen_t
    bs = cu_seqlens_k.size(0) - 1
    if debug:
        print(func)
    if func is not None:
        mask = torch.zeros((batch_func, seqlen, seqlen), device=device, dtype=torch.bool)
    else:
        mask = torch.zeros((seqlen, seqlen), device=device, dtype=torch.bool)
    mask[:] = False
    if func is not None:
        for b in range(batch_func):
            actual_seqlen_q = (cu_seqlens_q[b+1] - cu_seqlens_q[b]).item()
            actual_seqlen_k = (cu_seqlens_k[b+1] - cu_seqlens_k[b]).item()
            actual_offset = actual_seqlen_k - actual_seqlen_q
            for i in range(actual_seqlen_q):
                mask[b, actual_offset + i, 0 :func[b, 0, 0, i]] = True
            n_pair_fun = func.size(2) // 2
            for fun_idx in range(1, n_pair_fun+1):
                for i in range(actual_seqlen_q):
                    mask[b, actual_offset + i, func[b, 0, 2*fun_idx-1, i] : func[b, 0, 2*fun_idx, i]] = True
    elif window_size[0] < 0 and window_size[1] == 0:
        # causal mask
        for i in range(seqlen):
            mask[i, : i + 1] = True

        # context mask
        if seqlen_c != 0:
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
            for i in range(bs):
                target_start = (
                    num_contexts[i] + cu_seqlens_k[i + 1] - cu_seqlens_k[i]
                ).item()
                mask[i, 0, : num_contexts[i], :target_start] = True

        # target mask
        if seqlen_t != 0:
            mask = (
                mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
                if mask.ndim == 2
                else mask
            )
            for i in range(bs):
                target_start = (
                    num_contexts[i] + cu_seqlens_k[i + 1] - cu_seqlens_k[i]
                ).item()
                # target group mask
                if target_group_size > 1:
                    group_num = math.ceil((seqlen - target_start) / target_group_size)
                    for j in range(group_num):
                        for k in range(
                            min(
                                target_group_size,
                                seqlen - target_start - j * target_group_size,
                            )
                        ):
                            mask[
                                i,
                                0,
                                target_start + j * target_group_size + k,
                                target_start : target_start + j * target_group_size,
                            ] = False
                else:
                    for j in range(target_start, seqlen):
                        mask[i, 0, j, target_start:j] = False

    # local mask
    else:
        window_size_0 = window_size[0] if window_size[0] > 0 else seqlen
        window_size_1 = window_size[1] if window_size[1] > 0 else seqlen
        for i in range(seqlen):
            mask[i, max(0, i - window_size_0) : min(seqlen, i + window_size_1 + 1)] = (
                True
            )
    return mask

def make_heart_func(batch_func: int, L_q: int, L_k: int, device="cuda"):
    n_func = 3
    func = torch.zeros((batch_func, 1, n_func, L_q), dtype=torch.int32, device=device)

    x = torch.linspace(-1.3, 1.3, L_k, device=device)
    y = torch.linspace(-1.3, 1.3, L_q, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    inside = ((X*X + Y*Y - 1)**3 - (X*X)*(Y**3)) <= 0
    inside = inside.T  #

    if inside.size(0) == inside.size(1):
        inside = torch.rot90(inside, k=-1, dims=(0, 1))

    inside = torch.rot90(inside, k=2, dims=(0, 1))

    for i in range(L_q):
        row = inside[i]  # (L_k,)
        idx = torch.nonzero(row, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue

        start_col = idx.min().item()
        end_col_exclusive = idx.max().item() + 1
        func[:, :, 0, i] = 0
        func[:, :, 1, i] = start_col
        func[:, :, 2, i] = end_col_exclusive
    return func

def generate_input(
    batch_size: int,
    heads: int,
    heads_rab: Optional[int],
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_context_len: int,
    max_target_len: int,
    target_group_size: int,
    attn_dim: int,
    hidden_dim: int,
    window_size: tuple[int, int],
    dtype: torch.dtype,
    full_batch: bool,
    has_drab: bool,
    is_delta_q: bool,
    is_arbitrary: bool,
):
    has_context = max_context_len > 0
    has_target = max_target_len > 0
    # Generate lengths for context
    if max_context_len > 0:
        if full_batch:
            num_contexts = (
                torch.ones(
                    (batch_size,), device=torch.device("cuda"), dtype=torch.int32
                )
                * max_context_len
            )
        else:
            num_contexts = torch.randint(
                0,
                max_context_len + 1,
                size=(batch_size,),
                dtype=torch.int32,
                device=torch.device("cuda"),
            )
    else:
        num_contexts = torch.zeros(
            (batch_size,), dtype=torch.int32, device=torch.device("cuda")
        )
    cu_seqlens_c = torch.zeros(
        (batch_size + 1,),
        dtype=torch.int32,
        device=torch.accelerator.current_accelerator(),
    )
    cu_seqlens_c[1:] = torch.cumsum(num_contexts, dim=0)

    # Generate lengths for history qkv
    if full_batch:
        lengths_k = (
            torch.ones(
                (batch_size,),
                device=torch.accelerator.current_accelerator(),
                dtype=torch.int32,
            )
            * max_seq_len_k
        )
    else:
        lengths_k = torch.randint(
            1, max_seq_len_k + 1, size=(batch_size,), device=torch.device("cuda")
        )
    cu_seqlens_k = torch.zeros(
        (batch_size + 1,),
        dtype=torch.int32,
        device=torch.accelerator.current_accelerator(),
    )
    cu_seqlens_k[1:] = torch.cumsum(lengths_k, dim=0)

    # Generate lengths for target qkv
    if has_target:
        if full_batch:
            num_targets = (
                torch.ones(
                    (batch_size,), device=torch.device("cuda"), dtype=torch.int32
                )
                * max_target_len
            )
        else:
            num_targets = torch.randint(
                0,
                max_target_len + 1,
                size=(batch_size,),
                dtype=torch.int32,
                device=torch.device("cuda"),
            )
    else:
        num_targets = torch.zeros(
            (batch_size,), dtype=torch.int32, device=torch.device("cuda")
        )
    cu_seqlens_t = torch.zeros(
        (batch_size + 1,),
        dtype=torch.int32,
        device=torch.accelerator.current_accelerator(),
    )
    cu_seqlens_t[1:] = torch.cumsum(num_targets, dim=0)

    # Generate lengths for delta q
    if is_delta_q:
        if full_batch:
            lengths_q = (
                torch.ones(
                    (batch_size,), device=torch.device("cuda"), dtype=torch.int32
                )
                * max_seq_len_q
            )
        else:
            # lengths_q[i] is an integer between 1 and min(max_seq_len_q, lengths_k[i])
            lengths_q = torch.zeros(
                (batch_size,), device=torch.device("cuda"), dtype=torch.int32
            )
            for i in range(batch_size):
                lengths_q[i] = torch.randint(
                    1,
                    min(max_seq_len_q, lengths_k[i]) + 1,  # pyre-ignore[6]
                    size=(1,),
                    device=torch.device("cuda"),
                )
        cu_seqlens_q = torch.zeros(
            (batch_size + 1,),
            dtype=torch.int32,
            device=torch.accelerator.current_accelerator(),
        )
        cu_seqlens_q[1:] = torch.cumsum(lengths_q, dim=0)
    else:
        cu_seqlens_q = cu_seqlens_k

    # Lengths for whole q, kv
    cu_seqlens_q_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    cu_seqlens_q_wt = cu_seqlens_c + cu_seqlens_q + cu_seqlens_t
    cu_seqlens_k_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    cu_seqlens_k_wt = cu_seqlens_c + cu_seqlens_k + cu_seqlens_t

    L_q = int(cu_seqlens_q_wt[-1].item())
    L_k = int(cu_seqlens_k_wt[-1].item())
    if dtype == torch.float8_e4m3fn:
        dtype_init = torch.float16
    else:
        dtype_init = dtype

    # Generate q, k, v for context + history + target
    q = (
        torch.empty(
            (L_q, heads, attn_dim), dtype=dtype_init, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    )
    k = (
        torch.empty(
            (L_k, heads, attn_dim), dtype=dtype_init, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    )
    v = (
        torch.empty(
            (L_k, heads, hidden_dim), dtype=dtype_init, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    )

    if is_delta_q is False and dtype != torch.float8_e4m3fn:
        qkv = torch.empty((L_q, 3, heads, attn_dim), dtype=dtype, device=torch.device("cuda")).uniform_(-1, 1).requires_grad_()
        q = qkv[:, 0, :, :]
        k = qkv[:, 1, :, :]
        v = qkv[:, 2, :, :]
    else:
        qkv = None

    rab = torch.empty(
        (
            batch_size,
            heads if heads_rab is None else heads_rab,
            max_context_len + max_seq_len_k + max_target_len,
            max_context_len + max_seq_len_k + max_target_len,
        ),
        dtype=dtype_init,
        device=torch.accelerator.current_accelerator(),
    ).uniform_(-1, 1)

    head_func = 1
    batch_func = batch_size
    if is_arbitrary:
        n_func = 1
        max_seq_split = max_seq_len_k // n_func
        coef = 0.3
        func = torch.empty((batch_func, head_func, n_func, max_seq_len_q), dtype=torch.int32, device=torch.device("cuda"))
        for i in range(n_func):
            func[:, :, i, :] = torch.randint(i * max_seq_split, int((i+coef) * max_seq_split), size=(batch_func, head_func, max_seq_len_q), device=torch.device("cuda"))

        if example:
            # emulate casual mask
            n_func = 1  # export HSTU_ARBITRARY_NFUNC=1;
            func = torch.empty((batch_func, head_func, n_func, max_seq_len_q), dtype=torch.int32, device=torch.device("cuda"))
            for token_id in range(max_seq_len_k):
                func[:, :, 0, token_id] = token_id + 1

            # emulate local mask（2, 12）
            # [1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0]
            # [1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
            # [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
            # [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
            # [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
            # [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1]
            # [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1]
            # [0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
            # [0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
            # [0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
            # [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]
            # [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1]
            # [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1]
            # [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1]
            # [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1]
            # [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1]
            n_func = 3  # export HSTU_ARBITRARY_NFUNC=3;
            func = torch.empty((batch_func, head_func, n_func, max_seq_len_q), dtype=torch.int32, device=torch.device("cuda"))
            left_window_size = 2
            right_window_size = 12
            for token_id in range(max_seq_len_k):
                func[:, :, 0, token_id] = 0
                func[:, :, 1, token_id] = max(0, token_id - left_window_size)
                func[:, :, 2, token_id] = min(max_seq_len_k, token_id + right_window_size + 1)

            # emulate casual + dynamic + target
            # [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            # [1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            # [1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
            # [1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
            # [1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0]
            # [1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
            # [1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0]
            # [1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
            # [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
            # [1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
            # [1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
            # [1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
            # [1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0]
            # [1 1 1 1 0 0 0 0 0 0 0 0 0 1 0 0]
            # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
            # [1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1]
            n_func = 3  # export HSTU_ARBITRARY_NFUNC=3;
            func = torch.empty((batch_func, head_func, n_func, max_seq_len_q), dtype=torch.int32, device=torch.device("cuda"))
            left_window_size = 2
            right_window_size = 12
            casual_len = 8

            for token_id in range(casual_len):
                func[:, :, 0, token_id] = token_id + 1
                func[:, :, 1, token_id] = casual_len
                func[:, :, 2, token_id] = casual_len

            for token_id in range(casual_len, max_seq_len_k):
                func[:, :, 0, token_id] = torch.randint(0, casual_len+1, (1,), device=torch.device("cuda"))
                func[:, :, 1, token_id] = token_id
                func[:, :, 2, token_id] = token_id + 1

            # heart example
            func = make_heart_func(batch_func, L_q, L_k)
        if example:
            print("func", func)
    else:
        func = None
        batch_func = 1

    # kernel input is a variable-length function (var_fun). The following is copying the value of a fixed-length function(func) to a variable-length function(var_fun).
    # the variable-length function (var_func) can also be directly initialized. The initialization of the fixed-length function(func) is just provided as an example.

    # it is especially important to note that we need to do padding for the variable-length function.
    # +256 is to avoid the boundary problem in kernel, it can reduce BRA instructions, which helps improve kernel performance.
    # currently, we do not have a broadcasting mechanism, and each batch has its own mask shape.
    L_func = L_q + 256
    if is_arbitrary:
        var_fun = torch.empty((head_func, n_func, L_func), dtype=torch.int32, device=torch.device("cuda"))
        for i in range(batch_func):
            for j in range(head_func):
                var_fun[j, :, cu_seqlens_q_wt[i]:cu_seqlens_q_wt[i+1]] = func[i, j, :, 0:cu_seqlens_q_wt[i+1]-cu_seqlens_q_wt[i]]
    else:
        var_fun = None

    # TODO: add broadcast mask
    # # The mask shape will be broadcasted to each batch
    # if batch_func == 1:
    #     var_fun = func[0, 0, :, :]
    # else:
    #     # different batches use different mask shapes.
    #     for i in range(batch_func):
    #         for j in range(head_func):
    #             var_fun[j, :, cu_seqlens_q_wt[i]:cu_seqlens_q_wt[i+1]] = func[i, j, :, 0:cu_seqlens_q_wt[i+1]-cu_seqlens_q_wt[i]]
    if has_drab:
        rab = rab.requires_grad_()
    if window_size[0] == -1 and window_size[1] == -1 and func is None:
        attn_mask = None
    else:
        attn_mask = (
            construct_mask(
                batch_func=batch_func,
                seqlen_c=max_context_len,
                seqlen=max_seq_len_k,
                seqlen_t=max_target_len,
                target_group_size=target_group_size,
                window_size=window_size,
                func=func,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                num_contexts=num_contexts,
            )
            .cuda()
            .to(torch.float32)
        )
    if example and attn_mask is not None:
        print(attn_mask.to(torch.int32).squeeze().cpu().numpy())
    return (
        L_q,
        L_k,
        num_contexts if has_context else None,
        cu_seqlens_q_wt,
        cu_seqlens_k_wt,
        num_targets if has_target else None,
        qkv,
        q,
        k,
        v,
        rab,
        attn_mask,
        var_fun,
    )


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    rab: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    alpha: float,
    upcast: bool = True,
    is_delta_q: bool = False,
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    B: int = q_offsets.size(0) - 1
    dtype_out = q.dtype
    if is_delta_q:
        padded_q = pad_input_delta_q(q, q_offsets, k_offsets, B, seqlen_k)
    else:
        padded_q = pad_input(q, q_offsets, B, seqlen_q)
    padded_k = pad_input(k, k_offsets, B, seqlen_k)
    padded_v = pad_input(v, k_offsets, B, seqlen_k)

    padded_q = padded_q.view(B, seqlen_k, num_heads, attention_dim)
    padded_k = padded_k.view(B, seqlen_k, num_heads, attention_dim)
    padded_v = padded_v.view(B, seqlen_k, num_heads, linear_dim)
    if upcast:
        padded_q, padded_k, padded_v = (
            padded_q.float(),
            padded_k.float(),
            padded_v.float(),
        )
        if rab is not None:
            rab = rab.float()
    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q,
        padded_k,
    )

    if rab is not None:
        padding = (
            0,
            qk_attn.shape[-1] - rab.shape[-1],
            0,
            qk_attn.shape[-2] - rab.shape[-2],
        )
        rab = F.pad(rab, padding, value=0)
        masked_qk_attn = qk_attn + rab
    else:
        masked_qk_attn = qk_attn
    masked_qk_attn = masked_qk_attn * alpha
    masked_qk_attn = F.silu(masked_qk_attn)
    masked_qk_attn = masked_qk_attn / seqlen_q
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        elif invalid_attn_mask.ndim == 3:
            invalid_attn_mask = invalid_attn_mask.unsqueeze(1)
        masked_qk_attn = masked_qk_attn * invalid_attn_mask.type(masked_qk_attn.dtype)

    attn_output = torch.einsum(
        "bhnm,bmhd->bnhd",
        masked_qk_attn,
        padded_v,
    )

    attn_output = attn_output.reshape(B, seqlen_k, num_heads * linear_dim)
    if is_delta_q:
        attn_output = unpad_input_delta_q(
            attn_output, q_offsets, k_offsets, B, seqlen_k
        )
    else:
        attn_output = unpad_input(attn_output, q_offsets)
    attn_output = attn_output.reshape(-1, num_heads, linear_dim)

    return attn_output.to(dtype_out)


@unittest.skipIf(
    not torch.cuda.is_available()
    or (torch.version.hip is None and torch.version.cuda < "12.4"),
    "Skip when no GPU is available or CUDA version is older than `12.4`.",
)
class HSTU16Test(unittest.TestCase):
    """Test HSTU attention with float16 inputs."""

    @unittest.skipIf(
        running_on_github, "GitHub runners are unable to run the test at this time"
    )
    @given(
        batch_size=st.sampled_from([32]),
        heads=st.sampled_from([2]),
        seq_len_params=st.sampled_from(
            [
                (32, 32),
                (99, 99),
                (256, 256),
                (1111, 1111),
                (27, 32),
                (8, 99),
                (51, 256),
                (160, 2000),
            ]
        ),
        max_context_len=st.sampled_from([0, 99, 160, 333]),
        target_params=st.sampled_from(
            [
                (0, (-1, -1), 1, False),
                (0, (-1, -1), 1, True),
                (0, (111, 11), 1, False),
                (0, (111, 222), 1, False),
                (0, (-1, 0), 1, False),
                (32, (-1, 0), 1, False),
                (257, (-1, 0), 1, False),
                (1024, (-1, 0), 1, False),
                (111, (-1, 0), 11, False),
                (1111, (-1, 0), 222, False),
            ]
        ),
        attn_hidden_dims=st.sampled_from(
            [
                (32, 32),
                (64, 64),
                (128, 128),
                (256, 256),
            ]
        ),
        alpha=st.sampled_from([1.0, 0.1]),
        rab_params=st.sampled_from(
            [
                (False, False, None),
                (True, False, None),  # None means heads_rab=heads
                (True, True, None),
                (True, False, 1),
                (True, True, 1),
            ]
        ),
        dtype=st.sampled_from([torch.bfloat16]), #, torch.float16]),
        full_batch=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_hstu_attn(
        self,
        batch_size: int,
        heads: int,
        max_context_len: int,
        attn_hidden_dims: tuple[int, int],
        alpha: float,
        rab_params: Tuple[bool, bool, Optional[int]],
        seq_len_params: Tuple[int, int, bool],
        target_params: Tuple[int, Tuple[int, int], int, bool],
        dtype: torch.dtype,
        full_batch: bool,
    ) -> None:
        max_seq_len_q, max_seq_len_k = seq_len_params
        max_target_len, window_size, target_group_size, is_arbitrary = target_params
        attn_dim, hidden_dim = attn_hidden_dims
        has_rab, has_drab, heads_rab = rab_params

        has_context = max_context_len > 0
        has_target = max_target_len > 0
        is_causal = window_size[0] == -1 and window_size[1] == 0
        is_delta_q = max_seq_len_q < max_seq_len_k
        if is_delta_q and has_target:
            logger.info("Skipping test for is_delta_q and has_target")
            return
        if is_delta_q and has_context:
            logger.info("Skipping test for is_delta_q and has_context")
            return
        if not is_causal and has_context:
            logger.info("Skipping test for not is_causal and has_context")
            return
        if (window_size[0] > 0 or window_size[1] > 0) and has_context:
            logger.info(
                "Skipping test for (window_size[0] > 0 or window_size[1] > 0) and has_context"
            )
            return

        torch.cuda.synchronize()
        (
            L_q,
            L_k,
            num_contexts,
            cu_seqlens_q,
            cu_seqlens_k,
            num_targets,
            qkv,
            q,
            k,
            v,
            rab,
            attn_mask,
            func,
        ) = generate_input(
            batch_size=batch_size,
            heads=heads,
            heads_rab=heads_rab,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            max_context_len=max_context_len,
            max_target_len=max_target_len,
            target_group_size=target_group_size,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            window_size=window_size,
            dtype=dtype,
            full_batch=full_batch,
            has_drab=has_drab,
            is_delta_q=is_delta_q,
            is_arbitrary=is_arbitrary,
        )
        out_ref = _hstu_attention_maybe_from_cache(
            num_heads=heads,
            attention_dim=attn_dim,
            linear_dim=hidden_dim,
            seqlen_q=max_context_len + max_seq_len_q + max_target_len,
            seqlen_k=max_context_len + max_seq_len_k + max_target_len,
            q=q.view(L_q, -1),
            k=k.view(L_k, -1),
            v=v.view(L_k, -1),
            q_offsets=cu_seqlens_q,
            k_offsets=cu_seqlens_k,
            rab=rab if has_rab else None,
            invalid_attn_mask=(
                attn_mask.to(torch.float32) if attn_mask is not None else None
            ),
            alpha=alpha,
            is_delta_q=is_delta_q,
        )

        torch_out = _hstu_attention_maybe_from_cache(
            num_heads=heads,
            attention_dim=attn_dim,
            linear_dim=hidden_dim,
            seqlen_q=max_context_len + max_seq_len_q + max_target_len,
            seqlen_k=max_context_len + max_seq_len_k + max_target_len,
            q=q.view(L_q, -1),
            k=k.view(L_k, -1),
            v=v.view(L_k, -1),
            q_offsets=cu_seqlens_q,
            k_offsets=cu_seqlens_k,
            rab=rab if has_rab else None,
            invalid_attn_mask=(
                attn_mask.to(torch.float32) if attn_mask is not None else None
            ),
            alpha=alpha,
            upcast=False,
            is_delta_q=is_delta_q,
        )
        if qkv is None:
            hstu_out = hstu_attn_varlen_func(
                q=q.to(dtype),
                k=k.to(dtype),
                v=v.to(dtype),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_context_len + max_seq_len_q + max_target_len,
                max_seqlen_k=max_context_len + max_seq_len_k + max_target_len,
                num_contexts=num_contexts,
                num_targets=num_targets,
                target_group_size=target_group_size,
                window_size=window_size,
                alpha=alpha,
                rab=rab if has_rab else None,
                has_drab=has_drab,
                func=func,
            )
        else:
            hstu_out = hstu_attn_qkvpacked_func(
                qkv=qkv.to(dtype),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_context_len + max_seq_len_q + max_target_len,
                max_seqlen_k=max_context_len + max_seq_len_k + max_target_len,
                num_contexts=num_contexts,
                num_targets=num_targets,
                target_group_size=target_group_size,
                window_size=window_size,
                alpha=alpha,
                rab=rab if has_rab else None,
                has_drab=has_drab,
                func=func,
            )

        # print(f"Output max diff: {(hstu_out - out_ref).abs().max().item()}")
        # print(f"Pytorch max diff: {(torch_out - out_ref).abs().max().item()}")
        # print(f"Output mean diff: {(hstu_out - out_ref).abs().mean().item()}")
        # print(f"Pytorch mean diff: {(torch_out - out_ref).abs().mean().item()}")

        assert (hstu_out - out_ref).abs().max().item() <= 2 * (torch_out - out_ref).abs().max().item()

        g = torch.rand_like(torch_out)
        if not has_drab:
            (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(
                out_ref, (q, k, v), g, retain_graph=True
            )
            (dq_torch, dk_torch, dv_torch) = torch.autograd.grad(
                torch_out, (q, k, v), g, retain_graph=True
            )
            if qkv is None:
                (dq_hstu, dk_hstu, dv_hstu) = torch.autograd.grad(
                    hstu_out,
                    (q, k, v),
                    g,
                    retain_graph=True,
                )
            else:
                (dqkv_hstu) = torch.autograd.grad(
                    hstu_out, qkv, g, retain_graph=True,
                )
                dq_hstu = dqkv_hstu[0][:, 0, :, :]
                dk_hstu = dqkv_hstu[0][:, 1, :, :]
                dv_hstu = dqkv_hstu[0][:, 2, :, :]
        else:
            (dq_ref, dk_ref, dv_ref, drab_ref) = torch.autograd.grad(
                out_ref, (q, k, v, rab), g, retain_graph=True
            )
            (dq_torch, dk_torch, dv_torch, drab_torch) = torch.autograd.grad(
                torch_out, (q, k, v, rab), g, retain_graph=True
            )
            if qkv is None:
                (dq_hstu, dk_hstu, dv_hstu, drab_hstu) = torch.autograd.grad(
                hstu_out,
                    (q, k, v, rab),
                    g,
                    retain_graph=True,
                )
            else:
                (dqkv_hstu, drab_hstu) = torch.autograd.grad(
                    hstu_out, (qkv, rab), g, retain_graph=True,
                )
                dq_hstu = dqkv_hstu[:, 0, :, :]
                dk_hstu = dqkv_hstu[:, 1, :, :]
                dv_hstu = dqkv_hstu[:, 2, :, :]

        # print(f"dV max diff: {(dv_hstu - dv_ref).abs().max().item()}")
        # print(f"dV Pytorch max diff: {(dv_torch - dv_ref).abs().max().item()}")
        # print(f"dK max diff: {(dk_hstu - dk_ref).abs().max().item()}")
        # print(f"dK Pytorch max diff: {(dk_torch - dk_ref).abs().max().item()}")
        # print(f"dQ max diff: {(dq_hstu - dq_ref).abs().max().item()}")
        # print(f"dQ Pytorch max diff: {(dq_torch - dq_ref).abs().max().item()}")
        # if has_drab:
        #     print(f"dRab max diff: {(drab_hstu - drab_ref).abs().max().item()}")
        #     print(f"dRab Pytorch max diff: {(drab_torch - drab_ref).abs().max().item()}")

        assert (dv_hstu - dv_ref).abs().max().item() <= 5 * (  # pyre-ignore[58]
            dv_torch - dv_ref
        ).abs().max().item()
        assert (dk_hstu - dk_ref).abs().max().item() <= 5 * (  # pyre-ignore[58]
            dk_torch - dk_ref
        ).abs().max().item()
        assert (dq_hstu - dq_ref).abs().max().item() <= 5 * (  # pyre-ignore[58]
            dq_torch - dq_ref
        ).abs().max().item()
        if has_drab:
            assert (drab_hstu - drab_ref).abs().max().item() <= 5 * (  # pyre-ignore[58,61]
                drab_torch - drab_ref  # pyre-ignore[61]
            ).abs().max().item()
        torch.cuda.synchronize()


def generate_paged_kv_input(
    batch_size: int,
    heads: int,
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_target_len: int,
    attn_dim: int,
    hidden_dim: int,
    page_size: int,
    dtype: torch.dtype,
    full_batch: bool,
):
    # Generate lengths for new history qkv
    if full_batch:
        lengths_q = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int32)
            * max_seq_len_q
        )
    else:
        lengths_q = torch.randint(1, max_seq_len_q + 1, size=(batch_size,), device=torch.device("cuda"))
    cu_seqlens_q = torch.zeros((batch_size + 1,), dtype=torch.int32, device=torch.device("cuda"))
    cu_seqlens_q[1:] = torch.cumsum(lengths_q, dim=0)

    # Generate lengths for target qkv
    if full_batch:
        num_targets = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int32)
            * max_target_len
        )
    else:
        num_targets = torch.randint(0, max_target_len + 1, size=(batch_size,), dtype=torch.int32, device=torch.device("cuda"))
    cu_seqlens_t = torch.zeros((batch_size + 1,), dtype=torch.int32, device=torch.device("cuda"))
    cu_seqlens_t[1:] = torch.cumsum(num_targets, dim=0)

    # Lengths for new history + target qkv
    cu_seqlens_q_wt = cu_seqlens_q + cu_seqlens_t
    L_q = int(cu_seqlens_q_wt[-1].item())

    # Generate q, k, v for new history + target
    M_uvqk = torch.empty((L_q, 4, heads, attn_dim), dtype=dtype, device=torch.device("cuda")).uniform_(-1, 1)
    q = M_uvqk[:, 2, :, :]
    k = M_uvqk[:, 3, :, :]
    v = M_uvqk[:, 1, :, :]

    # Generate user feature + previous history
    if full_batch:
        lengths_k = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int32)
            * max_seq_len_k
        )
    else:
        lengths_k = torch.randint(1, max_seq_len_k + 1, size=(batch_size,), device=torch.device("cuda"))
    cu_seqlens_k = torch.zeros((batch_size + 1,), dtype=torch.int32, device=torch.device("cuda"))
    cu_seqlens_k[1:] = torch.cumsum(lengths_k, dim=0)

    # Lengths for user feature + previous history + new history
    lengths_k_cache = lengths_k + lengths_q
    lengths_page = (lengths_k_cache + page_size - 1) // page_size
    page_offsets = torch.zeros((batch_size + 1,), dtype=torch.int32, device=torch.device("cuda"))
    page_offsets[1:] = torch.cumsum(lengths_page, dim=0)

    total_page = int(page_offsets[-1].item())
    page_ids = torch.randperm(total_page, device=torch.device("cuda"), dtype=torch.int32)
    last_page_lens = ((lengths_k_cache - 1) % page_size + 1).to(torch.int32)
    kv_cache = torch.empty((total_page, 2, page_size, heads, attn_dim), dtype=dtype, device=torch.device("cuda")).uniform_(-1, 1)

    mask = torch.zeros((batch_size, 1, max_seq_len_q + max_seq_len_k + max_target_len, max_seq_len_q + max_seq_len_k + max_target_len),
                       device=torch.device("cuda"), dtype=torch.float32)
    for i in range(batch_size):
        # new history part
        for j in range(lengths_k_cache[i]):
            mask[i, 0, j, :j + 1] = 1.0

        # target part
        mask[i, 0, lengths_k_cache[i]:lengths_k_cache[i] + num_targets[i], :lengths_k_cache[i]] = 1.0
        for j in range(num_targets[i]):
            mask[i, 0, lengths_k_cache[i] + j, lengths_k_cache[i] + j] = 1.0

    return (
        L_q,
        cu_seqlens_q_wt, cu_seqlens_k + cu_seqlens_q_wt, num_targets,
        page_offsets, page_ids, last_page_lens,
        q, k, v, kv_cache, mask
    )


def _hstu_paged_kv_attention(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    alpha: float,
    upcast: bool = True,
    kv_cache: torch.Tensor = None,
    page_offsets: torch.Tensor = None,
    page_ids: torch.Tensor = None,
    last_page_lens: torch.Tensor = None,
):
    k_con = torch.empty((0, num_heads, attention_dim), device=k.device, dtype=k.dtype)
    v_con = torch.empty((0, num_heads, attention_dim), device=v.device, dtype=v.dtype)

    for i in range(len(last_page_lens)):
        page_num = page_offsets[i + 1] - page_offsets[i]
        new_history_len = q_offsets[i + 1] - q_offsets[i] - num_targets[i]
        for j in range(page_num - 1):
            k_con = torch.cat((k_con, kv_cache[page_ids[page_offsets[i] + j], 0, :, :, :]), dim=0)
            v_con = torch.cat((v_con, kv_cache[page_ids[page_offsets[i] + j], 1, :, :, :]), dim=0)
        k_con = torch.cat((k_con, kv_cache[page_ids[page_offsets[i + 1] - 1], 0, :last_page_lens[i], :, :]), dim=0)
        k_con = torch.cat((k_con, k[(q_offsets[i] + new_history_len):q_offsets[i + 1], :, :]), dim=0)
        v_con = torch.cat((v_con, kv_cache[page_ids[page_offsets[i + 1] - 1], 1, :last_page_lens[i], :, :]), dim=0)
        v_con = torch.cat((v_con, v[(q_offsets[i] + new_history_len):q_offsets[i + 1], :, :]), dim=0)

    return _hstu_attention_maybe_from_cache(
              num_heads=num_heads,
              attention_dim=attention_dim,
              linear_dim=linear_dim,
              seqlen_q=seqlen_q,
              seqlen_k=seqlen_k,
              q=q,
              k=k_con,
              v=v_con,
              q_offsets=q_offsets,
              k_offsets=k_offsets,
              rab=None,
              invalid_attn_mask=invalid_attn_mask,
              alpha=alpha,
              upcast=upcast,
              is_delta_q=True,
          )

@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (9, 0),
    "Skip when only Hopper GPU. This test is not supported."
)
class HSTUPagedKVTest(unittest.TestCase):
    """Test HSTU paged kv attention."""

    @given(
        batch_size=st.sampled_from([32]),
        heads=st.sampled_from([2]),
        max_seq_len_q=st.sampled_from([21, 53, 201, 302]),
        max_seq_len_k=st.sampled_from([10, 99, 111, 256, 717]),
        max_target_len=st.sampled_from([0, 32, 111, 501]),
        attn_hidden_dims=st.sampled_from([(32, 32), (64, 64), (128, 128), (256, 256)]),
        page_size=st.sampled_from([32, 64]),
        alpha=st.sampled_from([1.0, 1.0 / (100 ** 0.5)]),
        dtype=st.sampled_from([torch.bfloat16]),
        full_batch=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_paged_kv_attn(
        self,
        batch_size: int,
        heads: int,
        max_seq_len_q: int,  # new history length
        max_seq_len_k: int,  # user feature + previous history
        max_target_len: int,
        attn_hidden_dims: Tuple[int, int],
        page_size: int,
        alpha: float,
        dtype: torch.dtype,
        full_batch: bool,
    ) -> None:
        attn_dim, hidden_dim = attn_hidden_dims
        torch.cuda.synchronize()
        L_q, cu_seqlens_q, cu_seqlens_k, num_targets, page_offsets, page_ids, last_page_lens, q, k, v, kv_cache, mask = generate_paged_kv_input(
            batch_size=batch_size,
            heads=heads,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            max_target_len=max_target_len,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            page_size=page_size,
            dtype=dtype,
            full_batch=full_batch,
        )
        out_ref = _hstu_paged_kv_attention(
            num_heads=heads,
            attention_dim=attn_dim,
            linear_dim=hidden_dim,
            seqlen_q=max_seq_len_q + max_target_len,
            seqlen_k=max_seq_len_q + max_seq_len_k + max_target_len,
            q=q,
            k=k,
            v=v,
            q_offsets=cu_seqlens_q,
            k_offsets=cu_seqlens_k,
            num_targets=num_targets,
            invalid_attn_mask=mask,
            alpha=alpha,
            upcast=True,
            kv_cache=kv_cache,
            page_offsets=page_offsets,
            page_ids=page_ids,
            last_page_lens=last_page_lens
        )

        torch_out = _hstu_paged_kv_attention(
            num_heads=heads,
            attention_dim=attn_dim,
            linear_dim=hidden_dim,
            seqlen_q=max_seq_len_q + max_target_len,
            seqlen_k=max_seq_len_q + max_seq_len_k + max_target_len,
            q=q,
            k=k,
            v=v,
            q_offsets=cu_seqlens_q,
            k_offsets=cu_seqlens_k,
            num_targets=num_targets,
            invalid_attn_mask=mask,
            alpha=alpha,
            upcast=False,
            kv_cache=kv_cache,
            page_offsets=page_offsets,
            page_ids=page_ids,
            last_page_lens=last_page_lens
        )

        hstu_out = hstu_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seq_len_q + max_target_len,
            max_seqlen_k=max_seq_len_q + max_seq_len_k + max_target_len,
            num_contexts=None,
            num_targets=num_targets,
            target_group_size=1,
            window_size=(-1, 0),
            alpha=alpha,
            rab=None,
            has_drab=False,
            kv_cache=kv_cache,
            page_offsets=page_offsets,
            page_ids=page_ids,
            last_page_lens=last_page_lens,
            func=None
        )

        # print(f"Output max diff: {(hstu_out - out_ref).abs().max().item()}")
        # print(f"Pytorch max diff: {(torch_out - out_ref).abs().max().item()}")
        # print(f"Output mean diff: {(hstu_out - out_ref).abs().mean().item()}")
        # print(f"Pytorch mean diff: {(torch_out - out_ref).abs().mean().item()}")

        assert (hstu_out - out_ref).abs().max().item() <= 2 * (torch_out - out_ref).abs().max().item()
        torch.cuda.synchronize()


def P_blockwise_Vt_gemm_fp8(
    P: torch.Tensor, # (B, H, SQ, SK)
    Vt: torch.Tensor, # (B, H, SK, D), fp8
    Vt_descale: torch.Tensor, # (sum(ceil(actual, 128)), H, D)
    cu_seqlens_vt_descale: torch.Tensor, # (B + 1)
    BM: int,
    BN: int,
    swapQK: bool = False,
    start_ids: torch.Tensor = None, # (B)
    mode: int = 1,
    is_qdo_offset: bool = False
):
    P = P.contiguous()
    Vt = Vt.contiguous()
    B = P.shape[0]
    H= P.shape[1]
    seq_len_q = P.shape[2]
    seq_len_k = P.shape[3]
    dim = Vt.shape[3]
    BM = BN if swapQK else BM
    BN = BM if swapQK else BN

    is_delta_q_m = (swapQK == False) and (start_ids is not None)
    is_delta_q_n = (swapQK == True) and (start_ids is not None)

    output = torch.zeros(B, H, seq_len_q, dim, dtype=torch.float, device='cuda')
    descale_one = torch.tensor([1.0], dtype=torch.float32, device='cuda')
    for bs in range(B):
      for h in range(H):
        start_q = start_ids[bs] if start_ids is not None else 0
        start_q_m = start_q if is_delta_q_m else 0
        start_q_n = start_q if is_delta_q_n else 0
        start_q_B = start_q if is_qdo_offset else 0
        total_M = math.ceil((seq_len_q - start_q_m) / BM)
        if mode == 1:
            actual_len_vt_descale = (cu_seqlens_vt_descale[bs+1] - cu_seqlens_vt_descale[bs]).item()
            total_N = min(math.ceil((seq_len_k - start_q_n) / BN), actual_len_vt_descale * 128 // BN)
        elif mode == 2:
            total_N = min(math.ceil((seq_len_k - start_q_n) / BN), (cu_seqlens_vt_descale[bs+1] - cu_seqlens_vt_descale[bs]))
        else:
            total_N = math.ceil((seq_len_k - start_q_n) / BN)

        for kM in range(total_M):
          for kN in range(total_N):
            P_block = P[bs, h, start_q_m + kM * BM : start_q_m + (kM + 1) * BM, start_q_n + kN * BN : start_q_n + (kN + 1) * BN]
            descale_Pblock = torch.max(P_block.abs()) / e4m3_max
            descale_Pblock = torch.max(descale_Pblock, torch.tensor([1e-6], dtype=torch.float32, device='cuda'))
            P_block = (P_block / descale_Pblock).to(torch.float8_e4m3fn)
            Vt_block = Vt[bs, h, start_q_B + kN * BN : start_q_B + (kN + 1) * BN, :].transpose(0, 1).contiguous().transpose(0, 1)
            output_tmp = torch._scaled_mm(P_block, Vt_block, out_dtype=torch.float, scale_a=descale_Pblock, scale_b=descale_one)
            if mode == 1:
                V_scale = Vt_descale[cu_seqlens_vt_descale[bs] + kN * BN // 128, h, :]
            elif mode == 2:
                V_scale = Vt_descale[cu_seqlens_vt_descale[bs] + kN, h]
            elif mode == 3:
                V_scale = Vt_descale[bs, h]
            elif mode == 4:
                V_scale = Vt_descale[bs]
            else:
                V_scale = Vt_descale
            output[bs, h, start_q_m + kM * BM : start_q_m + (kM + 1) * BM, :] += output_tmp * V_scale
    return output


def AB_blockscale_gemm_fp8(
    ab_attn: torch.Tensor, # (SQ, SK)
    a_offsets: torch.Tensor, # (B + 1)
    b_offsets: torch.Tensor, # (B + 1)
    a_descale: torch.Tensor, # (sum(ceil(actual, 128)), H)
    b_descale: torch.Tensor, # (sum(ceil(actual, 128)), H)
    cu_seqlens_block_descale_a,
    cu_seqlens_block_descale_b,
    B,
    bm,
    bn,
):
    for bs in range(B):
        cur_q_start_offset = (b_offsets[bs+1]-b_offsets[bs]) - (a_offsets[bs+1]-a_offsets[bs])
        #step1: padding 0 at beginning, and do offset
        cur_bs_ab_atten = ab_attn[bs, :, cur_q_start_offset: cur_q_start_offset + (a_offsets[bs+1]-a_offsets[bs]), : b_offsets[bs+1]-b_offsets[bs]]
        actual_len_padding_block_num_a = ((a_offsets[bs+1]-a_offsets[bs]) + bm - 1) // bm
        actual_len_padding_block_num_b = ((b_offsets[bs+1]-b_offsets[bs]) + bn - 1) // bn
        target_a_len = actual_len_padding_block_num_a * bm
        target_b_len = actual_len_padding_block_num_b * bn
        a_padded_len = target_a_len - (a_offsets[bs+1]-a_offsets[bs])
        b_padded_len = target_b_len - (b_offsets[bs+1]-b_offsets[bs])
        #step2: after slicing the real data, padding 0 based on BM and BN
        cur_padded_bs_ab_atten = F.pad(cur_bs_ab_atten, (0, b_padded_len, 0, a_padded_len)) #padding after original tensor
        cur_padded_bs_ab_atten_view = cur_padded_bs_ab_atten.view(cur_padded_bs_ab_atten.shape[0], actual_len_padding_block_num_a, bm, actual_len_padding_block_num_b, bn) #[1, 1, 128, 5, 64])
        cur_descale_a = a_descale[cu_seqlens_block_descale_a[bs] : cu_seqlens_block_descale_a[bs+1], :].permute(1, 0)
        cur_descale_b = b_descale[cu_seqlens_block_descale_b[bs] : cu_seqlens_block_descale_b[bs+1], :].permute(1, 0)

        for row in range(actual_len_padding_block_num_a):
            for col in range(actual_len_padding_block_num_b):
                scale = (cur_descale_a[:, row] * cur_descale_b[:, col]).view(-1, 1, 1)  # (num_heads, 1, 1)
                cur_padded_bs_ab_atten_view[:, row, :, col, :] = cur_padded_bs_ab_atten_view[:, row, :, col, :] * scale
        cur_padded_bs_ab_atten_view = cur_padded_bs_ab_atten_view.view(cur_bs_ab_atten.shape[0], actual_len_padding_block_num_a * bm, actual_len_padding_block_num_b * bn)
        #step3: Restore the true shape of is_delat before padding
        cur_bs_ab_atten = cur_padded_bs_ab_atten_view[:, 0:(a_offsets[bs+1]-a_offsets[bs]), 0:(b_offsets[bs+1]-b_offsets[bs])]
        #step4: No longer using padding techniques, but directly assigning slice values
        ab_attn[bs, :, cur_q_start_offset: cur_q_start_offset + (a_offsets[bs+1]-a_offsets[bs]), : b_offsets[bs+1]-b_offsets[bs]] = cur_bs_ab_atten

    return ab_attn

def _hstu_attention_maybe_from_cache_fp8(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    rab: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    alpha: float,
    quant_mode: int,
    is_delta_q: bool,
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    B: int = q_offsets.size(0) - 1
    n_q: int = seqlen_q  # max_seq_len
    n_k: int = seqlen_k  # max_seq_len
    ori_n_q: int = n_q
    ori_n_k: int = n_k
    n_q = 16 * math.ceil(seqlen_q / 16)
    n_k = 16 * math.ceil(seqlen_k / 16)
    dtype_out = torch.float16
    bm, bn = get_bm_and_bn_block_size_fwd(rab, attention_dim)

    if quant_mode == 0:
        q_fp8 = q.to(torch.float8_e4m3fn)
        k_fp8 = k.to(torch.float8_e4m3fn)
        v_fp8 = v.to(torch.float8_e4m3fn)
    elif quant_mode == 1:
        q_fp8, q_descale, _, _, _ = quantize_for_two_directions(q, q_offsets)
        k_fp8, k_descale, _, _, _ = quantize_for_two_directions(k, k_offsets)
        _, _, v_fp8, v_descale, cu_seqlens_v_descale = quantize_for_two_directions(v, k_offsets)
    elif quant_mode == 2:
        q_fp8, q_descale, cu_seqlens_block_descale_q = quantize_for_block_scale(q, q_offsets, block_size=bm)
        k_fp8, k_descale, cu_seqlens_block_descale_k = quantize_for_block_scale(k, k_offsets, block_size=bn)
        v_fp8, v_descale, cu_seqlens_block_descale_k = quantize_for_block_scale(v, k_offsets, block_size=bn)
        q_descale = q_descale.transpose(1, 0)
        k_descale = k_descale.transpose(1, 0)
        v_descale = v_descale.transpose(1, 0)
    elif quant_mode == 3 or quant_mode == 4 or quant_mode == 5:
        q_fp8, q_descale = quantize_for_head_batch_tensor(q, q_offsets, quant_mode=quant_mode)
        k_fp8, k_descale = quantize_for_head_batch_tensor(k, k_offsets, quant_mode=quant_mode)
        v_fp8, v_descale = quantize_for_head_batch_tensor(v, k_offsets, quant_mode=quant_mode)

    if is_delta_q:
        padded_q = pad_input_delta_q(q_fp8, q_offsets, k_offsets, B, n_k) #padding at beginning
    else:
        padded_q = pad_input(q_fp8, q_offsets, B, n_q)
    padded_k = pad_input(k_fp8, k_offsets, B, n_k)
    padded_v = pad_input(v_fp8, k_offsets, B, n_k)

    padded_q = padded_q.view(B, n_k, num_heads, attention_dim).permute(0, 2, 1, 3).contiguous()
    padded_k = padded_k.view(B, n_k, num_heads, attention_dim).permute(0, 2, 1, 3).contiguous()
    padded_v = padded_v.view(B, n_k, num_heads, linear_dim).permute(0, 2, 1, 3).contiguous()

    padded_q = padded_q.reshape(-1, n_k, attention_dim)
    padded_k = padded_k.reshape(-1, n_k, attention_dim).permute(0, 2, 1)
    descale_one = torch.tensor([1.0], dtype=torch.float32, device='cuda')

    # only support MK @ KN
    qk_attn = torch._scaled_mm(padded_q[0], padded_k[0], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)
    for i in range(1, padded_q.size(0)):
        qk_attn = torch.cat((qk_attn, torch._scaled_mm(padded_q[i], padded_k[i], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)), dim=0)
    qk_attn = qk_attn.view(B, num_heads, n_k, n_k)

    if quant_mode == 1:
        for bs in range(B):
            actual_seqlen_q = q_offsets[bs+1] - q_offsets[bs]
            actual_seqlen_k = k_offsets[bs+1] - k_offsets[bs]
            qk_attn[bs, :, actual_seqlen_k - actual_seqlen_q : actual_seqlen_k, :] = qk_attn[bs, :, actual_seqlen_k - actual_seqlen_q : actual_seqlen_k, :] * q_descale[:, q_offsets[bs] : q_offsets[bs+1]].unsqueeze(-1)
            qk_attn[bs, :, :, : actual_seqlen_k] = qk_attn[bs, :, :, : actual_seqlen_k] * k_descale[:, k_offsets[bs] : k_offsets[bs+1]].unsqueeze(-2)
    elif quant_mode == 2:
        qk_attn = AB_blockscale_gemm_fp8(
            ab_attn=qk_attn,
            a_offsets=q_offsets,
            b_offsets=k_offsets,
            a_descale=q_descale,
            b_descale=k_descale,
            cu_seqlens_block_descale_a=cu_seqlens_block_descale_q,
            cu_seqlens_block_descale_b=cu_seqlens_block_descale_k,
            B=B,
            bm=bm,
            bn=bn,
        )
    elif quant_mode == 3:
        qk_attn = qk_attn * q_descale.unsqueeze(-1).unsqueeze(-1) * k_descale.unsqueeze(-1).unsqueeze(-1)
    elif quant_mode == 4:
        qk_attn = qk_attn * q_descale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k_descale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    elif quant_mode == 5:
        qk_attn = qk_attn * q_descale * k_descale

    if rab is not None:
        padding = (
            0,
            qk_attn.shape[-1] - rab.shape[-1],
            0,
            qk_attn.shape[-2] - rab.shape[-2],
        )
        rab = F.pad(rab, padding, value=0)
        masked_qk_attn = qk_attn + rab
    else:
        masked_qk_attn = qk_attn
    masked_qk_attn = masked_qk_attn * alpha
    masked_qk_attn = F.silu(masked_qk_attn)
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            if invalid_attn_mask.shape[0] != n_k or invalid_attn_mask.shape[1] != n_k:
                invalid_attn_mask = F.pad(
                    invalid_attn_mask, (0, n_k - ori_n_k, 0, n_k - ori_n_k), value=0
                )
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        elif invalid_attn_mask.ndim == 3:
            if invalid_attn_mask.shape[1] != n_k or invalid_attn_mask.shape[2] != n_k:
                invalid_attn_mask = F.pad(invalid_attn_mask, (0, n_k - ori_n_k, 0, n_k - ori_n_k, 0, 0), value=0)
            invalid_attn_mask = invalid_attn_mask.unsqueeze(1)
        elif invalid_attn_mask.shape[2] != n_k or invalid_attn_mask.shape[3] != n_k:
            # pad 3rd and 4th dim
            invalid_attn_mask = F.pad(
                invalid_attn_mask,
                (0, n_k - ori_n_k, 0, n_k - ori_n_k, 0, 0, 0, 0),
                value=0,
            )
        masked_qk_attn = masked_qk_attn * invalid_attn_mask.type(masked_qk_attn.dtype)

    if quant_mode == 0:
        masked_qk_attn = masked_qk_attn.to(torch.float8_e4m3fn).reshape(-1, n_k, n_k)
        padded_v = padded_v.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2).reshape(-1, n_k, linear_dim)

        attn_output = torch._scaled_mm(masked_qk_attn[0], padded_v[0], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)
        for i in range(1, masked_qk_attn.size(0)):
            attn_output = torch.cat((attn_output, torch._scaled_mm(masked_qk_attn[i], padded_v[i], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)), dim=0)
        attn_output = attn_output.view(B, num_heads, n_k, linear_dim).permute(0, 2, 1, 3)
    elif quant_mode == 1:
        attn_output = P_blockwise_Vt_gemm_fp8(masked_qk_attn, padded_v, v_descale, cu_seqlens_v_descale, 16, bn).permute(0, 2, 1, 3)
    elif quant_mode == 2:
        start_ids = torch.tensor([k_offsets[i+1] - k_offsets[i] - q_offsets[i+1] + q_offsets[i] for i in range(B)], dtype=torch.int32) if is_delta_q else None
        attn_output = P_blockwise_Vt_gemm_fp8(masked_qk_attn, padded_v, v_descale, cu_seqlens_block_descale_k, 16, bn, False, start_ids, mode=2).permute(0, 2, 1, 3)
    elif quant_mode == 3 or quant_mode == 4 or quant_mode == 5:
        attn_output = P_blockwise_Vt_gemm_fp8(masked_qk_attn, padded_v, v_descale, None, 16, bn, False, None, mode=quant_mode).permute(0, 2, 1, 3)

    attn_output = attn_output.reshape(B, n_k, num_heads * linear_dim)[:, :ori_n_k, :]

    if is_delta_q:
        attn_output = unpad_input_delta_q(attn_output, q_offsets, k_offsets, B, n_k)
    else:
        attn_output = unpad_input(attn_output, q_offsets)
    attn_output = attn_output.reshape(-1, num_heads, linear_dim) / ori_n_q

    return attn_output.to(dtype_out)

def _bwd_reference_fp8(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    rab: Optional[torch.Tensor],
    invalid_attn_mask: Optional[torch.Tensor],
    alpha: float,
    quant_mode: int,
    is_delta_q: bool,
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    B: int = q_offsets.size(0) - 1
    n_q: int = seqlen_q
    n_k: int = seqlen_k
    ori_n_q: int = n_q
    ori_n_k: int = n_k
    n_q = 16 * math.ceil(seqlen_q / 16)
    n_k = 16 * math.ceil(seqlen_k / 16)
    L: int = q.size(0)
    dtype_og = do.dtype
    bm, bn = get_bm_and_bn_block_size_bwd()

    if quant_mode == 0:
        q_fp8 = q.to(torch.float8_e4m3fn)
        k_fp8 = k.to(torch.float8_e4m3fn)
        v_fp8 = v.to(torch.float8_e4m3fn)
        do_fp8 = do.to(torch.float8_e4m3fn)
    elif quant_mode == 1:
        q_fp8, q_descale, qt_fp8, qt_descale, cu_seqlens_qt_descale = quantize_for_two_directions(q, q_offsets, torch.float8_e4m3fn)
        k_fp8, k_descale, kt_fp8, kt_descale, cu_seqlens_kt_descale = quantize_for_two_directions(k, k_offsets, torch.float8_e4m3fn)
        v_fp8, v_descale, _, _, _ = quantize_for_two_directions(v, k_offsets, torch.float8_e4m3fn)
        do_fp8, do_descale, dot_fp8, dot_descale, cu_seqlens_dot_descale = quantize_for_two_directions(do, q_offsets, torch.float8_e4m3fn)
        padded_qt = pad_input(qt_fp8, q_offsets, B, n_q)
        padded_kt = pad_input(kt_fp8, k_offsets, B, n_k)
        padded_dot = pad_input(dot_fp8, q_offsets, B, n_q)
        padded_qt = padded_qt.view(B, n_q, num_heads, attention_dim).permute(0, 2, 1, 3).contiguous()
        padded_kt = padded_kt.view(B, n_k, num_heads, attention_dim).permute(0, 2, 1, 3).contiguous()
        padded_dot = padded_dot.view(B, n_q, num_heads, linear_dim).permute(0, 2, 1, 3).contiguous()
    elif quant_mode == 2:
        q_fp8, q_descale, cu_seqlens_block_descale_q = quantize_for_block_scale(q, q_offsets, block_size=bm, fp8_type=torch.float8_e4m3fn)
        k_fp8, k_descale, cu_seqlens_block_descale_k = quantize_for_block_scale(k, k_offsets, block_size=bn, fp8_type=torch.float8_e4m3fn)
        v_fp8, v_descale, cu_seqlens_block_descale_k = quantize_for_block_scale(v, k_offsets, block_size=bn, fp8_type=torch.float8_e4m3fn)
        do_fp8, do_descale, cu_seqlens_block_descale_q = quantize_for_block_scale(do, q_offsets, block_size=bm, fp8_type=torch.float8_e4m3fn) # same with q, so using bm as block size
        q_descale = q_descale.transpose(1, 0)
        k_descale = k_descale.transpose(1, 0)
        v_descale = v_descale.transpose(1, 0)
        do_descale = do_descale.transpose(1, 0)
    elif quant_mode == 3 or quant_mode == 4 or quant_mode == 5:
        q_fp8, q_descale = quantize_for_head_batch_tensor(q, q_offsets, quant_mode=quant_mode)
        k_fp8, k_descale = quantize_for_head_batch_tensor(k, k_offsets, quant_mode=quant_mode)
        v_fp8, v_descale = quantize_for_head_batch_tensor(v, k_offsets, quant_mode=quant_mode)
        do_fp8, do_descale = quantize_for_head_batch_tensor(do, q_offsets, quant_mode=quant_mode)

    if is_delta_q:
        padded_q = pad_input_delta_q(q_fp8, q_offsets, k_offsets, B, n_k)
        padded_do = pad_input_delta_q(do_fp8, q_offsets, k_offsets, B, n_k)
    else:
        padded_q = pad_input(q_fp8, q_offsets, B, n_q)
        padded_do = pad_input(do_fp8, q_offsets, B, n_q)
    padded_k = pad_input(k_fp8, k_offsets, B, n_k)
    padded_v = pad_input(v_fp8, k_offsets, B, n_k)

    padded_q = padded_q.view(B, n_k, num_heads, attention_dim).permute(0, 2, 1, 3).contiguous()
    padded_k = padded_k.view(B, n_k, num_heads, attention_dim).permute(0, 2, 1, 3).contiguous()
    padded_v = padded_v.view(B, n_k, num_heads, linear_dim).permute(0, 2, 1, 3).contiguous()
    padded_do = padded_do.view(B, n_k, num_heads, linear_dim).permute(0, 2, 1, 3).contiguous()

    def dsilu(dy, x):
        dy = dy.to(torch.float32)
        x = x.to(torch.float32)
        sigmoid = F.sigmoid(x)
        return dy * sigmoid * (1 + x * (1 - sigmoid))
    padded_k_for_dQ = padded_k
    padded_q_for_dK = padded_q
    padded_q = padded_q.reshape(-1, n_k, attention_dim)
    padded_k = padded_k.reshape(-1, n_k, attention_dim).permute(0, 2, 1)
    descale_one = torch.tensor([1.0], dtype=torch.float32, device='cuda')
    # only support MK @ KN
    qk_attn = torch._scaled_mm(padded_q[0], padded_k[0], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)
    for i in range(1, padded_q.size(0)):
        qk_attn = torch.cat((qk_attn, torch._scaled_mm(padded_q[i], padded_k[i], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)), dim=0)
    qk_attn = qk_attn.view(B, num_heads, n_k, n_k)

    if quant_mode == 1:
        for bs in range(B):
            actual_seqlen_q = q_offsets[bs+1] - q_offsets[bs]
            actual_seqlen_k = k_offsets[bs+1] - k_offsets[bs]
            qk_attn[bs, :, actual_seqlen_k - actual_seqlen_q : actual_seqlen_k, :] = qk_attn[bs, :, actual_seqlen_k - actual_seqlen_q : actual_seqlen_k, :] * q_descale[:, q_offsets[bs] : q_offsets[bs+1]].unsqueeze(-1)
            qk_attn[bs, :, :, : actual_seqlen_k] = qk_attn[bs, :, :, : actual_seqlen_k] * k_descale[:, k_offsets[bs] : k_offsets[bs+1]].unsqueeze(-2)
    elif quant_mode == 2:
        qk_attn = AB_blockscale_gemm_fp8(
            ab_attn=qk_attn,
            a_offsets=q_offsets,
            b_offsets=k_offsets,
            a_descale=q_descale,
            b_descale=k_descale,
            cu_seqlens_block_descale_a=cu_seqlens_block_descale_q,
            cu_seqlens_block_descale_b=cu_seqlens_block_descale_k,
            B=B,
            bm=bm,
            bn=bn,
        )
    elif quant_mode == 3:
        qk_attn = qk_attn * q_descale.unsqueeze(-1).unsqueeze(-1) * k_descale.unsqueeze(-1).unsqueeze(-1)
    elif quant_mode == 4:
        qk_attn = qk_attn * q_descale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k_descale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    elif quant_mode == 5:
        qk_attn = qk_attn * q_descale * k_descale

    if rab is not None:
        padding = (0, qk_attn.shape[-1]-rab.shape[-1], 0, qk_attn.shape[-2]-rab.shape[-2])
        rab = F.pad(rab, padding, value=0)
        qk_attn = qk_attn + rab
    qk_attn = qk_attn * alpha
    qk_attn_silu = F.silu(qk_attn)
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            if invalid_attn_mask.shape[0] != n_k or invalid_attn_mask.shape[1] != n_k:
                invalid_attn_mask = F.pad(invalid_attn_mask, (0, n_k - ori_n_k, 0, n_k - ori_n_k), value=0)
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        elif invalid_attn_mask.ndim == 3:
            if invalid_attn_mask.shape[1] != n_k or invalid_attn_mask.shape[2] != n_k:
                invalid_attn_mask = F.pad(invalid_attn_mask, (0, n_k - ori_n_k, 0, n_k - ori_n_k, 0, 0), value=0)
            invalid_attn_mask = invalid_attn_mask.unsqueeze(1)
        elif invalid_attn_mask.shape[2] != n_k or invalid_attn_mask.shape[3] != n_k:
            # pad 3rd and 4th dim
            invalid_attn_mask = F.pad(invalid_attn_mask, (0, n_k - ori_n_k, 0, n_k - ori_n_k, 0, 0, 0, 0), value=0)
        masked_qk_attn = qk_attn_silu * invalid_attn_mask.type(qk_attn_silu.dtype)
    else:
        masked_qk_attn = qk_attn_silu

    if quant_mode == 0:
        masked_qk_attn = masked_qk_attn.to(torch.float8_e4m3fn).reshape(-1, n_k, n_k).permute(0, 2, 1).contiguous()
        padded_do = padded_do.permute(0, 1, 3, 2).reshape(-1, linear_dim, n_k).contiguous().permute(0, 2, 1)
        dv = torch._scaled_mm(masked_qk_attn[0], padded_do[0], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)
        for i in range(1, masked_qk_attn.size(0)):
            dv = torch.cat((dv, torch._scaled_mm(masked_qk_attn[i], padded_do[i], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)), dim=0)
        dv = dv.view(B, num_heads, n_k, linear_dim).permute(0, 2, 1, 3)
    elif quant_mode == 1:
        start_ids = torch.tensor([k_offsets[i+1] - k_offsets[i] - q_offsets[i+1] + q_offsets[i] for i in range(B)], dtype=torch.int32) if is_delta_q else None
        dv = P_blockwise_Vt_gemm_fp8(masked_qk_attn.permute(0, 1, 3, 2), padded_dot, dot_descale, cu_seqlens_dot_descale, bm, bn, True, start_ids).permute(0, 2, 1, 3)
    elif quant_mode == 2:
        start_ids = torch.tensor([k_offsets[i+1] - k_offsets[i] - q_offsets[i+1] + q_offsets[i] for i in range(B)], dtype=torch.int32) if is_delta_q else None
        dv = P_blockwise_Vt_gemm_fp8(masked_qk_attn.permute(0, 1, 3, 2), padded_do, do_descale, cu_seqlens_block_descale_q, bm, bn, True, start_ids=start_ids, mode=2, is_qdo_offset=True).permute(0, 2, 1, 3)
    elif quant_mode == 3 or quant_mode == 4 or quant_mode == 5:
        start_ids = torch.tensor([k_offsets[i+1] - k_offsets[i] - q_offsets[i+1] + q_offsets[i] for i in range(B)], dtype=torch.int32) if is_delta_q else None
        dv = P_blockwise_Vt_gemm_fp8(masked_qk_attn.permute(0, 1, 3, 2), padded_do, do_descale, None, bm, bn, True, None, mode=quant_mode, is_qdo_offset=True).permute(0, 2, 1, 3)

    dv = unpad_input(dv, k_offsets) / ori_n_q

    if quant_mode == 0:
        padded_do = padded_do.contiguous()
    else:
        padded_do = padded_do.reshape(-1, n_k, linear_dim)
    padded_v = padded_v.reshape(-1, n_k, linear_dim).permute(0, 2, 1)
    dp = torch._scaled_mm(padded_do[0], padded_v[0], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)
    for i in range(1, padded_do.size(0)):
        dp = torch.cat((dp, torch._scaled_mm(padded_do[i], padded_v[i], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)), dim=0)
    dp = dp.view(B, num_heads, n_k, n_k)

    if quant_mode == 1:
        for bs in range(B):
          actual_seqlen_q = q_offsets[bs+1] - q_offsets[bs]
          actual_seqlen_k = k_offsets[bs+1] - k_offsets[bs]
          dp[bs, :, actual_seqlen_k - actual_seqlen_q : actual_seqlen_k, :] = dp[bs, :, actual_seqlen_k - actual_seqlen_q : actual_seqlen_k, :] * do_descale[:, q_offsets[bs] : q_offsets[bs+1]].unsqueeze(-1)
          dp[bs, :, :, : actual_seqlen_k] = dp[bs, :, :, : actual_seqlen_k] * v_descale[:, k_offsets[bs] : k_offsets[bs+1]].unsqueeze(-2)
    elif quant_mode == 2:
        dp = AB_blockscale_gemm_fp8(
            ab_attn=dp,
            a_offsets=q_offsets,
            b_offsets=k_offsets,
            a_descale=do_descale,
            b_descale=v_descale,
            cu_seqlens_block_descale_a=cu_seqlens_block_descale_q,
            cu_seqlens_block_descale_b=cu_seqlens_block_descale_k,
            B=B,
            bm=bm,
            bn=bn,
        )
    elif quant_mode == 3:
        dp = dp * do_descale.unsqueeze(-1).unsqueeze(-1) * v_descale.unsqueeze(-1).unsqueeze(-1)
    elif quant_mode == 4:
        dp = dp * do_descale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * v_descale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    elif quant_mode == 5:
        dp = dp * do_descale * v_descale

    if invalid_attn_mask is not None:
        dp = dp * invalid_attn_mask.type(dp.dtype)

    drab = dp / ori_n_q * alpha
    drab = dsilu(drab, qk_attn)
    drab = drab[:, :, :seqlen_k, :seqlen_k]
    if rab is not None and rab.shape[1] == 1:
        drab = drab.sum(dim=1, keepdim=True)

    dp = dsilu(dp, qk_attn)

    if quant_mode == 0:
        dp = dp.to(torch.float8_e4m3fn).contiguous().reshape(-1, n_k, n_k)
        padded_k = padded_k.contiguous().permute(0, 2, 1)
        dq = torch._scaled_mm(dp[0], padded_k[0], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)
        for i in range(1, dp.size(0)):
            dq = torch.cat((dq, torch._scaled_mm(dp[i], padded_k[i], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)), dim=0)
        dq = dq.view(B, num_heads, n_k, attention_dim).permute(0, 2, 1, 3)
    elif quant_mode == 1:
        start_ids = torch.tensor([k_offsets[i+1] - k_offsets[i] - q_offsets[i+1] + q_offsets[i] for i in range(B)], dtype=torch.int32) if is_delta_q else None
        dq = P_blockwise_Vt_gemm_fp8(dp, padded_kt, kt_descale, cu_seqlens_kt_descale, bm, bn, start_ids=start_ids).permute(0, 2, 1, 3)
    elif quant_mode == 2:
        start_ids = torch.tensor([k_offsets[i+1] - k_offsets[i] - q_offsets[i+1] + q_offsets[i] for i in range(B)], dtype=torch.int32) if is_delta_q else None
        dq = P_blockwise_Vt_gemm_fp8(dp, padded_k_for_dQ, k_descale, cu_seqlens_block_descale_k, bm, bn, False, start_ids, mode=2).permute(0, 2, 1, 3)
    elif quant_mode == 3 or quant_mode == 4 or quant_mode == 5:
        start_ids = torch.tensor([k_offsets[i+1] - k_offsets[i] - q_offsets[i+1] + q_offsets[i] for i in range(B)], dtype=torch.int32) if is_delta_q else None
        dq = P_blockwise_Vt_gemm_fp8(dp, padded_k_for_dQ, k_descale, None, bm, bn, False, start_ids, mode=quant_mode).permute(0, 2, 1, 3)

    if is_delta_q:
        dq = unpad_input_delta_q(dq, q_offsets, k_offsets, B, n_k) / ori_n_q * alpha
    else:
        dq = unpad_input(dq, q_offsets) / ori_n_q * alpha

    if quant_mode == 0:
        dp = dp.permute(0, 2, 1).contiguous()
        padded_q = padded_q.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        dk = torch._scaled_mm(dp[0], padded_q[0], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)
        for i in range(1, dp.size(0)):
            dk = torch.cat((dk, torch._scaled_mm(dp[i], padded_q[i], out_dtype=torch.float, scale_a=descale_one, scale_b=descale_one)), dim=0)
        dk = dk.view(B, num_heads, n_k, attention_dim).permute(0, 2, 1, 3)
    elif quant_mode == 1:
        start_ids = torch.tensor([k_offsets[i+1] - k_offsets[i] - q_offsets[i+1] + q_offsets[i] for i in range(B)], dtype=torch.int32) if is_delta_q else None
        dk = P_blockwise_Vt_gemm_fp8(dp.permute(0, 1, 3, 2).contiguous(), padded_qt, qt_descale, cu_seqlens_qt_descale, bm, bn, True, start_ids).permute(0, 2, 1, 3)
    elif quant_mode == 2:
        start_ids = torch.tensor([k_offsets[i+1] - k_offsets[i] - q_offsets[i+1] + q_offsets[i] for i in range(B)], dtype=torch.int32) if is_delta_q else None
        dk = P_blockwise_Vt_gemm_fp8(dp.permute(0, 1, 3, 2).contiguous(), padded_q_for_dK, q_descale, cu_seqlens_block_descale_q, bm, bn, True, start_ids=start_ids, mode=2, is_qdo_offset=True).permute(0, 2, 1, 3)
    elif quant_mode == 3 or quant_mode == 4 or quant_mode == 5:
        start_ids = torch.tensor([k_offsets[i+1] - k_offsets[i] - q_offsets[i+1] + q_offsets[i] for i in range(B)], dtype=torch.int32) if is_delta_q else None
        dk = P_blockwise_Vt_gemm_fp8(dp.permute(0, 1, 3, 2).contiguous(), padded_q_for_dK, q_descale, None, bm, bn, True, start_ids, mode=quant_mode, is_qdo_offset=True).permute(0, 2, 1, 3)
    dk = unpad_input(dk, k_offsets) / ori_n_q * alpha

    return (
        dq.view(-1, num_heads, attention_dim),
        dk.view(-1, num_heads, attention_dim),
        dv.view(-1, num_heads, linear_dim),
        drab
    )

@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    "Skip when no Hopper GPU is available. This test is only for Hopper GPU.",
)
class HSTU8Test(unittest.TestCase):
    """Test HSTU attention with float8_e4m3 inputs."""

    @given(
        batch_size=st.sampled_from([32]),
        heads=st.sampled_from([2]),
        seq_len_params=st.sampled_from(
            [
                (32, 32),
                (99, 99),
                (272, 272),
                (848, 848),
                (8, 99),
                (51, 256),
                (531, 777),
                (160, 800),
            ]
        ),
        max_context_len=st.sampled_from([0, 99, 160, 333]),
        target_params=st.sampled_from(
            [
                (0, (-1, -1), 1, False),
                (0, (-1, -1), 1, True),
                (0, (111, 11), 1, False),
                (0, (111, 222), 1, False),
                (0, (-1, 0), 1, False),
                (32, (-1, 0), 1, False),
                (257, (-1, 0), 1, False),
                (1024, (-1, 0), 1, False),
                (111, (-1, 0), 11, False),
                (1111, (-1, 0), 222, False),
            ]
        ),
        attn_hidden_dims=st.sampled_from(
            [
                (64, 64),
                (128, 128),
                (256, 256),
            ]
        ),
        alpha=st.sampled_from([1.0, 0.1]),
        rab_params=st.sampled_from(
            [
                (False, False, None),
                (True, False, None),  # None means heads_rab=heads
                (True, True, None),
                (True, False, 1),
                (True, True, 1),
            ]
        ),
        dtype=st.just(torch.float8_e4m3fn),
        quant_mode_full_batch=st.sampled_from([
            (0, True), (0, False),
            (1, True), (2, True),
            (3, True), (4, True),
            (5, True),
        ]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_hstu_attn_fp8(
        self,
        batch_size: int,
        heads: int,
        seq_len_params: Tuple[int, int],
        max_context_len: int,
        target_params: Tuple[int, Tuple[int, int], int, bool],
        attn_hidden_dims: Tuple[int, int],
        alpha: float,
        rab_params: Tuple[bool, bool, Optional[int]],
        dtype: torch.dtype,
        quant_mode_full_batch: Tuple[int, bool],
    ) -> None:
        max_seq_len_q, max_seq_len_k = seq_len_params
        max_target_len, window_size, target_group_size, is_arbitrary = target_params
        attn_dim, hidden_dim = attn_hidden_dims
        has_rab, has_drab, heads_rab = rab_params
        quant_mode, full_batch = quant_mode_full_batch

        total_q = max_context_len + max_seq_len_q + max_target_len
        total_k = max_context_len + max_seq_len_k + max_target_len
        has_context = max_context_len > 0
        has_target = max_target_len > 0
        is_causal = window_size[0] == -1 and window_size[1] == 0
        is_delta_q = max_seq_len_q < max_seq_len_k
        if quant_mode == 0 and alpha < 0.5:
            logger.info("Skipping test for quant_mode == 0 and alpha < 0.5, might cause dQ accuracy issue")
            return
        if quant_mode > 0 and (total_q % 16 != 0 or total_k % 16 != 0 or full_batch == False):
            logger.info("Skipping test for quant_mode > 0 and (total_q % 16 != 0 or total_k % 16 != 0 or full_batch == False), not supported")
            return
        if is_delta_q and has_target:
            logger.info("Skipping test for is_delta_q and has_target")
            return
        if is_delta_q and has_context:
            logger.info("Skipping test for is_delta_q and has_context")
            return
        if not is_causal and has_context:
            logger.info("Skipping test for not is_causal and has_context")
            return
        if (window_size[0] > 0 or window_size[1] > 0) and has_context:
            logger.info(
                "Skipping test for (window_size[0] > 0 or window_size[1] > 0) and has_context"
            )
            return

        torch.cuda.synchronize()
        L_q, L_k, num_contexts, cu_seqlens_q, cu_seqlens_k, num_targets, _, q, k, v, rab, attn_mask, func = (
            generate_input(
                batch_size=batch_size,
                heads=heads,
                heads_rab=heads_rab,
                max_seq_len_q=max_seq_len_q,
                max_seq_len_k=max_seq_len_k,
                max_context_len=max_context_len,
                max_target_len=max_target_len,
                target_group_size=target_group_size,
                attn_dim=attn_dim,
                hidden_dim=hidden_dim,
                window_size=window_size,
                dtype=dtype,
                full_batch=full_batch,
                has_drab=has_drab,
                is_delta_q=is_delta_q,
                is_arbitrary=is_arbitrary,
            )
        )
        out_ref = _hstu_attention_maybe_from_cache(
            num_heads=heads,
            attention_dim=attn_dim,
            linear_dim=hidden_dim,
            seqlen_q=max_context_len + max_seq_len_q + max_target_len,
            seqlen_k=max_context_len + max_seq_len_k + max_target_len,
            q=q,
            k=k,
            v=v,
            q_offsets=cu_seqlens_q,
            k_offsets=cu_seqlens_k,
            rab=rab if has_rab else None,
            invalid_attn_mask=(
                attn_mask.to(torch.float32) if attn_mask is not None else None
            ),
            alpha=alpha,
            upcast=True,
            is_delta_q=is_delta_q,
        )

        torch_out = _hstu_attention_maybe_from_cache_fp8(
            num_heads=heads,
            attention_dim=attn_dim,
            linear_dim=hidden_dim,
            seqlen_q=max_context_len + max_seq_len_q + max_target_len,
            seqlen_k=max_context_len + max_seq_len_k + max_target_len,
            q=q,
            k=k,
            v=v,
            q_offsets=cu_seqlens_q,
            k_offsets=cu_seqlens_k,
            rab=rab if has_rab else None,
            invalid_attn_mask=(
                attn_mask.to(torch.float32) if attn_mask is not None else None
            ),
            alpha=alpha,
            quant_mode=quant_mode,
            is_delta_q=is_delta_q,
        )

        hstu_out = hstu_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_context_len + max_seq_len_q + max_target_len,
            max_seqlen_k=max_context_len + max_seq_len_k + max_target_len,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=target_group_size,
            window_size=window_size,
            alpha=alpha,
            rab=rab if has_rab else None,
            has_drab=has_drab,
            func=func,
            quant_mode=quant_mode,
        )

        # print(f"Output max diff: {(hstu_out - out_ref).abs().max().item()}")
        # print(f"Pytorch max diff: {(torch_out - out_ref).abs().max().item()}")
        # print(f"Output mean diff: {(hstu_out - out_ref).abs().mean().item()}")
        # print(f"Pytorch mean diff: {(torch_out - out_ref).abs().mean().item()}")

        assert (hstu_out - out_ref).abs().max().item() <= 2 * (torch_out - out_ref).abs().max().item()

        g = torch.rand_like(torch_out)
        if not has_drab:
            (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(
                out_ref, (q, k, v), g, retain_graph=True
            )
            (dq_hstu, dk_hstu, dv_hstu) = torch.autograd.grad(
                hstu_out, (q, k, v), g, retain_graph=True,
            )
        else:
            (dq_ref, dk_ref, dv_ref, drab_ref) = torch.autograd.grad(
                out_ref, (q, k, v, rab), g, retain_graph=True
            )
            (dq_hstu, dk_hstu, dv_hstu, drab_hstu) = torch.autograd.grad(
                hstu_out, (q, k, v, rab), g, retain_graph=True,
            )
        (dq_torch, dk_torch, dv_torch, drab_torch) = _bwd_reference_fp8(
            num_heads=heads,
            attention_dim=attn_dim,
            linear_dim=hidden_dim,
            seqlen_q=max_context_len+max_seq_len_q+max_target_len,
            seqlen_k=max_context_len+max_seq_len_k+max_target_len,
            do=g,
            q=q,
            k=k,
            v=v,
            q_offsets=cu_seqlens_q,
            k_offsets=cu_seqlens_k,
            rab=rab if has_rab else None,
            invalid_attn_mask=attn_mask,
            alpha=alpha,
            quant_mode=quant_mode,
            is_delta_q=is_delta_q,
        )

        # print(f"dV max diff: {(dv_hstu - dv_ref).abs().max().item()}")
        # print(f"dV Pytorch max diff: {(dv_torch - dv_ref).abs().max().item()}")
        # print(f"dK max diff: {(dk_hstu - dk_ref).abs().max().item()}")
        # print(f"dK Pytorch max diff: {(dk_torch - dk_ref).abs().max().item()}")
        # print(f"dQ max diff: {(dq_hstu - dq_ref).abs().max().item()}")
        # print(f"dQ Pytorch max diff: {(dq_torch - dq_ref).abs().max().item()}")
        # if has_drab:
        #     print(f"drab max diff: {(drab_hstu - drab_ref).abs().max().item()}")
        #     print(f"drab Pytorch max diff: {(drab_torch - drab_ref).abs().max().item()}")

        assert (dv_torch - dv_ref).abs().max().item() <= 2 * (
            dv_hstu - dv_ref
        ).abs().max().item()
        assert (dk_torch - dk_ref).abs().max().item() <= 2 * (
            dk_hstu - dk_ref
        ).abs().max().item()
        assert (dq_torch - dq_ref).abs().max().item() <= 2 * (
            dq_hstu - dq_ref
        ).abs().max().item()
        if has_drab:
            assert (drab_torch - drab_ref).abs().max().item() <= 2 * (
                drab_hstu - drab_ref
            ).abs().max().item()

        assert (dv_hstu - dv_ref).abs().max().item() <= 5 * (
            dv_torch - dv_ref
        ).abs().max().item()
        assert (dk_hstu - dk_ref).abs().max().item() <= 5 * (
            dk_torch - dk_ref
        ).abs().max().item()
        assert (dq_hstu - dq_ref).abs().max().item() <= 5 * (
            dq_torch - dq_ref
        ).abs().max().item()
        if has_drab:
            assert (drab_hstu - drab_ref).abs().max().item() <= 5 * (
                drab_torch - drab_ref
            ).abs().max().item()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":

    HSTU8Test().test_hstu_attn_fp8.hypothesis.inner_test(HSTU8Test(),
    1, 1, (32, 32), 0, (0, (-1, -1), 1, False), (64, 64), 1.0, (True, False, None), torch.float8_e4m3fn, (0, True))

    # unittest.main()
