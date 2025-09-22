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
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from fbgemm_gpu.experimental.hstu import hstu_attn_varlen_func

from hypothesis import given, settings, strategies as st, Verbosity

running_on_github: bool = os.getenv("GITHUB_ENV") is not None

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

_MAX_SAMPLES: int = 100


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
    seqlen_c,
    seqlen,
    seqlen_t=0,
    target_group_size=1,
    window_size=(-1, -1),  # -1 means infinite window size
    seq_offsets=None,
    num_contexts=None,
    device=None,
):
    seqlen = seqlen_c + seqlen + seqlen_t
    bs = seq_offsets.size(0) - 1

    mask = torch.zeros((seqlen, seqlen), device=device, dtype=torch.bool)
    if window_size[0] < 0 and window_size[1] == 0:
        # causal mask
        for i in range(seqlen):
            mask[i, : i + 1] = True

        # context mask
        if seqlen_c != 0:
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
            for i in range(bs):
                target_start = (
                    num_contexts[i] + seq_offsets[i + 1] - seq_offsets[i]
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
                    num_contexts[i] + seq_offsets[i + 1] - seq_offsets[i]
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
        window_size_0 = window_size[0] if window_size[0] >= 0 else seqlen
        window_size_1 = window_size[1] if window_size[1] >= 0 else seqlen
        for i in range(seqlen):
            mask[i, max(0, i - window_size_0) : min(seqlen, i + window_size_1 + 1)] = (
                True
            )
    return mask


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
    seq_offsets_c = torch.zeros(
        (batch_size + 1,),
        dtype=torch.int32,
        device=torch.accelerator.current_accelerator(),
    )
    seq_offsets_c[1:] = torch.cumsum(num_contexts, dim=0)

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
    seq_offsets_k = torch.zeros(
        (batch_size + 1,),
        dtype=torch.int32,
        device=torch.accelerator.current_accelerator(),
    )
    seq_offsets_k[1:] = torch.cumsum(lengths_k, dim=0)

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
    seq_offsets_t = torch.zeros(
        (batch_size + 1,),
        dtype=torch.int32,
        device=torch.accelerator.current_accelerator(),
    )
    seq_offsets_t[1:] = torch.cumsum(num_targets, dim=0)

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
        seq_offsets_q = torch.zeros(
            (batch_size + 1,),
            dtype=torch.int32,
            device=torch.accelerator.current_accelerator(),
        )
        seq_offsets_q[1:] = torch.cumsum(lengths_q, dim=0)
    else:
        seq_offsets_q = seq_offsets_k

    # Lengths for whole q, kv
    seq_offsets_q_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets_q_wt = seq_offsets_c + seq_offsets_q + seq_offsets_t
    seq_offsets_k_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets_k_wt = seq_offsets_c + seq_offsets_k + seq_offsets_t

    L_q = int(seq_offsets_q_wt[-1].item())
    L_k = int(seq_offsets_k_wt[-1].item())
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
    ).to(dtype)
    k = (
        torch.empty(
            (L_k, heads, attn_dim), dtype=dtype_init, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    v = (
        torch.empty(
            (L_k, heads, hidden_dim), dtype=dtype_init, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)

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
    if has_drab:
        rab = rab.requires_grad_()
    if window_size[0] == -1 and window_size[1] == -1:
        attn_mask = None
    else:
        attn_mask = (
            construct_mask(
                seqlen_c=max_context_len,
                seqlen=max_seq_len_k,
                seqlen_t=max_target_len,
                target_group_size=target_group_size,
                window_size=window_size,
                num_contexts=num_contexts,
                seq_offsets=seq_offsets_k,
            )
            .cuda()
            .to(torch.float32)
        )

    return (
        L_q,
        L_k,
        num_contexts if has_context else None,
        seq_offsets_q_wt,
        seq_offsets_k_wt,
        num_targets if has_target else None,
        q,
        k,
        v,
        rab,
        attn_mask,
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
    reorder_op: bool = False,
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
        masked_qk_attn = (
            masked_qk_attn * invalid_attn_mask.type(masked_qk_attn.dtype)[:, :, :, :]
        )

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
    attn_output = attn_output.reshape(-1, num_heads * linear_dim)

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
                (32, 32, False),
                (99, 99, False),
                (111, 111, False),
                (256, 256, False),
                (1111, 1111, False),
                (27, 32, True),
                (8, 99, True),
                (51, 256, True),
                (1000, 1111, True),
                (160, 2000, True),
            ]
        ),
        max_context_len=st.sampled_from([0, 11, 99, 160, 333]),
        target_params=st.sampled_from(
            [
                (0, (-1, -1), 1),
                (0, (11, 111), 1),
                (0, (111, 11), 1),
                (0, (111, 222), 1),
                (0, (-1, 0), 1),
                (32, (-1, 0), 1),
                (257, (-1, 0), 1),
                (1024, (-1, 0), 1),
                (111, (-1, 0), 11),
                (1111, (-1, 0), 222),
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
        dtype=st.sampled_from([torch.bfloat16, torch.float16]),
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
        rab_params: tuple[bool, bool, Optional[int]],
        seq_len_params: tuple[int, int, bool],
        target_params: tuple[int, tuple[int, int], int],
        dtype: torch.dtype,
        full_batch: bool,
    ) -> None:
        max_seq_len_q, max_seq_len_k, is_delta_q = seq_len_params
        max_target_len, window_size, target_group_size = target_params
        attn_dim, hidden_dim = attn_hidden_dims
        has_rab, has_drab, heads_rab = rab_params

        has_context = max_context_len > 0
        has_target = max_target_len > 0
        is_causal = window_size[0] == -1 and window_size[1] == 0
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
            seq_offsets_q,
            seq_offsets_k,
            num_targets,
            q,
            k,
            v,
            rab,
            attn_mask,
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
            q_offsets=seq_offsets_q,
            k_offsets=seq_offsets_k,
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
            q_offsets=seq_offsets_q,
            k_offsets=seq_offsets_k,
            rab=rab if has_rab else None,
            invalid_attn_mask=(
                attn_mask.to(torch.float32) if attn_mask is not None else None
            ),
            alpha=alpha,
            upcast=False,
            reorder_op=True,
            is_delta_q=is_delta_q,
        )
        hstu_out = hstu_attn_varlen_func(
            q=q.to(dtype),
            k=k.to(dtype),
            v=v.to(dtype),
            seq_offsets_q=seq_offsets_q,
            seq_offsets_k=seq_offsets_k,
            max_seqlen_q=max_context_len + max_seq_len_q + max_target_len,
            max_seqlen_k=max_context_len + max_seq_len_k + max_target_len,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=target_group_size,
            window_size=window_size,
            alpha=alpha,
            rab=rab if has_rab else None,
            has_drab=has_drab,
            is_delta_q=is_delta_q,
        )

        assert (hstu_out.view(L_q, -1) - out_ref).abs().max().item() <= 2 * (
            torch_out - out_ref
        ).abs().max().item()

        g = torch.rand_like(torch_out)
        if not has_drab:
            (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(
                out_ref, (q, k, v), g, retain_graph=True
            )
            (dq_torch, dk_torch, dv_torch) = torch.autograd.grad(
                torch_out, (q, k, v), g, retain_graph=True
            )
            (dq_hstu, dk_hstu, dv_hstu) = torch.autograd.grad(
                hstu_out,
                (q, k, v),
                g.view(-1, heads, hidden_dim),
                retain_graph=True,
            )
        else:
            (dq_ref, dk_ref, dv_ref, drab_ref) = torch.autograd.grad(
                out_ref, (q, k, v, rab), g, retain_graph=True
            )
            (dq_torch, dk_torch, dv_torch, drab_torch) = torch.autograd.grad(
                torch_out, (q, k, v, rab), g, retain_graph=True
            )
            (dq_hstu, dk_hstu, dv_hstu, drab_hstu) = torch.autograd.grad(
                hstu_out,
                (q, k, v, rab),
                g.view(-1, heads, hidden_dim),
                retain_graph=True,
            )

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
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
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
    padded_q = pad_input(q, q_offsets, B, n_q)
    padded_k = pad_input(k, k_offsets, B, n_k)
    padded_v = pad_input(v, k_offsets, B, n_k)

    padded_q = padded_q.view(B, n_q, num_heads, attention_dim)
    padded_k = padded_k.view(B, n_k, num_heads, attention_dim)
    padded_v = padded_v.view(B, n_k, num_heads, linear_dim)

    padded_q = padded_q.permute(0, 2, 1, 3).reshape(-1, n_q, attention_dim)
    padded_k = (
        padded_k.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1, n_k, attention_dim)
        .permute(0, 2, 1)
    )
    # only support MK @ KN
    qk_attn = torch._scaled_mm(
        padded_q[0],
        padded_k[0],
        out_dtype=torch.float,
        scale_a=descale_q,  # pyre-ignore[6]
        scale_b=descale_k,  # pyre-ignore[6]
    )
    for i in range(1, padded_q.size(0)):
        qk_attn = torch.cat(
            (
                qk_attn,
                torch._scaled_mm(
                    padded_q[i],
                    padded_k[i],
                    out_dtype=torch.float,
                    scale_a=descale_q,  # pyre-ignore[6]
                    scale_b=descale_k,  # pyre-ignore[6]
                ),
            ),
            dim=0,
        )
    qk_attn = qk_attn.view(B, num_heads, n_q, n_k)

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
    masked_qk_attn = masked_qk_attn / ori_n_q
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            if invalid_attn_mask.shape[0] != n_q or invalid_attn_mask.shape[1] != n_k:
                invalid_attn_mask = F.pad(
                    invalid_attn_mask, (0, n_q - ori_n_q, 0, n_k - ori_n_k), value=1
                )
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        elif invalid_attn_mask.shape[2] != n_q or invalid_attn_mask.shape[3] != n_k:
            # pad 3rd and 4th dim
            invalid_attn_mask = F.pad(
                invalid_attn_mask,
                (0, 0, 0, 0, 0, n_q - ori_n_q, 0, n_k - ori_n_k),
                value=1,
            )
        masked_qk_attn = masked_qk_attn * invalid_attn_mask.type(masked_qk_attn.dtype)

    descale_a = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    masked_qk_attn = masked_qk_attn.to(torch.float8_e4m3fn).reshape(-1, n_q, n_k)
    padded_v = (
        padded_v.permute(0, 2, 3, 1)
        .contiguous()
        .permute(0, 1, 3, 2)
        .reshape(-1, n_k, linear_dim)
    )
    attn_output = torch._scaled_mm(
        masked_qk_attn[0],
        padded_v[0],
        out_dtype=torch.float,
        scale_a=descale_a,
        scale_b=descale_v,  # pyre-ignore[6]
    )
    for i in range(1, masked_qk_attn.size(0)):
        attn_output = torch.cat(
            (
                attn_output,
                torch._scaled_mm(
                    masked_qk_attn[i],
                    padded_v[i],
                    out_dtype=torch.float,
                    scale_a=descale_a,
                    scale_b=descale_v,  # pyre-ignore[6]
                ),
            ),
            dim=0,
        )
    attn_output = attn_output.view(B, num_heads, n_q, linear_dim).permute(0, 2, 1, 3)

    attn_output = attn_output.reshape(B, n_q, num_heads * linear_dim)[:, :ori_n_q, :]
    attn_output = unpad_input(attn_output, q_offsets)
    attn_output = attn_output.reshape(-1, num_heads * linear_dim)

    return attn_output.to(dtype_out)


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
                (256, 256),
                (1024, 1024),
                (1111, 1111),
            ]
        ),
        window_size=st.sampled_from(
            [
                (-1, -1),
                (11, 111),
                (111, 222),
                (-1, 0),
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
                (True, False, None),  # None is heads_rab=heads
                (True, False, 1),
            ]
        ),
        full_batch=st.sampled_from([True, False]),
        dtype=st.just(torch.float8_e4m3fn),
        max_target_len=st.just(0),  # Fixed for fp8
        is_delta_q=st.just(False),  # Fixed for fp8
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_hstu_attn_fp8(
        self,
        batch_size: int,
        heads: int,
        seq_len_params: tuple[int, int],
        window_size: tuple[int, int],
        attn_hidden_dims: tuple[int, int],
        alpha: float,
        rab_params: tuple[bool, bool, Optional[int]],
        full_batch: bool,
        dtype: torch.dtype,
        max_target_len: int,
        is_delta_q: bool,
    ) -> None:
        max_seq_len_q, max_seq_len_k = seq_len_params
        attn_dim, hidden_dim = attn_hidden_dims
        has_rab, has_drab, heads_rab = rab_params

        torch.cuda.synchronize()
        L_q, L_k, _, seq_offsets_q, seq_offsets_k, _, q, k, v, rab, attn_mask = (
            generate_input(
                batch_size=batch_size,
                heads=heads,
                heads_rab=heads_rab,
                max_seq_len_q=max_seq_len_q,
                max_seq_len_k=max_seq_len_k,
                max_context_len=0,
                max_target_len=max_target_len,
                target_group_size=1,
                attn_dim=attn_dim,
                hidden_dim=hidden_dim,
                window_size=window_size,
                dtype=dtype,
                full_batch=full_batch,
                has_drab=has_drab,
                is_delta_q=is_delta_q,
            )
        )
        out_ref = _hstu_attention_maybe_from_cache(
            num_heads=heads,
            attention_dim=attn_dim,
            linear_dim=hidden_dim,
            seqlen_q=max_seq_len_q,
            seqlen_k=max_seq_len_k,
            q=q.view(L_q, -1).float(),
            k=k.view(L_k, -1).float(),
            v=v.view(L_k, -1).float(),
            q_offsets=seq_offsets_q,
            k_offsets=seq_offsets_k,
            rab=rab if has_rab else None,
            invalid_attn_mask=(
                attn_mask.to(torch.float32) if attn_mask is not None else None
            ),
            alpha=alpha,
            is_delta_q=is_delta_q,
        )

        torch_out = _hstu_attention_maybe_from_cache_fp8(
            num_heads=heads,
            attention_dim=attn_dim,
            linear_dim=hidden_dim,
            seqlen_q=max_seq_len_q,
            seqlen_k=max_seq_len_k,
            q=q.view(L_q, -1),
            k=k.view(L_k, -1),
            v=v.view(L_k, -1),
            q_offsets=seq_offsets_q,
            k_offsets=seq_offsets_k,
            rab=rab if has_rab else None,
            invalid_attn_mask=(
                attn_mask.to(torch.float32) if attn_mask is not None else None
            ),
            alpha=alpha,
            descale_q=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
            descale_k=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
            descale_v=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        )

        hstu_out = hstu_attn_varlen_func(
            q=q.to(dtype),
            k=k.to(dtype),
            v=v.to(dtype),
            seq_offsets_q=seq_offsets_q,
            seq_offsets_k=seq_offsets_k,
            max_seqlen_q=max_seq_len_q,
            max_seqlen_k=max_seq_len_k,
            num_contexts=None,  # pyre-ignore[6]
            num_targets=None,  # pyre-ignore[6]
            target_group_size=1,
            window_size=window_size,
            alpha=alpha,
            rab=rab if has_rab else None,
            has_drab=has_drab,
            is_delta_q=is_delta_q,
            descale_q=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
            descale_k=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
            descale_v=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        )

        assert (hstu_out.view(L_q, -1) - out_ref).abs().max().item() <= 2 * (
            torch_out - out_ref
        ).abs().max().item()


if __name__ == "__main__":

    unittest.main()
