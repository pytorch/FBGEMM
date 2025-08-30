# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha  # noqa

import torch
from torch.library import opcheck

from .test_utils import generate_qkv, generate_random_padding_mask


def get_varlen_args(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    causal: bool,
    fwd_only: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    float,
    bool,
    Optional[torch.Tensor],
]:
    device = torch.accelerator.current_accelerator()

    q_ref = torch.randn(
        batch_size, seqlen_q, q_heads, head_dim, device=device, dtype=dtype
    ).requires_grad_()
    k_ref = torch.randn(
        batch_size, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype
    ).requires_grad_()
    v_ref = torch.randn(
        batch_size, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype
    ).requires_grad_()

    q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
    # It fails with zero_lengths=True
    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch_size, device, mode="random", zero_lengths=False
    )
    key_padding_mask = generate_random_padding_mask(
        seqlen_k, batch_size, device, mode="random", zero_lengths=False
    )

    # Always have seqlen_k >= seqlen_q
    key_padding_mask[:, :seqlen_q] |= query_padding_mask

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        _,
        _,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        kvpacked=False,
        query_unused_mask=None,
        key_unused_mask=None,
    )

    return (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        1.0,
        causal,
        None,
    )


def get_sample_inputs():
    return [
        (
            torch.randn(
                16,
                128,
                4,
                128,
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
                requires_grad=True,
            ),
            torch.randn(
                16,
                128,
                4,
                128,
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
                requires_grad=True,
            ),
            torch.randn(
                16,
                128,
                4,
                128,
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
                requires_grad=True,
            ),  # v
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # max_seq_len_q
            None,  # max_seq_len_k
            1.0,  # softmax_scale
            False,  # causal
            None,  # seqlen_kv
        ),
        get_varlen_args(
            batch_size=512,
            seqlen_q=512,
            seqlen_k=512,
            q_heads=4,
            kv_heads=4,
            head_dim=128,
            dtype=torch.bfloat16,
            causal=False,
        ),
    ]


def get_sample_inputs_for_backward():
    return [
        (
            torch.randn(
                16,
                128,
                4,
                128,
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
            ),
            torch.randn(
                16,
                128,
                4,
                128,
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
            ),
            torch.randn(
                16,
                128,
                4,
                128,
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
            ),
            torch.randn(
                16,
                128,
                4,
                128,
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
            ),
            torch.randn(
                16,
                128,
                4,
                128,
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
            ),
            torch.randn(
                16,
                4,
                128,
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
            ),
            # v
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # max_seq_len_q
            None,  # max_seq_len_k
            False,  # causal
        ),
    ]


# expected test_utils to be subset of ('test_schema', 'test_autograd_registration', 'test_faketensor', 'test_aot_dispatch_static', 'test_aot_dispatch_dynamic')
sample_inputs = get_sample_inputs()
for i in range(len(sample_inputs)):
    opcheck(torch.ops.blackwell_fmha.fmha_fwd.default, sample_inputs[i])

sample_inputs_bwd = get_sample_inputs_for_backward()
for i in range(len(sample_inputs_bwd)):
    opcheck(
        torch.ops.blackwell_fmha.fmha_bwd.default,
        sample_inputs_bwd[i],
    )
