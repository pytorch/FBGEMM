#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Tuple

import hypothesis.strategies as st
import numpy as np
import torch

from hypothesis import given, settings, Verbosity

VERBOSITY: Verbosity = Verbosity.verbose


def quant_int4_dequant_bf16(
    in_tensor: torch.Tensor, num_groups: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A util function for quantizing a tensor from from a float type (including
    FP32, FP16, BF16) to INT4 and then dequantize the INT4 result to BF16
    (i.e., fake quantization)
    """
    in_shape = in_tensor.shape
    in_tensor = in_tensor.reshape(
        *in_shape[:-1], num_groups, in_shape[-1] // num_groups
    )

    # Find max and min for each group
    max_vals = torch.max(in_tensor, dim=-1, keepdim=True).values
    min_vals = torch.min(in_tensor, dim=-1, keepdim=True).values

    # Compute scale and shift
    scale: torch.Tensor = (max_vals - min_vals) / 15
    shift = torch.min(in_tensor, dim=-1, keepdim=True).values
    scale = scale.to(torch.float16)
    shift = shift.to(torch.float16)
    shift_expand = shift.expand(in_tensor.shape)
    scale_expand = scale.expand(in_tensor.shape)

    # Scale and shift
    in_bytes = ((in_tensor - shift_expand) / scale_expand).to(torch.uint8)

    # Get only 4 bits
    in_int4 = in_bytes & 0xF

    # Pack int4 in uint8
    in_int4_packed = in_int4[..., ::2] + (in_int4[..., 1::2] << 4)

    # Concat data
    scale_shift = torch.concat(
        [scale.view(torch.uint8), shift.view(torch.uint8)], dim=-1
    )
    in_quant = torch.concat(
        [
            scale_shift.flatten(start_dim=-2),
            in_int4_packed.flatten(start_dim=-2),
        ],
        dim=-1,
    )

    # Dequantize tensor for reference
    in_fp16 = in_int4.to(torch.float16)

    # Convert type based on the CUDA implementation
    in_quant_dequant_fp16 = (in_fp16 * scale_expand) + shift_expand
    in_quant_dequant_fp32 = in_quant_dequant_fp16.to(torch.float)
    in_quant_dequant_bf16 = in_quant_dequant_fp32.to(torch.bfloat16)

    return in_quant, in_quant_dequant_bf16.view(*in_shape)


def gqa_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    seq_lens: List[int],
    qk_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The reference GQA implementation
    """
    (B, T, H, D) = Q.shape
    (_, MAX_T, Hk, D) = K.shape
    (_, MAX_T, Hv, D) = V.shape
    assert T == 1
    assert Hk == Hv
    Y = torch.zeros_like(Q)
    attn_out = torch.zeros(B, H, MAX_T)
    for b in range(B):
        max_t = seq_lens[b]
        for h in range(H):
            # now, compute fused attention
            Q_ = Q[b, 0, h, :]
            assert Q_.shape == (D,)
            K_ = K[b, :max_t, 0, :]
            assert K_.shape == (max_t, D)
            S = (Q_.view(1, D) @ K_.T) * qk_scale  # 1.0 / np.sqrt(D)
            # max_qk_acc = torch.max(S)
            # softmax_denominator = torch.exp(S - max_qk_acc).sum()
            assert S.shape == (1, max_t)
            P = torch.nn.functional.softmax(S, dim=-1)

            assert P.shape == (1, max_t)

            V_ = V[b, :max_t, 0, :]
            assert V_.shape == (max_t, D)
            O_ = P.view(1, max_t) @ V_
            assert O_.shape == (1, D)
            Y[b, 0, h, :] = O_
            attn_out[b, h, :max_t] = P
    return Y, attn_out


class Int4GQATest(unittest.TestCase):
    @unittest.skipIf(
        not torch.version.cuda,
        "Skip when CUDA is not available",
    )
    @settings(verbosity=VERBOSITY, max_examples=40, deadline=None)
    # pyre-ignore
    @given(
        int4_kv=st.booleans(),
        num_groups=st.sampled_from([1, 4]),
        B=st.integers(min_value=1, max_value=128),
        MAX_T=st.integers(min_value=4, max_value=128),
        N_H_L=st.integers(min_value=1, max_value=128),
    )
    def test_gqa(
        self,
        int4_kv: bool,
        num_groups: int,
        B: int,
        MAX_T: int,
        N_H_L: int,
    ) -> None:
        """
        Test correctness of torch.ops.fbgemm.gqa_attn_splitk against the
        reference GQA implementation (testing both BF16 and INT4 KV caches)
        """

        # Constants
        D_H = 128
        N_KVH_L = 1  # gqa_attn_splitk only supports 1 currently
        SEQ_POSITION = MAX_T - 2

        seq_positions = torch.tensor(
            [SEQ_POSITION for _ in range(B)], device="cuda"
        ).int()
        kv_seqlens = [seq_position + 1 for seq_position in seq_positions]
        q = torch.randn((B, 1, N_H_L, D_H), dtype=torch.bfloat16, device="cuda")

        # Generate KV cache
        cache_k = torch.randn(
            (B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda"
        )
        cache_v = torch.randn_like(cache_k)
        if int4_kv:
            cache_k, cache_k_ref = quant_int4_dequant_bf16(cache_k, num_groups)
            cache_v, cache_v_ref = quant_int4_dequant_bf16(cache_v, num_groups)
            cache_k_ref = cache_k_ref.cpu().float()
            cache_v_ref = cache_v_ref.cpu().float()
        else:
            cache_k_ref = cache_k.cpu().float()
            cache_v_ref = cache_v.cpu().float()

        # Compute qk_scale
        qk_scale = 1.0 / np.sqrt(D_H)

        # Run reference
        z_ref, attn_out_ref = gqa_reference(
            q.cpu().float(),
            cache_k_ref,
            cache_v_ref,
            kv_seqlens,
            qk_scale=qk_scale,
        )

        # Run test
        for split_k in [1, 2, 4, 8, 13, 16]:
            z, _, _ = torch.ops.fbgemm.gqa_attn_splitk(
                q,
                cache_k,
                cache_v,
                seq_positions,
                qk_scale=qk_scale,
                num_split_ks=split_k,
                num_int4_kv_groups=num_groups,
            )
            torch.testing.assert_close(
                z.cpu().bfloat16(),
                z_ref.cpu().bfloat16(),
                atol=2.0e-2,
                rtol=6.0e-3,
            )
