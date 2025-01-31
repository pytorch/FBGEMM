# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import hypothesis.strategies as st
import torch
from fbgemm_gpu.sll.triton import multi_head_jagged_flash_attention
from hypothesis import given, settings
from torch.nn import functional as F

from .common import clone_tensor, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


@unittest.skipIf(
    open_source,
    "Test fails in OSS mode, see https://github.com/triton-lang/triton/issues/5435",
)
class MultiHeadJaggedFlashAttentionTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 5),
        max_L=st.integers(1, 100),
        num_heads=st.sampled_from([1, 2, 4, 8]),
        d_per_head=st.sampled_from([32]),
        device_type=st.sampled_from(["cuda"]),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(deadline=None)
    def test_multi_head_jagged_flash_attention(
        self,
        B: int,
        max_L: int,
        num_heads: int,
        d_per_head: int,
        device_type: str,
    ) -> None:
        D = num_heads * d_per_head
        allow_tf32 = False

        device: torch.device = torch.device(device_type)
        num_objects = torch.randint(1, max_L + 1, (B,)).to(device)

        offsets = torch.cat(
            [torch.IntTensor([0]).to(device), num_objects.cumsum(dim=0)], dim=0
        )

        q_weights = torch.rand(
            int(num_objects.sum().item()),
            D,
            device=device,
            requires_grad=True,
        )

        k_weights = torch.rand(
            int(num_objects.sum().item()),
            D,
            device=device,
            requires_grad=True,
        )

        v_weights = torch.rand(
            int(num_objects.sum().item()),
            D,
            device=device,
            requires_grad=True,
        )

        do = torch.rand_like(q_weights) * 0.1

        q_weights_clone = clone_tensor(q_weights)
        k_weights_clone = clone_tensor(k_weights)
        v_weights_clone = clone_tensor(v_weights)

        def ref_multi_head_attention(
            num_objects: torch.Tensor,
            q_weights: torch.Tensor,
            k_weights: torch.Tensor,
            v_weights: torch.Tensor,
            offsets: torch.Tensor,
            num_heads: int,
            max_L: int,
            d_per_head: int,
            do: torch.Tensor,
        ) -> torch.Tensor:
            # [B, H, N, d_per_head]
            padded_q = (
                torch.ops.fbgemm.jagged_to_padded_dense(
                    values=q_weights,
                    offsets=[offsets],
                    max_lengths=[max_L],
                    padding_value=0.0,
                )
                .reshape(-1, max_L, num_heads, d_per_head)
                .transpose(1, 2)
            )

            # [B, H, N, d_per_head]
            padded_k = (
                torch.ops.fbgemm.jagged_to_padded_dense(
                    values=k_weights,
                    offsets=[offsets],
                    max_lengths=[max_L],
                    padding_value=0.0,
                )
                .reshape(-1, max_L, num_heads, d_per_head)
                .transpose(1, 2)
            )

            # [B, H, N, d_per_head]
            padded_v = (
                torch.ops.fbgemm.jagged_to_padded_dense(
                    values=v_weights,
                    offsets=[offsets],
                    max_lengths=[max_L],
                    padding_value=0.0,
                )
                .reshape(-1, max_L, num_heads, d_per_head)
                .transpose(1, 2)
            )

            # [B, H, N, N]
            s = torch.einsum("bhxk,bhyk->bhxy", padded_q, padded_k)

            _, presence_mask = torch.ops.fbgemm.pack_segments_v2(
                q_weights,
                num_objects,
                max_length=max_L,
                pad_minf=False,
                return_presence_mask=True,
            )

            s = s - (1.0 - presence_mask.unsqueeze(2).unsqueeze(1).to(s.dtype)) * 5e4
            p = F.softmax(s, dim=-1) / max_L
            attn_out = torch.matmul(p, padded_v)
            attn_out = attn_out.transpose(1, 2).reshape(
                -1, max_L, num_heads * d_per_head
            )

            jagged_attn_out = torch.ops.fbgemm.dense_to_jagged(attn_out, [offsets])[0]
            jagged_attn_out.backward(do)

            return jagged_attn_out

        attn_out_ref = ref_multi_head_attention(
            num_objects=num_objects,
            q_weights=q_weights,
            k_weights=k_weights,
            v_weights=v_weights,
            offsets=offsets,
            num_heads=num_heads,
            max_L=max_L,
            d_per_head=d_per_head,
            do=do,
        )

        attn_out_test = multi_head_jagged_flash_attention(
            q_weights=q_weights_clone.reshape(-1, num_heads, d_per_head).transpose(
                0, 1
            ),
            k_weights=k_weights_clone.reshape(-1, num_heads, d_per_head).transpose(
                0, 1
            ),
            v_weights=v_weights_clone.reshape(-1, num_heads, d_per_head).transpose(
                0, 1
            ),
            offsets=offsets,
            max_seq_len=max_L,
            allow_tf32=allow_tf32,
        )
        attn_out_test = attn_out_test.transpose(0, 1).reshape(
            -1, num_heads * d_per_head
        )

        assert torch.allclose(attn_out_ref, attn_out_test, atol=1e-2, rtol=1e-2)

        attn_out_test.backward(do)

        assert v_weights.grad is not None
        assert v_weights_clone.grad is not None
        assert torch.allclose(
            v_weights.grad, v_weights_clone.grad, atol=1e-2, rtol=1e-2
        )

        assert k_weights.grad is not None
        assert k_weights_clone.grad is not None
        assert torch.allclose(
            k_weights.grad, k_weights_clone.grad, atol=1e-2, rtol=1e-2
        )

        assert q_weights.grad is not None
        assert q_weights_clone.grad is not None
        assert torch.allclose(
            q_weights.grad, q_weights_clone.grad, atol=1e-2, rtol=1e-2
        )
