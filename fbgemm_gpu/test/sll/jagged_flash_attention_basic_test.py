# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import fbgemm_gpu.sll  # noqa F401
import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from .common import clone_tensor, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, running_on_rocm
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, running_on_rocm


class JaggedFlashAttentionBasicTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 10),
        max_L=st.integers(1, 100),
        D=st.integers(16, 64),
        use_mask=st.booleans(),
        allow_tf32=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_rocm)
    @settings(deadline=40000)
    def test_triton_jagged_flash_attention_basic(
        self,
        B: int,
        max_L: int,
        D: int,
        use_mask: bool,
        allow_tf32: bool,
        device_type: str,
    ) -> None:
        device: torch.device = torch.device(device_type)
        num_objects = torch.randint(1, max_L + 1, (B,)).to(device)

        x_offsets = torch.cat(
            [torch.IntTensor([0]).to(device), num_objects.cumsum(dim=0)], dim=0
        )
        attn_lengths = num_objects * num_objects
        attn_offsets = torch.cat(
            [torch.IntTensor([0]).to(device), attn_lengths.cumsum(dim=0)], dim=0
        )

        q_weights = torch.rand(
            int(num_objects.sum().item()), D, device=device, requires_grad=True
        )

        k_weights = torch.rand(
            int(num_objects.sum().item()), D, device=device, requires_grad=True
        )

        v_weights = torch.rand(
            int(num_objects.sum().item()), D, device=device, requires_grad=True
        )

        do = torch.rand_like(q_weights) * 0.01

        q_weights_clone = clone_tensor(q_weights)
        k_weights_clone = clone_tensor(k_weights)
        v_weights_clone = clone_tensor(v_weights)

        def ref_attention_basic(
            num_objects: torch.Tensor,
            x_offsets: torch.Tensor,
            attn_lengths: torch.Tensor,
            attn_offsets: torch.Tensor,
            q_weights: torch.Tensor,
            k_weights: torch.Tensor,
            v_weights: torch.Tensor,
            max_seq_len: int,
            use_mask: bool,
            allow_tf32: bool,
            do: torch.Tensor,
        ) -> torch.Tensor:
            s = torch.ops.fbgemm.sll_jagged_jagged_bmm_jagged_out(
                x=q_weights,
                y=k_weights,  # transpose is done inside the function
                x_lengths=num_objects,
                x_offsets=x_offsets,
                y_lengths=num_objects,
                y_offsets=x_offsets,
                z_lengths=attn_lengths,
                z_offsets=attn_offsets,
                max_seq_len=max_seq_len,
                allow_tf32=allow_tf32,
            )

            p = (
                torch.ops.fbgemm.sll_jagged2_softmax(
                    x=s,
                    offsets=x_offsets,
                    offsets_total=attn_offsets,
                    max_seq_len=max_seq_len,
                    transpose=False,
                )
                / max_seq_len
            )

            if use_mask:
                attn_mask = torch.triu(
                    torch.ones(
                        (max_seq_len, max_seq_len),
                        dtype=torch.bool,
                        device=q_weights.device,
                    ),
                ).requires_grad_(False)
                # p = p * attn_mask
                p = torch.ops.fbgemm.sll_jagged_dense_elementwise_mul_jagged_out(
                    x=p,
                    y=attn_mask,
                    x_seq_lengths=num_objects,
                    x_offsets=attn_offsets,
                    max_seq_len=max_seq_len,
                )

            attn_out = torch.ops.fbgemm.sll_array_jagged_bmm_jagged_out(
                x=p,
                y=v_weights,
                x_lengths=attn_lengths,
                x_offsets=attn_offsets,
                y_lengths=num_objects,
                y_offsets=x_offsets,
                z_lengths=num_objects,
                z_offsets=x_offsets,
                max_seq_len=max_seq_len,
                allow_tf32=allow_tf32,
            )

            attn_out.backward(do)

            return attn_out

        attn_out_ref = ref_attention_basic(
            num_objects=num_objects,
            x_offsets=x_offsets,
            attn_lengths=attn_lengths,
            attn_offsets=attn_offsets,
            q_weights=q_weights,
            k_weights=k_weights,
            v_weights=v_weights,
            max_seq_len=max_L,
            use_mask=use_mask,
            allow_tf32=allow_tf32,
            do=do,
        )

        attn_out_test = torch.ops.fbgemm.sll_jagged_flash_attention_basic(
            q_weights=q_weights_clone,
            k_weights=k_weights_clone,
            v_weights=v_weights_clone,
            offsets=x_offsets,
            max_seq_len=max_L,
            use_mask=use_mask,
            allow_tf32=allow_tf32,
        )

        assert torch.allclose(attn_out_ref, attn_out_test, atol=1e-5, rtol=1e-3)

        attn_out_test.backward(do)

        assert v_weights.grad is not None
        assert v_weights_clone.grad is not None
        assert torch.allclose(
            v_weights.grad, v_weights_clone.grad, atol=1e-4, rtol=1e-3
        )

        assert k_weights.grad is not None
        assert k_weights_clone.grad is not None
        assert torch.allclose(
            k_weights.grad, k_weights_clone.grad, atol=1e-4, rtol=1e-3
        )

        assert q_weights.grad is not None
        assert q_weights_clone.grad is not None
        assert torch.allclose(
            q_weights.grad, q_weights_clone.grad, atol=1e-4, rtol=1e-3
        )
