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


class JaggedDenseFlashAttentionTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 10),
        D=st.sampled_from([16, 32, 64]),
        T=st.integers(1, 30),
        max_L=st.integers(1, 100),
        allow_tf32=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_rocm)
    @settings(deadline=30000)
    def test_jagged_dense_flash_attention(
        self,
        B: int,
        D: int,
        T: int,
        max_L: int,
        allow_tf32: bool,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)

        lengths = torch.randint(1, max_L + 1, (B,), device=device)
        offsets = torch.cat(
            [torch.tensor([0], device=device, dtype=torch.int), lengths.cumsum(dim=0)],
            dim=0,
        )

        q = torch.rand(
            size=(int(lengths.sum().item()), D),
            requires_grad=True,
            device=device,
        )

        v = torch.rand(
            size=(int(lengths.sum().item()), D),
            requires_grad=True,
            device=device,
        )

        k = torch.rand(
            size=(B, D, T),
            requires_grad=True,
            device=device,
        )

        attn_bias = torch.rand(size=(B, max_L, T), requires_grad=True, device=device)

        def ref_attention(
            do: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            offsets: torch.Tensor,
            attn_bias: torch.Tensor,
            B: int,
            D: int,
            T: int,
            max_L: int,
            allow_tf32: bool,
        ) -> torch.Tensor:
            # [sum_B, D] * [B, D, T] = [sum_B, T]
            qk = torch.ops.fbgemm.sll_jagged_dense_bmm(
                q,
                k.to(q.dtype),
                offsets,
                max_L,
                allow_tf32=allow_tf32,
                use_fbgemm_kernel=False,
            )

            softmax_input = torch.ops.fbgemm.sll_jagged_dense_elementwise_add(
                qk,
                offsets,
                attn_bias,
                max_L,
                use_fbgemm_kernel=False,
            )

            P = torch.ops.fbgemm.sll_jagged_softmax(
                softmax_input,
                offsets,
                max_L,
                use_fbgemm_kernel=False,
            )  # [sum_B, T]

            # [sum_B, T] * [sum_B, D] = [B, T, D]
            attn_out = torch.ops.fbgemm.sll_jagged_jagged_bmm(
                P,
                v.to(P.dtype),
                offsets,
                max_L,
                allow_tf32=allow_tf32,
                use_fbgemm_kernel=False,
            )

            attn_out.backward(do)
            return attn_out

        # Backward pass
        dout = torch.rand((B, T, D), dtype=q.dtype, device=q.device) * 0.01

        ref = ref_attention(
            dout, q, k, v, offsets, attn_bias, B, D, T, max_L, allow_tf32
        )

        q_clone = clone_tensor(q)
        k_clone = clone_tensor(k)
        v_clone = clone_tensor(v)
        attn_bias_clone = clone_tensor(attn_bias)

        test = torch.ops.fbgemm.sll_jagged_dense_flash_attention(
            q_clone,
            k_clone,
            v_clone,
            attn_bias_clone,
            offsets,
            max_L,
            allow_tf32=allow_tf32,
        )

        if allow_tf32:
            assert torch.allclose(ref, test, atol=1e-3, rtol=1e-3)
        else:
            assert torch.allclose(ref, test, atol=1e-5, rtol=1e-3)

        test.backward(dout)

        assert v.grad is not None
        assert v_clone.grad is not None
        if allow_tf32:
            assert torch.allclose(v.grad, v_clone.grad, atol=1e-3, rtol=1e-3)
        else:
            assert torch.allclose(v.grad, v_clone.grad, atol=1e-5, rtol=1e-3)

        assert attn_bias.grad is not None
        assert attn_bias_clone.grad is not None
        if allow_tf32:
            assert torch.allclose(
                attn_bias.grad, attn_bias_clone.grad, atol=1e-3, rtol=1e-3
            )
        else:
            assert torch.allclose(
                attn_bias.grad, attn_bias_clone.grad, atol=1e-5, rtol=1e-3
            )

        assert q.grad is not None
        assert q_clone.grad is not None
        if allow_tf32:
            assert torch.allclose(q.grad, q_clone.grad, atol=1e-2, rtol=1e-2)
        else:
            assert torch.allclose(q.grad, q_clone.grad, atol=1e-5, rtol=1e-3)

        assert k.grad is not None
        assert k_clone.grad is not None
        if allow_tf32:
            assert torch.allclose(k.grad, k_clone.grad, atol=1e-2, rtol=1e-2)
        else:
            assert torch.allclose(k.grad, k_clone.grad, atol=1e-5, rtol=1e-3)
