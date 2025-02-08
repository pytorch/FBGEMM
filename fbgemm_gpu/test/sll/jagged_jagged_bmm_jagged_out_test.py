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

from .common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable

if torch.cuda.is_available():
    from fbgemm_gpu.sll.triton import triton_jagged_jagged_bmm_jagged_out


class JaggedJaggedBmmJaggedOutTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 512),
        max_L=st.integers(1, 200),
        K=st.integers(1, 100),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(deadline=30000)
    def test_triton_jagged_jagged_bmm_jagged_out(
        self,
        B: int,
        max_L: int,
        K: int,
    ) -> None:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        lengths_m = torch.randint(1, max_L + 1, (B,)).cuda()
        lengths_n = lengths_m

        offsets_m = torch.cat(
            [torch.IntTensor([0]).cuda(), lengths_m.cumsum(dim=0)], dim=0
        )
        offsets_n = torch.cat(
            [torch.IntTensor([0]).cuda(), lengths_n.cumsum(dim=0)], dim=0
        )
        lengths_mn = lengths_m * lengths_n
        offsets_mn = torch.cat(
            [torch.IntTensor([0]).cuda(), lengths_mn.cumsum(dim=0)], dim=0
        )

        jagged_A = torch.rand(int(lengths_m.sum().item()), K).cuda()
        jagged_B = torch.rand(int(lengths_n.sum().item()), K).cuda()

        def ref_jagged_jagged_bmm_jagged_out(
            B: int,
            jagged_A: torch.Tensor,
            jagged_B: torch.Tensor,
            lengths_mn: torch.Tensor,
            offsets_mn: torch.Tensor,
            offsets_m: torch.Tensor,
            offsets_n: torch.Tensor,
        ) -> torch.Tensor:
            jagged_C = torch.empty(
                (int(lengths_mn.sum().item())), dtype=jagged_A.dtype
            ).to(jagged_A.device)

            for i in range(B):
                jagged_C[offsets_mn[i] : offsets_mn[i + 1]] = torch.matmul(
                    jagged_A[offsets_m[i] : offsets_m[i + 1]],
                    jagged_B[offsets_n[i] : offsets_n[i + 1]].T,
                ).flatten()
            return jagged_C

        jagged_C_ref = ref_jagged_jagged_bmm_jagged_out(
            B, jagged_A, jagged_B, lengths_mn, offsets_mn, offsets_m, offsets_n
        )
        jagged_C_test = triton_jagged_jagged_bmm_jagged_out(
            jagged_A,
            jagged_B.T,
            max_L,
            lengths_m,
            lengths_n,
            lengths_mn,
            offsets_m,
            offsets_n,
            offsets_mn,
            allow_tf32=False,
        )

        assert torch.allclose(jagged_C_ref, jagged_C_test)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers(10, 512)` to decorator factory
    #  `hypothesis.given`.
    @given(
        B=st.integers(10, 512),
        max_L=st.integers(1, 200),
        K=st.integers(1, 100),
        device_type=st.sampled_from(["meta"]),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(deadline=30000)
    def test_triton_jagged_jagged_bmm_jagged_out_meta_backend(
        self,
        B: int,
        max_L: int,
        K: int,
        device_type: str,
    ) -> None:
        lengths_m = torch.randint(1, max_L + 1, (B,))
        lengths_n = lengths_m
        device = torch.device(device_type)

        offsets_m = torch.cat([torch.IntTensor([0]), lengths_m.cumsum(dim=0)], dim=0)
        offsets_n = torch.cat([torch.IntTensor([0]), lengths_n.cumsum(dim=0)], dim=0)
        lengths_mn = lengths_m * lengths_n
        offsets_mn = torch.cat([torch.IntTensor([0]), lengths_mn.cumsum(dim=0)], dim=0)

        jagged_A = torch.rand(
            int(lengths_m.sum().item()), K, requires_grad=True, device=device
        )
        jagged_B = torch.rand(
            int(lengths_n.sum().item()), K, requires_grad=True, device=device
        )

        jagged_C_ref = torch.rand(int(lengths_mn.sum().item()), device=device)
        jagged_C_test = torch.ops.fbgemm.sll_jagged_jagged_bmm_jagged_out(
            jagged_A,
            jagged_B.T,
            lengths_m.to(device_type),
            offsets_m.to(device_type),
            lengths_n.to(device_type),
            offsets_n.to(device_type),
            lengths_mn,
            offsets_mn,
            max_L,
            allow_tf32=False,
        )
        assert jagged_C_test.is_meta and jagged_C_ref.size() == jagged_C_test.size()

        grad_output = torch.rand((jagged_C_test.shape), device=device_type) * 0.01
        jagged_C_test.backward(grad_output)

        # pyre-fixme[16]: Optional type has no attribute `is_meta`.
        # pyre-fixme[16]: Optional type has no attribute `size`.
        assert jagged_A.grad.is_meta and jagged_A.grad.size() == jagged_A.size()
        assert jagged_B.grad.is_meta and jagged_B.grad.size() == jagged_B.size()
