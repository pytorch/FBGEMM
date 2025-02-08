# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import fbgemm_gpu.sll  # noqa F401
import torch
from hypothesis import given, settings, strategies as st

from .common import open_source  # noqa

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, running_on_rocm
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, running_on_rocm


@unittest.skipIf(*gpu_unavailable)
@unittest.skipIf(*running_on_rocm)
class JaggedDenseElementwiseMulJaggedOutTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 512),
        L=st.integers(1, 200),
    )
    @settings(deadline=30000)
    def test_jagged_dense_elementwise_mul_jagged_out(
        self,
        B: int,
        L: int,
    ) -> None:
        torch.manual_seed(0)
        seq_lengths_a = torch.randint(0, L + 1, (B,)).cuda()

        offsets_a = torch.cat(
            [torch.IntTensor([0]).cuda(), torch.square(seq_lengths_a).cumsum(dim=0)],
            dim=0,
        )

        jagged_A = torch.rand(int(offsets_a[-1] - offsets_a[0])).cuda()

        # test zero out upper triangle
        mask = torch.tril(
            torch.ones(
                (L, L),
                dtype=torch.bool,
            ).cuda(),
        )
        mask = mask.fill_diagonal_(False)
        result = torch.ops.fbgemm.sll_jagged_dense_elementwise_mul_jagged_out(
            jagged_A,
            mask,
            seq_lengths_a,
            offsets_a,
            L,
        )

        for i in range(B):
            if seq_lengths_a[i] == 0:
                continue

            a = jagged_A[offsets_a[i] : offsets_a[i + 1]]
            a = a.view(int(seq_lengths_a[i]), int(seq_lengths_a[i]))
            ref = a * mask[0 : seq_lengths_a[i], 0 : seq_lengths_a[i]]

            assert torch.equal(result[offsets_a[i] : offsets_a[i + 1]], ref.flatten())

        # test general jagged dense elementwise mul
        dense_B = torch.rand((L, L)).cuda()
        result = torch.ops.fbgemm.sll_jagged_dense_elementwise_mul_jagged_out(
            jagged_A,
            dense_B,
            seq_lengths_a,
            offsets_a,
            L,
        )

        for i in range(B):
            if seq_lengths_a[i] == 0:
                continue

            a = jagged_A[offsets_a[i] : offsets_a[i + 1]]
            a = a.view(int(seq_lengths_a[i]), int(seq_lengths_a[i]))

            b = dense_B[: seq_lengths_a[i], : seq_lengths_a[i]]
            ref = a * b
            assert torch.equal(result[offsets_a[i] : offsets_a[i + 1]], ref.flatten())

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 512),
        L=st.integers(1, 200),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_jagged_dense_elementwise_mul_jagged_out_with_grad(
        self,
        B: int,
        L: int,
        device_type: str,
    ) -> None:
        torch.manual_seed(0)

        device = torch.device(device_type)
        seq_lengths_a = torch.randint(0, L + 1, (B,), device=device)

        offsets_a = torch.cat(
            [
                torch.IntTensor([0]).to(device_type),
                torch.square(seq_lengths_a).cumsum(dim=0),
            ],
            dim=0,
        )

        jagged_A = (
            torch.rand(int(offsets_a[-1] - offsets_a[0]), device=device)
            .detach()
            .requires_grad_(True)
        )
        dense_B = torch.rand((L, L), device=device)
        jagged_A_ref = jagged_A.clone().detach().requires_grad_(True)

        # check forward
        result = torch.ops.fbgemm.sll_jagged_dense_elementwise_mul_jagged_out(
            jagged_A,
            dense_B,
            seq_lengths_a,
            offsets_a,
            L,
        )

        ref = []
        for i in range(B):
            if seq_lengths_a[i] == 0:
                continue
            a = jagged_A_ref[offsets_a[i] : offsets_a[i + 1]].view(
                int(seq_lengths_a[i]), int(seq_lengths_a[i])
            )
            b = dense_B[: seq_lengths_a[i], : seq_lengths_a[i]]
            c = a * b
            ref.append(c.flatten())

        ref = torch.cat(ref)
        assert torch.equal(result, ref)

        # check backward
        grad_output = torch.rand(ref.shape, device=device) * 0.01
        ref.backward(grad_output)
        result.backward(grad_output)

        assert torch.allclose(jagged_A_ref.grad, jagged_A.grad)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 512),
        L=st.integers(1, 200),
        device_type=st.sampled_from(["meta"]),
    )
    @settings(deadline=30000)
    def test_jagged_dense_elementwise_mul_jagged_out_meta_backend(
        self,
        B: int,
        L: int,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)
        torch.manual_seed(0)

        device = torch.device(device_type)
        seq_lengths_a = torch.randint(0, L + 1, (B,))

        offsets_a = torch.cat(
            [
                torch.IntTensor([0]),
                torch.square(seq_lengths_a).cumsum(dim=0),
            ],
            dim=0,
        )

        jagged_A = (
            torch.rand(int(offsets_a[-1] - offsets_a[0]), device=device)
            .detach()
            .requires_grad_(True)
        )
        dense_B = torch.rand((L, L), device=device)
        jagged_A_ref = jagged_A.clone().detach().requires_grad_(True)

        # check forward
        result = torch.ops.fbgemm.sll_jagged_dense_elementwise_mul_jagged_out(
            jagged_A,
            dense_B,
            seq_lengths_a.to(device),
            offsets_a.to(device),
            L,
        )

        ref = []
        for i in range(B):
            if seq_lengths_a[i] == 0:
                continue
            a = jagged_A_ref[offsets_a[i] : offsets_a[i + 1]].view(
                int(seq_lengths_a[i]), int(seq_lengths_a[i])
            )
            b = dense_B[: seq_lengths_a[i], : seq_lengths_a[i]]
            c = a * b
            ref.append(c.flatten())

        ref = torch.cat(ref)
        assert result.is_meta and result.size() == ref.size()

        # check backward
        grad_output = torch.rand(ref.shape, device=device) * 0.01
        ref.backward(grad_output)
        result.backward(grad_output)

        assert (
            jagged_A.grad.is_meta and jagged_A_ref.grad.size() == jagged_A.grad.size()
        )
