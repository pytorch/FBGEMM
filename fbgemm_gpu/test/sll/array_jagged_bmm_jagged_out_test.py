# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from .common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, running_on_rocm
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, running_on_rocm

if torch.cuda.is_available():
    from fbgemm_gpu.sll.triton import triton_array_jagged_bmm_jagged_out


class ArrayJaggedBmmJaggedTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 512),
        max_L=st.integers(1, 200),
        D=st.integers(1, 100),
    )
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_rocm)
    @settings(deadline=30000)
    def test_triton_array_jagged_bmm_jagged_out(
        self,
        B: int,
        max_L: int,
        D: int,
    ) -> None:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        lengths_bk = torch.randint(0, max_L + 100, (B,)).cuda()
        lengths_cm = lengths_bk

        offsets_bk = torch.cat(
            [torch.IntTensor([0]).cuda(), lengths_bk.cumsum(dim=0)], dim=0
        )
        offsets_cm = torch.cat(
            [torch.IntTensor([0]).cuda(), lengths_cm.cumsum(dim=0)], dim=0
        )

        lengths_am = lengths_bk * lengths_cm

        offsets_am = torch.cat(
            [torch.IntTensor([0]).cuda(), lengths_am.cumsum(dim=0)], dim=0
        )

        array_A = torch.rand(int(lengths_am.sum().item())).cuda() * 0.01
        jagged_B = torch.rand(int(lengths_bk.sum().item()), D).cuda()

        def ref_array_jagged_bmm_jagged_out(
            array_A: torch.Tensor,
            jagged_B: torch.Tensor,
            lengths_am: torch.Tensor,
            lengths_bk: torch.Tensor,
            lengths_cm: torch.Tensor,
            offsets_am: torch.Tensor,
            offsets_bk: torch.Tensor,
            offsets_cm: torch.Tensor,
            max_seq_len: int,
            transpose: bool = False,
        ) -> torch.Tensor:
            B = lengths_am.size(0)
            D = jagged_B.size(1)
            jagged_C = torch.zeros(
                (int(lengths_cm.sum()), D), device=jagged_B.device, dtype=jagged_B.dtype
            )

            for i in range(B):
                seq_len = int(lengths_bk[i])
                capped_seq_len = min(seq_len, max_seq_len)
                a = array_A[offsets_am[i] : offsets_am[i + 1]].view(seq_len, seq_len)
                a = a[:capped_seq_len, :capped_seq_len]

                if transpose:
                    a = a.T
                b = jagged_B[offsets_bk[i] : offsets_bk[i] + capped_seq_len]
                jagged_C[offsets_cm[i] : offsets_cm[i] + capped_seq_len] = torch.matmul(
                    a.float(), b.float()
                )
            return jagged_C

        jagged_C_ref = ref_array_jagged_bmm_jagged_out(
            array_A,
            jagged_B,
            lengths_am,
            lengths_bk,
            lengths_cm,
            offsets_am,
            offsets_bk,
            offsets_cm,
            max_L,
            False,
        )
        jagged_C_test = triton_array_jagged_bmm_jagged_out(
            array_A,
            jagged_B,
            lengths_am,
            lengths_bk,
            lengths_cm,
            offsets_am,
            offsets_bk,
            offsets_cm,
            max_L,
            False,
            0,
        )

        assert torch.allclose(jagged_C_ref, jagged_C_test)

        jagged_C_ref = ref_array_jagged_bmm_jagged_out(
            array_A,
            jagged_B,
            lengths_am,
            lengths_bk,
            lengths_cm,
            offsets_am,
            offsets_bk,
            offsets_cm,
            max_L,
            True,
        )
        jagged_C_test = triton_array_jagged_bmm_jagged_out(
            array_A,
            jagged_B,
            lengths_am,
            lengths_bk,
            lengths_cm,
            offsets_am,
            offsets_bk,
            offsets_cm,
            max_L,
            False,
            1,
        )

        assert torch.allclose(jagged_C_ref, jagged_C_test)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 512),
        max_L=st.integers(1, 200),
        D=st.integers(1, 100),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_rocm)
    @settings(deadline=30000)
    def test_triton_array_jagged_bmm_jagged_out_with_grad(
        self,
        B: int,
        max_L: int,
        D: int,
        device_type: str,
    ) -> None:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        device = torch.device(device_type)
        lengths_y = torch.randint(0, max_L + 1, (B,), device=device)
        lengths_x = lengths_y * lengths_y

        offsets_x = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths_x.cumsum(dim=0),
            ],
            dim=0,
        )
        offsets_y = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths_y.cumsum(dim=0),
            ],
            dim=0,
        )

        torch.manual_seed(0)
        array_x1 = torch.rand(
            int(lengths_x.sum().item()), device=device, requires_grad=True
        )

        jagged_y1 = torch.rand(
            (int(lengths_y.sum().item()), D), device=device, requires_grad=True
        )

        torch.manual_seed(0)
        array_x2 = torch.rand(
            int(lengths_x.sum().item()), device=device, requires_grad=True
        )

        jagged_y2 = torch.rand(
            (int(lengths_y.sum().item()), D), device=device, requires_grad=True
        )

        ret = torch.ops.fbgemm.sll_array_jagged_bmm_jagged_out(
            array_x2,
            jagged_y2,
            lengths_x,
            offsets_x,
            lengths_y,
            offsets_y,
            lengths_y,
            offsets_y,
            max_L,
            allow_tf32=False,
        )

        ref = torch.empty(jagged_y2.shape, dtype=torch.float32, device=device)
        for i in range(B):
            ni = int(lengths_y[i].item())
            ref[offsets_y[i] : offsets_y[i + 1]] = torch.matmul(
                array_x1[offsets_x[i] : offsets_x[i + 1]].view((ni, ni)),
                jagged_y1[offsets_y[i] : offsets_y[i + 1]],
            )

        torch.testing.assert_close(ret, ref, rtol=1e-4, atol=1e-5)

        grad_output = torch.rand(ref.shape, device=device) * 0.01
        ref.backward(grad_output)
        ret.backward(grad_output)

        torch.testing.assert_close(array_x1.grad, array_x2.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(jagged_y1.grad, jagged_y2.grad, rtol=1e-4, atol=1e-5)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 512),
        max_L=st.integers(1, 200),
        D=st.integers(1, 100),
        device_type=st.sampled_from(["meta"]),
    )
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_rocm)
    @settings(deadline=30000)
    def test_triton_array_jagged_bmm_jagged_out_meta_backend(
        self,
        B: int,
        max_L: int,
        D: int,
        device_type: str,
    ) -> None:

        device = torch.device(device_type)
        lengths_y = torch.randint(0, max_L + 1, (B,))
        lengths_x = lengths_y * lengths_y

        offsets_x = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                lengths_x.cumsum(dim=0),
            ],
            dim=0,
        )
        offsets_y = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                lengths_y.cumsum(dim=0),
            ],
            dim=0,
        )

        torch.manual_seed(0)
        array_x1 = torch.rand(
            int(lengths_x.sum().item()), device=device, requires_grad=True
        )

        jagged_y1 = torch.rand(
            (int(lengths_y.sum().item()), D), device=device, requires_grad=True
        )

        torch.manual_seed(0)
        array_x2 = torch.rand(
            int(lengths_x.sum().item()), device=device, requires_grad=True
        )

        jagged_y2 = torch.rand(
            (int(lengths_y.sum().item()), D), device=device, requires_grad=True
        )

        ret = torch.ops.fbgemm.sll_array_jagged_bmm_jagged_out(
            array_x2,
            jagged_y2,
            lengths_x.to(device_type),
            offsets_x.to(device_type),
            lengths_y.to(device_type),
            offsets_y.to(device_type),
            lengths_y.to(device_type),
            offsets_y.to(device_type),
            max_L,
            allow_tf32=False,
        )

        ref = torch.empty(jagged_y2.shape, dtype=torch.float32, device=device)
        for i in range(B):
            ni = int(lengths_y[i].item())
            ref[offsets_y[i] : offsets_y[i + 1]] = torch.matmul(
                array_x1[offsets_x[i] : offsets_x[i + 1]].view((ni, ni)),
                jagged_y1[offsets_y[i] : offsets_y[i + 1]],
            )

        assert ret.is_meta and ret.size() == ref.size()

        # check backward
        grad_output = torch.rand(ref.shape, device=device) * 0.01
        ref.backward(grad_output.to(device_type))
        ret.backward(grad_output.to(device_type))

        assert (
            # pyre-fixme[16]: Optional type has no attribute `is_meta`.
            array_x1.grad.is_meta
            and array_x2.grad.is_meta
            # pyre-fixme[16]: Optional type has no attribute `size`.
            and array_x1.grad.size() == array_x2.grad.size()
        )
        assert (
            jagged_y1.grad.is_meta
            and jagged_y2.grad.is_meta
            and jagged_y1.grad.size() == jagged_y2.grad.size()
        )
