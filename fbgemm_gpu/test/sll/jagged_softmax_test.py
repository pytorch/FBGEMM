# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import fbgemm_gpu  # noqa F401
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
class JaggedSoftmaxTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 512),
        N=st.integers(1, 1000),
        H=st.integers(1, 20),
        use_fbgemm_kernel=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_triton_jagged_softmax(
        self, B: int, N: int, H: int, use_fbgemm_kernel: bool, device_type: str
    ) -> None:
        device = torch.device(device_type)
        lengths = torch.randint(0, N + 1, (B,), device=device)
        offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths.cumsum(dim=0),
            ],
            dim=0,
        )

        torch.manual_seed(0)
        x1 = torch.rand(
            (int(lengths.sum().item()), H), requires_grad=True, device=device
        )  # [Sum_B, H]
        padded_x1 = torch.ops.fbgemm.jagged_to_padded_dense(
            x1,
            [offsets],
            max_lengths=[N],
            padding_value=0.0,
        )  # [B, N, H]
        _, presences = torch.ops.fbgemm.pack_segments_v2(
            x1,
            lengths,
            max_length=N,
            return_presence_mask=True,
        )
        softmax_input = (
            padded_x1 - (1.0 - presences.unsqueeze(2).to(padded_x1.dtype)) * 5e7
        )
        padded_ref = torch.nn.functional.softmax(
            softmax_input.transpose(1, 2), dim=-1
        )  # [B, H, N]
        ref = torch.ops.fbgemm.dense_to_jagged(padded_ref.permute(0, 2, 1), [offsets])[
            0
        ]

        torch.manual_seed(0)
        x2 = torch.rand(
            (int(lengths.sum().item()), H), requires_grad=True, device=device
        )  # [Sum_B, H]
        ret = torch.ops.fbgemm.sll_jagged_softmax(
            x2, offsets, N, use_fbgemm_kernel=use_fbgemm_kernel
        )

        assert torch.allclose(ret, ref, 1e-5)

        grad_output = torch.rand((int(lengths.sum().item()), H), device=device) * 0.01
        ref.backward(grad_output)
        ret.backward(grad_output)

        # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
        assert torch.allclose(x1.grad, x2.grad, 1e-5)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 10),
        N=st.integers(10, 100),
        transpose=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_triton_jagged2_softmax(
        self,
        B: int,
        N: int,
        transpose: bool,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)
        lengths_n = torch.randint(1, N + 1, (B,))
        offsets_n = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                lengths_n.cumsum(dim=0),
            ],
            dim=0,
        ).to(device_type)

        lengths_total = lengths_n * lengths_n
        offsets_total = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                lengths_total.cumsum(dim=0),
            ],
            dim=0,
        ).to(device_type)

        torch.manual_seed(0)
        x = torch.rand(
            int(lengths_total.sum().item()), requires_grad=True, device=device
        )
        torch.manual_seed(0)
        x_ref = torch.rand(
            int(lengths_total.sum().item()), requires_grad=True, device=device
        )
        ref = torch.zeros((x.shape[0]), dtype=x.dtype, device=x.device)
        for i in range(B):
            submatrix = x_ref[offsets_total[i] : offsets_total[i + 1]]
            Ni = int(lengths_n[i].item())
            softmax_dim = 0 if transpose else 1
            ref[offsets_total[i] : offsets_total[i + 1]] = torch.nn.functional.softmax(
                submatrix.reshape((Ni, Ni)), dim=softmax_dim
            ).view(-1)

        ret = torch.ops.fbgemm.sll_jagged2_softmax(
            x, offsets_n, offsets_total, N, transpose=transpose
        )

        assert torch.allclose(ret, ref)

        grad_output = torch.rand((ref.shape), device=device) * 0.01
        ref.backward(grad_output)
        ret.backward(grad_output)

        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-5)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers(10, 512)` to decorator factory
    #  `hypothesis.given`.
    @given(
        B=st.integers(10, 512),
        N=st.integers(10, 1000),
        transpose=st.booleans(),
        device_type=st.sampled_from(["meta"]),
    )
    @settings(deadline=None)
    def test_triton_jagged2_softmax_meta_backend(
        self,
        B: int,
        N: int,
        transpose: bool,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)
        lengths_n = torch.randint(1, N + 1, (B,))
        offsets_n = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                lengths_n.cumsum(dim=0),
            ],
            dim=0,
        ).to(device_type)
        lengths_total = lengths_n * lengths_n
        offsets_total = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                lengths_total.cumsum(dim=0),
            ],
            dim=0,
        ).to(device_type)

        torch.manual_seed(0)
        x = torch.rand(
            int(lengths_total.sum().item()), requires_grad=True, device=device
        )

        ret = torch.ops.fbgemm.sll_jagged2_softmax(
            x, offsets_n, offsets_total, N, transpose=transpose
        )

        assert ret.is_meta and ret.size() == x.size()

        grad_output = torch.rand((ret.shape), device=device) * 0.01
        ret.backward(grad_output)

        # pyre-fixme[16]: Optional type has no attribute `is_meta`.
        # pyre-fixme[16]: Optional type has no attribute `size`.
        assert x.grad.is_meta and x.grad.size() == x.size()
