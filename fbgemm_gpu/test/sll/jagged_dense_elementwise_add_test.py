#!/usr/bin/env python3
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


class JaggedDenseElementwiseAddTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 512),
        D=st.integers(1, 100),
        N=st.integers(1, 200),
        use_fbgemm_kernel=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(deadline=30000)
    def test_triton_jagged_dense_add(
        self, B: int, D: int, N: int, use_fbgemm_kernel: bool, device_type: str
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
            (int(lengths.sum().item()), D), requires_grad=True, device=device
        )  # [Sum_B, D]
        padded_x1 = torch.ops.fbgemm.jagged_to_padded_dense(
            x1,
            [offsets],
            max_lengths=[N],
            padding_value=0.0,
        )  # [B, N, D]
        y1 = torch.rand((B, N, D), requires_grad=True, device=device)
        ref = torch.ops.fbgemm.dense_to_jagged(padded_x1 + y1, [offsets])[0]

        torch.manual_seed(0)
        x2 = torch.rand(
            (int(lengths.sum().item()), D), requires_grad=True, device=device
        )  # [Sum_B, D]
        y2 = torch.rand((B, N, D), requires_grad=True, device=device)
        ret = torch.ops.fbgemm.sll_jagged_dense_elementwise_add(
            x2, offsets, y2, N, use_fbgemm_kernel=use_fbgemm_kernel
        )

        if not torch.allclose(ref, ret, 1e-5):
            print(ret, ref)

        assert torch.allclose(ref, ret, 1e-5)

        grad_output = torch.rand((int(lengths.sum().item()), D), device=device) * 0.01
        ref.backward(grad_output)
        ret.backward(grad_output)

        # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
        assert torch.allclose(x1.grad, x2.grad, 1e-5)
        # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
        assert torch.allclose(y1.grad, y2.grad, 1e-5)
