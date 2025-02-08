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
class JaggedDenseBMMTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 512),
        D=st.integers(1, 256),
        N=st.integers(1, 1000),
        T=st.integers(1, 256),
        use_fbgemm_kernel=st.booleans(),
        allow_tf32=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_triton_jagged_dense_bmm(
        self,
        B: int,
        D: int,
        N: int,
        T: int,
        allow_tf32: bool,
        use_fbgemm_kernel: bool,
        device_type: str,
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
        x = torch.rand(int(lengths.sum().item()), D, device=device)
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
            x,
            [offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        y = torch.rand((B, D, T), device=device)
        padded_ref = torch.bmm(padded_x, y)
        ref = torch.ops.fbgemm.dense_to_jagged(padded_ref, [offsets])[0]
        ret = torch.ops.fbgemm.sll_jagged_dense_bmm(
            x, y, offsets, N, allow_tf32=allow_tf32, use_fbgemm_kernel=use_fbgemm_kernel
        )
        if allow_tf32:
            assert torch.allclose(ref, ret, atol=1e-3, rtol=1e-3)
        else:
            assert torch.allclose(ref, ret, 1e-5)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 512),
        D=st.integers(1, 256),
        N=st.integers(1, 1000),
        T=st.integers(1, 256),
        use_fbgemm_kernel=st.booleans(),
        allow_tf32=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_triton_jagged_dense_bmm_with_grad(
        self,
        B: int,
        D: int,
        N: int,
        T: int,
        allow_tf32: bool,
        use_fbgemm_kernel: bool,
        device_type: str,
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
        )
        padded_x1 = torch.ops.fbgemm.jagged_to_padded_dense(
            x1,
            [offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        y1 = torch.rand((B, D, T), requires_grad=True, device=device)
        padded_ref = torch.bmm(padded_x1, y1)
        ref = torch.ops.fbgemm.dense_to_jagged(padded_ref, [offsets])[0]

        torch.manual_seed(0)
        x2 = torch.rand(
            (int(lengths.sum().item()), D), requires_grad=True, device=device
        )
        y2 = torch.rand((B, D, T), requires_grad=True, device=device)
        ret = torch.ops.fbgemm.sll_jagged_dense_bmm(
            x2,
            y2,
            offsets,
            N,
            allow_tf32=allow_tf32,
            use_fbgemm_kernel=use_fbgemm_kernel,
        )

        if allow_tf32:
            assert torch.allclose(ref, ret, atol=1e-3, rtol=1e-3)
        else:
            assert torch.allclose(ref, ret, 1e-5)

        grad_output = torch.rand((int(lengths.sum().item()), T), device=device) * 0.01
        ref.backward(grad_output)
        ret.backward(grad_output)

        if allow_tf32:
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(x1.grad, x2.grad, atol=1e-3, rtol=1e-3)
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(y1.grad, y2.grad, atol=1e-3, rtol=1e-3)
        else:
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(x1.grad, x2.grad, 1e-5)
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(y1.grad, y2.grad, 1e-5)
