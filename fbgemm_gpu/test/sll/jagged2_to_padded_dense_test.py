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


class Jagged2ToPaddedDenseTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_rocm)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 10),
        max_L=st.integers(1, 100),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_jagged2_to_padded_dense(
        self,
        B: int,
        max_L: int,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)

        lengths = torch.randint(1, max_L + 1, (B,), device=device)
        lengths_square = lengths * lengths
        offsets = torch.cat(
            [
                torch.tensor([0], device=device, dtype=torch.int),
                lengths_square.cumsum(dim=0),
            ],
            dim=0,
        )

        x = torch.rand(
            int(lengths_square.sum().item()),
            requires_grad=True,
            device=device,
        )

        def ref_jagged2_to_padded_dense(
            x: torch.Tensor, offsets: torch.Tensor, max_L: int, padding_value: float
        ) -> torch.Tensor:
            B = offsets.size(0) - 1
            dense_output = torch.full(
                (B, max_L, max_L),
                padding_value,
                dtype=x.dtype,
                device=x.device,
            )
            for b in range(B):
                begin = offsets[b]
                end = offsets[b + 1]
                Ni = int(torch.sqrt(end - begin))
                if Ni == 0:
                    continue
                dense_output[b, 0:Ni, 0:Ni] = x[begin:end].view(Ni, Ni)

            return dense_output

        x_clone = (
            x.detach().clone().requires_grad_()
            if x.requires_grad
            else x.detach().clone()
        )
        padding_value = 0.0
        ref_out = ref_jagged2_to_padded_dense(x, offsets, max_L, padding_value)
        test_out = torch.ops.fbgemm.sll_jagged2_to_padded_dense(
            x_clone, offsets, max_L, padding_value
        )
        assert torch.allclose(ref_out, test_out)

        # Backward pass
        dout = torch.rand((B, max_L, max_L), dtype=x.dtype, device=x.device) * 0.1
        test_out.backward(dout)
        ref_out.backward(dout)

        assert x.grad is not None
        assert x_clone.grad is not None
        assert torch.allclose(x.grad, x_clone.grad)
