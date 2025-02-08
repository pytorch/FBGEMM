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


class DenseJaggedCatJaggedOutTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_rocm)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 512),
        max_L=st.integers(1, 200),
        device_type=st.sampled_from(["cpu", "cuda"]),
        enable_pt2=st.sampled_from([True, False]),
    )
    @settings(deadline=None)
    def test_dense_jagged_cat_jagged_out(
        self,
        B: int,
        max_L: int,
        device_type: str,
        enable_pt2: bool,
    ) -> None:
        device = torch.device(device_type)
        lengths = torch.randint(0, max_L + 1, (B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        c_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths + 1)
        a = torch.randint(0, 100000000, (B,), device=device)
        b = torch.randint(0, 100000000, (int(lengths.sum().item()),), device=device)

        ref = torch.cat(
            [
                (
                    torch.cat((a[i : i + 1], b[offsets[i] : offsets[i + 1]]), dim=-1)
                    if lengths[i] > 0
                    else a[i : i + 1]
                )
                for i in range(B)
            ],
            dim=-1,
        )

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def model(a, b, offsets, max_L):
            return torch.ops.fbgemm.sll_dense_jagged_cat_jagged_out(
                a, b, offsets, max_L
            )

        if enable_pt2:
            model = torch.compile(model)

        ret, c_offsets_computed = model(a, b, offsets, max_L)

        assert torch.allclose(ref, ret)
        assert torch.equal(c_offsets, c_offsets_computed)
