# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[16,21,53,56]

import logging
import unittest
from typing import Tuple

import torch
import triton  # noqa: F401
from fbgemm_gpu.experimental.gen_ai.moe import silu_mul
from fbgemm_gpu.experimental.gen_ai.moe.activation import silu_mul
from hypothesis import given, settings, strategies as st, Verbosity

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

_MAX_SAMPLES: int = 100


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when no GPU is available.",
)
class ActivationTests(unittest.TestCase):
    """Test activation kernels."""

    @given(
        T=st.sampled_from([1, 128, 2048, 4096, 16384]),
        D=st.sampled_from([5120, 7168]),
        contiguous=st.sampled_from([True, False]),
        compiled=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_silu_mul(
        self,
        T: int,
        D: int,
        contiguous: bool,
        compiled: bool,
    ) -> None:
        if contiguous:
            x0 = torch.randn([T, D], device="cuda", dtype=torch.bfloat16)
            x1 = torch.randn([T, D], device="cuda", dtype=torch.bfloat16)
        else:
            x = torch.randn([T, 2 * D], device="cuda", dtype=torch.bfloat16)
            x0 = x[:, :D]
            x1 = x[:, D:]

        def fn() -> torch.Tensor:
            op = silu_mul
            if compiled:
                op = torch.compile(op)
            return op(x0, x1)

        y = fn()

        def ref_fn() -> torch.Tensor:
            x0_fp32 = x0.to(torch.float32)
            x1_fp32 = x1.to(torch.float32)
            return (x0_fp32 * torch.sigmoid(x0_fp32) * x1_fp32).to(torch.bfloat16)

        y_ref = ref_fn()

        torch.testing.assert_allclose(y, y_ref, rtol=1.6e-2, atol=1e-3)


if __name__ == "__main__":

    unittest.main()
