# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[16,21,53,56]

import logging
import unittest
from typing import Optional, Tuple

import torch
import triton  # noqa: F401

if torch.cuda.is_available():
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        triton_quantize_fp8_row,
    )
    from fbgemm_gpu.experimental.gen_ai.moe import silu_mul, silu_mul_quant

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
        torch.manual_seed(0)

        x = torch.randn([T, 2 * D], device="cuda", dtype=torch.bfloat16)
        x0 = x[:, :D]
        x1 = x[:, D:]

        if contiguous:
            x0 = x0.contiguous()
            x1 = x1.contiguous()

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

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
        "Skip when H100 is not available",
    )
    @given(
        T=st.sampled_from([1, 128, 2048, 4096, 16384]),
        D=st.sampled_from([5120, 7168]),
        scale_ub=st.sampled_from([None, 1200.00]),
        contiguous=st.sampled_from([True, False]),
        compiled=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_silu_mul_quant(
        self,
        T: int,
        D: int,
        scale_ub: Optional[float],
        contiguous: bool,
        compiled: bool,
    ) -> None:
        torch.manual_seed(2025)

        x = torch.randn([T, 2 * D], device="cuda", dtype=torch.bfloat16)

        x_sign = torch.sign(x)
        x_clamp = torch.clamp(torch.abs(x), 0.01, 2.0)
        x = x_sign * x_clamp
        x0 = x[:, :D]
        x1 = x[:, D:]

        if contiguous:
            x0 = x0.contiguous()
            x1 = x1.contiguous()

        if scale_ub is not None:
            scale_ub_tensor = torch.tensor(
                [scale_ub], device="cuda", dtype=torch.float32
            )
        else:
            scale_ub_tensor = None

        def fn() -> Tuple[torch.Tensor, torch.Tensor]:
            op = silu_mul_quant
            if compiled:
                op = torch.compile(op)
            return op(x0, x1, scale_ub_tensor)

        y_fp8, y_scale = fn()
        y = y_fp8.to(torch.float32) * y_scale[:, None]

        def ref_fn() -> Tuple[torch.Tensor, torch.Tensor]:
            x0_fp32 = x0.to(torch.float32)
            x1_fp32 = x1.to(torch.float32)
            y = x0_fp32 * torch.sigmoid(x0_fp32) * x1_fp32
            return triton_quantize_fp8_row(y, scale_ub_tensor)

        y_fp8_ref, y_scale_ref = ref_fn()
        y_ref = y_fp8_ref.to(torch.float32) * y_scale_ref[:, None]

        torch.testing.assert_allclose(y, y_ref, rtol=1e-1, atol=1e-3)


if __name__ == "__main__":

    unittest.main()
