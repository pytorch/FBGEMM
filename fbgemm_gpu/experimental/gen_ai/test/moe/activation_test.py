# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[16,21,53,56]

import logging
import os
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

running_on_github: bool = os.getenv("GITHUB_ENV") is not None

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

_MAX_SAMPLES: int = 100


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when no GPU is available.",
)
@unittest.skipIf(
    running_on_github and torch.version.hip is not None,
    "type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e5',)",
)
class ActivationTests(unittest.TestCase):
    """Test activation kernels."""

    @given(
        T=st.sampled_from([0, 1, 128, 2048, 4096, 16384]),
        D=st.sampled_from([5120, 7168]),
        contiguous=st.sampled_from([True, False]),
        partial=st.sampled_from([True, False]),
        compiled=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_silu_mul(
        self,
        T: int,
        D: int,
        contiguous: bool,
        partial: bool,
        compiled: bool,
    ) -> None:
        torch.manual_seed(0)

        x = torch.randn([T, 2 * D], device="cuda", dtype=torch.bfloat16)
        x0 = x[:, :D]
        x1 = x[:, D:]

        if contiguous:
            x0 = x0.contiguous()
            x1 = x1.contiguous()

        num_valid_tokens: int = T
        valid_token_count: Optional[torch.Tensor] = None
        if partial:
            num_valid_tokens = T // 2
            valid_token_count = torch.tensor(
                [num_valid_tokens], dtype=torch.int32, device="cuda"
            )

        def fn() -> torch.Tensor:
            op = silu_mul
            if compiled:
                op = torch.compile(op)
            return op(x0, x1, valid_token_count)

        y = fn()

        def ref_fn() -> torch.Tensor:
            x0_fp32 = x0.to(torch.float32)
            x1_fp32 = x1.to(torch.float32)
            return (x0_fp32 * torch.sigmoid(x0_fp32) * x1_fp32).to(torch.bfloat16)

        y_ref = ref_fn()

        torch.testing.assert_allclose(
            y[:num_valid_tokens], y_ref[:num_valid_tokens], rtol=1.6e-2, atol=1e-3
        )

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
        "Skip when H100 is not available",
    )
    @given(
        T=st.sampled_from([0, 1, 128, 2048, 4096, 16384]),
        D=st.sampled_from([5120, 7168]),
        scale_ub=st.sampled_from([None, 1200.00]),
        contiguous=st.sampled_from([True, False]),
        partial=st.sampled_from([True, False]),
        compiled=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    def test_silu_mul_quant(
        self,
        T: int,
        D: int,
        scale_ub: Optional[float],
        contiguous: bool,
        partial: bool,
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

        num_valid_tokens: int = T
        valid_token_count: Optional[torch.Tensor] = None
        if partial:
            num_valid_tokens = T // 2
            valid_token_count = torch.tensor(
                [num_valid_tokens], dtype=torch.int32, device="cuda"
            )

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
            return op(x0, x1, scale_ub_tensor, valid_token_count)

        y_fp8, y_scale = fn()
        y = y_fp8.to(torch.float32) * y_scale[:, None]

        def ref_fn() -> Tuple[torch.Tensor, torch.Tensor]:
            x0_fp32 = x0.to(torch.float32)
            x1_fp32 = x1.to(torch.float32)
            y_fp32 = x0_fp32 * torch.sigmoid(x0_fp32) * x1_fp32
            return triton_quantize_fp8_row(y_fp32, scale_ub_tensor, valid_token_count)

        y_fp8_ref, y_scale_ref = ref_fn()
        y_ref = y_fp8_ref.to(torch.float32) * y_scale_ref[:, None]

        torch.testing.assert_allclose(
            y[:num_valid_tokens], y_ref[:num_valid_tokens], rtol=1e-1, atol=1e-3
        )


if __name__ == "__main__":

    unittest.main()
