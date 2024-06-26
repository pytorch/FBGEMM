# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Optional, Tuple

import torch

from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
    matmul_fp8_block,
    matmul_fp8_row,
    quantize_fp8_block,
    quantize_fp8_row,
    scale_fp8_row,
)


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp8Matmul(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_quantize_fp8_row(self) -> None:
        def _test_quantize_fp8_row(
            shape: Tuple[int, int],
            use_triton: bool,
            device: torch.device,
            output_device: Optional[torch.device] = None,
            use_scale_ub: bool = False,
        ) -> None:
            M, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)

            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device=device)
                if use_scale_ub
                else None
            )

            a_fp8, a_scale = quantize_fp8_row(
                a, scale_ub=scale_ub, use_triton=use_triton, output_device=output_device
            )

            # Undo scaling.
            a_torch = a_fp8.to(torch.bfloat16)
            a_torch *= a_scale[:, None]

            self.assertTrue(
                torch.allclose(
                    a.to(device=output_device), a_torch, atol=2e-1, rtol=1e-1
                )
            )

        _test_quantize_fp8_row((2, 3), True, torch.device("cuda"))
        _test_quantize_fp8_row((2, 3), True, torch.device("cuda"), use_scale_ub=True)
        _test_quantize_fp8_row((2, 3), False, torch.device("cpu"), torch.device("cuda"))
        _test_quantize_fp8_row(
            (2, 3), False, torch.device("cpu"), torch.device("cuda"), use_scale_ub=True
        )

    def test_scale_fp8_row(self) -> None:
        def _test_scale_fp8_row(
            shape: Tuple[int, int],
            device: torch.device,
        ) -> None:
            M, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)

            x_scale = torch.randn(M, dtype=torch.bfloat16, device=device)
            w_scale = torch.randn(K, dtype=torch.bfloat16, device=device)

            scaled_out = scale_fp8_row(a, x_scale, w_scale)

            # Compare with reference value.
            scaled_out_torch = a * x_scale[:, None] * w_scale[None, :]

            self.assertTrue(
                torch.allclose(
                    scaled_out,
                    scaled_out_torch,
                    atol=2e-1,
                    rtol=1e-1,
                )
            )

        _test_scale_fp8_row((2, 3), torch.device("cuda"))
        _test_scale_fp8_row((2, 3), torch.device("cpu"))

    def test_matmul_fp8_row(self) -> None:
        def _test_matmul_fp8_row(
            shape: Tuple[int, int, int], device: torch.device, fp8_fast_accum: bool
        ) -> None:
            M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)

            # Quantize inputs.
            a_fp8, a_scale = quantize_fp8_row(a)
            b_fp8, b_scale = quantize_fp8_row(b)

            result = matmul_fp8_row(
                a_fp8, b_fp8, a_scale, b_scale, fp8_fast_accum=fp8_fast_accum
            )
            self.assertTrue(result.shape == (M, N))

            expected_result = a @ b.T
            self.assertTrue(
                torch.allclose(result, expected_result, atol=2e-1, rtol=5e-2)
            )

        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), True)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), False)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cpu"), False)

    def test_quantize_fp8_block(self) -> None:
        def _test_quantize_fp8_block(
            shape: Tuple[int, int],
            block_shape: Tuple[int, int],
            use_scale_ub: bool = False,
        ) -> None:
            M, K = shape
            BLOCK_M, BLOCK_K = block_shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device="cuda")
                if use_scale_ub
                else None
            )

            a_fp8, a_scale = quantize_fp8_block(a, BLOCK_M, BLOCK_K, scale_ub=scale_ub)

            a_torch = a_fp8.to(torch.bfloat16)

            # Undo scaling.
            for i in range(0, M, BLOCK_M):
                for j in range(0, K, BLOCK_K):
                    block = a_torch[i : i + BLOCK_M, j : j + BLOCK_K]
                    scaling = a_scale[i // BLOCK_M, j // BLOCK_K]
                    scaled_block = block * scaling
                    a_torch[i : i + BLOCK_M, j : j + BLOCK_K] = scaled_block

            self.assertTrue(torch.allclose(a, a_torch, atol=2e-1, rtol=5e-2))

        _test_quantize_fp8_block((2, 4), (1, 2))
        _test_quantize_fp8_block((3, 6), (2, 8))
        _test_quantize_fp8_block((3, 6), (2, 8), use_scale_ub=True)

    def test_matmul_fp8_block(self) -> None:
        def _test_matmul_fp8_block(
            shape: Tuple[int, int, int],
            block_shape: Tuple[int, int, int],
            fp8_fast_accum: bool,
            device: str = "cuda",
        ) -> None:
            M, N, K = shape
            BLOCK_M, BLOCK_N, BLOCK_K = block_shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)

            # Quantize inputs.
            a_fp8, a_scale = quantize_fp8_block(
                a, BLOCK_M, BLOCK_K, output_device=torch.device("cuda")
            )
            b_fp8, b_scale = quantize_fp8_block(
                b, BLOCK_N, BLOCK_K, output_device=torch.device("cuda")
            )

            result = matmul_fp8_block(
                a_fp8,
                b_fp8,
                a_scale,
                b_scale,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                fp8_fast_accum=fp8_fast_accum,
            )
            self.assertTrue(result.shape == (M, N))

            expected_result = (a @ b.T).to("cuda")

            self.assertTrue(
                torch.allclose(result, expected_result, atol=1e2, rtol=5e-2)
            )

        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), True)
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), True)
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), False)
        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), False)
        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), True, "cpu")
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), True, "cpu")
