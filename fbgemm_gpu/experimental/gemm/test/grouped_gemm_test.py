# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import unittest
from typing import Tuple

import torch

if torch.cuda.is_available():
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import quantize_fp8_row
    from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import (
        grouped_gemm,
        grouped_gemm_fp8_rowwise,
    )


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when CUDA is not available",
)
class TestGroupedGEMM(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    # pyre-ignore [56]
    @unittest.skipIf(
        os.getenv("GITHUB_ENV") is not None,
        """This test fails on the GitHub runners: module 'triton.language' has no attribute 'async_task'""",
    )
    def test_grouped_gemm_fp8_rowwise(self) -> None:
        def _test_grouped_gemm_fp8_rowwise(
            shape: Tuple[int, int, int, int],
            device: torch.device,
            fast_accu: bool,
            use_warp_specialization: bool,
        ) -> None:
            G, M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
            m_ends, _ = torch.sort(
                torch.randint(
                    low=0, high=M, size=[G - 1], device=device, dtype=torch.int32
                )
                if M > 0
                else torch.zeros([G - 1], device=device, dtype=torch.int32)
            )
            m_ends = m_ends.tolist()
            m_starts = [0] + m_ends
            m_ends = m_ends + [M]
            m_sizes = torch.tensor(
                [m_ends[i] - m_starts[i] for i in range(G)], device=device
            ).to(torch.int32)

            a_fp8, a_scale = quantize_fp8_row(a)
            b_fp8, b_scale = quantize_fp8_row(b)

            result = grouped_gemm_fp8_rowwise(
                a_fp8,
                b_fp8,
                m_sizes,
                a_scale,
                b_scale,
                use_fast_accum=fast_accu,
                _use_warp_specialization=use_warp_specialization,
            )
            self.assertTrue(result.shape == (M, N))

            expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
            # Running baseline with quantization to exclude quantization error from the test as it has nothing to do with the correctness of the kernel implementation.
            for g in range(G):
                m_start = m_starts[g]
                m_end = m_ends[g]
                n_start = g * N
                n_end = (g + 1) * N

                expected_result[m_start:m_end, :] = (
                    a_fp8[m_start:m_end, :].to(torch.float32)
                    @ b_fp8[n_start:n_end, :].to(torch.float32).T
                    * a_scale[m_start:m_end][:, None]
                    * b_scale[n_start:n_end][None, :]
                ).to(torch.bfloat16)

            if M > 16384:
                torch.testing.assert_close(
                    result, expected_result, atol=5e-2, rtol=1.6e-2
                )
            else:
                torch.testing.assert_close(
                    result, expected_result, atol=2e-2, rtol=1.6e-2
                )

        for G in (1, 4, 16):
            for M in (0, 64, 512, 1000000):
                for fast_accu in (True, False):
                    for ws in (True, False):
                        logging.info(
                            f"Testing FP8 GMM with G={G}, M={M}, FastAccu={fast_accu}"
                        )
                        _test_grouped_gemm_fp8_rowwise(
                            (G, M, 256, 256),
                            torch.device("cuda"),
                            fast_accu=fast_accu,
                            use_warp_specialization=ws,
                        )

    @unittest.skipIf(  # pyre-ignore [56]
        os.getenv("GITHUB_ENV") is not None,
        """This test fails on the GitHub runners: "type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')""",
    )
    def test_grouped_gemm_bf16(self) -> None:
        def _test_grouped_gemm_bf16(
            shape: Tuple[int, int, int, int],
            device: torch.device,
            use_warp_specialization: bool,
        ) -> None:
            G, M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
            m_ends, _ = torch.sort(
                torch.randint(
                    low=0, high=M, size=[G - 1], device=device, dtype=torch.int32
                )
                if M > 0
                else torch.zeros([G - 1], device=device, dtype=torch.int32)
            )
            m_ends = m_ends.tolist()
            m_starts = [0] + m_ends
            m_ends = m_ends + [M]
            m_sizes = torch.tensor(
                [m_ends[i] - m_starts[i] for i in range(G)], device=device
            ).to(torch.int32)

            result = grouped_gemm(
                a,
                b,
                m_sizes,
                _use_warp_specialization=use_warp_specialization,
            )
            self.assertTrue(result.shape == (M, N))

            expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
            for g in range(G):
                m_start = m_starts[g]
                m_end = m_ends[g]
                expected_result[m_start:m_end, :] = (
                    a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].T
                )

            torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

        for G in (1, 4, 16):
            for M in (0, 64, 512, 1000000):
                for ws in (True, False):
                    logging.info(f"Testing BF16 GMM with G={G}, M={M}")
                    _test_grouped_gemm_bf16(
                        (G, M, 256, 256),
                        torch.device("cuda"),
                        use_warp_specialization=ws,
                    )
