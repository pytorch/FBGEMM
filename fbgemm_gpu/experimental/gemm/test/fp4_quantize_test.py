# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest
from typing import Tuple

import torch

from fbgemm_gpu.experimental.gemm.triton_gemm.fp4_quantize import (
    _to_blocked,
    triton_quantize_mx4_unpack,
    triton_rms_quantize_mx4_unpack,
    triton_silu_quantize_mx4_unpack,
)
from fbgemm_gpu.quantize_utils import fp32_to_mx4, RoundingMode


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp4Quantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_quantize_fp4(self) -> None:
        def _test_quantize_fp4(
            shape: Tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 32
            rounding_mode = RoundingMode.even
            packed_group_size = group_size // 2
            groups_per_row = math.ceil(N / group_size)
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            xq_ref, x_scale_ref = triton_quantize_mx4_unpack(
                x, group_size=group_size, rounding_mode=rounding_mode
            )
            xq_packed = fp32_to_mx4(
                x, group_size=group_size, rounding_mode=rounding_mode
            )

            xq = torch.empty([M, N // 2], device=x.device, dtype=torch.uint8)
            x_scale = torch.empty(
                [M, groups_per_row], device=x.device, dtype=torch.uint8
            )

            for i in range(groups_per_row):
                start_idx = i * (packed_group_size + 1)
                end_idx = start_idx + packed_group_size
                xq[:, i * packed_group_size : (i + 1) * packed_group_size] = xq_packed[
                    :, start_idx:end_idx
                ]
                x_scale[:, i] = xq_packed[:, end_idx]

            self.assertTrue(torch.equal(xq, xq_ref))
            self.assertTrue(
                torch.equal(_to_blocked(x_scale), x_scale_ref.view(torch.uint8))
            )

        _test_quantize_fp4((1, 128))
        _test_quantize_fp4((3, 512))
        _test_quantize_fp4((128, 1024))
        _test_quantize_fp4((4096, 10240))


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp4RmsQuantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_rms_quantize_fp4(self) -> None:
        def _test_rms_quantize_fp4(
            shape: Tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 32
            rounding_mode = RoundingMode.even
            packed_group_size = group_size // 2
            groups_per_row = math.ceil(N / group_size)
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            w = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            xq_ref, x_scale_ref = triton_rms_quantize_mx4_unpack(
                x, w, EPS=1e-5, group_size=group_size, rounding_mode=rounding_mode
            )

            intermediate = (
                x.to(torch.float32).reshape(-1, group_size)
                * torch.rsqrt(
                    torch.pow(x.to(torch.float32).reshape(-1, group_size), 2).mean(
                        dim=1
                    )
                    + 1e-5
                ).unsqueeze(1)
            ) * w.reshape(-1, group_size).to(torch.float32)

            intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
            xq_packed = fp32_to_mx4(
                intermediate, group_size=group_size, rounding_mode=rounding_mode
            )

            xq = torch.empty([M, N // 2], device=x.device, dtype=torch.uint8)
            x_scale = torch.empty(
                [M, groups_per_row], device=x.device, dtype=torch.uint8
            )

            for i in range(groups_per_row):
                start_idx = i * (packed_group_size + 1)
                end_idx = start_idx + packed_group_size
                xq[:, i * packed_group_size : (i + 1) * packed_group_size] = xq_packed[
                    :, start_idx:end_idx
                ]
                x_scale[:, i] = xq_packed[:, end_idx]

            self.assertTrue(torch.equal(xq, xq_ref))
            self.assertTrue(
                torch.equal(_to_blocked(x_scale), x_scale_ref.view(torch.uint8))
            )

        _test_rms_quantize_fp4((1, 32))
        _test_rms_quantize_fp4((1, 128))
        _test_rms_quantize_fp4((3, 512))
        _test_rms_quantize_fp4((128, 1024))
        # TODO: fix potential bug with large tensors
        # _test_rms_quantize_fp4((4096, 10240))


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp4SiluQuantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_silu_quantize_fp4(self) -> None:
        def _test_silu_quantize_fp4(
            shape: Tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 32
            rounding_mode = RoundingMode.even
            packed_group_size = group_size // 2
            groups_per_row = math.ceil(N / group_size)
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            w = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            xq_ref, x_scale_ref = triton_silu_quantize_mx4_unpack(
                x, w, group_size=group_size, rounding_mode=rounding_mode
            )
            intermediate = torch.nn.functional.silu(x.to(torch.float32)) * w.to(
                torch.float32
            )
            intermediate = intermediate.to(torch.bfloat16)
            xq_packed = fp32_to_mx4(
                intermediate, group_size=group_size, rounding_mode=rounding_mode
            )

            xq = torch.empty([M, N // 2], device=x.device, dtype=torch.uint8)
            x_scale = torch.empty(
                [M, groups_per_row], device=x.device, dtype=torch.uint8
            )

            for i in range(groups_per_row):
                start_idx = i * (packed_group_size + 1)
                end_idx = start_idx + packed_group_size
                xq[:, i * packed_group_size : (i + 1) * packed_group_size] = xq_packed[
                    :, start_idx:end_idx
                ]
                x_scale[:, i] = xq_packed[:, end_idx]

            self.assertTrue(torch.equal(xq, xq_ref))
            self.assertTrue(
                torch.equal(_to_blocked(x_scale), x_scale_ref.view(torch.uint8))
            )

        _test_silu_quantize_fp4((1, 128))
        _test_silu_quantize_fp4((3, 512))
        _test_silu_quantize_fp4((128, 1024))
        _test_silu_quantize_fp4((10240, 10240))
