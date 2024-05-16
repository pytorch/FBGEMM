# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest

from typing import Tuple

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import torch

from hypothesis import given, settings, strategies as st

# Supported FP8 format is different on NV and AMD.
if torch.version.hip is not None:
    fp8_e4m3: torch.dtype = torch.float8_e4m3fnuz
    fp8_e5m2: torch.dtype = torch.float8_e5m2fnuz
else:
    fp8_e4m3: torch.dtype = torch.float8_e4m3fn
    fp8_e5m2: torch.dtype = torch.float8_e5m2

E4M3_MAX_POS: float = torch.finfo(fp8_e4m3).max
EPS: float = 1e-12
FP16_MAX_POS: float = torch.finfo(torch.float16).max


def fp8_row_quantize_ref(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Quantize an input tensor and return the fp8 tensor and its inverse scale.
    x_row_max = torch.max(torch.abs(x), dim=1).values
    max_scaling_factor = E4M3_MAX_POS * 512.0  # Match kernel logics
    scale = torch.Tensor(E4M3_MAX_POS / x_row_max).clamp(max=max_scaling_factor)
    xq = (x * scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    return xq, scale.reciprocal().to(torch.float32)


def fp8_col_quantize_ref(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Quantize an input tensor and return the fp8 tensor and its inverse scale.
    x_col_max = torch.max(torch.abs(x), dim=0).values
    max_scaling_factor = E4M3_MAX_POS * 512.0  # Match kernel logics
    scale = torch.Tensor(E4M3_MAX_POS / x_col_max).clamp(max=max_scaling_factor)
    xq = (x * scale.unsqueeze(0)).to(torch.float8_e4m3fn)
    return xq, scale.reciprocal().to(torch.float32)


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when MI300 or H100 is not available",
)
class FP8Tests(unittest.TestCase):
    def test_fp8_python(self) -> None:
        src_float = torch.randn(1000, 1000).cuda()
        src_float[0, 0] = 1e6
        fp8_152 = src_float.to(torch.float8_e5m2)
        fp8_143 = src_float.to(torch.float8_e4m3fn)
        assert len(fp8_152.float().unique()) <= 256
        assert len(fp8_143.float().unique()) <= 256

    @settings(deadline=None)
    @given(
        kernel=st.sampled_from(
            ["cutlass"] + (["cublas"] if torch.version.cuda else [])
        ),
        use_fast_accum=st.booleans() if torch.version.cuda else st.sampled_from([True]),
    )
    def test_f8f8bf16(self, kernel: str, use_fast_accum: bool) -> None:
        M = 128
        N = 128
        K = 256
        fp8_max = E4M3_MAX_POS
        x = torch.randn(size=(M, K), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(size=(N, K), dtype=torch.bfloat16, device="cuda") * 0.01

        x_max = x.abs().max()
        w_max = w.abs().max()

        x_scale = (x_max / fp8_max).float()
        w_scale = (w_max / fp8_max).float()

        xq = (x * fp8_max / x_max).to(fp8_e4m3)
        wq = (w * fp8_max / w_max).to(fp8_e4m3)

        if kernel == "cutlass":
            zq = torch.ops.fbgemm.f8f8bf16(
                xq, wq, (x_scale * w_scale).item(), use_fast_accum
            )
        else:
            zq = torch.ops.fbgemm.f8f8bf16_cublas(
                xq, wq, x_scale, w_scale, use_fast_accum
            )

        # Fake quant
        x = xq.bfloat16() * x_scale
        w = wq.bfloat16() * w_scale

        zq_ref = (x @ w.T).to(torch.bfloat16)

        torch.testing.assert_close(zq, zq_ref, atol=1.0e-3, rtol=1.0e-3)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: built in quantize ops not yet suported."
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
        Mode=st.sampled_from(["tensorwise", "rowwise"]),
        QType=st.sampled_from([torch.float8_e4m3fn, torch.float8_e5m2]),
        Bias=st.sampled_from([True, False]),
    )
    def test_quantize_fp8_matmul(
        self, B_T: int, D: int, HD_L: int, Mode: str, QType: torch.dtype, Bias: bool
    ) -> None:
        x = torch.randn(size=(B_T, D), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(size=(HD_L, D), dtype=torch.bfloat16, device="cuda") * 0.01
        bias = (
            torch.randn(size=(HD_L,), dtype=torch.bfloat16, device="cuda")
            if Bias
            else None
        )

        if Mode == "tensorwise":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
            wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
            zq = torch.ops.fbgemm.f8f8bf16(xq, wq, x_scale * w_scale)
            if bias is not None:
                zq += bias
        elif Mode == "rowwise":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x, output_dtype=QType)
            wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
            zq = torch.ops.fbgemm.f8f8bf16_rowwise(xq, wq, x_scale, w_scale, bias=bias)
        else:
            raise ValueError(f"Invalid mode {Mode}")

        zq_ref = (x @ w.T).to(torch.bfloat16)
        if bias is not None:
            zq_ref += bias
        torch.testing.assert_close(zq, zq_ref, atol=8.0e-2, rtol=8.0e-2)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: built in quantize ops not yet suported."
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        Mode=st.sampled_from(["tensorwise", "rowwise", "colwise"]),
    )
    def test_quantize_fp8_per_tensor_row_col(self, B_T: int, D: int, Mode: str) -> None:
        x = torch.randn(size=(B_T, D), dtype=torch.bfloat16, device="cuda") * 0.1
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        if Mode == "tensorwise":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
            x = (xq.float() / x_scale).bfloat16()  # Fake quantization
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
            x_max = x.abs().max()
            x_scale_ref = (x_max / fp8_max).float()
            xq_ref = (x * fp8_max / x_max).to(torch.float8_e4m3fn)
        elif Mode == "rowwise":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
            x = (xq.float() / x_scale.unsqueeze(1)).bfloat16()  # Fake quantization
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
            xq_ref, x_scale_ref = fp8_row_quantize_ref(x)
        elif Mode == "colwise" and str(torch.version.cuda) >= "12.1":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_col(x)
            x = (xq.float() / x_scale.unsqueeze(0)).bfloat16()  # Fake quantization
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_col(x)
            xq_ref, x_scale_ref = fp8_col_quantize_ref(x)
        elif Mode == "colwise":
            # quantize_fp8_per_col is not defined for CUDA < 12.1
            return
        else:
            raise ValueError(f"Invalid mode {Mode} (on CUDA {torch.version.cuda})")

        torch.testing.assert_close(xq.float(), xq_ref.float(), atol=5.0e-2, rtol=5.0e-2)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: built in quantize ops not yet suported."
    )
    @settings(deadline=None)
    @given(
        G_B=st.sampled_from([64, 32]),  # graph batch size
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
    )
    def test_tensor_with_nan(self, G_B: int, D: int, HD_L: int) -> None:
        x = torch.randn(size=(G_B, D), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(size=(HD_L, D), dtype=torch.bfloat16, device="cuda") * 0.01

        # batch size (B) which is <= graph batch size (G_B)
        B = int(G_B / 2)
        B_t = torch.tensor(B, dtype=torch.int64, device="cuda")

        x[B:, :] = float("nan")
        x_ref = torch.randn(size=(B, D), dtype=torch.bfloat16, device="cuda")
        x_ref[:B, :] = x[:B, :]

        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x, B_t, None)
        zq = torch.ops.fbgemm.f8f8bf16(xq, wq, x_scale * w_scale)

        # Fake quant
        x = xq[:B, :].bfloat16() * x_scale
        w = wq.bfloat16() * w_scale

        zq_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(zq[:B, :], zq_ref, atol=2.0e-3, rtol=2.0e-3)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: built in quantize ops not yet suported."
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
        UB=st.sampled_from([1000, 10000]),
        Mode=st.sampled_from(["tensorwise", "rowwise"]),
    )
    def test_quantize_fp8_per_tensor_with_ub(
        self, B_T: int, D: int, HD_L: int, UB: int, Mode: str
    ) -> None:
        x = torch.randn(size=(B_T, D), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(size=(HD_L, D), dtype=torch.bfloat16, device="cuda") * 0.01

        UB_t = torch.tensor(UB, dtype=torch.int64, device="cuda")

        if Mode == "tensorwise":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x, None, UB_t)
            wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
            zq = torch.ops.fbgemm.f8f8bf16(xq, wq, x_scale * w_scale)
            # Fake quant
            x = xq.bfloat16() * x_scale
            w = wq.bfloat16() * w_scale
        elif Mode == "rowwise":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x, None, UB_t)
            wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
            zq = torch.ops.fbgemm.f8f8bf16_rowwise(xq, wq, x_scale, w_scale)
            # Fake quant
            x = xq.bfloat16() * x_scale.unsqueeze(1)
            w = wq.bfloat16() * w_scale.unsqueeze(1)
        else:
            raise ValueError(f"Invalid mode {Mode}")

        zq_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(zq, zq_ref, atol=1.0e-3, rtol=1.0e-3)
