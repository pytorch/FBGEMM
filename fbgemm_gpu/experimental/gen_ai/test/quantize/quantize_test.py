# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest

from typing import List, Tuple

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import torch
import triton  # noqa: F401

if torch.cuda.is_available():
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        matmul_fp8_block,
        matmul_fp8_row,
        quantize_fp8_block,
        quantize_fp8_row,
    )

from hypothesis import given, settings, strategies as st

# Marlin is currently only supported internally at Meta.
try:
    if not torch.version.hip:
        from marlin.quantize import marlin_quantize

        torch.ops.load_library("//ai_codesign/gen_ai/marlin:marlin_ops")
        MARLIN_ENABLED = True
except ImportError:
    MARLIN_ENABLED = False

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

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)


def fp8_row_quantize_ref(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Quantize an input tensor and return the fp8 tensor and its inverse scale.
    x_row_max = torch.max(torch.abs(x), dim=1).values
    max_scaling_factor = E4M3_MAX_POS * 512.0  # Match kernel logics
    scale = torch.Tensor(E4M3_MAX_POS / x_row_max).clamp(max=max_scaling_factor)
    xq = (x * scale.unsqueeze(1)).to(fp8_e4m3)
    return xq, scale.reciprocal().to(torch.float32)


def fp8_col_quantize_ref(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Quantize an input tensor and return the fp8 tensor and its inverse scale.
    x_col_max = torch.max(torch.abs(x), dim=0).values
    max_scaling_factor = E4M3_MAX_POS * 512.0  # Match kernel logics
    scale = torch.Tensor(E4M3_MAX_POS / x_col_max).clamp(max=max_scaling_factor)
    xq = (x * scale.unsqueeze(0)).to(fp8_e4m3)
    return xq, scale.reciprocal().to(torch.float32)


def int4_row_quantize(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_bit = 4  # Number of target bits.
    to_quant = x.reshape(-1, group_size).to(torch.float)

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int

    zeros = min_val + scales * (2 ** (n_bit - 1))

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)

    # Recenter output and move to int8.
    out = (out - 2 ** (n_bit - 1)).to(dtype=torch.int8).reshape(x.shape)

    # Cutlass expects column major layout for scale and zero point,
    # so we transpose here and make them contiguous.
    scales = scales.view(x.shape[0], -1).t().contiguous()
    zeros = zeros.view(x.shape[0], -1).t().contiguous()

    return out, scales, zeros


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    # Given int8 x, pack adjacent int4 values into a single int8.
    low_x = x[:, ::2]
    high_x = x[:, 1::2]

    # High bits need to left shift, this also masks off extra bits.
    high_x = torch.bitwise_left_shift(high_x, 4)
    # Low bits need to have sign bits removed.
    low_x = torch.bitwise_and(low_x, 0xF)

    # Recombine into a single value with bitwise or.
    return torch.bitwise_or(low_x, high_x).contiguous()


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when no GPU is available. This test is only for GPU.",
)
@unittest.skipIf(open_source, "Temporarily disabled in OSS.")
class FP8TorchExportTests(unittest.TestCase):
    """Test that FP8 ops can be exported."""

    def test_quantize_export(self) -> None:
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # let's go trough all our cuda quantize operations here
                _, _ = torch.ops.fbgemm.quantize_fp8_per_row(x)
                _, _ = torch.ops.fbgemm.quantize_fp8_per_col(x)
                o, _ = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
                return o

        model = TestModule().cuda()
        # bf16 required here
        _ = torch.export.export(
            model, (torch.randn(32, 32).to(torch.bfloat16).cuda(),), strict=True
        )

    def test_f8f8bf16_export(self) -> None:
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
                M, K = xq.shape
                N, _ = wq.shape
                row_scale = torch.randn(M).cuda()
                col_scale = torch.randn(N).cuda()
                block_scale = torch.randn(M // 128, K // 128).cuda()
                _ = torch.ops.fbgemm.f8f8bf16_blockwise(
                    xq, wq, block_scale, block_scale
                )
                _ = torch.ops.fbgemm.f8f8bf16_tensorwise(xq, wq, 1.0)
                o = torch.ops.fbgemm.f8f8bf16_rowwise(xq, wq, row_scale, col_scale)
                return o

        model = TestModule().cuda()
        M, N, K = 256, 256, 256
        fp8_dtype = torch.float8_e4m3fn
        if torch.version.hip:
            fp8_dtype = torch.float8_e4m3fnuz
        xq = torch.randn(M, K).to(fp8_dtype).cuda()
        wq = torch.randn(N, K).to(fp8_dtype).cuda()
        _ = torch.export.export(model, (xq, wq), strict=True)


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when MI300 or H100 is not available",
)
class FP8Tests(unittest.TestCase):
    def test_fp8_python(self) -> None:
        src_float = torch.randn(1000, 1000).cuda()
        src_float[0, 0] = 1e6
        fp8_152 = src_float.to(fp8_e5m2)
        fp8_143 = src_float.to(fp8_e4m3)
        assert len(fp8_152.float().unique()) <= 256
        assert len(fp8_143.float().unique()) <= 256

    @unittest.skipIf(not torch.version.cuda, "Skip on AMD: f8f8bf16 not yet suported.")
    @settings(deadline=None)
    @given(
        kernel=st.sampled_from(["cutlass", "cublas"]),
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
            zq = torch.ops.fbgemm.f8f8bf16(xq, wq, x_scale * w_scale, use_fast_accum)
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
        ((not torch.version.cuda) and (not torch.version.hip)),
        "Skip if no GPU is present.",
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([0, 2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
        Mode=st.sampled_from(
            ["rowwise", "blockwise"]
            + (["tensorwise_broadcast", "tensorwise"] if torch.version.cuda else [])
        ),
        QType=st.sampled_from([fp8_e4m3, fp8_e5m2]),
        Bias=st.sampled_from([True, False]),
        CudaGraph=st.sampled_from([True, False]),
        UseTriton=st.sampled_from([False] + ([True] if torch.version.cuda else [])),
        InputMultiDim=st.booleans(),
    )
    def test_quantize_fp8_matmul(
        self,
        B_T: int,
        D: int,
        HD_L: int,
        Mode: str,
        QType: torch.dtype,
        Bias: bool,
        CudaGraph: bool,
        UseTriton: bool,
        InputMultiDim: bool,
    ) -> None:
        if InputMultiDim and not torch.version.hip:
            x = torch.randn(size=(3, B_T, D), dtype=torch.bfloat16, device="cuda") * 0.1
        else:
            x = torch.randn(size=(B_T, D), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(size=(HD_L, D), dtype=torch.bfloat16, device="cuda") * 0.01
        bias = (
            torch.randn(size=(HD_L,), dtype=torch.bfloat16, device="cuda")
            if Bias
            else None
        )

        if Mode == "tensorwise":
            if CudaGraph:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
                    wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
                    zq = torch.ops.fbgemm.f8f8bf16(xq, wq, x_scale * w_scale)
                    if bias is not None:
                        zq += bias
                g.replay()
            else:
                xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
                wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
                zq = torch.ops.fbgemm.f8f8bf16(xq, wq, x_scale * w_scale)
                if bias is not None:
                    zq += bias
        elif Mode == "tensorwise_broadcast":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
            wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
            x_scale = x_scale.item()
            w_scale = w_scale.item()
            if CudaGraph:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    zq = torch.ops.fbgemm.f8f8bf16_tensorwise(xq, wq, x_scale * w_scale)
                    if bias is not None:
                        zq += bias
                g.replay()
            else:
                zq = torch.ops.fbgemm.f8f8bf16_tensorwise(xq, wq, x_scale * w_scale)
                if bias is not None:
                    zq += bias
        elif Mode == "rowwise":
            if CudaGraph:
                # Warm up triton functions before cuda graph.
                xq, x_scale = quantize_fp8_row(x)
                wq, w_scale = quantize_fp8_row(w)
                if UseTriton and torch.version.cuda:
                    zq = matmul_fp8_row(xq, wq, x_scale, w_scale)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    if torch.version.cuda:
                        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
                            x, output_dtype=QType
                        )
                        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
                    else:
                        xq, x_scale = quantize_fp8_row(x)
                        wq, w_scale = quantize_fp8_row(w)
                    if UseTriton and torch.version.cuda:
                        zq = matmul_fp8_row(xq, wq, x_scale, w_scale)
                        if bias is not None:
                            zq += bias
                    else:
                        zq = torch.ops.fbgemm.f8f8bf16_rowwise(
                            xq,
                            wq,
                            x_scale,
                            w_scale,
                            bias=bias if torch.version.cuda else None,
                        )
                        # Bias fusion not yet supported on AMD.
                        if bias is not None and torch.version.hip:
                            zq += bias
                g.replay()
            else:
                if torch.version.cuda:
                    xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
                        x, output_dtype=QType
                    )
                    wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
                else:
                    xq, x_scale = quantize_fp8_row(x)
                    wq, w_scale = quantize_fp8_row(w)
                if UseTriton and torch.version.cuda:
                    zq = matmul_fp8_row(xq, wq, x_scale, w_scale)
                    if bias is not None:
                        zq += bias
                else:
                    zq = torch.ops.fbgemm.f8f8bf16_rowwise(
                        xq,
                        wq,
                        x_scale,
                        w_scale,
                        bias=bias if torch.version.cuda else None,
                    )
                    # Bias fusion not yet supported on AMD.
                    if bias is not None and torch.version.hip:
                        zq += bias
        elif Mode == "blockwise":
            block_m = block_n = block_k = 128
            output_device = torch.device("cuda")
            if CudaGraph:
                #  Need a warmup to compile the Triton kernel before cuda graph

                wq, w_scale = quantize_fp8_block(
                    w, block_n, block_k, output_device=output_device
                )
                xq, x_scale = quantize_fp8_block(x, block_m, block_k)
                if UseTriton:
                    zq = matmul_fp8_block(
                        xq,
                        wq,
                        x_scale,
                        w_scale,
                        block_m,
                        block_n,
                        block_k,
                        fp8_fast_accum=True,
                    )
                else:
                    zq = torch.ops.fbgemm.f8f8bf16_blockwise(
                        xq, wq, x_scale, w_scale, block_m, block_n, block_k
                    )
                if bias is not None:
                    zq += bias

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    wq, w_scale = quantize_fp8_block(
                        w, block_n, block_k, output_device=output_device
                    )
                    xq, x_scale = quantize_fp8_block(x, block_m, block_k)
                    if UseTriton:
                        zq = matmul_fp8_block(
                            xq,
                            wq,
                            x_scale,
                            w_scale,
                            block_m,
                            block_n,
                            block_k,
                            fp8_fast_accum=True,
                        )
                    else:
                        zq = torch.ops.fbgemm.f8f8bf16_blockwise(
                            xq, wq, x_scale, w_scale, block_m, block_n, block_k
                        )
                    if bias is not None:
                        zq += bias
                g.replay()
            else:
                wq, w_scale = quantize_fp8_block(
                    w, block_n, block_k, output_device=output_device
                )
                xq, x_scale = quantize_fp8_block(x, block_m, block_k)
                if UseTriton:
                    zq = matmul_fp8_block(
                        xq,
                        wq,
                        x_scale,
                        w_scale,
                        block_m,
                        block_n,
                        block_k,
                        fp8_fast_accum=True,
                    )
                else:
                    zq = torch.ops.fbgemm.f8f8bf16_blockwise(
                        xq, wq, x_scale, w_scale, block_m, block_n, block_k
                    )
                if bias is not None:
                    zq += bias
        else:
            raise ValueError(f"Invalid mode {Mode}")

        zq_ref = (x @ w.T).to(torch.bfloat16)
        if bias is not None:
            zq_ref += bias

        # Blockwise seems to have slightly more noisy outputs.
        # Special case correctness to avoid flakiness.
        if Mode == "blockwise":
            atol = 1.3e-1
            rtol = 1.3e-1
        else:
            atol = 9.0e-2
            rtol = 9.0e-2
        torch.testing.assert_close(zq, zq_ref, atol=atol, rtol=rtol)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: built in quantize ops not yet suported."
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
        QType=st.sampled_from([torch.float8_e4m3fn, torch.float8_e5m2]),
        CudaGraph=st.sampled_from([True, False]),
    )
    def test_quantize_int4_fp8_matmul(
        self,
        B_T: int,
        D: int,
        HD_L: int,
        QType: torch.dtype,
        CudaGraph: bool,
    ) -> None:
        x = torch.randn(size=(B_T, D), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(size=(HD_L, D), dtype=torch.bfloat16, device="cuda") * 0.01

        wq, w_scale, w_zp = int4_row_quantize(w, 128)
        wq = pack_int4(wq).contiguous().to(device="cuda")
        w_scale = w_scale.contiguous().to(device="cuda")
        w_zp = w_zp.contiguous().to(device="cuda")

        if CudaGraph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
                zq = torch.ops.fbgemm.f8i4bf16_rowwise(xq, wq, x_scale, w_scale, w_zp)
            g.replay()
        else:
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
            zq = torch.ops.fbgemm.f8i4bf16_rowwise(xq, wq, x_scale, w_scale, w_zp)

        zq_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(zq, zq_ref, atol=8.0e-2, rtol=8.0e-2)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: built in quantize ops not yet suported."
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
        CudaGraph=st.sampled_from([True, False]),
    )
    def test_quantize_int4_bf16_matmul(
        self,
        B_T: int,
        D: int,
        HD_L: int,
        CudaGraph: bool,
    ) -> None:
        x = torch.randn(size=(B_T, D), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(size=(HD_L, D), dtype=torch.bfloat16, device="cuda") * 0.01

        wq, w_scale, w_zp = int4_row_quantize(w, 128)
        wq = pack_int4(wq).contiguous().to(device="cuda")
        w_scale = w_scale.contiguous().to(device="cuda")
        w_zp = w_zp.contiguous().to(device="cuda")

        if CudaGraph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                zq = torch.ops.fbgemm.bf16i4bf16_rowwise(x, wq, w_scale, w_zp)
            g.replay()
        else:
            zq = torch.ops.fbgemm.bf16i4bf16_rowwise(x, wq, w_scale, w_zp)

        zq_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(zq, zq_ref, atol=1.0e-1, rtol=8.0e-2)

    @unittest.skipIf(
        not torch.version.cuda and torch.version.hip < "6.2",
        "Skip on AMD with < RoCM 6.2",
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        Mode=st.sampled_from(["tensorwise", "rowwise", "colwise"]),
        stochastic_rounding=st.booleans(),
    )
    def test_quantize_fp8_per_tensor_row_col(
        self, B_T: int, D: int, Mode: str, stochastic_rounding: bool
    ) -> None:
        dtype = torch.bfloat16
        x = torch.randn(size=(B_T, D), dtype=dtype, device="cuda") * 0.1
        fp8_max = torch.finfo(fp8_e4m3).max

        if Mode == "tensorwise":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
            x = (xq.float() / x_scale).to(dtype)  # Fake quantization
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(
                x,
                stochastic_rounding=stochastic_rounding,
            )
            x_max = x.abs().max()
            x_scale_ref = (x_max / fp8_max).float()
            xq_ref = (x * fp8_max / x_max).to(fp8_e4m3)
        elif Mode == "rowwise":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
            x = (xq.float() / x_scale.unsqueeze(1)).to(dtype)  # Fake quantization
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
                x, stochastic_rounding=stochastic_rounding
            )
            xq_ref, x_scale_ref = fp8_row_quantize_ref(x)
        elif Mode == "colwise" and str(torch.version.cuda) >= "12.1":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_col(x)
            x = (xq.float() / x_scale.unsqueeze(0)).to(dtype)  # Fake quantization
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_col(x)
            xq_ref, x_scale_ref = fp8_col_quantize_ref(x)
        elif Mode == "colwise":
            # quantize_fp8_per_col is not defined for CUDA < 12.1
            return
        else:
            raise ValueError(f"Invalid mode {Mode} (on CUDA {torch.version.cuda})")

        # For stochastic_rounding, |(fl(x) - x)/x| < \epsilon. (2.4a in https://epubs.siam.org/doi/epdf/10.1137/20M1334796).
        # Machine epsilon for E4M3 should be 2^{-3} and for E5M2 is 2^{-2}.
        tol = 0.125 if stochastic_rounding else 5.0e-2
        torch.testing.assert_close(xq.float(), xq_ref.float(), atol=tol, rtol=tol)

    @unittest.skipIf(
        not torch.version.cuda and torch.version.hip < "6.2",
        "Skip on AMD with < RoCM 6.2",
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
    )
    def test_quantize_fp8_per_tensor_sr(self, B_T: int, D: int) -> None:
        import random

        rand_val = random.random()  # [0,1) random values
        x = torch.full((B_T, D), rand_val, dtype=torch.bfloat16, device="cuda")
        x[0, 0] = 1.0  # first element = 1 to set up the x_scale
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(
            x,
            stochastic_rounding=True,
        )
        xq_fp32 = xq.float() * x_scale.float()
        val, cnts = torch.unique(xq_fp32, return_counts=True)

        assert (
            len(val.tolist()) == 2 + 1
        ), f"fp8 quantization should have 3 unique values: {len(val.tolist())} unique values"

        mean_val = (torch.sum(xq_fp32) - x[0, 0]) / (x.numel() - 1)
        rtol = 0.125
        # atol = 1 / (1 << 9)
        atol = 0
        # verify that elementwise the SR is close to the original value
        torch.testing.assert_close(xq_fp32, x.float(), atol=atol, rtol=rtol)
        # verify the mean value of SR of is close to the original value
        torch.testing.assert_close(mean_val, x[0, -1].float(), atol=atol, rtol=rtol)

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

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: BMM ops are not yet suported."
    )
    @settings(deadline=None)
    @given(
        B=st.sampled_from([1, 4]),
        M=st.sampled_from([2048, 4096]),
        N=st.sampled_from([128, 256]),
        K=st.sampled_from([256, 512]),
        use_loopover=st.sampled_from([True, False]),
    )
    def test_fp8_batched_gemm(
        self,
        B: int,
        M: int,
        N: int,
        K: int,
        use_loopover: bool,
    ) -> None:
        x = torch.rand(size=(B, M, K), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.rand(size=(B, N, K), dtype=torch.bfloat16, device="cuda") * 0.01

        xq, x_scale = quantize_fp8_row(x)
        x_scale = x_scale.view(B, -1)
        assert x_scale.shape == (B, M)
        wq, w_scale = quantize_fp8_row(w)
        w_scale = w_scale.view(B, -1)
        assert w_scale.shape == (B, N)

        def fp8_loopover_bmm(
            xq: List[torch.Tensor],
            wq: List[torch.Tensor],
            x_scale: List[torch.Tensor],
            w_scale: List[torch.Tensor],
        ) -> torch.Tensor:
            B = len(xq)
            M = xq[0].shape[0]
            N = wq[0].shape[0]
            y = torch.empty((B, M, N), dtype=torch.bfloat16, device=xq[0].device)
            for i in range(B):
                y[i] = torch.ops.fbgemm.f8f8bf16_rowwise(
                    xq[i], wq[i], x_scale[i], w_scale[i]
                )
            return y

        y_ref = torch.bmm(x, w.transpose(1, 2))
        if use_loopover:
            y_fp8 = fp8_loopover_bmm(xq, wq, x_scale, w_scale)
        else:
            y_fp8 = torch.ops.fbgemm.f8f8bf16_rowwise_batched(xq, wq, x_scale, w_scale)
        torch.testing.assert_close(y_ref, y_fp8, atol=8.0e-2, rtol=8.0e-2)

    @unittest.skipIf(
        not torch.version.cuda and torch.version.hip < "6.2",
        "Skip on AMD with < RoCM 6.2",
    )
    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 5, 16]),
        M=st.sampled_from([2048, 3584]),
        N=st.sampled_from([1024, 6144]),
        K=st.sampled_from([512, 3584]),
        use_cudagraph=st.booleans(),
        use_padding_zeros=st.booleans(),
        return_stacked=st.booleans(),
    )
    def test_fp8_grouped_gemm(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        use_cudagraph: bool,
        use_padding_zeros: bool,
        return_stacked: bool,
    ) -> None:
        ms = (
            torch.randint(
                (258 // 64) + 1 if use_padding_zeros else 1,
                (M // 64) + 1,
                (G,),
                dtype=torch.int,
            )
            * 64
        )
        # When using padding or the dynamic kernel, Ns and Ks should be fixed.
        if use_padding_zeros or return_stacked:
            ns = [N] * G
            ks = [K] * G
        # Otherwise, any value is supported.
        else:
            # AMD requires N and K >= 512.
            ns = torch.randint(512 // 64, (N // 64) + 1, (G,), dtype=torch.int) * 64
            ks = torch.randint(512 // 64, (K // 64) + 1, (G,), dtype=torch.int) * 64

        x_group = []
        w_group = []
        xq_group = []
        wq_group = []
        x_scale_group = []
        w_scale_group = []
        zero_start_index_M = None

        # If padding, mark where zeros start for each input.
        if use_padding_zeros:
            zero_start_index_M = torch.tensor(ms, dtype=torch.long, device="cuda")

        for _, (m, n, k) in enumerate(zip(ms, ns, ks)):
            x = torch.rand(size=(m, k), dtype=torch.bfloat16, device="cuda")
            w = torch.rand(size=(n, k), dtype=torch.bfloat16, device="cuda")

            if use_padding_zeros:
                # When padding, all x values are made to have the same M.
                x = torch.nn.functional.pad(x, (0, 0, 0, max(ms) - m), value=0)

            xq, x_scale = quantize_fp8_row(x)
            wq, w_scale = quantize_fp8_row(w)
            x_group.append(x)
            w_group.append(w)
            xq_group.append(xq)
            wq_group.append(wq)
            x_scale_group.append(x_scale)
            w_scale_group.append(w_scale)

        # Make inputs contiguous in memory, this simulates the typical MOE use-case.
        if use_padding_zeros:
            x_group = torch.unbind(torch.stack(x_group, dim=0).contiguous())
            w_group = torch.unbind(torch.stack(w_group, dim=0).contiguous())
            xq_group = torch.stack(xq_group, dim=0).contiguous()
            wq_group = torch.stack(wq_group, dim=0).contiguous()
            x_scale_group = torch.stack(x_scale_group, dim=0).contiguous()
            w_scale_group = torch.stack(w_scale_group, dim=0).contiguous()
        elif return_stacked:
            xq_group = torch.cat(xq_group, dim=0).contiguous()
            wq_group = torch.stack(wq_group, dim=0).contiguous()
            x_scale_group = torch.cat(x_scale_group, dim=0).contiguous()
            w_scale_group = torch.stack(w_scale_group, dim=0).contiguous()

        # FP8 grouped gemm kernel
        if use_padding_zeros:
            fp8_op = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_dynamic
            fp8_args = [
                xq_group,
                wq_group,
                x_scale_group,
                w_scale_group,
                zero_start_index_M,
            ]
        elif return_stacked:
            fp8_op = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_stacked
            M_offsets = ms.to(device="cuda", dtype=torch.int64)
            fp8_args = [xq_group, wq_group, x_scale_group, w_scale_group, M_offsets]
        else:
            fp8_op = torch.ops.fbgemm.f8f8bf16_rowwise_grouped
            fp8_args = [xq_group, wq_group, x_scale_group, w_scale_group]

        if use_cudagraph:
            # warmup
            fp8_op(*fp8_args)
            # With cudagraph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y_fp8_group = fp8_op(*fp8_args)
            g.replay()
        else:
            y_fp8_group = fp8_op(*fp8_args)

        # Massage output into proper format.
        if not isinstance(y_fp8_group, (tuple, list)):
            if y_fp8_group.ndim == 2:
                y_fp8_group = torch.split(y_fp8_group, tuple(ms.tolist()), dim=0)
            else:
                y_fp8_group = torch.unbind(y_fp8_group)

        # BF16 grouped gemm kernel
        bf16_args = (
            [x_group, w_group, zero_start_index_M if use_padding_zeros else None]
            if return_stacked
            else [x_group, w_group]
        )
        bf16_op = (
            torch.ops.fbgemm.bf16bf16bf16_grouped_dynamic
            if return_stacked
            else torch.ops.fbgemm.bf16bf16bf16_grouped
        )
        if use_cudagraph:
            # warmup
            bf16_op(*bf16_args)
            # With cudagraph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y_bf16_group = bf16_op(*bf16_args)
            g.replay()
        else:
            y_bf16_group = bf16_op(*bf16_args)

        # View output as list if needed.
        if not isinstance(y_bf16_group, (tuple, list)):
            if y_bf16_group.ndim == 2:
                y_bf16_group = torch.split(y_bf16_group, tuple(ms.tolist()), dim=0)
            else:
                y_bf16_group = torch.unbind(y_bf16_group)

        # BF16 loopover gemm reference
        y_group_ref = []
        for i in range(len(x_group)):
            y = torch.matmul(x_group[i], w_group[i].t())
            y_group_ref.append(y)

        # Assert FP8 outputs
        for i in range(len(y_group_ref)):
            torch.testing.assert_close(
                y_fp8_group[i], y_group_ref[i], atol=8.0e-2, rtol=2.0e-1
            )

        # Assert BF16 outputs
        for i in range(len(y_group_ref)):
            torch.testing.assert_close(
                y_bf16_group[i], y_group_ref[i], atol=8.0e-2, rtol=8.0e-2
            )

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: GMM ops are not yet suported."
    )
    @settings(deadline=None)
    @given(
        G=st.sampled_from([4, 5]),
        M=st.sampled_from([2048, 3584]),
        N=st.sampled_from([1024, 6144]),
        K=st.sampled_from([512, 3584]),
        use_cudagraph=st.sampled_from([True, False]),
        use_padding_zeros=st.sampled_from([True, False]),
    )
    def test_bf16_grouped_gemm(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        use_cudagraph: bool,
        use_padding_zeros: bool,
    ) -> None:
        G = 16
        M = 64
        N = 1024
        K = 5120
        xs = torch.rand(size=(G, M, K), dtype=torch.bfloat16, device="cuda")
        ws = torch.rand(size=(G, N, K), dtype=torch.bfloat16, device="cuda")

        x_group = [x.squeeze() for x in xs.split(1, dim=0)]
        w_group = [w.squeeze() for w in ws.split(1, dim=0)]

        zero_start_index_M = None

        use_padding_zeros = True
        if use_padding_zeros:
            zero_start_index_M = torch.randint(
                1,
                M,
                (G,),
                device="cuda",
            )
            for i in range(len(x_group)):
                x_group[i][zero_start_index_M[i] :, :] = 0

        bf16_op = (
            torch.ops.fbgemm.bf16bf16bf16_grouped_dynamic
            if use_padding_zeros
            else torch.ops.fbgemm.bf16bf16bf16_grouped
        )
        bf16_args = (
            (x_group, w_group, zero_start_index_M)
            if use_padding_zeros
            else (x_group, w_group)
        )

        # BF16 grouped gemm kernel
        if use_cudagraph:
            # warmup
            bf16_op(*bf16_args)
            # With cudagraph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y_bf16_group = bf16_op(*bf16_args)
            g.replay()
        else:
            y_bf16_group = bf16_op(*bf16_args)

        # BF16 loopover gemm reference
        y_group_ref = torch.bmm(xs, ws.transpose(1, 2))

        torch.testing.assert_close(
            y_group_ref, y_bf16_group.view([G, M, N]), atol=8.0e-2, rtol=8.0e-2
        )

    @unittest.skipIf(torch.version.hip, "Skip on AMD: Marlin not yet suported.")
    @settings(deadline=None)
    @given(
        B=st.sampled_from([1, 4]),
        M=st.sampled_from([2048, 4096]),
        N=st.sampled_from([256, 512]),
        K=st.sampled_from([256, 512]),
        use_loopover=st.sampled_from([True, False]),
    )
    def test_int4_batched_gemm(
        self,
        B: int,
        M: int,
        N: int,
        K: int,
        use_loopover: bool,
    ) -> None:
        if not MARLIN_ENABLED:
            return
        x = torch.rand(size=(B, M, K), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.rand(size=(B, N, K), dtype=torch.bfloat16, device="cuda") * 0.01

        wq = []
        w_scale = []
        group_size = 128

        if use_loopover:
            for i in range(B):
                _, wq_, w_scale_ = marlin_quantize(
                    w[i].cuda().t().contiguous(), group_size
                )
                wq.append(wq_)
                w_scale.append(w_scale_)
            wq = torch.stack(wq)
            w_scale = torch.stack(w_scale)

            def int4_loopover_bmm(
                x: torch.Tensor,
                wq: torch.Tensor,
                w_scale: torch.Tensor,
            ) -> torch.Tensor:
                B = x.shape[0]
                M = x.shape[1]
                N = w_scale.shape[2]
                y = torch.empty((B, M, N), dtype=torch.bfloat16, device=x[0].device)
                for i in range(B):
                    y[i] = torch.ops.marlin.marlin_gemm(x[i], wq[i], w_scale[i])
                return y

            y_int4 = int4_loopover_bmm(x, wq, w_scale)
        else:
            w_zp = []
            for i in range(B):
                wq_, w_scale_, w_zp_ = int4_row_quantize(w[i], group_size)

                wq_ = pack_int4(wq_).contiguous().to(device="cuda")
                w_scale_ = w_scale_.contiguous().to(device="cuda")
                w_zp_ = w_zp_.contiguous().to(device="cuda")
                wq.append(wq_)
                w_scale.append(w_scale_)
                w_zp.append(w_zp_)
            wq = torch.stack(wq)
            w_scale = torch.stack(w_scale).view(-1, N)
            w_zp = torch.stack(w_zp).view(-1, N)
            y_int4 = torch.ops.fbgemm.bf16i4bf16_rowwise_batched(x, wq, w_scale, w_zp)

        y_ref = torch.bmm(x, w.transpose(1, 2))
        torch.testing.assert_close(y_ref, y_int4, atol=1e-1, rtol=8.0e-2)

    @unittest.skipIf(
        ((not torch.version.cuda) and (not torch.version.hip)),
        "Skip if no GPU is present.",
    )
    @unittest.skipIf(open_source, "Temporarily disabled in OSS.")
    def test_quantize_compile(self) -> None:
        # Test that quantize operators can be torch compiled.
        # Correctness is covered in other tests, we just want to make sure
        # compile doesnt fail.
        if torch.version.hip:
            fp8_dtype = torch.float8_e4m3fnuz
        else:
            fp8_dtype = torch.float8_e4m3fn
        # Initialize tensors for testing.
        M, N, K = 256, 256, 256
        X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        XQ = torch.randn(M, K, device="cuda").to(fp8_dtype)
        WQ = torch.randn(N, K, device="cuda").to(fp8_dtype)
        output = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        row_scale = torch.randn(M, device="cuda")
        col_scale = torch.randn(N, device="cuda")
        block_scale = torch.randn(M // 128, K // 128, device="cuda")
        tensor_scale = torch.tensor(1.0, device="cuda")

        # Run various compiled quantize ops.
        # Quantize ops.
        torch.compile(torch.ops.fbgemm.quantize_fp8_per_tensor)(X)
        torch.compile(torch.ops.fbgemm.quantize_fp8_per_row)(X)
        torch.compile(torch.ops.fbgemm.quantize_fp8_per_col)(X)

        # GEMM ops.
        torch.compile(torch.ops.fbgemm.f8f8bf16_blockwise)(
            XQ, WQ, block_scale, block_scale
        )
        torch.compile(torch.ops.fbgemm.f8f8bf16_tensorwise)(XQ, WQ, 1.0)
        torch.compile(torch.ops.fbgemm.f8f8bf16_rowwise)(XQ, WQ, row_scale, col_scale)

        # Check that preallocated output writing is correct.
        torch.compile(torch.ops.fbgemm.f8f8bf16_rowwise_out)(
            XQ, WQ, row_scale, col_scale, output
        )
        torch.testing.assert_close(
            output, torch.ops.fbgemm.f8f8bf16_rowwise(XQ, WQ, row_scale, col_scale)
        )

        # These ops are only supported on cuda for now.
        if torch.version.cuda:
            torch.compile(torch.ops.fbgemm.i8i8bf16)(
                XQ.view(torch.int8), WQ.view(torch.int8), 1.0, 1
            )
            torch.compile(torch.ops.fbgemm.f8f8bf16)(XQ, WQ, tensor_scale)
            torch.compile(torch.ops.fbgemm.f8f8bf16_cublas)(XQ, WQ)
            torch.compile(torch.ops.fbgemm.f8f8bf16_rowwise_batched)(
                XQ.view(1, M, K),
                WQ.view(1, N, K),
                row_scale.view(1, M),
                col_scale.view(1, N),
            )
            torch.compile(torch.ops.fbgemm.f8i4bf16_rowwise)(
                XQ,
                WQ[:, ::2].view(torch.int8).contiguous(),
                row_scale,
                block_scale[0],
                block_scale[0],
            )
            torch.compile(torch.ops.fbgemm.bf16i4bf16_rowwise)(
                X,
                WQ[:, ::2].view(torch.int8).contiguous(),
                block_scale[0],
                block_scale[0],
            )
            torch.compile(torch.ops.fbgemm.bf16i4bf16_rowwise_batched)(
                X.view(1, M, K),
                WQ[:, ::2].view(1, N, K // 2).view(torch.int8).contiguous(),
                block_scale[0],
                block_scale[0],
            )
            # test bf16_fast_gemv is torch compileable
            W_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            torch.compile(torch.ops.fbgemm.bf16_fast_gemv)(X, W_bf16)
            # test fp8fp8bf16_fast_gemv is torch compileable
            torch.compile(torch.ops.fbgemm.fp8fp8bf16_fast_gemv)(XQ, WQ, tensor_scale)
            # test bf16bf16bf16_fast_gemv is torch compileable
            torch.compile(torch.ops.fbgemm.bf16fp8bf16_fast_gemv)(X, WQ, tensor_scale)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: fast gemv op is not yet supported."
    )
    def run_gemv(
        self, test_cases, gemv_op, atol, rtol, quantize_w=False, quantize_x=False
    ):
        for M, N, K in test_cases:
            x = torch.randn(size=(M, K), dtype=torch.bfloat16, device="cuda") * 0.1
            w = torch.randn(size=(N, K), dtype=torch.bfloat16, device="cuda") * 0.01
            if quantize_w and not quantize_x:
                wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
                z = gemv_op(x, wq, w_scale)
            elif quantize_w and quantize_x:
                xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
                wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
                z = gemv_op(xq, wq, x_scale * w_scale)
            else:
                z = gemv_op(x, w)
            z_ref = (x @ w.T).to(torch.bfloat16).to("cuda")
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: fast gemv op is not yet supported."
    )
    def test_bf16_gemv(self) -> None:
        test_cases = [
            (1, 128, 256),
            (1, 256, 256),
            (1, 1280, 8192),
            (1, 8192, 1024),
            (1, 7168, 8192),
            (1, 8192, 3584),
        ]
        self.run_gemv(test_cases, torch.ops.fbgemm.bf16_fast_gemv, 9.0e-3, 9.0e-3)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: fast gemv op is not yet supported."
    )
    def test_bf16_fp8_gemv(self) -> None:
        test_cases = [
            (1, 128, 256),
            (1, 256, 256),
            (1, 1280, 8192),
            (1, 8192, 1024),
            (1, 7168, 8192),
            (1, 8192, 3584),
        ]
        self.run_gemv(
            test_cases,
            torch.ops.fbgemm.bf16fp8bf16_fast_gemv,
            1.0e-2,
            1.0e-2,
            quantize_w=True,
        )

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: fast gemv op is not yet supported."
    )
    def test_fp8_fp8_gemv(self) -> None:
        test_cases = [
            (1, 1280, 8192),
            (1, 8192, 1024),
            (1, 7168, 8192),
            (1, 8192, 3584),
        ]
        self.run_gemv(
            test_cases,
            torch.ops.fbgemm.fp8fp8bf16_fast_gemv,
            9.0e-2,
            9.0e-2,
            quantize_w=True,
            quantize_x=True,
        )

    @unittest.skipIf(
        torch.version.hip, "Skip on AMD: cuda quantize op is yet supported."
    )
    @settings(deadline=None)
    @given(
        K=st.sampled_from([0, 128]),
    )
    def test_quantize_zero_input(self, K) -> None:
        w = torch.randn(size=(0, K), dtype=torch.bfloat16, device="cuda")
        w_scale_ref = torch.empty(size=(0,), dtype=torch.float32, device="cuda")
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
        torch.testing.assert_close(w.shape, wq.shape)
        torch.testing.assert_close(w_scale.shape, w_scale_ref.shape)

    @unittest.skipIf(torch.version.hip, "Skip on AMD: fp8 lite op is yet suported.")
    @settings(deadline=None)
    @given(
        M=st.sampled_from([1, 4]),
        N=st.sampled_from([1024, 6144]),
        K=st.sampled_from([512, 3584]),
        CudaGraph=st.sampled_from([True, False]),
    )
    def test_fp8_lite_matmul(self, M: int, N: int, K: int, CudaGraph: bool) -> None:
        x = torch.randn(size=(M, K), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(size=(N, K), dtype=torch.bfloat16, device="cuda") * 0.01
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
        if CudaGraph:
            zq = torch.ops.fbgemm.f8f8bf16_lite(xq, wq, x_scale * w_scale)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                zq = torch.ops.fbgemm.f8f8bf16_lite(xq, wq, x_scale * w_scale)
            g.replay()
        else:
            zq = torch.ops.fbgemm.f8f8bf16_lite(xq, wq, x_scale * w_scale)
        zq_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(zq, zq_ref, atol=9.0e-2, rtol=9.0e-2)


if __name__ == "__main__":
    unittest.main()
