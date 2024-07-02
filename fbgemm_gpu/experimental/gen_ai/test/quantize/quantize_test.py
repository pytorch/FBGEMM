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

from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
    matmul_fp8_block,
    matmul_fp8_row,
    quantize_fp8_block,
    quantize_fp8_row,
)

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
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
        Mode=st.sampled_from(
            ["rowwise", "blockwise"]
            + (["tensorwise_broadcast", "tensorwise"] if torch.version.cuda else [])
        ),
        QType=st.sampled_from([fp8_e4m3, fp8_e5m2]),
        Bias=st.sampled_from([True, False]),
        CudaGraph=st.sampled_from([True, False]),
        UseTriton=st.sampled_from([True, False]),
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
    ) -> None:
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
            block_m = block_n = block_k = 256
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
            atol = 1.0e-1
            rtol = 1.0e-1
        else:
            atol = 8.0e-2
            rtol = 8.0e-2
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
        torch.testing.assert_close(zq, zq_ref, atol=8.0e-2, rtol=8.0e-2)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: built in quantize ops not yet suported."
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
        not torch.version.cuda, "Skip on AMD: built in quantize ops not yet suported."
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


if __name__ == "__main__":
    unittest.main()
