# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import os
import unittest

from typing import Optional, Tuple, Union

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import torch
import triton  # noqa: F401

if torch.cuda.is_available():
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        matmul_fp8_block,
        matmul_fp8_row,
        quantize_fp8_block,
        quantize_fp8_row,
        supports_float8_fnuz,
    )

    from fbgemm_gpu.experimental.gen_ai.quantize import quantize_int4_preshuffle

    if torch.cuda.get_device_capability() >= (10, 0):
        from fbgemm_gpu.experimental.gemm.triton_gemm.fp4_quantize import _to_blocked

from hypothesis import given, settings, strategies as st

# Marlin is currently only supported internally at Meta.
try:
    if not torch.version.hip:
        from marlin.quantize import marlin_quantize

        torch.ops.load_library("//ai_codesign/gen_ai/marlin:marlin_ops")
        MARLIN_ENABLED = True
except ImportError:
    MARLIN_ENABLED = False

running_on_github: bool = os.getenv("GITHUB_ENV") is not None


def evaluate_platform_supports_fp8():
    if torch.cuda.is_available():
        if torch.version.hip:
            return supports_float8_fnuz(throw_on_hip_incompatibility=False)
        else:
            # Only SM90 or later is supported
            return torch.cuda.get_device_capability() >= (9, 0)
    return False


def evaluate_platform_supports_mxfp8():
    if torch.cuda.is_available():
        if torch.version.hip:
            return False
        return torch.cuda.get_device_capability() >= (10, 0)
    return False


def evaluate_cuda_platform_version(major: int):
    if torch.version.cuda:
        return torch.cuda.get_device_capability() >= (major, 0)
    return False


SM90_OR_LATER = evaluate_cuda_platform_version(9)

SUPPORTS_FP8 = evaluate_platform_supports_fp8()

SUPPORTS_MXFP8 = evaluate_platform_supports_mxfp8()

if torch.cuda.is_available() and supports_float8_fnuz(
    throw_on_hip_incompatibility=(not running_on_github)
):
    # Supported FP8 format is different on NV and AMD.
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


def sample_scales() -> st.SearchStrategy[Optional[torch.Tensor]]:
    return st.sampled_from(
        [
            None,
            torch.tensor(
                [1.0],
                dtype=torch.float,
                device=torch.accelerator.current_accelerator(),
            ),
        ]
        if torch.cuda.is_available()
        else [None]
    )


# Source: https://github.com/pytorch/ao/blob/568c1932a16ae9f30d48da214a88dc0013e98ed8/torchao/prototype/moe_training/utils.py#L310
def generate_jagged_offs(E, M, multiple_of=16, dtype=torch.int32, device="cuda"):
    """
    Utility function for tests and benchmarks.

    Generates a tensor of length E, containing random values divisible by `multiple_of`,
    from 0 to M, in sorted order, and where the final value in the tensor is always M.
    Args:
        E (int): The length of the tensor.
        M (int): The maximum value in the tensor.
    Returns:
        torch.Tensor: A tensor of length E with the specified properties.
    """
    import random

    # Ensure M is divisible by 16
    if M % multiple_of != 0:
        raise ValueError(f"M must be divisible by {multiple_of}")

    # Generate a list of possible values
    possible_values = list(range(multiple_of, M + 1, multiple_of))

    # If E is larger than the number of possible values, raise an error
    if E > len(possible_values):
        raise ValueError("E cannot be larger than the number of possible values")

    # Randomly select E - 1 values from the possible values (excluding M)
    selected_values = torch.tensor(random.sample(possible_values[:-1], E - 1))

    # Append M to the selected values
    selected_values = torch.cat((selected_values, torch.tensor([M])))

    # Sort the selected values
    selected_values, _ = torch.sort(selected_values)

    return selected_values.to(dtype).to(device)


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
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

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
        x = (
            torch.randn(
                size=(M, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.randn(
                size=(N, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )

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
        torch.version.hip is not None and running_on_github,
        "type fp8e4b8 not supported in this architecture. The supported fp8 dtypes are ('fp8e5',)",
    )
    @unittest.skipIf(
        ((not torch.version.cuda) and (not torch.version.hip)),
        "Skip if no GPU is present.",
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([0, 2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512, 4096, 8192]),
        Mode=st.sampled_from(
            ["rowwise", "blockwise"]
            + (["tensorwise_broadcast", "tensorwise"] if torch.version.cuda else [])
        ),
        QType=(
            st.sampled_from([fp8_e4m3, fp8_e5m2] if torch.version.cuda else [fp8_e4m3])
        ),
        Bias=st.sampled_from([True, False]),
        CudaGraph=st.sampled_from([True, False]),
        UseTriton=st.sampled_from([False] + ([True] if torch.version.cuda else [])),
        UseFastAccum=st.booleans(),
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
        UseFastAccum: bool,
        InputMultiDim: bool,
    ) -> None:
        # Slow accumulation is only supported on Nvidia.
        if torch.version.hip:
            UseFastAccum = True
        # Setup input shapes.
        if InputMultiDim:
            x = (
                torch.randn(
                    size=(3, B_T, D),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.1
            )
        else:
            x = (
                torch.randn(
                    size=(B_T, D),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.1
            )
        w = (
            torch.randn(
                size=(HD_L, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )
        bias = (
            torch.randn(
                size=(HD_L,),
                dtype=torch.bfloat16,
                device=self.device,
            )
            if Bias
            else None
        )

        if Mode == "tensorwise":

            def f(
                x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor]
            ) -> torch.Tensor:
                xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
                wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
                zq = torch.ops.fbgemm.f8f8bf16(xq, wq, x_scale * w_scale)
                if bias is not None:
                    zq += bias
                return zq

            if CudaGraph:
                # Warm-up to avoid capture issues
                f(x, w, bias)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    zq = f(x, w, bias)
                g.replay()
            else:
                zq = f(x, w, bias)
        elif Mode == "tensorwise_broadcast":

            def f(
                xq: torch.Tensor,
                wq: torch.Tensor,
                scale: float,
                bias: Optional[torch.Tensor],
            ) -> torch.Tensor:
                zq = torch.ops.fbgemm.f8f8bf16_tensorwise(
                    xq, wq, scale, use_fast_accum=UseFastAccum
                )
                if bias is not None:
                    zq += bias
                return zq

            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
            wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
            x_scale = x_scale.item()
            w_scale = w_scale.item()

            if CudaGraph:
                # Warm-up to avoid capture issues
                f(xq, wq, x_scale * w_scale, bias)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    zq = f(xq, wq, x_scale * w_scale, bias)
                g.replay()
            else:
                zq = f(xq, wq, x_scale * w_scale, bias)
        elif Mode == "rowwise":

            def f(
                x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor]
            ) -> torch.Tensor:
                xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
                    x, output_dtype=QType
                )
                wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
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
                        use_fast_accum=UseFastAccum,
                    )
                    # Bias fusion not yet supported on AMD.
                    if bias is not None and torch.version.hip:
                        zq += bias

                return zq

            if CudaGraph:
                # Warm-up to avoid capture issues
                f(x, w, bias)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    zq = f(x, w, bias)
                g.replay()
            else:
                zq = f(x, w, bias)
        elif Mode == "blockwise":

            def f(
                x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor]
            ) -> torch.Tensor:
                block_m = block_n = block_k = 128
                wq, w_scale = quantize_fp8_block(
                    w, block_n, block_k, output_device=torch.device(self.device)
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
                        fp8_fast_accum=UseFastAccum,
                    )
                else:
                    zq = torch.ops.fbgemm.f8f8bf16_blockwise(
                        xq, wq, x_scale, w_scale, block_m, block_n, block_k
                    )
                if bias is not None:
                    zq += bias

                return zq

            if CudaGraph:
                # Warm-up to avoid capture issues
                f(x, w, bias)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    zq = f(x, w, bias)
                g.replay()
            else:
                zq = f(x, w, bias)
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
        B_T=st.sampled_from([0, 2048, 4096]),
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
        x = (
            torch.randn(
                size=(B_T, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.randn(
                size=(HD_L, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )

        # Standard i4 weight format.
        wq, w_scale, w_zp = int4_row_quantize(w, 128)
        wq = pack_int4(wq).contiguous().to(device=self.device)
        w_scale = w_scale.contiguous().to(device=self.device)
        w_zp = w_zp.contiguous().to(device=self.device)

        # Preshuffled i4 weight format.
        wq_shuffled, (w_scale_group, w_scale_row) = quantize_int4_preshuffle(w, 128)

        if CudaGraph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
                zq = torch.ops.fbgemm.f8i4bf16_rowwise(xq, wq, x_scale, w_scale, w_zp)
                zq_shuffled = torch.ops.fbgemm.f8i4bf16_shuffled(
                    xq, wq_shuffled, x_scale, w_scale_row, w_scale_group
                )
            g.replay()
        else:
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
            zq = torch.ops.fbgemm.f8i4bf16_rowwise(xq, wq, x_scale, w_scale, w_zp)
            zq_shuffled = torch.ops.fbgemm.f8i4bf16_shuffled(
                xq, wq_shuffled, x_scale, w_scale_row, w_scale_group
            )

        zq_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(zq, zq_ref, atol=8.0e-2, rtol=8.0e-2)
        torch.testing.assert_close(zq_shuffled, zq_ref, atol=8.0e-2, rtol=8.0e-2)

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: built in quantize ops not yet suported."
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
        CudaGraph=st.booleans(),
        Preshuffle=st.booleans(),
    )
    def test_quantize_int4_bf16_matmul(
        self,
        B_T: int,
        D: int,
        HD_L: int,
        CudaGraph: bool,
        Preshuffle: bool,
    ) -> None:
        x = (
            torch.randn(
                size=(B_T, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.randn(
                size=(HD_L, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )

        if Preshuffle:
            wq, (w_scale, w_zp) = quantize_int4_preshuffle(w, dtype="bf16")
        else:
            wq, w_scale, w_zp = int4_row_quantize(w, 128)
            wq = pack_int4(wq).contiguous().to(device=self.device)
            w_scale = w_scale.contiguous().to(device=self.device)
            w_zp = w_zp.contiguous().to(device=self.device)

        bf16i4_op = (
            torch.ops.fbgemm.bf16i4bf16_shuffled
            if Preshuffle
            else torch.ops.fbgemm.bf16i4bf16_rowwise
        )

        if CudaGraph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                zq = bf16i4_op(x, wq, w_scale, w_zp)
            g.replay()
        else:
            zq = bf16i4_op(x, wq, w_scale, w_zp)

        zq_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(zq, zq_ref, atol=1.0e-1, rtol=8.0e-2)

    @unittest.skipIf(running_on_github, "Test is currently unreliable on GitHub OSS CI")
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
        x = (
            torch.randn(
                size=(B_T, D),
                dtype=dtype,
                device=self.device,
            )
            * 0.1
        )
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
        x = torch.full(
            (B_T, D),
            rand_val,
            dtype=torch.bfloat16,
            device=self.device,
        )
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
        x = (
            torch.randn(
                size=(G_B, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.randn(
                size=(HD_L, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )

        # batch size (B) which is <= graph batch size (G_B)
        B = int(G_B / 2)
        B_t = torch.tensor(B, dtype=torch.int64, device=self.device)

        x[B:, :] = float("nan")
        x_ref = torch.randn(
            size=(B, D),
            dtype=torch.bfloat16,
            device=self.device,
        )
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
        x = (
            torch.randn(
                size=(B_T, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.randn(
                size=(HD_L, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )

        UB_t = torch.tensor(UB, dtype=torch.int64, device=self.device)

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

    @unittest.skipIf(not SUPPORTS_FP8, "FP8 not supported on this platform")
    @settings(deadline=None)
    @given(
        B=st.sampled_from([1, 4]),
        M=st.sampled_from([2048, 4096]),
        N=st.sampled_from([128, 256]),
        K=st.sampled_from([256, 512]),
        use_loopover=st.sampled_from([True, False]),
        Bias=st.sampled_from([False] + ([True] if torch.version.cuda else [])),
        mode=st.sampled_from(
            ["default"] + (["torch_3d3d"] if torch.version.hip else [])
        ),
    )
    def test_fp8_batched_gemm(
        self,
        B: int,
        M: int,
        N: int,
        K: int,
        Bias: bool,
        use_loopover: bool,
        mode: str,
    ) -> None:
        # AMD CK FP8 batched gemm does not support N < 512 or K < 512.
        # Funny enough, grouped gemm does not have this restriction.
        if mode == "default" and torch.version.hip and (N < 512 or K < 512):
            return

        x = (
            torch.rand(
                size=(B, M, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.rand(
                size=(B, N, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )
        bias = (
            torch.randn(
                size=(B, N),
                dtype=torch.bfloat16,
                device=self.device,
            )
            if Bias
            else None
        )

        xq, x_scale = quantize_fp8_row(x)
        x_scale = x_scale.view(B, -1)
        assert x_scale.shape == (B, M)
        wq, w_scale = quantize_fp8_row(w)
        w_scale = w_scale.view(B, -1)
        assert w_scale.shape == (B, N)

        def fp8_loopover_bmm(
            xq: torch.Tensor,
            wq: torch.Tensor,
            x_scale: torch.Tensor,
            w_scale: torch.Tensor,
            bias: Optional[torch.Tensor],
        ) -> torch.Tensor:
            B = len(xq)
            M = xq[0].shape[0]
            N = wq[0].shape[0]
            y = torch.empty((B, M, N), dtype=torch.bfloat16, device=xq[0].device)
            for i in range(B):
                y[i] = torch.ops.fbgemm.f8f8bf16_rowwise(
                    xq[i],
                    wq[i],
                    x_scale[i],
                    w_scale[i],
                    bias[i] if bias is not None else None,
                )
            return y

        y_ref = torch.bmm(x, w.transpose(1, 2))
        if bias is not None:
            y_ref += bias.unsqueeze(1)

        if use_loopover:
            y_fp8 = fp8_loopover_bmm(xq, wq, x_scale, w_scale, bias)
        else:
            if mode == "default":
                y_fp8 = torch.ops.fbgemm.f8f8bf16_rowwise_batched(
                    xq, wq, x_scale, w_scale, bias
                )
            elif mode == "torch_3d3d":
                y_fp8_ = torch.empty(
                    (B, M, N), dtype=torch.bfloat16, device=xq[0].device
                )
                y_fp8 = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_mm(
                    xq,
                    wq,
                    x_scale,
                    w_scale,
                    None,
                    y_fp8_,
                )

        torch.testing.assert_close(y_ref, y_fp8, atol=8.0e-2, rtol=8.0e-2)

    @unittest.skipIf(not SUPPORTS_FP8, "FP8 not supported on this platform")
    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 5, 16]),
        M=st.sampled_from([0, 2048, 3584]),
        N=st.sampled_from([256, 1024, 6144]),
        K=st.sampled_from([256, 512, 3584]),
        use_cudagraph=st.booleans(),
        mode=st.sampled_from(["default", "cat", "padded"]),
    )
    def test_grouped_gemm_fbgemm_api(
        self, G: int, M: int, N: int, K: int, use_cudagraph: bool, mode: str
    ):
        # TODO remove this restriction.
        if N < 512 or K < 512:
            return

        if M > 0:
            ms = (
                torch.randint(
                    (258 // 64) + 1 if mode == "padding" else 1,
                    (M // 64) + 1,
                    (G,),
                    dtype=torch.int,
                )
                * 64
            )
        else:
            ms = torch.zeros((G,), dtype=torch.int)
        # Only default supports true dynamism.
        if mode != "default":
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
        if mode == "padded":
            zero_start_index_M = torch.tensor(ms, dtype=torch.long, device=self.device)

        for _, (m, n, k) in enumerate(zip(ms, ns, ks)):
            x = torch.rand(
                size=(m, k),
                dtype=torch.bfloat16,
                device=self.device,
            )
            w = torch.rand(
                size=(n, k),
                dtype=torch.bfloat16,
                device=self.device,
            )

            if mode == "padded":
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
        if mode == "padded":
            x_group = torch.stack(x_group, dim=0).contiguous()
            w_group = torch.stack(w_group, dim=0).contiguous()
            xq_group = torch.stack(xq_group, dim=0).contiguous()
            wq_group = torch.stack(wq_group, dim=0).contiguous()
            x_scale_group = torch.stack(x_scale_group, dim=0).contiguous()
            w_scale_group = torch.stack(w_scale_group, dim=0).contiguous()

            fp8_op = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_dynamic
            bf16_op = torch.ops.fbgemm.bf16bf16bf16_grouped_dynamic
            fp8_args = [
                xq_group,
                wq_group,
                x_scale_group,
                w_scale_group,
                zero_start_index_M,
            ]
            bf16_args = [x_group, w_group, zero_start_index_M]
        else:
            if mode == "cat":
                fp8_op = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_cat
                bf16_op = torch.ops.fbgemm.bf16bf16bf16_grouped_cat
            else:
                fp8_op = torch.ops.fbgemm.f8f8bf16_rowwise_grouped
                bf16_op = torch.ops.fbgemm.bf16bf16bf16_grouped
            fp8_args = [xq_group, wq_group, x_scale_group, w_scale_group]
            bf16_args = [x_group, w_group]

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

        self.bf16_loopover_validate(
            x_group,
            w_group,
            y_fp8_group,
            y_bf16_group,
            # default mode is worse for some reason
            rtol_fp8=2.0e-1 if mode == "default" else 8.0e-2,
        )

    def bf16_loopover_validate(
        self,
        x: Union[torch.Tensor, list[torch.Tensor]],
        w: Union[torch.Tensor, list[torch.Tensor]],
        out_fp8: Union[torch.tensor, list[torch.Tensor]],
        out_bf16: Union[torch.tensor, list[torch.Tensor], None] = None,
        atol_fp8=8.0e-2,
        rtol_fp8=8.0e-2,
        atol_bf16=8.0e-3,
        rtol_bf16=8.0e-3,
    ):
        out_ref = [torch.matmul(x[i], w[i].t()) for i in range(len(x))]

        for i in range(len(out_fp8)):
            torch.testing.assert_close(
                out_fp8[i], out_ref[i], atol=atol_fp8, rtol=rtol_fp8
            )

        if out_bf16:
            for i in range(len(out_bf16)):
                torch.testing.assert_close(
                    out_bf16[i], out_ref[i], atol=atol_bf16, rtol=rtol_bf16
                )

    @unittest.skipIf(not SUPPORTS_FP8, "FP8 not supported on this platform")
    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 16]),
        M=st.sampled_from([0, 2048, 3584]),
        N=st.sampled_from([256, 1024, 6144]),
        K=st.sampled_from([256, 512, 3584]),
        use_cudagraph=st.booleans(),
        mode=st.sampled_from(
            ["stacked"] + (["torch_2d3d"] if torch.version.hip else [])
        ),
    )
    def test_grouped_gemm_2d_3d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        use_cudagraph: bool,
        mode: str,
    ) -> None:
        # TODO remove this restriction.
        if (N < 512 or K < 512) and mode == "stacked":
            return

        if M > 0:
            M_sizes = (
                torch.randint(
                    1,
                    (M // 64) + 1,
                    (G,),
                    dtype=torch.int,
                )
                * 64
            )
        else:
            M_sizes = torch.zeros((G,), dtype=torch.int)

        M = torch.sum(M_sizes).item()
        X = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((G, N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        xq, x_scale = quantize_fp8_row(X)
        wq, w_scale = quantize_fp8_row(W)

        # FP8 grouped gemm kernel
        if mode == "stacked":
            fp8_op = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_stacked
            M_sizes_gpu = M_sizes.clone().to(device=self.device, dtype=torch.int64)
            fp8_args = [xq, wq, x_scale, w_scale, M_sizes_gpu]

            bf16_op = torch.ops.fbgemm.bf16bf16bf16_grouped_stacked
            bf16_args = [X, W, M_sizes_gpu]
        elif mode == "torch_2d3d":
            fp8_op = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_mm
            M_offsets = torch.cumsum(M_sizes, dim=0).to(
                device=self.device, dtype=torch.int32
            )
            out = torch.empty(M, N).to(device=self.device, dtype=torch.bfloat16)
            fp8_args = [
                xq,
                wq,
                x_scale,
                w_scale,
                M_offsets,
                out,
            ]

            bf16_op = None
            bf16_args = None

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
        y_fp8_group = torch.split(y_fp8_group, tuple(M_sizes.tolist()), dim=0)

        # unstack input to make it compatible with loopover.
        x_group = torch.split(X, tuple(M_sizes.tolist()), dim=0)

        y_bf16_group = None
        if bf16_op is not None:
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

            y_bf16_group = torch.split(y_bf16_group, tuple(M_sizes.tolist()), dim=0)

        # BF16 loopover gemm reference
        self.bf16_loopover_validate(x_group, W, y_fp8_group, y_bf16_group)

    @unittest.skipIf(not SUPPORTS_MXFP8, "MXFP8 not supported on this platform")
    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 16]),
        K=st.sampled_from([2048, 3584]),
        N=st.sampled_from([256, 1024, 6144]),
        M=st.sampled_from([256, 512, 3584]),
    )
    def test_mx_grouped_gemm_2d_2d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        # Simulate 2d-2d grouped gemm in backward pass `grad_weight = grad_output_t @ input`,
        # where we use "K" as the contracting dim which has "G" groups.
        from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import to_mxfp8

        total_K = K  # Alias for clarity, communicating this consists of several groups along this dim
        input_group_end_offsets = generate_jagged_offs(
            G, total_K, multiple_of=32, device=self.device
        )
        X = torch.randn((M, total_K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((N, total_K), dtype=torch.bfloat16, device=self.device) * 0.01

        # Convert scales to blocked format.
        x_list = []
        w_list = []
        x_blocked_scale_list = []
        w_blocked_scale_list = []

        def round_up(x: int, y: int) -> int:
            return ((x + y - 1) // y) * y

        for group_idx in range(G):
            # to_mxfp8 per group
            prev_group_end_offset = (
                0 if group_idx == 0 else input_group_end_offsets[group_idx - 1]
            )
            curr_group_end_offset = input_group_end_offsets[group_idx]
            group_size = curr_group_end_offset - prev_group_end_offset
            if group_size > 0:
                x_slice = X[
                    :, prev_group_end_offset:curr_group_end_offset
                ].contiguous()  # (M, K_group)
                w_slice = W[
                    :, prev_group_end_offset:curr_group_end_offset
                ].contiguous()  # (N, K_group)
                x_scale_slice, xq_slice = to_mxfp8(
                    x_slice
                )  # scale shape -> (M, K_group // 32)
                w_scale_slice, wq_slice = to_mxfp8(
                    w_slice
                )  # scale shape -> (N, K_group // 32)
                x_list.append(xq_slice)
                w_list.append(wq_slice)

                # Convert scales to blocked format.
                x_scale_slice_blocked = _to_blocked(
                    x_scale_slice
                )  # (round_up(M, 128), round_up(K_group//32, 4))
                w_scale_slice_blocked = _to_blocked(
                    w_scale_slice
                )  # (round_up(N, 128), round_up(K_group//32, 4))
                x_blocked_scale_list.append(x_scale_slice_blocked)
                w_blocked_scale_list.append(w_scale_slice_blocked)

        # Assemble the full XQ and WQ
        xq = torch.cat(x_list, dim=1).contiguous()
        wq = torch.cat(w_list, dim=1).contiguous()

        # Combine all XQ groups blocked scales into one tensor.
        x_blocked_scales = torch.cat(x_blocked_scale_list, dim=0)
        M_rounded = round_up(M, 128)
        x_blocked_scales = x_blocked_scales.reshape(M_rounded, -1)

        # Combine all WQ groups blocked scales into one tensor.
        w_blocked_scales = torch.cat(w_blocked_scale_list, dim=0)
        N_rounded = round_up(N, 128)
        w_blocked_scales = w_blocked_scales.reshape(N_rounded, -1)

        # Compute mxfp8 grouped mm output
        out = torch.empty((G, M, N), dtype=torch.bfloat16, device=self.device)
        y_mxfp8 = torch.ops.fbgemm.mx8mx8bf16_grouped_mm(
            xq,  # (M, total_K)
            wq.transpose(-2, -1),  # (total_K, N)
            x_blocked_scales,  # to_blocked_per_group(M, total_K//32)
            w_blocked_scales,  # to_blocked_per_group(N, total_K//32)
            input_group_end_offsets,  # (G,)
            out,  # (G, M, N)
        )

        # bf16 reference output
        y_bf16 = torch._grouped_mm(
            X, W.t(), offs=input_group_end_offsets, out_dtype=torch.bfloat16
        )

        # Assert no NaNs
        assert not y_mxfp8.isnan().any(), "mxfp8 output contains NaN"

        # Assert outputs are close
        torch.testing.assert_close(y_mxfp8, y_bf16, atol=8.0e-2, rtol=8.0e-2)

    @unittest.skipIf(not SUPPORTS_MXFP8, "MXFP8 not supported on this platform")
    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 16]),
        M=st.sampled_from([2048, 3584]),
        N=st.sampled_from([256, 1024, 6144]),
        K=st.sampled_from([256, 512, 3584]),
    )
    def test_mx_grouped_gemm_2d_3d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import to_mxfp8

        # Simulate 2d-3d grouped gemm `out = input @ weight.t()`
        # 2D inputs with groups along M, 3D weights.
        block_size = 32
        total_M = M  # Alias for clarity that M dim contains groups.
        X = torch.randn((total_M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((G, N, K), dtype=torch.bfloat16, device=self.device) * 0.01
        input_group_end_offsets = generate_jagged_offs(
            G, total_M, multiple_of=32, device=self.device
        )

        # For each constituent 2d subtensor in the 3d weights, quantize and convert scale to blocked format separately,
        # as they each used for independent gemm in the grouped gemm.
        wq_list = []
        w_scale_list = []
        for i in range(G):
            w_scale, wq = to_mxfp8(W[i])
            w_scale = _to_blocked(w_scale)
            wq_list.append(wq)
            w_scale_list.append(w_scale)
        wq = torch.stack(wq_list, dim=0).contiguous()
        w_scale = torch.stack(w_scale_list, dim=0).contiguous()

        # For each group along `total_M` in the 2D tensor, quantize and convert scale to blocked format separately,
        # as they each used for independent gemm in the grouped gemm.
        xq_list = []
        x_scale_list = []
        for i in range(G):
            prev_group_end = 0 if i == 0 else input_group_end_offsets[i - 1]
            curr_group_end = input_group_end_offsets[i]
            group_size = curr_group_end - prev_group_end
            if group_size > 0:
                x_slice = X[prev_group_end:curr_group_end, :]
                x_scale, xq = to_mxfp8(x_slice)
                x_scale = _to_blocked(x_scale)
                xq_list.append(xq)
                x_scale_list.append(x_scale)
        xq = torch.cat(xq_list, dim=0).contiguous()
        x_scale = torch.cat(x_scale_list, dim=0).contiguous()
        x_scale = x_scale.reshape(-1, K // block_size)
        xq = xq.view(-1, xq.shape[-1])

        # Compute mxfp8 grouped gemm.
        out = torch.empty((total_M, N), dtype=torch.bfloat16, device=self.device)
        y_mxfp8 = torch.ops.fbgemm.mx8mx8bf16_grouped_mm(
            xq,
            wq.transpose(-2, -1),
            x_scale,
            w_scale,
            input_group_end_offsets,
            out,
        )

        # Compute reference bf16 grouped gemm.
        y_bf16 = torch._grouped_mm(
            X,
            W.transpose(-2, -1),
            offs=input_group_end_offsets,
            out_dtype=torch.bfloat16,
        )

        # Assert outputs are close.
        torch.testing.assert_close(y_mxfp8, y_bf16, atol=8.0e-2, rtol=8.0e-2)

    @unittest.skipIf(
        not torch.version.hip,
        "Only AMD supports torch 3D-2D grouped gemm API",
    )
    @unittest.skipIf(not SUPPORTS_FP8, "FP8 not supported on this platform")
    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 16]),
        M=st.sampled_from([0, 64, 2048, 3584]),
        N=st.sampled_from([64, 256, 1024, 6144]),
        K=st.sampled_from([64, 256, 512, 3584]),
    )
    def test_grouped_gemm_3d_2d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        N_sizes = (
            torch.randint(
                1,
                (N // 64) + 1,
                (G,),
                dtype=torch.int,
            )
            * 64
        )
        N = torch.sum(N_sizes).item()
        N_offsets = torch.cumsum(N_sizes, dim=0).to(
            device=self.device, dtype=torch.int32
        )

        X = torch.randn((G, M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
        out = torch.empty((M, N), dtype=torch.bfloat16, device=self.device)

        xq, x_scale = quantize_fp8_row(X)
        wq, w_scale = quantize_fp8_row(W)

        y = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_mm(
            xq, wq, x_scale, w_scale, N_offsets, out
        )

        # Compare using loopover BF16 gemm
        y_fp8 = torch.split(y, tuple(N_sizes), dim=1)
        W_split = torch.split(W, tuple(N_sizes), dim=0)
        self.bf16_loopover_validate(X, W_split, y_fp8)

    @unittest.skipIf(
        not torch.version.hip,
        "Only AMD supports torch 2D-2D grouped gemm API",
    )
    @unittest.skipIf(not SUPPORTS_FP8, "FP8 not supported on this platform")
    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 16]),
        M=st.sampled_from([16, 2048, 3584]),
        N=st.sampled_from([16, 256, 1024, 6144]),
        K=st.sampled_from([16, 256, 512, 3584]),
        use_cudagraph=st.booleans(),
    )
    def test_grouped_gemm_2d_2d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        use_cudagraph: bool,
    ) -> None:
        K_sizes = torch.ones((G,), dtype=torch.int, device=self.device) * K
        K_offsets = torch.cumsum(K_sizes, dim=0).to(
            device=self.device, dtype=torch.int32
        )

        # Each group should be quantized rowwise separately
        X_list = []
        W_list = []
        xq_list = []
        wq_list = []
        x_scale_list = []
        w_scale_list = []
        for k_size in K_sizes.tolist():
            X = torch.randn((M, k_size), dtype=torch.bfloat16, device=self.device) * 0.1
            W = (
                torch.randn((N, k_size), dtype=torch.bfloat16, device=self.device)
                * 0.01
            )
            xq, x_scale = quantize_fp8_row(X)
            wq, w_scale = quantize_fp8_row(W)

            X_list.append(X)
            W_list.append(W)
            xq_list.append(xq)
            wq_list.append(wq)
            x_scale_list.append(x_scale)
            w_scale_list.append(w_scale)

        xq = torch.cat(xq_list, dim=1)
        wq = torch.cat(wq_list, dim=1)
        x_scale = torch.cat(x_scale_list, dim=0)
        w_scale = torch.cat(w_scale_list, dim=0)

        out = torch.empty((G, M, N), dtype=torch.bfloat16, device=self.device)

        if use_cudagraph:
            # warmup
            torch.ops.fbgemm.f8f8bf16_rowwise_grouped_mm(
                xq, wq, x_scale, w_scale, K_offsets, out
            )
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_mm(
                    xq, wq, x_scale, w_scale, K_offsets, out
                )
            g.replay()
        else:
            y = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_mm(
                xq, wq, x_scale, w_scale, K_offsets, out
            )

        # Compare using loopover BF16 gemm
        self.bf16_loopover_validate(X_list, W_list, y)

    @unittest.skipIf(not torch.version.cuda, "Currently not supported on AMD.")
    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 5, 16]),
        M=st.sampled_from([0, 2048, 3584]),
        N=st.sampled_from([1024, 6144]),
        K=st.sampled_from([512, 3584]),
        use_cudagraph=st.booleans(),
    )
    def test_shuffled_grouped_gemm(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        use_cudagraph: bool,
    ) -> None:
        if M > 0:
            ms = (
                torch.randint(
                    1,
                    (M // 64) + 1,
                    (G,),
                    dtype=torch.int,
                )
                * 64
            )
        else:
            ms = torch.zeros((G,), dtype=torch.int)

        M_sizes = ms.to(device=self.device, dtype=torch.int32)
        ns = [N] * G
        ks = [K] * G

        x_group = []
        w_group = []
        xq_group = []
        x_scale_group = []
        w_fp8_group = []
        fp8_group_scales = []
        fp8_row_scales = []
        w_bf16_group = []
        bf16_group_scales = []
        bf16_group_zeros = []

        for _, (m, n, k) in enumerate(zip(ms, ns, ks)):
            x = torch.rand(
                size=(m, k),
                dtype=torch.bfloat16,
                device=self.device,
            )
            w = torch.rand(
                size=(n, k),
                dtype=torch.bfloat16,
                device=self.device,
            )

            xq, x_scale = quantize_fp8_row(x)
            w_fp8, (fp8_group_scale, fp8_row_scale) = quantize_int4_preshuffle(w)
            w_bf16, (bf16_group_scale, bf16_group_zero) = quantize_int4_preshuffle(
                w, dtype="bf16", use_zp=False
            )
            x_group.append(x)
            w_group.append(w)
            xq_group.append(xq)
            x_scale_group.append(x_scale)
            w_fp8_group.append(w_fp8)
            fp8_group_scales.append(fp8_group_scale)
            fp8_row_scales.append(fp8_row_scale)
            w_bf16_group.append(w_bf16)
            bf16_group_scales.append(bf16_group_scale)
            bf16_group_zeros.append(bf16_group_zero)

        # Only stacked API currently available for preshuffled grouped gemm.
        x_group = torch.cat(x_group, dim=0).contiguous()
        w_group = torch.stack(w_group, dim=0).contiguous()
        xq_group = torch.cat(xq_group, dim=0).contiguous()
        x_scale_group = torch.cat(x_scale_group, dim=0).contiguous()
        w_fp8_group = torch.stack(w_fp8_group, dim=0).contiguous()
        fp8_group_scales = torch.stack(fp8_group_scales, dim=0).contiguous()
        fp8_row_scales = torch.stack(fp8_row_scales, dim=0).contiguous()
        w_bf16_group = torch.stack(w_bf16_group, dim=0).contiguous()
        bf16_group_scales = torch.stack(bf16_group_scales, dim=0).contiguous()
        bf16_group_zeros = torch.stack(bf16_group_zeros, dim=0).contiguous()

        fp8_op = torch.ops.fbgemm.f8i4bf16_shuffled_grouped
        fp8_args = [
            xq_group,
            w_fp8_group,
            x_scale_group,
            fp8_row_scales,
            fp8_group_scales,
            M_sizes,
        ]
        bf16_op = torch.ops.fbgemm.bf16i4bf16_shuffled_grouped
        bf16_args = [
            x_group,
            w_bf16_group,
            bf16_group_scales,
            bf16_group_zeros,
            M_sizes,
        ]

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
        y_fp8_group = torch.split(y_fp8_group, tuple(ms.tolist()), dim=0)

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
        y_bf16_group = torch.split(y_bf16_group, tuple(ms.tolist()), dim=0)

        # BF16 loopover gemm reference
        # unstack input to make it compatible with loopover.
        x_group = torch.split(x_group, tuple(ms.tolist()), dim=0)
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
                y_bf16_group[i], y_group_ref[i], atol=8.0e-2, rtol=5.0e-2
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
        x = (
            torch.rand(
                size=(B, M, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.rand(
                size=(B, N, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )

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

                wq_ = pack_int4(wq_).contiguous().to(device=self.device)
                w_scale_ = w_scale_.contiguous().to(device=self.device)
                w_zp_ = w_zp_.contiguous().to(device=self.device)
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
        X = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        XQ = torch.randn(M, K, device=self.device).to(fp8_dtype)
        WQ = torch.randn(N, K, device=self.device).to(fp8_dtype)
        output = torch.empty(M, N, device=self.device, dtype=torch.bfloat16)
        row_scale = torch.randn(M, device=self.device)
        col_scale = torch.randn(N, device=self.device)
        block_scale = torch.randn(M // 128, K // 128, device=self.device)
        tensor_scale = torch.tensor(1.0, device=self.device)

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
        # These ops are only supported on hip for now.
        if torch.version.hip:
            torch.compile(torch.ops.fbgemm.f8f8f16_rowwise)(
                XQ, WQ, row_scale, col_scale
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
                block_scale[0].repeat(M).view(-1, M),
                block_scale[0].repeat(N).view(-1, N),
            )
            torch.compile(torch.ops.fbgemm.bf16i4bf16_rowwise_batched)(
                X.view(1, M, K),
                WQ[:, ::2].view(1, N, K // 2).view(torch.int8).contiguous(),
                block_scale[0].repeat(M).view(1, -1, M),
                block_scale[0].repeat(N).view(1, -1, N),
            )
            # test bf16_fast_gemv is torch compileable
            W_bf16 = torch.randn(
                N,
                K,
                device=self.device,
                dtype=torch.bfloat16,
            )
            torch.compile(torch.ops.fbgemm.bf16_fast_gemv)(X, W_bf16)

    @unittest.skipIf(
        torch.version.hip, "Skip on AMD: cuda quantize op is yet supported."
    )
    @settings(deadline=None)
    @given(
        K=st.sampled_from([0, 128]),
    )
    def test_quantize_zero_input(self, K) -> None:
        w = torch.randn(
            size=(0, K),
            dtype=torch.bfloat16,
            device=self.device,
        )
        w_scale_ref = torch.empty(
            size=(0,),
            dtype=torch.float32,
            device=self.device,
        )
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
        x = (
            torch.randn(
                size=(M, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.randn(
                size=(N, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )
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


@unittest.skipIf(not torch.cuda.is_available(), "Skip when GPU is not available")
@unittest.skipIf(not SM90_OR_LATER, "Skip when not SM90+")
class FastGemvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    def run_gemv(
        self, test_cases, gemv_op, atol, rtol, quantize_w=False, quantize_x=False
    ):
        for M, N, K in test_cases:
            x = (
                torch.randn(
                    size=(M, K),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.1
            )
            w = (
                torch.randn(
                    size=(N, K),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.01
            )
            if quantize_w and not quantize_x:
                wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
                z = gemv_op(x, wq, w_scale)
            elif quantize_w and quantize_x:
                # row-wise scaling
                xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
                wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
                z = gemv_op(xq, wq, x_scale, w_scale)
            else:
                z = gemv_op(x, w)
            z_ref = (x @ w.T).to(torch.bfloat16).to(self.device)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)

    def run_gemv_batched(self, test_cases, gemv_op, atol, rtol):
        for B, M, N, K in test_cases:
            x = (
                torch.randn(
                    size=(B, M, K),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.1
            )
            w = (
                torch.randn(
                    size=(B, N, K),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.01
            )
            xq, x_scale = quantize_fp8_row(x)
            x_scale = x_scale.view(B, -1)
            assert x_scale.shape == (B, M)
            wq, w_scale = quantize_fp8_row(w)
            w_scale = w_scale.view(B, -1)
            assert w_scale.shape == (B, N)
            z = gemv_op(xq, wq, x_scale, w_scale, is_batched=True)
            z_ref = torch.bmm(x, w.transpose(1, 2)).to(torch.bfloat16).to(self.device)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)

    def test_bf16_gemv(self) -> None:
        test_cases = [
            (1, 128, 256),
            (1, 256, 256),
            (1, 1280, 8192),
            (1, 8192, 1024),
            (1, 7168, 8192),
            (1, 8192, 3584),
            (2, 128, 256),
            (2, 256, 256),
            (2, 1280, 8192),
            (2, 8192, 1024),
            (2, 7168, 8192),
            (2, 8192, 3584),
            (4, 128, 256),
            (4, 256, 256),
            (4, 1280, 8192),
            (4, 8192, 1024),
            (4, 7168, 8192),
            (4, 8192, 3584),
        ]
        self.run_gemv(test_cases, torch.ops.fbgemm.bf16_fast_gemv, 9.0e-3, 9.0e-3)

    def test_bf16_fp8_gemv(self) -> None:
        test_cases = [
            (1, 1280, 8192),
            (1, 8192, 1024),
            (1, 7168, 8192),
            (1, 8192, 3584),
            (2, 1280, 8192),
            (2, 8192, 1024),
            (2, 7168, 8192),
            (2, 8192, 3584),
            (4, 1280, 8192),
            (4, 8192, 1024),
            (4, 7168, 8192),
            (4, 8192, 3584),
        ]
        self.run_gemv(
            test_cases,
            torch.ops.fbgemm.bf16fp8bf16_fast_gemv,
            9.0e-2,
            9.0e-2,
            quantize_w=True,
        )

    def test_fp8_fp8_gemv(self) -> None:
        test_cases = [
            (1, 1280, 8192),
            (1, 8192, 1024),
            (1, 7168, 8192),
            (1, 8192, 3584),
            (2, 1280, 8192),
            (2, 8192, 1024),
            (2, 7168, 8192),
            (2, 8192, 3584),
            (3, 1280, 8192),
            (3, 8192, 1024),
            (3, 7168, 8192),
            (3, 8192, 3584),
            (4, 1280, 8192),
            (4, 8192, 1024),
            (4, 7168, 8192),
            (4, 8192, 3584),
            (1, 4096, 5120),  # below are l4_17B_128E dense model shapes
            (1, 5120, 2048),
            (1, 896, 5120),
            (1, 5120, 640),
            (2, 4096, 5120),
            (2, 5120, 2048),
            (2, 896, 5120),
            (2, 5120, 640),
        ]
        self.run_gemv(
            test_cases,
            torch.ops.fbgemm.fp8fp8bf16_fast_gemv,
            9.0e-2,
            9.0e-2,
            quantize_w=True,
            quantize_x=True,
        )

    def test_fp8_gemv_batched(self) -> None:
        test_cases = [
            (2, 1, 4096, 5120),
            (2, 1, 5120, 2048),
            (2, 1, 896, 5120),
            (2, 1, 5120, 640),
            (2, 1, 8192, 1024),
            (2, 1, 7168, 8192),
            (2, 1, 8192, 3584),
            (2, 1, 1280, 8192),
            (2, 2, 8192, 1024),
            (2, 2, 7168, 8192),
            (2, 2, 8192, 3584),
            (2, 2, 1280, 8192),
            (32, 1, 1280, 8192),
            (128, 1, 1280, 8192),
        ]
        self.run_gemv_batched(
            test_cases,
            torch.ops.fbgemm.fp8fp8bf16_fast_gemv,
            1.0e-1,
            1.0e-1,
        )


@unittest.skipIf(
    not torch.cuda.is_available() or torch.version.hip,
    "Skip when cuda is not available or HIP is enabled",
)
class NVFP4Tests(unittest.TestCase):
    @unittest.skipIf(
        (not torch.version.cuda)
        or torch.version.hip is not None
        or str(torch.version.cuda) <= "12.8",
        "Skip if no cuda is present or HIP is enabled",
    )
    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
        static_scale=sample_scales(),
        scale_ub=sample_scales(),
    )
    def test_fake_quantize_nvfp4_per_tensor(
        self,
        B_T: int,
        D: int,
        HD_L: int,
        static_scale: Optional[torch.Tensor],
        scale_ub: Optional[torch.Tensor],
    ) -> None:
        x = (
            torch.randn(
                size=(B_T, D),
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
            )
            * 0.1
        )
        w = (
            torch.randn(
                size=(HD_L, D),
                dtype=torch.bfloat16,
                device=torch.accelerator.current_accelerator(),
            )
            * 0.01
        )

        xq, _ = torch.ops.fbgemm.fake_quantize_nvfp4_per_tensor(
            x, static_scales=static_scale, scale_ub=scale_ub
        )
        wq, _ = torch.ops.fbgemm.fake_quantize_nvfp4_per_tensor(
            w, static_scales=static_scale, scale_ub=scale_ub
        )
        fake_quant_y = xq @ wq.T
        fake_quant_y = fake_quant_y.to(torch.bfloat16)

        y_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(fake_quant_y, y_ref, atol=0.1, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
