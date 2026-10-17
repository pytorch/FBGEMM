# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import os
import unittest
from typing import Optional, Union

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import torch
import triton  # noqa: F401
from fbgemm_gpu.experimental.gemm.triton_gemm.fp4_quantize import (
    get_nvfp4_global_scales_naive,
    quantize_nvfp4_naive,
    triton_quantize_mx4_unpack,
)

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
            return torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).gcnArchName.split(":")[0] in ("gfx942", "gfx950")
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


def evaluate_platform_supports_nvfp4():
    if torch.cuda.is_available():
        if torch.version.cuda:
            return evaluate_cuda_platform_version(10) and torch.version.cuda >= "12.8"
    return False


def evaluate_platform_supports_mxfp4():
    if torch.cuda.is_available():
        if torch.version.cuda:
            return evaluate_cuda_platform_version(10) and torch.version.cuda >= "12.8"
        # TODO add AMD here later
    return False


def evaluate_cuda_platform_version(major: int):
    if torch.cuda.is_available():
        if torch.version.cuda:
            return torch.cuda.get_device_capability() >= (major, 0)
    return False


SM90_OR_LATER = evaluate_cuda_platform_version(9)

SUPPORTS_NVFP4 = evaluate_platform_supports_nvfp4()

SUPPORTS_MXFP4 = evaluate_platform_supports_mxfp4()

SUPPORTS_FP8 = evaluate_platform_supports_fp8()

SUPPORTS_MXFP8 = evaluate_platform_supports_mxfp8()

IS_GFX950 = (
    torch.cuda.is_available()
    and torch.version.hip is not None
    and torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(
        ":"
    )[0]
    == "gfx950"
)

# CK grouped FP8 GEMM intermittently leaves partial tiles unwritten for N=6144
# on gfx950. Keep the smaller shapes covered until the large-N kernel is fixed.
GROUPED_GEMM_N_VALUES = [256, 1024] if IS_GFX950 else [256, 1024, 6144]
GROUPED_GEMM_N_VALUES_WITH_SMALL = [64, *GROUPED_GEMM_N_VALUES]

if torch.cuda.is_available() and supports_float8_fnuz(
    throw_on_hip_incompatibility=False
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


def fp8_row_quantize_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Quantize an input tensor and return the fp8 tensor and its inverse scale.
    x_row_max = torch.max(torch.abs(x.float()), dim=1).values
    max_scaling_factor = E4M3_MAX_POS * 512.0  # Match kernel logics
    scale = torch.Tensor(E4M3_MAX_POS / x_row_max).clamp(max=max_scaling_factor)
    xq = (x.float() * scale.unsqueeze(1)).to(fp8_e4m3)
    return xq, scale.reciprocal().to(torch.float32)


def fp8_col_quantize_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Quantize an input tensor and return the fp8 tensor and its inverse scale.
    x_col_max = torch.max(torch.abs(x.float()), dim=0).values
    max_scaling_factor = E4M3_MAX_POS * 512.0  # Match kernel logics
    scale = torch.Tensor(E4M3_MAX_POS / x_col_max).clamp(max=max_scaling_factor)
    xq = (x.float() * scale.unsqueeze(0)).to(fp8_e4m3)
    return xq, scale.reciprocal().to(torch.float32)


def int4_row_quantize(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                # Citrine C7: create test tensors directly on the GPU.
                row_scale = torch.randn(
                    M, device=torch.accelerator.current_accelerator()
                )
                col_scale = torch.randn(
                    N, device=torch.accelerator.current_accelerator()
                )
                block_scale = torch.randn(
                    M // 128, K // 128, device=torch.accelerator.current_accelerator()
                )
                _ = torch.ops.fbgemm.f8f8bf16_blockwise(
                    xq, wq, block_scale, block_scale
                )
                _ = torch.ops.fbgemm.f8f8bf16_tensorwise(xq, wq, 1.0)
                o = torch.ops.fbgemm.f8f8bf16_rowwise(xq, wq, row_scale, col_scale)
                return o

        device = torch.accelerator.current_accelerator()
        model = TestModule().to(device)
        M, N, K = 256, 256, 256
        # Citrine C3: create export inputs directly on the accelerator.
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        xq, _ = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
        wq, _ = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
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
        # Citrine C7: create test tensors directly on the GPU.
        src_float = torch.randn(1000, 1000, device=self.device)
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
        QType=(st.sampled_from([fp8_e4m3])),
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
        QType=st.sampled_from([torch.float8_e4m3fn]),
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
            x_max = x.float().abs().max()
            x_scale_ref = (x_max / fp8_max).float()
            xq_ref = (x.float() * fp8_max / x_max).to(fp8_e4m3)
        elif Mode == "rowwise":
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
            x = (xq.float() / x_scale.unsqueeze(1)).to(dtype)  # Fake quantization
            xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
                x, stochastic_rounding=stochastic_rounding
            )
            xq_ref, x_scale_ref = fp8_row_quantize_ref(x)
        elif Mode == "colwise" and (
            torch.version.hip or str(torch.version.cuda) >= "12.1"
        ):
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
        self.assertEqual(xq.dtype, fp8_e4m3)
        torch.testing.assert_close(x_scale, x_scale_ref, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(xq.float(), xq_ref.float(), atol=tol, rtol=tol)

    def test_quantize_fp8_per_tensor_input_dtypes(self) -> None:
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.linspace(-1, 1, 4 * 128, device=self.device, dtype=dtype).reshape(
                4, 128
            )
            expected_scale = x.float().abs().max() / E4M3_MAX_POS
            expected = (x.float() / expected_scale).to(fp8_e4m3)
            for stochastic in (False, True):
                with self.subTest(dtype=dtype, stochastic=stochastic):
                    quantized, scale = torch.ops.fbgemm.quantize_fp8_per_tensor(
                        x, stochastic_rounding=stochastic
                    )
                    self.assertEqual(quantized.dtype, fp8_e4m3)
                    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
                    tolerance = 0.125 if stochastic else 0
                    torch.testing.assert_close(
                        quantized.float(),
                        expected.float(),
                        atol=tolerance,
                        rtol=tolerance,
                    )

        bf16_input = x.to(torch.bfloat16)
        scale = torch.ops.fbgemm.get_fp8_per_tensor_scale(bf16_input)
        expected, expected_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(bf16_input)
        fixed = torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(bf16_input, scale)
        torch.testing.assert_close(scale[0], expected_scale, atol=0, rtol=0)
        torch.testing.assert_close(fixed.float(), expected.float(), atol=0, rtol=0)
        self.assertEqual(fixed.dtype, fp8_e4m3)

        stochastic_fixed = torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(
            bf16_input, scale, None, True
        )
        torch.testing.assert_close(
            stochastic_fixed.float() * scale,
            bf16_input.float(),
            atol=0.125 / E4M3_MAX_POS,
            rtol=0.125,
        )
        with self.assertRaisesRegex(RuntimeError, "multiple of 4"):
            torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(
                bf16_input[:, :3].contiguous(), scale, None, True
            )

        # Citrine C3: create test storage directly on the accelerator.
        noncontiguous = torch.ones(4, 256, dtype=torch.bfloat16, device=self.device)[
            :, ::2
        ]
        self.assertFalse(noncontiguous.is_contiguous())
        with self.assertRaisesRegex(RuntimeError, "input must be contiguous"):
            torch.ops.fbgemm.get_fp8_per_tensor_scale(noncontiguous)
        with self.assertRaisesRegex(RuntimeError, "input must be contiguous"):
            torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(noncontiguous, scale)
        with self.assertRaisesRegex(RuntimeError, "input must be contiguous"):
            torch.ops.fbgemm.quantize_fp8_per_tensor(noncontiguous)
        if torch.version.hip or str(torch.version.cuda) >= "12.1":
            with self.assertRaisesRegex(RuntimeError, "input must be contiguous"):
                torch.ops.fbgemm.quantize_fp8_per_col(noncontiguous)

    def test_quantize_fp8_per_col_rank_three(self) -> None:
        if not torch.version.hip and str(torch.version.cuda) < "12.1":
            self.skipTest("quantize_fp8_per_col requires CUDA 12.1 or later")

        # Citrine C3: create test input directly on the accelerator.
        x = torch.linspace(
            -2.0,
            1.0,
            steps=2 * 3 * 128,
            dtype=torch.bfloat16,
            device=self.device,
        ).reshape(2, 3, 128)
        # Citrine C3: create the upper bound directly on the accelerator.
        scale_ub = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        expected_scale = (
            x.float().abs().amax(dim=0).clamp(max=scale_ub) / E4M3_MAX_POS
        ).clamp(min=1 / (E4M3_MAX_POS * 512))

        quantized, scale = torch.ops.fbgemm.quantize_fp8_per_col(x, scale_ub=scale_ub)
        expected = (
            (x.float() / scale.unsqueeze(0))
            .clamp(min=-E4M3_MAX_POS, max=E4M3_MAX_POS)
            .to(fp8_e4m3)
        )

        self.assertEqual(scale.shape, (3, 128))
        torch.testing.assert_close(scale, expected_scale, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(quantized.float(), expected.float(), atol=0, rtol=0)

        invalid_scale_ubs = (
            (torch.tensor(0.5, dtype=torch.float64, device=self.device), "float32"),
            (torch.ones(2, dtype=torch.float32, device=self.device), "one element"),
            (torch.tensor(0.5, dtype=torch.float32), "input device"),
        )
        for invalid_scale_ub, error in invalid_scale_ubs:
            with self.subTest(error=error):
                with self.assertRaisesRegex(RuntimeError, error):
                    torch.ops.fbgemm.quantize_fp8_per_col(x, scale_ub=invalid_scale_ub)

        cpu_quantized, cpu_scale = torch.ops.fbgemm.quantize_fp8_per_col(
            x.cpu(), scale_ub=scale_ub.cpu()
        )
        self.assertEqual(cpu_quantized.device.type, "cpu")
        self.assertEqual(cpu_scale.device.type, "cpu")
        with self.assertRaisesRegex(RuntimeError, "input device"):
            torch.ops.fbgemm.quantize_fp8_per_col(x.cpu(), scale_ub=scale_ub)

    def test_quantize_fp8_rank_three_scale_shapes(self) -> None:
        # Citrine C3: create test input directly on the accelerator.
        x = torch.ones(2, 3, 128, dtype=torch.bfloat16, device=self.device)
        for op, expected_shape in (
            (torch.ops.fbgemm.quantize_fp8_per_tensor, ()),
            (torch.ops.fbgemm.quantize_fp8_per_row, (6,)),
        ):
            eager = op(x)
            compiled = torch.compile(op)(x)
            with self.subTest(op=op):
                self.assertEqual(eager[1].shape, expected_shape)
                self.assertEqual(compiled[1].shape, expected_shape)

    def test_quantize_fp8_per_row_empty_width(self) -> None:
        # Citrine C3: create the empty input directly on the accelerator.
        x = torch.empty(2, 0, dtype=torch.bfloat16, device=self.device)

        quantized, scale = torch.ops.fbgemm.quantize_fp8_per_row(x)

        self.assertEqual(quantized.shape, x.shape)
        self.assertEqual(scale.shape, (2,))
        torch.testing.assert_close(scale, torch.zeros_like(scale), atol=0, rtol=0)

        tensor_quantized, tensor_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
        self.assertEqual(tensor_quantized.shape, x.shape)
        torch.testing.assert_close(
            tensor_scale, torch.zeros_like(tensor_scale), atol=0, rtol=0
        )

        if torch.version.hip or str(torch.version.cuda) >= "12.1":
            # Citrine C3: create the empty columnwise input on the accelerator.
            col_input = torch.empty(0, 128, dtype=torch.bfloat16, device=self.device)
            col_quantized, col_scale = torch.ops.fbgemm.quantize_fp8_per_col(col_input)
            self.assertEqual(col_quantized.shape, col_input.shape)
            torch.testing.assert_close(
                col_scale, torch.zeros_like(col_scale), atol=0, rtol=0
            )

    def test_quantize_fp8_meta_contract_validation(self) -> None:
        # Citrine C3: create contract-test inputs directly on the accelerator.
        rank_four = torch.ones(1, 2, 3, 128, dtype=torch.bfloat16, device=self.device)
        noncontiguous = torch.ones(2, 256, dtype=torch.bfloat16, device=self.device)[
            :, ::2
        ]
        invalid_dtype = torch.ones(2, 128, dtype=torch.int32, device=self.device)
        stochastic_width = torch.ones(2, 127, dtype=torch.bfloat16, device=self.device)
        ops = [
            torch.ops.fbgemm.quantize_fp8_per_tensor,
            torch.ops.fbgemm.quantize_fp8_per_row,
        ]
        if torch.version.hip or str(torch.version.cuda) >= "12.1":
            ops.append(torch.ops.fbgemm.quantize_fp8_per_col)

        for op in ops:
            for candidate in (op, torch.compile(op)):
                with self.subTest(op=op, candidate=candidate):
                    with self.assertRaisesRegex(RuntimeError, "dim.*2 or 3"):
                        candidate(rank_four)
                    with self.assertRaisesRegex(
                        RuntimeError, "input must be contiguous"
                    ):
                        candidate(noncontiguous)
                    with self.assertRaisesRegex(RuntimeError, "input must be"):
                        candidate(invalid_dtype)

        for op in (
            torch.ops.fbgemm.quantize_fp8_per_tensor,
            torch.ops.fbgemm.quantize_fp8_per_row,
        ):
            for candidate in (op, torch.compile(op)):
                with self.subTest(op=op, candidate=candidate):
                    with self.assertRaisesRegex(RuntimeError, "multiple of 4"):
                        candidate(stochastic_width, stochastic_rounding=True)

    def test_quantize_fp8_optional_tensor_validation(self) -> None:
        # Citrine C3: create option tensors directly on the accelerator.
        x = torch.ones(2, 128, dtype=torch.bfloat16, device=self.device)
        invalid_scale_ubs = (
            (torch.tensor(1.0, dtype=torch.float64, device=self.device), "float32"),
            (torch.ones(2, dtype=torch.float32, device=self.device), "one element"),
            (torch.tensor(1.0, dtype=torch.float32), "device"),
        )
        invalid_batch_sizes = (
            (torch.tensor(1.0, dtype=torch.float32, device=self.device), "int64"),
            (torch.ones(2, dtype=torch.int64, device=self.device), "one element"),
            (torch.tensor(1, dtype=torch.int64), "device"),
        )
        for op in (
            torch.ops.fbgemm.quantize_fp8_per_tensor,
            torch.ops.fbgemm.quantize_fp8_per_row,
        ):
            for scale_ub, error in invalid_scale_ubs:
                with self.subTest(op=op, scale_ub_error=error):
                    with self.assertRaisesRegex(RuntimeError, error):
                        op(x, scale_ub=scale_ub)
            for bs, error in invalid_batch_sizes:
                with self.subTest(op=op, bs_error=error):
                    with self.assertRaisesRegex(RuntimeError, error):
                        op(x, bs=bs)

        cpu_input = x.cpu()
        cpu_bs = torch.tensor(2, dtype=torch.int64)
        cpu_scale_ub = torch.tensor(0.5, dtype=torch.float32)
        # Citrine C3: create mismatched option tensors directly on the GPU.
        device_bs = torch.tensor(2, dtype=torch.int64, device=self.device)
        device_scale_ub = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        for op in (
            torch.ops.fbgemm.quantize_fp8_per_tensor,
            torch.ops.fbgemm.quantize_fp8_per_row,
        ):
            quantized, scale = op(cpu_input, bs=cpu_bs, scale_ub=cpu_scale_ub)
            with self.subTest(op=op):
                self.assertEqual(quantized.device.type, "cpu")
                self.assertEqual(scale.device.type, "cpu")
                with self.assertRaisesRegex(RuntimeError, "device"):
                    op(cpu_input, bs=device_bs)
                with self.assertRaisesRegex(RuntimeError, "device"):
                    op(cpu_input, scale_ub=device_scale_ub)

        for scale_ub, error in invalid_scale_ubs:
            with self.subTest(op="get_scale", scale_ub_error=error):
                with self.assertRaisesRegex(RuntimeError, error):
                    torch.ops.fbgemm.get_fp8_per_tensor_scale(x, scale_ub=scale_ub)
        for bs, error in invalid_batch_sizes:
            with self.subTest(op="get_scale", bs_error=error):
                with self.assertRaisesRegex(RuntimeError, error):
                    torch.ops.fbgemm.get_fp8_per_tensor_scale(x, bs=bs)

        invalid_scales = (
            (torch.tensor(1.0, dtype=torch.float64, device=self.device), "float32"),
            (torch.ones(2, dtype=torch.float32, device=self.device), "one element"),
            (torch.tensor(1.0, dtype=torch.float32), "device"),
        )
        for scale, error in invalid_scales:
            with self.subTest(op="fixed_scale", scale_error=error):
                with self.assertRaisesRegex(RuntimeError, error):
                    torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(x, scale)

    def test_quantize_fp8_invalid_values_mark_outputs(self) -> None:
        # Citrine C3: create inputs and option tensors directly on the accelerator.
        x = torch.ones(2, 128, dtype=torch.bfloat16, device=self.device)
        negative_bs = torch.tensor(-1, dtype=torch.int64, device=self.device)
        negative_scale_ub = torch.tensor(-1.0, dtype=torch.float32, device=self.device)
        invalid_scale = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for op in (
            torch.ops.fbgemm.quantize_fp8_per_tensor,
            torch.ops.fbgemm.quantize_fp8_per_row,
            torch.ops.fbgemm.get_fp8_per_tensor_scale,
        ):
            with self.subTest(op=op, optional="bs"):
                output = op(x, bs=negative_bs)
                output_scale = (
                    output[1] if isinstance(output, (tuple, list)) else output
                )
                self.assertTrue(torch.isnan(output_scale).all())
            with self.subTest(op=op, optional="scale_ub"):
                output = op(x, scale_ub=negative_scale_ub)
                output_scale = (
                    output[1] if isinstance(output, (tuple, list)) else output
                )
                self.assertTrue(torch.isnan(output_scale).all())

        fixed_quantized = torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(
            x, invalid_scale
        )
        self.assertTrue(torch.isnan(fixed_quantized.float()).all())

        # Citrine C3: create the follow-up input directly on the accelerator.
        valid_input = torch.ones(1, 128, dtype=torch.bfloat16, device=self.device)
        torch.ops.fbgemm.quantize_fp8_per_tensor(valid_input)
        torch.cuda.synchronize()

    def test_quantize_fp8_invalid_values_are_graph_safe(self) -> None:
        # Citrine C3: create graph inputs directly on the accelerator.
        x = torch.ones(2, 128, dtype=torch.bfloat16, device=self.device)
        bs = torch.tensor(2, dtype=torch.int64, device=self.device)
        scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        torch.ops.fbgemm.quantize_fp8_per_tensor(x, bs=bs)
        torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(x, scale, bs=bs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, tensor_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x, bs=bs)
            fixed_quantized = torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(
                x, scale, bs=bs
            )
        bs.fill_(-1)
        scale.zero_()
        graph.replay()

        self.assertTrue(torch.isnan(tensor_scale).all())
        self.assertTrue(torch.isnan(fixed_quantized.float()).all())

    def test_quantize_fp8_empty_optionals_are_graph_safe(self) -> None:
        # Citrine C3: create graph inputs directly on the accelerator.
        x = torch.empty(0, 128, dtype=torch.bfloat16, device=self.device)
        bs = torch.tensor(0, dtype=torch.int64, device=self.device)
        scale_ub = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        torch.ops.fbgemm.quantize_fp8_per_tensor(x, bs=bs, scale_ub=scale_ub)
        torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(x, scale, bs=bs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quantized, tensor_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(
                x, bs=bs, scale_ub=scale_ub
            )
            fixed_quantized = torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(
                x, scale, bs=bs
            )
        graph.replay()

        self.assertEqual(quantized.shape, x.shape)
        self.assertEqual(tensor_scale.shape, ())
        self.assertEqual(fixed_quantized.shape, x.shape)

    def test_quantize_fp8_per_row_large_bf16_width(self) -> None:
        # Citrine C3: create the large row directly on the accelerator.
        x = torch.ones(2, 32768, dtype=torch.bfloat16, device=self.device)

        quantized, scale = torch.ops.fbgemm.quantize_fp8_per_row(x)

        self.assertEqual(quantized.shape, x.shape)
        torch.testing.assert_close(
            scale,
            torch.full_like(scale, 1 / E4M3_MAX_POS),
            atol=0,
            rtol=0,
        )

    def test_quantize_fp8_per_row_uses_input_device(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("multiple accelerators are required")

        current_device = torch.cuda.current_device()
        input_device = (current_device + 1) % torch.cuda.device_count()
        # Citrine C3: create test input directly on the non-current accelerator.
        x = torch.linspace(
            -1.0,
            1.0,
            steps=2 * 128,
            dtype=torch.bfloat16,
            device=torch.device(
                torch.accelerator.current_accelerator().type, input_device
            ),
        ).reshape(2, 128)

        with torch.cuda.device(current_device):
            quantized, scale = torch.ops.fbgemm.quantize_fp8_per_row(x)

        self.assertEqual(quantized.device.index, input_device)
        self.assertEqual(scale.device.index, input_device)
        expected_scale = x.float().abs().amax(dim=1) / E4M3_MAX_POS
        torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires two GPUs")
    def test_quantizer_nondefault_device(self) -> None:
        current = torch.cuda.current_device()
        other = (current + 1) % torch.cuda.device_count()
        x = torch.ones((4, 128), device=f"cuda:{other}", dtype=torch.bfloat16)
        for name in (
            "quantize_fp8_per_tensor",
            "quantize_fp8_per_row",
            "quantize_fp8_per_col",
        ):
            with self.subTest(operator=name):
                q, scale = getattr(torch.ops.fbgemm, name)(x)
                self.assertEqual(q.device, x.device)
                self.assertEqual(scale.device, x.device)
                if name == "quantize_fp8_per_row":
                    scale = scale.unsqueeze(-1)
                torch.testing.assert_close(
                    q.float() * scale, x.float(), atol=1e-6, rtol=1e-6
                )
                self.assertEqual(torch.cuda.current_device(), current)
        q = torch.ops.fbgemm.silu_mul_quantize_i8(x, x, 1.0)
        self.assertEqual(q.device, x.device)
        torch.testing.assert_close(q, torch.ones_like(q), atol=0, rtol=0)
        self.assertEqual(torch.cuda.current_device(), current)
        with self.assertRaisesRegex(RuntimeError, "same device"):
            torch.ops.fbgemm.silu_mul_quantize_i8(x, x.to(f"cuda:{current}"), 1.0)
        self.assertEqual(torch.cuda.current_device(), current)

        wrong_device = f"cuda:{current}"
        bs = torch.tensor([x.size(0)], dtype=torch.int64, device=wrong_device)
        scale_ub = torch.tensor([1.0], dtype=torch.float32, device=wrong_device)
        scale = torch.tensor([1.0], dtype=torch.float32, device=wrong_device)
        valid_scale = scale.to(x.device)
        invalid_auxiliary_calls = {
            "tensorwise bs": lambda: torch.ops.fbgemm.quantize_fp8_per_tensor(x, bs),
            "tensorwise scale_ub": lambda: torch.ops.fbgemm.quantize_fp8_per_tensor(
                x, None, scale_ub
            ),
            "get scale bs": lambda: torch.ops.fbgemm.get_fp8_per_tensor_scale(x, bs),
            "get scale scale_ub": lambda: torch.ops.fbgemm.get_fp8_per_tensor_scale(
                x, None, scale_ub
            ),
            "fixed scale": lambda: torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(
                x, scale
            ),
            "fixed scale bs": lambda: torch.ops.fbgemm.quantize_fp8_per_tensor_fixed_scale(
                x, valid_scale, bs
            ),
            "rowwise scale_ub": lambda: torch.ops.fbgemm.quantize_fp8_per_row(
                x, None, scale_ub
            ),
        }
        for name, call in invalid_auxiliary_calls.items():
            with self.subTest(operator=name), self.assertRaisesRegex(
                RuntimeError, "device"
            ):
                call()
            self.assertEqual(torch.cuda.current_device(), current)

    def test_quantizer_cpu_bridge_auxiliaries(self) -> None:
        x = torch.ones((2, 128), dtype=torch.bfloat16)
        bs = torch.tensor([2], dtype=torch.int64)
        scale_ub = torch.tensor([1.0], dtype=torch.float32)

        quantized, scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x, bs, scale_ub)
        self.assertEqual(quantized.device.type, "cpu")
        self.assertEqual(scale.device.type, "cpu")

    def test_quantize_int8(self) -> None:
        values = torch.tensor(
            [
                -200,
                -128,
                -127,
                -1.5,
                -0.5,
                0,
                0.5,
                1.5,
                126,
                127,
                128,
                200,
                2,
                -2,
                4,
                -4,
            ],
            dtype=torch.bfloat16,
            device=self.device,
        )
        for x in (values, torch.zeros_like(values), torch.randn_like(values)):
            for scale in (1.0, 0.125):
                with self.subTest(operator="per_tensor_quantize_i8", scale=scale):
                    quantized = torch.ops.fbgemm.per_tensor_quantize_i8(x, scale)
                    expected = (
                        (x.float() / (scale + 1e-8))
                        .clamp(-128, 127)
                        .round()
                        .to(torch.int8)
                    )
                    self.assertEqual(quantized.shape, x.shape)
                    self.assertEqual(quantized.dtype, torch.int8)
                    self.assertEqual(quantized.device, x.device)
                    torch.testing.assert_close(quantized, expected, atol=0, rtol=0)
            quantized, scale = torch.ops.fbgemm.per_tensor_dynamic_quantize_i8(x)
            expected_scale = x.abs().max().div(127)
            expected = (
                (x.float() / (expected_scale.float() + 1e-8))
                .clamp(-128, 127)
                .round()
                .to(torch.int8)
            )
            self.assertEqual(quantized.dtype, torch.int8)
            self.assertEqual(quantized.shape, x.shape)
            self.assertEqual(quantized.device, x.device)
            torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
            torch.testing.assert_close(quantized, expected, atol=0, rtol=0)

        x1 = torch.linspace(
            -2, 2, 3 * 128, dtype=torch.bfloat16, device=self.device
        ).reshape(3, 128)
        x2 = torch.ones_like(x1)
        scale = 0.03125
        quantized = torch.ops.fbgemm.silu_mul_quantize_i8(x1, x2, scale)
        expected = (
            (torch.nn.functional.silu(x1.float()) * x2.float() / scale)
            .clamp(-128, 127)
            .round()
            .to(torch.int8)
        )
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(quantized.shape, x1.shape)
        self.assertEqual(quantized.device, x1.device)
        torch.testing.assert_close(quantized, expected, atol=1, rtol=0)
        with self.assertRaisesRegex(RuntimeError, "same shape"):
            torch.ops.fbgemm.silu_mul_quantize_i8(x1, x2[:2], scale)

    def test_quantize_fp8_per_row_stochastic_rounding(self) -> None:
        fp8_max = torch.finfo(fp8_e4m3).max
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                x = torch.full(
                    (16, 1024),
                    1.0625 / fp8_max,
                    dtype=dtype,
                    device=self.device,
                )
                x[:, 0] = 1.0

                scale_ref = x.abs().amax(dim=1).float() / fp8_max
                nearest = (x.float() / scale_ref.unsqueeze(1)).to(fp8_e4m3)
                torch.manual_seed(0)
                stochastic, scale = torch.ops.fbgemm.quantize_fp8_per_row(
                    x, stochastic_rounding=True
                )
                torch.manual_seed(0)
                stochastic_repeated, scale_repeated = (
                    torch.ops.fbgemm.quantize_fp8_per_row(x, stochastic_rounding=True)
                )

                self.assertFalse(torch.equal(stochastic, nearest))
                rounded_midpoints = stochastic[:, 1:].float()
                torch.testing.assert_close(
                    torch.unique(rounded_midpoints),
                    torch.tensor([1.0, 1.125], device=self.device),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    rounded_midpoints.mean(),
                    torch.tensor(1.0625, device=self.device),
                    atol=0.01,
                    rtol=0,
                )
                self.assertFalse(
                    torch.equal(stochastic[:, 1:512], stochastic[:, 513:1024])
                )
                torch.testing.assert_close(
                    stochastic.float(), stochastic_repeated.float()
                )
                torch.testing.assert_close(scale, scale_ref)
                torch.testing.assert_close(scale, scale_repeated)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "Stochastic rounding is only supported for FP8 E4M3 output",
                ):
                    torch.ops.fbgemm.quantize_fp8_per_row(
                        x,
                        output_dtype=fp8_e5m2,
                        stochastic_rounding=True,
                    )

    def test_quantize_fp8_per_row_stochastic_normal_boundary(self) -> None:
        fp8_info = torch.finfo(fp8_e4m3)
        normal_boundary = fp8_info.smallest_normal
        midpoint = normal_boundary + normal_boundary / 16
        # Citrine C3: create the stochastic sample directly on the accelerator.
        x = torch.full(
            (32, 4096),
            midpoint / fp8_info.max,
            dtype=torch.float32,
            device=self.device,
        )
        x[:, 0] = 1.0

        quantized, _ = torch.ops.fbgemm.quantize_fp8_per_row(
            x, stochastic_rounding=True
        )

        torch.testing.assert_close(
            quantized[:, 1:].float().mean(),
            torch.tensor(midpoint, device=self.device),
            atol=normal_boundary / 128,
            rtol=0,
        )

    def test_quantize_fp8_per_row_stochastic_subnormal_midpoint(self) -> None:
        fp8_info = torch.finfo(fp8_e4m3)
        smallest_subnormal = fp8_info.smallest_normal / 8
        midpoint = smallest_subnormal / 2
        # Citrine C3: create the stochastic sample directly on the accelerator.
        x = torch.full(
            (32, 4096),
            midpoint / fp8_info.max,
            dtype=torch.float32,
            device=self.device,
        )
        x[:, 0] = 1.0

        quantized, _ = torch.ops.fbgemm.quantize_fp8_per_row(
            x, stochastic_rounding=True
        )
        samples = quantized[:, 1:].float()

        torch.testing.assert_close(
            torch.unique(samples),
            torch.tensor([0.0, smallest_subnormal], device=self.device),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            samples.mean(),
            torch.tensor(midpoint, device=self.device),
            atol=smallest_subnormal / 128,
            rtol=0,
        )

    def test_quantize_fp8_per_row_input_dtypes(self) -> None:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for output_dtype in (fp8_e4m3, fp8_e5m2):
                for row_width in (127, 128):
                    with self.subTest(
                        dtype=dtype,
                        output_dtype=output_dtype,
                        row_width=row_width,
                    ):
                        fp8_max = torch.finfo(output_dtype).max
                        x = torch.linspace(
                            -1.0,
                            1.0,
                            steps=2 * row_width,
                            dtype=dtype,
                            device=self.device,
                        ).reshape(2, row_width)
                        scale_ref = x.abs().amax(dim=1).float() / fp8_max
                        quantized_ref = (x.float() / scale_ref.unsqueeze(1)).to(
                            output_dtype
                        )

                        quantized, scale = torch.ops.fbgemm.quantize_fp8_per_row(
                            x, output_dtype=output_dtype
                        )

                        torch.testing.assert_close(
                            quantized.float(), quantized_ref.float(), atol=0, rtol=0
                        )
                        torch.testing.assert_close(scale, scale_ref)

        invalid_dtype_input = torch.ones(
            2, 128, dtype=torch.bfloat16, device=self.device
        )
        for op in (
            torch.ops.fbgemm.quantize_fp8_per_row,
            torch.compile(torch.ops.fbgemm.quantize_fp8_per_row),
        ):
            with self.assertRaisesRegex(RuntimeError, "output must be e4m3 or e5m2"):
                op(invalid_dtype_input, output_dtype=torch.bfloat16)

        storage = torch.ones(2 * 128 + 1, dtype=torch.bfloat16, device=self.device)
        unaligned = storage[1:].view(2, 128)
        self.assertTrue(unaligned.is_contiguous())
        quantized, scale = torch.ops.fbgemm.quantize_fp8_per_row(
            unaligned, stochastic_rounding=True
        )
        self.assertEqual(quantized.shape, unaligned.shape)
        self.assertEqual(scale.shape, (2,))

        noncontiguous = torch.ones(2, 256, dtype=torch.bfloat16, device=self.device)[
            :, ::2
        ]
        self.assertFalse(noncontiguous.is_contiguous())
        with self.assertRaisesRegex(RuntimeError, "input must be contiguous"):
            torch.ops.fbgemm.quantize_fp8_per_row(noncontiguous)

        cpu_input = torch.linspace(
            -1.0, 1.0, steps=2 * 127, dtype=torch.bfloat16
        ).reshape(2, 127)
        cpu_quantized, cpu_scale = torch.ops.fbgemm.quantize_fp8_per_row(cpu_input)
        device_quantized, device_scale = torch.ops.fbgemm.quantize_fp8_per_row(
            cpu_input.to(self.device)
        )
        self.assertEqual(cpu_quantized.device.type, "cpu")
        self.assertEqual(cpu_scale.device.type, "cpu")
        torch.testing.assert_close(
            cpu_quantized.float(), device_quantized.cpu().float(), atol=0, rtol=0
        )
        torch.testing.assert_close(cpu_scale, device_scale.cpu(), atol=0, rtol=0)

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
        if torch.version.hip:
            reference, reference_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x_ref)
            torch.testing.assert_close(x_scale, reference_scale, atol=0, rtol=0)
            torch.testing.assert_close(
                xq[:B].float(), reference.float(), atol=0, rtol=0
            )
            return
        zq = torch.ops.fbgemm.f8f8bf16(xq, wq, x_scale * w_scale)

        # Fake quant
        x = xq[:B, :].bfloat16() * x_scale
        w = wq.bfloat16() * w_scale

        zq_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(zq[:B, :], zq_ref, atol=2.0e-3, rtol=2.0e-3)

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

        UB_t = torch.tensor(UB, dtype=torch.float32, device=self.device)

        if torch.version.hip:
            op = (
                torch.ops.fbgemm.quantize_fp8_per_tensor
                if Mode == "tensorwise"
                else torch.ops.fbgemm.quantize_fp8_per_row
            )
            for cap in (UB_t, x.abs().max().float() / 2):
                xq, x_scale = op(x, scale_ub=cap)
                maximum = (
                    x.float().abs().max()
                    if Mode == "tensorwise"
                    else x.float().abs().amax(dim=-1)
                )
                expected_scale = maximum.clamp(max=cap) / E4M3_MAX_POS
                expected_scale = expected_scale.clamp(min=1 / (E4M3_MAX_POS * 512))
                torch.testing.assert_close(
                    x_scale, expected_scale, atol=1e-7, rtol=1e-5
                )
                divisor = x_scale if Mode == "tensorwise" else x_scale.unsqueeze(-1)
                expected = (
                    (x.float() / divisor)
                    .clamp(-E4M3_MAX_POS, E4M3_MAX_POS)
                    .to(fp8_e4m3)
                )
                torch.testing.assert_close(xq.float(), expected.float(), atol=0, rtol=0)
            return

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
                dtype=torch.float32,
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

    @unittest.skipIf(
        not torch.version.cuda, "Skip on AMD: FP16 GMM ops not yet suported."
    )
    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 8]),
        M=st.sampled_from([512, 1024, 2048]),
        N=st.sampled_from([512, 1024, 2048]),
        K=st.sampled_from([512, 1024, 2048]),
        dtype=st.sampled_from([torch.float16, torch.bfloat16]),
    )
    def test_grouped_gemm_fp16_bf16_support(
        self, G: int, M: int, N: int, K: int, dtype: torch.dtype
    ):
        """Test that CUTLASS grouped GEMM supports both FP16 and BF16 dtypes."""
        # Setup: Create input tensors with specified dtype
        ms = torch.randint(1, (M // 64) + 1, (G,), dtype=torch.int64) * 64

        x_group = []
        w_group = []

        for m in ms:
            x = torch.rand(size=(m, K), dtype=dtype, device=self.device)
            w = torch.rand(size=(N, K), dtype=dtype, device=self.device)
            x_group.append(x)
            w_group.append(w)

        # Execute: Run CUTLASS grouped GEMM
        y_cutlass = torch.ops.fbgemm.bf16bf16bf16_grouped(x_group, w_group)

        # Assert: Verify output dtype matches input dtype
        if not isinstance(y_cutlass, (tuple, list)):
            y_cutlass = torch.split(y_cutlass, tuple(ms.tolist()), dim=0)

        for i, y in enumerate(y_cutlass):
            # Verify output dtype matches input dtype
            self.assertEqual(
                y.dtype,
                dtype,
                f"Output dtype {y.dtype} does not match input dtype {dtype} for group {i}",
            )

            # Compute reference output
            y_ref = torch.matmul(x_group[i], w_group[i].t())

            # Verify numerical correctness
            torch.testing.assert_close(
                y,
                y_ref,
                atol=1e-2 if dtype == torch.float16 else 8e-3,
                rtol=1e-2 if dtype == torch.float16 else 8e-3,
            )

    @unittest.skipIf(not SUPPORTS_FP8, "FP8 not supported on this platform")
    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 5, 16]),
        M=st.sampled_from([0, 2048, 3584]),
        N=st.sampled_from(GROUPED_GEMM_N_VALUES),
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
        N=st.sampled_from(GROUPED_GEMM_N_VALUES),
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
        N=st.sampled_from(GROUPED_GEMM_N_VALUES_WITH_SMALL),
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
        # Initialize tensors for testing.
        M, N, K = 256, 256, 256
        X = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        W = torch.randn(N, K, device=self.device, dtype=torch.bfloat16)
        XQ, _ = torch.ops.fbgemm.quantize_fp8_per_tensor(X)
        WQ, _ = torch.ops.fbgemm.quantize_fp8_per_tensor(W)
        output = torch.empty(M, N, device=self.device, dtype=torch.bfloat16)
        row_scale = torch.randn(M, device=self.device)
        col_scale = torch.randn(N, device=self.device)
        block_scale = torch.randn(M // 128, K // 128, device=self.device)
        tensor_scale = torch.tensor(1.0, device=self.device)

        # Run various compiled quantize ops.
        quantize_input = torch.randn(127, 128, device=self.device, dtype=torch.bfloat16)
        for op in (
            torch.ops.fbgemm.quantize_fp8_per_tensor,
            torch.ops.fbgemm.quantize_fp8_per_row,
            torch.ops.fbgemm.quantize_fp8_per_col,
        ):
            torch.library.opcheck(
                op.default,
                (quantize_input,),
                test_utils=("test_faketensor",),
            )
            eager, _ = op(quantize_input)
            compiled, _ = torch.compile(op)(quantize_input)
            self.assertEqual(eager.dtype, fp8_e4m3)
            self.assertEqual(compiled.dtype, eager.dtype)

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
            legacy_XQ = X.to(torch.float8_e4m3fnuz)
            legacy_WQ = W.to(torch.float8_e4m3fnuz)
            torch.compile(torch.ops.fbgemm.f8f8f16_rowwise)(
                legacy_XQ, legacy_WQ, row_scale, col_scale
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
            gemv_output = torch.compile(torch.ops.fbgemm.fp8fp8bf16_fast_gemv)(
                XQ[:1],
                WQ,
                row_scale[:1],
                col_scale,
                False,
            )
            self.assertEqual(gemv_output.shape, (1, N))
            self.assertEqual(gemv_output.dtype, torch.bfloat16)

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

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10,
        "Skip when SM100 (Blackwell) is not available",
    )
    def test_f8f8bf16_conv(self) -> None:
        # Setup: Create simple 3D convolution test data
        n, d, h, w, c = 1, 4, 8, 8, 16
        k, t, r, s = 16, 3, 3, 3
        fp8_max = E4M3_MAX_POS

        # Create random activation (NDHWC) and filter (KTRSC)
        activation = (
            torch.randn(n, d, h, w, c, dtype=torch.bfloat16, device=self.device) * 0.1
        )
        filter_tensor = (
            torch.randn(k, t, r, s, c, dtype=torch.bfloat16, device=self.device) * 0.01
        )

        # Quantize to FP8
        act_max = torch.max(torch.abs(activation.to(torch.float)))
        flt_max = torch.max(torch.abs(filter_tensor.to(torch.float)))
        act_scale = act_max / fp8_max
        flt_scale = flt_max / fp8_max

        activation_q = (activation * fp8_max / act_max).to(fp8_e4m3)
        filter_q = (filter_tensor * fp8_max / flt_max).to(fp8_e4m3)

        # Execute: Run convolution with padding=1, stride=1, dilation=1
        padding = [1, 1, 1]
        stride = [1, 1, 1]
        dilation = [1, 1, 1]

        output = torch.ops.fbgemm.f8f8bf16_conv(
            activation_q, filter_q, act_scale * flt_scale, padding, stride, dilation
        )

        # Assert: Check output shape is correct
        # Output dimensions: z = 1 + (d + 2*pad_d - ((t-1)*dil_d + 1)) / stride_d
        expected_z = (
            1 + (d + 2 * padding[0] - ((t - 1) * dilation[0] + 1)) // stride[0]
        )  # = 4
        expected_p = (
            1 + (h + 2 * padding[1] - ((r - 1) * dilation[1] + 1)) // stride[1]
        )  # = 8
        expected_q = (
            1 + (w + 2 * padding[2] - ((s - 1) * dilation[2] + 1)) // stride[2]
        )  # = 8

        self.assertEqual(output.shape, (n, expected_z, expected_p, expected_q, k))
        self.assertEqual(output.dtype, torch.bfloat16)

        # Basic sanity check: output should not be all zeros
        self.assertFalse(torch.all(output == 0))


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
            torch.manual_seed(hash((M, N, K)))
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
                # Compare against the dequantized fp8 weight the kernel actually
                # operates on, so the test validates the kernel rather than
                # charging it for the fp8 rounding it is required to incur.
                w_ref = wq.float() * w_scale.float()
                z_ref = (x.float() @ w_ref.T).to(torch.bfloat16)
            elif quantize_w and quantize_x:
                # row-wise scaling
                xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
                wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
                z = gemv_op(xq, wq, x_scale, w_scale)
                x_ref = xq.float() * x_scale.float().reshape(M, 1)
                w_ref = wq.float() * w_scale.float().reshape(N, 1)
                z_ref = (x_ref @ w_ref.T).to(torch.bfloat16)
            else:
                z = gemv_op(x, w)
                z_ref = (x @ w.T).to(torch.bfloat16)
            z_ref = z_ref.to(self.device)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)

    def run_gemv_batched(self, test_cases, gemv_op, atol, rtol):
        for B, M, N, K in test_cases:
            torch.manual_seed(hash((B, M, N, K)))
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
            # Compare against the dequantized fp8 operands the kernel uses.
            x_ref = xq.float() * x_scale.float().view(B, M, 1)
            w_ref = wq.float() * w_scale.float().view(B, N, 1)
            z_ref = (
                torch.bmm(x_ref, w_ref.transpose(1, 2))
                .to(torch.bfloat16)
                .to(self.device)
            )
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)

    def test_gemv_shared_reduction_batches(self) -> None:
        cases = [(2, 7168, 8192), (4, 8192, 3584)]
        for _ in range(10):
            self.run_gemv(cases, torch.ops.fbgemm.bf16_fast_gemv, 1.0e-3, 1.0e-2)
            self.run_gemv(
                cases,
                torch.ops.fbgemm.bf16fp8bf16_fast_gemv,
                1.0e-3,
                1.0e-2,
                quantize_w=True,
            )

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
        self.run_gemv(test_cases, torch.ops.fbgemm.bf16_fast_gemv, 1.0e-1, 1.0e-1)

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
            3.0e-1,
            3.0e-1,
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
            3.0e-1,
            3.0e-1,
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
            3.0e-1,
            3.0e-1,
        )


@unittest.skipIf(
    not torch.cuda.is_available() or torch.version.hip,
    "Skip when cuda is not available or HIP is enabled",
)
@unittest.skipIf(not SUPPORTS_NVFP4, "Skip if NVFP4 is not supported")
class NVFP4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

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

    @settings(deadline=None)
    @given(
        M=st.sampled_from([1, 250]),
        N=st.sampled_from([256, 1024]),
        K=st.sampled_from([2048, 3584]),
    )
    def test_gemm(self, M: int, N: int, K: int) -> None:
        A = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        B = torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        global_scales, a_global_scales, b_global_scales = get_nvfp4_global_scales_naive(
            [A],
            [B],
        )
        aqs, a_scales = quantize_nvfp4_naive([A], a_global_scales)
        bqs, b_scales = quantize_nvfp4_naive([B], b_global_scales)

        out_nvfp4 = torch.ops.fbgemm.f4f4bf16(
            aqs[0], bqs[0], a_scales[0], b_scales[0], None, global_scales[0]
        )
        out_bf16 = A @ B.t()

        torch.testing.assert_close(out_nvfp4, out_bf16, atol=5.0e-2, rtol=5.0e-2)

    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 16]),
        M=st.sampled_from([250, 500, 3500]),
        N=st.sampled_from([256, 1024, 6144]),
        K=st.sampled_from([2048, 3584]),
    )
    def test_grouped_gemm_2d_3d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        XS = [
            torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
            for _ in range(G)
        ]
        WS = [
            torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
            for _ in range(G)
        ]
        offsets = torch.arange(M, G * (M + 1), M, dtype=torch.int32, device=self.device)

        global_scales, x_global_scales, w_global_scales = get_nvfp4_global_scales_naive(
            XS, WS
        )
        xqs, x_scales = quantize_nvfp4_naive(XS, x_global_scales)
        wqs, w_scales = quantize_nvfp4_naive(WS, w_global_scales)

        xq = torch.cat(xqs, dim=0).view(torch.float4_e2m1fn_x2)
        wq = torch.stack(wqs, dim=0).view(torch.float4_e2m1fn_x2)
        x_scale = torch.stack(x_scales, dim=0).view(torch.float8_e4m3fn)
        w_scale = torch.stack(w_scales, dim=0).view(torch.float8_e4m3fn)
        global_scale = torch.stack(global_scales, dim=0)

        X = torch.cat(XS, dim=0)
        W = torch.stack(WS, dim=0)

        out_bf16 = torch._grouped_mm(
            X, W.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16
        )
        out_nvfp4 = torch.ops.fbgemm.f4f4bf16_grouped_mm(
            xq, wq.transpose(-2, -1), x_scale, w_scale, offsets, None, global_scale
        )
        self.assertTrue(out_nvfp4.isfinite().all(), "output contains non-finite values")

        torch.testing.assert_close(out_nvfp4, out_bf16, atol=5.0e-2, rtol=5.0e-2)

    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 16]),
        M=st.sampled_from([250, 500, 3500]),
        N=st.sampled_from([256, 1024, 6144]),
        K=st.sampled_from([2048, 3584]),
    )
    def test_grouped_gemm_2d_2d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        XS = [
            torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
            for _ in range(G)
        ]
        WS = [
            torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
            for _ in range(G)
        ]
        offsets = torch.arange(K, G * (K + 1), K, dtype=torch.int32, device=self.device)

        global_scales, x_global_scales, w_global_scales = get_nvfp4_global_scales_naive(
            XS, WS
        )
        xqs, x_scales = quantize_nvfp4_naive(XS, x_global_scales)
        wqs, w_scales = quantize_nvfp4_naive(WS, w_global_scales)

        xq = torch.cat(xqs, dim=1).view(torch.float4_e2m1fn_x2)
        wq = torch.cat(wqs, dim=1).view(torch.float4_e2m1fn_x2)
        x_scale = torch.stack(x_scales, dim=0).view(torch.float8_e4m3fn)
        w_scale = torch.stack(w_scales, dim=0).view(torch.float8_e4m3fn)
        global_scale = torch.stack(global_scales, dim=0)

        X = torch.cat(XS, dim=1)
        W = torch.cat(WS, dim=1)

        out_bf16 = torch._grouped_mm(
            X, W.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16
        )
        out_nvfp4 = torch.ops.fbgemm.f4f4bf16_grouped_mm(
            xq, wq.transpose(-2, -1), x_scale, w_scale, offsets, None, global_scale
        )
        self.assertTrue(out_nvfp4.isfinite().all(), "output contains non-finite values")

        torch.testing.assert_close(out_nvfp4, out_bf16, atol=5.0e-2, rtol=5.0e-2)


@unittest.skipIf(
    not torch.cuda.is_available() or torch.version.hip,
    "Skip when cuda is not available or HIP is enabled",
)
@unittest.skipIf(not SUPPORTS_MXFP4, "Skip if MXFP4 is not supported")
class MXFP4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @settings(deadline=None)
    @given(
        M=st.sampled_from([1, 250]),
        N=st.sampled_from([256, 1024]),
        K=st.sampled_from([2048, 3584]),
    )
    def test_gemm(self, M: int, N: int, K: int) -> None:
        A = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        B = torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        aq, a_scale = triton_quantize_mx4_unpack(A)
        bq, b_scale = triton_quantize_mx4_unpack(B)

        out_mxfp4 = torch.ops.fbgemm.f4f4bf16(aq, bq, a_scale, b_scale)
        out_bf16 = A @ B.t()

        torch.testing.assert_close(out_mxfp4, out_bf16, atol=8.0e-2, rtol=8.0e-2)

    @given(
        G=st.sampled_from([1, 4, 16]),
        M=st.sampled_from([250, 500, 3500]),
        N=st.sampled_from([256, 1024, 6144]),
        K=st.sampled_from([2048, 3584]),
    )
    def test_grouped_gemm_2d_3d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        XS = [
            torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
            for _ in range(G)
        ]
        WS = [
            torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
            for _ in range(G)
        ]
        offsets = torch.arange(M, G * (M + 1), M, dtype=torch.int32, device=self.device)

        wqs = []
        xqs = []
        w_scales = []
        x_scales = []

        for x, w in zip(XS, WS):
            xq, x_scale = triton_quantize_mx4_unpack(x)
            wq, w_scale = triton_quantize_mx4_unpack(w)

            xqs.append(xq)
            wqs.append(wq)
            x_scales.append(x_scale)
            w_scales.append(w_scale)

        xq = torch.cat(xqs, dim=0).view(torch.float4_e2m1fn_x2)
        wq = torch.stack(wqs, dim=0).view(torch.float4_e2m1fn_x2)
        x_scale = torch.stack(x_scales, dim=0).view(torch.float8_e8m0fnu)
        w_scale = torch.stack(w_scales, dim=0).view(torch.float8_e8m0fnu)

        X = torch.cat(XS, dim=0)
        W = torch.stack(WS, dim=0)

        out_bf16 = torch._grouped_mm(
            X, W.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16
        )
        out_mxfp4 = torch.ops.fbgemm.f4f4bf16_grouped_mm(
            xq, wq.transpose(-2, -1), x_scale, w_scale, offsets
        )
        self.assertTrue(out_mxfp4.isfinite().all(), "output contains non-finite values")

        torch.testing.assert_close(out_mxfp4, out_bf16, atol=8.0e-2, rtol=8.0e-2)

    @settings(deadline=None)
    @given(
        G=st.sampled_from([1, 4, 16]),
        M=st.sampled_from([250, 500, 3500]),
        N=st.sampled_from([256, 1024, 6144]),
        K=st.sampled_from([2048, 3584]),
    )
    def test_grouped_gemm_2d_2d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        XS = [
            torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
            for _ in range(G)
        ]
        WS = [
            torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
            for _ in range(G)
        ]
        offsets = torch.arange(K, G * (K + 1), K, dtype=torch.int32, device=self.device)

        wqs = []
        xqs = []
        w_scales = []
        x_scales = []

        for x, w in zip(XS, WS):
            xq, x_scale = triton_quantize_mx4_unpack(x)
            wq, w_scale = triton_quantize_mx4_unpack(w)

            xqs.append(xq)
            wqs.append(wq)
            x_scales.append(x_scale)
            w_scales.append(w_scale)

        xq = torch.cat(xqs, dim=1).view(torch.float4_e2m1fn_x2)
        wq = torch.cat(wqs, dim=1).view(torch.float4_e2m1fn_x2)
        x_scale = torch.stack(x_scales, dim=0).view(torch.float8_e8m0fnu)
        w_scale = torch.stack(w_scales, dim=0).view(torch.float8_e8m0fnu)

        X = torch.cat(XS, dim=1)
        W = torch.cat(WS, dim=1)

        out_bf16 = torch._grouped_mm(
            X, W.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16
        )
        out_mxfp4 = torch.ops.fbgemm.f4f4bf16_grouped_mm(
            xq, wq.transpose(-2, -1), x_scale, w_scale, offsets
        )
        self.assertTrue(out_mxfp4.isfinite().all(), "output contains non-finite values")

        torch.testing.assert_close(out_mxfp4, out_bf16, atol=8.0e-2, rtol=8.0e-2)


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when MI300 or H100 is not available",
)
class BF16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    def generate_random_splits(G: int, M: int) -> torch.Tensor:
        if M > 0:
            m_cumsums = torch.sort(
                torch.randint(
                    0,
                    M,
                    (G + 1,),
                    dtype=torch.int32,
                    device=torch.accelerator.current_accelerator(),
                )
            ).values
            m_cumsums[0], m_cumsums[-1] = 0, M
            m_sizes = m_cumsums[1:] - m_cumsums[:-1]
            return m_sizes
        else:
            return torch.zeros(
                (G,), dtype=torch.int32, device=torch.accelerator.current_accelerator()
            )

    @unittest.skipIf(
        not torch.version.cuda,
        "Skip on AMD: test_grouped_gemm_wgrad not yet suported.",
    )
    @settings(deadline=None)
    @given(
        G=st.sampled_from([2, 16]),
        M=st.sampled_from([0, 257, 2049]),
        N=st.sampled_from([256, 2048]),
        K=st.sampled_from([128, 1024]),
        dtype=st.sampled_from([torch.bfloat16, torch.float16]),
        output_accum=st.booleans(),
    )
    def test_grouped_gemm_wgrad(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
        output_accum: bool,
    ) -> None:
        torch.manual_seed(hash((G, M, N, K)))
        # Inputs
        dy_bf16 = torch.randn(
            (M, N), dtype=dtype, device=torch.accelerator.current_accelerator()
        )
        x_bf16 = torch.randn(
            (M, K), dtype=dtype, device=torch.accelerator.current_accelerator()
        )

        m_sizes = BF16Tests.generate_random_splits(G, M)

        # Test
        if output_accum:
            wgrad_accum = torch.randn(
                (G, N, K),
                dtype=torch.float32,
                device=torch.accelerator.current_accelerator(),
            )
        else:
            wgrad_accum = None

        sm_margin = 0
        num_sms = (
            torch.cuda.get_device_properties("cuda").multi_processor_count - sm_margin
        )

        test_wgrad = torch.ops.fbgemm.bf16bf16bf16_grouped_wgrad(
            dy_bf16,
            x_bf16,
            m_sizes.to(torch.int64),
            output=wgrad_accum.clone() if output_accum else None,
            output_accum=output_accum,
            num_sms=num_sms,
        )

        if output_accum:
            assert test_wgrad.dtype == torch.float32

        # Reference
        dy_fp32 = dy_bf16.to(torch.float32)
        x_fp32 = x_bf16.to(torch.float32)
        ref_wgrad = torch.zeros(
            (G, N, K),
            dtype=torch.float32,
            device=torch.accelerator.current_accelerator(),
        )

        # Track which groups have non-zero size for comparison
        non_zero_groups = []
        m_start = 0
        for g, m_size in enumerate(m_sizes.tolist()):
            if m_size > 0:
                # Actual slice - compute matrix multiplication
                ref_wgrad[g, :, :] = (
                    dy_fp32[m_start : m_start + m_size, :].T
                    @ x_fp32[m_start : m_start + m_size, :]
                )
                non_zero_groups.append(g)
            m_start += m_size

        if output_accum:
            assert wgrad_accum is not None
            ref_wgrad += wgrad_accum

        ref_wgrad = ref_wgrad.to(test_wgrad.dtype)

        # Float16 is slightly less accurate.
        atol = 1e-4 if dtype == torch.bfloat16 else 1e-3
        # Compare groups with non-zero m_size
        if non_zero_groups:
            if test_wgrad.dtype == torch.float32:
                torch.testing.assert_close(
                    test_wgrad[non_zero_groups],
                    ref_wgrad[non_zero_groups],
                    atol=atol,
                    rtol=1e-3,
                )
            else:
                torch.testing.assert_close(
                    test_wgrad[non_zero_groups],
                    ref_wgrad[non_zero_groups],
                    atol=atol,
                    rtol=1e-2,
                )

    @unittest.skipIf(
        not torch.version.cuda,
        "Skip on AMD: test_grouped_gemm_dgrad not yet suported.",
    )
    @settings(deadline=None)
    @given(
        G=st.sampled_from([2, 16]),
        M=st.sampled_from([0, 257, 2049]),
        N=st.sampled_from([256, 2048]),
        K=st.sampled_from([128, 1024]),
        dtype=st.sampled_from([torch.bfloat16, torch.float16]),
    )
    def test_grouped_gemm_dgrad(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(hash((G, M, N, K)))

        # Inputs
        dy_half = torch.randn(
            (M, N), dtype=dtype, device=torch.accelerator.current_accelerator()
        )
        w_half = torch.randn(
            (G, N, K),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(),
        )
        m_sizes = BF16Tests.generate_random_splits(G, M)

        sm_margin = 0
        num_sms = (
            torch.cuda.get_device_properties("cuda").multi_processor_count - sm_margin
        )

        y_half = torch.ops.fbgemm.bf16bf16bf16_grouped_grad(
            dy_half,
            w_half.permute(0, 2, 1),
            m_sizes.to(torch.int64),
            num_sms=num_sms,
        )

        Y_preallocated = torch.empty(
            (M * K),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(),
        )
        y_half_preallocated = torch.ops.fbgemm.bf16bf16bf16_grouped_grad(
            dy_half,
            w_half.permute(0, 2, 1),
            m_sizes.to(torch.int64),
            Y_preallocated,
            num_sms=num_sms,
        )

        # Reference
        dy_fp32 = dy_half.to(torch.float32)
        w_fp32 = w_half.to(torch.float32)

        ref_y_fp32 = torch.empty(
            (M, K), dtype=torch.float32, device=torch.accelerator.current_accelerator()
        )
        m_start = 0
        for g, m_size in enumerate(m_sizes.tolist()):
            ref_y_fp32[m_start : m_start + m_size, :] = dy_fp32[
                m_start : m_start + m_size, :
            ] @ w_fp32[g, :, :].view(N, K)
            m_start += m_size
        ref_y_half = ref_y_fp32.to(dtype)

        torch.testing.assert_close(y_half, ref_y_half, atol=1e-3, rtol=1.6e-2)
        torch.testing.assert_close(
            y_half_preallocated, ref_y_half, atol=1e-3, rtol=1.6e-2
        )

    @unittest.skipIf(
        not torch.version.cuda,
        "Skip on AMD: test_grouped_gemm_fprop not yet suported.",
    )
    @settings(deadline=None)
    @given(
        G=st.sampled_from([2, 16]),
        M=st.sampled_from([0, 257, 2049]),
        N=st.sampled_from([256, 2048]),
        K=st.sampled_from([128, 1024]),
        dtype=st.sampled_from([torch.bfloat16, torch.float16]),
    )
    def test_grouped_gemm_fprop(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(hash((G, M, N, K)))

        # Inputs
        x_half = torch.randn(
            (M, K), dtype=dtype, device=torch.accelerator.current_accelerator()
        )
        w_half = torch.randn(
            (G, N, K),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(),
        )
        m_sizes = BF16Tests.generate_random_splits(G, M)

        sm_margin = 0
        num_sms = (
            torch.cuda.get_device_properties("cuda").multi_processor_count - sm_margin
        )

        y_half = torch.ops.fbgemm.bf16bf16bf16_grouped_stacked(
            x_half, w_half, m_sizes.to(torch.int64), num_sms=num_sms
        )

        Y_preallocated = torch.empty(
            (M * N),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(),
        )
        y_half_Y_preallocated = torch.ops.fbgemm.bf16bf16bf16_grouped_stacked(
            x_half, w_half, m_sizes.to(torch.int64), Y_preallocated, num_sms=num_sms
        )

        # Reference
        x_fp32 = x_half.to(torch.float32)
        w_fp32 = w_half.to(torch.float32)

        ref_y_fp32 = torch.empty(
            (M, N), dtype=torch.float32, device=torch.accelerator.current_accelerator()
        )
        m_start = 0
        for g, m_size in enumerate(m_sizes.tolist()):
            ref_y_fp32[m_start : m_start + m_size, :] = (
                x_fp32[m_start : m_start + m_size, :] @ w_fp32[g, :, :].view(N, K).T
            )
            m_start += m_size
        ref_y_half = ref_y_fp32.to(dtype)

        torch.testing.assert_close(y_half, ref_y_half, atol=1e-3, rtol=1.6e-2)
        torch.testing.assert_close(
            y_half_Y_preallocated, ref_y_half, atol=1e-3, rtol=1.6e-2
        )

    @unittest.skipIf(
        not torch.version.cuda,
        "Skip on AMD: test not yet supported.",
    )
    @settings(deadline=None)
    @given(
        N=st.sampled_from([256, 2048]),
        K=st.sampled_from([128, 1024]),
        output_accum=st.booleans(),
        all_zero=st.booleans(),
    )
    def test_grouped_gemm_wgrad_zero_token_experts(
        self,
        N: int,
        K: int,
        output_accum: bool,
        all_zero: bool,
    ) -> None:
        """Test wgrad produces zeros (not NaN) for experts with zero tokens.

        Covers two cases:
        - all_zero=True:  total_M == 0, all experts receive zero tokens.
        - all_zero=False: total_M > 0, some experts receive tokens, others don't.
        """
        G = 8
        if all_zero:
            M = 0
            m_sizes = torch.zeros(
                G, dtype=torch.int32, device=torch.accelerator.current_accelerator()
            )
            zero_experts = list(range(G))
        else:
            M = 100
            m_sizes = torch.zeros(
                G, dtype=torch.int32, device=torch.accelerator.current_accelerator()
            )
            # Experts 0, 3, 5 get tokens; others get zero
            m_sizes[0] = 40
            m_sizes[3] = 30
            m_sizes[5] = 30
            zero_experts = [1, 2, 4, 6, 7]

        torch.manual_seed(hash((G, M, N, K)))
        dy_bf16 = torch.randn(
            (M, N), dtype=torch.bfloat16, device=torch.accelerator.current_accelerator()
        )
        x_bf16 = torch.randn(
            (M, K), dtype=torch.bfloat16, device=torch.accelerator.current_accelerator()
        )

        if output_accum:
            wgrad_accum = torch.randn(
                (G, N, K),
                dtype=torch.float32,
                device=torch.accelerator.current_accelerator(),
            )
        else:
            wgrad_accum = None

        sm_margin = 0
        num_sms = (
            torch.cuda.get_device_properties("cuda").multi_processor_count - sm_margin
        )

        # Poison the CUDA caching allocator by running the same kernel with all
        # experts having tokens, producing a non-zero output buffer of exactly
        # G*N*K elements. When freed, the caching allocator will reuse this
        # dirty block for the next at::empty(G*N*K, ...) call inside the kernel.
        # Zero-token experts then retain stale non-zero values.
        warmup_M = G * 10
        warmup_dy = torch.randn(
            (warmup_M, N),
            dtype=torch.bfloat16,
            device=torch.accelerator.current_accelerator(),
        )
        warmup_x = torch.randn(
            (warmup_M, K),
            dtype=torch.bfloat16,
            device=torch.accelerator.current_accelerator(),
        )
        warmup_m_sizes = torch.full(
            (G,), 10, dtype=torch.int64, device=torch.accelerator.current_accelerator()
        )
        warmup_result = torch.ops.fbgemm.bf16bf16bf16_grouped_wgrad(
            warmup_dy,
            warmup_x,
            warmup_m_sizes,
            output_accum=False,
            num_sms=num_sms,
        )
        # Sanity: warmup produced non-zero values that will pollute the buffer
        assert warmup_result.abs().sum() > 0, "Warmup should produce non-zero output"
        del warmup_result, warmup_dy, warmup_x, warmup_m_sizes

        test_wgrad = torch.ops.fbgemm.bf16bf16bf16_grouped_wgrad(
            dy_bf16,
            x_bf16,
            m_sizes.to(torch.int64),
            output=wgrad_accum.clone() if output_accum else None,
            output_accum=output_accum,
            num_sms=num_sms,
        )

        self.assertFalse(
            torch.isnan(test_wgrad).any().item(),
            f"NaN found in wgrad (all_zero={all_zero})",
        )

        for idx in zero_experts:
            if output_accum:
                expected = wgrad_accum[idx].to(test_wgrad.dtype)
            else:
                expected = torch.zeros(
                    (N, K),
                    dtype=test_wgrad.dtype,
                    device=torch.accelerator.current_accelerator(),
                )
            torch.testing.assert_close(
                test_wgrad[idx],
                expected,
                atol=1e-4,
                rtol=1e-3,
                msg=f"Expert {idx} (zero tokens, all_zero={all_zero}) has unexpected gradient values",
            )


if __name__ == "__main__":
    unittest.main()
