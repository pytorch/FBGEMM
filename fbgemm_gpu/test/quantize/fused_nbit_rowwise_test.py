# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from typing import Callable

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from hypothesis import assume, given, HealthCheck, settings

from . import common  # noqa E402

# pyre-fixme[21]: Could not find name `open_source` in
#  `deeplearning.fbgemm.fbgemm_gpu.test.quantize.common`.
from .common import (
    bytes_to_half_floats,
    fused_rowwise_nbit_quantize_dequantize_reference,
    fused_rowwise_nbit_quantize_reference,
    open_source,
)

# pyre-fixme[16]: Module `common` has no attribute `open_source`.
if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, optests
else:
    from fbgemm_gpu.test.test_utils import gpu_available, optests

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")

torch.ops.import_module("fbgemm_gpu.sparse_ops")

no_long_tests: bool = False


@optests.generate_opcheck_tests(fast=True)
class TestFusedNBitRowwiseQuantizationConversion(unittest.TestCase):
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
        bit_rate=st.sampled_from([2, 4]),
        is_half=st.booleans(),
        test_float_or_half_op=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(
        self,
        nrows: int,
        ncols: int,
        bit_rate: int,
        is_half: bool,
        test_float_or_half_op: bool,
    ) -> None:
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        assume(ncols % (2 * num_elem_per_byte) == 0)

        input_data = torch.rand(nrows, ncols).float()
        if is_half:
            input_data = input_data.half()

        if test_float_or_half_op:
            quantized_data = (
                torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                    input_data, bit_rate
                )
            )
        else:
            if not is_half:
                quantized_data = (
                    torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                        input_data, bit_rate
                    )
                )
            else:
                quantized_data = torch.ops.fbgemm.HalfToFusedNBitRowwiseQuantizedSBHalf(
                    input_data, bit_rate
                )
        if nrows == 0 or ncols == 0:
            assert quantized_data.numel() == nrows * (
                (ncols + bit_rate - 1) // bit_rate + 4
            )
            return

        quantized_data = quantized_data.numpy()

        reference = fused_rowwise_nbit_quantize_reference(
            input_data.float().numpy(), bit_rate
        )

        interleaved_dim = ncols // num_elem_per_byte
        # compare quantized data
        np.testing.assert_array_equal(
            quantized_data[:, :interleaved_dim], reference[:, :interleaved_dim]
        )
        # compare scales
        np.testing.assert_array_almost_equal(
            bytes_to_half_floats(
                quantized_data[:, interleaved_dim : interleaved_dim + 2]
            ),
            bytes_to_half_floats(reference[:, interleaved_dim : interleaved_dim + 2]),
        )
        # compare zero points
        np.testing.assert_array_equal(
            quantized_data[:, interleaved_dim + 2], reference[:, interleaved_dim + 2]
        )

        if gpu_available:
            input_data_gpu = input_data.cuda()
            if test_float_or_half_op:
                quantized_data_gpu = (
                    torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                        input_data_gpu, bit_rate
                    )
                )
            else:
                if not is_half:
                    quantized_data_gpu = (
                        torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                            input_data_gpu, bit_rate
                        )
                    )
                else:
                    quantized_data_gpu = (
                        torch.ops.fbgemm.HalfToFusedNBitRowwiseQuantizedSBHalf(
                            input_data_gpu, bit_rate
                        )
                    )
            quantized_data_numpy = quantized_data_gpu.cpu().numpy()
            # compare quantized data
            np.testing.assert_array_equal(
                quantized_data_numpy[:, :ncols], reference[:, :ncols]
            )

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
        bit_rate=st.sampled_from([2, 4, 8]),
        output_dtype=st.sampled_from(
            [SparseType.FP16, SparseType.FP32, SparseType.BF16]
        ),
        test_generic_op=st.booleans(),
        test_meta=st.booleans(),
        test_cuda=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op(
        self,
        nrows: int,
        ncols: int,
        bit_rate: int,
        output_dtype: SparseType,
        test_generic_op: bool,
        test_meta: bool,
        test_cuda: bool,
    ) -> None:
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        input_data = torch.rand(nrows, ncols).float()
        if output_dtype == SparseType.FP16:
            input_data = input_data.half()
        elif output_dtype == SparseType.BF16:
            input_data = input_data.bfloat16()

        assume(ncols % (2 * num_elem_per_byte) == 0)

        if not test_cuda:
            # cpu path does not support bf16
            if output_dtype == SparseType.BF16:
                return
            if test_generic_op:
                quantized_data = (
                    torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                        input_data, bit_rate
                    )
                )
                dequantized_data = (
                    torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
                        quantized_data,
                        bit_rate,
                        output_dtype.as_int(),
                    )
                )
                if test_meta:
                    dequantized_data_meta = (
                        torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
                            quantized_data.to(torch.device("meta")),
                            bit_rate,
                            output_dtype.as_int(),
                        )
                    )
                    assert dequantized_data_meta.device == torch.device("meta")
                    assert dequantized_data_meta.shape == dequantized_data.shape
            else:
                if output_dtype == SparseType.FP32:
                    quantized_data = (
                        torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                            input_data, bit_rate
                        )
                    )
                    dequantized_data = (
                        torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(
                            quantized_data, bit_rate
                        )
                    )
                elif output_dtype == SparseType.FP16:
                    quantized_data = (
                        torch.ops.fbgemm.HalfToFusedNBitRowwiseQuantizedSBHalf(
                            input_data, bit_rate
                        )
                    )
                    dequantized_data = (
                        torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToHalf(
                            quantized_data, bit_rate
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported output dtype {output_dtype} for cpu ops"
                    )
            if nrows == 0 or ncols == 0:
                assert dequantized_data.numel() == 0
                return
            if output_dtype == SparseType.FP32:
                reference = torch.from_numpy(
                    fused_rowwise_nbit_quantize_dequantize_reference(
                        input_data.float().numpy(), bit_rate
                    )
                )
            elif output_dtype == SparseType.FP16:
                reference = torch.from_numpy(
                    fused_rowwise_nbit_quantize_dequantize_reference(
                        input_data.float().numpy(), bit_rate
                    )
                ).half()
            else:
                raise NotImplementedError(
                    f"Unsupported output dtype {output_dtype} for cpu ops"
                )
            torch.testing.assert_close(dequantized_data, reference)

        if test_cuda and gpu_available:
            input_data_gpu = input_data.cuda()
            if test_generic_op:
                quantized_data_gpu = (
                    torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                        input_data_gpu, bit_rate
                    )
                )
                dequantized_data_gpu = (
                    torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
                        quantized_data_gpu,
                        bit_rate,
                        output_dtype.as_int(),
                    )
                )
            else:
                # legacy path does not support bf16
                if SparseType.BF16 == output_dtype:
                    return
                if output_dtype == SparseType.FP32:
                    quantized_data_gpu = (
                        torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                            input_data_gpu, bit_rate
                        )
                    )
                    dequantized_data_gpu = (
                        torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(
                            quantized_data_gpu, bit_rate
                        )
                    )
                elif output_dtype == SparseType.FP16:
                    quantized_data_gpu = (
                        torch.ops.fbgemm.HalfToFusedNBitRowwiseQuantizedSBHalf(
                            input_data_gpu, bit_rate
                        )
                    )
                    dequantized_data_gpu = (
                        torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToHalf(
                            quantized_data_gpu, bit_rate
                        )
                    )
            if nrows == 0 or ncols == 0:
                assert dequantized_data_gpu.numel() == 0
                return
            # compare quantized data
            if output_dtype == SparseType.FP32:
                reference = torch.from_numpy(
                    fused_rowwise_nbit_quantize_dequantize_reference(
                        input_data.float().numpy(), bit_rate
                    )
                )
            elif output_dtype == SparseType.FP16:
                reference = torch.from_numpy(
                    fused_rowwise_nbit_quantize_dequantize_reference(
                        input_data.float().numpy(), bit_rate
                    )
                ).half()
            elif output_dtype == SparseType.BF16:
                reference = torch.from_numpy(
                    fused_rowwise_nbit_quantize_dequantize_reference(
                        input_data.float().numpy(), bit_rate
                    )
                ).bfloat16()
            else:
                raise NotImplementedError(
                    f"Unsupported output dtype for gpu ops {output_dtype}"
                )
            torch.testing.assert_close(dequantized_data_gpu.cpu(), reference)

    @unittest.skipIf(no_long_tests, "Slow test, requires buck build to run.")  # noqa
    def test_quantize_and_dequantize_op_cuda_large_nrows(self) -> None:
        ncols = 256
        bit_rate = 4
        nrows = 65540

        num_elem_per_byte = 8 // bit_rate
        input_data = torch.rand(nrows, ncols).float()
        assume(ncols % (2 * num_elem_per_byte) == 0)

        reference = torch.from_numpy(
            fused_rowwise_nbit_quantize_dequantize_reference(
                input_data.numpy(), bit_rate
            )
        )

        if gpu_available:
            input_data_gpu = input_data.cuda()
            quantized_data_gpu = (
                torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                    input_data_gpu, bit_rate
                )
            )
            dequantized_data_gpu = (
                torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(
                    quantized_data_gpu, bit_rate
                )
            )
            # compare quantized data
            torch.testing.assert_close(dequantized_data_gpu.cpu(), reference)

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
        bit_rate=st.sampled_from([2, 4, 8]),
        output_dtype=st.sampled_from(
            # [SparseType.FP16, SparseType.FP32, SparseType.BF16]
            [SparseType.BF16]
        ),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op_cpu_and_cuda(
        self,
        nrows: int,
        ncols: int,
        bit_rate: int,
        output_dtype: SparseType,
    ) -> None:
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        input_data = torch.rand(nrows, ncols).float()

        assume(ncols % (2 * num_elem_per_byte) == 0)

        quantized_data = torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
            input_data, bit_rate
        )

        if gpu_available:
            # CPU quantized data, then dequantize on CUDA.
            quantized_data = quantized_data.cuda()
            dequantized_data = (
                torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
                    quantized_data,
                    bit_rate,
                    output_dtype.as_int(),
                )
            )
            if nrows == 0 or ncols == 0:
                self.assertEqual(dequantized_data.numel(), 0)
                return
            if output_dtype == SparseType.FP32:
                reference = torch.from_numpy(
                    fused_rowwise_nbit_quantize_dequantize_reference(
                        input_data.float().numpy(), bit_rate
                    )
                )
            elif output_dtype == SparseType.FP16:
                reference = torch.from_numpy(
                    fused_rowwise_nbit_quantize_dequantize_reference(
                        input_data.float().numpy(), bit_rate
                    )
                ).half()
            elif output_dtype == SparseType.BF16:
                reference = torch.from_numpy(
                    fused_rowwise_nbit_quantize_dequantize_reference(
                        input_data.float().numpy(), bit_rate
                    )
                ).bfloat16()
            else:
                raise NotImplementedError(
                    f"Unsupported output dtype for gpu ops {output_dtype}"
                )
            torch.testing.assert_close(dequantized_data.cpu(), reference)


def _bf16_round_trip(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).to(torch.float32)


def _quantize_dequantize_bf16_sb_reference(
    data: torch.Tensor, bit_rate: int
) -> torch.Tensor:
    """Roundtrip reference matching the bf16 scale/bias fbgemm kernel.

    Mirrors the fp16 SB reference but rounds scale/bias through bfloat16
    instead of float16, so the expected value matches what the kernel produces.
    """
    data_f32 = data.float()
    qmax = (1 << bit_rate) - 1
    minimum = _bf16_round_trip(data_f32.min(dim=1, keepdim=True).values)
    maximum = data_f32.max(dim=1, keepdim=True).values
    span = maximum - minimum
    scale = _bf16_round_trip(torch.where(span == 0, torch.ones_like(span), span / qmax))
    inverse_scale = 1.0 / scale
    quantized = torch.clamp(
        torch.round((data_f32 - minimum) * inverse_scale), 0.0, float(qmax)
    )
    return scale * quantized + minimum


class TestFusedNBitRowwiseQuantizationConversionBF16SB(unittest.TestCase):
    """Tests for FusedNBitRowwiseQuantizedSBBFloat16 ops (bf16 scale/bias)."""

    @unittest.skipUnless(gpu_available, "CUDA required for bf16 SB quantize op")
    # pyre-ignore [56]
    @given(
        nrows=st.integers(min_value=1, max_value=64),
        ncols=st.integers(min_value=8, max_value=128),
        bit_rate=st.sampled_from([2, 4]),
        output_dtype=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        ),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_bf16_sb_quantize_dequantize_roundtrip_cuda(
        self,
        nrows: int,
        ncols: int,
        bit_rate: int,
        output_dtype: SparseType,
    ) -> None:
        num_elem_per_byte = 8 // bit_rate
        assume(ncols % (2 * num_elem_per_byte) == 0)

        torch.manual_seed(0)
        input_data = torch.rand(nrows, ncols).float().cuda()

        quantized_gpu = (
            torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBBFloat16(
                input_data, bit_rate
            )
        )
        dequantized_gpu = (
            torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBBFloat16ToFloatOrHalf(
                quantized_gpu, bit_rate, output_dtype.as_int()
            )
        )

        reference = _quantize_dequantize_bf16_sb_reference(input_data.cpu(), bit_rate)
        if output_dtype == SparseType.FP16:
            reference = reference.half()
        elif output_dtype == SparseType.BF16:
            reference = reference.bfloat16()
        torch.testing.assert_close(dequantized_gpu.cpu(), reference)

    @unittest.skipUnless(
        gpu_available, "CUDA needed to produce quantized bytes for CPU dequant test"
    )
    # pyre-ignore [56]
    @given(
        nrows=st.integers(min_value=1, max_value=64),
        ncols=st.integers(min_value=8, max_value=128),
        bit_rate=st.sampled_from([2, 4]),
        output_dtype=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        ),
        scale_bias_last=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_bf16_sb_dequantize_cpu_matches_cuda(
        self,
        nrows: int,
        ncols: int,
        bit_rate: int,
        output_dtype: SparseType,
        scale_bias_last: bool,
    ) -> None:
        """CPU dequant produces the same result as CUDA dequant for bf16 SB bytes."""
        num_elem_per_byte = 8 // bit_rate
        assume(ncols % (2 * num_elem_per_byte) == 0)

        torch.manual_seed(0)
        input_data = torch.rand(nrows, ncols).float().cuda()
        # Quantize on CUDA (only backend that has bf16 SB quantize); dequant on CPU.
        # Quantize op always emits scale_bias_last layout, so for the
        # scale_bias_last=False case we synthesize a front-layout tensor.
        quantized_last = (
            torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBBFloat16(
                input_data, bit_rate
            )
        ).cpu()

        if scale_bias_last:
            quantized = quantized_last
        else:
            packed_dim = quantized_last.size(1) - 2 * 2  # 2 * sizeof(bf16)
            quantized = torch.cat(
                [quantized_last[:, packed_dim:], quantized_last[:, :packed_dim]],
                dim=1,
            ).contiguous()

        dequantized_cpu = (
            torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBBFloat16ToFloatOrHalf(
                quantized, bit_rate, output_dtype.as_int(), scale_bias_last
            )
        )
        dequantized_cuda = (
            torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBBFloat16ToFloatOrHalf(
                quantized.cuda(), bit_rate, output_dtype.as_int(), scale_bias_last
            )
        ).cpu()
        torch.testing.assert_close(dequantized_cpu, dequantized_cuda)

    def test_bf16_sb_dequantize_meta_dispatch(self) -> None:
        """Meta dispatch returns correct shape/dtype without materializing data."""
        bit_rate = 4
        num_elem_per_byte = 8 // bit_rate
        nrows, ncols = 8, 64
        # Quantized layout: ncols/num_elem_per_byte packed bytes + 2 * sizeof(bf16).
        packed_cols = ncols // num_elem_per_byte + 2 * 2
        meta_input = torch.empty((nrows, packed_cols), dtype=torch.uint8, device="meta")

        for output_dtype, expected_dtype in [
            (SparseType.FP32, torch.float32),
            (SparseType.FP16, torch.float16),
            (SparseType.BF16, torch.bfloat16),
        ]:
            out = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBBFloat16ToFloatOrHalf(
                meta_input, bit_rate, output_dtype.as_int()
            )
            self.assertEqual(out.device, torch.device("meta"))
            self.assertEqual(out.shape, (nrows, ncols))
            self.assertEqual(out.dtype, expected_dtype)

    @unittest.skipUnless(
        gpu_available, "CUDA required for fp16 vs bf16 loss comparison"
    )
    def test_fp16_vs_bf16_sb_quantization_loss_comparable(self) -> None:
        """fp16 and bf16 scale/bias produce comparable quantization loss across
        a range of input distributions. The 4-bit quantization grid dominates
        the error; SB precision should change loss by at most ~2x."""
        torch.manual_seed(0)
        bit_rate = 4
        nrows, ncols = 256, 64
        # 4-bit grid has 15 levels; absolute mean error is bounded by span / 30
        # for uniform inputs. We use a looser ceiling of span / 15 as a safety
        # check and require bf16 to stay within 2x of fp16.
        ranges = [
            ("unit_uniform", -1.0, 1.0),
            ("wide_dynamic", -100.0, 100.0),
            ("small_positive", 1e-3, 1.0),
            ("tiny_symmetric", -1e-4, 1e-4),
        ]
        for name, lo, hi in ranges:
            input_data = torch.empty(nrows, ncols).uniform_(lo, hi).float().cuda()
            span = hi - lo

            q_fp16 = torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                input_data, bit_rate
            )
            d_fp16 = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
                q_fp16, bit_rate, SparseType.FP32.as_int()
            )
            loss_fp16 = (input_data - d_fp16).abs().mean().item()

            q_bf16 = torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBBFloat16(
                input_data, bit_rate
            )
            d_bf16 = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBBFloat16ToFloatOrHalf(
                q_bf16, bit_rate, SparseType.FP32.as_int()
            )
            loss_bf16 = (input_data - d_bf16).abs().mean().item()

            ceiling = span / 15.0
            self.assertLess(
                loss_fp16,
                ceiling,
                f"[{name}] fp16 loss {loss_fp16} exceeds ceiling {ceiling}",
            )
            self.assertLess(
                loss_bf16,
                ceiling,
                f"[{name}] bf16 loss {loss_bf16} exceeds ceiling {ceiling}",
            )
            self.assertLess(
                loss_bf16,
                2.0 * loss_fp16 + 1e-7,
                f"[{name}] bf16 loss {loss_bf16} > 2x fp16 loss {loss_fp16}",
            )

    @unittest.skipUnless(gpu_available, "CUDA required for kernel perf benchmark")
    def test_fp16_vs_bf16_sb_dequantize_perf_no_regression(self) -> None:
        """bf16 SB dequant kernel must not regress vs fp16 SB by more than 50%.

        Times the GPU dequant kernel for a representative cache shape across
        several iterations after warmup, using CUDA events for accurate timing.
        """
        bit_rate = 4
        nrows, ncols = 65536, 128
        warmup_iters = 5
        bench_iters = 50

        torch.manual_seed(0)
        input_data = torch.rand(nrows, ncols).float().cuda()

        q_fp16 = torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
            input_data, bit_rate
        )
        q_bf16 = torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBBFloat16(
            input_data, bit_rate
        )

        def bench(op: Callable[[], torch.Tensor]) -> float:
            for _ in range(warmup_iters):
                op()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(bench_iters):
                op()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / bench_iters

        out_dtype_int = SparseType.FP32.as_int()
        fp16_us = (
            bench(
                lambda: torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
                    q_fp16, bit_rate, out_dtype_int
                )
            )
            * 1000.0
        )
        bf16_us = (
            bench(
                lambda: torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBBFloat16ToFloatOrHalf(
                    q_bf16, bit_rate, out_dtype_int
                )
            )
            * 1000.0
        )
        print(
            f"[dequant perf] shape=({nrows}x{ncols}) bit_rate={bit_rate} "
            f"fp16_sb={fp16_us:.2f}us bf16_sb={bf16_us:.2f}us "
            f"ratio={bf16_us / fp16_us:.3f}x"
        )
        self.assertLess(
            bf16_us,
            1.5 * fp16_us,
            f"bf16 SB dequant ({bf16_us:.2f}us) regresses >50% vs fp16 SB "
            f"({fp16_us:.2f}us)",
        )

    @unittest.skipUnless(
        gpu_available and os.environ.get("FBGEMM_BENCH_DEQUANT_SWEEP", "0") == "1",
        "Set FBGEMM_BENCH_DEQUANT_SWEEP=1 with CUDA to run perf sweep",
    )
    def test_fp16_vs_bf16_sb_dequantize_perf_sweep(self) -> None:
        """Wide sweep of dequant kernel perf across cache-sized shapes,
        bit rates and output dtypes. Gated behind FBGEMM_BENCH_DEQUANT_SWEEP=1
        to keep CI cheap. Prints a markdown table; asserts no >50% regression."""
        warmup_iters = 10
        bench_iters = 100
        out_dtype_int = SparseType.FP32.as_int()

        shapes = [
            (65536, 64),
            (65536, 128),
            (65536, 256),
            (262144, 64),
            (262144, 128),
            (262144, 256),
            (1048576, 64),
            (1048576, 128),
            (1048576, 256),
            (4194304, 64),
            (4194304, 128),
        ]
        bit_rates = [2, 4]

        def bench(op: Callable[[], torch.Tensor]) -> float:
            for _ in range(warmup_iters):
                op()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(bench_iters):
                op()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / bench_iters

        torch.manual_seed(0)
        print("\n| shape         | bit | bytes_in | fp16_sb_us | bf16_sb_us | ratio |")
        print("|---------------|-----|----------|-----------:|-----------:|------:|")
        max_ratio = 0.0
        for nrows, ncols in shapes:
            input_data = torch.rand(nrows, ncols).float().cuda()
            for bit_rate in bit_rates:
                q_fp16 = torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                    input_data, bit_rate
                )
                q_bf16 = (
                    torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBBFloat16(
                        input_data, bit_rate
                    )
                )
                bytes_in = q_fp16.numel()
                fp16_us = (
                    bench(
                        lambda q=q_fp16, b=bit_rate: torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
                            q, b, out_dtype_int
                        )
                    )
                    * 1000.0
                )
                bf16_us = (
                    bench(
                        lambda q=q_bf16, b=bit_rate: torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBBFloat16ToFloatOrHalf(
                            q, b, out_dtype_int
                        )
                    )
                    * 1000.0
                )
                ratio = bf16_us / fp16_us
                max_ratio = max(max_ratio, ratio)
                print(
                    f"| {nrows:>7}x{ncols:<3} | {bit_rate:>3} | "
                    f"{bytes_in:>8} | {fp16_us:>10.2f} | {bf16_us:>10.2f} | "
                    f"{ratio:>5.3f}x |"
                )
        print(f"\nWorst ratio (bf16/fp16): {max_ratio:.3f}x")
        self.assertLess(
            max_ratio,
            1.5,
            f"bf16 SB dequant kernel regresses >50% vs fp16 SB "
            f"(worst ratio {max_ratio:.3f}x)",
        )


if __name__ == "__main__":
    unittest.main()
