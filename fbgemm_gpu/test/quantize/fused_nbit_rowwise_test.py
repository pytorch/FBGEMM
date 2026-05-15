# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

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

    @unittest.skipIf(not gpu_available, "Test requires GPU")
    def test_nbit_dequant_fp32_intermediate_precision(self) -> None:
        """Verify that FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf uses fp32
        intermediate arithmetic when dequantizing to BF16.

        The kernel must upcast fp16 scale/bias to fp32 before multiply-add,
        then downcast the fp32 result to bf16. This test constructs inputs
        where computing in fp16 arithmetic (without fp32 upcast) would produce
        a different result from computing in fp32 then casting to bf16.
        """
        bit_rate = 4
        nrows = 4
        # ncols must be divisible by 2 * num_elem_per_byte = 4
        ncols = 8
        num_elem_per_byte = 8 // bit_rate

        # Construct quantized data manually with adversarial scale/bias.
        # We choose scale and bias values where fp16 intermediate arithmetic
        # would overflow or lose precision, but fp32 intermediate is fine.
        #
        # fp16 max is ~65504. If scale=1024 (representable in fp16) and
        # quantized_val=15 (max for 4-bit), then scale*15 = 15360 (fine in fp16).
        # But if bias = 50000 (representable in fp16), then
        # scale*15 + bias = 15360 + 50000 = 65360 (still in fp16 range).
        #
        # Instead, use a case where fp16 accumulation rounds differently from fp32:
        # scale = 2049.0 (in fp16: rounded to 2048.0 due to 10-bit mantissa)
        # but in fp32, 2049.0 is exact.
        # If the kernel does NOT upcast to fp32, it reads scale as fp16 (2048.0)
        # and computes 2048*val + bias. With fp32 upcast, it gets 2049*val + bias.
        #
        # Actually, scale/bias ARE stored as fp16 in the quantized tensor,
        # so 2049.0 stored as fp16 is 2048.0. The kernel reads fp16 bytes.
        # The fp32 upcast of fp16(2049) = fp16(2048) → fp32(2048).
        # So the upcast doesn't help with values that don't fit in fp16.
        #
        # The real difference is in the ARITHMETIC: fp16 multiply-add vs fp32.
        # Use values where fp16 multiply-add loses precision:
        # scale = 0.1 (fp16: 0.0999755859375), bias = 1000.0 (exact in fp16)
        # quantized_val = 15
        # fp32 arithmetic: 0.0999755859375 * 15 + 1000.0 = 1001.49963...
        # fp16 arithmetic: fp16(fp16(0.0999755859375 * 15) + 1000.0)
        #   = fp16(fp16(1.4996...) + 1000.0) = fp16(1.5 + 1000.0) = fp16(1001.5)
        # The key: in fp16 arithmetic, 1000.0 + 1.4996 rounds to 1001.5
        # (since fp16 at magnitude 1000 has precision of 0.5)
        # In fp32: 1000.0 + 1.4996... = 1001.4996... → bf16: 1001.0 or 1002.0
        #
        # Better approach: use exact fp16 representable values that demonstrate
        # the fp32 intermediate is used for the final store to bf16.
        # The kernel stores as: bf16(fp32_result) where fp32_result = scale*q+bias
        # vs hypothetical: bf16(fp16(scale*q+bias))
        #
        # Most reliable test: compute the expected output using fp32 arithmetic
        # on the EXACT fp16 scale/bias values, cast to bf16, and compare
        # bit-for-bit with kernel output.

        packed_dim = (ncols + num_elem_per_byte - 1) // num_elem_per_byte
        # Each row: packed_data (packed_dim bytes) + scale (2 bytes) + bias (2 bytes)
        row_size = packed_dim + 4

        quantized = torch.zeros(nrows, row_size, dtype=torch.uint8)

        # Fill with different scale/bias per row to test various precision scenarios
        test_cases = [
            # (scale_fp16, bias_fp16, quantized_vals)
            # Case 1: Large scale, moderate bias
            (np.float16(1000.0), np.float16(500.0), [15, 7, 3, 1, 14, 8, 2, 0]),
            # Case 2: Small scale, large bias — tests precision of addition
            (np.float16(0.1), np.float16(2000.0), [15, 15, 15, 15, 0, 0, 0, 0]),
            # Case 3: Scale near fp16 boundary
            (np.float16(60000.0), np.float16(0.0), [1, 0, 1, 0, 1, 0, 1, 0]),
            # Case 4: Very small scale and bias
            (np.float16(0.001), np.float16(0.001), [15, 8, 4, 2, 1, 0, 15, 8]),
        ]

        for row_idx, (scale_f16, bias_f16, qvals) in enumerate(test_cases):
            # Pack quantized values (4-bit each, 2 per byte)
            for j in range(ncols):
                byte_idx = j // num_elem_per_byte
                shift = (j % num_elem_per_byte) * bit_rate
                quantized[row_idx, byte_idx] |= (qvals[j] & 0xF) << shift

            # Store scale and bias as fp16 bytes (scale_bias_last=True)
            scale_bytes = np.array([scale_f16]).view(np.uint8)
            bias_bytes = np.array([bias_f16]).view(np.uint8)
            quantized[row_idx, packed_dim] = scale_bytes[0]
            quantized[row_idx, packed_dim + 1] = scale_bytes[1]
            quantized[row_idx, packed_dim + 2] = bias_bytes[0]
            quantized[row_idx, packed_dim + 3] = bias_bytes[1]

        # Compute expected output using fp32 arithmetic on exact fp16 scale/bias
        expected = torch.zeros(nrows, ncols, dtype=torch.bfloat16)
        for row_idx, (scale_f16, bias_f16, qvals) in enumerate(test_cases):
            scale_fp32 = float(scale_f16)  # exact fp16 → fp32
            bias_fp32 = float(bias_f16)
            for j in range(ncols):
                # fp32 arithmetic, then cast to bf16
                val_fp32 = scale_fp32 * qvals[j] + bias_fp32
                expected[row_idx, j] = torch.tensor(val_fp32, dtype=torch.float32).to(
                    torch.bfloat16
                )

        # Run kernel on CUDA with output_dtype=BF16 (5)
        quantized_gpu = quantized.cuda()
        dequantized_gpu = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
            quantized_gpu,
            bit_rate,
            5,  # SparseType.BF16.as_int()
        )

        # Bit-exact comparison: kernel output must match fp32-computed reference
        # cast to bf16, proving fp32 intermediate is used.
        torch.testing.assert_close(
            dequantized_gpu.cpu(),
            expected,
            rtol=0,
            atol=0,
            msg=(
                "FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf BF16 output does not "
                "match fp32-intermediate reference. The kernel may not be upcasting "
                "fp16 scale/bias to fp32 before computation."
            ),
        )

    @unittest.skipIf(not gpu_available, "Test requires GPU")
    @given(
        nrows=st.integers(min_value=1, max_value=50),
        ncols=st.sampled_from([8, 16, 32, 64]),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_fp16_input_int4_quant_dequant_bf16_vs_fp16_roundtrip(
        self,
        nrows: int,
        ncols: int,
    ) -> None:
        """Compare two dequantization paths for fp16 input quantized to INT4:

        Path A (direct bf16):  fp16 → INT4 quant → dequant to BF16
        Path B (via fp16):     fp16 → INT4 quant → dequant to FP16 → cast to BF16

        Both paths should produce identical BF16 results because the kernel
        uses fp32 intermediate arithmetic for both, and the only difference
        is the final fp32→output_dtype cast. Since fp16 has strictly more
        mantissa precision (10 bits) than bf16 (7 bits), casting
        fp32→fp16→bf16 should equal fp32→bf16 for values within fp16 range
        (which embedding values typically are).

        If they differ, it indicates the dequant paths have inconsistent
        intermediate precision.
        """
        bit_rate = 4
        num_elem_per_byte = 8 // bit_rate
        assume(ncols % (2 * num_elem_per_byte) == 0)

        # Start with fp16 input (simulating fp16-trained embeddings)
        input_data_fp16 = torch.rand(nrows, ncols).half()

        # Quantize fp16 → INT4
        quantized_data = torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
            input_data_fp16.cuda(), bit_rate
        )

        # Path A: INT4 → dequant directly to BF16
        dequant_bf16 = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
            quantized_data,
            bit_rate,
            5,  # SparseType.BF16.as_int()
        )

        # Path B: INT4 → dequant to FP16 → cast to BF16
        dequant_fp16 = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
            quantized_data,
            bit_rate,
            1,  # SparseType.FP16.as_int()
        )
        dequant_fp16_to_bf16 = dequant_fp16.to(torch.bfloat16)

        # Both paths should produce identical bf16 values.
        # The kernel uses fp32 arithmetic in both cases:
        #   Path A: fp32_result → bf16
        #   Path B: fp32_result → fp16 → bf16
        # For values in fp16 range, fp32→bf16 truncates to 7-bit mantissa,
        # while fp32→fp16→bf16 first rounds to 10-bit then truncates to 7-bit.
        # These can differ by at most 1 ULP in bf16 due to double-rounding.
        # We use atol=0, rtol=0 to detect ANY difference and document it.
        try:
            torch.testing.assert_close(
                dequant_bf16.cpu(),
                dequant_fp16_to_bf16.cpu(),
                rtol=0,
                atol=0,
            )
        except AssertionError:
            # If bit-exact fails, check that differences are at most 1 bf16 ULP.
            # Double-rounding (fp32→fp16→bf16 vs fp32→bf16) can cause 1 ULP diff.
            diff = (dequant_bf16.float() - dequant_fp16_to_bf16.float()).abs().cpu()
            # Compute 1 ULP for each bf16 value
            bf16_vals = dequant_bf16.cpu().float()
            # bf16 has 7 mantissa bits, so ULP = 2^(exponent - 7)
            # Use torch.finfo to check
            max_diff = diff.max().item()
            max_val = bf16_vals.abs().max().item()
            # Allow up to 1 ULP: for bf16 values around max_val,
            # 1 ULP ≈ max_val * 2^-7 ≈ max_val * 0.0078125
            self.assertLessEqual(
                max_diff,
                max_val * 0.0078125 + 1e-10,
                f"Dequant BF16 direct vs fp16→bf16 differ by more than 1 bf16 ULP. "
                f"max_diff={max_diff}, max_val={max_val}. "
                f"This suggests inconsistent intermediate precision in the kernel.",
            )


if __name__ == "__main__":
    unittest.main()
