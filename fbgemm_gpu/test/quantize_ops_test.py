# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import sys
import unittest
from ctypes import c_float, c_int32, cast, POINTER, pointer
from typing import Callable, Dict, List, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from hypothesis import assume, given, HealthCheck, settings, Verbosity
from torch import Tensor


try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import (  # noqa: F401
        bytes_to_half_floats,
        fused_rowwise_8bit_dequantize_reference,
        fused_rowwise_8bit_quantize_reference,
        fused_rowwise_nbit_quantize_dequantize_reference,
        fused_rowwise_nbit_quantize_reference,
        gpu_available,
        gpu_unavailable,
        optests,
        symint_vector_unsupported,
    )
except Exception:
    if torch.version.hip:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_hip")
    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    from fbgemm_gpu.test.test_utils import (
        bytes_to_half_floats,
        fused_rowwise_8bit_dequantize_reference,
        fused_rowwise_8bit_quantize_reference,
        fused_rowwise_nbit_quantize_dequantize_reference,
        fused_rowwise_nbit_quantize_reference,
        gpu_available,
        gpu_unavailable,
        optests,
        symint_vector_unsupported,
    )

no_long_tests: bool = False


class TestFused8BitRowwiseQuantizationConversion(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 0, $parameter$max_value =
    #  100)` to decorator factory `hypothesis.given`.
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
        is_half=st.booleans(),
        test_float_or_half_op=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(
        self,
        nrows: int,
        ncols: int,
        is_half: bool,
        test_float_or_half_op: bool,
    ) -> None:
        input_data = torch.rand(nrows, ncols).float()
        if is_half:
            input_data = torch.rand(nrows, ncols).half()

        if test_float_or_half_op:
            quantized_data = torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(
                input_data
            )
        else:
            if not is_half:
                quantized_data = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                    input_data
                )
            else:
                quantized_data = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(
                    input_data
                )

        if nrows == 0 or ncols == 0:
            assert quantized_data.numel() == nrows * ((ncols + 3) // 4 * 4 + 8)
            return

        reference = fused_rowwise_8bit_quantize_reference(input_data.float().numpy())

        np.testing.assert_array_almost_equal(quantized_data.numpy(), reference)

        if gpu_available:
            input_data_gpu = input_data.cuda()
            if test_float_or_half_op:
                quantized_data_gpu = (
                    torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(
                        input_data_gpu
                    )
                )
            else:
                if not is_half:
                    quantized_data_gpu = (
                        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                            input_data_gpu
                        )
                    )
                else:
                    quantized_data_gpu = (
                        torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input_data_gpu)
                    )
            quantized_data_numpy = quantized_data_gpu.cpu().numpy()
            ncols_aligned = (ncols + 4 - 1) // 4 * 4
            # compare quantized data
            np.testing.assert_allclose(
                quantized_data_numpy[:, :ncols],
                reference[:, :ncols],
                # Allow 1 mantissa bit difference (LSB)
                atol=1,
            )
            # compare scales
            np.testing.assert_array_almost_equal(
                quantized_data_numpy[:, ncols_aligned : ncols_aligned + 4],
                reference[:, ncols : ncols + 4],
            )
            # compare zero points
            np.testing.assert_array_equal(
                quantized_data_numpy[:, ncols_aligned + 4 : ncols_aligned + 8],
                reference[:, ncols + 4 : ncols + 8],
            )

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
        output_dtype=st.sampled_from(
            [SparseType.FP16, SparseType.FP32, SparseType.BF16]
        ),
        test_generic_op=st.booleans(),
        test_cuda=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op(
        self,
        nrows: int,
        ncols: int,
        output_dtype: SparseType,
        test_generic_op: bool,
        test_cuda: bool,
    ) -> None:
        num_elem_per_byte = 1
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
                    torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(input_data)
                )
                dequantized_data = (
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatOrHalf(
                        quantized_data,
                        output_dtype.as_int(),
                    )
                )
            else:
                if output_dtype == SparseType.FP32:
                    quantized_data = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                        input_data
                    )
                    dequantized_data = (
                        torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                            quantized_data
                        )
                    )
                elif output_dtype == SparseType.FP16:
                    quantized_data = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(
                        input_data
                    )
                    dequantized_data = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(
                        quantized_data
                    )
                else:
                    raise NotImplementedError("Unsupported dtype")

            if nrows == 0 or ncols == 0:
                assert dequantized_data.numel() == 0
                return

            reference = torch.from_numpy(
                fused_rowwise_8bit_dequantize_reference(quantized_data.numpy())
            )
            if output_dtype == SparseType.FP32:
                torch.testing.assert_close(dequantized_data.float(), reference.float())
            elif output_dtype == SparseType.FP16:
                torch.testing.assert_close(dequantized_data.half(), reference.half())
        if test_cuda and gpu_available:
            if nrows == 0 or ncols == 0:
                return
            input_data_gpu = input_data.cuda()

            if test_generic_op:
                quantized_data_gpu = (
                    torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(
                        input_data_gpu
                    )
                )
                dequantized_data_gpu = (
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatOrHalf(
                        quantized_data_gpu,
                        output_dtype.as_int(),
                    )
                )
            else:
                # legacy path does not support bf16
                if SparseType.BF16 == output_dtype:
                    return
                if output_dtype == SparseType.FP32:
                    quantized_data_gpu = (
                        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                            input_data_gpu
                        )
                    )
                    dequantized_data_gpu = (
                        torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                            quantized_data_gpu
                        )
                    )
                elif output_dtype == SparseType.FP16:
                    quantized_data_gpu = (
                        torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input_data_gpu)
                    )
                    dequantized_data_gpu = (
                        torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(
                            quantized_data_gpu
                        )
                    )

            dequantized_data_trimmed = dequantized_data_gpu[:, :ncols].cpu()
            quantize_data_numpy = quantized_data_gpu.cpu().numpy()
            reference = torch.from_numpy(
                fused_rowwise_8bit_dequantize_reference(quantize_data_numpy)[:, :ncols]
            )

            if output_dtype == SparseType.FP32:
                torch.testing.assert_close(
                    dequantized_data_trimmed.float(), reference.float()
                )
            elif output_dtype == SparseType.FP16:
                torch.testing.assert_close(
                    dequantized_data_trimmed.half(), reference.half()
                )
            elif output_dtype == SparseType.BF16:
                torch.testing.assert_close(
                    dequantized_data_trimmed.bfloat16(), reference.bfloat16()
                )

    @unittest.skipIf(no_long_tests, "Slow test, requires buck build to run.")  # noqa
    def test_quantize_and_dequantize_op_cuda_large_nrows(self) -> None:
        ncols = 256
        nrows = 65540

        input_data = torch.rand(nrows, ncols).float()
        quantized_data = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data)

        reference = torch.from_numpy(
            fused_rowwise_8bit_dequantize_reference(quantized_data.numpy())
        )

        if gpu_available:
            input_data_gpu = input_data.cuda()
            quantized_data_gpu = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                input_data_gpu
            )
            dequantized_data_gpu = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                quantized_data_gpu
            )
            reference = torch.from_numpy(
                fused_rowwise_8bit_dequantize_reference(
                    quantized_data_gpu.cpu().numpy()
                )
            )
            # compare quantized data
            torch.testing.assert_close(dequantized_data_gpu.cpu(), reference)


class TestMixedDimInt8DequantizationConversion(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # Pyre was not able to infer the type of argument `not torch.cuda.is_available()`
    # to decorator factory `unittest.skipIf`.
    @unittest.skipIf(*gpu_unavailable)
    def test_mixed_dim_8bit_dequantize_op_empty(self) -> None:
        # assert that kernel return empty tensor and not failing with cuda error
        input_refs = torch.empty((0, 0), dtype=torch.uint8).cuda()
        D_offsets = torch.tensor([0]).cuda()
        mixed_dim_dequant_output = (
            torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatMixedDim(
                input_refs, D_offsets, SparseType.FP32.as_int()
            )
        )
        assert mixed_dim_dequant_output.numel() == 0

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 0, $parameter$max_value =
    #  100)` to decorator factory `hypothesis.given`.
    @given(
        B=st.integers(min_value=1, max_value=100),
        T=st.integers(min_value=1, max_value=100),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        min_dim=st.just(1),
        max_dim=st.just(100),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_mixed_dim_8bit_dequantize_op(
        self,
        B: int,
        T: int,
        output_dtype: SparseType,
        min_dim: int,
        max_dim: int,
    ) -> None:
        self.run_mixed_dim_8bit_dequantize_op_test(B, T, output_dtype, min_dim, max_dim)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 0, $parameter$max_value =
    #  100)` to decorator factory `hypothesis.given`.
    @given(
        B=st.integers(min_value=1, max_value=100),
        T=st.integers(min_value=1, max_value=100),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        min_dim=st.just(100),
        max_dim=st.just(1000),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_mixed_dim_8bit_dequantize_op_large_dims(
        self,
        B: int,
        T: int,
        output_dtype: SparseType,
        min_dim: int,
        max_dim: int,
    ) -> None:
        self.run_mixed_dim_8bit_dequantize_op_test(B, T, output_dtype, min_dim, max_dim)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 0, $parameter$max_value =
    #  100)` to decorator factory `hypothesis.given`.
    @given(
        B=st.just(65540),
        T=st.just(5),
        output_dtype=st.just(SparseType.FP32),
        min_dim=st.just(1),
        max_dim=st.just(100),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_mixed_dim_8bit_dequantize_op_large_rows(
        self,
        B: int,
        T: int,
        output_dtype: SparseType,
        min_dim: int,
        max_dim: int,
    ) -> None:
        self.run_mixed_dim_8bit_dequantize_op_test(B, T, output_dtype, min_dim, max_dim)

    def run_mixed_dim_8bit_dequantize_op_test(
        self,
        B: int,
        T: int,
        output_dtype: SparseType,
        min_dim: int,
        max_dim: int,
    ) -> None:
        table_dims = [
            random.randint(min_dim, max_dim) * 8 for _ in range(T)
        ]  # assume table dimensions are multiples of 8
        table_dims_with_qparams = [d + 8 for d in table_dims]
        D_offsets = (
            torch.cumsum(torch.tensor([0] + table_dims_with_qparams), dim=0)
            .to(torch.int)
            .cuda()
        )
        input_refs = [torch.randn((B, d)).cuda() for d in table_dims]
        input_refs_int8 = [
            torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(t) for t in input_refs
        ]
        input_data = torch.concat(input_refs_int8, dim=1).contiguous()
        mixed_dim_dequant_output = (
            torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatMixedDim(
                input_data, D_offsets, output_dtype.as_int()
            )
        )

        table_output_split = [t + 8 for t in table_dims]
        output_ref = []

        for output_i8 in torch.split(input_data, table_output_split, dim=1):
            output_ref.append(
                torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                    output_i8.contiguous()
                )
            )
        output_ref_concat = torch.cat(output_ref, dim=1)
        if output_dtype == SparseType.FP16:
            output_ref_concat = output_ref_concat.half()

        torch.testing.assert_close(output_ref_concat, mixed_dim_dequant_output)


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
        bit_rate=st.sampled_from([2, 4]),
        is_output_half=st.booleans(),
        test_float_or_half_op=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op(
        self,
        nrows: int,
        ncols: int,
        bit_rate: int,
        is_output_half: bool,
        test_float_or_half_op: bool,
    ) -> None:
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        input_data = torch.rand(nrows, ncols).float()
        if is_output_half:
            input_data = input_data.half()

        assume(ncols % (2 * num_elem_per_byte) == 0)

        if test_float_or_half_op:
            quantized_data = (
                torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                    input_data, bit_rate
                )
            )
            dequantized_data = (
                torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
                    quantized_data,
                    bit_rate,
                    output_dtype=1 if is_output_half else 0,
                )
            )
        else:
            if not is_output_half:
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
            else:
                quantized_data = torch.ops.fbgemm.HalfToFusedNBitRowwiseQuantizedSBHalf(
                    input_data, bit_rate
                )
                dequantized_data = (
                    torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToHalf(
                        quantized_data, bit_rate
                    )
                )
        if nrows == 0 or ncols == 0:
            assert dequantized_data.numel() == 0
            return
        if not is_output_half:
            reference = torch.from_numpy(
                fused_rowwise_nbit_quantize_dequantize_reference(
                    input_data.float().numpy(), bit_rate
                )
            )
        else:
            reference = torch.from_numpy(
                fused_rowwise_nbit_quantize_dequantize_reference(
                    input_data.float().numpy(), bit_rate
                )
            ).half()
        torch.testing.assert_close(dequantized_data, reference)

        if gpu_available:
            input_data_gpu = input_data.cuda()
            if test_float_or_half_op:
                quantized_data_gpu = (
                    torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                        input_data_gpu, bit_rate
                    )
                )
                dequantized_data_gpu = (
                    torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
                        quantized_data_gpu,
                        bit_rate,
                        output_dtype=1 if is_output_half else 0,
                    )
                )
            else:
                if not is_output_half:
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
                else:
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
            # compare quantized data
            torch.testing.assert_close(
                dequantized_data_gpu.cpu().float(), dequantized_data.float()
            )

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


class TestHFP8QuantizationConversion(unittest.TestCase):
    # min_pos is the minimal of denormal numbers
    # min_normal_pos is the minimal of normal numbers
    def _get_hfp8_dynamic_range(
        self, ebits: int, mbits: int, bias: int
    ) -> Tuple[int, int, int]:
        max_pos = (1 << ((1 << ebits) - 2 - bias)) * (2 - 2 ** (-mbits))
        min_pos = 2 ** (1 - bias - mbits)
        min_normal_pos = 2 ** (1 - bias)
        return min_pos, max_pos, min_normal_pos

    def _get_hfp8_config(
        self,
    ) -> Tuple[int, int, Dict[int, int], Dict[int, int], Dict[int, int]]:
        # TODO: set up test for 1-5-2 format
        # TODO: parameterize ebits and mbits in unit test
        ebits = 4
        mbits = 3
        max_pos_dict = {}
        min_pos_dict = {}
        min_normal_pos_dict = {}
        for bias in [4, 5, 6, 7]:
            min_pos, max_pos, min_normal_pos = self._get_hfp8_dynamic_range(
                ebits, mbits, bias
            )
            min_pos_dict[bias] = min_pos
            max_pos_dict[bias] = max_pos
            min_normal_pos_dict[bias] = min_normal_pos

        return ebits, mbits, min_pos_dict, max_pos_dict, min_normal_pos_dict

    def _test_conversion(
        self,
        input_data: Tensor,
        reference_data: Tensor,
        ebits: int,
        exponent_bias: int,
        max_pos: float,
        atol: float = 0.0,
        rtol: float = 1e-7,
    ) -> None:
        if torch.cuda.is_available():
            input_data_gpu = input_data.cuda()
            quantized_data_gpu = torch.ops.fbgemm.FloatToHFP8Quantized(
                input_data_gpu, ebits, exponent_bias, max_pos
            )
            dequantized_data_gpu = torch.ops.fbgemm.HFP8QuantizedToFloat(
                quantized_data_gpu, ebits, exponent_bias
            )
            torch.testing.assert_close(
                dequantized_data_gpu.cpu(), reference_data, rtol=rtol, atol=atol
            )

    # pyre-ignore [56]
    @given(
        nrows=st.integers(min_value=1, max_value=100),
        ncols=st.integers(min_value=1, max_value=100),
        exponent_bias=st.integers(min_value=4, max_value=7),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op(
        self, nrows: int, ncols: int, exponent_bias: int
    ) -> None:
        ebits, mbits, min_pos, max_pos, min_normal_pos = self._get_hfp8_config()
        # test positive normal range
        input_data = torch.FloatTensor((nrows, ncols)).uniform_(
            min_normal_pos[exponent_bias], max_pos[exponent_bias]
        )

        self._test_conversion(
            input_data,
            input_data,
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
            rtol=(2 ** (-mbits - 1)),
            atol=0,
        )

        # test positive denormal range
        input_data = torch.FloatTensor((nrows, ncols)).uniform_(
            min_pos[exponent_bias], min_normal_pos[exponent_bias]
        )
        self._test_conversion(
            input_data,
            input_data,
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
            rtol=0.0,
            atol=(2 ** (1 - exponent_bias - mbits)),
        )

        # test negative normal range
        input_data = torch.FloatTensor((nrows, ncols)).uniform_(
            -max_pos[exponent_bias], -min_normal_pos[exponent_bias]
        )
        self._test_conversion(
            input_data,
            input_data,
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
            rtol=(2 ** (-mbits - 1)),
            atol=0,
        )

        # test negative denormal range
        input_data = torch.FloatTensor((nrows, ncols)).uniform_(
            -min_normal_pos[exponent_bias], -min_pos[exponent_bias]
        )
        self._test_conversion(
            input_data,
            input_data,
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
            rtol=0.0,
            atol=(2 ** (1 - exponent_bias - mbits)),
        )

        # test positive underflow
        input_data = torch.FloatTensor((nrows, ncols)).uniform_(
            0, 0.5 * min_pos[exponent_bias]
        )
        self._test_conversion(
            input_data,
            input_data.new_full(input_data.shape, 0),
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
        )

        # test negative underflow
        input_data = torch.FloatTensor((nrows, ncols)).uniform_(
            -0.5 * min_pos[exponent_bias], 0
        )
        self._test_conversion(
            input_data,
            input_data.new_full(input_data.shape, 0),
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
        )

        # test positive overflow
        input_data = torch.FloatTensor((nrows, ncols)).uniform_(
            max_pos[exponent_bias], max_pos[exponent_bias] * 2
        )
        self._test_conversion(
            input_data,
            input_data.new_full(input_data.shape, max_pos[exponent_bias]),
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
        )

        # test negative overflow
        input_data = torch.FloatTensor((nrows, ncols)).uniform_(
            -max_pos[exponent_bias] * 2, -max_pos[exponent_bias]
        )
        self._test_conversion(
            input_data,
            input_data.new_full(input_data.shape, -max_pos[exponent_bias]),
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
        )


class TestDenseMLPQuantizationConversion(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(self, nrows: int, ncols: int) -> None:
        ebits = 8
        mbits = 7
        bias = 127
        max_pos = (1 << ((1 << ebits) - 2 - bias)) * (2 - 2 ** (-mbits))
        min_pos = 2 ** (1 - bias - mbits)
        bounding_box_size = 16
        print("MSFP parameters", bounding_box_size, ebits, mbits, bias)
        input_data = torch.rand(nrows, ncols).float()
        quantized_data = torch.ops.fbgemm.FloatToMSFPQuantized(
            input_data.cuda(),
            bounding_box_size,
            ebits,
            mbits,
            bias,
            min_pos,
            max_pos,
        )
        dequantized_data = torch.ops.fbgemm.MSFPQuantizedToFloat(
            quantized_data.cuda(), ebits, mbits, bias
        )
        torch.testing.assert_close(dequantized_data.cpu(), input_data, rtol=1, atol=0)


class SparseNNOperatorsGPUTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.sampled_from(["BF16"])` to decorator factory
    #  `hypothesis.given`.
    @given(
        precision=st.just("BF16"),
        batch_size=st.integers(min_value=1, max_value=256),
        k=st.integers(min_value=2, max_value=2),
        n=st.integers(min_value=2, max_value=2),
    )
    def test_dense_mlp_quantize_ops(
        self, precision: str, batch_size: int, k: int, n: int
    ) -> None:
        if precision == "BF16":
            input_data = torch.rand((n, k), dtype=torch.float32)
            quantized_data = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data)
            dequantized_data = torch.ops.fbgemm.Bfloat16QuantizedToFloat(quantized_data)
            torch.testing.assert_close(
                dequantized_data, input_data, rtol=1e-2, atol=1e-2
            )


def bfloat_quantize(x_float: float) -> np.uint16:
    bits = cast(pointer(c_float(x_float)), POINTER(c_int32)).contents.value
    bits += 1 << 15
    bits = bits >> 16
    bits = np.uint16(bits)
    return bits


def bfloat_dequantize(x_bfloat: np.uint16) -> float:
    bits = np.int32(x_bfloat) << 16
    return cast(pointer(c_int32(bits)), POINTER(c_float)).contents.value


class TestBfloat16QuantizationConversion(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 0, $parameter$max_value =
    #  100)` to decorator factory `hypothesis.given`.
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(self, nrows: int, ncols: int) -> None:
        input_data = torch.rand(nrows, ncols).float()
        quantized_data = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data)
        if nrows == 0 or ncols == 0:
            assert quantized_data.numel() == 0
            return
        f = np.vectorize(lambda x: bfloat_quantize(x))
        reference = f(input_data.numpy())
        quantized_data_uint16 = quantized_data.numpy()
        quantized_data_uint16.dtype = np.uint16
        np.testing.assert_array_almost_equal(quantized_data_uint16, reference)

        if torch.cuda.is_available():
            input_data_gpu = input_data.cuda()
            quantized_data_gpu = torch.ops.fbgemm.FloatToBfloat16Quantized(
                input_data_gpu
            )
            quantized_data_numpy = quantized_data_gpu.cpu().numpy()
            quantized_data_numpy.dtype = np.uint16
            np.testing.assert_allclose(quantized_data_numpy, reference)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 0, $parameter$max_value =
    #  100)` to decorator factory `hypothesis.given`.
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op(self, nrows: int, ncols: int) -> None:
        input_data = torch.rand(nrows, ncols).float()
        quantized_data = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data)
        dequantized_data = torch.ops.fbgemm.Bfloat16QuantizedToFloat(quantized_data)
        if nrows == 0 or ncols == 0:
            assert dequantized_data.numel() == 0
            return
        f = np.vectorize(lambda x: bfloat_quantize(x))
        ref_bfloat16 = f(input_data.numpy())
        f = np.vectorize(lambda x: bfloat_dequantize(x))
        ref_fp32 = torch.from_numpy(f(ref_bfloat16)).float()
        torch.testing.assert_close(dequantized_data, ref_fp32)

        if torch.cuda.is_available():
            input_data_gpu = input_data.cuda()
            quantized_data_gpu = torch.ops.fbgemm.FloatToBfloat16Quantized(
                input_data_gpu
            )
            dequantized_data_gpu = torch.ops.fbgemm.Bfloat16QuantizedToFloat(
                quantized_data_gpu
            )
            # compare quantized data
            torch.testing.assert_close(dequantized_data_gpu.cpu(), ref_fp32)

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.sampled_from([(65540, 256), (256, 65540)])` to decorator
    #  factory `hypothesis.given`.
    @given(
        ncols_nrows=st.sampled_from([(65540, 256), (256, 65540)]),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op_cuda_large_nrows_bf16(
        self, ncols_nrows: Tuple[int, int]
    ) -> None:
        ncols, nrows = ncols_nrows
        input_data = torch.rand(nrows, ncols).float()
        quantized_data = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data)
        dequantized_data = torch.ops.fbgemm.Bfloat16QuantizedToFloat(quantized_data)

        if torch.cuda.is_available():
            input_data_gpu = input_data.cuda()
            quantized_data_gpu = torch.ops.fbgemm.FloatToBfloat16Quantized(
                input_data_gpu
            )
            dequantized_data_gpu = torch.ops.fbgemm.Bfloat16QuantizedToFloat(
                quantized_data_gpu
            )
            # compare quantized data
            torch.testing.assert_close(dequantized_data_gpu.cpu(), dequantized_data)


# e.g. "test_faketensor__test_cumsum": [unittest.expectedFailure]
# Please avoid putting tests here, you should put operator-specific
# skips and failures in deeplearning/fbgemm/fbgemm_gpu/test/failures_dict.json
# pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
additional_decorators: Dict[str, List[Callable]] = {
    "test_pt2_compliant_tag_fbgemm_jagged_dense_elementwise_add": [
        # This operator has been grandfathered in. We need to fix this test failure.
        unittest.expectedFailure,
    ],
    "test_pt2_compliant_tag_fbgemm_jagged_dense_elementwise_add_jagged_output": [
        # This operator has been grandfathered in. We need to fix this test failure.
        unittest.expectedFailure,
    ],
}


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class TestFP8RowwiseQuantizationConversion(unittest.TestCase):
    enable_logging: bool = False
    max_examples: int = 40

    def setUp(self) -> None:
        self.enable_logging = bool(os.getenv("FBGEMM_GPU_ENABLE_LOGGING", 0))
        if self.enable_logging:
            logging.info("Enabled logging for TestFP8RowwiseQuantizationConversion")

        torch._dynamo.config.cache_size_limit = self.max_examples
        logging.info(
            f"Setting torch._dynamo.config.cache_size_limit = {self.max_examples}"
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]:
    @given(
        batched=st.booleans(),
        bs=st.integers(min_value=1, max_value=100),
        m=st.integers(min_value=0, max_value=100),
        n=st.integers(min_value=0, max_value=100),
        forward=st.booleans(),
        given_last_dim=st.booleans(),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.bfloat16,
            ],
        ),
        # if before PT 2.1, we don't support symint_vector, so turn it off
        test_compile=st.booleans() if symint_vector_unsupported() else st.just(False),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=max_examples, deadline=None)
    def test_quantize_and_dequantize_op_fp8_rowwise(
        self,
        batched: bool,
        bs: int,
        m: int,
        n: int,
        forward: bool,
        given_last_dim: bool,
        dtype: torch.dtype,
        test_compile: bool,
    ) -> None:
        n = n * 4  # need (n % 4 == 0)
        input_data = (
            torch.rand(bs, m, n, dtype=dtype)
            if batched
            else torch.rand(bs * m, n, dtype=dtype)
        )

        input_data_gpu = input_data.cuda()
        quantized_data_gpu = torch.ops.fbgemm.FloatToFP8RowwiseQuantized(
            input_data_gpu, forward=forward
        )
        quantize_func = (
            torch.compile(
                torch.ops.fbgemm.FP8RowwiseQuantizedToFloat,
                dynamic=True,
                fullgraph=True,
            )
            if test_compile and sys.version_info < (3, 12, 0)
            else torch.ops.fbgemm.FP8RowwiseQuantizedToFloat
        )

        if test_compile:
            torch._dynamo.mark_dynamic(quantized_data_gpu, 0)
            torch._dynamo.mark_dynamic(quantized_data_gpu, 1)

        output_dtype = {
            torch.float: SparseType.FP32,
            torch.half: SparseType.FP16,
            torch.bfloat16: SparseType.BF16,
        }[dtype].as_int()

        dequantized_data_gpu = quantize_func(
            quantized_data_gpu,
            forward=forward,
            output_dtype=output_dtype,
        )

        if m == 0 or n == 0:
            assert dequantized_data_gpu.numel() == 0
            return

        assert (
            dequantized_data_gpu.dtype == dtype
        ), "Result is {dequantized_data_gpu.dtype} type, but expected {dtype}"

        qref = input_data_gpu.float()
        dq = dequantized_data_gpu.float()

        assert not torch.isnan(dq).any(), "Results contain nan"

        if self.enable_logging:
            # Logging quantization errors
            errors = (qref - dq) / (qref + 1e-5)
            logging.info(f"max relative error {errors.abs().max()}")
            val, idx = torch.topk(errors.flatten().abs(), k=min(10, errors.shape[-1]))
            logging.info(f"top-10 errors {val}")
            logging.info(f"ref data {input_data_gpu.flatten()}")
            logging.info(f"dequantized data {dequantized_data_gpu.flatten()}")
            logging.info(f"max relative error {errors.flatten()[idx]}")

        torch.testing.assert_close(qref.cpu(), dq.cpu(), rtol=0.1, atol=0.05)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]:
    @given(
        m=st.integers(min_value=1, max_value=1000),
        n1=st.integers(min_value=1, max_value=1000),
        n2=st.integers(min_value=1, max_value=1000),
        n3=st.integers(min_value=1, max_value=1000),
        row_dim=st.integers(min_value=1, max_value=2048),
        forward=st.booleans(),
        given_last_dim=st.booleans(),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.bfloat16,
            ],
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=max_examples, deadline=None)
    def test_quantize_and_dequantize_op_padded_fp8_rowwise(
        self,
        m: int,
        n1: int,
        n2: int,
        n3: int,
        row_dim: int,
        forward: bool,
        given_last_dim: bool,
        dtype: torch.dtype,
    ) -> None:
        row_dim = row_dim * 4
        device = "cuda"
        input1 = torch.rand(m, n1, device=device, dtype=dtype)
        input2 = torch.rand(m, n2, device=device, dtype=dtype)
        input3 = torch.rand(m, n3, device=device, dtype=dtype)
        output_dtype = {
            torch.float: SparseType.FP32,
            torch.half: SparseType.FP16,
            torch.bfloat16: SparseType.BF16,
        }[dtype].as_int()

        q1 = torch.ops.fbgemm.FloatToPaddedFP8RowwiseQuantized(
            input1, forward=forward, row_dim=row_dim
        )
        q2 = torch.ops.fbgemm.FloatToPaddedFP8RowwiseQuantized(
            input2, forward=forward, row_dim=row_dim
        )
        q3 = torch.ops.fbgemm.FloatToPaddedFP8RowwiseQuantized(
            input3, forward=forward, row_dim=row_dim
        )
        qcat = torch.cat([q1, q3, q2], dim=-1)
        if given_last_dim:
            d_qcat = torch.ops.fbgemm.PaddedFP8RowwiseQuantizedToFloat(
                qcat,
                forward=forward,
                row_dim=row_dim,
                output_last_dim=n1 + n2 + n3,
                output_dtype=output_dtype,
            )
        else:
            d_qcat = torch.ops.fbgemm.PaddedFP8RowwiseQuantizedToFloat(
                qcat,
                forward=forward,
                row_dim=row_dim,
                output_dtype=output_dtype,
            )

        assert (
            d_qcat.dtype == dtype
        ), "Result is {d_qcat.dtype} type, but expected {dtype}"
        qref = torch.cat([input1, input3, input2], dim=-1).cpu().float()
        dqcat = d_qcat.cpu().float()

        assert not torch.isnan(dqcat).any(), "Results contain nan"

        if self.enable_logging:
            # Logging quantization errors
            errors = (dqcat - qref) / (qref + 1e-5)
            assert not torch.isnan(errors).any()
            val, idx = torch.topk(errors.abs(), k=min(10, errors.shape[-1]))
            logging.info(f"top-10 errors {val}")
            logging.info(f"qref {torch.gather(qref, dim=1, index=idx)}")
            logging.info(f"dqcat {torch.gather(dqcat, dim=1, index=idx)}")
            logging.info(
                f"relative error: max: {errors.abs().max()*100:.1f}%, "
                f"median: {errors.abs().median()*100:.1f}%, "
                f"mean: {errors.abs().mean()*100:.1f}%"
            )

        torch.testing.assert_allclose(dqcat, qref, rtol=0.1, atol=0.05)


if __name__ == "__main__":
    unittest.main()
