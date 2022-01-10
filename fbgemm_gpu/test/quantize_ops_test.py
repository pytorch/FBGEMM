# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from hypothesis import HealthCheck, given, assume, settings


try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
    from test_utils import (  # pyre-ignore[21]
        fused_rowwise_8bit_quantize_reference,
        fused_rowwise_8bit_dequantize_reference,
        fused_rowwise_nbit_quantize_reference,
        fused_rowwise_nbit_quantize_dequantize_reference,
        bytes_to_half_floats,
        gpu_available,
    )
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library("//caffe2/torch/fb/sparsenn:sparsenn_operators_gpu")
    torch.ops.load_library("//caffe2/torch/fb/sparsenn:sparsenn_operators")
    from fbgemm_gpu.test.test_utils import (
        fused_rowwise_8bit_quantize_reference,
        fused_rowwise_8bit_dequantize_reference,
        fused_rowwise_nbit_quantize_reference,
        fused_rowwise_nbit_quantize_dequantize_reference,
        bytes_to_half_floats,
        gpu_available,
    )


class TestFused8BitRowwiseQuantizationConversion(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 0, $parameter$max_value =
    #  100)` to decorator factory `hypothesis.given`.
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
        is_half=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(
        self,
        nrows: int,
        ncols: int,
        is_half: bool,
    ) -> None:
        input_data = torch.rand(nrows, ncols).float()
        if is_half:
            input_data = torch.rand(nrows, ncols).half()

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
            if not is_half:
                quantized_data_gpu = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                    input_data_gpu
                )
            else:
                quantized_data_gpu = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(
                    input_data_gpu
                )

            quantized_data_numpy = quantized_data_gpu.cpu().numpy()
            ncols_aligned = (ncols + 4 - 1) // 4 * 4
            # compare quantized data
            np.testing.assert_allclose(
                quantized_data_numpy[:, :ncols], reference[:, :ncols]
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
        is_output_half=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op(
        self,
        nrows: int,
        ncols: int,
        is_output_half: bool,
    ) -> None:
        num_elem_per_byte = 1
        input_data = torch.rand(nrows, ncols).float()
        if is_output_half:
            input_data = input_data.half()

        assume(ncols % (2 * num_elem_per_byte) == 0)

        if not is_output_half:
            quantized_data = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                input_data
            )
            dequantized_data = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                quantized_data
            )
        else:
            quantized_data = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(
                input_data
            )
            dequantized_data = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(
                quantized_data
            )
        if nrows == 0 or ncols == 0:
            assert dequantized_data.numel() == 0
            return

        reference = torch.from_numpy(
            fused_rowwise_8bit_dequantize_reference(quantized_data.numpy())
        )
        if not is_output_half:
            torch.testing.assert_allclose(dequantized_data.float(), reference.float())
        else:
            torch.testing.assert_allclose(dequantized_data.half(), reference.half())

        if gpu_available:
            input_data_gpu = input_data.cuda()

            if not is_output_half:
                quantized_data_gpu = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                    input_data_gpu
                )
                dequantized_data_gpu = (
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                        quantized_data_gpu
                    )
                )
            else:
                quantized_data_gpu = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(
                    input_data_gpu
                )
                dequantized_data_gpu = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(
                    quantized_data_gpu
                )

            dequantized_data_numpy = dequantized_data_gpu.cpu().numpy()
            dequantized_data_trimmed = torch.from_numpy(
                dequantized_data_numpy[:, :ncols]
            )

            if not is_output_half:
                torch.testing.assert_allclose(
                    dequantized_data_trimmed.float(), reference.float()
                )
            else:
                torch.testing.assert_allclose(
                    dequantized_data_trimmed.half(), reference.half()
                )

    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
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
            torch.testing.assert_allclose(dequantized_data_gpu.cpu(), reference)


class TestMixedDimInt8DequantizationConversion(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # Pyre was not able to infer the type of argument `not torch.cuda.is_available()`
    # to decorator factory `unittest.skipIf`.
    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
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

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
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

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
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

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
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

        torch.testing.assert_allclose(output_ref_concat, mixed_dim_dequant_output)


class TestFusedNBitRowwiseQuantizationConversion(unittest.TestCase):
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
        bit_rate=st.sampled_from([2, 4]),
        is_half=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(
        self, nrows: int, ncols: int, bit_rate: int, is_half: bool
    ) -> None:
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        assume(ncols % (2 * num_elem_per_byte) == 0)

        input_data = torch.rand(nrows, ncols).float()
        if is_half:
            input_data = input_data.half()

        if not is_half:
            quantized_data = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                input_data, bit_rate
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
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op(
        self,
        nrows: int,
        ncols: int,
        bit_rate: int,
        is_output_half: bool,
    ) -> None:
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        input_data = torch.rand(nrows, ncols).float()
        if is_output_half:
            input_data = input_data.half()

        assume(ncols % (2 * num_elem_per_byte) == 0)

        if not is_output_half:
            quantized_data = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                input_data, bit_rate
            )
            dequantized_data = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(
                quantized_data, bit_rate
            )
        else:
            quantized_data = torch.ops.fbgemm.HalfToFusedNBitRowwiseQuantizedSBHalf(
                input_data, bit_rate
            )
            dequantized_data = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToHalf(
                quantized_data, bit_rate
            )
        if nrows == 0 or ncols == 0:
            assert dequantized_data.numel() == 0
            return
        reference = torch.from_numpy(
            fused_rowwise_nbit_quantize_dequantize_reference(
                input_data.float().numpy(), bit_rate
            )
        )
        torch.testing.assert_allclose(dequantized_data, reference)

        if gpu_available:
            input_data_gpu = input_data.cuda()
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
            torch.testing.assert_allclose(
                dequantized_data_gpu.cpu().float(), dequantized_data.float()
            )

    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
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
            torch.testing.assert_allclose(dequantized_data_gpu.cpu(), reference)


class TestDenseMLPQuantizationConversion(unittest.TestCase):
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
        quantized_data = torch.ops.fb.FloatToMSFPQuantized(
            input_data.cuda(),
            bounding_box_size,
            ebits,
            mbits,
            bias,
            min_pos,
            max_pos,
        )
        dequantized_data = torch.ops.fb.MSFPQuantizedToFloat(
            quantized_data.cuda(), ebits, mbits, bias
        )
        torch.testing.assert_allclose(
            dequantized_data.cpu(), input_data, rtol=1, atol=0
        )


if __name__ == "__main__":
    unittest.main()
