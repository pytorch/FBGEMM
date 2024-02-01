# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from ctypes import c_float, c_int32, cast, POINTER, pointer
from typing import Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, HealthCheck, settings

from . import common  # noqa E402


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


if __name__ == "__main__":
    unittest.main()
