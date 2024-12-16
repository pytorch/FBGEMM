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


if __name__ == "__main__":
    unittest.main()
