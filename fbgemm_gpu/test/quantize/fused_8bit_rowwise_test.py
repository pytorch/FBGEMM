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
    fused_rowwise_8bit_dequantize_2bytes_padding_scale_bias_first_reference,
    fused_rowwise_8bit_dequantize_reference,
    fused_rowwise_8bit_quantize_reference,
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

    def quantize_and_dequantize_op_test_helper(  # noqa: C901
        self,
        nrows: int,
        ncols: int,
        output_dtype: SparseType,
        quant_padding_float_type: bool,
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
            if not test_generic_op and not quant_padding_float_type:
                return
            if not quant_padding_float_type and output_dtype == SparseType.FP32:
                return
            if test_generic_op:
                quantized_data_gpu_ref = (
                    torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(
                        input_data_gpu
                    )
                )
                # fbgemm weight 2byte storages are scale_bias first layout
                if quant_padding_float_type is False:
                    scale_bias_last = False
                    quant_pad = quantized_data_gpu_ref[:, -8:]
                    quant_data = quantized_data_gpu_ref[:, :-8]
                    quantized_data_gpu = torch.cat(
                        [
                            quant_pad.view(torch.float)
                            .to(torch.half)
                            .view(torch.uint8),
                            quant_data,
                        ],
                        dim=1,
                    )
                else:
                    scale_bias_last = True
                    quantized_data_gpu = quantized_data_gpu_ref
                dequantized_data_gpu = (
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatOrHalf(
                        quantized_data_gpu,
                        output_dtype.as_int(),
                        quant_padding_float_type=quant_padding_float_type,
                        scale_bias_last=scale_bias_last,
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

            # pyre-fixme[61]: `dequantized_data_gpu` is undefined, or not always
            #  defined.
            dequantized_data_trimmed = dequantized_data_gpu[:, :ncols].cpu()
            quantize_data_numpy = quantized_data_gpu.cpu().numpy()
            if quant_padding_float_type:
                reference = torch.from_numpy(
                    fused_rowwise_8bit_dequantize_reference(quantize_data_numpy)[
                        :, :ncols
                    ]
                )
            else:
                reference = torch.from_numpy(
                    fused_rowwise_8bit_dequantize_2bytes_padding_scale_bias_first_reference(
                        quantize_data_numpy
                    )[
                        :, :ncols
                    ]
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

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.sampled_from([32, 128, 256, 384, 512, 1024]),
        output_dtype=st.sampled_from(
            [SparseType.FP16, SparseType.FP32, SparseType.BF16]
        ),
        quant_padding_float_type=st.sampled_from(
            [True, False],
        ),
        test_generic_op=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op_cpu(  # noqa: C901
        self,
        nrows: int,
        ncols: int,
        output_dtype: SparseType,
        quant_padding_float_type: bool,
        test_generic_op: bool,
    ) -> None:
        self.quantize_and_dequantize_op_test_helper(
            nrows, ncols, output_dtype, quant_padding_float_type, test_generic_op, False
        )

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.sampled_from([32, 128, 256, 384, 512, 1024]),
        output_dtype=st.sampled_from(
            [SparseType.FP16, SparseType.FP32, SparseType.BF16]
        ),
        quant_padding_float_type=st.sampled_from(
            [True, False],
        ),
        test_generic_op=st.booleans(),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op_cuda(  # noqa: C901
        self,
        nrows: int,
        ncols: int,
        output_dtype: SparseType,
        quant_padding_float_type: bool,
        test_generic_op: bool,
    ) -> None:
        self.quantize_and_dequantize_op_test_helper(
            nrows, ncols, output_dtype, quant_padding_float_type, test_generic_op, True
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


if __name__ == "__main__":
    unittest.main()
