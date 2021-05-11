import unittest

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.test.test_utils import (
    fused_rowwise_8bit_quantize_reference,
    fused_rowwise_8bit_dequantize_reference,
)
from hypothesis import HealthCheck, given, settings

torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


class TestFused8BitRowwiseQuantizationConversion(unittest.TestCase):
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
    )
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `[hypothesis.HealthCheck.filter_too_much]` to decorator factory
    #  `hypothesis.settings`.
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(self, nrows: int, ncols: int) -> None:
        input_data = torch.rand(nrows, ncols).float()
        quantized_data = torch.ops.fb.FloatToFused8BitRowwiseQuantized(input_data)

        if nrows == 0 or ncols == 0:
            assert quantized_data.numel() == nrows * ((ncols + 3) // 4 * 4 + 8)
            return

        reference = fused_rowwise_8bit_quantize_reference(input_data.numpy())
        np.testing.assert_array_almost_equal(quantized_data.numpy(), reference)

        if torch.cuda.is_available():
            input_data_gpu = input_data.cuda()
            quantized_data_gpu = torch.ops.fb.FloatToFused8BitRowwiseQuantized(
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

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `[hypothesis.HealthCheck.filter_too_much]` to decorator factory
    #  `hypothesis.settings`.
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op_cuda_large_nrows(self) -> None:
        ncols = 256
        nrows = 65540

        input_data = torch.rand(nrows, ncols).float()
        quantized_data = torch.ops.fb.FloatToFused8BitRowwiseQuantized(input_data)

        reference = torch.from_numpy(
            fused_rowwise_8bit_dequantize_reference(quantized_data.numpy())
        )

        if torch.cuda.is_available():
            input_data_gpu = input_data.cuda()
            quantized_data_gpu = torch.ops.fb.FloatToFused8BitRowwiseQuantized(
                input_data_gpu
            )
            dequantized_data_gpu = torch.ops.fb.Fused8BitRowwiseQuantizedToFloat(
                quantized_data_gpu
            )
            reference = torch.from_numpy(
                fused_rowwise_8bit_dequantize_reference(
                    quantized_data_gpu.cpu().numpy()
                )
            )
            # compare quantized data
            torch.testing.assert_allclose(dequantized_data_gpu.cpu(), reference)
