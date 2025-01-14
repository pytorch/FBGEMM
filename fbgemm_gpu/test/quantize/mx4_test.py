# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List, Tuple

import fbgemm_gpu.quantize.quantize_ops  # noqa F401
import hypothesis.strategies as st

import torch

from fbgemm_gpu.quantize_utils import fp32_to_mx4, mx4_to_fp32, RoundingMode
from fbgemm_gpu.triton.quantize_ref import py_dequantize_mx4, py_quantize_mx4

from hypothesis import given, settings, Verbosity

from . import common  # noqa E402

# pyre-fixme[21]: Could not find name `open_source` in
#  `deeplearning.fbgemm.fbgemm_gpu.test.quantize.common`.
from .common import open_source
from .mx.common import (
    _get_format_params,
    _quantize_elemwise_core,
    _reshape_to_blocks,
    _shared_exponents,
    _undo_reshape_to_blocks,
    all_encodings,
    check_diff_quantize,
)

# pyre-fixme[16]: Module `common` has no attribute `open_source`.
if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


FP32_EXPONENT_BIAS: int = 127
FP32_MIN_NORMAL: float = 2 ** (-FP32_EXPONENT_BIAS + 1)


def fake_quantize_mx_cuda(
    A: torch.Tensor,
    scale_bits: int = 8,
    ebits: int = 2,
    mbits: int = 3,
    emax: int = 2,
    max_norm: float = 6.0,
    group_size: int = 32,
) -> torch.Tensor:
    """Call MX* quantization CUDA implementation"""

    mx_quantized = torch.ops.fbgemm.quantize_mx_cuda(
        A,
        scale_bits,
        ebits,
        mbits,
        max_norm,
        group_size,
    )

    return torch.ops.fbgemm.dequantize_mx_cuda(
        mx_quantized,
        group_size,
    )


def fake_quantize_mx(
    A: torch.Tensor,
    scale_bits: int,
    ebits: int = 2,
    mbits: int = 3,
    emax: int = 2,
    max_norm: float = 6.0,
    group_size: int = 32,
    shared_exp_method: str = "max",
    axes: List[int] = [-1],  # noqa
    round: str = "nearest",
    flush_fp32_subnorms: bool = False,
) -> torch.Tensor:
    """Function used for MX* fake quantization"""

    # Make sure axes is a list of non-negative numbers
    axes = [x + A.ndim if x < 0 else x for x in axes]

    # Perform tiling to the hardware vector size
    if group_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(A, axes, group_size)

    shared_exp_axes = [x + 1 for x in axes] if group_size > 0 else axes

    # Get shared exponents
    shared_exp = _shared_exponents(
        A,
        method=shared_exp_method,
        axes=shared_exp_axes,
        ebits=0,
        rounding_mode="floor",
    )

    # Flush subnormal FP32 inputs to zero
    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    # Offset the max exponent by the largest representable exponent
    # in the element data format
    shared_exp = shared_exp - emax

    scale_emax = 2 ** (scale_bits - 1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    _shared_exp: torch.Tensor = shared_exp.cpu()
    scale = (2**_shared_exp).to(A.device)

    A = A / (scale)

    A = _quantize_elemwise_core(
        A,
        mbits,
        ebits,
        max_norm,
        round=round,
        allow_denorm=True,
        saturate_normals=True,
        custom_cuda=False,
    )

    A = A * scale

    # Undo tile reshaping
    if group_size:
        # pyre-fixme[61]: `padded_shape` is undefined, or not always defined.
        # pyre-fixme[61]: `orig_shape` is undefined, or not always defined.
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A


# @optests.generate_opcheck_tests()
class TestMXQuantizationConversion(unittest.TestCase):

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]:
    @given(
        power=st.integers(min_value=5, max_value=8),
        sizes=st.integers(min_value=4, max_value=12),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_mx4(self, power: int, sizes: int) -> None:
        group_size = 2**power
        device = torch.device("cuda")

        input = all_encodings(8, sizes, device=device)
        assert input.numel() % group_size == 0

        axes = [-1]
        element_format_str = "fp4_e2m1"
        ebits, mbits, emax, max_norm, _ = _get_format_params(element_format_str)
        scale_bits = 8

        # Reference from mx_github
        output_ref = fake_quantize_mx(
            input,
            scale_bits,
            ebits,
            mbits,
            emax,
            max_norm,
            axes=axes,
            group_size=group_size,
        )

        # Test CUDA implementation
        output_cuda = fake_quantize_mx_cuda(
            input,
            scale_bits,
            ebits,
            mbits,
            emax,
            max_norm,
            group_size=group_size,
        )

        # Test intercompatibility between implementations.
        # Test CPU implementation
        quantized_cpu = py_quantize_mx4(
            input, group_size, rounding_mode=RoundingMode.floor
        )
        output_cpu = py_dequantize_mx4(quantized_cpu, group_size)

        # Test Triton implementation
        quantized_triton = fp32_to_mx4(
            input, group_size, rounding_mode=RoundingMode.floor, use_triton=True
        )
        output_triton = mx4_to_fp32(quantized_triton, group_size, use_triton=True)

        # Test shim functions
        output_cuda_from_quantized_triton = mx4_to_fp32(
            quantized_triton, group_size, use_triton=False
        )

        # Test torch.ops
        quantized_from_ops = torch.ops.fbgemm.quantize_mx(
            input,
            scale_bits,
            ebits,
            mbits,
            max_norm,
            mx_group_size=group_size,
            rounding_mode=RoundingMode.floor,
        )
        output_from_ops = torch.ops.fbgemm.dequantize_mx(
            quantized_from_ops,
            mx_group_size=group_size,
        )

        assert check_diff_quantize(input, output_ref, output_cuda)
        assert check_diff_quantize(input, output_cuda, output_triton)
        assert check_diff_quantize(
            input, output_cuda, output_cuda_from_quantized_triton
        )
        assert check_diff_quantize(
            input, output_cuda_from_quantized_triton, output_triton
        )
        assert check_diff_quantize(input, output_triton, output_cpu)
        assert check_diff_quantize(input, output_cuda, output_from_ops)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]:
    @given(
        shape=st.sampled_from(
            [
                [32],  # Small shape with group_size = num_elements.
                [2, 16],  # Multi dimensional shape that is padded.
                [2, 2, 4, 32],  # Even more multi dimensional shape without padding.
                [96],  # Shape that cannot be made into even rows.
                [16, 1028],  # Large shape with multiple padded rows.
                [4, 30],  # Multiple small rows with padding.
            ]
        ),
        group_size=st.sampled_from([32, 64]),
        rounding_mode=st.sampled_from(list(RoundingMode)),
        magnitude=st.sampled_from([1.0, 1e3, 1e-3]),
        mx4_format=st.sampled_from([(2, 1), (3, 0)]),
        device=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_mx4_cases(
        self,
        shape: List[int],
        group_size: int,
        rounding_mode: RoundingMode,
        magnitude: int,
        mx4_format: Tuple[int, int],
        device: str,
    ) -> None:
        """Test correctness of mx4 routines with random inputs and unusual shapes."""
        # We only want to consider total sizes that are divisible by group_size.
        ebits, mbits = mx4_format

        # Generate a random input with the specified magnitude.
        input = torch.randn(shape, device=device, dtype=torch.float32) * magnitude

        # Perform quant then dequant to check that proper shape is maintained and
        # outputs are reasonably correct.
        mx_quantized = fp32_to_mx4(
            input, group_size, rounding_mode=rounding_mode, ebits=ebits, mbits=mbits
        )
        mx_dequantized = mx4_to_fp32(mx_quantized, group_size, ebits=ebits, mbits=mbits)

        # If the rows of input are not divisible by group_size, we expect the output
        # to be padded.
        if input.shape[-1] % group_size != 0:
            pad = group_size - (input.shape[-1] % group_size)
            input = torch.nn.functional.pad(input, (0, pad))

        # Check that output shape matches input shape.
        assert mx_dequantized.shape == input.shape

        # Check that values are reasonably close, based on expected variance.
        # I give quite a bit of wiggle room to make sure this isnt flaky.
        torch.testing.assert_close(input, mx_dequantized, rtol=1.0, atol=magnitude / 2)

    # pyre-fixme[56]:
    @unittest.skipIf(
        not (
            torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / (1024**3) >= 32
        ),
        "Test requires a gpu with at least 32GB of memory.",
    )
    def test_mx4_index_overflow(self) -> None:
        """Tests that mx4 quantization kernels can handle inputs that would overflow int32 indices."""
        large_input = torch.zeros(2**32, dtype=torch.float32).to("cuda")
        mx_quantized = fp32_to_mx4(large_input, 32)
        mx_dequantized = mx4_to_fp32(mx_quantized, 32)
        # We just need to check that everything ran without an illegal memory access.
        assert mx_dequantized[0] == 0

    @unittest.skipIf(
        not (
            torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / (1024**3) >= 64
        ),
        "Test requires a gpu with at least 64GB of memory.",
    )
    # pyre-fixme[56]:
    @given(
        shape=st.sampled_from([[1024 * 1024, 2020]]),
        group_size=st.sampled_from([32]),
        rounding_mode=st.sampled_from([RoundingMode.even]),
        magnitude=st.sampled_from([1e6]),
        mx4_format=st.sampled_from([(2, 1)]),
        device=st.sampled_from(["cuda"]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_mx4_large_cases(
        self,
        shape: List[int],
        group_size: int,
        rounding_mode: RoundingMode,
        magnitude: int,
        mx4_format: Tuple[int, int],
        device: str,
    ) -> None:
        """Test correctness of mx4 routines with random inputs and shapes that overflow int32."""
        # We only want to consider total sizes that are divisible by group_size.
        ebits, mbits = mx4_format

        # Generate a random input with the specified magnitude.
        input = torch.randn(shape, device=device, dtype=torch.float32) * magnitude

        # Perform quant then dequant to check that proper shape is maintained and
        # outputs are reasonably correct.
        mx_quantized = fp32_to_mx4(
            input, group_size, rounding_mode=rounding_mode, ebits=ebits, mbits=mbits
        )
        mx_dequantized = mx4_to_fp32(mx_quantized, group_size, ebits=ebits, mbits=mbits)

        # If the rows of input are not divisible by group_size, we expect the output
        # to be padded.
        if input.shape[-1] % group_size != 0:
            pad = group_size - (input.shape[-1] % group_size)
            input = torch.nn.functional.pad(input, (0, pad))

        # Check that output shape matches input shape.
        assert mx_dequantized.shape == input.shape

        # Check that values are reasonably close, based on expected variance.
        # I give quite a bit of wiggle room to make sure this isnt flaky.
        torch.testing.assert_close(input, mx_dequantized, rtol=1.0, atol=magnitude / 2)


if __name__ == "__main__":
    unittest.main()
