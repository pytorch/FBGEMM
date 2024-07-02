# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import hypothesis.strategies as st

import torch
from fbgemm_gpu.quantize_utils import fp32_to_mx4, mx4_to_fp32
from fbgemm_gpu.triton.quantize_ref import py_dequantize_mx4, py_quantize_mx4

from hypothesis import given, settings, Verbosity

from . import common  # noqa E402
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

    ####################
    # Python Quantize
    ####################

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

        output = fake_quantize_mx_cuda(
            input,
            scale_bits,
            ebits,
            mbits,
            emax,
            max_norm,
            group_size=group_size,
        )

        # Test intercompatibility between implementations.
        py_mx_q_input = py_quantize_mx4(input, group_size)
        py_mx_output = py_dequantize_mx4(py_mx_q_input, group_size)
        triton_mx_q_input = fp32_to_mx4(input, group_size, use_triton=True)
        cuda_mx_output = mx4_to_fp32(triton_mx_q_input, group_size, use_triton=False)
        triton_mx_output = mx4_to_fp32(triton_mx_q_input, group_size, use_triton=True)

        check_diff_quantize(input, py_mx_output, output_ref)
        check_diff_quantize(input, cuda_mx_output, output_ref)
        check_diff_quantize(input, triton_mx_output, output_ref)
        check_diff_quantize(input, output, output_ref)


if __name__ == "__main__":
    unittest.main()
