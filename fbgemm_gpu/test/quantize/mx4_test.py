# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import hypothesis.strategies as st

import numpy as np
import torch
from deeplearning.fbgemm.fbgemm_gpu.test.quantize.mx.common import check_diff_quantize

from hypothesis import given, settings, Verbosity

from . import common  # noqa E402
from .common import open_source
from .mx.common import (
    _get_format_params,
    _quantize_elemwise_core,
    _reshape_to_blocks,
    _shared_exponents,
    _undo_reshape_to_blocks,
    ElemFormat,
    RoundingMode,
)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


FP32_EXPONENT_BIAS: int = 127
FP32_MIN_NORMAL: float = 2 ** (-FP32_EXPONENT_BIAS + 1)


def _quantize_mx(
    A: torch.Tensor,
    scale_bits: int,
    elem_format: ElemFormat | str | None,  # can be None for no quantization
    shared_exp_method: str = "max",
    axes: List[int] = [-1],  # noqa
    group_size: int = 0,
    round: str = "nearest",
    flush_fp32_subnorms: bool = False,
    custom_cuda: bool = False,
) -> torch.Tensor:
    """Function used for MX* quantization"""
    # Shortcut for no quantization
    if elem_format is None:
        return A

    assert scale_bits > 0

    # Custom CUDA only supports limited rounding modes
    custom_cuda = custom_cuda and round in RoundingMode.string_enums()

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    ####################
    # C++ Quantize
    ####################

    if custom_cuda:
        axis = axes[0]
        if group_size == 0:
            group_size = A.shape[axis]

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
        custom_cuda=custom_cuda,
    )

    A = A * scale

    # Undo tile reshaping
    if group_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A


# @optests.generate_opcheck_tests()
class TestMXQuantizationConversion(unittest.TestCase):

    def all_encodings(
        self,
        _e: int,
        _m: int,
        device: torch.device,
        encodes_infs: bool = True,
    ) -> torch.Tensor:
        _CACHE = {}
        if (_e, _m, encodes_infs) in _CACHE:
            x = _CACHE[(_e, _m, encodes_infs)]
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        # Holds all positive and negative
        x = np.zeros((2 ** (_e + _m + 1)), dtype=np.float32)
        for _i in range(2 ** (_e + _m)):
            if _e > 0:
                _exp = _i >> _m
                # Skip exp == all ones
                if encodes_infs and _exp == 2**_e - 1:
                    continue
                # Normal or subnormal encoding
                if _exp == 0:
                    _exp = 1 - (2 ** (_e - 1) - 1)
                    _explicit = 0.0
                else:
                    _exp -= 2 ** (_e - 1) - 1
                    _explicit = 1.0
                # Obtain mantissa value
                _mant = _i & ((2**_m) - 1)
                _mmant = _mant / (2**_m)

                # FP8 e4m3 hack
                if _e == 4 and _m == 3 and _exp == 8 and _mmant == 0.875:
                    _value = 0
                else:
                    _value = 2 ** (_exp) * (_explicit + _mmant)
            else:
                _value = _i / (2 ** (_m - 1))

            x[_i] = _value
            x[_i + 2 ** (_e + _m)] = -_value

        _CACHE[(_e, _m, encodes_infs)] = x

        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _test_mx4(
        self, x: torch.Tensor, group_size: int
    ) -> bool:
        assert x.numel() % group_size == 0
        axes = [-1]
        element_format_str = "fp4_e2m1"
        scale_bits = 8

        ### PYTHON

        y1 = _quantize_mx(
            x,
            scale_bits,
            elem_format=element_format_str,
            axes=axes,
            group_size=group_size,
        )

        ### CUDA
        y2 = _quantize_mx(
            x,
            scale_bits,
            elem_format=element_format_str,
            axes=axes,
            custom_cuda=True,
            group_size=group_size,
        )

        return check_diff_quantize(x, y1, y2)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]:
    @given(
        power=st.integers(min_value=5, max_value=7),
        sizes=st.integers(min_value=4, max_value=12),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_mx4(self, power: int, sizes: int) -> None:
        group_size = 2**power
        device = torch.device("cuda")
        res = []

        x = self.all_encodings(8, sizes, device=device)

        res.append(self._test_mx4(x, group_size))
        if False in res:
            raise ValueError(f"Test failed at {res.index(False)}")


if __name__ == "__main__":
    unittest.main()
