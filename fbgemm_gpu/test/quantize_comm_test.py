#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional, Tuple

import hypothesis.strategies as st
import torch
from fbgemm_gpu.quantize_comm import QuantizedCommCodec
from fbgemm_gpu.split_embedding_configs import SparseType
from hypothesis import assume, given, settings


class QuantizedCommCodecTest(unittest.TestCase):
    @settings(deadline=2000)
    # pyre-ignore
    @given(
        comm_precisions_loss_scale=st.sampled_from(
            [
                (SparseType.FP32, None),
                (SparseType.FP16, None),
                (SparseType.FP16, 4.0),
                (SparseType.BF16, None),
                (SparseType.BF16, 2.0),
                (SparseType.FP8, None),
                (SparseType.FP8, 3.0),
            ]
        ),
        row_size=st.integers(4, 256),
        col_size=st.integers(4, 256),
        rand_seed=st.integers(0, 65534),
    )
    def test_quantized_comm_codec(
        self,
        comm_precisions_loss_scale: Tuple[SparseType, Optional[float]],
        row_size: int,
        col_size: int,
        rand_seed: int,
    ) -> None:

        (comm_precision, loss_scale) = comm_precisions_loss_scale
        if comm_precision == SparseType.FP8:
            assume(col_size % 4 == 0)

        torch.manual_seed(rand_seed)
        shape = (row_size, col_size)

        quant_codec = QuantizedCommCodec(comm_precision, loss_scale)

        input_tensor = torch.rand(shape, requires_grad=True)

        quant_tensor = quant_codec.encode(input_tensor)
        output_tensor = quant_codec.decode(quant_tensor)

        rtol = 0.005
        atol = 0.005
        if comm_precision == SparseType.FP8:
            rtol = 0.05
            atol = 0.05

        torch.testing.assert_close(
            input_tensor.detach(), output_tensor.detach(), rtol=rtol, atol=atol
        )
