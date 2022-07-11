#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional

import hypothesis.strategies as st
import torch
from fbgemm_gpu.quantize_comm import QuantizeCommCodec
from fbgemm_gpu.split_embedding_configs import SparseType
from hypothesis import assume, given, settings


class QuantizeCommCodecTest(unittest.TestCase):
    @settings(deadline=2000)
    # pyre-ignore
    @given(
        fwd_comm_precision=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.FP8]
        ),
        bwd_comm_precision=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16, SparseType.FP8]
        ),
        loss_scale=st.sampled_from([None, 4.0]),
        row_size=st.integers(4, 256),
        col_size=st.integers(4, 256),
        rand_seed=st.integers(0, 65534),
    )
    def test_quantize_comm_codec(
        self,
        fwd_comm_precision: SparseType,
        bwd_comm_precision: SparseType,
        loss_scale: Optional[float],
        row_size: int,
        col_size: int,
        rand_seed: int,
    ) -> None:
        assume(fwd_comm_precision.bit_rate() == bwd_comm_precision.bit_rate())
        if fwd_comm_precision == SparseType.FP8 or bwd_comm_precision == SparseType.FP8:
            assume(col_size % 4 == 0)
        rtol = 0.005
        atol = 0.003
        if fwd_comm_precision == SparseType.FP8:
            rtol = 0.1
            atol = 0.1
        torch.manual_seed(rand_seed)

        shape = (row_size, col_size)

        quant_codec = QuantizeCommCodec(
            fwd_comm_precision,
            bwd_comm_precision,
            loss_scale=loss_scale,
        )

        input_tensor = torch.rand(shape, requires_grad=True)
        # pyre-ignore
        quant_tensor = quant_codec.encoder.apply(input_tensor)
        output_tensor = quant_codec.decoder.apply(quant_tensor)

        torch.testing.assert_close(
            input_tensor.detach(), output_tensor.detach(), atol=atol, rtol=rtol
        )

        expected_grad = torch.rand(shape)
        output_tensor.backward(expected_grad)
        actual_grad = input_tensor.grad

        self.assertIsNotNone(actual_grad)
        torch.testing.assert_close(
            expected_grad.detach(), actual_grad.detach(), atol=atol, rtol=rtol
        )
