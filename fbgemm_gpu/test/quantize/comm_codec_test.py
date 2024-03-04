#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Optional, Tuple

import hypothesis.strategies as st
import torch
from fbgemm_gpu.quantize_comm import none_throws, QuantizedCommCodec
from fbgemm_gpu.split_embedding_configs import SparseType
from hypothesis import assume, given, settings


class QuantizedCommCodecTest(unittest.TestCase):
    @settings(deadline=4000)
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
                (SparseType.INT8, None),
            ]
        ),
        row_size=st.integers(4, 256),
        col_size=st.integers(4, 256),
        rand_seed=st.integers(0, 65534),
        row_dim=st.sampled_from([-1, 4, 8, 16, 32]),
    )
    def test_quantized_comm_codec(
        self,
        comm_precisions_loss_scale: Tuple[SparseType, Optional[float]],
        row_size: int,
        col_size: int,
        rand_seed: int,
        row_dim: int,
    ) -> None:
        (comm_precision, loss_scale) = comm_precisions_loss_scale

        if comm_precision == SparseType.FP8:
            if row_dim > 0:
                assume((col_size * row_size) % row_dim == 0)
            assume(col_size % 4 == 0)

        torch.manual_seed(rand_seed)
        shape = (row_size, col_size)
        input_tensor = torch.rand(shape, requires_grad=True)

        cur_row_dim = None

        if (
            comm_precision == SparseType.FP8
            and torch.cuda.device_count() != 0
            and row_dim > 0
        ):
            cur_row_dim = row_dim
            input_tensor = input_tensor.view(-1).cuda()

        quant_codec = QuantizedCommCodec(
            comm_precision, loss_scale, row_dim=cur_row_dim
        )
        ctx = quant_codec.create_context()

        if comm_precision == SparseType.INT8:
            ctx = none_throws(ctx)
            assume(row_size * col_size % ctx.row_dim == 0)
            input_tensor = input_tensor.view(-1)

        quant_tensor = quant_codec.encode(input_tensor, ctx)

        self.assertEqual(
            quant_tensor.numel(),
            quant_codec.calc_quantized_size(input_tensor.numel(), ctx),
        )

        output_tensor = quant_codec.decode(quant_tensor, ctx)
        self.assertEqual(output_tensor.shape, input_tensor.shape)

        rtol = 0.005
        atol = 0.005
        if comm_precision == SparseType.FP8:
            rtol = 0.05
            atol = 0.05

        torch.testing.assert_close(
            input_tensor.detach().cpu(),
            output_tensor.detach().cpu(),
            rtol=rtol,
            atol=atol,
        )
