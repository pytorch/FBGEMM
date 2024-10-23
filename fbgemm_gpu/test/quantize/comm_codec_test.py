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
    @settings(deadline=8000)
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
                (SparseType.MX4, None),
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

        dim_sum_per_rank, rank = [col_size], 0
        if ctx is not None:
            padded_dim_sum, padding_size = quant_codec.padded_size(
                input_tensor, dim_sum_per_rank, rank, ctx
            )
        else:
            padded_dim_sum, padding_size = shape[1], 0
        quant_tensor = quant_codec.encode(input_tensor, ctx)

        padded_numel = (
            padded_dim_sum
            if input_tensor.ndim == 1
            else padded_dim_sum * input_tensor.shape[0]
        )
        self.assertEqual(
            quant_tensor.numel(),
            quant_codec.calc_quantized_size(padded_numel, ctx),
        )

        output_tensor = quant_codec.decode(quant_tensor, ctx)

        # MX4 may flatten tensors if they are too small. Thats ok.
        if comm_precision == SparseType.MX4:
            output_tensor = output_tensor.view(input_tensor.shape[0], -1)
        # padding is done on dimension 1
        if padding_size != 0:
            output_tensor = output_tensor[:, :-padding_size]
        self.assertEqual(output_tensor.shape, input_tensor.shape)

        rtol = 0.005
        atol = 0.005
        # Lower precision datatypes will have some error.
        if comm_precision == SparseType.FP8:
            rtol = 0.05
            atol = 0.05
        elif comm_precision == SparseType.MX4:
            rtol = 0.3
            atol = 0.3

        torch.testing.assert_close(
            input_tensor.detach().cpu(),
            output_tensor.detach().cpu(),
            rtol=rtol,
            atol=atol,
        )
