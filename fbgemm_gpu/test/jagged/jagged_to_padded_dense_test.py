#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest

import hypothesis.strategies as st
import torch
import torch._dynamo
from hypothesis import assume, given, settings, Verbosity

from .common import (
    additional_decorators,
    generate_jagged_tensor,
    open_source,
    to_padded_dense,
)

if open_source:
    # pyre-ignore[21]
    from test_utils import cpu_and_maybe_gpu, gradcheck, optests
else:
    from fbgemm_gpu.test.test_utils import cpu_and_maybe_gpu, gradcheck, optests


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class JaggedToPaddedDenseTest(unittest.TestCase):
    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        fold_inner_dense=st.booleans(),
        padding_value=st.sampled_from([0, -1e-8]),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_to_padded_dense(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        fold_inner_dense: bool,
        padding_value: float,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # CPU doesn't support bfloat16
        assume(device_type != "cpu" or dtype != torch.bfloat16)
        assume(not fold_inner_dense or inner_dense_size == 1)

        # Testing with a basic crafted example.
        # dense representation is
        # [[[[0, 1], [ 0,  0], [0, 0]],
        #   [[2, 3], [ 4,  5], [6, 7]],
        #   [[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]]],
        #  [[[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]]],
        #  [[[8, 9], [10, 11], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]]]],
        # inner_dense_size = 2
        # x_offsets = [
        #     torch.LongTensor([0, 2, 2, 3]),  # lengths torch.Tensor([2, 0, 1]),
        #     torch.LongTensor([0, 1, 4, 6]),  # lengths torch.Tensor([1, 3, 2]),
        # ]
        # outer_dense_size = len(x_offsets[0]) - 1
        # max_lengths = [4, 3]

        device = torch.device(device_type)

        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            torch.float,
            device,
            fold_inner_dense,
        )

        output_ref = to_padded_dense(
            x_values, x_offsets, max_lengths, padding_value=padding_value
        )
        output = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values,
            x_offsets,
            max_lengths,
            padding_value=padding_value,
        )

        torch.testing.assert_close(output, output_ref)

        gradcheck(
            torch.ops.fbgemm.jagged_to_padded_dense,
            (
                x_values.float().requires_grad_(True),
                x_offsets,
                max_lengths,
                padding_value,
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        padding_value=st.just(0),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("meta"),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_to_padded_dense_meta_backend(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        padding_value: float,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        assume(device_type != "cpu" or dtype != torch.bfloat16)
        device = torch.device("cpu")

        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, torch.float, device
        )

        output_ref = to_padded_dense(
            x_values, x_offsets, max_lengths, padding_value=padding_value
        )
        x_values.to(device_type)
        output = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values,
            x_offsets,
            max_lengths,
            padding_value=padding_value,
        )

        assert output.size() == output_ref.size()


if __name__ == "__main__":
    unittest.main()
