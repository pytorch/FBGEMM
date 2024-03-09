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
import numpy as np
import torch
import torch._dynamo
from hypothesis import given, settings, Verbosity

from .common import additional_decorators, open_source, torch_compiled, var_list_to_coo

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        cpu_and_maybe_gpu,
        gpu_unavailable,
        optests,
        symint_vector_unsupported,
    )
else:
    from fbgemm_gpu.test.test_utils import (
        cpu_and_maybe_gpu,
        gpu_unavailable,
        optests,
        symint_vector_unsupported,
    )


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class Jagged2DToDenseTest(unittest.TestCase):
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(min_value=1, max_value=128),
        D=st.integers(min_value=1, max_value=128),
        max_sequence_length=st.integers(min_value=1, max_value=200),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
    )
    def test_jagged_2d_to_dense(
        self,
        B: int,
        D: int,
        max_sequence_length: int,
        dtype: torch.dtype,
    ) -> None:
        D = D * 4
        lengths_ = np.random.randint(low=0, high=max_sequence_length, size=B)
        total_lengths = lengths_.sum()
        lengths = torch.from_numpy(lengths_)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

        ref_values = torch.rand(total_lengths, D)
        ref_output_values = var_list_to_coo(
            lengths,
            ref_values,
            max_sequence_length,
            D,
        ).to_dense()
        ref_output_values = ref_output_values.to(dtype)

        # test cpu forward
        values = ref_values.clone().to(dtype).detach().requires_grad_(True)
        output_values = torch.ops.fbgemm.jagged_2d_to_dense(
            values=values,
            offsets=offsets,
            max_sequence_length=max_sequence_length,
        )
        torch.testing.assert_close(ref_output_values, output_values)

        if torch.cuda.is_available():
            # test gpu forward
            ref_values = ref_values.cuda()
            values = ref_values.clone().to(dtype).detach().requires_grad_(True)
            offsets = offsets.cuda()
            ref_output_values = ref_output_values.cuda()
            output_values = torch.ops.fbgemm.jagged_2d_to_dense(
                values=values,
                offsets=offsets,
                max_sequence_length=max_sequence_length,
            )
            torch.testing.assert_close(ref_output_values, output_values)

            # test gpu backward
            output_values.backward(ref_output_values)
            ref_values = ref_values.to(dtype)
            torch.testing.assert_close(ref_values, values.grad)

    def test_jagged_2d_to_dense_truncation(self) -> None:
        # Test the case where max_sequence_length < max(lengths[i])
        lengths_ = np.array([2, 3, 0, 1])
        lengths = torch.from_numpy(lengths_)
        total_lengths = lengths_.sum()
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

        embedding_dim = 16
        max_sequence_length = 2
        ref_values = torch.rand(total_lengths, embedding_dim)
        ref_output_values = var_list_to_coo(
            lengths,
            ref_values,
            3,
            embedding_dim,
        ).to_dense()[:, :max_sequence_length, :]

        # test cpu forward
        values = ref_values.clone().detach().requires_grad_(True)
        output_values = torch.ops.fbgemm.jagged_2d_to_dense(
            values=values,
            offsets=offsets,
            max_sequence_length=max_sequence_length,
        )
        torch.testing.assert_close(ref_output_values, output_values)

        if torch.cuda.is_available():
            # test gpu forward
            ref_values = ref_values.cuda()
            values = ref_values.clone().detach().requires_grad_(True)
            offsets = offsets.cuda()
            ref_output_values = ref_output_values.cuda()
            output_values = torch.ops.fbgemm.jagged_2d_to_dense(
                values=values,
                offsets=offsets,
                max_sequence_length=max_sequence_length,
            )
            torch.testing.assert_close(ref_output_values, output_values)

            # test gpu backward
            expected_grad = ref_values
            expected_grad[4, :] = 0  # due to truncation
            expected_grad = expected_grad.cuda()
            output_values.backward(ref_output_values)
            torch.testing.assert_close(expected_grad, values.grad)

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @unittest.skipIf(*symint_vector_unsupported())
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(min_value=2, max_value=128),
        D=st.integers(min_value=2, max_value=128),
        max_sequence_length=st.integers(min_value=1, max_value=200),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
    )
    def test_jagged_2d_to_dense_dynamic_shape(
        self,
        B: int,
        D: int,
        max_sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        D = D * 4
        lengths_ = np.random.randint(low=0, high=max_sequence_length, size=B)
        total_lengths = lengths_.sum()
        lengths = torch.from_numpy(lengths_)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

        ref_values = torch.rand(total_lengths, D)
        ref_output_values = var_list_to_coo(
            lengths,
            ref_values,
            max_sequence_length,
            D,
        ).to_dense()
        ref_output_values = ref_output_values.to(dtype)

        ref_values = ref_values.to(device)
        values = ref_values.clone().to(dtype).detach().requires_grad_(True)
        offsets = offsets.to(device)
        ref_output_values = ref_output_values.to(device)
        output_values = torch_compiled(
            torch.ops.fbgemm.jagged_2d_to_dense, dynamic=True, fullgraph=True
        )(
            values=values,
            offsets=offsets,
            max_sequence_length=max_sequence_length,
        )
        torch.testing.assert_close(ref_output_values, output_values)

        output_values.backward(ref_output_values)
        ref_values = ref_values.to(dtype)
        torch.testing.assert_close(ref_values, values.grad)

    @unittest.skipIf(*gpu_unavailable)
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    @given(
        T=st.integers(min_value=1, max_value=5),
        B=st.integers(min_value=1, max_value=64),
        D=st.integers(min_value=1, max_value=128),
        max_sequence_length=st.integers(min_value=1, max_value=300),
        device=cpu_and_maybe_gpu(),
    )
    def test_stacked_jagged_2d_to_dense(
        self,
        T: int,
        B: int,
        D: int,
        max_sequence_length: int,
        device: torch.device,
    ) -> None:
        D = D * 4
        lengths_ = np.random.randint(low=0, high=max_sequence_length, size=B * T)
        total_lengths = lengths_.sum()
        lengths = torch.from_numpy(lengths_).to(device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        ref_values = torch.rand(total_lengths, D, device=device)
        ref_output_values = var_list_to_coo(
            lengths,
            ref_values,
            max_sequence_length,
            D,
        ).to_dense()
        lengths = lengths.view(T, B)

        values = ref_values.clone().detach().requires_grad_(True)
        output_values_per_table = torch.ops.fbgemm.stacked_jagged_2d_to_dense(
            values=values,
            lengths=lengths,
            offset_per_key=[0]
            + np.cumsum([lengths[t].sum().item() for t in range(T)]).tolist(),
            max_lengths_per_key=[max_sequence_length] * T,
        )
        ref_output_values = torch.ops.fbgemm.jagged_2d_to_dense(
            values=ref_values,
            offsets=offsets,
            max_sequence_length=max_sequence_length,
        )
        torch.testing.assert_close(
            ref_output_values, torch.cat(output_values_per_table)
        )

        # test backward
        output_values = torch.cat(output_values_per_table)
        output_values.backward(ref_output_values)
        torch.testing.assert_close(ref_values, values.grad)


if __name__ == "__main__":
    unittest.main()
