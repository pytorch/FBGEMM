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

from .common import additional_decorators, open_source, torch_compiled

if open_source:
    # pyre-ignore[21]
    from test_utils import cpu_and_maybe_gpu, optests, symint_vector_unsupported
else:
    from fbgemm_gpu.test.test_utils import (
        cpu_and_maybe_gpu,
        optests,
        symint_vector_unsupported,
    )


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class DenseBmmTest(unittest.TestCase):
    @given(
        B=st.integers(10, 512),
        M=st.integers(1, 32),
        N=st.integers(1, 32),
        max_L=st.integers(1, 32),
        dtype=st.sampled_from([torch.float]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_jagged_bmm(
        self,
        B: int,
        M: int,
        N: int,
        max_L: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        assume(B != 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = torch.randint(max_L + 1, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        total_length = int(lengths.sum().item())
        x_values = torch.rand(
            (total_length, M), requires_grad=True, dtype=dtype, device=device
        )
        y_values = torch.rand(
            (total_length, N), requires_grad=True, dtype=dtype, device=device
        )
        output = torch.ops.fbgemm.jagged_jagged_bmm(
            x_values,
            y_values,
            offsets,
            max_L,
        )

        x_values_ref = x_values.detach().clone().requires_grad_(True)
        y_values_ref = y_values.detach().clone().requires_grad_(True)
        x_dense_ref = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values_ref,
            [offsets],
            max_lengths=[max_L],
        )
        y_dense_ref = torch.ops.fbgemm.jagged_to_padded_dense(
            y_values_ref,
            [offsets],
            max_lengths=[max_L],
        )
        output_ref = torch.bmm(x_dense_ref.transpose(2, 1), y_dense_ref)

        # verify forward
        torch.testing.assert_close(output, output_ref)

        # verify backward
        grad_output = output.detach().clone().requires_grad_(True)

        output.backward(grad_output)
        output_ref.backward(grad_output)

        torch.testing.assert_close(x_values.grad, x_values_ref.grad)
        torch.testing.assert_close(y_values.grad, y_values_ref.grad)

    @given(
        B=st.integers(10, 512),
        M=st.integers(1, 32),
        N=st.integers(1, 32),
        max_L=st.integers(1, 32),
        dtype=st.sampled_from([torch.float]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_jagged_dense_bmm(
        self,
        B: int,
        M: int,
        N: int,
        max_L: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        assume(B != 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = torch.randint(max_L + 1, size=(B,), device=device)
        total_length = int(lengths.sum().item())
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        x_values = torch.rand(
            (total_length, M), requires_grad=True, dtype=dtype, device=device
        )
        y = torch.rand((B, M, N), requires_grad=True, dtype=dtype, device=device)

        output, _ = torch.ops.fbgemm.jagged_dense_bmm(
            x_values,
            offsets,
            y,
            max_L,
        )

        x_values_ref = x_values.detach().clone().requires_grad_(True)
        x_dense_ref = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values_ref,
            [offsets],
            max_lengths=[max_L],
        )
        y_ref = y.detach().clone().requires_grad_(True)
        output_dense = torch.bmm(x_dense_ref, y_ref)

        output_ref, _ = torch.ops.fbgemm.dense_to_jagged(
            output_dense, [offsets], total_length
        )

        # verify forward
        torch.testing.assert_close(output, output_ref)
        # verify backward
        grad_output = output.detach().clone().requires_grad_(True)

        output.backward(grad_output)
        output_ref.backward(grad_output)

        torch.testing.assert_close(x_values.grad, x_values_ref.grad)
        torch.testing.assert_close(y.grad, y_ref.grad)

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @unittest.skipIf(*symint_vector_unsupported())
    @given(
        B=st.integers(10, 512),
        M=st.integers(2, 32),
        N=st.integers(2, 32),
        max_L=st.integers(2, 32),
        dtype=st.sampled_from([torch.float]),
        device=st.just(torch.device("cpu")),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_dense_bmm_dynamic_shape(
        self,
        B: int,
        M: int,
        N: int,
        max_L: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        assume(B != 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = torch.randint(low=1, high=max_L + 1, size=(B,), device=device)
        total_length = int(lengths.sum().item())
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        x_values = torch.rand(
            (total_length, M), requires_grad=True, dtype=dtype, device=device
        )
        y = torch.rand((B, M, N), requires_grad=True, dtype=dtype, device=device)

        torch._dynamo.mark_dynamic(x_values, 0)
        torch._dynamo.mark_dynamic(x_values, 1)
        torch._dynamo.mark_dynamic(lengths, 0)  # offsets = lengths + 1

        output, _ = torch_compiled(
            torch.ops.fbgemm.jagged_dense_bmm, fullgraph=True, dynamic=True
        )(
            x_values,
            offsets,
            y,
            max_L,
        )

        x_values_ref = x_values.detach().clone().requires_grad_(True)
        x_dense_ref = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values_ref,
            [offsets],
            max_lengths=[max_L],
        )
        y_ref = y.detach().clone().requires_grad_(True)
        output_dense = torch.bmm(x_dense_ref, y_ref)

        output_ref, _ = torch.ops.fbgemm.dense_to_jagged(
            output_dense, [offsets], total_length
        )

        # verify forward
        torch.testing.assert_close(output, output_ref)
        # verify backward
        grad_output = output.detach().clone().requires_grad_(True)

        output.backward(grad_output)
        output_ref.backward(grad_output)

        torch.testing.assert_close(x_values.grad, x_values_ref.grad)
        torch.testing.assert_close(y.grad, y_ref.grad)


if __name__ == "__main__":
    unittest.main()
