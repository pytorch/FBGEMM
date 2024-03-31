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

from .common import (
    additional_decorators,
    generate_jagged_tensor,
    open_source,
    to_padded_dense,
    torch_compiled,
)

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        cpu_and_maybe_gpu,
        gpu_unavailable,
        gradcheck,
        optests,
        symint_vector_unsupported,
    )
else:
    from fbgemm_gpu.test.test_utils import (
        cpu_and_maybe_gpu,
        gpu_unavailable,
        gradcheck,
        optests,
        symint_vector_unsupported,
    )


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class DenseDenseElementwiseAddTest(unittest.TestCase):
    def _test_jagged_dense_dense_elementwise_add_jagged_output(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)
        # create a jagged tensor and then densify
        y_0 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        y_1 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        output_ref = x_padded + y_0 + y_1
        (
            output,
            output_offsets,
        ) = torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
            x_values, x_offsets, y_0, y_1
        )
        output = to_padded_dense(output, output_offsets, max_lengths)

        torch.testing.assert_close(output, output_ref)

        # pyre-fixme[2]: Parameter must be annotated.
        def add_jagged_output_func(*args) -> torch.Tensor:
            return torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
                *args
            )[0]

        f = add_jagged_output_func

        gradcheck(
            f,
            (
                x_values.float().requires_grad_(True),
                x_offsets,
                y_0.float().requires_grad_(True),
                y_1.float().requires_grad_(True),
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(0, 4),
        inner_dense_size=st.integers(0, 4),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._test_jagged_dense_dense_elementwise_add_jagged_output(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.integers(0, 8),
        inner_dense_size=st.sampled_from([16, 64, 96, 192]),
        dtype=st.just(torch.half),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output_opt(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._test_jagged_dense_dense_elementwise_add_jagged_output(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(0, 4),
        inner_dense_size=st.integers(0, 4),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("meta"),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output_meta_backend(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        device = torch.device("cpu")

        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)
        # create a jagged tensor and then densify
        y_0 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        y_1 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        output_ref = x_padded + y_0 + y_1
        x_values.to(device_type)
        (
            output,
            output_offsets,
        ) = torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
            x_values, x_offsets, y_0, y_1
        )
        output.to("cpu")
        output = to_padded_dense(output, output_offsets, max_lengths)

        assert output.size() == output_ref.size()

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @unittest.skipIf(*symint_vector_unsupported())
    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(2, 4),
        inner_dense_size=st.integers(2, 4),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("cpu"),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output_dynamic_shape(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            torch.device(device_type),
            mark_dynamic=True,
        )

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)
        # create a jagged tensor and then densify
        y_0 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=torch.device(device_type),
            ),
            x_offsets,
            max_lengths,
        )
        y_1 = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=torch.device(device_type),
            ),
            x_offsets,
            max_lengths,
        )
        output_ref = x_padded + y_0 + y_1
        x_values.to(device_type)
        (output, output_offsets) = torch_compiled(
            torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output,
            fullgraph=True,
            dynamic=True,
        )(x_values, x_offsets, y_0, y_1)
        output.to("cpu")
        output = to_padded_dense(output, output_offsets, max_lengths)

        assert output.size() == output_ref.size()


if __name__ == "__main__":
    unittest.main()
