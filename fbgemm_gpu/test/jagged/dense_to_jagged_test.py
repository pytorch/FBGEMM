#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest
from typing import List, Tuple

import hypothesis.strategies as st
import torch
import torch._dynamo
from hypothesis import given, settings, Verbosity

from .common import additional_decorators, generate_jagged_tensor, open_source

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
class DenseToJaggedTest(unittest.TestCase):
    def _test_dense_to_jagged(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        precompute_total_L: bool,
    ) -> None:
        # Generate multi-dim jagged tensor
        values_2d, offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)

        # jagged -> dense
        dense = torch.ops.fbgemm.jagged_to_padded_dense(values_2d, offsets, max_lengths)

        # dense -> jagged (op which is being tested)
        if precompute_total_L:
            total_L = values_2d.size(0)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets, total_L
            )
        else:
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets
            )

        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        torch.testing.assert_close(dense, dense2)

        # verify backward
        dense.retain_grad()
        ref_output_values = jagged_values.clone().detach().requires_grad_(True)
        ref_values = dense.clone().detach().requires_grad_(True)
        jagged_values.backward(ref_output_values)
        torch.testing.assert_close(dense.grad, ref_values)

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            precompute_total_L,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.integers(0, 6000),
        inner_dense_size=st.sampled_from([8, 16, 23, 24, 48, 50, 64, 72, 96, 192]),
        dtype=st.just(torch.half),
        device=cpu_and_maybe_gpu(),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_opt(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            precompute_total_L,
        )

    # (8000+1) * 8 (size of the element of LongTensor/int64_t offsets)
    # = ~62.5KB > 48KB default shared memory on V100/A100.
    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.just(8000),
        inner_dense_size=st.just(16),
        dtype=st.just(torch.half),
        device=cpu_and_maybe_gpu(),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_dense_to_jagged_opt_large_batch(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            precompute_total_L,
        )

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=st.sampled_from([torch.device("meta")]),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_meta_backend(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        precompute_total_L: bool,
    ) -> None:
        device = torch.device("cpu")
        values_2d, offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)

        # jagged -> dense
        dense = torch.ops.fbgemm.jagged_to_padded_dense(values_2d, offsets, max_lengths)

        # dense -> jagged (op which is being tested)
        if precompute_total_L:
            total_L = values_2d.size(0)
            dense.to(device)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets, total_L
            )
        else:
            dense.to(device)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets
            )

        jagged_values.to(device)
        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        assert dense.size() == dense2.size()

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @unittest.skipIf(*symint_vector_unsupported())
    @given(
        num_jagged_dim=st.integers(1, 5),
        # TODO: size = 0/1 will be incorrectly specialized
        outer_dense_size=st.integers(2, 5),
        inner_dense_size=st.integers(2, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_dynamic_shape(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        values_2d, offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            mark_dynamic=True,
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)

        def jagged_to_dense(
            values: torch.Tensor,
            offsets: List[torch.LongTensor],
            max_lengths: List[int],
        ) -> torch.Tensor:
            return torch.ops.fbgemm.jagged_to_padded_dense(values, offsets, max_lengths)

        # jagged -> dense
        dense = jagged_to_dense(values_2d, offsets, max_lengths.tolist())

        # dense -> jagged, it is required to pre-compute totalL
        total_L = values_2d.size(0)
        dense = dense.clone().detach().to(device)

        torch._dynamo.mark_dynamic(dense, 0)
        torch._dynamo.mark_dynamic(dense, -1)

        def dense_to_jagged_withL(
            dense: torch.Tensor, offsets: List[torch.LongTensor], total_L: List[int]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.fbgemm.dense_to_jagged(dense, offsets, total_L)

        def dense_to_jagged_noL(
            dense: torch.Tensor, offsets: List[torch.LongTensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.fbgemm.dense_to_jagged(dense, offsets)

        jagged_values, jagged_offsets = dense_to_jagged_noL(dense, offsets)
        jagged_values, jagged_offsets = dense_to_jagged_withL(dense, offsets, total_L)

        jagged_values.to(device)
        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        assert dense.size() == dense2.size()


if __name__ == "__main__":
    unittest.main()
