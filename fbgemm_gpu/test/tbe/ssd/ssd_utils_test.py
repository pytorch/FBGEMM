# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Callable

import fbgemm_gpu.tbe.ssd  # noqa F401
import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .. import common  # noqa E402
from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, running_on_github
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, running_on_github


MAX_EXAMPLES = 20


@unittest.skipIf(*running_on_github)
@unittest.skipIf(*gpu_unavailable)
class SSDUtilsTest(unittest.TestCase):
    def execute_masked_index_test(
        self,
        D: int,
        max_index: int,
        num_indices: int,
        num_value_rows: int,
        num_output_rows: int,
        dtype: torch.dtype,
        test_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        is_index_put: bool,
    ) -> None:
        """
        A helper function that generates inputs/outputs, runs
        torch.ops.fbgemm.masked_index_* against the PyTorch counterpart, and
        compares the output results"""
        device = "cuda"

        # Number of columns must be multiple of 4 (embedding requirement)
        D = D * 4

        # Generate indices
        indices = torch.randint(
            low=0, high=max_index, size=(num_indices,), dtype=torch.long, device=device
        )

        # Compute/set unique indices (indices have to be unique to avoid race
        # condition)
        indices_unique = indices.unique()
        count_val = indices_unique.numel()
        indices[:count_val] = indices_unique

        # Permute unique indices
        rand_pos = torch.randperm(indices_unique.numel(), device=device)
        indices[:count_val] = indices[rand_pos]

        # Set some indices to -1
        indices[rand_pos[: max(count_val // 2, 1)]] = -1

        # Generate count tensor
        count = torch.as_tensor([count_val], dtype=torch.int, device=device)

        # Generate values
        values = torch.rand(num_value_rows, D, dtype=dtype, device=device)

        # Allocate output and output_ref
        output = torch.zeros(num_output_rows, D, dtype=dtype, device=device)
        output_ref = torch.zeros(num_output_rows, D, dtype=dtype, device=device)

        # Run test
        output = test_fn(output, indices, values, count)

        # Run reference
        indices = indices[:count_val]
        filter_ = indices >= 0
        indices_ = indices[filter_]
        filter_locs = filter_.nonzero().flatten()
        if is_index_put:
            output_ref[indices_] = values[filter_locs]
        else:
            output_ref[filter_locs] = values[indices_]

        # Compare results
        assert torch.equal(output_ref, output)

    # pyre-ignore [56]
    @given(
        num_indices=st.integers(min_value=10, max_value=100),
        D=st.integers(min_value=2, max_value=256),
        num_output_rows=st.integers(min_value=10, max_value=100),
        dtype=st.sampled_from([torch.float, torch.half]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_masked_index_put(
        self,
        num_indices: int,
        D: int,
        num_output_rows: int,
        dtype: torch.dtype,
    ) -> None:
        """
        Test correctness of torch.ops.fbgemm.masked_index_put against PyTorch's
        index_put
        """
        self.execute_masked_index_test(
            D=D,
            max_index=num_output_rows,
            num_indices=num_indices,
            num_value_rows=num_indices,
            num_output_rows=num_output_rows,
            dtype=dtype,
            test_fn=torch.ops.fbgemm.masked_index_put,
            is_index_put=True,
        )

    # pyre-ignore [56]
    @given(
        num_indices=st.integers(min_value=10, max_value=100),
        D=st.integers(min_value=2, max_value=256),
        num_value_rows=st.integers(min_value=10, max_value=100),
        dtype=st.sampled_from([torch.float, torch.half]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_masked_index_select(
        self,
        num_indices: int,
        D: int,
        num_value_rows: int,
        dtype: torch.dtype,
    ) -> None:
        """
        Test correctness of torch.ops.fbgemm.masked_index_select aginst
        PyTorch's index_select
        """
        self.execute_masked_index_test(
            D=D,
            max_index=num_value_rows,
            num_indices=num_indices,
            num_value_rows=num_value_rows,
            num_output_rows=num_indices,
            dtype=dtype,
            test_fn=torch.ops.fbgemm.masked_index_select,
            is_index_put=False,
        )
