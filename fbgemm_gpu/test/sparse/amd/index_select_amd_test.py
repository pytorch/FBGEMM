#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import fbgemm_gpu.sparse_ops  # noqa: F401
import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None,
    "ROCm GPU required",
)
class IndexSelectAmdTest(unittest.TestCase):
    def test_group_index_select_backward_col_tile_boundary(self) -> None:
        """Exercise the ROCm contiguous-warp cache at a column-tile boundary."""
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        dtype = torch.float
        num_cols = 64
        num_embedding_rows = 10
        num_indices = 100003

        # Citrine C3: create the regression inputs directly on the AMD device.
        indices = torch.zeros(num_indices, device=device, dtype=torch.long)
        input_tensor = torch.randn(
            num_embedding_rows, num_cols, device=device, dtype=dtype
        ).requires_grad_(True)
        input_ref = input_tensor.detach().clone().requires_grad_(True)

        output = torch.ops.fbgemm.group_index_select_dim0([input_tensor], [indices])
        output_ref = [torch.index_select(input_ref, 0, indices)]
        grad = torch.ones(num_indices, num_cols, device=device, dtype=dtype)
        output_ref[0].backward(grad)
        output[0].backward(grad)

        torch.testing.assert_close(
            input_tensor.grad,
            input_ref.grad,
            atol=0.5,
            rtol=0,
            msg="grad_input mismatch at the ROCm column-tile cache boundary",
        )
