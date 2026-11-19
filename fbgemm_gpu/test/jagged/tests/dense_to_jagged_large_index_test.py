# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import fbgemm_gpu
import torch

if TYPE_CHECKING or not getattr(fbgemm_gpu, "open_source", False):
    from fbgemm_gpu.test.large_regression_utils import GpuRegressionTest
else:
    from large_regression_utils import GpuRegressionTest


class DenseToJaggedLargeIndexTest(GpuRegressionTest):
    minimum_memory_gib = 16

    def test_dense_to_jagged_backward_int32_overflow(self) -> None:
        device = self.device
        B = 1024
        max_L = 40960
        D = 64
        dtype = torch.bfloat16
        self.assertTrue(
            B * max_L < 1 << 31,
            "test must keep B * max_L below INT_MAX (we are testing the PTA32 stride-overflow path, not the outer-loop overflow path)",
        )
        self.assertTrue(
            B * max_L * D > (1 << 31) - 1, "test must exceed INT_MAX on B * max_L * D"
        )
        wrap_oidx = ((1 << 31) - 1 + max_L * D - 1) // (max_L * D)
        self.assertTrue(
            B - 1 >= wrap_oidx,
            "last batch must fall inside the PTA32 stride-overflow zone",
        )
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[B] = 1
        grad_jagged = torch.ones((1, D), dtype=dtype, device=device)
        out = torch.ops.fbgemm.jagged_to_padded_dense(
            grad_jagged, [offsets], [max_L], padding_value=0.0
        )
        self.assertEqual(tuple(out.shape), (B, max_L, D))
        self.assertEqual(out[B - 1, 0, 0].item(), 1.0)
        self.assertEqual(out[B - 1, 0, D - 1].item(), 1.0)
        for pos in [1, max_L // 2, max_L - 1]:
            self.assertEqual(out[B - 1, pos, 0].item(), 0.0)
            self.assertEqual(out[0, pos, 0].item(), 0.0)
        self.assertEqual(out[wrap_oidx, 0, 0].item(), 0.0)
        self.assertEqual(out[wrap_oidx, max_L - 1, 0].item(), 0.0)
