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


class DenseDenseElementwiseAddLargeGridTest(GpuRegressionTest):
    minimum_memory_gib = 40

    def test_jagged_dense_dense_elementwise_add_jagged_output_large_grid(self) -> None:
        rows, features = self.launch_boundary()
        device = self.device
        x = torch.full((rows, features), 1.0, dtype=torch.float32, device=device)
        offsets = [torch.tensor([0, rows], dtype=torch.int64, device=device)]
        y0 = torch.full((1, rows, features), 2.0, dtype=torch.float32, device=device)
        y1 = torch.full((1, rows, features), 4.0, dtype=torch.float32, device=device)
        output, output_offsets = (
            torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
                x, offsets, y0, y1
            )
        )
        self.assertEqual(output.shape, (rows, features))
        for row in (0, rows // 2, rows - 1):
            torch.testing.assert_close(
                output[row, [0, features - 1]],
                torch.full((2,), 7.0, device=device),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(output_offsets[0], offsets[0])
