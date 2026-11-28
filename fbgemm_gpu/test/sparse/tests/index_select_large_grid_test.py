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


class IndexSelectLargeGridTest(GpuRegressionTest):
    minimum_memory_gib = 36

    def test_index_select_dim0_forward_large_grid(self) -> None:
        N = (1 << 23) + 1
        D = 1024
        self.assertGreater(N * (D // 2), 1 << 32)
        device = self.device
        with torch.no_grad():
            input_t = torch.zeros((N, D), dtype=torch.float16, device=device)
            input_t[0, 0] = 1.0
            input_t[N // 2, D // 2] = 2.0
            input_t[N - 1, D - 1] = 3.0
            indices = torch.arange(N, dtype=torch.int32, device=device)
            output = torch.ops.fbgemm.index_select_dim0(input_t, indices)
            self.assertEqual(output.shape, (N, D))
            self.assertEqual(output[0, 0].item(), 1.0)
            self.assertEqual(output[N // 2, D // 2].item(), 2.0)
            self.assertEqual(output[N - 1, D - 1].item(), 3.0)
            del input_t, indices, output
        torch.cuda.empty_cache()
