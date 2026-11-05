# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import unittest
from typing import NoReturn

import fbgemm_gpu
import fbgemm_gpu.sparse_ops  # noqa: F401
import torch

if not getattr(fbgemm_gpu, "open_source", False):
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")


class GpuRegressionTest(unittest.TestCase):
    minimum_memory_gib: int = 0
    rocm_only: bool = False
    device: torch.device

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self._unsupported("GPU unavailable")
        self.device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        properties = torch.cuda.get_device_properties(self.device)
        if self.rocm_only and torch.version.hip is None:
            self._unsupported("ROCm regression")
        if properties.total_memory < self.minimum_memory_gib * (1 << 30):
            self._unsupported(f"Requires {self.minimum_memory_gib} GiB of GPU memory")
        torch.cuda.reset_peak_memory_stats(self.device)
        print(
            json.dumps(
                {
                    "gpu": properties.name,
                    "memory_bytes": properties.total_memory,
                    "free_memory_bytes": torch.cuda.mem_get_info(self.device)[0],
                    "warp_size": properties.warp_size,
                    "hip": torch.version.hip,
                    "cuda": torch.version.cuda,
                }
            )
        )

    def _unsupported(self, reason: str) -> NoReturn:
        if os.environ.get("FBGEMM_LARGE_REGRESSION_CI") == "1":
            self.fail(reason)
        self.skipTest(reason)

    def tearDown(self) -> None:
        print(
            json.dumps(
                {
                    "allocated_peak_bytes": torch.cuda.max_memory_allocated(
                        self.device
                    ),
                    "reserved_peak_bytes": torch.cuda.max_memory_reserved(self.device),
                }
            )
        )

    def launch_boundary(self) -> tuple[int, int]:
        warp = torch.cuda.get_device_properties(self.device).warp_size
        self.assertIn(warp, (32, 64))
        features = warp // 2
        rows_per_block = 1024 // warp
        rows = (1 << 22) * rows_per_block - rows_per_block + 1
        self.assertLess(rows * features, (1 << 31) - 1)
        self.assertEqual(
            ((rows + rows_per_block - 1) // rows_per_block) * 1024, 1 << 32
        )
        return rows, features
