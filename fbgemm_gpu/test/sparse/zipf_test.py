#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest

import torch

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests


class ZipfTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @optests.dontGenerateOpCheckTests(
        "zipf_cuda has no FakeTensor/meta dispatch; this test validates its CUDA "
        "contract directly"
    )
    def test_zipf_grid_stride_smoke(self) -> None:
        n = 1024
        output = torch.ops.fbgemm.zipf_cuda(1.5, n, 0)

        self.assertEqual(output.shape, (n,))
        self.assertEqual(output.dtype, torch.int64)
        self.assertGreaterEqual(int(output.min().item()), 0)
        self.assertTrue(torch.isfinite(output.to(torch.float64)).all().item())

        repeated_output = torch.ops.fbgemm.zipf_cuda(1.5, n, 0)
        torch.testing.assert_close(output, repeated_output)

        unique_count = torch.unique(output).numel()
        self.assertGreaterEqual(unique_count, 2)
        self.assertLessEqual(unique_count, n)


extend_test_class(ZipfTest)

if __name__ == "__main__":
    unittest.main()
