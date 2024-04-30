# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from fbgemm_gpu.experimental.example import utils

from . import gpu_unavailable


class SgemmFloatTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_sgemm_float(self) -> None:
        alpha = 3.14
        beta = 2.71

        A = torch.rand(4, 3, dtype=torch.float, device="cuda")
        B = torch.rand(3, 5, dtype=torch.float, device="cuda")
        C = torch.rand(4, 5, dtype=torch.float, device="cuda")
        D = utils.sgemm(alpha, A, B, beta, C)

        expected = torch.add(alpha * torch.matmul(A, B), beta * C)
        torch.testing.assert_close(D.cpu(), expected.cpu())


if __name__ == "__main__":
    unittest.main()
