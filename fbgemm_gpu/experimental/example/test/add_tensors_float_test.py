# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch

from fbgemm_gpu.experimental.example import utils


class ExampleTest(unittest.TestCase):
    def test_add_tensors_float(self) -> None:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        expected = torch.tensor([5, 7, 9], dtype=torch.float)
        c = utils.add_tensors(a, b)
        torch.testing.assert_close(c.cpu(), expected.cpu())


if __name__ == "__main__":
    unittest.main()
