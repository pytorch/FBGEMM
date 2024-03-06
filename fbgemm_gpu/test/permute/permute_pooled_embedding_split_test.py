#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest
from itertools import accumulate
from typing import List, Tuple

import torch

from .common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable

typed_gpu_unavailable: Tuple[bool, str] = gpu_unavailable


class PermutePooledEmbeddingSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = "cuda"

    @unittest.skipIf(*gpu_unavailable)
    def test_duplicate_permutations(self) -> None:
        # self.device = "cuda"
        embs_dims = [2, 3, 1, 4]
        permute = [3, 0, 2, 0, 1, 3]
        expected_result = [6, 7, 8, 9, 0, 1, 5, 0, 1, 2, 3, 4, 6, 7, 8, 9]
        input = torch.Tensor([range(10)]).to(device="cuda")

        _permute = torch.tensor(permute, device=self.device, dtype=torch.int64)
        _offset_dim_list = torch.tensor(
            [0] + list(accumulate(embs_dims)), device=self.device, dtype=torch.int64
        )
        inv_permute: List[int] = [0] * len(permute)
        for i, p in enumerate(permute):
            inv_permute[p] = i
        _inv_permute = torch.tensor(inv_permute, device=self.device, dtype=torch.int64)
        inv_embs_dims = [embs_dims[i] for i in permute]
        _inv_offset_dim_list = torch.tensor(
            [0] + list(accumulate(inv_embs_dims)),
            device=self.device,
            dtype=torch.int64,
        )

        result = torch.ops.fbgemm.permute_duplicate_pooled_embs_auto_grad_split(
            input,
            _offset_dim_list.to(device=input.device),
            _permute.to(device=input.device),
            _inv_offset_dim_list.to(device=input.device),
            _inv_permute.to(device=input.device),
        )
        self.assertEqual(
            result.view(16).tolist(),
            expected_result,
        )

        input = input.to(device="cpu")
        result = torch.ops.fbgemm.permute_duplicate_pooled_embs_auto_grad_split(
            input,
            _offset_dim_list.to(device=input.device),
            _permute.to(device=input.device),
            _inv_offset_dim_list.to(device=input.device),
            _inv_permute.to(device=input.device),
        )
        self.assertEqual(
            result.view(16).tolist(),
            expected_result,
        )


if __name__ == "__main__":
    unittest.main()
