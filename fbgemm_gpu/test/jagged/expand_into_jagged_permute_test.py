#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import itertools
import random
import unittest
from typing import List

import hypothesis.strategies as st
import torch
import torch._dynamo
from hypothesis import given, settings, Verbosity

from .common import additional_decorators, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, on_oss_clang, optests
else:
    from fbgemm_gpu.test.test_utils import gpu_available, on_oss_clang, optests


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class ExpandIntoJaggedPermuteTest(unittest.TestCase):
    @staticmethod
    def expand_into_jagged_permute_ref_(
        permute: List[int],
        length: List[int],
    ) -> List[int]:
        offsets = [0] + list(itertools.accumulate(length))
        output_permute = []
        for r in permute:
            output_permute.extend(
                range(
                    offsets[r],
                    offsets[r + 1],
                )
            )
        return output_permute

    @unittest.skipIf(*on_oss_clang)
    @given(
        T=st.integers(min_value=10, max_value=20),
        W=st.integers(min_value=8, max_value=64),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_expand_into_jagged_permute(
        self,
        T: int,
        W: int,
    ) -> None:
        length_per_w = [random.randint(5000, 10000) for i in range(W)]
        length_1d = list(
            itertools.chain.from_iterable(itertools.repeat(x, T) for x in length_per_w)
        )
        permute_list = list(range(T * W))
        random.shuffle(permute_list)
        permuted_length_1d = [length_1d[r] for r in permute_list]
        permute_tensor = torch.tensor(permute_list)

        # compute offsets
        offsets_1d = [0] + list(itertools.accumulate(length_1d))
        permuted_offsets_1d = [0] + list(itertools.accumulate(permuted_length_1d))
        offsets_1d_tensor = torch.tensor(offsets_1d)
        permuted_offsets_1d_tensor = torch.tensor(permuted_offsets_1d)

        # cpu op
        output_permute_cpu = torch.ops.fbgemm.expand_into_jagged_permute(
            permute_tensor,
            offsets_1d_tensor,
            permuted_offsets_1d_tensor,
            offsets_1d[-1],
        )

        # reference solution
        output_permute_ref = self.expand_into_jagged_permute_ref_(
            permute_list,
            length_1d,
        )
        output_permute_ref_tensor = torch.tensor(output_permute_ref)

        # assert cpu and gpu ops
        torch.testing.assert_close(output_permute_cpu, output_permute_ref_tensor)
        if gpu_available:
            # gpu op
            output_permute_gpu = torch.ops.fbgemm.expand_into_jagged_permute(
                permute_tensor.cuda(),
                offsets_1d_tensor.cuda(),
                permuted_offsets_1d_tensor.cuda(),
                offsets_1d[-1],
            )
            torch.testing.assert_close(
                output_permute_gpu.cpu(), output_permute_ref_tensor
            )


if __name__ == "__main__":
    unittest.main()
