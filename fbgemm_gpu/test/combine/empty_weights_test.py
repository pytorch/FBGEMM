#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from fbgemm_gpu import sparse_ops  # noqa: F401
from hypothesis import given, settings

from .common import open_source, TBEInputPrepareReference

if open_source:
    # pyre-ignore[21]
    from test_utils import cpu_and_maybe_gpu, optests
else:
    from fbgemm_gpu.test.test_utils import cpu_and_maybe_gpu, optests


@optests.generate_opcheck_tests()
class EmptyWeightsTest(unittest.TestCase):

    def _get_inputs(
        self, device: torch.device, all_weights_empty: bool
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        arg0_list = [
            [88, 55],
            [80, 29],
            [2, 85],
            [39, 51],
            [84, 35],
            [12, 6],
            [94, 43],
            [98, 59],
            [19, 68],
            [97, 89],
        ]
        arg0 = [torch.tensor(t, dtype=torch.int32, device=device) for t in arg0_list]

        arg1_list = [
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        ]
        arg1 = [torch.tensor(t, dtype=torch.int32, device=device) for t in arg1_list]

        if all_weights_empty:
            arg2_list = [[] for _ in range(len(arg0))]
        else:
            arg2_list = [
                [],
                [],
                [],
                [],
                [3.0, 3.0],
                [],
                [],
                [3.0, 3.0],
                [3.0, 3.0],
                [],
            ]
        arg2 = [torch.tensor(t, dtype=torch.float, device=device) for t in arg2_list]

        return (arg0, arg1, arg2)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(device=cpu_and_maybe_gpu())
    @settings(deadline=None)
    def test_tbe_input_combine_with_length_partially_empty_weights(
        self, device: torch.device
    ) -> None:
        arg0, arg1, arg2 = self._get_inputs(device, all_weights_empty=False)
        outputs = torch.ops.fbgemm.tbe_input_combine_with_length(arg0, arg1, arg2)

        include_last_offsets = [False] * (len(arg0) - 1) + [True]
        ref_mod = TBEInputPrepareReference(include_last_offsets)
        ref_outputs = ref_mod(arg0, arg1, arg2)

        # indices
        self.assertTrue(ref_outputs[0].allclose(outputs[0]))
        # per sample weights
        self.assertTrue(ref_outputs[2].allclose(outputs[2]))

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(device=cpu_and_maybe_gpu())
    @settings(deadline=None)
    def test_tbe_input_combine_with_length_all_empty_weights(
        self, device: torch.device
    ) -> None:
        arg0, arg1, arg2 = self._get_inputs(device, all_weights_empty=True)
        outputs = torch.ops.fbgemm.tbe_input_combine_with_length(arg0, arg1, arg2)

        include_last_offsets = [False] * (len(arg0) - 1) + [True]
        ref_mod = TBEInputPrepareReference(include_last_offsets)
        ref_outputs = ref_mod(arg0, arg1, arg2)

        # indices
        self.assertTrue(ref_outputs[0].allclose(outputs[0]))
        # per sample weights
        self.assertEqual(outputs[2].numel(), 0)


if __name__ == "__main__":
    unittest.main()
