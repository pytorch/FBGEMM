# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import hypothesis.strategies as st
import torch
from hypothesis import given, settings


class CoalesceTest(unittest.TestCase):
    # pyre-ignore
    @given(
        device=st.sampled_from(
            [torch.device("cpu"), torch.device("cuda")]
            if torch.cuda.is_available()
            else [torch.device("cpu")]
        ),
        batch_size=st.integers(min_value=10, max_value=5000),
        num_inputs=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=40, deadline=None)
    def test_coalesce_batches(
        self, device: torch.device, batch_size: int, num_inputs: int
    ) -> None:
        move_size = batch_size // 3
        new_bids = list(range(move_size))
        old_bids = []
        while len(old_bids) < len(new_bids):
            bid = random.randint(move_size, batch_size - 1)
            if bid not in old_bids:
                old_bids.append(bid)
        new_bids = torch.tensor(new_bids).to(device)
        old_bids = torch.tensor(old_bids).to(device)
        inputs = []
        inputs_ref = []
        for _ in range(num_inputs):
            dtype = random.choice([torch.bool, torch.int, torch.float])
            if dtype == torch.float:
                s = torch.rand([batch_size, 10], dtype=dtype)
                t = s[:, :5]
            elif dtype == torch.int:
                s = torch.randint(-100, 100, [batch_size, 20], dtype=dtype)
                t = s[:, :5]
            else:
                assert dtype == torch.bool
                t = torch.randint(0, 2, [batch_size], dtype=dtype)
            t = t.to(device)
            inputs.append(t)
            inputs_ref.append(t.clone())

        torch.ops.fbgemm.coalesce_batches(inputs, inputs, old_bids, new_bids)

        for i in range(num_inputs):
            inputs_ref[i][new_bids] = inputs_ref[i][old_bids]

        for i in range(num_inputs):
            torch.testing.assert_close(inputs[i], inputs_ref[i])
