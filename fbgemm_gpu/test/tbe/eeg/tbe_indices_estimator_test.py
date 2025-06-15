#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import random
import unittest

import fbgemm_gpu  # noqa F401

import torch
from fbgemm_gpu.tbe.bench import EEG_MAX_HEAVY_HITTERS


class TBEIndicesEstimatorTest(unittest.TestCase):
    def test_indices_estimation(self) -> None:
        max_i = random.randint(1, 200)
        num_i = random.randint(100, 1000)
        indices = torch.randint(0, max_i, (num_i,), dtype=torch.int64)

        heavy_hitters, q, s, max_index, num_indices = (
            torch.ops.fbgemm.tbe_estimate_indices_distribution(indices)
        )

        assert (
            heavy_hitters.numel() <= EEG_MAX_HEAVY_HITTERS
        ), "Materialized too many heavy hitters"
        assert torch.all(heavy_hitters >= 0) and torch.all(
            heavy_hitters < 1
        ), "Invalid heavy hitter values"
        assert q > 0, "Invalid q"
        assert s >= 0, "Invalid s"
        assert max_index >= 0 and max_index <= max_i, "Invalid max_index"
        assert num_indices == num_i, "num_indices does not match num_i"
