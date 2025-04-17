#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import unittest

import fbgemm_gpu.tbe.bench  # noqa F401
import hypothesis.strategies as st
import torch
from hypothesis import given, settings


def rand_int(min_value: int, max_value: int) -> int:
    return torch.randint(min_value, max_value, (1,)).tolist()[0]


class TBEIndicesGeneratorTest(unittest.TestCase):
    # pyre-ignore[56]
    @given(
        zipf_q=st.sampled_from([0.0001, 0.001, 0.01, 0.1, 0.5, 0.7]),
        zipf_s=st.sampled_from([0.0001, 0.001, 0.01, 0.1, 0.5, 0.7]),
        max_index=st.integers(100, 10000),
        num_indices=st.integers(1000, 100000),
        dtype=st.sampled_from([torch.float32, torch.float64]),
    )
    @settings(max_examples=20, deadline=None)
    def test_indices_generation(
        self,
        zipf_q: float,
        zipf_s: float,
        max_index: int,
        num_indices: int,
        dtype: torch.dtype,
    ) -> None:
        # Generate the heavy hitters distribution
        # The length needs to be < max_index
        # Furthermore, the sum of the distribution needs to be between 0 and 1
        heavy_hitters = torch.rand(
            rand_int(10, max_index - 1),
            dtype=dtype,
        )
        epsilon = torch.rand(1)[0] * 1000 + 0.01
        heavy_hitters /= torch.sum(heavy_hitters) + epsilon

        indices = torch.ops.fbgemm.tbe_generate_indices_from_distribution(
            heavy_hitters,
            zipf_q,
            zipf_s,
            max_index,
            num_indices,
        )

        assert indices.shape == (num_indices,)
        assert indices.dtype == torch.int64
        assert not torch.any(indices > max_index)
        assert not torch.any(indices < 0)
