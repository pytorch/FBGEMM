#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest

import hypothesis.strategies as st
import torch
from fbgemm_gpu.tbe.bench import (
    BatchParams,
    IndicesParams,
    PoolingParams,
    TBEDataConfig,
)
from hypothesis import given


def rand_int(min_value: int, max_value: int) -> int:
    return torch.randint(min_value, max_value, (1,)).tolist()[0]


class TBEDataConfigModelsTest(unittest.TestCase):
    # pyre-ignore[56]
    @given(
        dtype=st.sampled_from([torch.float32, torch.float64]),
    )
    def test_indices_params_serialization(
        self,
        dtype: torch.dtype,
    ) -> None:
        params1 = IndicesParams(
            torch.rand(20, dtype=dtype),
            torch.rand(1).tolist()[0],
            torch.rand(1).tolist()[0],
            torch.int32,
            torch.int64,
        )
        assert IndicesParams.from_json(params1.json()) == params1

    def test_batch_params_serialization(self) -> None:
        params1 = BatchParams(
            rand_int(10, 100), rand_int(10, 100), "normal", rand_int(10, 100)
        )
        assert BatchParams.from_json(params1.json()) == params1

    def test_pooling_params_serialization(self) -> None:
        params1 = PoolingParams(rand_int(10, 100), rand_int(10, 100), "normal")
        assert PoolingParams.from_json(params1.json()) == params1

    # pyre-ignore[56]
    @given(
        dtype=st.sampled_from([torch.float32, torch.float64]),
    )
    def test_tbe_data_config_serialization(
        self,
        dtype: torch.dtype,
    ) -> None:
        config = TBEDataConfig(
            rand_int(10, 100),
            rand_int(10, 100),
            rand_int(10, 100),
            random.random() < 0.5,
            random.random() < 0.5,
            BatchParams(
                rand_int(10, 100), rand_int(10, 100), "normal", rand_int(10, 100)
            ),
            IndicesParams(
                torch.rand(20, dtype=torch.float32),
                torch.rand(1).tolist()[0],
                torch.rand(1).tolist()[0],
                torch.int32,
                torch.int64,
            ),
            PoolingParams(rand_int(10, 100), rand_int(10, 100), "normal"),
            not torch.cuda.is_available(),
        )
        assert TBEDataConfig.from_json(config.json()) == config
